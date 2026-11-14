"""Token containers: per-cell detected features, one view of a cell, and a padded batch.

Index spaces (AGENTS.md section 5): ``col_idx`` is a column of the user's AnnData; ``emb_idx`` is a
row of the reference table (``genes.parquet`` / ``peaks.parquet``). Everything in this module is in
``emb_idx`` space. Padding and collation live here, not in the strategies.

Contract (docs/data-schema.md section 2.3)::

    TokenSet:  emb_idx int32[L], kind int8[L]  (1 = gene, 2 = peak), L <= lmax
    Batch:     emb_idx int32[B, lmax] (pad = -1), kind int8[B, lmax] (pad = 0),
               lengths int64[B], cell_index int64[B], view_index int64[B]

``torch`` is imported only inside :func:`collate` / :class:`Batch` so that strategies and the
feature parser stay importable without it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    import torch

__all__ = [
    "GENE",
    "PAD_EMB_IDX",
    "PAD_KIND",
    "PEAK",
    "Batch",
    "CellFeatures",
    "Modality",
    "TokenSet",
    "collate",
    "detected_from_csr_row",
    "key_padding_mask",
]

GENE: int = 1
"""``kind`` value of a gene token."""
PEAK: int = 2
"""``kind`` value of a peak token."""
PAD_EMB_IDX: int = -1
"""``emb_idx`` written at padded positions of a :class:`Batch`."""
PAD_KIND: int = 0
"""``kind`` written at padded positions of a :class:`Batch`."""

Modality = Literal["gene", "peak"]

_KIND_OF: dict[str, int] = {"gene": GENE, "peak": PEAK}


def _as_int32_sorted(a: np.ndarray | Sequence[int] | None, name: str) -> np.ndarray:
    """Return ``a`` as a 1-D int32 array and check it is strictly increasing and >= 0."""
    arr = np.asarray(a if a is not None else [], dtype=np.int64).reshape(-1)
    if arr.size and arr.min() < 0:
        raise ValueError(f"{name} must not contain negative (unmapped) emb_idx values")
    if arr.size > 1 and not np.all(np.diff(arr) > 0):
        raise ValueError(f"{name} must be strictly increasing (sorted, unique emb_idx)")
    return arr.astype(np.int32, copy=False)


def detected_from_csr_row(
    col_idx: np.ndarray,
    values: np.ndarray | None,
    col_to_emb_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Translate one CSR row (dataset columns) into sorted detected ``emb_idx`` + aligned values.

    Detection is ``value > 0`` (docs/architecture.md section 2). Columns whose ``emb_idx`` is ``-1``
    (unmapped) are dropped. Several columns mapping to the same ``emb_idx`` (duplicate symbols, R3)
    are OR-ed: the feature is detected once and carries the largest of its values.

    Args:
        col_idx: ``(nnz,)`` column indices of the CSR row (any order).
        values: ``(nnz,)`` values aligned with ``col_idx``; ``None`` for binary modalities (peaks).
        col_to_emb_idx: ``(n_var,)`` int array, ``var["bloom_emb_idx"]`` (``-1`` = unmapped).

    Returns:
        ``(emb_idx int32 (n,) strictly increasing, values float32 (n,) aligned or None)``.
    """
    col_idx = np.asarray(col_idx).reshape(-1)
    col_to_emb_idx = np.asarray(col_to_emb_idx)
    emb = col_to_emb_idx[col_idx].astype(np.int64)
    if values is None:
        keep = emb >= 0
        return np.unique(emb[keep]).astype(np.int32), None
    vals = np.asarray(values, dtype=np.float32).reshape(-1)
    if vals.shape != col_idx.shape:
        raise ValueError("values must be aligned with col_idx")
    keep = (emb >= 0) & (vals > 0)
    emb, vals = emb[keep], vals[keep]
    if emb.size == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)
    # sort by (emb_idx asc, value desc) so the first row of each emb_idx group holds its max value
    order = np.lexsort((-vals, emb))
    emb, vals = emb[order], vals[order]
    first = np.ones(emb.size, dtype=bool)
    first[1:] = emb[1:] != emb[:-1]
    return emb[first].astype(np.int32), vals[first].astype(np.float32)


@dataclass(frozen=True)
class CellFeatures:
    """Detected features of one cell in ``emb_idx`` space (the input to a :class:`ViewStrategy`).

    One instance carries **one modality** (genes from an RNA file or peaks from an ATAC file); a
    strategy applied to it can only emit tokens of that modality (S7). Both arrays are strictly
    increasing, so set operations are plain merges and ties in ``TopExpressed`` break by ascending
    ``emb_idx``.

    Attributes:
        cell_index: Row of the cell in the dataset; seeds the per-cell RNG.
        detected_gene_emb_idx: ``(n_genes,)`` int32, strictly increasing.
        gene_values: ``(n_genes,)`` float32 aligned with ``detected_gene_emb_idx`` (the raw or
            normalized expression; only used for ranking by ``TopExpressed``).
        detected_peak_emb_idx: ``(n_peaks,)`` int32, strictly increasing.
        peak_values: ``(n_peaks,)`` float32 aligned with ``detected_peak_emb_idx`` (summed fragment
            counts of the user peaks mapped onto each reference peak; only used by ``TopExpressed``
            when applied to ATAC), or ``None`` for binary peak input.
    """

    cell_index: int
    detected_gene_emb_idx: np.ndarray
    gene_values: np.ndarray
    detected_peak_emb_idx: np.ndarray
    peak_values: np.ndarray | None = None

    def __post_init__(self) -> None:
        genes = _as_int32_sorted(self.detected_gene_emb_idx, "detected_gene_emb_idx")
        peaks = _as_int32_sorted(self.detected_peak_emb_idx, "detected_peak_emb_idx")
        vals = np.asarray(
            self.gene_values if self.gene_values is not None else [], dtype=np.float32
        ).reshape(-1)
        if vals.shape != genes.shape:
            raise ValueError(
                f"gene_values {vals.shape} must align with detected_gene_emb_idx {genes.shape}"
            )
        if genes.size and peaks.size:
            raise ValueError(
                "CellFeatures carries one modality per instance (mixed gene+peak views are not a "
                "v1 preset); build one CellFeatures per modality file"
            )
        pvals = None
        if self.peak_values is not None:
            pvals = np.asarray(self.peak_values, dtype=np.float32).reshape(-1)
            if pvals.shape != peaks.shape:
                raise ValueError(
                    f"peak_values {pvals.shape} must align with detected_peak_emb_idx {peaks.shape}"
                )
        object.__setattr__(self, "cell_index", int(self.cell_index))
        object.__setattr__(self, "detected_gene_emb_idx", genes)
        object.__setattr__(self, "gene_values", vals)
        object.__setattr__(self, "detected_peak_emb_idx", peaks)
        object.__setattr__(self, "peak_values", pvals)

    # -- constructors -------------------------------------------------------------------------

    @classmethod
    def genes(
        cls,
        cell_index: int,
        emb_idx: np.ndarray | Sequence[int],
        values: np.ndarray | Sequence[float],
    ) -> CellFeatures:
        """Gene-modality cell from already sorted ``emb_idx`` and aligned ``values``."""
        return cls(cell_index, np.asarray(emb_idx), np.asarray(values), np.zeros(0, np.int32))

    @classmethod
    def peaks(
        cls,
        cell_index: int,
        emb_idx: np.ndarray | Sequence[int],
        values: np.ndarray | Sequence[float] | None = None,
    ) -> CellFeatures:
        """Peak-modality cell from already sorted ``emb_idx`` (+ optional aligned count values)."""
        return cls(
            cell_index,
            np.zeros(0, np.int32),
            np.zeros(0, np.float32),
            np.asarray(emb_idx),
            None if values is None else np.asarray(values),
        )

    @classmethod
    def from_csr_row(
        cls,
        cell_index: int,
        col_idx: np.ndarray,
        values: np.ndarray | None,
        col_to_emb_idx: np.ndarray,
        modality: Modality,
    ) -> CellFeatures:
        """Build from one CSR row of a prepared matrix (see :func:`detected_from_csr_row`).

        Args:
            cell_index: Row of the cell.
            col_idx: ``(nnz,)`` column indices of the row.
            values: ``(nnz,)`` values aligned with ``col_idx``; for peaks they are optional and,
                when given, kept as ``peak_values`` (counts) so ``TopExpressed`` can rank peaks.
            col_to_emb_idx: ``var["bloom_emb_idx"]`` of the modality file (``-1`` = unmapped).
            modality: ``"gene"`` or ``"peak"``.
        """
        if modality == "gene":
            if values is None:
                raise ValueError("gene features need values (ranking by TopExpressed)")
            emb, vals = detected_from_csr_row(col_idx, values, col_to_emb_idx)
            return cls.genes(cell_index, emb, vals)
        if modality == "peak":
            emb, vals = detected_from_csr_row(col_idx, values, col_to_emb_idx)
            return cls.peaks(cell_index, emb, vals)
        raise ValueError(f"modality must be 'gene' or 'peak', got {modality!r}")

    # -- accessors ------------------------------------------------------------------------------

    @property
    def modality(self) -> Modality | None:
        """``"gene"`` / ``"peak"`` by which array is populated; ``None`` for an empty cell."""
        if self.detected_gene_emb_idx.size:
            return "gene"
        if self.detected_peak_emb_idx.size:
            return "peak"
        return None

    @property
    def n_detected(self) -> int:
        """Number of detected features (of the populated modality)."""
        return int(self.detected_gene_emb_idx.size + self.detected_peak_emb_idx.size)

    def restrict(self, universe_emb_idx: np.ndarray | None) -> CellFeatures:
        """Intersect the detected set with ``universe_emb_idx`` (sorted or not); ``None`` = no-op."""
        if universe_emb_idx is None:
            return self
        uni = np.asarray(universe_emb_idx).reshape(-1)
        if self.modality == "gene":
            keep = np.isin(self.detected_gene_emb_idx, uni, assume_unique=False)
            return CellFeatures.genes(
                self.cell_index, self.detected_gene_emb_idx[keep], self.gene_values[keep]
            )
        if self.modality == "peak":
            keep = np.isin(self.detected_peak_emb_idx, uni, assume_unique=False)
            pv = None if self.peak_values is None else self.peak_values[keep]
            return CellFeatures.peaks(self.cell_index, self.detected_peak_emb_idx[keep], pv)
        return self


@dataclass(frozen=True)
class TokenSet:
    """One view of one cell: ``L <= lmax`` tokens in ``emb_idx`` space.

    Attributes:
        emb_idx: ``(L,)`` int32 rows of the reference table (never ``-1``).
        kind: ``(L,)`` int8, ``1`` = gene, ``2`` = peak — all the same within one view (a view
            holds one modality; a mix raises ``ValueError``, issue #46).
    """

    emb_idx: np.ndarray
    kind: np.ndarray

    def __post_init__(self) -> None:
        emb = np.asarray(self.emb_idx, dtype=np.int32).reshape(-1)
        kind = np.asarray(self.kind, dtype=np.int8).reshape(-1)
        if emb.shape != kind.shape:
            raise ValueError(f"emb_idx {emb.shape} and kind {kind.shape} must have the same length")
        if emb.size and emb.min() < 0:
            raise ValueError("TokenSet.emb_idx must not contain -1 (unmapped) entries")
        if kind.size and not np.isin(kind, (GENE, PEAK)).all():
            raise ValueError(f"TokenSet.kind must be {GENE} (gene) or {PEAK} (peak)")
        if kind.size and (kind == GENE).any() and (kind == PEAK).any():
            raise ValueError(
                "TokenSet mixes gene (kind 1) and peak (kind 2) tokens: a view holds one modality "
                "(issue #46); build one TokenSet per modality"
            )
        object.__setattr__(self, "emb_idx", emb)
        object.__setattr__(self, "kind", kind)

    def __len__(self) -> int:
        return int(self.emb_idx.size)

    @classmethod
    def of(cls, emb_idx: np.ndarray | Sequence[int], modality: Modality) -> TokenSet:
        """Single-modality view: every token gets the ``kind`` of ``modality``."""
        emb = np.asarray(emb_idx, dtype=np.int32).reshape(-1)
        return cls(emb, np.full(emb.shape, _KIND_OF[modality], dtype=np.int8))

    @classmethod
    def genes(cls, emb_idx: np.ndarray | Sequence[int]) -> TokenSet:
        """Gene view (``kind == 1``)."""
        return cls.of(emb_idx, "gene")

    @classmethod
    def peaks(cls, emb_idx: np.ndarray | Sequence[int]) -> TokenSet:
        """Peak view (``kind == 2``)."""
        return cls.of(emb_idx, "peak")


@dataclass
class Batch:
    """Collated views, padded to ``lmax`` (docs/data-schema.md section 2.3).

    Attributes:
        emb_idx: ``(B, lmax)`` int32, ``-1`` at pads.
        kind: ``(B, lmax)`` int8, ``0`` at pads.
        lengths: ``(B,)`` int64 number of real tokens per row.
        cell_index: ``(B,)`` int64 dataset row of each view's cell.
        view_index: ``(B,)`` int64 index of the view within its cell's strategy output.
    """

    emb_idx: torch.Tensor  # (B, lmax) int32, pad = -1
    kind: torch.Tensor  # (B, lmax) int8, pad = 0
    lengths: torch.Tensor  # (B,) int64
    cell_index: torch.Tensor  # (B,) int64
    view_index: torch.Tensor  # (B,) int64

    def __len__(self) -> int:
        return int(self.emb_idx.shape[0])

    @property
    def lmax(self) -> int:
        """Padded length of every row."""
        return int(self.emb_idx.shape[1])

    def key_padding_mask(self) -> torch.Tensor:
        """``(B, lmax)`` bool, **True = pad** (the model ignores True positions)."""
        return key_padding_mask(self)

    def to(self, device: torch.device | str) -> Batch:
        """Move every tensor to ``device`` (``emb_idx``/``kind`` stay int32/int8)."""
        return Batch(*(t.to(device) for t in self._tensors()))

    def _tensors(self) -> tuple[torch.Tensor, ...]:
        return (self.emb_idx, self.kind, self.lengths, self.cell_index, self.view_index)


def collate(
    token_sets: Sequence[TokenSet],
    lmax: int | None = None,
    cell_index: Sequence[int] | np.ndarray | None = None,
    view_index: Sequence[int] | np.ndarray | None = None,
) -> Batch:
    """Pad ``token_sets`` to ``lmax`` and stack them into a :class:`Batch`.

    Args:
        token_sets: ``B`` views; each ``len(ts) <= lmax``.
        lmax: Padded length. ``None`` = the longest view in the batch (at least 1).
        cell_index: ``(B,)`` cell row of each view; ``None`` = ``0..B-1``.
        view_index: ``(B,)`` view number within each cell; ``None`` = zeros.

    Returns:
        :class:`Batch` with ``emb_idx (B, lmax)`` int32 (pad ``-1``), ``kind (B, lmax)`` int8
        (pad ``0``), ``lengths``, ``cell_index``, ``view_index`` all int64 ``(B,)``.
    """
    import torch

    B = len(token_sets)
    lengths = np.fromiter((len(ts) for ts in token_sets), dtype=np.int64, count=B)
    if lmax is None:
        lmax = int(lengths.max()) if B else 1
        lmax = max(lmax, 1)
    if B and lengths.max() > lmax:
        raise ValueError(f"a view has {int(lengths.max())} tokens > lmax={lmax}")
    emb = np.full((B, lmax), PAD_EMB_IDX, dtype=np.int32)  # (B, lmax)
    kind = np.full((B, lmax), PAD_KIND, dtype=np.int8)  # (B, lmax)
    for row, ts in enumerate(token_sets):
        n = len(ts)
        emb[row, :n] = ts.emb_idx
        kind[row, :n] = ts.kind
    ci = np.arange(B, dtype=np.int64) if cell_index is None else np.asarray(cell_index, np.int64)
    vi = np.zeros(B, dtype=np.int64) if view_index is None else np.asarray(view_index, np.int64)
    if ci.shape != (B,) or vi.shape != (B,):
        raise ValueError("cell_index and view_index must have one entry per token set")
    return Batch(
        emb_idx=torch.from_numpy(emb),
        kind=torch.from_numpy(kind),
        lengths=torch.from_numpy(lengths),
        cell_index=torch.from_numpy(ci.copy()),
        view_index=torch.from_numpy(vi.copy()),
    )


def key_padding_mask(batch: Batch) -> torch.Tensor:
    """``(B, lmax)`` bool mask derived from ``lengths``; **True = pad**."""
    import torch

    positions = torch.arange(batch.lmax, device=batch.lengths.device)  # (lmax,)
    return positions[None, :] >= batch.lengths[:, None]  # (B, lmax)
