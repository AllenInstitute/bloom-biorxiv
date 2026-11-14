"""``ViewStrategy``: the extension point that turns one cell into one or more token views.

docs/architecture.md section 5.2. A strategy is bound to a modality by the caller
(``strategies={"rna": ..., "atac": ...}``): it receives :class:`~bloom.tokens.CellFeatures` of one
modality and emits :class:`~bloom.tokens.TokenSet` views of that modality only. Padding and
collation are not the strategy's job (``bloom.tokens.collate``).

Determinism: the caller derives the per-cell generator with :func:`cell_rng`, so a view depends
only on ``(seed, cell_index)`` and never on batch size, worker count or device.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from bloom.feature_lists import FeatureList, FeatureReference
from bloom.tokens import CellFeatures, TokenSet

__all__ = ["Universe", "ViewStrategy", "cell_rng"]

logger = logging.getLogger(__name__)

Universe = FeatureList | np.ndarray | None
"""A ``universe``: a parsed feature list (resolved later against the reference), an array of
``emb_idx`` (already resolved), or ``None`` (no restriction)."""


def cell_rng(seed: int, cell_index: int) -> np.random.Generator:
    """Per-cell generator ``np.random.default_rng([seed, cell_index])`` (SeedSequence spawn)."""
    return np.random.default_rng([int(seed), int(cell_index)])


class ViewStrategy(ABC):
    """Abstract view strategy.

    Subclasses implement :meth:`sample` on an already universe-restricted cell; :meth:`views` is
    the public entry point (restriction, then sampling, then truncation checks).

    Attributes:
        lmax: Maximum tokens per view (``L <= lmax``).
        universe: Optional feature list / ``emb_idx`` array that the detected set is intersected
            with before sampling or ranking. A :class:`FeatureList` must be resolved against the
            reference tables with :meth:`resolve` before :meth:`views` is called.
    """

    #: preset name used by the CLI / config (``strategy:`` key); subclasses override
    name: str = "custom"

    def __init__(self, lmax: int, universe: Universe = None) -> None:
        if int(lmax) < 1:
            raise ValueError(f"lmax must be >= 1, got {lmax}")
        self.lmax: int = int(lmax)
        self.universe: FeatureList | np.ndarray | None
        self._universe_emb_idx: np.ndarray | None
        if universe is None:
            self.universe = None
            self._universe_emb_idx = None
        elif isinstance(universe, FeatureList):
            self.universe = universe
            self._universe_emb_idx = None  # resolved later
        else:
            arr = np.unique(np.asarray(universe, dtype=np.int64).reshape(-1))
            if arr.size and arr.min() < 0:
                raise ValueError("universe emb_idx must be >= 0 (drop unmapped -1 entries first)")
            self.universe = arr.astype(np.int32)
            self._universe_emb_idx = self.universe

    # -- universe -------------------------------------------------------------------------------

    @property
    def universe_emb_idx(self) -> np.ndarray | None:
        """Sorted unique ``emb_idx`` of the universe, or ``None`` when unrestricted / unresolved."""
        return self._universe_emb_idx

    @property
    def needs_resolve(self) -> bool:
        """True when ``universe`` (or a panel) is a :class:`FeatureList` not yet mapped to ``emb_idx``."""
        return isinstance(self.universe, FeatureList) and self._universe_emb_idx is None

    def resolve(self, reference: FeatureReference) -> ViewStrategy:
        """Map a :class:`FeatureList` universe to ``emb_idx`` through ``reference`` (in place).

        Returns ``self`` so it can be chained. Unknown identifiers are counted and logged once by
        :meth:`FeatureList.resolve`.
        """
        if isinstance(self.universe, FeatureList):
            res = self.universe.resolve(reference)
            self._universe_emb_idx = np.unique(res.emb_idx).astype(np.int32)
        return self

    def restrict(self, cell: CellFeatures) -> CellFeatures:
        """Apply ``universe`` to ``cell`` (detected intersect universe); identity when unrestricted."""
        if self.needs_resolve:
            raise RuntimeError(
                f"{type(self).__name__}: universe {self.universe.source or '<memory>'} is a "
                "FeatureList; call strategy.resolve(reference) before views()"
            )
        return cell.restrict(self._universe_emb_idx)

    # -- views ----------------------------------------------------------------------------------

    @abstractmethod
    def sample(self, cell: CellFeatures, rng: np.random.Generator) -> list[TokenSet]:
        """Produce the views of an already restricted cell (implemented by presets/subclasses).

        Must return ``[]`` for a cell with no detected features and never a view longer than
        ``self.lmax``.
        """

    def views(self, cell: CellFeatures, rng: np.random.Generator) -> list[TokenSet]:
        """Views of ``cell``: restrict to the universe, then :meth:`sample`.

        Args:
            cell: Detected features of one cell (one modality).
            rng: Per-cell generator from :func:`cell_rng`; deterministic presets ignore it.

        Returns:
            ``list[TokenSet]`` with ``len(ts) <= lmax`` each; empty for a cell without features.
        """
        out = self.sample(self.restrict(cell), rng)
        for ts in out:
            if len(ts) > self.lmax:
                raise ValueError(
                    f"{type(self).__name__} emitted {len(ts)} tokens > lmax={self.lmax}"
                )
        return out

    # -- provenance -----------------------------------------------------------------------------

    def params(self) -> dict[str, object]:
        """JSON-serialisable description for ``uns["bloom"]`` provenance."""
        uni: object
        if self.universe is None:
            uni = None
        elif isinstance(self.universe, FeatureList):
            uni = {
                "source": self.universe.source,
                "n_ids": len(self.universe),
                "kind": self.universe.kind,
            }
        else:
            uni = {"n_emb_idx": int(self.universe.size)}
        return {"strategy": self.name, "lmax": self.lmax, "universe": uni}

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v!r}" for k, v in self.params().items() if k != "strategy")
        return f"{type(self).__name__}({items})"


def _candidates(cell: CellFeatures) -> tuple[np.ndarray, np.ndarray | None, str | None]:
    """``(emb_idx, values_or_None, modality)`` of the populated modality of ``cell``."""
    mod = cell.modality
    if mod == "gene":
        return cell.detected_gene_emb_idx, cell.gene_values, "gene"
    if mod == "peak":
        return cell.detected_peak_emb_idx, cell.peak_values, "peak"
    return np.zeros(0, dtype=np.int32), None, None
