"""``prepare``: user AnnData(s) -> :class:`PreparedDataset` (mapping, detection, report, bundle I/O).

docs/data-schema.md sections 1 and 2; docs/architecture.md section 3. One AnnData per modality,
any non-empty subset: ``rna=`` (genes, section 1.1), ``atac=`` (peaks, section 1.2), or both.

Pipeline for one RNA AnnData:

1. value source: ``layer`` -> ``layers["counts"]`` -> ``raw.X`` -> ``X`` (recorded as
   ``value_source``); converted to CSR float32 with **sorted indices** (unsorted CSR rows changed
   tie-breaks in the original pipeline); negative values are an error;
2. gene identifiers (``var_names`` or ``var[gene_col]``) mapped onto reference rows with
   :meth:`bloom.reference.ReferenceTables.map_genes` (mouse symbol / ``ENSMUSG`` id / human
   ortholog) -> ``var["bloom_symbol" | "bloom_emb_idx" | "bloom_mapped_via"]``;
3. detection = ``X > 0`` on mapped columns (duplicate columns of one ``emb_idx`` are OR-ed) ->
   ``obs["bloom_n_genes_detected"]`` and the mapping report with its quality warnings;
4. ``uns["bloom"]`` provenance; :meth:`PreparedDataset.write` / :meth:`PreparedDataset.read`
   round-trip the bundle (``rna.h5ad`` + ``manifest.json``, ``schema_version`` ``"1.0"``).

Pipeline for one ATAC AnnData (:func:`prepare_atac`):

1. ``X`` -> CSR float32 (fragment / insertion counts; negative values are an error);
2. peak coordinates parsed from ``var_names`` or ``var[chrom, start, end]`` (:mod:`bloom.peaks`),
   non-canonical contigs dropped and counted;
3. matched onto the reference peak table exactly or by overlap (``peak_match`` ``auto`` = exact when
   >= 95 % of the parsed peaks match exactly, else overlap at ``min_overlap``);
4. the prepared ``X`` is the **reference-peak** matrix restricted to the hit peaks (OR over the user
   peaks mapped onto one reference peak; values = summed counts, detection = ``X > 0``), with
   ``var["bloom_emb_idx"]`` (never -1), ``var["n_user_peaks"]``, ``obs["bloom_n_peaks_detected"]``.

With both modalities the cells are aligned on ``obs_names`` (intersection, RNA order); the cells
present in one file only are dropped, or kept in :attr:`PreparedDataset.unmatched` when
``keep_unmatched`` is set (they are embedded separately, docs/data-schema.md section 4).

Input guards: a wrong ``species`` is detected from the identifiers (issue #33,
:class:`SpeciesMismatchError`, overridable with ``force_species``) and an input where **no** cell
has a detected mapped feature is an error (issue #34, :class:`NoMappedFeaturesError`) instead of an
all-NaN output.

Only ``X > 0`` enters the model: library size is not collected in v1.
"""

from __future__ import annotations

import hashlib
import json
import logging
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from bloom.peaks import (
    DEFAULT_MIN_OVERLAP,
    PEAK_MATCH_MODES,
    ParsedPeaks,
    detection_matrix,
    parse_peaks,
    peak_names,
)
from bloom.reference import (
    MAPPED_VIA_CATEGORIES,
    ReferenceTables,
    Species,
    mapping_report,
)
from bloom.tokens import CellFeatures

logger = logging.getLogger(__name__)

__all__ = [
    "LOW_MAPPING_THRESHOLD",
    "MIN_MEDIAN_DETECTED",
    "SCHEMA_VERSION",
    "SPECIES_MISMATCH_THRESHOLD",
    "ModalityName",
    "NoMappedFeaturesError",
    "PreparedDataset",
    "SchemaVersionError",
    "SpeciesMismatchError",
    "check_species",
    "prepare",
    "prepare_atac",
    "prepare_rna",
    "select_values",
]

SCHEMA_VERSION = "1.0"
"""``schema_version`` written to ``uns["bloom"]`` and ``manifest.json`` (docs/data-schema.md section 5)."""
MANIFEST_NAME = "manifest.json"
RNA_FILE = "rna.h5ad"
ATAC_FILE = "atac.h5ad"
MODALITY_FILES: dict[str, str] = {"rna": RNA_FILE, "atac": ATAC_FILE}
UNMATCHED_FILES: dict[str, str] = {"rna": "rna_only.h5ad", "atac": "atac_only.h5ad"}
SPECIES_MISMATCH_THRESHOLD = 0.50
"""Issue #33: error when more than this fraction of the gene ids look like the *other* species."""
LOW_MAPPING_THRESHOLD = 0.70
"""Warn when less than this fraction of the detected mass falls on mapped genes."""
MIN_MEDIAN_DETECTED = 200
"""Warn when the median number of detected mapped genes per cell is below this."""
NEGATIVE_VALUES_MESSAGE = (
    "{what} matrix contains negative values ({n_negative} entries, min {minimum:g}) in {source}: "
    "scaled/centered data cannot be binarized. Provide raw counts or a non-negative "
    "normalization (e.g. layers['counts'], raw.X, or --layer <name>)."
)

ModalityName = Literal["rna", "atac"]
_VAR_COLUMNS = ("bloom_symbol", "bloom_emb_idx", "bloom_mapped_via")
_ATAC_VAR_COLUMNS = ("bloom_emb_idx", "n_user_peaks")


class SchemaVersionError(ValueError):
    """A prepared bundle was written by an incompatible (higher major) schema version."""


class SpeciesMismatchError(ValueError):
    """The gene identifiers look like the other species than ``--species`` (issue #33)."""


class NoMappedFeaturesError(ValueError):
    """No cell has a detected feature that maps onto the reference (issue #34)."""


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------


def _schema_major(version: str) -> int:
    try:
        return int(str(version).split(".")[0])
    except ValueError as exc:
        raise SchemaVersionError(f"unparseable schema_version {version!r}") from exc


def check_schema_version(version: str, *, where: str) -> None:
    """Refuse bundles whose ``schema_version`` major is above ours (test P9)."""
    ours = _schema_major(SCHEMA_VERSION)
    theirs = _schema_major(version)
    if theirs > ours:
        raise SchemaVersionError(
            f"{where} has schema_version {version!r}; this bloom version reads schema "
            f"{ours}.x only (SCHEMA_VERSION={SCHEMA_VERSION}). Upgrade bloom or re-run "
            "`bloom prepare`."
        )
    if theirs < ours:
        logger.warning("%s has older schema_version %s (ours %s)", where, version, SCHEMA_VERSION)


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Hex sha256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _json_safe(obj: Any) -> Any:
    """Convert numpy scalars/arrays and drop ``None`` so the dict is h5ad- and JSON-writable."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list | tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _to_csr_float32(X: Any, *, source: str, what: str = "RNA") -> sp.csr_matrix:
    """Any matrix-like (dense, CSR, CSC, COO, backed dataset) -> CSR float32, sorted, no explicit zeros.

    Raises:
        ValueError: when the matrix holds negative values (docs/data-schema.md section 1.1).
    """
    if not sp.issparse(X):
        if hasattr(X, "to_memory"):  # anndata backed sparse dataset
            X = X.to_memory()
        elif not isinstance(X, np.ndarray):
            X = X[...]  # h5py dataset or array-like
    if sp.issparse(X):
        M = sp.csr_matrix(X, dtype=np.float32, copy=True)
    else:
        M = sp.csr_matrix(np.asarray(X, dtype=np.float32))
    if M.nnz and float(M.data.min()) < 0:
        n_negative = int((M.data < 0).sum())
        raise ValueError(
            NEGATIVE_VALUES_MESSAGE.format(
                what=what, n_negative=n_negative, minimum=float(M.data.min()), source=source
            )
        )
    if not np.isfinite(M.data).all():
        raise ValueError(f"{what} matrix in {source} contains NaN or inf values")
    M.eliminate_zeros()
    M.sort_indices()  # tie-breaks depend on sorted rows (AGENTS.md section 6)
    return M


def select_values(adata: AnnData, layer: str | None = None) -> tuple[Any, str, pd.DataFrame]:
    """Pick the value matrix by precedence ``layer`` -> ``layers["counts"]`` -> ``raw.X`` -> ``X``.

    Args:
        adata: user AnnData.
        layer: explicit layer name (``--layer``); must exist.

    Returns:
        ``(matrix, value_source, var)`` where ``value_source`` is one of ``"layers['<name>']"``,
        ``"raw.X"`` or ``"X"`` and ``var`` is the ``var`` frame the matrix columns belong to
        (``adata.raw.var`` when ``raw.X`` is used, since raw may hold more genes than ``var``).
    """
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(
                f"layer {layer!r} not found; available layers: {sorted(k for k in adata.layers.keys() if k)}"
            )
        return adata.layers[layer], f"layers[{layer!r}]", adata.var
    if "counts" in adata.layers:
        return adata.layers["counts"], "layers['counts']", adata.var
    if adata.raw is not None:
        return adata.raw.X, "raw.X", adata.raw.var
    return adata.X, "X", adata.var


def _gene_ids(var: pd.DataFrame, gene_col: str | None) -> np.ndarray:
    if gene_col is None:
        return np.asarray(var.index.astype(str), dtype=object)
    if gene_col not in var.columns:
        raise KeyError(
            f"gene_col {gene_col!r} is not a var column; available: {list(var.columns)[:20]}"
        )
    return np.asarray(var[gene_col].astype(str), dtype=object)


def _detection_matrix(X: sp.csr_matrix, emb_idx: np.ndarray) -> tuple[sp.csr_matrix, np.ndarray]:
    """``X > 0`` restricted to mapped columns and OR-ed per ``emb_idx``.

    Returns:
        ``(D, emb_unique)``: ``D`` CSR bool ``(n_cells, n_unique_emb)`` with sorted indices,
        ``emb_unique`` int32 ``(n_unique_emb,)`` strictly increasing column labels.
    """
    emb_idx = np.asarray(emb_idx, dtype=np.int64)
    mapped_cols = np.flatnonzero(emb_idx >= 0)
    emb_unique, inverse = np.unique(emb_idx[mapped_cols], return_inverse=True)
    n_cells = X.shape[0]
    if mapped_cols.size == 0:
        return sp.csr_matrix((n_cells, 0), dtype=bool), emb_unique.astype(np.int32)
    # column -> unique emb_idx incidence, (n_var, n_unique)
    P = sp.csr_matrix(
        (np.ones(mapped_cols.size, dtype=np.int8), (mapped_cols, inverse)),
        shape=(X.shape[1], emb_unique.size),
    )
    B = X.copy()
    B.data = np.ones_like(B.data, dtype=np.int8)  # X > 0 (explicit zeros already removed)
    D = (B @ P) > 0
    D = sp.csr_matrix(D, dtype=bool)
    D.sort_indices()
    return D, emb_unique.astype(np.int32)


# --------------------------------------------------------------------------------------------
# PreparedDataset
# --------------------------------------------------------------------------------------------


@dataclass
class PreparedDataset:
    """Prepared modalities + mapping reports (docs/data-schema.md section 2).

    Attributes:
        rna: prepared RNA AnnData (``X`` CSR float32 original values, ``var["bloom_*"]``,
            ``obs["bloom_n_genes_detected"]``, ``uns["bloom"]``) or ``None``.
        atac: prepared ATAC AnnData (``X`` CSR float32 reference-peak counts, cells x hit
            reference peaks; ``var["bloom_emb_idx"]``, ``var["n_user_peaks"]``,
            ``obs["bloom_n_peaks_detected"]``, ``uns["bloom"]``; docs/data-schema.md section 2.2)
            or ``None``.
        species: ``"mouse"`` or ``"human"``.
        reference_id: id of the reference tables the features were mapped onto.
        report: ``{"rna": {...}, "atac": {...}}`` mapping/quality report per modality present.
            RNA keys: ``n_var``, ``n_mapped``, ``mapping_rate``, ``detected_mass_mapped``,
            ``low_mapping``, ``n_obs``, ``median_detected``, ``n_empty_cells``, ``value_source``,
            ``unmapped_examples``, ``warnings``, plus the counters of
            :class:`bloom.reference.MappingResult`. ATAC keys: ``peak_match`` (mode used),
            ``peak_match_requested``, ``min_overlap``, ``exact_rate``, ``n_var`` (user peaks),
            ``n_valid``, ``n_unparsed``, ``dropped_contigs``, ``n_mapped`` (user peaks with a
            partner), ``mapping_rate``, ``n_ref_peaks_hit``, ``n_pairs``, ``detected_mass_mapped``,
            ``median_detected``, ``n_empty_cells``, ``unmapped_examples``, ``warnings``.
        source: ``{"rna": {"path": str | None, "sha256": str | None}, "atac": {...}}``.
        alignment: with both modalities, ``{"n_rna", "n_atac", "n_common", "n_rna_only",
            "n_atac_only", "keep_unmatched"}``; ``None`` for a single modality.
        unmatched: ``{"rna": AnnData, "atac": AnnData}`` prepared cells present in one file only,
            filled when ``keep_unmatched`` was set (embedded to ``embeddings_<mod>_only.h5ad``).
        created: ISO-8601 UTC timestamp.
    """

    rna: AnnData | None = None
    atac: AnnData | None = None
    species: str = "mouse"
    reference_id: str = ""
    report: dict[str, dict[str, Any]] = field(default_factory=dict)
    source: dict[str, dict[str, Any]] = field(default_factory=dict)
    created: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    schema_version: str = SCHEMA_VERSION
    alignment: dict[str, Any] | None = None
    unmatched: dict[str, AnnData] = field(default_factory=dict)

    # ---- basics ---------------------------------------------------------------------------

    @property
    def modalities(self) -> tuple[str, ...]:
        """Modalities present, in the fixed order ``("rna", "atac")``."""
        return tuple(m for m in ("rna", "atac") if getattr(self, m) is not None)

    def _adata(self, modality: ModalityName) -> AnnData:
        a = getattr(self, modality, None)
        if a is None:
            raise ValueError(f"modality {modality!r} is not present (have {self.modalities})")
        return a

    @property
    def n_obs(self) -> int:
        """Cells of the first modality present."""
        return int(self._adata(self.modalities[0]).n_obs) if self.modalities else 0

    @property
    def obs_names(self) -> pd.Index:
        return self._adata(self.modalities[0]).obs_names

    def __repr__(self) -> str:
        parts = []
        for m in self.modalities:
            a = getattr(self, m)
            rep = self.report.get(m, {})
            parts.append(
                f"{m}: {a.n_obs} cells x {a.n_vars} features, mapped {rep.get('n_mapped')}"
            )
        if self.unmatched:
            parts.append(
                "unmatched " + ", ".join(f"{m}: {a.n_obs}" for m, a in self.unmatched.items())
            )
        return f"PreparedDataset(species={self.species!r}, {'; '.join(parts) or 'empty'})"

    def single(self, modality: ModalityName, *, unmatched: bool = False) -> PreparedDataset:
        """A one-modality view of this dataset (``unmatched=True``: the cells only in that file)."""
        a = self.unmatched[modality] if unmatched else self._adata(modality)
        return PreparedDataset(
            **{modality: a},
            species=self.species,
            reference_id=self.reference_id,
            report={modality: self.report.get(modality, {})},
            source={modality: self.source.get(modality, {})},
            created=self.created,
            schema_version=self.schema_version,
        )

    # ---- per-cell access (Embed track) ----------------------------------------------------

    def col_to_emb_idx(self, modality: ModalityName = "rna") -> np.ndarray:
        """``var["bloom_emb_idx"]`` as int32 ``(n_var,)`` (``-1`` = unmapped)."""
        return self._adata(modality).var["bloom_emb_idx"].to_numpy().astype(np.int32)

    def cell_features(self, i: int, modality: ModalityName = "rna") -> CellFeatures:
        """Detected features of cell ``i`` in ``emb_idx`` space (sorted, with aligned values).

        For RNA this is ``CellFeatures.genes`` built from the CSR row of ``X`` (``X > 0`` on mapped
        columns; duplicates OR-ed, keeping the largest value); for ATAC ``CellFeatures.peaks`` with
        the summed counts as ``peak_values``. An empty cell gives empty arrays.
        """
        a = self._adata(modality)
        X = a.X
        if not isinstance(X, sp.csr_matrix | sp.csr_array):
            raise TypeError("prepared X must be CSR (use prepare() or PreparedDataset.read())")
        if not 0 <= i < a.n_obs:
            raise IndexError(f"cell index {i} out of range [0, {a.n_obs})")
        lo, hi = X.indptr[i], X.indptr[i + 1]
        kind = "gene" if modality == "rna" else "peak"
        return CellFeatures.from_csr_row(
            i, X.indices[lo:hi], X.data[lo:hi], self.col_to_emb_idx(modality), kind
        )

    def cells(self, modality: ModalityName = "rna") -> Iterator[CellFeatures]:
        """Iterate :meth:`cell_features` over all cells in ``obs`` order."""
        for i in range(self._adata(modality).n_obs):
            yield self.cell_features(i, modality)

    def detection_matrix(self, modality: ModalityName = "rna") -> tuple[sp.csr_matrix, np.ndarray]:
        """Binary detection matrix ``(n_cells, n_unique_mapped_emb_idx)`` and its column labels.

        Independent of the value type (counts / log-CPM / positive floats): only ``X > 0`` and the
        mapping enter (tests P1, P4, P8).
        """
        a = self._adata(modality)
        return _detection_matrix(sp.csr_matrix(a.X), self.col_to_emb_idx(modality))

    # ---- report ---------------------------------------------------------------------------

    def report_text(self) -> str:
        """Human-readable report (what ``bloom prepare`` prints)."""
        lines = [f"species {self.species}; reference {self.reference_id}"]
        for m in self.modalities:
            r = self.report[m]
            a = getattr(self, m)
            if m == "rna":
                lines.append(
                    f"[rna] {a.n_obs} cells x {a.n_vars} features; value source {r['value_source']}"
                )
                lines.append(
                    f"  mapped {r['n_mapped']}/{r['n_var']} ids ({100 * r['mapping_rate']:.1f}%); "
                    f"detected mass mapped {100 * (r['detected_mass_mapped'] or 0):.1f}%; "
                    f"via exact {r.get('n_exact', 0)}, gene_id {r.get('n_gene_id', 0)}, "
                    f"ortholog {r.get('n_ortholog', 0)}"
                )
            else:
                lines.append(
                    f"[atac] {a.n_obs} cells x {r['n_var']} user peaks ({r.get('source')}) -> "
                    f"{a.n_vars} reference peaks hit; peak match {r['peak_match']}"
                    + (f" (min_overlap {r['min_overlap']})" if r["peak_match"] == "overlap" else "")
                    + (
                        f"; exact rate {100 * r['exact_rate']:.1f}%"
                        if r.get("exact_rate") is not None
                        else ""
                    )
                )
                lines.append(
                    f"  mapped {r['n_mapped']}/{r['n_valid']} parsed peaks "
                    f"({100 * r['mapping_rate']:.1f}%); detected mass mapped "
                    f"{100 * (r['detected_mass_mapped'] or 0):.1f}%; unparsed ids {r['n_unparsed']}; "
                    f"dropped on non-canonical contigs {r['n_dropped_contigs']}"
                )
            lines.append(
                f"  detected mapped features per cell: median {r['median_detected']}, "
                f"min {r['min_detected']}, max {r['max_detected']}; "
                f"empty cells {r['n_empty_cells']}"
            )
            if r.get("unmapped_examples"):
                lines.append(
                    f"  most frequent unmapped ids: {', '.join(r['unmapped_examples'][:10])}"
                )
            for w in r.get("warnings", []):
                lines.append(f"  WARNING: {w}")
        if self.alignment:
            al = self.alignment
            lines.append(
                f"[alignment] {al['n_common']} cells in both files (rna {al['n_rna']}, atac "
                f"{al['n_atac']}); rna-only {al['n_rna_only']}, atac-only {al['n_atac_only']}"
                + (" kept (keep_unmatched)" if al.get("keep_unmatched") else " dropped")
            )
        return "\n".join(lines)

    # ---- bundle I/O -----------------------------------------------------------------------

    def manifest(self) -> dict[str, Any]:
        """``manifest.json`` content (docs/data-schema.md section 2)."""
        from bloom import __version__

        return _json_safe({
            "schema_version": self.schema_version,
            "package_version": __version__,
            "reference_id": self.reference_id,
            "species": self.species,
            "created": self.created,
            "modalities": list(self.modalities),
            "files": {m: MODALITY_FILES[m] for m in self.modalities},
            "unmatched_files": {m: UNMATCHED_FILES[m] for m in self.unmatched},
            "alignment": self.alignment,
            "source": self.source,
            "mapping": self.report,
        })

    def write(self, directory: str | Path, *, overwrite: bool = True) -> Path:
        """Write the bundle ``<dir>/{manifest.json, rna.h5ad[, atac.h5ad]}``; returns ``<dir>``.

        ``X`` is written as CSR float32 (original values); the user's ``obs``/``var``/``uns`` are
        kept alongside the ``bloom_*`` columns.
        """
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        if not self.modalities:
            raise ValueError("nothing to write: no modality present")
        todo = [(getattr(self, m), d / MODALITY_FILES[m]) for m in self.modalities]
        todo += [(a, d / UNMATCHED_FILES[m]) for m, a in self.unmatched.items()]
        for a, path in todo:
            if path.exists() and not overwrite:
                raise FileExistsError(path)
            a.uns["bloom"] = _json_safe(a.uns.get("bloom", {}))
            a.write_h5ad(path)
        (d / MANIFEST_NAME).write_text(json.dumps(self.manifest(), indent=2) + "\n")
        logger.info("wrote prepared bundle %s (%s)", d, ", ".join(self.modalities))
        return d

    @classmethod
    def read(cls, directory: str | Path) -> PreparedDataset:
        """Read a bundle written by :meth:`write`; refuses a higher schema major (test P9)."""
        import anndata as ad

        d = Path(directory)
        mpath = d / MANIFEST_NAME
        if not mpath.exists():
            raise FileNotFoundError(f"{d} is not a prepared bundle ({MANIFEST_NAME} missing)")
        man = json.loads(mpath.read_text())
        check_schema_version(str(man.get("schema_version", "0")), where=str(mpath))
        files = man.get("files") or {m: MODALITY_FILES[m] for m in man.get("modalities", [])}

        def _load(m: str, fname: str) -> AnnData:
            if m not in ("rna", "atac"):
                raise ValueError(f"unknown modality {m!r} in {mpath}")
            path = d / fname
            if not path.exists():
                raise FileNotFoundError(f"{mpath} lists {fname} but {path} is missing")
            a = ad.read_h5ad(path)
            uns = dict(a.uns.get("bloom", {}))
            check_schema_version(str(uns.get("schema_version", "0")), where=str(path))
            required = _VAR_COLUMNS if m == "rna" else _ATAC_VAR_COLUMNS
            missing = [c for c in required if c not in a.var.columns]
            if missing:
                raise ValueError(f"{path} lacks var columns {missing}")
            a.X = _to_csr_float32(a.X, source=f"{path}:X")
            a.var["bloom_emb_idx"] = a.var["bloom_emb_idx"].astype(np.int32)
            if m == "rna":
                a.var["bloom_mapped_via"] = pd.Categorical(
                    a.var["bloom_mapped_via"].astype(str), categories=list(MAPPED_VIA_CATEGORIES)
                )
            else:
                a.var["n_user_peaks"] = a.var["n_user_peaks"].astype(np.int32)
            a.uns["bloom"] = uns
            return a

        loaded = {m: _load(m, fname) for m, fname in files.items()}
        unmatched = {m: _load(m, fname) for m, fname in (man.get("unmatched_files") or {}).items()}
        mapping = man.get("mapping", {})
        return cls(
            rna=loaded.get("rna"),
            atac=loaded.get("atac"),
            species=str(man.get("species", "")),
            reference_id=str(man.get("reference_id", "")),
            report={m: dict(v) for m, v in mapping.items()},
            source={m: dict(v) for m, v in man.get("source", {}).items()},
            created=str(man.get("created", "")),
            schema_version=str(man.get("schema_version", SCHEMA_VERSION)),
            alignment=dict(man["alignment"]) if man.get("alignment") else None,
            unmatched=unmatched,
        )


# --------------------------------------------------------------------------------------------
# prepare
# --------------------------------------------------------------------------------------------


def _load_anndata(x: AnnData | str | Path) -> tuple[AnnData, str | None]:
    import anndata as ad

    if isinstance(x, AnnData):
        a = x
        path = getattr(getattr(a, "file", None), "filename", None)
        if a.isbacked:
            a = a.to_memory()
        return a, (str(path) if path else None)
    p = Path(x)
    if not p.exists():
        raise FileNotFoundError(p)
    return ad.read_h5ad(p), str(p)


# --------------------------------------------------------------------------------------------
# input guards (#33, #34)
# --------------------------------------------------------------------------------------------


def _is_hgnc_style(ids: np.ndarray) -> np.ndarray:
    """HGNC-style identifiers: all letters upper-case and at least one letter (``PVALB``, ``ENSG...``)."""
    s = pd.Series(ids, dtype=object).astype(str).str.strip()
    has_alpha = s.str.contains(r"[A-Za-z]", regex=True)
    return (has_alpha & (s == s.str.upper())).to_numpy()


def check_species(
    ids: np.ndarray | Sequence[str],
    species: str,
    reference: ReferenceTables,
    *,
    threshold: float = SPECIES_MISMATCH_THRESHOLD,
) -> dict[str, Any]:
    """Detect ``--species`` contradicting the gene identifiers (issue #33).

    ``species="human"``: more than ``threshold`` of the ids match a **mouse** symbol exactly
    (case-sensitive, before normalisation) or carry the ``ENSMUSG`` prefix.
    ``species="mouse"``: more than ``threshold`` of the ids are HGNC-style (all upper-case, e.g.
    ``PVALB``) without matching a mouse symbol exactly, or carry the ``ENSG`` prefix.

    Returns:
        ``{"species", "n_ids", "n_other_species_like", "fraction", "mismatch", "suggested"}``.

    Raises:
        SpeciesMismatchError: when ``mismatch`` is true (the message names ``--force-species``).
    """
    arr = np.asarray([str(x).strip() for x in ids], dtype=object)
    n = int(arr.size)
    if n == 0:
        return {
            "species": species,
            "n_ids": 0,
            "n_other_species_like": 0,
            "fraction": 0.0,
            "mismatch": False,
            "suggested": species,
        }
    mouse_exact = np.fromiter((x in reference.symbol_to_emb_idx for x in arr), dtype=bool, count=n)
    ensmusg = pd.Series(arr).str.startswith("ENSMUSG").to_numpy()
    ensg = pd.Series(arr).str.startswith("ENSG0").to_numpy()
    if species == "human":
        other = mouse_exact | ensmusg
        suggested = "mouse"
        what = "match mouse symbols / ENSMUSG ids exactly"
    else:
        other = (_is_hgnc_style(arr) & ~mouse_exact & ~ensmusg) | ensg
        suggested = "human"
        what = "are HGNC-style (all upper-case) / ENSG ids"
    frac = float(other.sum() / n)
    info = {
        "species": species,
        "n_ids": n,
        "n_other_species_like": int(other.sum()),
        "fraction": frac,
        "mismatch": bool(frac > threshold),
        "suggested": suggested if frac > threshold else species,
    }
    if info["mismatch"]:
        examples = arr[other][:8].tolist()
        raise SpeciesMismatchError(
            f"--species {species} but {100 * frac:.1f}% of the {n} gene identifiers {what} "
            f"(e.g. {examples}); this looks like {suggested} data. Re-run with --species "
            f"{suggested}, or pass --force-species to map with --species {species} anyway."
        )
    return info


def _check_any_mapped(
    n_detected: np.ndarray,
    *,
    modality: str,
    unmapped_examples: Sequence[str],
    n_ids: int,
    n_mapped: int,
) -> None:
    """Issue #34: no cell with a detected mapped feature is an error, not an all-NaN output."""
    if n_detected.size and int((n_detected > 0).sum()) == 0:
        what = "gene" if modality == "rna" else "peak"
        ex = ", ".join(str(x) for x in list(unmapped_examples)[:10]) or "<none>"
        raise NoMappedFeaturesError(
            f"[{modality}] no cell has a detected {what} that maps onto the reference "
            f"({n_mapped}/{n_ids} identifiers mapped); embeddings would be all NaN. Most frequent "
            f"unmapped identifiers: {ex}. "
            + (
                "Typical causes: wrong --species, Ensembl ids in a symbol column (--gene-col), "
                "a non-mouse/human organism, or an all-zero matrix."
                if modality == "rna"
                else "Typical causes: peaks not in mm10 coordinates, identifiers that are not "
                "chr:start-end (use var[chrom,start,end]), or --peak-match exact on a different "
                "peak set (use --peak-match overlap)."
            )
        )


# --------------------------------------------------------------------------------------------
# prepare: workers
# --------------------------------------------------------------------------------------------


def _check_unique_obs(adata: AnnData) -> None:
    """obs_names must be unique: they align modalities and index the output (P6)."""
    if not adata.obs_names.is_unique:
        dups = adata.obs_names[adata.obs_names.duplicated()].unique()[:5].tolist()
        raise ValueError(
            f"obs_names must be unique; {int(adata.obs_names.duplicated().sum())} duplicate(s), "
            f"e.g. {dups}. Run `adata.obs_names_make_unique()` first."
        )


def _source_of(path: str | None, source_sha256: bool) -> dict[str, Any]:
    return _json_safe({
        "path": path,
        "sha256": sha256_file(path) if (path and source_sha256 and Path(path).exists()) else None,
    })


def prepare_rna(
    rna: AnnData | str | Path,
    *,
    species: Species,
    reference: ReferenceTables,
    gene_col: str | None = None,
    layer: str | None = None,
    source_sha256: bool = True,
    force_species: bool = False,
) -> tuple[AnnData, dict[str, Any], dict[str, Any]]:
    """Prepare one RNA AnnData; returns ``(prepared AnnData, report, source)``.

    See the module docstring for the steps. Prefer :func:`prepare` (builds the
    :class:`PreparedDataset`); this function is the modality worker.

    Raises:
        SpeciesMismatchError: the identifiers look like the other species (#33) and
            ``force_species`` is false.
        NoMappedFeaturesError: no cell has a detected mapped gene (#34).
    """
    if species not in ("mouse", "human"):
        raise ValueError(f"species must be 'mouse' or 'human', got {species!r}")
    adata, path = _load_anndata(rna)
    _check_unique_obs(adata)

    # value source (P3) -> CSR float32 sorted (P4), non-negative (P2)
    values, value_source, var = select_values(adata, layer)
    X = _to_csr_float32(values, source=value_source)  # (n_cells, n_var)
    if X.shape != (adata.n_obs, len(var)):
        raise ValueError(
            f"{value_source} has shape {X.shape} but obs x var is {(adata.n_obs, len(var))}"
        )

    # species sanity (#33) then gene mapping (R1-R5 live in reference.py)
    ids = _gene_ids(var, gene_col)
    species_check: dict[str, Any] | None = None
    if force_species:
        species_check = {"species": species, "forced": True}
    else:
        species_check = check_species(ids, species, reference)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = reference.map_genes(ids, species=species)
    mapping_warnings = [str(w.message) for w in caught]
    for w in caught:  # re-emit so callers still see them
        warnings.warn(str(w.message), category=w.category, stacklevel=2)
    rep = mapping_report(result, X, low_rate_threshold=LOW_MAPPING_THRESHOLD)

    # detection (P1, P7)
    D, emb_unique = _detection_matrix(X, result.emb_idx)
    n_detected = np.asarray(D.sum(axis=1)).ravel().astype(np.int32)  # (n_cells,)
    _check_any_mapped(
        n_detected,
        modality="rna",
        unmapped_examples=rep["unmapped_examples"],
        n_ids=rep["n_var"],
        n_mapped=rep["n_mapped"],
    )
    n_empty = int((n_detected == 0).sum())
    median_detected = float(np.median(n_detected)) if n_detected.size else 0.0

    out_warnings = list(mapping_warnings)
    if rep["low_mapping"]:
        rate = rep["detected_mass_mapped"]
        msg = (
            f"only {100 * rate:.1f}% of the detected gene mass maps onto the reference "
            f"(threshold {100 * LOW_MAPPING_THRESHOLD:.0f}%); most frequent unmapped ids: "
            f"{rep['unmapped_examples'][:10]}. Typical causes: Ensembl ids in a symbol column, "
            "wrong --species, lower-cased symbols, or --gene-col needed."
        )
        out_warnings.append(msg)
    if n_detected.size and median_detected < MIN_MEDIAN_DETECTED:
        out_warnings.append(
            f"median detected mapped genes per cell is {median_detected:.0f} (< "
            f"{MIN_MEDIAN_DETECTED}); views will be short relative to training"
        )
    if n_empty:
        out_warnings.append(
            f"{n_empty} cell(s) have no detected mapped genes; they are kept and their "
            "embeddings will be NaN rows"
        )
    for msg in out_warnings[len(mapping_warnings) :]:
        warnings.warn(msg, stacklevel=2)

    # assemble the prepared AnnData
    var_out = var.copy()
    var_out["bloom_symbol"] = pd.Series(result.symbol, index=var_out.index, dtype=object).astype(
        str
    )
    var_out["bloom_emb_idx"] = result.emb_idx.astype(np.int32)
    var_out["bloom_mapped_via"] = pd.Categorical(
        np.asarray(result.mapped_via.astype(str)), categories=list(MAPPED_VIA_CATEGORIES)
    )
    obs_out = adata.obs.copy()
    obs_out["bloom_n_genes_detected"] = n_detected
    report: dict[str, Any] = {
        **rep,
        "n_obs": int(adata.n_obs),
        "value_source": value_source,
        "gene_col": gene_col,
        "median_detected": median_detected,
        "min_detected": int(n_detected.min()) if n_detected.size else 0,
        "max_detected": int(n_detected.max()) if n_detected.size else 0,
        "n_empty_cells": n_empty,
        "n_unique_emb_idx": int(emb_unique.size),
        "species_check": species_check,
        "warnings": out_warnings,
    }
    report = _json_safe(report)
    source = _source_of(path, source_sha256)
    uns = dict(adata.uns)
    uns["bloom"] = _json_safe({
        "schema_version": SCHEMA_VERSION,
        "reference_id": reference.reference_id,
        "species": species,
        "modality": "rna",
        "value_source": value_source,
        "gene_col": gene_col,
        "mapping": report,
        "source": source,
    })
    prepared = AnnData(X=X, obs=obs_out, var=var_out, uns=uns)
    return prepared, report, source


def _parse_user_peaks(var: pd.DataFrame) -> ParsedPeaks:
    parsed = parse_peaks(var)
    if parsed.n_valid == 0:
        ex = ", ".join(str(x) for x in parsed.ids[:5])
        raise ValueError(
            f"no ATAC var identifier parses as a peak coordinate ({len(parsed)} ids, "
            f"{parsed.n_unparsed} unparsable, {parsed.n_dropped} on non-canonical contigs; e.g. "
            f"{ex}). Expected var_names like chr1:3000000-3000500 (also chr1-...-... / "
            "chr1_..._...) or var columns chrom,start,end in mm10 coordinates."
        )
    return parsed


def _collapse_duplicate_peaks(
    X: sp.csr_matrix, parsed: ParsedPeaks
) -> tuple[sp.csr_matrix, dict[str, Any]]:
    """Issue #48: var rows with identical (valid) coordinates are one feature.

    Their values are OR-ed into the first row of each group (detection = any ``> 0``, value = the
    largest, mirroring R3 for duplicate gene symbols); the other rows become empty columns, so
    ``parsed`` / the matches stay aligned with ``X`` and nothing is summed twice.

    Returns:
        ``(X, info)`` with ``info = {n_duplicate_rows, n_duplicate_coordinates, examples}``.
    """
    info: dict[str, Any] = {"n_duplicate_rows": 0, "n_duplicate_coordinates": 0, "examples": []}
    valid = np.flatnonzero(parsed.valid)
    if valid.size < 2:
        return X, info
    keys = pd.DataFrame({
        "chrom": parsed.chrom[valid],
        "start": parsed.start[valid],
        "end": parsed.end[valid],
    })
    dup_mask = keys.duplicated(keep="first").to_numpy()
    if not dup_mask.any():
        return X, info
    group = keys.groupby(["chrom", "start", "end"], sort=False).ngroup().to_numpy()
    _, first_pos = np.unique(
        group, return_index=True
    )  # group ids are 0..G-1 in order of appearance
    first_of_group = valid[first_pos]  # var column of each group's first occurrence
    remap = np.arange(X.shape[1], dtype=np.int64)
    remap[valid[dup_mask]] = first_of_group[group[dup_mask]]
    coo = X.tocoo()
    cols = remap[coo.col]
    key = coo.row.astype(np.int64) * X.shape[1] + cols
    order = np.lexsort((-coo.data, key))  # largest value first within a (row, col) group
    key_sorted = key[order]
    keep = np.ones(order.size, dtype=bool)
    keep[1:] = key_sorted[1:] != key_sorted[:-1]
    sel = order[keep]
    out = sp.csr_matrix((coo.data[sel], (coo.row[sel], cols[sel])), shape=X.shape, dtype=np.float32)
    out.sort_indices()
    dup_rows = valid[dup_mask]
    info = {
        "n_duplicate_rows": int(dup_rows.size),
        "n_duplicate_coordinates": int(np.unique(group[dup_mask]).size),
        "examples": [str(x) for x in parsed.names()[dup_rows[:5]]],
    }
    return out, info


def prepare_atac(
    atac: AnnData | str | Path,
    *,
    reference: ReferenceTables,
    peak_match: str = "auto",
    min_overlap: float = DEFAULT_MIN_OVERLAP,
    species: Species = "mouse",
    source_sha256: bool = True,
    _parsed_override: ParsedPeaks | None = None,
) -> tuple[AnnData, dict[str, Any], dict[str, Any]]:
    """Prepare one ATAC AnnData; returns ``(prepared AnnData, report, source)``.

    Args:
        atac: cells x peaks AnnData (counts in ``X``, binarised downstream) or path.
        reference: loaded tables with the peak table available (``bloom download --what peaks``).
        peak_match: ``"auto"`` (exact when >= 95 % of the parsed peaks match exactly, else
            overlap), ``"exact"`` or ``"overlap"`` (docs/data-schema.md section 1.2).
        min_overlap: overlap mode threshold, fraction of the 501-bp reference peak covered.
        species: recorded only; peak coordinates must be mm10 regardless.
        source_sha256: hash the input file for provenance.
        _parsed_override: pre-parsed coordinates (tests / the G6 shifted-coordinate study).

    Raises:
        ValueError: negative values, duplicate ``obs_names``, no parsable peak, bad ``peak_match``.
        NoMappedFeaturesError: no cell has a detected mapped peak (#34).
        FileNotFoundError: peak table not downloaded.
    """
    if peak_match not in PEAK_MATCH_MODES:
        raise ValueError(f"peak_match must be one of {PEAK_MATCH_MODES}, got {peak_match!r}")
    if not 0.0 < float(min_overlap) <= 1.0:
        raise ValueError(f"min_overlap must be in (0, 1], got {min_overlap}")
    adata, path = _load_anndata(atac)
    _check_unique_obs(adata)
    X = _to_csr_float32(adata.X, source="X", what="ATAC")  # (n_cells, n_user_peaks)

    parsed = _parsed_override if _parsed_override is not None else _parse_user_peaks(adata.var)
    if len(parsed) != adata.n_vars:
        raise ValueError(f"parsed {len(parsed)} peaks for {adata.n_vars} var rows")
    # issue #48: duplicate coordinates are one feature -> OR-ed (largest value kept), like R3 for genes
    X, dup_info = _collapse_duplicate_peaks(X, parsed)
    index = reference.peak_index  # loads peaks.parquet (72 MB) on first use
    matches, info = index.match(parsed, mode=peak_match, min_overlap=min_overlap)

    # reference-peak matrix restricted to hit peaks (section 2.2)
    M, ref_emb, n_user_per_ref = detection_matrix(X, matches)  # (n_cells, k)
    n_detected = np.diff(M.indptr).astype(np.int32)  # (n_cells,) nnz per row == detected peaks
    user_mapped = np.zeros(len(parsed), dtype=bool)
    user_mapped[matches.user_idx] = True
    unmapped_valid = parsed.valid & ~user_mapped
    per_col = np.diff(X.tocsc().indptr)  # detected entries per user column
    total_mass = int(per_col.sum())
    detected_mass_mapped = float(per_col[user_mapped].sum() / total_mass) if total_mass else 0.0
    # most frequent unmapped parsed peaks (by number of cells detecting them)
    unm_idx = np.flatnonzero(unmapped_valid)
    unm_idx = unm_idx[np.argsort(-per_col[unm_idx], kind="stable")][:10]
    unmapped_examples = [str(x) for x in parsed.ids[unm_idx]]
    n_valid = parsed.n_valid
    n_mapped = matches.n_user_mapped
    _check_any_mapped(
        n_detected,
        modality="atac",
        unmapped_examples=unmapped_examples,
        n_ids=len(parsed),
        n_mapped=n_mapped,
    )
    n_empty = int((n_detected == 0).sum())
    median_detected = float(np.median(n_detected)) if n_detected.size else 0.0
    mapping_rate = float(n_mapped / n_valid) if n_valid else 0.0

    out_warnings: list[str] = []
    if dup_info["n_duplicate_rows"]:
        out_warnings.append(
            f"{dup_info['n_duplicate_rows']} var row(s) repeat {dup_info['n_duplicate_coordinates']} "
            f"peak coordinate(s) already present, e.g. {dup_info['examples']}; duplicates are "
            "OR-ed (detection) and the largest value is kept, as for duplicate gene symbols"
        )
    if parsed.n_dropped:
        top = dict(sorted(parsed.dropped_contigs.items(), key=lambda kv: -kv[1])[:5])
        out_warnings.append(
            f"{parsed.n_dropped} peak(s) on {len(parsed.dropped_contigs)} non-canonical contig(s) "
            f"dropped (chr1..19, X, Y, M kept), e.g. {top}"
        )
    if parsed.n_unparsed:
        out_warnings.append(
            f"{parsed.n_unparsed} var identifier(s) are not peak coordinates and were ignored"
        )
    if detected_mass_mapped < LOW_MAPPING_THRESHOLD:
        out_warnings.append(
            f"only {100 * detected_mass_mapped:.1f}% of the detected peak mass maps onto the "
            f"reference peaks ({info['peak_match']} matching; threshold "
            f"{100 * LOW_MAPPING_THRESHOLD:.0f}%); most frequent unmapped peaks: "
            f"{unmapped_examples[:5]}. Typical causes: a different peak caller with "
            "--peak-match exact (use overlap), non-mm10 coordinates, or very wide peaks."
        )
    if n_detected.size and median_detected < MIN_MEDIAN_DETECTED:
        out_warnings.append(
            f"median detected mapped peaks per cell is {median_detected:.0f} (< "
            f"{MIN_MEDIAN_DETECTED}); views will be short relative to training"
        )
    if n_empty:
        out_warnings.append(
            f"{n_empty} cell(s) have no detected mapped peaks; they are kept and their "
            "embeddings will be NaN rows"
        )
    for msg in out_warnings:
        warnings.warn(msg, stacklevel=2)

    var_out = pd.DataFrame(index=pd.Index(reference.peak_name_of(ref_emb).astype(str), name=None))
    var_out["bloom_emb_idx"] = ref_emb.astype(np.int32)
    var_out["n_user_peaks"] = n_user_per_ref.astype(np.int32)
    obs_out = adata.obs.copy()
    obs_out["bloom_n_peaks_detected"] = n_detected
    report: dict[str, Any] = {
        **parsed.report(),
        **matches.report(),
        **info,
        "n_obs": int(adata.n_obs),
        "n_var": len(parsed),
        "n_duplicate_peaks": dup_info["n_duplicate_rows"],
        "n_mapped": n_mapped,
        "mapping_rate": mapping_rate,
        "detected_mass_mapped": detected_mass_mapped,
        "low_mapping": bool(detected_mass_mapped < LOW_MAPPING_THRESHOLD),
        "unmapped_examples": unmapped_examples,
        "value_source": "X",
        "median_detected": median_detected,
        "min_detected": int(n_detected.min()) if n_detected.size else 0,
        "max_detected": int(n_detected.max()) if n_detected.size else 0,
        "n_empty_cells": n_empty,
        "warnings": out_warnings,
    }
    report = _json_safe(report)
    source = _source_of(path, source_sha256)
    uns = dict(adata.uns)
    uns["bloom"] = _json_safe({
        "schema_version": SCHEMA_VERSION,
        "reference_id": reference.reference_id,
        "species": species,
        "modality": "atac",
        "peak_match": info["peak_match"],
        "peak_match_requested": info["peak_match_requested"],
        "min_overlap": info["min_overlap"] if info["peak_match"] == "overlap" else None,
        "exact_rate": info.get("exact_rate"),
        "n_user_peaks": len(parsed),
        "n_user_peaks_valid": n_valid,
        "n_user_peaks_mapped": n_mapped,
        "n_ref_peaks_hit": int(ref_emb.size),
        "dropped_contigs": parsed.dropped_contigs,
        "n_unparsed": parsed.n_unparsed,
        "mapping": report,
        "source": source,
    })
    prepared = AnnData(X=M, obs=obs_out, var=var_out, uns=uns)
    return prepared, report, source


def _align(
    rna: AnnData, atac: AnnData, *, keep_unmatched: bool
) -> tuple[AnnData, AnnData, dict[str, Any], dict[str, AnnData]]:
    """Intersect the two ``obs_names`` sets (RNA order); remainders kept only with ``keep_unmatched``."""
    common = rna.obs_names[rna.obs_names.isin(atac.obs_names)]
    n_rna_only = int(rna.n_obs - common.size)
    n_atac_only = int(atac.n_obs - common.size)
    if common.size == 0:
        raise ValueError(
            "the RNA and ATAC files share no obs_names (cells are aligned by name): rna e.g. "
            f"{rna.obs_names[:5].tolist()}, atac e.g. {atac.obs_names[:5].tolist()}. Rename the "
            "cells so that the same cell has the same obs_name in both files, or embed the files "
            "separately."
        )
    unmatched: dict[str, AnnData] = {}
    if keep_unmatched:
        if n_rna_only:
            unmatched["rna"] = rna[~rna.obs_names.isin(common)].copy()
        if n_atac_only:
            unmatched["atac"] = atac[~atac.obs_names.isin(common)].copy()
    elif n_rna_only or n_atac_only:
        warnings.warn(
            f"{n_rna_only} RNA-only and {n_atac_only} ATAC-only cell(s) are dropped from the "
            f"combined dataset ({common.size} cells in both files); pass keep_unmatched=True / "
            "--keep-unmatched to embed them separately",
            stacklevel=3,
        )
    alignment = {
        "n_rna": int(rna.n_obs),
        "n_atac": int(atac.n_obs),
        "n_common": int(common.size),
        "n_rna_only": n_rna_only,
        "n_atac_only": n_atac_only,
        "keep_unmatched": bool(keep_unmatched),
    }
    rna_c = rna[common].copy() if common.size != rna.n_obs else rna
    atac_c = atac[common].copy() if not atac.obs_names.equals(common) else atac
    for a in (rna_c, atac_c):
        a.X = sp.csr_matrix(a.X)
        a.X.sort_indices()
    return rna_c, atac_c, alignment, unmatched


def prepare(
    rna: AnnData | str | Path | None = None,
    atac: AnnData | str | Path | None = None,
    *,
    species: Species,
    reference: ReferenceTables | None = None,
    gene_col: str | None = None,
    layer: str | None = None,
    peak_match: str = "auto",
    min_overlap: float = DEFAULT_MIN_OVERLAP,
    keep_unmatched: bool = False,
    force_species: bool = False,
) -> PreparedDataset:
    """Map user AnnData(s) onto the reference tables and build a :class:`PreparedDataset`.

    Args:
        rna: RNA AnnData (cells x genes) or path to an ``.h5ad``; dense, CSR, CSC or backed.
        atac: ATAC AnnData (cells x peaks, mm10 coordinates in ``var_names`` or
            ``var[chrom,start,end]``) or path. Either or both of ``rna`` / ``atac``.
        species: ``"mouse"`` (MGI symbols / ``ENSMUSG`` ids) or ``"human"`` (HGNC symbols /
            ``ENSG`` ids through the shipped one-to-one ortholog table). ATAC peaks are mm10 in
            either case, so ``"human"`` with an ATAC file is refused.
        reference: loaded :class:`~bloom.reference.ReferenceTables`; ``None`` loads the default
            release layout under ``$BLOOM_HOME`` (``ReferenceTables.from_dir()``).
        gene_col: ``var`` column holding the gene identifiers; default ``var_names``.
        layer: explicit RNA layer to take values from; default precedence
            ``layers["counts"]`` -> ``raw.X`` -> ``X``.
        peak_match: ``"auto"`` | ``"exact"`` | ``"overlap"`` (docs/data-schema.md section 1.2).
        min_overlap: overlap threshold as a fraction of the reference peak (default 0.5).
        keep_unmatched: with both files, keep the cells present in one file only in
            :attr:`PreparedDataset.unmatched` (else they are dropped with a warning).
        force_species: skip the species-mismatch guard (#33).

    Returns:
        :class:`PreparedDataset` with the prepared modalities and ``.report`` per modality.

    Raises:
        ValueError: negative values, duplicate ``obs_names``, bad species, nothing given, disjoint
            ``obs_names``, no parsable peak; :class:`SpeciesMismatchError` (#33) and
            :class:`NoMappedFeaturesError` (#34) are subclasses.
        KeyError: ``layer`` / ``gene_col`` absent.
        FileNotFoundError: peak table not downloaded (``bloom download --what peaks``).
    """
    if rna is None and atac is None:
        raise ValueError("prepare() needs at least one of rna= or atac=")
    if species not in ("mouse", "human"):
        raise ValueError(f"species must be 'mouse' or 'human', got {species!r}")
    if atac is not None and species == "human":
        raise ValueError(
            "ATAC input must be in mm10 coordinates: --species human is only supported for RNA "
            "(human orthologs); embed the human RNA file alone, or supply mouse ATAC data"
        )
    if reference is None:
        reference = ReferenceTables.from_dir(require_peaks=atac is not None)
    if atac is not None and not reference.has_peaks:
        raise FileNotFoundError("peak table not downloaded; run `bloom download --what peaks`")

    report: dict[str, dict[str, Any]] = {}
    source: dict[str, dict[str, Any]] = {}
    rna_p = atac_p = None
    if rna is not None:
        rna_p, report["rna"], source["rna"] = prepare_rna(
            rna,
            species=species,
            reference=reference,
            gene_col=gene_col,
            layer=layer,
            force_species=force_species,
        )
    if atac is not None:
        atac_p, report["atac"], source["atac"] = prepare_atac(
            atac,
            reference=reference,
            peak_match=peak_match,
            min_overlap=min_overlap,
            species=species,
        )
    alignment = None
    unmatched: dict[str, AnnData] = {}
    if rna_p is not None and atac_p is not None:
        rna_p, atac_p, alignment, unmatched = _align(rna_p, atac_p, keep_unmatched=keep_unmatched)
    return PreparedDataset(
        rna=rna_p,
        atac=atac_p,
        species=species,
        reference_id=reference.reference_id,
        report=report,
        source=source,
        alignment=alignment,
        unmatched=unmatched,
    )
