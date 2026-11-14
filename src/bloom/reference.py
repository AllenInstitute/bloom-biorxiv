"""Reference tables of the ``mm10-evo2-v1`` release: genes, peaks and human orthologs.

The model consumes features as rows of two reference tables (``emb_idx``, see AGENTS.md
section 5 for the two index spaces). This module loads those tables from a reference
directory (``$BLOOM_HOME/references/mm10-evo2-v1`` by default; docs/assets.md section 2)
and resolves user identifiers onto ``emb_idx``:

* mouse gene symbols (``genes.parquet.gene_name``, exact, then case/whitespace-normalised),
* mouse Ensembl ids (``genes.parquet.gene_id``, version suffix ignored),
* human HGNC symbols / ``ENSG`` ids through the shipped one-to-one ortholog table, joined on
  the mouse Ensembl ``gene_id`` (docs/data-schema.md section 1.3),
* peak names ``chr:start-end`` (``peaks.parquet.name``, exact; coordinate parsing and overlap
  matching live in ``bloom.peaks``; :attr:`ReferenceTables.peak_index` caches the interval index).

Tensors (``cseq`` f32 ``(N, 768)``, ``chr`` i8 ``(N,)``, ``bp`` i32 ``(N,)``) are read from the
safetensors files on demand; the 5.4 GB peak table is never touched unless a peak accessor is
called (test M6).
"""

from __future__ import annotations

import logging
import os
import re
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

if TYPE_CHECKING:
    import torch

    from bloom.peaks import ReferencePeakIndex

logger = logging.getLogger(__name__)

REFERENCE_ID = "mm10-evo2-v1"
GENE_FILES = ("genes.parquet", "genes.safetensors")
PEAK_FILES = ("peaks.parquet", "peaks.safetensors")
ORTHOLOG_FILE = "orthologs_human_mouse_one2one.tsv"
MAPPED_VIA_CATEGORIES = ("exact", "gene_id", "ortholog", "none")
Species = Literal["mouse", "human"]

_ENSEMBL_VERSION = re.compile(r"\.\d+$")
_ENSMUSG = "ENSMUSG"
_ENSG = "ENSG0"  # human ids are ENSG + 11 digits; ENSMUSG also starts with "ENS"


def bloom_home() -> Path:
    """Asset cache directory: ``$BLOOM_HOME`` or ``~/.cache/bloom`` (docs/assets.md section 2)."""
    return Path(os.environ.get("BLOOM_HOME", Path.home() / ".cache" / "bloom")).expanduser()


def default_reference_dir(reference_id: str = REFERENCE_ID) -> Path:
    """``$BLOOM_HOME/references/<reference_id>``."""
    return bloom_home() / "references" / reference_id


def strip_ensembl_version(gene_id: str) -> str:
    """``ENSMUSG00000005716.9`` -> ``ENSMUSG00000005716``."""
    return _ENSEMBL_VERSION.sub("", gene_id)


def _normalise_symbol(symbol: str) -> str:
    """Case- and whitespace-insensitive key for symbol lookups."""
    return symbol.strip().lower()


def _as_str_array(ids: Sequence[str] | np.ndarray | pd.Index) -> np.ndarray:
    arr = np.asarray(list(ids) if not isinstance(ids, np.ndarray | pd.Index) else ids, dtype=object)
    if arr.ndim != 1:
        raise ValueError(f"expected a 1-d sequence of identifiers, got shape {arr.shape}")
    return np.array([str(x) for x in arr], dtype=object)


# --------------------------------------------------------------------------------------------
# Mapping result
# --------------------------------------------------------------------------------------------


@dataclass
class MappingResult:
    """Outcome of mapping a list of user feature identifiers onto reference rows.

    Attributes:
        ids: the identifiers as given (``object`` array, length ``n``).
        emb_idx: ``int32[n]`` row in the reference table, ``-1`` when unmapped.
        mapped_via: categorical ``[n]`` with categories ``exact`` (mouse symbol, after optional
            case/whitespace normalisation), ``gene_id`` (mouse Ensembl id), ``ortholog`` (human
            symbol / ``ENSG`` id through the ortholog table), ``none``.
        symbol: ``object[n]`` reference symbol (``genes.parquet.gene_name``) of the mapped row,
            ``""`` when unmapped. This is what ``prepare`` writes to ``var["bloom_symbol"]``.
        report: counts (``n_ids``, ``n_mapped``, ``n_unmapped``, ``n_exact``, ``n_gene_id``,
            ``n_ortholog``, ``n_normalised`` case/whitespace fixes, ``n_duplicate_ids``
            identifiers given more than once, ``n_duplicate_emb_idx`` distinct ids resolving to
            the same row, ``species``, ``unmapped_examples``).
    """

    ids: np.ndarray
    emb_idx: np.ndarray  # (n,) int32, -1 = unmapped
    mapped_via: pd.Categorical
    symbol: np.ndarray
    report: dict[str, Any] = field(default_factory=dict)

    @property
    def mapped(self) -> np.ndarray:
        """Boolean mask ``[n]`` of mapped identifiers."""
        return self.emb_idx >= 0

    @property
    def n_mapped(self) -> int:
        return int(self.mapped.sum())

    def __len__(self) -> int:
        return len(self.emb_idx)


def mapping_report(
    result: MappingResult, X: Any | None = None, *, low_rate_threshold: float = 0.70
) -> dict[str, Any]:
    """Mapping summary for ``uns["bloom"]["mapping"]`` (docs/data-schema.md section 2.1; test R6).

    Args:
        result: output of :meth:`ReferenceTables.map_genes` for the columns of ``X``.
        X: optional cells x features matrix (dense array, scipy sparse, or anything with
            ``.nonzero()``); only ``X > 0`` is used. When given, ``detected_mass_mapped`` is the
            fraction of detected (non-zero) entries that fall in mapped columns -- the quantity
            the low-mapping warning is gated on (docs/data-schema.md section 1.1).

    Returns:
        ``{"n_var", "n_mapped", "mapping_rate", "detected_mass_mapped", "low_mapping", ...}``
        plus the counters of ``result.report``.
    """
    rep = dict(result.report)
    rep["n_var"] = len(result)
    rep["n_mapped"] = result.n_mapped
    rep["mapping_rate"] = float(result.n_mapped / len(result)) if len(result) else 0.0
    detected_mass_mapped: float | None = None
    if X is not None:
        if X.shape[1] != len(result):
            raise ValueError(f"X has {X.shape[1]} columns but the mapping has {len(result)} ids")
        per_col = np.asarray((X > 0).sum(axis=0)).ravel()  # detected entries per column
        total = int(per_col.sum())
        detected_mass_mapped = float(per_col[result.mapped].sum() / total) if total else 0.0
    rep["detected_mass_mapped"] = detected_mass_mapped
    rate = detected_mass_mapped if detected_mass_mapped is not None else rep["mapping_rate"]
    rep["low_mapping"] = bool(rate < low_rate_threshold)
    return rep


# --------------------------------------------------------------------------------------------
# Orthologs
# --------------------------------------------------------------------------------------------


@dataclass
class Orthologs:
    """One-to-one human -> mouse ortholog table (``orthologs_human_mouse_one2one.tsv``).

    Attributes:
        table: columns ``human_gene_id, human_gene_name, mouse_gene_id, mouse_gene_name`` (plus
            ``emb_idx`` in the shipped file). Ensembl ids carry no version suffix.
        header: ``key: value`` pairs parsed from the leading ``#`` lines (source, release, ...).
        path: file the table was read from, if any.
    """

    table: pd.DataFrame
    header: dict[str, str] = field(default_factory=dict)
    path: Path | None = None

    REQUIRED: ClassVar[tuple[str, ...]] = (
        "human_gene_id",
        "human_gene_name",
        "mouse_gene_id",
        "mouse_gene_name",
    )

    def __post_init__(self) -> None:
        missing = [c for c in self.REQUIRED if c not in self.table.columns]
        if missing:
            raise ValueError(f"ortholog table lacks columns {missing} ({self.path})")
        t = self.table
        t = t.assign(
            human_gene_id=t["human_gene_id"].astype(str).map(strip_ensembl_version),
            mouse_gene_id=t["mouse_gene_id"].astype(str).map(strip_ensembl_version),
            human_gene_name=t["human_gene_name"].astype(str),
            mouse_gene_name=t["mouse_gene_name"].astype(str),
        ).reset_index(drop=True)
        self.table = t
        # human -> mouse gene_id; one2one tables have unique keys, keep the first otherwise
        self._by_human_id: dict[str, str] = dict(
            zip(t["human_gene_id"][::-1], t["mouse_gene_id"][::-1], strict=True)
        )
        self._by_human_symbol: dict[str, str] = dict(
            zip(t["human_gene_name"][::-1], t["mouse_gene_id"][::-1], strict=True)
        )
        self._by_human_symbol_norm: dict[str, str] = {}
        for sym, mid in zip(t["human_gene_name"][::-1], t["mouse_gene_id"][::-1], strict=True):
            self._by_human_symbol_norm[_normalise_symbol(sym)] = mid
        self._mouse_symbol_of_id: dict[str, str] = dict(
            zip(t["mouse_gene_id"], t["mouse_gene_name"], strict=True)
        )

    def __len__(self) -> int:
        return len(self.table)

    @classmethod
    def from_tsv(cls, path: str | Path) -> Orthologs:
        """Read a ``#``-headed TSV (the shipped table or a legacy BioMart export)."""
        path = Path(path)
        header: dict[str, str] = {}
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.startswith("#"):
                    break
                body = line[1:].strip()
                if ":" in body:
                    k, v = body.split(":", 1)
                    header[k.strip()] = v.strip()
        table = pd.read_csv(path, sep="\t", comment="#", dtype=str, keep_default_na=False)
        if "emb_idx" in table.columns:
            table["emb_idx"] = table["emb_idx"].astype(np.int32)
        return cls(table=table, header=header, path=path)

    def mouse_gene_id(self, human_id_or_symbol: str) -> tuple[str | None, bool]:
        """Resolve a human ``ENSG`` id or HGNC symbol to the mouse Ensembl ``gene_id``.

        Returns:
            ``(mouse_gene_id or None, normalised)`` where ``normalised`` is True when the symbol
            matched only after case/whitespace normalisation.
        """
        key = human_id_or_symbol.strip()
        if key.startswith(_ENSG):
            return self._by_human_id.get(strip_ensembl_version(key)), key != human_id_or_symbol
        hit = self._by_human_symbol.get(human_id_or_symbol)
        if hit is not None:
            return hit, False
        hit = self._by_human_symbol_norm.get(_normalise_symbol(key))
        return hit, hit is not None

    def mouse_symbol_of(self, mouse_gene_id: str) -> str | None:
        """Mouse symbol recorded in the ortholog table for ``mouse_gene_id`` (may drift from the gene table)."""
        return self._mouse_symbol_of_id.get(mouse_gene_id)


# --------------------------------------------------------------------------------------------
# Reference tables
# --------------------------------------------------------------------------------------------


class ReferenceTables:
    """Gene and peak reference tables of one release, with identifier -> ``emb_idx`` indices.

    Construct with :meth:`from_dir` (release layout) or directly with explicit file paths
    (tests build row-subset tables whose ``emb_idx`` column holds the full-table
    rows). Parquet metadata is loaded eagerly for genes; peaks are loaded on first access;
    safetensors are opened on first tensor request only.

    Attributes:
        genes: ``DataFrame`` indexed by ``emb_idx`` with ``gene_name, gene_id, chrom, start,
            end, strand`` (docs/assets.md section 1).
        reference_id: ``"mm10-evo2-v1"``.
    """

    def __init__(
        self,
        genes_parquet: str | Path,
        genes_safetensors: str | Path | None = None,
        peaks_parquet: str | Path | None = None,
        peaks_safetensors: str | Path | None = None,
        orthologs_tsv: str | Path | None = None,
        reference_id: str = REFERENCE_ID,
    ) -> None:
        self.reference_id = reference_id
        self.paths: dict[str, Path | None] = {
            "genes_parquet": Path(genes_parquet),
            "genes_safetensors": Path(genes_safetensors) if genes_safetensors else None,
            "peaks_parquet": Path(peaks_parquet) if peaks_parquet else None,
            "peaks_safetensors": Path(peaks_safetensors) if peaks_safetensors else None,
            "orthologs_tsv": Path(orthologs_tsv) if orthologs_tsv else None,
        }
        self.genes: pd.DataFrame = self._read_table(self.paths["genes_parquet"])
        for col in ("gene_name", "gene_id"):
            if col not in self.genes.columns:
                raise ValueError(f"{genes_parquet} lacks column {col!r}")
        self._peaks: pd.DataFrame | None = None
        self._peak_index: ReferencePeakIndex | None = None
        self._orthologs: Orthologs | None = None
        self._tensors: dict[str, dict[str, torch.Tensor]] = {}
        self._build_gene_index()

    # ---- construction -----------------------------------------------------------------------

    @classmethod
    def from_dir(
        cls,
        reference_dir: str | Path | None = None,
        *,
        require_peaks: bool = False,
        reference_id: str = REFERENCE_ID,
    ) -> ReferenceTables:
        """Load the release layout from ``reference_dir`` (default ``$BLOOM_HOME/references/mm10-evo2-v1``).

        ``genes.parquet`` and ``genes.safetensors`` are required; peak and ortholog files are
        optional unless ``require_peaks`` is set (peak strategies).
        """
        d = (
            Path(reference_dir)
            if reference_dir is not None
            else default_reference_dir(reference_id)
        )
        missing = [f for f in GENE_FILES if not (d / f).exists()]
        if require_peaks:
            missing += [f for f in PEAK_FILES if not (d / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"reference directory {d} lacks {missing}; run `bloom download --what genes"
                + (",peaks" if require_peaks else "")
                + "` (docs/assets.md section 2)"
            )
        opt = {f: (d / f if (d / f).exists() else None) for f in (*PEAK_FILES, ORTHOLOG_FILE)}
        return cls(
            genes_parquet=d / GENE_FILES[0],
            genes_safetensors=d / GENE_FILES[1],
            peaks_parquet=opt["peaks.parquet"],
            peaks_safetensors=opt["peaks.safetensors"],
            orthologs_tsv=opt[ORTHOLOG_FILE],
            reference_id=reference_id,
        )

    @staticmethod
    def _read_table(path: Path) -> pd.DataFrame:
        df = pq.read_table(path).to_pandas()
        if "emb_idx" in df.columns:
            df = df.set_index(df["emb_idx"].astype(np.int64).to_numpy())
        else:
            df.index = pd.RangeIndex(len(df))
        df.index.name = "emb_idx"
        return df

    def _build_gene_index(self) -> None:
        g = self.genes
        emb = g.index.to_numpy()
        names = g["gene_name"].astype(str).to_numpy()
        ids = g["gene_id"].astype(str).to_numpy()
        if pd.Series(names).duplicated().any():
            dups = pd.Series(names)[pd.Series(names).duplicated()].unique()[:5].tolist()
            raise ValueError(f"gene_name must be unique in the gene table; duplicates: {dups}")
        self._symbol_to_emb: dict[str, int] = dict(zip(names, emb.tolist(), strict=True))
        # normalised lookup; on collisions keep the first row (none in mm10-evo2-v1: checked at export)
        self._symbol_norm_to_emb: dict[str, int] = {}
        for n, e in zip(names, emb.tolist(), strict=True):
            self._symbol_norm_to_emb.setdefault(_normalise_symbol(n), e)
        self._gene_id_to_emb: dict[str, int] = {
            strip_ensembl_version(i): e for i, e in zip(ids, emb.tolist(), strict=True)
        }

    # ---- sizes -----------------------------------------------------------------------------

    @property
    def n_genes(self) -> int:
        """Rows in the loaded gene table (32,129 for the full release)."""
        return len(self.genes)

    @property
    def n_peaks(self) -> int:
        """Rows in the peak table, read from parquet metadata (does not load the table)."""
        p = self.paths["peaks_parquet"]
        if p is None:
            return 0
        if self._peaks is not None:
            return len(self._peaks)
        return pq.read_metadata(p).num_rows

    # ---- lazy members ----------------------------------------------------------------------

    @property
    def peaks(self) -> pd.DataFrame:
        """Peak table indexed by ``emb_idx`` (``chrom, start, end, name``); loaded on first access."""
        if self._peaks is None:
            p = self.paths["peaks_parquet"]
            if p is None:
                raise FileNotFoundError(
                    "peak table not downloaded; run `bloom download --what peaks`"
                )
            self._peaks = self._read_table(p)
            names = self._peaks["name"].astype(str).to_numpy()
            self._peak_name_to_emb: dict[str, int] = dict(
                zip(names, self._peaks.index.to_numpy().tolist(), strict=True)
            )
            logger.info("loaded peak table %s (%d rows)", p, len(self._peaks))
        return self._peaks

    @property
    def has_peaks(self) -> bool:
        return self.paths["peaks_parquet"] is not None

    @property
    def peak_index(self) -> ReferencePeakIndex:
        """Interval index over :attr:`peaks` for exact / overlap matching (``bloom.peaks``); built once."""
        if self._peak_index is None:
            from bloom.peaks import ReferencePeakIndex

            self._peak_index = ReferencePeakIndex.from_table(self.peaks)
        return self._peak_index

    # ---- FeatureReference protocol (bloom.feature_lists; universe / panel resolution) ----------

    @property
    def symbol_to_emb_idx(self) -> Mapping[str, int]:
        """``gene_name -> emb_idx`` (exact symbols), for :meth:`bloom.feature_lists.FeatureList.resolve`."""
        return self._symbol_to_emb

    @property
    def peak_to_emb_idx(self) -> Mapping[str, int]:
        """``"chr:start-end" -> emb_idx``; loads the peak parquet on first access."""
        _ = self.peaks
        return self._peak_name_to_emb

    @property
    def orthologs(self) -> Orthologs | None:
        """Shipped ortholog table, or ``None`` when the file is absent."""
        if self._orthologs is None and self.paths["orthologs_tsv"] is not None:
            self._orthologs = Orthologs.from_tsv(self.paths["orthologs_tsv"])
        return self._orthologs

    def set_orthologs(self, orthologs: Orthologs) -> None:
        """Attach an in-memory ortholog table (tests, custom tables)."""
        self._orthologs = orthologs

    # ---- tensors ---------------------------------------------------------------------------

    def _load_tensors(self, kind: str) -> dict[str, torch.Tensor]:
        if kind not in self._tensors:
            path = self.paths[f"{kind}_safetensors"]
            if path is None:
                what = "peaks" if kind == "peaks" else "genes"
                raise FileNotFoundError(
                    f"{what} table not downloaded; run `bloom download --what {what}`"
                )
            from safetensors import safe_open

            out: dict[str, torch.Tensor] = {}
            # safetensors memory-maps the file; tensors are materialised per key on CPU
            with safe_open(str(path), framework="pt", device="cpu") as f:
                for key in ("cseq", "chr", "bp"):
                    out[key] = f.get_tensor(key)
                if "emb_idx" in f.keys():
                    out["emb_idx"] = f.get_tensor("emb_idx")
            n = len(self.genes) if kind == "genes" else self.n_peaks
            if out["cseq"].shape[0] != n:
                raise ValueError(
                    f"{path}: cseq has {out['cseq'].shape[0]} rows but the parquet has {n}"
                )
            self._tensors[kind] = out
            logger.info("loaded %s tensors from %s", kind, path)
        return self._tensors[kind]

    def gene_tensors(self) -> dict[str, torch.Tensor]:
        """``{"cseq": f32 (N_gene, 768), "chr": i8 (N_gene,), "bp": i32 (N_gene,)}`` on CPU.

        For a row-subset table the dict also carries ``emb_idx`` i32 ``(N_gene,)``.
        """
        return self._load_tensors("genes")

    def peak_tensors(self) -> dict[str, torch.Tensor]:
        """``{"cseq": f32 (N_peak, 768), "chr": i8, "bp": i32}``; opens ``peaks.safetensors`` (5.4 GB)."""
        return self._load_tensors("peaks")

    def safetensors_metadata(self, kind: Literal["genes", "peaks"]) -> dict[str, str]:
        """Header metadata of the safetensors file (``chr_encoding``, ``source_cache``, ...)."""
        from safetensors import safe_open

        path = self.paths[f"{kind}_safetensors"]
        if path is None:
            raise FileNotFoundError(f"{kind}.safetensors not available")
        with safe_open(str(path), framework="pt", device="cpu") as f:
            return dict(f.metadata() or {})

    # ---- lookups ---------------------------------------------------------------------------

    def _resolve(self, raw: str, species: str, orth: Orthologs | None) -> tuple[int, str, bool]:
        """Resolve one identifier -> ``(emb_idx or -1, mapped_via, normalised)``."""
        key = raw.strip()
        if key.startswith(_ENSMUSG):
            hit = self._gene_id_to_emb.get(strip_ensembl_version(key))
            return (hit, "gene_id", key != raw) if hit is not None else (-1, "none", False)
        if species == "mouse":
            hit = self._symbol_to_emb.get(raw)
            if hit is not None:
                return hit, "exact", False
            hit = self._symbol_norm_to_emb.get(_normalise_symbol(key))
            return (hit, "exact", True) if hit is not None else (-1, "none", False)
        # human: ortholog table -> mouse gene_id -> gene table; mouse symbol as fallback
        if orth is None:
            raise FileNotFoundError(
                "species='human' needs the ortholog table "
                f"{ORTHOLOG_FILE}; run `bloom download --what orthologs`"
            )
        mouse_id, normalised = orth.mouse_gene_id(raw)
        if mouse_id is None:
            return -1, "none", False
        hit = self._gene_id_to_emb.get(mouse_id)
        if hit is None:
            msym = orth.mouse_symbol_of(mouse_id)
            hit = self._symbol_to_emb.get(msym) if msym is not None else None
        return (hit, "ortholog", normalised) if hit is not None else (-1, "none", False)

    def _finish(
        self, arr: np.ndarray, emb_idx: np.ndarray, via: np.ndarray, n_normalised: int, species: str
    ) -> MappingResult:
        mapped = emb_idx >= 0
        symbol = self.symbol_of(emb_idx)
        ids_s = pd.Series(arr)
        dup_mask = ids_s.duplicated(keep=False)
        n_dup_ids = int(ids_s[dup_mask].nunique()) if dup_mask.any() else 0
        rows = pd.Series(emb_idx[mapped])
        n_dup_rows = int(rows[rows.duplicated(keep=False)].nunique())
        report: dict[str, Any] = {
            "species": species,
            "n_ids": len(arr),
            "n_mapped": int(mapped.sum()),
            "n_unmapped": int((~mapped).sum()),
            "n_exact": int((via == "exact").sum()),
            "n_gene_id": int((via == "gene_id").sum()),
            "n_ortholog": int((via == "ortholog").sum()),
            "n_normalised": int(n_normalised),
            "n_duplicate_ids": n_dup_ids,
            "n_duplicate_emb_idx": n_dup_rows,
            "unmapped_examples": ids_s[~mapped].value_counts().index[:10].tolist(),
        }
        if n_normalised:
            warnings.warn(
                f"{n_normalised} gene identifier(s) matched only after case/whitespace "
                "normalisation (e.g. 'pvalb' -> 'Pvalb'); check var_names",
                stacklevel=3,
            )
        if n_dup_ids:
            warnings.warn(
                f"{n_dup_ids} gene identifier(s) occur more than once in var; duplicates map to "
                "the same reference row and their detection is OR-ed",
                stacklevel=3,
            )
        return MappingResult(
            ids=arr,
            emb_idx=emb_idx,
            mapped_via=pd.Categorical(via, categories=list(MAPPED_VIA_CATEGORIES)),
            symbol=symbol,
            report=report,
        )

    def map_genes(
        self, ids: Sequence[str] | np.ndarray | pd.Index, species: Species = "mouse"
    ) -> MappingResult:
        """Map gene identifiers onto reference rows.

        Mouse: ``ENSMUSG`` ids via ``gene_id`` (version suffix ignored), otherwise symbols via
        ``gene_name`` (exact first, then lower-cased/stripped; such fixes are counted and
        reported once as a warning). Human: ``ENSG`` ids and HGNC symbols are resolved through
        the ortholog table to a mouse ``gene_id`` and joined on ``genes.parquet.gene_id``; when
        that id is absent from the gene table the table's mouse symbol is tried as a fallback.
        Unknown identifiers get ``-1`` / ``none``. Duplicate identifiers all receive the same
        row and are counted (``n_duplicate_ids``) with a warning.

        Args:
            ids: feature identifiers (``var_names`` or a ``var`` column).
            species: ``"mouse"`` or ``"human"``.

        Returns:
            :class:`MappingResult` (``emb_idx`` int32 ``(n,)`` with ``-1`` for unmapped).
        """
        if species not in ("mouse", "human"):
            raise ValueError(f"species must be 'mouse' or 'human', got {species!r}")
        arr = _as_str_array(ids)
        orth = self.orthologs if species == "human" else None
        if species == "human" and orth is None:
            raise FileNotFoundError(
                "species='human' needs the ortholog table "
                f"{ORTHOLOG_FILE}; run `bloom download --what orthologs`"
            )
        emb_idx = np.full(len(arr), -1, dtype=np.int32)
        via = np.full(len(arr), "none", dtype=object)
        n_normalised = 0
        for i, raw in enumerate(arr):
            emb_idx[i], via[i], normalised = self._resolve(raw, species, orth)
            n_normalised += normalised
        return self._finish(arr, emb_idx, via, n_normalised, species)

    def map_symbols(
        self, symbols: Sequence[str] | np.ndarray | pd.Index, species: Species = "mouse"
    ) -> MappingResult:
        """Alias of :meth:`map_genes` for symbol input (issue #14 interface)."""
        return self.map_genes(symbols, species=species)

    def map_gene_ids(self, ensembl_ids: Sequence[str] | np.ndarray | pd.Index) -> MappingResult:
        """Map Ensembl ids: ``ENSMUSG`` via ``gene_id``, ``ENSG`` via the ortholog table.

        Species is inferred per identifier from the prefix; strings that are not Ensembl ids
        stay unmapped (``none``). ``ENSG`` ids require the ortholog table.
        """
        arr = _as_str_array(ensembl_ids)
        emb_idx = np.full(len(arr), -1, dtype=np.int32)
        via = np.full(len(arr), "none", dtype=object)
        n_normalised = 0
        n_human = n_mouse = 0
        for i, raw in enumerate(arr):
            key = raw.strip()
            if key.startswith(_ENSMUSG):
                n_mouse += 1
                emb_idx[i], via[i], normalised = self._resolve(raw, "mouse", None)
            elif key.startswith(_ENSG):
                n_human += 1
                emb_idx[i], via[i], normalised = self._resolve(raw, "human", self.orthologs)
            else:
                continue
            n_normalised += normalised
        species = (
            "human" if n_human and not n_mouse else "mouse" if n_mouse and not n_human else "mixed"
        )
        return self._finish(arr, emb_idx, via, n_normalised, species)

    def map_peaks(
        self,
        peaks: Iterable[str] | Iterable[tuple[str, int, int]] | np.ndarray | pd.Index,
    ) -> np.ndarray:
        """Exact peak lookup by name ``chr:start-end`` (or ``(chrom, start, end)`` tuples).

        Returns:
            ``int32[n]`` ``emb_idx`` with ``-1`` for peaks not in the reference set. Coordinate
            parsing of other formats and overlap matching belong to ``peaks.py`` (Phase 2).
        """
        _ = self.peaks  # loads the table and builds the name index
        out = []
        for p in peaks:
            if isinstance(p, tuple | list):
                chrom, start, end = p
                name = f"{chrom}:{int(start)}-{int(end)}"
            else:
                name = str(p).strip()
            out.append(self._peak_name_to_emb.get(name, -1))
        return np.asarray(out, dtype=np.int32)

    def symbol_of(self, emb_idx: Sequence[int] | np.ndarray) -> np.ndarray:
        """Gene symbols (``gene_name``) for reference rows; ``object[n]``, ``""`` for ``-1``."""
        idx = np.asarray(emb_idx, dtype=np.int64)
        names = self.genes["gene_name"].reindex(idx).astype(object).to_numpy(copy=True)
        names[(idx < 0) | pd.isna(names)] = ""
        return names

    def gene_id_of(self, emb_idx: Sequence[int] | np.ndarray) -> np.ndarray:
        """Ensembl ``gene_id`` for reference rows; ``object[n]``, ``""`` for ``-1``."""
        idx = np.asarray(emb_idx, dtype=np.int64)
        ids = self.genes["gene_id"].reindex(idx).astype(object).to_numpy(copy=True)
        ids[(idx < 0) | pd.isna(ids)] = ""
        return ids

    def peak_name_of(self, emb_idx: Sequence[int] | np.ndarray) -> np.ndarray:
        """Peak names ``chr:start-end`` for reference rows; ``object[n]``, ``""`` for ``-1``."""
        idx = np.asarray(emb_idx, dtype=np.int64)
        names = self.peaks["name"].reindex(idx).astype(object).to_numpy(copy=True)
        names[(idx < 0) | pd.isna(names)] = ""
        return names

    def __repr__(self) -> str:
        peaks = f"{self.n_peaks:,}" if self.has_peaks else "absent"
        orth = (
            "absent"
            if self.paths["orthologs_tsv"] is None and self._orthologs is None
            else "present"
        )
        return (
            f"ReferenceTables({self.reference_id}: genes={self.n_genes:,}, peaks={peaks}, "
            f"orthologs={orth})"
        )


__all__ = [
    "GENE_FILES",
    "MAPPED_VIA_CATEGORIES",
    "ORTHOLOG_FILE",
    "PEAK_FILES",
    "REFERENCE_ID",
    "MappingResult",
    "Orthologs",
    "ReferenceTables",
    "bloom_home",
    "default_reference_dir",
    "mapping_report",
    "strip_ensembl_version",
]
