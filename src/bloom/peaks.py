"""Peak coordinates: parsing, exact and overlap matching against the reference peak table.

docs/data-schema.md section 1.2. A user's ATAC ``var`` names peaks as ``chr1:100-600``,
``chr1-100-600`` or ``chr1_100_600`` (``chr`` prefix optional), or carries ``chrom`` / ``start`` /
``end`` columns. :func:`parse_peaks` turns either into arrays; contigs outside ``chr1..19, X, Y, M``
are dropped and counted. :class:`ReferencePeakIndex` matches parsed peaks onto the reference table
(``peaks.parquet``: 1,747,701 mm10 peaks of 501 bp) either **exactly** (same ``chrom, start, end``;
how the training data was built) or by **overlap** (a reference peak is hit when a user peak covers
at least ``min_overlap`` of it, as a fraction of the reference peak's width). Both return the
user -> reference pairs (:class:`PeakMatches`); many-to-one and one-to-many pairs are kept and
OR-ed by :func:`detection_matrix`.

The overlap index is a per-chromosome sorted-start array with ``searchsorted`` (two binary searches
per user peak, then an exact filter on the candidate window), so 100k user peaks against the full
table take about a second (test R10: < 10 s). Coordinates are treated as half-open ``[start, end)``,
matching the ``end - start == 501`` width of the reference names.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from bloom.feature_lists import _PEAK_RE

__all__ = [
    "CANONICAL_CONTIGS",
    "DEFAULT_MIN_OVERLAP",
    "EXACT_RATE_THRESHOLD",
    "PEAK_MATCH_MODES",
    "ParsedPeaks",
    "PeakMatches",
    "ReferencePeakIndex",
    "detection_matrix",
    "parse_peaks",
    "peak_names",
]

logger = logging.getLogger(__name__)

CANONICAL_CONTIGS: tuple[str, ...] = (*(f"chr{i}" for i in range(1, 20)), "chrX", "chrY", "chrM")
"""mm10 contigs kept; anything else (``chrUn_*``, ``chr1_random``, ``GL456210.1``, ...) is dropped."""
DEFAULT_MIN_OVERLAP: float = 0.5
"""Default ``--min-overlap``: fraction of the reference peak a user peak must cover (G6 study)."""
EXACT_RATE_THRESHOLD: float = 0.95
"""``peak_match='auto'`` picks ``exact`` when at least this fraction of parsed user peaks match exactly."""
PEAK_MATCH_MODES: tuple[str, ...] = ("auto", "exact", "overlap")
COORD_COLUMNS: tuple[str, str, str] = ("chrom", "start", "end")

_CANONICAL = set(CANONICAL_CONTIGS)


def _normalise_contig(raw: pd.Series) -> pd.Series:
    """``1`` -> ``chr1``, ``chrx`` -> ``chrX``, ``MT``/``chrMT`` -> ``chrM``; other names kept as given."""
    s = raw.astype(str).str.strip()
    body = s.str.replace(r"^[cC][hH][rR]", "", regex=True)
    upper = body.str.upper()
    body = body.where(~upper.isin(["X", "Y", "M", "MT"]), upper.replace({"MT": "M"}))
    return "chr" + body


@dataclass
class ParsedPeaks:
    """Peak coordinates of a user ``var`` after parsing and contig filtering.

    Attributes:
        chrom: ``object[n]`` normalised contig (``"chr1"``); ``""`` where unparsable.
        start: ``int64[n]`` (0 where unparsable).
        end: ``int64[n]``.
        valid: ``bool[n]`` parsed **and** on a canonical contig; only these are matched.
        n_unparsed: identifiers that are not coordinates (or ``start >= end``).
        dropped_contigs: ``{contig: count}`` of parsed peaks on non-canonical contigs.
        source: ``"var_names"`` or ``"var[chrom,start,end]"``.
        ids: the identifiers as given (``object[n]``), for reports.
    """

    chrom: np.ndarray
    start: np.ndarray
    end: np.ndarray
    valid: np.ndarray
    n_unparsed: int
    dropped_contigs: dict[str, int]
    source: str
    ids: np.ndarray

    def __len__(self) -> int:
        return int(self.chrom.size)

    @property
    def n_valid(self) -> int:
        return int(self.valid.sum())

    @property
    def n_dropped(self) -> int:
        """Parsed peaks dropped for being on a non-canonical contig."""
        return int(sum(self.dropped_contigs.values()))

    def names(self) -> np.ndarray:
        """Canonical ``chr:start-end`` names (``object[n]``; ``""`` where unparsable)."""
        return peak_names(self.chrom, self.start, self.end)

    def shifted(self, offset: int) -> ParsedPeaks:
        """Copy with every coordinate shifted by ``offset`` bp (for synthetic peak-caller studies, G6)."""
        return ParsedPeaks(
            chrom=self.chrom.copy(),
            start=self.start + int(offset),
            end=self.end + int(offset),
            valid=self.valid.copy(),
            n_unparsed=self.n_unparsed,
            dropped_contigs=dict(self.dropped_contigs),
            source=self.source,
            ids=self.ids.copy(),
        )

    def report(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "n_peaks": len(self),
            "n_valid": self.n_valid,
            "n_unparsed": int(self.n_unparsed),
            "n_dropped_contigs": self.n_dropped,
            "dropped_contigs": dict(self.dropped_contigs),
        }

    def by_contig(self) -> dict[str, np.ndarray]:
        """``{contig: int64 positions}`` of the valid peaks, grouped once (used by both matchers)."""
        valid = np.flatnonzero(self.valid)
        if valid.size == 0:
            return {}
        codes, uniques = pd.factorize(pd.Series(self.chrom[valid]), sort=True)
        order = np.argsort(codes, kind="stable")
        bounds = np.flatnonzero(np.r_[True, codes[order][1:] != codes[order][:-1], True])
        return {
            str(uniques[codes[order[lo]]]): valid[order[lo:hi]]
            for lo, hi in itertools.pairwise(bounds)
        }


def peak_names(chrom: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """``chr:start-end`` strings (``object[n]``); ``""`` where ``chrom`` is empty."""
    chrom = np.asarray(chrom, dtype=object)
    out = np.array(
        [f"{c}:{int(s)}-{int(e)}" if c else "" for c, s, e in zip(chrom, start, end, strict=True)],
        dtype=object,
    )
    return out


def parse_peaks(
    var: pd.DataFrame | pd.Index | Sequence[str] | np.ndarray,
    *,
    columns: tuple[str, str, str] = COORD_COLUMNS,
    canonical: Sequence[str] = CANONICAL_CONTIGS,
) -> ParsedPeaks:
    """Parse peak identifiers (``var_names``) or ``var[chrom, start, end]`` into coordinates (R7).

    Args:
        var: the ATAC ``var`` frame (its ``chrom``/``start``/``end`` columns are used when **all**
            three are present, else its index), or a sequence of identifier strings.
        columns: names of the coordinate columns to look for in a frame.
        canonical: contigs kept; others are dropped and counted per contig.

    Returns:
        :class:`ParsedPeaks` (``valid`` marks the peaks that take part in matching).
    """
    canon = set(canonical)
    if isinstance(var, pd.DataFrame) and all(c in var.columns for c in columns):
        ids = np.asarray(var.index.astype(str), dtype=object)
        chrom = _normalise_contig(var[columns[0]])
        start = pd.to_numeric(var[columns[1]], errors="coerce")
        end = pd.to_numeric(var[columns[2]], errors="coerce")
        parsed = start.notna() & end.notna()
        source = f"var[{','.join(columns)}]"
    else:
        names = var.index if isinstance(var, pd.DataFrame) else var
        s = pd.Series(np.asarray(list(names), dtype=object).astype(str)).str.strip()
        ids = s.to_numpy(dtype=object)
        m = s.str.extract(_PEAK_RE)  # (n, 3): contig, start, end
        parsed = m[0].notna()
        chrom = _normalise_contig(m[0].fillna(""))
        start = pd.to_numeric(m[1], errors="coerce")
        end = pd.to_numeric(m[2], errors="coerce")
        source = "var_names"
    start_i = start.fillna(0).to_numpy(dtype=np.int64, copy=True)
    end_i = end.fillna(0).to_numpy(dtype=np.int64, copy=True)
    parsed = parsed.to_numpy() & (end_i > start_i) & (start_i >= 0)
    chrom_arr = chrom.to_numpy(dtype=object)
    chrom_arr[~parsed] = ""
    on_canon = pd.Series(chrom_arr).isin(canon).to_numpy()
    valid = parsed & on_canon
    dropped = pd.Series(chrom_arr[parsed & ~on_canon]).value_counts()
    dropped_contigs = {str(k): int(v) for k, v in dropped.items()}
    n_unparsed = int((~parsed).sum())
    if dropped_contigs:
        logger.info(
            "dropped %d peak(s) on %d non-canonical contig(s): %s",
            sum(dropped_contigs.values()),
            len(dropped_contigs),
            dict(list(dropped_contigs.items())[:5]),
        )
    return ParsedPeaks(
        chrom=chrom_arr,
        start=start_i,
        end=end_i,
        valid=valid,
        n_unparsed=n_unparsed,
        dropped_contigs=dropped_contigs,
        source=source,
        ids=ids,
    )


@dataclass
class PeakMatches:
    """User-peak -> reference-peak pairs from :meth:`ReferencePeakIndex.match`.

    Attributes:
        user_idx: ``int64[m]`` column of the user ``var`` (``col_idx`` space).
        ref_emb_idx: ``int32[m]`` row of ``peaks.parquet`` (``emb_idx`` space).
        overlap: ``float32[m]`` covered fraction of the reference peak (1.0 for exact matches).
        n_user: number of user peaks the matching was run on (all columns, valid or not).
        mode: ``"exact"`` or ``"overlap"``.
        min_overlap: threshold used (``None`` for exact).
    """

    user_idx: np.ndarray
    ref_emb_idx: np.ndarray
    overlap: np.ndarray
    n_user: int
    mode: str
    min_overlap: float | None = None

    def __len__(self) -> int:
        return int(self.user_idx.size)

    @property
    def n_user_mapped(self) -> int:
        """Distinct user peaks with at least one reference partner."""
        return int(np.unique(self.user_idx).size)

    @property
    def n_ref_hit(self) -> int:
        """Distinct reference peaks hit by at least one user peak."""
        return int(np.unique(self.ref_emb_idx).size)

    def user_to_ref(self) -> np.ndarray:
        """``int32[n_user]`` first reference partner per user peak, ``-1`` when unmapped.

        For exact matches this is the complete mapping (one partner at most); for overlap matches
        a user peak may have several partners and only the first (lowest ``emb_idx``) is returned.
        """
        out = np.full(self.n_user, -1, dtype=np.int32)
        if len(self):
            order = np.lexsort((self.ref_emb_idx, self.user_idx))
            u, r = self.user_idx[order], self.ref_emb_idx[order]
            first = np.ones(u.size, dtype=bool)
            first[1:] = u[1:] != u[:-1]
            out[u[first]] = r[first]
        return out

    def report(self) -> dict[str, Any]:
        return {
            "peak_match": self.mode,
            "min_overlap": self.min_overlap,
            "n_pairs": len(self),
            "n_user_peaks": int(self.n_user),
            "n_user_peaks_mapped": self.n_user_mapped,
            "n_ref_peaks_hit": self.n_ref_hit,
        }


@dataclass
class ReferencePeakIndex:
    """Per-chromosome sorted reference peaks for exact and overlap lookups.

    Build with :meth:`from_table` from ``ReferenceTables.peaks`` (a frame with ``chrom``, ``start``,
    ``end`` indexed by ``emb_idx``; ``bloom.reference.ReferenceTables.peak_index`` caches one).

    Attributes:
        contigs: ``{chrom: (start int64[k], end int64[k], emb_idx int32[k])}`` sorted by ``start``.
        max_width: largest reference peak width (501 for ``mm10-evo2-v1``), bounds the search window.
        n_peaks: rows in the table.
    """

    contigs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=dict)
    max_width: int = 0
    n_peaks: int = 0

    @classmethod
    def from_table(cls, peaks: pd.DataFrame) -> ReferencePeakIndex:
        """Index a peak table (columns ``chrom, start, end``; index or column ``emb_idx``)."""
        if "emb_idx" in peaks.columns:
            emb = peaks["emb_idx"].to_numpy(dtype=np.int64)
        else:
            emb = peaks.index.to_numpy(dtype=np.int64)
        chrom = _normalise_contig(peaks["chrom"]).to_numpy(dtype=object)
        start = peaks["start"].to_numpy(dtype=np.int64)
        end = peaks["end"].to_numpy(dtype=np.int64)
        if (end <= start).any():
            raise ValueError("reference peak table has end <= start rows")
        order = np.lexsort((start, chrom.astype(str)))
        chrom, start, end, emb = chrom[order], start[order], end[order], emb[order]
        contigs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        bounds = np.flatnonzero(np.r_[True, chrom[1:] != chrom[:-1], True])
        for lo, hi in itertools.pairwise(bounds):
            contigs[str(chrom[lo])] = (
                np.ascontiguousarray(start[lo:hi]),
                np.ascontiguousarray(end[lo:hi]),
                np.ascontiguousarray(emb[lo:hi].astype(np.int32)),
            )
        return cls(
            contigs=contigs,
            max_width=int((end - start).max()) if len(start) else 0,
            n_peaks=len(start),
        )

    # ---- matching ---------------------------------------------------------------------------

    def match_exact(self, peaks: ParsedPeaks) -> PeakMatches:
        """Pairs where ``(chrom, start, end)`` equals a reference peak (R8)."""
        users: list[np.ndarray] = []
        refs: list[np.ndarray] = []
        groups = peaks.by_contig()
        for name, (rs, re_, remb) in self.contigs.items():
            sel = groups.get(name)
            if sel is None:
                continue
            us, ue = peaks.start[sel], peaks.end[sel]
            # starts are unique per contig in the release table; handle duplicates by scanning
            # the run of equal starts (rare: never in mm10-evo2-v1)
            lo = np.searchsorted(rs, us, side="left")
            hi = np.searchsorted(rs, us, side="right")
            width = hi - lo
            simple = width <= 1
            hit_pos = np.where(simple & (lo < rs.size), np.minimum(lo, rs.size - 1), 0)
            ok = simple & (lo < rs.size) & (rs[hit_pos] == us) & (re_[hit_pos] == ue)
            users.append(sel[ok])
            refs.append(remb[hit_pos[ok]])
            for j in np.flatnonzero(~simple):  # duplicate starts: scan the run
                for p in range(lo[j], hi[j]):
                    if re_[p] == ue[j]:
                        users.append(sel[j : j + 1])
                        refs.append(remb[p : p + 1])
                        break
        u = np.concatenate(users) if users else np.zeros(0, dtype=np.int64)
        r = np.concatenate(refs) if refs else np.zeros(0, dtype=np.int32)
        order = np.argsort(u, kind="stable")
        return PeakMatches(
            user_idx=u[order].astype(np.int64),
            ref_emb_idx=r[order].astype(np.int32),
            overlap=np.ones(u.size, dtype=np.float32),
            n_user=len(peaks),
            mode="exact",
        )

    def match_overlap(
        self, peaks: ParsedPeaks, min_overlap: float = DEFAULT_MIN_OVERLAP
    ) -> PeakMatches:
        """Pairs where the user peak covers ``>= min_overlap`` of the reference peak (R9, R10).

        The covered fraction is ``overlap_bp / (ref_end - ref_start)``; ``min_overlap = 0`` means any
        positive overlap. One user peak may hit several reference peaks and vice versa.
        """
        if not 0.0 <= float(min_overlap) <= 1.0:
            raise ValueError(f"min_overlap must be in [0, 1], got {min_overlap}")
        thr = float(min_overlap)
        users: list[np.ndarray] = []
        refs: list[np.ndarray] = []
        fracs: list[np.ndarray] = []
        groups = peaks.by_contig()
        for name, (rs, re_, remb) in self.contigs.items():
            sel = groups.get(name)
            if sel is None:
                continue
            us, ue = peaks.start[sel], peaks.end[sel]
            # candidates: ref_start < user_end and ref_start > user_start - max_width
            lo = np.searchsorted(rs, us - self.max_width, side="right")
            hi = np.searchsorted(rs, ue, side="left")
            counts = hi - lo
            keep = counts > 0
            if not keep.any():
                continue
            sel, us, ue, lo, counts = sel[keep], us[keep], ue[keep], lo[keep], counts[keep]
            total = int(counts.sum())
            rep = np.repeat(np.arange(sel.size), counts)  # (total,) user position per pair
            offsets = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
            pos = lo[rep] + offsets  # (total,) reference position per pair
            ov = np.minimum(ue[rep], re_[pos]) - np.maximum(us[rep], rs[pos])  # (total,) bp
            frac = ov / (re_[pos] - rs[pos])
            ok = (ov > 0) & (frac >= thr)
            users.append(sel[rep[ok]])
            refs.append(remb[pos[ok]])
            fracs.append(frac[ok].astype(np.float32))
        u = np.concatenate(users) if users else np.zeros(0, dtype=np.int64)
        r = np.concatenate(refs) if refs else np.zeros(0, dtype=np.int32)
        f = np.concatenate(fracs) if fracs else np.zeros(0, dtype=np.float32)
        order = np.lexsort((r, u))
        return PeakMatches(
            user_idx=u[order].astype(np.int64),
            ref_emb_idx=r[order].astype(np.int32),
            overlap=f[order],
            n_user=len(peaks),
            mode="overlap",
            min_overlap=thr,
        )

    def match(
        self,
        peaks: ParsedPeaks,
        mode: str = "auto",
        min_overlap: float = DEFAULT_MIN_OVERLAP,
        exact_rate_threshold: float = EXACT_RATE_THRESHOLD,
    ) -> tuple[PeakMatches, dict[str, Any]]:
        """Match with ``mode`` ``auto`` | ``exact`` | ``overlap``.

        ``auto`` runs the exact match first and keeps it when at least ``exact_rate_threshold`` of
        the valid user peaks matched; otherwise it falls back to overlap matching. The decision is
        recorded in the returned info dict (``peak_match_requested``, ``peak_match``,
        ``exact_rate``).

        Returns:
            ``(matches, info)``.
        """
        if mode not in PEAK_MATCH_MODES:
            raise ValueError(f"peak_match must be one of {PEAK_MATCH_MODES}, got {mode!r}")
        info: dict[str, Any] = {"peak_match_requested": mode, "min_overlap": float(min_overlap)}
        exact = None
        if mode in ("auto", "exact"):
            exact = self.match_exact(peaks)
            n_valid = peaks.n_valid
            info["exact_rate"] = float(exact.n_user_mapped / n_valid) if n_valid else 0.0
        if mode == "exact" or (mode == "auto" and info["exact_rate"] >= exact_rate_threshold):
            assert exact is not None
            info["peak_match"] = "exact"
            return exact, info
        if mode == "auto":
            logger.info(
                "peak_match=auto: exact rate %.3f < %.2f, using overlap matching (min_overlap %.2f)",
                info["exact_rate"],
                exact_rate_threshold,
                min_overlap,
            )
        ov = self.match_overlap(peaks, min_overlap)
        info["peak_match"] = "overlap"
        return ov, info


def detection_matrix(
    X: sp.spmatrix | np.ndarray, matches: PeakMatches
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Reference-peak matrix ``(n_cells, n_ref_hit)`` from a user ``cells x peaks`` matrix (OR / sum).

    A reference peak is *detected* in a cell when any user peak mapped onto it has ``X > 0``
    (OR over many-to-one pairs). The stored value is the **sum** of the mapped user peaks' values,
    so ``X > 0`` is the detection and the magnitude can rank peaks (``top`` strategy); with exact
    one-to-one matching the values are the user's counts unchanged.

    Args:
        X: user matrix ``(n_cells, n_user_peaks)``, non-negative; any scipy format or dense.
        matches: pairs from :class:`ReferencePeakIndex`.

    Returns:
        ``(M, ref_emb_idx, n_user_per_ref)``: ``M`` CSR float32 ``(n_cells, k)`` with sorted
        indices and no explicit zeros; ``ref_emb_idx`` int32 ``(k,)`` strictly increasing column
        labels; ``n_user_per_ref`` int32 ``(k,)`` user peaks mapped onto each column.
    """
    Xc = sp.csr_matrix(X, dtype=np.float32)
    n_cells, n_user = Xc.shape
    if matches.n_user != n_user:
        raise ValueError(
            f"X has {n_user} columns but the matches were computed for {matches.n_user}"
        )
    ref_unique, inverse, counts = np.unique(
        matches.ref_emb_idx, return_inverse=True, return_counts=True
    )
    k = ref_unique.size
    if k == 0:
        return (
            sp.csr_matrix((n_cells, 0), dtype=np.float32),
            ref_unique.astype(np.int32),
            np.zeros(0, dtype=np.int32),
        )
    P = sp.csr_matrix(
        (np.ones(len(matches), dtype=np.float32), (matches.user_idx, inverse)),
        shape=(n_user, k),
    )  # (n_user, k) incidence; duplicate (user, ref) pairs cannot occur
    M = sp.csr_matrix(Xc @ P, dtype=np.float32)
    M.eliminate_zeros()
    M.sort_indices()
    return M, ref_unique.astype(np.int32), counts.astype(np.int32)
