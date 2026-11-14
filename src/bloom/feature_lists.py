"""Feature-list files (``universe`` / ``panel``): parsing and resolution to ``emb_idx``.

Format (docs/data-schema.md section 3): plain text, one identifier per line, ``#`` comments and
blank lines allowed. Identifiers are gene symbols or peak coordinates in any of the accepted forms
``chr1:100-600``, ``chr1-100-600``, ``chr1_100_600`` (a missing ``chr`` prefix is added). Peak
identifiers are normalised to ``chr:start-end``. Identifiers absent from the reference are
counted, reported once (``logging``) and ignored.

Resolution needs the reference tables (``bloom.reference``, issue #14); this module only requires
an object exposing ``symbol_to_emb_idx`` and ``peak_to_emb_idx`` mappings
(:class:`FeatureReference`), so tests can pass plain dicts.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "FeatureKind",
    "FeatureList",
    "FeatureReference",
    "ResolvedFeatures",
    "normalize_feature_id",
    "parse_peak_id",
]

logger = logging.getLogger(__name__)

FeatureKind = Literal["gene", "peak"]

# chr1:100-600 | chr1-100-600 | chr1_100_600 | 1:100-600 (separators may be mixed; the contig is
# non-greedy so chrUn_GL456389:1-2 keeps its underscore)
_PEAK_RE = re.compile(r"^(?:chr)?(.+?)[:_-](\d+)[-_:](\d+)$", re.IGNORECASE)


def parse_peak_id(text: str) -> tuple[str, int, int] | None:
    """Parse a peak coordinate string; ``None`` if ``text`` is not a coordinate.

    Accepts ``chr1:100-600``, ``chr1-100-600``, ``chr1_100_600`` and the same without the ``chr``
    prefix. Returns ``("chr1", 100, 600)`` with the ``chr`` prefix normalised (``MT`` -> ``chrM``).
    Full contig validation (dropping ``chrUn_*`` etc.) belongs to ``bloom.peaks`` (Phase 2).
    """
    m = _PEAK_RE.match(text.strip())
    if m is None:
        return None
    chrom, start, end = m.group(1), int(m.group(2)), int(m.group(3))
    if chrom.upper() == "MT":
        chrom = "M"
    if chrom.upper() in {"X", "Y", "M"}:
        chrom = chrom.upper()
    return f"chr{chrom}", start, end


def normalize_feature_id(text: str) -> tuple[str, FeatureKind]:
    """Return ``(canonical_id, kind)``: peaks become ``chr:start-end``; symbols are stripped."""
    peak = parse_peak_id(text)
    if peak is not None:
        chrom, start, end = peak
        return f"{chrom}:{start}-{end}", "peak"
    return text.strip(), "gene"


@runtime_checkable
class FeatureReference(Protocol):
    """What :meth:`FeatureList.resolve` needs from ``ReferenceTables`` (issue #14).

    Both mappings go from canonical identifier (gene symbol as in ``genes.parquet.gene_name``;
    peak ``chr:start-end`` as in ``peaks.parquet``) to ``emb_idx``. A missing key means unknown.
    """

    symbol_to_emb_idx: Mapping[str, int]
    peak_to_emb_idx: Mapping[str, int]


@dataclass(frozen=True)
class ResolvedFeatures:
    """Outcome of :meth:`FeatureList.resolve`.

    Attributes:
        emb_idx: ``(n_known,)`` int32 in list order (duplicates already removed by the parser).
        kind: ``"gene"`` or ``"peak"``.
        n_unknown: Number of identifiers absent from the reference.
        unknown: Up to ``max_unknown_kept`` of those identifiers, for the report.
    """

    emb_idx: np.ndarray
    kind: FeatureKind
    n_unknown: int
    unknown: tuple[str, ...] = ()

    @property
    def n_known(self) -> int:
        """Number of resolved identifiers."""
        return int(self.emb_idx.size)


@dataclass(frozen=True)
class FeatureList:
    """An ordered, de-duplicated list of gene symbols **or** peak coordinates.

    Attributes:
        ids: Canonical identifiers in file order (first occurrence kept).
        kind: ``"gene"`` or ``"peak"``; inferred from the identifiers (a list mixing both raises).
        source: Path the list came from, for provenance (``None`` when built in memory).
        n_duplicates: Identifiers dropped because they repeated an earlier line.
    """

    ids: tuple[str, ...]
    kind: FeatureKind
    source: str | None = None
    n_duplicates: int = 0
    _index: dict[str, int] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_index", {s: i for i, s in enumerate(self.ids)})

    def __len__(self) -> int:
        return len(self.ids)

    def __contains__(self, item: str) -> bool:
        return normalize_feature_id(item)[0] in self._index

    @classmethod
    def from_ids(cls, ids: Iterable[str], source: str | None = None) -> FeatureList:
        """Build from identifiers (lines); comments after ``#`` and blank entries are ignored."""
        seen: dict[str, None] = {}
        kinds: set[str] = set()
        n_dup = 0
        for raw in ids:
            text = raw.split("#", 1)[0].strip()
            if not text:
                continue
            canon, kind = normalize_feature_id(text)
            kinds.add(kind)
            if canon in seen:
                n_dup += 1
                continue
            seen[canon] = None
        if len(kinds) > 1:
            raise ValueError(
                f"feature list {source or '<memory>'} mixes gene symbols and peak coordinates; "
                "a universe/panel is one modality"
            )
        kind_out: FeatureKind = "peak" if kinds == {"peak"} else "gene"
        return cls(tuple(seen), kind_out, source, n_dup)

    @classmethod
    def from_file(cls, path: str | PathLike[str]) -> FeatureList:
        """Parse a one-identifier-per-line text file (``#`` comments, blank lines allowed)."""
        p = Path(path)
        with p.open("r", encoding="utf-8") as fh:
            return cls.from_ids(fh, source=str(p))

    def resolve(self, reference: FeatureReference, max_unknown_kept: int = 20) -> ResolvedFeatures:
        """Map identifiers to ``emb_idx`` through ``reference``; unknown ones are counted and logged.

        Args:
            reference: Any object with ``symbol_to_emb_idx`` and ``peak_to_emb_idx`` mappings.
            max_unknown_kept: How many unknown identifiers to keep in the result for reporting.

        Returns:
            :class:`ResolvedFeatures` with ``emb_idx`` in list order (unknowns skipped).
        """
        table = reference.symbol_to_emb_idx if self.kind == "gene" else reference.peak_to_emb_idx
        out: list[int] = []
        unknown: list[str] = []
        for s in self.ids:
            idx = table.get(s)
            if idx is None or int(idx) < 0:
                unknown.append(s)
            else:
                out.append(int(idx))
        if unknown:
            logger.warning(
                "feature list %s: %d of %d %s identifiers are not in the reference and were "
                "ignored (e.g. %s)",
                self.source or "<memory>",
                len(unknown),
                len(self.ids),
                self.kind,
                ", ".join(unknown[:5]),
            )
        return ResolvedFeatures(
            emb_idx=np.asarray(out, dtype=np.int32),
            kind=self.kind,
            n_unknown=len(unknown),
            unknown=tuple(unknown[:max_unknown_kept]),
        )
