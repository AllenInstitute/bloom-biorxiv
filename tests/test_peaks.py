"""`bloom.peaks`: R7-R10 (docs/test-plan.md section 2) and the reference-peak detection matrix.

Unit tests use a hand-written reference table (``_REF``: 501-bp mm10-style peaks on chr1, chr2, chrX);
R10 (100k random user peaks vs the full 1.75 M-row ``peaks.parquet`` in < 10 s) and the S6b peak part
(``universe`` = the DA peak panel on the 20-cell fixture) need the staged assets (``assets`` tier).
"""

from __future__ import annotations

import time

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from bloom import strategies as S
from bloom.feature_lists import FeatureList
from bloom.peaks import (
    CANONICAL_CONTIGS,
    ParsedPeaks,
    PeakMatches,
    ReferencePeakIndex,
    detection_matrix,
    parse_peaks,
    peak_names,
)
from bloom.reference import ReferenceTables
from bloom.tokens import PEAK, CellFeatures

# emb_idx, chrom, start, end  (501 bp each; chr1 peaks 1000 bp apart, two adjacent ones at 10000/10501)
_REF = [
    (0, "chr1", 1000, 1501),
    (1, "chr1", 2000, 2501),
    (2, "chr1", 3000, 3501),
    (3, "chr1", 10000, 10501),
    (4, "chr1", 10501, 11002),
    (7, "chr2", 100000, 100501),
    (9, "chr2", 200000, 200501),
    (12, "chrX", 5000, 5501),
]


@pytest.fixture(scope="module")
def ref_table() -> pd.DataFrame:
    df = pd.DataFrame(_REF, columns=["emb_idx", "chrom", "start", "end"])
    df["name"] = [f"{c}:{s}-{e}" for c, s, e in zip(df.chrom, df.start, df.end, strict=True)]
    return df.set_index("emb_idx")


@pytest.fixture(scope="module")
def index(ref_table) -> ReferencePeakIndex:
    return ReferencePeakIndex.from_table(ref_table)


# -- R7: coordinate parsing -----------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "ids",
    [
        ["chr1:100-600"],
        ["chr1-100-600"],
        ["chr1_100_600"],
        ["1:100-600"],
        ["CHR1:100-600"],
        [" chr1:100-600 "],
    ],
)
def test_r7_formats_parse_to_the_same_tuple(ids):
    p = parse_peaks(ids)
    assert p.source == "var_names"
    assert p.chrom.tolist() == ["chr1"] and p.start.tolist() == [100] and p.end.tolist() == [600]
    assert p.valid.tolist() == [True] and p.n_unparsed == 0 and p.dropped_contigs == {}
    assert p.names().tolist() == ["chr1:100-600"]


@pytest.mark.unit
def test_r7_var_columns_take_precedence_and_normalise_contigs():
    var = pd.DataFrame(
        {
            "chrom": ["1", "chrX", "MT", "chrUn_GL456389", "2"],
            "start": [100, 5, 10, 1, "7"],
            "end": [600, 20, 30, 2, "x"],
        },
        index=["a", "b", "c", "d", "e"],
    )
    p = parse_peaks(var)
    assert p.source == "var[chrom,start,end]"
    assert p.chrom.tolist() == ["chr1", "chrX", "chrM", "chrUn_GL456389", ""]
    assert p.valid.tolist() == [True, True, True, False, False]
    assert p.n_unparsed == 1  # "x" is not a coordinate
    assert p.dropped_contigs == {"chrUn_GL456389": 1}
    assert p.ids.tolist() == ["a", "b", "c", "d", "e"]
    # the same frame without one coordinate column falls back to the index
    assert parse_peaks(var.drop(columns=["end"])).source == "var_names"


@pytest.mark.unit
def test_r7_non_canonical_contigs_dropped_and_counted():
    ids = [
        "chr1:100-600",
        "chrUn_GL456389:1-500",
        "chr1_random:100-600",
        "GL456210.1:100-600",
        "chrUn_GL456389:1-500",
        "chr1_GL456211_random_100_600",
        "Pvalb",
        "chr1:600-100",
    ]
    p = parse_peaks(ids)
    assert len(p) == 8 and p.n_valid == 1
    assert p.dropped_contigs == {
        "chrUn_GL456389": 2,
        "chr1_random": 1,
        "chrGL456210.1": 1,
        "chr1_GL456211_random": 1,
    }
    assert p.n_dropped == 5 and p.n_unparsed == 2  # a symbol and an end <= start
    assert set(CANONICAL_CONTIGS) == {f"chr{i}" for i in range(1, 20)} | {"chrX", "chrY", "chrM"}
    rep = p.report()
    assert rep["n_peaks"] == 8 and rep["n_valid"] == 1 and rep["n_dropped_contigs"] == 5
    assert p.shifted(50).start[0] == 150 and p.shifted(50).valid.tolist() == p.valid.tolist()
    assert peak_names(p.chrom[:1], p.start[:1], p.end[:1]).tolist() == ["chr1:100-600"]


# -- R8: exact matching ---------------------------------------------------------------------------


@pytest.mark.unit
def test_r8_exact_match_and_one_bp_shift(ref_table, index):
    names = ref_table["name"].tolist()
    exact = index.match_exact(parse_peaks(names))
    assert exact.mode == "exact" and len(exact) == len(names)
    assert exact.user_idx.tolist() == list(range(len(names)))
    assert exact.ref_emb_idx.tolist() == ref_table.index.tolist()
    assert exact.overlap.tolist() == [1.0] * len(names)
    assert exact.user_to_ref().tolist() == ref_table.index.tolist()
    shifted = index.match_exact(parse_peaks(names).shifted(1))
    assert len(shifted) == 0 and shifted.n_user_mapped == 0
    assert shifted.user_to_ref().tolist() == [-1] * len(names)
    # a different width at the same start is not an exact match either
    assert len(index.match_exact(parse_peaks(["chr1:1000-1500"]))) == 0
    # an unknown contig / an unparsed id never matches and never raises
    assert len(index.match_exact(parse_peaks(["chr7:1000-1501", "chrUn_x:1-2", "gene"]))) == 0
    assert index.n_peaks == len(_REF) and index.max_width == 501


@pytest.mark.unit
def test_r8_auto_picks_exact_above_the_threshold(index, ref_table):
    names = ref_table["name"].tolist()
    m, info = index.match(parse_peaks(names), mode="auto")
    assert info["peak_match"] == "exact" and info["exact_rate"] == pytest.approx(1.0)
    assert info["peak_match_requested"] == "auto"
    # 7 of 8 shifted by 1 bp -> exact rate 1/8 < 0.95 -> overlap
    shifted = [f"{r.chrom}:{r.start + 1}-{r.end + 1}" for r in ref_table.iloc[:-1].itertuples()]
    p = parse_peaks([*shifted, names[-1]])
    m, info = index.match(p, mode="auto", min_overlap=0.5)
    assert info["peak_match"] == "overlap" and info["exact_rate"] == pytest.approx(1 / 8)
    assert m.mode == "overlap" and m.n_user_mapped == 8  # 500/501 covered
    m, info = index.match(p, mode="exact")
    assert m.n_user_mapped == 1 and info["peak_match"] == "exact"
    with pytest.raises(ValueError, match="peak_match"):
        index.match(p, mode="fuzzy")


# -- R9: overlap matching -------------------------------------------------------------------------


@pytest.mark.unit
def test_r9_overlap_covering_two_reference_peaks(index):
    # user peak chr1:10100-10900 covers 401/501 of emb 3 and 399/501 of emb 4
    m = index.match_overlap(parse_peaks(["chr1:10100-10900"]), min_overlap=0.5)
    assert m.ref_emb_idx.tolist() == [3, 4] and m.user_idx.tolist() == [0, 0]
    assert m.overlap.tolist() == pytest.approx([401 / 501, 399 / 501])
    assert m.n_user_mapped == 1 and m.n_ref_hit == 2
    assert m.user_to_ref().tolist() == [3]  # first partner only
    # min_overlap = 0 keeps any positive overlap; 0.9 drops both
    assert index.match_overlap(parse_peaks(["chr1:10100-10900"]), 0.0).ref_emb_idx.tolist() == [
        3,
        4,
    ]
    assert len(index.match_overlap(parse_peaks(["chr1:10100-10900"]), 0.9)) == 0
    # touching but not overlapping (half-open intervals) is not a hit
    assert len(index.match_overlap(parse_peaks(["chr1:1501-2000"]), 0.0)) == 0


@pytest.mark.unit
def test_r9_coverage_flips_at_min_overlap(index):
    # reference emb 0 = chr1:1000-1501 (501 bp). 0.49 * 501 = 245.49 bp; 0.51 * 501 = 255.51 bp
    covers_245 = parse_peaks(["chr1:500-1245"])  # 245 bp covered -> 0.489
    covers_256 = parse_peaks(["chr1:500-1256"])  # 256 bp covered -> 0.511
    assert len(index.match_overlap(covers_245, 0.5)) == 0
    m = index.match_overlap(covers_256, 0.5)
    assert m.ref_emb_idx.tolist() == [0] and m.overlap[0] == pytest.approx(256 / 501)
    # the threshold is on the *reference* peak's width: a huge user peak fully covers it
    big = index.match_overlap(parse_peaks(["chr1:0-20000"]), 0.5)
    assert big.ref_emb_idx.tolist() == [0, 1, 2, 3, 4] and big.overlap.tolist() == [1.0] * 5
    with pytest.raises(ValueError, match="min_overlap"):
        index.match_overlap(covers_256, 1.5)


@pytest.mark.unit
def test_r9_many_user_peaks_to_one_reference_peak_are_or_ed(index):
    # three user peaks each covering >= 0.5 of emb 7 (chr2:100000-100501); one unrelated on chrX
    ids = ["chr2:99800-100300", "chr2:100200-100700", "chr2:100000-100501", "chrX:5000-5501"]
    p = parse_peaks(ids)
    m = index.match_overlap(p, 0.5)
    assert m.user_idx.tolist() == [0, 1, 2, 3] and m.ref_emb_idx.tolist() == [7, 7, 7, 12]
    assert m.n_user_mapped == 4 and m.n_ref_hit == 2
    X = sp.csr_matrix(
        np.array(
            [
                [1, 0, 0, 0],  # cell 0: one of the three -> detected
                [0, 2, 3, 0],  # cell 1: two of the three -> detected once, value summed
                [0, 0, 0, 5],  # cell 2: only the chrX peak
                [0, 0, 0, 0],  # cell 3: empty
            ],
            dtype=np.float32,
        )
    )
    M, ref_emb, n_user = detection_matrix(X, m)
    assert ref_emb.tolist() == [7, 12] and n_user.tolist() == [3, 1]
    assert M.shape == (4, 2) and M.dtype == np.float32
    assert M.toarray().tolist() == [[1, 0], [5, 0], [0, 5], [0, 0]]
    assert (M > 0).sum(axis=1).A1.tolist() == [1, 1, 1, 0]
    assert M.has_sorted_indices and M.nnz == 3
    rep = m.report()
    assert (
        rep["n_pairs"] == 4
        and rep["n_ref_peaks_hit"] == 2
        and rep["min_overlap"] == pytest.approx(0.5)
    )
    with pytest.raises(ValueError, match="columns"):
        detection_matrix(X[:, :3], m)
    # no match at all -> (n_cells, 0)
    empty = detection_matrix(
        X,
        PeakMatches(
            np.zeros(0, np.int64), np.zeros(0, np.int32), np.zeros(0, np.float32), 4, "exact"
        ),
    )
    assert empty[0].shape == (4, 0) and empty[1].size == 0


@pytest.mark.unit
def test_detection_matrix_exact_keeps_user_values(ref_table, index):
    names = ref_table["name"].tolist()
    m = index.match_exact(parse_peaks(names))
    rng = np.random.default_rng(0)
    X = sp.random(5, len(names), density=0.5, format="csr", random_state=0, dtype=np.float32)
    X.data = rng.integers(1, 9, X.nnz).astype(np.float32)
    M, ref_emb, n_user = detection_matrix(X, m)
    assert ref_emb.tolist() == ref_table.index.tolist() and n_user.tolist() == [1] * len(names)
    np.testing.assert_array_equal(M.toarray(), X.toarray())  # one-to-one: counts unchanged


@pytest.mark.unit
def test_from_table_accepts_emb_idx_column_and_rejects_bad_rows(ref_table):
    idx = ReferencePeakIndex.from_table(ref_table.reset_index())
    assert set(idx.contigs) == {"chr1", "chr2", "chrX"}
    assert idx.contigs["chr1"][2].tolist() == [0, 1, 2, 3, 4]  # emb_idx sorted by start
    bad = ref_table.copy()
    bad.loc[0, "end"] = 900
    with pytest.raises(ValueError, match="end <= start"):
        ReferencePeakIndex.from_table(bad)


# -- R10 + S6b peak part: real tables (assets) -------------------------------------------------------


@pytest.mark.assets
def test_r10_overlap_100k_random_peaks_vs_full_table_under_10s(peak_table_dir):
    ref = ReferenceTables.from_dir(peak_table_dir)
    t0 = time.perf_counter()
    index = ref.peak_index
    t_index = time.perf_counter() - t0
    assert index.n_peaks == 1_747_701 and index.max_width == 501
    rng = np.random.default_rng(0)
    peaks = ref.peaks
    # 100k random 501-bp windows within the span of the reference peaks on each contig
    chroms = rng.choice(np.array(list(index.contigs)), size=100_000)
    starts = np.empty(100_000, dtype=np.int64)
    for c in index.contigs:
        sel = chroms == c
        rs = index.contigs[c][0]
        starts[sel] = rng.integers(rs.min(), rs.max(), size=int(sel.sum()))
    parsed = ParsedPeaks(
        chrom=chroms.astype(object),
        start=starts,
        end=starts + 501,
        valid=np.ones(100_000, dtype=bool),
        n_unparsed=0,
        dropped_contigs={},
        source="synthetic",
        ids=np.array(
            [f"{c}:{s}-{s + 501}" for c, s in zip(chroms, starts, strict=True)], dtype=object
        ),
    )
    t0 = time.perf_counter()
    m = index.match_overlap(parsed, 0.5)
    t_overlap = time.perf_counter() - t0
    t0 = time.perf_counter()
    e = index.match_exact(parsed)
    t_exact = time.perf_counter() - t0
    assert t_overlap < 10.0, f"overlap matching took {t_overlap:.1f} s"
    assert t_exact < 10.0
    # every pair reported really covers >= 50 % of its reference peak
    rs = peaks.loc[m.ref_emb_idx, "start"].to_numpy()
    re_ = peaks.loc[m.ref_emb_idx, "end"].to_numpy()
    ov = np.minimum(parsed.end[m.user_idx], re_) - np.maximum(parsed.start[m.user_idx], rs)
    assert (ov / (re_ - rs) >= 0.5).all() and np.allclose(ov / (re_ - rs), m.overlap, atol=1e-6)
    # exact matches of the real names map onto their own row
    sample = peaks.sample(5000, random_state=0)
    ex = index.match_exact(parse_peaks(sample["name"].tolist()))
    assert ex.ref_emb_idx.tolist() == sample.index.tolist()
    print(
        f"\nR10: index build {t_index:.1f} s; 100k overlap {t_overlap:.2f} s ({len(m)} pairs, "
        f"{m.n_user_mapped} user peaks mapped); 100k exact {t_exact:.2f} s"
    )


@pytest.mark.assets
def test_s6b_da_peak_universe_restricts_views_on_multiome_20(
    multiome_20_dir, peak_table_dir, da_panel_path
):
    """S6b (peak part): ``universe`` = the DA peak list keeps every emitted token in detected ∩ DA."""
    from bloom.prepare import prepare

    ref = ReferenceTables.from_dir(peak_table_dir)
    ds = prepare(atac=ad.read_h5ad(multiome_20_dir / "atac.h5ad"), species="mouse", reference=ref)
    da = FeatureList.from_file(da_panel_path).resolve(ref)
    assert da.kind == "peak" and da.n_known > 2000
    strat = S.Resample(lmax=8192, draws=2, universe=FeatureList.from_file(da_panel_path))
    strat.resolve(ref)
    da_set = set(da.emb_idx.tolist())
    for i in range(ds.n_obs):
        cell: CellFeatures = ds.cell_features(i, "atac")
        detected = set(cell.detected_peak_emb_idx.tolist())
        views = strat.views(cell, S.cell_rng(0, i))
        assert views, f"cell {i} has no DA peak detected"
        for v in views:
            assert (v.kind == PEAK).all()
            toks = set(v.emb_idx.tolist())
            assert toks <= (detected & da_set) and len(v) == min(8192, len(detected & da_set))
