"""Strategies: test-plan S1-S7, S6b (gene + synthetic peak part), S6c, S10, and the
ranking-parity regression of ``TopExpressed`` against ``evaluate.py``'s
``StrategyTopExpressedGenes`` on the ``rna_100`` fixture.

S9 (padding invariance of the model) needs ``Bloom.encode`` (issue #13) and lives with the model
tests; S6b's peak part on the example cells needs ATAC ``prepare`` (issue #18, ``assets`` tier).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from bloom import strategies as S
from bloom.feature_lists import FeatureList, normalize_feature_id, parse_peak_id
from bloom.tokens import GENE, PEAK, CellFeatures, TokenSet, collate

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "data" / "rna_100.h5ad"
RNA_100_REFS_JSON = REPO / "tests" / "data" / "ref" / "rna_100_reference_embeddings.json"
#: rna_100 var_names found in genes.parquet.gene_name (see tests/test_prepare.py).
N_FIXTURE_GENES_MAPPED = 31_859


# -- synthetic cells -----------------------------------------------------------------------------


def _gene_cell(cell_index: int, n: int, seed: int = 0, n_universe: int = 50_000) -> CellFeatures:
    """``n`` detected genes with distinct emb_idx and integer-like values (many ties)."""
    rng = np.random.default_rng(seed + cell_index)
    emb = np.sort(rng.choice(n_universe, size=n, replace=False))
    values = rng.integers(1, 6, size=n).astype(np.float32)
    return CellFeatures.genes(cell_index, emb, values)


def _peak_cell(cell_index: int, n: int, seed: int = 0) -> CellFeatures:
    rng = np.random.default_rng(seed + cell_index)
    return CellFeatures.peaks(cell_index, np.sort(rng.choice(1_000_000, size=n, replace=False)))


class FakeReference:
    """Dict-based stand-in for ``ReferenceTables`` (issue #14)."""

    def __init__(self, symbols: dict[str, int] | None = None, peaks: dict[str, int] | None = None):
        self.symbol_to_emb_idx = symbols or {}
        self.peak_to_emb_idx = peaks or {}


# -- S1 / S2: TopExpressed --------------------------------------------------------------------------


@pytest.mark.unit
def test_s1_top_deterministic_and_ties_by_ascending_emb_idx():
    cell = _gene_cell(0, 300)
    strat = S.TopExpressed(lmax=100)
    outs = [strat.views(cell, S.cell_rng(s, 0))[0].emb_idx for s in (0, 1, 2)]
    assert all(np.array_equal(outs[0], o) for o in outs[1:])
    # rank order: values non-increasing along the view
    pos = np.searchsorted(cell.detected_gene_emb_idx, outs[0])
    vals = cell.gene_values[pos]
    assert np.all(np.diff(vals) <= 0)
    # within a tie group the emb_idx is ascending
    for v in np.unique(vals):
        assert np.all(np.diff(outs[0][vals == v]) > 0)
    # all-equal values: the first lmax emb_idx are picked
    flat = CellFeatures.genes(1, np.arange(10, 400), np.ones(390, np.float32))
    assert S.TopExpressed(lmax=5).views(flat, S.cell_rng(0, 1))[0].emb_idx.tolist() == [
        10,
        11,
        12,
        13,
        14,
    ]


@pytest.mark.unit
def test_s1_top_shorter_than_lmax_and_empty_cell():
    cell = _gene_cell(0, 30)
    view = S.TopExpressed(lmax=100).views(cell, None)
    assert len(view) == 1 and len(view[0]) == 30 and view[0].kind.tolist() == [GENE] * 30
    assert S.TopExpressed(lmax=100).views(CellFeatures.genes(0, [], []), None) == []


@pytest.mark.unit
def test_s2_top_ranking_invariant_to_monotone_normalisation():
    rng = np.random.default_rng(3)
    emb = np.sort(rng.choice(20_000, size=2_000, replace=False))
    counts = rng.poisson(2.0, size=2_000).astype(np.float32) + 1.0
    cpm = np.log1p(counts / counts.sum() * 1e6).astype(np.float32)
    a = S.TopExpressed(lmax=500).views(CellFeatures.genes(0, emb, counts), None)[0].emb_idx
    b = S.TopExpressed(lmax=500).views(CellFeatures.genes(0, emb, cpm), None)[0].emb_idx
    assert np.array_equal(a, b)


@pytest.mark.unit
def test_top_rejects_binary_peaks_but_ranks_counted_peaks():
    with pytest.raises(ValueError, match="binary"):
        S.TopExpressed(lmax=10).views(_peak_cell(0, 20), None)
    emb = np.array([3, 10, 25, 40, 41], dtype=np.int32)
    counts = np.array([1, 5, 2, 5, 1], dtype=np.float32)
    cell = CellFeatures.peaks(0, emb, counts)
    assert cell.peak_values.dtype == np.float32 and cell.modality == "peak"
    v = S.TopExpressed(lmax=3).views(cell, None)[0]
    assert (
        v.emb_idx.tolist() == [10, 40, 25] and (v.kind == PEAK).all()
    )  # ties -> ascending emb_idx
    # universe restriction keeps the values aligned
    r = cell.restrict(np.array([25, 41, 999]))
    assert r.detected_peak_emb_idx.tolist() == [25, 41] and r.peak_values.tolist() == [2.0, 1.0]
    with pytest.raises(ValueError, match="peak_values"):
        CellFeatures.peaks(0, emb, counts[:3])


# -- S3 / S4: Resample -----------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("n", [1, 7, 100, 2_000])
def test_s3_resample_sizes_no_duplicates(n):
    lmax, draws = 100, 5
    cell = _gene_cell(0, n)
    views = S.Resample(lmax=lmax, draws=draws).views(cell, S.cell_rng(0, 0))
    assert len(views) == draws
    for v in views:
        assert len(v) == min(lmax, n)
        assert len(np.unique(v.emb_idx)) == len(v)
        assert np.isin(v.emb_idx, cell.detected_gene_emb_idx).all()
        assert v.kind.tolist() == [GENE] * len(v)


@pytest.mark.unit
def test_s4_resample_seed_semantics():
    cells = [_gene_cell(i, 500) for i in range(6)]
    strat = S.Resample(lmax=64, draws=3)

    def run(indices, seed):
        return {
            i: [v.emb_idx.tolist() for v in strat.views(cells[i], S.cell_rng(seed, i))]
            for i in indices
        }

    a = run(range(6), seed=0)  # one "batch" of 6
    b = run([3], seed=0)  # cell 3 alone
    c = run([5, 4, 3], seed=0)  # different batch composition / order
    assert a[3] == b[3] == c[3]
    assert a[3] != run([3], seed=1)[3]
    # distinct cells with the same seed get distinct streams
    same = CellFeatures.genes(0, cells[0].detected_gene_emb_idx, cells[0].gene_values)
    other = CellFeatures.genes(1, cells[0].detected_gene_emb_idx, cells[0].gene_values)
    assert [v.emb_idx.tolist() for v in strat.views(same, S.cell_rng(0, 0))] != [
        v.emb_idx.tolist() for v in strat.views(other, S.cell_rng(0, 1))
    ]
    # draws are independent samples, not copies
    assert a[0][0] != a[0][1]


@pytest.mark.unit
def test_cell_rng_matches_contract():
    r1 = S.cell_rng(7, 3).integers(0, 1 << 30, 4)
    r2 = np.random.default_rng([7, 3]).integers(0, 1 << 30, 4)
    assert np.array_equal(r1, r2)


# -- S5: Tile --------------------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("n", "lmax", "max_draws"), [(1, 10, 3), (25, 10, 3), (30, 10, 3), (95, 10, 4), (0, 10, 3)]
)
def test_s5_tile_partition(n, lmax, max_draws):
    cell = _gene_cell(0, n) if n else CellFeatures.genes(0, [], [])
    views = S.Tile(lmax=lmax, max_draws=max_draws).views(cell, S.cell_rng(0, 0))
    assert len(views) == (min(max_draws, math.ceil(n / lmax)) if n else 0)
    seen: set[int] = set()
    for v in views:
        assert len(v) <= lmax
        s = set(v.emb_idx.tolist())
        assert len(s) == len(v) and not (s & seen)
        seen |= s
    if n <= max_draws * lmax:
        assert seen == set(cell.detected_gene_emb_idx.tolist())
    else:
        assert len(seen) == max_draws * lmax
    # full chunks before the (possibly short) last one
    assert all(len(v) == lmax for v in views[:-1])


@pytest.mark.unit
def test_tile_is_seeded():
    cell = _gene_cell(0, 50)
    a = S.Tile(lmax=10, max_draws=5).views(cell, S.cell_rng(0, 0))
    b = S.Tile(lmax=10, max_draws=5).views(cell, S.cell_rng(0, 0))
    c = S.Tile(lmax=10, max_draws=5).views(cell, S.cell_rng(1, 0))
    assert [v.emb_idx.tolist() for v in a] == [v.emb_idx.tolist() for v in b]
    assert [v.emb_idx.tolist() for v in a] != [v.emb_idx.tolist() for v in c]


# -- S6 / S6b / S6c: Panel, universe, feature lists ---------------------------------------------------


@pytest.mark.unit
def test_s6_panel_order_and_unknown_reported_once(caplog):
    ref = FakeReference(symbols={"Gad1": 40, "Pvalb": 10, "Sst": 30, "Vip": 20})
    panel = FeatureList.from_ids(["Vip", "Pvalb", "NotAGene", "Gad1", "Sst", "AlsoNot"])
    strat = S.Panel(panel, lmax=8192)
    with pytest.raises(RuntimeError, match="resolve"):
        strat.views(_gene_cell(0, 10), None)
    with caplog.at_level(logging.WARNING, logger="bloom.feature_lists"):
        strat.resolve(ref)
    assert sum("not in the reference" in r.message for r in caplog.records) == 1
    assert "2 of 6" in caplog.text
    assert strat.panel_emb_idx.tolist() == [20, 10, 40, 30]  # panel order
    cell = CellFeatures.genes(
        0, [5, 10, 30, 40, 99], np.ones(5, np.float32)
    )  # Vip (20) not detected
    views = strat.views(cell, None)
    assert len(views) == 1 and views[0].emb_idx.tolist() == [10, 40, 30]
    assert S.Panel(panel, lmax=2).resolve(ref).views(cell, None)[0].emb_idx.tolist() == [10, 40]
    assert strat.views(CellFeatures.genes(1, [5, 99], [1.0, 1.0]), None) == []
    # emb_idx panel: order kept, duplicates dropped
    assert S.Panel(np.array([30, 10, 30, 40])).views(cell, None)[0].emb_idx.tolist() == [30, 10, 40]


@pytest.mark.unit
@pytest.mark.parametrize(
    "make",
    [
        lambda u: S.Resample(50, 3, universe=u),
        lambda u: S.Tile(50, 4, universe=u),
        lambda u: S.TopExpressed(50, universe=u),
    ],
)
def test_s6b_universe_restricts_gene_presets(make):
    cell = _gene_cell(0, 400)
    universe = cell.detected_gene_emb_idx[::3].copy()
    rng = np.random.default_rng(0)
    universe = np.concatenate([universe, rng.choice(10**6, 100) + 100_000])  # plus undetected ids
    allowed = set(cell.detected_gene_emb_idx.tolist()) & set(universe.tolist())
    views = make(universe).views(cell, S.cell_rng(0, 0))
    assert views
    for v in views:
        assert set(v.emb_idx.tolist()) <= allowed
    # Tile covers the whole restricted set (4 x 50 >= |allowed|); Top is truncated to lmax
    if isinstance(make(universe), S.Tile):
        assert set().union(*(set(v.emb_idx.tolist()) for v in views)) == allowed
    if isinstance(make(universe), S.TopExpressed):
        assert len(views) == 1 and len(views[0]) == min(50, len(allowed))


@pytest.mark.unit
def test_s6b_universe_feature_list_for_peaks_and_params():
    ref = FakeReference(peaks={"chr1:100-600": 3, "chr1:1000-1500": 5, "chr2:7-507": 8})
    uni = FeatureList.from_ids(["chr1-100-600", "chr2_7_507", "chr9:1-2"])
    strat = S.Resample(lmax=10, draws=2, universe=uni)
    assert strat.needs_resolve
    with pytest.raises(RuntimeError, match="resolve"):
        strat.views(_peak_cell(0, 5), S.cell_rng(0, 0))
    strat.resolve(ref)
    assert strat.universe_emb_idx.tolist() == [3, 8]
    cell = CellFeatures.peaks(0, [1, 3, 5, 8])
    for v in strat.views(cell, S.cell_rng(0, 0)):
        assert v.emb_idx.tolist() == [3, 8] and v.kind.tolist() == [PEAK, PEAK]
    p = strat.params()
    assert p["strategy"] == "random" and p["draws"] == 2 and p["universe"]["n_ids"] == 3
    assert "Resample(" in repr(strat)


@pytest.mark.unit
def test_s6c_feature_list_parsing(tmp_path):
    text = """# gene panel
Pvalb
  Sst   # trailing comment

Gad1
Pvalb
"""
    p = tmp_path / "genes.txt"
    p.write_text(text)
    fl = FeatureList.from_file(p)
    assert fl.ids == ("Pvalb", "Sst", "Gad1") and fl.kind == "gene" and fl.n_duplicates == 1
    assert fl.source == str(p) and "Sst" in fl and "Vip" not in fl
    res = fl.resolve(FakeReference(symbols={"Pvalb": 1, "Gad1": 2}))
    assert res.emb_idx.tolist() == [1, 2] and res.n_unknown == 1 and res.unknown == ("Sst",)
    assert res.n_known == 2 and res.kind == "gene"

    peaks = FeatureList.from_ids([
        "chr1:100-600",
        "chr1-100-600",
        "chr1_100_600",
        "1:100-600",
        "chr2:5-10 # x",
        "",
        "# c",
    ])
    assert (
        peaks.kind == "peak"
        and peaks.ids == ("chr1:100-600", "chr2:5-10")
        and peaks.n_duplicates == 3
    )
    with pytest.raises(ValueError, match="mixes"):
        FeatureList.from_ids(["Pvalb", "chr1:1-2"])
    assert FeatureList.from_ids([]).kind == "gene" and len(FeatureList.from_ids([])) == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("chr1:100-600", ("chr1", 100, 600)),
        ("chr1-100-600", ("chr1", 100, 600)),
        ("chr1_100_600", ("chr1", 100, 600)),
        ("1:100-600", ("chr1", 100, 600)),
        ("chrX:5-6", ("chrX", 5, 6)),
        ("MT:1-2", ("chrM", 1, 2)),
        ("chrUn_GL456389:1-2", ("chrUn_GL456389", 1, 2)),
        ("Pvalb", None),
        ("Nkx2-1", None),
        ("HLA-A", None),
    ],
)
def test_s6c_peak_id_parsing(text, expected):
    assert parse_peak_id(text) == expected
    kind = "peak" if expected else "gene"
    assert normalize_feature_id(text)[1] == kind


# -- S7: modality routing ----------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "strat", [S.Resample(8, 2), S.Tile(8, 2), S.Panel(np.arange(0, 1_000_000, 7))]
)
def test_s7_strategy_emits_only_the_cell_modality(strat):
    for v in strat.views(_peak_cell(0, 40), S.cell_rng(0, 0)):
        assert (v.kind == PEAK).all()
    for v in strat.views(_gene_cell(0, 40), S.cell_rng(0, 0)):
        assert (v.kind == GENE).all()


@pytest.mark.unit
def test_s7_mixed_modality_cell_is_rejected():
    with pytest.raises(ValueError, match="one modality"):
        CellFeatures(0, np.array([1]), np.array([1.0]), np.array([2]))


# -- factory -----------------------------------------------------------------------------------------


@pytest.mark.unit
def test_build_factory(tmp_path):
    assert isinstance(S.build("random"), S.Resample)
    r = S.build("random", lmax=100, draws=16)
    assert r.lmax == 100 and r.draws == 16
    assert S.get("random") is S.Resample and S.get("TOP") is S.TopExpressed
    with pytest.warns(DeprecationWarning, match="renamed to 'random'"):  # pre-release name
        assert isinstance(S.build("resample", draws=3), S.Resample)
    with pytest.warns(DeprecationWarning):
        assert S.canonical_name("resample") == "random"
    t = S.build(
        "tile", draws=3
    )  # exposed as a preset since v0.1.0 (#31): draws = the cap max_draws
    assert isinstance(t, S.Tile) and t.lmax == 8192 and t.max_draws == 3
    assert t.params() == {"strategy": "tile", "lmax": 8192, "universe": None, "draws": 3}
    assert S.get("tile") is S.Tile and S.build("tile").max_draws == S.DEFAULT_DRAWS
    assert sorted(S.PRESETS) == ["panel", "random", "tile", "top"]
    with pytest.raises(ValueError, match="unknown strategy"):
        S.build("tiles")
    top = S.build("top")
    assert isinstance(top, S.TopExpressed) and top.lmax == 4096
    f = tmp_path / "u.txt"
    f.write_text("Pvalb\nSst\n")
    u = S.build("Top", universe=f)
    assert isinstance(u.universe, FeatureList) and u.needs_resolve
    p = S.build("panel", panel=f, lmax=10)
    assert isinstance(p, S.Panel) and p.needs_resolve and p.lmax == 10
    with pytest.raises(ValueError, match="unknown strategy"):
        S.build("shuffle")
    with pytest.raises(ValueError, match="needs `panel`"):
        S.build("panel")
    with pytest.raises(ValueError, match="only valid"):
        S.build("random", panel=f)
    with pytest.raises(ValueError, match="draws"):
        S.build("top", draws=4)
    assert set(S.PRESETS) == {"random", "top", "panel", "tile"} and S.ALIASES == {
        "resample": "random"
    }
    with pytest.raises(ValueError, match="lmax"):
        S.Resample(lmax=0)


# -- S10: custom strategy through the ABC -------------------------------------------------------------


@pytest.mark.unit
def test_s10_custom_strategy_runs_through_collate():
    class OddEmbIdxOnly(S.ViewStrategy):
        """Keep detected features with odd emb_idx, one view (a 7-line subclass)."""

        name = "odd"

        def sample(self, cell, rng):
            emb = cell.detected_gene_emb_idx
            keep = emb[emb % 2 == 1][: self.lmax]
            return [TokenSet.genes(keep)] if keep.size else []

    strat = OddEmbIdxOnly(lmax=64, universe=np.arange(0, 50_000, 3))
    cells = [_gene_cell(i, 300) for i in range(5)]
    views, cell_index, view_index = [], [], []
    for cell in cells:
        for j, v in enumerate(strat.views(cell, S.cell_rng(0, cell.cell_index))):
            views.append(v)
            cell_index.append(cell.cell_index)
            view_index.append(j)
    batch = collate(views, lmax=strat.lmax, cell_index=cell_index, view_index=view_index)
    assert len(batch) == len(views) > 0 and batch.lmax == 64
    assert (batch.emb_idx[~batch.key_padding_mask()] % 2 == 1).all()
    assert (batch.emb_idx[~batch.key_padding_mask()] % 3 == 0).all()
    assert batch.key_padding_mask().shape == (len(views), 64)
    assert strat.params()["strategy"] == "odd"


@pytest.mark.unit
def test_views_guard_against_overlong_custom_output():
    class Bad(S.ViewStrategy):
        def sample(self, cell, rng):
            return [TokenSet.genes(cell.detected_gene_emb_idx)]

    with pytest.raises(ValueError, match="> lmax"):
        Bad(lmax=5).views(_gene_cell(0, 10), None)


# -- regression on the rna_100 fixture ------------------------------------------------------------------


@pytest.fixture(scope="module")
def rna_100():
    ad = pytest.importorskip("anndata")
    a = ad.read_h5ad(FIXTURE)
    return a


def _col_to_emb(adata, genes_parquet: Path) -> np.ndarray:
    """var_names -> emb_idx (row of genes.parquet; its ``emb_idx`` column when present), -1 unmapped."""
    pd = pytest.importorskip("pandas")
    g = pd.read_parquet(genes_parquet)
    emb = g["emb_idx"].to_numpy() if "emb_idx" in g.columns else np.arange(len(g))
    lut = dict(zip(g["gene_name"], emb.astype(int), strict=True))
    return np.array([lut.get(s, -1) for s in adata.var_names], dtype=np.int64)


def _evaluate_py_top(X: sp.csr_matrix, row: int, col_to_emb: np.ndarray, lmax: int) -> np.ndarray:
    """Replicates ``StrategyTopExpressedGenes.get_idx`` (alleninstitute/bloom@141f0d3): nonzero
    columns in ascending **column** order, valid-masked (mapped only), stable argsort of
    ``-clocal``, leading ``lmax``; then translated to emb_idx."""
    a, b = X.indptr[row], X.indptr[row + 1]
    nz, vals = X.indices[a:b], X.data[a:b]
    order_cols = np.argsort(nz, kind="stable")  # CSR rows of the fixture are sorted; be safe
    nz, vals = nz[order_cols], vals[order_cols]
    valid = col_to_emb[nz] >= 0
    nz, vals = nz[valid], vals[valid]
    order = np.argsort(-vals.astype(np.float32), kind="stable")
    return col_to_emb[nz[order[:lmax]]]


@pytest.mark.regression
def test_top_parity_with_evaluate_py(rna_100, gene_table_paths):
    """TopExpressed(4096) == the token sets evaluate.py fed the model for the ``rna_top4096`` reference.

    The sidecar JSON records the strategy (``StrategyTopExpressedGenes``, ``Lmax 4096``) but no
    per-cell token lists, so the expected sets are derived by replicating that code on the fixture
    with the fixture's ``var_names -> emb_idx`` map (genes.parquet). Ordered equality is
    asserted for all 100 cells; this pins the tie-break semantics (emb_idx vs column order).
    """
    sidecar = json.loads(RNA_100_REFS_JSON.read_text())["arrays"]["rna_top4096"]
    assert sidecar["strategy"]["_target_"].endswith("StrategyTopExpressedGenes")
    lmax = int(sidecar["strategy"]["Lmax"])
    assert lmax == 4096
    X = sp.csr_matrix(rna_100.X)
    col_to_emb = _col_to_emb(rna_100, gene_table_paths[1])
    assert (col_to_emb >= 0).sum() == N_FIXTURE_GENES_MAPPED
    strat = S.TopExpressed(lmax=lmax)
    n_truncated = n_tied_at_cut = 0
    for i in range(rna_100.n_obs):
        a, b = X.indptr[i], X.indptr[i + 1]
        cell = CellFeatures.from_csr_row(i, X.indices[a:b], X.data[a:b], col_to_emb, "gene")
        ours = strat.views(cell, S.cell_rng(0, i))[0].emb_idx
        expected = _evaluate_py_top(X, i, col_to_emb, lmax)
        assert np.array_equal(ours, expected), f"cell {i} ({rna_100.obs_names[i]}) differs"
        if cell.n_detected > lmax:
            n_truncated += 1
            pos = np.searchsorted(cell.detected_gene_emb_idx, ours)
            v = cell.gene_values[pos]
            cut = v[-1]
            n_tied_at_cut += int((cell.gene_values == cut).sum() > (v == cut).sum())
    assert n_truncated > 0  # the tie-break actually matters on this fixture
    assert n_tied_at_cut > 0


# S2 on the fixture itself is deliberately NOT asserted: ``rna_100.layers["counts"]`` is a
# largest-remainder reconstruction from log1p(CPM) (``uns["bloom_phase0"]["logcpm_to_count"]``), so
# e.g. counts 7 and 8 of cell 0 collapse onto one stored value (5.878950) and the two rankings
# legitimately differ. The S2 contract is covered by the synthetic test above.


@pytest.mark.regression
def test_s3_s5_properties_on_fixture_cells_lmax_8192(rna_100):
    """S3/S5 at the paper's lmax on real cells (identity col->emb map; no gene table needed)."""
    X = sp.csr_matrix(rna_100.X)
    ident = np.arange(rna_100.n_vars)
    lmax = 8192
    resample = S.Resample(lmax=lmax, draws=4)
    tile = S.Tile(lmax=lmax, max_draws=4)
    sizes = []
    for i in range(rna_100.n_obs):
        a, b = X.indptr[i], X.indptr[i + 1]
        cell = CellFeatures.from_csr_row(i, X.indices[a:b], X.data[a:b], ident, "gene")
        n = cell.n_detected
        sizes.append(n)
        detected = set(cell.detected_gene_emb_idx.tolist())
        views = resample.views(cell, S.cell_rng(0, i))
        assert len(views) == 4
        for v in views:
            assert len(v) == min(lmax, n) and len(np.unique(v.emb_idx)) == len(v)
            assert set(v.emb_idx.tolist()) <= detected
        again = resample.views(cell, S.cell_rng(0, i))
        assert all(np.array_equal(x.emb_idx, y.emb_idx) for x, y in zip(views, again, strict=True))
        tiles = tile.views(cell, S.cell_rng(0, i))
        assert len(tiles) == min(4, math.ceil(n / lmax))
        union: set[int] = set()
        for t in tiles:
            s = set(t.emb_idx.tolist())
            assert not (s & union)
            union |= s
        assert union == detected
    assert min(sizes) > 0 and max(sizes) < lmax  # every rna_100 cell fits in one 8192 view
