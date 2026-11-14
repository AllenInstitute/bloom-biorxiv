"""`bloom.prepare`: P1-P9 (docs/test-plan.md section 3; P5 = modality subsets / alignment), the input
guards of issues #33 (species mismatch) and #34 (nothing mappable), the human/ortholog path
(E3's mapping-rate check) and a regression of `cell_features()` against the token sets the
Strategies regression test derives from `evaluate.py` on the `rna_100` fixture.

Unit tests use (a) the synthetic 8-gene reference from conftest and (b) a *fixture-derived*
reference: a gene table built from the `rna_100` var_names with a permuted `emb_idx` (so
`emb_idx != col_idx`) and 100 genes left out (so some columns are unmapped). Nothing is
downloaded; the regression / human tests use the real gene table + the shipped ortholog table through
the conftest fixtures (auto-fetched from the bucket, else skipped).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import scipy.sparse as sp
from typer.testing import CliRunner

from bloom import strategies as S
from bloom.cli import app
from bloom.prepare import (
    SCHEMA_VERSION,
    NoMappedFeaturesError,
    PreparedDataset,
    SchemaVersionError,
    SpeciesMismatchError,
    check_species,
    prepare,
    select_values,
)
from bloom.reference import ReferenceTables
from bloom.tokens import CellFeatures

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "data" / "rna_100.h5ad"
N_DROPPED = 100  # fixture-derived reference leaves this many var_names unmapped


# -- fixtures -------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rna_100() -> ad.AnnData:
    return ad.read_h5ad(FIXTURE)


@pytest.fixture(scope="module")
def fixture_reference(tmp_path_factory, rna_100) -> ReferenceTables:
    """Gene table = rna_100 var_names minus the last 100, emb_idx a fixed permutation."""
    d = tmp_path_factory.mktemp("fixture_ref")
    names = rna_100.var_names[:-N_DROPPED].to_numpy()
    rng = np.random.default_rng(16)
    emb = rng.permutation(len(names) + 500)[: len(names)].astype(np.int32)  # sparse, permuted
    g = pd.DataFrame({
        "emb_idx": emb,
        "gene_name": names,
        "gene_id": [f"ENSMUSG{i:011d}" for i in range(len(names))],
        "chrom": "chr1",
        "start": np.arange(len(names)) * 1000,
        "end": np.arange(len(names)) * 1000 + 500,
        "strand": "+",
    }).sort_values("emb_idx")
    pq.write_table(pa.Table.from_pandas(g, preserve_index=False), d / "genes.parquet")
    return ReferenceTables(genes_parquet=d / "genes.parquet")


@pytest.fixture(scope="module")
def synthetic_reference(synthetic_reference_dir) -> ReferenceTables:
    return ReferenceTables.from_dir(synthetic_reference_dir)


def _small_adata(X, var_names=None, obs_names=None, **kw) -> ad.AnnData:
    X = np.asarray(X, dtype=np.float32) if not sp.issparse(X) else X
    n, m = X.shape
    var_names = list(var_names) if var_names is not None else [f"g{j}" for j in range(m)]
    obs_names = list(obs_names) if obs_names is not None else [f"c{i}" for i in range(n)]
    a = ad.AnnData(X=X, obs=pd.DataFrame(index=obs_names), var=pd.DataFrame(index=var_names), **kw)
    return a


SYMBOLS = ["Pvalb", "Gad1", "Gad2", "Sst", "Vip", "Xkr4", "Gm1992", "Notagene"]  # synthetic ref


def _detection_equal(ds_a: PreparedDataset, ds_b: PreparedDataset) -> None:
    Da, ea = ds_a.detection_matrix()
    Db, eb = ds_b.detection_matrix()
    assert np.array_equal(ea, eb)
    assert Da.shape == Db.shape and (Da != Db).nnz == 0
    assert np.array_equal(ds_a.rna.var["bloom_emb_idx"], ds_b.rna.var["bloom_emb_idx"])
    assert np.array_equal(
        ds_a.rna.obs["bloom_n_genes_detected"], ds_b.rna.obs["bloom_n_genes_detected"]
    )


# -- P1 value-type independence ---------------------------------------------------------------------


@pytest.mark.unit
def test_p1_detection_independent_of_value_type(rna_100, fixture_reference):
    counts = rna_100.layers["counts"]
    logcpm = rna_100.X
    assert ((sp.csr_matrix(counts) > 0) != (sp.csr_matrix(logcpm) > 0)).nnz == 0
    rng = np.random.default_rng(0)
    floats = sp.csr_matrix(counts, dtype=np.float32, copy=True)
    floats.data = rng.uniform(0.01, 1e4, floats.nnz).astype(np.float32)
    runs = []
    for M in (counts, logcpm, floats):
        a = ad.AnnData(X=M, obs=rna_100.obs.copy(), var=rna_100.var.copy())
        runs.append(prepare(a, species="mouse", reference=fixture_reference))
    for other in runs[1:]:
        _detection_equal(runs[0], other)
    rep = runs[0].report["rna"]
    assert rep["n_var"] == rna_100.n_vars
    assert rep["n_mapped"] == rna_100.n_vars - N_DROPPED
    assert rep["value_source"] == "X"
    assert rep["n_empty_cells"] == 0
    assert 0.9 < rep["detected_mass_mapped"] <= 1.0


@pytest.mark.unit
def test_p1_detection_matches_independent_computation(rna_100, fixture_reference):
    ds = prepare(rna_100, species="mouse", reference=fixture_reference)
    assert ds.report["rna"]["value_source"] == "layers['counts']"  # fixture carries counts
    emb = ds.rna.var["bloom_emb_idx"].to_numpy()
    X = sp.csr_matrix(rna_100.layers["counts"])
    for i in (0, 17, 99):
        cols = X.indices[X.indptr[i] : X.indptr[i + 1]]
        vals = X.data[X.indptr[i] : X.indptr[i + 1]]
        expected = np.unique(emb[cols[(vals > 0) & (emb[cols] >= 0)]])
        cf = ds.cell_features(i)
        assert np.array_equal(cf.detected_gene_emb_idx, expected)
        assert cf.n_detected == ds.rna.obs["bloom_n_genes_detected"].iloc[i]
        assert np.all(np.diff(cf.detected_gene_emb_idx) > 0)  # sorted, unique
    # R6-style check of the report
    per_col = np.asarray((X > 0).sum(axis=0)).ravel()
    assert ds.report["rna"]["detected_mass_mapped"] == pytest.approx(
        per_col[emb >= 0].sum() / per_col.sum()
    )


# -- P2 negative values -------------------------------------------------------------------------------


@pytest.mark.unit
def test_p2_negative_values_raise(synthetic_reference):
    X = np.array([[1.0, -0.5, 2.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    a = _small_adata(X, var_names=SYMBOLS[:3])
    with pytest.raises(ValueError, match="scaled/centered data cannot be binarized"):
        prepare(a, species="mouse", reference=synthetic_reference)


# -- P3 layer precedence ------------------------------------------------------------------------------


def _layered_adata() -> ad.AnnData:
    """X, layers['counts'], layers['other'], raw.X with four distinct sparsity patterns."""
    n, m = 3, 6
    base = np.zeros((n, m), dtype=np.float32)
    X = base.copy()
    X[0, 0] = 1
    counts = base.copy()
    counts[0, 1] = 1
    other = base.copy()
    other[0, 2] = 1
    raw = base.copy()
    raw[0, 3] = 1
    a = _small_adata(X, var_names=SYMBOLS[:m])
    a.layers["counts"] = sp.csr_matrix(counts)
    a.layers["other"] = other
    a.raw = _small_adata(raw, var_names=SYMBOLS[:m])
    return a


def _first_cell_detected(ds: PreparedDataset) -> list[str]:
    cf = ds.cell_features(0)
    sym = ds.rna.var.set_index("bloom_emb_idx")["bloom_symbol"]
    return sorted(sym.loc[cf.detected_gene_emb_idx].tolist())


@pytest.mark.unit
def test_p3_layer_precedence_recorded(synthetic_reference):
    a = _layered_adata()
    cases = [
        ({"layer": "other"}, "layers['other']", "Gad2"),
        ({}, "layers['counts']", "Gad1"),
    ]
    for kw, src, gene in cases:
        ds = prepare(a, species="mouse", reference=synthetic_reference, **kw)
        assert ds.rna.uns["bloom"]["value_source"] == src
        assert ds.report["rna"]["value_source"] == src
        assert _first_cell_detected(ds) == [gene]
    # no counts layer -> raw.X
    b = _layered_adata()
    del b.layers["counts"]
    ds = prepare(b, species="mouse", reference=synthetic_reference)
    assert ds.rna.uns["bloom"]["value_source"] == "raw.X"
    assert _first_cell_detected(ds) == ["Sst"]
    # no counts, no raw -> X
    b.raw = None
    ds = prepare(b, species="mouse", reference=synthetic_reference)
    assert ds.rna.uns["bloom"]["value_source"] == "X"
    assert _first_cell_detected(ds) == ["Pvalb"]
    # explicit layer wins over everything, missing layer is an error
    assert select_values(_layered_adata(), "counts")[1] == "layers['counts']"
    with pytest.raises(KeyError, match="not found"):
        prepare(_layered_adata(), species="mouse", reference=synthetic_reference, layer="nope")


# -- P4 dense / CSR / CSC / backed --------------------------------------------------------------------


@pytest.mark.unit
def test_p4_storage_formats_identical(rna_100, fixture_reference, tmp_path):
    counts = sp.csr_matrix(rna_100.layers["counts"])
    obs, var = rna_100.obs.copy(), rna_100.var.copy()
    variants = {
        "csr": ad.AnnData(X=counts.copy(), obs=obs, var=var),
        "csc": ad.AnnData(X=counts.tocsc(), obs=obs, var=var),
        "dense": ad.AnnData(X=counts.toarray(), obs=obs, var=var),
    }
    # an unsorted CSR variant: reverse the column order inside every row
    unsorted = counts.copy()
    for i in range(unsorted.shape[0]):
        lo, hi = unsorted.indptr[i], unsorted.indptr[i + 1]
        unsorted.indices[lo:hi] = unsorted.indices[lo:hi][::-1]
        unsorted.data[lo:hi] = unsorted.data[lo:hi][::-1]
    unsorted.has_sorted_indices = False
    variants["csr_unsorted"] = ad.AnnData(X=unsorted, obs=obs, var=var)
    p = tmp_path / "backed.h5ad"
    variants["csr"].write_h5ad(p)
    variants["backed"] = ad.read_h5ad(p, backed="r")
    runs = {
        k: prepare(v, species="mouse", reference=fixture_reference) for k, v in variants.items()
    }
    for k, ds in runs.items():
        _detection_equal(runs["csr"], ds)
        assert ds.rna.X.has_sorted_indices, k
        assert ds.rna.X.dtype == np.float32
        assert (ds.rna.X != runs["csr"].rna.X).nnz == 0, k
        for i in (0, 50):
            assert np.array_equal(
                ds.cell_features(i).detected_gene_emb_idx,
                runs["csr"].cell_features(i).detected_gene_emb_idx,
            )
            assert np.array_equal(
                ds.cell_features(i).gene_values, runs["csr"].cell_features(i).gene_values
            )


# -- P6 duplicate obs_names ---------------------------------------------------------------------------


@pytest.mark.unit
def test_p6_duplicate_obs_names_raise(synthetic_reference):
    a = _small_adata(np.ones((3, 3), np.float32), var_names=SYMBOLS[:3], obs_names=["a", "b", "a"])
    with pytest.raises(ValueError, match="obs_names must be unique"):
        prepare(a, species="mouse", reference=synthetic_reference)


# -- P7 empty cells -----------------------------------------------------------------------------------


@pytest.mark.unit
def test_p7_empty_cell_kept_and_reported(synthetic_reference):
    X = np.array(
        [[1, 0, 0, 2], [0, 0, 0, 0], [0, 0, 5, 0]], dtype=np.float32
    )  # cell 1 empty, cell 2 only on an unmapped gene
    a = _small_adata(X, var_names=["Pvalb", "Gad1", "Notagene", "Sst"])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ds = prepare(a, species="mouse", reference=synthetic_reference)
    assert ds.rna.n_obs == 3
    assert ds.rna.obs["bloom_n_genes_detected"].tolist() == [2, 0, 0]
    assert ds.report["rna"]["n_empty_cells"] == 2
    assert any("2 cell(s) have no detected mapped genes" in str(x.message) for x in w)
    cf = ds.cell_features(1)
    assert cf.n_detected == 0 and cf.detected_gene_emb_idx.dtype == np.int32
    assert isinstance(cf, CellFeatures)
    # strategies emit no view for an empty cell; embed() turns that into a NaN row
    assert S.TopExpressed(lmax=10).views(cf, S.cell_rng(0, 1)) == []
    # median < 200 and low mapping warnings are also present on this tiny example
    msgs = " ".join(ds.report["rna"]["warnings"])
    assert "median detected" in msgs


@pytest.mark.unit
def test_low_mapping_warning_names_unmapped_ids(synthetic_reference):
    X = np.array([[0, 3, 3, 3], [1, 2, 2, 2]], dtype=np.float32)
    a = _small_adata(X, var_names=["Pvalb", "ENSG00000999999", "foo", "foo"])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ds = prepare(a, species="mouse", reference=synthetic_reference)
    rep = ds.report["rna"]
    assert rep["low_mapping"] is True
    assert rep["detected_mass_mapped"] == pytest.approx(1 / 7)
    assert rep["unmapped_examples"][0] == "foo"
    low = [str(x.message) for x in w if "detected gene mass" in str(x.message)]
    assert low and "foo" in low[0]
    assert any("more than once" in str(x.message) for x in w)  # duplicate ids (R3) surfaced


@pytest.mark.unit
def test_ensembl_ids_and_gene_col(synthetic_reference):
    X = np.eye(3, dtype=np.float32)
    a = _small_adata(X, var_names=["x", "y", "z"])
    a.var["ensembl"] = ["ENSMUSG00000005716.9", "ENSMUSG00000070880", "ENSMUSG00000000000"]
    ds = prepare(a, species="mouse", reference=synthetic_reference, gene_col="ensembl")
    assert ds.rna.var["bloom_emb_idx"].tolist() == [25359, 2498, -1]
    assert ds.rna.var["bloom_mapped_via"].tolist() == ["gene_id", "gene_id", "none"]
    assert ds.rna.var["bloom_symbol"].tolist() == ["Pvalb", "Gad1", ""]
    assert ds.rna.uns["bloom"]["gene_col"] == "ensembl"
    with pytest.raises(KeyError, match="gene_col"):
        prepare(a, species="mouse", reference=synthetic_reference, gene_col="missing")


@pytest.mark.unit
def test_bad_species_and_nothing_given_rejected(synthetic_reference):
    a = _small_adata(np.ones((1, 2), np.float32), var_names=SYMBOLS[:2])
    with pytest.raises(ValueError, match="species"):
        prepare(a, species="rat", reference=synthetic_reference)
    with pytest.raises(ValueError, match="at least one"):
        prepare(species="mouse", reference=synthetic_reference)


# -- P8 bundle round-trip -----------------------------------------------------------------------------


@pytest.mark.unit
def test_p8_bundle_round_trip(rna_100, fixture_reference, tmp_path):
    ds = prepare(FIXTURE, species="mouse", reference=fixture_reference)
    out = ds.write(tmp_path / "bundle")
    assert (out / "rna.h5ad").exists() and (out / "manifest.json").exists()
    man = json.loads((out / "manifest.json").read_text())
    assert man["schema_version"] == SCHEMA_VERSION == "1.0"
    assert man["modalities"] == ["rna"]
    assert man["species"] == "mouse"
    assert man["reference_id"] == fixture_reference.reference_id
    assert man["source"]["rna"]["path"] == str(FIXTURE)
    assert len(man["source"]["rna"]["sha256"]) == 64
    assert man["mapping"]["rna"]["n_mapped"] == rna_100.n_vars - N_DROPPED
    back = PreparedDataset.read(out)
    _detection_equal(ds, back)
    assert (back.rna.X != ds.rna.X).nnz == 0 and back.rna.X.dtype == np.float32
    assert back.rna.var["bloom_mapped_via"].tolist() == ds.rna.var["bloom_mapped_via"].tolist()
    assert back.rna.var["bloom_symbol"].tolist() == ds.rna.var["bloom_symbol"].tolist()
    assert back.report == ds.report
    assert back.species == ds.species and back.reference_id == ds.reference_id
    assert back.rna.uns["bloom"]["value_source"] == "layers['counts']"
    assert set(back.rna.uns["bloom"]["mapping"]) >= {"n_var", "n_mapped", "detected_mass_mapped"}
    assert back.rna.obs.index.equals(rna_100.obs.index)
    assert (
        back.rna.obs["AIT33_subclass"].astype(str).tolist()
        == rna_100.obs["AIT33_subclass"].astype(str).tolist()
    )
    for i in range(0, 100, 25):
        x, y = ds.cell_features(i), back.cell_features(i)
        assert np.array_equal(x.detected_gene_emb_idx, y.detected_gene_emb_idx)
        assert np.array_equal(x.gene_values, y.gene_values)


# -- P9 schema guard ----------------------------------------------------------------------------------


@pytest.mark.unit
def test_p9_schema_major_guard(synthetic_reference, tmp_path):
    a = _small_adata(np.ones((2, 2), np.float32), var_names=SYMBOLS[:2])
    ds = prepare(a, species="mouse", reference=synthetic_reference)
    out = ds.write(tmp_path / "b")
    man = json.loads((out / "manifest.json").read_text())
    man["schema_version"] = "2.0"
    (out / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(SchemaVersionError, match=r"2\.0"):
        PreparedDataset.read(out)
    # a minor bump is accepted; a bad h5ad-level version is also refused
    man["schema_version"] = "1.7"
    (out / "manifest.json").write_text(json.dumps(man))
    assert PreparedDataset.read(out).rna.n_obs == 2
    b = ad.read_h5ad(out / "rna.h5ad")
    b.uns["bloom"]["schema_version"] = "3.0"
    b.write_h5ad(out / "rna.h5ad")
    with pytest.raises(SchemaVersionError, match=r"3\.0"):
        PreparedDataset.read(out)
    with pytest.raises(FileNotFoundError):
        PreparedDataset.read(tmp_path / "nothing")


# -- CLI ------------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_cli_prepare_rna(synthetic_reference_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("BLOOM_HOME", str(synthetic_reference_dir.parents[1]))
    a = _small_adata(
        np.array([[1, 0, 2], [0, 1, 0]], np.float32), var_names=["Pvalb", "Gad1", "Sst"]
    )
    src = tmp_path / "in.h5ad"
    a.write_h5ad(src)
    out = tmp_path / "prepared"
    res = CliRunner().invoke(
        app, ["prepare", "--rna", str(src), "--species", "mouse", "-o", str(out)]
    )
    assert res.exit_code == 0, res.output
    assert "mapped 3/3 ids (100.0%)" in res.output
    assert "wrote" in res.output and (out / "rna.h5ad").exists()
    back = PreparedDataset.read(out)
    assert back.rna.obs["bloom_n_genes_detected"].tolist() == [2, 1]
    # explicit --reference-dir, --layer error path, atac placeholder
    res = CliRunner().invoke(
        app,
        [
            "prepare",
            "--rna",
            str(src),
            "--species",
            "mouse",
            "-o",
            str(out),
            "--layer",
            "nope",
            "--reference-dir",
            str(synthetic_reference_dir),
        ],
    )
    assert res.exit_code == 1 and "not found" in res.output
    res = CliRunner().invoke(app, ["prepare", "--species", "mouse", "-o", str(out)])
    assert res.exit_code == 2
    res = CliRunner().invoke(app, ["prepare", "--rna", str(src), "-o", str(out)])
    assert res.exit_code == 2 and "--species" in res.output  # species required with --rna


# -- regression: cell_features() vs the evaluate.py-derived token sets (real gene table) ---------------


def _evaluate_py_top(X: sp.csr_matrix, row: int, col_to_emb: np.ndarray, lmax: int) -> np.ndarray:
    """Same derivation as tests/test_strategies.py::_evaluate_py_top (StrategyTopExpressedGenes)."""
    try:
        from test_strategies import _evaluate_py_top as helper  # pytest prepends tests/ to sys.path

        return helper(X, row, col_to_emb, lmax)
    except ImportError:  # pragma: no cover - fallback when tests/ is not importable
        a, b = X.indptr[row], X.indptr[row + 1]
        nz, vals = X.indices[a:b], X.data[a:b]
        order_cols = np.argsort(nz, kind="stable")
        nz, vals = nz[order_cols], vals[order_cols]
        valid = col_to_emb[nz] >= 0
        nz, vals = nz[valid], vals[valid]
        order = np.argsort(-vals.astype(np.float32), kind="stable")
        return col_to_emb[nz[order[:lmax]]]


#: rna_100 var_names (31,899 symbols) found in genes.parquet.gene_name (measured on the real table);
#: all 18,509 *detected* fixture genes that map are among them, so the token sets below are unchanged.
N_FIXTURE_GENES_MAPPED = 31_859


@pytest.fixture(scope="module")
def real_gene_reference(gene_table_dir) -> ReferenceTables:
    return ReferenceTables(
        genes_parquet=gene_table_dir / "genes.parquet",
        genes_safetensors=gene_table_dir / "genes.safetensors",
    )


@pytest.mark.regression
def test_cell_features_reproduce_evaluate_py_token_sets(rna_100, real_gene_reference):
    """prepare(rna_100) -> cell_features(i) -> TopExpressed(4096) == evaluate.py's token sets,
    using X (log1p CPM, what the reference embeddings were computed from) as the value source."""
    a = ad.AnnData(X=rna_100.X.copy(), obs=rna_100.obs.copy(), var=rna_100.var.copy())
    ds = prepare(a, species="mouse", reference=real_gene_reference)
    rep = ds.report["rna"]
    assert rep["value_source"] == "X"
    assert rep["n_mapped"] == N_FIXTURE_GENES_MAPPED
    assert rep["n_exact"] == N_FIXTURE_GENES_MAPPED and rep["n_normalised"] == 0
    assert rep["n_empty_cells"] == 0
    X = sp.csr_matrix(rna_100.X)
    col_to_emb = ds.col_to_emb_idx()
    strat = S.TopExpressed(lmax=4096)
    for i, cell in enumerate(ds.cells()):
        assert cell.cell_index == i
        lo, hi = X.indptr[i], X.indptr[i + 1]
        direct = CellFeatures.from_csr_row(i, X.indices[lo:hi], X.data[lo:hi], col_to_emb, "gene")
        assert np.array_equal(cell.detected_gene_emb_idx, direct.detected_gene_emb_idx)
        assert np.array_equal(cell.gene_values, direct.gene_values)
        ours = strat.views(cell, S.cell_rng(0, i))[0].emb_idx
        assert np.array_equal(ours, _evaluate_py_top(X, i, col_to_emb, 4096)), f"cell {i}"
        assert cell.n_detected == ds.rna.obs["bloom_n_genes_detected"].iloc[i]


@pytest.mark.regression
def test_human_path_uppercased_fixture_maps_via_orthologs(
    rna_100, gene_table_dir, shipped_orthologs_tsv
):
    """E3-adjacent: upper-cased rna_100 var_names as HGNC symbols through the shipped ortholog table."""
    ref = ReferenceTables(
        genes_parquet=gene_table_dir / "genes.parquet",
        genes_safetensors=gene_table_dir / "genes.safetensors",
        orthologs_tsv=shipped_orthologs_tsv,
    )
    a = ad.AnnData(X=rna_100.X.copy(), obs=rna_100.obs.copy(), var=rna_100.var.copy())
    a.var_names = a.var_names.str.upper()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = prepare(a, species="human", reference=ref)
    rep = ds.report["rna"]
    via = ds.rna.var["bloom_mapped_via"]
    mapped = ds.rna.var["bloom_emb_idx"] >= 0
    assert set(via[mapped].unique()) == {"ortholog"}
    assert rep["n_ortholog"] == rep["n_mapped"] > 10_000
    assert rep["detected_mass_mapped"] > 0.5
    assert rep["n_empty_cells"] == 0
    # the symbol written is the mouse gene-table symbol, e.g. PVALB -> Pvalb
    sym = ds.rna.var.loc["PVALB", "bloom_symbol"]
    assert sym == "Pvalb"
    assert ds.rna.var.loc["PVALB", "bloom_emb_idx"] == 25359
    print(
        f"human/ortholog path: mapped {rep['n_mapped']}/{rep['n_var']} ids "
        f"({100 * rep['mapping_rate']:.1f}%), detected mass mapped "
        f"{100 * rep['detected_mass_mapped']:.1f}%, low_mapping={rep['low_mapping']}"
    )


# ==================================================================================================
# ATAC path (#18): P5 modality subsets / alignment, prepared atac.h5ad schema, bundle round-trip
# ==================================================================================================
# synthetic reference peaks (conftest SYNTH_PEAKS): emb 0 chr1:3003635-3004136, 1 chr1:3007195-3007696,
# 7 chr2:100000-100501, 12 chrX:5000-5501
REF_PEAKS = ["chr1:3003635-3004136", "chr1:3007195-3007696", "chr2:100000-100501", "chrX:5000-5501"]
REF_EMB = [0, 1, 7, 12]


def _atac_adata(obs_names=("c0", "c1", "c2"), extra=(), X=None) -> ad.AnnData:
    """3 cells x (4 reference peaks + extra ids); counts chosen so every reference peak is hit."""
    names = [*REF_PEAKS, *extra]
    if X is None:
        X = np.zeros((len(obs_names), len(names)), dtype=np.float32)
        X[0, 0] = 3
        X[0, 2] = 1
        X[1, 1] = 2
        X[1, 3] = 1
        X[2, 0] = 1
        X[2, 1] = 1
        X[2, 2] = 1
        X[2, 3] = 1
        for j in range(4, len(names)):
            X[j % len(obs_names), j] = 1
    a = _small_adata(X, var_names=names, obs_names=list(obs_names))
    a.obs["cell_type"] = [f"t{i}" for i in range(len(obs_names))]
    a.X = sp.csr_matrix(a.X).astype(np.uint32)  # counts as ATAC callers write them
    return a


@pytest.mark.unit
def test_p5_atac_alone_exact(synthetic_reference):
    a = _atac_adata()
    ds = prepare(atac=a, species="mouse", reference=synthetic_reference)
    assert ds.modalities == ("atac",) and ds.rna is None and ds.n_obs == 3
    p = ds.atac
    assert p.var_names.tolist() == REF_PEAKS  # reference peak names, hit ones only
    assert p.var["bloom_emb_idx"].tolist() == REF_EMB and p.var["bloom_emb_idx"].dtype == np.int32
    assert p.var["n_user_peaks"].tolist() == [1, 1, 1, 1]
    assert p.obs["bloom_n_peaks_detected"].tolist() == [2, 2, 4]
    assert p.obs["cell_type"].tolist() == ["t0", "t1", "t2"]  # user obs kept
    assert sp.isspmatrix_csr(p.X) and p.X.dtype == np.float32
    np.testing.assert_array_equal(p.X.toarray(), a.X.toarray().astype(np.float32))  # 1:1 -> counts
    rep = ds.report["atac"]
    assert rep["peak_match"] == "exact" and rep["peak_match_requested"] == "auto"
    assert rep["exact_rate"] == pytest.approx(1.0) and rep["n_var"] == 4 and rep["n_mapped"] == 4
    assert rep["mapping_rate"] == pytest.approx(1.0)
    assert rep["detected_mass_mapped"] == pytest.approx(1.0)
    assert (
        rep["n_ref_peaks_hit"] == 4 and rep["n_empty_cells"] == 0 and rep["dropped_contigs"] == {}
    )
    u = p.uns["bloom"]
    assert u["modality"] == "atac" and u["peak_match"] == "exact" and u["n_user_peaks"] == 4
    assert u["n_user_peaks_mapped"] == 4 and u["n_ref_peaks_hit"] == 4 and "min_overlap" not in u
    assert u["schema_version"] == SCHEMA_VERSION
    # per-cell features: peaks in emb_idx space with the counts as values (for `top`)
    cf = ds.cell_features(0, "atac")
    assert cf.modality == "peak" and cf.detected_peak_emb_idx.tolist() == [0, 7]
    assert cf.peak_values.tolist() == [3.0, 1.0]
    D, emb = ds.detection_matrix("atac")
    assert emb.tolist() == REF_EMB and D.toarray().astype(int).tolist() == [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 1, 1, 1],
    ]
    txt = ds.report_text()
    assert "[atac] 3 cells x 4 user peaks" in txt and "peak match exact" in txt
    # top on ATAC ranks by count (ties -> ascending emb_idx)
    v = S.TopExpressed(lmax=1).views(cf, None)[0]
    assert v.emb_idx.tolist() == [0] and v.kind.tolist() == [2]


@pytest.mark.unit
def test_p5_atac_overlap_auto_and_modes(synthetic_reference):
    # 4 exact + one 1-bp-shifted copy of peak 0 + a non-canonical contig + an off-reference peak
    extra = ["chr1:3003636-3004137", "chrUn_GL456389:1-500", "chr3:1000-1501"]
    a = _atac_adata(extra=extra)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ds = prepare(atac=a, species="mouse", reference=synthetic_reference)
    rep = ds.report["atac"]
    # exact rate 4/6 valid peaks < 0.95 -> auto falls back to overlap
    assert rep["peak_match"] == "overlap" and rep["exact_rate"] == pytest.approx(4 / 6)
    assert rep["min_overlap"] == pytest.approx(0.5)
    assert rep["n_valid"] == 6 and rep["n_dropped_contigs"] == 1
    assert rep["dropped_contigs"] == {"chrUn_GL456389": 1}
    assert rep["n_mapped"] == 5 and rep["mapping_rate"] == pytest.approx(5 / 6)
    assert rep["unmapped_examples"] == ["chr3:1000-1501"]
    p = ds.atac
    assert p.var_names.tolist() == REF_PEAKS and p.var["n_user_peaks"].tolist() == [2, 1, 1, 1]
    # OR / sum: cell 1 detects the shifted copy (col 4) -> reference peak 0 is detected there too
    assert p.obs["bloom_n_peaks_detected"].tolist() == [2, 3, 4]
    assert p.X[1, 0] == pytest.approx(1.0) and p.X[0, 0] == pytest.approx(3.0)
    assert p.uns["bloom"]["min_overlap"] == pytest.approx(0.5)
    assert p.uns["bloom"]["peak_match"] == "overlap"
    assert any("non-canonical contig" in str(x.message) for x in w)
    # explicit exact: the shifted copy is unmapped, n_user_peaks back to 1
    ds_e = prepare(atac=a, species="mouse", reference=synthetic_reference, peak_match="exact")
    assert (
        ds_e.report["atac"]["n_mapped"] == 4 and ds_e.atac.var["n_user_peaks"].tolist() == [1] * 4
    )
    assert ds_e.atac.obs["bloom_n_peaks_detected"].tolist() == [2, 2, 4]
    # explicit overlap with a stricter threshold than 500/501 still maps the shifted copy
    ds_o = prepare(
        atac=a,
        species="mouse",
        reference=synthetic_reference,
        peak_match="overlap",
        min_overlap=0.99,
    )
    assert ds_o.report["atac"]["n_mapped"] == 5
    with pytest.raises(ValueError, match="peak_match"):
        prepare(atac=a, species="mouse", reference=synthetic_reference, peak_match="fuzzy")


@pytest.mark.unit
def test_p5_atac_var_columns_and_input_errors(synthetic_reference, synthetic_reference_dir):
    a = _atac_adata()
    a.var["chrom"] = ["1", "chr1", "2", "X"]
    a.var["start"] = [3003635, 3007195, 100000, 5000]
    a.var["end"] = [3004136, 3007696, 100501, 5501]
    a.var_names = ["p0", "p1", "p2", "p3"]
    ds = prepare(atac=a, species="mouse", reference=synthetic_reference)
    assert (
        ds.report["atac"]["source"] == "var[chrom,start,end]" and ds.report["atac"]["n_mapped"] == 4
    )
    # human + ATAC is refused (peaks are mm10)
    with pytest.raises(ValueError, match="mm10"):
        prepare(atac=_atac_adata(), species="human", reference=synthetic_reference)
    # nothing parses as a coordinate
    b = _small_adata(np.ones((2, 2), np.float32), var_names=["Pvalb", "Gad1"])
    with pytest.raises(ValueError, match="no ATAC var identifier parses"):
        prepare(atac=b, species="mouse", reference=synthetic_reference)
    # negative values name the ATAC matrix
    c = _atac_adata()
    c.X = sp.csr_matrix(np.array([[1, -1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1]], dtype=np.float32))
    with pytest.raises(ValueError, match="ATAC matrix contains negative"):
        prepare(atac=c, species="mouse", reference=synthetic_reference)
    # duplicate obs_names
    d = _atac_adata(obs_names=("a", "a", "b"))
    with pytest.raises(ValueError, match="obs_names must be unique"):
        prepare(atac=d, species="mouse", reference=synthetic_reference)
    # reference without the peak table
    bare = ReferenceTables(genes_parquet=synthetic_reference_dir / "genes.parquet")
    with pytest.raises(FileNotFoundError, match="bloom download --what peaks"):
        prepare(atac=_atac_adata(), species="mouse", reference=bare)


@pytest.mark.unit
def test_p5_both_modalities_align_on_obs_names(synthetic_reference, tmp_path):
    rna = _small_adata(
        np.ones((4, 3), np.float32), var_names=SYMBOLS[:3], obs_names=["c0", "c1", "c2", "c3"]
    )
    rna.obs["cell_type"] = ["r0", "r1", "r2", "r3"]
    rna.obs["rna_only_col"] = [1, 2, 3, 4]
    atac = _atac_adata(obs_names=("c3", "c2", "c4"))  # c3/c2 shared (reversed order), c4 ATAC-only
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ds = prepare(rna=rna, atac=atac, species="mouse", reference=synthetic_reference)
    assert ds.modalities == ("rna", "atac") and ds.n_obs == 2
    assert ds.obs_names.tolist() == ["c2", "c3"]  # RNA order
    assert ds.atac.obs_names.tolist() == ["c2", "c3"] and ds.rna.obs_names.tolist() == ["c2", "c3"]
    assert ds.alignment == {
        "n_rna": 4,
        "n_atac": 3,
        "n_common": 2,
        "n_rna_only": 2,
        "n_atac_only": 1,
        "keep_unmatched": False,
    }
    assert ds.unmatched == {}
    assert any("2 RNA-only and 1 ATAC-only" in str(x.message) for x in w)
    # the ATAC rows followed their cells: c3 was row 0 of the ATAC file (detects peaks 0 and 7)
    assert ds.atac.obs["bloom_n_peaks_detected"].tolist() == [2, 2]
    assert ds.cell_features(1, "atac").detected_peak_emb_idx.tolist() == [0, 7]
    assert "[alignment] 2 cells in both files (rna 4, atac 3); rna-only 2, atac-only 1 dropped" in (
        ds.report_text()
    )
    # keep_unmatched: remainders kept as one-modality datasets and round-tripped through the bundle
    ds2 = prepare(
        rna=rna, atac=atac, species="mouse", reference=synthetic_reference, keep_unmatched=True
    )
    assert sorted(ds2.unmatched) == ["atac", "rna"]
    assert ds2.unmatched["rna"].obs_names.tolist() == ["c0", "c1"]
    assert ds2.unmatched["atac"].obs_names.tolist() == ["c4"]
    sub = ds2.single("atac", unmatched=True)
    assert (
        sub.modalities == ("atac",) and sub.n_obs == 1 and sub.report["atac"] == ds2.report["atac"]
    )
    out = ds2.write(tmp_path / "bundle")
    assert {p.name for p in out.iterdir()} == {
        "manifest.json",
        "rna.h5ad",
        "atac.h5ad",
        "rna_only.h5ad",
        "atac_only.h5ad",
    }
    man = json.loads((out / "manifest.json").read_text())
    assert man["modalities"] == ["rna", "atac"] and man["alignment"]["keep_unmatched"] is True
    assert man["unmatched_files"] == {"rna": "rna_only.h5ad", "atac": "atac_only.h5ad"}
    back = PreparedDataset.read(out)
    assert back.modalities == ("rna", "atac") and back.alignment == ds2.alignment
    assert back.obs_names.tolist() == ["c2", "c3"]
    assert back.atac.var["bloom_emb_idx"].tolist() == REF_EMB
    assert back.atac.var["n_user_peaks"].dtype == np.int32
    assert (back.atac.X != ds2.atac.X).nnz == 0
    assert back.unmatched["rna"].obs_names.tolist() == ["c0", "c1"]
    assert back.unmatched["atac"].obs_names.tolist() == ["c4"]
    assert back.report["atac"] == ds2.report["atac"]
    # disjoint obs_names: error naming ids of both files
    far = _atac_adata(obs_names=("x0", "x1", "x2"))
    with pytest.raises(ValueError, match=r"share no obs_names.*c0.*x0"):
        prepare(rna=rna, atac=far, species="mouse", reference=synthetic_reference)


@pytest.mark.unit
def test_p8_atac_bundle_round_trip(synthetic_reference, tmp_path):
    ds = prepare(
        atac=_atac_adata(extra=["chr1:3003636-3004137"]),
        species="mouse",
        reference=synthetic_reference,
    )
    out = ds.write(tmp_path / "b")
    assert not (out / "rna.h5ad").exists() and (out / "atac.h5ad").exists()
    back = PreparedDataset.read(out)
    assert back.modalities == ("atac",) and back.species == "mouse"
    assert back.atac.var_names.tolist() == ds.atac.var_names.tolist()
    assert back.atac.var["bloom_emb_idx"].tolist() == REF_EMB
    assert back.atac.var["n_user_peaks"].tolist() == [2, 1, 1, 1]
    assert (back.atac.X != ds.atac.X).nnz == 0
    assert (
        back.atac.obs["bloom_n_peaks_detected"].tolist()
        == ds.atac.obs["bloom_n_peaks_detected"].tolist()
    )
    assert back.report == ds.report and back.atac.uns["bloom"]["peak_match"] == "overlap"
    for i in range(3):
        a, b = ds.cell_features(i, "atac"), back.cell_features(i, "atac")
        assert a.detected_peak_emb_idx.tolist() == b.detected_peak_emb_idx.tolist()
        assert a.peak_values.tolist() == b.peak_values.tolist()
    # a bundle whose atac.h5ad lacks the ATAC columns is refused
    bad = ad.read_h5ad(out / "atac.h5ad")
    del bad.var["n_user_peaks"]
    bad.write_h5ad(out / "atac.h5ad")
    with pytest.raises(ValueError, match="n_user_peaks"):
        PreparedDataset.read(out)


# ==================================================================================================
# input guards: #33 species mismatch, #34 nothing mappable
# ==================================================================================================


@pytest.mark.unit
def test_issue33_species_mismatch_detected_and_overridable(synthetic_reference):
    mouse_ids = ["Pvalb", "Gad1", "Gad2", "Sst", "Vip", "Xkr4"]
    a = _small_adata(np.ones((2, 6), np.float32), var_names=mouse_ids)
    # mouse symbols with --species human: > 50 % match mouse symbols case-sensitively
    with pytest.raises(
        SpeciesMismatchError, match=r"looks like mouse data.*--species mouse.*--force-species"
    ):
        prepare(a, species="human", reference=synthetic_reference)
    # the override maps anyway (through the ortholog table after upper-casing)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = prepare(a, species="human", reference=synthetic_reference, force_species=True)
    assert ds.report["rna"]["species_check"] == {"species": "human", "forced": True}
    assert ds.report["rna"]["n_ortholog"] >= 4
    # HGNC-style (all upper-case) symbols with --species mouse
    human = _small_adata(np.ones((2, 4), np.float32), var_names=["PVALB", "GAD1", "GAD2", "SST"])
    with pytest.raises(SpeciesMismatchError, match="looks like human data"):
        prepare(human, species="mouse", reference=synthetic_reference)
    # ENSG ids with --species mouse
    ensg = _small_adata(
        np.ones((1, 2), np.float32), var_names=["ENSG00000100362", "ENSG00000128683"]
    )
    with pytest.raises(SpeciesMismatchError, match="ENSG"):
        prepare(ensg, species="mouse", reference=synthetic_reference)
    # correct species: no error; the check is recorded
    ds_m = prepare(a, species="mouse", reference=synthetic_reference)
    chk = ds_m.report["rna"]["species_check"]
    assert (
        chk["mismatch"] is False
        and chk["fraction"] == pytest.approx(0.0)
        and chk["suggested"] == "mouse"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds_h = prepare(human, species="human", reference=synthetic_reference)
    assert ds_h.report["rna"]["species_check"]["mismatch"] is False
    # below the 50 % threshold nothing fires (mixed / mostly unknown ids)
    info = check_species(["Pvalb", "FOO1", "bar", "Baz2"], "human", synthetic_reference)
    assert info["n_other_species_like"] == 1 and info["mismatch"] is False
    info = check_species(["Pvalb", "FOO1", "bar", "Baz2"], "mouse", synthetic_reference)
    assert info["n_other_species_like"] == 1 and info["mismatch"] is False
    # mouse Ensembl ids / Riken-style ids are never HGNC-style
    info = check_species(
        ["ENSMUSG00000005716", "0610009B22Rik", "AW551984"], "mouse", synthetic_reference
    )
    assert info["n_other_species_like"] == 1  # only AW551984 (all caps, not in the synthetic table)
    assert check_species([], "mouse", synthetic_reference)["mismatch"] is False


@pytest.mark.unit
def test_issue34_nothing_mappable_is_an_error(synthetic_reference):
    # RNA: every id unknown -> error listing the most frequent unmapped ids
    a = _small_adata(np.ones((3, 3), np.float32), var_names=["Foo1", "Bar2", "Baz3"])
    with pytest.raises(NoMappedFeaturesError, match=r"\[rna\] no cell has a detected gene.*Foo1"):
        prepare(a, species="mouse", reference=synthetic_reference)
    # RNA: ids known but every detected entry sits on an unmapped column
    b = _small_adata(
        np.array([[0, 0, 1], [0, 0, 2]], np.float32), var_names=["Pvalb", "Gad1", "Nope"]
    )
    with pytest.raises(NoMappedFeaturesError, match=r"0/3|1/3|2/3"):
        prepare(b, species="mouse", reference=synthetic_reference)
    # RNA: an all-zero matrix
    c = _small_adata(np.zeros((2, 2), np.float32), var_names=["Pvalb", "Gad1"])
    with pytest.raises(NoMappedFeaturesError, match="all-zero"):
        prepare(c, species="mouse", reference=synthetic_reference)
    # partial emptiness stays a warning (P7)
    d = _small_adata(np.array([[1, 0], [0, 0]], np.float32), var_names=["Pvalb", "Gad1"])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ds = prepare(d, species="mouse", reference=synthetic_reference)
    assert ds.report["rna"]["n_empty_cells"] == 1 and any("1 cell(s)" in str(x.message) for x in w)
    # ATAC: parsable mm10 peaks that hit nothing (different peak set under exact matching)
    e = _small_adata(np.ones((2, 2), np.float32), var_names=["chr3:1000-1501", "chr5:2000-2501"])
    with pytest.raises(
        NoMappedFeaturesError, match=r"\[atac\].*chr3:1000-1501.*--peak-match overlap"
    ):
        prepare(atac=e, species="mouse", reference=synthetic_reference, peak_match="exact")
    assert issubclass(NoMappedFeaturesError, ValueError) and issubclass(
        SpeciesMismatchError, ValueError
    )


@pytest.mark.unit
def test_p10_duplicate_peak_coordinates_warn_and_or(synthetic_reference):
    """Issue #48: var rows repeating a coordinate are OR-ed (largest value kept) with a counted warning."""
    base = _atac_adata()
    X = base.X.toarray().astype(np.float32)  # (3, 4)
    # duplicate REF_PEAKS[0] twice (one copy spelled with a different contig style) and REF_PEAKS[2] once
    dup_names = ["1:3003635-3004136", REF_PEAKS[0], REF_PEAKS[2]]
    Xd = np.zeros((3, 3), dtype=np.float32)
    Xd[0, 0] = 5  # cell 0 already has 3 on peak 0 -> max 5, not 8
    Xd[1, 1] = 2  # cell 1 had 0 on peak 0 -> detected via the duplicate
    Xd[2, 2] = 7  # cell 2 had 1 on peak 2 -> 7
    a = _small_adata(
        np.hstack([X, Xd]), var_names=[*REF_PEAKS, *dup_names], obs_names=["c0", "c1", "c2"]
    )
    with pytest.warns(UserWarning, match="3 var row\\(s\\) repeat 2 peak coordinate\\(s\\)"):
        ds = prepare(atac=a, species="mouse", reference=synthetic_reference)
    p = ds.atac
    assert p.var_names.tolist() == REF_PEAKS and p.var["n_user_peaks"].tolist() == [3, 1, 2, 1]
    expected = X.copy()
    expected[0, 0] = 5
    expected[1, 0] = 2
    expected[2, 2] = 7
    np.testing.assert_array_equal(p.X.toarray(), expected)
    assert p.obs["bloom_n_peaks_detected"].tolist() == [2, 3, 4]
    rep = ds.report["atac"]
    assert rep["n_duplicate_peaks"] == 3 and rep["n_var"] == 7 and rep["n_mapped"] == 7
    assert any("OR-ed" in w for w in rep["warnings"])
    assert rep["exact_rate"] == pytest.approx(1.0) and rep["peak_match"] == "exact"
    # min_overlap must be in (0, 1] at the library level too
    with pytest.raises(ValueError, match=r"min_overlap must be in \(0, 1\]"):
        prepare(atac=a, species="mouse", reference=synthetic_reference, min_overlap=0.0)
    # without duplicates: no duplicate warning, no change
    plain = _atac_adata()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ds0 = prepare(atac=plain, species="mouse", reference=synthetic_reference)
    assert not [w for w in caught if "repeat" in str(w.message)]
    assert ds0.report["atac"]["n_duplicate_peaks"] == 0


@pytest.mark.unit
def test_cli_prepare_atac_and_guards(synthetic_reference_dir, tmp_path):
    a = _atac_adata(extra=["chr1:3003636-3004137"])
    src = tmp_path / "atac.h5ad"
    a.write_h5ad(src)
    out = tmp_path / "prepared"
    ref = ["--reference-dir", str(synthetic_reference_dir)]
    res = CliRunner().invoke(app, ["prepare", "--atac", str(src), "-o", str(out), *ref])
    assert res.exit_code == 0, res.output
    assert "peak match overlap (min_overlap 0.5); exact rate 80.0%" in res.output
    assert (
        "wrote" in res.output and (out / "atac.h5ad").exists() and not (out / "rna.h5ad").exists()
    )
    back = PreparedDataset.read(out)
    assert back.species == "mouse" and back.atac.obs["bloom_n_peaks_detected"].tolist() == [2, 3, 4]
    res = CliRunner().invoke(
        app, ["prepare", "--atac", str(src), "-o", str(out), "--peak-match", "exact", *ref]
    )
    assert res.exit_code == 0 and "peak match exact" in res.output
    res = CliRunner().invoke(
        app, ["prepare", "--atac", str(src), "-o", str(out), "--peak-match", "x", *ref]
    )
    assert res.exit_code == 1 and "--peak-match" in res.output
    res = CliRunner().invoke(
        app, ["prepare", "--atac", str(src), "-o", str(out), "--min-overlap", "2", *ref]
    )
    assert res.exit_code == 1 and "--min-overlap" in res.output
    res = CliRunner().invoke(
        app, ["prepare", "--atac", str(src), "-o", str(out), "--min-overlap", "0", *ref]
    )
    assert res.exit_code == 1 and "--min-overlap must be in (0, 1]" in res.output  # issue #48
    res = CliRunner().invoke(
        app, ["prepare", "--atac", str(tmp_path / "missing.h5ad"), "-o", str(out), *ref]
    )
    assert res.exit_code == 1 and "does not exist" in res.output
    res = CliRunner().invoke(
        app, ["prepare", "--atac", str(src), "--species", "human", "-o", str(out), *ref]
    )
    assert res.exit_code == 1 and "mm10" in res.output
    # #33 / #34 through the CLI
    rna = _small_adata(np.ones((2, 4), np.float32), var_names=["PVALB", "GAD1", "GAD2", "SST"])
    rsrc = tmp_path / "rna.h5ad"
    rna.write_h5ad(rsrc)
    res = CliRunner().invoke(
        app, ["prepare", "--rna", str(rsrc), "--species", "mouse", "-o", str(out), *ref]
    )
    assert res.exit_code == 1 and "--force-species" in res.output
    res = CliRunner().invoke(
        app,
        [
            "prepare",
            "--rna",
            str(rsrc),
            "--species",
            "mouse",
            "--force-species",
            "-o",
            str(out),
            *ref,
        ],
    )
    assert (
        res.exit_code == 0 and "matched only after case/whitespace" in res.output
    )  # PVALB -> Pvalb
    res = CliRunner().invoke(
        app, ["prepare", "--rna", str(rsrc), "--species", "human", "-o", str(out), *ref]
    )
    assert res.exit_code == 0, res.output
    # both files, keep-unmatched: five files written
    both = tmp_path / "both"
    a2 = _atac_adata(obs_names=("c1", "c9", "c8"))
    a2.write_h5ad(tmp_path / "atac2.h5ad")
    res = CliRunner().invoke(
        app,
        [
            "prepare",
            "--rna",
            str(rsrc),
            "--atac",
            str(tmp_path / "atac2.h5ad"),
            "--species",
            "human",
            "-o",
            str(both),
            "--keep-unmatched",
            *ref,
        ],
    )
    assert res.exit_code == 1 and "mm10" in res.output  # human RNA + ATAC refused
    mouse_rna = _small_adata(np.ones((2, 2), np.float32), var_names=["Pvalb", "Gad1"])
    mouse_rna.write_h5ad(rsrc)
    res = CliRunner().invoke(
        app,
        [
            "prepare",
            "--rna",
            str(rsrc),
            "--atac",
            str(tmp_path / "atac2.h5ad"),
            "--species",
            "mouse",
            "-o",
            str(both),
            "--keep-unmatched",
            *ref,
        ],
    )
    assert res.exit_code == 0, res.output
    assert (
        "[alignment] 1 cells in both files (rna 2, atac 3); rna-only 1, atac-only 2 kept"
        in res.output
    )
    assert "rna_only.h5ad" in res.output and "atac_only.h5ad" in res.output
    assert PreparedDataset.read(both).unmatched["atac"].n_obs == 2
