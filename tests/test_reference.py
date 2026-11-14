"""Reference tables and feature mapping -- tests R1-R6 (docs/test-plan.md section 2) + lazy loading.

Every mapping test runs against the synthetic reference (always) and against the real
`mm10-evo2-v1` gene table when it is available under `$BLOOM_HOME` (`bloom download --what genes`;
skipped otherwise). R7-R10 (coordinate parsing, overlap) belong to `peaks.py` (issue #18).
"""

from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from bloom.reference import (
    MAPPED_VIA_CATEGORIES,
    MappingResult,
    Orthologs,
    ReferenceTables,
    mapping_report,
    strip_ensembl_version,
)

PVALB_ID = "ENSMUSG00000005716"


@pytest.fixture(params=["synthetic", "real_genes"], scope="module")
def ref(request, synthetic_reference_dir) -> ReferenceTables:
    """ReferenceTables over the synthetic tables or over the real gene table (+ the synthetic ortholog slice)."""
    if request.param == "synthetic":
        return ReferenceTables.from_dir(synthetic_reference_dir)
    d = request.getfixturevalue("gene_table_dir")
    return ReferenceTables(
        genes_parquet=d / "genes.parquet",
        genes_safetensors=d / "genes.safetensors",
        orthologs_tsv=synthetic_reference_dir / "orthologs_human_mouse_one2one.tsv",
    )


def _expected_row(ref: ReferenceTables, symbol: str) -> int:
    """emb_idx of `symbol` computed independently from the parquet table."""
    g = ref.genes
    rows = g.index[g["gene_name"] == symbol]
    assert len(rows) == 1, symbol
    return int(rows[0])


# ---- R1 ---------------------------------------------------------------------------------------


@pytest.mark.unit
def test_r1_symbol_lookup(ref):
    res = ref.map_genes(["Pvalb", "Gad1", "NotAGene"], species="mouse")
    assert isinstance(res, MappingResult)
    assert res.emb_idx.dtype == np.int32
    assert res.emb_idx[0] == _expected_row(ref, "Pvalb") == 25359
    assert res.emb_idx[1] == _expected_row(ref, "Gad1") == 2498
    assert res.emb_idx[2] == -1
    assert list(res.mapped_via) == ["exact", "exact", "none"]
    assert list(res.mapped_via.categories) == list(MAPPED_VIA_CATEGORIES)
    assert list(res.symbol) == ["Pvalb", "Gad1", ""]
    assert res.report["n_mapped"] == 2 and res.report["n_unmapped"] == 1
    assert res.report["unmapped_examples"] == ["NotAGene"]
    assert list(ref.symbol_of(res.emb_idx)) == ["Pvalb", "Gad1", ""]
    assert list(ref.gene_id_of([25359, -1])) == [PVALB_ID, ""]


# ---- R2 ---------------------------------------------------------------------------------------


@pytest.mark.unit
def test_r2_case_and_whitespace(ref):
    with pytest.warns(UserWarning, match=r"2 gene identifier\(s\) matched only after"):
        res = ref.map_genes(["pvalb", " Pvalb ", "Gad1"], species="mouse")
    assert res.emb_idx[0] == res.emb_idx[1] == 25359
    assert list(res.mapped_via) == ["exact", "exact", "exact"]
    assert res.report["n_normalised"] == 2
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # exact input must not warn
        ref.map_genes(["Pvalb", "Gad1"], species="mouse")


# ---- R3 ---------------------------------------------------------------------------------------


@pytest.mark.unit
def test_r3_duplicate_symbols(ref):
    with pytest.warns(UserWarning, match="occur more than once"):
        res = ref.map_genes(["Pvalb", "Gad1", "Pvalb", "Pvalb"], species="mouse")
    assert res.emb_idx[0] == res.emb_idx[2] == res.emb_idx[3] == 25359
    assert res.report["n_duplicate_ids"] == 1
    assert res.report["n_duplicate_emb_idx"] == 1
    assert res.n_mapped == 4


# ---- R4 ---------------------------------------------------------------------------------------


@pytest.mark.unit
def test_r4_ensembl_ids(ref):
    res = ref.map_genes([PVALB_ID, f"{PVALB_ID}.9", "ENSMUSG00000000000"], species="mouse")
    assert res.emb_idx[0] == res.emb_idx[1] == 25359
    assert res.emb_idx[2] == -1
    assert list(res.mapped_via) == ["gene_id", "gene_id", "none"]
    assert res.report["n_gene_id"] == 2
    # the same through map_gene_ids
    res2 = ref.map_gene_ids([PVALB_ID, "ENSMUSG00000070880"])
    assert res2.emb_idx.tolist() == [25359, 2498]
    assert res2.report["species"] == "mouse"
    # a var full of Ensembl ids maps completely -> no low-mapping flag
    ids = ref.genes["gene_id"].to_numpy()
    full = ref.map_genes(ids, species="mouse")
    assert full.n_mapped == len(ids)
    X = sp.random(5, len(ids), density=0.3, format="csr", random_state=0)
    rep = mapping_report(full, X)
    assert rep["low_mapping"] is False and rep["detected_mass_mapped"] == pytest.approx(1.0)
    assert strip_ensembl_version("ENSG00000100362.12") == "ENSG00000100362"


# ---- R5 ---------------------------------------------------------------------------------------


@pytest.mark.unit
def test_r5_human_orthologs_synthetic_table(ref):
    res = ref.map_genes(["PVALB", "GAD1", "ATP5PO", "TP53", "ENSG00000100362.4"], species="human")
    assert res.emb_idx[0] == 25359 and res.symbol[0] == "Pvalb"
    assert res.emb_idx[1] == 2498
    assert res.emb_idx[3] == -1  # human symbol absent from the one2one slice
    assert res.emb_idx[4] == 25359  # ENSG id, version stripped
    assert list(res.mapped_via[[0, 1, 3, 4]]) == ["ortholog", "ortholog", "none", "ortholog"]
    # symbol-drifted gene: ortholog table says Atp5po, gene table says Atp5o -> join on gene_id
    if "Atp5o" in set(ref.genes["gene_name"]):
        assert res.emb_idx[2] == _expected_row(ref, "Atp5o") == 100 and res.symbol[2] == "Atp5o"
        assert res.mapped_via[2] == "ortholog"
    assert res.report["n_ortholog"] == res.n_mapped


@pytest.mark.unit
def test_r5_symbol_fallback_and_missing(synthetic_reference_dir):
    ref = ReferenceTables.from_dir(synthetic_reference_dir)
    with pytest.warns(UserWarning, match="1 gene identifier"):
        res = ref.map_genes(["GAD2", "NOTINGENETABLE", "sst"], species="human")
    assert res.emb_idx[0] == 1938  # mouse id absent -> mouse symbol fallback
    assert res.emb_idx[1] == -1  # neither id nor symbol in the gene table
    assert res.emb_idx[2] == 26231 and res.report["n_normalised"] == 1  # lower-case human symbol
    # human mapping without an ortholog table is a clear error
    bare = ReferenceTables(
        genes_parquet=synthetic_reference_dir / "genes.parquet",
        genes_safetensors=synthetic_reference_dir / "genes.safetensors",
    )
    with pytest.raises(FileNotFoundError, match="orthologs"):
        bare.map_genes(["PVALB"], species="human")


@pytest.mark.unit
def test_r5_shipped_table_on_real_genes(gene_table_dir, shipped_orthologs_tsv):
    ref = ReferenceTables(
        genes_parquet=gene_table_dir / "genes.parquet",
        genes_safetensors=gene_table_dir / "genes.safetensors",
        orthologs_tsv=shipped_orthologs_tsv,
    )
    orth = ref.orthologs
    assert orth is not None and "ensembl_release" in orth.header
    assert {
        "human_gene_id",
        "human_gene_name",
        "mouse_gene_id",
        "mouse_gene_name",
        "emb_idx",
    } <= set(orth.table.columns)
    # one-to-one: unique on both sides
    assert orth.table["human_gene_id"].is_unique and orth.table["mouse_gene_id"].is_unique
    res = ref.map_genes(["PVALB", "GAD1", "ZZZNOTAGENE"], species="human")
    assert res.emb_idx.tolist()[:2] == [25359, 2498] and res.emb_idx[2] == -1
    assert list(res.mapped_via) == ["ortholog", "ortholog", "none"]
    # every shipped pair is in the gene universe and its emb_idx column agrees with the gene table
    sub = orth.table[orth.table["mouse_gene_id"].isin(ref.genes["gene_id"])]
    assert len(sub) == len(orth.table) == 15_676
    by_id = pd.Series(ref.genes.index, index=ref.genes["gene_id"])
    assert (sub["emb_idx"].to_numpy() == by_id.loc[sub["mouse_gene_id"]].to_numpy()).all()


# ---- R6 ---------------------------------------------------------------------------------------


@pytest.mark.unit
def test_r6_mapping_report_numbers(ref):
    ids = ["Pvalb", "Gad1", "Nope1", "Nope2", PVALB_ID]
    res = ref.map_genes(ids, species="mouse")
    X = sp.csr_matrix(
        np.array(
            [
                [1.0, 0.0, 2.0, 0.0, 0.0],
                [0.0, 3.0, 0.0, 1.0, 5.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    rep = mapping_report(res, X)
    # independent computation: detected entries 5, in mapped columns {0,1,4}: 1 + 1 + 1 = 3
    assert rep["n_var"] == 5 and rep["n_mapped"] == 3
    assert rep["detected_mass_mapped"] == pytest.approx(3 / 5)
    assert rep["mapping_rate"] == pytest.approx(3 / 5)
    assert rep["low_mapping"] is True
    dense = mapping_report(res, X.toarray())
    assert dense["detected_mass_mapped"] == rep["detected_mass_mapped"]
    assert mapping_report(res)["detected_mass_mapped"] is None
    with pytest.raises(ValueError, match="columns"):
        mapping_report(res, X[:, :3])


# ---- tensors, peaks, laziness (M6 for the reference side) --------------------------------------


@pytest.mark.unit
def test_gene_tensors_shapes(ref):
    t = ref.gene_tensors()
    n = ref.n_genes
    assert tuple(t["cseq"].shape) == (n, 768) and str(t["cseq"].dtype) == "torch.float32"
    assert tuple(t["chr"].shape) == (n,) and str(t["chr"].dtype) == "torch.int8"
    assert tuple(t["bp"].shape) == (n,) and str(t["bp"].dtype) == "torch.int32"
    if "emb_idx" in t:  # only row-subset tables carry it; the full table is row == emb_idx
        assert t["emb_idx"].numpy().tolist() == ref.genes.index.tolist()
    else:
        assert ref.genes.index.tolist() == list(range(n))
    assert ref.safetensors_metadata("genes")["reference_id"] == "mm10-evo2-v1"


@pytest.mark.unit
def test_peaks_exact_lookup_and_lazy_open(synthetic_reference_dir, monkeypatch):
    import safetensors

    opened: list[str] = []
    real_safe_open = safetensors.safe_open

    def spy(path, *a, **k):
        opened.append(Path(path).name)
        return real_safe_open(path, *a, **k)

    monkeypatch.setattr(safetensors, "safe_open", spy)
    ref = ReferenceTables.from_dir(synthetic_reference_dir, require_peaks=True)
    ref.map_genes(["Pvalb"])
    ref.gene_tensors()
    assert ref.n_peaks == 4  # parquet metadata only
    assert ref._peaks is None and "peaks.safetensors" not in opened
    idx = ref.map_peaks([
        "chr1:3003635-3004136",
        ("chr2", 100000, 100501),
        "chr1:1-2",
        " chrX:5000-5501 ",
    ])
    assert idx.tolist() == [0, 7, -1, 12] and idx.dtype == np.int32
    assert list(ref.peak_name_of([7, -1])) == ["chr2:100000-100501", ""]
    assert "peaks.safetensors" not in opened  # the parquet is enough for name lookups
    t = ref.peak_tensors()
    assert tuple(t["cseq"].shape) == (4, 768) and opened.count("peaks.safetensors") == 1
    assert "ReferenceTables(mm10-evo2-v1: genes=8, peaks=4, orthologs=present)" == repr(ref)


@pytest.mark.unit
def test_from_dir_missing_files(tmp_path, synthetic_reference_dir):
    with pytest.raises(FileNotFoundError, match="bloom download"):
        ReferenceTables.from_dir(tmp_path)
    ref = ReferenceTables.from_dir(synthetic_reference_dir)
    bare_dir = tmp_path / "bare"
    bare_dir.mkdir()
    for f in ("genes.parquet", "genes.safetensors"):
        (bare_dir / f).symlink_to(synthetic_reference_dir / f)
    bare = ReferenceTables.from_dir(bare_dir)
    assert not bare.has_peaks and bare.orthologs is None and bare.n_peaks == 0
    with pytest.raises(FileNotFoundError, match="bloom download --what peaks"):
        bare.map_peaks(["chr1:1-2"])
    with pytest.raises(FileNotFoundError):
        ReferenceTables.from_dir(bare_dir, require_peaks=True)
    assert ref.n_genes == 8


@pytest.mark.unit
def test_orthologs_header_and_lookup(synthetic_reference_dir):
    orth = Orthologs.from_tsv(synthetic_reference_dir / "orthologs_human_mouse_one2one.tsv")
    assert orth.header["source"] == "synthetic" and orth.header["ensembl_release"] == "0"
    assert len(orth) == 7
    assert orth.mouse_gene_id("PVALB") == ("ENSMUSG00000005716", False)
    assert orth.mouse_gene_id(" pvalb") == ("ENSMUSG00000005716", True)
    assert orth.mouse_gene_id("ENSG00000100362.3") == ("ENSMUSG00000005716", False)
    assert orth.mouse_gene_id("TP53") == (None, False)
    assert orth.mouse_symbol_of("ENSMUSG00000022956") == "Atp5po"
    with pytest.raises(ValueError, match="lacks columns"):
        Orthologs(pd.DataFrame({"human_gene_id": ["x"]}))


# ---- assets tier: the full release tables (bloom download --what genes,peaks) --------------------


@pytest.mark.assets
def test_full_reference_tables_load_time_and_rss():
    """Full mm10-evo2-v1 tables: sizes, load time and RSS (numbers go in the PR / model card)."""
    import resource

    from bloom.reference import default_reference_dir

    d = default_reference_dir()
    if not all(
        (d / f).exists()
        for f in ("genes.parquet", "genes.safetensors", "peaks.parquet", "peaks.safetensors")
    ):
        pytest.skip(f"full reference tables not under {d}")
    log = logging.getLogger("bloom.tests")
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    t0 = time.perf_counter()
    ref = ReferenceTables.from_dir(d, require_peaks=True)
    genes = ref.gene_tensors()
    t_genes = time.perf_counter() - t0
    assert ref.n_genes == 32129 and tuple(genes["cseq"].shape) == (32129, 768)
    assert ref.genes["gene_name"].is_unique and ref.genes["gene_id"].is_unique
    t0 = time.perf_counter()
    assert ref.n_peaks == 1747701
    peaks = ref.peaks
    t_peaks_parquet = time.perf_counter() - t0
    sample = peaks["name"].sample(1000, random_state=0)
    assert (ref.map_peaks(sample.to_numpy()) == sample.index.to_numpy()).all()
    t0 = time.perf_counter()
    pt = ref.peak_tensors()
    t_peaks_st = time.perf_counter() - t0
    assert tuple(pt["cseq"].shape) == (1747701, 768)
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    log.warning(
        "full tables: genes %.2fs, peaks.parquet %.2fs, peaks.safetensors %.2fs, maxRSS %.0f -> %.0f MB",
        t_genes,
        t_peaks_parquet,
        t_peaks_st,
        rss0,
        rss1,
    )
    if ref.orthologs is not None:
        res = ref.map_genes(ref.orthologs.table["human_gene_name"].to_numpy(), species="human")
        assert res.n_mapped == len(ref.orthologs)  # every shipped pair resolves by gene_id
        assert (res.emb_idx == ref.orthologs.table["emb_idx"].to_numpy()).all()
