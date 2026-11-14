"""``bloom.embed`` and ``bloom embed``: unit tests with a fake model, end-to-end E1-E8, gates G1/G4/G7/G8.

Tiers (docs/test-plan.md):

- ``unit``: a deterministic fake model (features of ``emb_idx``, no assets) exercises the view
  queue, batching, float64 averaging, NaN rows, ``--keep-views``, provenance, modality routing
  (S7, RNA and ATAC), the two-modality output / ``--keep-unmatched`` (P5, E4 with the fake model),
  the lazy peak-table load, a custom strategy (S10) and the CLI error paths (E7).
- ``regression`` (skipped without ``$BLOOM_HOME/models/hyacinth``, i.e. on CI): the real CLI on
  ``tests/data/rna_100.h5ad`` (E1, E2, E3, E8, G8) and the G1 gate (``max|d| < 1e-4``) against
  ``tests/data/ref/rna_top4096_tf32off`` (regenerated 2026-11-13 from column-sorted inputs with
  the fused CUDA fast path off, #26 / #27).
- ``assets``: G4 (``random`` statistical parity) and G7 (CPU vs CUDA) on a machine with the
  downloaded assets (``bloom download --what model,genes``) and a CUDA GPU.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import time
import types
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch
from conftest import load_reference, total_ram_gb
from torch import nn
from typer.testing import CliRunner

import bloom
from bloom.cli import app
from bloom.embed import DEFAULT_BATCH_SIZE, default_strategies, embed, obsm_key
from bloom.feature_lists import FeatureList
from bloom.prepare import PreparedDataset
from bloom.reference import ReferenceTables
from bloom.strategies import Panel, Resample, Tile, TopExpressed, ViewStrategy
from bloom.tokens import CellFeatures, TokenSet

TESTS_DATA = Path(__file__).resolve().parent / "data"
FIXTURE = TESTS_DATA / "rna_100.h5ad"
REFS_NPZ = TESTS_DATA / "ref" / "rna_100_reference_embeddings.npz"
REF_TOP = "rna_top4096_tf32off"  # TF32 off + fused fast path off: the G1 reference
REF_RESAMP = "rna_random_8192x4"  # paper default strategy: statistical parity (G4)
PROVENANCE_KEYS = {
    "schema_version",
    "package_version",
    "git_sha",
    "model_id",
    "model_sha256",
    "reference_id",
    "modalities",
    "device",
    "precision",
    "torch_version",
    "created",
}
G1_TOLERANCE = 1e-4
G7_TOLERANCE = 1e-3
REF_TOP_TF32 = (
    "rna_top4096"  # paper-faithful (TF32 on, fused fast path in evaluate.py); reported only
)
runner = CliRunner()


# ============================================================================================
# fake model + synthetic dataset
# ============================================================================================
class FakeBloom(nn.Module):
    """Deterministic stand-in for ``Bloom``: ``h = mean_k cos((emb_idx + 1) * k)`` over real tokens.

    Padding- and batch-invariant by construction (per-row reduction), NaN for empty rows, so the
    averaging / batching logic of ``embed`` can be checked exactly.
    """

    def __init__(self, d_model: int = 16, reference=None) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(
            d_model=d_model,
            model_id="fake",
            reference_id="mm10-evo2-v1",
            weights={"sha256": "f" * 64},
            strategy_defaults={"gene": {"strategy": "random", "lmax": 8192, "draws": 4}},
        )
        self.anchor = nn.Parameter(torch.zeros(1))
        self.has_gene_table = True
        self.has_peak_table = False
        self.reference = reference
        self.calls: list[tuple[int, int]] = []  # (B, L) of every encode call

    @property
    def device(self) -> torch.device:
        return self.anchor.device

    @staticmethod
    def features(emb_idx: np.ndarray, d_model: int = 16) -> np.ndarray:
        e = np.asarray(emb_idx, dtype=np.float64) + 1.0
        ks = np.arange(1, d_model + 1, dtype=np.float64)
        return np.cos(e[:, None] * ks)  # (L, d)

    def encode(
        self, emb_idx: torch.Tensor, kind: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        self.calls.append(tuple(emb_idx.shape))
        out = torch.full((emb_idx.shape[0], self.config.d_model), float("nan"), dtype=torch.float32)
        for b in range(emb_idx.shape[0]):
            n = int(lengths[b])
            if n == 0:
                continue
            assert bool((kind[b, :n] != 0).all()) and bool((kind[b, n:] == 0).all())
            f = self.features(emb_idx[b, :n].cpu().numpy(), self.config.d_model)
            out[b] = torch.from_numpy(f.mean(0).astype(np.float32))
        return out.to(emb_idx.device)


# 6 cells x 10 genes; columns 8 and 9 are unmapped; cell 4 is all zero; cell 5 detects only unmapped genes.
EMB_OF_COL = np.array([5, 0, 7, 2, 3, 6, 1, 4, -1, -1], dtype=np.int32)
SYMBOLS = ["g5", "g0", "g7", "g2", "g3", "g6", "g1", "g4", "unmapped_a", "unmapped_b"]


def make_dataset(n_cells: int = 6, seed: int = 0) -> PreparedDataset:
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 5, size=(n_cells, 10)).astype(np.float32)
    X[:, 0] += 1.0  # every normal cell detects column 0 (emb 5)
    X[4] = 0.0
    X[5] = 0.0
    X[5, 8:] = 3.0
    obs = pd.DataFrame(
        {"celltype": [f"t{i % 2}" for i in range(n_cells)], "depth": np.arange(n_cells) * 10.0},
        index=[f"cell{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(
        {
            "bloom_symbol": SYMBOLS,
            "bloom_emb_idx": EMB_OF_COL,
            "bloom_mapped_via": pd.Categorical(
                ["exact"] * 8 + ["none"] * 2, categories=["exact", "gene_id", "ortholog", "none"]
            ),
        },
        index=SYMBOLS,
    )
    a = ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=var)
    a.X.sort_indices()
    a.uns["bloom"] = {"schema_version": "1.0", "reference_id": "mm10-evo2-v1", "species": "mouse"}
    return PreparedDataset(
        rna=a,
        species="mouse",
        reference_id="mm10-evo2-v1",
        report={"rna": {"n_var": 10, "n_mapped": 8, "mapping_rate": 0.8, "n_empty_cells": 2}},
        source={"rna": {"path": "synthetic.h5ad", "sha256": None}},
    )


def expected_mean(ds: PreparedDataset, strategy: ViewStrategy, seed: int, i: int) -> np.ndarray:
    """Reference computation of the fake model's per-cell mean over the strategy's views."""
    views = strategy.views(ds.cell_features(i), bloom.strategies.cell_rng(seed, i))
    if not views:
        return np.full(16, np.nan, dtype=np.float32)
    acc = np.zeros(16, dtype=np.float64)
    for ts in views:
        acc += FakeBloom.features(ts.emb_idx).mean(0)
    return (acc / len(views)).astype(np.float32)


# ============================================================================================
# unit: embed() with the fake model
# ============================================================================================
@pytest.mark.unit
def test_embed_output_schema_averaging_and_nan_rows():
    ds = make_dataset()
    model = FakeBloom()
    strat = Resample(lmax=3, draws=4)
    with pytest.warns(UserWarning, match="2 of 6 cells have no detected gene"):
        out = embed(
            model, ds, {"rna": strat}, batch_size=5, seed=7, keep_views=True, progress=False
        )

    # schema (docs/data-schema.md section 4)
    assert out.n_obs == 6 and out.n_vars == 0 and sp.issparse(out.X) and out.X.shape == (6, 0)
    assert out.obs.index.tolist() == ds.rna.obs.index.tolist()
    assert out.obs["celltype"].tolist() == ds.rna.obs["celltype"].tolist()  # user obs preserved
    assert out.obs["bloom_n_views_rna"].tolist() == [4, 4, 4, 4, 0, 0]
    assert out.obs["bloom_n_tokens_rna"].dtype == np.float32
    X = out.obsm[obsm_key("rna")]
    assert X.dtype == np.float32 and X.shape == (6, 16)
    assert np.isnan(X[4]).all() and np.isnan(X[5]).all() and np.isfinite(X[:4]).all()
    assert "X_bloom_atac" not in out.obsm  # S7: only the modality given
    # bloom_n_tokens = mean view length = min(lmax, n_detected)
    for i in range(4):
        n_det = ds.cell_features(i).n_detected
        assert out.obs["bloom_n_tokens_rna"].iloc[i] == pytest.approx(min(3, n_det))
    assert np.isnan(out.obs["bloom_n_tokens_rna"].iloc[4])
    # averaging over views in float64 -> float32 equals the reference computation
    for i in range(6):
        exp = expected_mean(ds, strat, 7, i)
        np.testing.assert_allclose(X[i], exp, rtol=0, atol=1e-6, equal_nan=True)
    # E8: per-view embeddings and their mean
    V = out.obsm[obsm_key("rna", views=True)]
    assert V.shape == (6, 4, 16) and V.dtype == np.float32
    np.testing.assert_allclose(np.nanmean(V[:4], axis=1), X[:4], atol=2e-6)
    assert np.isnan(V[4]).all()
    # provenance
    prov = out.uns["bloom"]
    assert PROVENANCE_KEYS <= set(prov)
    assert prov["model_id"] == "fake" and prov["model_sha256"] == "f" * 64
    strat_prov = prov["modalities"]["rna"]["strategy"]
    assert strat_prov == {
        "strategy": "random",
        "lmax": 3,
        "draws": 4,
    }  # None keys are dropped (h5ad)
    assert strat_prov.get("universe") is None
    assert prov["modalities"]["rna"]["seed"] == 7 and prov["precision"] == "fp32"
    assert prov["modalities"]["rna"]["run"]["n_views"] == 16
    assert prov["batch_size"] == 5


@pytest.mark.unit
def test_view_queue_fills_batches_regardless_of_draws():
    ds = make_dataset()
    model = FakeBloom()
    embed(model, ds, {"rna": Resample(lmax=3, draws=3)}, batch_size=5, progress=False)
    # 4 cells x 3 views = 12 views -> batches of 5, 5, 2
    assert [b for b, _ in model.calls] == [5, 5, 2]


@pytest.mark.unit
@pytest.mark.parametrize("batch_size", [1, 3, 32])
def test_g8_batch_size_invariance_fake(batch_size):
    ds = make_dataset()
    ref = embed(FakeBloom(), ds, {"rna": TopExpressed(lmax=4)}, batch_size=1, progress=False)
    out = embed(
        FakeBloom(), ds, {"rna": TopExpressed(lmax=4)}, batch_size=batch_size, progress=False
    )
    np.testing.assert_array_equal(out.obsm["X_bloom_rna"], ref.obsm["X_bloom_rna"])


@pytest.mark.unit
def test_top_strategy_uses_highest_values():
    ds = make_dataset()
    out = embed(FakeBloom(), ds, {"rna": TopExpressed(lmax=2)}, progress=False)
    cell = ds.cell_features(0)
    order = np.argsort(-cell.gene_values, kind="stable")[:2]
    exp = FakeBloom.features(cell.detected_gene_emb_idx[order]).mean(0).astype(np.float32)
    np.testing.assert_allclose(out.obsm["X_bloom_rna"][0], exp, atol=1e-6)
    assert out.obs["bloom_n_views_rna"].iloc[0] == 1


@pytest.mark.unit
def test_s10_custom_strategy_runs_end_to_end():
    class EvenGenesOnly(ViewStrategy):
        """Keep detected genes with an even emb_idx; one view."""

        name = "even"

        def __init__(self) -> None:
            super().__init__(lmax=8)

        def sample(self, cell: CellFeatures, rng: np.random.Generator) -> list[TokenSet]:
            emb = cell.detected_gene_emb_idx
            emb = emb[emb % 2 == 0]
            return [TokenSet.genes(emb)] if emb.size else []

    ds = make_dataset()
    out = embed(FakeBloom(), ds, {"rna": EvenGenesOnly()}, progress=False)
    emb = ds.cell_features(0).detected_gene_emb_idx
    exp = FakeBloom.features(emb[emb % 2 == 0]).mean(0).astype(np.float32)
    np.testing.assert_allclose(out.obsm["X_bloom_rna"][0], exp, atol=1e-6)
    assert out.uns["bloom"]["modalities"]["rna"]["strategy"]["strategy"] == "even"


@pytest.mark.unit
def test_default_strategies_from_config_and_overrides():
    ds = make_dataset()
    model = FakeBloom()
    s = default_strategies(ds, model)
    assert isinstance(s["rna"], Resample) and (s["rna"].lmax, s["rna"].draws) == (8192, 4)
    s = default_strategies(ds, model, {"rna": {"strategy": "top", "lmax": None, "draws": None}})
    assert isinstance(s["rna"], TopExpressed) and s["rna"].lmax == 4096  # preset default
    s = default_strategies(ds, model, {"rna": {"strategy": "random", "lmax": 100, "draws": 2}})
    assert (s["rna"].lmax, s["rna"].draws) == (100, 2)
    with pytest.warns(DeprecationWarning, match="renamed to 'random'"):  # old name, one release
        s = default_strategies(ds, model, {"rna": {"strategy": "resample", "draws": 2}})
    assert isinstance(s["rna"], Resample) and s["rna"].draws == 2
    s = default_strategies(ds, model, {"rna": {"strategy": "tile", "lmax": 50, "draws": None}})
    assert isinstance(s["rna"], Tile) and (s["rna"].lmax, s["rna"].max_draws) == (50, 4)  # #31
    with pytest.raises(ValueError, match="needs `panel`"):
        default_strategies(ds, model, {"rna": {"strategy": "panel"}})


@pytest.mark.unit
def test_universe_and_panel_resolved_against_model_reference():
    ref = types.SimpleNamespace(
        symbol_to_emb_idx={f"g{i}": i for i in range(8)}, peak_to_emb_idx={}
    )
    ds = make_dataset()
    uni = FeatureList.from_ids(["g0", "g1", "g2", "notthere"])
    out = embed(
        FakeBloom(reference=ref),
        ds,
        {"rna": Resample(lmax=8, draws=1, universe=uni)},
        progress=False,
    )
    for i in range(4):
        det = ds.cell_features(i).detected_gene_emb_idx
        keep = det[det <= 2]
        exp = (
            FakeBloom.features(keep).mean(0).astype(np.float32)
            if keep.size
            else np.full(16, np.nan)
        )
        np.testing.assert_allclose(out.obsm["X_bloom_rna"][i], exp, atol=1e-6, equal_nan=True)
    prov = out.uns["bloom"]["modalities"]["rna"]["strategy"]["universe"]
    assert prov["n_ids"] == 4 and prov["kind"] == "gene"
    # panel: detected intersect panel in panel order
    panel = Panel(FeatureList.from_ids(["g7", "g5", "g0"]), lmax=8)
    out = embed(FakeBloom(reference=ref), ds, {"rna": panel}, progress=False)
    det = ds.cell_features(0).detected_gene_emb_idx
    keep = [e for e in (7, 5, 0) if e in set(det.tolist())]
    np.testing.assert_allclose(
        out.obsm["X_bloom_rna"][0], FakeBloom.features(np.array(keep)).mean(0), atol=1e-6
    )
    # no reference on the model -> clear error
    with pytest.raises(ValueError, match="no `reference`"):
        embed(FakeBloom(), ds, {"rna": Resample(lmax=8, universe=uni)}, progress=False)


@pytest.mark.unit
def test_embed_errors_and_guards():
    ds = make_dataset()
    with pytest.raises(ValueError, match="precision"):
        embed(FakeBloom(), ds, precision="fp16", progress=False)
    with pytest.raises(ValueError, match="batch_size"):
        embed(FakeBloom(), ds, batch_size=0, progress=False)
    with pytest.raises(ValueError, match="absent modalities"):
        embed(FakeBloom(), ds, {"atac": Resample(lmax=8)}, progress=False)
    # an ATAC dataset on a model without the peak table and without a reference -> actionable error
    atac = PreparedDataset(atac=ds.rna.copy(), species="mouse", reference_id="mm10-evo2-v1")
    with pytest.raises(RuntimeError, match="bloom download --what peaks"):
        embed(FakeBloom(), atac, progress=False)
    assert DEFAULT_BATCH_SIZE["cpu"] == 1


@pytest.mark.unit
def test_output_h5ad_round_trip(tmp_path):
    ds = make_dataset()
    out = embed(
        FakeBloom(), ds, {"rna": Resample(lmax=3, draws=2)}, keep_views=True, progress=False
    )
    p = tmp_path / "emb.h5ad"
    out.write_h5ad(p)
    back = ad.read_h5ad(p)
    np.testing.assert_array_equal(back.obsm["X_bloom_rna"], out.obsm["X_bloom_rna"])
    np.testing.assert_array_equal(back.obsm["X_bloom_rna_views"], out.obsm["X_bloom_rna_views"])
    assert back.shape == (6, 0) and PROVENANCE_KEYS <= set(back.uns["bloom"])
    assert back.obs["bloom_n_views_rna"].tolist() == out.obs["bloom_n_views_rna"].tolist()


@pytest.mark.unit
def test_reference_tables_satisfy_feature_reference_protocol(synthetic_reference_dir):
    ref = ReferenceTables.from_dir(synthetic_reference_dir)
    assert ref.symbol_to_emb_idx["Pvalb"] == 25359
    res = FeatureList.from_ids(["Pvalb", "Gad1", "Nope"]).resolve(ref)
    assert res.emb_idx.tolist() == [25359, 2498] and res.n_unknown == 1
    assert ref.peak_to_emb_idx["chr1:3003635-3004136"] == 0


# ============================================================================================
# unit: CLI with the fake model (errors E7 / --atac; success path through prepare)
# ============================================================================================
def _tiny_h5ad(path: Path, symbols: list[str], n_cells: int = 5) -> Path:
    rng = np.random.default_rng(1)
    X = rng.integers(0, 4, size=(n_cells, len(symbols))).astype(np.float32)
    X[:, 0] += 1
    a = ad.AnnData(
        X=sp.csr_matrix(X),
        obs=pd.DataFrame(
            {"label": list("abcde")[:n_cells]}, index=[f"c{i}" for i in range(n_cells)]
        ),
        var=pd.DataFrame(index=symbols),
    )
    a.write_h5ad(path)
    return path


@pytest.fixture
def fake_cli(monkeypatch, synthetic_reference_dir):
    """Patch ``bloom.embed.load_model`` so the CLI runs with the fake model + synthetic reference."""
    ref = ReferenceTables.from_dir(synthetic_reference_dir)
    model = FakeBloom(reference=ref)

    def _load(
        model_id="hyacinth", device="auto", *, model_dir=None, reference_dir=None, peaks=False
    ):
        return model

    monkeypatch.setattr(importlib.import_module("bloom.embed"), "load_model", _load)
    return model


@pytest.mark.unit
def test_cli_embed_fake_model_rna_and_prepared_dir(tmp_path, fake_cli, synthetic_reference_dir):
    symbols = ["Pvalb", "Gad1", "Gad2", "Sst", "Vip", "Xkr4", "NotAGene"]
    h5 = _tiny_h5ad(tmp_path / "tiny.h5ad", symbols)
    out = tmp_path / "out.h5ad"
    r = runner.invoke(
        app,
        [
            "embed",
            "--rna",
            str(h5),
            "--species",
            "mouse",
            "--strategy",
            "top",
            "--lmax",
            "3",
            "-o",
            str(out),
        ],
    )
    assert r.exit_code == 0, r.output
    assert "wrote" in r.output and "mapped 6/7" in r.output
    a = ad.read_h5ad(out)
    assert a.obsm["X_bloom_rna"].shape == (5, 16) and a.obs["label"].tolist() == list("abcde")
    assert a.uns["bloom"]["modalities"]["rna"]["strategy"]["strategy"] == "top"
    assert PROVENANCE_KEYS <= set(a.uns["bloom"])
    # E2 (fake): prepare -> embed PREPARED_DIR is bitwise identical to the in-memory path
    bundle = tmp_path / "bundle"
    r = runner.invoke(
        app,
        [
            "prepare",
            "--rna",
            str(h5),
            "--species",
            "mouse",
            "-o",
            str(bundle),
            "--reference-dir",
            str(synthetic_reference_dir),
        ],
    )
    assert r.exit_code == 0, r.output
    out2 = tmp_path / "out2.h5ad"
    r = runner.invoke(
        app, ["embed", str(bundle), "--strategy", "top", "--lmax", "3", "-o", str(out2)]
    )
    assert r.exit_code == 0, r.output
    b = ad.read_h5ad(out2)
    np.testing.assert_array_equal(a.obsm["X_bloom_rna"], b.obsm["X_bloom_rna"])
    # tile (#31): draws partition the detected set -> with lmax 2 a cell with 5 detected genes gets 3 views
    r = runner.invoke(
        app,
        [
            "embed",
            str(bundle),
            "--strategy",
            "tile",
            "--lmax",
            "2",
            "--draws",
            "8",
            "-o",
            str(out2),
        ],
    )
    assert r.exit_code == 0, r.output
    t = ad.read_h5ad(out2)
    assert t.uns["bloom"]["modalities"]["rna"]["strategy"] == {
        "strategy": "tile",
        "lmax": 2,
        "draws": 8,
    }
    assert (t.obs["bloom_n_views_rna"] == np.ceil(t.obs["bloom_n_genes_detected"] / 2)).all()
    # the pre-release preset name is still accepted (one release) with a warning on stderr
    r = runner.invoke(
        app, ["embed", str(bundle), "--strategy", "resample", "--draws", "1", "-o", str(out2)]
    )
    assert r.exit_code == 0, r.output
    assert "renamed to random" in r.output
    assert ad.read_h5ad(out2).uns["bloom"]["modalities"]["rna"]["strategy"]["strategy"] == "random"
    # species mismatch with a bundle is a user error
    r = runner.invoke(app, ["embed", str(bundle), "--species", "human", "-o", str(out2)])
    assert r.exit_code == 1 and "prepared for species" in r.output


@pytest.mark.unit
def test_cli_embed_user_errors(tmp_path, fake_cli):
    h5 = _tiny_h5ad(tmp_path / "tiny.h5ad", ["Pvalb", "Gad1"])
    out = str(tmp_path / "o.h5ad")
    cases = [
        (
            ["embed", "--atac", str(tmp_path / "missing.h5ad"), "--species", "mouse", "-o", out],
            1,
            "does not exist",
        ),
        (
            ["embed", "--rna", str(h5), "--species", "mouse", "--peak-match", "x", "-o", out],
            1,
            "--peak-match",
        ),
        (
            ["embed", "--rna", str(h5), "--species", "mouse", "--min-overlap", "1.5", "-o", out],
            1,
            "--min-overlap",
        ),
        (
            ["embed", "--rna", str(h5), "--species", "mouse", "--min-overlap", "0", "-o", out],
            1,
            "--min-overlap must be in (0, 1]",
        ),
        (["embed", "-o", out], 1, "give --rna"),
        (["embed", "--rna", str(h5), "-o", out], 1, "--species"),
        (["embed", "--rna", str(h5), "--species", "rat", "-o", out], 1, "mouse or human"),
        (
            ["embed", "--rna", str(h5), "--species", "mouse", "--strategy", "nope", "-o", out],
            1,
            "--strategy",
        ),
        (
            ["embed", "--rna", str(h5), "--species", "mouse", "--strategy", "panel", "-o", out],
            1,
            "--panel",
        ),
        (
            ["embed", "--rna", str(h5), "--species", "mouse", "--precision", "fp16", "-o", out],
            1,
            "--precision",
        ),
        (
            ["embed", "--rna", str(tmp_path / "missing.h5ad"), "--species", "mouse", "-o", out],
            1,
            "does not exist",
        ),
        (
            [
                "embed",
                "--rna",
                str(h5),
                "--species",
                "mouse",
                "--universe",
                str(tmp_path / "u.txt"),
                "-o",
                out,
            ],
            1,
            "--universe",
        ),
    ]
    for args, code, needle in cases:
        r = runner.invoke(app, args)
        assert r.exit_code == code, (args, r.output)
        assert needle in r.output, (args, r.output)
        assert r.output.count("error:") == 1, (args, r.output)  # one message, no empty second line
    # --layer: the KeyError message is unwrapped and lists the real layers only
    r = runner.invoke(
        app, ["embed", "--rna", str(h5), "--species", "mouse", "--layer", "nope", "-o", out]
    )
    assert r.exit_code == 1 and "layer 'nope' not found" in r.output and "None" not in r.output


@pytest.mark.unit
def test_e7_missing_model_assets_message(tmp_path, monkeypatch):
    monkeypatch.setenv("BLOOM_HOME", str(tmp_path))  # empty cache
    h5 = _tiny_h5ad(tmp_path / "tiny.h5ad", ["Pvalb", "Gad1"])
    r = runner.invoke(
        app, ["embed", "--rna", str(h5), "--species", "mouse", "-o", str(tmp_path / "o.h5ad")]
    )
    assert r.exit_code == 1, r.output
    assert "not downloaded" in r.output and "bloom download --what model" in r.output
    assert "bloom download --what model,genes" in r.output and "public bucket" in r.output


@pytest.mark.unit
def test_e6_info_still_exits_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("BLOOM_HOME", str(tmp_path))
    assert runner.invoke(app, ["info"]).exit_code == 0


# ============================================================================================
# regression (host): the real CLI on the 100-cell fixture
# ============================================================================================
def run_cli(
    args: list[str], env: dict[str, str] | None = None
) -> tuple[subprocess.CompletedProcess, float]:
    """Run ``python -m bloom.cli embed ...`` in a subprocess (fresh model per run) and time it."""
    e = dict(os.environ)
    e.update(env or {})
    t0 = time.perf_counter()
    r = subprocess.run(
        [sys.executable, "-m", "bloom.cli", *args], capture_output=True, text=True, env=e
    )
    return r, time.perf_counter() - t0


def diff_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    d = np.abs(a - b)
    cos = np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return {
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "min_cosine": float(cos.min()),
        "mean_cosine": float(cos.mean()),
    }


def fixture_tie_at_cut(lmax: int = 4096) -> np.ndarray:
    """Cells whose top-``lmax`` set is ambiguous (value tie across the cut) -> token sets may differ from the paper's (#26)."""
    a = ad.read_h5ad(FIXTURE)
    X = a.X.tocsr()
    tie = np.zeros(a.n_obs, dtype=bool)
    for i in range(a.n_obs):
        v = np.sort(X.data[X.indptr[i] : X.indptr[i + 1]])[::-1]
        if v.size > lmax and v[lmax - 1] == v[lmax]:
            tie[i] = True
    return tie


@pytest.fixture(scope="module")
def host_env(hyacinth_model_dir) -> dict[str, str]:
    """Environment for subprocess runs: BLOOM_HOME with the model + full gene table (skips otherwise)."""
    home = Path(os.environ.get("BLOOM_HOME", Path.home() / ".cache" / "bloom")).expanduser()
    if not (home / "references" / "mm10-evo2-v1" / "genes.safetensors").is_file():
        pytest.skip(
            f"full gene table not under {home}/references/mm10-evo2-v1 (bloom download --what genes)"
        )
    return {"BLOOM_HOME": str(home)}


@pytest.fixture(scope="module")
def e1_run(host_env, tmp_path_factory):
    """E1: ``bloom embed --rna rna_100.h5ad --species mouse --strategy top --lmax 4096`` on CPU (+ --keep-views)."""
    out = tmp_path_factory.mktemp("e1") / "out.h5ad"
    r, wall = run_cli(
        [
            "embed",
            "--rna",
            str(FIXTURE),
            "--species",
            "mouse",
            "--strategy",
            "top",
            "--lmax",
            "4096",
            "--device",
            "cpu",
            "--keep-views",
            "-o",
            str(out),
        ],
        host_env,
    )
    assert r.returncode == 0, r.stderr[-3000:]
    return {"path": out, "adata": ad.read_h5ad(out), "wall_s": wall, "stderr": r.stderr}


@pytest.mark.regression
def test_e1_cli_top4096_cpu_schema_and_provenance(e1_run, record_property):
    a = e1_run["adata"]
    src = ad.read_h5ad(FIXTURE)
    assert a.n_obs == 100 and a.n_vars == 0
    assert a.obs.index.tolist() == src.obs.index.tolist()
    for col in src.obs.columns:  # obs preserved
        assert a.obs[col].astype(str).tolist() == src.obs[col].astype(str).tolist()
    assert set(a.obs.columns) >= {
        "bloom_n_tokens_rna",
        "bloom_n_views_rna",
        "bloom_n_genes_detected",
    }
    X = a.obsm["X_bloom_rna"]
    assert X.shape == (100, 1024) and X.dtype == np.float32 and np.isfinite(X).all()
    assert (a.obs["bloom_n_views_rna"] == 1).all()
    assert a.obs["bloom_n_tokens_rna"].max() == 4096
    prov = a.uns["bloom"]
    assert PROVENANCE_KEYS <= set(prov)
    assert prov["model_id"] == "hyacinth" and len(prov["model_sha256"]) == 64
    assert (
        prov["reference_id"] == "mm10-evo2-v1"
        and prov["device"] == "cpu"
        and prov["precision"] == "fp32"
    )
    assert prov["modalities"]["rna"]["strategy"] == {"strategy": "top", "lmax": 4096}
    assert prov["modalities"]["rna"]["source"]["path"].endswith("rna_100.h5ad")
    # E8: --keep-views mean == X_bloom_rna (single view for top)
    V = a.obsm["X_bloom_rna_views"]
    assert V.shape == (100, 1, 1024)
    np.testing.assert_allclose(V.mean(1), X, atol=1e-6)
    record_property("e1_wall_s", e1_run["wall_s"])
    record_property("e1_run", prov["modalities"]["rna"]["run"])
    print(
        "\n[E1 top4096 CPU] wall",
        round(e1_run["wall_s"], 1),
        "s;",
        json.dumps(prov["modalities"]["rna"]["run"]),
    )


@pytest.mark.regression
def test_e1_vs_reference_report(e1_run, record_property):
    """Report max|d| / cosine vs ``rna_top4096_tf32off`` for all cells and for the cells whose
    top-4096 set has no expression tie at the cut (the two groups differed before the references were
    regenerated, #26); the gate itself is ``test_g1_cli_top4096_cpu_strict``."""
    X = e1_run["adata"].obsm["X_bloom_rna"]
    ref = load_reference(REFS_NPZ, REF_TOP)
    tie = fixture_tie_at_cut()
    report = {
        "all_cells": diff_stats(X, ref),
        "cells_without_tie_at_cut": diff_stats(X[~tie], ref[~tie]),
        "n_tie_cells": int(tie.sum()),
        "vs_paper_faithful_tf32_ref": diff_stats(X, load_reference(REFS_NPZ, REF_TOP_TF32)),
    }
    print("\n[E1 vs rna_top4096_tf32off] " + json.dumps(report, indent=1))
    for k, v in report.items():
        record_property(k, v)
    assert report["all_cells"]["min_cosine"] > 0.9999


@pytest.mark.regression
def test_g1_cli_top4096_cpu_strict(e1_run, record_property):
    """G1: CPU fp32 ``top`` 4096 through the CLI vs ``rna_top4096_tf32off``: ``max|d| < 1e-4``."""
    stats = diff_stats(e1_run["adata"].obsm["X_bloom_rna"], load_reference(REFS_NPZ, REF_TOP))
    print("\n[G1 CLI top4096 CPU vs rna_top4096_tf32off] " + json.dumps(stats))
    record_property("g1", stats)
    assert stats["max_abs"] < G1_TOLERANCE, stats


@pytest.mark.regression
def test_e2_prepare_then_embed_equals_in_memory(e1_run, host_env, tmp_path):
    bundle = tmp_path / "bundle"
    r, _ = run_cli(
        ["prepare", "--rna", str(FIXTURE), "--species", "mouse", "-o", str(bundle)], host_env
    )
    assert r.returncode == 0, r.stderr[-2000:]
    out = tmp_path / "out.h5ad"
    r, _ = run_cli(
        [
            "embed",
            str(bundle),
            "--strategy",
            "top",
            "--lmax",
            "4096",
            "--device",
            "cpu",
            "-o",
            str(out),
        ],
        host_env,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    b = ad.read_h5ad(out)
    np.testing.assert_array_equal(b.obsm["X_bloom_rna"], e1_run["adata"].obsm["X_bloom_rna"])
    assert b.uns["bloom"]["modalities"]["rna"]["source"]["path"].endswith("rna_100.h5ad")


@pytest.mark.regression
@pytest.mark.parametrize("batch_size", [4, 32])
def test_g8_batch_size_invariance_cpu(e1_run, host_env, tmp_path, batch_size, record_property):
    """G8: batch_size 1 (E1) vs 4 and 32 give identical ``top`` embeddings on CPU (< 1e-5).

    Batch 32 at L = 4096 needs ~42 GB RSS (``B x H x L x L`` attention) and skips on machines with less
    than 48 GB (CI runners); batch 4 (~6 GB) runs everywhere.
    """
    if batch_size >= 32 and 0 < total_ram_gb() < 48:
        pytest.skip(
            f"batch {batch_size} at L = 4096 needs ~42 GB RSS; this machine has {total_ram_gb():.0f} GB"
        )
    out = tmp_path / f"b{batch_size}.h5ad"
    r, wall = run_cli(
        [
            "embed",
            "--rna",
            str(FIXTURE),
            "--species",
            "mouse",
            "--strategy",
            "top",
            "--lmax",
            "4096",
            "--device",
            "cpu",
            "--batch-size",
            str(batch_size),
            "-o",
            str(out),
        ],
        host_env,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    stats = diff_stats(ad.read_h5ad(out).obsm["X_bloom_rna"], e1_run["adata"].obsm["X_bloom_rna"])
    print(f"\n[G8 batch {batch_size} vs 1] " + json.dumps(stats) + f" wall {wall:.1f}s")
    record_property(f"g8_batch{batch_size}", {**stats, "wall_s": wall})
    assert stats["max_abs"] < 1e-5, stats


@pytest.mark.regression
def test_e3_human_symbols_map_through_orthologs(host_env, tmp_path, record_property):
    """E3: the fixture with upper-cased symbols as a human input runs; mapping goes through the ortholog table."""
    a = ad.read_h5ad(FIXTURE)
    a.var_names = [s.upper() for s in a.var_names]
    a.var_names_make_unique()
    del a.layers["counts"]
    a = a[:20].copy()  # 20 cells keep the CPU cost at ~10 s
    h5 = tmp_path / "human_like.h5ad"
    a.write_h5ad(h5)
    out = tmp_path / "out.h5ad"
    r, wall = run_cli(
        [
            "embed",
            "--rna",
            str(h5),
            "--species",
            "human",
            "--strategy",
            "top",
            "--lmax",
            "4096",
            "--device",
            "cpu",
            "-o",
            str(out),
        ],
        host_env,
    )
    assert r.returncode == 0, r.stderr[-3000:]
    b = ad.read_h5ad(out)
    assert b.obsm["X_bloom_rna"].shape == (20, 1024) and np.isfinite(b.obsm["X_bloom_rna"]).all()
    bundle = tmp_path / "bundle"
    r2, _ = run_cli(
        ["prepare", "--rna", str(h5), "--species", "human", "-o", str(bundle)], host_env
    )
    assert r2.returncode == 0, r2.stderr[-2000:]
    rep = json.loads((bundle / "manifest.json").read_text())["mapping"]["rna"]
    print(
        "\n[E3 human-like] "
        + json.dumps({
            k: rep.get(k)
            for k in (
                "n_var",
                "n_mapped",
                "mapping_rate",
                "n_exact",
                "n_ortholog",
                "detected_mass_mapped",
            )
        })
    )
    record_property("e3_report", rep)
    assert rep["n_ortholog"] > 0 and rep["n_ortholog"] >= 0.95 * rep["n_mapped"]  # ortholog path
    assert rep["n_mapped"] >= 10_000  # ~half of the 32k universe has a 1:1 human ortholog
    assert "ortholog" in r.stderr


# ============================================================================================
# assets (CUDA host): G4 statistical parity, G7 CPU vs CUDA
# ============================================================================================
@pytest.mark.assets
def test_g4_resample_8192_d4_statistical_parity(host_env, tmp_path, record_property):
    """G4: ``random --lmax 8192 --draws 4 --seed 0`` vs ``rna_random_8192x4``: mean cosine > 0.999.

    Runs on the GPU when available (TF32 default), else CPU (~10 min at 1.4 s per view).
    """
    out = tmp_path / "resamp.h5ad"
    r, wall = run_cli(
        [
            "embed",
            "--rna",
            str(FIXTURE),
            "--species",
            "mouse",
            "--strategy",
            "random",
            "--lmax",
            "8192",
            "--draws",
            "4",
            "--seed",
            "0",
            "--keep-views",
            "-o",
            str(out),
        ],
        host_env,
    )
    assert r.returncode == 0, r.stderr[-3000:]
    a = ad.read_h5ad(out)
    X = a.obsm["X_bloom_rna"]
    stats = diff_stats(X, load_reference(REFS_NPZ, REF_RESAMP))
    V = a.obsm["X_bloom_rna_views"]
    view_cos = np.sum(V[:, 0] * V[:, 1], 1) / (
        np.linalg.norm(V[:, 0], axis=1) * np.linalg.norm(V[:, 1], axis=1)
    )
    report = {
        **stats,
        "wall_s": wall,
        "device": a.uns["bloom"]["device"],
        "run": a.uns["bloom"]["modalities"]["rna"]["run"],
        "view0_vs_view1_mean_cosine": float(view_cos.mean()),
    }
    print("\n[G4 random 8192x4 vs rna_random_8192x4] " + json.dumps(report, indent=1))
    for k, v in report.items():
        record_property(k, v)
    assert (a.obs["bloom_n_views_rna"] == 4).all()
    assert stats["mean_cosine"] > 0.999, stats


@pytest.mark.assets
def test_g7_cpu_vs_cuda_top4096(e1_run, host_env, tmp_path, record_property):
    """G7: CPU vs GPU ``top`` embeddings through the CLI default (``--device cuda``).

    Gate ``max|d| < 1e-3``: since #44 ``encode`` always runs with the fused CUDA fast path off, so the
    CLI default on CUDA (TF32 off) reproduces the CPU run to ~1e-5 and ``uns["bloom"]["fastpath"]``
    is ``False``. The bf16 gap vs fp32 is recorded for the hardware table (G3), as is the wall time.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    cpu = e1_run["adata"].obsm["X_bloom_rna"]
    # 1. CLI default on CUDA (unfused, TF32 off)
    out = tmp_path / "cuda.h5ad"
    r, wall = run_cli(
        [
            "embed",
            "--rna",
            str(FIXTURE),
            "--species",
            "mouse",
            "--strategy",
            "top",
            "--lmax",
            "4096",
            "--device",
            "cuda",
            "--batch-size",
            "8",
            "-o",
            str(out),
        ],
        host_env,
    )
    assert r.returncode == 0, r.stderr[-3000:]
    b = ad.read_h5ad(out)
    assert b.uns["bloom"]["device"].startswith("cuda") and b.uns["bloom"]["tf32_matmul"] is False
    assert b.uns["bloom"]["fastpath"] is False
    gpu_default = b.obsm["X_bloom_rna"]
    # 2. embed() on CUDA (same path, in-process wall) and bf16
    model = bloom.load_model("hyacinth", device="cuda")
    ds = bloom.prepare(rna=FIXTURE, species="mouse", reference=model.reference)
    t0 = time.perf_counter()
    gpu = embed(model, ds, {"rna": TopExpressed(lmax=4096)}, batch_size=8, progress=False)
    wall_embed = time.perf_counter() - t0
    gpu_bf16 = embed(
        model,
        ds,
        {"rna": TopExpressed(lmax=4096)},
        batch_size=8,
        precision="bf16",
        progress=False,
    )
    assert torch.backends.mha.get_fastpath_enabled()  # the toggle is scoped inside encode()
    report = {
        "G7_cpu_vs_gpu_cli_default": diff_stats(gpu_default, cpu),
        "gpu_embed_vs_gpu_cli": diff_stats(gpu.obsm["X_bloom_rna"], gpu_default),
        "gpu_cli_default_vs_tf32off_ref": diff_stats(
            gpu_default, load_reference(REFS_NPZ, REF_TOP)
        ),
        "gpu_cli_default_vs_paper_faithful_ref": diff_stats(
            gpu_default, load_reference(REFS_NPZ, REF_TOP_TF32)
        ),
        "bf16_vs_fp32_gpu": diff_stats(gpu_bf16.obsm["X_bloom_rna"], gpu.obsm["X_bloom_rna"]),
        "wall_cli_default_s": wall,
        "wall_embed_s": wall_embed,
        "run": b.uns["bloom"]["modalities"]["rna"]["run"],
    }
    print("\n[G7 CPU vs CUDA top4096] " + json.dumps(report, indent=1))
    for k, v in report.items():
        record_property(k, v)
    assert gpu_bf16.uns["bloom"]["precision"] == "bf16"
    assert np.isfinite(gpu_bf16.obsm["X_bloom_rna"]).all()
    assert report["G7_cpu_vs_gpu_cli_default"]["max_abs"] < G7_TOLERANCE, report
    assert report["gpu_cli_default_vs_tf32off_ref"]["max_abs"] < G7_TOLERANCE, report


# ============================================================================================
# unit: ATAC with the fake model (S7 routing, lazy peak table, P5 two-modality output, E4-fake)
# ============================================================================================
PEAK_EMB_OF_COL = np.array([3, 0, 9, 12, 7], dtype=np.int32)  # prepared atac var is in emb space
PEAK_NAMES = ["chr1:3000-3501", "chr1:1000-1501", "chr2:5000-5501", "chrX:1-502", "chr2:4000-4501"]


class FakePeakBloom(FakeBloom):
    """Fake model that also records the token kinds it saw and supports the lazy peak-table load."""

    def __init__(self, d_model: int = 16, reference=None, peak_table: bool = False) -> None:
        super().__init__(d_model, reference)
        self.has_peak_table = peak_table
        self.kinds: list[set[int]] = []
        self.peak_loads: list[str] = []

    def load_peak_table(self, source, n_rows=None, device=None):
        self.peak_loads.append(str(source))
        self.has_peak_table = True
        return self

    def encode(self, emb_idx, kind, lengths):
        for b in range(emb_idx.shape[0]):
            n = int(lengths[b])
            self.kinds.append(set(kind[b, :n].tolist()))
        return super().encode(emb_idx, kind, lengths)


def make_atac_dataset(n_cells: int = 6, seed: int = 1, obs_names=None) -> PreparedDataset:
    """6 cells x 5 reference peaks (prepared layout); cell 4 detects nothing."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 4, size=(n_cells, 5)).astype(np.float32)
    X[:, 0] += 1.0
    X[4] = 0.0
    names = obs_names or [f"cell{i}" for i in range(n_cells)]
    obs = pd.DataFrame(
        {
            "celltype": [f"a{i % 3}" for i in range(n_cells)],
            "frip": np.linspace(0.1, 0.6, n_cells),
            "bloom_n_peaks_detected": (X > 0).sum(1).astype(np.int32),
        },
        index=names,
    )
    var = pd.DataFrame(
        {"bloom_emb_idx": PEAK_EMB_OF_COL, "n_user_peaks": np.ones(5, dtype=np.int32)},
        index=PEAK_NAMES,
    )
    a = ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=var)
    a.X.sort_indices()
    a.uns["bloom"] = {"schema_version": "1.0", "reference_id": "mm10-evo2-v1", "modality": "atac"}
    return PreparedDataset(
        atac=a,
        species="mouse",
        reference_id="mm10-evo2-v1",
        report={
            "atac": {
                "n_var": 5,
                "n_mapped": 5,
                "mapping_rate": 1.0,
                "n_empty_cells": 1,
                "peak_match": "exact",
                "exact_rate": 1.0,
                "n_ref_peaks_hit": 5,
            }
        },
        source={"atac": {"path": "atac.h5ad", "sha256": None}},
    )


def expected_mean_atac(
    ds: PreparedDataset, strategy: ViewStrategy, seed: int, i: int
) -> np.ndarray:
    views = strategy.views(ds.cell_features(i, "atac"), bloom.strategies.cell_rng(seed, i))
    if not views:
        return np.full(16, np.nan, dtype=np.float32)
    acc = np.zeros(16, dtype=np.float64)
    for ts in views:
        acc += FakeBloom.features(ts.emb_idx).mean(0)
    return (acc / len(views)).astype(np.float32)


@pytest.mark.unit
def test_embed_atac_routing_defaults_and_top_by_count():
    ds = make_atac_dataset()
    model = FakePeakBloom(peak_table=True)
    with pytest.warns(UserWarning, match="1 of 6 cells have no detected peak"):
        out = embed(model, ds, {"atac": Resample(lmax=3, draws=2)}, seed=3, progress=False)
    assert "X_bloom_atac" in out.obsm and "X_bloom_rna" not in out.obsm  # S7
    assert all(k == {2} for k in model.kinds)  # only peak tokens reach the model
    assert out.obs["bloom_n_views_atac"].tolist() == [2, 2, 2, 2, 0, 2]
    assert np.isnan(out.obsm["X_bloom_atac"][4]).all()
    for i in (0, 1, 5):
        np.testing.assert_allclose(
            out.obsm["X_bloom_atac"][i],
            expected_mean_atac(ds, Resample(lmax=3, draws=2), 3, i),
            rtol=1e-6,
        )
    assert out.uns["bloom"]["modalities"]["atac"]["mapping"]["peak_match"] == "exact"
    assert out.uns["bloom"]["modalities"]["atac"]["mapping"]["n_ref_peaks_hit"] == 5
    # paper defaults for an ATAC-only dataset: random 8192 x 16 (config.json has no peak block here)
    strats = default_strategies(ds, model)
    assert set(strats) == {"atac"} and strats["atac"].params()["strategy"] == "random"
    assert strats["atac"].params()["lmax"] == 8192 and strats["atac"].params()["draws"] == 16
    # top on ATAC ranks by the prepared counts, ties by ascending emb_idx; deterministic
    top = embed(model, ds, {"atac": TopExpressed(lmax=2)}, progress=False)
    for i in range(6):
        np.testing.assert_allclose(
            top.obsm["X_bloom_atac"][i],
            expected_mean_atac(ds, TopExpressed(lmax=2), 0, i),
            rtol=1e-6,
        )
    cell0 = ds.cell_features(0, "atac")
    order = np.argsort(-cell0.peak_values, kind="stable")
    assert (
        TopExpressed(lmax=2).views(cell0, None)[0].emb_idx.tolist()
        == cell0.detected_peak_emb_idx[order[:2]].tolist()
    )
    # binary peaks (no counts) still refuse `top`
    binary = CellFeatures.peaks(0, cell0.detected_peak_emb_idx)
    with pytest.raises(ValueError, match="binary"):
        TopExpressed(lmax=2).views(binary, None)


@pytest.mark.unit
def test_embed_loads_peak_table_lazily_and_only_for_atac(synthetic_reference_dir):
    ref = ReferenceTables.from_dir(synthetic_reference_dir, require_peaks=True)
    model = FakePeakBloom(reference=ref)
    assert not model.has_peak_table
    # RNA-only: the peak table is never touched (M6 at the embed level)
    embed(model, make_dataset(), {"rna": TopExpressed(lmax=3)}, progress=False)
    assert model.peak_loads == [] and not model.has_peak_table
    # ATAC: loaded once from reference.paths, then reused
    ds = make_atac_dataset()
    embed(model, ds, {"atac": TopExpressed(lmax=3)}, progress=False)
    embed(model, ds, {"atac": TopExpressed(lmax=3)}, progress=False)
    assert model.peak_loads == [str(synthetic_reference_dir / "peaks.safetensors")]
    assert model.has_peak_table
    from bloom.embed import ensure_peak_table

    assert ensure_peak_table(model) is False  # already loaded
    # a reference without the peak files -> actionable error
    bare = FakePeakBloom(
        reference=ReferenceTables(genes_parquet=synthetic_reference_dir / "genes.parquet")
    )
    with pytest.raises(RuntimeError, match="bloom download --what peaks"):
        embed(bare, ds, {"atac": TopExpressed(lmax=3)}, progress=False)


@pytest.mark.unit
def test_embed_both_modalities_merges_obs_and_embeds_unmatched():
    from bloom.embed import embed_unmatched

    rna = make_dataset()
    atac = make_atac_dataset()  # same obs_names cell0..cell5
    atac.atac.obs["celltype"] = ["x"] * 6  # conflicts with the RNA column -> suffixed _atac
    extra = make_atac_dataset(n_cells=6, seed=5, obs_names=[f"only{i}" for i in range(6)])
    ds = PreparedDataset(
        rna=rna.rna,
        atac=atac.atac,
        species="mouse",
        reference_id="mm10-evo2-v1",
        report={**rna.report, **atac.report},
        source={**rna.source, **atac.source},
        alignment={
            "n_rna": 6,
            "n_atac": 12,
            "n_common": 6,
            "n_rna_only": 0,
            "n_atac_only": 6,
            "keep_unmatched": True,
        },
        unmatched={"atac": extra.atac},
    )
    assert ds.modalities == ("rna", "atac")
    model = FakePeakBloom(peak_table=True)
    strategies = {"rna": TopExpressed(lmax=3), "atac": Resample(lmax=3, draws=2)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = embed(model, ds, strategies, seed=1, progress=False)
    assert out.n_obs == 6 and set(out.obsm) == {"X_bloom_rna", "X_bloom_atac"}
    assert out.obs["celltype"].tolist() == rna.rna.obs["celltype"].tolist()  # RNA obs first
    assert out.obs["celltype_atac"].tolist() == ["x"] * 6
    assert out.obs["frip"].tolist() == atac.atac.obs["frip"].tolist()  # ATAC-only columns added
    assert (
        out.obs["bloom_n_peaks_detected"].tolist()
        == atac.atac.obs["bloom_n_peaks_detected"].tolist()
    )
    assert {
        "bloom_n_views_rna",
        "bloom_n_views_atac",
        "bloom_n_tokens_rna",
        "bloom_n_tokens_atac",
    } <= set(out.obs)
    assert out.uns["bloom"]["alignment"]["n_common"] == 6
    assert set(out.uns["bloom"]["modalities"]) == {"rna", "atac"}
    # each modality equals its single-modality run (same cell indices -> same draws)
    rna_only = embed(model, rna, {"rna": TopExpressed(lmax=3)}, seed=1, progress=False)
    np.testing.assert_array_equal(out.obsm["X_bloom_rna"], rna_only.obsm["X_bloom_rna"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        atac_only = embed(model, atac, {"atac": Resample(lmax=3, draws=2)}, seed=1, progress=False)
    np.testing.assert_array_equal(out.obsm["X_bloom_atac"], atac_only.obsm["X_bloom_atac"])
    # the remainder (--keep-unmatched) is embedded as its own one-modality dataset
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rest = embed_unmatched(model, ds, strategies, seed=1, progress=False)
    assert set(rest) == {"atac"} and rest["atac"].n_obs == 6
    assert rest["atac"].obs_names.tolist() == [f"only{i}" for i in range(6)]
    assert set(rest["atac"].obsm) == {"X_bloom_atac"}
    # strategies for an absent modality are still refused
    with pytest.raises(ValueError, match="absent modalities"):
        embed(model, rna, {"atac": Resample(lmax=3)}, progress=False)


def _tiny_atac_h5ad(path: Path, obs_names: list[str]) -> Path:
    """cells x the 4 synthetic reference peaks (+ one off-reference peak), counts."""
    names = [
        "chr1:3003635-3004136",
        "chr1:3007195-3007696",
        "chr2:100000-100501",
        "chrX:5000-5501",
        "chr7:100-601",
    ]
    rng = np.random.default_rng(2)
    X = rng.integers(0, 3, size=(len(obs_names), 5)).astype(np.float32)
    X[:, 0] += 1
    a = ad.AnnData(
        X=sp.csr_matrix(X),
        obs=pd.DataFrame({"atac_qc": np.arange(len(obs_names))}, index=obs_names),
        var=pd.DataFrame(index=names),
    )
    a.write_h5ad(path)
    return path


@pytest.fixture
def fake_peak_cli(monkeypatch, synthetic_reference_dir):
    """Patch ``bloom.embed.load_model`` with a peak-capable fake model + the synthetic reference."""
    ref = ReferenceTables.from_dir(synthetic_reference_dir, require_peaks=True)
    model = FakePeakBloom(reference=ref)

    def _load(
        model_id="hyacinth", device="auto", *, model_dir=None, reference_dir=None, peaks=False
    ):
        return model

    monkeypatch.setattr(importlib.import_module("bloom.embed"), "load_model", _load)
    return model


@pytest.mark.unit
def test_cli_embed_fake_model_atac_and_both(tmp_path, fake_peak_cli):
    """E4 with the fake model: --rna + --atac -> one file with both keys; --atac alone identical."""
    cells = [f"c{i}" for i in range(5)]
    rna = _tiny_h5ad(
        tmp_path / "rna.h5ad", ["Pvalb", "Gad1", "Gad2", "Sst", "Vip", "Xkr4", "NotAGene"]
    )
    atac = _tiny_atac_h5ad(tmp_path / "atac.h5ad", cells)
    both = tmp_path / "both.h5ad"
    r = runner.invoke(
        app,
        [
            "embed",
            "--rna",
            str(rna),
            "--atac",
            str(atac),
            "--species",
            "mouse",
            "--rna-strategy",
            "top",
            "--rna-lmax",
            "3",
            "--atac-draws",
            "2",
            "--atac.lmax",
            "3",
            "-o",
            str(both),
        ],
    )
    assert r.exit_code == 0, r.output
    assert "strategy rna: TopExpressed" in r.output and "strategy atac: Resample" in r.output
    assert "[atac] 5 cells x 5 user peaks" in r.output
    assert "peak match overlap (min_overlap 0.5); exact rate 80.0%" in r.output  # auto: 4/5 exact
    assert fake_peak_cli.peak_loads  # lazy load happened for the ATAC modality
    b = ad.read_h5ad(both)
    assert set(b.obsm) == {"X_bloom_rna", "X_bloom_atac"} and b.n_obs == 5
    assert b.obs["label"].tolist() == list("abcde") and b.obs["atac_qc"].tolist() == list(range(5))
    assert (
        b.obs["bloom_n_views_atac"].tolist() == [2] * 5
        and b.obs["bloom_n_views_rna"].tolist() == [1] * 5
    )
    mods = b.uns["bloom"]["modalities"]
    assert mods["rna"]["strategy"]["strategy"] == "top" and mods["rna"]["strategy"]["lmax"] == 3
    assert mods["atac"]["strategy"] == {"strategy": "random", "lmax": 3, "draws": 2}
    assert (
        mods["atac"]["mapping"]["peak_match"] == "overlap"
        and mods["atac"]["mapping"]["n_ref_peaks_hit"] == 4
    )
    assert b.uns["bloom"]["alignment"]["n_common"] == 5
    # --atac alone (species implied): identical X_bloom_atac for the same seed
    alone = tmp_path / "alone.h5ad"
    r = runner.invoke(
        app, ["embed", "--atac", str(atac), "--draws", "2", "--lmax", "3", "-o", str(alone)]
    )
    assert r.exit_code == 0, r.output
    a = ad.read_h5ad(alone)
    assert set(a.obsm) == {"X_bloom_atac"} and "bloom_n_views_rna" not in a.obs
    np.testing.assert_array_equal(a.obsm["X_bloom_atac"], b.obsm["X_bloom_atac"])
    # issue #47: an override for a modality that is not in the input is ignored with a warning
    r = runner.invoke(
        app,
        [
            "embed",
            "--atac",
            str(atac),
            "--rna-strategy",
            "top",
            "--rna-lmax",
            "3",
            "-o",
            str(alone),
        ],
    )
    assert r.exit_code == 0, r.output
    assert "warning: --rna-strategy, --rna-lmax ignored: the input has no rna modality" in r.output
    np.testing.assert_array_equal(ad.read_h5ad(alone).obsm["X_bloom_atac"], b.obsm["X_bloom_atac"])
    # deterministic ATAC `top` is batch-size invariant with the fake model as well
    t1, t4 = tmp_path / "t1.h5ad", tmp_path / "t4.h5ad"
    for p, bs in ((t1, "1"), (t4, "4")):
        r = runner.invoke(
            app,
            [
                "embed",
                "--atac",
                str(atac),
                "--strategy",
                "top",
                "--lmax",
                "3",
                "--batch-size",
                bs,
                "-o",
                str(p),
            ],
        )
        assert r.exit_code == 0, r.output
    np.testing.assert_array_equal(
        ad.read_h5ad(t1).obsm["X_bloom_atac"], ad.read_h5ad(t4).obsm["X_bloom_atac"]
    )
    # --keep-unmatched: cells present in one file only go to <stem>_<mod>_only.h5ad
    atac2 = _tiny_atac_h5ad(tmp_path / "atac2.h5ad", ["c3", "c4", "z1", "z2"])
    out = tmp_path / "emb.h5ad"
    r = runner.invoke(
        app,
        [
            "embed",
            "--rna",
            str(rna),
            "--atac",
            str(atac2),
            "--species",
            "mouse",
            "--strategy",
            "top",
            "--lmax",
            "3",
            "--keep-unmatched",
            "-o",
            str(out),
        ],
    )
    assert r.exit_code == 0, r.output
    assert "rna-only 3, atac-only 2 kept" in r.output
    e = ad.read_h5ad(out)
    assert e.obs_names.tolist() == ["c3", "c4"] and set(e.obsm) == {"X_bloom_rna", "X_bloom_atac"}
    ro = ad.read_h5ad(tmp_path / "emb_rna_only.h5ad")
    ao = ad.read_h5ad(tmp_path / "emb_atac_only.h5ad")
    assert ro.obs_names.tolist() == ["c0", "c1", "c2"] and set(ro.obsm) == {"X_bloom_rna"}
    assert ao.obs_names.tolist() == ["z1", "z2"] and set(ao.obsm) == {"X_bloom_atac"}
    assert "wrote" in r.output and "2 atac-only cell(s)" in r.output
    # without --keep-unmatched the remainder is dropped (no extra files), with a note in the report
    r = runner.invoke(
        app,
        [
            "embed",
            "--rna",
            str(rna),
            "--atac",
            str(atac2),
            "--species",
            "mouse",
            "--strategy",
            "top",
            "--lmax",
            "3",
            "-o",
            str(tmp_path / "d.h5ad"),
        ],
    )
    assert r.exit_code == 0 and "atac-only 2 dropped" in r.output
    assert not (tmp_path / "d_atac_only.h5ad").exists()
    # user errors specific to ATAC
    r = runner.invoke(app, ["embed", "--atac", str(atac), "--species", "human", "-o", str(alone)])
    assert r.exit_code == 1 and "mm10" in r.output
    far = _tiny_atac_h5ad(tmp_path / "far.h5ad", ["q0", "q1"])
    r = runner.invoke(
        app,
        ["embed", "--rna", str(rna), "--atac", str(far), "--species", "mouse", "-o", str(alone)],
    )
    assert r.exit_code == 1 and "share no obs_names" in r.output
    # a prepared two-modality bundle embeds the same way
    bundle = tmp_path / "bundle"
    r = runner.invoke(
        app,
        [
            "prepare",
            "--rna",
            str(rna),
            "--atac",
            str(atac),
            "--species",
            "mouse",
            "-o",
            str(bundle),
            "--reference-dir",
            str(fake_peak_cli.reference.paths["genes_parquet"].parent),
        ],
    )
    assert r.exit_code == 0, r.output
    r = runner.invoke(
        app,
        [
            "embed",
            str(bundle),
            "--rna-strategy",
            "top",
            "--rna-lmax",
            "3",
            "--atac-draws",
            "2",
            "--atac-lmax",
            "3",
            "-o",
            str(tmp_path / "pb.h5ad"),
        ],
    )
    assert r.exit_code == 0, r.output
    pb = ad.read_h5ad(tmp_path / "pb.h5ad")
    np.testing.assert_array_equal(pb.obsm["X_bloom_atac"], b.obsm["X_bloom_atac"])
    np.testing.assert_array_equal(pb.obsm["X_bloom_rna"], b.obsm["X_bloom_rna"])
