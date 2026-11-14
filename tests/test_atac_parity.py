"""ATAC parity on the 20-cell multiome fixture (``assets`` tier; issue #18, docs/test-plan.md G5 / G7 / G8 / E4).

The fixture (``examples/multiome_20/{rna,atac}.h5ad``, ``cells.tsv``) and its reference embeddings
(``reference_embeddings.npz``: ``rna_top4096[_tf32off]``, ``rna_random_8192x4[_tf32off]``,
``atac_random_8192x16[_tf32off]``; sidecar ``reference_embeddings.json``) come from the public bucket
(``bloom download --what examples``); the ATAC runs need the 5.4 GB peak table (``--what peaks``) and the
model weights. Decision 2026-11-13 (user): the ATAC gates run on the **CPU path**, plus **one** GPU smoke:

- G5 (statistical): ``bloom embed --atac ... --device cpu`` with the paper defaults (``random`` 8192 x 16,
  seed 0) vs ``atac_random_8192x16_tf32off`` -> mean per-cell cosine > 0.999 (max |d| and min cosine are
  reported). ``BLOOM_ATAC_G5_DRAWS`` (default 16) shortens the CPU run (16 draws x 20 cells at L = 8192 is
  ~7-8 min on 16 threads).
- deterministic ``top`` (by count, lmax 8192): batch 1 vs 4 on CPU < 1e-5 (G8 for peaks) and one GPU run vs
  CPU < 1e-3 (G7 gate; ``encode`` is unfused on CUDA by default since #44).
- E4: ``--rna ... --atac ...`` -> one file with both keys; ``--atac`` alone gives an identical
  ``X_bloom_atac`` (same seed, same cell order).
"""

from __future__ import annotations

import json
import os
import resource
import time

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from conftest import load_reference

import bloom
from bloom.embed import embed
from bloom.strategies import TopExpressed

ATAC_ARRAYS = ("atac_random_8192x16", "atac_random_8192x16_tf32off")
RNA_ARRAYS = (
    "rna_top4096",
    "rna_top4096_tf32off",
    "rna_random_8192x4",
    "rna_random_8192x4_tf32off",
)
G5_MEAN_COSINE = 0.999
TOP_BATCH_TOLERANCE = 1e-5
TOP_CPU_GPU_GATE = 1e-3  # G7 gate; encode is unfused on CUDA by default (#44)


def diff_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    d = np.abs(a - b)
    cos = np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return {
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "min_cosine": float(cos.min()),
        "mean_cosine": float(cos.mean()),
    }


def peak_rss_gb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / 1e6 if os.uname().sysname == "Linux" else ru / 1e9  # KB on Linux, bytes on macOS


@pytest.mark.assets
def test_multiome_20_fixture_and_references_are_consistent(multiome_20_dir):
    """Fixture files agree with each other and with the npz / sidecar (row order, cell ids, shapes)."""
    cells = pd.read_csv(multiome_20_dir / "cells.tsv", sep="\t")
    rna = ad.read_h5ad(multiome_20_dir / "rna.h5ad")
    atac = ad.read_h5ad(multiome_20_dir / "atac.h5ad")
    side = json.loads((multiome_20_dir / "reference_embeddings.json").read_text())
    assert rna.n_obs == atac.n_obs == len(cells) == 20
    assert (
        list(rna.obs_names)
        == list(atac.obs_names)
        == cells["obs_name"].tolist()
        == side["obs_names"]
    )
    assert rna.obs["source_row"].tolist() == atac.obs["source_row"].tolist() == side["source_rows"]
    assert rna.obs["AIT33_subclass"].nunique() == 10
    assert "counts" in rna.layers and rna.X.dtype == np.float32
    assert atac.X.dtype == np.uint32 and "emb_idx" in atac.var.columns
    assert int(np.diff(atac.X.tocsr().indptr).min()) >= 8192  # every ATAC view is full length
    with np.load(multiome_20_dir / "reference_embeddings.npz") as z:
        assert list(z["obs_names"]) == side["obs_names"]
        for name in RNA_ARRAYS + ATAC_ARRAYS:
            assert z[name].shape == (20, 1024) and z[name].dtype == np.float32, name
            assert np.isfinite(z[name]).all(), name
            assert name in side["arrays"]
    for name in ATAC_ARRAYS:
        assert side["arrays"][name]["modality"] == "atac"
        assert side["arrays"][name]["strategy"]["feature_modality"] == "peak"


@pytest.fixture(scope="module")
def cpu_model(hyacinth_model_dir, peak_tensors_dir):
    """The released model on CPU with the gene table; the peak table is loaded lazily by ``embed``."""
    return bloom.load_model("hyacinth", device="cpu")


@pytest.fixture(scope="module")
def atac_ds(multiome_20_dir, cpu_model):
    """``prepare(atac=multiome_20/atac.h5ad)``: exact matching of the 483k fixture peaks onto the reference."""
    t0 = time.perf_counter()
    ds = bloom.prepare(
        atac=ad.read_h5ad(multiome_20_dir / "atac.h5ad"),
        species="mouse",
        reference=cpu_model.reference,
    )
    rep = ds.report["atac"]
    print(
        f"\n[prepare --atac multiome_20] {time.perf_counter() - t0:.1f} s; peak_match {rep['peak_match']} "
        f"(exact rate {rep['exact_rate']:.4f}); mapped {rep['n_mapped']}/{rep['n_var']} user peaks; "
        f"{rep['n_ref_peaks_hit']} reference peaks hit; detected per cell median {rep['median_detected']}"
    )
    return ds


@pytest.mark.assets
def test_prepare_atac_multiome_20_exact_matches_fixture_emb_idx(multiome_20_dir, atac_ds):
    """The fixture's own ``var["emb_idx"]`` equals our exact match for every peak (R8 on real data)."""
    atac = ad.read_h5ad(multiome_20_dir / "atac.h5ad")
    rep = atac_ds.report["atac"]
    assert rep["peak_match"] == "exact" and rep["exact_rate"] == pytest.approx(1.0)
    assert rep["n_mapped"] == rep["n_var"] == atac.n_vars and rep["n_empty_cells"] == 0
    # the hit reference peaks are exactly those detected in >= 1 cell; their emb_idx match the fixture
    nz_cols = np.flatnonzero(np.diff(atac.X.tocsc().indptr) > 0)  # peaks detected in >= 1 cell
    expected = np.sort(atac.var["emb_idx"].to_numpy()[nz_cols])
    assert atac_ds.atac.var["bloom_emb_idx"].to_numpy().tolist() == expected.tolist()
    assert (atac_ds.atac.var["n_user_peaks"] == 1).all()
    assert (
        atac_ds.atac.obs["bloom_n_peaks_detected"].tolist()
        == np.diff(atac.X.tocsr().indptr).tolist()
    )


@pytest.mark.assets
def test_g5_atac_random_8192x16_cpu_statistical_parity(
    multiome_20_dir, cpu_model, atac_ds, record_property
):
    """G5: ``random`` 8192 x 16, seed 0 on CPU vs ``atac_random_8192x16_tf32off``: mean cosine > 0.999."""
    draws = int(os.environ.get("BLOOM_ATAC_G5_DRAWS", "16"))
    torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))
    strategies = {"atac": bloom.strategies.Resample(lmax=8192, draws=draws)}
    rss0 = peak_rss_gb()
    t0 = time.perf_counter()
    out = embed(cpu_model, atac_ds, strategies, seed=0, device="cpu", progress=False)
    wall = time.perf_counter() - t0
    X = out.obsm["X_bloom_atac"]
    assert X.shape == (20, 1024) and np.isfinite(X).all()
    assert (out.obs["bloom_n_views_atac"] == draws).all()
    assert (out.obs["bloom_n_tokens_atac"] == 8192).all()  # every fixture cell has >= 8,276 peaks
    report = {
        "draws": draws,
        "vs_tf32off": diff_stats(
            X, load_reference(multiome_20_dir / "reference_embeddings.npz", ATAC_ARRAYS[1])
        ),
        "vs_paper_faithful": diff_stats(
            X, load_reference(multiome_20_dir / "reference_embeddings.npz", ATAC_ARRAYS[0])
        ),
        "wall_s": round(wall, 1),
        "s_per_view": round(wall / (20 * draws), 3),
        "s_per_cell": round(wall / 20, 2),
        "peak_rss_gb_before": round(rss0, 2),
        "peak_rss_gb_after": round(peak_rss_gb(), 2),
        "torch_threads": torch.get_num_threads(),
        "run": out.uns["bloom"]["modalities"]["atac"]["run"],
    }
    print("\n[G5 atac random 8192x%d CPU vs references] " % draws + json.dumps(report, indent=1))
    for k, v in report.items():
        record_property(k, v)
    assert report["vs_tf32off"]["mean_cosine"] > G5_MEAN_COSINE, report


@pytest.fixture(scope="module")
def top_cpu(cpu_model, atac_ds):
    """Deterministic ATAC ``top`` (by count, lmax 8192) on CPU, batch 1."""
    t0 = time.perf_counter()
    out = embed(
        cpu_model,
        atac_ds,
        {"atac": TopExpressed(lmax=8192)},
        batch_size=1,
        device="cpu",
        progress=False,
    )
    return {"X": out.obsm["X_bloom_atac"], "wall_s": time.perf_counter() - t0, "adata": out}


@pytest.mark.assets
def test_atac_top_batch_size_invariance_cpu(cpu_model, atac_ds, top_cpu, record_property):
    """G8 for peaks: ``top`` 8192 with batch 1 vs 4 on CPU differs by < 1e-5."""
    out4 = embed(
        cpu_model,
        atac_ds,
        {"atac": TopExpressed(lmax=8192)},
        batch_size=4,
        device="cpu",
        progress=False,
    )
    stats = diff_stats(out4.obsm["X_bloom_atac"], top_cpu["X"])
    assert (top_cpu["adata"].obs["bloom_n_tokens_atac"] == 8192).all()
    report = {
        **stats,
        "wall_batch1_s": round(top_cpu["wall_s"], 1),
        "s_per_cell_batch1": round(top_cpu["wall_s"] / 20, 2),
    }
    print("\n[ATAC top 8192 batch 4 vs 1, CPU] " + json.dumps(report, indent=1))
    for k, v in report.items():
        record_property(k, v)
    assert stats["max_abs"] < TOP_BATCH_TOLERANCE, stats


@pytest.mark.assets
def test_atac_top_cuda_vs_cpu_smoke(
    multiome_20_dir, atac_ds, top_cpu, peak_tensors_dir, hyacinth_model_dir, record_property
):
    """The one GPU smoke: ``top`` 8192 on the GPU vs CPU (G7 for peaks).

    Gate ``max |d| < 1e-3``: ``encode`` runs with the fused CUDA fast path off by default (#44), so the
    plain ``embed()`` on CUDA (what ``bloom embed --device cuda`` does; TF32 off) reproduces the CPU
    run to ~1e-5. Wall, s/cell and peak VRAM feed the hardware table (docs/validation.md).
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model = bloom.load_model("hyacinth", device="cuda")
    t_load = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = embed(model, atac_ds, {"atac": TopExpressed(lmax=8192)}, batch_size=8, progress=False)
    wall = time.perf_counter() - t0
    assert model.has_peak_table and model.peak_cseq.device.type == "cuda"
    assert out.uns["bloom"]["fastpath"] is False
    stats = diff_stats(out.obsm["X_bloom_atac"], top_cpu["X"])
    report = {
        "G7_atac_cpu_vs_gpu": stats,
        "wall_s": round(wall, 1),
        "s_per_cell": round(wall / 20, 3),
        "model_load_s": round(t_load, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "device_name": torch.cuda.get_device_name(0),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "run": out.uns["bloom"]["modalities"]["atac"]["run"],
    }
    print("\n[ATAC top 8192 GPU vs CPU] " + json.dumps(report, indent=1))
    for k, v in report.items():
        record_property(k, v)
    model.clear_peak_table()
    del model
    torch.cuda.empty_cache()
    assert stats["max_abs"] < TOP_CPU_GPU_GATE, stats


@pytest.mark.assets
def test_e4_cli_rna_and_atac_one_output_and_atac_alone_identical(
    multiome_20_dir, hyacinth_model_dir, peak_tensors_dir, tmp_path
):
    """E4: one output with ``X_bloom_rna`` + ``X_bloom_atac``; ``--atac`` alone reproduces ``X_bloom_atac``.

    ``top`` on both modalities keeps the CPU run short (one view per cell); the two-modality alignment keeps
    the fixture's cell order, so the per-cell seeds coincide and the ATAC embeddings are identical.
    """
    import subprocess
    import sys

    env = dict(os.environ)
    both = tmp_path / "both.h5ad"
    args = [
        sys.executable, "-m", "bloom.cli", "embed",
        "--rna", str(multiome_20_dir / "rna.h5ad"), "--atac", str(multiome_20_dir / "atac.h5ad"),
        "--species", "mouse", "--rna-strategy", "top", "--atac-strategy", "top", "--atac-lmax", "8192",
        "--device", "cpu", "-o", str(both),
    ]  # fmt: skip
    r = subprocess.run(args, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "[atac] 20 cells x 483249 user peaks" in r.stderr and "peak match exact" in r.stderr
    b = ad.read_h5ad(both)
    assert set(b.obsm) == {"X_bloom_rna", "X_bloom_atac"} and b.n_obs == 20
    assert (
        b.obs_names.tolist()
        == json.loads((multiome_20_dir / "reference_embeddings.json").read_text())["obs_names"]
    )
    assert np.isfinite(b.obsm["X_bloom_rna"]).all() and np.isfinite(b.obsm["X_bloom_atac"]).all()
    assert b.uns["bloom"]["modalities"]["rna"]["strategy"]["lmax"] == 4096
    assert b.uns["bloom"]["modalities"]["atac"]["strategy"]["lmax"] == 8192
    assert b.uns["bloom"]["modalities"]["atac"]["mapping"]["peak_match"] == "exact"
    assert b.uns["bloom"]["alignment"]["n_common"] == 20
    # RNA half equals the RNA reference (top 4096, CPU math)
    rna_ref = load_reference(multiome_20_dir / "reference_embeddings.npz", "rna_top4096_tf32off")
    assert diff_stats(b.obsm["X_bloom_rna"], rna_ref)["max_abs"] < 1e-4
    alone = tmp_path / "alone.h5ad"
    args = [
        sys.executable, "-m", "bloom.cli", "embed", "--atac", str(multiome_20_dir / "atac.h5ad"),
        "--strategy", "top", "--lmax", "8192", "--device", "cpu", "-o", str(alone),
    ]  # fmt: skip
    r = subprocess.run(args, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-3000:]
    a = ad.read_h5ad(alone)
    assert set(a.obsm) == {"X_bloom_atac"} and a.uns["bloom"]["species"] == "mouse"
    np.testing.assert_array_equal(a.obsm["X_bloom_atac"], b.obsm["X_bloom_atac"])
