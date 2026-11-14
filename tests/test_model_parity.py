"""Direct-encode parity of the model port against the reference embeddings.

G1 (docs/test-plan.md section 5): ``top``-4096 embeddings of the 100-cell fixture on CPU fp32 vs the
``rna_top4096_tf32off`` array of ``tests/data/ref/rna_100_reference_embeddings.npz`` (TF32 off, fused
CUDA fast path off, column-sorted inputs). Token sets are rebuilt here exactly as
``StrategyTopExpressedGenes`` builds them (stable argsort of ``-X`` over the cell's mapped nonzero
genes), bypassing ``strategies/``, ``tokens.py`` and ``embed.py`` so a failure isolates the model port.
The sidecar records provenance, not per-cell token lists, hence the rebuild. The ``assets`` tier adds
G2a (GPU, TF32 off, same reference), G2b (GPU, TF32 "medium", vs the paper-faithful ``rna_top4096``
array; min cosine gate, the paper ran the fused kernel) and the CPU-vs-GPU check. ``encode`` always
runs with the fused CUDA fast path disabled (#44), so no test toggles it.

Needs ``$BLOOM_HOME/models/hyacinth/model.safetensors`` (497 MB) and the gene table (``genes.*``,
101 MB), both from the public bucket (``bloom download --what model,genes``). The ``regression``
tests skip with a message when they are absent (CI); the ``assets`` tests run on any machine with
the assets (GPU gates need CUDA).
"""

from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from conftest import cpu_batch_size, load_reference

from bloom.config import ModelConfig
from bloom.model import KIND_GENE, KIND_PAD, build_from_config, load_pretrained

TESTS_DATA = Path(__file__).resolve().parent / "data"
FIXTURE = TESTS_DATA / "rna_100.h5ad"
REFS_NPZ = TESTS_DATA / "ref" / "rna_100_reference_embeddings.npz"
SIDECAR = TESTS_DATA / "ref" / "rna_100_reference_embeddings.json"
REF_TF32OFF = "rna_top4096_tf32off"  # TF32 off, fused fast path off (G1 / G2a)
REF_TF32 = "rna_top4096"  # paper-faithful: TF32 on, fused fast path (evaluate.py); G2b sanity only
LMAX = 4096
G1_TOLERANCE = 1e-4
G2B_MIN_COSINE = 0.9999  # sanity vs the paper-faithful (fused-kernel) run; max |d| reported only
_CACHE: dict[str, object] = {}


# ------------------------------------------------------------------------------------------
# token sets
# ------------------------------------------------------------------------------------------
def symbol_to_emb(genes_parquet: Path) -> pd.Series:
    """``gene_name -> emb_idx`` (the parquet row, or its ``emb_idx`` column for a subset table)."""
    genes = pd.read_parquet(genes_parquet)
    emb = genes["emb_idx"].to_numpy() if "emb_idx" in genes.columns else np.arange(len(genes))
    return pd.Series(emb, index=genes["gene_name"].to_numpy())


def _rank_top(cols: np.ndarray, vals: np.ndarray, col2emb: np.ndarray, lmax: int) -> np.ndarray:
    """``StrategyTopExpressedGenes.get_idx``: mapped nonzeros, stable argsort of ``-value``."""
    keep = (vals > 0) & (col2emb[cols] >= 0)
    cols, vals = cols[keep], vals[keep]
    order = np.argsort(-vals, kind="stable")
    return col2emb[cols[order[:lmax]]]


def top_tokens(adata: ad.AnnData, sym2emb: pd.Series, lmax: int = LMAX) -> list[np.ndarray]:
    """Per cell ``emb_idx`` of the top-``lmax`` expressed mapped genes from the committed fixture.

    The fixture's CSR rows are column-sorted, so expression ties break by ascending column index
    (the S1 contract for ``TopExpressed``).
    """
    X = adata.X.tocsr()
    assert X.has_sorted_indices, "fixture CSR rows must be column-sorted (tie-break contract)"
    col2emb = sym2emb.reindex(adata.var_names).fillna(-1).to_numpy().astype(np.int64)
    return [
        _rank_top(
            X.indices[X.indptr[i] : X.indptr[i + 1]],
            X.data[X.indptr[i] : X.indptr[i + 1]],
            col2emb,
            lmax,
        )
        for i in range(X.shape[0])
    ]


def collate(
    token_lists: list[np.ndarray], lmax: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Left-packed ``(emb_idx int32, kind int8, lengths int64)`` padded to ``lmax`` (pads: -1 / 0)."""
    B = len(token_lists)
    emb_idx = torch.full((B, lmax), -1, dtype=torch.int32)
    kind = torch.full((B, lmax), KIND_PAD, dtype=torch.int8)
    lengths = torch.zeros(B, dtype=torch.int64)
    for b, t in enumerate(token_lists):
        n = len(t)
        emb_idx[b, :n] = torch.from_numpy(np.asarray(t, dtype=np.int32))
        kind[b, :n] = KIND_GENE
        lengths[b] = n
    return emb_idx, kind, lengths


def encode_all(model, tokens: list[np.ndarray], batch_size: int = 8) -> np.ndarray:
    out = np.empty((len(tokens), model.config.d_model), dtype=np.float32)
    with torch.inference_mode():
        for s in range(0, len(tokens), batch_size):
            emb_idx, kind, lengths = collate(tokens[s : s + batch_size], LMAX)
            out[s : s + batch_size] = model.encode(emb_idx, kind, lengths).cpu().numpy()
    return out


def _tf32_off() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def diff_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    d = np.abs(a - b)
    cos = np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()), "min_cosine": float(cos.min())}


# ------------------------------------------------------------------------------------------
# G1: CPU fp32 vs the TF32-off reference, committed fixture token sets
# ------------------------------------------------------------------------------------------
def encode_fixture_top4096(
    model_dir: Path, gene_paths: tuple[Path, Path], batch_size: int | None = None
) -> dict[str, object]:
    """Encode the 100 fixture cells with the top-4096 token sets on CPU fp32 (TF32 off) + timing.

    ``batch_size`` defaults to 8 on a host with enough RAM and 1 on a ~16 GB CI runner (``cpu_batch_size``;
    the embeddings do not depend on it, G8).
    """
    if batch_size is None:
        batch_size = cpu_batch_size()
    key = f"{model_dir}|{gene_paths}|{batch_size}"
    if key in _CACHE:
        return _CACHE[key]  # type: ignore[return-value]
    _tf32_off()
    genes_st, genes_pq = gene_paths
    adata = ad.read_h5ad(FIXTURE)
    tokens = top_tokens(adata, symbol_to_emb(genes_pq))
    t0 = time.perf_counter()
    model = load_pretrained(model_dir, device="cpu", strict=True, genes=genes_st)
    t_load = time.perf_counter() - t0
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0, c0 = time.perf_counter(), time.process_time()
    out = encode_all(model, tokens, batch_size)
    result = {
        "embeddings": out,
        "n_tokens": np.array([len(t) for t in tokens]),
        "wall_s": time.perf_counter() - t0,
        "cpu_s": time.process_time() - c0,
        "load_s": t_load,
        "max_rss_before_mb": rss0 / 1024,
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,  # KiB on Linux
        "batch_size": batch_size,
        "threads": torch.get_num_threads(),
        "gene_table": str(genes_st),
        "model_dir": str(model_dir),
    }
    _CACHE[key] = result
    return result


@pytest.mark.regression
def test_g1_top4096_cpu_fp32_regression(hyacinth_model_dir, gene_table_paths, record_property):
    """G1: CPU fp32 ``top`` 4096 vs ``rna_top4096_tf32off (rna_100_reference_embeddings.npz)`` (skips without assets)."""
    result = encode_fixture_top4096(hyacinth_model_dir, gene_table_paths)
    ref = load_reference(REFS_NPZ, REF_TF32OFF)
    emb = result["embeddings"]
    assert emb.shape == ref.shape == (100, 1024) and np.isfinite(emb).all()
    report = {
        **{f"G1_{k}": v for k, v in diff_stats(emb, ref).items()},
        "tolerance": G1_TOLERANCE,
        "n_tokens_min_median_max": [
            int(result["n_tokens"].min()),
            float(np.median(result["n_tokens"])),
            int(result["n_tokens"].max()),
        ],
        "cells_at_lmax": int((result["n_tokens"] == LMAX).sum()),
        **{k: v for k, v in result.items() if k not in ("embeddings", "n_tokens")},
    }
    print(
        "\n[G1 top4096 CPU fp32 vs rna_top4096_tf32off (rna_100_reference_embeddings.npz)] "
        + json.dumps(report, indent=1)
    )
    for k, v in report.items():
        record_property(k, v)
    assert report["G1_max_abs"] < G1_TOLERANCE, (
        f"G1: max|diff| = {report['G1_max_abs']:.3e} >= {G1_TOLERANCE}"
    )


# ------------------------------------------------------------------------------------------
# assets tier (CUDA host): GPU gates G2a / G2b (#44: encode always runs unfused)
# ------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def fixture_tokens(gene_table_paths):
    return top_tokens(ad.read_h5ad(FIXTURE), symbol_to_emb(gene_table_paths[1]))


def _cuda_or_skip() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")


@pytest.mark.assets
def test_g2a_gpu_tf32off_vs_reference(
    hyacinth_model_dir, gene_table_paths, fixture_tokens, record_property
):
    """G2a: GPU fp32, TF32 off (``encode`` is unfused by default, #44) vs ``rna_top4096_tf32off`` < 1e-4."""
    _cuda_or_skip()
    _tf32_off()
    assert torch.backends.mha.get_fastpath_enabled()  # the toggle is scoped inside encode()
    model = load_pretrained(hyacinth_model_dir, device="cuda", genes=gene_table_paths[0])
    out = encode_all(model, fixture_tokens)
    assert torch.backends.mha.get_fastpath_enabled()  # restored after every call
    ref = load_reference(REFS_NPZ, REF_TF32OFF)
    stats = {"G2a_gpu_tf32off_vs_ref": diff_stats(out, ref)}
    print(
        "\n[G2a GPU TF32-off vs rna_top4096_tf32off (rna_100_reference_embeddings.npz)] "
        + json.dumps(stats)
    )
    for k, v in stats.items():
        record_property(k, v)
    assert stats["G2a_gpu_tf32off_vs_ref"]["max_abs"] < G1_TOLERANCE, stats


@pytest.mark.assets
def test_g2b_gpu_tf32_vs_paper_faithful_reference(
    hyacinth_model_dir, gene_table_paths, fixture_tokens, record_property
):
    """G2b: GPU fp32 with TF32 "medium" (as evaluate.py ran; unfused since #44) vs the paper-faithful
    ``rna_top4096`` array. The paper's run used the fused kernel, so this is a sanity check only
    (min cosine > 0.9999); the max |d| is reported, not gated (docs/validation.md section 2)."""
    _cuda_or_skip()
    try:
        torch.set_float32_matmul_precision("medium")
        torch.backends.cuda.matmul.allow_tf32 = True
        model = load_pretrained(hyacinth_model_dir, device="cuda", genes=gene_table_paths[0])
        out = encode_all(model, fixture_tokens)
    finally:
        _tf32_off()
    stats = diff_stats(out, load_reference(REFS_NPZ, REF_TF32))
    stats["min_cosine_gate"] = G2B_MIN_COSINE
    print(
        "\n[G2b GPU TF32 (unfused) vs rna_top4096 (rna_100_reference_embeddings.npz)] "
        + json.dumps(stats)
    )
    for k, v in stats.items():
        record_property(k, v)
    assert stats["min_cosine"] > G2B_MIN_COSINE, stats


@pytest.mark.assets
def test_cpu_matches_gpu(hyacinth_model_dir, gene_table_paths, fixture_tokens, record_property):
    """CPU fp32 == GPU fp32 through plain ``encode`` (no toggles: unfused by default, #44): < 1e-4."""
    _cuda_or_skip()
    _tf32_off()
    tokens = fixture_tokens[:32]  # 4 batches are enough; the CPU side costs ~0.5 s per cell
    model_cpu = load_pretrained(hyacinth_model_dir, device="cpu", genes=gene_table_paths[0])
    model_gpu = load_pretrained(hyacinth_model_dir, device="cuda", genes=gene_table_paths[0])
    gpu = encode_all(model_gpu, tokens)
    cpu = encode_all(model_cpu, tokens)
    stats = {"cpu_vs_gpu": diff_stats(cpu, gpu)}
    print("\n[CPU vs GPU encode] " + json.dumps(stats))
    for k, v in stats.items():
        record_property(k, v)
    assert stats["cpu_vs_gpu"]["max_abs"] < G1_TOLERANCE, stats


# ------------------------------------------------------------------------------------------
# cheap checks that run in CI
# ------------------------------------------------------------------------------------------
@pytest.mark.regression
def test_token_recomputation_matches_sidecar_rows(gene_table_paths):
    """Rebuilt token sets are consistent with the sidecar (row order) and with the ranking rule."""
    sidecar = json.loads(SIDECAR.read_text())
    adata = ad.read_h5ad(FIXTURE)
    assert list(adata.obs_names) == sidecar["obs_names"]
    assert adata.obs["source_row"].astype(int).tolist() == sidecar["source_rows"]
    sym2emb = symbol_to_emb(gene_table_paths[1])
    tokens = top_tokens(adata, sym2emb)
    n = np.array([len(t) for t in tokens])
    assert n.max() == LMAX and n.min() > 0
    assert all(len(np.unique(t)) == len(t) for t in tokens)  # no duplicate rows within a cell
    # among *mapped* detected genes, every kept gene has a value >= every dropped gene
    X = adata.X.tocsr()
    col2emb = sym2emb.reindex(adata.var_names).fillna(-1).to_numpy().astype(np.int64)
    i = int(np.argmax(np.diff(X.indptr)))
    row = X.getrow(i)
    mapped = col2emb[row.indices] >= 0
    kept = np.isin(col2emb[row.indices], tokens[i]) & mapped
    dropped = mapped & ~kept
    assert row.data[kept].min() >= row.data[dropped].max()
    assert kept.sum() == LMAX


@pytest.mark.unit
def test_build_from_phase0_config_cpu_smoke(hyacinth_config_path):
    """Full-size model forward at a small L on CPU (no weights): shapes and finiteness."""
    torch.manual_seed(0)
    cfg = ModelConfig.from_json(hyacinth_config_path)
    model = build_from_config(cfg)
    n = 50
    model.load_gene_table({
        "cseq": torch.randn(n, cfg.d_cseq),
        "chr": torch.randint(0, 21, (n,)).to(torch.int8),
        "bp": torch.randint(0, 150_000_000, (n,)).to(torch.int32),
    })
    emb_idx, kind, lengths = collate([np.arange(0, 30), np.arange(10, 25)], lmax=32)
    with torch.inference_mode():
        h = model.encode(emb_idx, kind, lengths)
    assert h.shape == (2, 1024) and torch.isfinite(h).all()
