"""Tile (preset ``tile``) parity on the 20-cell multiome fixture (``assets`` tier; issue #31, docs/test-plan.md G9).

References ``rna_tile_4096x2[_tf32off]`` and ``atac_tile_8192x4[_tf32off]`` in ``examples/multiome_20/reference_embeddings.npz``
were produced by the original ``evaluate.py`` (``StrategyTiledPool(draw_mode="tile")``, ``tools/hydra/
multiome_20_tile_reference_embeddings.yaml``): RNA ``Lmax 4096, max_draws 2`` (every fixture cell detects 3,407-6,834 genes, so
17/20 cells get two disjoint draws), ATAC ``Lmax 8192, max_draws 4`` (cells detect 8,276-36,504 peaks -> 2-4 draws).

Two gates, both on the **CPU path** like G5:

- G9a (deterministic): ``evaluate.py`` seeds each cell with ``np.random.default_rng(42 + row_idx)`` and permutes the detected
  features in dataset *column* order (``np.intersect1d`` output). :class:`EvaluatePyTile` reproduces that permutation on top of
  the same ``Tile`` chunking (``ceil(n / lmax)`` disjoint chunks capped at ``max_draws``), so the token sets are identical and
  the embeddings must agree to numerics: ``max |d| < 1e-4`` vs the TF32-off reference (G1 level; measured ~1e-6).
- G9b (statistical): the exposed preset with bloom's own per-cell seeds (``--strategy tile``, seed 0) vs the same reference:
  mean per-cell cosine > 0.995. Two (RNA) or <= 4 (ATAC) *disjoint* random draws have more sampling variance than ``random``
  (RNA ``random`` 8192 draws are the full detected set -> no variance; ATAC ``random`` averages 16 draws), so the ``random``
  gate of 0.999 is not reachable: the strategy's own seed-to-seed self-consistency on this fixture is mean cosine 0.9976-0.9984
  (RNA) / 0.9983-0.9987 (ATAC), and the parity vs ``evaluate.py`` sits at exactly that level (docs/validation.md section 10).
"""

from __future__ import annotations

import json
import os
import time

import anndata as ad
import numpy as np
import pytest
import torch
from conftest import load_reference

import bloom
from bloom.embed import embed
from bloom.strategies import Tile, ViewStrategy
from bloom.tokens import CellFeatures, TokenSet

RNA_LMAX, RNA_DRAWS = 4096, 2
ATAC_LMAX, ATAC_DRAWS = 8192, 4
G9A_TOLERANCE = 1e-4  # deterministic: evaluate.py's partition reproduced
G9B_MEAN_COSINE = 0.995  # statistical: bloom's own seeds (self-consistency floor ~0.998)
EVALUATE_PY_SEED = 42  # evaluate.py: cell_seed = seed + file_idx * 100_000 + row_idx, file_idx = 0


def diff_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    d = np.abs(a - b)
    cos = np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return {
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "min_cosine": float(cos.min()),
        "mean_cosine": float(cos.mean()),
    }


class EvaluatePyTile(ViewStrategy):
    """``Tile`` with ``evaluate.py``'s RNG stream: ``default_rng(42 + row_idx)`` over the detected features in column order.

    ``StrategyTiledPool._draw_chunks``: ``P = np.intersect1d(all_idx, nonzero_idx)`` (sorted by dataset column),
    ``P = P[rng.permutation(len(P))]``, ``min(max_draws, ceil(len(P) / Lmax))`` consecutive ``Lmax`` chunks. Token order
    within a view does not enter the model, so each chunk is emitted in ascending ``emb_idx`` like :class:`Tile`.
    """

    name = "tile_evaluate_py"

    def __init__(self, lmax: int, max_draws: int, emb_to_col: np.ndarray, modality: str) -> None:
        super().__init__(lmax)
        self.max_draws, self.emb_to_col, self.modality = int(max_draws), emb_to_col, modality

    def sample(self, cell: CellFeatures, rng: np.random.Generator) -> list[TokenSet]:
        emb = cell.detected_gene_emb_idx if self.modality == "gene" else cell.detected_peak_emb_idx
        if emb.size == 0:
            return []
        cols = self.emb_to_col[emb]
        assert (cols >= 0).all(), "every detected feature must map back to a dataset column"
        P = emb[np.argsort(cols)]  # ascending column order, as np.intersect1d returns it
        r = np.random.default_rng(EVALUATE_PY_SEED + int(cell.cell_index))
        P = P[r.permutation(len(P))]
        n = min(self.max_draws, int(np.ceil(len(P) / self.lmax)))
        return [
            TokenSet.of(np.sort(P[k * self.lmax : (k + 1) * self.lmax]), self.modality)
            for k in range(n)
        ]

    def params(self) -> dict[str, object]:
        return {**super().params(), "draws": self.max_draws, "rng": "evaluate.py"}


def _emb_to_col(var, n_rows: int) -> np.ndarray:
    """``emb_idx -> column of the fixture file`` (the eval inputs share the fixture's column order)."""
    m = np.full(n_rows, -1, dtype=np.int64)
    e = var["bloom_emb_idx"].to_numpy()
    ok = e >= 0
    m[e[ok]] = np.flatnonzero(ok)
    return m


@pytest.fixture(scope="module")
def cpu_model(hyacinth_model_dir, peak_tensors_dir):
    torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))
    return bloom.load_model("hyacinth", device="cpu")


@pytest.fixture(scope="module")
def both_ds(multiome_20_dir, cpu_model):
    """``prepare(rna, atac)`` of the fixture: 20 aligned cells, exact peak matching."""
    return bloom.prepare(
        rna=ad.read_h5ad(multiome_20_dir / "rna.h5ad"),
        atac=ad.read_h5ad(multiome_20_dir / "atac.h5ad"),
        species="mouse",
        reference=cpu_model.reference,
    )


def _refs(multiome_20_dir, flavour: str) -> dict[str, np.ndarray]:
    npz = multiome_20_dir / "reference_embeddings.npz"
    return {
        "rna": load_reference(npz, f"rna_tile_{RNA_LMAX}x{RNA_DRAWS}{flavour}"),
        "atac": load_reference(npz, f"atac_tile_{ATAC_LMAX}x{ATAC_DRAWS}{flavour}"),
    }


@pytest.mark.assets
def test_tile_reference_arrays_present(multiome_20_dir):
    side = json.loads((multiome_20_dir / "reference_embeddings.json").read_text())
    for name in ("rna_tile_4096x2", "atac_tile_8192x4"):
        for suffix in ("", "_tf32off"):
            arr = side["arrays"][name + suffix]
            assert arr["strategy"]["draw_mode"] == "tile" and arr["shape"] == [20, 1024]
    assert side["arrays"]["rna_tile_4096x2"]["strategy"]["max_draws"] == RNA_DRAWS
    assert side["arrays"]["atac_tile_8192x4"]["strategy"]["max_draws"] == ATAC_DRAWS


@pytest.mark.assets
def test_g9a_tile_deterministic_parity_with_evaluate_py_partition(
    multiome_20_dir, cpu_model, both_ds, record_property
):
    """G9a: reproducing evaluate.py's per-cell permutation makes Tile bit-level (< 1e-4) equal to the reference."""
    strategies = {
        "rna": EvaluatePyTile(
            RNA_LMAX, RNA_DRAWS, _emb_to_col(both_ds.rna.var, cpu_model.reference.n_genes), "gene"
        ),
        "atac": EvaluatePyTile(
            ATAC_LMAX,
            ATAC_DRAWS,
            _emb_to_col(both_ds.atac.var, cpu_model.reference.n_peaks),
            "peak",
        ),
    }
    t0 = time.perf_counter()
    out = embed(cpu_model, both_ds, strategies, device="cpu", progress=False)
    wall = time.perf_counter() - t0
    refs = _refs(multiome_20_dir, "_tf32off")
    report = {
        "rna_vs_tf32off": diff_stats(out.obsm["X_bloom_rna"], refs["rna"]),
        "atac_vs_tf32off": diff_stats(out.obsm["X_bloom_atac"], refs["atac"]),
        "rna_n_views": out.obs["bloom_n_views_rna"].tolist(),
        "atac_n_views": out.obs["bloom_n_views_atac"].tolist(),
        "wall_s": round(wall, 1),
        "torch_threads": torch.get_num_threads(),
    }
    print(
        "\n[G9a tile, evaluate.py partition reproduced, CPU vs *_tile_*_tf32off] "
        + json.dumps(report, indent=1)
    )
    for k, v in report.items():
        record_property(k, v)
    # every RNA cell with > 4096 detected genes gets 2 disjoint draws; ATAC cells 2-4 (ceil(n / 8192), capped at 4)
    n_genes = out.obs["bloom_n_genes_detected"].to_numpy()
    assert (
        out.obs["bloom_n_views_rna"].to_numpy()
        == np.minimum(RNA_DRAWS, np.ceil(n_genes / RNA_LMAX))
    ).all()
    n_peaks = out.obs["bloom_n_peaks_detected"].to_numpy()
    assert (
        out.obs["bloom_n_views_atac"].to_numpy()
        == np.minimum(ATAC_DRAWS, np.ceil(n_peaks / ATAC_LMAX))
    ).all()
    assert report["rna_vs_tf32off"]["max_abs"] < G9A_TOLERANCE, report["rna_vs_tf32off"]
    assert report["atac_vs_tf32off"]["max_abs"] < G9A_TOLERANCE, report["atac_vs_tf32off"]


@pytest.mark.assets
def test_g9b_tile_preset_statistical_parity(multiome_20_dir, cpu_model, both_ds, record_property):
    """G9b: the exposed preset (bloom seeds, seed 0) vs the evaluate.py reference: mean cosine > 0.995 per modality."""
    strategies = {"rna": Tile(RNA_LMAX, RNA_DRAWS), "atac": Tile(ATAC_LMAX, ATAC_DRAWS)}
    t0 = time.perf_counter()
    out = embed(cpu_model, both_ds, strategies, seed=0, device="cpu", progress=False)
    wall = time.perf_counter() - t0
    refs_off, refs_on = _refs(multiome_20_dir, "_tf32off"), _refs(multiome_20_dir, "")
    report = {
        "rna_vs_tf32off": diff_stats(out.obsm["X_bloom_rna"], refs_off["rna"]),
        "atac_vs_tf32off": diff_stats(out.obsm["X_bloom_atac"], refs_off["atac"]),
        "rna_vs_paper_faithful": diff_stats(out.obsm["X_bloom_rna"], refs_on["rna"]),
        "atac_vs_paper_faithful": diff_stats(out.obsm["X_bloom_atac"], refs_on["atac"]),
        "strategy": {m: out.uns["bloom"]["modalities"][m]["strategy"] for m in ("rna", "atac")},
        "run": {m: out.uns["bloom"]["modalities"][m]["run"] for m in ("rna", "atac")},
        "wall_s": round(wall, 1),
    }
    print("\n[G9b tile preset seed 0, CPU vs *_tile_*_tf32off] " + json.dumps(report, indent=1))
    for k, v in report.items():
        record_property(k, v)
    # uns provenance drops None values (`universe`)
    assert report["strategy"]["rna"] == {"strategy": "tile", "lmax": RNA_LMAX, "draws": RNA_DRAWS}
    assert report["strategy"]["atac"] == {
        "strategy": "tile",
        "lmax": ATAC_LMAX,
        "draws": ATAC_DRAWS,
    }
    assert report["rna_vs_tf32off"]["mean_cosine"] > G9B_MEAN_COSINE, report["rna_vs_tf32off"]
    assert report["atac_vs_tf32off"]["mean_cosine"] > G9B_MEAN_COSINE, report["atac_vs_tf32off"]
