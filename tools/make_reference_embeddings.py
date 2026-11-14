"""Convert evaluate.py outputs to fixture-aligned .npy + .json sidecars and run the parity check."""

import datetime
import hashlib
import json
import os
import subprocess
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from _paths import env_str

ASSETS = env_str("BLOOM_STAGING")
ARCH_WORKTREE = env_str("BLOOM_ARCH_WORKTREE")
CKPT = env_str("BLOOM_TRAINING_CKPT")
RAW = f"{ASSETS}/examples/reference_embeddings/raw"
OUT = f"{ASSETS}/examples/reference_embeddings"
PRESET = sys.argv[1]
LOG = sys.argv[2]
TIMING = json.load(open(sys.argv[3])) if len(sys.argv) > 3 and os.path.exists(sys.argv[3]) else {}
ORIG = env_str("BLOOM_PAPER_RESULTS").rstrip("/") + "/"
cfg = yaml.safe_load(open(PRESET))
cells = {
    "rna_100": pd.read_csv(f"{ASSETS}/examples/fixtures/cells.tsv", sep="\t"),
    "allen_multiome_300": pd.read_csv(f"{ASSETS}/examples/allen_multiome_300/cells.tsv", sep="\t"),
}
scale_lines = [l.strip() for l in open(LOG) if "[scale-check]" in l]
export_commit = subprocess.check_output(
    ["git", "-C", os.path.expanduser("~/code/bloom"), "rev-parse", "HEAD"], text=True
).strip()
S = {}
for task, tcfg in cfg["evaluate"]["tasks"].items():
    ds = tcfg["dataset"]
    fx = cells[ds]
    a = ad.read_h5ad(f"{RAW}/{task}/embeddings.h5ad")
    emb = np.asarray(a.obsm["bloom_emb"], dtype=np.float32)
    n = emb.shape[0]
    info = {
        "task": task,
        "n_cells": int(n),
        "dim": int(emb.shape[1]),
        "obs_index_examples": a.obs_names[:3].tolist(),
        "obsm_keys": list(a.obsm.keys()),
    }
    assert n == len(fx), (n, len(fx))
    if "valid_mask" in a.obsm:
        info["n_valid"] = int(np.asarray(a.obsm["valid_mask"]).sum())
    # alignment: evaluate.py orders rows by (file_idx, row_idx) == subset-file row order == fixture obs order (both sorted by source_row)
    real_names = a.obs_names.tolist() == fx["obs_name"].tolist()
    info["obs_names_match_fixture"] = bool(real_names)
    if not real_names:
        assert list(a.obs_names) == [str(i) for i in range(n)], "unexpected obs index; cannot align"
    info["nan_rows"] = int(np.isnan(emb).any(1).sum())
    info["row_norm_mean"] = float(np.linalg.norm(emb, axis=1).mean())
    stem = {
        "rna_100_top4096": "rna_100_top4096",
        "rna_100_resamp_8192_d4": "rna_100_resamp_8192_d4",
        "allen_multiome_300_top4096": "allen_multiome_300_top4096",
        "allen_multiome_300_resamp_8192_d4": "allen_multiome_300_resamp_8192_d4",
        "allen_multiome_300_resamp_atac_8192_d16": "allen_multiome_300_resamp_atac_8192_d16",
    }[task]
    np.save(f"{OUT}/{stem}.npy", emb)
    side = {
        "file": f"{stem}.npy",
        "shape": list(emb.shape),
        "dtype": "float32",
        "row_order": f"identical to obs order of examples/{'fixtures/rna_100.h5ad' if ds == 'rna_100' else 'allen_multiome_300/rna.h5ad'} (cells.tsv; sorted by source_row)",
        "obs_names": fx["obs_name"].tolist(),
        "source_rows": fx["source_row"].tolist(),
        "producer": {
            "repo": "alleninstitute/bloom",
            "commit": export_commit,
            "script": "src/bloom/evaluate.py (unchanged) via tools/run_evaluate_checked.py",
            "env": os.path.basename(sys.prefix),
            "bloom_arch_worktree": f"{ARCH_WORKTREE} (commit ef5fb8a8e5768071da9d9542a595dd8e05b0cb03)",
            "PYTHONPATH": f"{ARCH_WORKTREE}/src",
            "seq_head_scale_checked": scale_lines[:4],
            "host": "maintainer GPU host (docs/maintainers.md)",
            "gpu": "NVIDIA L4 (23 GB)",
            "checkpoint": f"{CKPT} (hyacinth, step 530000)",
            "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "evaluate": {
            "precision": cfg["evaluate"]["precision"],
            "float32_matmul_precision": "medium (set by evaluate.py; TF32 matmuls enabled)",
            "batch_size": cfg["evaluate"]["batch_size"],
            "num_workers": cfg["evaluate"]["num_workers"],
            "seed": cfg["evaluate"]["seed"],
            "per_cell_seed": "seed + file_idx*100000 + row_idx, with file_idx=0 and row_idx = row in the subset file (== position in cells.tsv); NOT the source row, so resample draws differ from the full-dataset run",
            "peak_modality": cfg["evaluate"]["peak_modality"],
            "dataset_name_for_feature_map": "eval_balanced_50_allen_multiome",
            "reference_space": "mm10 (evo2_mean_ctx_2048_pca_768_norm)",
        },
        "strategy": tcfg["strategy"],
        "aggregation": "mean of per-draw h_cell over draws (EvalWriter online mean)",
        "timing": TIMING.get(task),
    }
    json.dump(side, open(f"{OUT}/{stem}.json", "w"), indent=2)
    info["npy_bytes"] = os.path.getsize(f"{OUT}/{stem}.npy")
    S[task] = info
    print(task, {k: v for k, v in info.items() if k != "obs_index_examples"}, flush=True)

# ---- parity check vs the paper's full-dataset run (same cells by source_row) ----
par = {}
for task, od in [
    ("allen_multiome_300_top4096", "eval_balanced_50_allen_multiome_nz_top4096"),
    ("rna_100_top4096", "eval_balanced_50_allen_multiome_nz_top4096"),
    ("allen_multiome_300_resamp_8192_d4", "eval_balanced_50_allen_multiome_nz_resamp_8192_d4"),
    (
        "allen_multiome_300_resamp_atac_8192_d16",
        "eval_balanced_50_allen_multiome_nz_resamp_atac_8192_d16",
    ),
]:
    p = f"{ORIG}/{od}/embeddings.h5ad"
    if not os.path.exists(p):
        par[task] = {"error": f"missing {p}"}
        continue
    o = ad.read_h5ad(p)
    ds = cfg["evaluate"]["tasks"][task]["dataset"]
    fx = cells[ds]
    src = fx["source_row"].to_numpy()
    O = np.asarray(o.obsm["bloom_emb"], dtype=np.float32)
    assert O.shape[0] == 15750, O.shape
    ref = O[src]
    mine = np.load(f"{OUT}/{task}.npy")
    # obs-name sanity if the original has real names
    names_ok = None
    if not o.obs_names[0].isdigit():
        names_ok = bool((o.obs_names[src] == fx["obs_name"].to_numpy()).all())
    d = np.abs(mine - ref)
    cos = (mine * ref).sum(1) / (np.linalg.norm(mine, axis=1) * np.linalg.norm(ref, axis=1))
    par[task] = {
        "orig_dir": od,
        "orig_obs_index_real_names_match": names_ok,
        "max_abs_diff": float(d.max()),
        "mean_abs_diff": float(d.mean()),
        "p99_abs_diff": float(np.quantile(d, 0.99)),
        "cos_min": float(cos.min()),
        "cos_mean": float(cos.mean()),
        "n_cells_cos_below_0.99999": int((cos < 0.99999).sum()),
        "n_cells_maxdiff_above_1e-4": int((d.max(1) > 1e-4).sum()),
        "ref_row_norm_mean": float(np.linalg.norm(ref, axis=1).mean()),
        "passes_1e-4": bool(d.max() < 1e-4),
        "note": "deterministic strategy: expected bit-close"
        if "top" in task
        else "resample: per-cell seeds differ (row_idx in subset != source row) -> statistical comparison only",
    }
    print("PARITY", task, par[task], flush=True)
json.dump({"tasks": S, "parity": par}, open("reference_embeddings_summary.json", "w"), indent=2)
print("DONE")
