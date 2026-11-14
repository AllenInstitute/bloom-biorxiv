"""Phase 1 review (issues #26 / #27): turn the re-run evaluate.py outputs into the top4096 reference embeddings.

Inputs (written by the evaluate.py job, see PHASE1_REVIEW / docs/assets.md):
  <RUNS>/tf32/<task>/embeddings.h5ad      paper-faithful run (TF32 "medium", fused CUDA fast path on)
  <RUNS>/tf32off/<task>/embeddings.h5ad   TF32 off ("highest", cuDNN TF32 off) AND torch.backends.mha fast path off
  <RUNS>/{tf32,tf32off}.log               run logs ([scale-check], [tf32-check], [fastpath-check] lines)
  <RUNS>/timing.json                      per-run wall / GPU peak written by the job
for task in {rna_100_top4096, allen_multiome_300_top4096}; eval inputs = evalinputs_sorted/ (column-sorted CSR).

What it does, per task:
  1. moves the Phase 0 (unsorted-input) files to examples/reference_embeddings/legacy_unsorted/ (npy, json, raw/),
  2. writes <task>.npy / <task>_tf32off.npy (float32, rows = fixture obs order = cells.tsv) + .json sidecars,
  3. copies the raw embeddings.h5ad under raw/<task>/ and raw/<task>_tf32off/,
  4. reports parity vs the paper's full-dataset run (by source_row), vs the legacy references, and tf32 vs tf32off,
  5. updates eval_timing_per_task.json + reference_embeddings_summary.json in the cwd (inputs of tools/make_manifest.py).
Run from a copy of tools/phase0_outputs/reference_embeddings/ (e.g. tools/phase1_outputs/reference_embeddings/).
Usage: python regenerate_top4096_refs.py <RUNS>
"""

import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd
from _paths import env_str

ASSETS = env_str("BLOOM_STAGING")
OUT = f"{ASSETS}/examples/reference_embeddings"
LEGACY = f"{OUT}/legacy_unsorted"
RUNS = sys.argv[1]
ORIG = (
    f"{env_str('BLOOM_PAPER_RESULTS')}/eval_balanced_50_allen_multiome_nz_top4096/embeddings.h5ad"
)
SORTED = env_str("BLOOM_EVAL_INPUTS_SORTED")
WORKTREE = env_str("BLOOM_ARCH_WORKTREE")
CKPT = env_str("BLOOM_TRAINING_CKPT")
TASKS = {"rna_100_top4096": "rna_100", "allen_multiome_300_top4096": "allen_multiome_300"}
cells = {
    "rna_100": pd.read_csv(f"{ASSETS}/examples/fixtures/cells.tsv", sep="\t"),
    "allen_multiome_300": pd.read_csv(f"{ASSETS}/examples/allen_multiome_300/cells.tsv", sep="\t"),
}
fixtures = {
    "rna_100": f"{ASSETS}/examples/fixtures/rna_100.h5ad",
    "allen_multiome_300": f"{ASSETS}/examples/allen_multiome_300/rna.h5ad",
}
sorted_info = json.load(open(f"{SORTED}/SORTED_INPUTS.json"))
timing = json.load(open(f"{RUNS}/timing.json"))
export_commit = subprocess.check_output(
    ["git", "-C", os.path.expanduser("~/code/bloom"), "rev-parse", "HEAD"], text=True
).strip()
arch_commit = subprocess.check_output(
    ["git", "-C", WORKTREE, "rev-parse", "HEAD"], text=True
).strip()
now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def log_lines(flavour, tag):
    return [l.strip() for l in open(f"{RUNS}/{flavour}.log") if tag in l]


def stats(a, b):
    d = np.abs(a - b)
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return {
        "max_abs_diff": float(d.max()),
        "mean_abs_diff": float(d.mean()),
        "p99_abs_diff": float(np.quantile(d, 0.99)),
        "cos_min": float(cos.min()),
        "cos_mean": float(cos.mean()),
        "n_cells_maxdiff_above_1e-4": int((d.max(1) > 1e-4).sum()),
        "n_cells_maxdiff_above_1e-3": int((d.max(1) > 1e-3).sum()),
    }


def tie_at_cut(h5ad, lmax=4096):
    a = ad.read_h5ad(h5ad)
    X = a.X.tocsr()
    tie = np.zeros(a.n_obs, dtype=bool)
    for i in range(a.n_obs):
        v = np.sort(X.data[X.indptr[i] : X.indptr[i + 1]])[::-1]
        tie[i] = v.size > lmax and v[lmax - 1] == v[lmax]
    return tie


def read_raw(flavour, task):
    a = ad.read_h5ad(f"{RUNS}/{flavour}/{task}/embeddings.h5ad")
    emb = np.asarray(a.obsm["bloom_emb"], dtype=np.float32)
    assert list(a.obs_names) == [str(i) for i in range(len(emb))], "unexpected obs index"
    return emb


os.makedirs(f"{LEGACY}/raw", exist_ok=True)
paper = np.asarray(ad.read_h5ad(ORIG).obsm["bloom_emb"], dtype=np.float32)
assert paper.shape[0] == 15750
TIMING = json.load(open("eval_timing_per_task.json"))
SUMMARY = json.load(open("reference_embeddings_summary.json"))
report = {"date": now, "runs_dir": RUNS, "tasks": {}}
for task, ds in TASKS.items():
    fx = cells[ds]
    src = fx["source_row"].to_numpy()
    tie = tie_at_cut(fixtures[ds])
    # 1. legacy
    legacy = {}
    for stem in (task, f"{task}_tf32off"):
        for ext in ("npy", "json"):
            p, q = f"{OUT}/{stem}.{ext}", f"{LEGACY}/{stem}.{ext}"
            if os.path.exists(p) and not os.path.exists(q):
                shutil.move(p, q)
        legacy[stem] = np.load(f"{LEGACY}/{stem}.npy")
    if os.path.isdir(f"{OUT}/raw/{task}") and not os.path.isdir(f"{LEGACY}/raw/{task}"):
        shutil.move(f"{OUT}/raw/{task}", f"{LEGACY}/raw/{task}")
    # 2./3. new references + raw
    new = {task: read_raw("tf32", task), f"{task}_tf32off": read_raw("tf32off", task)}
    for stem, emb in new.items():
        assert emb.shape == (len(fx), 1024) and np.isfinite(emb).all(), (stem, emb.shape)
        np.save(f"{OUT}/{stem}.npy", emb)
        os.makedirs(f"{OUT}/raw/{stem}", exist_ok=True)
        flavour = "tf32off" if stem.endswith("_tf32off") else "tf32"
        shutil.copy(f"{RUNS}/{flavour}/{task}/embeddings.h5ad", f"{OUT}/raw/{stem}/embeddings.h5ad")
    # 4. comparisons
    cmp = {
        "tf32_vs_paper_run_all_cells": stats(new[task], paper[src]),
        "tf32_vs_paper_run_cells_without_tie_at_cut": stats(new[task][~tie], paper[src][~tie]),
        "tf32off_vs_paper_run_all_cells": stats(new[f"{task}_tf32off"], paper[src]),
        "tf32_vs_tf32off": stats(new[task], new[f"{task}_tf32off"]),
        "tf32_vs_legacy_tf32": stats(new[task], legacy[task]),
        "tf32_vs_legacy_tf32_cells_without_tie_at_cut": stats(new[task][~tie], legacy[task][~tie]),
        "tf32off_vs_legacy_tf32off": stats(new[f"{task}_tf32off"], legacy[f"{task}_tf32off"]),
        "tf32off_vs_legacy_tf32off_cells_without_tie_at_cut": stats(
            new[f"{task}_tf32off"][~tie], legacy[f"{task}_tf32off"][~tie]
        ),
        "n_cells": len(fx),
        "n_cells_with_tie_at_cut": int(tie.sum()),
    }
    for k, v in cmp.items():
        print(task, k, v, flush=True)
    # sidecars
    common = {
        "shape": [len(fx), 1024],
        "dtype": "float32",
        "row_order": f"identical to obs order of examples/{'fixtures/rna_100.h5ad' if ds == 'rna_100' else 'allen_multiome_300/rna.h5ad'} (cells.tsv; sorted by source_row)",
        "obs_names": fx["obs_name"].tolist(),
        "source_rows": fx["source_row"].tolist(),
        "regenerated": {
            "date": now,
            "reason": "Phase 0 eval inputs had unsorted CSR rows, so StrategyTopExpressedGenes broke ties at the 4096 cut by storage order (#26); "
            "the TF32-off flavour is now also produced with the fused CUDA TransformerEncoder fast path disabled so CPU fp32 can meet G1 (#27)",
            "issues": ["#26", "#27"],
            "eval_input_rna": f"{SORTED}/{ds}/rna.h5ad",
            "eval_input_csr_rows_column_sorted": True,
            "eval_input_sha256": sorted_info[ds]["sha256"],
            "eval_input_rows_unsorted_in_phase0_input": sorted_info[ds]["n_rows_unsorted_before"],
            "legacy_files": f"examples/reference_embeddings/legacy_unsorted/{task}*.{{npy,json}} (staging tree only, not released)",
            "preset": "tools/hydra/phase1_reference_embeddings_sorted.yaml",
        },
        "producer": {
            "repo": "alleninstitute/bloom",
            "commit": export_commit,
            "script": "src/bloom/evaluate.py (unchanged) via tools/run_evaluate_checked.py",
            "env": os.path.basename(sys.prefix),
            "bloom_arch_worktree": f"{WORKTREE} (commit {arch_commit})",
            "PYTHONPATH": f"{WORKTREE}/src",
            "host": "maintainer GPU host (docs/maintainers.md)",
            "gpu": "NVIDIA L4 (23 GB)",
            "checkpoint": f"{CKPT} (hyacinth, step 530000)",
            "date": now,
        },
        "strategy": {
            "_target_": "bloom.utils.strategy.StrategyTopExpressedGenes",
            "idx_file": None,
            "Lmax": 4096,
            "Lq": 128,
            "binarize_input_genes": True,
            "peak_modality": "atac",
        },
        "aggregation": "mean of per-draw h_cell over draws (EvalWriter online mean); 1 draw for top4096",
    }
    for stem, flavour in ((task, "tf32"), (f"{task}_tf32off", "tf32off")):
        side = {"file": f"{stem}.npy", **common}
        side["producer"] = {
            **common["producer"],
            "seq_head_scale_checked": log_lines(flavour, "[scale-check]")[:4],
        }
        side["evaluate"] = {
            "precision": "32-true",
            "float32_matmul_precision": "medium (set by evaluate.py; TF32 matmuls enabled)"
            if flavour == "tf32"
            else "highest (TF32 off: BLOOM_MATMUL_PRECISION=highest; cuda.matmul.allow_tf32=False, cudnn.allow_tf32=False)",
            "mha_fastpath": "enabled (torch default; fused CUDA nn.TransformerEncoder path, as the paper's evaluations)"
            if flavour == "tf32"
            else "disabled (BLOOM_MHA_FASTPATH=0 -> torch.backends.mha.set_fastpath_enabled(False); unfused math == CPU)",
            "numerics_log_lines": log_lines(flavour, "[tf32-check]")
            + log_lines(flavour, "[fastpath-check]"),
            "batch_size": 8,
            "num_workers": 8,
            "seed": 42,
            "per_cell_seed": "seed + file_idx*100000 + row_idx (unused: top4096 is deterministic)",
            "peak_modality": "atac",
            "dataset_name_for_feature_map": "eval_balanced_50_allen_multiome",
            "reference_space": "mm10 (evo2_mean_ctx_2048_pca_768_norm)",
        }
        side["variant"] = (
            "paper-faithful numerics (TF32 on, fused fast path on) on column-sorted inputs; reference for G2b (GPU default, < 5e-4) and G7"
            if flavour == "tf32"
            else "strict reference for CPU fp32 (G1) and GPU TF32-off (G2a) at max|d| < 1e-4"
        )
        side["timing"] = timing.get(flavour)
        side["parity_vs_paper_run"] = {
            "orig_dir": "eval_balanced_50_allen_multiome_nz_top4096",
            "note": "the paper's run consumed the unsorted eval_balanced_50 rows, so its token sets differ on cells with an expression tie at the 4096 cut; compare on the cells without a tie",
            "all_cells": cmp[f"{flavour}_vs_paper_run_all_cells"],
            "cells_without_tie_at_cut": stats(new[stem][~tie], paper[src][~tie]),
            "n_cells_with_tie_at_cut": int(tie.sum()),
        }
        side["diff_vs_legacy_unsorted_reference"] = cmp[f"{flavour}_vs_legacy_{flavour}"]
        side["diff_tf32_vs_tf32off"] = cmp["tf32_vs_tf32off"]
        json.dump(side, open(f"{OUT}/{stem}.json", "w"), indent=2)
        TIMING[stem] = {
            **(timing.get(flavour) or {}),
            "note": f"wall of the {flavour} evaluate.py run (both top4096 tasks, incl. model load)",
        }
        SUMMARY["tasks"][stem] = {
            "task": stem,
            "n_cells": len(fx),
            "dim": 1024,
            "obs_names_match_fixture": False,
            "nan_rows": 0,
            "row_norm_mean": float(np.linalg.norm(new[stem], axis=1).mean()),
            "npy_bytes": os.path.getsize(f"{OUT}/{stem}.npy"),
            "sha256": sha256(f"{OUT}/{stem}.npy"),
            "regenerated": now,
        }
        SUMMARY["parity"][stem] = side["parity_vs_paper_run"]
    report["tasks"][task] = cmp
TIMING["_phase1_regeneration"] = {**timing, "date": now}
json.dump(TIMING, open("eval_timing_per_task.json", "w"), indent=2)
json.dump(SUMMARY, open("reference_embeddings_summary.json", "w"), indent=2)
json.dump(report, open("top4096_regeneration_report.json", "w"), indent=2)
shutil.copy(f"{SORTED}/SORTED_INPUTS.json", "sorted_inputs.json")
print("DONE", flush=True)
