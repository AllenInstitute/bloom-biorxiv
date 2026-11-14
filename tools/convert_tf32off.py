import datetime
import json
import os
import subprocess

import anndata as ad
import numpy as np
import pandas as pd
from _paths import env_str

ASSETS = env_str("BLOOM_STAGING")
RE = f"{ASSETS}/examples/reference_embeddings"
RC = f"{RE}/raw_check"
cells = {
    "rna_100": pd.read_csv(f"{ASSETS}/examples/fixtures/cells.tsv", sep="\t"),
    "allen_multiome_300": pd.read_csv(f"{ASSETS}/examples/allen_multiome_300/cells.tsv", sep="\t"),
}
export_commit = subprocess.check_output(
    ["git", "-C", os.path.expanduser("~/code/bloom"), "rev-parse", "HEAD"], text=True
).strip()
out = {}
for ds in ["rna_100", "allen_multiome_300"]:
    task = f"{ds}_top4096"
    a = ad.read_h5ad(f"{RC}/tf32off_b8/{task}/embeddings.h5ad")
    emb = np.asarray(a.obsm["bloom_emb"], dtype=np.float32)
    assert emb.shape[0] == len(cells[ds]) and list(a.obs_names) == [str(i) for i in range(len(emb))]
    tf32 = np.load(f"{RE}/{task}.npy")
    d = np.abs(emb - tf32)
    stem = f"{task}_tf32off"
    np.save(f"{RE}/{stem}.npy", emb)
    base = json.load(open(f"{RE}/{task}.json"))
    side = dict(base)
    side.update({
        "file": f"{stem}.npy",
        "variant": "TF32 OFF: torch.set_float32_matmul_precision('highest') forced via run_evaluate_checked.py (evaluate.py's 'medium' call ignored); everything else identical to the TF32 run",
        "purpose": "strict reference for CPU fp32 regression (TEST_PLAN G1); the TF32 run is the paper-faithful reference (use 1e-3 tolerance)",
        "diff_vs_tf32_run": {"max_abs": float(d.max()), "mean_abs": float(d.mean())},
        "timing": None,
        "parity_vs_paper_run": None,
    })
    side["evaluate"] = dict(base["evaluate"])
    side["evaluate"]["float32_matmul_precision"] = "highest (TF32 off)"
    json.dump(side, open(f"{RE}/{stem}.json", "w"), indent=2)
    out[stem] = {
        "shape": list(emb.shape),
        "max_abs_vs_tf32": float(d.max()),
        "mean_abs_vs_tf32": float(d.mean()),
        "nan_rows": int(np.isnan(emb).any(1).sum()),
    }
    print(stem, out[stem], flush=True)
json.dump(out, open("tf32off_refs.json", "w"), indent=2)
