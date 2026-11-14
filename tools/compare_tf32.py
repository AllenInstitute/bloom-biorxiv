import json
import sys

import anndata as ad
import numpy as np
import pandas as pd
from _paths import env_str

ASSETS = env_str("BLOOM_STAGING")
fx = pd.read_csv(f"{ASSETS}/examples/allen_multiome_300/cells.tsv", sep="\t")
src = fx["source_row"].to_numpy()
orig = np.asarray(
    ad.read_h5ad(
        f"{env_str('BLOOM_PAPER_RESULTS')}/eval_balanced_50_allen_multiome_nz_top4096/embeddings.h5ad"
    ).obsm["bloom_emb"],
    dtype=np.float32,
)[src]
runs = {
    "L4_tf32_b8 (release)": np.load(
        f"{ASSETS}/examples/reference_embeddings/allen_multiome_300_top4096.npy"
    )
}
for k in ["tf32off_b8", "tf32on_b32", "tf32on_b1"]:
    runs[f"L4_{k}"] = np.asarray(
        ad.read_h5ad(
            f"{ASSETS}/examples/reference_embeddings/raw_check/{k}/allen_multiome_300_top4096/embeddings.h5ad"
        ).obsm["bloom_emb"],
        dtype=np.float32,
    )
runs["H200_tf32_b64 (paper)"] = orig
names = list(runs)
out = {}


def stats(a, b):
    d = np.abs(a - b)
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return {
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "cos_min": float(cos.min()),
        "bitwise_equal": bool(d.max() == 0),
    }


for i, a in enumerate(names):
    for b in names[i + 1 :]:
        out[f"{a} vs {b}"] = stats(runs[a], runs[b])
        print(f"{a:28s} vs {b:28s}", out[f"{a} vs {b}"], flush=True)
json.dump(out, open("tf32_check.json", "w"), indent=2)
