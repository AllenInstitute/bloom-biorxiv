"""Column-sort the Phase 0 eval inputs (rna.h5ad CSR rows) for the top4096 reference regeneration (#26).

Writes evalinputs_sorted/<ds>/rna.h5ad with sort_indices() applied; metadata.h5ad, atac.h5ad and the
feature-maps dir are symlinked to the Phase 0 originals so the evaluate.py wiring is identical.
"""

import hashlib
import json
import os
import sys

import anndata as ad
import h5py
import numpy as np
from _paths import env_str

SRC = env_str("BLOOM_EVAL_INPUTS")
DST = env_str("BLOOM_EVAL_INPUTS_SORTED")
out = {}
for ds in ("rna_100", "allen_multiome_300"):
    os.makedirs(f"{DST}/{ds}", exist_ok=True)
    a = ad.read_h5ad(f"{SRC}/{ds}/rna.h5ad")
    X = a.X.tocsr()
    n_unsorted = 0
    for i in range(X.shape[0]):
        idx = X.indices[X.indptr[i] : X.indptr[i + 1]]
        if idx.size > 1 and np.any(np.diff(idx) < 0):
            n_unsorted += 1
    Xs = X.copy()
    Xs.sort_indices()
    assert Xs.has_sorted_indices
    # same (col, value) multiset per row
    for i in range(X.shape[0]):
        s0 = sorted(
            zip(
                X.indices[X.indptr[i] : X.indptr[i + 1]].tolist(),
                X.data[X.indptr[i] : X.indptr[i + 1]].tolist(),
            )
        )
        s1 = list(
            zip(
                Xs.indices[Xs.indptr[i] : Xs.indptr[i + 1]].tolist(),
                Xs.data[Xs.indptr[i] : Xs.indptr[i + 1]].tolist(),
            )
        )
        assert s0 == s1, i
    b = ad.AnnData(X=Xs, obs=a.obs.copy(), var=a.var.copy(), uns=dict(a.uns))
    b.uns["bloom_phase1"] = {
        "note": "column-sorted copy (sort_indices) of evalinputs/%s/rna.h5ad; issue #26" % ds
    }
    p = f"{DST}/{ds}/rna.h5ad"
    b.write_h5ad(p)
    with h5py.File(p) as f:
        ind = f["X/indices"][:]
        ptr = f["X/indptr"][:]
        ok = all(np.all(np.diff(ind[ptr[i] : ptr[i + 1]]) > 0) for i in range(len(ptr) - 1))
    for name in ("metadata.h5ad", "atac.h5ad"):
        lnk = f"{DST}/{ds}/{name}"
        if not os.path.lexists(lnk):
            os.symlink(f"{SRC}/{ds}/{name}", lnk)
    out[ds] = {
        "n_cells": int(X.shape[0]),
        "n_rows_unsorted_before": n_unsorted,
        "sorted_on_disk": bool(ok),
        "obs_names_equal": bool((a.obs_names == b.obs_names).all()),
        "var_names_equal": bool((a.var_names == b.var_names).all()),
        "nnz": int(X.nnz),
        "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest(),
    }
    print(ds, out[ds], flush=True)
if not os.path.lexists(f"{DST}/feature-maps"):
    os.symlink(f"{SRC}/feature-maps", f"{DST}/feature-maps")
json.dump(out, open(f"{DST}/SORTED_INPUTS.json", "w"), indent=2)
print("DONE")
