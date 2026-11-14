"""Phase 0 step 3: cut the 100-cell RNA test fixture, the 300-cell multiome example, genes_mini tables,
and hyacinth-v1-format evaluate.py inputs for both."""

import json
import os
import time
from collections import OrderedDict

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from _paths import env_str
from anndata.io import read_elem
from safetensors.torch import load_file, save_file

_DATA = env_str("BLOOM_TRAINING_DATA").rstrip("/")
SRC = f"{_DATA}/datasets/hyacinth-v1/eval_balanced_50/allen_multiome/"
FMAP = f"{_DATA}/feature-maps/eval_balanced_50_allen_multiome/"
ASSETS = env_str("BLOOM_STAGING")
REF = f"{ASSETS}/references/mm10-evo2-v1"
FIX = f"{ASSETS}/examples/fixtures"
EX = f"{ASSETS}/examples/allen_multiome_300"
EVALIN = env_str("BLOOM_EVAL_INPUTS")
SEED = 0
for d in (FIX, EX, f"{EVALIN}/rna_100", f"{EVALIN}/allen_multiome_300", f"{EVALIN}/feature-maps"):
    os.makedirs(d, exist_ok=True)
link = f"{EVALIN}/feature-maps/eval_balanced_50_allen_multiome"
if not os.path.islink(link):
    os.symlink(FMAP, link)
T0 = time.time()
S = OrderedDict(seed=SEED, source=SRC)


def tick(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


meta = ad.read_h5ad(SRC + "metadata.h5ad")
obs = meta.obs
N = len(obs)
S["n_source_cells"] = N
sub = obs["AIT33_subclass"].astype(str)
cls = obs["AIT33_class"].astype(str)
sub2cls = (
    sub.groupby(sub).agg(lambda s: cls[s.index].iloc[0])
    if False
    else pd.Series({s: cls[sub == s].iloc[0] for s in sub.unique()})
)
S["n_subclasses"] = int(sub.nunique())
S["n_classes"] = int(cls.nunique())
tick(
    f"metadata {N} cells, {S['n_subclasses']} subclasses, {S['n_classes']} classes; obs_names real: {obs.index[:2].tolist()}"
)

rng = np.random.default_rng(SEED)


def pick_subclasses(k, rng):
    order = list(rng.permutation(sorted(sub.unique())))
    chosen, seen = [], set()
    for s in order:  # first pass: cover as many classes as possible
        if sub2cls[s] not in seen and len(chosen) < k:
            chosen.append(s)
            seen.add(sub2cls[s])
    for s in order:
        if len(chosen) >= k:
            break
        if s not in chosen:
            chosen.append(s)
    return chosen


def pick_cells(subclasses, per, exclude, rng):
    rows = []
    for s in subclasses:
        cand = np.setdiff1d(np.where(sub.to_numpy() == s)[0], exclude)
        rows.extend(sorted(rng.choice(cand, size=per, replace=False).tolist()))
    return np.array(sorted(rows), dtype=np.int64)


fix_sub = pick_subclasses(50, rng)
fix_rows = pick_cells(fix_sub, 2, np.array([], dtype=int), rng)
ex_sub = pick_subclasses(30, rng)
ex_rows = pick_cells(ex_sub, 10, fix_rows, rng)
assert len(fix_rows) == 100 and len(ex_rows) == 300 and not set(fix_rows) & set(ex_rows)
S["fixture_selection"] = (
    "50 subclasses (greedy class coverage over a seeded permutation) x 2 cells; obs ordered by source_row"
)
S["example_selection"] = (
    "30 subclasses (same procedure, independent draw) x 10 cells, cells disjoint from the fixture; obs ordered by source_row"
)
S["fixture_n_classes"] = int(cls.iloc[fix_rows].nunique())
S["example_n_classes"] = int(cls.iloc[ex_rows].nunique())
S["subclasses_shared_between_fixture_and_example"] = len(set(fix_sub) & set(ex_sub))
tick(
    f"fixture: 100 cells / {len(fix_sub)} subclasses / {S['fixture_n_classes']} classes; example: 300 cells / {len(ex_sub)} subclasses / {S['example_n_classes']} classes; shared subclasses {S['subclasses_shared_between_fixture_and_example']}"
)


def read_rows(path, rows):
    """CSR row subset via h5py, preserving within-row index order. Returns (csr, var_names, obs_names)."""
    with h5py.File(path) as f:
        X = f["X"]
        shape = tuple(int(x) for x in X.attrs["shape"])
        indptr = X["indptr"][:]
        data, idx, ptr = [], [], [0]
        for r in rows:
            a, b = int(indptr[r]), int(indptr[r + 1])
            data.append(X["data"][a:b])
            idx.append(X["indices"][a:b])
            ptr.append(ptr[-1] + (b - a))
        data = np.concatenate(data)
        idx = np.concatenate(idx).astype(np.int32)
        ptr = np.asarray(ptr, dtype=np.int64)
        var_names = f["var"][f["var"].attrs["_index"]].asstr()[:]
        obs_names = f["obs"][f["obs"].attrs["_index"]].asstr()[:]
    return sp.csr_matrix((data, idx, ptr), shape=(len(rows), shape[1])), var_names, obs_names


genes = pd.read_parquet(f"{REF}/genes.parquet")
peaks_names = pd.read_parquet(f"{REF}/peaks.parquet", columns=["name"])["name"].to_numpy()
d2e = pd.read_csv(FMAP + "rna_dataset_to_embedding.csv")
e2d = pd.read_csv(FMAP + "rna_embedding_to_dataset.csv")
S["feature_map_columns"] = {
    "rna_dataset_to_embedding": list(d2e.columns),
    "rna_embedding_to_dataset": list(e2d.columns),
    "n_rows": [len(d2e), len(e2d)],
}
tick(f"feature maps: {S['feature_map_columns']}")
recipe = None
with h5py.File(SRC + "rna_count.h5ad") as f:
    recipe = read_elem(f["uns"]["logcpm_to_count"])


def build(name, rows, out_dir, with_atac):
    info = OrderedDict(n_cells=len(rows))
    X, gsym, rna_obs_names = read_rows(SRC + "rna.h5ad", rows)
    C, gsym2, _ = read_rows(SRC + "rna_count.h5ad", rows)
    assert (gsym == gsym2).all()
    # ---- hyacinth-v1-format eval inputs (full column space, original within-row order, source obs names) ----
    ev = f"{EVALIN}/{name}"
    ad.AnnData(
        X=X.copy(), obs=pd.DataFrame(index=rna_obs_names[rows]), var=pd.DataFrame(index=gsym)
    ).write_h5ad(f"{ev}/rna.h5ad")
    meta[rows].copy().write_h5ad(f"{ev}/metadata.h5ad")
    A, pnames, atac_obs_names = read_rows(SRC + "atac.h5ad", rows)
    assert (
        A.dtype == np.uint32 and len(pnames) == len(peaks_names) and (pnames == peaks_names).all()
    ), "atac var != reference peaks"
    ad.AnnData(
        X=A.copy(), obs=pd.DataFrame(index=atac_obs_names[rows]), var=pd.DataFrame(index=pnames)
    ).write_h5ad(f"{ev}/atac.h5ad")
    info["eval_inputs"] = {
        k: os.path.getsize(f"{ev}/{k}.h5ad") for k in ("rna", "metadata", "atac")
    }
    info["source_rna_obs_names_equal_metadata"] = bool(
        (rna_obs_names == obs.index.to_numpy().astype(str)).all()
    )
    tick(f"{name}: eval inputs written {info['eval_inputs']}")
    # ---- public fixture / example ----
    Xs = X.copy()
    Xs.sort_indices()
    Cs = C.copy().astype(np.int32)
    Cs.sort_indices()
    o = pd.DataFrame(
        {
            "AIT33_class": cls.iloc[rows].to_numpy(),
            "AIT33_subclass": sub.iloc[rows].to_numpy(),
            "source_row": rows.astype(np.int64),
        },
        index=obs.index[rows].astype(str),
    )
    o["AIT33_class"] = o["AIT33_class"].astype("category")
    o["AIT33_subclass"] = o["AIT33_subclass"].astype("category")
    umi = obs["rna_umi_counts"].iloc[rows].to_numpy().astype(np.int64)
    rowsum = np.asarray(Cs.sum(1)).ravel()
    info["counts_rowsum_equals_rna_umi_counts"] = int((rowsum == umi).sum())
    info["detection_pattern_identical_X_vs_counts"] = bool(((Xs > 0) != (Cs > 0)).nnz == 0)
    info["expm1_rowsum_median"] = float(np.median(np.asarray(np.expm1(Xs).sum(1)).ravel()))
    info["n_detected_genes_per_cell"] = {
        "min": int(np.diff(Xs.indptr).min()),
        "median": float(np.median(np.diff(Xs.indptr))),
        "max": int(np.diff(Xs.indptr).max()),
    }
    uns = {
        "bloom_phase0": {
            "source_dataset": "hyacinth-v1/eval_balanced_50/allen_multiome",
            "source_files": ["rna.h5ad", "rna_count.h5ad", "metadata.h5ad"]
            + (["atac.h5ad"] if with_atac else []),
            "seed": SEED,
            "selection": S["fixture_selection"] if name == "rna_100" else S["example_selection"],
            "X": "log1p(CPM) float32 CSR, copied from source rna.h5ad (indices sorted)",
            "layers_counts": "int32 CSR copied from source rna_count.h5ad",
            "logcpm_to_count": dict(recipe),
            "labels": "AIT33 (Allen whole-mouse-brain taxonomy) from metadata.h5ad",
            "reference_id": "mm10-evo2-v1",
        }
    }
    rna = ad.AnnData(
        X=Xs,
        obs=o,
        var=pd.DataFrame(index=pd.Index(gsym, name="gene_symbol")),
        layers={"counts": Cs},
        uns=uns,
    )
    rna.var_names.name = None
    rna_path = f"{out_dir}/{'rna_100' if name == 'rna_100' else 'rna'}.h5ad"
    rna.write_h5ad(rna_path, compression="gzip")
    info["rna_h5ad_bytes"] = os.path.getsize(rna_path)
    pd.DataFrame({
        "obs_name": o.index,
        "source_row": o["source_row"].to_numpy(),
        "AIT33_subclass": o["AIT33_subclass"].astype(str).to_numpy(),
        "AIT33_class": o["AIT33_class"].astype(str).to_numpy(),
    }).to_csv(f"{out_dir}/cells.tsv", sep="\t", index=False)
    info["subclass_counts"] = o["AIT33_subclass"].astype(str).value_counts().to_dict()
    tick(
        f"{name}: {rna_path} ({info['rna_h5ad_bytes'] / 1e6:.2f} MB); counts rowsum==umi for {info['counts_rowsum_equals_rna_umi_counts']}/{len(rows)}; pattern identical {info['detection_pattern_identical_X_vs_counts']}"
    )
    # ---- gene mapping (detected genes -> emb_idx), two routes ----
    det_cols = np.unique(Xs.indices)
    sym2emb = pd.Series(genes["emb_idx"].to_numpy(), index=genes["gene_name"].to_numpy())
    via_symbol = sym2emb.reindex(gsym[det_cols]).to_numpy()
    fm_col = next(c for c in d2e.columns if "emb" in c)
    fm = (
        pd.Series(d2e[fm_col].to_numpy(), index=d2e["feature_name"].to_numpy())
        if "feature_name" in d2e
        else None
    )
    via_fmap = fm.reindex(gsym[det_cols]).to_numpy() if fm is not None else None
    info["n_detected_genes_union"] = len(det_cols)
    info["n_mapped_via_symbol"] = int(np.isfinite(via_symbol).sum())
    if via_fmap is not None:
        both = np.isfinite(via_symbol) & (via_fmap >= 0)
        info["n_mapped_via_feature_map"] = int((via_fmap >= 0).sum())
        info["n_symbol_vs_fmap_disagree"] = int((via_symbol[both] != via_fmap[both]).sum())
    unm = gsym[det_cols][~np.isfinite(via_symbol)]
    info["unmapped_examples"] = unm[:10].tolist()
    # top4096 union (what G1 needs) for size reference
    top_union = set()
    for i in range(Xs.shape[0]):
        r = Xs.getrow(i)
        order = np.lexsort((r.indices, -r.data))
        top_union.update(r.indices[order[:4096]].tolist())
    info["n_top4096_union_genes"] = len(top_union)
    emb_rows = np.sort(via_symbol[np.isfinite(via_symbol)].astype(np.int64))
    if name == "rna_100":
        gt = load_file(f"{REF}/genes.safetensors")
        idx = torch.from_numpy(emb_rows)
        save_file(
            {
                "cseq": gt["cseq"][idx].contiguous(),
                "chr": gt["chr"][idx].contiguous(),
                "bp": gt["bp"][idx].contiguous(),
                "emb_idx": idx.to(torch.int32).contiguous(),
            },
            f"{out_dir}/genes_mini.safetensors",
            metadata={
                "reference_id": "mm10-evo2-v1",
                "subset_of": "genes.safetensors",
                "n_rows": str(len(emb_rows)),
                "note": "rows = genes detected in rna_100.h5ad that map by symbol; emb_idx = row in the full gene table",
            },
        )
        genes.iloc[emb_rows].reset_index(drop=True).to_parquet(
            f"{out_dir}/genes_mini.parquet", index=False
        )
        info["genes_mini_bytes"] = {
            "safetensors": os.path.getsize(f"{out_dir}/genes_mini.safetensors"),
            "parquet": os.path.getsize(f"{out_dir}/genes_mini.parquet"),
        }
    tick(
        f"{name}: detected genes {len(det_cols)}, mapped by symbol {info['n_mapped_via_symbol']}, via fmap {info.get('n_mapped_via_feature_map')}, disagree {info.get('n_symbol_vs_fmap_disagree')}, top4096 union {len(top_union)}"
    )
    # ---- atac for the example ----
    if with_atac:
        As = A.copy()
        As.sort_indices()
        keep = np.unique(As.indices)
        Ak = As[:, keep]
        Ak.sort_indices()
        var = pd.DataFrame({"emb_idx": keep.astype(np.int32)}, index=pnames[keep])
        atac = ad.AnnData(
            X=Ak,
            obs=o.copy(),
            var=var,
            uns={
                "bloom_phase0": {
                    "source_dataset": "hyacinth-v1/eval_balanced_50/allen_multiome",
                    "source_files": ["atac.h5ad", "metadata.h5ad"],
                    "seed": SEED,
                    "X": "raw fragment counts uint32 CSR (NOT binarized), copied from source atac.h5ad; columns restricted to reference peaks with >=1 nonzero across these cells",
                    "var_emb_idx": "row in references/mm10-evo2-v1/peaks.* (== source atac column index; the atac feature map is the identity)",
                    "reference_id": "mm10-evo2-v1",
                }
            },
        )
        atac.write_h5ad(f"{out_dir}/atac.h5ad", compression="gzip")
        info["atac"] = {
            "n_peaks_kept": len(keep),
            "n_peaks_total": int(A.shape[1]),
            "nnz": int(Ak.nnz),
            "dtype": str(Ak.dtype),
            "bytes": os.path.getsize(f"{out_dir}/atac.h5ad"),
            "n_detected_peaks_per_cell": {
                "min": int(np.diff(Ak.indptr).min()),
                "median": float(np.median(np.diff(Ak.indptr))),
                "max": int(np.diff(Ak.indptr).max()),
            },
            "max_count": int(Ak.data.max()),
        }
        tick(f"{name}: atac.h5ad {info['atac']}")
    return info


S["rna_100"] = build("rna_100", fix_rows, FIX, with_atac=False)
S["allen_multiome_300"] = build("allen_multiome_300", ex_rows, EX, with_atac=True)
S["elapsed_s"] = time.time() - T0
json.dump(S, open("fixtures_summary.json", "w"), indent=2, default=str)
import shutil

shutil.copy(f"{FIX}/cells.tsv", "rna_100_cells.tsv")
shutil.copy(f"{EX}/cells.tsv", "allen_multiome_300_cells.tsv")
tick("DONE")
