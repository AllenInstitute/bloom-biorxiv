"""Cut the 20-cell multiome fixture ``examples/multiome_20`` and its ``evaluate.py`` inputs.

Source: the staged 300-cell example (``examples/allen_multiome_300/{rna,atac}.h5ad`` + ``cells.tsv``;
their ``obs["source_row"]`` points into ``hyacinth-v1/eval_balanced_50/allen_multiome``) and the
column-sorted eval inputs of the same 300 cells (``evalinputs_sorted/allen_multiome_300/rna.h5ad``,
``evalinputs/allen_multiome_300/{atac,metadata}.h5ad``).

Selection (seed 0): the 10 AIT33 subclasses with the largest median number of nonzero ATAC peaks per
cell, 2 cells drawn at random (``default_rng(0)``) from each -> 20 cells, ordered by ``source_row``.

Outputs
  <fixture>/rna.h5ad    X log1p(CPM) float32 CSR (column-sorted) + layers["counts"] int32; obs AIT33_class,
                        AIT33_subclass, source_row; uns["bloom_fixture"] (+ the logcpm_to_count recipe)
  <fixture>/atac.h5ad   raw uint32 counts; var_names = reference peak strings restricted to the peaks with
                        >= 1 nonzero in these 20 cells; var["emb_idx"] = row in references/mm10-evo2-v1/peaks.*
  <fixture>/cells.tsv   obs_name, source_row, AIT33_subclass, AIT33_class
  <evalin>/multiome_20/{rna,atac,metadata}.h5ad   hyacinth-v1-format inputs with the FULL column spaces,
                        CSR rows column-sorted (rna AND atac), real obs names; feature maps are the
                        ``evalinputs_sorted/feature-maps`` symlink (eval_balanced_50_allen_multiome)
  --selection-out (default ./multiome_20_selection.json)  what was picked and why (also printed)

Usage: python tools/make_multiome_20.py [--staging ROOT] [--evalin ROOT] [--example-dir DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

SEED = 0
N_SUBCLASSES = 10
PER_SUBCLASS = 2
MIN_ATAC_NNZ = 8192  # every ATAC view of the fixture is then a full-length (lmax 8192) view


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def obj_index(values) -> pd.Index:
    """Object-dtype string index (pandas 3 defaults to ``str`` dtype, which anndata 0.13 writes as a
    ``nullable-string-array`` group; the plain ``string-array`` encoding is what every anndata reads)."""
    return pd.Index(np.array([str(v) for v in values], dtype=object))


def compress_string_index(path: str, group: str, expected: ad.AnnData) -> None:
    """Re-store ``<group>/_index`` as a gzip-compressed fixed-length string dataset (anndata writes variable-length
    strings, which HDF5 cannot compress: 461k peak names cost ~17 MB). Verified by reading the file back."""
    with h5py.File(path, "r+") as f:
        g = f[group]
        key = g.attrs["_index"]
        obj = g[key]
        if isinstance(obj, h5py.Group):  # nullable-string-array: values + mask
            assert not obj["mask"][:].any()
            names = obj["values"].asstr()[:]
        else:
            names = obj.asstr()[:]
        del g[key]
        width = max(len(n.encode()) for n in names)
        new = g.create_dataset(
            key,
            data=np.array(names, dtype=f"S{width}"),
            compression="gzip",
            compression_opts=6,
            chunks=True,
        )
        new.attrs["encoding-type"] = "string-array"
        new.attrs["encoding-version"] = "0.2.0"
    # HDF5 does not reclaim the space of the deleted dataset: repack by copying every object into a new file
    tmp = path + ".repack"
    with h5py.File(path) as fi, h5py.File(tmp, "w") as fo:
        for k, v in fi.attrs.items():
            fo.attrs[k] = v
        for k in fi:
            fi.copy(k, fo, name=k)
    os.replace(tmp, path)
    back = ad.read_h5ad(path)
    assert back.var_names.tolist() == expected.var_names.tolist()
    assert back.obs_names.tolist() == expected.obs_names.tolist()
    assert (back.var["emb_idx"].to_numpy() == expected.var["emb_idx"].to_numpy()).all()
    assert (
        sp.csr_matrix(back.X) != sp.csr_matrix(expected.X)
    ).nnz == 0 and back.X.dtype == expected.X.dtype


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--staging", default=os.environ.get("BLOOM_STAGING"), help="default $BLOOM_STAGING"
    )
    ap.add_argument(
        "--example-dir",
        default=None,
        help="300-cell example dir (default <staging>/examples/allen_multiome_300)",
    )
    ap.add_argument(
        "--evalin", default=os.environ.get("BLOOM_EVAL_INPUTS"), help="default $BLOOM_EVAL_INPUTS"
    )
    ap.add_argument(
        "--evalin-sorted",
        default=os.environ.get("BLOOM_EVAL_INPUTS_SORTED"),
        help="default $BLOOM_EVAL_INPUTS_SORTED",
    )
    ap.add_argument(
        "--out", default=None, help="fixture dir (default <staging>/examples/multiome_20)"
    )
    ap.add_argument(
        "--selection-out",
        default="multiome_20_selection.json",
        help="where to write the selection report (outside the staging tree)",
    )
    args = ap.parse_args()
    t0 = time.time()
    ex = args.example_dir or f"{args.staging}/examples/allen_multiome_300"
    out = args.out or f"{args.staging}/examples/multiome_20"
    ev_out = f"{args.evalin_sorted}/multiome_20"
    os.makedirs(out, exist_ok=True)
    os.makedirs(ev_out, exist_ok=True)

    rna300 = ad.read_h5ad(f"{ex}/rna.h5ad")
    atac300 = ad.read_h5ad(f"{ex}/atac.h5ad")
    assert (rna300.obs_names == atac300.obs_names).all()
    cells300 = pd.read_csv(f"{ex}/cells.tsv", sep="\t")
    assert cells300["obs_name"].tolist() == rna300.obs_names.tolist()
    print(
        f"[{time.time() - t0:6.1f}s] loaded 300-cell example: rna {rna300.shape}, atac {atac300.shape}",
        flush=True,
    )

    # ---- selection -------------------------------------------------------------------------------
    A300 = sp.csr_matrix(atac300.X)
    nnz = np.diff(A300.indptr)
    sub = rna300.obs["AIT33_subclass"].astype(str).to_numpy()
    df = pd.DataFrame({"subclass": sub, "atac_nnz": nnz, "pos": np.arange(len(sub))})
    med = df.groupby("subclass")["atac_nnz"].median().sort_values(ascending=False)
    eligible = df[df["atac_nnz"] >= MIN_ATAC_NNZ]
    n_elig = eligible.groupby("subclass").size()
    chosen_sub = [sc for sc in med.index if n_elig.get(sc, 0) >= PER_SUBCLASS][:N_SUBCLASSES]
    assert len(chosen_sub) == N_SUBCLASSES
    rng = np.random.default_rng(SEED)
    pos = []
    for s in chosen_sub:
        cand = eligible.loc[eligible["subclass"] == s, "pos"].to_numpy()
        pos.extend(sorted(rng.choice(cand, size=PER_SUBCLASS, replace=False).tolist()))
    pos = np.array(
        sorted(pos), dtype=np.int64
    )  # obs order = source_row order (the 300 are sorted by it)
    src_rows = rna300.obs["source_row"].to_numpy().astype(np.int64)[pos]
    assert np.all(np.diff(src_rows) > 0)
    obs_names = rna300.obs_names[pos].to_numpy()
    sel = {
        "seed": SEED,
        "rule": f"{N_SUBCLASSES} subclasses with the largest median ATAC nonzeros per cell in allen_multiome_300 "
        f"(among subclasses with >= {PER_SUBCLASS} cells of >= {MIN_ATAC_NNZ} nonzero peaks), {PER_SUBCLASS} cells each "
        f"drawn from those cells with default_rng({SEED}).choice(replace=False); obs ordered by source_row",
        "min_atac_nnz": MIN_ATAC_NNZ,
        "subclass_median_atac_nnz_all": {k: float(v) for k, v in med.items()},
        "subclasses": chosen_sub,
        "cells": [
            {
                "obs_name": str(o),
                "source_row": int(r),
                "pos_in_300": int(p),
                "AIT33_subclass": str(sub[p]),
                "AIT33_class": str(rna300.obs["AIT33_class"].astype(str).iloc[p]),
                "atac_nnz": int(nnz[p]),
                "rna_detected": int(np.diff(sp.csr_matrix(rna300.X).indptr)[p]),
            }
            for o, r, p in zip(obs_names, src_rows, pos, strict=True)
        ],
    }
    print(f"[{time.time() - t0:6.1f}s] picked {len(pos)} cells from {chosen_sub}", flush=True)

    # ---- public fixture: rna -------------------------------------------------------------------------
    rna = rna300[pos].copy()
    X = sp.csr_matrix(rna.X)
    X.sort_indices()
    C = sp.csr_matrix(rna.layers["counts"]).astype(np.int32)
    C.sort_indices()
    obs = pd.DataFrame(
        {
            "AIT33_class": pd.Categorical(rna.obs["AIT33_class"].astype(str)),
            "AIT33_subclass": pd.Categorical(rna.obs["AIT33_subclass"].astype(str)),
            "source_row": src_rows,
        },
        index=obj_index(obs_names),
    )
    recipe = dict(rna300.uns.get("bloom_phase0", {}).get("logcpm_to_count", {}))
    uns = {
        "bloom_fixture": {
            "name": "multiome_20",
            "source_dataset": "hyacinth-v1/eval_balanced_50/allen_multiome",
            "via": "examples/allen_multiome_300 (row subset; source_row recorded)",
            "seed": SEED,
            "selection": sel["rule"],
            "X": "log1p(CPM) float32 CSR, column-sorted",
            "layers_counts": "int32 CSR (reconstructed counts; uns['logcpm_to_count'] recipe)",
            "logcpm_to_count": recipe,
            "labels": "AIT33 (Allen whole-mouse-brain taxonomy)",
            "reference_id": "mm10-evo2-v1",
        }
    }
    rna_out = ad.AnnData(
        X=X,
        obs=obs,
        var=pd.DataFrame(index=obj_index(rna.var_names)),
        layers={"counts": C},
        uns=uns,
    )
    rna_out.var_names.name = None
    rna_out.write_h5ad(f"{out}/rna.h5ad", compression="gzip")

    # ---- public fixture: atac -------------------------------------------------------------------------
    A = sp.csr_matrix(atac300.X)[pos]
    A.sort_indices()
    keep = np.unique(A.indices)
    Ak = sp.csr_matrix(A[:, keep])
    Ak.sort_indices()
    assert Ak.dtype == np.uint32, Ak.dtype
    var = pd.DataFrame(
        {"emb_idx": atac300.var["emb_idx"].to_numpy().astype(np.int32)[keep]},
        index=obj_index(atac300.var_names.to_numpy()[keep]),
    )
    atac_out = ad.AnnData(
        X=Ak,
        obs=obs.copy(),
        var=var,
        uns={
            "bloom_fixture": {
                **uns["bloom_fixture"],
                "X": "raw fragment counts uint32 CSR (NOT binarized), column-sorted; columns = reference peaks with >= 1 nonzero across these 20 cells",
                "var_emb_idx": "row in references/mm10-evo2-v1/peaks.* (== source atac column index)",
            }
        },
    )
    atac_out.write_h5ad(f"{out}/atac.h5ad", compression="gzip")
    compress_string_index(f"{out}/atac.h5ad", "var", atac_out)
    pd.DataFrame({
        "obs_name": obs.index,
        "source_row": src_rows,
        "AIT33_subclass": obs["AIT33_subclass"].astype(str).to_numpy(),
        "AIT33_class": obs["AIT33_class"].astype(str).to_numpy(),
    }).to_csv(f"{out}/cells.tsv", sep="\t", index=False)
    print(
        f"[{time.time() - t0:6.1f}s] fixture written: rna {os.path.getsize(out + '/rna.h5ad'):,} B, atac {os.path.getsize(out + '/atac.h5ad'):,} B ({len(keep):,} peaks, nnz {Ak.nnz:,})",
        flush=True,
    )

    # ---- evaluate.py inputs (full column spaces, sorted rows) -----------------------------------------
    ev300 = f"{args.evalin_sorted}/allen_multiome_300"
    r300 = ad.read_h5ad(f"{ev300}/rna.h5ad")
    epos = np.array([r300.obs_names.get_loc(o) for o in obs_names])
    assert (r300.obs_names[epos].to_numpy() == obs_names).all()
    ev_rna = r300[epos].copy()
    Xe = sp.csr_matrix(ev_rna.X)
    Xe.sort_indices()
    ev_rna = ad.AnnData(
        X=Xe,
        obs=pd.DataFrame(index=obj_index(ev_rna.obs_names)),
        var=pd.DataFrame(index=obj_index(ev_rna.var_names)),
    )
    ev_rna.uns["bloom_fixture"] = {
        "note": "multiome_20 rows of evalinputs_sorted/allen_multiome_300/rna.h5ad (column-sorted CSR)"
    }
    ev_rna.write_h5ad(f"{ev_out}/rna.h5ad")
    # the detection pattern of the eval input equals the fixture's
    assert ((Xe > 0) != (X > 0)).nnz == 0
    a300 = ad.read_h5ad(f"{args.evalin}/allen_multiome_300/atac.h5ad", backed="r")
    assert (a300.obs_names[epos].to_numpy() == obs_names).all()
    sub = a300[epos].to_memory()
    Ae = sp.csr_matrix(sub.X)
    assert Ae.dtype == np.uint32, Ae.dtype
    Ae.sort_indices()
    pnames = sub.var_names.to_numpy().astype(str)
    assert (
        len(pnames) == 1_747_701
        and (pnames[atac300.var["emb_idx"].to_numpy()] == atac300.var_names.to_numpy()).all()
    )
    ev_atac = ad.AnnData(
        X=Ae,
        obs=pd.DataFrame(index=obj_index(obs_names)),
        var=pd.DataFrame(index=obj_index(pnames)),
    )
    ev_atac.write_h5ad(f"{ev_out}/atac.h5ad")
    # same detection pattern as the public fixture (columns restricted to `keep`)
    emb_keep = atac300.var["emb_idx"].to_numpy()[keep]
    assert ((Ae[:, emb_keep] > 0) != (Ak > 0)).nnz == 0
    meta = ad.read_h5ad(f"{args.evalin}/allen_multiome_300/metadata.h5ad")
    mpos = np.array([meta.obs_names.get_loc(o) for o in obs_names])
    meta[mpos].copy().write_h5ad(f"{ev_out}/metadata.h5ad")
    for p in (f"{ev_out}/rna.h5ad", f"{ev_out}/atac.h5ad"):
        with h5py.File(p) as f:
            ind, ptr = f["X/indices"][:], f["X/indptr"][:]
            assert all(
                np.all(np.diff(ind[ptr[i] : ptr[i + 1]]) > 0) for i in range(len(ptr) - 1)
            ), p
    sel["eval_inputs"] = {
        os.path.basename(p): {"bytes": os.path.getsize(p), "sha256": sha256(p)}
        for p in (f"{ev_out}/rna.h5ad", f"{ev_out}/atac.h5ad", f"{ev_out}/metadata.h5ad")
    }
    sel["fixture"] = {
        os.path.basename(p): {"bytes": os.path.getsize(p), "sha256": sha256(p)}
        for p in (f"{out}/rna.h5ad", f"{out}/atac.h5ad", f"{out}/cells.tsv")
    }
    sel["atac"] = {
        "n_peaks_kept": len(keep),
        "nnz": int(Ak.nnz),
        "n_detected_peaks_per_cell": {
            "min": int(np.diff(Ak.indptr).min()),
            "median": float(np.median(np.diff(Ak.indptr))),
            "max": int(np.diff(Ak.indptr).max()),
        },
        "max_count": int(Ak.data.max()),
    }
    sel["rna"] = {
        "n_genes": int(X.shape[1]),
        "n_detected_genes_per_cell": {
            "min": int(np.diff(X.indptr).min()),
            "median": float(np.median(np.diff(X.indptr))),
            "max": int(np.diff(X.indptr).max()),
        },
    }
    sel["elapsed_s"] = round(time.time() - t0, 1)
    with open(args.selection_out, "w") as fh:
        json.dump(sel, fh, indent=2)
    print(
        json.dumps(
            {k: v for k, v in sel.items() if k not in ("subclass_median_atac_nnz_all",)}, indent=1
        ),
        flush=True,
    )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
