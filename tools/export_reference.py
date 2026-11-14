"""Phase 0 step 2: export mm10-evo2-v1 reference tables (genes, peaks, orthologs, panels)."""

import gzip
import hashlib
import json
import os
import shutil
import sys
import time
from collections import OrderedDict

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from _paths import env_str
from anndata.io import read_elem
from safetensors.torch import load_file, save_file

CACHE = env_str("BLOOM_EVO2_CACHE").rstrip("/") + "/"
_DATA = env_str("BLOOM_TRAINING_DATA").rstrip("/")
EMB = f"{_DATA}/embeddings/"
POOLS = f"{_DATA}/strategies/eval/"
OUT = f"{env_str('BLOOM_STAGING')}/references/mm10-evo2-v1"
_LEGACY = env_str("BLOOM_LEGACY_ORTHOLOGS").rstrip("/")
ORTHO_HOUSE = f"{_LEGACY}/orthologs_one2one.tsv"
ORTHO_MU = f"{_LEGACY}/orthologs_one2one_mouseuniverse.tsv"
CHROM_TO_IDX = {f"chr{i}": i - 1 for i in range(1, 20)}
CHROM_TO_IDX.update({"chrX": 19, "chrY": 20, "chrM": 21})
CHROM_TO_IDX.update({"chr20": 22, "chr21": 23, "chr22": 24})
os.makedirs(f"{OUT}/orthologs", exist_ok=True)
os.makedirs(f"{OUT}/panels", exist_ok=True)
T0 = time.time()
S = OrderedDict()


def tick(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def export_table(kind, h5ad, fname_prefix):
    cseq = torch.load(f"{CACHE}/{kind}_cseq.pt", weights_only=True, map_location="cpu")
    chr_ = torch.load(f"{CACHE}/{kind}_chr.pt", weights_only=True, map_location="cpu")
    bp = torch.load(f"{CACHE}/{kind}_bp.pt", weights_only=True, map_location="cpu")
    tick(
        f"{kind}: cseq {tuple(cseq.shape)} {cseq.dtype} chr {tuple(chr_.shape)} {chr_.dtype} bp {tuple(bp.shape)} {bp.dtype}"
    )
    with h5py.File(EMB + h5ad) as f:
        obs = read_elem(f["obs"])
        xshape = tuple(f["X"].shape) if hasattr(f["X"], "shape") else tuple(f["X"].attrs["shape"])
    N = len(obs)
    info = OrderedDict(
        N_pt=int(cseq.shape[0]),
        N_obs=N,
        X_shape=list(xshape),
        dtypes={"cseq": str(cseq.dtype), "chr": str(chr_.dtype), "bp": str(bp.dtype)},
    )
    assert cseq.shape[0] == N == chr_.shape[0] == bp.shape[0], info
    assert cseq.dtype == torch.float32 and chr_.dtype == torch.int8 and bp.dtype == torch.int32, (
        info
    )
    chrom = obs["chrom"].astype(str).to_numpy()
    start = obs["start"].to_numpy().astype(np.int64)
    end = obs["end"].to_numpy().astype(np.int64)
    chr_re = np.array([CHROM_TO_IDX[c] for c in chrom], dtype=np.int8)
    bp_re = ((start + end) // 2).astype(np.int32)
    info["chr_recompute_equal"] = bool(np.array_equal(chr_re, chr_.numpy()))
    info["bp_recompute_equal"] = bool(np.array_equal(bp_re, bp.numpy()))
    info["n_chr_mismatch"] = int((chr_re != chr_.numpy()).sum())
    info["n_bp_mismatch"] = int((bp_re != bp.numpy()).sum())
    info["chrom_counts"] = {
        k: int(v) for k, v in pd.Series(chrom).value_counts().sort_index().items()
    }
    info["cseq_row_norm_mean"] = float(cseq[: min(N, 20000)].norm(dim=1).mean())
    info["cseq_finite"] = bool(torch.isfinite(cseq).all())
    tick(f"{kind}: chr equal={info['chr_recompute_equal']} bp equal={info['bp_recompute_equal']}")
    st = f"{OUT}/{fname_prefix}.safetensors"
    save_file(
        {"cseq": cseq.contiguous(), "chr": chr_.contiguous(), "bp": bp.contiguous()},
        st,
        metadata={
            "reference_id": "mm10-evo2-v1",
            "kind": kind,
            "assembly": "mm10",
            "source_h5ad": h5ad,
            "source_cache": CACHE,
            "n_rows": str(N),
            "chr_encoding": json.dumps(CHROM_TO_IDX),
            "bp": "(start+end)//2",
        },
    )
    back = load_file(st)
    assert (
        torch.equal(back["cseq"], cseq)
        and torch.equal(back["chr"], chr_)
        and torch.equal(back["bp"], bp)
    )
    info["safetensors_bytes"] = os.path.getsize(st)
    tick(f"{kind}: wrote {st} ({info['safetensors_bytes'] / 1e6:.1f} MB), reload bitwise OK")
    return obs, chrom, start, end, info


# ---- genes ----
gobs, gchrom, gstart, gend, ginfo = export_table(
    "gene", "evo2_genes_mean_ctx_2048_pca_768_norm.h5ad", "genes"
)
gdf = pd.DataFrame({
    "emb_idx": np.arange(len(gobs), dtype=np.int32),
    "gene_name": gobs["gene_name"].astype(str).to_numpy(),
    "gene_id": gobs["gene_id"].astype(str).to_numpy(),
    "chrom": gchrom,
    "start": gstart,
    "end": gend,
    "strand": gobs["strand"].astype(str).to_numpy(),
    "obs_name": gobs.index.astype(str).to_numpy(),
})
ginfo["obs_index_col_equals_obs_name"] = (
    bool((gobs["index"].astype(str).to_numpy() == gdf["obs_name"].to_numpy()).all())
    if "index" in gobs
    else None
)
ginfo["n_unique_gene_name"] = int(gdf["gene_name"].nunique())
ginfo["n_dup_gene_name"] = int(gdf["gene_name"].duplicated().sum())
ginfo["n_unique_gene_id"] = int(gdf["gene_id"].nunique())
ginfo["n_missing_gene_id"] = int((gdf["gene_id"].isin(["nan", "", "None", "<NA>"])).sum())
ginfo["dup_gene_name_examples"] = (
    gdf.loc[gdf["gene_name"].duplicated(keep=False), "gene_name"].unique()[:10].tolist()
)
gdf.to_parquet(f"{OUT}/genes.parquet", index=False)
ginfo["parquet_bytes"] = os.path.getsize(f"{OUT}/genes.parquet")
S["genes"] = ginfo
tick(
    f"genes.parquet written; dup gene names={ginfo['n_dup_gene_name']} e.g. {ginfo['dup_gene_name_examples'][:5]}"
)

# ---- peaks ----
pobs, pchrom, pstart, pend, pinfo = export_table(
    "peak", "evo2_peaks_mean_ctx_2048_pca_768_norm.h5ad", "peaks"
)
names = np.char.add(
    np.char.add(np.char.add(pchrom.astype(str), ":"), pstart.astype(str)),
    np.char.add("-", pend.astype(str)),
)
pdf = pd.DataFrame({
    "emb_idx": np.arange(len(pobs), dtype=np.int32),
    "chrom": pchrom,
    "start": pstart,
    "end": pend,
    "name": names,
    "obs_name": pobs.index.astype(str).to_numpy(),
})
pinfo["name_equals_obs_name_all"] = bool((pdf["name"] == pdf["obs_name"]).all())
pinfo["n_unique_name"] = int(pdf["name"].nunique())
pinfo["width_counts"] = {int(k): int(v) for k, v in pd.Series(pend - pstart).value_counts().items()}
pdf.to_parquet(f"{OUT}/peaks.parquet", index=False)
pinfo["parquet_bytes"] = os.path.getsize(f"{OUT}/peaks.parquet")
with gzip.open(f"{OUT}/peaks.bed.gz", "wt", compresslevel=6) as fh:
    pdf[["chrom", "start", "end", "name"]].to_csv(fh, sep="\t", header=False, index=False)
pinfo["bed_gz_bytes"] = os.path.getsize(f"{OUT}/peaks.bed.gz")
pinfo["bed_compressor"] = "python gzip (not bgzip)"
S["peaks"] = pinfo
tick(
    f"peaks.parquet + peaks.bed.gz written; name==obs_name: {pinfo['name_equals_obs_name_all']}; widths {pinfo['width_counts']}"
)

# ---- orthologs ----
oinfo = OrderedDict()
shutil.copyfile(ORTHO_HOUSE, f"{OUT}/orthologs/orthologs_one2one.tsv")
shutil.copyfile(ORTHO_MU, f"{OUT}/orthologs/orthologs_one2one_mouseuniverse.tsv")
A = pd.read_csv(ORTHO_HOUSE, sep="\t")
B = pd.read_csv(ORTHO_MU, sep="\t")
gset = set(gdf["gene_name"])
gids = set(gdf["gene_id"])


def pairs(df):
    return set(zip(df["human_gene_name"].astype(str), df["mouse_gene_name"].astype(str)))


pa, pb = pairs(A), pairs(B)
onlyA, onlyB = sorted(pa - pb), sorted(pb - pa)


def stats(df, nm):
    return OrderedDict(
        rows=len(df),
        columns=list(df.columns),
        n_unique_human=int(df["human_gene_name"].nunique()),
        n_unique_mouse=int(df["mouse_gene_name"].nunique()),
        orthology_type=df["orthology_type"].value_counts().to_dict(),
        confidence=df["orthology_confidence"].value_counts().to_dict(),
        mouse_symbol_in_genes_parquet=int(df["mouse_gene_name"].isin(gset).sum()),
        mouse_id_in_genes_parquet=int(df["mouse_gene_id"].isin(gids).sum()),
        mouse_symbol_missing_examples=df
        .loc[~df["mouse_gene_name"].isin(gset), "mouse_gene_name"]
        .head(8)
        .tolist(),
    )


oinfo["house"] = stats(A, "house")
oinfo["mouseuniverse"] = stats(B, "mouseuniverse")
oinfo["pairs_only_in_house"] = len(onlyA)
oinfo["pairs_only_in_mouseuniverse"] = len(onlyB)
oinfo["pairs_common"] = len(pa & pb)
oinfo["only_house_examples"] = onlyA[:15]
oinfo["only_mouseuniverse_examples"] = onlyB[:15]
oinfo["only_house_mouse_in_genes"] = int(sum(m in gset for _, m in onlyA))
oinfo["only_mu_mouse_in_genes"] = int(sum(m in gset for _, m in onlyB))
S["orthologs"] = oinfo


def fmt_pairs(ps):
    return "\n".join(f"  - {h} -> {m}" for h, m in ps) if ps else "  (none)"


md = f"""# Ortholog tables — diff summary (mm10-evo2-v1)

Two one-to-one human->mouse ortholog tables are staged side by side because the choice is SPEC.md open question #1.

| table | file | rows | unique human | unique mouse | mouse symbols found in genes.parquet | mouse Ensembl ids found in genes.parquet |
|---|---|---|---|---|---|---|
| house (manuscript) | `orthologs_one2one.tsv` (legacy house table, `$BLOOM_LEGACY_ORTHOLOGS`) | {len(A)} | {oinfo["house"]["n_unique_human"]} | {oinfo["house"]["n_unique_mouse"]} | {oinfo["house"]["mouse_symbol_in_genes_parquet"]} | {oinfo["house"]["mouse_id_in_genes_parquet"]} |
| mouseuniverse (eval feature maps) | `orthologs_one2one_mouseuniverse.tsv` (from `s3://bloom-output/mm-hg-data/`) | {len(B)} | {oinfo["mouseuniverse"]["n_unique_human"]} | {oinfo["mouseuniverse"]["n_unique_mouse"]} | {oinfo["mouseuniverse"]["mouse_symbol_in_genes_parquet"]} | {oinfo["mouseuniverse"]["mouse_id_in_genes_parquet"]} |

Columns (identical in both): `{", ".join(A.columns)}`.
orthology_type: house {oinfo["house"]["orthology_type"]}, mouseuniverse {oinfo["mouseuniverse"]["orthology_type"]}.
orthology_confidence: house {oinfo["house"]["confidence"]}, mouseuniverse {oinfo["mouseuniverse"]["confidence"]}.

## Pair overlap (human_gene_name, mouse_gene_name)

- common pairs: {oinfo["pairs_common"]}
- only in house: {len(onlyA)} (mouse symbol present in genes.parquet for {oinfo["only_house_mouse_in_genes"]} of them)
- only in mouseuniverse: {len(onlyB)} (mouse symbol present in genes.parquet for {oinfo["only_mu_mouse_in_genes"]} of them)

Examples only in house:
{fmt_pairs(onlyA[:15])}

Examples only in mouseuniverse:
{fmt_pairs(onlyB[:15])}

Mouse symbols in house not found in genes.parquet (examples): {oinfo["house"]["mouse_symbol_missing_examples"]}
Mouse symbols in mouseuniverse not found in genes.parquet (examples): {oinfo["mouseuniverse"]["mouse_symbol_missing_examples"]}

The release layout in ASSETS.md names a single `orthologs_human_mouse_one2one.tsv`; it will be a copy of whichever table is chosen.
"""
open(f"{OUT}/orthologs/DIFF.md", "w").write(md)
open("DIFF.md", "w").write(md)
tick(
    f"orthologs: house {len(A)} mu {len(B)} common {len(pa & pb)} onlyA {len(onlyA)} onlyB {len(onlyB)}"
)

# ---- panels ----
PANELS = [
    ("mm-brain-adult_tf_dummy", "tf_534", "gene"),
    ("mm-brain-adult_hvg_dummy", "hvg_2000", "gene"),
    ("mm-brain-adult_hvg-8k_dummy", "hvg_8054", "gene"),
    ("mm-brain-adult_merfish-658801_dummy", "merfish_658801", "gene"),
    ("mm-brain-adult_da_dummy", "da_2127", "peak"),
    ("mm-brain-adult_enhancer_dummy", "enhancer_2298", "peak"),
]
pan = OrderedDict()
for src, name, kind in PANELS:
    y = yaml.safe_load(open(f"{POOLS}/{src}.yaml"))
    sections = {
        sec: {k: (len(v) if v is not None else None) for k, v in (y.get(sec) or {}).items()}
        for sec in y
    }
    key = "gene_idx" if kind == "gene" else "peak_idx"
    idx = np.asarray(y["input"][key], dtype=np.int64)
    table = gdf if kind == "gene" else pdf
    col = "gene_name" if kind == "gene" else "name"
    assert idx.min() >= 0 and idx.max() < len(table), (name, idx.min(), idx.max())
    ids = table[col].to_numpy()[idx]
    q = y.get("query") or {}
    qsame = None
    if q.get(key) is not None:
        qsame = bool(np.array_equal(np.asarray(q[key]), idx))
    with open(f"{OUT}/panels/{name}.txt", "w") as fh:
        fh.write(
            f"# bloom panel {name} | kind={kind} | source={src}.yaml (input.{key}) | n={len(idx)} | reference=mm10-evo2-v1 | ids resolved via {'genes' if kind == 'gene' else 'peaks'}.parquet row=emb_idx\n"
        )
        fh.write("\n".join(ids) + "\n")
    pan[name] = OrderedDict(
        source=f"{src}.yaml",
        kind=kind,
        n=len(idx),
        n_unique_ids=len(set(ids)),
        sections=sections,
        query_equals_input=qsame,
        sorted=bool(np.all(np.diff(idx) > 0)),
        examples=ids[:5].tolist(),
    )
    tick(
        f"panel {name}: n={len(idx)} unique={len(set(ids))} sections={sections} query==input: {qsame}"
    )
S["panels"] = pan
S["elapsed_s"] = time.time() - T0
json.dump(S, open("reference_summary.json", "w"), indent=2, default=str)
tick("DONE")
