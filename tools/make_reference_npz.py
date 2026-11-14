"""Build the consolidated reference-embedding files (one ``.npz`` + one ``.json`` per fixture).

Two modes:

``multiome-20``  -- from the two ``evaluate.py`` runs of ``tools/hydra/multiome_20_reference_embeddings.yaml``
  (``<runs>/tf32/<task>/embeddings.h5ad`` paper-faithful, ``<runs>/tf32off/<task>/embeddings.h5ad`` TF32 off +
  fused fast path off; ``<runs>/{tf32,tf32off}.log``; ``<runs>/timing.json``) write
  ``examples/multiome_20/reference_embeddings.npz`` with arrays ``rna_top4096``, ``rna_top4096_tf32off``,
  ``rna_random_8192x4``, ``rna_random_8192x4_tf32off``, ``atac_random_8192x16``, ``atac_random_8192x16_tf32off``
  (float32 ``(20, 1024)``, rows = fixture obs order = ``cells.tsv``) plus ``obs_names``, and the sidecar
  ``reference_embeddings.json`` (per array: strategy config, seed, TF32 / fast-path flags, export commit, bloom-arch
  commit, eval-input sha256, run wall / GPU peak, parity vs the paper's full-dataset run by ``source_row`` and vs
  the 300-cell references of the same cells).

``multiome-20-add``  -- append arrays from a further pair of ``evaluate.py`` runs (another preset under ``tools/hydra/``,
  e.g. ``multiome_20_tile_reference_embeddings.yaml`` for the ``tile`` preset, issue #31) to the existing
  ``reference_embeddings.npz`` / ``.json`` without touching the arrays already there (byte-identical; checked):
  ``--tasks <task>=<array_base>,...`` names the new arrays (``<array_base>`` and ``<array_base>_tf32off``).

``rna-100``  -- fold the three legacy ``tests/data/ref/rna_100_*.{npy,json}`` pairs into
  ``tests/data/ref/rna_100_reference_embeddings.npz`` (arrays ``rna_top4096``, ``rna_top4096_tf32off``,
  ``rna_random_8192x4``, ``obs_names``) + one ``.json`` that keeps each legacy sidecar verbatim under ``arrays``.

Both write ``allow_pickle``-free npz files: ``np.load(path)[name]`` is all a test needs.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
from _paths import env_str

ORIG = env_str("BLOOM_PAPER_RESULTS")
PAPER_DIRS = {
    "rna_top4096": "eval_balanced_50_allen_multiome_nz_top4096",
    "rna_random_8192x4": "eval_balanced_50_allen_multiome_nz_resamp_8192_d4",
    "atac_random_8192x16": "eval_balanced_50_allen_multiome_nz_resamp_atac_8192_d16",
}
TASK_TO_ARRAY = {
    "multiome_20_top4096": "rna_top4096",
    "multiome_20_resamp_8192_d4": "rna_random_8192x4",
    "multiome_20_resamp_atac_8192_d16": "atac_random_8192x16",
}
ARRAY_TO_300 = {  # the same cells in the (superseded) 300-cell references, for a cross-check
    "rna_top4096": "allen_multiome_300_top4096",
    "rna_top4096_tf32off": "allen_multiome_300_top4096_tf32off",
    "rna_random_8192x4": "allen_multiome_300_resamp_8192_d4",
    "atac_random_8192x16": "allen_multiome_300_resamp_atac_8192_d16",
}
ARRAY_ORDER = [
    "rna_top4096",
    "rna_top4096_tf32off",
    "rna_random_8192x4",
    "rna_random_8192x4_tf32off",
    "atac_random_8192x16",
    "atac_random_8192x16_tf32off",
]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def stats(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.abs(a - b)
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return {
        "n_cells": int(a.shape[0]),
        "max_abs_diff": float(d.max()),
        "mean_abs_diff": float(d.mean()),
        "p99_abs_diff": float(np.quantile(d, 0.99)),
        "cos_min": float(cos.min()),
        "cos_mean": float(cos.mean()),
        "n_cells_maxdiff_above_1e-4": int((d.max(1) > 1e-4).sum()),
        "n_cells_maxdiff_above_1e-3": int((d.max(1) > 1e-3).sum()),
    }


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_head(path: str) -> str:
    return subprocess.check_output(
        ["git", "-C", os.path.expanduser(path), "rev-parse", "HEAD"], text=True
    ).strip()


# --------------------------------------------------------------------------------------------------
def multiome_20(args: argparse.Namespace) -> None:
    import anndata as ad
    import pandas as pd
    import yaml

    fx = args.fixture
    cells = pd.read_csv(f"{fx}/cells.tsv", sep="\t")
    obs_names = cells["obs_name"].astype(str).to_numpy()
    src = cells["source_row"].to_numpy().astype(np.int64)
    n = len(cells)
    cfg = yaml.safe_load(open(args.preset))
    timing = json.load(open(f"{args.runs}/timing.json"))
    selection = (
        json.load(open(args.selection)) if args.selection and os.path.exists(args.selection) else {}
    )

    def log_lines(flavour: str, tag: str) -> list[str]:
        p = f"{args.runs}/{flavour}.log"
        return [ln.strip() for ln in open(p) if tag in ln]

    # tie at the 4096 cut on the fixture (top4096 parity vs the paper's UNSORTED run holds only on tie-free cells)
    a = ad.read_h5ad(f"{fx}/rna.h5ad")
    X = a.X.tocsr()
    assert list(a.obs_names) == obs_names.tolist()
    tie = np.zeros(n, dtype=bool)
    for i in range(n):
        v = np.sort(X.data[X.indptr[i] : X.indptr[i + 1]])[::-1]
        tie[i] = v.size > 4096 and v[4095] == v[4096]

    arrays: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}
    for task, base in TASK_TO_ARRAY.items():
        tcfg = cfg["evaluate"]["tasks"][task]
        for flavour in ("tf32", "tf32off"):
            name = base if flavour == "tf32" else f"{base}_tf32off"
            raw = f"{args.runs}/{flavour}/{task}/embeddings.h5ad"
            e = ad.read_h5ad(raw)
            emb = np.asarray(e.obsm["bloom_emb"], dtype=np.float32)
            assert emb.shape == (n, 1024), (name, emb.shape)
            assert np.isfinite(emb).all(), name
            assert list(e.obs_names) == [str(i) for i in range(n)], "unexpected obs index"
            arrays[name] = emb
            m: dict = {
                "shape": [n, 1024],
                "dtype": "float32",
                "task": task,
                "strategy": tcfg["strategy"],
                "modality": "atac" if base.startswith("atac") else "rna",
                "seed": cfg["evaluate"]["seed"],
                "per_cell_seed": "seed + file_idx*100000 + row_idx (row_idx = row in the 20-cell file = position in cells.tsv)",
                "batch_size": cfg["evaluate"]["batch_size"],
                "num_workers": cfg["evaluate"]["num_workers"],
                "precision": cfg["evaluate"]["precision"],
                "float32_matmul_precision": "medium (TF32 matmuls on; set by evaluate.py)"
                if flavour == "tf32"
                else "highest (TF32 off: BLOOM_MATMUL_PRECISION=highest; cuda.matmul.allow_tf32=False, cudnn.allow_tf32=False)",
                "mha_fastpath": flavour == "tf32",
                "mha_fastpath_note": "torch default (fused CUDA nn.TransformerEncoder path, as the paper's evaluations)"
                if flavour == "tf32"
                else "BLOOM_MHA_FASTPATH=0 -> torch.backends.mha.set_fastpath_enabled(False); unfused math == CPU",
                "numerics_log_lines": log_lines(flavour, "[tf32-check]")
                + log_lines(flavour, "[fastpath-check]"),
                "seq_head_scale_checked": log_lines(flavour, "[scale-check]")[:4],
                "aggregation": "mean of per-draw h_cell over draws (EvalWriter online mean)",
                "variant": "paper-faithful numerics on column-sorted inputs (reference for the GPU default, G2b-style < 5e-4)"
                if flavour == "tf32"
                else "strict reference for CPU fp32 and GPU TF32-off (G1-style < 1e-4)",
                "run": {
                    **(timing.get(flavour) or {}),
                    "note": "wall of the whole evaluate.py run (3 tasks incl. model load)",
                },
                "raw": f"{raw}",
                "raw_sha256": sha256(raw),
            }
            # parity vs the paper's full-dataset run (same cells by source_row)
            pdir = f"{ORIG}/{PAPER_DIRS[base]}/embeddings.h5ad"
            if os.path.exists(pdir):
                P = np.asarray(ad.read_h5ad(pdir).obsm["bloom_emb"], dtype=np.float32)
                assert P.shape[0] == 15750, P.shape
                ref = P[src]
                par = {"orig_dir": PAPER_DIRS[base], "all_cells": stats(emb, ref)}
                if base == "rna_top4096":
                    par["cells_without_tie_at_cut"] = stats(emb[~tie], ref[~tie])
                    par["n_cells_with_tie_at_cut"] = int(tie.sum())
                    par["note"] = (
                        "the paper's run consumed unsorted CSR rows: ties at the 4096 cut break differently; compare tie-free cells"
                    )
                else:
                    par["note"] = (
                        "random strategy: per-cell seeds differ from the full-dataset run -> statistical comparison only"
                    )
                m["parity_vs_paper_run"] = par
            # cross-check vs the 300-cell references of the same cells
            p300 = f"{args.refs300}/{ARRAY_TO_300.get(name, '')}.npy"
            if ARRAY_TO_300.get(name) and os.path.exists(p300) and selection.get("cells"):
                pos300 = np.array([c["pos_in_300"] for c in selection["cells"]], dtype=np.int64)
                R = np.load(p300)[pos300]
                m["parity_vs_allen_multiome_300_reference"] = {
                    "file": os.path.basename(p300),
                    **stats(emb, R),
                    "note": "same cells, same inputs (column-sorted), same numerics flavour; differs only by batch composition"
                    if base == "rna_top4096"
                    else "random: per-cell seeds differ (row index in a 20-cell vs a 300-cell file) -> statistical only",
                }
            meta[name] = m
            print(
                name,
                json.dumps({
                    k: m[k]
                    for k in ("parity_vs_paper_run", "parity_vs_allen_multiome_300_reference")
                    if k in m
                }),
                flush=True,
            )
    for base in TASK_TO_ARRAY.values():
        meta[base]["diff_tf32_vs_tf32off"] = stats(arrays[base], arrays[f"{base}_tf32off"])
        meta[f"{base}_tf32off"]["diff_tf32_vs_tf32off"] = meta[base]["diff_tf32_vs_tf32off"]
        print(base, "tf32 vs tf32off", json.dumps(meta[base]["diff_tf32_vs_tf32off"]), flush=True)

    out_npz = f"{fx}/reference_embeddings.npz"
    np.savez_compressed(
        out_npz, obs_names=obs_names.astype(str), **{k: arrays[k] for k in ARRAY_ORDER}
    )
    chk = np.load(out_npz)
    assert list(chk["obs_names"]) == obs_names.tolist() and all(
        np.array_equal(chk[k], arrays[k]) for k in ARRAY_ORDER
    )
    for k in ARRAY_ORDER:
        meta[k]["sha256"] = hashlib.sha256(np.ascontiguousarray(arrays[k]).tobytes()).hexdigest()
    ev = selection.get("eval_inputs", {})
    side = {
        "file": "reference_embeddings.npz",
        "fixture": "examples/multiome_20",
        "n_cells": n,
        "dim": 1024,
        "row_order": "identical to obs order of examples/multiome_20/{rna,atac}.h5ad (cells.tsv; sorted by source_row)",
        "obs_names": obs_names.tolist(),
        "source_rows": src.tolist(),
        "arrays": {k: meta[k] for k in ARRAY_ORDER},
        "inputs": {
            "eval_inputs_dir": f"{env_str('BLOOM_EVAL_INPUTS_SORTED')}/multiome_20",
            "csr_rows_column_sorted": True,
            "sha256": {k: v.get("sha256") for k, v in ev.items()},
            "fixture_sha256": {k: v.get("sha256") for k, v in selection.get("fixture", {}).items()},
            "selection": selection.get("rule"),
            "subclasses": selection.get("subclasses"),
        },
        "producer": {
            "repo": "alleninstitute/bloom",
            "commit": git_head(env_str("BLOOM_TRAINING_REPO")),
            "script": "src/bloom/evaluate.py (unchanged) via tools/run_evaluate_checked.py",
            "preset": "tools/hydra/multiome_20_reference_embeddings.yaml",
            "env": os.path.basename(sys.prefix),
            "bloom_arch_commit": git_head(args.arch_worktree),
            "bloom_arch_worktree": args.arch_worktree,
            "PYTHONPATH": f"{args.arch_worktree}/src",
            "host": "maintainer GPU host (docs/maintainers.md)",
            "gpu": "NVIDIA L4 (23 GB)",
            "checkpoint": f"{env_str('BLOOM_TRAINING_CKPT')} (hyacinth, step 530000)",
            "reference_space": "mm10 (evo2_mean_ctx_2048_pca_768_norm)",
            "dataset_name_for_feature_map": "eval_balanced_50_allen_multiome",
            "converter": "tools/make_reference_npz.py multiome-20",
            "date": now(),
        },
    }
    with open(f"{fx}/reference_embeddings.json", "w") as fh:
        json.dump(side, fh, indent=2)
    print("wrote", out_npz, os.path.getsize(out_npz), "B and sidecar", flush=True)


# --------------------------------------------------------------------------------------------------
def multiome_20_add(args: argparse.Namespace) -> None:
    """Append the arrays of another preset's two runs to the fixture's npz + sidecar (existing arrays untouched)."""
    import anndata as ad
    import yaml

    fx = args.fixture
    npz_path, json_path = f"{fx}/reference_embeddings.npz", f"{fx}/reference_embeddings.json"
    side = json.load(open(json_path))
    with np.load(npz_path) as z:
        existing = {k: np.asarray(z[k]) for k in z.files}
    obs_names = existing["obs_names"].astype(str)
    n = len(obs_names)
    assert side["obs_names"] == obs_names.tolist()
    cfg = yaml.safe_load(open(args.preset))
    timing = json.load(open(f"{args.runs}/timing.json"))
    tasks = dict(kv.split("=", 1) for kv in args.tasks.split(","))

    def log_lines(flavour: str, tag: str) -> list[str]:
        return [ln.strip() for ln in open(f"{args.runs}/{flavour}.log") if tag in ln]

    new_arrays: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}
    for task, base in tasks.items():
        tcfg = cfg["evaluate"]["tasks"][task]
        for flavour in ("tf32", "tf32off"):
            name = base if flavour == "tf32" else f"{base}_tf32off"
            if name in existing and not args.replace:
                raise SystemExit(f"{name} already in {npz_path}; pass --replace to overwrite it")
            raw = f"{args.runs}/{flavour}/{task}/embeddings.h5ad"
            e = ad.read_h5ad(raw)
            emb = np.asarray(e.obsm["bloom_emb"], dtype=np.float32)
            assert emb.shape == (n, 1024) and np.isfinite(emb).all(), (name, emb.shape)
            assert list(e.obs_names) == [str(i) for i in range(n)], "unexpected obs index"
            new_arrays[name] = emb
            meta[name] = {
                "shape": [n, 1024],
                "dtype": "float32",
                "task": task,
                "strategy": tcfg["strategy"],
                "modality": "atac" if base.startswith("atac") else "rna",
                "seed": cfg["evaluate"]["seed"],
                "per_cell_seed": "seed + file_idx*100000 + row_idx (row_idx = row in the 20-cell file = position in cells.tsv)",
                "batch_size": cfg["evaluate"]["batch_size"],
                "num_workers": cfg["evaluate"]["num_workers"],
                "precision": cfg["evaluate"]["precision"],
                "float32_matmul_precision": "medium (TF32 matmuls on; set by evaluate.py)"
                if flavour == "tf32"
                else "highest (TF32 off: BLOOM_MATMUL_PRECISION=highest; cuda.matmul.allow_tf32=False, cudnn.allow_tf32=False)",
                "mha_fastpath": flavour == "tf32",
                "mha_fastpath_note": "torch default (fused CUDA nn.TransformerEncoder path, as the paper's evaluations)"
                if flavour == "tf32"
                else "BLOOM_MHA_FASTPATH=0 -> torch.backends.mha.set_fastpath_enabled(False); unfused math == CPU",
                "numerics_log_lines": log_lines(flavour, "[tf32-check]")
                + log_lines(flavour, "[fastpath-check]"),
                "seq_head_scale_checked": log_lines(flavour, "[scale-check]")[:4],
                "aggregation": "mean of per-draw h_cell over draws (EvalWriter online mean)",
                "variant": "paper-faithful numerics on column-sorted inputs"
                if flavour == "tf32"
                else "TF32 off + unfused (== CPU math == what `bloom` computes on any device since v0.1.0)",
                "run": {
                    **(timing.get(flavour) or {}),
                    "note": f"wall of the whole evaluate.py run ({len(tasks)} task(s) incl. model load)",
                },
                "raw": raw,
                "raw_sha256": sha256(raw),
                "parity_note": "stochastic strategy (per-cell RNG stream of evaluate.py is not ported): statistical parity only",
                "preset": os.path.relpath(args.preset),
                "added": now(),
                "producer": {
                    "commit": git_head(env_str("BLOOM_TRAINING_REPO")),
                    "bloom_arch_commit": git_head(args.arch_worktree),
                    "env": os.path.basename(sys.prefix),
                },
            }
    for base in tasks.values():
        d = stats(new_arrays[base], new_arrays[f"{base}_tf32off"])
        meta[base]["diff_tf32_vs_tf32off"] = meta[f"{base}_tf32off"]["diff_tf32_vs_tf32off"] = d
        print(base, "tf32 vs tf32off", json.dumps(d), flush=True)
    for k in new_arrays:
        meta[k]["sha256"] = hashlib.sha256(
            np.ascontiguousarray(new_arrays[k]).tobytes()
        ).hexdigest()

    order = [k for k in existing if k != "obs_names"] + [k for k in new_arrays if k not in existing]
    merged = {**{k: v for k, v in existing.items() if k != "obs_names"}, **new_arrays}
    np.savez_compressed(npz_path, obs_names=obs_names, **{k: merged[k] for k in order})
    with np.load(npz_path) as chk:
        assert list(chk["obs_names"]) == obs_names.tolist()
        for k, v in existing.items():
            if k != "obs_names" and k not in new_arrays:
                assert np.array_equal(chk[k], v), f"existing array {k} changed"
        for k, v in new_arrays.items():
            assert np.array_equal(chk[k], v)
    side["arrays"].update(meta)
    side.setdefault("additions", []).append({
        "date": now(),
        "arrays": sorted(new_arrays),
        "preset": os.path.relpath(args.preset),
        "runs": args.runs,
        "converter": "tools/make_reference_npz.py multiome-20-add",
    })
    with open(json_path, "w") as fh:
        json.dump(side, fh, indent=2)
    print(
        "wrote",
        npz_path,
        os.path.getsize(npz_path),
        "B (+",
        sorted(new_arrays),
        ") and sidecar",
        flush=True,
    )


# --------------------------------------------------------------------------------------------------
def rna_100(args: argparse.Namespace) -> None:
    d = args.ref_dir
    legacy = {
        "rna_top4096": "rna_100_top4096",
        "rna_top4096_tf32off": "rna_100_top4096_tf32off",
        "rna_random_8192x4": "rna_100_resamp_8192_d4",
    }
    arrays, sides = {}, {}
    for name, stem in legacy.items():
        arrays[name] = np.load(f"{d}/{stem}.npy").astype(np.float32)
        sides[name] = json.load(open(f"{d}/{stem}.json"))
        assert arrays[name].shape == (100, 1024)
    obs = sides["rna_top4096"]["obs_names"]
    src = sides["rna_top4096"]["source_rows"]
    for s in sides.values():
        assert s["obs_names"] == obs and s["source_rows"] == src
    out_npz = f"{d}/rna_100_reference_embeddings.npz"
    np.savez_compressed(out_npz, obs_names=np.array(obs, dtype=str), **arrays)
    chk = np.load(out_npz)
    assert all(np.array_equal(chk[k], arrays[k]) for k in arrays)
    side = {
        "file": "rna_100_reference_embeddings.npz",
        "fixture": "tests/data/rna_100.h5ad",
        "n_cells": 100,
        "dim": 1024,
        "row_order": "identical to obs order of tests/data/rna_100.h5ad (rna_100_cells.tsv; sorted by source_row)",
        "obs_names": obs,
        "source_rows": src,
        "arrays": {},
        "consolidated": {
            "date": now(),
            "from": [f"{v}.npy + {v}.json" for v in legacy.values()],
            "converter": "tools/make_reference_npz.py rna-100",
            "note": "arrays are the legacy .npy files byte-for-byte; each legacy sidecar is kept verbatim below (minus the repeated obs_names/source_rows)",
        },
    }
    for name, stem in legacy.items():
        s = {k: v for k, v in sides[name].items() if k not in ("obs_names", "source_rows")}
        s["legacy_file"] = f"{stem}.npy"
        s["legacy_npy_sha256"] = sha256(f"{d}/{stem}.npy")
        s["sha256"] = hashlib.sha256(np.ascontiguousarray(arrays[name]).tobytes()).hexdigest()
        s["modality"] = "rna"
        s["mha_fastpath"] = not name.endswith("_tf32off")
        side["arrays"][name] = s
    with open(f"{d}/rna_100_reference_embeddings.json", "w") as fh:
        json.dump(side, fh, indent=2)
    print("wrote", out_npz, os.path.getsize(out_npz), "B", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="mode", required=True)
    m = sub.add_parser("multiome-20")
    m.add_argument("--runs", required=True)
    m.add_argument(
        "--fixture",
        required=True,
        help="examples/multiome_20 dir (rna.h5ad, cells.tsv; npz + json written here)",
    )
    m.add_argument("--preset", default="tools/hydra/multiome_20_reference_embeddings.yaml")
    m.add_argument(
        "--selection", default=None, help="multiome_20_selection.json from make_multiome_20.py"
    )
    m.add_argument(
        "--refs300",
        default=None,
        help="default <$BLOOM_STAGING>/archive/examples/reference_embeddings",
    )
    m.add_argument(
        "--arch-worktree",
        default=os.environ.get("BLOOM_ARCH_WORKTREE"),
        help="default $BLOOM_ARCH_WORKTREE",
    )
    a = sub.add_parser("multiome-20-add")
    a.add_argument("--runs", required=True)
    a.add_argument("--fixture", required=True)
    a.add_argument("--preset", required=True, help="the tools/hydra preset the runs were made with")
    a.add_argument(
        "--tasks",
        required=True,
        help="<task>=<array_base>,... e.g. multiome_20_tile_4096_d2=rna_tile_4096x2",
    )
    a.add_argument("--replace", action="store_true", help="overwrite arrays that already exist")
    a.add_argument("--arch-worktree", default=os.environ.get("BLOOM_ARCH_WORKTREE"))
    r = sub.add_parser("rna-100")
    r.add_argument("--ref-dir", default="tests/data/ref")
    args = ap.parse_args(argv)
    {"multiome-20": multiome_20, "multiome-20-add": multiome_20_add, "rna-100": rna_100}[args.mode](
        args
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
