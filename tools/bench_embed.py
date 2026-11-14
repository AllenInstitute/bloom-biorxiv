"""Measure ``bloom embed`` cost (s/cell, peak RSS, peak VRAM) for docs/validation.md (hardware table).

Maintainer tool (not installed). Each configuration runs ``python -m bloom.cli embed`` in a fresh
subprocess under ``/usr/bin/time -v`` so the RSS is the whole process (model + tables + attention
workspace); VRAM is read from ``uns["bloom"]`` is not available, so it is sampled with
``torch.cuda.max_memory_allocated`` by an in-process run when ``--device cuda`` is requested.

Usage::

    BLOOM_HOME=<assets dir> \\
    <env>/bin/python tools/bench_embed.py --out bench.json --cells 20 \\
        --config top,4096,1,cpu --config top,4096,8,cpu --config resample,8192,1,cpu \\
        --config top,4096,8,cuda --config resample,8192,8,cuda --config resample,8192,8,cuda,bf16

Each ``--config`` is ``strategy,lmax,batch,device[,precision]``; ``resample`` uses ``--draws 1`` so
the cost is per view (= per cell for ``top``). Output: JSON list + a Markdown table on stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import anndata as ad
import numpy as np

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "data" / "rna_100.h5ad"


def select_cells(cells: int, min_nnz: int) -> ad.AnnData:
    """First ``cells`` fixture cells with at least ``min_nnz`` nonzero genes (so padded L reaches lmax)."""
    a = ad.read_h5ad(FIXTURE)
    nnz = np.diff(a.X.tocsr().indptr)
    keep = np.flatnonzero(nnz >= min_nnz)[:cells]
    if keep.size == 0:
        raise SystemExit(f"no fixture cell has >= {min_nnz} nonzero genes (max {nnz.max()})")
    return a[keep].copy()


def run_one(
    cells: int,
    strategy: str,
    lmax: int,
    batch: int,
    device: str,
    precision: str,
    threads: int | None,
    min_nnz: int = 0,
) -> dict:
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as td:
        src = Path(td) / "in.h5ad"
        a = select_cells(cells, min_nnz)
        a.write_h5ad(src)
        out = Path(td) / "out.h5ad"
        args = [
            sys.executable,
            "-m",
            "bloom.cli",
            "embed",
            "--rna",
            str(src),
            "--species",
            "mouse",
            "--strategy",
            strategy,
            "--lmax",
            str(lmax),
            "--batch-size",
            str(batch),
            "--device",
            device,
            "--precision",
            precision,
            "--quiet",
            "-o",
            str(out),
        ]
        if strategy in ("resample", "tile"):
            args += ["--draws", "1"]
        time_bin = shutil.which("time") or "/usr/bin/time"
        env = dict(os.environ)
        if threads:
            env["OMP_NUM_THREADS"] = str(threads)
        t0 = time.perf_counter()
        r = subprocess.run([time_bin, "-v", *args], capture_output=True, text=True, env=env)
        wall = time.perf_counter() - t0
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-3000:])
        m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", r.stderr)
        rss_gb = int(m.group(1)) / 1024**2 if m else float("nan")
        res = ad.read_h5ad(out)
        run = res.uns["bloom"]["modalities"]["rna"]["run"]
        vram_gb = None
        if device == "cuda":
            vram_gb = _vram_in_process(src, strategy, lmax, batch, precision)
        return {
            "strategy": strategy,
            "lmax": lmax,
            "lmax_padded": run["lmax_padded"],
            "batch": batch,
            "device": device,
            "device_name": res.uns["bloom"].get("device_name") or "cpu",
            "precision": precision,
            "cells": int(a.n_obs),
            "min_nnz": min_nnz,
            "views": run["n_views"],
            "s_per_view": round(run["wall_s"] / max(run["n_views"], 1), 3),
            "embed_wall_s": run["wall_s"],
            "process_wall_s": round(wall, 1),
            "peak_rss_gb": round(rss_gb, 2),
            "peak_vram_gb": vram_gb,
            "threads": threads or _torch_threads(),
            "mean_tokens": float(np.nanmean(res.obs["bloom_n_tokens_rna"])),
        }


def _torch_threads() -> int:
    import torch

    return torch.get_num_threads()


def _vram_in_process(src: Path, strategy: str, lmax: int, batch: int, precision: str) -> float:
    import torch

    import bloom
    from bloom.strategies import build

    torch.cuda.reset_peak_memory_stats()
    model = bloom.load_model("hyacinth", device="cuda")
    ds = bloom.prepare(rna=src, species="mouse", reference=model.reference)
    strat = build(strategy, lmax=lmax, draws=1 if strategy in ("resample", "tile") else None)
    bloom.embed(model, ds, {"rna": strat}, batch_size=batch, precision=precision, progress=False)
    peak = torch.cuda.max_memory_allocated() / 1024**3
    del model
    torch.cuda.empty_cache()
    return round(peak, 2)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--config", action="append", required=True, help="strategy,lmax,batch,device[,precision]"
    )
    ap.add_argument("--cells", type=int, default=20)
    ap.add_argument("--threads", type=int, default=None, help="OMP_NUM_THREADS for CPU runs")
    ap.add_argument(
        "--min-nnz",
        type=int,
        default=0,
        help="only cells with >= this many nonzero genes (full-length views)",
    )
    ap.add_argument("--out", type=Path, default=Path("bench_embed.json"))
    args = ap.parse_args()
    rows = []
    for spec in args.config:
        parts = spec.split(",")
        strategy, lmax, batch, device = parts[0], int(parts[1]), int(parts[2]), parts[3]
        precision = parts[4] if len(parts) > 4 else "fp32"
        print(f"== {spec}", file=sys.stderr, flush=True)
        try:
            row = run_one(
                args.cells, strategy, lmax, batch, device, precision, args.threads, args.min_nnz
            )
        except RuntimeError as exc:
            row = {"config": spec, "error": str(exc)[-500:]}
        rows.append(row)
        print(json.dumps(row), file=sys.stderr, flush=True)
        args.out.write_text(json.dumps(rows, indent=1) + "\n")
    print(
        "| strategy | L (padded) | batch | device | precision | s / view | peak RSS (GB) | peak VRAM (GB) |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        if "error" in r:
            print(f"| {r['config']} | error | | | | | | |")
            continue
        vram = "" if r["peak_vram_gb"] is None else f"{r['peak_vram_gb']:.1f}"
        print(
            f"| {r['strategy']} | {r['lmax']} ({r['lmax_padded']}) | {r['batch']} | {r['device_name']} | "
            f"{r['precision']} | {r['s_per_view']:.2f} | {r['peak_rss_gb']:.1f} | {vram} |"
        )


if __name__ == "__main__":
    main()
