"""G6: overlap-matched vs exact-matched ATAC embeddings on the 20-cell fixture (docs/test-plan.md G6).

Synthetic "different peak caller": the fixture's peak coordinates are shifted by ``--shifts`` bp
(default 50, 150, 250 of the 501-bp reference width) and matched back onto the reference with
``--peak-match overlap`` at each ``--min-overlap`` in ``--thresholds``. For every (shift, threshold)
the script records the mapping rate, the detected reference peaks per cell and the cosine of the
deterministic ``top`` (by count, lmax 8192) embedding against the exact-matched run of the unshifted
fixture. ``top`` keeps the comparison free of sampling noise; the numbers feed the ``--min-overlap``
recommendation in docs/input-data.md.

Usage (host, CPU; ~30 s per configuration at L = 8192 on 16 threads)::

    BLOOM_HOME=... python tools/atac_overlap_study.py --out g6.json [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np

import bloom
from bloom.peaks import parse_peaks
from bloom.prepare import NoMappedFeaturesError, prepare_atac
from bloom.strategies import TopExpressed


def _stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    d = np.abs(a - b)
    cos = np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return {
        "max_abs": float(d.max()),
        "mean_cosine": float(cos.mean()),
        "min_cosine": float(cos.min()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--fixture", type=Path, default=None, help="examples/multiome_20 dir (default: BLOOM_HOME)"
    )
    ap.add_argument("--shifts", type=int, nargs="+", default=[50, 150, 250])
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--lmax", type=int, default=8192)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    model = bloom.load_model("hyacinth", device=args.device)
    fixture = args.fixture or (bloom.assets.bloom_home() / "examples" / "multiome_20")
    atac = ad.read_h5ad(fixture / "atac.h5ad")
    parsed = parse_peaks(atac.var)
    strat = {"atac": TopExpressed(lmax=args.lmax)}
    bs = 8 if args.device.startswith("cuda") else 1

    def run(peak_match: str, min_overlap: float, shift: int) -> tuple[dict, np.ndarray | None]:
        t0 = time.perf_counter()
        try:
            prepared, rep, _ = prepare_atac(
                atac,
                reference=model.reference,
                peak_match=peak_match,
                min_overlap=min_overlap,
                _parsed_override=parsed.shifted(shift) if shift else None,
            )
        except NoMappedFeaturesError:  # exact matching of shifted coordinates maps nothing (#34)
            return {
                "shift_bp": shift,
                "peak_match": peak_match,
                "min_overlap": min_overlap if peak_match == "overlap" else None,
                "n_user_peaks": int(atac.n_vars),
                "n_user_peaks_mapped": 0,
                "mapping_rate": 0.0,
                "n_pairs": 0,
                "n_ref_peaks_hit": 0,
                "median_detected_per_cell": 0.0,
                "prepare_s": round(time.perf_counter() - t0, 1),
                "embed_s": 0.0,
                "vs_exact": {"note": "nothing mapped (NoMappedFeaturesError)"},
            }, None
        t_prep = time.perf_counter() - t0
        ds = bloom.PreparedDataset(
            atac=prepared,
            species="mouse",
            reference_id=model.reference.reference_id,
            report={"atac": rep},
        )
        t0 = time.perf_counter()
        out = bloom.embed(model, ds, strat, batch_size=bs, progress=False)
        row = {
            "shift_bp": shift,
            "peak_match": rep["peak_match"],
            "min_overlap": min_overlap if rep["peak_match"] == "overlap" else None,
            "n_user_peaks": rep["n_var"],
            "n_user_peaks_mapped": rep["n_mapped"],
            "mapping_rate": round(rep["mapping_rate"], 4),
            "n_pairs": rep["n_pairs"],
            "n_ref_peaks_hit": rep["n_ref_peaks_hit"],
            "median_detected_per_cell": rep["median_detected"],
            "prepare_s": round(t_prep, 1),
            "embed_s": round(time.perf_counter() - t0, 1),
        }
        return row, out.obsm["X_bloom_atac"]

    rows = []
    base_row, base = run("exact", 0.5, 0)
    base_row["vs_exact"] = _stats(base, base)
    rows.append(base_row)
    print(json.dumps(base_row))
    for shift in args.shifts:
        ex_row, ex = run("exact", 0.5, shift)
        if ex is not None:
            ex_row["vs_exact"] = (
                _stats(ex, base) if np.isfinite(ex).all() else {"note": "NaN rows (unmapped cells)"}
            )
        rows.append(ex_row)
        print(json.dumps(ex_row))
        for thr in args.thresholds:
            row, X = run("overlap", thr, shift)
            if X is not None:
                row["vs_exact"] = (
                    _stats(X, base)
                    if np.isfinite(X).all()
                    else {"note": "NaN rows (unmapped cells)"}
                )
            rows.append(row)
            print(json.dumps(row))
    report = {
        "fixture": str(fixture),
        "strategy": strat["atac"].params(),
        "device": str(model.device),
        "n_cells": int(atac.n_obs),
        "reference_peak_width_bp": 501,
        "rows": rows,
    }
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    # markdown table for docs/validation.md
    lines = [
        "| shift (bp) | match | min_overlap | user peaks mapped | ref peaks hit | median detected / cell | mean cos vs exact | min cos | max abs |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        v = r["vs_exact"]
        cos = (
            f"{v['mean_cosine']:.5f} | {v['min_cosine']:.5f} | {v['max_abs']:.2e}"
            if "mean_cosine" in v
            else "n/a | n/a | n/a"
        )
        lines.append(
            f"| {r['shift_bp']} | {r['peak_match']} | {r['min_overlap'] if r['min_overlap'] is not None else '-'} | "
            f"{r['n_user_peaks_mapped']:,} ({100 * r['mapping_rate']:.1f}%) | {r['n_ref_peaks_hit']:,} | "
            f"{r['median_detected_per_cell']:.0f} | {cos} |"
        )
    md = "\n".join(lines) + "\n"
    args.out.with_suffix(".md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
