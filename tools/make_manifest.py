#!/usr/bin/env python
"""Write ``MANIFEST.json`` for the staging tree and sync the packaged copy (maintainer tool).

Walks the staging root (``--staging``, default ``$BLOOM_STAGING``) and lists every regular file
except ``MANIFEST.json`` itself, anything under ``archive/`` (superseded assets kept on the host only) and
symlinks. Each entry carries ``path`` (= bucket key), ``size_bytes``, ``sha256``, ``group``, ``version`` and
``license``. The shipped ortholog table's ``#`` header (Ensembl release, counts, filter) is copied into
``versions["mm10-evo2-v1"]["orthologs"]``.

With ``--repo-root`` (default: this checkout) the manifest is also written to ``src/bloom/data/manifest.json``
and ``MANIFEST_SHA256`` in ``src/bloom/assets.py`` is updated, so the package pin always matches the staged
tree. Follow with ``python tools/publish_assets.py --dry-run`` / ``--prune``.

Usage::

    python tools/make_manifest.py [--staging ROOT] [--repo-root DIR] [--no-sync]
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SKIP_TOP = {"MANIFEST.json", "archive"}
#: SPDX id of the asset license (weights, reference tables, example data, model card); decided 2026-11-13 (#10).
LICENSE_ASSETS = "CC-BY-NC-4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-nc/4.0/"


def sha256(path: Path, bufsize: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def classify(rel: str) -> tuple[str, str, str]:
    """``(group, version, license)`` for a manifest path."""
    if rel.startswith("models/"):
        return "model", "hyacinth", LICENSE_ASSETS
    if rel.startswith("references/"):
        return "reference", "mm10-evo2-v1", LICENSE_ASSETS
    if rel.startswith("examples/"):
        return "example", "examples-v2", LICENSE_ASSETS
    sys.exit(f"unexpected staged file outside models/ references/ examples/: {rel}")


def tsv_header(path: Path) -> dict[str, str]:
    """``key: value`` pairs from the leading ``#`` lines of a TSV."""
    out: dict[str, str] = {}
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            body = line[1:].strip()
            if ":" in body:
                k, v = body.split(":", 1)
                out[k.strip()] = v.strip()
    return out


def build(staging: Path) -> dict:
    files = []
    for root, dirs, fns in os.walk(staging):
        rel_root = os.path.relpath(root, staging)
        if rel_root == ".":
            dirs[:] = sorted(d for d in dirs if d not in SKIP_TOP)
        else:
            dirs.sort()
        for fn in sorted(fns):
            full = Path(root) / fn
            rel = os.path.relpath(full, staging)
            if rel in SKIP_TOP or full.is_symlink() or fn.startswith("."):
                continue
            g, v, lic = classify(rel)
            files.append({
                "path": rel,
                "size_bytes": full.stat().st_size,
                "sha256": sha256(full),
                "group": g,
                "version": v,
                "license": lic,
            })
            print(f"{files[-1]['sha256'][:12]}  {files[-1]['size_bytes']:>13,}  {rel}", flush=True)
    orth = staging / "references/mm10-evo2-v1/orthologs_human_mouse_one2one.tsv"
    ortholog_meta = (
        {
            k: tsv_header(orth).get(k)
            for k in ("ensembl_release", "counts", "filter", "join", "generated")
        }
        if orth.exists()
        else None
    )
    return {
        "manifest_version": 1,
        "created": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "base_url": "https://bloom-biorxiv.s3.us-west-2.amazonaws.com/",
        "model_id": "hyacinth",
        "reference_id": "mm10-evo2-v1",
        "examples_version": "examples-v2",
        "versions": {
            "hyacinth": {
                "group": "model",
                "source": "alleninstitute/bloom training run hyacinth, last.ckpt step 530000",
            },
            "mm10-evo2-v1": {
                "group": "reference",
                "source": "evo2_mean_ctx_2048_pca_768_norm (genes 32,129; peaks 1,747,701)",
                "orthologs": ortholog_meta,
            },
            "examples-v2": {
                "group": "example",
                "source": "examples/multiome_20: 20 cells (10 AIT33 subclasses x 2) of hyacinth-v1/eval_balanced_50/allen_multiome, "
                "seed 0, with evaluate.py reference embeddings (reference_embeddings.{npz,json})",
            },
        },
        "licenses": {
            LICENSE_ASSETS: (
                "Creative Commons Attribution-NonCommercial 4.0 International: model weights, derived "
                f"reference tables, example data and the model card ({LICENSE_URL}); the companion code "
                "is MIT (repository LICENSE)"
            ),
        },
        "n_files": len(files),
        "total_bytes": sum(f["size_bytes"] for f in files),
        "files": files,
    }


def sync_package(repo_root: Path, manifest_bytes: bytes) -> None:
    """Write the packaged copy and update ``MANIFEST_SHA256`` in ``src/bloom/assets.py``."""
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    pkg = repo_root / "src/bloom/data/manifest.json"
    pkg.write_bytes(manifest_bytes)
    assets_py = repo_root / "src/bloom/assets.py"
    src = assets_py.read_text()
    new, n = re.subn(r'MANIFEST_SHA256 = "[0-9a-f_A-Z]*"', f'MANIFEST_SHA256 = "{digest}"', src)
    if n != 1:
        sys.exit("could not find MANIFEST_SHA256 in src/bloom/assets.py")
    assets_py.write_text(new)
    print(f"synced {pkg} and MANIFEST_SHA256 = {digest}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--staging",
        type=Path,
        default=os.environ.get("BLOOM_STAGING"),
        help="staging tree (= bucket content); default $BLOOM_STAGING",
    )
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    ap.add_argument("--no-sync", action="store_true", help="only write <staging>/MANIFEST.json")
    args = ap.parse_args(argv)
    if args.staging is None:
        ap.error("--staging DIR or $BLOOM_STAGING is required")
    doc = build(args.staging)
    data = (json.dumps(doc, indent=2) + "\n").encode()
    (args.staging / "MANIFEST.json").write_bytes(data)
    print(
        f"n_files {doc['n_files']} total_bytes {doc['total_bytes']:,} sha256 {hashlib.sha256(data).hexdigest()}"
    )
    if not args.no_sync:
        sync_package(args.repo_root, data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
