#!/usr/bin/env python
"""Publish the staged assets to the public bucket ``s3://bloom-biorxiv`` (maintainer tool).

Reads ``MANIFEST.json`` from the staging root and uploads every listed file to the bucket under
its manifest path (the bucket layout *is* the release layout, so ``bloom.assets.UrlSource`` needs no
mapping). Files whose remote size and ``sha256`` metadata already match the manifest are skipped,
so re-runs are cheap. Two unmanifested, human-readable documents (``tools/bucket/README.md`` ->
``README.md`` and ``tools/bucket/MODEL_CARD.md`` -> ``models/hyacinth/MODEL_CARD.md``) are uploaded
too, and ``MANIFEST.json`` goes **last** so a reader that sees the new manifest also finds its files.

Anything not in the manifest (the staging tree's ``archive/``, scratch) is never uploaded. ``--prune`` deletes
bucket objects that are neither manifest entries nor the three documents (dry-run prints the list first; the
option refuses to run against any bucket other than ``bloom-biorxiv``).

Uses the AWS CLI (``aws s3 cp`` multipart defaults; ``aws s3api head-object`` for the skip check) with
the host's instance role; ``--verify`` afterwards talks to the bucket **anonymously** over plain HTTPS
(``HEAD`` per file, size check; small files re-downloaded and sha256-checked) so it exercises exactly
what a user without credentials gets.

Examples::

    python tools/publish_assets.py --dry-run
    python tools/publish_assets.py                     # upload (skips up-to-date objects)
    python tools/publish_assets.py --verify            # anonymous HEAD + sha check of small files
    python tools/publish_assets.py --verify --verify-max-bytes 0   # HEAD only
    python tools/publish_assets.py --prune --dry-run   # list objects that are not in the manifest
    python tools/publish_assets.py --prune --verify    # upload new objects, delete stale ones, verify

Runs on the maintainers' host with write access to the bucket (6.2 GB; ``peaks.safetensors`` alone is 5.4 GB).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGING = Path(os.environ["BLOOM_STAGING"]) if os.environ.get("BLOOM_STAGING") else None
DEFAULT_BUCKET = "bloom-biorxiv"
DEFAULT_REGION = "us-west-2"

#: Unmanifested human-readable files: repo path -> bucket key.
EXTRA_FILES: dict[str, str] = {
    "tools/bucket/README.md": "README.md",
    "tools/bucket/MODEL_CARD.md": "models/hyacinth/MODEL_CARD.md",
}

#: Content types for the extensions ``mimetypes`` does not know (or gets wrong for our purpose).
CONTENT_TYPES: dict[str, str] = {
    ".safetensors": "application/octet-stream",
    ".parquet": "application/vnd.apache.parquet",
    ".h5ad": "application/x-hdf5",
    ".npy": "application/octet-stream",
    ".tsv": "text/tab-separated-values",
    ".txt": "text/plain",
    ".json": "application/json",
    ".md": "text/markdown",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
    ".gz": "application/gzip",
}

_CHUNK = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    """sha256 hex digest of a file (streamed)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(_CHUNK):
            h.update(block)
    return h.hexdigest()


def content_type_for(key: str) -> str:
    """Content-Type for a bucket key (``.bed.gz`` -> gzip, ``.md`` -> markdown, ...)."""
    suffix = Path(key).suffix.lower()
    if suffix in CONTENT_TYPES:
        return CONTENT_TYPES[suffix]
    guess, _ = mimetypes.guess_type(key)
    return guess or "application/octet-stream"


def public_url(bucket: str, region: str, key: str) -> str:
    """Regional virtual-hosted URL (the region-less hostname answers 307)."""
    return f"https://{bucket}.s3.{region}.amazonaws.com/{quote(key)}"


@dataclass(frozen=True)
class Item:
    """One object to publish."""

    key: str
    local: Path
    size: int
    sha256: str
    manifest: bool  # False for the unmanifested README / MODEL_CARD


def load_items(staging: Path, repo_root: Path) -> tuple[list[Item], Item]:
    """Manifest entries (+ extras) in upload order, and the manifest itself (uploaded last)."""
    manifest_path = staging / "MANIFEST.json"
    doc = json.loads(manifest_path.read_bytes())
    items: list[Item] = []
    for f in doc["files"]:
        local = staging / f["path"]
        if not local.is_file():
            sys.exit(f"missing staged file {local}")
        size = local.stat().st_size
        if size != int(f["size_bytes"]):
            sys.exit(f"{f['path']}: staged size {size} != manifest {f['size_bytes']}")
        items.append(Item(f["path"], local, size, str(f["sha256"]).lower(), True))
    for rel, key in EXTRA_FILES.items():
        local = repo_root / rel
        if not local.is_file():
            sys.exit(f"missing {local} (unmanifested bucket document)")
        items.append(Item(key, local, local.stat().st_size, sha256_file(local), False))
    man = Item(
        "MANIFEST.json",
        manifest_path,
        manifest_path.stat().st_size,
        sha256_file(manifest_path),
        False,
    )
    return items, man


# ----------------------------------------------------------------------------------------------
# AWS CLI helpers


def aws(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run ``aws <args>`` and return the completed process (stdout captured)."""
    cmd = ["aws", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if check and proc.returncode != 0:
        sys.exit(f"{' '.join(cmd)}\n{proc.stderr.strip()}")
    return proc


def list_bucket(bucket: str) -> dict[str, int]:
    """``{key: size}`` of every object in the bucket (paginated by the CLI)."""
    proc = aws(
        "s3api",
        "list-objects-v2",
        "--bucket",
        bucket,
        "--output",
        "json",
        "--query",
        "Contents[].{Key: Key, Size: Size}",
    )
    data = json.loads(proc.stdout or "null") or []
    return {o["Key"]: int(o["Size"]) for o in data}


def head_object(bucket: str, key: str) -> dict | None:
    """``head-object`` JSON (``ContentLength``, ``Metadata``, ...) or ``None`` when absent."""
    proc = aws(
        "s3api", "head-object", "--bucket", bucket, "--key", key, "--output", "json", check=False
    )
    if proc.returncode != 0:
        if "404" in proc.stderr or "Not Found" in proc.stderr:
            return None
        sys.exit(f"head-object {key}: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def remote_matches(bucket: str, item: Item, sizes: dict[str, int]) -> bool:
    """True when the object exists with the manifest size and ``sha256`` metadata."""
    if sizes.get(item.key) != item.size:
        return False
    head = head_object(bucket, item.key)
    if head is None or int(head.get("ContentLength", -1)) != item.size:
        return False
    return (head.get("Metadata") or {}).get("sha256", "").lower() == item.sha256


def delete_objects(bucket: str, keys: list[str], *, dry_run: bool) -> None:
    """``aws s3api delete-objects`` in batches of 1000 (the API maximum)."""
    for i in range(0, len(keys), 1000):
        batch = keys[i : i + 1000]
        for k in batch:
            print(f"{'would delete' if dry_run else 'delete':<12} {k}")
        if dry_run:
            continue
        payload = json.dumps({"Objects": [{"Key": k} for k in batch], "Quiet": True})
        aws("s3api", "delete-objects", "--bucket", bucket, "--delete", payload)


def upload(bucket: str, item: Item, *, dry_run: bool) -> float:
    """``aws s3 cp`` one object (multipart handled by the CLI); returns wall seconds."""
    cmd = [
        "s3",
        "cp",
        str(item.local),
        f"s3://{bucket}/{item.key}",
        "--no-progress",
        "--content-type",
        content_type_for(item.key),
        "--metadata",
        f"sha256={item.sha256}",
    ]
    if dry_run:
        return 0.0
    t0 = time.perf_counter()
    aws(*cmd)
    return time.perf_counter() - t0


# ----------------------------------------------------------------------------------------------
# anonymous verification


def anon_head(url: str) -> tuple[int, int, str]:
    """Anonymous ``HEAD``: (status, Content-Length, Content-Type); status 0 on connection error."""
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return (
                resp.status,
                int(resp.headers.get("Content-Length", -1)),
                resp.headers.get("Content-Type", ""),
            )
    except urllib.error.HTTPError as exc:
        return exc.code, -1, ""
    except OSError:
        return 0, -1, ""


def anon_sha256(url: str) -> tuple[str, int, float]:
    """Anonymous GET; returns (sha256, bytes, seconds)."""
    h = hashlib.sha256()
    n = 0
    t0 = time.perf_counter()
    with urllib.request.urlopen(url, timeout=600) as resp:
        while block := resp.read(_CHUNK):
            h.update(block)
            n += len(block)
    return h.hexdigest(), n, time.perf_counter() - t0


def verify(
    bucket: str, region: str, items: list[Item], *, max_bytes: int
) -> tuple[int, list[dict]]:
    """HEAD every object anonymously; GET + sha256 those up to ``max_bytes``.

    Returns:
        ``(n_failures, rows)`` with one row per object (status, Content-Length, sha check).
    """
    rows = []
    failures = 0
    print(f"{'status':<8} {'size':>14} {'sha':<6} {'MB/s':>7}  key")
    for it in items:
        url = public_url(bucket, region, it.key)
        status, length, ctype = anon_head(url)
        ok = status == 200 and length == it.size
        sha_state, speed = "-", ""
        if ok and it.size <= max_bytes:
            digest, n, secs = anon_sha256(url)
            ok = digest == it.sha256 and n == it.size
            sha_state = "ok" if ok else "BAD"
            speed = f"{n / secs / 1e6:7.1f}" if secs > 0 else ""
        failures += not ok
        print(f"{status:<8} {length:>14,} {sha_state:<6} {speed:>7}  {it.key}")
        rows.append({
            "key": it.key,
            "url": url,
            "http_status": status,
            "content_length": length,
            "content_type": ctype,
            "expected_size": it.size,
            "expected_sha256": it.sha256,
            "sha256_checked": sha_state,
            "ok": ok,
        })
    print(f"verified {len(items)} objects anonymously, {failures} failure(s)")
    return failures, rows


# ----------------------------------------------------------------------------------------------
# main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--staging", type=Path, default=DEFAULT_STAGING, help="staging root with MANIFEST.json"
    )
    ap.add_argument("--bucket", default=DEFAULT_BUCKET)
    ap.add_argument("--region", default=DEFAULT_REGION)
    ap.add_argument(
        "--repo-root", type=Path, default=REPO_ROOT, help="repo holding tools/bucket/*.md"
    )
    ap.add_argument("--dry-run", action="store_true", help="print the plan, upload nothing")
    ap.add_argument(
        "--force", action="store_true", help="upload even when the remote object matches"
    )
    ap.add_argument(
        "--verify", action="store_true", help="after (or instead of) uploading, verify anonymously"
    )
    ap.add_argument(
        "--verify-only", action="store_true", help="skip uploading; just verify anonymously"
    )
    ap.add_argument(
        "--verify-max-bytes",
        type=int,
        default=2 * 1024 * 1024,
        help="GET + sha256 files up to this size during --verify (default 2 MiB; 0 = HEAD only)",
    )
    ap.add_argument(
        "--prune",
        action="store_true",
        help="delete bucket objects that are not manifest entries / README / MODEL_CARD / MANIFEST.json "
        "(only against the default bucket; combine with --dry-run to list first)",
    )
    ap.add_argument(
        "--report",
        type=Path,
        default=None,
        help="write a JSON report (upload table + verification)",
    )
    args = ap.parse_args(argv)
    if args.staging is None:
        ap.error("--staging DIR or $BLOOM_STAGING is required")

    if shutil.which("aws") is None and not args.verify_only:
        sys.exit("aws CLI not found")
    if args.prune and args.bucket != DEFAULT_BUCKET:
        sys.exit(f"--prune is only allowed against s3://{DEFAULT_BUCKET}")

    items, man = load_items(args.staging, args.repo_root)
    total = sum(i.size for i in items if i.manifest)
    n_manifest = sum(i.manifest for i in items)
    print(f"staging  {args.staging}")
    print(
        f"bucket   s3://{args.bucket} ({args.region}); {n_manifest} manifest files, {total:,} B, "
        f"+{len(items) - n_manifest} unmanifested docs, + MANIFEST.json last"
    )

    table: list[dict] = []
    if not args.verify_only:
        sizes = list_bucket(args.bucket)
        print(f"remote   {len(sizes)} object(s) present")
        print(f"{'action':<9} {'size':>14} {'secs':>7} {'MB/s':>7}  key")
        for it in [*items, man]:
            if not args.force and remote_matches(args.bucket, it, sizes):
                action, secs = "skip", 0.0
            else:
                action = "plan" if args.dry_run else "upload"
                secs = upload(args.bucket, it, dry_run=args.dry_run)
            speed = f"{it.size / secs / 1e6:7.1f}" if secs > 0 else ""
            print(f"{action:<9} {it.size:>14,} {secs:>7.1f} {speed:>7}  {it.key}")
            table.append({
                "key": it.key,
                "size": it.size,
                "sha256": it.sha256,
                "action": action,
                "seconds": round(secs, 2),
                "content_type": content_type_for(it.key),
                "manifest": it.manifest,
            })
        if not args.dry_run:
            remote = list_bucket(args.bucket)
            expected = {i.key for i in items} | {man.key}
            extra = sorted(set(remote) - expected)
            missing = sorted(expected - set(remote))
            print(
                f"remote now {len(remote)} object(s); expected {len(expected)}; "
                f"missing {len(missing)}; unexpected {len(extra)}"
            )
            if extra and not args.prune:
                for k in extra:
                    print(f"  unexpected object (not removed; use --prune): {k}")
            if missing:
                print("  missing:", *missing, sep="\n  ")
                return 1
        if args.prune:
            remote = list_bucket(args.bucket)
            expected = {i.key for i in items} | {man.key}
            extra = sorted(set(remote) - expected)
            print(f"prune    {len(extra)} object(s) not in the manifest / documents")
            delete_objects(args.bucket, extra, dry_run=args.dry_run)
            if not args.dry_run:
                remote = list_bucket(args.bucket)
                left = sorted(set(remote) - expected)
                print(
                    f"remote now {len(remote)} object(s) = {len(expected)} expected; stale left {len(left)}"
                )
                if left:
                    return 1

    failures, vrows = 0, []
    if args.verify or args.verify_only:
        failures, vrows = verify(
            args.bucket, args.region, [*items, man], max_bytes=args.verify_max_bytes
        )
    if args.report is not None:
        args.report.write_text(
            json.dumps(
                {"uploads": table, "verification": vrows, "verify_failures": failures}, indent=1
            )
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
