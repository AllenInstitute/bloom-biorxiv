"""Tests for ``bloom.assets`` (docs/test-plan.md E5-E7; manifest pin; resume logic).

HTTP is served by a local ``http.server`` thread with ``Range`` support so the ``unit`` tier needs
no network. The ``assets``-marked test downloads small files anonymously from the public bucket
(the default source).
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bloom import assets
from bloom.assets import (
    DEFAULT_ASSET_URL,
    GROUPS,
    MANIFEST_SHA256,
    AssetMissingError,
    AssetVerificationError,
    FileSource,
    UrlSource,
    download,
    load_manifest,
    locate,
    manifest,
    parse_groups,
    sha256_file,
    verify,
)
from bloom.cli import app

runner = CliRunner()

# ----------------------------------------------------------------------------------------------
# local HTTP server with Range support


@dataclass
class ServerConfig:
    """Knobs for the test server, mutated by tests."""

    ignore_range: bool = False  # answer 200 + full body even when Range is sent
    truncate_next: int = 0  # send only this many bytes on the next request (0 = off)
    truncate_every: int = 0  # send at most this many bytes on EVERY request (-1 = none; 0 = off)
    requests: list[dict] = field(default_factory=list)  # (path, Range header) per request


class _Handler(BaseHTTPRequestHandler):
    root: Path
    config: ServerConfig

    def log_message(self, *args) -> None:  # silence
        pass

    def do_GET(self) -> None:
        rel = self.path.split("?", 1)[0].lstrip("/")
        self.config.requests.append({"path": rel, "range": self.headers.get("Range")})
        path = self.root / rel
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = path.read_bytes()
        start = 0
        rng = self.headers.get("Range")
        if rng and not self.config.ignore_range:
            start = int(rng.split("=", 1)[1].rstrip("-"))
            if start >= len(data):
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{len(data)}")
                self.end_headers()
                return
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        else:
            self.send_response(HTTPStatus.OK)
        body = data[start:]
        if self.config.truncate_next:
            body = body[: self.config.truncate_next]
            self.config.truncate_next = 0
            # announce the full length so the client sees a short read, as a dropped link would
            self.send_header("Content-Length", str(len(data) - start))
        elif self.config.truncate_every:
            body = body[: max(self.config.truncate_every, 0)]
            self.send_header("Content-Length", str(len(data) - start))
        else:
            self.send_header("Content-Length", str(len(body)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass


@dataclass
class LocalServer:
    url: str
    root: Path
    config: ServerConfig


@pytest.fixture
def served(tmp_path_factory) -> Iterator[LocalServer]:
    root = tmp_path_factory.mktemp("served")
    config = ServerConfig()
    handler = type("Handler", (_Handler,), {"root": root, "config": config})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield LocalServer(f"http://127.0.0.1:{httpd.server_address[1]}", root, config)
    finally:
        httpd.shutdown()
        httpd.server_close()


# ----------------------------------------------------------------------------------------------
# a tiny manifest over files we generate


TINY_FILES = {
    # path -> (manifest group, size)
    "models/hyacinth/model.safetensors": ("model", 50_000),
    "models/hyacinth/config.json": ("model", 300),
    "references/mm10-evo2-v1/genes.safetensors": ("reference", 40_000),
    "references/mm10-evo2-v1/genes.parquet": ("reference", 2_000),
    "references/mm10-evo2-v1/peaks.safetensors": ("reference", 120_000),
    "references/mm10-evo2-v1/peaks.parquet": ("reference", 3_000),
    "references/mm10-evo2-v1/panels/hvg_1999.txt": ("reference", 500),
    "references/mm10-evo2-v1/orthologs_human_mouse_one2one.tsv": ("reference", 900),
    "examples/multiome_20/rna.h5ad": ("example", 7_000),
    "examples/multiome_20/cells.tsv": ("example", 400),
    "examples/multiome_20/reference_embeddings.npz": ("example", 800),
}


def _write_tiny_tree(root: Path, seed: int = 0) -> dict:
    files = []
    for path, (group, size) in TINY_FILES.items():
        data = hashlib.blake2b(f"{seed}:{path}".encode(), digest_size=32).digest()
        data = (data * (size // 32 + 1))[:size]
        p = root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        files.append({
            "path": path,
            "size_bytes": size,
            "sha256": hashlib.sha256(data).hexdigest(),
            "group": group,
            "version": "test",
            "license": "test",
        })
    return {
        "manifest_version": 1,
        "created": "2026-01-01T00:00:00Z",
        "model_id": "hyacinth",
        "reference_id": "mm10-evo2-v1",
        "examples_version": "examples-v1",
        "files": files,
    }


@pytest.fixture
def tiny(served: LocalServer, monkeypatch):
    """Tiny manifest served over HTTP (release layout)."""
    doc = _write_tiny_tree(served.root)
    m = load_manifest(json.dumps(doc).encode())
    monkeypatch.setattr(assets, "_MANIFEST", m)
    monkeypatch.setattr(assets.time, "sleep", lambda *_: None)  # no backoff in tests
    return m


# ----------------------------------------------------------------------------------------------
# manifest


@pytest.mark.unit
def test_packaged_manifest_matches_pin():
    m = manifest()
    assert m.sha256 == MANIFEST_SHA256
    assert m.manifest_version == 1
    assert m.model_id == "hyacinth" and m.reference_id == "mm10-evo2-v1"
    assert len(m.files) == 15
    assert m.total_bytes == 6_076_889_558  # examples-v2 + tile reference arrays (#31)
    counts = {g: len(m.group(g)) for g in GROUPS}
    assert counts == {
        "model": 2,
        "genes": 2,
        "peaks": 3,
        "orthologs": 1,
        "panels": 2,
        "examples": 5,
    }
    assert sum(counts.values()) == len(m.files)  # every entry has exactly one group
    assert {f.name for f in m.group("examples")} == {
        "rna.h5ad",
        "atac.h5ad",
        "cells.tsv",
        "reference_embeddings.npz",
        "reference_embeddings.json",
    }
    assert {f.name for f in m.group("panels")} == {"hvg_1999.txt", "da_2127.txt"}


@pytest.mark.unit
def test_manifest_pin_mismatch_is_rejected():
    data = json.dumps({"manifest_version": 1, "files": []}).encode()
    with pytest.raises(AssetVerificationError, match="does not match the pinned"):
        load_manifest(data, expected_sha256="0" * 64)
    load_manifest(data, expected_sha256=hashlib.sha256(data).hexdigest())  # correct pin passes


@pytest.mark.unit
def test_manifest_version_guard():
    data = json.dumps({"manifest_version": 2, "files": []}).encode()
    with pytest.raises(AssetVerificationError, match="manifest_version 2"):
        load_manifest(data)


@pytest.mark.unit
def test_parse_groups():
    assert parse_groups(None) == GROUPS
    assert parse_groups("all") == GROUPS
    assert parse_groups("genes, model") == ("model", "genes")  # canonical order
    assert parse_groups(["peaks"]) == ("peaks",)
    with pytest.raises(ValueError, match="unknown asset group"):
        parse_groups("weights")


# ----------------------------------------------------------------------------------------------
# download / verify (UrlSource over the local server)


@pytest.mark.unit
def test_download_fresh_then_idempotent(tiny, served, tmp_path):
    src = UrlSource(served.url)
    res = download("genes", tmp_path, source=src)
    assert [r.status for r in res] == ["downloaded", "downloaded"]
    for r in res:
        assert r.path.is_file() and sha256_file(r.path) == r.asset.sha256
        assert not assets._part_path(r.path).exists()
    n_requests = len(served.config.requests)
    res2 = download("genes", tmp_path, source=src)
    assert [r.status for r in res2] == ["present", "present"]
    assert len(served.config.requests) == n_requests  # nothing fetched
    assert all(v.status == "ok" for v in verify("genes", tmp_path))


@pytest.mark.unit
def test_e5_tampered_file_is_detected_and_refetched(tiny, served, tmp_path):
    """E5: a cached file with the right size but wrong content is re-fetched."""
    src = UrlSource(served.url)
    (res,) = download(
        files=[tiny.by_path()["references/mm10-evo2-v1/genes.parquet"]], dest=tmp_path, source=src
    )
    path = res.path
    good = path.read_bytes()
    tampered = bytearray(good)
    tampered[10] ^= 0xFF
    path.write_bytes(bytes(tampered))  # same size, wrong sha
    (chk,) = verify(files=[res.asset], dest=tmp_path)
    assert chk.status == "sha_mismatch"
    (res2,) = download(files=[res.asset], dest=tmp_path, source=src)
    assert res2.status == "refetched"
    assert path.read_bytes() == good
    # truncated file (wrong size) is also re-fetched, without hashing
    path.write_bytes(good[:-5])
    (chk,) = verify(files=[res.asset], dest=tmp_path, checksum=False)
    assert chk.status == "size_mismatch"
    (res3,) = download(files=[res.asset], dest=tmp_path, source=src, verify_existing=False)
    assert res3.status == "refetched" and path.read_bytes() == good


@pytest.mark.unit
def test_force_refetches_verified_file(tiny, served, tmp_path):
    src = UrlSource(served.url)
    entry = tiny.by_path()["models/hyacinth/config.json"]
    download(files=[entry], dest=tmp_path, source=src)
    n = len(served.config.requests)
    (res,) = download(files=[entry], dest=tmp_path, source=src, force=True)
    assert res.status == "refetched" and len(served.config.requests) == n + 1


@pytest.mark.unit
def test_resume_partial_download_with_range(tiny, served, tmp_path):
    """A leftover ``.part`` is resumed with ``Range: bytes=N-`` and only the tail is fetched."""
    entry = tiny.by_path()["references/mm10-evo2-v1/peaks.safetensors"]
    full = (served.root / entry.path).read_bytes()
    final = tmp_path / entry.path
    part = assets._part_path(final)
    part.parent.mkdir(parents=True)
    n_have = 70_000
    part.write_bytes(full[:n_have])
    (chk,) = verify(files=[entry], dest=tmp_path)
    assert chk.status == "partial" and chk.size_bytes == n_have
    (res,) = download(files=[entry], dest=tmp_path, source=UrlSource(served.url))
    assert res.status == "downloaded"
    assert final.read_bytes() == full and not part.exists()
    assert served.config.requests[-1]["range"] == f"bytes={n_have}-"


@pytest.mark.unit
def test_resume_when_server_ignores_range(tiny, served, tmp_path):
    """A 200 answer to a Range request restarts the transfer instead of appending."""
    served.config.ignore_range = True
    entry = tiny.by_path()["references/mm10-evo2-v1/peaks.safetensors"]
    full = (served.root / entry.path).read_bytes()
    final = tmp_path / entry.path
    part = assets._part_path(final)
    part.parent.mkdir(parents=True)
    part.write_bytes(b"x" * 1000)  # garbage prefix that a naive append would keep
    (res,) = download(files=[entry], dest=tmp_path, source=UrlSource(served.url))
    assert res.status == "downloaded" and final.read_bytes() == full


@pytest.mark.unit
def test_oversized_part_is_discarded(tiny, served, tmp_path):
    entry = tiny.by_path()["references/mm10-evo2-v1/genes.parquet"]
    final = tmp_path / entry.path
    part = assets._part_path(final)
    part.parent.mkdir(parents=True)
    part.write_bytes(b"y" * (entry.size_bytes + 10))
    (res,) = download(files=[entry], dest=tmp_path, source=UrlSource(served.url))
    assert res.status == "downloaded" and sha256_file(final) == entry.sha256


@pytest.mark.unit
def test_interrupted_transfer_is_resumed(tiny, served, tmp_path, monkeypatch):
    """A connection that drops mid-body is retried with Range from the bytes already on disk."""
    monkeypatch.setattr(assets, "_STREAM_CHUNK", 4096)  # small blocks so bytes land before EOF
    entry = tiny.by_path()["references/mm10-evo2-v1/peaks.safetensors"]
    served.config.truncate_next = 30_000
    (res,) = download(files=[entry], dest=tmp_path, source=UrlSource(served.url))
    assert res.status == "downloaded" and sha256_file(res.path) == entry.sha256
    ranges = [r["range"] for r in served.config.requests if r["path"] == entry.path]
    assert ranges[0] is None and len(ranges) == 2
    resumed_from = int(ranges[1].removeprefix("bytes=").rstrip("-"))
    assert 0 < resumed_from <= 30_000 and resumed_from % 4096 == 0


@pytest.mark.unit
def test_repeated_interruptions_succeed_while_progress_is_made(tiny, served, tmp_path, monkeypatch):
    """More than ``_MAX_ATTEMPTS`` drops are fine as long as every reconnect adds bytes (5.4 GB file)."""
    monkeypatch.setattr(assets, "_STREAM_CHUNK", 4096)
    entry = tiny.by_path()["references/mm10-evo2-v1/peaks.safetensors"]  # 120,000 B
    n_drops = assets._MAX_ATTEMPTS + 3
    served.config.truncate_every = 12_288  # every answer carries 3 blocks, then the link drops
    (res,) = download(files=[entry], dest=tmp_path, source=UrlSource(served.url))
    assert res.status == "downloaded" and sha256_file(res.path) == entry.sha256
    ranges = [r["range"] for r in served.config.requests if r["path"] == entry.path]
    assert len(ranges) > n_drops  # resumed many more times than the no-progress budget
    assert ranges[0] is None and all(r.startswith("bytes=") for r in ranges[1:])


@pytest.mark.unit
def test_no_progress_attempts_are_bounded(tiny, served, tmp_path, monkeypatch):
    """A server that never delivers a byte fails after ``_MAX_ATTEMPTS`` with a resume hint."""
    entry = tiny.by_path()["references/mm10-evo2-v1/genes.parquet"]
    served.config.truncate_every = -1  # headers only, zero-byte bodies
    (res,) = download(files=[entry], dest=tmp_path, source=UrlSource(served.url))
    assert res.status == "failed" and "without progress" in res.message
    n = len([r for r in served.config.requests if r["path"] == entry.path])
    assert n == assets._MAX_ATTEMPTS


@pytest.mark.unit
def test_wrong_content_from_source_fails_verification(tiny, served, tmp_path):
    entry = tiny.by_path()["references/mm10-evo2-v1/genes.parquet"]
    (served.root / entry.path).write_bytes(b"z" * entry.size_bytes)  # source serves wrong bytes
    (res,) = download(files=[entry], dest=tmp_path, source=UrlSource(served.url))
    assert res.status == "failed" and "sha256" in res.message
    assert not res.path.exists() and not assets._part_path(res.path).exists()


@pytest.mark.unit
def test_url_source_404_is_unavailable(tiny, served, tmp_path):
    entry = tiny.by_path()["models/hyacinth/model.safetensors"]
    (served.root / entry.path).unlink()
    (res,) = download(files=[entry], dest=tmp_path, source=UrlSource(served.url))
    assert res.status == "failed" and "404" in res.message


# ----------------------------------------------------------------------------------------------
# sources


@pytest.mark.unit
def test_file_source_links_without_hashing(tiny, served, tmp_path):
    src = FileSource(served.root)
    res = download("peaks", tmp_path, source=src)
    assert [r.status for r in res] == ["linked"] * 2
    for r in res:
        assert r.path.is_symlink() and r.path.resolve() == (served.root / r.asset.path).resolve()
    assert all(v.status == "ok" for v in verify("peaks", tmp_path))  # hashing on demand works
    res2 = download("peaks", tmp_path, source=src)
    assert [r.status for r in res2] == ["present"] * 2
    copied = download("genes", tmp_path, source=FileSource(served.root, link=False))
    assert [r.status for r in copied] == ["downloaded"] * 2
    assert not copied[0].path.is_symlink()


@pytest.mark.unit
def test_default_asset_url_is_the_regional_bucket_host():
    """The region-less S3 hostname answers 307; the default must be the regional one over https."""
    assert DEFAULT_ASSET_URL == "https://bloom-biorxiv.s3.us-west-2.amazonaws.com/"
    assert DEFAULT_ASSET_URL.startswith("https://")
    assert ".s3.us-west-2.amazonaws.com/" in DEFAULT_ASSET_URL
    src = UrlSource()
    assert src.base_url == DEFAULT_ASSET_URL.rstrip("/")
    entry = manifest().by_path()["models/hyacinth/model.safetensors"]
    assert src.url_for(entry) == (
        "https://bloom-biorxiv.s3.us-west-2.amazonaws.com/models/hyacinth/model.safetensors"
    )


@pytest.mark.unit
def test_source_from_spec(tmp_path, monkeypatch):
    monkeypatch.delenv(assets.SOURCE_ENV, raising=False)
    default = assets.source_from_spec(None)
    assert isinstance(default, UrlSource) and default.base_url == DEFAULT_ASSET_URL.rstrip("/")
    assert isinstance(assets.default_source(), UrlSource)
    assert assets.source_from_spec("bucket").base_url == default.base_url
    with pytest.raises(ValueError, match="unrecognised asset source"):
        assets.source_from_spec("github")
    assert (
        assets.source_from_spec("url:https://b.example/bloom/").base_url
        == "https://b.example/bloom"
    )
    fs = assets.source_from_spec(f"file://{tmp_path}")
    assert isinstance(fs, FileSource) and fs.root == tmp_path and fs.link
    assert not assets.source_from_spec(f"file://{tmp_path}?copy").link
    monkeypatch.setenv(assets.SOURCE_ENV, f"file://{tmp_path}")
    assert isinstance(assets.source_from_spec(None), FileSource)
    with pytest.raises(ValueError, match="unrecognised asset source"):
        assets.source_from_spec("ftp://nope")


# ----------------------------------------------------------------------------------------------
# locate (E7)


@pytest.mark.unit
def test_e7_locate_missing_peak_table_names_the_fix(tmp_path):
    with pytest.raises(AssetMissingError) as exc:
        locate("peaks", tmp_path)
    msg = str(exc.value)
    assert "peak table not downloaded" in msg
    assert "bloom download --what peaks" in msg
    assert isinstance(exc.value, FileNotFoundError)
    with pytest.raises(AssetMissingError, match="bloom download --what model"):
        locate("model", tmp_path)
    with pytest.raises(AssetMissingError, match="bloom download --what genes"):
        locate("references/mm10-evo2-v1/genes.parquet", tmp_path)
    with pytest.raises(AssetMissingError, match="bloom download --what orthologs"):
        locate("orthologs", tmp_path)
    assert locate("peaks", tmp_path, required=False) == (
        tmp_path / "references/mm10-evo2-v1/peaks.safetensors"
    )


@pytest.mark.unit
def test_locate_present_uses_bloom_home(tmp_path, monkeypatch):
    monkeypatch.setenv("BLOOM_HOME", str(tmp_path))
    p = tmp_path / assets.ASSET_KINDS["genes_table"]
    p.parent.mkdir(parents=True)
    p.write_bytes(b"")
    assert locate("genes_table") == p
    assert assets.bloom_home() == tmp_path


# ----------------------------------------------------------------------------------------------
# CLI


@pytest.mark.unit
def test_e6_info_without_assets_lists_groups_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("BLOOM_HOME", str(tmp_path))
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0, result.output
    for g in GROUPS:
        assert f"{g.ljust(10)} missing" in result.output
    assert "bloom download --what model,genes,peaks,orthologs,panels,examples" in result.output
    assert MANIFEST_SHA256[:16] in result.output


@pytest.mark.unit
def test_cli_download_and_info_round_trip(tiny, served, tmp_path, monkeypatch):
    monkeypatch.setenv("BLOOM_HOME", str(tmp_path))
    result = runner.invoke(
        app, ["download", "--what", "genes,model", "--source", f"url:{served.url}"]
    )
    assert result.exit_code == 0, result.output
    assert "downloaded" in result.output and "done: model ok, genes ok" in result.output
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0, result.output
    assert "model      present" in result.output
    assert "genes      present" in result.output
    assert "peaks      missing" in result.output
    # corrupt a file: size-only info says present, --verify flags it
    p = tmp_path / "references/mm10-evo2-v1/genes.parquet"
    data = bytearray(p.read_bytes())
    data[0] ^= 0xFF
    p.write_bytes(bytes(data))
    assert "genes      present" in runner.invoke(app, ["info"]).output
    out = runner.invoke(app, ["info", "--verify"]).output
    assert "genes      corrupt" in out and "genes.parquet" in out
    # and `bloom download` repairs it (E5 through the CLI)
    result = runner.invoke(app, ["download", "--what", "genes", "--source", f"url:{served.url}"])
    assert result.exit_code == 0 and "refetched" in result.output


@pytest.mark.unit
def test_cli_download_group_missing_from_source_exits_nonzero(tiny, served, monkeypatch, tmp_path):
    """A source lacking a whole group exits 1; one lacking part of a group completes with a warning."""
    (served.root / "models/hyacinth/model.safetensors").unlink()  # model: partial
    for f in (served.root / "examples/multiome_20").iterdir():
        f.unlink()  # examples: nothing
    monkeypatch.setenv(
        assets.SOURCE_ENV, f"file://{served.root}"
    )  # env override selects the source
    result = runner.invoke(app, ["download", "--what", "examples,genes", "--dir", str(tmp_path)])
    assert result.exit_code == 1, result.output
    assert "not available from file://" in result.output
    assert "incomplete group(s): examples" in result.output
    assert (tmp_path / "references/mm10-evo2-v1/genes.parquet").exists()  # other group still done
    # a group the source carries only partially completes with a warning (exit 0) ...
    monkeypatch.delenv(assets.SOURCE_ENV, raising=False)
    result = runner.invoke(
        app,
        [
            "download",
            "--what",
            "model",
            "--dir",
            str(tmp_path),
            "--source",
            f"file://{served.root}",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "partial group(s): model" in result.output
    assert (tmp_path / "models/hyacinth/config.json").exists()
    # ... unless --strict is given
    result = runner.invoke(
        app,
        [
            "download",
            "--what",
            "model",
            "--dir",
            str(tmp_path),
            "--source",
            f"file://{served.root}",
            "--strict",
        ],
    )
    assert result.exit_code == 1 and "incomplete group(s): model" in result.output


@pytest.mark.unit
def test_cli_default_source_is_the_bucket(monkeypatch):
    monkeypatch.delenv(assets.SOURCE_ENV, raising=False)
    monkeypatch.setattr(assets, "download", lambda **kw: [])  # no network
    result = runner.invoke(app, ["download", "--what", "model"])
    assert result.exit_code == 0, result.output
    assert f"source  url:{DEFAULT_ASSET_URL.rstrip('/')}" in result.output


@pytest.mark.unit
def test_cli_download_rejects_unknown_group():
    result = runner.invoke(app, ["download", "--what", "weights"])
    assert result.exit_code == 2 and "unknown asset group" in result.output


# ----------------------------------------------------------------------------------------------
# real sources (assets tier; network)


@pytest.mark.assets
def test_bucket_anonymous_download_genes_parquet_and_orthologs(tmp_path, monkeypatch):
    """Fetch two small files from the public bucket with no credentials and verify their sha256."""
    for var in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        assets.SOURCE_ENV,
    ):
        monkeypatch.delenv(var, raising=False)
    by_path = manifest().by_path()
    entries = [
        by_path["references/mm10-evo2-v1/genes.parquet"],
        by_path["references/mm10-evo2-v1/orthologs_human_mouse_one2one.tsv"],
    ]
    src = assets.default_source()
    assert isinstance(src, UrlSource) and src.base_url == DEFAULT_ASSET_URL.rstrip("/")
    results = download(files=entries, dest=tmp_path, source=src)
    assert [r.status for r in results] == ["downloaded", "downloaded"], [r.message for r in results]
    for r in results:
        assert r.path.stat().st_size == r.asset.size_bytes
        assert sha256_file(r.path) == r.asset.sha256
    assert results[0].path.stat().st_size == 1_733_063
    assert results[1].path.stat().st_size == 837_340
    # the bucket also serves MANIFEST.json, byte-identical to the packaged copy
    resp = src._session.get(f"{src.base_url}/MANIFEST.json", timeout=60)
    assert resp.status_code == 200
    assert hashlib.sha256(resp.content).hexdigest() == MANIFEST_SHA256
