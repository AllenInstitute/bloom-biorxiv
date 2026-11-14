"""Asset manifest, cache directory and verified downloads (docs/assets.md sections 2-3).

Every released file (model weights, reference tables, example / fixture data) is listed in
``MANIFEST.json`` with its path, size and sha256. A copy of that manifest is packaged with the code
(``bloom/data/manifest.json``) and pinned by sha256 (:data:`MANIFEST_SHA256`), so a given package
version always knows exactly which bytes it was tested against.

Files are cached under ``BLOOM_HOME`` (``$BLOOM_HOME`` or ``~/.cache/bloom``) in the release
layout. :func:`download` fetches a *group* of files from an :class:`AssetSource`, verifying size and
sha256, resuming partial transfers (HTTP ``Range``) and renaming atomically. :func:`locate` is the
read side used by the model and reference loaders and raises :class:`AssetMissingError` with the
``bloom download --what ...`` fix when a file is absent (test-plan E7).

Groups (``bloom download --what``)::

    model      models/hyacinth/{model.safetensors, config.json}
    genes      references/<ref>/genes.{safetensors,parquet}
    peaks      references/<ref>/peaks.{safetensors,parquet,bed.gz}
    orthologs  references/<ref>/orthologs_human_mouse_one2one.tsv
    panels     references/<ref>/panels/*.txt
    examples   examples/multiome_20/*  (20-cell RNA + ATAC fixture + reference_embeddings.{npz,json})

Sources: :class:`UrlSource` (default: the public bucket ``s3://bloom-biorxiv``, anonymous read at
:data:`DEFAULT_ASSET_URL`; any other ``https://`` mirror of the layout works) and :class:`FileSource`
(``file://`` staging root, the maintainers' local copy of the bucket layout).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Literal
from urllib.parse import quote, urlparse

import requests

__all__ = [
    "ASSET_KINDS",
    "DEFAULT_ASSET_URL",
    "GROUPS",
    "MANIFEST_SHA256",
    "AssetError",
    "AssetFile",
    "AssetMissingError",
    "AssetSource",
    "AssetUnavailableError",
    "AssetVerificationError",
    "DownloadResult",
    "FileSource",
    "GroupStatus",
    "Manifest",
    "UrlSource",
    "VerifyResult",
    "bloom_home",
    "default_source",
    "download",
    "group_status",
    "load_manifest",
    "locate",
    "manifest",
    "sha256_file",
    "source_from_spec",
    "verify",
]

# ----------------------------------------------------------------------------------------------
# constants

#: sha256 of the packaged ``bloom/data/manifest.json`` (= the bucket's ``MANIFEST.json``; 15 files).
#: Update together with the file (tests/test_assets.py checks the pin and the file count).
MANIFEST_SHA256 = "38f0274496524a7da4e5015eedf7dc21f87f987243a40dc4a75395b253d99d2e"

#: ``manifest_version`` this package understands.
SUPPORTED_MANIFEST_VERSION = 1

#: Public bucket ``s3://bloom-biorxiv`` (us-west-2, anonymous ``GetObject``) mirroring the release layout
#: (docs/assets.md section 2). Always the *regional* hostname: the region-less one answers HTTP 307.
DEFAULT_ASSET_URL = "https://bloom-biorxiv.s3.us-west-2.amazonaws.com/"

#: Environment variable that overrides the default source (``bucket``, ``url:<base>``,
#: ``file://<root>``; see :func:`source_from_spec`).
SOURCE_ENV = "BLOOM_ASSET_SOURCE"

GroupName = Literal["model", "genes", "peaks", "orthologs", "panels", "examples"]

#: Download groups in the order ``bloom download`` processes them.
GROUPS: tuple[str, ...] = ("model", "genes", "peaks", "orthologs", "panels", "examples")

#: Named files for :func:`locate` -> manifest path (release layout, relative to ``BLOOM_HOME``).
ASSET_KINDS: dict[str, str] = {
    "model": "models/hyacinth/model.safetensors",
    "config": "models/hyacinth/config.json",
    "genes": "references/mm10-evo2-v1/genes.safetensors",
    "genes_table": "references/mm10-evo2-v1/genes.parquet",
    "peaks": "references/mm10-evo2-v1/peaks.safetensors",
    "peaks_table": "references/mm10-evo2-v1/peaks.parquet",
    "peaks_bed": "references/mm10-evo2-v1/peaks.bed.gz",
    "orthologs": "references/mm10-evo2-v1/orthologs_human_mouse_one2one.tsv",
    "panels": "references/mm10-evo2-v1/panels",
    "examples": "examples/multiome_20",
    "reference_embeddings": "examples/multiome_20/reference_embeddings.npz",
}

#: Human-readable names used in error messages (E7: "peak table not downloaded; ...").
_KIND_LABEL: dict[str, str] = {
    "model": "model weights",
    "config": "model config",
    "genes": "gene table",
    "genes_table": "gene table (parquet)",
    "peaks": "peak table",
    "peaks_table": "peak table (parquet)",
    "peaks_bed": "peak table (bed)",
    "orthologs": "ortholog table",
    "panels": "panels",
    "examples": "example cells",
    "reference_embeddings": "reference embeddings",
}

#: Read block size for hashing.
_CHUNK = 8 * 1024 * 1024
#: Streaming block size for HTTP downloads (bytes on disk advance in these steps).
_STREAM_CHUNK = 1024 * 1024
#: Consecutive transfer attempts *without progress* per file before giving up on a flaky connection.
#: An attempt that adds bytes to the ``.part`` file resets the counter, so a 5.4 GB file survives many
#: interruptions as long as each reconnect moves it forward (HTTP ``Range`` resume).
_MAX_ATTEMPTS = 4
_HTTP_TIMEOUT = (15, 120)  # (connect, read) seconds


# ----------------------------------------------------------------------------------------------
# errors


class AssetError(RuntimeError):
    """Base class for asset errors."""


class AssetMissingError(AssetError, FileNotFoundError):
    """A required asset is not in ``BLOOM_HOME`` (test-plan E7)."""


class AssetVerificationError(AssetError):
    """A downloaded or cached file does not match the manifest size / sha256."""


class AssetUnavailableError(AssetError):
    """The selected source does not carry the requested file(s) (e.g. weights not released yet)."""


# ----------------------------------------------------------------------------------------------
# cache directory


def bloom_home() -> Path:
    """Asset cache directory: ``$BLOOM_HOME`` or ``~/.cache/bloom`` (docs/assets.md section 2)."""
    return Path(os.environ.get("BLOOM_HOME", Path.home() / ".cache" / "bloom")).expanduser()


# ----------------------------------------------------------------------------------------------
# manifest


@dataclass(frozen=True, slots=True)
class AssetFile:
    """One manifest entry.

    Attributes:
        path: Release-layout path relative to ``BLOOM_HOME`` (``references/.../genes.parquet``).
        size_bytes: Exact file size.
        sha256: Hex digest of the file content.
        manifest_group: ``group`` field of the manifest (``model`` / ``reference`` / ``example`` /
            ``fixture``). The download group (:attr:`group`) is finer-grained.
        version: Asset version id (``hyacinth``, ``mm10-evo2-v1``, ``examples-v1``).
        license: License tag from the manifest.
    """

    path: str
    size_bytes: int
    sha256: str
    manifest_group: str
    version: str
    license: str

    @property
    def group(self) -> str:
        """Download group (one of :data:`GROUPS`), derived from the manifest group and path."""
        return _group_of(self.manifest_group, self.path)

    @property
    def name(self) -> str:
        """File name (last path component)."""
        return self.path.rsplit("/", 1)[-1]


def _group_of(manifest_group: str, path: str) -> str:
    if manifest_group == "model":
        return "model"
    if manifest_group == "example":
        return "examples"
    if manifest_group == "reference":
        parts = path.split("/")
        name = parts[-1]
        if "panels" in parts[:-1]:
            return "panels"
        if "orthologs" in parts[:-1] or name.startswith("orthologs"):
            return "orthologs"
        if name.startswith("genes."):
            return "genes"
        if name.startswith("peaks."):
            return "peaks"
    raise ValueError(f"cannot assign manifest entry {path!r} (group {manifest_group!r}) to a group")


@dataclass(frozen=True)
class Manifest:
    """Parsed ``MANIFEST.json``.

    Attributes:
        files: Entries in manifest order.
        manifest_version: Schema version (must equal :data:`SUPPORTED_MANIFEST_VERSION`).
        model_id: Released model id (``hyacinth``).
        reference_id: Reference table id (``mm10-evo2-v1``).
        examples_version: Example/fixture set id.
        created: ISO timestamp of the manifest.
        sha256: sha256 of the manifest bytes that were parsed.
        raw: The full parsed JSON document (``versions``, ``licenses``, ...).
    """

    files: tuple[AssetFile, ...]
    manifest_version: int
    model_id: str
    reference_id: str
    examples_version: str
    created: str
    sha256: str
    raw: dict

    def by_path(self) -> dict[str, AssetFile]:
        """Map ``path -> entry``."""
        return {f.path: f for f in self.files}

    def group(self, name: str) -> tuple[AssetFile, ...]:
        """Entries of one download group (``name`` in :data:`GROUPS`)."""
        if name not in GROUPS:
            raise ValueError(f"unknown asset group {name!r}; choose from {', '.join(GROUPS)}")
        return tuple(f for f in self.files if f.group == name)

    def select(self, what: str | Iterable[str] | None) -> tuple[AssetFile, ...]:
        """Entries for a group list (``"model,genes"`` or iterable); ``None`` = all groups."""
        return tuple(f for g in parse_groups(what) for f in self.group(g))

    @property
    def total_bytes(self) -> int:
        """Sum of ``size_bytes`` over all entries."""
        return sum(f.size_bytes for f in self.files)


def parse_groups(what: str | Iterable[str] | None) -> tuple[str, ...]:
    """Normalise a ``--what`` value to a tuple of group names (order of :data:`GROUPS`).

    Accepts ``None`` / ``"all"`` (every group), a comma-separated string or an iterable.
    """
    if what is None:
        return GROUPS
    items = [s.strip() for s in what.split(",")] if isinstance(what, str) else list(what)
    items = [s for s in items if s]
    if not items or "all" in items:
        return GROUPS
    unknown = [s for s in items if s not in GROUPS]
    if unknown:
        raise ValueError(
            f"unknown asset group(s) {', '.join(map(repr, unknown))}; choose from {', '.join(GROUPS)}"
        )
    return tuple(g for g in GROUPS if g in items)


def load_manifest(source: str | Path | bytes, *, expected_sha256: str | None = None) -> Manifest:
    """Parse a manifest from a path or bytes and validate its schema version.

    Args:
        source: Path to a ``MANIFEST.json`` or its raw bytes.
        expected_sha256: If given, the bytes must hash to this value.

    Raises:
        AssetVerificationError: sha256 mismatch or unsupported ``manifest_version``.
    """
    data = source if isinstance(source, bytes) else Path(source).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise AssetVerificationError(
            f"manifest sha256 {digest[:16]}... does not match the pinned {expected_sha256[:16]}..."
        )
    doc = json.loads(data)
    version = int(doc.get("manifest_version", -1))
    if version != SUPPORTED_MANIFEST_VERSION:
        raise AssetVerificationError(
            f"manifest_version {version} is not supported by this package "
            f"(expects {SUPPORTED_MANIFEST_VERSION}); upgrade bloom-biorxiv"
        )
    files = tuple(
        AssetFile(
            path=str(f["path"]),
            size_bytes=int(f["size_bytes"]),
            sha256=str(f["sha256"]).lower(),
            manifest_group=str(f["group"]),
            version=str(f.get("version", "")),
            license=str(f.get("license", "")),
        )
        for f in doc["files"]
    )
    return Manifest(
        files=files,
        manifest_version=version,
        model_id=str(doc.get("model_id", "")),
        reference_id=str(doc.get("reference_id", "")),
        examples_version=str(doc.get("examples_version", "")),
        created=str(doc.get("created", "")),
        sha256=digest,
        raw=doc,
    )


_MANIFEST: Manifest | None = None


def manifest() -> Manifest:
    """The packaged manifest, verified against :data:`MANIFEST_SHA256` (cached after first call)."""
    global _MANIFEST  # module-level cache
    if _MANIFEST is None:
        data = resources.files("bloom.data").joinpath("manifest.json").read_bytes()
        _MANIFEST = load_manifest(data, expected_sha256=MANIFEST_SHA256)
    return _MANIFEST


# ----------------------------------------------------------------------------------------------
# hashing / verification


def sha256_file(path: Path, *, chunk: int = _CHUNK) -> str:
    """sha256 hex digest of a file, streamed in ``chunk`` bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


VerifyStatus = Literal["ok", "missing", "size_mismatch", "sha_mismatch", "partial", "unverified"]


@dataclass(frozen=True, slots=True)
class VerifyResult:
    """Outcome of checking one cached file.

    Attributes:
        asset: Manifest entry.
        path: Absolute path checked.
        status: ``ok`` | ``missing`` | ``size_mismatch`` | ``sha_mismatch`` | ``partial``
            (a ``.part`` file exists but no final file) | ``unverified`` (size matches, hash not
            computed because ``checksum=False``).
        size_bytes: Observed size (0 when missing).
    """

    asset: AssetFile
    path: Path
    status: VerifyStatus
    size_bytes: int

    @property
    def ok(self) -> bool:
        """True for ``ok`` and ``unverified`` (present with the right size)."""
        return self.status in ("ok", "unverified")


def _part_path(path: Path) -> Path:
    return path.with_name(path.name + ".part")


def _check_file(asset: AssetFile, path: Path, *, checksum: bool) -> VerifyResult:
    if not path.exists():
        part = _part_path(path)
        if part.exists():
            return VerifyResult(asset, path, "partial", part.stat().st_size)
        return VerifyResult(asset, path, "missing", 0)
    size = path.stat().st_size
    if size != asset.size_bytes:
        return VerifyResult(asset, path, "size_mismatch", size)
    if not checksum:
        return VerifyResult(asset, path, "unverified", size)
    if sha256_file(path) != asset.sha256:
        return VerifyResult(asset, path, "sha_mismatch", size)
    return VerifyResult(asset, path, "ok", size)


def verify(
    what: str | Iterable[str] | None = None,
    dest: Path | None = None,
    *,
    checksum: bool = True,
    files: Sequence[AssetFile] | None = None,
) -> list[VerifyResult]:
    """Check the cached files of the given groups against the manifest.

    Args:
        what: Groups to check (``"model,genes"``; ``None`` = all).
        dest: Cache root (default :func:`bloom_home`).
        checksum: Compute sha256 (slow for ``peaks.safetensors``, 5.4 GB). With ``False`` only
            the size is compared and present files report ``unverified``.
        files: Explicit entries to check instead of ``what``.

    Returns:
        One :class:`VerifyResult` per manifest entry, in manifest order.
    """
    root = Path(dest) if dest is not None else bloom_home()
    entries = tuple(files) if files is not None else manifest().select(what)
    return [_check_file(a, root / a.path, checksum=checksum) for a in entries]


# ----------------------------------------------------------------------------------------------
# sources

#: ``progress(asset, bytes_done, bytes_total)`` callback type.
ProgressFn = Callable[[AssetFile, int, int], None]


class AssetSource(ABC):
    """Where asset bytes come from.

    Implementations resolve a manifest entry to a URL (HTTP sources) or copy/link it (file
    source). :meth:`available` tells :func:`download` which files the source carries so that groups
    not yet published (weights, full reference tables) fail with a clear message.
    """

    #: Short name used in messages (``url:https://...``, ``file:///...``).
    name: str = "source"

    @abstractmethod
    def available(self, asset: AssetFile) -> bool:
        """Whether this source can deliver ``asset``."""

    @abstractmethod
    def fetch(
        self,
        asset: AssetFile,
        part: Path,
        *,
        resume: bool = True,
        progress: ProgressFn | None = None,
    ) -> None:
        """Write the bytes of ``asset`` to ``part`` (appending when resuming)."""

    def describe_unavailable(self, assets: Sequence[AssetFile]) -> str:
        """Message for files this source does not carry."""
        names = ", ".join(a.name for a in assets)
        return f"{names}: not available from {self.name}"


class ChunkedTransferError(AssetError):
    """A transfer ended before ``size_bytes`` were received; the caller may resume."""


class _HttpSource(AssetSource):
    """Shared HTTP download logic: streaming, ``Range`` resume, retries."""

    def __init__(self, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()

    @abstractmethod
    def url_for(self, asset: AssetFile) -> str | None:
        """Download URL for ``asset`` or ``None`` when the source does not have it."""

    def headers_for(self, asset: AssetFile) -> dict[str, str]:
        """Extra request headers (auth, accept)."""
        return {}

    def available(self, asset: AssetFile) -> bool:
        return self.url_for(asset) is not None

    def fetch(
        self,
        asset: AssetFile,
        part: Path,
        *,
        resume: bool = True,
        progress: ProgressFn | None = None,
    ) -> None:
        url = self.url_for(asset)
        if url is None:
            raise AssetUnavailableError(self.describe_unavailable([asset]))
        part.parent.mkdir(parents=True, exist_ok=True)
        if not resume and part.exists():
            part.unlink()
        last_error: Exception | None = None
        stalled = 0  # attempts in a row that added no bytes
        while True:
            before = part.stat().st_size if part.exists() else 0
            try:
                self._fetch_once(asset, url, part, progress)
                return
            except (
                requests.ConnectionError,
                requests.Timeout,
                requests.exceptions.ChunkedEncodingError,
                ChunkedTransferError,
            ) as exc:
                last_error = exc
                after = part.stat().st_size if part.exists() else 0
                stalled = 0 if after > before else stalled + 1
                if stalled >= _MAX_ATTEMPTS:
                    break
                time.sleep(min(2**stalled, 10))
        raise AssetError(
            f"{asset.name}: download failed after {_MAX_ATTEMPTS} attempts without progress "
            f"({last_error}); re-run `bloom download` to resume from the .part file"
        ) from last_error

    def _fetch_once(
        self, asset: AssetFile, url: str, part: Path, progress: ProgressFn | None
    ) -> None:
        have = part.stat().st_size if part.exists() else 0
        if have > asset.size_bytes:
            part.unlink()
            have = 0
        if have == asset.size_bytes:
            return  # complete .part from an earlier run; caller verifies and renames
        headers = dict(self.headers_for(asset))
        if have:
            headers["Range"] = f"bytes={have}-"
        with self._session.get(
            url, headers=headers, stream=True, timeout=_HTTP_TIMEOUT, allow_redirects=True
        ) as resp:
            if resp.status_code == 416:  # range not satisfiable: server disagrees on size
                part.unlink(missing_ok=True)
                have = 0
                headers.pop("Range", None)
                return self._fetch_once(asset, url, part, progress)
            if resp.status_code == 404:
                raise AssetUnavailableError(f"{asset.name}: HTTP 404 at {url}")
            resp.raise_for_status()
            if have and resp.status_code == 206:
                mode = "ab"
            else:  # 200: the server ignored Range (or nothing to resume) -> start over
                mode = "wb"
                have = 0
            done = have
            with open(part, mode) as fh:
                for block in resp.iter_content(chunk_size=_STREAM_CHUNK):
                    if block:
                        fh.write(block)
                        done += len(block)
                        if progress is not None:
                            progress(asset, done, asset.size_bytes)
        if done < asset.size_bytes:
            # connection closed early: let the retry loop resume from ``done``
            raise ChunkedTransferError(f"{asset.name}: got {done} of {asset.size_bytes} bytes")


class UrlSource(_HttpSource):
    """Any HTTP(S) base that mirrors the release layout (default: the public bucket).

    ``url_for`` = ``<base_url>/<manifest path>``. Every manifest entry is assumed available; a
    missing object surfaces as HTTP 404 -> :class:`AssetUnavailableError`. No credentials are sent.

    Args:
        base_url: e.g. :data:`DEFAULT_ASSET_URL` (``https://bloom-biorxiv.s3.us-west-2.amazonaws.com/``).
        session: Optional :class:`requests.Session` (tests inject one).
    """

    def __init__(
        self, base_url: str = DEFAULT_ASSET_URL, session: requests.Session | None = None
    ) -> None:
        super().__init__(session)
        self.base_url = base_url.rstrip("/")
        self.name = f"url:{self.base_url}"

    def url_for(self, asset: AssetFile) -> str | None:
        return f"{self.base_url}/{quote(asset.path)}"

    def describe_unavailable(self, assets: Sequence[AssetFile]) -> str:
        names = ", ".join(a.name for a in assets)
        return (
            f"{len(assets)} file(s) not found at {self.base_url} (HTTP 404): {names}. "
            "Check `bloom info` for the manifest version this package expects; the public bucket "
            "is listed in docs/assets.md section 2."
        )


class FileSource(AssetSource):
    """A local directory in the release layout (e.g. the maintainers' staging root).

    Args:
        root: Directory containing ``models/``, ``references/``, ``examples/``.
        link: Create symlinks instead of copying (default). Linked files are size-checked but not
            hashed by :func:`download` (5.4 GB peak table); ``bloom info --verify`` hashes them.
    """

    def __init__(self, root: str | Path, *, link: bool = True) -> None:
        self.root = Path(root).expanduser()
        self.link = link
        self.name = f"file://{self.root}"

    def source_path(self, asset: AssetFile) -> Path:
        """Path of ``asset`` under the root."""
        return self.root / asset.path

    def available(self, asset: AssetFile) -> bool:
        return self.source_path(asset).is_file()

    def fetch(
        self,
        asset: AssetFile,
        part: Path,
        *,
        resume: bool = True,
        progress: ProgressFn | None = None,
    ) -> None:
        src = self.source_path(asset)
        if not src.is_file():
            raise AssetUnavailableError(self.describe_unavailable([asset]))
        part.parent.mkdir(parents=True, exist_ok=True)
        part.unlink(missing_ok=True)
        if self.link:
            part.symlink_to(src.resolve())
        else:
            shutil.copyfile(src, part)
        if progress is not None:
            progress(asset, asset.size_bytes, asset.size_bytes)


def source_from_spec(spec: str | None) -> AssetSource:
    """Build a source from a CLI / environment spec.

    ``None`` -> ``$BLOOM_ASSET_SOURCE`` or ``bucket``; ``bucket`` -> :class:`UrlSource` at
    :data:`DEFAULT_ASSET_URL` (the public bucket, anonymous); ``url:<base>`` / ``https://...`` ->
    :class:`UrlSource`; ``file://<root>`` or an existing directory -> :class:`FileSource` (append
    ``?copy`` to copy instead of symlink).
    """
    if spec is None:
        spec = os.environ.get(SOURCE_ENV) or "bucket"
    spec = spec.strip()
    if spec in ("bucket", "default"):
        return UrlSource(DEFAULT_ASSET_URL)
    if spec.startswith("url:"):
        return UrlSource(spec[len("url:") :])
    if spec.startswith(("http://", "https://")):
        return UrlSource(spec)
    if spec.startswith("file://"):
        parsed = urlparse(spec)
        return FileSource(parsed.path, link=parsed.query != "copy")
    if Path(spec).expanduser().is_dir():
        return FileSource(spec)
    raise ValueError(
        f"unrecognised asset source {spec!r}; use bucket, url:<base>, or file://<root>"
    )


def default_source() -> AssetSource:
    """Source selected by ``$BLOOM_ASSET_SOURCE``, else the public bucket (:data:`DEFAULT_ASSET_URL`)."""
    return source_from_spec(None)


# ----------------------------------------------------------------------------------------------
# download


DownloadStatus = Literal["present", "downloaded", "refetched", "linked", "unavailable", "failed"]


@dataclass(frozen=True, slots=True)
class DownloadResult:
    """Outcome of :func:`download` for one file.

    Attributes:
        asset: Manifest entry.
        path: Final path under the cache root.
        status: ``present`` (verified, nothing done) | ``downloaded`` | ``refetched`` (cached copy
            was corrupt or ``force``) | ``linked`` (symlink from a :class:`FileSource`) |
            ``unavailable`` (source lacks the file) | ``failed`` (verification after transfer).
        message: Detail for ``unavailable`` / ``failed``.
    """

    asset: AssetFile
    path: Path
    status: DownloadStatus
    message: str = ""

    @property
    def ok(self) -> bool:
        """True when the file is in place."""
        return self.status in ("present", "downloaded", "refetched", "linked")


def _finalize(asset: AssetFile, part: Path, final: Path, *, checksum: bool) -> None:
    """Verify ``part`` against the manifest and atomically rename it to ``final``."""
    size = part.stat().st_size
    if size != asset.size_bytes:
        part.unlink(missing_ok=True)
        raise AssetVerificationError(
            f"{asset.name}: size {size} != manifest {asset.size_bytes} after download"
        )
    if checksum:
        digest = sha256_file(part)
        if digest != asset.sha256:
            part.unlink(missing_ok=True)
            raise AssetVerificationError(
                f"{asset.name}: sha256 {digest[:16]}... != manifest {asset.sha256[:16]}... "
                "after download (corrupt transfer or wrong source)"
            )
    os.replace(part, final)


def download(
    what: str | Iterable[str] | None = None,
    dest: Path | None = None,
    *,
    force: bool = False,
    source: AssetSource | str | None = None,
    verify_existing: bool = True,
    files: Sequence[AssetFile] | None = None,
    progress: ProgressFn | None = None,
    log: Callable[[str], None] | None = None,
) -> list[DownloadResult]:
    """Fetch and verify asset groups into the cache (idempotent).

    For each manifest entry of the requested groups: a cached file with the right size and
    sha256 is kept; a missing, truncated or tampered file is (re-)fetched (test-plan E5);
    a partial ``<name>.part`` is resumed with an HTTP ``Range`` request; the completed file is
    verified and moved into place with an atomic rename. Files the source does not carry are
    reported as ``unavailable`` (never raised per file, so one call can report a whole group).

    Args:
        what: Groups (``"model,genes"``, iterable, or ``None`` for all of :data:`GROUPS`).
        dest: Cache root (default :func:`bloom_home`).
        force: Re-fetch even when the cached copy verifies.
        source: An :class:`AssetSource`, a spec string for :func:`source_from_spec`, or ``None``
            for :func:`default_source`.
        verify_existing: Hash cached files before deciding they are present. ``False`` compares
            sizes only (fast path for ``bloom info``-style checks).
        files: Explicit manifest entries instead of ``what``.
        progress: Optional ``progress(asset, bytes_done, bytes_total)`` callback.
        log: Optional ``log(message)`` callback for one line per file.

    Returns:
        One :class:`DownloadResult` per entry, in manifest order.
    """
    root = Path(dest) if dest is not None else bloom_home()
    src = source if isinstance(source, AssetSource) else source_from_spec(source)
    entries = tuple(files) if files is not None else manifest().select(what)
    results: list[DownloadResult] = []
    for asset in entries:
        final = root / asset.path
        part = _part_path(final)
        was_present = final.exists()
        if was_present and not force:
            hash_existing = verify_existing and not final.is_symlink()
            check = _check_file(asset, final, checksum=hash_existing)
            if check.ok:
                results.append(DownloadResult(asset, final, "present"))
                if log:
                    log(f"present      {asset.path} ({asset.size_bytes:,} B)")
                continue
            if log:
                log(f"{check.status:<12} {asset.path}: re-fetching")
        if not src.available(asset):
            msg = src.describe_unavailable([asset])
            results.append(DownloadResult(asset, final, "unavailable", msg))
            if log:
                log(f"unavailable  {asset.path}")
            continue
        try:
            src.fetch(asset, part, resume=not force, progress=progress)
            linked = part.is_symlink()
            _finalize(asset, part, final, checksum=not linked)
        except AssetError as exc:
            results.append(DownloadResult(asset, final, "failed", str(exc)))
            if log:
                log(f"FAILED       {asset.path}: {exc}")
            continue
        status: DownloadStatus = (
            "linked" if linked else ("refetched" if was_present else "downloaded")
        )
        results.append(DownloadResult(asset, final, status))
        if log:
            log(f"{status:<12} {asset.path} ({asset.size_bytes:,} B)")
    return results


# ----------------------------------------------------------------------------------------------
# locate / status


def locate(kind: str, dest: Path | None = None, *, required: bool = True) -> Path:
    """Path of a cached asset by kind (:data:`ASSET_KINDS`) or manifest path.

    Args:
        kind: One of :data:`ASSET_KINDS` (``"peaks"``, ``"genes_table"``, ...) or a manifest
            path (``"references/mm10-evo2-v1/peaks.parquet"``).
        dest: Cache root (default :func:`bloom_home`).
        required: Raise when the file is absent (default); ``False`` returns the path anyway.

    Raises:
        AssetMissingError: The file is not in the cache; the message names the
            ``bloom download --what <group>`` fix (test-plan E7).
    """
    root = Path(dest) if dest is not None else bloom_home()
    rel = ASSET_KINDS.get(kind, kind)
    path = root / rel
    if required and not path.exists():
        label = _KIND_LABEL.get(kind, kind)
        group = _group_for_path(rel)
        hint = f"run `bloom download --what {group}`" if group else "run `bloom download`"
        raise AssetMissingError(
            f"{label} not downloaded ({path}); {hint} (cache root: BLOOM_HOME={root})"
        )
    return path


def _group_for_path(rel: str) -> str | None:
    entry = manifest().by_path().get(rel)
    if entry is not None:
        return entry.group
    for group, kind_rel in ((g, ASSET_KINDS[g]) for g in GROUPS if g in ASSET_KINDS):
        if rel == kind_rel or rel.startswith(kind_rel + "/"):
            return group
    parts = rel.split("/")
    if parts[0] == "models":
        return "model"
    if parts[0] == "references" and len(parts) > 2:
        name = parts[-1]
        if "panels" in parts:
            return "panels"
        if name.startswith("orthologs") or "orthologs" in parts:
            return "orthologs"
        if name.startswith("genes"):
            return "genes"
        if name.startswith("peaks"):
            return "peaks"
    if parts[0] == "examples":
        return "examples"
    return None


@dataclass(frozen=True, slots=True)
class GroupStatus:
    """Summary of one download group in the cache (``bloom info``).

    Attributes:
        group: Group name.
        n_files: Manifest entries in the group.
        n_present: Entries present with the right size (and sha, when hashed).
        n_corrupt: Entries present but with wrong size / sha.
        n_partial: Entries with only a ``.part`` file.
        bytes_total: Manifest bytes of the group.
        bytes_present: Bytes of entries counted in ``n_present``.
        hashed: Whether sha256 was computed.
        results: Per-file results.
    """

    group: str
    n_files: int
    n_present: int
    n_corrupt: int
    n_partial: int
    bytes_total: int
    bytes_present: int
    hashed: bool
    results: tuple[VerifyResult, ...]

    @property
    def state(self) -> str:
        """``present`` | ``missing`` | ``corrupt`` | ``partial`` (some files present)."""
        if self.n_corrupt:
            return "corrupt"
        if self.n_present == self.n_files:
            return "present"
        if self.n_present == 0 and self.n_partial == 0:
            return "missing"
        return "partial"


def group_status(dest: Path | None = None, *, checksum: bool = False) -> list[GroupStatus]:
    """Per-group presence summary of the cache (sizes only unless ``checksum=True``)."""
    out: list[GroupStatus] = []
    for g in GROUPS:
        res = verify(g, dest, checksum=checksum)
        present = [r for r in res if r.ok]
        out.append(
            GroupStatus(
                group=g,
                n_files=len(res),
                n_present=len(present),
                n_corrupt=sum(r.status in ("size_mismatch", "sha_mismatch") for r in res),
                n_partial=sum(r.status == "partial" for r in res),
                bytes_total=sum(r.asset.size_bytes for r in res),
                bytes_present=sum(r.size_bytes for r in present),
                hashed=checksum,
                results=tuple(res),
            )
        )
    return out


def iter_group_files(what: str | Iterable[str] | None = None) -> Iterator[AssetFile]:
    """Iterate manifest entries of the given groups."""
    yield from manifest().select(what)
