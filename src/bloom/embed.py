"""``embed``: the predict loop (``PreparedDataset`` -> strategy views -> batches -> ``h_cell`` -> AnnData).

docs/architecture.md sections 3.2, 5.2 and 5.4; output schema in docs/data-schema.md section 4.

For every modality present in the dataset (``rna`` -> gene tokens, ``atac`` -> peak tokens), each
cell's :class:`~bloom.tokens.CellFeatures` is turned into views by the modality's
:class:`~bloom.strategies.ViewStrategy` (seeded per cell with :func:`bloom.strategies.cell_rng`),
the views are streamed into fixed-size batches independent of how many views a cell produced,
encoded with :meth:`bloom.model.Bloom.encode` under ``torch.inference_mode`` and averaged per cell
in a float64 accumulator (then cast to float32), matching ``evaluate.py``'s ``EvalWriter``. Cells
without detected features are **kept** as NaN rows with a warning. The 5.4 GB peak table is loaded
onto the model lazily, only when an ``atac`` modality is embedded (read into RAM on CPU, moved
once on CUDA; architecture.md section 5.1, issue #43). With both modalities one AnnData carries
``X_bloom_rna`` and ``X_bloom_atac`` for the aligned cells; :func:`embed_unmatched` embeds the
cells present in one file only (``--keep-unmatched``).

Numerics: fp32 by default; ``precision="bf16"`` wraps ``encode`` in ``torch.autocast``. TF32 on
CUDA is left at the torch defaults (``torch.backends.cuda.matmul.allow_tf32``, off) and the fused
CUDA fast path is disabled inside ``Bloom.encode`` (issue #44), so ``--device cuda`` reproduces
the CPU embeddings to ~1e-5 (gate G7); the tolerance policy is in docs/test-plan.md section 5. On CPU the attention of the first layer
materialises an ``L x L`` matrix per head, so the default batch size is 1 there (L = 8192 at
batch 8 needs ~40 GB RSS; docs/validation.md (hardware table)).
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from anndata import AnnData

from bloom import assets
from bloom.model import FASTPATH_ENABLED, Bloom, load_pretrained, resolve_device
from bloom.prepare import SCHEMA_VERSION, PreparedDataset
from bloom.reference import ReferenceTables
from bloom.strategies import ViewStrategy, build, cell_rng
from bloom.tokens import TokenSet, collate

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "EmbedResult",
    "ModalityStats",
    "default_strategies",
    "embed",
    "embed_unmatched",
    "ensure_peak_table",
    "git_sha",
    "load_model",
    "obsm_key",
]

#: Default ``batch_size`` per device type (docs/architecture.md section 5.4, docs/validation.md (hardware table)).
DEFAULT_BATCH_SIZE: dict[str, int] = {"cpu": 1, "cuda": 8, "mps": 4}
PRECISIONS = ("fp32", "bf16")
_MODALITY_KIND = {"rna": "gene", "atac": "peak"}
_PAPER_DEFAULTS = {
    "rna": {"strategy": "random", "lmax": 8192, "draws": 4},
    "atac": {"strategy": "random", "lmax": 8192, "draws": 16},
}


def obsm_key(modality: str, views: bool = False) -> str:
    """``obsm`` key of a modality: ``X_bloom_rna`` / ``X_bloom_atac`` (+ ``_views``)."""
    return f"X_bloom_{modality}" + ("_views" if views else "")


# --------------------------------------------------------------------------------------------
# model loading
# --------------------------------------------------------------------------------------------
def git_sha(start: str | Path | None = None) -> str | None:
    """Short git sha of the checkout that contains ``start`` (default: this package), else ``None``.

    Only meaningful for editable installs; a wheel install returns ``None``.
    """
    d = Path(start) if start is not None else Path(__file__).resolve().parent
    try:
        r = subprocess.run(
            ["git", "-C", str(d), "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None


def _asset_hint(exc: Exception) -> str:
    return (
        f"{exc}\nFetch the model weights and the gene table from the public bucket with "
        "`bloom download --what model,genes` (597 MB into BLOOM_HOME; no credentials needed), or set "
        "BLOOM_HOME to a tree that already has them."
    )


def load_model(
    model_id: str = "hyacinth",
    device: str | torch.device = "auto",
    *,
    model_dir: str | Path | None = None,
    reference_dir: str | Path | None = None,
    peaks: bool = False,
) -> Bloom:
    """Load the released model with its gene table and attach the reference tables.

    Args:
        model_id: Released model id (only ``"hyacinth"`` exists).
        device: ``"auto"`` (cuda -> mps -> cpu), ``"cpu"``, ``"cuda"``, ``"mps"``.
        model_dir: Directory with ``config.json`` + ``model.safetensors``; default
            ``$BLOOM_HOME/models/<model_id>`` (``bloom.assets.locate``).
        reference_dir: Directory with ``genes.{parquet,safetensors}`` (+ peaks, orthologs);
            default ``$BLOOM_HOME/references/<reference_id>``.
        peaks: Also load the 5.4 GB peak table now. Not needed for :func:`embed`, which loads it
            lazily when an ``atac`` modality is present (:func:`ensure_peak_table`).

    Returns:
        The :class:`~bloom.model.Bloom` in ``eval()`` mode with ``model.reference`` set to the
        :class:`~bloom.reference.ReferenceTables`.

    Raises:
        bloom.assets.AssetMissingError: when the weights or the gene table are not in the cache
            (message names the ``bloom download`` fix; test-plan E7).
    """
    if model_id != "hyacinth":
        raise ValueError(f"unknown model id {model_id!r}; the released model is 'hyacinth'")
    try:
        if model_dir is None:
            model_dir = assets.locate("config").parent
            assets.locate("model")
        else:
            model_dir = Path(model_dir)
            if not (model_dir / "config.json").is_file():
                raise FileNotFoundError(f"config.json not found in {model_dir}")
        ref = ReferenceTables.from_dir(reference_dir, require_peaks=peaks)
    except (assets.AssetMissingError, FileNotFoundError) as exc:
        raise assets.AssetMissingError(_asset_hint(exc)) from exc
    t0 = time.perf_counter()
    model = load_pretrained(model_dir, device=device, strict=True)
    model.load_gene_table(str(ref.paths["genes_safetensors"]), n_rows=ref.n_genes)
    if peaks:
        model.load_peak_table(str(ref.paths["peaks_safetensors"]), n_rows=ref.n_peaks)
    model.reference = ref  # plain attribute on the nn.Module (architecture.md section 3.2)
    model.model_dir = Path(model_dir)
    logger.info(
        "loaded %s from %s on %s in %.1f s",
        model_id,
        model_dir,
        model.device,
        time.perf_counter() - t0,
    )
    return model


# --------------------------------------------------------------------------------------------
# strategies
# --------------------------------------------------------------------------------------------
def default_strategies(
    dataset: PreparedDataset,
    model: Bloom | None = None,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, ViewStrategy]:
    """Paper-default strategy per modality present, from ``config.json`` ``strategy_defaults``.

    Args:
        dataset: Decides which modalities get a strategy.
        model: Its ``config.strategy_defaults`` (keys ``gene`` / ``peak``) override the built-in
            paper defaults (``rna: random 8192 x 4``, ``atac: random 8192 x 16``).
        overrides: Per-modality ``{strategy, lmax, draws, universe, panel}`` keys (CLI flags);
            ``None`` values are ignored. Changing ``strategy`` resets ``lmax``/``draws`` to that
            preset's defaults unless given explicitly.

    Returns:
        ``{"rna": ViewStrategy, ...}``; a :class:`~bloom.feature_lists.FeatureList` universe or
        panel still needs ``resolve()`` (done by :func:`embed`).
    """
    out: dict[str, ViewStrategy] = {}
    cfg_defaults = dict(getattr(getattr(model, "config", None), "strategy_defaults", {}) or {})
    for mod in dataset.modalities:
        block = dict(_PAPER_DEFAULTS[mod])
        block.update({k: v for k, v in cfg_defaults.get(_MODALITY_KIND[mod], {}).items()})
        ov = {k: v for k, v in (overrides or {}).get(mod, {}).items() if v is not None}
        if "strategy" in ov and ov["strategy"] != block["strategy"]:
            block = {"strategy": ov["strategy"]}  # preset defaults apply unless overridden
        block.update(ov)
        out[mod] = build(
            block["strategy"],
            lmax=block.get("lmax"),
            draws=block.get("draws"),
            universe=block.get("universe"),
            panel=block.get("panel"),
        )
    return out


# --------------------------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------------------------
@dataclass
class ModalityStats:
    """Per-modality bookkeeping of one :func:`embed` run (also written to ``uns["bloom"]``)."""

    modality: str
    n_cells: int
    n_views: int = 0
    n_empty: int = 0
    n_batches: int = 0
    wall_s: float = 0.0
    lmax_padded: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_cells": self.n_cells,
            "n_views": self.n_views,
            "n_empty_cells": self.n_empty,
            "n_batches": self.n_batches,
            "wall_s": round(self.wall_s, 3),
            "s_per_cell": round(self.wall_s / max(self.n_cells, 1), 4),
            "lmax_padded": self.lmax_padded,
        }


@dataclass
class EmbedResult:
    """Return value of :func:`_embed_modality`: accumulators turned into arrays."""

    mean: np.ndarray  # (n_cells, d_model) float32, NaN rows for empty cells
    n_views: np.ndarray  # (n_cells,) int64
    n_tokens: np.ndarray  # (n_cells,) float32 mean tokens per view, NaN for empty cells
    views: np.ndarray | None  # (n_cells, max_views, d_model) float32 or None
    stats: ModalityStats = field(default_factory=lambda: ModalityStats("", 0))


class _Progress:
    """Minimal stderr progress line (no extra dependency)."""

    def __init__(self, total: int, label: str, enabled: bool, every: float = 2.0) -> None:
        self.total, self.label, self.enabled, self.every = total, label, enabled, every
        self.t0 = time.perf_counter()
        self.last = 0.0
        self.done = 0

    def update(self, done: int, force: bool = False) -> None:
        self.done = done
        if not self.enabled:
            return
        now = time.perf_counter()
        if not force and now - self.last < self.every:
            return
        self.last = now
        el = now - self.t0
        rate = done / el if el > 0 else 0.0
        eta = (self.total - done) / rate if rate > 0 else float("nan")
        sys.stderr.write(
            f"\r{self.label}: {done}/{self.total} cells  {el:6.0f}s elapsed  "
            f"{rate:5.2f} cells/s  ETA {eta:5.0f}s"
        )
        sys.stderr.flush()

    def close(self) -> None:
        if self.enabled:
            self.update(self.done, force=True)
            sys.stderr.write("\n")
            sys.stderr.flush()


def _feature_reference(model: Bloom, dataset: PreparedDataset) -> Any:
    ref = getattr(model, "reference", None)
    if ref is None:
        raise ValueError(
            "the strategy has an unresolved feature list (universe/panel) but the model carries "
            "no `reference`; use bloom.load_model() or call strategy.resolve(ReferenceTables) first"
        )
    return ref


def ensure_peak_table(model: Bloom) -> bool:
    """Load the peak reference table onto ``model`` if it is not loaded yet (lazy, ATAC only).

    Uses ``model.reference.paths["peaks_safetensors"]`` (set by :func:`load_model`). On CPU the
    safetensors tensors are read into RAM (5.4 GB RSS; #43); on CUDA they are moved once (5.4 GB of VRAM).

    Returns:
        True when the table was loaded by this call.

    Raises:
        RuntimeError: no reference attached or the peak files are not downloaded.
    """
    if getattr(model, "has_peak_table", False):
        return False
    ref = getattr(model, "reference", None)
    path = ref.paths.get("peaks_safetensors") if ref is not None else None
    if path is None or not Path(path).is_file():
        raise RuntimeError(
            "peak table not downloaded / not loaded; run `bloom download --what peaks` (5.4 GB) "
            "or call model.load_peak_table(peaks.safetensors)"
        )
    t0 = time.perf_counter()
    model.load_peak_table(str(path), n_rows=ref.n_peaks)
    logger.info(
        "loaded peak table %s (%d rows) on %s in %.1f s",
        path,
        ref.n_peaks,
        model.device,
        time.perf_counter() - t0,
    )
    return True


def _check_tables(model: Bloom, modality: str) -> None:
    if modality == "rna" and not getattr(model, "has_gene_table", True):
        raise RuntimeError(
            "gene reference table not loaded on the model; use bloom.load_model() or "
            "model.load_gene_table(genes.safetensors) (`bloom download --what genes`)"
        )
    if modality == "atac" and not getattr(model, "has_peak_table", True):
        ensure_peak_table(model)


def _encode_fn(model: Bloom, device: torch.device, precision: str) -> Callable[..., torch.Tensor]:
    if precision == "fp32":
        return lambda b: model.encode(b.emb_idx, b.kind, b.lengths)
    dtype = torch.bfloat16

    def _bf16(b: Any) -> torch.Tensor:
        with torch.autocast(device_type=device.type, dtype=dtype):
            return model.encode(b.emb_idx, b.kind, b.lengths)

    return _bf16


def _embed_modality(
    model: Bloom,
    dataset: PreparedDataset,
    modality: str,
    strategy: ViewStrategy,
    *,
    batch_size: int,
    seed: int,
    device: torch.device,
    precision: str,
    keep_views: bool,
    progress: bool,
) -> EmbedResult:
    """Run one modality: views -> batches -> ``encode`` -> per-cell float64 running mean."""
    n = dataset.n_obs
    d = int(model.config.d_model)
    stats = ModalityStats(modality, n)
    acc = np.zeros((n, d), dtype=np.float64)  # (n_cells, d_model) running sum
    n_views = np.zeros(n, dtype=np.int64)  # (n_cells,)
    tok_sum = np.zeros(n, dtype=np.float64)  # (n_cells,) sum of view lengths
    per_view: dict[int, list[np.ndarray]] = {} if keep_views else None  # type: ignore[assignment]
    encode = _encode_fn(model, device, precision)
    queue: list[tuple[int, int, TokenSet]] = []  # (cell_index, view_index, tokens)
    bar = _Progress(n, f"embed {modality} [{strategy.name}, {device.type}]", progress)
    t0 = time.perf_counter()

    def flush() -> None:
        if not queue:
            return
        cells = np.fromiter((q[0] for q in queue), dtype=np.int64, count=len(queue))
        views = np.fromiter((q[1] for q in queue), dtype=np.int64, count=len(queue))
        batch = collate([q[2] for q in queue], cell_index=cells, view_index=views)
        stats.lmax_padded = max(stats.lmax_padded, batch.lmax)
        with torch.inference_mode():
            h = encode(batch.to(device))  # (B, d_model)
        h64 = h.detach().to(device="cpu", dtype=torch.float64).numpy()
        np.add.at(acc, cells, h64)
        np.add.at(n_views, cells, 1)
        np.add.at(tok_sum, cells, batch.lengths.numpy().astype(np.float64))
        if per_view is not None:
            for row, ci in enumerate(cells.tolist()):
                per_view.setdefault(ci, []).append(h64[row].astype(np.float32))
        stats.n_batches += 1
        stats.n_views += len(queue)
        queue.clear()

    for i in range(n):
        cell = dataset.cell_features(i, modality)  # type: ignore[arg-type]
        views = strategy.views(cell, cell_rng(seed, i))
        if not views:
            stats.n_empty += 1
        for j, ts in enumerate(views):
            queue.append((i, j, ts))
            if len(queue) >= batch_size:
                flush()
        if i % 8 == 7 or i == n - 1:
            bar.update(i + 1)
    flush()
    bar.close()
    stats.wall_s = time.perf_counter() - t0

    mean = np.full((n, d), np.nan, dtype=np.float32)
    has = n_views > 0
    mean[has] = (acc[has] / n_views[has, None]).astype(np.float32)
    n_tokens = np.full(n, np.nan, dtype=np.float32)
    n_tokens[has] = (tok_sum[has] / n_views[has]).astype(np.float32)
    views_arr = None
    if per_view is not None:
        max_v = int(n_views.max()) if n else 0
        views_arr = np.full((n, max_v, d), np.nan, dtype=np.float32)
        for ci, rows in per_view.items():
            views_arr[ci, : len(rows)] = np.stack(rows)
    if stats.n_empty:
        warnings.warn(
            f"{stats.n_empty} of {n} cells have no detected {_MODALITY_KIND[modality]} features "
            f"({modality}); their {obsm_key(modality)} rows are NaN",
            stacklevel=3,
        )
    return EmbedResult(mean=mean, n_views=n_views, n_tokens=n_tokens, views=views_arr, stats=stats)


def _resolve_batch_size(batch_size: int | None, device: torch.device) -> int:
    if batch_size is None:
        return DEFAULT_BATCH_SIZE.get(device.type, 1)
    if int(batch_size) < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    return int(batch_size)


def embed(
    model: Bloom,
    dataset: PreparedDataset,
    strategies: Mapping[str, ViewStrategy] | None = None,
    *,
    batch_size: int | None = None,
    seed: int = 0,
    device: str | torch.device | None = None,
    precision: str = "fp32",
    keep_views: bool = False,
    progress: bool = True,
) -> AnnData:
    """Embed every cell of ``dataset`` for every modality present (docs/architecture.md 4.2).

    Args:
        model: Loaded :class:`~bloom.model.Bloom` (``bloom.load_model``) with its gene table.
        dataset: :class:`~bloom.prepare.PreparedDataset` (``bloom.prepare`` or a bundle read
            with ``PreparedDataset.read``).
        strategies: ``{"rna": ViewStrategy, "atac": ...}``; missing modalities get the paper
            defaults (:func:`default_strategies`). A strategy carrying an unresolved feature list
            is resolved against ``model.reference``.
        batch_size: Views per ``encode`` call; ``None`` = :data:`DEFAULT_BATCH_SIZE` of the
            device (1 on CPU: the first-layer attention materialises ``L x L`` per head).
        seed: Base seed; cell ``i`` uses ``np.random.default_rng([seed, i])``.
        device: ``None`` keeps the model where it is; ``"auto"`` resolves cuda -> mps -> cpu and
            moves the model; explicit ``"cpu"``/``"cuda"``/``"mps"`` move it there.
        precision: ``"fp32"`` (paper; default) or ``"bf16"`` (``torch.autocast``; GPU).
        keep_views: Also store per-view embeddings in ``obsm["X_bloom_<mod>_views"]``
            ``(n_cells, max_views, 1024)``, NaN-padded for cells with fewer views.
        progress: Print a progress line to stderr.

    Returns:
        AnnData with ``obs`` = dataset ``obs`` + ``bloom_n_tokens_<mod>`` (float32, mean tokens
        per view) + ``bloom_n_views_<mod>`` (int64); ``obsm["X_bloom_<mod>"]`` float32
        ``(n_cells, 1024)`` with NaN rows for cells without features; empty ``X`` ``(n_cells, 0)``
        CSR; ``uns["bloom"]`` provenance (docs/data-schema.md section 4).

    Raises:
        ValueError: bad ``precision`` / ``batch_size``, strategies for absent modalities.
        RuntimeError: peak table not available for an ``atac`` modality.
    """
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    if not dataset.modalities:
        raise ValueError("dataset has no modality (prepare an RNA and/or ATAC file first)")

    if device is not None:
        dev = resolve_device(device)
        if dev != model.device:
            model.to(dev)
    dev = model.device
    bs = _resolve_batch_size(batch_size, dev)
    if dev.type == "cpu" and precision == "bf16":
        warnings.warn(
            "precision='bf16' on CPU is slow and untested; fp32 is the reference", stacklevel=2
        )
    model.eval()

    strats = default_strategies(dataset, model)
    strats.update(strategies or {})
    extra = set(strats) - set(dataset.modalities)
    if extra:
        raise ValueError(
            f"strategies given for absent modalities {sorted(extra)} (have {dataset.modalities})"
        )

    model_cfg = getattr(model, "config", None)
    model_ref_id = getattr(model_cfg, "reference_id", None)
    if model_ref_id and dataset.reference_id and model_ref_id != dataset.reference_id:
        warnings.warn(
            f"dataset was prepared against reference {dataset.reference_id!r} but the model expects "
            f"{model_ref_id!r}",
            stacklevel=2,
        )

    results: dict[str, EmbedResult] = {}
    for mod in dataset.modalities:
        strat = strats[mod]
        if strat.needs_resolve:
            strat.resolve(_feature_reference(model, dataset))
        _check_tables(model, mod)
        logger.info(
            "embedding %s: %d cells, %r, batch %d, %s, %s",
            mod,
            dataset.n_obs,
            strat,
            bs,
            dev,
            precision,
        )
        results[mod] = _embed_modality(
            model,
            dataset,
            mod,
            strat,
            batch_size=bs,
            seed=seed,
            device=dev,
            precision=precision,
            keep_views=keep_views,
            progress=progress,
        )

    return _assemble(
        model, dataset, strats, results, seed=seed, device=dev, precision=precision, batch_size=bs
    )


def embed_unmatched(
    model: Bloom,
    dataset: PreparedDataset,
    strategies: Mapping[str, ViewStrategy] | None = None,
    **kwargs: Any,
) -> dict[str, AnnData]:
    """Embed the cells present in one modality file only (``prepare(..., keep_unmatched=True)``).

    Args:
        model, dataset: as for :func:`embed`; ``dataset.unmatched`` holds the per-modality
            remainders.
        strategies: the same mapping given to :func:`embed`; the entry of each modality is used.
        **kwargs: forwarded to :func:`embed` (``batch_size``, ``seed``, ``device``, ...).

    Returns:
        ``{"rna": AnnData, "atac": AnnData}`` for the modalities with unmatched cells (the
        ``embeddings_rna_only.h5ad`` / ``embeddings_atac_only.h5ad`` outputs of the CLI). Cell
        indices (and hence ``random`` draws) are those of the remainder, not of the original file.
    """
    out: dict[str, AnnData] = {}
    for mod in dataset.unmatched:
        sub = dataset.single(mod, unmatched=True)  # type: ignore[arg-type]
        strat = {mod: strategies[mod]} if strategies and mod in strategies else None
        out[mod] = embed(model, sub, strat, **kwargs)
    return out


def _merge_obs(dataset: PreparedDataset) -> pd.DataFrame:
    """``obs`` of the output: RNA ``obs`` (else ATAC); extra ATAC columns added, conflicts ``_atac``."""
    first = dataset.modalities[0]
    obs = getattr(dataset, first).obs.copy()
    if first == "rna" and dataset.atac is not None:
        atac_obs = dataset.atac.obs
        if not atac_obs.index.equals(obs.index):
            atac_obs = atac_obs.reindex(obs.index)
        for col in atac_obs.columns:
            if col not in obs.columns:
                obs[col] = atac_obs[col].to_numpy()
            elif not obs[col].astype(object).equals(atac_obs[col].astype(object)):
                obs[f"{col}_atac"] = atac_obs[col].to_numpy()
    return obs


def _assemble(
    model: Bloom,
    dataset: PreparedDataset,
    strats: Mapping[str, ViewStrategy],
    results: Mapping[str, EmbedResult],
    *,
    seed: int,
    device: torch.device,
    precision: str,
    batch_size: int,
) -> AnnData:
    from bloom import __version__
    from bloom.prepare import _json_safe

    obs = _merge_obs(dataset)
    for mod, res in results.items():
        obs[f"bloom_n_tokens_{mod}"] = res.n_tokens.astype(np.float32)
        obs[f"bloom_n_views_{mod}"] = res.n_views.astype(np.int64)
    n = len(obs)
    out = AnnData(
        X=sp.csr_matrix((n, 0), dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=pd.Index([], dtype=str)),
    )
    for mod, res in results.items():
        out.obsm[obsm_key(mod)] = res.mean
        if res.views is not None:
            out.obsm[obsm_key(mod, views=True)] = res.views

    cfg = getattr(model, "config", None)
    weights = dict(getattr(cfg, "weights", {}) or {})
    modalities: dict[str, Any] = {}
    for mod, strat in strats.items():
        src = dict(dataset.source.get(mod, {}) or {})
        modalities[mod] = {
            "strategy": strat.params(),
            "seed": int(seed),
            "source": {k: v for k, v in src.items() if v is not None} or {"path": "<memory>"},
            "mapping": {
                k: dataset.report.get(mod, {}).get(k)
                for k in (
                    "n_var",
                    "n_mapped",
                    "mapping_rate",
                    "detected_mass_mapped",
                    "n_empty_cells",
                    "peak_match",
                    "min_overlap",
                    "exact_rate",
                    "n_ref_peaks_hit",
                )
            },
            "run": results[mod].stats.as_dict(),
        }
    cuda_tf32 = bool(torch.backends.cuda.matmul.allow_tf32) if device.type == "cuda" else None
    out.uns["bloom"] = _json_safe({
        "schema_version": SCHEMA_VERSION,
        "package_version": __version__,
        "git_sha": git_sha() or "unknown",
        "model_id": getattr(cfg, "model_id", "unknown"),
        "model_sha256": weights.get("sha256", "unknown"),
        "model_dir": str(getattr(model, "model_dir", "")) or None,
        "reference_id": dataset.reference_id or getattr(cfg, "reference_id", "unknown"),
        "species": dataset.species,
        "modalities": modalities,
        "alignment": dataset.alignment,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "precision": precision,
        "tf32_matmul": cuda_tf32,
        "fastpath": FASTPATH_ENABLED,
        "batch_size": int(batch_size),
        "torch_version": str(torch.__version__),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    })
    return out
