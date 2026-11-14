"""``Bloom``: the inference model as a plain ``torch.nn.Module``.

``encode`` reproduces ``Gardenia.get_cell_representation`` of ``alleninstitute/bloom@141f0d3``:
reference-table gather -> ``BloomFirstLayer`` -> ``PerceiverEncoder`` (cross-attention into 256
latents, self-attention, mean-pool, LayerNorm) -> ``h_cell (B, 1024)``.

Reference tables (``gene_*``, ``peak_*``) are **non-persistent buffers**: they move with
``model.to(device)`` but are not part of ``state_dict()``. The peak table (5.4 GB fp32) is loaded
lazily through ``load_peak_table`` and only when a peak strategy needs it; on CPU it is
read into RAM (5.4 GB; a file-backed mmap is issue #43).

``encode(..., return_internals=True)`` returns the structured internals of the forward pass
(``bloom.model.internals.EncodeOutput``; docs/internals.md) with an identical ``h_cell``.

Numerics: ``encode`` runs with torch's fused ``nn.MultiheadAttention`` / ``nn.TransformerEncoder``
CUDA fast path **disabled** (:func:`fastpath_disabled`, issue #44), so the same weights give the
same embeddings on CPU and GPU to ~1e-5 (G7). The fused kernel is a future optimisation that is
evaluated outside this repository.
"""

from __future__ import annotations

import contextlib
import logging
import math
import warnings
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors import safe_open
from torch import nn

from bloom.config import ModelConfig
from bloom.model.decoder import MultiDecoderHead
from bloom.model.embedding import FeatureEmbedding
from bloom.model.first_layer import BloomFirstLayer
from bloom.model.perceiver import PerceiverEncoder

if TYPE_CHECKING:
    from bloom.model.internals import EncodeOutput

__all__ = [
    "FASTPATH_ENABLED",
    "KIND_GENE",
    "KIND_PAD",
    "KIND_PEAK",
    "REFERENCE_BUFFERS",
    "Bloom",
    "MixedModalityViewError",
    "SeqHeadScaleMismatchError",
    "build_from_config",
    "check_seq_head_scale",
    "check_single_modality",
    "fastpath_disabled",
    "inverse_state_dict_remap",
    "load_pretrained",
    "remap_state_dict",
    "resolve_device",
]

logger = logging.getLogger(__name__)

#: ``encode`` always runs with torch's fused MHA / TransformerEncoder fast path off (issue #44);
#: recorded in ``uns["bloom"]["fastpath"]`` by ``bloom.embed``.
FASTPATH_ENABLED: bool = False


@contextlib.contextmanager
def fastpath_disabled() -> Iterator[None]:
    """Disable torch's fused ``nn.MultiheadAttention`` / ``nn.TransformerEncoder`` fast path.

    Scoped: the previous ``torch.backends.mha`` setting is restored on exit. With the fast path on,
    CUDA inference differs from CPU by up to ~1e-2 at L = 8192 (fused kernel numerics); with it
    off, CPU and GPU agree to ~1e-5 (docs/validation.md, gate G7). ``Bloom.encode`` and
    ``encode_with_internals`` wrap their forward pass in this context.
    """
    mha = torch.backends.mha
    previous = bool(mha.get_fastpath_enabled())
    mha.set_fastpath_enabled(False)
    try:
        yield
    finally:
        mha.set_fastpath_enabled(previous)


#: Token kinds (``TokenSet.kind`` / ``Batch.kind``, AGENTS.md section 5).
KIND_PAD: int = 0
KIND_GENE: int = 1
KIND_PEAK: int = 2

#: Non-persistent reference-table buffers registered on ``Bloom``.
REFERENCE_BUFFERS: tuple[str, ...] = (
    "gene_cseq",
    "gene_chr",
    "gene_bp",
    "peak_cseq",
    "peak_chr",
    "peak_bp",
)

_TABLE_FIELDS = ("cseq", "chr", "bp")


class SeqHeadScaleMismatchError(RuntimeError):
    """The installed ``bloom-arch`` uses a sequence-head scale different from ``config.json``."""


class MixedModalityViewError(ValueError):
    """A view (one batch row) mixes gene (``kind == 1``) and peak (``kind == 2``) tokens (issue #46).

    The model was trained on one modality per view; a mixed view runs mechanically (each token is
    resolved by its own ``kind``) but its embedding and its cross-modal attention logits are undefined,
    so ``Bloom.encode``, ``encode_with_internals`` and ``pair_logits`` refuse it. Embed the RNA and the
    ATAC file as two modalities instead (``bloom embed --rna ... --atac ...``).
    """


def check_single_modality(kind: torch.Tensor) -> None:
    """Raise :class:`MixedModalityViewError` if any row of ``kind (B, L)`` holds both genes and peaks.

    Pads (``kind == 0``) are ignored. Cost: two ``(B, L)`` comparisons.
    """
    has_gene = (kind == KIND_GENE).any(dim=-1)
    has_peak = (kind == KIND_PEAK).any(dim=-1)
    mixed = (has_gene & has_peak).reshape(-1)
    if bool(mixed.any()):
        rows = torch.nonzero(mixed).reshape(-1).tolist()
        n_gene = int((kind.reshape(-1, kind.shape[-1])[rows[0]] == KIND_GENE).sum())
        n_peak = int((kind.reshape(-1, kind.shape[-1])[rows[0]] == KIND_PEAK).sum())
        raise MixedModalityViewError(
            f"a view holds one modality (kind 1 = gene or kind 2 = peak); batch row(s) {rows[:8]} mix "
            f"both (row {rows[0]}: {n_gene} gene + {n_peak} peak tokens). Mixed gene + peak views were "
            "never seen in training and their embeddings / cross-modal attention logits are undefined "
            "(issue #46): embed RNA and ATAC as two modalities (`bloom embed --rna ... --atac ...`, one "
            "`TokenSet` per modality)."
        )


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """``"auto"`` -> ``cuda`` if available, else ``mps``, else ``cpu`` (architecture.md 6.4)."""
    if isinstance(device, torch.device):
        return device
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------------------------
# reference-table coercion
# --------------------------------------------------------------------------------------------
def _read_table(source: Any, prefix: str) -> dict[str, torch.Tensor]:
    """Return ``{"cseq", "chr", "bp"[, "emb_idx"]}`` CPU tensors from a path, mapping or object.

    Accepted sources: a ``.safetensors`` path (tensors ``cseq``/``chr``/``bp`` or
    ``<prefix>_cseq``/...; an optional ``emb_idx`` marks a row subset, as in
    a row-subset table such as the test fixtures build), a mapping with the same keys, or an object exposing them as
    attributes (e.g. ``bloom.reference.ReferenceTables``).
    """
    names = {f: (f, f"{prefix}_{f}") for f in (*_TABLE_FIELDS, "emb_idx")}
    out: dict[str, torch.Tensor] = {}
    if isinstance(source, (str, Path)):
        with safe_open(str(source), framework="pt", device="cpu") as f:
            keys = set(f.keys())
            for field, candidates in names.items():
                for k in candidates:
                    if k in keys:
                        out[field] = f.get_tensor(k)
                        break
    elif isinstance(source, Mapping):
        for field, candidates in names.items():
            for k in candidates:
                if k in source:
                    out[field] = torch.as_tensor(source[k])
                    break
    else:
        for field, candidates in names.items():
            for k in candidates:
                v = getattr(source, k, None)
                if v is not None:
                    out[field] = torch.as_tensor(v)
                    break
    missing = [f for f in _TABLE_FIELDS if f not in out]
    if missing:
        raise ValueError(
            f"{prefix} table source {source!r} lacks {missing}; expected tensors "
            f"{[f'{prefix}_{f}' for f in _TABLE_FIELDS]} or {list(_TABLE_FIELDS)}"
        )
    return out


def _expand_subset(table: dict[str, torch.Tensor], n_rows: int | None) -> dict[str, torch.Tensor]:
    """Scatter a row-subset table (with ``emb_idx``) into a full-size table of ``n_rows`` rows.

    Rows absent from the subset are zero (``cseq``), 0 (``chr``) and 0 (``bp``) and must never be
    referenced by a token; the caller is responsible for restricting ``emb_idx`` accordingly.
    """
    emb_idx = table.pop("emb_idx", None)
    if emb_idx is None:
        return table
    emb_idx = emb_idx.long()
    n = int(emb_idx.max().item()) + 1 if n_rows is None else int(n_rows)
    if n <= int(emb_idx.max().item()):
        raise ValueError(f"n_rows={n} too small for subset emb_idx max {int(emb_idx.max())}")
    full: dict[str, torch.Tensor] = {}
    for f in _TABLE_FIELDS:
        src = table[f]
        dst = torch.zeros((n, *src.shape[1:]), dtype=src.dtype)
        dst[emb_idx] = src
        full[f] = dst
    return full


# --------------------------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------------------------
class Bloom(nn.Module):
    """Bloom cell-embedding model (``hyacinth``): first layer + Perceiver encoder (+ decoder).

    Args:
        config: Architecture read from ``config.json`` (``ModelConfig.from_json``).

    Attributes:
        first_layer: ``BloomFirstLayer`` (wraps ``bloom_arch.BloomAttentionBlock``).
        encoder: ``PerceiverEncoder``.
        decoder: ``MultiDecoderHead`` (weights loaded; not used by ``encode``).
        feature_emb: ``FeatureEmbedding`` (query side; decoder only).
        gene_cseq, gene_chr, gene_bp: reference gene table buffers (``None`` until loaded);
            ``(N_gene, d_cseq)`` f32, ``(N_gene,)`` int8, ``(N_gene,)`` int32.
        peak_cseq, peak_chr, peak_bp: reference peak table buffers (lazy; ``None`` until loaded).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.first_layer = BloomFirstLayer(config)
        self.encoder = PerceiverEncoder(
            d_model=config.d_model,
            n_latents=config.n_latents,
            n_self_layers=config.n_transformer_layers,
            n_heads=config.n_heads,
            dropout=config.dropout,
        )
        self.decoder = MultiDecoderHead(config, hidden_dim=config.decoder_hidden_dim)
        self.feature_emb = FeatureEmbedding(config)
        for name in REFERENCE_BUFFERS:
            self.register_buffer(name, None, persistent=False)

    # ---- properties ------------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        """Device of the parameters (reference tables may live elsewhere)."""
        return next(self.parameters()).device

    @property
    def n_parameters(self) -> int:
        """``sum(p.numel() for p in parameters())`` (124,144,831 for ``hyacinth``)."""
        return sum(p.numel() for p in self.parameters())

    @property
    def has_gene_table(self) -> bool:
        """Whether the gene reference table is loaded."""
        return self.gene_cseq is not None

    @property
    def has_peak_table(self) -> bool:
        """Whether the peak reference table is loaded."""
        return self.peak_cseq is not None

    # ---- reference tables ------------------------------------------------------------------
    def _set_table(self, prefix: str, table: dict[str, torch.Tensor]) -> None:
        cseq, chrom, bp = table["cseq"], table["chr"], table["bp"]
        if cseq.ndim != 2 or cseq.shape[1] != self.config.d_cseq:
            raise ValueError(
                f"{prefix}_cseq must be (N, {self.config.d_cseq}), got {tuple(cseq.shape)}"
            )
        n = cseq.shape[0]
        if chrom.shape != (n,) or bp.shape != (n,):
            raise ValueError(
                f"{prefix}_chr / {prefix}_bp must be ({n},), got {tuple(chrom.shape)} / "
                f"{tuple(bp.shape)}"
            )
        setattr(self, f"{prefix}_cseq", cseq)
        setattr(self, f"{prefix}_chr", chrom)
        setattr(self, f"{prefix}_bp", bp)
        logger.info("%s table set: %d rows x %d (%s)", prefix, n, cseq.shape[1], cseq.dtype)

    def load_gene_table(
        self, source: Any, n_rows: int | None = None, device: str | torch.device | None = None
    ) -> Bloom:
        """Set the gene reference table (``mm10-evo2-v1`` ``genes.safetensors``, 32,129 rows).

        Args:
            source: ``genes.safetensors`` path, a mapping or an object with ``gene_cseq`` /
                ``gene_chr`` / ``gene_bp`` (or ``cseq`` / ``chr`` / ``bp``). A subset table
                carrying ``emb_idx`` (a row-subset table such as the test fixtures build) is scattered to full size.
            n_rows: Full table size for subset tables (default ``emb_idx.max() + 1``).
            device: Where the buffers live; default: the model's device.

        Returns:
            ``self`` (for chaining).
        """
        table = _expand_subset(_read_table(source, "gene"), n_rows)
        dev = self.device if device is None else torch.device(device)
        self._set_table("gene", {k: v.to(dev) for k, v in table.items()})
        return self

    def load_peak_table(
        self, source: Any, n_rows: int | None = None, device: str | torch.device | None = None
    ) -> Bloom:
        """Set the peak reference table lazily (``peaks.safetensors``, 1,747,701 x 768 f32).

        On CPU the safetensors tensors are read into process memory (5.4 GB; #43 tracks a
        file-backed mmap); moving the model (or passing ``device="cuda"``) moves them to the accelerator once.

        Args:
            source: ``peaks.safetensors`` path, mapping or object (see ``load_gene_table``).
            n_rows: Full table size for subset tables.
            device: Where the buffers live; default: the model's device.

        Returns:
            ``self``.
        """
        table = _expand_subset(_read_table(source, "peak"), n_rows)
        dev = self.device if device is None else torch.device(device)
        self._set_table("peak", {k: v.to(dev) for k, v in table.items()})
        return self

    def clear_peak_table(self) -> None:
        """Drop the peak table buffers (frees the 5.4 GB on the accelerator)."""
        for f in _TABLE_FIELDS:
            setattr(self, f"peak_{f}", None)

    # ---- input resolution ------------------------------------------------------------------
    def resolve(
        self, emb_idx: torch.Tensor, kind: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather ``(cseq, chr, bp)`` for a batch of tokens (``lookup_tables.resolve_embeddings``).

        Args:
            emb_idx: ``(B, L)`` int reference row per token; ``-1`` at pads (never indexed).
            kind: ``(B, L)`` int8, ``0`` pad, ``1`` gene, ``2`` peak.

        Returns:
            ``cseq (B, L, d_cseq)`` f32, ``chrom (B, L)`` int64, ``bp (B, L)`` f32; zeros at pads.
        """
        B, L = emb_idx.shape
        device = self.device
        emb_idx = emb_idx.to(device)
        kind = kind.to(device)
        cseq = torch.zeros(B, L, self.config.d_cseq, device=device, dtype=torch.float32)
        chrom = torch.zeros(B, L, device=device, dtype=torch.long)
        bp = torch.zeros(B, L, device=device, dtype=torch.float32)
        for code, prefix in ((KIND_GENE, "gene"), (KIND_PEAK, "peak")):
            mask = kind == code
            if not mask.any():
                continue
            table_cseq = getattr(self, f"{prefix}_cseq")
            if table_cseq is None:
                raise RuntimeError(
                    f"{prefix} reference table not loaded; call `model.load_{prefix}_table(...)` "
                    f"(peaks: `bloom download --what peaks`)"
                )
            idx = emb_idx[mask].long()
            if idx.numel() and (idx.min() < 0 or idx.max() >= table_cseq.shape[0]):
                raise IndexError(
                    f"{prefix} emb_idx out of range [0, {table_cseq.shape[0]}): "
                    f"min {int(idx.min())}, max {int(idx.max())}"
                )
            tdev = table_cseq.device
            idx_t = idx.to(tdev)
            cseq[mask] = table_cseq[idx_t].to(device=device, dtype=cseq.dtype)
            chrom[mask] = getattr(self, f"{prefix}_chr")[idx_t].to(device).long()
            bp[mask] = getattr(self, f"{prefix}_bp")[idx_t].to(device).float()
        return cseq, chrom, bp

    @staticmethod
    def key_padding_mask(lengths: torch.Tensor, L: int) -> torch.Tensor:
        """``(B, L)`` bool, ``True`` at pads: position ``>= lengths[b]`` (tokens are left-packed)."""
        pos = torch.arange(L, device=lengths.device).unsqueeze(0)  # (1, L)
        return pos >= lengths.unsqueeze(1).to(pos.device)  # (B, L)

    def first_layer_inputs(
        self,
        emb_idx: torch.Tensor,
        kind: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Everything ``BloomFirstLayer.forward`` consumes for a padded batch.

        Raises:
            MixedModalityViewError: a row mixes gene and peak tokens (issue #46; also reached
                through :meth:`encode` and ``encode_with_internals``).

        Returns:
            ``cseq (B, L, d_cseq)``, ``chrom (B, L)``, ``bp (B, L)``, ``clocal (B, L)`` and
            ``key_padding_mask (B, L)`` (``True`` = pad).
        """
        if lengths is None:
            lengths = (kind != KIND_PAD).sum(dim=1)
        check_single_modality(kind)  # issue #46: one modality per view
        cseq, chrom, bp = self.resolve(emb_idx, kind)  # (B, L, 768), (B, L), (B, L)
        key_padding_mask = self.key_padding_mask(lengths.to(self.device), emb_idx.shape[1])
        # Binarized input: every real token carries clocal = 1, pads 0 (strategy.py
        # ``_binarize_by_modality`` + ``PadFeats.fetch_value``; AGENTS.md section 5).
        clocal = (~key_padding_mask).to(cseq.dtype)  # (B, L)
        return cseq, chrom, bp, clocal, key_padding_mask

    def token_states(
        self,
        emb_idx: torch.Tensor,
        kind: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the first layer.

        Returns:
            ``token_h (B, L, d_model)`` and ``key_padding_mask (B, L)`` (``True`` = pad).
        """
        cseq, chrom, bp, clocal, key_padding_mask = self.first_layer_inputs(emb_idx, kind, lengths)
        token_h = self.first_layer(
            cseq=cseq, chrom=chrom, bp=bp, clocal=clocal, key_padding_mask=key_padding_mask
        )  # (B, L, d_model)
        return token_h, key_padding_mask

    def encode(
        self,
        emb_idx: torch.Tensor,
        kind: torch.Tensor,
        lengths: torch.Tensor | None = None,
        *,
        return_internals: bool = False,
        **internals_kwargs: Any,
    ) -> torch.Tensor | EncodeOutput:
        """Cell embeddings for a padded batch of token sets (= ``get_cell_representation``).

        Wrap calls in ``torch.inference_mode()`` for inference; gradients flow when it is not.
        The fused CUDA fast path is disabled for the duration of the call
        (:func:`fastpath_disabled`): CPU and GPU results agree to ~1e-5.

        Args:
            emb_idx: ``(B, L)`` int32 reference rows, ``-1`` at pads; real tokens left-packed.
            kind: ``(B, L)`` int8, ``1`` gene, ``2`` peak, ``0`` pad.
            lengths: ``(B,)`` int64 number of real tokens per row; derived from ``kind`` if
                ``None``.
            return_internals: Return an :class:`~bloom.model.internals.EncodeOutput` (token
                states, latents, per-group attention ``q`` / ``k``, cross-attention weights,
                token ids) instead of ``h_cell`` alone; ``h_cell`` is bitwise the same (test I2).
            **internals_kwargs: Forwarded to
                :func:`~bloom.model.internals.encode_with_internals` (``reference``,
                ``cross_attn``, ``per_head``, ``requires_grad``, ``max_tokens``); only valid
                with ``return_internals=True``.

        Returns:
            ``h_cell (B, d_model)`` float32 (rows with ``lengths == 0`` are ``NaN``), or an
            ``EncodeOutput`` whose ``h_cell`` is that tensor.

        Raises:
            MixedModalityViewError: a row mixes gene (``kind == 1``) and peak (``kind == 2``) tokens;
                one modality per view (issue #46).
        """
        if return_internals:
            from bloom.model.internals import encode_with_internals

            return encode_with_internals(self, emb_idx, kind, lengths, **internals_kwargs)
        if internals_kwargs:
            raise TypeError(
                f"encode() got {sorted(internals_kwargs)}: these options need return_internals=True"
            )
        if lengths is None:
            lengths = (kind != KIND_PAD).sum(dim=1)
        with fastpath_disabled():
            token_h, key_padding_mask = self.token_states(emb_idx, kind, lengths)
            h_cell = self.encoder(token_h, key_padding_mask=key_padding_mask)  # (B, d_model)
        empty = lengths.to(h_cell.device) == 0
        if bool(empty.any()):
            h_cell = h_cell.masked_fill(empty.unsqueeze(1), float("nan"))
        return h_cell

    def encode_batch(self, batch: Any) -> torch.Tensor:
        """``encode`` on a ``Batch``-like object exposing ``emb_idx``, ``kind``, ``lengths``."""
        return self.encode(batch.emb_idx, batch.kind, getattr(batch, "lengths", None))

    def forward(
        self,
        emb_idx: torch.Tensor,
        kind: torch.Tensor,
        lengths: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | EncodeOutput:
        """Alias of :meth:`encode`."""
        return self.encode(emb_idx, kind, lengths, **kwargs)


# --------------------------------------------------------------------------------------------
# construction and loading
# --------------------------------------------------------------------------------------------
def check_seq_head_scale(model: Bloom, config: ModelConfig | None = None) -> float:
    """Assert the installed ``bloom-arch`` sequence-head scale equals ``config.seq_head_scale``.

    Returns:
        The scale in use.

    Raises:
        SeqHeadScaleMismatchError: on mismatch (AGENTS.md section 6, test M3).
    """
    config = model.config if config is None else config
    actual = model.first_layer.seq_head_scale
    expected = config.seq_head_scale
    if actual is None or not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=0.0):
        raise SeqHeadScaleMismatchError(
            f"bloom-arch sequence-head scale is {actual!r} but config.json expects "
            f"{expected!r} (1/sqrt(32) = {1 / math.sqrt(32):.17g} for hyacinth). The installed "
            "bloom-arch does not match the weights: install the pinned commit "
            "(pyproject.toml; AllenInstitute/bloom-sprint-attn@ef5fb8a) -- branch mask-dev uses "
            "1/sqrt(768) and gives wrong embeddings."
        )
    return actual


def build_from_config(config: ModelConfig, check_scale: bool = True) -> Bloom:
    """Instantiate ``Bloom`` (random init, ``eval()`` mode) and verify the attention scale."""
    model = Bloom(config)
    if check_scale:
        check_seq_head_scale(model, config)
    model.eval()
    return model


def remap_state_dict(
    state_dict: Mapping[str, torch.Tensor], remap: Mapping[str, str]
) -> dict[str, torch.Tensor]:
    """Rename keys by prefix (``config.json["state_dict_remap"]``); longest prefix wins.

    Idempotent on already-remapped keys (they match no source prefix).
    """
    prefixes = sorted(remap.items(), key=lambda kv: -len(kv[0]))
    out: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        new = k
        for src, dst in prefixes:
            if k.startswith(src):
                new = dst + k[len(src) :]
                break
        out[new] = v
    return out


def inverse_state_dict_remap(remap: Mapping[str, str]) -> dict[str, str]:
    """``{new_prefix: old_prefix}`` so ``remap_state_dict(sd, inverse)`` restores training names."""
    inv = {v: k for k, v in remap.items()}
    if len(inv) != len(remap):
        raise ValueError(f"state_dict_remap is not invertible: {dict(remap)}")
    return inv


def load_pretrained(
    path: str | Path,
    device: str | torch.device = "cpu",
    strict: bool = True,
    genes: Any | None = None,
    peaks: Any | None = None,
) -> Bloom:
    """Build ``Bloom`` from ``config.json`` + ``model.safetensors`` and load the weights.

    Args:
        path: Model directory (``config.json`` and the weights file named in
            ``config["weights"]["file"]``, default ``model.safetensors``) or the ``config.json``
            path itself.
        device: Target device (``"auto"``, ``"cpu"``, ``"cuda"``, ...).
        strict: Passed to ``load_state_dict``; with ``False`` missing / unexpected keys are
            logged as warnings instead of raising.
        genes: Optional gene table source for ``load_gene_table`` (path, mapping or object).
        peaks: Optional peak table source for ``load_peak_table``; **not** opened when ``None``.

    Returns:
        The model in ``eval()`` mode on ``device``.

    Raises:
        SeqHeadScaleMismatchError: if the installed ``bloom-arch`` scale differs from the config.
        FileNotFoundError: if the config or weights file is missing.
    """
    path = Path(path)
    cfg_path = path / "config.json" if path.is_dir() else path
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config.json not found at {cfg_path}")
    config = ModelConfig.from_json(cfg_path)
    weights_file = config.weights.get("file", "model.safetensors")
    weights_path = cfg_path.parent / weights_file
    if not weights_path.is_file():
        raise FileNotFoundError(f"weights not found at {weights_path}")

    model = build_from_config(config)  # raises on a scale mismatch before touching weights

    with safe_open(str(weights_path), framework="pt", device="cpu") as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
    state_dict = remap_state_dict(state_dict, config.state_dict_remap)
    result = model.load_state_dict(state_dict, strict=strict)
    if result.missing_keys or result.unexpected_keys:
        warnings.warn(
            f"load_pretrained({weights_path}): {len(result.missing_keys)} missing, "
            f"{len(result.unexpected_keys)} unexpected keys; missing={result.missing_keys[:5]}, "
            f"unexpected={result.unexpected_keys[:5]}",
            RuntimeWarning,
            stacklevel=2,
        )
    expected_n = config.n_parameters_expected
    if expected_n is not None and model.n_parameters != expected_n:
        warnings.warn(
            f"parameter count {model.n_parameters:,} differs from config.json "
            f"parameters.total {expected_n:,}",
            RuntimeWarning,
            stacklevel=2,
        )

    dev = resolve_device(device)
    model.to(dev)
    if genes is not None:
        model.load_gene_table(genes)
    if peaks is not None:
        model.load_peak_table(peaks)
    model.eval()
    return model
