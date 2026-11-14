"""Structured internals of ``Bloom.encode`` for the ML track (docs/internals.md).

What this module adds on top of ``Bloom.encode`` (which stays a plain ``h_cell`` return):

* :class:`EncodeOutput` -- ``h_cell``, post-first-layer ``token_h``, Perceiver ``latents``,
  per-head-group attention queries / keys ``qk``, the latent -> token ``cross_attn`` weights and
  resolved ``token_ids``; produced by :func:`encode_with_internals` or
  ``model.encode(..., return_internals=True)``.
* :func:`pair_logits` -- first-layer attention logits ``q_i . k_j * scale`` for *selected* token
  pairs only; the ``L x L`` attention matrix is never materialised (at ``L = 8192`` it would be
  20 heads x 256 MB per cell).
* :func:`capture` -- a forward-hook recorder for any named submodule, hooks removed on exit.

Numerics. ``h_cell`` is computed by exactly the same ops as ``Bloom.encode`` (test I2, bitwise).
The queries / keys are obtained by **re-applying the block's projection modules** (``W_Q`` /
``W_K`` linears, the position-head ``[beta * u ; w * r_phi] * sqrt(tau)`` construction) to the
same inputs the block saw; the attention itself is not recomputed. ``bloom_arch@ef5fb8a`` keeps
``q`` / ``k`` local to each head's ``forward`` and does not expose them, so this duplicates three
cheap ``(B, L, d_in) x (d_in, H d)`` matmuls per call. The cross-attention weights come from a
second call of ``PerceiverEncoder.cross_attend(..., need_weights=True)``; the default path that
feeds ``h_cell`` is untouched.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from bloom.model.bloom import (
    KIND_GENE,
    KIND_PAD,
    KIND_PEAK,
    Bloom,
    check_single_modality,
    fastpath_disabled,
)

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "HEAD_GROUPS",
    "EncodeOutput",
    "InternalsMemoryError",
    "capture",
    "encode_with_internals",
    "estimate_internals_bytes",
    "first_layer_qk",
    "pair_logits",
    "qk_scales",
    "resolve_token_ids",
]

#: Head groups of ``BloomAttentionBlock`` in output order (``hyacinth``: 8 / 8 / 4 heads).
HEAD_GROUPS: tuple[str, ...] = ("seq", "pos", "std")

#: Default ceiling on ``B * L`` for ``return_internals=True`` (test I7): 4 x 8192 or 8 x 4096
#: tokens = 0.41 GiB of returned tensors for ``hyacinth`` (13.3 KB per token).
DEFAULT_MAX_TOKENS: int = 32_768

#: A ``token_ids`` resolver: ``(emb_idx (n,) int64, kind (n,) int64) -> sequence of n strings``.
TokenIdResolver = Callable[[np.ndarray, np.ndarray], Sequence[str]]


class InternalsMemoryError(RuntimeError):
    """``return_internals=True`` was requested for more tokens than ``max_tokens`` allows."""


@dataclass
class EncodeOutput:
    """Everything ``Bloom.encode`` computes on the way to ``h_cell`` (docs/architecture.md 5.3).

    All tensors live on the model's device. Pad positions (``key_padding_mask == True``) carry
    zeros in ``token_h`` / ``q`` / ``k`` only where the model puts them there -- treat them as
    undefined and index real tokens through ``lengths`` or ``key_padding_mask``.

    Attributes:
        h_cell: ``(B, d_model)`` cell embeddings, identical to ``Bloom.encode``.
        token_h: ``(B, L, d_model)`` token states after ``BloomFirstLayer`` (attention block + FFN).
        latents: ``(B, M, d_model)`` Perceiver latents after the self-attention stack, before
            mean-pool + LayerNorm (``M = 256`` for ``hyacinth``).
        qk: ``{"seq": (q, k), "pos": (q, k), "std": (q, k)}``; each ``(B, L, H_g, d_g)`` with
            ``H = (8, 8, 4)`` and ``d = (32, 160, 32)`` for ``hyacinth`` (``d_pos = d_u + 2K``).
            Position heads have ``q is k`` (the block uses one tensor for both).
        scales: Softmax scale per group: ``seq`` = ``1/sqrt(32)`` (the ``bloom-arch`` pin),
            ``pos`` = ``1.0`` (``tau`` is baked into ``q``), ``std`` = ``1/sqrt(d_v)``.
        cross_attn: ``(B, M, L)`` latent -> token attention averaged over heads, or ``(B, H, M, L)``
            with ``per_head=True``; ``None`` when not requested. Rows sum to 1 and are 0 at pads.
        token_ids: Per row, ``L`` entries: gene symbol / peak ``"chr:start-end"``, ``None`` at
            pads; ``None`` altogether when no reference was available to resolve them.
        lengths: ``(B,)`` int64 real tokens per row.
        key_padding_mask: ``(B, L)`` bool, ``True`` = pad.
        emb_idx: ``(B, L)`` int32 reference rows as passed in (``-1`` at pads).
        kind: ``(B, L)`` int8 token kinds as passed in.
    """

    h_cell: Tensor  # (B, d_model)
    token_h: Tensor  # (B, L, d_model)
    latents: Tensor  # (B, M, d_model)
    qk: dict[str, tuple[Tensor, Tensor]]  # group -> ((B, L, H_g, d_g), (B, L, H_g, d_g))
    scales: dict[str, float]
    cross_attn: Tensor | None  # (B, M, L) or (B, H, M, L)
    token_ids: list[list[str | None]] | None
    lengths: Tensor  # (B,)
    key_padding_mask: Tensor  # (B, L)
    emb_idx: Tensor = field(repr=False)  # (B, L)
    kind: Tensor = field(repr=False)  # (B, L)

    @property
    def batch_size(self) -> int:
        """``B``."""
        return int(self.h_cell.shape[0])

    @property
    def n_tokens(self) -> int:
        """Padded length ``L``."""
        return int(self.token_h.shape[1])

    def position_of(self, token_id: str, batch: int = 0) -> int:
        """Position of ``token_id`` (symbol or ``chr:start-end``) in row ``batch``.

        Raises:
            KeyError: when the id is not among the row's real tokens.
            RuntimeError: when ``token_ids`` were not resolved (no reference).
        """
        if self.token_ids is None:
            raise RuntimeError("token_ids were not resolved; pass `reference=` to encode")
        row = self.token_ids[batch]
        try:
            return row.index(token_id)
        except ValueError:
            raise KeyError(f"{token_id!r} is not a token of row {batch}") from None

    def to(self, device: str | torch.device) -> EncodeOutput:
        """Copy of this output with every tensor moved to ``device``."""

        def mv(t: Tensor | None) -> Tensor | None:
            return None if t is None else t.to(device)

        return EncodeOutput(
            h_cell=mv(self.h_cell),
            token_h=mv(self.token_h),
            latents=mv(self.latents),
            qk={g: (mv(q), mv(k)) for g, (q, k) in self.qk.items()},
            scales=dict(self.scales),
            cross_attn=mv(self.cross_attn),
            token_ids=self.token_ids,
            lengths=mv(self.lengths),
            key_padding_mask=mv(self.key_padding_mask),
            emb_idx=mv(self.emb_idx),
            kind=mv(self.kind),
        )

    def detach(self) -> EncodeOutput:
        """Copy with every tensor detached from the autograd graph."""

        def dt(t: Tensor | None) -> Tensor | None:
            return None if t is None else t.detach()

        return EncodeOutput(
            h_cell=dt(self.h_cell),
            token_h=dt(self.token_h),
            latents=dt(self.latents),
            qk={g: (dt(q), dt(k)) for g, (q, k) in self.qk.items()},
            scales=dict(self.scales),
            cross_attn=dt(self.cross_attn),
            token_ids=self.token_ids,
            lengths=self.lengths,
            key_padding_mask=self.key_padding_mask,
            emb_idx=self.emb_idx,
            kind=self.kind,
        )


# --------------------------------------------------------------------------------------------
# memory guard
# --------------------------------------------------------------------------------------------
def estimate_internals_bytes(
    model: Bloom, n_tokens: int, cross_attn: bool = True, per_head: bool = False
) -> int:
    """Bytes of fp32 tensors an :class:`EncodeOutput` holds for ``n_tokens = B * L`` tokens.

    Counts ``token_h``, ``q`` / ``k`` of every group and ``cross_attn``; ``latents`` and
    ``h_cell`` are per cell and negligible. The attention matrix of the first layer is **not**
    included: ``encode`` never materialises it (scaled-dot-product attention) and neither does
    this module.
    """
    cfg = model.config
    attn = cfg.attention
    block = model.first_layer.attn_block
    per_token = cfg.d_model  # token_h
    per_token += 2 * attn.n_sequence_heads * attn.d_v  # seq q, k
    if block.pos_heads is not None:
        per_token += attn.n_position_heads * block.pos_heads.head_dim_qk  # pos q (== k)
    per_token += 2 * attn.n_standard_heads * attn.d_v  # std q, k
    if cross_attn:
        per_token += cfg.n_latents * (cfg.n_heads if per_head else 1)
    return 4 * per_token * int(n_tokens)


def _check_memory(
    model: Bloom, B: int, L: int, max_tokens: int | None, cross_attn: bool, per_head: bool
) -> None:
    if max_tokens is None:
        return
    n = B * L
    if n > int(max_tokens):
        est = estimate_internals_bytes(model, n, cross_attn, per_head) / 2**30
        raise InternalsMemoryError(
            f"return_internals=True for B x L = {B} x {L} = {n:,} tokens exceeds max_tokens="
            f"{int(max_tokens):,} (about {est:.2f} GiB of returned tensors on {model.device}). "
            "Encode fewer cells per call, pass a larger `max_tokens` (or `None` to disable the "
            "guard), or use `pair_logits` on a small batch instead of keeping all internals."
        )


# --------------------------------------------------------------------------------------------
# first-layer queries / keys
# --------------------------------------------------------------------------------------------
def qk_scales(first_layer: nn.Module) -> dict[str, float]:
    """Softmax scale per head group as ``bloom_arch@ef5fb8a`` applies it.

    ``seq``: ``SequenceHeads._scale`` (``1/sqrt(d_v)`` = ``1/sqrt(32)`` for ``hyacinth``; asserted
    against ``config.json`` at load). ``pos``: ``1.0`` (``sqrt(tau)`` is multiplied into ``q = k``).
    ``std``: SDPA default ``1/sqrt(d_v)``.
    """
    block = first_layer.attn_block
    out: dict[str, float] = {}
    if block.seq_heads is not None:
        out["seq"] = float(block.seq_heads._scale)
    if block.pos_heads is not None:
        out["pos"] = 1.0
    if block.std_heads is not None:
        out["std"] = 1.0 / math.sqrt(block.std_heads.d_v)
    return out


def first_layer_qk(
    first_layer: nn.Module,
    cseq: Tensor,
    chrom: Tensor,
    bp: Tensor,
    clocal: Tensor,
) -> dict[str, tuple[Tensor, Tensor]]:
    """Queries and keys of every head group of ``BloomFirstLayer.attn_block``.

    Re-applies the block's projections to the same inputs ``BloomFirstLayer.forward`` builds
    (``n = cseq``, ``m = m_encoder(clocal)``, ``u`` / ``r_phi`` from ``pos_encoder(bp, chrom)``);
    the attention is not recomputed.

    Args:
        first_layer: ``model.first_layer``.
        cseq: ``(B, L, d_cseq)`` reference sequence embeddings (``Bloom.resolve``).
        chrom: ``(B, L)`` chromosome index.
        bp: ``(B, L)`` base-pair coordinate.
        clocal: ``(B, L)`` measurement scalar (1 real, 0 pad).

    Returns:
        ``{"seq": (q, k), "pos": (q, k), "std": (q, k)}`` with ``q`` / ``k`` ``(B, L, H_g, d_g)``;
        groups the block was built without are absent. For ``pos`` the same tensor is returned
        twice (``q is k``).
    """
    block = first_layer.attn_block
    B, L, _ = cseq.shape
    n = cseq
    m = first_layer.m_encoder(clocal.unsqueeze(-1))  # (B, L, d_val)
    pos = block.pos_encoder(bp.float(), chrom.long())
    u = pos["u"].to(dtype=n.dtype)  # (B, L, d_u)
    r_phi = pos["r_phi"].to(dtype=n.dtype)  # (B, L, 2K)
    out: dict[str, tuple[Tensor, Tensor]] = {}

    heads = block.seq_heads
    if heads is not None:
        q = heads.W_Q(n).view(B, L, heads.H, heads.d_v)  # (B, L, H_seq, d_v)
        k = heads.W_K(n).view(B, L, heads.H, heads.d_v)
        out["seq"] = (q, k)

    heads = block.pos_heads
    if heads is not None:
        H = heads.H
        w_tiled = heads.w_tiled()  # (H, 2K)
        sqrt_tau = heads.tau.sqrt().view(1, 1, H, 1)
        beta = heads.beta.view(1, 1, H, 1)
        u_part = u.unsqueeze(2).expand(-1, -1, H, -1) * beta  # (B, L, H, d_u)
        r_part = r_phi.unsqueeze(2) * w_tiled.view(1, 1, H, -1)  # (B, L, H, 2K)
        qk = torch.cat([u_part, r_part], dim=-1) * sqrt_tau  # (B, L, H_pos, d_u + 2K)
        out["pos"] = (qk, qk)

    heads = block.std_heads
    if heads is not None:
        p = torch.cat([block.beta_std * u, r_phi], dim=-1)  # (B, L, d_u + 2K)
        f = torch.cat([n, p, m], dim=-1)  # (B, L, d_seq + d_u + 2K + d_val)
        q = heads.W_Q(f).view(B, L, heads.H, heads.d_v)  # (B, L, H_std, d_v)
        k = heads.W_K(f).view(B, L, heads.H, heads.d_v)
        out["std"] = (q, k)
    return out


# --------------------------------------------------------------------------------------------
# token ids
# --------------------------------------------------------------------------------------------
def resolve_token_ids(
    emb_idx: Tensor,
    kind: Tensor,
    reference: Any | TokenIdResolver,
) -> list[list[str | None]]:
    """Gene symbols / peak names per token; ``None`` at pads.

    Args:
        emb_idx: ``(B, L)`` reference rows.
        kind: ``(B, L)`` token kinds.
        reference: A ``bloom.reference.ReferenceTables`` (``symbol_of`` / ``peak_name_of``) or a
            callable ``(emb_idx (n,), kind (n,)) -> n strings`` for custom tables.
    """
    e = emb_idx.detach().cpu().numpy().astype(np.int64)
    kd = kind.detach().cpu().numpy().astype(np.int64)
    B, L = e.shape
    real = kd != KIND_PAD
    flat_e, flat_k = e[real], kd[real]
    if callable(reference) and not hasattr(reference, "symbol_of"):
        names = np.asarray(list(reference(flat_e, flat_k)), dtype=object)
        if names.shape != flat_e.shape:
            raise ValueError(f"token-id resolver returned {names.shape}, expected {flat_e.shape}")
    else:
        names = np.empty(flat_e.shape, dtype=object)
        g = flat_k == KIND_GENE
        if g.any():
            names[g] = reference.symbol_of(flat_e[g])
        p = flat_k == KIND_PEAK
        if p.any():
            names[p] = reference.peak_name_of(flat_e[p])
        other = ~(g | p)
        if other.any():
            raise ValueError(f"unknown token kinds {np.unique(flat_k[other]).tolist()}")
    grid = np.full((B, L), None, dtype=object)
    grid[real] = names
    return [row.tolist() for row in grid]


# --------------------------------------------------------------------------------------------
# the forward with internals
# --------------------------------------------------------------------------------------------
def encode_with_internals(
    model: Bloom,
    emb_idx: Tensor,
    kind: Tensor,
    lengths: Tensor | None = None,
    *,
    reference: Any | TokenIdResolver | None = None,
    cross_attn: bool = True,
    per_head: bool = False,
    requires_grad: bool = False,
    max_tokens: int | None = DEFAULT_MAX_TOKENS,
) -> EncodeOutput:
    """``Bloom.encode`` that also returns the intermediate tensors (docs/internals.md).

    ``h_cell`` is produced by the same operations as :meth:`Bloom.encode` (test I2: bitwise
    equal). Runs under ``torch.inference_mode()`` unless ``requires_grad=True``, in which case
    gradients are enabled and ``token_h.retain_grad()`` is set so ``h_cell.sum().backward()``
    populates ``out.token_h.grad`` (test I6).

    Args:
        model: The model (``eval()`` mode; gene / peak tables loaded).
        emb_idx: ``(B, L)`` int32 reference rows, ``-1`` at pads; real tokens left-packed.
        kind: ``(B, L)`` int8, ``1`` gene, ``2`` peak, ``0`` pad.
        lengths: ``(B,)`` real tokens per row; derived from ``kind`` when ``None``.
        reference: Resolves ``token_ids``: a ``ReferenceTables`` or a callable
            ``(emb_idx, kind) -> names``; ``None`` uses ``model.reference`` when
            ``bloom.load_model`` attached one, else ``token_ids`` is ``None``.
        cross_attn: Compute the latent -> token attention weights (a second cross-attention
            call with ``need_weights=True``; ``O(B * H * M * L)``).
        per_head: Return ``cross_attn`` as ``(B, H, M, L)`` instead of the head average.
        requires_grad: Build the autograd graph (saliency / attribution).
        max_tokens: Raise :class:`InternalsMemoryError` before any allocation when
            ``B * L > max_tokens`` (test I7); ``None`` disables the guard.

    Returns:
        :class:`EncodeOutput`.

    Raises:
        MixedModalityViewError: a row mixes gene and peak tokens (one modality per view, issue #46).
    """
    B, L = emb_idx.shape
    _check_memory(model, B, L, max_tokens, cross_attn, per_head)
    check_single_modality(kind)
    if lengths is None:
        lengths = (kind != KIND_PAD).sum(dim=1)
    if reference is None:
        reference = getattr(model, "reference", None)

    if requires_grad:
        grad_ctx: Any = torch.enable_grad()
        infer_ctx: Any = torch.inference_mode(False)
    else:
        grad_ctx = torch.no_grad()
        infer_ctx = torch.inference_mode()
    with infer_ctx, grad_ctx, fastpath_disabled():
        cseq, chrom, bp, clocal, key_padding_mask = model.first_layer_inputs(emb_idx, kind, lengths)
        token_h = model.first_layer(
            cseq=cseq, chrom=chrom, bp=bp, clocal=clocal, key_padding_mask=key_padding_mask
        )  # (B, L, d_model)
        if requires_grad and token_h.requires_grad:
            token_h.retain_grad()
        enc = model.encoder
        q_lat, _ = enc.cross_attend(token_h, key_padding_mask)  # default path (feeds h_cell)
        latents = enc.self_layers(q_lat)  # (B, M, d_model)
        h_cell = enc.pool(latents)  # (B, d_model)
        empty = lengths.to(h_cell.device) == 0
        if bool(empty.any()):
            h_cell = h_cell.masked_fill(empty.unsqueeze(1), float("nan"))

        qk = first_layer_qk(model.first_layer, cseq, chrom, bp, clocal)
        weights: Tensor | None = None
        if cross_attn:
            _, weights = enc.cross_attend(
                token_h,
                key_padding_mask,
                need_weights=True,
                average_attn_weights=not per_head,
            )  # (B, M, L) or (B, H, M, L)

    token_ids = None if reference is None else resolve_token_ids(emb_idx, kind, reference)
    return EncodeOutput(
        h_cell=h_cell,
        token_h=token_h,
        latents=latents,
        qk=qk,
        scales=qk_scales(model.first_layer),
        cross_attn=weights,
        token_ids=token_ids,
        lengths=lengths,
        key_padding_mask=key_padding_mask,
        emb_idx=emb_idx,
        kind=kind,
    )


# --------------------------------------------------------------------------------------------
# pair logits
# --------------------------------------------------------------------------------------------
def pair_logits(
    out: EncodeOutput,
    pairs: Sequence[tuple[int, int]] | np.ndarray | Tensor,
    group: str = "seq",
    batch: int = 0,
    heads: Sequence[int] | None = None,
) -> Tensor:
    """First-layer attention logits ``q_i . k_j * scale`` for selected token pairs.

    Only the requested pairs are computed; the ``L x L`` matrix never exists (test I3).

    Args:
        out: An :class:`EncodeOutput`.
        pairs: ``(P, 2)`` token positions ``(i, j)`` within row ``batch`` (``i`` = query,
            ``j`` = key); both must be real tokens (``< lengths[batch]``).
        group: ``"seq"``, ``"pos"`` or ``"std"``.
        batch: Row of the batch.
        heads: Optional subset of head indices; default all ``H_g`` heads of the group.

    Returns:
        ``(P, H_sel)`` float32 logits (pre-softmax, after the group's scale).

    Raises:
        MixedModalityViewError: row ``batch`` mixes gene and peak tokens (cross-modal logits are
            undefined, issue #46).
    """
    if group not in out.qk:
        raise KeyError(f"group {group!r} not in {sorted(out.qk)}")
    check_single_modality(out.kind[batch])
    q, k = out.qk[group]
    idx = torch.as_tensor(np.asarray(pairs), dtype=torch.long, device=q.device).reshape(-1, 2)
    n_real = int(out.lengths[batch])
    if idx.numel() and (idx.min() < 0 or idx.max() >= n_real):
        raise IndexError(
            f"pair positions must be in [0, lengths[{batch}]={n_real}); got min {int(idx.min())}, "
            f"max {int(idx.max())}"
        )
    qb, kb = q[batch], k[batch]  # (L, H_g, d_g)
    if heads is not None:
        h = torch.as_tensor(list(heads), dtype=torch.long, device=q.device)
        qb, kb = qb[:, h], kb[:, h]
    qi = qb[idx[:, 0]]  # (P, H, d)
    kj = kb[idx[:, 1]]  # (P, H, d)
    return (qi * kj).sum(dim=-1) * out.scales[group]  # (P, H)


# --------------------------------------------------------------------------------------------
# hook capture
# --------------------------------------------------------------------------------------------
@contextmanager
def capture(
    model: nn.Module,
    names: str | Iterable[str],
    *,
    detach: bool = False,
) -> Iterator[dict[str, Any]]:
    """Record the forward output of named submodules while the block runs (test I5).

    Args:
        model: Any module; names are matched against ``dict(model.named_modules())``.
        names: One or more module names, e.g. ``"first_layer.attn_block.W_O"``,
            ``"encoder.self_layers.layers.3"``, ``"encoder.cross_attn"``.
        detach: Store ``.detach()``-ed tensors (default keeps the graph).

    Yields:
        A dict filled during the block: ``name -> output`` of the **last** call of that module
        (an ``nn.MultiheadAttention`` yields its ``(attn_output, weights)`` tuple). Entries for
        modules that never ran are absent. Hooks are removed when the block exits, also on
        error; nesting is fine.

    Raises:
        KeyError: for a name that is not a submodule.
    """
    if isinstance(names, str):
        names = [names]
    names = list(names)
    modules: Mapping[str, nn.Module] = dict(model.named_modules())
    missing = [n for n in names if n not in modules]
    if missing:
        hint = [m for m in modules if any(m.startswith(x.split(".")[0]) for x in missing)][:8]
        raise KeyError(f"not submodules of {type(model).__name__}: {missing}; e.g. {hint}")
    store: dict[str, Any] = {}
    handles = []

    def _hook(name: str):
        def fn(_mod: nn.Module, _inp: Any, out: Any) -> None:
            if detach:
                out = torch.utils._pytree.tree_map(
                    lambda t: t.detach() if isinstance(t, Tensor) else t, out
                )
            store[name] = out

        return fn

    try:
        for n in names:
            handles.append(modules[n].register_forward_hook(_hook(n)))
        yield store
    finally:
        for h in handles:
            h.remove()
