"""First encoder layer: ``bloom_arch.BloomAttentionBlock`` + FFN residual.

Port of ``bloom/model/bloom_attention_layer.py`` (alleninstitute/bloom@141f0d3); module and
parameter names are unchanged so the exported state dict loads without remapping.
"""

from __future__ import annotations

import torch
from bloom_arch.attention import BloomAttentionBlock
from torch import nn

from bloom.config import ModelConfig

__all__ = ["BloomFirstLayer"]


class BloomFirstLayer(nn.Module):
    """Sole input-side encoder: genomic-position-aware attention over the token set.

    Each token is ``(cseq[768], chr, bp, clocal)``. The block mixes three head types
    (sequence similarity, genomic proximity, standard) and is followed by a pre-norm FFN
    residual. Pad positions are excluded through ``key_padding_mask``; they arrive with
    ``cseq = 0, chr = 0, bp = 0, clocal = 0`` from ``Bloom.resolve``.

    Args:
        config: Model configuration (``d_cseq``, ``d_model``, ``dropout`` and ``attention``).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        attn = config.attention
        self.attn_block = BloomAttentionBlock(
            d_seq=config.d_cseq,
            d_val=attn.d_val,
            d_u=attn.d_u,
            K=attn.K,
            d_v=attn.d_v,
            n_sequence_heads=attn.n_sequence_heads,
            n_position_heads=attn.n_position_heads,
            n_standard_heads=attn.n_standard_heads,
            d_out=config.d_model,
            max_G=attn.max_G,
            lambda_min=attn.lambda_min,
            lambda_max=attn.lambda_max,
            target_scales=attn.target_scales,
            rho=attn.rho,
            beta_init=attn.beta_init,
            tau_init=attn.tau_init,
            learn_tau=attn.learn_tau,
            pad_v_for_flash=attn.pad_v_for_flash,
            seed=attn.seed,
        )

        # Measurement value encoder: scalar -> d_val, LayerNorm'ed.
        self.m_encoder = nn.Sequential(
            nn.Linear(1, attn.d_val),
            nn.GELU(),
            nn.Linear(attn.d_val, attn.d_val),
            nn.LayerNorm(attn.d_val),
        )

        # Pre-norm FFN residual (same convention as the Perceiver layers).
        self.ln_ffn = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.GELU(),
            nn.Linear(config.d_model * 4, config.d_model),
            nn.Dropout(config.dropout),
        )

    @property
    def seq_head_scale(self) -> float | None:
        """Softmax scale the installed ``bloom-arch`` uses for the sequence heads."""
        heads = self.attn_block.seq_heads
        return None if heads is None else float(heads._scale)

    def forward(
        self,
        cseq: torch.Tensor,
        chrom: torch.Tensor,
        bp: torch.Tensor,
        clocal: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attention block followed by the FFN residual.

        Args:
            cseq: ``(B, L, d_cseq)`` float reference sequence embeddings.
            chrom: ``(B, L)`` integer chromosome index (0-based).
            bp: ``(B, L)`` base-pair coordinate (cast to float).
            clocal: ``(B, L)`` measurement scalar (1 for detected features, 0 at pads).
            key_padding_mask: ``(B, L)`` bool, ``True`` = pad (excluded as a key); ``None`` =
                no masking. Same convention as ``bloom_arch`` and ``nn.MultiheadAttention``.

        Returns:
            ``(B, L, d_model)`` token states.
        """
        m = self.m_encoder(clocal.unsqueeze(-1))  # (B, L, d_val)
        h = self.attn_block(
            n=cseq, x=bp.float(), g=chrom.long(), m=m, key_padding_mask=key_padding_mask
        )  # (B, L, d_model)
        h = h + self.ffn(self.ln_ffn(h))  # (B, L, d_model)
        return h
