"""PerceiverIO-style encoder downstream of ``BloomFirstLayer``.

Port of ``bloom/model/perceiver_encoder.py`` (alleninstitute/bloom@141f0d3). Cross-attention
compresses the variable-length, pad-masked token sequence into ``n_latents`` learned latents;
the latents evolve through an unmasked self-attention stack; the cell embedding is the
LayerNorm of their mean.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["PerceiverEncoder"]


class PerceiverEncoder(nn.Module):
    """Latent-bottleneck encoder: cross-attention into latents, then self-attention.

    Args:
        d_model: Token / latent width.
        n_latents: Number of learned latents.
        n_self_layers: Number of ``nn.TransformerEncoderLayer`` blocks on the latents.
        n_heads: Attention heads of the cross- and self-attention.
        dropout: Dropout rate (identity in ``eval()``).
    """

    def __init__(
        self,
        d_model: int,
        n_latents: int,
        n_self_layers: int,
        n_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, n_latents, d_model))  # (1, M, d_model)

        self.cross_q_norm = nn.LayerNorm(d_model)
        self.cross_kv_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.cross_ffn_norm = nn.LayerNorm(d_model)
        self.cross_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.self_layers = nn.TransformerEncoder(
            encoder_layer, num_layers=n_self_layers, enable_nested_tensor=False
        )
        self.final_norm = nn.LayerNorm(d_model)

    @property
    def n_latents(self) -> int:
        """Number of latents (256 for ``hyacinth``)."""
        return int(self.latents.shape[1])

    def cross_attend(
        self,
        h: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
        average_attn_weights: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Cross-attention of the latents into the token states (+ FFN residual).

        Args:
            h: ``(B, L, d_model)`` token states from ``BloomFirstLayer``.
            key_padding_mask: ``(B, L)`` bool, ``True`` = pad.
            need_weights: Also return the attention weights (``encode`` never sets it; the
                internals API calls this a second time for the weights only, docs/internals.md).
            average_attn_weights: With ``need_weights``, return the head average ``(B, M, L)``
                (default) or the per-head weights ``(B, H, M, L)``.

        Returns:
            ``(q, weights)`` with ``q`` ``(B, M, d_model)`` and ``weights`` ``None`` unless requested.
        """
        B = h.shape[0]
        q = self.latents.expand(B, -1, -1)  # (B, M, d_model)
        q_n = self.cross_q_norm(q)
        kv_n = self.cross_kv_norm(h)
        attn_out, weights = self.cross_attn(
            q_n,
            kv_n,
            kv_n,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=average_attn_weights,
        )
        q = q + attn_out
        q = q + self.cross_ffn(self.cross_ffn_norm(q))
        return q, weights

    def latent_states(
        self, h: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Latents after the self-attention stack, before pooling.

        Returns:
            ``(B, M, d_model)`` latent states.
        """
        q, _ = self.cross_attend(h, key_padding_mask)
        return self.self_layers(q)  # (B, M, d_model)

    def pool(self, z: torch.Tensor) -> torch.Tensor:
        """Mean over latents followed by LayerNorm: ``(B, M, d_model) -> (B, d_model)``."""
        return self.final_norm(z.mean(dim=1))

    def forward(
        self, h: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compress token states into a cell embedding.

        Args:
            h: ``(B, L, d_model)`` token states.
            key_padding_mask: ``(B, L)`` bool, ``True`` = pad (PyTorch MHA convention).

        Returns:
            ``(B, d_model)`` cell embedding.
        """
        return self.pool(self.latent_states(h, key_padding_mask))
