"""Query-side feature embedding (used by the decoder only).

Port of ``bloom/model/embedding.py`` (alleninstitute/bloom@141f0d3). The input side of the
model never goes through this module (``BloomFirstLayer`` is the sole input encoder); it is
kept so the released weights load completely and the decoder can be exposed later
(docs/architecture.md section 6).
"""

from __future__ import annotations

import torch
from torch import nn

from bloom.config import ModelConfig

__all__ = [
    "DESCRIPTOR_PAD",
    "DESCRIPTOR_UNUSED",
    "DescriptorEncoder",
    "FeatureEmbedding",
    "MeasurementEncoder",
    "MetadataEncoder",
]

#: ``theta`` descriptor-type codes of the training data (``DescriptorType`` enum).
DESCRIPTOR_PAD: int = 0
DESCRIPTOR_UNUSED: int = 1  # gene and peak tokens


class DescriptorEncoder(nn.Module):
    """MLP on ``cseq`` for real tokens; pads encode to zeros."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.mlp_cseq = nn.Sequential(
            nn.Linear(config.d_cseq, config.d_descriptor),
            nn.GELU(),
            nn.Linear(config.d_descriptor, config.d_descriptor),
        )

    def forward(self, cseq: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """``cseq (B, L, d_cseq)``, ``theta (B, L)`` -> ``(B, L, d_descriptor)``."""
        B, L, _ = cseq.shape
        output = torch.zeros(B, L, self.config.d_descriptor, device=cseq.device)
        seq_mask = theta.long() == DESCRIPTOR_UNUSED  # (B, L)
        if seq_mask.any():
            output[seq_mask] = self.mlp_cseq(cseq[seq_mask]).float()
        return output


class MetadataEncoder(nn.Module):
    """Embeddings of the categorical metadata (``ftype``, ``mtype``, ``batch``)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        d_each = config.d_metadata_each
        self.ftype_emb = nn.Embedding(config.n_ftype, d_each)
        self.mtype_emb = nn.Embedding(config.n_mtype, d_each)
        self.batch_emb = nn.Embedding(config.n_batch, d_each)

    def forward(
        self, ftype: torch.Tensor, mtype: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """Three ``(B, L)`` index tensors -> ``(B, L, d_metadata)``."""
        return torch.cat(
            [
                self.ftype_emb(ftype.long()),
                self.mtype_emb(mtype.long()),
                self.batch_emb(batch.long()),
            ],
            dim=-1,
        )


class MeasurementEncoder(nn.Module):
    """MLPs on the global / local measurement scalars; pads use a learned embedding."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        d_each = config.d_measurement_each
        self.mlp_global = nn.Sequential(nn.Linear(1, d_each), nn.GELU(), nn.Linear(d_each, d_each))
        self.mlp_local = nn.Sequential(nn.Linear(1, d_each), nn.GELU(), nn.Linear(d_each, d_each))
        self.pad_emb = nn.Parameter(torch.randn(d_each))

    def forward(
        self, cglobal: torch.Tensor, clocal: torch.Tensor, theta: torch.Tensor
    ) -> torch.Tensor:
        """``cglobal, clocal, theta (B, L)`` -> ``(B, L, d_measurement)`` = ``[global, local]``."""
        B, L = cglobal.shape
        d_each = self.config.d_measurement_each
        device = cglobal.device
        theta_long = theta.long()

        h_global = torch.zeros(B, L, d_each, device=device)
        h_local = torch.zeros(B, L, d_each, device=device)

        meas_mask = theta_long == DESCRIPTOR_UNUSED
        if meas_mask.any():
            h_global[meas_mask] = self.mlp_global(cglobal[meas_mask].unsqueeze(-1)).float()
            h_local[meas_mask] = self.mlp_local(clocal[meas_mask].unsqueeze(-1)).float()

        pad_mask = theta_long == DESCRIPTOR_PAD
        if pad_mask.any():
            pad = self.pad_emb.expand(int(pad_mask.sum()), -1).float()
            h_global[pad_mask] = pad
            h_local[pad_mask] = pad

        return torch.cat([h_global, h_local], dim=-1)


class FeatureEmbedding(nn.Module):
    """Descriptor + metadata + measurement encoders, each LayerNorm'ed, concatenated.

    Output layout follows ``ModelConfig.component_slices``.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.descriptor_enc = DescriptorEncoder(config)
        self.metadata_enc = MetadataEncoder(config)
        self.measurement_enc = MeasurementEncoder(config)
        self.ln_desc = nn.LayerNorm(config.d_descriptor)
        self.ln_ftype = nn.LayerNorm(config.d_metadata_each)
        self.ln_mtype = nn.LayerNorm(config.d_metadata_each)
        self.ln_batch = nn.LayerNorm(config.d_metadata_each)
        self.ln_global = nn.LayerNorm(config.d_measurement_each)
        self.ln_local = nn.LayerNorm(config.d_measurement_each)

    @staticmethod
    def unpack(h: torch.Tensor, config: ModelConfig) -> dict[str, torch.Tensor]:
        """Views of the concatenated embedding, keyed as in ``config.component_slices``."""
        return {name: h[..., a:b] for name, (a, b) in config.component_slices.items()}

    def forward(
        self,
        cseq: torch.Tensor,
        theta: torch.Tensor,
        ftype: torch.Tensor,
        mtype: torch.Tensor,
        batch: torch.Tensor,
        cglobal: torch.Tensor,
        clocal: torch.Tensor,
    ) -> torch.Tensor:
        """Embed query tokens.

        Args:
            cseq: ``(B, L, d_cseq)`` sequence embeddings.
            theta: ``(B, L)`` descriptor type (``DESCRIPTOR_UNUSED`` for real tokens, 0 at pads).
            ftype: ``(B, L)`` feature type; ``mtype``: modality type; ``batch``: dataset type.
            cglobal: ``(B, L)`` per-cell global measurement (log1p total counts).
            clocal: ``(B, L)`` per-token measurement.

        Returns:
            ``(B, L, d_query)`` with ``d_query = d_descriptor + d_metadata + d_measurement``.
        """
        h_desc = self.ln_desc(self.descriptor_enc(cseq, theta))
        h_meta = self.metadata_enc(ftype, mtype, batch)
        h_meas = self.measurement_enc(cglobal, clocal, theta)

        d_each = self.config.d_metadata_each
        h_ftype = self.ln_ftype(h_meta[..., :d_each])
        h_mtype = self.ln_mtype(h_meta[..., d_each : 2 * d_each])
        h_batch = self.ln_batch(h_meta[..., 2 * d_each :])

        m_each = self.config.d_measurement_each
        h_global = self.ln_global(h_meas[..., :m_each])
        h_local = self.ln_local(h_meas[..., m_each:])

        return torch.cat([h_desc, h_ftype, h_mtype, h_batch, h_global, h_local], dim=-1)
