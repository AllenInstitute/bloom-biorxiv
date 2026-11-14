"""Decoder heads (weights are loaded; not exposed by the v1 CLI).

Port of ``bloom/model/decoder.py`` (alleninstitute/bloom@141f0d3) without the
``torch.distributions`` wrapper: heads return Bernoulli logits. The reconstruction API is a
roadmap item (docs/architecture.md section 6); this module exists so the released state dict
loads with ``strict=True`` and the numerics are available when that API lands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import nn

from bloom.config import ModelConfig
from bloom.model.embedding import FeatureEmbedding

__all__ = ["BernoulliDecoderHead", "DecoderInput", "MultiDecoderHead"]

#: Feature / modality type codes of the training data (``FeatureType``, ``ModalityType``).
FTYPE_GENE: int = 1
FTYPE_PEAK: int = 2
MTYPE_SEQUENCING: int = 1


@dataclass
class DecoderInput:
    """Per-query-token decoder inputs (``h_local`` deliberately excluded: it is the target).

    All tensors share a leading dimension ``N`` = number of selected (cell, query) pairs.
    """

    h_cell: torch.Tensor  # (N, d_model)
    h_desc: torch.Tensor  # (N, d_descriptor)
    h_ftype: torch.Tensor  # (N, d_metadata_each)
    h_mtype: torch.Tensor  # (N, d_metadata_each)
    h_batch: torch.Tensor  # (N, d_metadata_each)
    h_global: torch.Tensor  # (N, d_measurement_each)
    ftype: torch.Tensor  # (N,)
    mtype: torch.Tensor  # (N,)
    batch: torch.Tensor  # (N,)


class BernoulliDecoderHead(nn.Module):
    """PeakVI-style factorized detection probability ``p = s(a) * s(b) * s(c)``.

    ``a = f_bio(h_cell, h_desc)`` (cell x feature), ``b = f_depth(h_global)`` (cell depth),
    ``c = f_tech(h_ftype, h_mtype, h_batch)`` (technical). The logit of ``p`` is computed in
    log-space with a ``logsumexp`` for ``log(1 - p)`` (numerically stable in bf16).

    Args:
        config: Model configuration (component widths).
        hidden_dim: Hidden width of the three MLPs (64 for ``hyacinth``).
    """

    def __init__(self, config: ModelConfig, hidden_dim: int = 256) -> None:
        super().__init__()
        self.mlp_bio = nn.Sequential(
            nn.Linear(config.d_model + config.d_descriptor, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.mlp_depth = nn.Sequential(
            nn.Linear(config.d_measurement_each, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.mlp_tech = nn.Sequential(
            nn.Linear(3 * config.d_metadata_each, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, inputs: DecoderInput) -> torch.Tensor:
        """Return Bernoulli logits ``(N,)`` of feature detection."""
        a = self.mlp_bio(torch.cat([inputs.h_cell, inputs.h_desc], dim=-1)).squeeze(-1)  # (N,)
        b = self.mlp_depth(inputs.h_global).squeeze(-1)  # (N,)
        c = self.mlp_tech(
            torch.cat([inputs.h_ftype, inputs.h_mtype, inputs.h_batch], dim=-1)
        ).squeeze(-1)  # (N,)

        log_pa, log_pb, log_pc = F.logsigmoid(a), F.logsigmoid(b), F.logsigmoid(c)
        log_qa, log_qb, log_qc = F.logsigmoid(-a), F.logsigmoid(-b), F.logsigmoid(-c)
        log_p = log_pa + log_pb + log_pc  # (N,)
        log_1mp = torch.logsumexp(
            torch.stack([log_qa, log_pa + log_qb, log_pa + log_pb + log_qc], dim=0), dim=0
        )  # (N,)
        return log_p - log_1mp  # (N,)


class MultiDecoderHead(nn.Module):
    """One ``BernoulliDecoderHead`` per ``(ftype, mtype)``: ``1_1`` genes, ``2_1`` peaks.

    Args:
        config: Model configuration.
        hidden_dim: Hidden width of each head.
    """

    HEAD_KEYS: ClassVar[list[tuple[int, int]]] = [
        (FTYPE_GENE, MTYPE_SEQUENCING),
        (FTYPE_PEAK, MTYPE_SEQUENCING),
    ]

    def __init__(self, config: ModelConfig, hidden_dim: int = 256) -> None:
        super().__init__()
        self.config = config
        self.heads = nn.ModuleDict()
        for ftype, mtype in self.HEAD_KEYS:
            self.heads[self._make_key(ftype, mtype)] = BernoulliDecoderHead(config, hidden_dim)

    @staticmethod
    def _make_key(ftype: int, mtype: int) -> str:
        return f"{ftype}_{mtype}"

    def _build_decoder_input(
        self,
        h_cell: torch.Tensor,
        h_query: torch.Tensor,
        ftype: torch.Tensor,
        mtype: torch.Tensor,
        batch: torch.Tensor,
        mask: torch.Tensor,
    ) -> DecoderInput:
        components = FeatureEmbedding.unpack(h_query, self.config)
        B, Lq = mask.shape
        h_cell_expanded = h_cell.unsqueeze(1).expand(B, Lq, -1)  # (B, Lq, d_model)
        return DecoderInput(
            h_cell=h_cell_expanded[mask],
            h_desc=components["desc"][mask],
            h_ftype=components["h_ftype"][mask],
            h_mtype=components["h_mtype"][mask],
            h_batch=components["h_batch"][mask],
            h_global=components["global"][mask],  # 'local' excluded: it is the target
            ftype=ftype[mask],
            mtype=mtype[mask],
            batch=batch[mask],
        )

    def forward(
        self,
        h_cell: torch.Tensor,
        h_query: torch.Tensor,
        ftype: torch.Tensor,
        mtype: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Route query tokens to their head.

        Args:
            h_cell: ``(B, d_model)`` cell embeddings.
            h_query: ``(B, Lq, d_query)`` query embeddings from ``FeatureEmbedding``.
            ftype: ``(B, Lq)`` feature type; ``mtype``: modality type; ``batch``: dataset type.

        Returns:
            ``(logits, valid_mask)``: ``(B, Lq)`` detection logits (0 where no head applies) and
            ``(B, Lq)`` bool mask of positions a head produced.
        """
        B, Lq, _ = h_query.shape
        ftype_l, mtype_l, batch_l = ftype.long(), mtype.long(), batch.long()
        logits = torch.zeros(B, Lq, device=h_query.device, dtype=h_query.dtype)
        valid_mask = torch.zeros(B, Lq, dtype=torch.bool, device=h_query.device)
        for ftype_val, mtype_val in self.HEAD_KEYS:
            mask = (ftype_l == ftype_val) & (mtype_l == mtype_val)  # (B, Lq)
            if mask.any():
                inputs = self._build_decoder_input(h_cell, h_query, ftype_l, mtype_l, batch_l, mask)
                logits[mask] = self.heads[self._make_key(ftype_val, mtype_val)](inputs)
                valid_mask[mask] = True
        return logits, valid_mask
