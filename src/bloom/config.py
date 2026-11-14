"""Model configuration dataclasses, serialized as ``config.json`` (docs/assets.md section 4).

``ModelConfig`` carries every architectural scalar needed to rebuild ``hyacinth`` without the
training repository, including the sequence-head attention scale (1/sqrt(32)) that the training
checkpoint does not store. The dataclasses mirror ``EmbeddingConfig`` / ``BloomAttentionConfig``
in ``alleninstitute/bloom`` (``bloom/model/components.py``) but drop training-only fields.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "SEQ_HEAD_SCALE_HYACINTH",
    "AttentionConfig",
    "ModelConfig",
]

#: Sequence-head attention scale used by the released ``hyacinth`` weights: 1/sqrt(d_v) with d_v = 32.
#: ``bloom-arch`` commit ef5fb8a hard-codes this value; later commits use 1/sqrt(768) (AGENTS.md section 6).
SEQ_HEAD_SCALE_HYACINTH: float = 1.0 / math.sqrt(32.0)

_ARCH_KEYS = (
    "d_cseq",
    "d_descriptor",
    "d_metadata",
    "d_measurement",
    "d_model",
    "n_latents",
    "n_transformer_layers",
    "n_heads",
    "decoder_hidden_dim",
    "dropout",
)
_HPARAM_KEYS = ("n_ftype", "n_mtype", "n_batch")
_META_KEYS = (
    "model_id",
    "reference_id",
    "config_version",
    "state_dict_remap",
    "training_context",
    "strategy_defaults",
    "weights",
    "source",
    "parameters",
)


def _check_known_keys(d: dict[str, Any], allowed: set[str], what: str) -> None:
    unknown = sorted(set(d) - allowed)
    if unknown:
        raise ValueError(f"Unknown {what} keys in config: {unknown}; allowed: {sorted(allowed)}")


@dataclass
class AttentionConfig:
    """Hyper-parameters of ``bloom_arch.attention.BloomAttentionBlock`` (the first layer).

    Attributes:
        d_val: Width of the encoded measurement value fed to the block.
        d_u: Chromosome-codebook (baseline) width of the positional encoding.
        K: Number of Fourier wavelengths; the positional encoding has ``2 * K`` dimensions.
        d_v: Per-head query/key/value width, shared by all head types.
        n_sequence_heads: Heads attending by DNA-sequence similarity (Q, K from ``cseq``).
        n_position_heads: Heads attending by genomic proximity (Q = K = positional encoding).
        n_standard_heads: Heads attending over the full token feature.
        max_G: Maximum number of distinct chromosome indices in the codebook.
        lambda_min: Finest Fourier wavelength in base pairs.
        lambda_max: Coarsest Fourier wavelength in base pairs.
        target_scales: Optional per-position-head centre wavelengths (``None`` = log-spaced).
        rho: Log-Gaussian spectral bandwidth of the position heads.
        beta_init: Initial same-chromosome baseline strength.
        tau_init: Initial attention temperature of the position heads.
        learn_tau: Whether ``tau`` is a learnable parameter (changes the state-dict key set).
        pad_v_for_flash: Pad V in the position heads for flash dispatch (numerically neutral).
        seed: Seed of the deterministic chromosome codebook.
        seq_head_scale: Softmax scale of the sequence heads. ``bloom-arch`` derives it from
            ``d_v`` at construction time; it is stored here so ``load_pretrained`` can assert the
            installed ``bloom-arch`` matches the weights (test M3).
    """

    d_val: int = 40
    d_u: int = 32
    K: int = 64
    d_v: int = 32
    n_sequence_heads: int = 8
    n_position_heads: int = 8
    n_standard_heads: int = 4
    max_G: int = 32  # ruff: ignore[mixed-case-variable-in-class-scope]
    lambda_min: float = 100.0
    lambda_max: float = 1.0e8
    target_scales: list[float] | None = None
    rho: float = 1.15
    beta_init: float = 0.0
    tau_init: float = 1.0
    learn_tau: bool = False
    pad_v_for_flash: bool = False
    seed: int = 42
    seq_head_scale: float = SEQ_HEAD_SCALE_HYACINTH

    def __post_init__(self) -> None:
        if self.n_sequence_heads + self.n_position_heads + self.n_standard_heads <= 0:
            raise ValueError("At least one attention head is required")
        if self.seq_head_scale <= 0:
            raise ValueError(f"seq_head_scale must be positive, got {self.seq_head_scale}")

    @property
    def n_heads_total(self) -> int:
        """Total number of first-layer heads (20 for ``hyacinth``)."""
        return self.n_sequence_heads + self.n_position_heads + self.n_standard_heads

    @property
    def head_groups(self) -> dict[str, int]:
        """Heads per group in the order the block concatenates them: ``seq``, ``pos``, ``std``."""
        return {
            "seq": self.n_sequence_heads,
            "pos": self.n_position_heads,
            "std": self.n_standard_heads,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AttentionConfig:
        """Build from the ``architecture.attention`` mapping of ``config.json``."""
        allowed = {f.name for f in dataclasses.fields(cls)}
        _check_known_keys(d, allowed, "attention")
        return cls(**d)

    def to_dict(self) -> dict[str, Any]:
        """Inverse of :meth:`from_dict`."""
        return dataclasses.asdict(self)


@dataclass
class ModelConfig:
    """Architecture and provenance of a released Bloom model (``config.json``).

    Attributes:
        d_cseq: Width of the reference sequence embeddings (Evo2 PCA, 768).
        d_descriptor: Query-side descriptor width (``feature_emb``; decoder only).
        d_metadata: Query-side metadata width; must be divisible by 3 (ftype / mtype / batch).
        d_measurement: Query-side measurement width; must be divisible by 2 (global / local).
        d_model: Token / latent / cell-embedding width (1024). Independent of the query-side
            component widths (``d_model_override`` in the training code).
        n_latents: Number of Perceiver latents (256).
        n_transformer_layers: Self-attention layers on the latents (8).
        n_heads: Heads of the Perceiver cross- and self-attention (8).
        decoder_hidden_dim: Hidden width of the Bernoulli decoder heads (64).
        dropout: Dropout rate used at training time (identity in ``eval()``).
        n_ftype: Feature-type vocabulary size of the query metadata embedding (PAD, GENE, PEAK).
        n_mtype: Modality-type vocabulary size (PAD, SEQUENCING).
        n_batch: Batch-type vocabulary size (PAD, SCRNASEQ, MULTIOME, SNRNASEQ).
        attention: First-layer attention block hyper-parameters.
        model_id: Released model name (``hyacinth``).
        reference_id: Reference-table id the model was trained against (``mm10-evo2-v1``).
        config_version: Schema version of ``config.json``.
        state_dict_remap: Key-prefix remap applied at export (training name -> package name).
        training_context: Training facts needed to interpret inputs (``Lmax``, binarization, ...).
        strategy_defaults: Paper-default view strategies per modality.
        weights: ``{"file", "size_bytes", "sha256"}`` of the weights next to the config.
        source: Checkpoint provenance (run, step, commits).
        parameters: Parameter counts recorded at export (``total`` is checked by test M1).
        extra: Any other top-level ``config.json`` keys, preserved for a faithful ``to_json``.
    """

    d_cseq: int = 768
    d_descriptor: int = 128
    d_metadata: int = 48
    d_measurement: int = 80
    d_model: int = 1024
    n_latents: int = 256
    n_transformer_layers: int = 8
    n_heads: int = 8
    decoder_hidden_dim: int = 64
    dropout: float = 0.1
    n_ftype: int = 3
    n_mtype: int = 2
    n_batch: int = 4
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    model_id: str = "hyacinth"
    reference_id: str = "mm10-evo2-v1"
    config_version: str = "1.0"
    state_dict_remap: dict[str, str] = field(default_factory=dict)
    training_context: dict[str, Any] = field(default_factory=dict)
    strategy_defaults: dict[str, Any] = field(default_factory=dict)
    weights: dict[str, Any] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.attention, dict):
            self.attention = AttentionConfig.from_dict(self.attention)
        if self.d_metadata % 3 != 0:
            raise ValueError(f"d_metadata ({self.d_metadata}) must be divisible by 3")
        if self.d_measurement % 2 != 0:
            raise ValueError(f"d_measurement ({self.d_measurement}) must be divisible by 2")
        if self.d_model <= 0:
            raise ValueError(f"d_model ({self.d_model}) must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )

    # ---- derived widths (shared with FeatureEmbedding / decoder) -------------------------------
    @property
    def d_metadata_each(self) -> int:
        """Width of each metadata embedding (``d_metadata / 3``)."""
        return self.d_metadata // 3

    @property
    def d_measurement_each(self) -> int:
        """Width of each measurement component (``d_measurement / 2``)."""
        return self.d_measurement // 2

    @property
    def d_query(self) -> int:
        """Width of a query-side token embedding: ``d_descriptor + d_metadata + d_measurement``."""
        return self.d_descriptor + self.d_metadata + self.d_measurement

    @property
    def component_slices(self) -> dict[str, tuple[int, int]]:
        """Slice boundaries of the concatenated query embedding (see ``FeatureEmbedding``)."""
        d = self.d_descriptor
        me = self.d_metadata_each
        ms = self.d_measurement_each
        return {
            "desc": (0, d),
            "h_ftype": (d, d + me),
            "h_mtype": (d + me, d + 2 * me),
            "h_batch": (d + 2 * me, d + 3 * me),
            "global": (d + 3 * me, d + 3 * me + ms),
            "local": (d + 3 * me + ms, d + 3 * me + 2 * ms),
        }

    @property
    def seq_head_scale(self) -> float:
        """Shortcut for ``attention.seq_head_scale``."""
        return self.attention.seq_head_scale

    @property
    def n_parameters_expected(self) -> int | None:
        """Parameter count recorded at export (``parameters.total``), if present."""
        total = self.parameters.get("total")
        return int(total) if total is not None else None

    @property
    def lmax_training(self) -> int | None:
        """``Lmax`` used at training time (8192 for ``hyacinth``), if recorded."""
        v = self.training_context.get("Lmax")
        return int(v) if v is not None else None

    # ---- (de)serialization --------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelConfig:
        """Build from the parsed ``config.json`` mapping (docs/assets.md section 4)."""
        d = dict(d)
        arch = dict(d.pop("architecture", {}))
        attention = arch.pop("attention", {})
        _check_known_keys(arch, set(_ARCH_KEYS), "architecture")
        hparams = dict(d.pop("embedding_config_hparams", {}))
        kwargs: dict[str, Any] = dict(arch)
        for k in _HPARAM_KEYS:
            if k in hparams:
                kwargs[k] = hparams[k]
        for k in _META_KEYS:
            if k in d:
                kwargs[k] = d.pop(k)
        return cls(attention=AttentionConfig.from_dict(attention), extra=d, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Inverse of :meth:`from_dict`; reproduces the ``config.json`` layout."""
        arch = {k: getattr(self, k) for k in _ARCH_KEYS}
        arch["attention"] = self.attention.to_dict()
        hparams = {
            "d_cseq": self.d_cseq,
            "d_descriptor": self.d_descriptor,
            "d_metadata": self.d_metadata,
            "d_measurement": self.d_measurement,
            "d_model_override": self.d_model,
            "n_ftype": self.n_ftype,
            "n_mtype": self.n_mtype,
            "n_batch": self.n_batch,
            "n_transformer_layers": self.n_transformer_layers,
            "n_latents": self.n_latents,
            "n_heads": self.n_heads,
            "dropout": self.dropout,
        }
        out: dict[str, Any] = {
            "model_id": self.model_id,
            "reference_id": self.reference_id,
            "config_version": self.config_version,
            "architecture": arch,
            "embedding_config_hparams": hparams,
        }
        for k in _META_KEYS[3:]:
            out[k] = getattr(self, k)
        out.update(self.extra)
        return out

    @classmethod
    def from_json(cls, path: str | Path) -> ModelConfig:
        """Read ``config.json``."""
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_json(self, path: str | Path, indent: int = 2) -> None:
        """Write ``config.json``."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent)
            f.write("\n")
