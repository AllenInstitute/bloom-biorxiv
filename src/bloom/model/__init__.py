"""``bloom.model``: the Bloom inference model and its components (docs/architecture.md 6.1)."""

from bloom.model.bloom import (
    FASTPATH_ENABLED,
    KIND_GENE,
    KIND_PAD,
    KIND_PEAK,
    REFERENCE_BUFFERS,
    Bloom,
    MixedModalityViewError,
    SeqHeadScaleMismatchError,
    build_from_config,
    check_seq_head_scale,
    check_single_modality,
    fastpath_disabled,
    inverse_state_dict_remap,
    load_pretrained,
    remap_state_dict,
    resolve_device,
)
from bloom.model.decoder import BernoulliDecoderHead, MultiDecoderHead
from bloom.model.embedding import FeatureEmbedding
from bloom.model.first_layer import BloomFirstLayer
from bloom.model.internals import (
    EncodeOutput,
    InternalsMemoryError,
    capture,
    encode_with_internals,
    pair_logits,
)
from bloom.model.perceiver import PerceiverEncoder

__all__ = [
    "FASTPATH_ENABLED",
    "KIND_GENE",
    "KIND_PAD",
    "KIND_PEAK",
    "REFERENCE_BUFFERS",
    "BernoulliDecoderHead",
    "Bloom",
    "BloomFirstLayer",
    "EncodeOutput",
    "FeatureEmbedding",
    "InternalsMemoryError",
    "MixedModalityViewError",
    "MultiDecoderHead",
    "PerceiverEncoder",
    "SeqHeadScaleMismatchError",
    "build_from_config",
    "capture",
    "check_seq_head_scale",
    "check_single_modality",
    "encode_with_internals",
    "fastpath_disabled",
    "inverse_state_dict_remap",
    "load_pretrained",
    "pair_logits",
    "remap_state_dict",
    "resolve_device",
]
