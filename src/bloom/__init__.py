"""Bloom: inference-only cell embeddings from single-cell RNA / ATAC AnnData.

Public API: ``load_model``, ``prepare``, ``embed``, ``strategies``::

    import bloom
    model = bloom.load_model("hyacinth", device="auto")
    ds = bloom.prepare(rna=adata, atac=atac_adata, species="mouse", reference=model.reference)
    emb = bloom.embed(model, ds)            # AnnData with obsm["X_bloom_rna"] and/or obsm["X_bloom_atac"]

CLI: ``bloom info``, ``bloom download``, ``bloom prepare``, ``bloom embed``. Modules:
``bloom.tokens`` (containers), ``bloom.strategies`` (views), ``bloom.prepare`` (mapping),
``bloom.config`` / ``bloom.model`` (``Bloom``, ``load_pretrained``), ``bloom.embed`` (the loop).

ML track (docs/internals.md): ``model.encode(..., return_internals=True)`` -> ``EncodeOutput``,
``bloom.pair_logits(out, pairs, group)``, ``bloom.capture(model, names)``.
"""

__version__ = "0.1.0"

from bloom import strategies, tokens
from bloom.config import AttentionConfig, ModelConfig
from bloom.embed import embed, load_model
from bloom.model.internals import EncodeOutput, capture, encode_with_internals, pair_logits
from bloom.prepare import PreparedDataset, prepare

__all__ = [
    "AttentionConfig",
    "EncodeOutput",
    "ModelConfig",
    "PreparedDataset",
    "__version__",
    "capture",
    "embed",
    "encode_with_internals",
    "load_model",
    "pair_logits",
    "prepare",
    "strategies",
    "tokens",
]
