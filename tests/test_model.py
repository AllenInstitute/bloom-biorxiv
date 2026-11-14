"""Model construction and loading tests M1-M7 (docs/test-plan.md section 1) + config round trip."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from bloom.config import AttentionConfig, ModelConfig
from bloom.model import (
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
)

HYACINTH_N_PARAMS = 124_144_831
TRAINING_PREFIXES = ("bloom_first_layer", "encoder", "decoder", "feature_emb")


# ------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------
def _write_gene_table(path: Path, n_rows: int, d_cseq: int, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    table = {
        "cseq": torch.randn(n_rows, d_cseq, generator=g),
        "chr": torch.randint(0, 4, (n_rows,), generator=g).to(torch.int8),
        "bp": torch.randint(0, 10_000_000, (n_rows,), generator=g).to(torch.int32),
    }
    save_file(table, str(path))
    return table


def _tokens(rows: list[list[int]], L: int, kind_code: int = KIND_GENE):
    """Left-packed ``(emb_idx, kind, lengths)`` for a list of per-cell reference rows."""
    B = len(rows)
    emb_idx = torch.full((B, L), -1, dtype=torch.int32)
    kind = torch.full((B, L), KIND_PAD, dtype=torch.int8)
    lengths = torch.zeros(B, dtype=torch.int64)
    for b, r in enumerate(rows):
        emb_idx[b, : len(r)] = torch.tensor(r, dtype=torch.int32)
        kind[b, : len(r)] = kind_code
        lengths[b] = len(r)
    return emb_idx, kind, lengths


def _save_tiny_model(tmp_path: Path, config: ModelConfig, seed: int = 0) -> tuple[Path, Bloom]:
    torch.manual_seed(seed)
    model = build_from_config(config)
    save_file(
        {k: v.contiguous() for k, v in model.state_dict().items()},
        str(tmp_path / "model.safetensors"),
    )
    config.to_json(tmp_path / "config.json")
    return tmp_path, model


# ------------------------------------------------------------------------------------------
# config
# ------------------------------------------------------------------------------------------
@pytest.mark.unit
def test_config_roundtrip_real_file(hyacinth_config_path):
    raw = json.loads(hyacinth_config_path.read_text())
    cfg = ModelConfig.from_json(hyacinth_config_path)
    assert cfg.model_id == "hyacinth" and cfg.reference_id == "mm10-evo2-v1"
    assert (cfg.d_cseq, cfg.d_model, cfg.n_latents, cfg.n_transformer_layers) == (768, 1024, 256, 8)
    assert (cfg.n_ftype, cfg.n_mtype, cfg.n_batch) == (3, 2, 4)
    assert cfg.attention.head_groups == {"seq": 8, "pos": 8, "std": 4}
    assert cfg.seq_head_scale == pytest.approx(1 / math.sqrt(32), rel=0, abs=0)
    assert cfg.n_parameters_expected == HYACINTH_N_PARAMS
    assert cfg.lmax_training == 8192
    assert cfg.component_slices["local"] == (128 + 48 + 40, 128 + 48 + 80)
    # to_dict reproduces the file (including keys we do not model, kept in `extra`)
    assert cfg.to_dict() == raw


@pytest.mark.unit
def test_config_json_roundtrip_tmp(tiny_config, tmp_path):
    p = tmp_path / "config.json"
    tiny_config.to_json(p)
    again = ModelConfig.from_json(p)
    assert again == tiny_config


@pytest.mark.unit
def test_config_validation():
    with pytest.raises(ValueError, match="d_metadata"):
        ModelConfig(d_metadata=7)
    with pytest.raises(ValueError, match="d_measurement"):
        ModelConfig(d_measurement=3)
    with pytest.raises(ValueError, match="n_heads"):
        ModelConfig(d_model=30, n_heads=8)
    with pytest.raises(ValueError, match="Unknown attention keys"):
        AttentionConfig.from_dict({"d_val": 40, "bogus": 1})
    with pytest.raises(ValueError, match="Unknown architecture keys"):
        ModelConfig.from_dict({"architecture": {"d_model": 1024, "bogus": 1}})


# ------------------------------------------------------------------------------------------
# M1 parameter count
# ------------------------------------------------------------------------------------------
@pytest.mark.unit
def test_m1_parameter_count(hyacinth_config_path):
    cfg = ModelConfig.from_json(hyacinth_config_path)
    model = build_from_config(cfg)
    assert model.n_parameters == HYACINTH_N_PARAMS
    per_module = {
        "feature_emb": model.feature_emb,
        "bloom_first_layer": model.first_layer,
        "encoder": model.encoder,
        "decoder": model.decoder,
    }
    counts = {k: sum(p.numel() for p in m.parameters()) for k, m in per_module.items()}
    assert counts == cfg.parameters["per_module"]
    # the state dict has the 198 tensors of the export: parameters + 7 persistent bloom-arch buffers
    sd = model.state_dict()
    assert len(sd) == cfg.parameters["n_state_dict_tensors"] == 198
    assert sum(v.numel() for v in sd.values()) == cfg.parameters["n_state_dict_elements"]
    persistent = {k for k in sd if k not in dict(model.named_parameters())}
    expected = {
        k.replace("bloom_first_layer.", "first_layer.")
        for k in cfg.parameters["persistent_buffers"]
    }
    assert persistent == expected


# ------------------------------------------------------------------------------------------
# M2 strict load of the released weights (assets tier: needs model.safetensors)
# ------------------------------------------------------------------------------------------
@pytest.mark.assets
def test_m2_strict_load_hyacinth(hyacinth_model_dir, recwarn):
    from safetensors import safe_open

    model = load_pretrained(hyacinth_model_dir, device="cpu", strict=True)
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]
    with safe_open(str(hyacinth_model_dir / "model.safetensors"), "pt") as f:
        keys = set(f.keys())
        probe = "first_layer.attn_block.seq_heads.W_Q.weight"
        assert torch.equal(model.state_dict()[probe], f.get_tensor(probe))
    assert keys == set(model.state_dict())
    assert model.n_parameters == HYACINTH_N_PARAMS
    assert not model.training
    assert check_seq_head_scale(model) == pytest.approx(1 / math.sqrt(32), rel=0, abs=0)


# ------------------------------------------------------------------------------------------
# M3 sequence-head scale assertion
# ------------------------------------------------------------------------------------------
@pytest.mark.unit
def test_m3_scale_assertion(hyacinth_config_path, tiny_config, tmp_path):
    cfg = ModelConfig.from_json(hyacinth_config_path)
    model = build_from_config(cfg)
    assert model.first_layer.attn_block.seq_heads._scale == cfg.attention.seq_head_scale
    assert check_seq_head_scale(model) == 1 / math.sqrt(32)

    # tampered config: the mask-dev scale 1/sqrt(768)
    raw = json.loads(hyacinth_config_path.read_text())
    raw["architecture"]["attention"]["seq_head_scale"] = 1 / math.sqrt(768)
    bad = ModelConfig.from_dict(raw)
    with pytest.raises(SeqHeadScaleMismatchError, match=re.escape("1/sqrt(32)")):
        build_from_config(bad)

    # load_pretrained raises before reading any weights (the weights file is a decoy)
    model_dir, _ = _save_tiny_model(tmp_path, tiny_config)
    raw = json.loads((model_dir / "config.json").read_text())
    raw["architecture"]["attention"]["seq_head_scale"] = (
        0.5 * raw["architecture"]["attention"]["seq_head_scale"]
    )
    (model_dir / "config.json").write_text(json.dumps(raw))
    with pytest.raises(SeqHeadScaleMismatchError):
        load_pretrained(model_dir)


# ------------------------------------------------------------------------------------------
# M4 (reduced): state-dict key remap round trip
# ------------------------------------------------------------------------------------------
@pytest.mark.unit
def test_m4_state_dict_remap_roundtrip(hyacinth_config_path):
    cfg = ModelConfig.from_json(hyacinth_config_path)
    remap = cfg.state_dict_remap
    inverse = inverse_state_dict_remap(remap)
    model = build_from_config(cfg)
    sd = model.state_dict()
    training_sd = remap_state_dict(sd, inverse)
    assert all(k.startswith("model.") for k in training_sd)
    assert {k.split(".")[1] for k in training_sd} == set(TRAINING_PREFIXES)
    back = remap_state_dict(training_sd, remap)
    assert list(back) == list(sd)
    assert all(back[k] is sd[k] for k in sd)  # same tensors, only names changed
    # idempotent on already-remapped keys; the training `model.` prefix is required for a hit
    assert remap_state_dict(sd, remap).keys() == sd.keys()
    assert remap_state_dict({"probe.weight": torch.zeros(1)}, remap) == {
        "probe.weight": torch.zeros(1)
    }


# ------------------------------------------------------------------------------------------
# M5 non-persistent reference buffers
# ------------------------------------------------------------------------------------------
@pytest.mark.unit
def test_m5_reference_buffers_not_in_state_dict(tiny_config, tmp_path):
    model = build_from_config(tiny_config)
    table = _write_gene_table(tmp_path / "genes.safetensors", 10, tiny_config.d_cseq)
    model.load_gene_table(tmp_path / "genes.safetensors")
    model.load_peak_table({
        "peak_cseq": table["cseq"],
        "peak_chr": table["chr"],
        "peak_bp": table["bp"],
    })
    buffers = dict(model.named_buffers())
    assert set(REFERENCE_BUFFERS) <= set(buffers)
    sd = model.state_dict()
    assert not [k for k in sd if re.search(r"(_cseq|_chr|_bp)$", k)]
    assert not [k for k in sd if k in REFERENCE_BUFFERS]
    # the tables are still usable after a state-dict round trip
    model2 = build_from_config(tiny_config)
    model2.load_state_dict(sd, strict=True)
    assert not model2.has_gene_table and not model2.has_peak_table


# ------------------------------------------------------------------------------------------
# M6 lazy peak table
# ------------------------------------------------------------------------------------------
@pytest.mark.unit
def test_m6_genes_only_never_opens_peaks(tiny_config, tmp_path, monkeypatch):
    import bloom.model.bloom as bloom_mod

    model_dir, _ = _save_tiny_model(tmp_path, tiny_config)
    genes_path = tmp_path / "genes.safetensors"
    peaks_path = tmp_path / "peaks.safetensors"
    _write_gene_table(genes_path, 10, tiny_config.d_cseq, seed=1)
    _write_gene_table(peaks_path, 10, tiny_config.d_cseq, seed=2)

    opened: list[str] = []
    real_safe_open = bloom_mod.safe_open

    def spy(path, *args, **kwargs):
        opened.append(str(path))
        if str(path).endswith("peaks.safetensors"):
            raise AssertionError("peaks.safetensors must not be opened for a gene-only run")
        return real_safe_open(path, *args, **kwargs)

    monkeypatch.setattr(bloom_mod, "safe_open", spy)
    model = load_pretrained(model_dir, device="cpu", genes=genes_path)
    emb_idx, kind, lengths = _tokens([[0, 3, 5], [1]], L=4)
    with torch.inference_mode():
        h = model.encode(emb_idx, kind, lengths)
    assert h.shape == (2, tiny_config.d_model) and torch.isfinite(h).all()
    assert model.has_gene_table and not model.has_peak_table
    assert not any(p.endswith("peaks.safetensors") for p in opened)
    assert any(p.endswith("genes.safetensors") for p in opened)

    # peak tokens without the table -> actionable error, still no file access
    pe, pk, pl = _tokens([[2]], L=2, kind_code=KIND_PEAK)
    with pytest.raises(RuntimeError, match="load_peak_table"):
        model.encode(pe, pk, pl)
    assert not any(p.endswith("peaks.safetensors") for p in opened)

    # explicit lazy load works and peak tokens then resolve
    monkeypatch.setattr(bloom_mod, "safe_open", real_safe_open)
    model.load_peak_table(peaks_path)
    with torch.inference_mode():
        h2 = model.encode(pe, pk, pl)
    assert torch.isfinite(h2).all()
    model.clear_peak_table()
    assert not model.has_peak_table


# ------------------------------------------------------------------------------------------
# M7 device moves
# ------------------------------------------------------------------------------------------
@pytest.mark.unit
def test_m7_device_moves(tiny_config, tmp_path):
    torch.manual_seed(0)
    model = build_from_config(tiny_config)
    _write_gene_table(tmp_path / "genes.safetensors", 12, tiny_config.d_cseq)
    model.load_gene_table(tmp_path / "genes.safetensors")
    emb_idx, kind, lengths = _tokens([[0, 1, 2, 3], [4, 5]], L=6)

    def devices(m: Bloom) -> set[str]:
        return {p.device.type for p in m.parameters()} | {
            b.device.type for b in m.buffers() if b is not None
        }

    model.to("cpu")
    assert devices(model) == {"cpu"}
    with torch.inference_mode():
        h_cpu = model.encode(emb_idx, kind, lengths)
    assert h_cpu.device.type == "cpu" and torch.isfinite(h_cpu).all()

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available: cuda half of M7 skipped")
    model.to("cuda")
    assert devices(model) == {"cuda"}
    with torch.inference_mode():
        h_gpu = model.encode(emb_idx, kind, lengths)  # inputs on CPU are moved by the model
    assert h_gpu.device.type == "cuda"
    assert torch.allclose(h_gpu.cpu(), h_cpu, atol=1e-3, rtol=0)
    model.to("cpu")
    assert devices(model) == {"cpu"}


# ------------------------------------------------------------------------------------------
# encode semantics: padding invariance (S9-style), batch invariance, empty rows
# ------------------------------------------------------------------------------------------
@pytest.mark.unit
def test_encode_padding_and_batch_invariance(tiny_config, tmp_path):
    torch.manual_seed(0)
    model = build_from_config(tiny_config)
    _write_gene_table(tmp_path / "genes.safetensors", 20, tiny_config.d_cseq)
    model.load_gene_table(tmp_path / "genes.safetensors")
    rows = [[0, 5, 7, 9, 11], [2, 3], [1, 4, 6, 8, 10, 12, 14]]
    with torch.inference_mode():
        h8 = model.encode(*_tokens(rows, L=8))
        h16 = model.encode(*_tokens(rows, L=16))
        h_single = torch.cat([model.encode(*_tokens([r], L=8)) for r in rows])
        h_nolen = model.encode(*_tokens(rows, L=8)[:2])
    assert (h8 - h16).abs().max() < 1e-5
    assert (h8 - h_single).abs().max() < 1e-5
    assert torch.equal(h8, h_nolen)
    # key_padding_mask is True exactly at pads
    emb_idx, kind, lengths = _tokens(rows, L=8)
    kpm = Bloom.key_padding_mask(lengths, 8)
    assert torch.equal(kpm, kind == KIND_PAD)
    # a zero-length row yields NaN without touching the others
    with torch.inference_mode():
        h = model.encode(*_tokens([rows[0], []], L=8))
    assert torch.isnan(h[1]).all() and torch.allclose(h[0], h8[0], atol=1e-6)


@pytest.mark.unit
def test_encode_disables_fused_fast_path_scoped(tiny_config, tmp_path, monkeypatch):
    """#44: ``encode`` runs with ``torch.backends.mha`` fast path off and restores the caller's setting."""
    import bloom.model.bloom as bloom_module

    torch.manual_seed(0)
    model = build_from_config(tiny_config)
    _write_gene_table(tmp_path / "genes.safetensors", 20, tiny_config.d_cseq)
    model.load_gene_table(tmp_path / "genes.safetensors")
    seen: list[bool] = []
    original_encoder = model.encoder.forward

    def spy(*args, **kwargs):
        seen.append(bool(torch.backends.mha.get_fastpath_enabled()))
        return original_encoder(*args, **kwargs)

    monkeypatch.setattr(model.encoder, "forward", spy)
    for caller_setting in (True, False):
        torch.backends.mha.set_fastpath_enabled(caller_setting)
        with torch.inference_mode():
            model.encode(*_tokens([[0, 5, 7]], L=8))
        assert torch.backends.mha.get_fastpath_enabled() is caller_setting  # restored
    torch.backends.mha.set_fastpath_enabled(True)
    assert seen == [False, False]
    assert bloom_module.FASTPATH_ENABLED is False
    # the context manager itself restores on exceptions too
    with pytest.raises(RuntimeError), fastpath_disabled():
        assert torch.backends.mha.get_fastpath_enabled() is False
        raise RuntimeError("x")
    assert torch.backends.mha.get_fastpath_enabled() is True


@pytest.mark.unit
def test_resolve_and_subset_table(tiny_config, tmp_path):
    model = build_from_config(tiny_config)
    full = _write_gene_table(tmp_path / "genes.safetensors", 20, tiny_config.d_cseq)
    sub_rows = torch.tensor([3, 7, 11, 19])
    save_file(
        {
            "cseq": full["cseq"][sub_rows].contiguous(),
            "chr": full["chr"][sub_rows].contiguous(),
            "bp": full["bp"][sub_rows].contiguous(),
            "emb_idx": sub_rows.to(torch.int32),
        },
        str(tmp_path / "genes_mini.safetensors"),
    )
    model.load_gene_table(tmp_path / "genes_mini.safetensors")
    assert model.gene_cseq.shape == (20, tiny_config.d_cseq)
    emb_idx, kind, _ = _tokens([[7, 19], [3]], L=3)
    cseq, chrom, bp = model.resolve(emb_idx, kind)
    assert cseq.shape == (2, 3, tiny_config.d_cseq) and chrom.dtype == torch.long
    assert torch.equal(cseq[0, 0], full["cseq"][7]) and torch.equal(cseq[1, 0], full["cseq"][3])
    assert int(chrom[0, 1]) == int(full["chr"][19])
    assert torch.equal(bp[0, 1], full["bp"][19].float())
    assert torch.equal(cseq[1, 1:], torch.zeros(2, tiny_config.d_cseq))  # pads are zero
    assert int(chrom[1, 2]) == 0 and bp[1, 2].item() == pytest.approx(0.0)
    with pytest.raises(IndexError, match="out of range"):
        model.resolve(*_tokens([[25]], L=1)[:2])
    with pytest.raises(ValueError, match="lacks"):
        model.load_gene_table({"cseq": full["cseq"]})


@pytest.mark.unit
def test_s11_mixed_gene_peak_view_is_rejected(tiny_config, tmp_path):
    """S11 (#46): a batch row mixing kind 1 and 2 is refused by `encode` (plain and with internals).

    Single-modality rows in the same batch are fine; the error names the offending row and the fix.
    """
    model = build_from_config(tiny_config)
    model.load_gene_table(_write_gene_table(tmp_path / "genes.safetensors", 20, tiny_config.d_cseq))
    model.load_peak_table(_write_gene_table(tmp_path / "peaks.safetensors", 30, tiny_config.d_cseq))
    emb_idx, kind, lengths = _tokens([[1, 2, 3, 4], [5, 6, 7], [8, 9, 10, 11]], L=4)
    kind[1, :3] = KIND_PEAK  # row 1: a pure peak view -> allowed
    with torch.inference_mode():
        h = model.encode(emb_idx, kind, lengths)
    assert h.shape == (3, tiny_config.d_model) and torch.isfinite(h).all()
    check_single_modality(kind)  # no raise

    kind[2, 2:] = KIND_PEAK  # row 2: 2 genes + 2 peaks -> rejected
    with pytest.raises(MixedModalityViewError, match=r"row\(s\) \[2\].*2 gene \+ 2 peak") as e:
        model.encode(emb_idx, kind, lengths)
    assert isinstance(e.value, ValueError) and "two modalities" in str(e.value)
    with pytest.raises(MixedModalityViewError):
        model.encode(emb_idx, kind, lengths, return_internals=True)
    with pytest.raises(MixedModalityViewError):
        check_single_modality(kind)
    # pads do not count as a modality; a single row (1-D kind) is accepted too
    check_single_modality(kind[0])
