"""Internals API (docs/test-plan.md section 7, I1-I7) plus one ``assets`` end-to-end check.

The unit tests build ``hyacinth``'s architecture from the committed ``config.json`` at **random
init** (no weights) with a 4,096-row random gene table, so shapes and head counts are the released
ones (``H = (8, 8, 4)``, ``d = (32, 160, 32)``, 256 latents) while the tier stays download-free.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from bloom.config import ModelConfig
from bloom.model import (
    KIND_GENE,
    KIND_PAD,
    KIND_PEAK,
    EncodeOutput,
    InternalsMemoryError,
    MixedModalityViewError,
    build_from_config,
    capture,
    encode_with_internals,
    pair_logits,
)
from bloom.model.internals import DEFAULT_MAX_TOKENS, estimate_internals_bytes, qk_scales

B, L, L_SHORT = 2, 512, 300
N_ROWS = 4096
SEED = 0


# ------------------------------------------------------------------------------------------
# fixtures
# ------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def model(hyacinth_config_path, tmp_path_factory):
    """``hyacinth`` architecture at random init with a random 4,096-row gene table (CPU)."""
    cfg = ModelConfig.from_json(hyacinth_config_path)
    torch.manual_seed(SEED)
    m = build_from_config(cfg)
    g = torch.Generator().manual_seed(SEED)
    table = {
        "cseq": torch.randn(N_ROWS, cfg.d_cseq, generator=g),
        "chr": torch.randint(0, 21, (N_ROWS,), generator=g).to(torch.int8),
        "bp": torch.randint(0, 100_000_000, (N_ROWS,), generator=g).to(torch.int32),
    }
    path = tmp_path_factory.mktemp("internals") / "genes.safetensors"
    save_file(table, str(path))
    m.load_gene_table(str(path))
    return m


@pytest.fixture(scope="module")
def batch():
    """``(emb_idx, kind, lengths)``: row 0 full (512 tokens), row 1 with 212 pads."""
    g = torch.Generator().manual_seed(SEED + 1)
    emb_idx = torch.full((B, L), -1, dtype=torch.int32)
    kind = torch.full((B, L), KIND_PAD, dtype=torch.int8)
    lengths = torch.tensor([L, L_SHORT], dtype=torch.int64)
    for b in range(B):
        n = int(lengths[b])
        emb_idx[b, :n] = torch.randperm(N_ROWS, generator=g)[:n].to(torch.int32)
        kind[b, :n] = KIND_GENE
    return emb_idx, kind, lengths


def _resolver(emb_idx: np.ndarray, kind: np.ndarray) -> list[str]:
    return [f"G{int(e)}" for e in emb_idx]


@pytest.fixture(scope="module")
def out(model, batch) -> EncodeOutput:
    emb_idx, kind, lengths = batch
    return model.encode(emb_idx, kind, lengths, return_internals=True, reference=_resolver)


# ------------------------------------------------------------------------------------------
# I1 shapes
# ------------------------------------------------------------------------------------------
@pytest.mark.unit
def test_i1_shapes(model, batch, out):
    cfg = model.config
    attn = cfg.attention
    d_model, M = cfg.d_model, cfg.n_latents
    assert (attn.n_sequence_heads, attn.n_position_heads, attn.n_standard_heads) == (8, 8, 4)
    assert out.h_cell.shape == (B, d_model)
    assert out.token_h.shape == (B, L, d_model)
    assert out.latents.shape == (B, M, d_model)
    assert set(out.qk) == {"seq", "pos", "std"}
    d_pos = attn.d_u + 2 * attn.K
    for group, h, d in (("seq", 8, attn.d_v), ("pos", 8, d_pos), ("std", 4, attn.d_v)):
        q, k = out.qk[group]
        assert q.shape == (B, L, h, d), (group, q.shape)
        assert k.shape == (B, L, h, d), (group, k.shape)
    assert out.cross_attn.shape == (B, M, L)
    assert out.lengths.tolist() == [L, L_SHORT]
    assert out.key_padding_mask.shape == (B, L)
    assert out.key_padding_mask[1, L_SHORT:].all() and not out.key_padding_mask[1, :L_SHORT].any()
    assert len(out.token_ids) == B and all(len(row) == L for row in out.token_ids)
    assert out.token_ids[1][L_SHORT - 1] is not None
    assert out.token_ids[1][L_SHORT:] == [None] * (L - L_SHORT)
    assert out.token_ids[0][3] == f"G{int(batch[0][0, 3])}"
    assert out.position_of(out.token_ids[0][7]) == 7
    assert out.batch_size == B and out.n_tokens == L
    # per-head cross-attention
    emb_idx, kind, lengths = batch
    out_h = model.encode(emb_idx, kind, lengths, return_internals=True, per_head=True)
    assert out_h.cross_attn.shape == (B, cfg.n_heads, M, L)
    assert torch.allclose(out_h.cross_attn.mean(dim=1), out.cross_attn, atol=1e-6)
    assert out_h.token_ids is None  # no reference attached to this model
    # opt out of the weights
    out_n = model.encode(emb_idx, kind, lengths, return_internals=True, cross_attn=False)
    assert out_n.cross_attn is None and torch.equal(out_n.h_cell, out.h_cell)


# ------------------------------------------------------------------------------------------
# I2 h_cell unchanged
# ------------------------------------------------------------------------------------------
@pytest.mark.unit
def test_i2_h_cell_bitwise(model, batch, out):
    emb_idx, kind, lengths = batch
    with torch.inference_mode():
        plain = model.encode(emb_idx, kind, lengths)
    assert torch.equal(plain, out.h_cell)
    # encode() runs unfused in every mode (#44): a grad-enabled call differs from the
    # inference_mode call only by autograd bookkeeping noise (~1e-6), on CPU and CUDA alike.
    assert torch.allclose(model.encode(emb_idx, kind, lengths), out.h_cell, atol=1e-5)
    plain_internals = encode_with_internals(model, emb_idx, kind, lengths)
    assert torch.equal(plain_internals.h_cell, out.h_cell)
    assert torch.equal(plain_internals.token_h, out.token_h)
    assert torch.equal(plain_internals.latents, out.latents)


@pytest.mark.unit
def test_internals_kwargs_need_flag(model, batch):
    emb_idx, kind, lengths = batch
    with pytest.raises(TypeError, match="return_internals=True"):
        model.encode(emb_idx, kind, lengths, per_head=True)


# ------------------------------------------------------------------------------------------
# I3 pair logits
# ------------------------------------------------------------------------------------------
@pytest.mark.unit
def test_i3_scales(model, out):
    assert out.scales == qk_scales(model.first_layer)
    assert math.isclose(out.scales["seq"], 1.0 / math.sqrt(32.0), rel_tol=0, abs_tol=0)  # exact
    assert math.isclose(out.scales["seq"], model.config.seq_head_scale, rel_tol=0, abs_tol=0)
    assert math.isclose(out.scales["pos"], 1.0, rel_tol=0, abs_tol=0)
    assert math.isclose(
        out.scales["std"], 1.0 / math.sqrt(model.config.attention.d_v), rel_tol=1e-12
    )


@pytest.mark.unit
@pytest.mark.parametrize("group", ["seq", "pos", "std"])
def test_i3_pair_logits_match_explicit(out, group):
    rng = np.random.default_rng(SEED)
    for b in range(B):
        n = int(out.lengths[b])
        pairs = rng.integers(0, n, size=(100, 2))
        got = pair_logits(out, pairs, group=group, batch=b)  # (100, H_g)
        q, k = out.qk[group]
        full = torch.einsum("ihd,jhd->hij", q[b], k[b]) * out.scales[group]  # (H, L, L)
        want = full[:, pairs[:, 0], pairs[:, 1]].T  # (100, H)
        assert got.shape == want.shape
        assert torch.allclose(got, want, atol=1e-5, rtol=0), (got - want).abs().max()
    # head subset and tensor pairs
    sub = pair_logits(out, torch.as_tensor(pairs), group=group, batch=1, heads=[0, 2])
    assert torch.equal(sub, got[:, [0, 2]])
    with pytest.raises(IndexError):
        pair_logits(out, [(0, L_SHORT)], group=group, batch=1)  # a pad position
    with pytest.raises(KeyError):
        pair_logits(out, pairs, group="nope")


@pytest.mark.unit
@pytest.mark.parametrize("group", ["seq", "pos", "std"])
def test_i3_qk_reproduce_block_attention(model, batch, out, group):
    """softmax(q k^T * scale, pad-masked) @ W_V(.) == the head module's own output (< 1e-5).

    An independent check that the returned q / k and scale are the ones the block attends with.
    """
    emb_idx, kind, lengths = batch
    block = model.first_layer.attn_block
    heads = {"seq": block.seq_heads, "pos": block.pos_heads, "std": block.std_heads}[group]
    with torch.inference_mode(), capture(model, f"first_layer.attn_block.{group}_heads") as rec:
        model.encode(emb_idx, kind, lengths)
        cseq, chrom, bp, clocal, kpm = model.first_layer_inputs(emb_idx, kind, lengths)
        m = model.first_layer.m_encoder(clocal.unsqueeze(-1))
        s = torch.cat([cseq, m], dim=-1)
        if group == "std":
            pos = block.pos_encoder(bp.float(), chrom.long())
            p = torch.cat([block.beta_std * pos["u"], pos["r_phi"]], dim=-1)
            v_in = torch.cat([cseq, p, m], dim=-1)
        else:
            v_in = s
        v = heads.W_V(v_in).view(B, L, heads.H, heads.d_v)  # (B, L, H, d_v)
    module_out = rec[f"first_layer.attn_block.{group}_heads"]  # (B, L, H * d_v)
    q, k = out.qk[group]
    for b in range(B):
        n = int(lengths[b])
        logits = torch.einsum("ihd,jhd->hij", q[b], k[b]) * out.scales[group]  # (H, L, L)
        logits = logits.masked_fill(kpm[b].view(1, 1, L), float("-inf"))
        attn = torch.softmax(logits, dim=-1)
        o = torch.einsum("hij,jhd->ihd", attn, v[b]).reshape(L, heads.H * heads.d_v)
        diff = (o[:n] - module_out[b, :n]).abs().max().item()
        assert diff < 1e-5, (group, b, diff)


@pytest.mark.unit
def test_s11_internals_and_pair_logits_refuse_mixed_views(model, batch, out):
    """S11 (#46) at the internals level: `encode_with_internals` and `pair_logits` refuse mixed rows."""
    emb_idx, kind, lengths = batch
    mixed = kind.clone()
    mixed[1, :10] = KIND_PEAK  # row 1 now holds 10 peak + 290 gene tokens
    with pytest.raises(MixedModalityViewError, match=r"row\(s\) \[1\]"):
        encode_with_internals(model, emb_idx, mixed, lengths, reference=_resolver)
    # a hand-built EncodeOutput whose row 1 is mixed: cross-modal logits are refused, row 0 still works
    fake = EncodeOutput(**{**out.__dict__, "kind": mixed})
    assert pair_logits(fake, [(0, 1)], group="seq", batch=0).shape == (1, 8)
    with pytest.raises(MixedModalityViewError):
        pair_logits(fake, [(0, 100)], group="seq", batch=1)


# ------------------------------------------------------------------------------------------
# I4 cross-attention rows
# ------------------------------------------------------------------------------------------
@pytest.mark.unit
def test_i4_cross_attn_rows(model, batch, out):
    w = out.cross_attn  # (B, M, L)
    assert torch.isfinite(w).all()
    assert torch.allclose(w.sum(dim=-1), torch.ones(B, model.config.n_latents), atol=1e-5)
    assert (w[1, :, L_SHORT:] == 0).all()
    assert (w[1, :, :L_SHORT] > 0).all()
    emb_idx, kind, lengths = batch
    out_h = model.encode(emb_idx, kind, lengths, return_internals=True, per_head=True)
    wh = out_h.cross_attn  # (B, H, M, L)
    assert torch.allclose(wh.sum(dim=-1), torch.ones_like(wh[..., 0]), atol=1e-5)
    assert (wh[1, :, :, L_SHORT:] == 0).all()


# ------------------------------------------------------------------------------------------
# I5 capture
# ------------------------------------------------------------------------------------------
def _n_hooks(model) -> int:
    return sum(len(m._forward_hooks) for m in model.modules())


@pytest.mark.unit
def test_i5_capture(model, batch, out):
    emb_idx, kind, lengths = batch
    before = _n_hooks(model)
    names = ["encoder.final_norm", "first_layer.attn_block.W_O", "encoder.cross_attn"]
    with torch.inference_mode(), capture(model, names) as rec:
        assert _n_hooks(model) == before + len(names)
        h = model.encode(emb_idx, kind, lengths)
    assert _n_hooks(model) == before
    assert set(rec) == set(names)
    assert torch.equal(rec["encoder.final_norm"], h)  # h_cell = final_norm(mean(latents))
    assert rec["first_layer.attn_block.W_O"].shape == (B, L, model.config.d_model)
    attn_out, weights = rec["encoder.cross_attn"]  # MultiheadAttention returns a tuple
    assert attn_out.shape == (B, model.config.n_latents, model.config.d_model) and weights is None
    # nested
    with torch.inference_mode(), capture(model, "encoder") as outer:
        with capture(model, "first_layer", detach=True) as inner:
            assert _n_hooks(model) == before + 2
            model.encode(emb_idx, kind, lengths)
        assert _n_hooks(model) == before + 1
        assert torch.equal(inner["first_layer"], out.token_h)
        assert torch.equal(outer["encoder"], out.h_cell)
    assert _n_hooks(model) == before
    # removed on error
    with pytest.raises(RuntimeError, match="boom"):
        with capture(model, "encoder"):
            raise RuntimeError("boom")
    assert _n_hooks(model) == before
    with pytest.raises(KeyError, match="not submodules"):
        with capture(model, ["encoder.nope"]):
            pass
    assert _n_hooks(model) == before


# ------------------------------------------------------------------------------------------
# I6 gradients
# ------------------------------------------------------------------------------------------
@pytest.mark.unit
def test_i6_gradients_reach_token_h(model, batch, out):
    emb_idx, kind, lengths = batch
    assert not out.token_h.requires_grad  # default path is inference_mode
    with torch.inference_mode():  # the flag must win over the caller's context
        g = model.encode(emb_idx, kind, lengths, return_internals=True, requires_grad=True)
    assert g.token_h.requires_grad and g.h_cell.requires_grad
    # At random init every LayerNorm has gamma == 1, so h_cell.sum() (and h_cell.norm()) are
    # constants with an exactly-zero gradient; a random projection is not. The released weights
    # have non-constant gamma and the assets test below uses the plain h_cell.sum().
    w = torch.randn(model.config.d_model, generator=torch.Generator().manual_seed(SEED))
    (g.h_cell * w).sum().backward()
    assert g.token_h.grad is not None
    assert g.token_h.grad.shape == (B, L, model.config.d_model)
    assert torch.isfinite(g.token_h.grad).all()
    assert g.token_h.grad[0].abs().sum() > 0
    assert torch.allclose(g.h_cell.detach(), out.h_cell, atol=1e-5)  # see I2
    model.zero_grad(set_to_none=True)


# ------------------------------------------------------------------------------------------
# I7 memory guard
# ------------------------------------------------------------------------------------------
@pytest.mark.unit
def test_i7_memory_guard(model, monkeypatch):
    big_L = DEFAULT_MAX_TOKENS // 2 + 1
    emb_idx = torch.zeros((2, big_L), dtype=torch.int32)
    kind = torch.full((2, big_L), KIND_GENE, dtype=torch.int8)

    def boom(*a, **k):
        raise AssertionError("resolve() ran: the guard must fire before any allocation")

    monkeypatch.setattr(model, "resolve", boom)
    with pytest.raises(InternalsMemoryError, match="max_tokens"):
        model.encode(emb_idx, kind, return_internals=True)
    with pytest.raises(InternalsMemoryError):
        encode_with_internals(model, emb_idx, kind, max_tokens=1000)
    # the plain path has no guard
    monkeypatch.undo()
    est = estimate_internals_bytes(model, 2 * big_L)
    per_token = 4 * (1024 + 2 * 8 * 32 + 8 * 160 + 2 * 4 * 32 + 256)  # 13,312 B for hyacinth
    assert est == per_token * 2 * big_L  # 0.41 GiB at the default ceiling
    small = torch.zeros((1, 8), dtype=torch.int32)
    small_kind = torch.full((1, 8), KIND_GENE, dtype=torch.int8)
    res = model.encode(small, small_kind, return_internals=True, max_tokens=8)
    assert res.h_cell.shape == (1, model.config.d_model)
    with pytest.raises(InternalsMemoryError):
        model.encode(small, small_kind, return_internals=True, max_tokens=7)
    assert model.encode(small, small_kind, return_internals=True, max_tokens=None).h_cell.shape == (
        1,
        model.config.d_model,
    )


# ------------------------------------------------------------------------------------------
# token ids through ReferenceTables (synthetic release-layout tables)
# ------------------------------------------------------------------------------------------
@pytest.mark.unit
def test_token_ids_from_reference_tables(hyacinth_config_path, synthetic_reference_dir):
    from bloom.reference import ReferenceTables

    ref = ReferenceTables.from_dir(synthetic_reference_dir)
    cfg = ModelConfig.from_json(hyacinth_config_path)
    torch.manual_seed(SEED)
    m = build_from_config(cfg)
    m.load_gene_table(str(ref.paths["genes_safetensors"]))  # subset table scattered to full size
    emb_idx = torch.tensor([[25359, 2498, 26231, -1]], dtype=torch.int32)  # Pvalb, Gad1, Sst, pad
    kind = torch.tensor([[1, 1, 1, 0]], dtype=torch.int8)
    res = m.encode(emb_idx, kind, return_internals=True, reference=ref)
    assert res.token_ids == [["Pvalb", "Gad1", "Sst", None]]
    assert res.position_of("Gad1") == 1
    with pytest.raises(KeyError):
        res.position_of("Vip")
    m.reference = ref  # bloom.load_model attaches it like this
    assert m.encode(emb_idx, kind, return_internals=True).token_ids == res.token_ids
    assert torch.isfinite(pair_logits(res, [(0, 1), (1, 2)], "seq")).all()
    assert res.to("cpu").h_cell is not None and res.detach().token_h.requires_grad is False


# ------------------------------------------------------------------------------------------
# assets: the released weights on 2 fixture cells
# ------------------------------------------------------------------------------------------
@pytest.mark.assets
def test_internals_hyacinth_end_to_end(hyacinth_model_dir, gene_table_dir):
    import anndata as ad

    import bloom
    from bloom.strategies import TopExpressed, cell_rng
    from bloom.tokens import collate

    model = bloom.load_model("hyacinth", device="auto", reference_dir=gene_table_dir)
    adata = ad.read_h5ad("tests/data/rna_100.h5ad")[:2].copy()
    ds = bloom.prepare(rna=adata, species="mouse", reference=model.reference)
    strat = TopExpressed(lmax=4096)
    views = [strat.views(ds.cell_features(i, "rna"), cell_rng(0, i))[0] for i in range(2)]
    batch = collate(views).to(model.device)
    res = model.encode(batch.emb_idx, batch.kind, batch.lengths, return_internals=True)
    with torch.inference_mode():
        plain = model.encode(batch.emb_idx, batch.kind, batch.lengths)
    assert torch.equal(plain, res.h_cell)
    assert res.token_h.shape == (2, batch.lmax, 1024) and res.latents.shape == (2, 256, 1024)
    symbols = set(model.reference.genes["gene_name"])
    for b in range(2):
        n = int(res.lengths[b])
        ids = res.token_ids[b]
        assert all(s in symbols for s in ids[:n]) and ids[n:] == [None] * (batch.lmax - n)
    assert torch.allclose(
        res.cross_attn.sum(-1), torch.ones_like(res.cross_attn[..., 0]), atol=1e-4
    )
    # I6 with the released weights: the plain h_cell.sum() has a non-zero gradient
    g = model.encode(
        batch.emb_idx[:1],
        batch.kind[:1],
        batch.lengths[:1],
        return_internals=True,
        requires_grad=True,
    )
    g.h_cell.sum().backward()
    n0 = int(g.lengths[0])
    assert g.token_h.grad is not None and torch.isfinite(g.token_h.grad).all()
    assert (g.token_h.grad[0, :n0].norm(dim=-1) > 0).all()
    model.zero_grad(set_to_none=True)
    # seq-head logits for a few gene pairs present in cell 0's top-4096 view
    ids0 = res.token_ids[0]
    wanted = [g for g in ("Malat1", "Snhg11", "Meg3", "Gad1", "Gad2", "Slc17a7") if g in ids0]
    genes = (wanted + ids0[:4])[:4]
    pos = [res.position_of(g, 0) for g in genes]
    pairs = [(i, j) for i in pos for j in pos if i != j]
    logits = pair_logits(res, pairs, group="seq", batch=0)  # (P, 8)
    assert logits.shape == (len(pairs), 8) and torch.isfinite(logits).all()
    assert res.scales["seq"] == pytest.approx(1 / math.sqrt(32))
