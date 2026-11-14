> Repository: `github.com/alleninstitute/bloom-biorxiv`.

# Internals — the ML practitioner's view of `Bloom.encode`

`Bloom` is a plain `torch.nn.Module`; `model.encode(emb_idx, kind, lengths)` returns `h_cell (B, 1024)`. This page is
about what else the forward pass computes and how to get at it without materialising the first-layer attention matrix.
Spec: [architecture.md](architecture.md) §5.3; tests I1–I7 in [test-plan.md](test-plan.md) §7; module
`src/bloom/model/internals.py`. The model and its inputs: [architecture.md](architecture.md) §2 and §5.1,
[data-schema.md](data-schema.md).

## 1. What you get

```python
import torch, bloom
from bloom.strategies import TopExpressed, cell_rng
from bloom.tokens import collate

model = bloom.load_model("hyacinth", device="auto")                  # eval(), gene table + model.reference attached
ds    = bloom.prepare(rna=adata, species="mouse", reference=model.reference)
views = [TopExpressed(lmax=4096).views(ds.cell_features(i, "rna"), cell_rng(0, i))[0] for i in range(8)]
batch = collate(views).to(model.device)                              # emb_idx (B, L) int32 pad -1, kind (B, L) int8, lengths (B,)

out = model.encode(batch.emb_idx, batch.kind, batch.lengths, return_internals=True)   # -> EncodeOutput
```

`EncodeOutput` (a dataclass; every tensor on the model's device, fp32):

| field | shape | what |
|---|---|---|
| `h_cell` | `(B, 1024)` | the cell embedding, **bitwise identical** to `model.encode(...)` under `inference_mode` (I2) |
| `token_h` | `(B, L, 1024)` | token states after `BloomFirstLayer` (attention block + FFN residual) |
| `latents` | `(B, 256, 1024)` | Perceiver latents after the 8-layer self-attention stack, before mean-pool + LayerNorm |
| `qk["seq"]` | `q, k` each `(B, L, 8, 32)` | sequence heads: `q = W_Q(cseq)`, `k = W_K(cseq)` |
| `qk["pos"]` | `q, k` each `(B, L, 8, 160)` | position heads: `q is k = sqrt(tau_h) · [beta_h · u_chr ; w_h ⊙ r_phi(bp)]` (`d = d_u + 2K = 32 + 128`) |
| `qk["std"]` | `q, k` each `(B, L, 4, 32)` | standard heads on `[cseq ; beta_std · u ; r_phi ; m]` |
| `scales` | dict | softmax scale per group: `seq` **1/√32** (the `bloom-arch@ef5fb8a` pin asserted at load), `pos` 1.0, `std` 1/√32 |
| `cross_attn` | `(B, 256, L)` or `(B, 8, 256, L)` | latent → token attention of the Perceiver cross-attention, head-averaged (default) or per head (`per_head=True`); rows sum to 1, 0 at pads (I4); `None` with `cross_attn=False` |
| `token_ids` | `list[list[str \| None]]` | gene symbol / peak `chr:start-end` per position, `None` at pads; `None` altogether when no reference is available |
| `lengths`, `key_padding_mask` | `(B,)`, `(B, L)` | real tokens per row; `True` = pad |
| `emb_idx`, `kind` | `(B, L)` | the inputs, for bookkeeping |

Helpers: `out.position_of("Gad1", batch=0)` → token position; `out.to(device)`, `out.detach()`.

**Peak tokens** (`kind == 2`, views from an ATAC `PreparedDataset`: `ds.cell_features(i, "atac")`) need the 5.4 GB peak table
on the model — `bloom.load_model("hyacinth", device="auto", peaks=True)`, or `bloom.embed.ensure_peak_table(model)` later
(`bloom.embed()` does this itself); without it `encode` raises `RuntimeError: peak reference table not loaded`
(`bloom download --what peaks` first). `token_ids` are then `chr:start-end` strings (`peaks.parquet.name`). A view holds one
modality (`CellFeatures` carries genes *or* peaks, S7), so gene–peak pair logits across modalities are not available: a
hand-built view that mixes `kind` 1 and 2 is **rejected** (#46, test S11) — `TokenSet(...)` raises `ValueError`, and
`encode`, `encode(..., return_internals=True)` / `encode_with_internals` and `pair_logits` raise
`bloom.model.MixedModalityViewError` (a `ValueError`) naming the offending batch row. The model was trained on one modality
per view, so the embedding of a mixed view and its cross-modal attention logits are undefined; embed RNA and ATAC as two
modalities (two `TokenSet`s, two `encode` calls) instead.

Keyword options of `encode(..., return_internals=True, ...)` (also `bloom.encode_with_internals(model, ...)`):

| option | default | effect |
|---|---|---|
| `reference` | `model.reference` | a `ReferenceTables` (`bloom.load_model` attaches one) or a callable `(emb_idx, kind) -> names` for your own tables |
| `cross_attn` | `True` | compute the cross-attention weights (a second cross-attention call, see §3) |
| `per_head` | `False` | `(B, 8, 256, L)` instead of the head average |
| `requires_grad` | `False` | build the autograd graph; `out.token_h.retain_grad()` is set, so `out.h_cell.sum().backward()` fills `out.token_h.grad` (I6) |
| `max_tokens` | `32_768` | raise `InternalsMemoryError` **before any allocation** when `B·L` exceeds it (I7); `None` disables |

## 2. Pair logits — never materialise attention at L = 8192

The first layer attends over up to 8,192 tokens with 20 heads: the `B × H × L × L` fp32 attention matrix is 5 GB **per
cell**. `encode` never builds it (scaled-dot-product attention) and neither does this API. The paper's attention analyses were
done from exported Q/K, and that is the supported route here:

```python
from bloom import pair_logits

i, j = out.position_of("Gad1"), out.position_of("Gad2")         # positions in row 0
pair_logits(out, [(i, j), (j, i)], group="seq", batch=0)         # (2, 8): q_i · k_j · (1/sqrt 32) per sequence head
pair_logits(out, pairs, group="pos", heads=[0, 3])               # (P, 2): position heads 0 and 3
```

`pair_logits(out, pairs, group, batch=0, heads=None)` computes `q_i · k_j · scale` for the `(P, 2)` positions given
and nothing else (I3: equals the explicitly indexed `L × L` product to < 1e-5; `scales["seq"] == 1/sqrt(32)`). Pairs
must be real tokens (`< lengths[batch]`), otherwise `IndexError`. For a full row of one query (`i` against every real
`j`) build the pairs with `[(i, j) for j in range(int(out.lengths[b]))]` — that is `L` dot products, not `L²`.

Row semantics: `i` is the **query**, `j` the **key**: softmax over `j` of `pair_logits(out, [(i, j) for j in ...])`
is the attention row of token `i` in that head (pad keys excluded), exactly what the block uses (test
`test_i3_qk_reproduce_block_attention` recomposes the head outputs from `q`, `k`, `W_V` this way).

## 3. How the tensors are obtained (numerics)

- `h_cell`, `token_h`, `latents`: the same operations as `Bloom.encode` (`Bloom.first_layer_inputs` → `first_layer` →
  `encoder.cross_attend` (default path) → `encoder.self_layers` → `encoder.pool`). Under `inference_mode` the result is
  bitwise `encode`'s (I2). Outside it (`requires_grad=True`, or your own grad-enabled call of `encode`) the result differs
  only by autograd bookkeeping noise (~1e-6): since v0.1.0 both `encode` and this function run with torch's fused
  `nn.MultiheadAttention` / `TransformerEncoder` fast path disabled on every device (`bloom.model.fastpath_disabled`, #44),
  so there is no fused-vs-unfused gap between inference and saliency calls.
- `qk`: `bloom_arch@ef5fb8a` keeps `q`/`k` local to each head module's `forward`, so `internals.first_layer_qk`
  **re-applies the block's projections** (`W_Q`/`W_K` linears, the position-head `[beta·u ; w⊙r_phi]·sqrt(tau)`
  construction) to the inputs the block saw. This duplicates three `(B·L, d_in) × (d_in, H·d)` matmuls (cheap) and does
  **not** recompute the attention. I3 checks that `softmax(q kᵀ · scale) W_V` reproduces each head module's output.
- `cross_attn`: a second `PerceiverEncoder.cross_attend(token_h, mask, need_weights=True, average_attn_weights=...)`.
  `need_weights=True` switches `nn.MultiheadAttention` to its unfused path, so it runs on a separate call whose output
  tensor is discarded; the call that feeds `h_cell` is unchanged. Cost `O(B · 8 · 256 · L)` — 67 MB at `L = 8192`, B = 1.
- The token-level `key_padding_mask` is `True` at pads; `token_h`, `q`, `k` at pad positions are whatever the model
  computes for zero inputs — index real tokens with `lengths` / `~key_padding_mask`.

## 4. Memory guidance

Per token the internals hold `1024 (token_h) + 2·8·32 + 8·160 + 2·4·32 (q, k) + 256 (cross_attn)` = 3,328 fp32
values = 13.3 KB (`per_head=True` adds 7·256 values → 20.5 KB):

| B × L | internals | note |
|---|---|---|
| 1 × 4096 | 55 MB | `top` view, one cell |
| 8 × 4096 = 32,768 | 0.41 GiB | default `max_tokens` |
| 1 × 8192 | 109 MB | one `random` view |
| 4 × 8192 = 32,768 | 0.41 GiB | default `max_tokens` |

`estimate_internals_bytes(model, B·L)` gives the number the guard prints. The guard does not model the first-layer
attention itself: on CPU fp32 SDPA materialises `B × 20 × L × L` (1.3 GB per cell at L = 4096, 5 GB at 8192, see
`embed()`'s CPU batch-size default of 1 in [architecture.md](architecture.md) §5.4); on CUDA the SDPA kernels do not.
With `requires_grad=True` autograd additionally saves activations of every layer — use B = 1 and a GPU for saliency.
Do not keep `EncodeOutput`s for many cells; reduce them (pair logits, pooled `cross_attn`, `token_h` projections) per
batch and keep the reductions.

## 5. Hook capture

```python
from bloom import capture

with torch.inference_mode(), capture(model, ["encoder.self_layers.layers.3", "first_layer.attn_block.W_O"]) as rec:
    h = model.encode(batch.emb_idx, batch.kind, batch.lengths)
rec["encoder.self_layers.layers.3"]        # (B, 256, 1024) output of the 4th latent self-attention layer
```

`capture(model, names, detach=False)` registers forward hooks on the named submodules (`dict(model.named_modules())`
lists the names; a wrong name raises `KeyError` with suggestions), records each module's **last** output while the block
runs — an `nn.MultiheadAttention` yields its `(attn_output, weights)` tuple — and removes the hooks on exit, also on
exceptions; nesting is fine (I5). It works on any `nn.Module`, not only `Bloom`.

## 6. Worked example — seq-head logits for gene pairs on the fixture

```python
import math, torch, anndata as ad, bloom
from bloom import pair_logits
from bloom.strategies import TopExpressed, cell_rng
from bloom.tokens import collate

model = bloom.load_model("hyacinth", device="auto")
adata = ad.read_h5ad("tests/data/rna_100.h5ad")[:2].copy()
ds = bloom.prepare(rna=adata, species="mouse", reference=model.reference)
views = [TopExpressed(lmax=4096).views(ds.cell_features(i, "rna"), cell_rng(0, i))[0] for i in range(2)]
batch = collate(views).to(model.device)

out = model.encode(batch.emb_idx, batch.kind, batch.lengths, return_internals=True)
assert out.scales["seq"] == 1 / math.sqrt(32)

genes = [g for g in ("Malat1", "Snhg11", "Meg3", "Gad1") if g in out.token_ids[0]]
pos = [out.position_of(g, batch=0) for g in genes]
pairs = [(i, j) for i in pos for j in pos if i != j]
logits = pair_logits(out, pairs, group="seq", batch=0)              # (P, 8)
for (i, j), row in zip(pairs, logits.tolist()):
    print(out.token_ids[0][i], "->", out.token_ids[0][j], [f"{v:+.2f}" for v in row])

# which tokens does latent 0 attend to in cell 0?
top = out.cross_attn[0, 0].topk(5).indices.tolist()
print([out.token_ids[0][t] for t in top])
```

The same script lives in `tools/examples/attention_internals.py` (runs on CPU in ~20 s with the assets downloaded).

Saliency (gradient of the embedding norm w.r.t. token states, one cell):

```python
g = model.encode(batch.emb_idx[:1], batch.kind[:1], batch.lengths[:1], return_internals=True, requires_grad=True)
g.h_cell.norm().backward()
saliency = g.token_h.grad.norm(dim=-1)[0, : int(g.lengths[0])]  # (L_real,)
```
