> Repository: `github.com/alleninstitute/bloom-biorxiv`.

# bloom-biorxiv — specification

The design of the v1 release: what the package does, its user surface, the internal contracts and the layout. Measured
state: [validation.md](validation.md); status per module: `AGENTS.md`. Companion documents: [data-schema.md](data-schema.md),
[assets.md](assets.md), [test-plan.md](test-plan.md), [strategies.md](strategies.md), [standards.md](standards.md).

Naming: the released checkpoint is **`hyacinth`** (the training project name). Training-run ids / step numbers appear only in
provenance fields, never in user-facing names. Reference tables: **`mm10-evo2-v1`**.

## 1. Purpose

`bloom-biorxiv` is the public, inference-only companion repository for the Bloom manuscript
(Gala, Convy, Marghi et al., bioRxiv 2026, doi:10.64898/2026.11.11.765881). It must let

- **a bioinformatician** go from their own AnnData to Bloom cell embeddings with one
  command, on a laptop CPU or a single GPU, with no cluster, no Hydra, no Lightning; and
- **an ML practitioner** load the model as a plain `torch.nn.Module`, construct input views
  with a small, documented strategy interface, and pull out attention Q/K, token states,
  Perceiver latents and cross-attention weights.

It is *not* a training repository. The training code stays in `alleninstitute/bloom`.

Decisions that shape v1 (details in the sections below): any non-negative matrix is accepted and only detection (`X > 0`)
matters; one AnnData per modality (`--rna`, `--atac`); three strategy presets `random` (paper default), `top`, `panel`, each
with an optional `universe`; mouse natively and human through 1:1 orthologs; assets on a public S3 bucket with a sha256
manifest; plain PyTorch (no Hydra, no Lightning); `bloom-arch` pinned as a git dependency; one released checkpoint; output
keys `obsm["X_bloom_rna"]` / `obsm["X_bloom_atac"]`; no `atacpb` path, no mixed gene+peak views in v1.

## 2. What the model needs — the fact that shapes the whole design

Verified in `alleninstitute/bloom@141f0d3` (`strategy.py::_mask_metadata`, `_binarize_by_modality`;
`bloom_attention_layer.py::forward`; `model.py::get_cell_representation`) and the training run's
Hydra config (`binarize_input_* = true`, `mask_metadata_keys = [ftype, mtype, batch]`):

- The encoder input per token is `(cseq[768], chr, bp, clocal)`. With binarization `clocal ≡ 1`
  for every real token. Feature/modality/batch metadata is zeroed on the input side.
- Therefore **`h_cell = f({detected genes} ∪ {detected peaks})`**. Library size, normalization,
  `batch_type`, species label and count depth do not enter the embedding. They enter only the
  decoder (`cglobal`, `batch`), which v1 does not expose.
- Consequences: (1) the only hard input requirement is a non-negative matrix and feature
  identifiers we can map to the reference tables; (2) library-size columns
  (`rna_umi_counts`/`atac_umi_counts`) are **not collected** — that path is dropped until the
  decoder is exposed; (3) the `top` strategy ranks by magnitude, and log1p(CPM) is monotonic in
  counts within a cell, so ranking is identical for raw or CPM-normalized input (ties aside).

## 3. User-facing surface

### 3.1 CLI (`bloom`)

```
bloom download  [--what model,genes,peaks,orthologs,panels,examples] [--dir ~/.cache/bloom] [--force] [--source ...]
bloom prepare   [--rna RNA.h5ad] [--atac ATAC.h5ad] [--species {mouse,human}] [--gene-col COL] [--layer L]
                [--peak-match {auto,exact,overlap}] [--min-overlap 0.5] [--keep-unmatched] [--force-species]
                -o PREPARED_DIR
bloom embed     ([--rna RNA.h5ad] [--atac ATAC.h5ad] | PREPARED_DIR) [--species {mouse,human}]
                [--strategy {random,top,panel}] [--lmax N] [--draws N]
                [--universe features.txt] [--panel features.txt] [--seed 0]
                [--rna-<key> V] [--atac-<key> V]        # per-modality overrides of strategy/lmax/draws/universe/panel
                                                        #   (also spelled --rna.<key> / --atac.<key>)
                [--peak-match {auto,exact,overlap}] [--min-overlap 0.5] [--keep-unmatched] [--force-species]
                [--device {auto,cpu,cuda,mps}] [--batch-size 8] [--precision {fp32,bf16}] [--keep-views]
                -o embeddings.h5ad

At least one of `--rna` / `--atac` is required; the modality is implied by the flag, never
stated separately. `--species` is required with `--rna` on a raw file; ATAC alone implies mouse (mm10
peaks) and `--species human` is refused for ATAC. Each file given is embedded with its own strategy block
(defaults below) and written as `obsm["X_bloom_rna"]` / `obsm["X_bloom_atac"]`; with both files the output
holds the `obs_names` intersection and `--keep-unmatched` writes `embeddings_{rna,atac}_only.h5ad`.
bloom info      (package version, model/asset versions and sha256, device, torch version)
```

`embed` on a raw AnnData runs `prepare` in memory; `prepare` exists so the mapping report can
be inspected and the mapping cached for large datasets.

Defaults reproduce the paper: `--rna` → `random --lmax 8192 --draws 4`;
`--atac` → `random --lmax 8192 --draws 16`; `top` uses `--lmax 4096`.

### 3.2 Python API

```python
import bloom

model = bloom.load_model("hyacinth", device="auto")           # downloads + verifies assets
ref   = model.reference                                        # gene/peak tables, ortholog map

ds  = bloom.prepare(rna=adata, atac=atac_adata, species="mouse", reference=ref)  # either/both -> PreparedDataset
rep = ds.report                                                # mapping rates, dropped features

emb = bloom.embed(model, ds, batch_size=8, seed=0)              # paper defaults per modality present
                                                               # -> AnnData with obsm["X_bloom_rna"] and/or ["X_bloom_atac"]
# explicit per-modality strategies; ATAC restricted to a peak subset (e.g. the DA peaks)
emb = bloom.embed(model, ds, strategies={"rna":  bloom.strategies.Resample(lmax=8192, draws=4),
                                         "atac": bloom.strategies.Resample(lmax=8192, draws=16,
                                                                          universe="da_peaks.txt")})

# ML track: the model is a plain nn.Module (docs/internals.md, docs/extending-strategies.md)
h   = model.encode(emb_idx, kind, lengths)                     # (B, 1024); batch from bloom.tokens.collate([...])
out = model.encode(emb_idx, kind, lengths, return_internals=True)   # EncodeOutput: token_h, latents, qk, cross_attn, token_ids
lg  = bloom.pair_logits(out, [(i, j)], group="seq")           # q_i . k_j / sqrt(32) for selected pairs, never L x L
with bloom.capture(model, ["encoder.self_layers.layers.3"]) as rec:  # forward hooks, removed on exit
    model.encode(emb_idx, kind, lengths)
```

### 3.3 Output

`embeddings.h5ad`: `obs` = input `obs` (+ `bloom_n_tokens_*`), `obsm["X_bloom_rna"]` /
`obsm["X_bloom_atac"]` float32 (n_cells × 1024) for each modality given, `uns["bloom"]` = provenance (model id + sha, strategy + params, seed, draws,
package version, git sha, timestamp, reference id). Full schema in data-schema.md §4.

## 4. Package layout

```
bloom-biorxiv/
├── README.md, LICENSE, AGENTS.md (CLAUDE.md imports it), CONTRIBUTING.md
├── pyproject.toml            # the only dependency file: hatchling, src layout, `bloom` console script,
│                             #   bloom-arch @ git commit ef5fb8a, [dev] extra; torch floor only (build chosen by the user)
├── scripts/setup_env.sh      # conda env + torch (cpu|cu128) + `pip install -e .[dev]` + pre-commit hook
├── scripts/check.sh          # ruff + unit/regression tests (= CI)
├── src/bloom/
│   ├── __init__.py           # load_model, prepare, embed, strategies
│   ├── assets.py, config.py, reference.py, prepare.py, tokens.py, feature_lists.py, embed.py, cli.py
│   ├── strategies/           # base.py (ViewStrategy ABC), resample.py, top.py, panel.py, tile.py, factory.py
│   ├── model/                # bloom.py (Bloom, load_pretrained), first_layer.py, perceiver.py, decoder.py, embedding.py,
│   │                         #   internals.py (EncodeOutput, pair_logits, capture)
│   └── data/manifest.json    # packaged copy of the bucket MANIFEST.json (pinned by assets.MANIFEST_SHA256)
├── tools/                    # maintainer scripts, not installed (export_*, make_*, run_evaluate_checked.py, publish_assets.py,
│                             #   hydra/ presets, panels/ = the four unpublished panels, bucket/ = README + MODEL_CARD sources)
├── tests/                    # pytest tiers unit | regression | assets; tests/data/ = rna_100 fixture + consolidated references
└── docs/                     # this spec, data-schema, assets, input-data, strategies, test-plan, validation, development,
                              #   maintainers, standards, history/ (Phase 0/1 reports and frozen outputs)
```

Phase 2 added `src/bloom/peaks.py` (coordinate parsing, exact + overlap matching with per-contig sorted arrays +
`searchsorted`, reference-peak detection matrix). `src/bloom/model/internals.py`
(`EncodeOutput`, `encode_with_internals()`, `pair_logits()`, `capture()`) is the Phase 3 ML surface; `tools/examples/` holds
runnable scripts. Notebooks and a documentation site are roadmap items.

Dependencies (runtime): `torch` (version floor only), `numpy`, `pandas`, `scipy`, `anndata`, `h5py`, `pyarrow`,
`safetensors`, `pyyaml`, `typer`, `requests`, `bloom-arch` (git). `[dev]`: `pytest`, `ruff`, `pre-commit`.

## 5. Design details

### 5.1 Model

- `Bloom(nn.Module)` holds `first_layer`, `encoder` (Perceiver), `decoder`, `feature_emb`, and
  **non-persistent buffers** for the reference tables (`gene_cseq`, `peak_cseq`, `*_chr`, `*_bp`).
  Peak table (5.4 GB fp32) is loaded lazily and only when a peak strategy is requested;
  on CPU it is read into RAM (≈ 5.4 GB RSS; a file-backed mmap is #43); on GPU it is moved once.
- `encode(batch) -> h_cell` reproduces `Gardenia.get_cell_representation` exactly:
  `key_padding_mask` from lengths, `BloomFirstLayer`, `PerceiverEncoder`, mean-pool + LayerNorm.
- `config.json` carries every architectural scalar including the **sequence-head scale
  (1/√32)** that the training checkpoint does not store; `load_pretrained` asserts
  `first_layer.attn_block.seq_heads._scale == config.seq_head_scale`.
- `tools/export_checkpoint.py` keeps only `model.*` keys (≈124 M params, ≈497 MB fp32), drops
  optimizer state, `align_proj.*`, `probe*.*`, remaps key prefixes to the new module names, and
  writes the remap table into `config.json["state_dict_remap"]` for auditability.

### 5.2 Strategies (the extension point)

```python
class ViewStrategy(ABC):
    lmax: int                      # modality is supplied by the caller: the strategy is bound to
                                   # the modality file it is applied to (`strategies={"rna": ..., "atac": ...}`)
    universe: FeatureList | np.ndarray | None   # optional restriction, applied by the base class
    @abstractmethod
    def sample(self, cell: CellFeatures, rng: np.random.Generator) -> list[TokenSet]: ...  # subclasses implement this
    def views(self, cell, rng) -> list[TokenSet]: ...   # public entry point: restrict to `universe`, then `sample`, check `lmax`
```

A custom strategy therefore implements `sample()` on an already universe-restricted cell (worked example:
[strategies.md](strategies.md) §4, test S10).

`CellFeatures` exposes `detected_gene_emb_idx`, `detected_peak_emb_idx`, `gene_values`
(sorted CSR row). Every preset takes `universe: FeatureList | None`; when given, the detected
set is intersected with it before sampling/ranking (this is `StrategyTiledPool`'s `idx_file`
generalised to all presets, and the ATAC analogue of restricting to a gene set). `TokenSet` is `(emb_idx int32[L], kind int8[L])` with `L ≤ lmax`; padding and
collation happen in `tokens.py`, not in strategies. `embed()` averages `h_cell` over the views
a strategy returns (float64 accumulator → float32), matching `evaluate.py`'s `EvalWriter`.

| preset | class | views | paper equivalent |
|---|---|---|---|
| `random` (default) | `Resample(lmax, draws, universe=None)` | `draws` independent random draws of `min(lmax, n_detected)` detected features (without replacement within a draw; features may recur across draws) | `nz_resamp_{lmax}_d{draws}`; with `universe` = `StrategyTiledPool(idx_file=...)` |
| `top` | `TopExpressed(lmax, universe=None)` | rank detected genes by value desc (stable, ties by reference row), take `lmax`; 1 view, no RNG | `nz_top{lmax}` |
| `panel` | `Panel(features, lmax)` | detected ∩ panel, in panel order, truncated to `lmax`; 1 view | `StrategyFixed` with a `*_dummy.yaml` pool (cell-typing panels, MERFISH) |
| `tile` | `Tile(lmax, max_draws, universe=None)` | shuffle once, disjoint `lmax` chunks, `min(max_draws, ceil(n / lmax))` views — the draws partition the detected features (no feature in more than one draw); validated against `evaluate.py` in v0.1.0 (#31, G9) | `nz_nonrepeat_{lmax}_d{n}` (`StrategyTiledPool(draw_mode="tile")`) |

`bloom.strategies.get(name)` / `build(name, ...)` map preset names to classes; `resample` (the pre-review name
of `random`) is accepted with a `DeprecationWarning` for one release.

Determinism: per-cell `rng = np.random.default_rng([seed, cell_index])` (SeedSequence
spawn), so results do not depend on batch size, worker count or device.

### 5.3 Internals (ML track)

Attention weights at L = 8192 cannot be materialized (`B×H×L×L` fp32 = 5 GB per cell for the
20 first-layer heads); the paper's attention analyses were done from exported Q/K. So the API returns Q/K and computes
logits for selected pairs only (`src/bloom/model/internals.py`; how-to and memory table in [internals.md](internals.md)):

```python
@dataclass
class EncodeOutput:
    h_cell: Tensor            # (B, 1024)  bitwise == encode() under inference_mode (I2)
    token_h: Tensor           # (B, L, 1024)  post-BloomFirstLayer token states
    latents: Tensor           # (B, 256, 1024) after the self-attention stack (pre-pool)
    qk: dict[str, tuple[Tensor, Tensor]]   # {"seq": (q,k), "pos": (q,k), "std": (q,k)}, each (B, L, H_g, d_g);
                                            #   H = (8, 8, 4), d = (32, 160, 32); pos has q is k
    scales: dict[str, float]  # softmax scale per group: seq 1/sqrt(32) (asserted at load), pos 1.0, std 1/sqrt(32)
    cross_attn: Tensor | None # (B, 256, L) latent→token attention averaged over heads; (B, 8, 256, L) with per_head=True
    token_ids: list[list[str | None]] | None  # gene symbol / "chr:start-end" per token, None at pads; None without a reference
    lengths: Tensor           # (B,)
    key_padding_mask: Tensor  # (B, L) True = pad
    emb_idx, kind: Tensor     # the inputs

out = model.encode(emb_idx, kind, lengths, return_internals=True,
                   reference=None, cross_attn=True, per_head=False, requires_grad=False, max_tokens=32_768)
bloom.pair_logits(out, pairs=[(i, j), ...], group="seq", batch=0, heads=None) -> Tensor  # (P, H_g) logits, selected pairs only
bloom.capture(model, names) -> context manager yielding {name: last forward output}; hooks removed on exit, nestable
```

Numerics: `h_cell`/`token_h`/`latents` come from the same ops as `encode` (I2 bitwise). `bloom-arch@ef5fb8a` keeps q/k
local to each head's `forward`, so `internals.first_layer_qk` re-applies the block's projection modules (`W_Q`/`W_K`, the
position-head `[β·u ; w⊙r_φ]·√τ` construction) to the same inputs — three cheap matmuls duplicated, the attention is not
recomputed (I3 verifies `softmax(qkᵀ·scale)·W_V` reproduces each head's output). `cross_attn` comes from a second
`PerceiverEncoder.cross_attend(..., need_weights=True)` call; the call that feeds `h_cell` is unchanged (`encode` runs with the
fused CUDA fast path disabled on every device since v0.1.0, #44). Dropout is
always in eval mode; `torch.inference_mode` is used unless `requires_grad=True` (then `token_h.retain_grad()` is set and
`h_cell.sum().backward()` fills `token_h.grad`, I6). A memory guard raises `InternalsMemoryError` before any allocation when
`B·L > max_tokens` (default 32,768 = 0.41 GiB of internals; I7). `return_internals` is never used by `embed()`.

### 5.4 Device & memory policy

- `device="auto"` → `cuda` → `mps` → `cpu`. On CPU the first layer at L = 8192 materialises the attention
  matrix (PyTorch's CPU flash kernel does not engage for fp32) — a measured fact: it does not for fp32, so `embed()` defaults to `batch_size=1` on CPU (hardware table in
  [validation.md](validation.md) §6).
- Precision default fp32 (the paper's evaluations used `32-true`); `bf16` optional on GPU with a
  documented tolerance.

## 6. Roadmap

Phases 0–3 are done and released as `v0.1.0` ([CHANGELOG.md](../CHANGELOG.md)). What comes next — per milestone, with issue
links — is maintained in [ROADMAP.md](../ROADMAP.md) and mirrored by the GitHub milestones `v0.2.0` and `later`; this
document no longer carries its own list.

## 7. On `bloom-sprint-attn`: tag pin vs submodule vs PyPI

Recommendation: **pinned git tag as a pip dependency** in `pyproject.toml` (`bloom-arch @ git+https://github.com/AllenInstitute/bloom-sprint-attn@v0.x.y`).

- A git **submodule** adds a second failure mode for every cloner (`--recurse-submodules`
  forgotten → `ImportError`), does not give pip a package to install, and makes the version
  pin live in `.gitmodules` rather than next to the other dependencies.
- **PyPI first** is only necessary if `bloom` itself goes to PyPI (PyPI rejects git URLs in
  dependencies). Since v1 is clone+conda, a tag is sufficient; publishing bloom-arch to PyPI
  later costs nothing extra and is the right first step of the PyPI roadmap item.
- Two requirements on the refactor, whichever route: (1) the seq-head scale becomes an explicit
  constructor argument (stored in our `config.json`, asserted at load), not a module constant;
  (2) either state-dict key names are preserved or `tools/export_checkpoint.py` carries the
  remap. Add a test in bloom-biorxiv that constructs the block from `config.json` and checks
  `sum(p.numel())` and the scale against pinned values.
