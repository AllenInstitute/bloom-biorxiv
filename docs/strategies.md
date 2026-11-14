# View strategies — how a cell becomes model input

A *view* is one token set of ≤ `lmax` detected features that the model encodes into a 1024-d
`h_cell`. A *strategy* decides which views a cell gets; `bloom embed` averages `h_cell` over them
(float64 accumulator → float32), exactly as the paper's `evaluate.py` did. The interface
([architecture.md](architecture.md) §5.2):

```python
class ViewStrategy(ABC):
    lmax: int
    def views(self, cell: CellFeatures, rng: np.random.Generator) -> list[TokenSet]: ...
```

## 1. The four presets

| preset (`--strategy`) | Python class | views per cell | tokens per view | RNG | paper name |
|---|---|---|---|---|---|
| `random` (default) | `Resample(lmax=8192, draws=4)` | `draws` | up to `lmax` detected features, sampled without replacement **within** a draw; draws are independent, so a feature may recur across draws | yes | `nz_resamp_8192_d4` (RNA); `_d16` for ATAC |
| `top` | `TopExpressed(lmax=4096)` | 1 | the `lmax` most highly expressed detected features; ties by ascending reference row; deterministic | no | `nz_top4096` |
| `panel` | `Panel(features, lmax=8192)` | 1 | `detected ∩ panel`, in panel order (your own feature list) | no | `StrategyFixed` with a `*_dummy.yaml` pool |
| `tile` | `Tile(lmax=8192, max_draws=4)` | `min(draws, ceil(n_detected / lmax))` | one shuffle of the detected set cut into **disjoint** `lmax` chunks — the draws partition the detected features, no feature appears in more than one draw; the tail beyond `draws × lmax` is dropped | yes | `nz_nonrepeat_{lmax}_d{n}` (`StrategyTiledPool(draw_mode="tile")`) |

In words: **`random`** is Monte-Carlo averaging over random subsets of the cell's detected features
(the paper's embeddings); **`top`** is deterministic; **`panel`** embeds every cell on your own feature
list; **`tile`** covers the whole detected set once with disjoint views (sampling *without replacement across
draws*) — `--draws` is the cap on the number of views, the actual count is `ceil(n_detected / lmax)`. Note that with the default `lmax = 8192` an RNA view is almost always the *full* detected set
(the fixture's cells detect 1,087–7,478 mapped genes), so for typical RNA cells `random` and a single
draw coincide and the four draws only average out numerical noise; the draws matter when a cell detects
more than `lmax` features — routinely the case for ATAC (Phase 2, median ~20 k detected peaks).

Defaults (from `config.json` `strategy_defaults`): `--rna` → `random --lmax 8192 --draws 4`;
`top` → `--lmax 4096`. Changing `--strategy` resets `lmax`/`draws` to that preset's defaults unless you
pass them. A cell with more than `lmax` detected genes is sub-sampled (`random`) or truncated
(`top`/`panel`); a cell with fewer gets a shorter view (no padding enters the model).

Which to use:

- **`random`** — the paper's embeddings; independent random draws average out the sampling noise.
  Cost: `draws` forward passes per cell (4 × L ≤ 8192).
- **`top`** — deterministic (bit-reproducible across batch sizes and devices within the tolerances in
  [test-plan.md](test-plan.md) §5) and 1 pass per cell: the choice for regression gates and for cheap
  CPU runs. Depends on expression *values*, so ranking ties (common with counts) follow the reference-table
  row order.
- **`panel`** — embed every cell on the same fixed gene list (cell-typing panels, spatial panels). The
  shipped `hvg_1999` panel is the example used in the tests and tutorials.
- **`tile`** — every detected feature enters exactly one view, so `draws × lmax` tokens cover the cell with no
  overlap between views: the choice when a cell detects far more than `lmax` features (ATAC: 20–40 k peaks,
  `--strategy tile --lmax 8192 --draws 4` covers up to 32,768 of them in 4 passes where `random` re-draws overlapping
  8,192-peak subsets 16 times). For RNA with `lmax = 8192` it coincides with `random`'s single full-set draw. A cell
  with `≤ lmax` detected features gets one view; with `> draws × lmax` the tail of the shuffle is not embedded. Fewer
  draws than `random` means more sampling variance per run (seed-to-seed mean cosine ≈ 0.998 on the 20-cell fixture at
  2–4 draws, validation.md §10).

Naming history: before the Phase 1 review the `random` preset was called `resample`; the old name is still
accepted on the CLI and in `bloom.strategies.build()` with a `DeprecationWarning` for one release. The paper's
`nz_resamp_*` result names and the reference array `rna_random_8192x4` keeps the paper strategy name in its sidecar.

`tile` was kept out of the CLI until it had been validated against the paper pipeline (#31): the references
`rna_tile_4096x2` / `atac_tile_8192x4` in `examples/multiome_20/reference_embeddings.npz` come from the original
`evaluate.py` (`StrategyTiledPool(draw_mode="tile")`); reproducing its per-cell permutation gives `max |Δ|` 1e-6 (G9a),
and with bloom's own seeds the statistical parity equals the strategy's seed-to-seed variance (G9b, validation.md §10).

## 2. `--universe` and `--panel` files

Plain text, one identifier per line, `#` comments and blank lines allowed; gene symbols (as in
`genes.parquet.gene_name`) or peak coordinates — never both in one file. Identifiers absent from the
reference are counted and logged once, then ignored.

- `--universe F` (any preset except `panel`) intersects the *detected* set with `F` before
  sampling/ranking — the generalisation of the paper's `StrategyTiledPool(idx_file=…)`.
- `--panel F` (`panel` only) *is* the view: `detected ∩ F` in file order, truncated to `lmax`.

Shipped lists (`bloom download --what panels` → `$BLOOM_HOME/references/mm10-evo2-v1/panels/`): `hvg_1999.txt` (genes) and
`da_2127.txt` (peaks). Four more lists in the same format are in the repository under `tools/panels/` (not downloaded):
`hvg_8054.txt`, `tf_533.txt`, `merfish_658801.txt` (genes), `enhancer_2298.txt` (peaks).

```bash
bloom embed --rna cells.h5ad --species mouse --strategy panel \
    --panel $BLOOM_HOME/references/mm10-evo2-v1/panels/hvg_1999.txt -o emb_hvg.h5ad
bloom embed --rna cells.h5ad --species mouse --strategy random \
    --universe tools/panels/hvg_8054.txt -o emb_hvg8k.h5ad                                 # repo file, not a downloaded panel
```

## 3. Seeds and reproducibility

Cell `i` of the dataset draws from `np.random.default_rng([seed, i])` (`--seed`, default 0). So
`random` views depend only on `(seed, cell_index)` — not on batch size, device, worker count or
on which other cells are in the file *as long as the cell order is the same*. Reordering or subsetting
cells changes their indices and therefore their draws; the embedding distribution is unchanged.

`top` and `panel` ignore the seed; `tile` uses it for the one shuffle per cell. The paper's `evaluate.py` RNG stream is not reproduced
(its seed was `42 + file_idx·100000 + row_idx` on a different generator), which is why `random`
parity with the Phase 0 references is statistical (mean per-cell cosine > 0.999, test G4), while
`top` is compared element-wise (G1/G2).

## 4. Python API

```python
import bloom
from bloom.strategies import Resample, TopExpressed, Panel, Tile, ViewStrategy, get
from bloom.feature_lists import FeatureList

model = bloom.load_model("hyacinth", device="auto")
ds = bloom.prepare(rna="cells.h5ad", species="mouse", reference=model.reference)

emb = bloom.embed(model, ds)                                       # paper default: random 8192 x 4
emb = bloom.embed(model, ds, {"rna": TopExpressed(lmax=4096)})     # deterministic
emb = bloom.embed(model, ds, {"rna": Resample(8192, 4, universe=FeatureList.from_file("hvg_8054.txt"))})
emb = bloom.embed(model, ds, {"rna": Panel(FeatureList.from_file("hvg_1999.txt"))}, keep_views=True)
emb = bloom.embed(model, ds, {"atac": Tile(lmax=8192, max_draws=4)})          # disjoint draws cover the detected peaks
assert get("random") is Resample                                   # preset name -> class
```

A `FeatureList` universe/panel is resolved against `model.reference` inside `embed()`; call
`strategy.resolve(reference)` yourself when you use a strategy without `embed()`.

A custom strategy is a subclass implementing `sample()` (the universe restriction and the `lmax`
check are done by the base class); it runs through `embed()` unchanged (test S10):

```python
class EvenRows(ViewStrategy):
    name = "even"
    def __init__(self): super().__init__(lmax=8192)
    def sample(self, cell, rng):
        emb = cell.detected_gene_emb_idx
        emb = emb[emb % 2 == 0]
        return [TokenSet.genes(emb)] if emb.size else []
```

## 5. Cost

One forward pass costs roughly the same for every strategy at the same `L`; the strategy sets how
many passes a cell needs (`draws`) and how long they are (`lmax`, bounded by the cell's detected
genes). The first-layer attention is quadratic in `L`: on 16 CPU threads a `top` view (L = 4096) takes
0.45 s, a 7,500-token `random` view 1.36 s; on the L4 0.055 s and ~0.1 s. On CPU keep `batch_size=1`
(RSS 2.7 GB at L = 4096, 5.4 GB at L = 7,500; x4 at batch 4). Full table: [validation.md](validation.md) §6.
