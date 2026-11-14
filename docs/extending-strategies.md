> Repository: `github.com/alleninstitute/bloom-biorxiv`.

# Extending strategies — writing your own `ViewStrategy`

A *strategy* decides which detected features of a cell become the tokens of one or more forward passes
(`views`); `embed()` averages `h_cell` over the views. The three presets (`random`, `top`, `panel`) are documented in
[strategies.md](strategies.md). This page is the how-to for a custom one. Spec: [architecture.md](architecture.md)
§5.2; contract test S10 in [test-plan.md](test-plan.md).

## 1. The ABC

```python
class ViewStrategy(ABC):                      # bloom.strategies.ViewStrategy (src/bloom/strategies/base.py)
    name: str = "custom"                      # goes into uns["bloom"] provenance; override it
    def __init__(self, lmax: int, universe: FeatureList | np.ndarray | None = None): ...
    @abstractmethod
    def sample(self, cell: CellFeatures, rng: np.random.Generator) -> list[TokenSet]: ...
    def views(self, cell, rng) -> list[TokenSet]:     # the entry point embed() calls: restrict(universe) -> sample -> lmax check
    def params(self) -> dict: ...                     # provenance; extend it if your strategy has parameters
```

You implement `sample()` and nothing else. What the base class does around it:

- `universe` (optional) is intersected with the detected set **before** `sample()` sees the cell — a `FeatureList`
  (ids, resolved against `model.reference` inside `embed()`) or an `emb_idx` array. Your `sample()` never has to
  filter by universe itself.
- every emitted view is checked against `lmax`; a longer one raises `ValueError`.
- `rng` is `np.random.default_rng([seed, cell_index])` (`bloom.strategies.cell_rng`), so a stochastic strategy is
  reproducible per cell and independent of batch size, worker count and device. Deterministic strategies ignore it.

Inputs and outputs (`bloom.tokens`):

| object | fields | notes |
|---|---|---|
| `CellFeatures` | `cell_index`, `detected_gene_emb_idx (n,) int32` strictly increasing, `gene_values (n,) float32` aligned, `detected_peak_emb_idx (n,) int32` | one modality populated at a time (`cell.modality` is `"gene"` or `"peak"`); `emb_idx` = row of `genes.parquet` / `peaks.parquet`, never the AnnData column |
| `TokenSet` | `emb_idx (L,) int32`, `kind (L,) int8` | build with `TokenSet.genes(emb_idx)` / `TokenSet.peaks(emb_idx)` / `TokenSet.of(emb_idx, cell.modality)`; `L <= lmax`; order is the token order the model sees (the first layer is permutation-equivariant up to padding, so order does not change `h_cell`) |

Return `[]` for a cell with nothing to emit; `embed()` writes a NaN row and warns with the count.

## 2. A 12-line example — exclude a chromosome

Genes on chrX are excluded through the reference table; everything else is the paper's `random` view. The universe is
passed as an `emb_idx` array, so the base class does the restriction and the subclass stays a plain `Resample`:

```python
import numpy as np, bloom
from bloom.strategies import Resample

class NoChrX(Resample):
    """random 8192 x 4 over autosomal + chrY genes only (chrX dropped via genes.parquet)."""
    name = "no_chrx"

    def __init__(self, reference, lmax=8192, draws=4):
        genes = reference.genes                                  # DataFrame indexed by emb_idx
        keep = genes.index.to_numpy()[genes["chrom"].to_numpy() != "chrX"]
        super().__init__(lmax=lmax, draws=draws, universe=keep)

model = bloom.load_model("hyacinth", device="auto")
ds = bloom.prepare(rna=adata, species="mouse", reference=model.reference)
emb = bloom.embed(model, ds, strategies={"rna": NoChrX(model.reference)})   # obsm["X_bloom_rna"], uns["bloom"]...["strategy"] == "no_chrx"
```

Measured in the Phase 1 review (validation.md §5): 6.3 s / 100 fixture cells on the L4; cosine to the all-gene view mean
0.9990 (min 0.9975).

A strategy that is not a restriction of a preset implements `sample()` directly, e.g. "the `lmax` detected genes with the
*smallest* non-zero values" (a 9-line subclass):

```python
from bloom.strategies import ViewStrategy
from bloom.tokens import TokenSet

class BottomExpressed(ViewStrategy):
    name = "bottom"

    def sample(self, cell, rng):
        emb, val = cell.detected_gene_emb_idx, cell.gene_values
        order = np.argsort(val, kind="stable")[: self.lmax]      # ascending value, ties by emb_idx
        return [TokenSet.genes(emb[order])] if order.size else []
```

## 3. Passing it to `embed()` and the CLI

```python
emb = bloom.embed(model, ds, strategies={"rna": BottomExpressed(lmax=4096)}, seed=0, batch_size=8, keep_views=True)
```

`strategies` maps modality → strategy instance; a modality present in `ds` but absent from the dict gets the paper
default. `uns["bloom"]["rna"]["strategy"]` records `strategy.params()` — override `params()` to add your own
parameters (keep it JSON-serialisable). Custom classes are Python-only: the CLI's `--strategy` accepts preset names
(`bloom.strategies.PRESETS`); to run a custom strategy from a shell, write a 10-line script that calls `bloom.embed`.

Without `embed()` (the ML path): `views = strat.views(ds.cell_features(i, "rna"), cell_rng(seed, i))`, then
`batch = collate(views)` and `model.encode(batch.emb_idx, batch.kind, batch.lengths)` — or `return_internals=True`
([internals.md](internals.md)). A `FeatureList` universe needs `strat.resolve(model.reference)` first in this path.

## 4. Checklist before you trust a custom strategy

1. `len(ts) <= lmax` for every view and `[]` on an empty cell (the base class enforces the first; S10 shows the shape).
2. Determinism: `strat.views(cell, cell_rng(seed, i))` twice gives identical views (S4 semantics).
3. Modality: a gene strategy emits only `kind == 1`, a peak strategy only `kind == 2` (S7); never mix in one view (v1).
4. Compare to the paper default on a few hundred cells (`tools/bench_embed.py`-style cosine, or kNN label accuracy as in
   validation.md §3) before using the embeddings downstream.
