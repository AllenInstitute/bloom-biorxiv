# Input data — what `bloom embed --rna` / `--atac` expects

Companion to [data-schema.md](data-schema.md) §1 (the schema) and [strategies.md](strategies.md)
(how the detected features become model input). Sections 1–6 cover RNA; §7 covers ATAC (`--atac`,
issue #18) and §8 the two-file case.

## 1. The one thing that matters: which genes are detected

The encoder sees a cell as its **set of detected genes** (`X > 0`), looked up in the `mm10-evo2-v1`
gene table (32,129 mouse genes). Nothing else enters the embedding: no library size, no normalization,
no batch label, no cell metadata ([architecture.md](architecture.md) §2). Consequently:

- raw counts, CPM, log1p(CPM) or any positive transform of the same matrix give **identical**
  embeddings for `random` and `panel`; for `top` they give the same ranking as long as the
  transform is monotone within a cell (ties aside);
- scaled / centered matrices (negative entries) are rejected — pass the counts layer instead;
- a cell with no detected *mapped* gene cannot be embedded; it is kept in the output as a NaN row with
  a warning (not silently dropped).

## 2. The AnnData

| Field | Requirement |
|---|---|
| `X`, `layers["counts"]`, `raw.X` or `--layer NAME` | cells × genes, non-negative; dense or sparse (CSR/CSC); `backed="r"` files are read into memory. Precedence: `--layer` → `layers["counts"]` → `raw.X` → `X`. The chosen source is recorded as `value_source`. |
| gene identifiers | `var_names` (default) or `--gene-col COL`. Mouse: MGI symbols (`Pvalb`, `Gad1`, …) or Ensembl ids (`ENSMUSG…`, version suffix ignored). Human: HGNC symbols (`PVALB`) or `ENSG…` ids. |
| `obs` | anything; copied verbatim to the output. |
| `obs_names` | unique (`adata.obs_names_make_unique()` first); duplicates are an error. |

Minimal example (mouse, counts in `X`):

```python
import anndata as ad, numpy as np, scipy.sparse as sp
adata = ad.AnnData(X=sp.csr_matrix(counts), obs=obs_df, var=pd.DataFrame(index=gene_symbols))
adata.write_h5ad("cells.h5ad")
```

```bash
bloom embed --rna cells.h5ad --species mouse -o embeddings.h5ad
```

## 3. Species

- `--species mouse`: symbols are matched exactly against `genes.parquet.gene_name`, then
  case/whitespace-normalised (counted in a warning), Ensembl ids against `gene_id`.
- `--species human`: HGNC symbols / `ENSG` ids go through the shipped one-to-one ortholog table
  (`orthologs_human_mouse_one2one.tsv`, Ensembl 115, 15,676 pairs; [data-schema.md](data-schema.md)
  §1.3) and are joined to the gene table on the **mouse Ensembl id**. About half of the gene universe
  has a strict 1:1 ortholog, so a human dataset maps fewer genes than a mouse one; this is the path the
  paper's human evaluations used.

`var["bloom_mapped_via"]` in the prepared bundle records `exact` / `gene_id` / `ortholog` / `none`
per gene.

## 4. The mapping report

`bloom embed` prints (to stderr) the same report `bloom prepare` prints:

```
species mouse; reference mm10-evo2-v1
[rna] 100 cells x 31899 features; value source layers['counts']
  mapped 31859/31899 ids (99.9%); detected mass mapped 99.9%; via exact 31859, gene_id 0, ortholog 0
  detected mapped features per cell: median 4438.5, min 1087, max 7478; empty cells 0
  most frequent unmapped ids: Ptp4a1, Arhgef4, Gm28040, Zc3h11a, Gm16701, A530058N18Rik, 4933434E20Rik, Pakap, Gm28710, Jakmip1
```
(`bloom prepare --rna tests/data/rna_100.h5ad --species mouse`, 2026-11-13.)

Quality gates (warnings, never errors):

- **detected mass mapped < 70 %** → the most frequent unmapped identifiers are listed. Typical
  causes: Ensembl ids in a column you did not name (`--gene-col gene_ids`), wrong `--species`,
  lower-cased or suffixed symbols, a non-mouse/human organism.
- **median detected genes per cell < 200** → views will be short relative to training (`Lmin` 124,
  `Lmax` 8192 during training); embeddings are still produced.
- **empty cells** → counted; their rows are NaN.

Use `bloom prepare -o PREPARED_DIR` to inspect the report and cache the mapping for a large dataset,
then `bloom embed PREPARED_DIR -o embeddings.h5ad` (bitwise identical to the one-step command).

## 5. Common failures

| Message | Fix |
|---|---|
| `model weights not downloaded (...); run bloom download --what model` | `bloom download --what model,genes` (597 MB from the public bucket, [assets.md](assets.md) §2); or point `BLOOM_HOME` at a tree that already has them. |
| `RNA matrix contains negative values ... scaled/centered data cannot be binarized` | Point at the counts: `--layer counts`, or write `adata.raw` / `layers["counts"]`. |
| `obs_names must be unique` | `adata.obs_names_make_unique()`. |
| `layer 'x' not found` / `gene_col 'x' is not a var column` | Check `adata.layers.keys()` / `adata.var.columns`. |
| `only 12.3% of the detected gene mass maps onto the reference` (warning) | Wrong `--species`, ids instead of symbols, or an unsupported organism. |
| `--species human but 97.3% of the 31899 gene identifiers match mouse symbols ... --force-species` (exit 1) | The ids look like the other species (#33): re-run with the suggested `--species`, or `--force-species` to map anyway. |
| `[rna] no cell has a detected gene that maps onto the reference ... embeddings would be all NaN` (exit 1) | Nothing is mappable (#34): wrong species, Ensembl ids in a symbol column, non-mouse/human organism or an all-zero matrix. Partially empty inputs stay a warning. |
| `peak table not downloaded; run bloom download --what peaks` | `--atac` needs `peaks.parquet` (72 MB, matching) and `peaks.safetensors` (5.4 GB, embedding). |
| `no ATAC var identifier parses as a peak coordinate` | `var_names` must be `chr:start-end` (or `-`/`_` separated), or give `var["chrom"]`, `var["start"]`, `var["end"]` (§7). |
| `ATAC input must be in mm10 coordinates: --species human is only supported for RNA` | Peaks are matched on mm10 coordinates; embed a human RNA file alone. |
| `the RNA and ATAC files share no obs_names` | With two files, cells are aligned by `obs_names`; rename the cells consistently or embed the files separately. |
| `--species mouse|human is required for a raw .h5ad input` | A raw file needs the species; a prepared bundle carries it. |

Exit codes: `0` ok, `1` user error (the message names the fix), `2` not implemented / missing argument.

## 6. Output

`embeddings.h5ad` ([data-schema.md](data-schema.md) §4): `obs` = your `obs` +
`bloom_n_genes_detected` + `bloom_n_tokens_rna` (mean tokens per view) + `bloom_n_views_rna`;
`obsm["X_bloom_rna"]` float32 `(n_cells, 1024)` (NaN rows for empty cells);
`obsm["X_bloom_rna_views"]` `(n_cells, draws, 1024)` with `--keep-views`; `X` empty `(n_cells, 0)`;
`uns["bloom"]` = provenance (package + git version, model id and sha256, reference id, strategy and seed,
device, precision, torch version, timestamp, mapping summary, run timing).

```python
import anndata as ad
emb = ad.read_h5ad("embeddings.h5ad")
X = emb.obsm["X_bloom_rna"]            # (n_cells, 1024)
emb.uns["bloom"]["modalities"]["rna"]  # strategy, seed, source, mapping, run
```

## 7. ATAC input (`--atac`)

The encoder sees an ATAC cell as its **set of detected reference peaks** (`X > 0`), looked up in the
`mm10-evo2-v1` peak table (1,747,701 peaks of 501 bp, mm10). Fragment / insertion counts are binarized;
only which reference peaks are hit matters (plus the counts for the `top` strategy, which ranks by count).

| Field | Requirement |
|---|---|
| `X` | cells × peaks, non-negative counts; dense or sparse. Only `X` is read (no layer precedence for ATAC). |
| peak coordinates | `var_names` as `chr1:3000000-3000500`, `chr1-3000000-3000500` or `chr1_3000000_3000500` (`chr` prefix optional, `MT` → `chrM`), **or** `var["chrom"]`, `var["start"]`, `var["end"]` (used when all three are present). Must be **mm10**. |
| contigs | `chr1..19`, `chrX`, `chrY`, `chrM` are kept; everything else (`chrUn_*`, `*_random`, `GL*`, `JH*`) is dropped and counted in the report. |
| `obs_names` | unique; used to align with an RNA file when both are given (§8). |

Matching the user's peaks onto the reference (`--peak-match`, [data-schema.md](data-schema.md) §1.2):

- `exact` — `(chrom, start, end)` identical to a reference peak. This is how the training data was built and
  the right mode when the data were called on the same peak set (e.g. the Allen multiome releases).
- `overlap` — a reference peak is detected when any detected user peak covers at least `--min-overlap`
  (fraction of the 501-bp reference peak, default **0.5**) of it. One user peak can hit several reference
  peaks and several user peaks one reference peak (detection OR-ed, counts summed). This is the mode for
  peaks called by another pipeline (MACS2, ArchR, SnapATAC2, Cell Ranger ARC ...).
- `auto` (default) — `exact` when ≥ 95 % of the parsed peaks match exactly, else `overlap`; the choice and
  the exact rate are printed and recorded in `uns["bloom"]`.

`--min-overlap` recommendation: **keep the default 0.5** (G6 table in [validation.md](validation.md) §7:
the fixture's coordinates shifted by 50 / 150 / 250 bp as a synthetic "other peak caller").
At 0.5 a peak shifted by up to half a reference peak (250 bp) still maps onto exactly its originating reference peak
(cosine 1.0 vs exact matching); 0.25 lets neighbouring reference peaks in (mean cosine 0.97 at 150 bp, 0.90 at 250 bp);
0.75 maps nothing beyond a 125-bp shift. Lower the threshold only when your peaks are narrow (< 250 bp) and the
report shows a low mapping rate; raise it when you want near-exact correspondence and accept losing peaks. Wide peaks
(≥ 1 kb) cover several reference peaks fully and detect all of them at any threshold. The report line
`peak match overlap (min_overlap 0.5); exact rate 12.3%` plus `mapped N/M parsed peaks` tells you what happened.

Quality gates (warnings): detected peak mass mapped < 70 %, median detected reference peaks per cell < 200,
empty cells (NaN rows). Peaks that parse but hit nothing are listed as *most frequent unmapped ids*. An input
in which **no** cell has a detected mapped peak is an error (exit 1, #34).

Defaults reproduce the paper's ATAC evaluation: `random`, `--lmax 8192`, `--draws 16` (per-modality block,
`config.json` `strategy_defaults`). `--atac-strategy tile --atac-lmax 8192 --atac-draws 4` covers the detected peaks
once with disjoint views (no peak in more than one draw; `ceil(n_detected / 8192)` views, at most 4; docs/strategies.md
§1). `--atac-strategy top` ranks the reference peaks by summed count (ties by
reference row) and is deterministic; `--atac-universe peaks.txt` restricts the detected set (e.g. the shipped
`panels/da_2127.txt`).

```bash
bloom embed --atac cells_atac.h5ad -o embeddings.h5ad                     # species implied (mm10 peaks)
bloom embed --atac cells_atac.h5ad --peak-match overlap --min-overlap 0.5 -o embeddings.h5ad
bloom prepare --atac cells_atac.h5ad -o prepared/                           # inspect the matching report first
```

Output: `obsm["X_bloom_atac"]` float32 `(n_cells, 1024)`, `obs["bloom_n_peaks_detected"]`,
`obs["bloom_n_tokens_atac"]`, `obs["bloom_n_views_atac"]`; `uns["bloom"]["modalities"]["atac"]["mapping"]`
carries `peak_match`, `min_overlap`, `exact_rate`, `n_ref_peaks_hit`.

Resources (measured on the 20-cell fixture, [validation.md](validation.md) §7): matching 483k user peaks against the
full table takes ~4 s (interval index, 1.4 s to build); the peak table itself is 5.4 GB (`bloom download --what peaks`) and
is loaded only when an ATAC file is embedded — on CPU it is read into RAM (peak RSS 12.3 GB for the default run), on a GPU it
takes 5.4 GB of VRAM (8.9 GB peak at batch 8). The paper default (`random`, 16 draws at L = 8192) costs ≈ 1.4 s per view on
16 CPU threads, i.e. ≈ 22 s per cell (7.3 min for 20 cells) versus ≈ 0.1 s per view on an L4: use a GPU for real datasets, or
`--atac-draws 4` / `--atac-strategy top` on CPU.

## 8. RNA and ATAC together

`bloom embed --rna RNA.h5ad --atac ATAC.h5ad --species mouse -o embeddings.h5ad` embeds each file with its
own strategy block and writes **one** output for the cells present in both files (`obs_names`
intersection, RNA order), with `obsm["X_bloom_rna"]` and `obsm["X_bloom_atac"]`; `obs` is the RNA `obs`
plus the ATAC-only columns (conflicting ATAC columns are suffixed `_atac`). Cells present in one file only
are dropped with a note in the report, or written to `embeddings_rna_only.h5ad` / `embeddings_atac_only.h5ad`
with `--keep-unmatched`. Disjoint `obs_names` are an error. The two embeddings are independent runs (not a
mixed view); with identical cell order the ATAC embedding equals the `--atac`-alone run for the same seed.

Per-modality overrides: `--rna-strategy/--rna-lmax/--rna-draws/--rna-universe/--rna-panel` and the
`--atac-*` counterparts (the dotted spelling `--rna.lmax` is accepted too); the generic `--strategy`,
`--lmax`, `--draws`, `--universe`, `--panel` apply to every modality given.
