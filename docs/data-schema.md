<!-- Imported verbatim from the design document DATA_SCHEMA.md (2026-11-12); only the repository name and cross-links between the five documents were updated. Section 1.3 and the panel list in section 3 updated 2026-11-13 to the staged assets (issue #7); section 1.3 updated 2026-11-13 to the derived ortholog table (issue #14). -->
> Repository: `github.com/alleninstitute/bloom-biorxiv`.

# bloom-biorxiv — data schema

> Section 1.2, 2 and 4 updated 2026-11-13 with the Phase 2 (ATAC) behaviour as implemented (issue #18): `auto` matching,
> summed counts in the prepared `atac.h5ad`, `*_only.h5ad` bundle members, `--species human` refused for ATAC.

Three layers: (1) what the user provides, (2) the prepared/canonical dataset, (3) the output.
Reference-table formats are in assets.md.

## 1. User input

The user supplies **one AnnData per modality** and nothing else: `--rna RNA.h5ad`, `--atac ATAC.h5ad`,
or both. Neither is required on its own; at least one must be given. Each modality file is
prepared and embedded independently (one strategy, one embedding per modality); when both are
given, cells are aligned by `obs_names` only so that the two embeddings can be written for the
same cells. There is no "primary" modality.

### 1.1 RNA AnnData (`--rna`, optional)

| Field | Requirement | Notes |
|---|---|---|
| `X` (or `layers[counts_layer]`, or `raw.X`) | cells × genes, non-negative, dense or CSR | Lookup order: `--layer` if given → `layers["counts"]` → `raw.X` → `X`. Negative values → error ("scaled/centered data cannot be binarized"). |
| gene identifiers | `var_names` by default, or `--gene-col` | Mouse: MGI symbols (as in the mm10 gene table, e.g. `Pvalb`). Human: HGNC symbols. Ensembl IDs are detected (`ENSMUSG`/`ENSG` prefix) and mapped through the reference table's `gene_id` column when present. |
| `obs` | any | Copied verbatim to the output. |
| `obs_names` | unique | Used to align with the ATAC file when both are given. Duplicates → error. |

Value type (counts vs normalized) is not inferred and not needed: only `X > 0` is used. Library-size
columns are not collected in v1 (dropped until the decoder path is exposed).

Quality gates (warn, do not fail): mapping rate < 70% of detected-gene mass → warning naming the
most frequent unmapped identifiers (typical cause: Ensembl IDs, wrong species, lowercase symbols);
median detected genes per cell < 200 → warning (views will be short relative to training).

### 1.2 ATAC AnnData (`--atac`, optional)

| Field | Requirement | Notes |
|---|---|---|
| `X` | cells × peaks, non-negative | Fragment or insertion counts; binarized. |
| peak coordinates | `var_names` as `chr1:3000000-3000500` or `chr1-3000000-3000500` or `chr1_3000000_3000500`; or columns `chrom`,`start`,`end` in `var` | Must be mm10. `chr` prefix normalized; contigs outside `chr1..19, X, Y, M` dropped with a count. |
| `obs_names` | unique | When an RNA file is also given, the two `obs_names` sets are intersected for the combined output; cells present in only one file get only that modality's embedding. ATAC alone is a complete input. |

Peak matching modes (`--peak-match`, default `auto`; `bloom.peaks`):
- `exact`: `(chrom, start, end)` equality with the reference peak set (how the training data was built).
- `overlap`: a reference peak is *detected* in a cell if any detected user peak covers ≥ `--min-overlap`
  (fraction of the 501-bp reference peak; default 0.5, measured in Phase 2: validation.md §7 / input-data.md §7) of
  it. Many-to-one and one-to-many mappings are OR-ed (the stored value is the sum of the mapped user counts).
  Coordinates are half-open `[start, end)`; a user peak that merely touches a reference peak does not overlap it.
- `auto`: exact when ≥ 95 % of the parsed user peaks match exactly, else overlap; the decision
  (`peak_match`, `exact_rate`) is recorded in the report and in `uns["bloom"]`.

ATAC coordinates are mm10 in every case, so `--species human` is accepted for RNA only; an ATAC file with
`--species human` is refused. ATAC alone implies `--species mouse`.

### 1.3 Species and orthologs

`--species human` maps HGNC symbols (or `ENSG` ids, version suffix ignored) → mouse genes through the single shipped
table `orthologs_human_mouse_one2one.tsv`: Ensembl **release 115** one-to-one human→mouse orthologs (human
protein-coding genes, orthology confidence 1) restricted to the 32,129-gene mm10 universe — 15,676 pairs (decision
2026-11-13, issue #9; derived in issue #14, provenance in the file's `#` header and docs/assets.md §1). The join to the gene
table is on the mouse Ensembl `gene_id` (`genes.parquet.gene_id`; the shipped table also carries the resolved `emb_idx`),
never on the mouse symbol (symbol lookup alone misses 486 of the 15,676 pairs whose symbol drifted between Ensembl 115 and
the Evo2 gene table, e.g. `ATP5PO`); the table's mouse symbol is used only as a fallback for a custom table whose
`mouse_gene_id` is absent from the gene table. Human-side matching is by HGNC symbol (exact, then case/whitespace-normalised
with a counted warning) with `ENSG` ids accepted. `var["bloom_mapped_via"]` records `exact` (mouse symbol) / `gene_id`
(mouse `ENSMUSG` id) / `ortholog` / `none`; `var["bloom_symbol"]` is the gene-table symbol of the mapped row
(`bloom.reference.ReferenceTables.map_genes`).

## 2. Prepared dataset (`bloom prepare` output; also the in-memory object)

Directory bundle, mirroring the modality-split layout the model was trained on but with the
feature maps embedded in `var` (no side CSVs):

```
PREPARED_DIR/
├── manifest.json       # schema_version, reference_id, species, created, modalities present, mapping reports,
│                       #   source file hashes, alignment (both modalities), unmatched_files (--keep-unmatched)
├── rna.h5ad            # cells × genes   (present iff --rna was given)
├── atac.h5ad           # cells × peaks   (present iff --atac was given)
├── rna_only.h5ad       # --keep-unmatched with both files: cells present in the RNA file only
└── atac_only.h5ad      # --keep-unmatched with both files: cells present in the ATAC file only
```

With both files the two prepared AnnData hold the **same cells in the same (RNA) order** — the
intersection of the two `obs_names` sets; the remainders are dropped with a warning, or kept in the
`*_only.h5ad` members with `--keep-unmatched`. Disjoint `obs_names` are an error naming ids of both files.

### 2.1 `rna.h5ad`

| Location | dtype | Meaning |
|---|---|---|
| `X` | CSR float32 | Original values (not binarized) — needed for `top` ranking. |
| `obs[...]` | — | User `obs`, verbatim. |
| `obs["bloom_n_genes_detected"]` | int32 | Number of detected *and mapped* genes. |
| `var["bloom_symbol"]` | str | Symbol used for lookup (after ortholog mapping). |
| `var["bloom_emb_idx"]` | int32 | Row in the reference gene table; `-1` if unmapped. |
| `var["bloom_mapped_via"]` | category | `exact` / `gene_id` / `ortholog` / `none` (section 1.3). |
| `uns["bloom"]` | dict | `{schema_version, reference_id, species, value_source, mapping: {n_var, n_mapped, detected_mass_mapped}}` |

### 2.2 `atac.h5ad`

| Location | dtype | Meaning |
|---|---|---|
| `X` | CSR float32 | **Reference-peak** matrix (cells × n_reference_peaks_hit); columns are reference peaks that were hit by ≥ 1 user peak, so the matrix stays small. The value is the sum of the mapped user peaks' counts (unchanged counts under 1:1 exact matching); detection is `X > 0`, the magnitude only serves the `top` strategy (rank by count). |
| `var_names` | str | Reference peak `chr:start-end`. |
| `var["bloom_emb_idx"]` | int32 | Row in the reference peak table (never −1 here). |
| `var["n_user_peaks"]` | int32 | How many user peaks mapped onto this reference peak. |
| `obs["bloom_n_peaks_detected"]` | int32 | Detected reference peaks per cell. |
| `uns["bloom"]` | dict | `{schema_version, reference_id, species, modality: "atac", peak_match, peak_match_requested, min_overlap (overlap mode only), exact_rate, n_user_peaks, n_user_peaks_valid, n_user_peaks_mapped, n_ref_peaks_hit, dropped_contigs, n_unparsed, mapping: {...the report...}, source}` |

### 2.3 Internal batch (what strategies produce; `tokens.py`)

```
TokenSet:  emb_idx int32[L], kind int8[L]  (1=gene, 2=peak; one value per view), L ≤ lmax
Batch:     emb_idx int32[B, lmax] (pad = -1), kind int8[B, lmax] (pad = 0),
           lengths int64[B], cell_index int64[B], view_index int64[B]
```
The model resolves `emb_idx/kind → cseq, chr, bp` on device (as `resolve_embeddings` does
today); `clocal` is a constant 1 for real tokens and is created inside `encode()`. A view holds **one**
modality: a `TokenSet` or a batch row mixing `kind` 1 and 2 is rejected (`ValueError` /
`MixedModalityViewError`, #46).

## 3. Strategy config (user-editable; serialized into the output)

```yaml
# one block per modality file given; defaults below are the paper's
rna:
  strategy: random          # random | top | panel | tile   (`resample` = deprecated name of random)
  lmax: 8192
  draws: 4
atac:
  strategy: random
  lmax: 8192
  draws: 16
# full key set per block:
strategy: random          # random | top | panel | tile
lmax: 8192
draws: 4                  # random: independent draws of up to lmax detected features (exactly `draws` views);
                          # tile: cap on the disjoint draws — views = min(draws, ceil(n_detected / lmax))
universe: null            # optional path: one gene symbol or peak `chr:start-end` per line;
                          #   detected features are intersected with it before sampling/ranking
panel: null               # panel only: ordered feature list that *is* the view
seed: 0
```

Feature-list files: plain text, one identifier per line, `#` comments allowed; gene symbols or
peak coordinates in any of the accepted formats (§1.2). Identifiers absent from the reference
are reported once and ignored. The shipped panels (`tf_533`, `hvg_1999`, `hvg_8054`,
`merfish_658801` — 500 gene ids, named after the MERFISH panel id — for genes; `da_2127`, `enhancer_2298` for peaks;
see assets.md §1) use the same format.

## 4. Output (`embeddings.h5ad`)

| Location | dtype | Meaning |
|---|---|---|
| `obs` | — | User `obs` (from the RNA file if given, else the ATAC file; columns only in the ATAC file are added, conflicting columns from the ATAC file are suffixed `_atac`), plus `bloom_n_genes_detected` / `bloom_n_peaks_detected`, `bloom_n_tokens_{rna,atac}` (float32: mean tokens per view) and `bloom_n_views_{rna,atac}` (int). |
| `obsm["X_bloom_rna"]`, `obsm["X_bloom_atac"]` | float32 (n_cells × 1024), one per modality given | Mean over views of `h_cell`. Cells with zero detected features are **kept** with NaN rows and a warning (not silently dropped as `evaluate.py` does). |
| `obsm["X_bloom_rna_views"]`, `obsm["X_bloom_atac_views"]` | float32 (n_cells × draws × 1024), optional (`--keep-views`) | Per-view embeddings for variance analysis. |
| `var` | empty | `X` is an empty sparse matrix (n_cells × 0), as today. |
| `uns["bloom"]` | dict | `{package_version, git_sha, model_id, model_sha256, reference_id, species, modalities: {rna: {strategy: {...}, seed, source: sha/path, mapping, run}, atac: {... mapping also carries peak_match, min_overlap, exact_rate, n_ref_peaks_hit}}, alignment (both files), device, precision, torch_version, created}` |

One output per modality file given: `--rna` → `obsm["X_bloom_rna"]`, `--atac` → `obsm["X_bloom_atac"]`.
With a single modality the key is still modality-specific (there is no generic `X_bloom`), so
downstream code never has to guess which modality an embedding came from. With both files, one
`embeddings.h5ad` is written whose `obs` is the intersection of the two `obs_names` sets (cells
present in only one file are written to `embeddings_rna_only.h5ad` / `embeddings_atac_only.h5ad`
when `--keep-unmatched` is set). The two embeddings are independent runs, not a mixed view.

## 5. Schema versioning

`schema_version = "1.0"` in every `uns["bloom"]` / `manifest.json`. Breaking changes bump the
major; additions bump the minor. The loader refuses prepared bundles with a higher major.
