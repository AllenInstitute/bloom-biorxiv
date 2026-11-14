# bloom-biorxiv assets — `s3://bloom-biorxiv`

Public, anonymous-read bucket holding the inference assets of **Bloom**, a multimodal single-cell
transformer trained on mouse brain RNA and ATAC data (Allen Institute). Bloom turns a cell's set of
detected genes (RNA) or accessible peaks (ATAC) into a 1024-d embedding by attending over
sequence-derived feature embeddings of the detected features; the released checkpoint is
**`hyacinth`** and the reference tables it was trained against are **`mm10-evo2-v1`**
(mm10 genes and peaks with Evo2-derived 768-d sequence embeddings). Everything here is what the
companion code needs to run `bloom embed`; training code and training data are not included.

- Code: <https://github.com/alleninstitute/bloom-biorxiv>
- Preprint: Gala, Convy, Marghi et al., *Bloom unifies brain cell representations across modalities and contexts*, bioRxiv 2026, doi:10.64898/2026.11.11.765881 (<https://doi.org/10.64898/2026.11.11.765881>)
- Regional base URL (use this one; the region-less hostname answers 307):
  `https://bloom-biorxiv.s3.us-west-2.amazonaws.com/`

## Layout

The bucket layout is the release layout: every key is the `path` of a `MANIFEST.json` entry, so a
local cache (`BLOOM_HOME`, default `~/.cache/bloom`) is a verbatim mirror.

| Path | What it is | Size | Version |
|---|---|---:|---|
| `MANIFEST.json` | single source of truth: path, `size_bytes`, `sha256`, group, version, license per file (15 files, 6,076,570,255 B) | 6 KB | manifest v1 |
| `README.md` | this file (not a manifest entry) | — | — |
| `models/hyacinth/model.safetensors` | released weights, fp32, weights-only, 198 tensors, 124,144,831 parameters | 496,624,516 B | `hyacinth` |
| `models/hyacinth/config.json` | architecture scalars, `seq_head_scale`, strategy defaults, `reference_id`, provenance | 3,923 B | `hyacinth` |
| `models/hyacinth/MODEL_CARD.md` | model card (not a manifest entry) | — | `hyacinth` |
| `references/mm10-evo2-v1/genes.safetensors` | gene table: `cseq` f32 `[32129, 768]`, `chr` i8, `bp` i32 | 98,861,805 B | `mm10-evo2-v1` |
| `references/mm10-evo2-v1/genes.parquet` | gene metadata (`gene_name`, `gene_id`, `chrom`, `start`, `end`, `strand`; row = `emb_idx`) | 1,733,063 B | `mm10-evo2-v1` |
| `references/mm10-evo2-v1/peaks.safetensors` | peak table: `cseq` f32 `[1747701, 768]`, `chr`, `bp` — needed for ATAC only | 5,377,676,873 B | `mm10-evo2-v1` |
| `references/mm10-evo2-v1/peaks.parquet` | peak metadata (`chrom`, `start`, `end`, `name = chr:start-end`; row = `emb_idx`) | 72,448,807 B | `mm10-evo2-v1` |
| `references/mm10-evo2-v1/peaks.bed.gz` | same peaks as gzip BED for re-quantification (snapatac2 / ArchR / featureCounts) | 20,520,259 B | `mm10-evo2-v1` |
| `references/mm10-evo2-v1/orthologs_human_mouse_one2one.tsv` | Ensembl 115 one-to-one human -> mouse pairs (15,676), restricted to the gene universe; `#` header = provenance | 837,340 B | `mm10-evo2-v1` |
| `references/mm10-evo2-v1/panels/{hvg_1999,da_2127}.txt` | gene panel `hvg_1999` (1,999 genes) and peak panel `da_2127` (2,127 peaks); one id per line | 12,548 B / 51,207 B | `mm10-evo2-v1` |
| `examples/multiome_20/rna.h5ad` | 20 example cells (10 AIT33 subclasses x 2) from the Allen whole-mouse-brain multiome data: RNA log1p(CPM) + counts layer over 31,899 genes; `obs` carries `AIT33_class`, `AIT33_subclass`, `source_row` | 1,398,379 B | `examples-v2` |
| `examples/multiome_20/atac.h5ad` | the same 20 cells: raw ATAC counts on the 483,249 reference peaks they detect (`var["emb_idx"]` = row of the peak table) | 5,912,106 B | `examples-v2` |
| `examples/multiome_20/cells.tsv` | `obs_name, source_row, AIT33_subclass, AIT33_class` | 1,452 B | `examples-v2` |
| `examples/multiome_20/reference_embeddings.npz` | reference embeddings of the 20 cells from the original pipeline: `rna_top4096`, `rna_random_8192x4`, `rna_tile_4096x2`, `atac_random_8192x16`, `atac_tile_8192x4`, each also as `*_tf32off` (TF32 off + unfused attention), plus `obs_names`; float32 `(20, 1024)` | 767,956 B | `examples-v2` |
| `examples/multiome_20/reference_embeddings.json` | provenance per array: strategy, seed, numerics flags, code commits, input sha256, run wall, parity vs the paper's embeddings | 40,473 B | `examples-v2` |

## Fetching

With the package (recommended — verifies size and sha256, resumes interrupted transfers):

```bash
git clone https://github.com/alleninstitute/bloom-biorxiv.git && cd bloom-biorxiv
scripts/setup_env.sh bloom-biorxiv cpu          # or cu128; prints the env prefix
<prefix>/bin/bloom download --what model,genes  # RNA inference: 0.6 GB into ~/.cache/bloom
<prefix>/bin/bloom download --what examples     # 20-cell RNA + ATAC fixture + reference embeddings: 8 MB
<prefix>/bin/bloom download --what peaks        # ATAC: + 5.4 GB
<prefix>/bin/bloom download                      # everything, 6.1 GB
<prefix>/bin/bloom info --verify                 # re-hash the cache against MANIFEST.json
```

Without the package, any HTTP client works (no credentials, no AWS tooling needed):

```bash
B=https://bloom-biorxiv.s3.us-west-2.amazonaws.com
curl -fsSLO $B/MANIFEST.json
curl -fsSL --create-dirs -o models/hyacinth/model.safetensors $B/models/hyacinth/model.safetensors
curl -fsSL --create-dirs -o references/mm10-evo2-v1/genes.parquet $B/references/mm10-evo2-v1/genes.parquet
sha256sum -c <(python -c "import json; [print(f['sha256'], ' ', f['path']) for f in json.load(open('MANIFEST.json'))['files']]") 2>/dev/null | grep -v 'No such file'
```

`aws s3 cp --no-sign-request s3://bloom-biorxiv/<key> .` also works. Objects carry a `sha256` user
metadata header (`x-amz-meta-sha256`) equal to the manifest digest.

## Integrity

`MANIFEST.json` lists every file with its exact size and sha256; the package pins the manifest's own
sha256 (`bloom.assets.MANIFEST_SHA256`) so a given code version always knows which bytes it was tested
against. `bloom download` refuses to keep a file whose hash does not match; `bloom info --verify`
re-hashes a cache (about 6 s for all 6.1 GB on a local SSD). `examples/multiome_20/reference_embeddings.json`
records how every reference array was produced.

## Licensing

- **Everything in this bucket** — model weights (`models/`), derived reference tables (`references/`),
  example cells (`examples/`, a subset of the Allen Institute whole-mouse-brain multiome dataset) and the
  model card — is released under **[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)**
  (Creative Commons Attribution-NonCommercial 4.0 International). Every `MANIFEST.json` entry carries
  `"license": "CC-BY-NC-4.0"`; attribution: Allen Institute and the Bloom preprint (Gala, Convy, Marghi et al. 2026, doi:10.64898/2026.11.11.765881).
- **Companion code**: MIT (repository `LICENSE`).

## Contact

Questions and problems: open an issue at <https://github.com/alleninstitute/bloom-biorxiv/issues>
(use the `use-case.md` template for a reproducible report).

## Changelog

- **2026-11-14** — public repository `alleninstitute/bloom-biorxiv`. `models/hyacinth/config.json`,
  `examples/multiome_20/reference_embeddings.json` and the header of `orthologs_human_mouse_one2one.tsv` lost host-,
  environment- and path-specific provenance strings (numeric facts, checksums and commit ids unchanged; no array or
  table row changed). `MANIFEST.json` regenerated: 15 files, sha256 `38f02744…`. `README.md` and `MODEL_CARD.md`
  point at the public repository.
- **2026-11-14** — `examples/multiome_20/reference_embeddings.{npz,json}` gained the `tile` preset references
  `rna_tile_4096x2`, `atac_tile_8192x4` (+ `*_tf32off`; original `evaluate.py`, `StrategyTiledPool(draw_mode="tile")`),
  appended by `tools/make_reference_npz.py multiome-20-add` (the six existing arrays are byte-identical). `MANIFEST.json`
  regenerated: 15 files, sha256 `d6eee1b9…`. `v0.1.0` was re-cut on the commit that closes #31 / #46 / #51.
- **2026-11-13** — `v0.1.0` tagged (first public inference release of the companion code). Licenses decided: assets
  CC BY-NC 4.0, code MIT; `MANIFEST.json` regenerated with `license: CC-BY-NC-4.0` on every entry (same 15 files,
  sha256 `ea2a1145…`); `README.md` and `models/hyacinth/MODEL_CARD.md` updated. No data file changed.
- **2026-11-13**: pruned to minimal set; `multiome_20` replaces `allen_multiome_300`. Removed: the 300-cell example, the
  `fixtures/` group (`rna_100.h5ad` is in the repository; `genes_mini.*` superseded by the full gene table), per-array
  `.npy` references and raw `evaluate.py` outputs, legacy `orthologs/` tables, panels `tf_533`, `hvg_8054`,
  `merfish_658801`, `enhancer_2298` (kept in the repository under `tools/panels/`). Added `examples/multiome_20/`
  (`examples-v2`). `MANIFEST.json` now 15 files, sha256 `ffd460d2…`.
- **2026-11-13** — initial publish: `hyacinth` weights + config, `mm10-evo2-v1` tables (genes, peaks,
  orthologs, panels), `examples-v1` (300-cell example, 100-cell fixture, reference embeddings),
  `MANIFEST.json` (47 files, sha256 `f6de4028…`), this README and `models/hyacinth/MODEL_CARD.md`.
