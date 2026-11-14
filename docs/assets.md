# Assets — the public bucket `s3://bloom-biorxiv`

Everything `bloom` needs besides the code is served from the public bucket **`s3://bloom-biorxiv`** (AWS us-west-2,
anonymous `s3:GetObject`, no credentials or AWS tooling). Base URL — always the **regional** hostname (the region-less one
answers HTTP 307):

```
https://bloom-biorxiv.s3.us-west-2.amazonaws.com/
```

## 1. Layout (15 manifest files, 6,076,570,255 B; `MANIFEST.json` sha256 `ffd460d2…`)

The bucket layout is the manifest's relative paths, so a local `BLOOM_HOME` is a verbatim mirror.

```
https://bloom-biorxiv.s3.us-west-2.amazonaws.com/
├── MANIFEST.json                         # path, size_bytes, sha256, group, version, license per file (+ base_url, versions)
├── README.md                             # human guide to the bucket (not a manifest entry; source tools/bucket/README.md)
├── models/hyacinth/
│   ├── model.safetensors                 # fp32 weights-only, 198 tensors, 124,144,831 params        496,624,516 B
│   ├── config.json                       # architecture + seq_head_scale (1/sqrt 32) + strategy defaults + reference_id + provenance
│   └── MODEL_CARD.md                     # model card (not a manifest entry; source tools/bucket/MODEL_CARD.md)
├── references/mm10-evo2-v1/
│   ├── genes.safetensors                 # cseq f32 [32129, 768], chr i8, bp i32                      98,861,805 B
│   ├── genes.parquet                     # gene_name, gene_id, chrom, start, end, strand (row = emb_idx) 1,733,063 B
│   ├── peaks.safetensors                 # cseq f32 [1747701, 768], chr i8, bp i32               5,377,676,873 B
│   ├── peaks.parquet                     # chrom, start, end, name "chr:start-end" (row = emb_idx)    72,448,807 B
│   ├── peaks.bed.gz                      # same peaks as gzip BED, for re-quantification             20,520,259 B
│   ├── orthologs_human_mouse_one2one.tsv # Ensembl 115 1:1 human->mouse, 15,676 pairs, gene_id join, emb_idx; '#' header = provenance
│   └── panels/{hvg_1999.txt, da_2127.txt} # 1,999 genes / 2,127 peaks, one id per line
└── examples/multiome_20/
    ├── rna.h5ad                          # 20 cells x 31,899 genes: X log1p(CPM) float32 CSR + layers["counts"] int32    1,398,379 B
    ├── atac.h5ad                         # 20 cells x 483,249 reference peaks (>= 1 nonzero): raw uint32 counts, var["emb_idx"]  5,912,106 B
    ├── cells.tsv                         # obs_name, source_row, AIT33_subclass, AIT33_class
    ├── reference_embeddings.npz          # rna_top4096, rna_random_8192x4, rna_tile_4096x2, atac_random_8192x16, atac_tile_8192x4 (each also *_tf32off), obs_names
    └── reference_embeddings.json         # per array: strategy, seed, TF32 / fast-path flags, commits, input sha256, wall, parity vs the paper
```

Every object carries `x-amz-meta-sha256` = the manifest digest and a sensible `Content-Type`.

| group (`--what`) | files | size | needed for |
|---|---|---:|---|
| `model` | `models/hyacinth/{model.safetensors, config.json}` | 497 MB | `bloom embed`, `load_model`, parity tests |
| `genes` | `genes.{safetensors,parquet}` | 101 MB | RNA mapping + embedding; the `regression` tests (auto-fetched) |
| `peaks` | `peaks.{safetensors,parquet}`, `peaks.bed.gz` | 5.47 GB | ATAC (Phase 2) only |
| `orthologs` | `orthologs_human_mouse_one2one.tsv` | 0.8 MB | `--species human` |
| `panels` | `panels/{hvg_1999,da_2127}.txt` | 64 KB | `--strategy panel`, `--universe` examples |
| `examples` | `examples/multiome_20/*` | 7.8 MB | the 20-cell RNA + ATAC fixture and its reference embeddings (`assets` tests, Phase 2) |

## 2. Fetching

```bash
bloom download --what model,genes           # RNA inference (0.6 GB)
bloom download --what examples              # 20-cell multiome fixture + references (8 MB)
bloom download                              # everything (6.1 GB; peaks.safetensors is 5.4 GB)
bloom info --verify                         # per-group present / partial / missing / corrupt; --verify re-hashes
```

`bloom download` (`src/bloom/assets.py`) is idempotent, verifies size + sha256 against the packaged manifest, resumes partial
downloads (`<name>.part` + HTTP `Range`; the retry budget counts only attempts that add no bytes) and renames atomically.
Cache: `~/.cache/bloom/` (override `BLOOM_HOME`), same layout as the bucket. Options: `--dir DIR`, `--force`, `--no-verify`,
`--strict` (exit 1 when a group is only partially available), `--source bucket | url:<base> | file://<root>[?copy]`
(or `$BLOOM_ASSET_SOURCE`; `file://` symlinks a local staging tree — the maintainers' shortcut, `$BLOOM_STAGING`).
Exit 1 when a requested group has no file in the source or a transfer fails verification.

Python: `bloom.assets.download(what, dest, force=, source=)`, `verify(what)`, `locate(kind)` (raises `AssetMissingError`
naming the `bloom download --what <group>` fix), `manifest()`, `default_source()`.

Without the package any HTTP client works, e.g. `curl -fsSLO https://bloom-biorxiv.s3.us-west-2.amazonaws.com/MANIFEST.json`
or `aws s3 cp --no-sign-request s3://bloom-biorxiv/<key> .`.

## 3. Manifest semantics

- `manifest_version` 1; the package ships a byte-identical copy (`src/bloom/data/manifest.json`) pinned by
  `bloom.assets.MANIFEST_SHA256`, so a given code version always knows which bytes it was tested against; a corrupt or
  foreign file never stays in `BLOOM_HOME`.
- Entry fields: `path` (= bucket key = path under `BLOOM_HOME`), `size_bytes`, `sha256`, `group` (`model | reference | example`),
  `version` (`hyacinth`, `mm10-evo2-v1`, `examples-v2`), `license`. Download groups (`--what`) refine `reference` by file name
  (`genes.*`, `peaks.*`, `orthologs*`, `panels/`).
- `model_id` (`hyacinth`) and `reference_id` (`mm10-evo2-v1`) are pinned together in `config.json`: a model never loads a
  reference it was not trained against. `versions["mm10-evo2-v1"]["orthologs"]` carries the ortholog table's `#` header.
- Committed in the repo (not in the bucket): `tests/data/rna_100.h5ad` (100-cell RNA fixture) and
  `tests/data/ref/rna_100_reference_embeddings.{npz,json}`; `tests/data/hyacinth_config.json` is a copy of `models/hyacinth/config.json`.
- Licenses (decided 2026-11-13, #10): everything in the bucket — weights, derived reference tables, example cells and the model
  card — is [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/); every manifest entry carries `"license": "CC-BY-NC-4.0"`.
  The code is MIT (`LICENSE`).

How the tables and fixtures were built, and how to regenerate / republish: [maintainers.md](maintainers.md).
