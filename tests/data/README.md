# tests/data — committed fixtures

Small files only (sha256 in `CHECKSUMS.sha256`; `sha256sum -c CHECKSUMS.sha256` from this directory). Everything else
(gene / peak tables, the 20-cell multiome fixture) comes from the public bucket (docs/assets.md).

| file | bytes | what |
|---|---:|---|
| `rna_100.h5ad` | 2,827,994 | 100 cells = 50 AIT33 subclasses x 2 (40 classes) of `hyacinth-v1/eval_balanced_50/allen_multiome`, seed 0; `X` log1p(CPM) float32 CSR over 31,899 genes (column-sorted), `layers["counts"]` int32 (row sums == `rna_umi_counts`); `obs`: `AIT33_class`, `AIT33_subclass`, `source_row`; `obs_names` = source sample ids. Cut by `tools/make_fixtures.py`. |
| `rna_100_cells.tsv` | 6,969 | `obs_name, source_row, AIT33_subclass, AIT33_class` in `obs` order |
| `ref/rna_100_reference_embeddings.npz` | 1,149,171 | arrays `rna_top4096` (paper-faithful: TF32 "medium", fused CUDA fast path), `rna_top4096_tf32off` (TF32 off, fast path off — the G1/G2a reference), `rna_random_8192x4` (paper default strategy; statistical parity only), `obs_names`; float32 `(100, 1024)`, rows in `obs` order. Produced by the original `evaluate.py` (`hyacinth`, L4, batch 8) from column-sorted inputs; folded from the former per-array `.npy` files by `tools/make_reference_npz.py rna-100` (byte-identical arrays, legacy sha256 recorded in the sidecar). |
| `ref/rna_100_reference_embeddings.json` | 20,242 | one sidecar: `obs_names`, `source_rows`, and per array the strategy, seed formula, `bloom-arch` commit, TF32 / fast-path flags, timing and parity vs the paper's run |
| `hyacinth_config.json` | 3,923 | copy of the released `models/hyacinth/config.json` (sha256 `6e8b8399…`) for the unit tests that build the model without weights |

Load a reference in a test with `conftest.load_reference(RNA_100_REFS, "rna_top4096_tf32off")`. Tolerances: docs/test-plan.md §5;
measured values: docs/validation.md §1. Regeneration: docs/maintainers.md §3.
