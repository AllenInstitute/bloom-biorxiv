---
name: embedding-parity
description: Validate a change to the Bloom model, strategies, tokens or prepare code against the reference embeddings produced by the original alleninstitute/bloom evaluate.py (docs/test-plan.md section 5); run the regression and assets tiers and paste the evidence into the PR; regenerate the references when the inputs or numerics change.
---

# embedding-parity

## When to use

Any PR touching `src/bloom/model/`, `src/bloom/strategies/`, `src/bloom/tokens.py`, `src/bloom/prepare.py`,
`config.json`, or the `bloom-arch` pin.

## Contract: docs/test-plan.md section 5 (G1-G8); measured values in docs/validation.md

References: `tests/data/ref/rna_100_reference_embeddings.npz` (arrays `rna_top4096`, `rna_top4096_tf32off`,
`rna_random_8192x4`) and `examples/multiome_20/reference_embeddings.npz` on the bucket (+ `atac_random_8192x16`).

| ID | comparison | tolerance | test |
|---|---|---|---|
| G1 | `top` 4096, CPU fp32 vs `rna_top4096_tf32off` | `max |delta| < 1e-4` | `tests/test_model_parity.py::test_g1_*`, `tests/test_embed.py::test_g1_cli_*` (`regression`) |
| G2a | GPU fp32, TF32 off vs the same reference (`encode` is unfused on CUDA by default, #44) | `< 1e-4` | `test_g2a_*` (`assets`) |
| G2b | GPU TF32 "medium" vs the paper-faithful `rna_top4096` (fused kernel in `evaluate.py`) | sanity: cosine > 0.9999; max abs reported | `test_g2b_*` (`assets`) |
| G3 | GPU bf16 vs fp32 | cosine > 0.999 (documented, not gated) | `test_g7b_*` |
| G4 | `random` 8192 x 4 vs `rna_random_8192x4` | mean cosine > 0.999 | `test_g4_*` (`assets`) |
| G5 | ATAC `random` 8192 x 16 vs `atac_random_8192x16` (multiome_20) | as G4 | `tests/test_atac_parity.py` (scaffold, skips until #18) |
| G7 | CPU vs GPU `top` through the CLI default (`--device cuda`; unfused, `uns["bloom"]["fastpath"] == False`) | `< 1e-3` | `test_g7_cpu_vs_cuda_top4096` (`assets`) |
| G8 | batch size in {1, 4, 32}, `top` | `< 1e-5` | `test_g8_*` (`regression`) |

## Procedure

1. In the PR's own env, with the assets in `BLOOM_HOME` (`bloom download --what model,genes,examples`; maintainers may point
   `BLOOM_HOME` at the staging tree, `$BLOOM_STAGING`):
   `<env>/bin/pytest -m "regression or assets" tests/test_model_parity.py tests/test_embed.py tests/test_atac_parity.py -s`
   (about 5 min; G1 alone 45 s on CPU). CI downloads the weights and runs G1 on every PR (since v0.1.0, #45); the GPU gates
   still need this manual run — it is the evidence.
2. Paste into the PR "Validation" section: the tier run, the device, the observed `max |delta|` / cosine numbers per gate
   (the tests print `[G1 ...]`, `[G2a ...]`, ... JSON lines), and for a reference change the sha256 of every reference file.
3. Regenerating references (maintainer, only when the inputs or the numerics of the original pipeline change):
   docs/maintainers.md section 3 — column-sorted eval inputs, the unchanged `evaluate.py` via `tools/run_evaluate_checked.py`
   in two numerics flavours, `tools/make_reference_npz.py` to consolidate, `tools/make_manifest.py` + `tools/publish_assets.py`
   to publish; update `tests/test_assets.py` counts, `tests/data/CHECKSUMS.sha256`, `tests/data/README.md`.
