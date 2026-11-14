# Changelog

All notable changes to `bloom-biorxiv` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [Semantic Versioning](https://semver.org/)
with the output schema (`docs/data-schema.md`) and the CLI as the public interface. Assets are versioned independently
through `MANIFEST.json` (`hyacinth`, `mm10-evo2-v1`, `examples-v2`). Planned work: [ROADMAP.md](ROADMAP.md).

## [Unreleased]

## [0.1.0] - 2026-11-14

First public inference release — the companion code of the Bloom preprint
(Gala, Convy, Marghi et al., *Bloom unifies brain cell representations across modalities and contexts*, bioRxiv 2026,
doi:[10.64898/2026.11.11.765881](https://doi.org/10.64898/2026.11.11.765881)).
Public release on alleninstitute/bloom-biorxiv (2026-11-14), as a single commit without the pre-release development
history; issue and pull-request numbers quoted in this entry refer to the pre-release development tracker, not to this
repository's issues. This bioRxiv release is an early-access courtesy; an archival release (with a DOI) will accompany
the publication.

### Added

- `bloom download` / `bloom info`: sha256-verified, resumable fetch of the released model `hyacinth` (497 MB) and the
  `mm10-evo2-v1` reference tables (genes 101 MB, peaks 5.4 GB, orthologs, panels) and the 20-cell multiome example from the
  public bucket `https://bloom-biorxiv.s3.us-west-2.amazonaws.com/` into `BLOOM_HOME` (default `~/.cache/bloom`).
- `bloom prepare` / `bloom embed --rna`: mouse RNA AnnData → 1024-d `obsm["X_bloom_rna"]` with the paper default (`random`,
  4 draws of ≤ 8192 detected genes), `top` (4096 highest counts) and `panel` strategies; human symbols / Ensembl ids through the
  shipped Ensembl 115 one-to-one ortholog table (`--species human`); species-mismatch and no-mapped-feature guards.
- `bloom embed --atac`: mm10 peaks matched onto the 1,747,701 reference peaks (`--peak-match exact | overlap | auto`,
  `--min-overlap`), `obsm["X_bloom_atac"]`; both files at once give one AnnData for the shared cells (`--keep-unmatched` for
  the remainders); prepared bundles (`bloom prepare … -o DIR && bloom embed DIR`) are bitwise identical to the one-step run.
- Provenance in `uns["bloom"]`: package version, git sha, model sha256, reference id, per-modality strategy / seed / mapping
  report, device, precision, `tf32_matmul`, `fastpath`, batch size, torch version.
- ML API: `bloom.load_model`, `model.encode(..., return_internals=True)` → `EncodeOutput` (token states, latents, per-group
  attention Q/K, cross-attention weights, token ids), `bloom.pair_logits`, `bloom.capture`, custom `ViewStrategy` subclasses
  (`docs/internals.md`, `docs/extending-strategies.md`, `tools/examples/attention_internals.py`).
- `--strategy tile` (#31): the fourth preset — one shuffle of the detected features cut into disjoint `lmax` chunks, so the
  draws partition the detected set (no feature in more than one draw; `ceil(n_detected / lmax)` views capped by `--draws`,
  default 4). Validated against the original pipeline's `StrategyTiledPool(draw_mode="tile")` on the 20-cell fixture: with
  `evaluate.py`'s per-cell permutation reproduced the embeddings agree to 1e-6 (G9a); with bloom's own seeds the statistical
  parity (mean cosine 0.9977 RNA / 0.9985 ATAC) equals the strategy's seed-to-seed variance (G9b gate 0.995;
  `docs/validation.md` §10). Reference arrays `rna_tile_4096x2`, `atac_tile_8192x4` (+ `*_tf32off`) were appended to
  `examples/multiome_20/reference_embeddings.npz` (manifest sha256 `38f0274496524a7da4e5015eedf7dc21f87f987243a40dc4a75395b253d99d2e`).
- Issue #48: duplicate peak coordinates in the ATAC `var` are OR-ed (largest value kept, as for duplicate gene symbols) with a
  counted warning; `--min-overlap` must be in (0, 1].
- Issue #47: `--rna-*` / `--atac-*` overrides for a modality that is not in the input now print a warning instead of being
  silently ignored.
- CI in two CPU tiers (#45, #51): every pull request runs lint + `(unit or regression) and not weights` with only the gene
  table downloaded (`scripts/check.sh fast`; 1 min 55 s per job on a 4-vCPU runner, pytest 47 s, vs 21 min 27 s before); the weight-dependent regression tests (G1,
  G8, E1–E3, E8; pytest marker `weights`, added automatically to tests that use the released weights) run on pushes to
  `main`, nightly and on demand after `bloom download --what model,genes,orthologs` (cached on the manifest sha;
  20 min 31 s per job). Regression tests pick their CPU batch size from the machine's RAM (`tests/conftest.py::cpu_batch_size`;
  1 on a 16 GB runner, 8 on the maintainers' host) and the G8 batch-32 case (~42 GB) skips below 48 GB.
- One modality per view (#46): a `TokenSet` mixing gene and peak tokens raises `ValueError`; a batch row mixing `kind` 1 and 2
  is refused by `Bloom.encode`, `encode(..., return_internals=True)` and `bloom.pair_logits` with
  `bloom.model.MixedModalityViewError` (a `ValueError` naming the row) — cross-modal attention logits were never validated
  and the model was trained on single-modality views (test S11, `docs/internals.md`).
- `CHANGELOG.md`, `ROADMAP.md`, `.github/PULL_REQUEST_TEMPLATE.md`, `.github/release.yml` (release notes grouped by `type:*`
  label), GitHub milestones `v0.2.0` / `later`; developer guides (`CONTRIBUTING.md`, `docs/development.md`) cover Linux / macOS
  setup, the CPU-only path, `BLOOM_HOME`, test tiers, how to add a strategy or a test gate.

### Changed

- **CUDA numerics (#44).** `Bloom.encode` now disables torch's fused `nn.MultiheadAttention` / `nn.TransformerEncoder` fast
  path for the duration of every call (`bloom.model.fastpath_disabled`), on every device. `bloom embed --device cuda`
  (and `auto`) therefore reproduces the CPU embeddings to ~1e-5 instead of differing by up to 9.2e-3 at L = 8192; the
  fused-vs-unfused comparison tests and the documented fused bounds were removed; `uns["bloom"]["fastpath"]` is `False`.
  The fused kernel (~1.5× faster on long views) is a future optimisation that will be evaluated outside this repository.
  Cost on an L4 with the unfused path: RNA `top` 4096 0.037 s/cell, RNA `random` 8192×4 0.060 s/view, ATAC
  `top` 8192 0.097 s/cell, ATAC `random` 8192×16 0.098 s/view (`docs/validation.md` §9).
- Licenses decided (#10): code **MIT** (`LICENSE`, Allen Institute 2026); model weights, reference tables, example data and the
  model card **CC BY-NC 4.0** (model card, bucket README, `license: CC-BY-NC-4.0` on every `MANIFEST.json` entry; the manifest
  was regenerated with no data change, sha256 `ea2a1145ad19147481b5a2eda61787dd42ca28586dc06ffe95f3bee55aa4827a`).
- Repository surface no longer references the maintainers' internal infrastructure; maintainer tools read their locations
  from `BLOOM_*` environment variables (`tools/_paths.py`, `docs/maintainers.md` §1).
- Version `0.0.1` → `0.1.0` (`bloom info`, `uns["bloom"]["package_version"]`).

### Validated

Measured on 1× NVIDIA L4 / 16 CPU threads, torch 2.11.0+cu128 (`docs/validation.md`, `docs/test-plan.md` §5):

| gate | result | tolerance |
|---|---|---|
| G1 `top` 4096 CPU fp32 vs the TF32-off reference (100 cells; also through the CLI) | max abs 1.34e-5 | < 1e-4 |
| G2a GPU fp32, TF32 off vs the same reference | 1.91e-6 (min cosine 0.99999988) | < 1e-4 |
| G7 CPU vs GPU (`bloom embed --device cuda`, `top` 4096) | 1.29e-5 (min cosine 0.99999976; GPU vs `rna_top4096_tf32off` 1.43e-6) | < 1e-3 |
| G7 (peaks) CPU vs GPU `top` 8192 | 9.30e-6 (min cosine 0.99999988) | < 1e-3 |
| G8 batch 1 / 4 / 32 (RNA), 1 / 4 (ATAC) | 1.43e-6 | < 1e-5 |
| G4 RNA `random` 8192×4 vs the paper pipeline | mean cosine 1.0000 (min 0.999999; max abs 4.8e-3 vs the fused-kernel reference) | mean cosine > 0.999 |
| G5 ATAC `random` 8192×16 vs the paper pipeline (CPU) | mean cosine 0.99949 (min 0.9978; 16 draws, 450 s = 1.41 s/view, 22.5 s/cell, peak RSS 12.2 GB) | mean cosine > 0.999 |
| G6 peaks shifted by ≤ 250 bp, `--min-overlap 0.5` | bitwise identical to exact matching | — |
| Q1 kNN AIT33-subclass accuracy, 300 cells, k = 15 | 0.923, identical to the paper's embeddings of the same cells | — |
| I2 `h_cell` with / without `return_internals` | bitwise | — |
| `scripts/check.sh` (CPU, weights) / `pytest -m assets -s` (L4) | 191 passed, 17 deselected in 278 s (peak RSS 42.4 GB, G8 batch 32 on CPU) / 17 passed, 191 deselected in 707 s (peak RSS 27.0 GB; G5 CPU 450 s of it) | — |

### Known limitations

- **ATAC validation scope.** ATAC embeddings are validated by parity with the paper pipeline (mean cosine 0.9995 on the 20-cell
  fixture) and a deterministic GPU/CPU check (9.30e-6, G7 peaks); a label-transfer (kNN) sanity check on ATAC embeddings has **not** been
  run ([#2](https://github.com/alleninstitute/bloom-biorxiv/issues/2), milestone `later`).
- **CPU memory for ATAC.** The 5.4 GB peak table is read into RAM on CPU (not memory-mapped): a default ATAC run needs ≈ 12 GB
  RSS; 16 draws of 20 cells take ≈ 7.3 min on 16 threads (1.37 s/view, 22 s/cell). Use a GPU (9 GB VRAM) for real datasets
  ([#1](https://github.com/alleninstitute/bloom-biorxiv/issues/1), milestone `v0.2.0`, deferred past this release).
- **Performance.** RNA `top` 4096: 0.45 s/cell on 16 CPU threads (2.7 GB RSS), 0.037 s/cell on an L4; RNA `random`
  defaults: 4 × 0.6–1.4 s/cell on CPU, 4 × 0.060 s on an L4. Batch size never changes results and on CPU only adds
  memory (`B × H × L × L` attention: `top` 4096 at batch 8 = 11 GB RSS); `--batch-size` defaults to 1 on CPU, 8 on CUDA.
- Mouse (mm10) reference only; human RNA goes through one-to-one orthologs (15,676 pairs), human ATAC is refused; MPS
  (Apple silicon) is untested against the gates; `bf16` differs from fp32 by up to 6.8e-3 (G3, documented).

[Unreleased]: https://github.com/alleninstitute/bloom-biorxiv/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/alleninstitute/bloom-biorxiv/releases/tag/v0.1.0
