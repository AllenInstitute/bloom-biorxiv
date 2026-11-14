> Repository: `github.com/alleninstitute/bloom-biorxiv`.

# bloom-biorxiv — test plan

Catalogue of contract tests with their current status (last column: `pass` / `skipped-on-CI` = passes on a machine with
the asset, skips on CI / `not implemented #N`). Tiers by marker: `unit` (CPU, no downloads), `regression` (CPU, committed
fixtures + the gene table, weights when present), `assets` (`bloom download`; GPU for the GPU gates); tests that need the
released weights also carry `weights` (added automatically by `tests/conftest.py`). CI runs `(unit or regression) and not
weights` on every PR and `weights and not assets` on pushes to `main` / nightly (#51), so "skipped-on-CI without weights"
below means "runs in the `weights` job". Measured values: [validation.md](validation.md).

## 1. Model construction and loading

| ID | Test | Pass criterion | Status |
|---|---|---|---|
| M1 | Build `Bloom` from `config.json`; count parameters | `sum(p.numel()) == 124,144,831` (measured at export 2026-11-12 from `hyacinth` `model.*` keys; encoder 113.63 M + first layer 10.24 M + decoder 0.16 M + feature_emb 0.12 M) | pass (unit) |
| M2 | `load_pretrained` state-dict coverage | zero missing keys, zero unexpected keys after remap | pass (unit; strict load `assets`) |
| M3 | Sequence-head scale assertion | `first_layer.attn_block.seq_heads._scale == 1/sqrt(32)`; a config with a wrong scale raises | pass (unit) |
| M4 | Checkpoint export round-trip (`tools/export_checkpoint.py`) | re-loading the safetensors into the *original* `Gardenia` via the inverse remap gives identical tensors (bitwise) | not implemented — maintainer tool, no test |
| M5 | Reference buffers are non-persistent | `state_dict()` contains no `*_cseq/*_chr/*_bp` | pass (unit) |
| M6 | Lazy peak table | a run with only `--rna` never touches `peaks.safetensors` (file open mocked); `embed()` loads the peak table once, only for an `atac` modality (`ensure_peak_table`) | pass (unit: `test_model.py::test_m6`, `test_embed.py::test_embed_loads_peak_table_lazily_and_only_for_atac`); the gene table is still loaded for ATAC-only runs (`load_model`, 99 MB) |
| M7 | Device moves | `model.to("cpu")`/`"cuda"` keeps buffers and parameters on the same device; `encode` works on both | pass (unit; the CUDA half skips on CI) |

## 2. Reference tables and feature mapping

| ID | Test | Pass criterion | Status |
|---|---|---|---|
| R1 | Gene symbol lookup | `Pvalb`, `Gad1` map to the rows whose `genes.parquet.gene_name` matches; unknown → −1 | pass (unit) |
| R2 | Case and whitespace | `pvalb`, ` Pvalb ` map after normalization, with a warning counting such fixes | pass (unit) |
| R3 | Duplicate symbols in user `var` | all duplicates map to the same `emb_idx`; detection is OR-ed; warning emitted | pass (unit) |
| R4 | Ensembl IDs | `ENSMUSG00000005716` maps via `gene_id`; a var full of Ensembl IDs does **not** trigger the low-mapping warning | pass (unit) |
| R5 | Human orthologs | `PVALB` → `Pvalb`; `var["bloom_mapped_via"] == "ortholog"`; human symbol absent from the one2one table → −1 | pass (unit; real table skipped-on-CI without genes) |
| R6 | Mapping report numbers | `n_mapped`, `detected_mass_mapped` equal values computed independently in the test | pass (unit) |
| R7 | Peak coordinate parsing | `chr1:100-600`, `chr1-100-600`, `chr1_100_600`, `1:100-600`, and `var[["chrom","start","end"]]` all parse to the same tuple; `chrUn_*`, `chr1_random` dropped and counted | pass (unit, `tests/test_peaks.py`) |
| R8 | Exact peak match | a var equal to a sample of reference peaks maps 100%; shifted by 1 bp maps 0% in `exact` mode | pass (unit; real table: 5,000 sampled names + the fixture's `var["emb_idx"]` for all 483,249 peaks, `assets`) |
| R9 | Overlap match | synthetic cases: user peak covering 2 reference peaks → both detected; coverage 0.49 vs 0.51 of a reference peak flips with `min_overlap=0.5`; many user peaks → one ref peak OR-ed | pass (unit) |
| R10 | Overlap match scale | 100k random user peaks against the full 1.75 M table in < 10 s (interval index, not O(n·m)) | pass — 0.03 s for 100k peaks (index build 1.4 s) (assets) |

## 3. Input validation (`prepare`)

Issue guards (2026-11-13): #33 — `--species human` with > 50 % of the ids matching mouse symbols case-sensitively (or `ENSMUSG`), or `--species mouse` with > 50 % HGNC-style all-upper-case ids (or `ENSG`) → `SpeciesMismatchError` (exit 1) naming the suggested species, `--force-species` overrides (unit: `test_issue33_*`). #34 — no cell with a detected mapped feature (RNA or ATAC) → `NoMappedFeaturesError` (exit 1) listing the most frequent unmapped ids; partial emptiness stays a warning (P7) (unit: `test_issue34_*`).

| ID | Test | Pass criterion | Status |
|---|---|---|---|
| P1 | Value-type independence | the same cells as integer counts, log1p-CPM and arbitrary positive floats with identical sparsity pattern give identical `PreparedDataset` detection matrices | pass (unit) |
| P2 | Negative values | scaled matrix raises `ValueError` with the message naming the fix | pass (unit) |
| P3 | Layer precedence | `--layer`, `layers["counts"]`, `raw.X`, `X` chosen in that order; recorded in `uns["bloom"]["value_source"]` | pass (unit) |
| P4 | Dense vs sparse vs backed | identical `PreparedDataset` from `X` dense, CSR, CSC, and `backed="r"` | pass (unit) |
| P5 | Modality subsets | `--rna` alone, `--atac` alone, and both all run; with both, output `obs` is the `obs_names` intersection and `--keep-unmatched` writes the per-modality remainders; disjoint `obs_names` with both files → error naming the first few ids of each | pass (unit: `test_prepare.py::test_p5_*`, `test_embed.py::test_embed_both_modalities_*`, CLI with the fake model) |
| P6 | Duplicate `obs_names` | error | pass (unit) |
| P7 | Empty cells | a cell with zero detected mapped features is kept; embedding row is NaN; warning lists the count | pass (unit) |
| P8 | Prepared bundle round-trip | `prepare → write → read` equals in-memory `prepare` (emb_idx, detection matrices, report) | pass (unit; RNA, ATAC and the `*_only.h5ad` members) |
| P9 | Schema version guard | bundle with `schema_version: "2.0"` refused by a 1.x loader | pass (unit) |

## 4. Strategies

| ID | Test | Pass criterion | Status |
|---|---|---|---|
| S1 | `TopExpressed` determinism | identical output across 3 calls, across batch sizes, and across worker counts; ties broken by ascending column index | pass (unit) |
| S2 | `TopExpressed` ranking invariance | raw counts vs the same cell CPM-normalized give the same token order | pass (unit) |
| S3 | `Resample` (`random`) sizes | every view has `L == min(lmax, n_detected)`; no duplicates within a view | pass (unit) |
| S4 | `Resample` seed semantics | same `(seed, cell_index)` → identical views regardless of batch composition; different seeds differ | pass (unit) |
| S5 | `Tile` partition (preset `tile` since v0.1.0, #31) | views are pairwise disjoint; union == detected set when `n_detected ≤ max_draws·lmax`; count == `min(max_draws, ceil(n/lmax))`; `build("tile", draws=k)` → `Tile(max_draws=k)`; CLI `--strategy tile` writes `strategy: tile` provenance | pass (unit; CLI with the fake model) |
| S6 | `Panel` | tokens are `detected ∩ panel` in panel order; features missing from the reference are dropped and reported once | pass (unit) |
| S6b | `universe` on `Resample`/`Tile`/`Top` | every emitted token is in `detected ∩ universe`; for peaks, a `universe` of the DA list on the example cells reproduces `StrategyTiledPool(idx_file=da pool)` behaviour (same token universe per cell) | pass (gene part unit; peak part with `panels/da_2127.txt` on `multiome_20`, `assets`: every token in detected ∩ DA, `L = min(8192, |detected ∩ DA|)`) |
| S6c | Feature-list parsing | gene and peak lists, comments, blank lines, mixed coordinate formats; unknown identifiers reported once with a count | pass (unit) |
| S7 | Modality routing | a strategy applied to the ATAC file never emits gene tokens and vice versa; `X_bloom_atac` is absent when only `--rna` is given (and vice versa) | pass (unit; both directions through `embed()` with the fake model) |
| S8 | Padding/collate | `lengths` correct; pad positions have `emb_idx == -1`, `kind == 0`; `key_padding_mask` True exactly at pads | pass (unit) |
| S9 | Padding invariance of the model | `encode` on a cell padded to 4096 vs 8192 differs by < 1e-5 (mask correctness) | pass (unit) |
| S10 | Custom strategy via ABC | a 15-line subclass in the test runs end-to-end through `embed()` | pass (unit) |
| S11 | One modality per view (#46) | `TokenSet` mixing `kind` 1 and 2 raises `ValueError`; a batch row mixing them raises `MixedModalityViewError` from `encode`, `encode(return_internals=True)` and `pair_logits` (message names the row); single-modality rows in the same batch are unaffected | pass (unit: `test_model.py::test_s11_*`, `test_internals.py::test_s11_*`, `test_tokens.py`) |

## 5. Regression against `alleninstitute/bloom` (the key tests)

References: `tests/data/ref/rna_100_reference_embeddings.npz` (100-cell fixture; arrays `rna_top4096` = paper-faithful TF32
"medium" + fused CUDA fast path, `rna_top4096_tf32off` = TF32 off + fast path off, `rna_random_8192x4`) and
`examples/multiome_20/reference_embeddings.npz` on the bucket (20 cells; the same three plus `atac_random_8192x16`, each in
both flavours). All produced by the original `evaluate.py` (`alleninstitute/bloom@141f0d3`, `hyacinth`, `32-true`, batch 8,
L4) from column-sorted inputs; `random` references use `evaluate.py`'s own per-cell seeds, so `random` is compared
statistically only. Row order of every array = the fixture's `obs` order (`cells.tsv`).

| ID | Test | Reference | Pass criterion | Status |
|---|---|---|---|---|
| G1 | `top`, `lmax=4096`, **CPU fp32** | `rna_top4096_tf32off` | `max |Δ| < 1e-4` | pass — 1.34e-5 (regression; skipped-on-CI without weights) |
| G2a | `top`, `lmax=4096`, **GPU fp32 with TF32 off** (`torch.backends.cuda.matmul.allow_tf32 = False`, `set_float32_matmul_precision("highest")`; `encode` is unfused by default since v0.1.0, #44) | `rna_top4096_tf32off` | `max |Δ| < 1e-4` | pass — 1.91e-6 (assets, GPU) |
| G2b | `top`, `lmax=4096`, GPU with TF32 on (`"medium"`, as the paper ran; unfused) | `rna_top4096` (paper-faithful: TF32 **and** the fused kernel) | **sanity only** since v0.1.0: per-cell cosine > 0.9999; `max |Δ|` reported (the fused-kernel gap, a few 1e-3, is expected) | pass (assets, GPU; validation.md §9) |
| G3 | `top` GPU bf16 | `rna_top4096` | per-cell cosine > 0.999; **documented in the hardware table, not a gate** | documented — 6.83e-3 (assets) |
| G4 | `random 8192×4` (RNA; class `Resample`) | `rna_random_8192x4` and the paper run `nz_resamp_8192_d4` | **statistical parity only**: mean per-cell cosine > 0.999 and kNN (k = 15) AIT33-subclass accuracy within ±1 pt of the reference. Phase 0 measured mean cosine 0.999998 (min 0.99973) between the Phase 0 draws and the paper's draws — different seeds, same model. | pass — mean cos 1.0 (assets) |
| G5 | `random peak 8192×16` (ATAC), exact-matched peaks (`assets`, **CPU** — user decision 2026-11-13: CPU path + one GPU smoke) | `atac_random_8192x16_tf32off` (multiome_20; also reported vs `atac_random_8192x16`) | mean per-cell cosine > 0.999 (statistical: different per-cell seeds). | pass — mean cos 0.99949, min 0.99781, max abs 0.30; 16 draws in 439.5 s on 16 threads (assets, CPU) |
| G6 | Overlap-matched vs exact-matched ATAC on the 20-cell fixture (`assets`, `tools/atac_overlap_study.py`) | exact-matched `top` 8192 run | mapping rate and cosine for coordinates shifted 50 / 150 / 250 bp at `min_overlap` 0.25 / 0.5 / 0.75; the `--min-overlap` default is chosen from this table (kNN accuracy is not meaningful on 20 cells and was not computed) | measured — validation.md §7; default 0.5 kept |
| G7 | Cross-device consistency | — | CPU vs GPU `top` embeddings through the CLI default (`bloom embed --device cuda`, TF32 off) `max |Δ| < 1e-3`; `uns["bloom"]["fastpath"]` is `False`. Since v0.1.0 `encode` always disables the fused CUDA fast path (#44: the fused kernel reached 9.2e-3 at L = 8192), so no test toggles it and there is no separate documented bound. ATAC: one GPU `top` 8192 run vs CPU `< 1e-3` | pass — 1.38e-5 RNA, 9.78e-6 ATAC (assets, GPU; v0.1.0 re-run in validation.md §9) |
| G8 | Batch-size invariance | — | `batch_size ∈ {1, 4, 32}` give identical `top` embeddings (< 1e-5 on CPU; on GPU with TF32 off); ATAC `top` 8192 batch 1 vs 4 on CPU | pass — 1.43e-6 (regression; skipped-on-CI without weights, the fake-model variant runs there); ATAC 1.43e-6 (assets, CPU) |
| G9a | `tile` (RNA `lmax 4096 × ≤ 2`, ATAC `lmax 8192 × ≤ 4`) with `evaluate.py`'s per-cell permutation reproduced (`tests/test_tile_parity.py::EvaluatePyTile`: `default_rng(42 + row_idx)` over the detected features in column order), **CPU** | `rna_tile_4096x2_tf32off`, `atac_tile_8192x4_tf32off` (multiome_20; original `evaluate.py`, `StrategyTiledPool(draw_mode="tile")`) | identical token sets ⇒ `max |Δ| < 1e-4`; view counts `min(draws, ceil(n_detected / lmax))` | pass — RNA 1.05e-5, ATAC 6.68e-6 on CPU (9.5e-7 / 1.4e-6 on the L4) (assets; validation.md §10) |
| G9b | `--strategy tile` with bloom's own seeds (seed 0) vs the same references | as above | **statistical parity only**: mean per-cell cosine > 0.995 per modality. Two / ≤ 4 disjoint random draws carry more sampling variance than `random` (RNA `random` 8192 is the full set, ATAC `random` averages 16 draws), so the `random` gate 0.999 does not apply: the strategy's seed-to-seed self-consistency is 0.9976–0.9984 (RNA) / 0.9983–0.9987 (ATAC) on this fixture and the parity sits at that level | pass — mean cos 0.9977 RNA (min 0.9796), 0.9985 ATAC (min 0.9918) (assets, CPU) |

Policy notes: `random` and `tile` are never bit-compared through the CLI (`evaluate.py`'s per-cell RNG stream is not ported;
G9a reproduces it inside the test only to prove the partition semantics); bit-parity holds for `top` and `panel`. The fused CUDA kernel is a future optimisation evaluated outside this repository
([ROADMAP.md](../ROADMAP.md)); G3 as a gate is also a roadmap item.

## 6. End-to-end (CLI)

| ID | Test | Pass criterion | Status |
|---|---|---|---|
| E1 | `bloom embed --rna tests/data/rna_100.h5ad --species mouse --strategy top -o out.h5ad` (CPU) | exits 0; output schema validates (DATA_SCHEMA §4); `obs` identical to input; `uns["bloom"]` has every provenance key | pass (regression; skipped-on-CI without weights) |
| E2 | `bloom prepare` then `bloom embed PREPARED_DIR` | identical embeddings to E1 | pass (regression; skipped-on-CI) |
| E3 | `bloom embed … --species human` on a human fixture (symbols upper-cased from the mouse fixture) | mapping rate > 80% of fixture genes; runs to completion | pass (regression; skipped-on-CI) |
| E4 | `bloom embed --rna … --atac …` on the 20-cell fixture (`assets`; `top` on both modalities, CPU) | one output with `X_bloom_rna` and `X_bloom_atac`; `bloom embed --atac …` alone gives `X_bloom_atac` identical to the combined run's (same seed) | pass (assets, CPU; identical bitwise; also with the fake model in `unit`) |
| E5 | `bloom download` with a tampered local file | sha mismatch detected, file re-fetched (HTTP mocked in `unit`; real in `assets`) | pass (unit; real bucket in assets) |
| E6 | `bloom info` | prints versions; exit 0 without any assets present | pass (unit) |
| E7 | Missing asset for the requested modality | clear error: "peak table not downloaded; run `bloom download --what peaks`" | pass (unit; `prepare --atac` without `peaks.parquet`, `embed` without `peaks.safetensors`) |
| E8 | `--keep-views` | `obsm["X_bloom_rna_views"].mean(1) == obsm["X_bloom_rna"]` (same for atac) | pass (regression; skipped-on-CI) |

## 7. Internals (ML track)

| ID | Test | Pass criterion | Status |
|---|---|---|---|
| I1 | `return_internals=True` shapes | `token_h (B,L,1024)`, `latents (B,256,1024)`, `qk[g]` q,k `(B,L,H_g,d)` with `H = (8,8,4)`, `d = (32,160,32)`, `cross_attn (B,256,L)` (`(B,8,256,L)` per head), `token_ids` length L with `None` at pads | pass (unit; hyacinth architecture at random init, B=2, L=512) |
| I2 | `h_cell` unchanged | `encode(x) == encode(x, return_internals=True).h_cell` bitwise under `inference_mode`; `< 1e-5` from a grad-enabled call (CPU and CUDA alike since v0.1.0: `encode` is always unfused, #44) | pass (unit; bitwise also on the released weights, `assets`) |
| I3 | `pair_logits` | equals the explicitly indexed `q kᵀ · scale` for 100 random `(i,j)` pairs per head group (< 1e-5); `scales["seq"] == 1/sqrt(32) == config.seq_head_scale`; `softmax(q kᵀ·scale, pad-masked) · W_V` reproduces each head module's output (< 1e-5) | pass (unit; 3 groups x 2 rows) |
| I4 | Cross-attention rows | each row of `cross_attn` sums to 1 (1e-5) and is 0 at pad positions, > 0 at real tokens; same per head | pass (unit) |
| I5 | `capture()` | hook count returns to baseline on exit and on exception; `encoder.final_norm` capture == `h_cell`; nested blocks; unknown name → `KeyError` | pass (unit) |
| I6 | Gradients | `requires_grad=True` (also inside a caller's `inference_mode`): `backward()` populates a finite, non-zero `token_h.grad (B,L,1024)`; `h_cell.sum()` on the released weights (at random init LayerNorm γ=1 makes `h_cell.sum()` constant, so the unit test uses a random projection) | pass (unit + assets) |
| I7 | Memory guard | `B·L > max_tokens` (default 32,768) raises `InternalsMemoryError` naming `max_tokens` before `resolve()` runs; `max_tokens=None` disables; `estimate_internals_bytes` = 13,312 B/token | pass (unit) |
| I8 | End-to-end on the released weights | `load_model` + `TopExpressed(4096)` on 2 cells of `rna_100`, `return_internals=True`: `token_ids` are `genes.parquet` symbols, seq-head pair logits of known gene pairs finite, `h_cell` bitwise `encode()` | pass (assets, L4) |

## 8. Sanity / scientific checks (`assets`, run before release, numbers go in the model card)

| ID | Check | Expectation | Status |
|---|---|---|---|
| Q1 | kNN AIT33 subclass accuracy on `allen_multiome_300` (archived set), RNA `random` (`hyacinth`) | within noise of the paper's `eval_balanced_50_allen_multiome` value for that preset | measured — 0.923 = paper (validation.md §3) |
| Q2 | Same, ATAC `random d16` | within noise of the paper's ATAC value | not implemented — needs the archived 300-cell set on the host (ATAC is in place; tracked in #42) |
| Q3 | Human: `eval_human_siletti_balanced_100` subset via orthologs | within noise of the paper's `_mouse_ortho` value | not implemented |
| Q4 | Timing table | cells/s and peak RSS/VRAM for `{top4096, random 8192×4}` × `{laptop CPU, L4 GPU}` × `{fp32, bf16}` | measured (validation.md §6) |
| Q5 | Attention ordering (notebook 03) | median seq-head logit for enhancer pairs > proximal cis > background on the example cells | not implemented — peak tokens exist since #18; `pair_logits` is the tool (`tools/examples/attention_internals.py`) |

## 9. Fixtures

Committed (`tests/data/`, sha256 in `CHECKSUMS.sha256`): `rna_100.h5ad` (100 cells = 50 AIT33 subclasses x 2, `X` log1p(CPM) +
`layers["counts"]`, `obs` `AIT33_class/subclass/source_row`), `rna_100_cells.tsv`, `ref/rna_100_reference_embeddings.{npz,json}`,
`hyacinth_config.json`. From the bucket into `BLOOM_HOME` (`tests/conftest.py` fetches them when missing): `references/mm10-evo2-v1/genes.*`
(`regression`), `examples/multiome_20/*` (20 cells, RNA + ATAC, references; `assets`), `peaks.parquet` + `panels/da_2127.txt`
(R10, S6b; `assets`). `peaks.safetensors` (5.4 GB) is never auto-fetched (`bloom download --what peaks`). R7–R9 use an
8-peak table written inline in `tests/test_peaks.py` (no `peaks_synthetic.bed` file was needed).
