# Validation — current measured state

All numbers measured on the maintainers' GPU host (1x NVIDIA L4 24 GB, 32 vCPU / 16 torch threads, torch 2.11.0+cu128)
on 2026-11-13 unless stated. Test IDs: [test-plan.md](test-plan.md); how to re-run: [maintainers.md](maintainers.md) §2.

> **v0.1.0 numerics change (#44).** Since v0.1.0 `encode` always runs with torch's fused CUDA fast path disabled
> (`torch.backends.mha.set_fastpath_enabled(False)`, scoped inside `Bloom.encode`), so `bloom embed --device cuda` reproduces the
> CPU embeddings to ~1e-5 and `uns["bloom"]["fastpath"]` is `False`. Rows below marked *(fused, pre-v0.1.0)* were measured with
> the torch-default fused kernel and are kept for the record; §9 holds the v0.1.0 re-measurements. The fused kernel is a future
> optimisation that is evaluated outside this repository.

## 1. Parity gates (100-cell fixture `tests/data/rna_100.h5ad`)

References: `tests/data/ref/rna_100_reference_embeddings.npz` — arrays `rna_top4096` (paper-faithful: TF32 "medium", fused
CUDA fast path), `rna_top4096_tf32off` (TF32 off, fast path off = CPU math), `rna_random_8192x4`; produced by the original
`evaluate.py` from column-sorted inputs (sidecar `rna_100_reference_embeddings.json`).

| gate | comparison (`top` 4096 unless stated) | max abs diff | min cosine | tolerance | reproduce |
|---|---|---:|---:|---|---|
| G1 | CPU fp32 direct encode vs `rna_top4096_tf32off` | 1.34e-5 | 0.99999988 | < 1e-4 | `pytest -m regression tests/test_model_parity.py::test_g1_top4096_cpu_fp32_regression` |
| G1 (CLI) | `bloom embed --strategy top --device cpu` vs same | 1.34e-5 | 0.99999988 | < 1e-4 | `pytest -m regression tests/test_embed.py::test_g1_cli_top4096_cpu_strict` |
| G2a | L4 fp32, TF32 off (unfused) vs same | 1.91e-6 | 0.99999988 | < 1e-4 | `pytest -m assets tests/test_model_parity.py -k g2a` |
| G2b *(fused, pre-v0.1.0)* | L4 TF32 "medium", fast path on vs `rna_top4096` | 1.79e-4 | 0.99999988 | was < 5e-4 | — (since v0.1.0 G2b is a sanity check only, §9) |
| G4 | `random 8192x4` (L4) vs `rna_random_8192x4` | 5.89e-4 | 0.99999988 | mean cos > 0.999 (= 1.0) | `pytest -m assets tests/test_embed.py -k g4` |
| G7 | CPU vs L4 `embed()` (unfused = the `bloom embed --device cuda` default since v0.1.0) | 1.38e-5 | 0.99999976 | < 1e-3 | `pytest -m assets tests/test_embed.py -k g7` |
| G7 *(fused, pre-v0.1.0)* | CPU vs `bloom embed --device cuda` with the torch-default fused kernel | 4.65e-3 | 0.99999917 | was bound 5e-3 | no longer reachable through `bloom` |
| G8 | batch 4 / 32 vs 1 (CPU) | 1.43e-6 | 0.99999976 | < 1e-5 | `pytest -m regression tests/test_embed.py -k g8` |
| G3 (documented) | L4 bf16 vs fp32 | 6.83e-3 | 0.9999975 | not gated | `pytest -m assets tests/test_embed.py -k g7` (bf16 is measured inside `test_g7_cpu_vs_cuda_top4096`) |

The fused CUDA kernel (pre-v0.1.0 default) sat between the two reference flavours: 4.65e-3 vs `rna_top4096_tf32off` and 6.99e-4
vs `rna_top4096`. Paper-faithful references vs the paper's own `nz_top4096` run on cells **without** an expression tie at the 4096 cut: 2.30e-4
(the TF32 floor); on the 57 tie cells the paper's run holds storage-order token sets (its inputs were unsorted). Full numbers:
`docs/history/phase1_outputs/reference_embeddings/top4096_regeneration_report.json`.

## 2. 20-cell multiome fixture `examples/multiome_20` (RNA + ATAC; `bloom download --what examples`)

20 cells = 10 AIT33 subclasses x 2 (10 classes) from `allen_multiome_300`, every cell with ≥ 8,276 nonzero ATAC peaks (ATAC
views are full-length). References in `reference_embeddings.npz`, made by the original `evaluate.py` in two numerics flavours
(runs of 101 s / 104 s for the 3 tasks on the L4, peak 9,706 MiB; `reference_embeddings.json`):

| array | vs paper's full-dataset run (same cells by `source_row`) | vs the 300-cell reference, same cells | TF32 vs TF32-off/unfused |
|---|---|---|---:|
| `rna_top4096` | tie-free cells (3/20): max abs 1.16e-4, cos 1.0; all cells 0.57 (17 tie cells) | 6.18e-5 (batch-composition noise under TF32) | 3.53e-3 |
| `rna_top4096_tf32off` | tie-free 3.36e-3 (fast-path effect) | 1.4e-6 (deterministic) | — |
| `rna_random_8192x4` | 9.44e-5, min cos 0.9999998 (all cells < 8,192 genes: every draw is the full set) | 3.59e-5 | 3.32e-3 |
| `atac_random_8192x16` | mean cos 0.99967, min 0.99938 (different per-cell seeds: statistical only) | mean cos 0.99952 | 5.86e-3 |

Consistency checks of the fixture itself and the ATAC gates: `pytest -m assets tests/test_atac_parity.py` (§7).

## 3. Q1 — kNN AIT33-subclass accuracy on `allen_multiome_300` (RNA; historical measurement, 2026-11-13)

The 300-cell set (`archive/examples/allen_multiome_300` on the host; cells = `hyacinth-v1/eval_balanced_50/allen_multiome` rows
listed in `docs/history/phase0_outputs/fixtures/allen_multiome_300_cells.tsv`) is no longer published; the measurement stands.
`bloom embed` defaults (`random` 8192x4, seed 0, L4; 60.8 s for 1,200 views = 0.20 s/cell) vs the paper's embeddings of the same
cells (`eval_balanced_50_allen_multiome_nz_resamp_8192_d4`, by `source_row`). Leave-one-out kNN, cosine; 30 subclasses x 10 cells.

| embeddings | subclass k=5 | k=10 | **k=15 (nearest)** | k=15 (weighted) | class k=15 |
|---|---:|---:|---:|---:|---:|
| `bloom embed` defaults | 0.957 | 0.953 | **0.923** | 0.927 | 0.923 |
| paper run, same cells | 0.957 | 0.953 | **0.923** | 0.927 | 0.923 |
| `evaluate.py` on the 300-cell subset | 0.957 | 0.953 | **0.923** | 0.927 | 0.923 |
| `bloom embed --strategy top` | 0.953 | 0.947 | **0.933** | 0.933 | 0.930 |

Identical predictions at every k; mean per-cell cosine `bloom embed` vs paper 0.999998 (min 0.99981). Only 4/300 cells detect
more than 8,192 genes, so the `random` sets differ by numerics alone. `top` vs `random`: mean cosine 0.9787.

## 4. Use-case paths (README commands, fresh clone)

| path | device | wall | result |
|---|---|---:|---|
| `bloom download --what model,genes` (bucket) | – | 17.8 s for 0.76 GB on the host; your bandwidth elsewhere | sha256-verified |
| `bloom embed --rna tests/data/rna_100.h5ad --species mouse` (defaults) | CPU / L4 | 231 s / 22.9 s | CPU vs L4 4.58e-3, min cos 0.9999992 |
| `--strategy top` | CPU / L4 | 45.3 s / 7.2 s | G1 1.34e-5; CPU vs L4 4.65e-3 |
| `--strategy panel --panel hvg_1999.txt` | CPU / L4 | 12.5 s / 4.6 s | 1,999-gene views; CPU vs L4 6.60e-3 |
| `--species human` (upper-cased symbols), `top` | CPU / L4 | 42.2 s / 7.3 s | 14,602 / 31,899 ids via orthologs |
| `bloom prepare` → `bloom embed PREPARED_DIR` | CPU | 2.6 s + 45.6 s | bitwise identical to the one-step run |
| `peaks.safetensors` download (5.38 GB) | – | 98 s on the host; SIGKILL after 10 s then re-run resumed from the `.part` | sha verified |

Output schema (data-schema.md §4) checked on six outputs; error paths: missing assets → exit 1 naming `bloom download`;
`--species rat` / no `--species` → exit 1; negative values → exit 1; duplicate `obs_names` → exit 1;
`--strategy tile` → exit 1 (#31). Findings #33, #34 (guards) and #35 (internals API) were closed by PRs #41 and #40; `--atac` is
Phase 2 (§7). The Phase 2+3 review re-ran these paths from a fresh clone (2026-11-13): §8.

## 5. ML path

`bloom.load_model("hyacinth", device="auto")` → `model.reference.genes` → a 9-line `ViewStrategy` subclass → `bloom.embed(model,
ds, strategies={"rna": NoChrX(universe)})`: 6.3 s / 100 cells on the L4, provenance `strategy: no_chrx`, cosine vs the all-gene
view mean 0.9990 / min 0.9975. `collate([TokenSet.genes(...), ...])` → `model.encode(emb_idx, kind, lengths)` equals
`embed()`'s single full view at 1.4e-6; `backward()` runs outside `inference_mode`. How-to: [extending-strategies.md](extending-strategies.md).

Internals (I1–I8, `tests/test_internals.py`, measured 2026-11-13 in env `bloom-biorxiv-internals`, PR #40 branch):

| check | result |
|---|---|
| I1–I7 on the `hyacinth` architecture at random init (B = 2, L = 512 / 300 real tokens; CPU) | 15 unit tests pass in 5 s |
| I2 `h_cell` with vs without `return_internals` | bitwise equal under `inference_mode` (random init and released weights, CPU and L4) |
| I3 `pair_logits` vs explicitly indexed `q kᵀ · scale`, 100 pairs × 3 groups × 2 rows | < 1e-5 (allclose, atol 1e-5); `softmax(q kᵀ·scale)·W_V` reproduces each head module's output < 1e-5 |
| I4 `cross_attn` row sums | 1 ± 1e-5 at random init; released weights on 2 × 4096 tokens: min 0.99999988, max 1.00000012 |
| I8 released weights, 2 `rna_100` cells, `TopExpressed(4096)`, L4 | `token_ids` all in `genes.parquet.gene_name`; seq-head logits for Malat1/Snhg11/Meg3/Gad1 pairs finite (range −6.5 … +9.0); `h_cell.sum().backward()` gives non-zero `token_h.grad` on all 4,096 real tokens |
| parity after the change (`pytest -m "regression or assets" tests/test_model_parity.py tests/test_embed.py`) | 14 passed in 312 s: G1 1.34e-5, G2a 1.91e-6 (unchanged; values re-printed in the PR), G7 1.38e-5, G8 batch invariance, G4 mean cosine 1.0 |
| `tools/examples/attention_internals.py` on the L4 | 2 × 4096 tokens: `qk` shapes `(2,4096,8,32) / (2,4096,8,160) / (2,4096,4,32)`, seq scale 0.17677669529663687 |

Memory: `estimate_internals_bytes` = 13,312 B per token (`token_h` 4 KB + q/k 9.2 KB + head-averaged `cross_attn` 1 KB);
the default guard `max_tokens = 32,768` corresponds to 0.41 GiB of returned tensors. The guard does not cover the
first-layer attention matrix itself (see §6 for the CPU cost of `encode` at L = 4096 / 8192).

## 6. Hardware — measured cost of `bloom embed`

`tools/bench_embed.py`, fp32 unless stated, `hyacinth`, RNA fixture; "s / view" = wall of the embed loop per forward pass
(= per cell for `top`; x `--draws` for `random`); RSS = whole process incl. the 497 MB model and 99 MB gene table; VRAM =
`torch.cuda.max_memory_allocated`. Padded L in parentheses (no fixture cell reaches 8,192 mapped genes; max 7,478).

| strategy | L (padded) | batch | device | precision | s / view | peak RSS (GB) | peak VRAM (GB) |
|---|---|---|---|---|---|---|---|
| top | 4096 (4096) | 1 | CPU 16 thr | fp32 | 0.45 | 2.7 | – |
| top | 4096 (4096) | 8 | CPU 16 thr | fp32 | 0.48 | 11.2 | – |
| random | 8192 (6032) | 1 | CPU 16 thr | fp32 | 0.58 | 4.1 | – |
| random | 8192 (6032) | 4 | CPU 16 thr | fp32 | 0.78 | 11.8 | – |
| random | 8192 (7478) | 1 | CPU 16 thr | fp32 | 1.36 | 5.4 | – |
| random | 8192 (7478) | 4 | CPU 16 thr | fp32 | 1.36 | 13.2 | – |
| top | 4096 (4096) | 8 | L4 | fp32 | 0.055 | 1.8 | 1.9 |
| top | 4096 (4096) | 32 | L4 | fp32 | 0.053 | 1.8 | 3.9 |
| random | 8192 (6032) | 8 | L4 | fp32 | 0.075 | 1.8 | 2.5 |
| random | 8192 (6032) | 16 | L4 | fp32 | 0.074 | 1.8 | 4.5 |
| random | 8192 (6032) | 8 | L4 | bf16 | 0.048 | 1.8 | 1.7 |
| random | 8192 (7478) | 8 (4 views) | L4 | fp32 | 0.22* | 1.8 | 1.5 |
| random (defaults) | 8192 | 8 | L4 | fp32 | 0.051 (300 cells: 1,200 views in 60.8 s) | 1.9 | – |
| panel hvg_1999 | 1999 | 1 / 8 | CPU / L4 | fp32 | 0.097 / 0.013 | – | – |
| synthetic full-length | 8192 | 1 / 8 | CPU 16 thr | fp32 | 1.4 | 6.2 / 39.8 | – |

\* 4 views only: CUDA warm-up dominates; steady state at L ≈ 7.5k is ~0.1 s / view.

The L4 rows above were measured with the torch-default fused kernel (pre-v0.1.0). v0.1.0 runs unfused on CUDA; the re-measured
L4 costs are in §9 (RNA `top` 4096 0.037 s/cell, RNA `random` 8192×4 0.060 s/view, ATAC `top` 8192 0.097 s/cell).
CPU rows are unaffected (the CPU path was never fused).

- CPU: batch size buys nothing and costs memory (the first-layer attention materialises `B x H x L x L`); `embed()` defaults to
  `--batch-size 1` on CPU; results do not depend on batch size (G8).
- GPU: batch 8 is enough; ≈ 0.25 GB VRAM per extra view at L = 4096; model + gene table ≈ 1.5 GB; the 5.4 GB peak table (ATAC) comes on top.
- bf16 (`--precision bf16`) is ~1.5x faster on the L4 and halves activation memory; max abs 6.8e-3 vs fp32 (G3, documented).
- Laptop CPU: expect 2–4x the 16-thread numbers per view; memory is the binding constraint (`top` 4096 ≈ 3 GB; L ≈ 7.5k ≈ 5.5 GB at batch 1).

Reproduce: `BLOOM_HOME=... <env>/bin/python tools/bench_embed.py --cells 20 --config top,4096,1,cpu ...` (`--min-nnz 7000` for long views).

## 7. ATAC (Phase 2, issue #18) — measured 2026-11-13, env `bloom-biorxiv-atac`, feat/atac branch

User decision (2026-11-13): the ATAC gates run on the **CPU path**, plus **one** L4 smoke. Inputs: `examples/multiome_20/atac.h5ad`
(20 cells x 483,249 user peaks, counts uint32, every cell >= 8,276 nonzero peaks); `peaks.parquet` (1,747,701 reference peaks,
501 bp) + `peaks.safetensors` (5.4 GB).

### 7.1 Matching (`bloom.peaks`, `prepare --atac`)

| step | measured |
|---|---|
| `ReferencePeakIndex` build (full table, per-contig sorted arrays) | 1.4 s |
| R10: 100k random 501-bp user peaks vs the full table, `overlap` 0.5 / `exact` | 0.03 s / 0.02 s (32,517 pairs; gate < 10 s) |
| `prepare(atac=multiome_20)` end to end (parse 483k names, exact match, reference-peak matrix) | 4.3 s; `auto` -> `exact` (exact rate 1.0000); 483,249 / 483,249 peaks mapped; `var["bloom_emb_idx"]` equals the fixture's own `var["emb_idx"]` for every peak; median 36,504.5 detected reference peaks per cell |

### 7.2 Parity gates (`tests/test_atac_parity.py`, `pytest -m assets`)

| gate | comparison | result | tolerance |
|---|---|---:|---|
| G5 | `random` 8192 x 16, seed 0, **CPU fp32** (16 threads, batch 1) vs `atac_random_8192x16_tf32off` | mean cosine **0.99949**, min 0.99781, max abs 0.301, mean abs 0.0167 | mean cosine > 0.999 (statistical: `evaluate.py` per-cell seeds differ) |
| G5 (info) | same run vs the paper-faithful `atac_random_8192x16` | mean cosine 0.99949, min 0.99778 | — |
| G8 (peaks) | `top` by count, lmax 8192, CPU batch 4 vs batch 1 | max abs **1.43e-6**, min cosine 0.99999988 | < 1e-5 |
| G7 (peaks), gate | `top` 8192 on the L4 (unfused, TF32 off = the v0.1.0 default) vs CPU | max abs **9.78e-6**, mean abs 9.1e-7, min cosine 0.99999988 | < 1e-3 |
| G7 (peaks) *(fused, pre-v0.1.0)* | `top` 8192 on the L4 with the torch-default fused kernel vs CPU | max abs **9.24e-3**, mean abs 9.4e-4, min cosine 0.999997 | was bounded at 2e-2; the fused kernel exceeded 5e-3 at L = 8192 — the reason `encode` is unfused by default since v0.1.0 (#44) |
| E4 | `bloom embed --rna --atac` (`top` both, CPU) -> one file with `X_bloom_rna` + `X_bloom_atac`; `--atac` alone | pass: both keys, 20 cells, `alignment.n_common` 20; `--atac` alone bitwise identical; RNA half vs `rna_top4096_tf32off` < 1e-4 (gate; value not printed) | identical `X_bloom_atac`; RNA half vs `rna_top4096_tf32off` < 1e-4 |

For comparison, Phase 0 measured mean cosine 0.99973 (min 0.9982) between two `evaluate.py` runs of the same preset with
different per-cell seeds; the 16 draws of 20 cells here sit at the same level (0.99949 / 0.99781), i.e. the gap is sampling,
not numerics. The draw count was kept at 16 (the CPU run finished in 7.3 min, under the 10-min threshold at which the plan
allowed `--atac-draws 4`); `BLOOM_ATAC_G5_DRAWS` shortens the test when needed.

### 7.3 Cost

| run | device | s / view | s / cell | memory |
|---|---|---:|---:|---|
| `random` 8192 x 16 (padded L = 8192, batch 1) | CPU, 16 threads | 1.374 | 21.98 (439.5 s for 20 cells) | peak RSS 12.25 GB (process incl. model 0.5 GB, gene table 0.1 GB, peak table 5.4 GB read into RAM, L = 8192 attention) |
| `top` 8192 (1 view, batch 1) | CPU, 16 threads | 1.36 | 1.36 (27.2 s for 20 cells) | as above |
| `top` 8192 (batch 8) | L4 | ~0.1 | 0.12 (2.4 s incl. warm-up; 1.95 s embed loop) | peak VRAM 8.88 GB = model + gene table ≈ 1.5 GB + peak table 5.4 GB + activations; `load_model` 0.4 s, the 5.4 GB peak table is read and moved on the first `embed` (included in the 2.4 s wall) |

The peak table is loaded by `embed()` only when an `atac` modality is present (`ensure_peak_table`); `load_model` never
opens `peaks.safetensors` (M6). On CPU the safetensors tensors are read into RAM by `get_tensor` (5.4 GB), not kept as a
file-backed mmap — a documented deviation from the architecture note "memory-mapped on CPU" that would need a change in
`model/bloom.py` (out of scope for #18; the 124 GB host did not need it).

### 7.4 G6 — overlap-matched vs exact-matched (`tools/atac_overlap_study.py`)

Synthetic "other peak caller": the fixture's coordinates shifted by 50 / 150 / 250 bp (reference width 501 bp), matched with
`--peak-match overlap` at `--min-overlap` 0.25 / 0.5 / 0.75; `top` by count, lmax 8192, CPU (deterministic, so the cosine
measures the mapping change only). kNN accuracy is not meaningful on 20 cells and was not computed.

| shift (bp) | match | min_overlap | user peaks mapped | ref peaks hit | median detected / cell | mean cos vs exact | min cos | max abs |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | exact | - | 483,249 (100.0%) | 483,249 | 36504 | 1.00000 | 1.00000 | 0.00e+00 |
| 50 | exact | - | 0 (0.0%) | 0 | 0 | n/a | n/a | n/a |
| 50 | overlap | 0.25 | 483,249 (100.0%) | 483,249 | 36504 | 1.00000 | 1.00000 | 0.00e+00 |
| 50 | overlap | 0.5 | 483,249 (100.0%) | 483,249 | 36504 | 1.00000 | 1.00000 | 0.00e+00 |
| 50 | overlap | 0.75 | 483,249 (100.0%) | 483,249 | 36504 | 1.00000 | 1.00000 | 0.00e+00 |
| 150 | exact | - | 0 (0.0%) | 0 | 0 | n/a | n/a | n/a |
| 150 | overlap | 0.25 | 483,249 (100.0%) | 498,469 | 38306 | 0.96841 | 0.89802 | 1.86e+00 |
| 150 | overlap | 0.5 | 483,249 (100.0%) | 483,249 | 36504 | 1.00000 | 1.00000 | 0.00e+00 |
| 150 | overlap | 0.75 | 0 (0.0%) | 0 | 0 | n/a | n/a | n/a |
| 250 | exact | - | 0 (0.0%) | 0 | 0 | n/a | n/a | n/a |
| 250 | overlap | 0.25 | 483,249 (100.0%) | 541,961 | 43268 | 0.90104 | 0.76604 | 2.39e+00 |
| 250 | overlap | 0.5 | 483,249 (100.0%) | 483,249 | 36504 | 1.00000 | 1.00000 | 0.00e+00 |
| 250 | overlap | 0.75 | 0 (0.0%) | 0 | 0 | n/a | n/a | n/a |

Reading: the reference peaks do not overlap each other, so a shifted peak keeps covering its own reference peak by
`(501 - shift) / 501` (0.90 / 0.70 / 0.50 for 50 / 150 / 250 bp) and touches a neighbour by at most `shift / 501`. At
**`min_overlap = 0.5`** every shift up to half a peak width maps back onto exactly its originating reference peak and the
embedding is unchanged (cosine 1.0, 483,249 reference peaks hit in all three cases); at 0.25 the neighbours start to count
(+15k / +59k spurious reference peaks at 150 / 250 bp, mean cosine 0.968 / 0.901, min 0.898 / 0.766); at 0.75 anything
shifted by more than 125 bp maps nothing. **Decision: keep the default `--min-overlap 0.5`** (docs/input-data.md §7). The
study shifts coordinates only; peaks from another caller also differ in width — a wide user peak covers several
reference peaks fully (the fraction is relative to the 501-bp reference peak) and all of them are detected at any threshold,
which is the intended behaviour for broad peaks.

## 8. Phase 2+3 review — fresh clone, both user paths (2026-11-13, `main` @ `cf5989d`)

Fresh clone, `scripts/setup_env.sh … cu128` (85 s), `bloom download --what all` into an empty `BLOOM_HOME` (6.1 GB in 124 s on the
host), then the README commands on `examples/multiome_20` (20 cells; the 2-cell CPU timing is on the first two cells).

| path | device | wall | result |
|---|---|---:|---|
| `bloom embed --rna --atac --species mouse` (defaults: RNA `random` 8192×4, ATAC `random` 8192×16) | L4 | 43 s (RNA 80 views 5.0 s; ATAC 320 views 30.1 s = 0.094 s/view) | ATAC vs `atac_random_8192x16_tf32off` mean cos 0.99949, min 0.99783 (= G5 on CPU); RNA vs `rna_random_8192x4` mean cos 1.000000, max abs 4.5e-4 |
| same, 2 cells, `--device cpu` | CPU 16 thr | 57 s (ATAC 1.375 s/view = 22.0 s/cell; RNA 0.76 s/view) | → ≈ 8.4 min for the 20 cells at defaults (§7.3: 7.3 min ATAC + RNA); peak RSS 9.1 GB |
| `--rna --atac --atac-draws 4 --device cpu`, 20 cells | CPU | 178 s | peak RSS 12.1 GB (§7.3: 12.25) |
| `--atac` alone (`--atac-draws 4`, CPU) vs the two-file run | CPU | 119 s | `X_bloom_atac` bitwise identical |
| `bloom prepare --rna --atac` → `bloom embed PREPARED_DIR` vs one-step (`top` both, ATAC lmax 8192) | CPU | 7 s + 42 s vs 45 s | both `obsm` keys bitwise identical |
| `--rna-strategy top` RNA half vs `rna_top4096_tf32off` | CPU | – | max abs 1.17e-5 (G1 level) |
| CPU vs L4 *(fused, pre-v0.1.0)* | – | L4 `top` both 11 s | RNA `top` 4096 3.65e-3; ATAC `top` 8192 9.24e-3 (= §7.2, #44); ATAC `random` ×4 5.95e-3 — all ~1e-5 since v0.1.0 (§9) |
| `--keep-unmatched`, RNA file missing 3 cells (`top` both) | CPU | 43 s | `ku.h5ad` 17 cells with both keys + `ku_atac_only.h5ad` 3 cells; `alignment` `n_common 17, n_atac_only 3` |
| `--peak-match exact / overlap / auto` on the fixture | – | 6.7 s each | 483,249/483,249 mapped; `auto` → `exact` (rate 100 %) |
| same on a +50 bp-shifted copy | – | 6–7 s | `exact` → exit 1 (`NoMappedFeaturesError`); `overlap` and `auto` map 483,249/483,249, `auto` prints "peak match overlap (min_overlap 0.5); exact rate 0.0%"; the `top` embedding is bitwise identical to the exact-matched one (G6) |
| `var[chrom,start,end]` instead of names; 200 duplicated peaks; 10 % peaks on `chrUn_*` | – | 5–7 s | all accepted: 483,249 reference peaks hit; duplicates summed (no warning, #48); 48,325 dropped peaks reported with the contig name |
| error paths: `--species human` with `--atac`; only non-mm10 contigs; `--min-overlap 1.5`; `PREPARED_DIR` + `--atac`; disjoint `obs_names`; `--rna` without `--species` | – | – | all exit 1 with the fix in the message |
| ML path (docs only): `encode(return_internals=True)` on 2 RNA and 2 ATAC `top` views, `pair_logits` gene–gene / peak–peak, `capture("encoder.cross_attn")`, 12-line `NoChrX`, `h_cell.sum().backward()` on a peak view | L4 | model 0.7 s, peak table 0.5 s (page cache) | `token_ids` 4,096/4,096 in `genes.parquet` / `peaks.parquet`; seq logits gene–gene −5.4…+9.0, peak–peak −26…+6.4; `NoChrX` cos vs default 0.9994 / 0.9987 in 1.0 s; saliency finite and non-zero on all 4,096 peaks; 8.4 GB VRAM peak; `h_cell` bitwise = `encode()` under `inference_mode`, 6.1e-3 from a grad-enabled `encode()` *(fused, pre-v0.1.0: the grad-enabled call left the fast path; ~1e-6 since v0.1.0)* |

Output schema (data-schema.md §4) verified on six outputs: `X` empty `(n, 0)`, `obsm` keys per modality given, `obs` carries
`bloom_n_{genes,peaks}_detected`, `bloom_n_tokens_<mod>`, `bloom_n_views_<mod>`, `uns["bloom"]` with `alignment` and per-modality
`mapping` (`peak_match`, `min_overlap`, `exact_rate`, `n_ref_peaks_hit`). Findings filed: #45 (CI skips G1 without weights), #46 (mixed
gene+peak view accepted by `encode`), #47 (`--atac-*` override on an RNA-only bundle silently ignored), #48 (duplicate peak coordinates
summed without a warning; `--min-overlap 0` accepted).

## 9. v0.1.0 release measurements — unfused CUDA default (#44), 2026-11-13, `chore/release-v0.1.0`

`pytest -m assets -s` in the release env on the L4 (same host as above), after the change that disables the fused fast path inside
`encode`. GPU gates compare against the TF32-off references (G2a-style) and CPU (G7); the paper-faithful `rna_top4096` array (fused
kernel + TF32 in `evaluate.py`) is kept as a sanity reference only.

| gate | comparison | result | tolerance |
|---|---|---:|---|
| G2a | L4 fp32, TF32 off vs `rna_top4096_tf32off` | 1.91e-6 (min cosine 0.99999988) | < 1e-4 |
| G2b (sanity) | L4 TF32 "medium" (unfused) vs paper-faithful `rna_top4096` | max abs 4.65e-3, min cosine 0.9999991 (the fused-kernel gap, expected) | min cosine > 0.9999 (max abs reported only) |
| CPU vs GPU `encode` (32 cells) | direct encode, no toggles | 1.38e-5 | < 1e-4 |
| G7 | CPU vs `bloom embed --device cuda` (`top` 4096, CLI default) | 1.29e-5 (min cosine 0.99999976; GPU vs `rna_top4096_tf32off` 1.43e-6) | < 1e-3 |
| G7 (peaks) | CPU vs L4 `top` 8192 (CLI default) | 9.30e-6 (min cosine 0.99999988) | < 1e-3 |
| G4 | `random` 8192×4 on the L4 vs `rna_random_8192x4` | mean cosine 1.0000 (min 0.999999; max abs 4.8e-3 vs the fused-kernel reference) | mean cosine > 0.999 |
| G5 | ATAC `random` 8192×16 CPU vs `atac_random_8192x16_tf32off` | mean cosine 0.99949 (min 0.9978; 16 draws, 450 s = 1.41 s/view, 22.5 s/cell, peak RSS 12.2 GB) | mean cosine > 0.999 |
| G3 (documented) | L4 bf16 vs fp32 | 6.83e-3 (min cosine 0.9999975) | not gated |
| provenance | `uns["bloom"]["fastpath"]` | `False` on CPU and CUDA | — |

Cost on the L4 with the unfused path (in-process `embed()` on the 20 `multiome_20` cells after one warm-up pass, batch 8, fp32,
model + peak table resident; `s / cell` = loop wall / 20). The earlier §6 table used the `rna_100` fixture and the fused kernel,
so the two are not directly comparable; the fused kernel is a future optimisation evaluated outside this repository.

| strategy | L | device | s / view | s / cell | peak VRAM (GB) |
|---|---|---|---:|---:|---:|
| RNA `top` 4096 | 4096 | L4 | 0.037 | 0.037 | 7.4 (incl. the 5.4 GB peak table loaded by the bench; ≈ 2.0 without it) |
| RNA `random` 8192×4 (defaults) | ≤ 8192 | L4 | 0.060 | 0.24 | 8.4 (as above) |
| ATAC `top` 8192 | 8192 | L4 | 0.097 | 0.097 | 8.9 |
| ATAC `random` 8192×16 (defaults) | 8192 | L4 | 0.098 | 1.56 | 8.9 |

`scripts/check.sh` (CPU, weights present): 191 passed, 17 deselected in 278 s (peak RSS 42.4 GB, G8 batch 32 on CPU). `pytest -m assets -s`: 17 passed, 191 deselected in 707 s (peak RSS 27.0 GB; G5 CPU 450 s of it).

## 10. `tile` preset — parity with `evaluate.py` (issue #31), 2026-11-14

References `rna_tile_4096x2[_tf32off]` and `atac_tile_8192x4[_tf32off]` (added to `examples/multiome_20/reference_embeddings.npz`,
`tools/hydra/multiome_20_tile_reference_embeddings.yaml`, `tools/make_reference_npz.py multiome-20-add`) come from the original
`evaluate.py` with `StrategyTiledPool(draw_mode="tile")`: RNA `Lmax 4096, max_draws 2` (17/20 cells detect > 4,096 genes and get two
disjoint draws, 3 get one), ATAC `Lmax 8192, max_draws 4` (2–4 draws; cells detect 8,276–36,504 peaks). Runs of 59 s (TF32) / 52 s
(TF32 off + unfused) on the L4, peak 9,698 MiB; TF32 vs TF32-off max abs 3.3e-3 / 5.2e-3, cos ≥ 0.9999993.

| check | comparison | result | gate |
|---|---|---|---|
| G9a RNA | `Tile` with `evaluate.py`'s per-cell permutation reproduced (`default_rng(42 + row_idx)` over the detected genes in column order), CPU, vs `rna_tile_4096x2_tf32off` | max abs **1.05e-5** (L4: 9.54e-7, min cosine 1.0) | < 1e-4 |
| G9a ATAC | same for peaks vs `atac_tile_8192x4_tf32off` | max abs **6.68e-6** (L4: 1.43e-6) | < 1e-4 |
| G9b RNA | `--strategy tile --lmax 4096 --draws 2`, seed 0, CPU vs `rna_tile_4096x2_tf32off` | mean cosine **0.99771** (min 0.97962, max abs 0.73) | mean cosine > 0.995 |
| G9b ATAC | `--strategy tile --lmax 8192 --draws 4`, seed 0, CPU vs `atac_tile_8192x4_tf32off` | mean cosine **0.99853** (min 0.99183, max abs 0.47) | mean cosine > 0.995 |
| view counts | `bloom_n_views_*` | RNA 37 views / 20 cells, ATAC 72 views / 20 cells = `min(draws, ceil(n_detected / lmax))` for every cell, identical to the reference runs | — |

Why G9b is gated at 0.995 and not at the `random` level of 0.999: with 2 (RNA) or ≤ 4 (ATAC) *disjoint* random draws the mean over
draws depends on which features fall into the same view, so two seeds of the **same** strategy differ by as much as the parity
with `evaluate.py` does — seed-to-seed self-consistency of `Tile` on these 20 cells (L4, seeds 0/1/2): mean cosine 0.9976–0.9984
(min 0.979) for RNA, 0.9983–0.9987 (min 0.985) for ATAC; parity vs the reference for seeds 0/1/2: 0.9977 / 0.9978 / 0.9980 (RNA),
0.9985 / 0.9987 / 0.9982 (ATAC). `random` has no such floor: its RNA draws at `lmax = 8192` are the full detected set (identical
draws) and its ATAC default averages 16 draws. G9a shows the remaining difference is sampling only: with the same partition the
embeddings agree to ~1e-6. Cost on CPU (16 threads, batch 1): RNA 4096 views 0.35 s / view, ATAC 8192 views 1.35 s / view (G9b run
110 s for 109 views). For context, `tile` vs the `random` references of the same cells: mean cosine 0.9967 (RNA) / 0.9971 (ATAC) —
the two presets are different estimators of the cell embedding, not interchangeable.
