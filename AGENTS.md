# AGENTS.md — bloom-biorxiv (alleninstitute/bloom-biorxiv)

Single guide for humans and coding agents; `CLAUDE.md` imports this file. If something here conflicts with `docs/`,
`docs/standards.md` and `docs/architecture.md` win and this file needs a fix. Maintainer procedures (regenerating
references, publishing, releasing) are in [docs/maintainers.md](docs/maintainers.md); what comes next is in
[ROADMAP.md](ROADMAP.md) and what changed in [CHANGELOG.md](CHANGELOG.md).

## Purpose

Public, inference-only companion to the Bloom manuscript (multimodal single-cell transformer, Allen Institute): it turns
a user's AnnData (RNA and/or ATAC) into 1024-d cell embeddings with the released model `hyacinth` (124,144,831
parameters) and the `mm10-evo2-v1` reference tables, and exposes the model as a plain `torch.nn.Module`. Training stays
in `alleninstitute/bloom`.

## Module map

| file | responsibility |
|---|---|
| `src/bloom/assets.py` | `MANIFEST.json` (packaged copy + sha pin), `BLOOM_HOME`, `download` (verify, resume), `locate`, groups `model genes peaks orthologs panels examples` |
| `src/bloom/config.py` | `ModelConfig` / `AttentionConfig` dataclasses <-> `config.json` (incl. `seq_head_scale`) |
| `src/bloom/model/` | `Bloom` (`encode`, `return_internals`; runs with the fused CUDA fast path off, `fastpath_disabled`; rejects views mixing gene and peak tokens, `MixedModalityViewError` #46), `BloomFirstLayer` (wraps `bloom_arch`), Perceiver encoder, decoder (loaded, unused), `load_pretrained`; `internals.py`: `EncodeOutput`, `pair_logits`, `capture` (docs/internals.md) |
| `src/bloom/reference.py` | `ReferenceTables`: gene/peak parquet + safetensors (lazy), symbol / Ensembl / ortholog mapping, mapping report, `peak_index` |
| `src/bloom/peaks.py` | peak coordinate parsing (`parse_peaks`), `ReferencePeakIndex` (exact / overlap / auto matching, per-contig sorted arrays), `detection_matrix` (user peaks -> reference peaks, OR / sum) |
| `src/bloom/prepare.py` | AnnData(s) -> `PreparedDataset` (RNA: value source, mapping; ATAC: `prepare_atac`, reference-peak matrix, duplicate coordinates OR-ed with a warning #48; `obs_names` alignment, `keep_unmatched`; guards #33 `SpeciesMismatchError` / #34 `NoMappedFeaturesError`; report, bundle I/O) |
| `src/bloom/tokens.py` | `CellFeatures`, `TokenSet`, `Batch`, `collate` (pads `emb_idx = -1`, `kind = 0`) |
| `src/bloom/feature_lists.py` | gene / peak list files (`--universe`, `--panel`), id normalisation |
| `src/bloom/strategies/` | `ViewStrategy` ABC + presets `random` (`Resample`), `top` (`TopExpressed`), `panel` (`Panel`), `tile` (`Tile`: disjoint draws that partition the detected set; validated G9, #31) |
| `src/bloom/embed.py` | the predict loop: dataset -> strategy views -> batches -> `h_cell` mean -> `obsm["X_bloom_<mod>"]` + `uns["bloom"]`; lazy peak table (`ensure_peak_table`), `embed_unmatched` |
| `src/bloom/cli.py` | typer: `bloom info | download | prepare | embed` (`--rna`/`--atac`, `--peak-match`, `--min-overlap` in (0, 1], `--keep-unmatched`, `--force-species`, per-modality `--rna-*` / `--atac-*` overrides — warned when the modality is absent, #47) |
| `tools/` | maintainer scripts (export, fixtures, references, manifest, publish; locations via `BLOOM_*` env vars, `tools/_paths.py`); `tools/examples/` runnable ML-path scripts; not installed; see docs/maintainers.md |
| `tests/` | pytest tiers `unit | regression | assets` + marker `weights` (auto-added to tests using `hyacinth_model_dir`; CI runs them on `main` / nightly only, #51); `tests/data/` = `rna_100.h5ad` + `ref/rna_100_reference_embeddings.{npz,json}` + `hyacinth_config.json` |
| `docs/` | architecture (spec), data-schema, assets, input-data, strategies, extending-strategies, internals, test-plan (catalogue + status), validation (measured), development, maintainers, standards, history/ |

## Commands

```bash
scripts/setup_env.sh <env> [cpu|cu128]     # conda env + torch + pip install -e .[dev] + pre-commit hook; prints <prefix>
<prefix>/bin/bloom info                     # env bin is NOT on PATH: call <prefix>/bin/{python,bloom,pytest,ruff,pre-commit}
<prefix>/bin/bloom download --what genes[,model]   # from the public bucket into BLOOM_HOME (~/.cache/bloom)
scripts/check.sh [fast]                     # ruff check + ruff format --check + pytest -m "unit or regression"; `fast` = the PR CI job (`and not weights`)
<prefix>/bin/pre-commit run --all-files     # ruff + file hygiene
<prefix>/bin/pytest -m unit | regression | assets
```

| tier | assets | disk | device | time |
|---|---|---|---:|---|
| `unit` | none (synthetic tables, local HTTP server, `tests/data/`) | 0 | CPU | < 1 min |
| `regression` | `genes` (101 MB, auto-fetched by `tests/conftest.py`; `BLOOM_FETCH_FIXTURES=0` disables) + `model` (497 MB) for the parity gates. Weight-dependent tests carry the `weights` marker: CI runs `(unit or regression) and not weights` on every PR (~3 min) and `weights and not assets` (G1, G8, E1–E3, E8) on pushes to `main` + nightly (~20 min on 4 vCPU); locally they skip without the weights | 0.6 GB | CPU | < 5 min with weights on 16 threads |
| `assets` | `bloom download` everything (6.1 GB; `peaks` 5.5 GB only for ATAC) | 6.1 GB | CPU + CUDA for G2/G7 | ~12 min on an L4 |

## Contracts (do not break without a test + a docs change)

- Two index spaces, always suffixed: `col_idx` = column of the user's AnnData; `emb_idx` = row of `genes.parquet` /
  `peaks.parquet` (stored as `var["bloom_emb_idx"]`, `-1` unmapped). They coincide for `allen_multiome` ATAC only by accident.
- `key_padding_mask` `(B, lmax)` bool, **True = pad**; pads carry `emb_idx = -1`, `kind = 0`; `kind` 1 = gene, 2 = peak. One
  modality per view: a `TokenSet` or batch row mixing 1 and 2 raises (`MixedModalityViewError`, #46, test S11).
- Seeds: per cell `rng = np.random.default_rng([seed, cell_index])`; results never depend on batch size, worker count or device.
- Only `X > 0` matters (binarised input, metadata zeroed): library size, normalisation and batch do not enter the embedding.
- ATAC: user peaks are matched onto the 1,747,701 reference peaks (`exact` | `overlap` ≥ `--min-overlap` 0.5 of the 501-bp
  reference peak | `auto` = exact when ≥ 95 % match exactly); the prepared `atac.h5ad` holds the **reference-peak** matrix
  (hit peaks only; summed counts, detection = `X > 0`); peaks are mm10, so `--species human` is RNA-only. With both files
  cells are aligned by `obs_names` (intersection, RNA order); `--keep-unmatched` embeds the remainders separately.
- Output: `obsm["X_bloom_rna"]` / `obsm["X_bloom_atac"]` float32 `(n_cells, 1024)` per modality given, never a generic `X_bloom`;
  `uns["bloom"]` provenance (package version, git sha, model sha256, reference id, strategy + seed per modality, device, precision).
- Tolerances (docs/test-plan.md §5): `top` CPU fp32 and GPU TF32-off vs the `*_tf32off` reference `max|d| < 1e-4`; CPU vs GPU
  `< 1e-3` (G7; `encode` is unfused on every device, `uns["bloom"]["fastpath"] == False`); `random` statistical only (mean
  cosine > 0.999); the paper-faithful (fused-kernel + TF32) reference is a sanity check only (cosine > 0.9999).

## Gotchas

- `bloom-arch` is pinned to `AllenInstitute/bloom-sprint-attn@ef5fb8a` (seq-head scale 1/sqrt(32) = 0.17677669529663687, not in
  the checkpoint). Its development branch `mask-dev` uses 1/sqrt(768): never pip-install it into a bloom-biorxiv env;
  `load_pretrained` asserts the scale.
- Sort CSR indices before ranking: `TopExpressed` breaks ties by ascending reference row; `prepare` always `sort_indices()`.
- Torch's fused CUDA `nn.TransformerEncoder` fast path differs from CPU by up to 1e-2 at L = 8192, so `Bloom.encode` disables it
  for the duration of every call (`fastpath_disabled`, #44): CPU == GPU to ~1e-5, TF32 stays at the torch default (off). The
  fused kernel (~1.5× faster) is a future optimisation evaluated outside this repo; do not re-enable it in tests.
- CPU memory scales with batch size (`B x H x L x L` attention): `top` 4096 is 2.7 GB RSS at batch 1, 11 GB at batch 8; `embed()` defaults to batch 1 on CPU. Batch size never changes results (1.4e-6).
- The peak table (5.4 GB fp32) is loaded by `embed()` only when an `atac` modality is present (`ensure_peak_table`); on CPU it
  is read into RAM (safetensors `get_tensor`, ≈ 5.4 GB RSS on top), on CUDA moved once (5.4 GB VRAM). `load_model(peaks=True)`
  loads it eagerly. ATAC `random` 8192 × 16 on CPU is ≈ 1.4 s per view (≈ 7.5 min for 20 cells); use the GPU for real datasets.
- Env `bin` is not on `PATH` after `setup_env.sh`; `conda activate` fails in non-interactive shells.
- `setup_env.sh` installs the pre-commit hook into the clone's shared `.git/hooks`; a second worktree's install leaves
  `pre-commit.legacy` and commits fail in "migration mode" — fix with `<prefix>/bin/pre-commit install -f --hook-type pre-commit`.
- `torch>=2.6` `torch.load` defaults to `weights_only=True`; the Lightning `.ckpt` (maintainer `tools/` only) needs `False`.
- Ruff `RUF001` flags non-ASCII in Python strings: use `\u` escapes.
- Internals: never materialise first-layer attention (`B x 20 x L x L`); use `bloom.pair_logits(out, pairs, group)` on the returned Q/K.
  `return_internals=True` runs under `inference_mode` (bitwise `encode`); a grad-enabled call differs by ~1e-6 on any device. At
  random init LayerNorm gamma = 1 makes `h_cell.sum()` constant (zero gradient): use the released weights or a random projection.

## Status

`v0.1.0` (2026-11-13) is the first public inference release: Phases 1–3 merged and reviewed from a fresh clone
(validation.md §8–§9); licenses decided (#10: code MIT, assets CC BY-NC 4.0); `--device cuda` unfused by default (#44); CI
runs G1 (#45) in a separate `weights` job (#51); #31 (`tile` preset), #46 / #47 / #48 fixed. Open issues carry a milestone:
`v0.2.0` — #1 (mmap peak table); `later` — #2 (ATAC kNN sanity). Themes and the full list: [ROADMAP.md](ROADMAP.md).

| module | state | tests passing | remaining |
|---|---|---|---|
| assets.py + `download`/`info` | done | E5, E6, E7, manifest pin (`unit`); anonymous bucket download (`assets`) | — |
| config.py, model/ | done | M1–M3, M5, M7, S9, fast-path scoping (`unit`), G1 (`regression`), G2a, G2b, G7, G8 (`assets`) | — |
| reference.py | done | R1–R6 on synthetic + real gene table; `peak_index` | — |
| peaks.py | done | R7–R10 (`unit`; R10 + S6b peak part `assets`) | — |
| prepare.py (RNA + ATAC) | done | P1–P10 (P5 modality subsets / alignment, P10 duplicate peaks), #33 / #34 guards, token-set parity | — |
| tokens.py, strategies/ | done | S1–S8, S10, `top` parity with `evaluate.py`; `tile` parity G9a/G9b (`assets`) | — |
| embed.py + `bloom embed` | done | E1–E4, E8, G1 CLI, G4; ATAC: G5 (CPU), `top` batch invariance, one GPU smoke, E4 (`assets`) | CPU peak-table mmap (#1, v0.2.0) |
| ATAC | done | `tests/test_peaks.py`, ATAC parts of `test_prepare.py` / `test_embed.py`, `tests/test_atac_parity.py` (G5, G7/G8 for peaks, E4); G6 overlap study | ATAC kNN sanity (#2, later) |
| model/internals.py | done | I1–I7, S11 mixed-view guard (`unit`), I8 end-to-end (`assets`, GPU) | — |

## Picking up work

1. `git clone` → `scripts/setup_env.sh <env> cpu|cu128` → `<prefix>/bin/bloom download --what genes,model` → `scripts/check.sh`.
2. Read `docs/architecture.md` §3–§5 (interfaces) and the test IDs in `docs/test-plan.md`; take an open issue (`gh issue list
   --milestone v0.2.0`); `docs/development.md` §5–§6 explain how to add a strategy or a test gate.
3. Own worktree **and** env per task (`git worktree add ../wt-<task> -b <type>/<task>`), Conventional Commits, never commit to `main`.
4. Every behaviour change to `model/`, `strategies/`, `tokens.py`, `prepare.py` adds/updates a contract test with a tolerance.
5. `pre-commit run --all-files` + `scripts/check.sh` green → PR (template: problem / result / validation evidence; `type:*` label;
   CHANGELOG `Unreleased` line; `Closes #N`) → squash merge.
