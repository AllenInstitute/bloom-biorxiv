# Maintainers

Everything a developer needs is on GitHub and the public bucket ([development.md](development.md)). This page covers the
maintainer tasks that need more than that: regenerating assets or reference embeddings, publishing to the bucket, running the
GPU gates, cutting a release. Issue tracking and the roadmap: [ROADMAP.md](../ROADMAP.md), GitHub milestones `v0.2.0` / `later`.

## 1. Maintainer environment (internal)

The maintainer tasks run on **a Linux host with one ≥ 24 GB CUDA GPU**, the **original training checkpoint** and training
artefacts, and **write access to the bucket** (`s3://bloom-biorxiv`, us-west-2, anonymous read; written through an IAM role
with `aws` CLI). No hostnames or mount paths appear in this repository; the tools under `tools/` take their locations from
environment variables (`tools/_paths.py` documents each one and exits with its purpose when unset):

| variable | points at | needed by |
|---|---|---|
| `BLOOM_STAGING` | staging tree = bucket content (`models/`, `references/`, `examples/`, `MANIFEST.json`); `archive/` holds unpublished material and is skipped | `make_manifest.py`, `publish_assets.py`, `make_multiome_20.py`, `make_fixtures.py`, `make_reference_*.py`, `bench_embed.py` (as `BLOOM_HOME`) |
| `BLOOM_TRAINING_CKPT`, `BLOOM_TRAINING_CONFIG` | the released run's Lightning `last.ckpt` and its Hydra `config.yaml` | `export_checkpoint.py`, reference regeneration |
| `BLOOM_EVO2_CACHE` | Evo2 feature-embedding cache (`evo2_{genes,peaks}_mean_ctx_2048_pca_768_norm`) | `export_checkpoint.py`, `export_reference.py` |
| `BLOOM_TRAINING_DATA` | training-data root (`embeddings/`, `strategies/eval/`, `feature-maps/`, `datasets/hyacinth-v1/`) | `export_reference.py`, `make_fixtures.py`, Hydra presets |
| `BLOOM_PAPER_RESULTS` | the paper's `evaluate.py` results (`<task>/embeddings.h5ad`) | `make_reference_npz.py`, `compare_tf32.py`, `regenerate_top4096_refs.py` |
| `BLOOM_EVAL_INPUTS`, `BLOOM_EVAL_INPUTS_SORTED` | `evaluate.py` inputs per fixture, raw and column-sorted | `sort_evalinputs.py`, `make_multiome_20.py`, Hydra presets |
| `BLOOM_ARCH_WORKTREE` | `bloom-arch` checkout at the pinned commit (seq-head scale 1/sqrt(32)); put `<dir>/src` **first** on `PYTHONPATH` for `evaluate.py` | reference regeneration |
| `BLOOM_TRAINING_REPO` | checkout of `alleninstitute/bloom` whose unchanged `evaluate.py` produces the references (own env with Lightning + Hydra) | reference regeneration |
| `BLOOM_LEGACY_ORTHOLOGS` | legacy ortholog tables (`orthologs_one2one.tsv`, `biomart_raw_response.tsv`, `*_mouseuniverse.tsv`) | `make_orthologs.py`, `export_reference.py` |

Conventions that hold on any such host: one worktree **and** one conda env per task (`git worktree add ../wt-<task> -b
<type>/<task> origin/main`; `scripts/setup_env.sh bloom-biorxiv-<task> cu128`, ~2 min); env `bin` is not on `PATH`, call
`<prefix>/bin/{python,bloom,pytest,ruff,pre-commit}`; `setup_env.sh` installs the pre-commit hook into the clone's shared
`.git/hooks` — a second worktree's install leaves `pre-commit.legacy`, fix with `<prefix>/bin/pre-commit install -f --hook-type
pre-commit`. Shortcut for the `assets` tier: `export BLOOM_HOME=$BLOOM_STAGING` (the staging tree is already in the release
layout) or `bloom download --source file://$BLOOM_STAGING` (symlinks). Never install the `bloom-arch` development branch
(`mask-dev`, scale 1/sqrt(768)) into a `bloom-biorxiv` env; `load_pretrained` asserts the scale. The training checkpoint and
training data are not redistributed; only the exported weights and tables in the bucket are public.

`gh` notes: `gh issue view` / `gh pr edit` fail on gh 2.45 (GraphQL projectCards) — use `gh api repos/<o>/<r>/issues/<n>` /
`gh api -X PATCH .../pulls/<n>`; the bot account cannot approve its own PRs (comment, then `gh pr merge --squash --delete-branch`).

## 2. Running the GPU gates

```bash
cd ../wt-<task>
bash scripts/setup_env.sh bloom-biorxiv-<task> cu128           # once per worktree; prints <prefix>
P=<prefix>
$P/bin/pre-commit install -f --hook-type pre-commit
BLOOM_HOME=$BLOOM_STAGING $P/bin/pytest -m assets -s           # ~12 min: G2a/G2b, G4, G5 (CPU, ~7 min), G7 RNA + ATAC, G8, E4, I8
```

The tests print one JSON line per gate (`[G1 ...]`, `[G2a ...]`); paste them into the PR's validation section. Measured
values live in [validation.md](validation.md). Since v0.1.0 `encode` runs with the fused CUDA fast path disabled (#44), so
no test toggles `torch.backends.mha`; CPU and GPU agree to ~1e-5.

## 3. Regenerating reference embeddings (`tools/`)

References are produced by the **unchanged** `alleninstitute/bloom` `evaluate.py` and consolidated into one `.npz` + one
`.json` per fixture. Regenerate only when the inputs or the numerics of the original pipeline change.

1. Eval inputs must be hyacinth-v1-format h5ads with the **full** column space and **column-sorted CSR rows**
   (`tools/sort_evalinputs.py` for the 100/300-cell sets; `tools/make_multiome_20.py` writes the 20-cell set directly).
2. Hydra preset under `tools/hydra/` (dataset `name: eval_balanced_50_allen_multiome` for the feature map, `MULTIOME`,
   batch 8, seed 42; paths resolve through `${oc.env:BLOOM_*}`); copy it to `<scratch>/experiment_evaluate/` and pass
   `--config-dir <scratch>`.
3. Run twice from the training repo's env through `tools/run_evaluate_checked.py` (asserts the seq-head scale on every load):
   ```bash
   export PYTHONPATH=$BLOOM_ARCH_WORKTREE/src
   RG=<training-env>/bin/python
   $RG tools/run_evaluate_checked.py --config-dir <scratch> experiment_evaluate=<preset> \
       evaluate.checkpoint_path=$BLOOM_TRAINING_CKPT evaluate.output_base_path=<runs>/tf32
   BLOOM_MATMUL_PRECISION=highest BLOOM_MHA_FASTPATH=0 $RG tools/run_evaluate_checked.py ... evaluate.output_base_path=<runs>/tf32off
   ```
   `tf32` = paper-faithful (TF32 "medium", fused CUDA fast path); `tf32off` = TF32 off + unfused (== CPU math, and == what
   `bloom` computes on any device since v0.1.0). `multiome_20` (3 tasks) takes ~100 s per run on an L4 (peak 9.7 GB);
   write `<runs>/timing.json` (`wall_s`, `gpu_peak_MiB`).
4. `tools/make_reference_npz.py multiome-20 --runs <runs> --fixture $BLOOM_STAGING/examples/multiome_20 --selection <selection.json>`
   writes `reference_embeddings.npz` (arrays `rna_top4096[_tf32off]`, `rna_random_8192x4[_tf32off]`, `atac_random_8192x16[_tf32off]`,
   `obs_names`) + `reference_embeddings.json` (strategy, seed, flags, commits, input sha256, wall, parity vs the paper's run by
   `source_row`). `tools/make_reference_npz.py rna-100` folds `.npy` references into `tests/data/ref/rna_100_reference_embeddings.{npz,json}`.
5. Update `tests/data/CHECKSUMS.sha256`, `tests/data/README.md`, then publish (§5).

## 4. How the 20-cell fixture was cut (`tools/make_multiome_20.py`)

From the 300-cell example (`$BLOOM_STAGING/archive/examples/allen_multiome_300`, rows of `hyacinth-v1/eval_balanced_50/allen_multiome`,
`source_row` recorded): the 10 AIT33 subclasses with the largest median number of nonzero ATAC peaks per cell, among
subclasses with at least two cells of ≥ 8,192 nonzero peaks; 2 cells per subclass drawn from those cells with
`default_rng(0)`; obs ordered by `source_row`. RNA: `X` log1p(CPM) + `layers["counts"]`, column-sorted; ATAC: raw `uint32`
counts restricted to the 483,249 peaks with ≥ 1 nonzero in these cells, `var["emb_idx"]` = reference row, peak-name index
stored as a compressed fixed-length string dataset (5.9 MB). The script also writes the matching `evaluate.py` inputs.

## 5. Publishing to the bucket

```bash
P=<prefix>; export BLOOM_STAGING=<staging tree>
$P/bin/python tools/make_manifest.py                   # MANIFEST.json for the staging tree (skips archive/); syncs src/bloom/data/manifest.json + MANIFEST_SHA256
$P/bin/python tools/publish_assets.py --prune --dry-run # plan: uploads + objects that would be deleted
$P/bin/python tools/publish_assets.py --prune --verify  # upload changed objects, delete stale ones, MANIFEST.json last, anonymous HEAD/sha check
```

`tools/bucket/README.md` and `tools/bucket/MODEL_CARD.md` are uploaded by `publish_assets.py` as `README.md` and
`models/hyacinth/MODEL_CARD.md` (not manifest entries). Also update `tests/test_assets.py` (file count, total bytes, group
counts), the bucket README changelog and `docs/assets.md`. Bucket objects carry `x-amz-meta-sha256`; `--prune` refuses any
bucket other than `bloom-biorxiv` and never touches `archive/` on the host. Licenses: every manifest entry says
`CC-BY-NC-4.0` (`tools/make_manifest.py`, decided 2026-11-13, #10).

## 6. Provenance of the tables (10 lines)

- Weights: `tools/export_checkpoint.py` keeps the `model.*` keys of the released run's `last.ckpt` (step 530000), remaps prefixes
  (`model.bloom_first_layer.` → `first_layer.`, …) and writes `config.json` with `seq_head_scale = 1/sqrt(32)` and the remap.
- Gene / peak tables: `tools/export_reference.py` from the Evo2 caches `evo2_{genes,peaks}_mean_ctx_2048_pca_768_norm`
  (`cseq` f32 `[N, 768]`, `chr` i8, `bp` i32) + the hyacinth-v1 gene/peak metadata → safetensors + parquet (+ `peaks.bed.gz`).
- Orthologs: `tools/make_orthologs.py` — Ensembl 115 BioMart one-to-one human→mouse pairs (confidence 1, duplicated symbols
  removed) restricted to mouse `gene_id`s in the 32,129-gene universe → 15,676 pairs; `#` header records the query and counts.
- Panels: `hvg_1999` (genes) and `da_2127` (peaks) are the published panels; `tf_533`, `hvg_8054`, `merfish_658801`,
  `enhancer_2298` live in `tools/panels/` (repo) and `archive/` (staging). All ids resolve against the parquet tables.
- Fixtures: `tools/make_fixtures.py` (rna_100, allen_multiome_300 — the latter now archived), `tools/make_multiome_20.py` (§4).
- The frozen sidecar JSONs (`tests/data/ref/*.json`, `tests/data/hyacinth_config.json` = `models/hyacinth/config.json`,
  `docs/history/`) keep the provenance fields they were written with (`export_host`, env, checkpoint path, run id): they are
  sha-pinned outputs and are not rewritten.

## 7. Release checklist (next: `v0.2.0`)

`v0.1.0` followed this list on 2026-11-13 (CHANGELOG.md). For the next tag:

1. **Gates on `main`** — CI green (CI downloads the weights and runs G1 since v0.1.0); on the GPU host
   `BLOOM_HOME=$BLOOM_STAGING <prefix>/bin/pytest -m assets -s` (G2a/G2b, G4, G5, G7, G8, E4, I8). Paste the `[G…]` JSON lines
   into the release notes; update validation.md if a value moved.
2. **Fresh-clone smoke** — the README quickstart in a new env and an empty `BLOOM_HOME` (`setup_env.sh` → `bloom info` →
   `bloom download --what all` → `bloom embed --rna`, `--atac`, both); `bloom embed --help` flags all documented in the README.
3. **Docs consistency** — validation.md numbers = the runs above; test-plan.md status column vs `pytest --collect-only -q`;
   README status line; AGENTS.md status table; CHANGELOG.md `Unreleased` → the new version; ROADMAP.md moved items.
4. **Issues** — every open issue carries `use-case:*` and a milestone (`v0.x.0` / `later`); close what the release resolves;
   the rest are "Known limitations" in the CHANGELOG entry.
5. **Assets** — `tools/make_manifest.py` (only if the staging tree changed) → `publish_assets.py --prune --dry-run` → `--prune --verify`;
   `tests/test_assets.py` counts; `tools/bucket/README.md` changelog entry naming the tag and the manifest sha256;
   `bloom info --verify` from a fresh download.
6. **Version** — `src/bloom/__init__.py` `__version__` (shown by `bloom info`, written to `uns["bloom"]["package_version"]`);
   PR `chore(release): vX.Y.Z preparation`.
7. **Tag + release** — `git tag -a vX.Y.Z` on the squash-merged `main`, `gh release create vX.Y.Z --generate-notes --notes-file …`
   (`.github/release.yml` groups PRs by `type:*` label). No archival DOI for the bioRxiv releases (user decision 2026-11-14): an
   archival release accompanies the eventual publication. Re-cutting a tag (done for `v0.1.0` on 2026-11-14): `gh release delete
   vX.Y.Z -y; git tag -d vX.Y.Z; git push origin :refs/tags/vX.Y.Z`, then tag + release again and say so in the release body.
8. **Migration** — done 2026-11-14: the repository was published at `alleninstitute/bloom-biorxiv` as a single commit (the
   pre-release development history was not carried over and its tracker was archived). The provenance sidecars
   (`tests/data/hyacinth_config.json`, `tests/data/ref/rna_100_reference_embeddings.json`,
   `examples/multiome_20/reference_embeddings.json`) name no host, environment or absolute path; keep it that way when
   regenerating them.
