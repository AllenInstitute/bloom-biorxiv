# Development on any machine

The repository is machine-independent: the code is on GitHub, the assets are on the public bucket
`s3://bloom-biorxiv` (`https://bloom-biorxiv.s3.us-west-2.amazonaws.com/`, anonymous read; [assets.md](assets.md) §2),
and every test either ships its data in `tests/data/`, builds synthetic tables, or fetches / skips cleanly.
Only *regenerating* assets needs the maintainers' host ([maintainers.md](maintainers.md)).

## 1. Setup

```bash
git clone https://github.com/alleninstitute/bloom-biorxiv.git && cd bloom-biorxiv
scripts/setup_env.sh bloom-dev cpu        # conda env + torch (cpu | cu128) + `pip install -e .[dev]` + pre-commit hook; ~2 min
export PATH=<prefix>/bin:$PATH            # the script prints <prefix>; or call <prefix>/bin/{python,bloom,pytest,ruff,pre-commit}
bloom info                                # versions, device, BLOOM_HOME, which asset groups are present
```

- **Linux** (x86-64): `cpu` or `cu128` (CUDA 12.8 wheels; any NVIDIA GPU with ≥ 2 GB VRAM runs RNA, ≥ 10 GB runs ATAC).
- **macOS** (Apple silicon or Intel): use `cpu`; the default torch wheel on macOS includes MPS, and `--device mps` /
  `--device auto` will use it, but MPS is **untested** against the parity gates — the documented and validated path is CPU
  (`--device cpu`). CPU numbers (validation.md §6) were measured with 16 threads; a laptop is 2–4× slower per view.
- **Windows**: not tested; WSL2 with the Linux instructions is the expected route.
- **Python ≥ 3.12**; `pyproject.toml` is the only dependency file. Without conda: create any Python 3.12 venv, install the
  torch wheel for your platform first (`pip install "torch==2.11.*" --index-url https://download.pytorch.org/whl/cpu`), then
  `pip install -e ".[dev]"` and `pre-commit install`. `bloom-arch` is pulled from GitHub at the sha pinned in `pyproject.toml`.
- One env per worktree / task is the convention (`git worktree add ../wt-<task> -b <type>/<task>` + `scripts/setup_env.sh bloom-<task> cpu`).

## 2. Assets and `BLOOM_HOME`

`bloom download` fills `BLOOM_HOME` (default `~/.cache/bloom`; set the variable to share one cache between
envs, worktrees or users) in the release layout, verifying size + sha256 against the packaged `MANIFEST.json`
and resuming interrupted transfers. Groups and sizes:

| group | files | size | needed for |
|---|---|---|---|
| `genes` | `references/mm10-evo2-v1/genes.{safetensors,parquet}` | 101 MB | `bloom embed` (RNA), `load_model`; the `regression` tests |
| `model` | `models/hyacinth/{model.safetensors, config.json}` | 497 MB | `bloom embed`, model / parity tests (G1) |
| `orthologs` | `orthologs_human_mouse_one2one.tsv` | 0.8 MB | `--species human`, ortholog tests (E3) |
| `panels` | `panels/{hvg_1999,da_2127}.txt` | 64 KB | `--strategy panel`, panel tests |
| `examples` | `examples/multiome_20/*` | 7.8 MB | 20-cell RNA + ATAC fixture with reference embeddings (`assets` tests) |
| `peaks` | `peaks.{safetensors,parquet}`, `peaks.bed.gz` | 5.47 GB | ATAC only (12 GB RAM on CPU, 9 GB VRAM on GPU) |

```bash
bloom download --what genes,model               # 0.6 GB: everything the regression tier needs (CI does exactly this)
bloom download --what orthologs,panels,examples # + 9 MB: the rest of the `assets` tier except peaks
bloom download --what peaks                     # + 5.47 GB: ATAC
bloom info --verify                             # re-hash everything present
```

Other sources (`--source`, or `BLOOM_ASSET_SOURCE`): `url:<base>` = another mirror of the layout, `file://<root>` = a
local directory in the release layout (symlinked). Offline runs: `BLOOM_FETCH_FIXTURES=0` stops `tests/conftest.py` from
attempting a download; the affected tests then skip with a message naming the `bloom download` command.

## 3. Test matrix

| tier | marker | what | assets | CPU | GPU | wall |
|---|---|---|---|---|---|---|
| lint | — | `ruff check .` + `ruff format --check .` (also pre-commit) | none | yes | — | 5 s |
| unit | `-m unit` | construction, parsing, strategies, CLI smoke, download / resume logic against a local HTTP server, manifest pin, internals I1–I7 (hyacinth architecture at random init), fast-path scoping (#44), duplicate-peak and override warnings (#47, #48) | none | yes | — | < 1 min |
| regression | `-m regression` | G1 CPU parity (`top` 4096 vs `rna_top4096_tf32off`, < 1e-4), G8 batch invariance, E1–E3 / E8 CLI runs, token-set parity and mapping on the real gene table | `genes` (fetched by `tests/conftest.py` when missing) + `model` for the tests that also carry the `weights` marker (auto-added for the `hyacinth_model_dir` fixture; they skip without the weights) | yes (16 threads ≈ 5 min) | optional | < 5 min |
| regression, no weights | `-m "(unit or regression) and not weights"` = `scripts/check.sh fast` | the above minus the weight-dependent tests — what CI runs on every pull request (#51) | `genes` | yes | — | ≈ 3 min on 4 vCPU |
| weights | `-m "weights and not assets"` | G1, G8 (batch 1/4), E1–E3, E8: five `top` 4096 runs of the 100-cell fixture — CI runs them on pushes to `main`, nightly (09:00 UTC) and `workflow_dispatch` | `genes` + `model` | yes | — | ≈ 20 min on 4 vCPU |
| assets | `-m assets` | anonymous bucket download + sha check, `load_model` on the real weights, G2a/G2b GPU parity, G4, G7 CPU-vs-GPU, 20-cell fixture consistency, ATAC gates G5 / G7-G8 (peaks) / E4 (`tests/test_atac_parity.py`), I8 internals on the released weights, timing | `bloom download` everything (6.1 GB; `peaks` only for the ATAC tests) | partially (GPU gates skip) | L4 ≈ 12 min | 10–15 min |

```bash
scripts/check.sh                       # lint + unit + regression, CPU (= both CI jobs together)
scripts/check.sh fast                  # = the CI `check` job on PRs: deselects the `weights` marker
pytest -m "weights and not assets"     # = the CI `weights` job (main / nightly); needs `bloom download --what model,genes`
pytest -m assets -s                    # after `bloom download`; GPU gates skip without CUDA; -s prints the [G...] JSON lines
pytest -m "regression or assets" tests/test_model_parity.py tests/test_embed.py   # the parity gates only
```

CI (`.github/workflows/ci.yml`) has two CPU jobs on ubuntu / Python 3.12 with the pinned CPU torch wheel, each with a
`BLOOM_HOME` under the runner workspace restored from `actions/cache` keyed on the packaged manifest's sha256 (a manifest
change triggers a fresh download):

| job | trigger | downloads | runs | wall (4 vCPU) |
|---|---|---|---|---|
| `check` | every pull request and push | `genes,orthologs` (102 MB) | `scripts/check.sh fast` = lint + `(unit or regression) and not weights` | 1 min 55 s job (pytest 47 s: 184 passed, 1 skipped, 25 deselected; PR #53) |
| `weights` | pushes to `main`, nightly `0 9 * * *` UTC, `workflow_dispatch` | `model,genes,orthologs` (0.6 GB) | `pytest -m "weights and not assets"` (G1, G8, E1–E3, E8) | 20 min 31 s job (pytest 19 min 39 s: 7 passed, 1 skipped — G8 batch 32; `workflow_dispatch` on PR #53's branch) |

Before #51 one job did both (21 min 27 s on PR #50). The weight-dependent tests size their CPU batch from the runner's RAM
(`cpu_batch_size`, 1 on 16 GB); only G8's batch-32 case (~42 GB RSS) and the CUDA variant of M7 skip there. A PR that touches
`model/`, `strategies/`, `tokens.py` or `embed.py` should still be run with the weights locally (`scripts/check.sh` after
`bloom download --what model,genes`) or by dispatching the `weights` job on the branch. Test IDs and tolerances:
[test-plan.md](test-plan.md); measured gate values: [validation.md](validation.md).

## 4. Before a PR

```bash
pre-commit run --all-files             # ruff + hygiene; the hook is installed by setup_env.sh
scripts/check.sh
gh pr create                           # the template asks for problem / result / validation evidence; `Closes #N`
```

Conventional Commits, squash merge, one PR per branch, `type:*` label, CHANGELOG `Unreleased` entry
([CONTRIBUTING.md](../CONTRIBUTING.md)).

## 5. How to add a view strategy

A strategy turns one cell's detected features into one or more token views (`docs/strategies.md`, API how-to with a
runnable 12-line example in [extending-strategies.md](extending-strategies.md)).

1. Subclass `bloom.strategies.ViewStrategy` in `src/bloom/strategies/<name>.py`: implement `views(cell_features, rng)`
   returning `list[TokenSet]`, `params()` (what goes into `uns["bloom"]["modalities"][mod]["strategy"]`) and set `name`.
   Use `cell_rng` / the `rng` argument for any randomness so results depend only on `(seed, cell_index)`.
2. Register it: `PRESETS` / `ALIASES` in `src/bloom/strategies/__init__.py` and the factory in `strategies/factory.py`
   (CLI `--strategy <name>`, per-modality `--rna-strategy` / `--atac-strategy`); document the preset in `docs/strategies.md`
   and the flag in README + `docs/input-data.md` §8 if it needs new options.
3. Tests (`tests/test_strategies.py`, marker `unit`): shape / dtype of the views, determinism under the same seed, behaviour at
   the `lmax` boundary, and the padding contract (`emb_idx = -1`, `kind = 0` at pads). If the strategy is deterministic add a
   `top`-style parity test against a reference embedding; if stochastic, a statistical one (mean cosine) — see §6.
4. Record it in `docs/test-plan.md` (S-series IDs) and CHANGELOG `Unreleased`.

## 6. How to add a test gate

Gates are contract tests with a stated tolerance ([test-plan.md](test-plan.md) §5 lists G1–G8, I1–I8, E-, P-, R-, S-series).

1. Pick the tier: `unit` (synthetic, no downloads), `regression` (committed fixture `tests/data/rna_100.h5ad` + the
   reference arrays in `tests/data/ref/rna_100_reference_embeddings.npz`; needs `genes` + `model`), `assets` (bucket
   fixtures, GPU). Fixtures that provide the assets and skip cleanly live in `tests/conftest.py` (`hyacinth_model_dir`,
   `gene_table_paths`, `multiome_20_dir`, `peak_tensors_dir`).
2. Write the test with a module-level tolerance constant, print one JSON line `[Gx <what>] {...}` with `max_abs`,
   `mean_abs`, `min_cosine` (helpers `diff_stats` / `load_reference` in the parity modules) and `record_property` the numbers,
   then assert the tolerance. Deterministic paths (`top`, `panel`) are compared by `max |Δ|`; stochastic ones (`random`)
   statistically (mean per-cell cosine).
3. New reference arrays come from the **unchanged** original `evaluate.py` only (maintainers, [maintainers.md](maintainers.md) §3)
   and ship in the bucket or `tests/data/ref/` with a sidecar JSON recording how they were produced.
4. Add the row to test-plan.md §5 (ID, reference, pass criterion, status) and the measured value to validation.md; mention the
   gate in the PR's validation evidence.

## 7. Maintainers

Regenerating assets or references, publishing to the bucket and the release checklist: [maintainers.md](maintainers.md).
