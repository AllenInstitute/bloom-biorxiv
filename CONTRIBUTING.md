# Contributing

Working standards are decided in [docs/standards.md](docs/standards.md); this file is the operational summary.
Developer setup in depth (Linux / macOS, CPU-only, `BLOOM_HOME`, test tiers, adding a strategy or a test gate):
[docs/development.md](docs/development.md). Agents: read `AGENTS.md` first.

## Getting started (any machine)

```bash
git clone https://github.com/alleninstitute/bloom-biorxiv.git && cd bloom-biorxiv
scripts/setup_env.sh bloom-dev cpu                     # or cu128 with a CUDA GPU; prints <prefix> (conda + torch + pip install -e .[dev])
export PATH=<prefix>/bin:$PATH                         # or call <prefix>/bin/{bloom,pytest,ruff,pre-commit} by full path
bloom download --what genes,model                      # 0.6 GB from the public bucket into ~/.cache/bloom ($BLOOM_HOME)
scripts/check.sh                                       # = CI: ruff + unit + regression (G1 runs when the weights are present)
```

No host-specific setup is needed: the assets come from `https://bloom-biorxiv.s3.us-west-2.amazonaws.com/` (anonymous).
Issues labelled `good first issue` are self-contained starting points; the roadmap is [ROADMAP.md](ROADMAP.md).

## Branches

- One `main`, protected by convention: **never commit to `main`**, not even docs.
- Short-lived branches named `<type>/<short-desc>` from `main`, e.g. `feat/prepare-orthologs`,
  `fix/padding-mask`, `docs/hardware-table`, `chore/release-v0.2.0`.
- One PR per branch, focused on one change. Squash merge; the squash title is the PR title and must
  itself be a valid Conventional Commit.

## Commits: Conventional Commits

`<type>(<scope>)!: <summary>` — header <= 72 characters, imperative mood, body explains *why*.

- Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`, `ci`.
- Project scopes: `prepare`, `strategies`, `model`, `internals`, `assets`, `cli`
  (and `env`, `tooling`, `docs`, `release` for repository plumbing).
- `!` marks a breaking change to the **output schema** (`docs/data-schema.md`) or the **CLI**; say
  what breaks in the body under `BREAKING CHANGE:`.
- Enforced by review first; a commit-msg hook is added only if it slips.

## Pull requests

`.github/PULL_REQUEST_TEMPLATE.md` pre-fills the body; the three sections are mandatory:

1. **Problem** — what is wrong or missing, with the issue link.
2. **Result** — what changed, user-visible behaviour first.
3. **Validation evidence** — which test tier ran (`unit` / `regression` / `assets`), on which device (CPU, which GPU),
   tolerances observed; for model or strategy changes paste the `[G…]` JSON lines / the `embedding-parity` skill output.

Then `Closes #N`. Before pushing: `pre-commit run --all-files` and `scripts/check.sh` must pass. CI runs the fast tier
(`scripts/check.sh fast`: lint + `(unit or regression) and not weights`, gene table only) on every PR and the weight-dependent
tier (`pytest -m "weights and not assets"`, model downloaded and cached on the manifest sha; G1 among them) on pushes to
`main`, nightly and on demand (#51). Add a `type:<type>` label matching the title (`type:feat`, `type:fix`, `type:docs`, `type:test`, `type:chore`,
`type:ci`) — `.github/release.yml` groups the release notes by it. Document user-visible changes under `## [Unreleased]`
in `CHANGELOG.md`.

## Review loop (use-case driven)

The reviewer — human or agent — runs the end-user paths rather than reading diffs line by line:

```bash
gh pr checkout <n>
scripts/setup_env.sh review-<n> cpu              # fresh env per review (cu128 on a CUDA machine)
bloom embed --rna my.h5ad --species mouse -o out.h5ad      # bioinformatician path
python -c "..."                                  # ML path: model.encode(emb_idx, kind, lengths, return_internals=True), custom ViewStrategy
gh issue create --template use-case.md --label use-case:bioinformatician   # one issue per finding
gh issue list --label "use-case:ml"
```

Each finding becomes **one GitHub issue** using `.github/ISSUE_TEMPLATE/use-case.md` (persona, command/snippet run,
expected vs observed, data shape, `bloom info` output). Labels: exactly one of `use-case:bioinformatician` / `use-case:ml`;
`good first issue` when self-contained; `roadmap` for design-level items; every open issue is assigned to a milestone
(`v0.2.0` or `later`). Issues, not PR comments, so findings outlive the PR. The maintainer merges; a second human review
when available. Reviewers have triage permission on the repo.

## Test tiers

| marker | runs | contents | assets |
|---|---|---|---|
| `unit` | always, CI, < 1 min, CPU, no downloads | construction, parsing, strategies, CLI smoke, download/resume logic against a local HTTP server, internals at random init | none |
| `regression` | CI, CPU | parity vs committed reference embeddings on `tests/data/` fixtures (G1, G8, E1–E3) | `tests/data/` (in git) + `genes.*`; the weight-dependent ones also carry the marker `weights` (auto-added for the `hyacinth_model_dir` fixture): CI runs `(unit or regression) and not weights` on PRs (~3 min) and `weights and not assets` on pushes to `main` / nightly (~20 min, downloads `model`); locally `bloom download --what genes,model`, else they skip |
| `assets` | manually before a tag, any machine with the assets (GPU gates need CUDA) | real downloads from the bucket, model loading, GPU parity (G2a/G2b/G7), G4/G5 statistical parity, 20-cell ATAC fixture, I8 internals, timing | `bloom download` (everything ≈ 6.1 GB; `peaks` 5.4 GB only for ATAC) |

`scripts/check.sh` = `ruff check` + `ruff format --check` + `pytest -m "unit or regression"` (`scripts/check.sh fast` = the PR
job, `and not weights`). Contract list with IDs and
tolerances: [docs/test-plan.md](docs/test-plan.md). Every behaviour change to `model/`, `strategies/`, `tokens.py` or
`prepare.py` adds or updates one of those tests (how: [docs/development.md](docs/development.md) §5–§6).

## Environments

`pyproject.toml` is the only dependency file (no lockfile, no `environment.yml`). Create one conda env
per worktree / task: `scripts/setup_env.sh <env> [cpu|cu128]` (torch build chosen by the user; `uv pip`
inside the env is fine, not required). Assets live in `BLOOM_HOME` (default `~/.cache/bloom`), shared across
envs and worktrees. There is no GPU CI; any CUDA machine with ≥ 10 GB VRAM can run the GPU gates after `bloom download`
(the maintainers use a single L4; [docs/maintainers.md](docs/maintainers.md)).

## Releases

Release = git tag `vX.Y.Z` + GitHub release notes (auto-generated from PR labels, `.github/release.yml`) + the matching
`CHANGELOG.md` section ([Keep a Changelog](https://keepachangelog.com)); no archival DOI for the bioRxiv releases (one
accompanies the publication). Assets are versioned independently via `MANIFEST.json` (`model_id` `hyacinth`,
`reference_id` `mm10-evo2-v1`, `examples-v2`). Checklist: [docs/maintainers.md](docs/maintainers.md) §7. No PyPI yet
([ROADMAP.md](ROADMAP.md)).
