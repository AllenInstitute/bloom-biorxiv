<!-- Title = Conventional Commit header (<= 72 chars), e.g. `fix(prepare): warn on duplicate peak coordinates`.
     It becomes the squash-commit message. CONTRIBUTING.md has the full rules. -->

## Problem

<!-- What is wrong or missing, and for whom (bioinformatician CLI path / ML API path). Link the issue. -->

## Result

<!-- What changed, user-visible behaviour first. Output schema or CLI changes need a `!` in the title and a
     `BREAKING CHANGE:` note here. -->

## Validation evidence

<!-- Which test tier ran (`unit` / `regression` / `assets`), on which device (CPU, which GPU), and the numbers:
     paste the `[G...]` JSON lines for model / strategy / prepare changes, or the `embedding-parity` skill output.
     `pre-commit run --all-files` and `scripts/check.sh` must be green before pushing. -->

- [ ] `scripts/check.sh` green locally (lint + unit + regression)
- [ ] new / changed behaviour has a contract test (docs/test-plan.md) with a tolerance
- [ ] docs updated (README / docs/*.md / CHANGELOG.md `Unreleased`)

Closes #
