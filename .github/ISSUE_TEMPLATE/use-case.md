---
name: Use-case finding
about: A reviewer or user exercised an end-user path and it did not do what they expected
title: "[use-case] <persona>: <one-line summary>"
labels: ""
assignees: ""
---

<!-- One finding per issue. Add exactly one of the labels use-case:bioinformatician / use-case:ml,
     and a phase:1..4 or roadmap label if you can place it. -->

## Persona

<!-- bioinformatician (own AnnData -> embeddings via the CLI) | ml (torch.nn.Module, internals, custom ViewStrategy) -->

## Command or snippet run

```bash
# exact command line, or the Python snippet, including the git sha / PR number checked out
```

## Expected vs observed

**Expected:**

**Observed:** <!-- full error message / wrong numbers / unexpected output schema -->

## Data shape

- cells x features:
- species: <!-- mouse | human -->
- modality: <!-- rna | atac | both -->
- value type / layer used: <!-- counts, log1p(CPM), ... (should not matter: only X > 0 is used) -->

## Environment

```
# paste the output of `bloom info`
```

## Notes

<!-- optional: minimal reproducer, workaround, which contract in docs/test-plan.md this touches -->
