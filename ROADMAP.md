# Roadmap

What comes after `v0.1.0` ([CHANGELOG.md](CHANGELOG.md)), by theme and milestone. The GitHub milestones
[`v0.2.0`](https://github.com/alleninstitute/bloom-biorxiv/issues?q=is%3Aopen+milestone%3Av0.2.0) and [`later`](https://github.com/alleninstitute/bloom-biorxiv/issues?q=is%3Aopen+milestone%3Alater) are the live
view; every open issue is assigned to one of them. Items without an issue get one when work starts.

## v0.2.0 — scale and completeness of the inference path

| theme | item | issue |
|---|---|---|
| memory | memory-map the 5.4 GB peak table on CPU instead of reading it into RAM (12 GB RSS today); keep if < 10 % slower — deferred from the v0.1.0 re-cut (2026-11-14) | [#1](https://github.com/alleninstitute/bloom-biorxiv/issues/1) |
| performance | evaluate the fused CUDA attention kernel (~1.5× faster on long views; 9.2e-3 vs CPU at L = 8192) outside this repo and decide whether an opt-in flag is worth a documented tolerance | — |
| reference | native hg38 reference tables (genes, peaks) so human input no longer goes through orthologs | — |
| ML API | decoder / reconstruction API (the loaded-but-unused decoder head; needs library size + batch in the schema) | — |
| docs | MkDocs site on GitHub Pages; notebook conventions and first notebooks (attention ordering Q5, ATAC walk-through) | — |
| distribution | PyPI release (`bloom-arch` must be on PyPI or vendored first); archival DOI with the publication | — |

## later

| theme | item | issue |
|---|---|---|
| validation | label-transfer (kNN) sanity check on ATAC embeddings (the 300-cell archived set) | [#2](https://github.com/alleninstitute/bloom-biorxiv/issues/2) |
| scale | multi-GPU / streaming embedding for > 1 M cells | — |
| reference | custom peak sets: Evo2 → PCA-768 embedding pipeline for peaks outside the released table; pseudobulk peak tables | — |
| training | fine-tuning hooks; ablation checkpoints | — |
| numerics | fp16 / bf16 peak table; G3 (bf16) as a gate | — |

Not planned: training code (lives in `alleninstitute/bloom`), Lightning / Hydra dependencies, a lockfile.
