# Model card — `hyacinth` (Bloom)

**Model id** `hyacinth` · **reference tables** `mm10-evo2-v1` · **weights** `model.safetensors`
(496,624,516 B, sha256 `79fbd1092ded8e88c9d8d7ba3771e14a20933c18bd14da1dad81a3fd23eb5dcc`) ·
**config** `config.json` (3,923 B) · **released** 2026-11-13 (`bloom-biorxiv` v0.1.0) · **license** [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (see below).

## Summary

Bloom is a multimodal single-cell transformer for the mouse brain. A cell is represented by the *set
of features it detects* — genes with non-zero RNA counts and, for ATAC, accessible peaks — and every
feature enters the model through a sequence-derived embedding of its genomic sequence (Evo2 mean
embedding, PCA to 768 d) plus genomic position (chromosome, bp). A first attention layer with
sequence-, position- and standard heads contextualises the feature tokens; a Perceiver-style
encoder compresses them into 256 latents through 8 self-attention layers; the cell embedding is the
mean-pooled, layer-normalised latent state (1024 d). Expression magnitude is not used: only
`X > 0` matters, so library size and normalisation do not enter the embedding. `hyacinth` is the
checkpoint evaluated in the Bloom preprint (Gala, Convy, Marghi et al., *Bloom unifies brain cell representations across modalities and contexts*, bioRxiv 2026, doi:10.64898/2026.11.11.765881).

## Architecture (from `config.json`)

| quantity | value |
|---|---:|
| parameters | 124,144,831 (encoder 113,632,256; first layer 10,231,569; decoder 161,926; feature embedding 119,080) + 7 persistent buffers |
| `d_model` | 1024 |
| latents | 256 |
| transformer layers | 8 (`n_heads` 8) |
| first-layer heads | 8 sequence / 8 position / 4 standard (`d_val` 40, `d_u` 32, `K` 64, `d_v` 32, `max_G` 32) |
| sequence-head scale | 1/sqrt(32) = 0.17677669529663687 (not stored in the checkpoint; asserted at load) |
| feature embedding input | `d_cseq` 768 + descriptor 128 + metadata 48 + measurement 80 |
| context length | `Lmax` 8192 tokens per view (`Lq` 8054) |
| weights dtype | float32, weights-only (`safetensors`, 198 tensors) |

## Training data and provenance

- Training repository `alleninstitute/bloom`, training commit `671c8cb`; attention block
  `AllenInstitute/bloom-sprint-attn @ ef5fb8a8e5768071da9d9542a595dd8e05b0cb03` (public).
- Training run `hyacinth`, checkpoint `last.ckpt` (step 530,000; sha256 `45ba9313…`), global step 530,000,
  epoch 2069; PyTorch Lightning 2.6.1, precision `bf16-mixed`; training views
  from `StrategyRandomContrastiveNonzeroOnly` over the `mm-brain-adult_free_hvg-8k` pool, inputs
  binarised, metadata tokens (`ftype`, `mtype`, `batch`) masked.
- Exported 2026-11-12 with `tools/export_checkpoint.py` (torch 2.11.0+cu128, export commit `141f0d3` of the training repo):
  `state_dict` keys remapped (`model.bloom_first_layer.` -> `first_layer.`, `model.encoder.` ->
  `encoder.`, `model.decoder.` -> `decoder.`, `model.feature_emb.` -> `feature_emb.`); the
  alignment projection, probe heads, optimizer and scheduler states were dropped.
- Training data: mouse brain single-cell / single-nucleus RNA and ATAC datasets (the `hyacinth-v1`
  dataset collection of the Bloom project), mapped to the `mm10-evo2-v1` gene (32,129) and peak
  (1,747,701 x 501 bp) tables. Dataset list and preprocessing: see the manuscript.

## Intended use

- Embedding mouse brain cells from RNA (`bloom embed --rna ... --species mouse`) and from ATAC on the
  mm10 reference peak set (`bloom embed --atac ...`, exact or overlap peak matching); downstream clustering,
  label transfer, integration across modalities and datasets, and interpretability of the attention layer
  (`torch.nn.Module` API, `docs/internals.md`).
- Human RNA via the shipped Ensembl 115 one-to-one ortholog table (`--species human`): human symbols
  or Ensembl ids are mapped to their mouse orthologs (15,676 one-to-one pairs) and embedded with the
  mouse reference. This is a documented route, not a human model.

## Limitations

- **Mouse-only reference.** Feature embeddings exist only for mm10 genes and peaks; any human input is
  projected through orthologs and loses non-orthologous genes. There is no hg38 table yet.
- **Brain-trained.** Trained on adult mouse brain; behaviour on other tissues is unvalidated.
- **Binarised input.** Expression level, batch and library size are ignored by design; cells with no
  detectable reference gene get a NaN embedding.
- **View sampling.** The paper default (`random`: 4 draws of up to 8,192 detected genes, averaged) is
  stochastic for cells detecting more than 8,192 genes (seeded per cell); `top` (4,096 highest-count
  genes) is deterministic but differs from `random` (mean cosine 0.979 on the example cells).
- **Numerics.** The companion code runs the attention layers with the fused CUDA fast path disabled, so GPU
  and CPU embeddings agree to ~1e-5; the paper's own embeddings were computed with the fused kernel and TF32
  and differ from these by up to a few 1e-3 (`docs/validation.md`).
- **ATAC.** Peak-based inference needs the 5.4 GB peak table (`bloom download --what peaks`), ~12 GB RAM on
  CPU, and has been validated by parity with the paper pipeline (mean cosine 0.9995) and a deterministic
  GPU/CPU check; a label-transfer sanity check on ATAC embeddings has not been run.

## Evaluation (companion code `bloom-biorxiv` v0.1.0, `docs/validation.md`, 2026-11-13)

| check | result |
|---|---|
| reproduces the paper pipeline (`top` 4096, CPU fp32 vs reference embeddings, 100 cells) | max abs diff 1.34e-5, min cosine 0.99999988 |
| same on an NVIDIA L4 (TF32 off; CPU vs GPU) | 1.91e-6 (GPU vs reference), 1.4e-5 (CPU vs GPU) |
| `random` 8192 x 4 vs reference | mean cosine 1.0 (max abs 5.89e-4) |
| ATAC `random` 8192 x 16 (20 example cells) vs reference | mean cosine 0.99949 (min 0.9978) |
| kNN AIT33 subclass accuracy, 300 example cells (30 subclasses x 10), k = 15 leave-one-out | 0.923 (identical to the paper's embeddings of the same cells; per-cell cosine 0.999998) |
| `top` strategy, same kNN | 0.933 |
| throughput | `top` 4096: 0.45 s/cell on 16 CPU threads, 0.055 s/cell on an L4; `random` default 0.20 s/cell on an L4 |

## License

The model weights (`models/hyacinth/*`), the derived reference tables (`references/mm10-evo2-v1/*`),
the example data (`examples/*`) and this model card are released under the
**Creative Commons Attribution-NonCommercial 4.0 International** license
([CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)). You may share and adapt them for
non-commercial purposes with attribution to the Allen Institute and the Bloom manuscript; every
`MANIFEST.json` entry carries `"license": "CC-BY-NC-4.0"`. The example cells are a subset of the Allen
Institute whole-mouse-brain multiome data redistributed under the same terms. The companion code is
released separately under the MIT license (repository `LICENSE`).

## Citation

Gala, Convy, Marghi et al., *Bloom unifies brain cell representations across modalities and contexts*, bioRxiv 2026, doi:[10.64898/2026.11.11.765881](https://doi.org/10.64898/2026.11.11.765881)

```bibtex
@article{gala2026bloom,
  title   = {Bloom unifies brain cell representations across modalities and contexts},
  author  = {Gala, R. and Convy, I. and Marghi, Y. and Liu, R. and Roghair, J. and Xue, R. and Gouwens, N. and Otto, S. and Caldwell, S. and Khinvasara, K. and Martin, J. and Sengupta, S. and Yao, Z. and Hawrylycz, M. and Mufti, S. and Zeng, H. and Sumbul, U.},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.11.11.765881},
  url     = {https://doi.org/10.64898/2026.11.11.765881}
}
```

This bioRxiv release is provided as an early-access courtesy so that scientists can use the model in good faith; an archival
release will accompany the publication.
