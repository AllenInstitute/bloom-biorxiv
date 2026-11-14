"""Host-independent locations for the maintainer tools (docs/maintainers.md, "Maintainer environment").

Every tool reads its inputs and outputs from environment variables instead of hard-coded host paths;
``env_path("BLOOM_STAGING")`` exits with the variable's purpose when it is unset. ``BLOOM_STAGING`` is the
only one most tools need; the rest refer to the original training artefacts (checkpoint, Evo2 caches,
training-data pools, the paper's ``evaluate.py`` outputs) that exist only on the maintainers' host.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ENV_DOC: dict[str, str] = {
    "BLOOM_STAGING": "staging tree = bucket content (models/, references/, examples/, MANIFEST.json; archive/ is host-only)",
    "BLOOM_TRAINING_CKPT": "the original training checkpoint (Lightning last.ckpt of the released run)",
    "BLOOM_TRAINING_CONFIG": "the Hydra config.yaml of that training run",
    "BLOOM_EVO2_CACHE": "Evo2 feature-embedding cache dir (evo2_{genes,peaks}_mean_ctx_2048_pca_768_norm)",
    "BLOOM_TRAINING_DATA": "training-data root: embeddings/, strategies/eval/, feature-maps/, datasets/<collection>/",
    "BLOOM_PAPER_RESULTS": "the paper's evaluate.py results root (<task>/embeddings.h5ad per eval task)",
    "BLOOM_EVAL_INPUTS": "raw evaluate.py inputs per fixture (rna.h5ad, atac.h5ad, metadata.h5ad, feature-maps/)",
    "BLOOM_EVAL_INPUTS_SORTED": "column-sorted copies of BLOOM_EVAL_INPUTS (tools/sort_evalinputs.py)",
    "BLOOM_ARCH_WORKTREE": "bloom-arch checkout at the pinned commit (seq-head scale 1/sqrt(32)); its src/ goes first on PYTHONPATH",
    "BLOOM_TRAINING_REPO": "checkout of alleninstitute/bloom whose unchanged evaluate.py produces the references",
    "BLOOM_LEGACY_ORTHOLOGS": "dir with the legacy ortholog tables (orthologs_one2one.tsv, biomart_raw_response.tsv, *_mouseuniverse.tsv)",
}


def env_path(var: str, default: str | None = None) -> Path:
    """``Path(os.environ[var])`` or exit 2 naming the variable and what it must point at."""
    value = os.environ.get(var, default)
    if not value:
        sys.exit(
            f"{var} is not set: it must point at the {ENV_DOC.get(var, 'required path')} (docs/maintainers.md)"
        )
    return Path(value).expanduser()


def env_str(var: str, default: str | None = None) -> str:
    return str(env_path(var, default))
