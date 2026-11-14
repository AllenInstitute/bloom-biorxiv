"""``Resample`` = the ``random`` preset: the paper's default (``nz_resamp_{lmax}_d{draws}``)."""

from __future__ import annotations

import numpy as np

from bloom.strategies.base import Universe, ViewStrategy, _candidates
from bloom.tokens import CellFeatures, TokenSet

__all__ = ["Resample"]


class Resample(ViewStrategy):
    """``draws`` random draws of up to ``lmax`` detected features (preset name ``random``).

    Each draw is a uniform sample of ``min(lmax, n_detected)`` features without replacement **within**
    the draw (no duplicates in a view); draws are independent of each other, so the same feature may
    recur across draws. Tokens within a view are emitted in ascending ``emb_idx`` (token order does
    not enter the model). With the default ``lmax = 8192`` a typical RNA cell detects fewer features
    than ``lmax``, so every draw is the full detected set and the draws coincide.

    Paper equivalent: ``nz_resamp_8192_d4`` for RNA, ``nz_resamp_atac_8192_d16`` for ATAC; with a
    ``universe`` it is ``StrategyTiledPool(idx_file=...)`` restricted sampling.

    Args:
        lmax: Tokens per view (``8192`` in the paper).
        draws: Number of views per cell (``4`` RNA / ``16`` ATAC in the paper).
        universe: Optional feature list / ``emb_idx`` array; detected features are intersected
            with it first.
    """

    name = "random"

    def __init__(self, lmax: int = 8192, draws: int = 4, universe: Universe = None) -> None:
        super().__init__(lmax, universe)
        if int(draws) < 1:
            raise ValueError(f"draws must be >= 1, got {draws}")
        self.draws: int = int(draws)

    def sample(self, cell: CellFeatures, rng: np.random.Generator) -> list[TokenSet]:
        emb, _, mod = _candidates(cell)
        n = int(emb.size)
        if n == 0 or mod is None:
            return []
        size = min(self.lmax, n)
        out: list[TokenSet] = []
        for _ in range(self.draws):
            pick = rng.choice(n, size=size, replace=False)  # (size,) positions into emb
            out.append(TokenSet.of(np.sort(emb[pick]), mod))
        return out

    def params(self) -> dict[str, object]:
        return {**super().params(), "draws": self.draws}
