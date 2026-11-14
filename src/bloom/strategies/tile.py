"""``Tile`` = the ``tile`` preset: shuffle once, cover the detected set with disjoint ``lmax`` chunks.

Exposed as a ``--strategy`` / config preset since v0.1.0 after statistical parity with the paper
pipeline's ``StrategyTiledPool(draw_mode="tile")`` on the 20-cell multiome fixture (issue #31; gate
G9 in docs/test-plan.md).
"""

from __future__ import annotations

import math

import numpy as np

from bloom.strategies.base import Universe, ViewStrategy, _candidates
from bloom.tokens import CellFeatures, TokenSet

__all__ = ["Tile"]


class Tile(ViewStrategy):
    """Disjoint views: one random permutation of the detected set, cut into ``lmax`` chunks (preset ``tile``).

    The draws *partition* the detected features: no feature appears in more than one draw, so
    ``max_draws x lmax`` tokens cover the whole detected set of a cell (ATAC cells detect tens of
    thousands of peaks; ``random`` re-draws overlapping subsets instead). CLI: ``--strategy tile
    --lmax 8192 --draws 4`` (``--draws`` is the cap ``max_draws``).

    The number of views is ``min(max_draws, ceil(n_detected / lmax))``; views are pairwise
    disjoint and their union is the whole detected set whenever ``n_detected <= max_draws * lmax``
    (otherwise the tail of the permutation is not emitted). The last chunk may be shorter than
    ``lmax``. Tokens within a view are emitted in ascending ``emb_idx``.

    Paper equivalent: ``nz_nonrepeat_{lmax}_d{n}``.

    Args:
        lmax: Tokens per view.
        max_draws: Upper bound on the number of views per cell.
        universe: Optional feature list / ``emb_idx`` array applied before shuffling.
    """

    name = "tile"

    def __init__(self, lmax: int = 8192, max_draws: int = 4, universe: Universe = None) -> None:
        super().__init__(lmax, universe)
        if int(max_draws) < 1:
            raise ValueError(f"max_draws must be >= 1, got {max_draws}")
        self.max_draws: int = int(max_draws)

    def sample(self, cell: CellFeatures, rng: np.random.Generator) -> list[TokenSet]:
        emb, _, mod = _candidates(cell)
        n = int(emb.size)
        if n == 0 or mod is None:
            return []
        perm = rng.permutation(n)  # (n,) positions into emb
        n_views = min(self.max_draws, math.ceil(n / self.lmax))
        return [
            TokenSet.of(np.sort(emb[perm[i * self.lmax : (i + 1) * self.lmax]]), mod)
            for i in range(n_views)
        ]

    def params(self) -> dict[str, object]:
        return {**super().params(), "draws": self.max_draws}
