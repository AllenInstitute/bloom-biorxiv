"""``TopExpressed``: the deterministic ``nz_top{lmax}`` view."""

from __future__ import annotations

import numpy as np

from bloom.strategies.base import Universe, ViewStrategy, _candidates
from bloom.tokens import CellFeatures, TokenSet

__all__ = ["TopExpressed"]


class TopExpressed(ViewStrategy):
    """One view: the ``lmax`` detected genes with the largest values, no RNG.

    Ranking is a **stable** argsort of ``-value`` over the detected genes held in ascending
    ``emb_idx`` order, so ties break by ascending ``emb_idx`` (reference-table row). The emitted
    view is in rank order (highest value first).

    Tie-break note. ``alleninstitute/bloom`` ``StrategyTopExpressedGenes`` (``evaluate.py``) does
    the same stable sort over the cell's nonzero columns in **dataset column order**, i.e. ties
    break by ascending ``col_idx``. We rank in ``emb_idx`` space so the result does not depend on
    how a user's ``var`` happens to be ordered (a permuted AnnData gives the same embedding). The
    two rules coincide whenever ``col_idx -> emb_idx`` is monotone over the mapped genes, which
    holds for genomically ordered inputs such as the paper's ``allen_multiome`` data (verified on
    the ``rna_100`` fixture: identical top-4096 sets for all 100 cells although 57 cells have a
    value tie at the cut; ``tests/test_strategies.py::test_top_parity_with_evaluate_py``).

    Values are compared as given (raw counts or log1p(CPM)); any monotone per-cell transform
    gives the same ranking (S2). Peaks can be ranked only when the cell carries ``peak_values``
    (fragment counts, as ``prepare`` keeps for ATAC input); binary peak input raises.

    Args:
        lmax: Tokens in the view (``4096`` in the paper's ``nz_top4096``).
        universe: Optional feature list / ``emb_idx`` array applied before ranking.
    """

    name = "top"

    def __init__(self, lmax: int = 4096, universe: Universe = None) -> None:
        super().__init__(lmax, universe)

    def sample(self, cell: CellFeatures, rng: np.random.Generator | None = None) -> list[TokenSet]:
        emb, values, mod = _candidates(cell)
        if mod is None:
            return []
        if values is None:
            raise ValueError(
                "TopExpressed ranks by value; these peak features are binary (no counts) - use "
                "Resample, Tile or Panel, or prepare the ATAC file from a count matrix"
            )
        order = np.argsort(-values.astype(np.float32), kind="stable")  # (n,) ties -> asc emb_idx
        return [TokenSet.of(emb[order[: self.lmax]], mod)]
