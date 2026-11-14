"""``Panel``: a fixed, ordered feature list is the view (``StrategyFixed`` with a dummy pool)."""

from __future__ import annotations

import numpy as np

from bloom.feature_lists import FeatureList, FeatureReference
from bloom.strategies.base import ViewStrategy, _candidates
from bloom.tokens import CellFeatures, TokenSet

__all__ = ["Panel"]


class Panel(ViewStrategy):
    """One view: ``detected intersect panel`` in **panel order**, truncated to ``lmax``; no RNG.

    Features of the panel absent from the reference are dropped and reported once when the panel
    is resolved; panel features not detected in a cell are simply absent from that cell's view.
    ``universe`` is not a parameter: the panel *is* the restriction.

    Paper equivalent: ``StrategyFixed`` with a ``*_dummy.yaml`` pool (cell-typing panels, MERFISH).

    Args:
        features: Ordered panel as a :class:`FeatureList` (resolve against the reference with
            :meth:`resolve` before use) or an ``emb_idx`` array (already resolved).
        lmax: Maximum view length; a panel longer than ``lmax`` is truncated in panel order.
    """

    name = "panel"

    def __init__(self, features: FeatureList | np.ndarray, lmax: int = 8192) -> None:
        super().__init__(lmax, universe=None)
        self.features: FeatureList | np.ndarray
        self._panel_emb_idx: np.ndarray | None
        if isinstance(features, FeatureList):
            self.features = features
            self._panel_emb_idx = None
        else:
            arr = np.asarray(features, dtype=np.int64).reshape(-1)
            if arr.size and arr.min() < 0:
                raise ValueError("panel emb_idx must be >= 0 (drop unmapped -1 entries first)")
            _, first = np.unique(arr, return_index=True)
            self.features = arr[np.sort(first)].astype(np.int32)  # panel order, duplicates dropped
            self._panel_emb_idx = self.features

    @property
    def panel_emb_idx(self) -> np.ndarray | None:
        """Resolved panel in panel order, or ``None`` before :meth:`resolve`."""
        return self._panel_emb_idx

    @property
    def needs_resolve(self) -> bool:
        return self._panel_emb_idx is None

    def resolve(self, reference: FeatureReference) -> Panel:
        if isinstance(self.features, FeatureList):
            res = self.features.resolve(reference)
            _, first = np.unique(res.emb_idx, return_index=True)
            self._panel_emb_idx = res.emb_idx[np.sort(first)].astype(np.int32)
        return self

    def restrict(self, cell: CellFeatures) -> CellFeatures:
        if self.needs_resolve:
            raise RuntimeError(
                "Panel features are a FeatureList; call strategy.resolve(reference) before views()"
            )
        return cell

    def sample(self, cell: CellFeatures, rng: np.random.Generator | None = None) -> list[TokenSet]:
        emb, _, mod = _candidates(cell)
        if mod is None or self._panel_emb_idx is None:
            return []
        hit = self._panel_emb_idx[np.isin(self._panel_emb_idx, emb)]  # panel order
        if hit.size == 0:
            return []
        return [TokenSet.of(hit[: self.lmax], mod)]

    def params(self) -> dict[str, object]:
        feats: object
        if isinstance(self.features, FeatureList):
            feats = {
                "source": self.features.source,
                "n_ids": len(self.features),
                "kind": self.features.kind,
            }
        else:
            feats = {"n_emb_idx": int(self.features.size)}
        return {"strategy": self.name, "lmax": self.lmax, "panel": feats}
