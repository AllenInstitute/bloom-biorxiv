"""Preset factory: the config block ``{strategy, lmax, draws, universe, panel}`` -> a preset.

User-facing preset names: ``random`` (the paper default: ``draws`` independent random draws of up to
``lmax`` detected features; a feature may recur across draws), ``top`` (the ``lmax`` most highly
expressed detected features; deterministic), ``panel`` (your own ordered feature list) and ``tile``
(the draws *partition* the detected features: one shuffle cut into disjoint ``lmax`` chunks, no feature
in more than one draw; ``ceil(n_detected / lmax)`` draws capped by ``draws``; exposed in v0.1.0 after
validation against the paper pipeline, issue #31). ``resample`` is the pre-release name of ``random``
and is still accepted with a ``DeprecationWarning`` for one release.
"""

from __future__ import annotations

import warnings
from os import PathLike

import numpy as np

from bloom.feature_lists import FeatureList
from bloom.strategies.base import Universe, ViewStrategy
from bloom.strategies.panel import Panel
from bloom.strategies.resample import Resample
from bloom.strategies.tile import Tile
from bloom.strategies.top import TopExpressed

__all__ = ["ALIASES", "DEFAULT_DRAWS", "DEFAULT_LMAX", "PRESETS", "build", "canonical_name", "get"]

PRESETS: dict[str, type[ViewStrategy]] = {
    Resample.name: Resample,
    TopExpressed.name: TopExpressed,
    Panel.name: Panel,
    Tile.name: Tile,
}
"""Preset name (the ``strategy:`` config key) -> class: ``random``, ``top``, ``panel``, ``tile``."""

ALIASES: dict[str, str] = {"resample": "random"}
"""Deprecated preset names -> current name (accepted with a ``DeprecationWarning`` for one release)."""

DEFAULT_LMAX: dict[str, int] = {"random": 8192, "top": 4096, "panel": 8192, "tile": 8192}
"""Paper defaults per preset (docs/architecture.md section 3.1)."""
DEFAULT_DRAWS: int = 4
"""``random`` RNA default (the ATAC default, 16, is applied by the caller that knows the modality) and
the default cap on the number of ``tile`` draws (``max_draws``)."""


def canonical_name(strategy: str) -> str:
    """Return the current preset name for ``strategy`` (case-insensitive; aliases warn).

    Raises:
        ValueError: unknown name.
    """
    key = str(strategy).strip().lower()
    if key in ALIASES:
        warnings.warn(
            f"strategy {key!r} was renamed to {ALIASES[key]!r}; the old name is accepted for one "
            "release",
            DeprecationWarning,
            stacklevel=3,
        )
        key = ALIASES[key]
    if key in PRESETS:
        return key
    raise ValueError(f"unknown strategy {strategy!r}; choose from {sorted(PRESETS)}")


def get(strategy: str) -> type[ViewStrategy]:
    """Preset class for a name: ``bloom.strategies.get("random") is Resample``."""
    return PRESETS[canonical_name(strategy)]


def _as_feature_list(x: FeatureList | np.ndarray | str | PathLike[str] | None) -> Universe:
    if x is None or isinstance(x, FeatureList | np.ndarray):
        return x
    return FeatureList.from_file(x)


def build(
    strategy: str,
    lmax: int | None = None,
    draws: int | None = None,
    universe: FeatureList | np.ndarray | str | PathLike[str] | None = None,
    panel: FeatureList | np.ndarray | str | PathLike[str] | None = None,
) -> ViewStrategy:
    """Preset factory for the config block ``{strategy, lmax, draws, universe, panel}``.

    Args:
        strategy: ``"random" | "top" | "panel" | "tile"`` (case-insensitive; ``"resample"`` is a
            deprecated alias of ``"random"``).
        lmax: Tokens per view; default ``8192`` (``4096`` for ``top``).
        draws: Views per cell (``random``: exactly ``draws``; ``tile``: at most ``draws`` =
            ``Tile.max_draws``, the actual count is ``min(draws, ceil(n_detected / lmax))``); default
            ``4``. The ATAC ``random`` default of ``16`` is applied by the caller that knows the modality.
        universe: Feature-list path, parsed :class:`FeatureList` or ``emb_idx`` array; not
            accepted for ``panel``.
        panel: Required for ``panel``: feature-list path, :class:`FeatureList` or ``emb_idx`` array.

    Returns:
        The preset instance. A :class:`FeatureList` universe/panel still needs
        ``strategy.resolve(reference)`` before ``views()``.
    """
    key = canonical_name(strategy)
    L = int(lmax) if lmax is not None else DEFAULT_LMAX[key]
    uni = _as_feature_list(universe)
    if key == "panel":
        if panel is None:
            raise ValueError("strategy 'panel' needs `panel` (a feature-list file)")
        if universe is not None:
            raise ValueError(
                "strategy 'panel' does not take a universe (the panel is the restriction)"
            )
        return Panel(_as_feature_list(panel), lmax=L)
    if panel is not None:
        raise ValueError(f"`panel` is only valid with strategy 'panel', not {key!r}")
    if key == "top":
        if draws is not None and int(draws) != 1:
            raise ValueError("strategy 'top' is a single deterministic view; draws must be 1/unset")
        return TopExpressed(lmax=L, universe=uni)
    d = int(draws) if draws is not None else DEFAULT_DRAWS
    if key == "tile":
        return Tile(lmax=L, max_draws=d, universe=uni)
    return Resample(lmax=L, draws=d, universe=uni)
