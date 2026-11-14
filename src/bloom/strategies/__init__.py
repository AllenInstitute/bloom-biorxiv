"""View strategies: the v1 presets and the ``ViewStrategy`` extension point.

Presets (docs/architecture.md section 5.2): ``random`` -> :class:`Resample` (the paper default:
independent random draws of detected features), ``top`` -> :class:`TopExpressed` (deterministic),
``panel`` -> :class:`Panel` (a fixed feature list), ``tile`` -> :class:`Tile` (draws that partition
the detected set: no feature in more than one draw; issue #31). :func:`build` maps the config keys
``strategy, lmax, draws, universe, panel`` (docs/data-schema.md section 3) to a preset; :func:`get`
returns the class for a name (``resample`` is a deprecated alias of ``random``).
"""

from bloom.strategies.base import Universe, ViewStrategy, cell_rng
from bloom.strategies.factory import (
    ALIASES,
    DEFAULT_DRAWS,
    DEFAULT_LMAX,
    PRESETS,
    build,
    canonical_name,
    get,
)
from bloom.strategies.panel import Panel
from bloom.strategies.resample import Resample
from bloom.strategies.tile import Tile
from bloom.strategies.top import TopExpressed

__all__ = [
    "ALIASES",
    "DEFAULT_DRAWS",
    "DEFAULT_LMAX",
    "PRESETS",
    "Panel",
    "Resample",
    "Tile",
    "TopExpressed",
    "Universe",
    "ViewStrategy",
    "build",
    "canonical_name",
    "cell_rng",
    "get",
]
