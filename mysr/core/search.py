"""High-level search orchestration.

The Python side prepares data + config, delegates the actual evolutionary
search to the Julia backend (``MySR.jl`` -> SymbolicRegression.jl) through
:mod:`mysr.julia_bridge`, and returns a :class:`HallOfFame`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np

from .config import SearchConfig
from ..julia_bridge import run_search


class HallOfFame:
    """Collection of discovered equations, ordered by loss."""

    def __init__(self, equations: List[Dict[str, Any]]):
        self.equations = equations

    def __len__(self) -> int:
        return len(self.equations)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        return self.equations[i]

    def __repr__(self) -> str:
        head = "\n".join(
            f"  loss={e.get('loss'):.6g}  complexity={e.get('complexity')}  "
            f"eq={e.get('equation')}"
            for e in self.equations[:5]
        )
        return f"HallOfFame({len(self.equations)} equations)\n{head}"


def fit(
    X: Union[np.ndarray, List],
    y: Union[np.ndarray, List],
    config: Optional[SearchConfig] = None,
    **kwargs: Any,
) -> HallOfFame:
    """Run a MySR search.

    Parameters
    ----------
    X : (n_samples, n_features) array of features.
    y : (n_samples,) array of targets.
    config : optional SearchConfig; defaults to defaults.
    **kwargs : override individual SearchConfig fields.
    """
    if config is None:
        config = SearchConfig()
    for key, value in kwargs.items():
        if not hasattr(config, key):
            raise TypeError(f"unknown SearchConfig field: {key!r}")
        setattr(config, key, value)

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError(
            f"shape mismatch: X={X.shape}, y={y.shape}; expected (n, d) and (n,)"
        )

    raw = run_search(X, y, config.to_julia_dict(), verbose=config.verbose)
    return HallOfFame(raw)
