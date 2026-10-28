"""Error norms and observed orders of accuracy for convergence studies."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
Norm = Literal["l2", "max"]


def error_norms(numerical: ArrayLike, exact: ArrayLike, volume: ArrayLike) -> tuple[float, float]:
    """Volume-weighted RMS (discrete L2) and maximum norms of ``numerical - exact``.

    The L2 norm is ``sqrt(sum(V e^2) / sum(V))``, so it does not grow with the
    number of cells and is comparable across grids.
    """
    e = np.asarray(numerical, dtype=np.float64) - np.asarray(exact, dtype=np.float64)
    v = np.broadcast_to(np.asarray(volume, dtype=np.float64), e.shape)
    l2 = float(np.sqrt(np.sum(v * e**2) / np.sum(v)))
    return l2, float(np.max(np.abs(e)))


def observed_orders(sizes: Sequence[float], errors: Sequence[float]) -> FloatArray:
    """Order between consecutive levels: ``log(e_k / e_{k+1}) / log(h_k / h_{k+1})``."""
    h = np.asarray(sizes, dtype=np.float64)
    e = np.asarray(errors, dtype=np.float64)
    if h.shape != e.shape or h.size < 2:
        raise ValueError("need matching sizes and errors on at least two levels")
    if np.any(e <= 0.0) or np.any(h <= 0.0):
        raise ValueError("sizes and errors must be positive")
    return np.asarray(np.log(e[:-1] / e[1:]) / np.log(h[:-1] / h[1:]), dtype=np.float64)


def fitted_order(sizes: Sequence[float], errors: Sequence[float]) -> float:
    """Least-squares slope of ``log(error)`` against ``log(size)`` over all levels."""
    h = np.log(np.asarray(sizes, dtype=np.float64))
    e = np.log(np.asarray(errors, dtype=np.float64))
    if h.shape != e.shape or h.size < 2:
        raise ValueError("need matching sizes and errors on at least two levels")
    slope = np.polyfit(h, e, 1)[0]
    return float(slope)


@dataclass(frozen=True)
class ConvergenceStudy:
    """Errors on a sequence of refined discretizations.

    ``parameter`` is the quantity being refined ("h" for the cell size, "dt"
    for the time step); ``sizes`` are its values (coarse to fine) and
    ``levels`` the matching cell or step counts. ``worst_cell``, when
    recorded, is the centre of the cell with the largest error on the finest
    level.
    """

    name: str
    description: str
    parameter: str
    levels: tuple[int, ...]
    sizes: tuple[float, ...]
    errors_l2: tuple[float, ...]
    errors_max: tuple[float, ...]
    expected_order: float
    worst_cell: tuple[float, ...] = ()

    def errors(self, norm: Norm = "l2") -> tuple[float, ...]:
        return self.errors_l2 if norm == "l2" else self.errors_max

    def orders(self, norm: Norm = "l2") -> FloatArray:
        return observed_orders(self.sizes, self.errors(norm))

    def finest_order(self, norm: Norm = "l2") -> float:
        """Observed order between the two finest levels."""
        return float(self.orders(norm)[-1])

    def fitted_order(self, norm: Norm = "l2") -> float:
        return fitted_order(self.sizes, self.errors(norm))
