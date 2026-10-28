"""Boundary conditions for one side of the domain.

Sign convention: ``Neumann.flux`` is the heat flux *entering* the domain
(positive heats the body). With ``n`` the outward unit normal,

* Dirichlet:  ``T = T_b``
* Neumann:    ``k dT/dn = q_in``
* Robin:      ``k dT/dn = h (T_inf - T)``  (convection to an ambient ``T_inf``)

Boundary data may be a constant or a function. In 1D a function takes the
time, ``f(t)``. In 2D it takes the coordinate along the side and the time,
``f(s, t)``, where ``s`` is the array of face-centre coordinates on that side
(``y`` on the left and right sides, ``x`` on the bottom and top sides) and the
result must broadcast to the shape of ``s``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

BoundaryData = float | Callable[..., ArrayLike]
"""A constant, ``f(t)`` in 1D, or ``f(s, t)`` in 2D."""


@dataclass(frozen=True)
class Dirichlet:
    """Prescribed surface temperature ``T_b``."""

    temperature: BoundaryData


@dataclass(frozen=True)
class Neumann:
    """Prescribed heat flux entering the domain (``0.0`` is an insulated side)."""

    flux: BoundaryData


@dataclass(frozen=True)
class Robin:
    """Convection: heat entering is ``h * (ambient - T_surface)``.

    ``h`` is a positive constant so the system matrix does not change in time
    (one factorization serves the whole run); ``ambient`` may vary.
    """

    h: float
    ambient: BoundaryData

    def __post_init__(self) -> None:
        if not (math.isfinite(self.h) and self.h > 0.0):
            raise ValueError(f"the heat transfer coefficient h must be positive, got {self.h!r}")


BoundaryCondition = Dirichlet | Neumann | Robin


def boundary_data(condition: BoundaryCondition) -> BoundaryData:
    """The time- or space-dependent datum of a condition (temperature, flux or ambient)."""
    if isinstance(condition, Dirichlet):
        return condition.temperature
    if isinstance(condition, Neumann):
        return condition.flux
    if isinstance(condition, Robin):
        return condition.ambient
    raise TypeError(f"not a boundary condition: {condition!r}")


def evaluate_data(data: BoundaryData, s: FloatArray | None, t: float, size: int) -> FloatArray:
    """Evaluate boundary data on ``size`` faces at time ``t``.

    ``s`` is ``None`` in 1D (``data`` is called as ``data(t)``) and the
    face-centre coordinates along the side in 2D (``data(s, t)``).
    """
    value: ArrayLike
    if not callable(data):
        value = data
    elif s is None:
        value = data(t)
    else:
        value = data(s, t)
    out = np.broadcast_to(np.asarray(value, dtype=np.float64), (size,)).copy()
    if not np.all(np.isfinite(out)):
        raise ValueError(f"boundary data is not finite at t={t!r}")
    return out
