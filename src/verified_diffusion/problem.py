"""Problem definitions: geometry, material properties, source and boundary conditions.

The equation is

    rho_c * dT/dt = div(k grad T) + q(x, t)

with ``rho_c`` the volumetric heat capacity, ``k`` the conductivity and ``q``
the volumetric heat source. Material properties may vary in space but not in
time or with temperature, so the discrete operator is linear and constant.

Coefficients (``conductivity`` and ``heat_capacity``) are a constant, an
array of cell values, or a function of the cell-centre coordinates (``f(x)``
in 1D, ``f(X, Y)`` in 2D). The source is a constant, an array of cell values,
or a function of the coordinates and time (``q(x, t)`` or ``q(X, Y, t)``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .boundary import BoundaryCondition, Dirichlet, Neumann, Robin
from .grid import Grid1D, Grid2D

FloatArray = NDArray[np.float64]

Coefficient = float | ArrayLike | Callable[..., ArrayLike]
Source = float | ArrayLike | Callable[..., ArrayLike]
FaceAverage = Literal["harmonic", "arithmetic"]

SIDES_1D = ("left", "right")
SIDES_2D = ("left", "right", "bottom", "top")


def _check_condition(name: str, condition: object) -> None:
    if not isinstance(condition, Dirichlet | Neumann | Robin):
        raise TypeError(f"{name} must be Dirichlet, Neumann or Robin, got {condition!r}")


def _check_average(face_average: str) -> None:
    if face_average not in ("harmonic", "arithmetic"):
        raise ValueError(f"face_average must be 'harmonic' or 'arithmetic', not {face_average!r}")


@dataclass(frozen=True)
class Problem1D:
    """Heat conduction on ``[grid.start, grid.stop]`` (per unit cross-section area).

    ``face_average`` selects how the conductivity is averaged at a face between
    two cells. "harmonic" (the default) is the series-resistance formula, exact
    for a piecewise-constant conductivity whose jumps sit on faces.
    "arithmetic" (distance-weighted linear interpolation) exists to compare
    against it.
    """

    grid: Grid1D
    left: BoundaryCondition
    right: BoundaryCondition
    conductivity: Coefficient = 1.0
    heat_capacity: Coefficient = 1.0
    source: Source = 0.0
    face_average: FaceAverage = "harmonic"

    def __post_init__(self) -> None:
        for side in SIDES_1D:
            _check_condition(side, getattr(self, side))
        _check_average(self.face_average)

    @property
    def shape(self) -> tuple[int]:
        return (self.grid.n,)

    def conditions(self) -> dict[str, BoundaryCondition]:
        return {"left": self.left, "right": self.right}

    def centres(self) -> tuple[FloatArray]:
        return (self.grid.centres,)


@dataclass(frozen=True)
class Problem2D:
    """Heat conduction on a rectangle (per unit depth).

    Sides are ``left`` (x = x.start), ``right`` (x = x.stop), ``bottom``
    (y = y.start) and ``top`` (y = y.stop). See :class:`Problem1D` for
    ``face_average``.
    """

    grid: Grid2D
    left: BoundaryCondition
    right: BoundaryCondition
    bottom: BoundaryCondition
    top: BoundaryCondition
    conductivity: Coefficient = 1.0
    heat_capacity: Coefficient = 1.0
    source: Source = 0.0
    face_average: FaceAverage = "harmonic"

    def __post_init__(self) -> None:
        for side in SIDES_2D:
            _check_condition(side, getattr(self, side))
        _check_average(self.face_average)

    @property
    def shape(self) -> tuple[int, int]:
        return self.grid.shape

    def conditions(self) -> dict[str, BoundaryCondition]:
        return {"left": self.left, "right": self.right, "bottom": self.bottom, "top": self.top}

    def centres(self) -> tuple[FloatArray, FloatArray]:
        return self.grid.centres()


Problem = Problem1D | Problem2D


def cell_field(
    spec: Coefficient, centres: tuple[FloatArray, ...], shape: tuple[int, ...], name: str
) -> FloatArray:
    """Evaluate a coefficient on the cell centres as an array of ``shape``."""
    value = spec(*centres) if callable(spec) else spec
    try:
        out = np.broadcast_to(np.asarray(value, dtype=np.float64), shape).copy()
    except ValueError as exc:
        raise ValueError(f"{name} cannot be broadcast to the cell shape {shape}") from exc
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be finite")
    return out


def positive_cell_field(
    spec: Coefficient, centres: tuple[FloatArray, ...], shape: tuple[int, ...], name: str
) -> FloatArray:
    out = cell_field(spec, centres, shape, name)
    if not np.all(out > 0.0):
        raise ValueError(f"{name} must be positive in every cell")
    return out
