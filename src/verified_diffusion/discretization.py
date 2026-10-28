"""Cell-centred finite-volume discretization.

Integrating the equation over cell ``i`` gives the semi-discrete system

    M dT/dt = -A T + b(t) + s(t)

* ``M = diag(rho_c_i V_i)`` is the heat capacity of each cell,
* ``A`` is the conductance matrix: an interior face between cells ``i`` and
  ``j`` carries the heat flow ``G_ij (T_j - T_i)`` with
  ``G_ij = area / (d_i / k_i + d_j / k_j)`` (two thermal resistances in
  series, which is the harmonic mean of ``k`` weighted by the centre-to-face
  distances ``d_i`` and ``d_j``),
* ``b(t)`` carries the boundary data and ``s(t) = V q(x_i, t)`` the source.

A boundary face is closed by a two-point flux across the half cell between the
first cell centre and the face (the same as a mirror ghost cell with linear
extrapolation):

* Dirichlet: ``Q = G_b (T_b - T_i)`` with ``G_b = area k_i / d_i``,
* Robin:     ``Q = G_b (T_inf - T_i)`` with ``G_b = area / (d_i / k_i + 1 / h)``,
* Neumann:   ``Q = area q_in`` (``G_b = 0``).

Every face flow appears once with each sign in the two cells it separates, so
the scheme is conservative: the column sums of ``A`` are the boundary
conductances, and summing the cell equations leaves only boundary and source
terms. That is the discrete energy balance the solvers report.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

from .boundary import BoundaryCondition, Dirichlet, Neumann, Robin, boundary_data, evaluate_data
from .problem import (
    FaceAverage,
    Problem,
    Problem1D,
    Problem2D,
    Source,
    cell_field,
    positive_cell_field,
)
from .tridiagonal import TridiagonalLU, tridiagonal_matvec

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]
Solve = Callable[[FloatArray], FloatArray]


class ConductanceMatrix(Protocol):
    """The operations the solvers need from the conductance matrix ``A``."""

    def matvec(self, x: FloatArray) -> FloatArray:
        """``A @ x``."""
        ...

    def factorize(self, mass: FloatArray | None, scale: float) -> Solve:
        """Factorize ``diag(mass) + scale * A`` (or ``A`` if ``mass`` is None) once."""
        ...

    def diagonal(self) -> FloatArray: ...

    def abs_row_sums(self) -> FloatArray:
        """``sum_j |A_ij|`` for each row."""
        ...

    def to_sparse(self) -> sp.csr_matrix: ...


class TridiagonalConductance:
    """Conductance matrix of a 1D problem, stored as three bands and solved by Thomas."""

    def __init__(self, lower: FloatArray, diag: FloatArray, upper: FloatArray) -> None:
        self.lower = lower
        self.diag = diag
        self.upper = upper

    def matvec(self, x: FloatArray) -> FloatArray:
        return tridiagonal_matvec(self.lower, self.diag, self.upper, x)

    def factorize(self, mass: FloatArray | None, scale: float) -> Solve:
        if mass is None:
            lu = TridiagonalLU(self.lower, self.diag, self.upper)
        else:
            lu = TridiagonalLU(scale * self.lower, mass + scale * self.diag, scale * self.upper)
        return lu.solve

    def diagonal(self) -> FloatArray:
        return self.diag.copy()

    def abs_row_sums(self) -> FloatArray:
        out = np.abs(self.diag)
        out[:-1] += np.abs(self.upper)
        out[1:] += np.abs(self.lower)
        return out

    def to_sparse(self) -> sp.csr_matrix:
        return sp.diags([self.lower, self.diag, self.upper], [-1, 0, 1], format="csr")


class SparseConductance:
    """Conductance matrix of a 2D problem, factorized by SuperLU (``scipy.sparse.linalg.splu``)."""

    def __init__(self, matrix: sp.csr_matrix) -> None:
        self.matrix = matrix

    def matvec(self, x: FloatArray) -> FloatArray:
        return np.asarray(self.matrix @ x, dtype=np.float64)

    def factorize(self, mass: FloatArray | None, scale: float) -> Solve:
        system = self.matrix if mass is None else sp.diags(mass) + scale * self.matrix
        lu = spla.splu(sp.csc_matrix(system))

        def solve(rhs: FloatArray) -> FloatArray:
            return np.asarray(lu.solve(rhs), dtype=np.float64)

        return solve

    def diagonal(self) -> FloatArray:
        return np.asarray(self.matrix.diagonal(), dtype=np.float64)

    def abs_row_sums(self) -> FloatArray:
        return np.asarray(abs(self.matrix).sum(axis=1), dtype=np.float64).ravel()

    def to_sparse(self) -> sp.csr_matrix:
        return self.matrix.copy()


@dataclass(frozen=True)
class BoundarySide:
    """The faces of one side of the domain and the condition applied there.

    ``cells`` are the flat indices of the cells next to each face, ``area``
    the face areas, ``resistance`` the thermal resistance of the half cell
    between each cell centre and its face (``d / (k area)``), and
    ``conductance`` the boundary conductances ``G_b`` (zero for Neumann).
    ``coordinate`` holds the face-centre coordinates along the side in 2D and
    is ``None`` in 1D.
    """

    name: str
    condition: BoundaryCondition
    cells: IntArray
    area: FloatArray
    resistance: FloatArray
    conductance: FloatArray
    coordinate: FloatArray | None

    def data(self, t: float) -> FloatArray:
        """The boundary datum on each face at time ``t``."""
        return evaluate_data(boundary_data(self.condition), self.coordinate, t, self.cells.size)

    def drive(self, t: float) -> FloatArray:
        """Contribution of each face to ``b(t)``: ``G_b T_ref`` or ``area q_in``."""
        if isinstance(self.condition, Neumann):
            return self.area * self.data(t)
        return self.conductance * self.data(t)

    def face_heat_flow(self, T: FloatArray, t: float) -> FloatArray:
        """Heat flow entering the domain through each face (W, or W/m in 2D)."""
        return self.drive(t) - self.conductance * T[self.cells]

    def surface_temperature(self, T: FloatArray, t: float) -> FloatArray:
        """Temperature on each face implied by the two-point boundary flux."""
        if isinstance(self.condition, Dirichlet):
            return self.data(t)
        return T[self.cells] + self.face_heat_flow(T, t) * self.resistance


def _boundary_side(
    name: str,
    condition: BoundaryCondition,
    cells: IntArray,
    area: FloatArray,
    distance: FloatArray,
    k: FloatArray,
    coordinate: FloatArray | None,
) -> BoundarySide:
    resistance = distance / (k * area)
    if isinstance(condition, Dirichlet):
        conductance = 1.0 / resistance
    elif isinstance(condition, Robin):
        conductance = 1.0 / (resistance + 1.0 / (condition.h * area))
    else:
        conductance = np.zeros_like(area)
    return BoundarySide(name, condition, cells, area, resistance, conductance, coordinate)


@dataclass(frozen=True)
class FiniteVolumeSystem:
    """The assembled semi-discrete system ``M dT/dt = -A T + b(t) + s(t)``.

    All cell vectors are flat; :attr:`shape` is the shape of the cell field
    (``(n,)`` in 1D, ``(nx, ny)`` in 2D, flattened in C order).
    """

    shape: tuple[int, ...]
    centres: tuple[FloatArray, ...]
    volume: FloatArray
    mass: FloatArray
    conductivity: FloatArray
    matrix: ConductanceMatrix
    sides: tuple[BoundarySide, ...]
    source_spec: Source

    @property
    def size(self) -> int:
        return int(self.volume.size)

    @property
    def side_names(self) -> tuple[str, ...]:
        return tuple(side.name for side in self.sides)

    def has_fixed_level(self) -> bool:
        """True if some side ties the temperature level (Dirichlet or Robin)."""
        return any(not isinstance(side.condition, Neumann) for side in self.sides)

    def source_rate(self, t: float) -> FloatArray:
        """``s(t)``: heat released per unit time in each cell, ``V q``."""
        spec = self.source_spec
        value = spec(*self.centres, t) if callable(spec) else spec
        q = cell_field(value, (), self.shape, "source").ravel()
        return self.volume * q

    def boundary_rate(self, t: float) -> FloatArray:
        """``b(t)``: the boundary data gathered into the cells next to each face."""
        out = np.zeros(self.size)
        for side in self.sides:
            np.add.at(out, side.cells, side.drive(t))
        return out

    def rhs(self, t: float) -> FloatArray:
        """``b(t) + s(t)``."""
        return self.boundary_rate(t) + self.source_rate(t)

    def boundary_heat_flow(self, T: FloatArray, t: float) -> dict[str, float]:
        """Heat flow entering through each side at time ``t`` for the flat field ``T``."""
        return {side.name: float(side.face_heat_flow(T, t).sum()) for side in self.sides}

    def time_derivative(self, t: float, T: FloatArray) -> FloatArray:
        """Right-hand side of the semi-discrete ODE, ``dT/dt = M^-1 (b + s - A T)``.

        This is the method-of-lines form; a general ODE integrator applied to
        it gives the time-exact solution of the spatial discretization.
        """
        return (self.rhs(t) - self.matrix.matvec(T)) / self.mass

    def stability_bound(self) -> float:
        """Gershgorin upper bound on the eigenvalues of ``M^-1 A``.

        ``max_i (A_ii + sum_j |A_ij|) / M_i``. ``A`` is symmetric and ``M``
        diagonal and positive, so ``M^-1 A`` has real non-negative eigenvalues,
        all below this bound.
        """
        return float(np.max(self.matrix.abs_row_sums() / self.mass))


def assemble(problem: Problem) -> FiniteVolumeSystem:
    """Build the finite-volume system of a 1D or 2D problem."""
    if isinstance(problem, Problem1D):
        return _assemble_1d(problem)
    if isinstance(problem, Problem2D):
        return _assemble_2d(problem)
    raise TypeError(f"expected Problem1D or Problem2D, got {type(problem).__name__}")


def _interior_conductance(
    area: FloatArray,
    d_left: FloatArray,
    d_right: FloatArray,
    k_left: FloatArray,
    k_right: FloatArray,
    face_average: FaceAverage,
) -> FloatArray:
    if face_average == "harmonic":
        return area / (d_left / k_left + d_right / k_right)
    k_face = (d_right * k_left + d_left * k_right) / (d_left + d_right)
    return area * k_face / (d_left + d_right)


def _assemble_1d(problem: Problem1D) -> FiniteVolumeSystem:
    grid = problem.grid
    centres = problem.centres()
    shape = problem.shape
    k = positive_cell_field(problem.conductivity, centres, shape, "conductivity")
    rho_c = positive_cell_field(problem.heat_capacity, centres, shape, "heat_capacity")
    w = grid.widths
    half = 0.5 * w
    one = np.ones(1)

    G = _interior_conductance(
        np.ones(grid.n - 1), half[:-1], half[1:], k[:-1], k[1:], problem.face_average
    )
    sides = (
        _boundary_side("left", problem.left, np.array([0]), one, half[:1], k[:1], None),
        _boundary_side(
            "right", problem.right, np.array([grid.n - 1]), one, half[-1:], k[-1:], None
        ),
    )
    diag = np.zeros(grid.n)
    diag[:-1] += G
    diag[1:] += G
    for side in sides:
        np.add.at(diag, side.cells, side.conductance)
    matrix = TridiagonalConductance(-G, diag, -G.copy())
    return FiniteVolumeSystem(
        shape=shape,
        centres=centres,
        volume=w.copy(),
        mass=rho_c * w,
        conductivity=k,
        matrix=matrix,
        sides=sides,
        source_spec=problem.source,
    )


def _assemble_2d(problem: Problem2D) -> FiniteVolumeSystem:
    gx, gy = problem.grid.x, problem.grid.y
    nx, ny = problem.shape
    centres = problem.centres()
    shape = problem.shape
    k = positive_cell_field(problem.conductivity, centres, shape, "conductivity")
    rho_c = positive_cell_field(problem.heat_capacity, centres, shape, "heat_capacity")
    hx = 0.5 * gx.widths
    hy = 0.5 * gy.widths
    index = np.arange(nx * ny).reshape(nx, ny)

    # Faces normal to x, between (i, j) and (i + 1, j); their area is the cell height.
    Gx = _interior_conductance(
        np.broadcast_to(gy.widths, (nx - 1, ny)),
        np.broadcast_to(hx[:-1, None], (nx - 1, ny)),
        np.broadcast_to(hx[1:, None], (nx - 1, ny)),
        k[:-1, :],
        k[1:, :],
        problem.face_average,
    )
    # Faces normal to y, between (i, j) and (i, j + 1); their area is the cell width.
    Gy = _interior_conductance(
        np.broadcast_to(gx.widths[:, None], (nx, ny - 1)),
        np.broadcast_to(hy[:-1], (nx, ny - 1)),
        np.broadcast_to(hy[1:], (nx, ny - 1)),
        k[:, :-1],
        k[:, 1:],
        problem.face_average,
    )

    sides = (
        _boundary_side(
            "left",
            problem.left,
            index[0, :],
            gy.widths.copy(),
            np.full(ny, hx[0]),
            k[0, :],
            gy.centres,
        ),
        _boundary_side(
            "right",
            problem.right,
            index[-1, :],
            gy.widths.copy(),
            np.full(ny, hx[-1]),
            k[-1, :],
            gy.centres,
        ),
        _boundary_side(
            "bottom",
            problem.bottom,
            index[:, 0],
            gx.widths.copy(),
            np.full(nx, hy[0]),
            k[:, 0],
            gx.centres,
        ),
        _boundary_side(
            "top",
            problem.top,
            index[:, -1],
            gx.widths.copy(),
            np.full(nx, hy[-1]),
            k[:, -1],
            gx.centres,
        ),
    )

    diag = np.zeros(nx * ny)
    np.add.at(diag, index[:-1, :].ravel(), Gx.ravel())
    np.add.at(diag, index[1:, :].ravel(), Gx.ravel())
    np.add.at(diag, index[:, :-1].ravel(), Gy.ravel())
    np.add.at(diag, index[:, 1:].ravel(), Gy.ravel())
    for side in sides:
        np.add.at(diag, side.cells, side.conductance)

    rows = np.concatenate(
        [
            index.ravel(),
            index[:-1, :].ravel(),
            index[1:, :].ravel(),
            index[:, :-1].ravel(),
            index[:, 1:].ravel(),
        ]
    )
    cols = np.concatenate(
        [
            index.ravel(),
            index[1:, :].ravel(),
            index[:-1, :].ravel(),
            index[:, 1:].ravel(),
            index[:, :-1].ravel(),
        ]
    )
    data = np.concatenate([diag, -Gx.ravel(), -Gx.ravel(), -Gy.ravel(), -Gy.ravel()])
    matrix = sp.csr_matrix((data, (rows, cols)), shape=(nx * ny, nx * ny))
    volume = problem.grid.volumes().ravel()
    return FiniteVolumeSystem(
        shape=shape,
        centres=centres,
        volume=volume,
        mass=rho_c.ravel() * volume,
        conductivity=k,
        matrix=SparseConductance(matrix),
        sides=sides,
        source_spec=problem.source,
    )
