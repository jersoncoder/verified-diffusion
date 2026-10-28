"""The verification studies, shared by the test suite and the report script.

Each function runs one study and returns its errors on every level, so the
numbers in ``docs/verification.md`` are the numbers the tests assert on.
Requires sympy (the development extra) for the manufactured solutions.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import sympy
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from . import analytic
from .boundary import Dirichlet, Neumann, Robin
from .discretization import FiniteVolumeSystem
from .gci import GridConvergence, grid_convergence_index
from .grid import Grid1D, Grid2D
from .mms import Kind, ManufacturedSolution, t, x, y
from .problem import Problem1D, Problem2D
from .solver import TransientSolver, solve_steady
from .verification import ConvergenceStudy, error_norms

FloatArray = NDArray[np.float64]

ORDER_TOLERANCE = 0.1
"""The observed order between the two finest levels must lie this close to the theory."""

STRETCHED_2D_MAX_NORM_FLOOR = 1.75
"""Lower bound for the max-norm order on the stretched 2D grid (see docs/verification.md)."""

# Manufactured solutions ------------------------------------------------------

STEADY_1D = ManufacturedSolution(sympy.exp(x) * sympy.sin(2 * sympy.pi * x) + 1)
"""Constant conductivity."""

STEADY_1D_VARIABLE_K = ManufacturedSolution(
    sympy.cos(3 * x) + x**3, conductivity=1 + sympy.sin(2 * sympy.pi * x) / 2
)
"""Conductivity varying by a factor of three across the domain."""

STEADY_2D = ManufacturedSolution(
    sympy.exp(x * y) * sympy.cos(x + 2 * y), conductivity=2 + sympy.sin(x) * sympy.cos(y), dim=2
)

TRANSIENT_1D = ManufacturedSolution(
    sympy.exp(-t) * sympy.sin(sympy.pi * x + sympy.Rational(1, 2)) + x**2 * sympy.cos(2 * t) / 2,
    conductivity=1 + x**2 / 2,
    heat_capacity=1 + sympy.sin(sympy.pi * x) / 4,
)
"""Time-dependent source and boundary data, variable conductivity and heat capacity."""

TRANSIENT_2D = ManufacturedSolution(
    sympy.cos(sympy.pi * x / 2 + t) * sympy.exp(y - x * y),
    conductivity=1 + x**2 * y / 2,
    heat_capacity=1 + x * y / 4,
    dim=2,
)

ROBIN_H = 4.0
"""Heat transfer coefficient used by every manufactured Robin condition."""


# Spatial order ---------------------------------------------------------------


def _stretched_2d(n: int) -> Grid2D:
    """Unit square, clustered at both ends in x and at the bottom in y."""
    return Grid2D(
        Grid1D.stretched(0.0, 1.0, n, beta=1.5, cluster="both"),
        Grid1D.stretched(0.0, 1.0, n, beta=1.5, cluster="left"),
    )


MIXED_2D: dict[str, Kind] = {
    "left": "dirichlet",
    "right": "robin",
    "bottom": "neumann",
    "top": "dirichlet",
}


def steady_space_1d_uniform(levels: Sequence[int] = (8, 16, 32, 64, 128)) -> ConvergenceStudy:
    """Steady MMS, uniform grid, constant k, Dirichlet on both sides."""
    exact = STEADY_1D.exact()
    l2, mx = [], []
    for n in levels:
        grid = Grid1D.uniform(0.0, 1.0, n)
        T = solve_steady(STEADY_1D.problem_1d(grid, "dirichlet", "dirichlet")).temperature
        e = error_norms(T, exact(grid.centres, 0.0), grid.widths)
        l2.append(e[0])
        mx.append(e[1])
    return _study(
        "steady-1d-uniform",
        "1D steady, uniform grid, constant k, Dirichlet/Dirichlet",
        "h",
        levels,
        l2,
        mx,
        2.0,
    )


def steady_space_1d_stretched(levels: Sequence[int] = (8, 16, 32, 64, 128)) -> ConvergenceStudy:
    """Steady MMS, tanh-stretched grid (beta=2), variable k, Robin and Neumann sides."""
    exact = STEADY_1D_VARIABLE_K.exact()
    l2, mx = [], []
    for n in levels:
        grid = Grid1D.stretched(0.0, 1.0, n, beta=2.0)
        problem = STEADY_1D_VARIABLE_K.problem_1d(grid, "robin", "neumann", ROBIN_H)
        e = error_norms(solve_steady(problem).temperature, exact(grid.centres, 0.0), grid.widths)
        l2.append(e[0])
        mx.append(e[1])
    return _study(
        "steady-1d-stretched",
        "1D steady, stretched grid, variable k, Robin/Neumann",
        "h",
        levels,
        l2,
        mx,
        2.0,
    )


def steady_space_2d(levels: Sequence[int] = (8, 16, 32, 64, 128, 256)) -> ConvergenceStudy:
    """Steady MMS in 2D, stretched in both directions, variable k, all three kinds of side."""
    exact = STEADY_2D.exact()
    l2, mx = [], []
    for n in levels:
        grid = _stretched_2d(n)
        problem = STEADY_2D.problem_2d(grid, MIXED_2D, ROBIN_H)
        X, Y = grid.centres()
        T, T_exact = solve_steady(problem).temperature, exact(X, Y, 0.0)
        e = error_norms(T, T_exact, grid.volumes())
        l2.append(e[0])
        mx.append(e[1])
    return _study(
        "steady-2d-stretched",
        "2D steady, stretched grid, variable k, Dirichlet/Robin/Neumann/Dirichlet",
        "h",
        levels,
        l2,
        mx,
        2.0,
        worst_cell=_worst_cell(T - T_exact, (X, Y)),
    )


def transient_space_1d(
    levels: Sequence[int] = (8, 16, 32, 64, 128), t_end: float = 0.5
) -> ConvergenceStudy:
    """Crank-Nicolson with ``dt = h / 4``, stretched grid, time-dependent Dirichlet and Robin data.

    Both errors are second order, so the total error falls as ``h^2``.
    """
    exact = TRANSIENT_1D.exact()
    l2, mx = [], []
    for n in levels:
        grid = Grid1D.stretched(0.0, 1.0, n, beta=2.0)
        problem = TRANSIENT_1D.problem_1d(grid, "dirichlet", "robin", ROBIN_H)
        run = TransientSolver(problem, "crank-nicolson").run(
            exact(grid.centres, 0.0), t_end, 0.25 / n
        )
        e = error_norms(run.temperature, exact(grid.centres, t_end), grid.widths)
        l2.append(e[0])
        mx.append(e[1])
    return _study(
        "transient-1d-cn",
        "1D Crank-Nicolson, dt = h/4, stretched grid, variable k and rho*c, Dirichlet/Robin(t)",
        "h",
        levels,
        l2,
        mx,
        2.0,
    )


def transient_space_2d(
    levels: Sequence[int] = (8, 16, 32, 64, 128), t_end: float = 0.25
) -> ConvergenceStudy:
    """2D Crank-Nicolson with ``dt = h / 4`` and time-dependent data on all four sides."""
    exact = TRANSIENT_2D.exact()
    l2, mx = [], []
    for n in levels:
        grid = _stretched_2d(n)
        problem = TRANSIENT_2D.problem_2d(grid, MIXED_2D, ROBIN_H)
        X, Y = grid.centres()
        run = TransientSolver(problem, "crank-nicolson").run(exact(X, Y, 0.0), t_end, 0.25 / n)
        T, T_exact = run.temperature, exact(X, Y, t_end)
        e = error_norms(T, T_exact, grid.volumes())
        l2.append(e[0])
        mx.append(e[1])
    return _study(
        "transient-2d-cn",
        "2D Crank-Nicolson, dt = h/4, stretched grid, variable k and rho*c, mixed sides (t)",
        "h",
        levels,
        l2,
        mx,
        2.0,
        worst_cell=_worst_cell(T - T_exact, (X, Y)),
    )


# Temporal order --------------------------------------------------------------

TIME_STUDY_CELLS = 16
TIME_STUDY_END = 0.5


def semi_discrete_reference(
    system: FiniteVolumeSystem, T0: FloatArray, t_end: float, rtol: float = 1e-12
) -> FloatArray:
    """Time-exact solution of the semi-discrete system on the same grid.

    Integrates ``dT/dt = M^-1 (b + s - A T)`` with scipy's Radau (implicit
    Runge-Kutta, order 5) at a tight tolerance. Comparing a time-stepping run
    against it isolates the time error: the spatial error is common to both.
    """
    jacobian = -(system.matrix.to_sparse().toarray() / system.mass[:, None])
    result = solve_ivp(
        system.time_derivative,
        (0.0, t_end),
        T0,
        method="Radau",
        rtol=rtol,
        atol=rtol * 1e-2,
        jac=jacobian,
    )
    if not result.success:
        raise RuntimeError(f"reference integration failed: {result.message}")
    return np.asarray(result.y[:, -1], dtype=np.float64)


def _time_problem() -> Problem1D:
    grid = Grid1D.uniform(0.0, 1.0, TIME_STUDY_CELLS)
    return TRANSIENT_1D.problem_1d(grid, "neumann", "robin", ROBIN_H)


def time_order(scheme: str, refinements: int = 5) -> ConvergenceStudy:
    """Temporal order of a scheme on a fixed 16-cell grid, against the semi-discrete reference.

    Implicit schemes start at ``dt = t_end / 10``; explicit Euler starts at the
    largest equal step within its stability limit. Each level halves ``dt``.
    """
    problem = _time_problem()
    solver = TransientSolver(problem, scheme)
    grid = problem.grid
    T0 = TRANSIENT_1D.exact()(grid.centres, 0.0)
    reference = semi_discrete_reference(solver.system, T0, TIME_STUDY_END)
    # Explicit Euler starts at the largest equal step within its stability limit.
    first = math.ceil(TIME_STUDY_END / solver.max_stable_dt) if solver.theta < 0.5 else 10
    steps = [first * 2**k for k in range(refinements)]
    l2, mx = [], []
    for n in steps:
        run = solver.run(T0, TIME_STUDY_END, TIME_STUDY_END / n)
        e = error_norms(run.temperature, reference, grid.widths)
        l2.append(e[0])
        mx.append(e[1])
    expected = 2.0 if solver.theta == 0.5 else 1.0
    return ConvergenceStudy(
        name=f"time-{scheme}",
        description=(
            f"{scheme}, 1D, {TIME_STUDY_CELLS} cells, variable k and rho*c, Neumann(t)/Robin(t), "
            "error against the semi-discrete reference"
        ),
        parameter="dt",
        levels=tuple(steps),
        sizes=tuple(TIME_STUDY_END / n for n in steps),
        errors_l2=tuple(l2),
        errors_max=tuple(mx),
        expected_order=expected,
    )


# Analytic solutions ----------------------------------------------------------

ERFC_ALPHA = 1.0
ERFC_END = 0.05
ERFC_LENGTH = 2.0  # erfc(L / (2 sqrt(alpha t))) is below 1e-9 at t = ERFC_END


def erfc_study(
    scheme: str = "crank-nicolson",
    startup_steps: int = 2,
    levels: Sequence[int] = (20, 40, 80, 160, 320),
) -> ConvergenceStudy:
    """Semi-infinite solid at 0 whose surface is raised to 1 at ``t = 0``.

    A finite domain ``[0, 2]`` with the far side held at the initial
    temperature stands in for the half-space. The grid is stretched towards
    the heated surface, and the time step is proportional to the cell size,
    with 10 steps on the coarsest grid.
    """
    l2, mx = [], []
    for n in levels:
        grid = Grid1D.stretched(0.0, ERFC_LENGTH, n, beta=3.0)
        problem = Problem1D(
            grid, Dirichlet(1.0), Dirichlet(0.0), conductivity=ERFC_ALPHA, heat_capacity=1.0
        )
        run = TransientSolver(problem, scheme).run(
            0.0, ERFC_END, ERFC_END * levels[0] / (10 * n), startup_steps=startup_steps
        )
        exact = analytic.semi_infinite_step(
            grid.centres, ERFC_END, alpha=ERFC_ALPHA, T_initial=0.0, T_surface=1.0
        )
        e = error_norms(run.temperature, exact, grid.widths)
        l2.append(e[0])
        mx.append(e[1])
    expected = 2.0 if scheme == "crank-nicolson" else 1.0
    startup = f", {startup_steps} start-up steps" if startup_steps else ", no start-up steps"
    return _study(
        f"erfc-{scheme}-{startup_steps}",
        f"semi-infinite solid, step surface temperature, {scheme}{startup}, dt proportional to h",
        "h",
        levels,
        l2,
        mx,
        expected,
        length=ERFC_LENGTH,
    )


SLAB_BIOT = 5.0
SLAB_FOURIER = 0.3


def slab_study(
    startup_steps: int = 2, levels: Sequence[int] = (10, 20, 40, 80, 160)
) -> ConvergenceStudy:
    """Plane wall at 1, insulated at x = 0, convection to 0 at x = 1 (Bi = 5), at Fo = 0.3.

    Crank-Nicolson with ``dt = Fo / n``; compared with the Fourier series over
    the roots of ``lambda tan(lambda) = Bi``.
    """
    l2, mx = [], []
    for n in levels:
        grid = Grid1D.uniform(0.0, 1.0, n)
        problem = Problem1D(grid, Neumann(0.0), Robin(SLAB_BIOT, 0.0))
        run = TransientSolver(problem, "crank-nicolson").run(
            1.0, SLAB_FOURIER, SLAB_FOURIER / n, startup_steps=startup_steps
        )
        exact = analytic.slab_convection(
            grid.centres,
            SLAB_FOURIER,
            length=1.0,
            alpha=1.0,
            conductivity=1.0,
            h=SLAB_BIOT,
            T_initial=1.0,
            T_ambient=0.0,
        )
        e = error_norms(run.temperature, exact, grid.widths)
        l2.append(e[0])
        mx.append(e[1])
    startup = f", {startup_steps} start-up steps" if startup_steps else ", no start-up steps"
    return _study(
        f"slab-{startup_steps}",
        f"slab with convection (Bi = 5), Fourier series at Fo = 0.3, Crank-Nicolson{startup}",
        "h",
        levels,
        l2,
        mx,
        2.0,
    )


PLATE_TERMS = 1001


def plate_study(levels: Sequence[int] = (8, 16, 32, 64, 128)) -> ConvergenceStudy:
    """Unit square at 0 on three sides, ``4 x (1 - x)`` on the top; sine-series solution."""
    coefficients = analytic.parabola_sine_coefficients(1.0, PLATE_TERMS)
    l2, mx = [], []
    for n in levels:
        grid = Grid2D.uniform((0.0, 1.0), (0.0, 1.0), n, n)
        zero = Dirichlet(0.0)
        problem = Problem2D(
            grid,
            left=zero,
            right=zero,
            bottom=zero,
            top=Dirichlet(lambda s, _t: 4.0 * s * (1.0 - s)),
        )
        X, Y = grid.centres()
        exact = analytic.rectangular_plate(
            X, Y, width=1.0, height=1.0, top_coefficients=coefficients
        )
        e = error_norms(solve_steady(problem).temperature, exact, grid.volumes())
        l2.append(e[0])
        mx.append(e[1])
    return _study(
        "plate-2d",
        "2D steady plate, parabolic top edge, Fourier sine series",
        "h",
        levels,
        l2,
        mx,
        2.0,
    )


@dataclass(frozen=True)
class CompositeWallResult:
    """Maximum error at the cell centres of a two-layer wall, per face-averaging rule."""

    cells_per_layer: tuple[int, ...]
    harmonic: tuple[float, ...]
    arithmetic: tuple[float, ...]
    harmonic_stretched: tuple[float, ...]


COMPOSITE = {"thicknesses": (0.3, 0.7), "conductivities": (1.0, 20.0)}


def composite_wall_study(levels: Sequence[int] = (4, 8, 16, 32, 64)) -> CompositeWallResult:
    """Two layers (k = 1 and 20) between 0 and 1, Dirichlet 0 and 1; interface on a face."""
    L1, L2 = COMPOSITE["thicknesses"]
    k1, k2 = COMPOSITE["conductivities"]

    def conductivity(xc: FloatArray) -> FloatArray:
        return np.where(xc < L1, k1, k2)

    def layered(n: int, stretched: bool) -> Grid1D:
        if stretched:
            a = Grid1D.stretched(0.0, L1, n, beta=2.0, cluster="right").faces
            b = Grid1D.stretched(L1, L1 + L2, n, beta=2.0, cluster="left").faces
        else:
            a, b = np.linspace(0.0, L1, n + 1), np.linspace(L1, L1 + L2, n + 1)
        return Grid1D(np.concatenate([a, b[1:]]))

    results: dict[tuple[bool, str], list[float]] = {}
    for stretched, average in ((False, "harmonic"), (False, "arithmetic"), (True, "harmonic")):
        errors = []
        for n in levels:
            grid = layered(n, stretched)
            problem = Problem1D(
                grid,
                Dirichlet(0.0),
                Dirichlet(1.0),
                conductivity=conductivity,
                face_average="harmonic" if average == "harmonic" else "arithmetic",
            )
            exact, _ = analytic.composite_wall(
                grid.centres,
                thicknesses=(L1, L2),
                conductivities=(k1, k2),
                T_left=0.0,
                T_right=1.0,
            )
            errors.append(float(np.max(np.abs(solve_steady(problem).temperature - exact))))
        results[(stretched, average)] = errors
    return CompositeWallResult(
        cells_per_layer=tuple(levels),
        harmonic=tuple(results[(False, "harmonic")]),
        arithmetic=tuple(results[(False, "arithmetic")]),
        harmonic_stretched=tuple(results[(True, "harmonic")]),
    )


# Grid convergence index -----------------------------------------------------


@dataclass(frozen=True)
class GCIStudy:
    """A quantity of interest on four nested grids and two three-grid GCI analyses."""

    name: str
    cells: tuple[int, ...]
    values: tuple[float, ...]
    fine: GridConvergence
    coarse: GridConvergence


def _heated_plate(n: int) -> Problem2D:
    """A 2 x 1 plate with a Gaussian heat source and a conductivity that varies with position.

    Left side held at 0, convection to 0 on the right (h = 10), insulated
    bottom, and a prescribed heat loss on the top that grows along x. There is
    no closed-form solution.
    """
    grid = Grid2D(Grid1D.uniform(0.0, 2.0, 2 * n), Grid1D.uniform(0.0, 1.0, n))
    return Problem2D(
        grid,
        left=Dirichlet(0.0),
        right=Robin(10.0, 0.0),
        bottom=Neumann(0.0),
        top=Neumann(lambda s, _t: -2.0 * s),
        conductivity=lambda X, Y: 1.0 + 4.0 * np.exp(-((X - 1.4) ** 2 + (Y - 0.3) ** 2) / 0.08),
        source=lambda X, Y, _t: 100.0 * np.exp(-((X - 0.6) ** 2 + (Y - 0.6) ** 2) / 0.02),
    )


def gci_study(levels: Sequence[int] = (16, 32, 64, 128)) -> list[GCIStudy]:
    """GCI of the mean temperature and of the heat leaving through the convective side."""
    if len(levels) != 4:
        raise ValueError("the GCI study uses exactly four grids")
    quantities: dict[str, list[float]] = {
        "mean temperature": [],
        "heat flow out of the right side": [],
    }
    for n in levels:
        solution = solve_steady(_heated_plate(n))
        volume = solution.system.volume
        T = solution.temperature.ravel()
        quantities["mean temperature"].append(float(volume @ T / volume.sum()))
        quantities["heat flow out of the right side"].append(-solution.balance.boundary["right"])
    ratio = levels[1] / levels[0]
    studies = []
    for name, values in quantities.items():
        f = values[::-1]  # fine first
        studies.append(
            GCIStudy(
                name=name,
                cells=tuple(2 * n * n for n in levels),
                values=tuple(values),
                fine=grid_convergence_index(f[0], f[1], f[2], ratio),
                coarse=grid_convergence_index(f[1], f[2], f[3], ratio),
            )
        )
    return studies


# Energy balance --------------------------------------------------------------


@dataclass(frozen=True)
class BalanceRow:
    case: str
    scheme: str
    steps: int
    max_relative_residual: float
    stored: float
    boundary: float
    source: float


def _busy_1d() -> Problem1D:
    return Problem1D(
        Grid1D.stretched(0.0, 1.0, 40, beta=2.0),
        left=Robin(8.0, lambda tt: 300.0 + 20.0 * math.sin(3.0 * tt)),
        right=Neumann(lambda tt: -500.0 * tt),
        conductivity=lambda xc: 1.0 + 3.0 * xc,
        heat_capacity=lambda xc: 2.0 - xc,
        source=lambda xc, tt: 1e3 * np.exp(-5.0 * xc) * np.cos(2.0 * tt),
    )


def _busy_2d() -> Problem2D:
    grid = Grid2D(
        Grid1D.stretched(0.0, 1.0, 24, beta=1.5, cluster="both"), Grid1D.uniform(0.0, 0.5, 12)
    )
    return Problem2D(
        grid,
        left=Dirichlet(lambda s, tt: 300.0 + 50.0 * s * tt),
        right=Robin(3.0, 280.0),
        bottom=Neumann(lambda s, _tt: 100.0 * np.sin(math.pi * s)),
        top=Dirichlet(300.0),
        conductivity=lambda X, Y: 1.0 + X + 4.0 * Y,
        heat_capacity=1.5,
        source=lambda X, Y, tt: 200.0 * np.exp(-((X - 0.5) ** 2) / 0.02) * (1.0 + tt),
    )


BALANCE_CASES: dict[str, Callable[[], Problem1D | Problem2D]] = {
    "1D: Robin(t), Neumann(t), source(x, t), variable k and rho*c, stretched": _busy_1d,
    "2D: Dirichlet(y, t), Robin, Neumann(x), Dirichlet, source(x, y, t), variable k": _busy_2d,
}


def energy_balance_study(t_end: float = 0.1) -> list[BalanceRow]:
    """Run every balance case with every scheme and record the worst step residual."""
    rows = []
    for case, make in BALANCE_CASES.items():
        for scheme in ("explicit-euler", "crank-nicolson", "implicit-euler"):
            solver = TransientSolver(make(), scheme)
            dt = 0.9 * solver.max_stable_dt if solver.theta < 0.5 else 2e-3
            run = solver.run(300.0, t_end, dt)
            stored, boundary, source = run.total_balance()
            rows.append(
                BalanceRow(
                    case=case,
                    scheme=scheme,
                    steps=len(run.balances),
                    max_relative_residual=run.max_relative_residual,
                    stored=stored,
                    boundary=boundary,
                    source=source,
                )
            )
    return rows


def _study(
    name: str,
    description: str,
    parameter: str,
    levels: Sequence[int],
    l2: Sequence[float],
    mx: Sequence[float],
    expected: float,
    length: float = 1.0,
    worst_cell: tuple[float, ...] = (),
) -> ConvergenceStudy:
    return ConvergenceStudy(
        name=name,
        description=description,
        parameter=parameter,
        levels=tuple(levels),
        sizes=tuple(length / n for n in levels),
        errors_l2=tuple(l2),
        errors_max=tuple(mx),
        expected_order=expected,
        worst_cell=worst_cell,
    )


def _worst_cell(error: FloatArray, centres: tuple[FloatArray, ...]) -> tuple[float, ...]:
    """Centre coordinates of the cell where ``|error|`` is largest."""
    index = np.unravel_index(int(np.argmax(np.abs(error))), error.shape)
    return tuple(float(c[index]) for c in centres)
