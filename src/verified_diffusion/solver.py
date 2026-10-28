"""Steady and transient solvers with a per-step energy balance.

Time integration uses the theta method on ``M dT/dt = -A T + r(t)`` with
``r = b + s`` (boundary data plus source):

    (M + theta dt A) T^{n+1} = (M - (1 - theta) dt A) T^n
                               + dt (theta r^{n+1} + (1 - theta) r^n)

``theta = 0`` is explicit Euler, ``1/2`` Crank-Nicolson and ``1`` implicit
Euler. Boundary data and sources are evaluated at both ends of the step and
weighted the same way, which keeps Crank-Nicolson second order with
time-dependent data.

Summing the rows of the update gives the energy balance of the step,

    sum_i M_i (T_i^{n+1} - T_i^n) = dt (theta Q^{n+1} + (1 - theta) Q^n)
                                    + dt (theta S^{n+1} + (1 - theta) S^n)

where ``Q`` is the heat flow entering through the boundary and ``S`` the total
source. Interior face flows cancel in pairs, so the identity is exact for the
discrete scheme and its residual measures round-off alone.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .discretization import FiniteVolumeSystem, Forcing, Solve, assemble
from .problem import Problem, cell_field

FloatArray = NDArray[np.float64]

SCHEMES: Mapping[str, float] = {
    "explicit-euler": 0.0,
    "crank-nicolson": 0.5,
    "implicit-euler": 1.0,
}
"""Named schemes and their theta."""

InitialCondition = float | ArrayLike | Callable[..., ArrayLike]


class StabilityError(ValueError):
    """A time step beyond the stability limit of a conditionally stable scheme."""


def scheme_theta(scheme: str | float) -> float:
    """The theta of a named scheme, or ``scheme`` itself if it is a number in ``[0, 1]``."""
    if isinstance(scheme, str):
        try:
            return SCHEMES[scheme]
        except KeyError:
            names = ", ".join(SCHEMES)
            raise ValueError(f"unknown scheme {scheme!r}; expected one of {names}") from None
    theta = float(scheme)
    if not 0.0 <= theta <= 1.0:
        raise ValueError(f"theta must lie in [0, 1], got {theta!r}")
    return theta


def _scheme_name(theta: float) -> str:
    for name, value in SCHEMES.items():
        if value == theta:
            return name
    return f"theta={theta:g}"


@dataclass(frozen=True)
class StepBalance:
    """Energy balance of one time step (J, or J/m in 2D).

    ``stored`` is the change in stored energy, ``boundary`` the heat that
    entered through each side during the step and ``source`` the heat the
    source released. ``scale`` is the sum of the magnitudes of the terms that
    enter the step (stored energy at both ends, conductance-weighted
    temperatures and boundary and source data); round-off in the step is a
    small multiple of machine epsilon times ``scale``.
    """

    t: float
    dt: float
    stored: float
    boundary: Mapping[str, float]
    source: float
    scale: float

    @property
    def boundary_total(self) -> float:
        return float(sum(self.boundary.values()))

    @property
    def residual(self) -> float:
        """``stored - boundary_total - source``; zero up to round-off."""
        return self.stored - self.boundary_total - self.source

    @property
    def relative_residual(self) -> float:
        return abs(self.residual) / self.scale if self.scale > 0.0 else 0.0


@dataclass(frozen=True)
class SteadyBalance:
    """Steady energy balance (W, or W/m in 2D): boundary inflow plus source is zero."""

    boundary: Mapping[str, float]
    source: float
    scale: float

    @property
    def boundary_total(self) -> float:
        return float(sum(self.boundary.values()))

    @property
    def residual(self) -> float:
        return self.boundary_total + self.source

    @property
    def relative_residual(self) -> float:
        return abs(self.residual) / self.scale if self.scale > 0.0 else 0.0


@dataclass(frozen=True)
class SteadySolution:
    temperature: FloatArray
    balance: SteadyBalance
    system: FiniteVolumeSystem


@dataclass(frozen=True)
class TransientSolution:
    """Result of :meth:`TransientSolver.run`.

    ``times`` holds the start time and the end time of every step (start-up
    half steps included), matching ``balances`` one to one after the first
    entry. ``temperature`` is the field at ``times[-1]``. ``snapshots`` holds
    the fields at ``snapshot_times`` when the run was asked to save them.
    """

    times: FloatArray
    temperature: FloatArray
    balances: tuple[StepBalance, ...]
    snapshot_times: FloatArray
    snapshots: tuple[FloatArray, ...]
    system: FiniteVolumeSystem

    @property
    def max_relative_residual(self) -> float:
        return max((b.relative_residual for b in self.balances), default=0.0)

    def total_balance(self) -> tuple[float, float, float]:
        """Cumulative ``(stored, boundary, source)`` over the whole run."""
        stored = math.fsum(b.stored for b in self.balances)
        boundary = math.fsum(b.boundary_total for b in self.balances)
        source = math.fsum(b.source for b in self.balances)
        return stored, boundary, source


def solve_steady(problem: Problem, t: float = 0.0) -> SteadySolution:
    """Solve ``A T = b(t) + s(t)``; time-dependent data are evaluated at ``t``.

    Raises ``ValueError`` if every side is Neumann: the temperature level is
    then undetermined.
    """
    system = assemble(problem)
    if not system.has_fixed_level():
        raise ValueError(
            "a steady problem with Neumann conditions on every side has no unique solution"
        )
    b = system.boundary_rate(t)
    s = system.source_rate(t)
    T = system.matrix.factorize(None, 1.0)(b + s)
    scale = (
        float(system.matrix.abs_row_sums() @ np.abs(T))
        + float(np.abs(b).sum())
        + float(np.abs(s).sum())
    )
    balance = SteadyBalance(
        boundary=system.boundary_heat_flow(T, t), source=float(s.sum()), scale=scale
    )
    return SteadySolution(T.reshape(system.shape), balance, system)


class TransientSolver:
    """Theta-method time integration of a :class:`Problem1D` or :class:`Problem2D`.

    The matrix ``M + theta dt A`` is factorized once per time step size (Thomas
    in 1D, SuperLU in 2D) and reused while ``dt`` stays the same. Implicit
    Euler start-up steps (see :meth:`run`) use a second factorization.

    For ``theta < 1/2`` the scheme is stable only for ``dt <= 2 / ((1 - 2 theta)
    lambda)``, where ``lambda`` bounds the eigenvalues of ``M^-1 A``
    (:meth:`FiniteVolumeSystem.stability_bound`). The solver reports this as a
    grid Fourier number ``Fo = dt lambda / 4``, which equals ``alpha dt / dx^2``
    on a uniform 1D grid and ``alpha dt (1/dx^2 + 1/dy^2)`` on a uniform 2D grid,
    with the limit ``Fo <= 1 / (2 (1 - 2 theta))`` (``1/2`` for explicit Euler),
    and refuses a larger step with :class:`StabilityError`.
    """

    def __init__(self, problem: Problem, scheme: str | float = "crank-nicolson") -> None:
        self.problem = problem
        self.theta = scheme_theta(scheme)
        self.scheme = _scheme_name(self.theta)
        self.system = assemble(problem)
        self._lambda = self.system.stability_bound()
        self._abs_rows = self.system.matrix.abs_row_sums()
        self._factors: dict[tuple[float, float], Solve] = {}
        self._forcing_cache: tuple[Forcing, ...] = ()

    # Stability -------------------------------------------------------------

    @property
    def max_stable_dt(self) -> float:
        """Largest stable time step (infinite for ``theta >= 1/2``)."""
        if self.theta >= 0.5 or self._lambda == 0.0:
            return math.inf
        return 2.0 / ((1.0 - 2.0 * self.theta) * self._lambda)

    @property
    def fourier_limit(self) -> float:
        """Largest stable grid Fourier number (infinite for ``theta >= 1/2``)."""
        if self.theta >= 0.5:
            return math.inf
        return 0.5 / (1.0 - 2.0 * self.theta)

    def fourier_number(self, dt: float) -> float:
        """Grid Fourier number ``dt lambda / 4`` of a time step."""
        return dt * self._lambda / 4.0

    def check_stability(self, dt: float) -> None:
        if dt > self.max_stable_dt * (1.0 + 1e-12):
            raise StabilityError(
                f"{self.scheme} is unstable for dt={dt:.6g}: the grid Fourier number "
                f"Fo = {self.fourier_number(dt):.6g} exceeds the stability limit "
                f"{self.fourier_limit:.6g}; the largest stable time step is "
                f"dt_max = {self.max_stable_dt:.6g}"
            )

    # Stepping --------------------------------------------------------------

    def initial_field(self, T0: InitialCondition) -> FloatArray:
        """Evaluate an initial condition (constant, cell array or function of the centres)."""
        return cell_field(T0, self.system.centres, self.system.shape, "initial temperature")

    def step(self, T: ArrayLike, t: float, dt: float) -> tuple[FloatArray, StepBalance]:
        """Advance the field ``T`` (cell shape) from ``t`` to ``t + dt``."""
        field = np.asarray(T, dtype=np.float64)
        if field.shape != self.system.shape:
            raise ValueError(f"T must have shape {self.system.shape}, got {field.shape}")
        new, balance = self._step(field.ravel(), t, dt, self.theta)
        return new.reshape(self.system.shape), balance

    def run(
        self,
        T0: InitialCondition,
        t_end: float,
        dt: float,
        t_start: float = 0.0,
        save_every: int = 0,
        startup_steps: int = 0,
    ) -> TransientSolution:
        """Integrate from ``t_start`` to ``t_end`` with steps of at most ``dt``.

        The run takes ``n = ceil((t_end - t_start) / dt)`` equal steps, so the
        step actually used is ``(t_end - t_start) / n``, never larger than
        ``dt``. With ``save_every = k > 0`` the field is stored every ``k``
        steps, plus the first and last.

        ``startup_steps = m`` replaces each of the first ``m`` steps by two
        implicit Euler steps of half the size (Rannacher time-stepping).
        Crank-Nicolson damps stiff components only weakly (its amplification
        factor tends to -1), so data that jump at the start, such as a sudden
        surface temperature, leave an error that does not converge; two or
        more start-up steps remove it while keeping the scheme second order.
        """
        if not t_end > t_start:
            raise ValueError("t_end must be greater than t_start")
        if not dt > 0.0:
            raise ValueError("dt must be positive")
        if startup_steps < 0:
            raise ValueError("startup_steps must be non-negative")
        n = max(1, math.ceil((t_end - t_start) / dt - 1e-9))
        step = (t_end - t_start) / n
        T = self.initial_field(T0).ravel()
        times = [t_start]
        balances: list[StepBalance] = []
        saved_t: list[float] = []
        saved: list[FloatArray] = []
        if save_every > 0:
            saved_t.append(t_start)
            saved.append(T.reshape(self.system.shape).copy())
        for k in range(n):
            t = t_start + k * step
            if k < startup_steps:
                for half in (0.0, 0.5):
                    T, balance = self._step(T, t + half * step, 0.5 * step, 1.0)
                    balances.append(balance)
                    times.append(balance.t)
            else:
                T, balance = self._step(T, t, step, self.theta)
                balances.append(balance)
                times.append(balance.t)
            if save_every > 0 and ((k + 1) % save_every == 0 or k + 1 == n):
                saved_t.append(times[-1])
                saved.append(T.reshape(self.system.shape).copy())
        times[-1] = t_end
        return TransientSolution(
            times=np.array(times),
            temperature=T.reshape(self.system.shape),
            balances=tuple(balances),
            snapshot_times=np.array(saved_t),
            snapshots=tuple(saved),
            system=self.system,
        )

    def _solver(self, theta: float, dt: float) -> Solve:
        # One factorization per (theta, dt); a run uses at most two (the
        # optional implicit start-up and the main scheme).
        key = (theta, dt)
        if key not in self._factors:
            if len(self._factors) >= 2:
                self._factors.pop(next(iter(self._factors)))
            self._factors[key] = self.system.matrix.factorize(self.system.mass, theta * dt)
        return self._factors[key]

    def _forcing(self, t: float) -> Forcing:
        # A step needs the data at both of its ends, and the end of one step is
        # the start of the next: keep the two most recent evaluations.
        for cached in self._forcing_cache:
            if cached.t == t:
                return cached
        forcing = self.system.forcing(t)
        self._forcing_cache = (*self._forcing_cache[-1:], forcing)
        return forcing

    def _step(
        self, T: FloatArray, t: float, dt: float, theta: float
    ) -> tuple[FloatArray, StepBalance]:
        if not dt > 0.0:
            raise ValueError("dt must be positive")
        if theta == self.theta:
            self.check_stability(dt)
        system = self.system
        t_new = t + dt
        f0, f1 = self._forcing(t), self._forcing(t_new)
        r0, r1 = f0.boundary + f0.source, f1.boundary + f1.source
        rhs = system.mass * T + dt * (theta * r1 + (1.0 - theta) * r0)
        if theta < 1.0:
            rhs -= (1.0 - theta) * dt * system.matrix.matvec(T)
        T_new = rhs / system.mass if theta == 0.0 else self._solver(theta, dt)(rhs)

        flow0 = system.boundary_heat_flow(T, t, f0)
        flow1 = system.boundary_heat_flow(T_new, t_new, f1)
        boundary = {
            name: dt * (theta * flow1[name] + (1.0 - theta) * flow0[name]) for name in flow0
        }
        source = dt * (theta * float(f1.source.sum()) + (1.0 - theta) * float(f0.source.sum()))
        stored = float(system.mass @ (T_new - T))
        scale = float(
            system.mass @ (np.abs(T) + np.abs(T_new))
            + dt * theta * (self._abs_rows @ np.abs(T_new) + _abs_sum(f1))
            + dt * (1.0 - theta) * (self._abs_rows @ np.abs(T) + _abs_sum(f0))
        )
        balance = StepBalance(
            t=t_new, dt=dt, stored=stored, boundary=boundary, source=source, scale=scale
        )
        return T_new, balance


def _abs_sum(forcing: Forcing) -> float:
    return float(np.abs(forcing.boundary).sum() + np.abs(forcing.source).sum())
