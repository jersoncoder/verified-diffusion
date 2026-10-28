"""Time integration: stability limit, energy balance and stepping mechanics."""

import math
import re

import numpy as np
import pytest

from verified_diffusion import (
    Dirichlet,
    Grid1D,
    Grid2D,
    Neumann,
    Problem1D,
    Problem2D,
    Robin,
    StabilityError,
    TransientSolver,
    solve_steady,
)
from verified_diffusion.discretization import Solve

SCHEME_NAMES = ["explicit-euler", "crank-nicolson", "implicit-euler"]

# Round-off bound for the energy residual, relative to the magnitude of the
# terms in the balance (see StepBalance.scale). Measured values are near 1e-17.
ROUND_OFF = 1e-14


def busy_problem_1d() -> Problem1D:
    """Every kind of condition, time-dependent data, variable properties, stretched grid."""
    return Problem1D(
        Grid1D.stretched(0.0, 1.0, 40, beta=2.0),
        left=Robin(8.0, lambda t: 300.0 + 20.0 * math.sin(3.0 * t)),
        right=Neumann(lambda t: -500.0 * t),
        conductivity=lambda x: 1.0 + 3.0 * x,
        heat_capacity=lambda x: 2.0 - x,
        source=lambda x, t: 1e3 * np.exp(-5.0 * x) * np.cos(2.0 * t),
    )


def busy_problem_2d() -> Problem2D:
    grid = Grid2D(
        Grid1D.stretched(0.0, 1.0, 16, beta=1.5, cluster="both"), Grid1D.uniform(0.0, 0.5, 10)
    )
    return Problem2D(
        grid,
        left=Dirichlet(lambda s, t: 300.0 + 50.0 * s * t),
        right=Robin(3.0, lambda s, t: 280.0 + 0.0 * s),
        bottom=Neumann(lambda s, t: 100.0 * np.sin(math.pi * s)),
        top=Dirichlet(300.0),
        conductivity=lambda X, Y: 1.0 + X + 4.0 * Y,
        heat_capacity=1.5,
        source=lambda X, Y, t: 200.0 * np.exp(-((X - 0.5) ** 2) / 0.02) * (1.0 + t),
    )


def time_step(solver: TransientSolver, implicit_dt: float) -> float:
    return 0.9 * solver.max_stable_dt if solver.theta < 0.5 else implicit_dt


@pytest.mark.parametrize("scheme", SCHEME_NAMES)
@pytest.mark.parametrize("make_problem", [busy_problem_1d, busy_problem_2d])
def test_energy_balance_closes_to_round_off(scheme: str, make_problem: object) -> None:
    problem = make_problem()  # type: ignore[operator]
    solver = TransientSolver(problem, scheme)
    run = solver.run(300.0, 0.05, time_step(solver, 2e-3))
    assert run.max_relative_residual < ROUND_OFF
    # The balance is not trivially zero: every term is active.
    last = run.balances[-1]
    assert all(abs(q) > 0.0 for q in last.boundary.values())
    assert abs(last.source) > 0.0
    assert abs(last.stored) > 0.0
    # Cumulative stored energy equals the change of the total heat content.
    stored, boundary, source = run.total_balance()
    mass = run.system.mass
    content_change = float(mass @ (run.temperature.ravel() - 300.0))
    assert stored == pytest.approx(content_change, rel=1e-9)
    assert stored == pytest.approx(boundary + source, rel=1e-9)


def test_insulated_bar_stores_exactly_the_heat_it_receives() -> None:
    q_in = 250.0
    problem = Problem1D(
        Grid1D.uniform(0.0, 0.1, 20),
        left=Neumann(q_in),
        right=Neumann(0.0),
        conductivity=45.0,
        heat_capacity=3.6e6,
    )
    run = TransientSolver(problem, "implicit-euler").run(20.0, 600.0, 10.0)
    stored, boundary, source = run.total_balance()
    assert boundary == pytest.approx(q_in * 600.0, rel=1e-12)
    assert stored == pytest.approx(q_in * 600.0, rel=1e-9)
    assert source == 0.0


def test_steady_balance_closes() -> None:
    solution = solve_steady(busy_problem_2d(), t=0.3)
    balance = solution.balance
    assert balance.relative_residual < ROUND_OFF
    assert balance.boundary_total == pytest.approx(-balance.source, rel=1e-10)


def test_steady_problem_needs_a_fixed_temperature_level() -> None:
    problem = Problem1D(Grid1D.uniform(0.0, 1.0, 5), Neumann(1.0), Neumann(-1.0))
    with pytest.raises(ValueError, match="no unique solution"):
        solve_steady(problem)


def test_explicit_step_beyond_the_limit_is_refused_with_the_fourier_number() -> None:
    k, rho_c, dx = 2.0, 4.0, 0.1
    alpha = k / rho_c
    problem = Problem1D(
        Grid1D.uniform(0.0, 1.0, 10), Dirichlet(0.0), Dirichlet(0.0), k, heat_capacity=rho_c
    )
    solver = TransientSolver(problem, "explicit-euler")
    dt_max = dx**2 / (2.0 * alpha)
    assert solver.max_stable_dt == pytest.approx(dt_max)
    assert solver.fourier_number(dt_max) == pytest.approx(0.5)
    dt = 1.2 * dt_max
    with pytest.raises(StabilityError) as info:
        solver.step(np.zeros(10), 0.0, dt)
    message = str(info.value)
    assert "Fourier number" in message
    match = re.search(r"Fo = ([0-9.e+-]+).*dt_max = ([0-9.e+-]+)", message)
    assert match is not None
    assert float(match.group(1)) == pytest.approx(alpha * dt / dx**2, rel=1e-5)
    assert float(match.group(2)) == pytest.approx(dt_max, rel=1e-5)
    solver.run(0.0, 10 * dt_max, dt_max)  # the limit itself is accepted


def test_fourier_number_in_2d_is_the_sum_of_both_directions() -> None:
    grid = Grid2D.uniform((0.0, 1.0), (0.0, 1.0), 10, 20)
    zero = Dirichlet(0.0)
    problem = Problem2D(grid, left=zero, right=zero, bottom=zero, top=zero)
    solver = TransientSolver(problem, "explicit-euler")
    dt = 1e-4
    assert solver.fourier_number(dt) == pytest.approx(dt * (1 / 0.1**2 + 1 / 0.05**2))
    assert solver.max_stable_dt == pytest.approx(0.5 / (1 / 0.1**2 + 1 / 0.05**2))


def test_theta_below_one_half_has_a_wider_limit() -> None:
    problem = Problem1D(Grid1D.uniform(0.0, 1.0, 10), Dirichlet(0.0), Dirichlet(0.0))
    explicit = TransientSolver(problem, 0.0)
    theta = TransientSolver(problem, 0.25)
    assert theta.fourier_limit == pytest.approx(1.0)
    assert theta.max_stable_dt == pytest.approx(2.0 * explicit.max_stable_dt)
    for scheme in ("crank-nicolson", "implicit-euler"):
        solver = TransientSolver(problem, scheme)
        assert solver.max_stable_dt == math.inf
        solver.run(0.0, 10.0, 5.0)  # any step is accepted


def test_explicit_euler_at_the_limit_does_not_grow() -> None:
    # Zero data and a rough initial field: the energy norm sum(M T^2) must not
    # grow at the largest stable step.
    problem = Problem1D(Grid1D.stretched(0.0, 1.0, 30, beta=1.0), Robin(2.0, 0.0), Dirichlet(0.0))
    solver = TransientSolver(problem, "explicit-euler")
    rng = np.random.default_rng(0)
    T = rng.normal(size=30)
    mass = solver.system.mass
    norm = float(mass @ T**2)
    for k in range(400):
        T, _ = solver.step(T, k * solver.max_stable_dt, solver.max_stable_dt)
        new_norm = float(mass @ T**2)
        assert new_norm <= norm * (1.0 + 1e-12)
        norm = new_norm


def test_run_takes_equal_steps_that_land_on_the_end_time() -> None:
    problem = Problem1D(Grid1D.uniform(0.0, 1.0, 8), Dirichlet(1.0), Neumann(0.0))
    solver = TransientSolver(problem, "crank-nicolson")
    calls = 0
    factorize = solver.system.matrix.factorize

    def counting(mass: np.ndarray | None, scale: float) -> Solve:
        nonlocal calls
        calls += 1
        return factorize(mass, scale)

    solver.system.matrix.factorize = counting  # type: ignore[method-assign]
    run = solver.run(0.0, 1.0, 0.3, save_every=2)
    assert len(run.balances) == 4
    np.testing.assert_allclose(np.diff(run.times), 0.25)
    assert run.times[-1] == 1.0
    np.testing.assert_allclose(run.snapshot_times, [0.0, 0.5, 1.0])
    assert calls == 1


def test_long_implicit_run_reaches_the_steady_state() -> None:
    busy = busy_problem_2d()
    frozen = Problem2D(
        busy.grid,
        left=Dirichlet(lambda s, t: 300.0 + 50.0 * s),
        right=busy.right,
        bottom=busy.bottom,
        top=busy.top,
        conductivity=busy.conductivity,
        heat_capacity=busy.heat_capacity,
        source=lambda X, Y, t: 400.0 * np.exp(-((X - 0.5) ** 2) / 0.02),
    )
    steady = solve_steady(frozen)
    run = TransientSolver(frozen, "implicit-euler").run(300.0, 50.0, 1.0)
    np.testing.assert_allclose(run.temperature, steady.temperature, rtol=1e-9)


def test_argument_validation() -> None:
    problem = Problem1D(Grid1D.uniform(0.0, 1.0, 4), Dirichlet(0.0), Dirichlet(0.0))
    with pytest.raises(ValueError, match="unknown scheme"):
        TransientSolver(problem, "leapfrog")
    with pytest.raises(ValueError, match="theta"):
        TransientSolver(problem, 1.5)
    solver = TransientSolver(problem)
    with pytest.raises(ValueError, match="shape"):
        solver.step(np.zeros(5), 0.0, 0.1)
    with pytest.raises(ValueError, match="t_end"):
        solver.run(0.0, 0.0, 0.1)
    with pytest.raises(ValueError, match="dt"):
        solver.run(0.0, 1.0, -0.1)
