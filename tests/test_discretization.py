"""Structure and exactness properties of the finite-volume operator."""

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
    assemble,
    solve_steady,
)
from verified_diffusion.discretization import FiniteVolumeSystem


def problem_1d() -> Problem1D:
    return Problem1D(
        Grid1D.stretched(0.0, 1.0, 17, beta=2.0),
        left=Robin(4.0, 1.0),
        right=Dirichlet(0.0),
        conductivity=lambda x: 1.0 + x**2,
        heat_capacity=lambda x: 2.0 + x,
    )


def problem_2d() -> Problem2D:
    grid = Grid2D(
        Grid1D.stretched(0.0, 1.0, 9, beta=1.5, cluster="both"), Grid1D.uniform(0.0, 2.0, 7)
    )
    return Problem2D(
        grid,
        left=Dirichlet(1.0),
        right=Robin(2.0, 0.0),
        bottom=Neumann(0.5),
        top=Dirichlet(0.0),
        conductivity=lambda X, Y: 1.0 + X * Y,
    )


def check_m_matrix(system: FiniteVolumeSystem) -> None:
    A = system.matrix.to_sparse().toarray()
    np.testing.assert_allclose(A, A.T, rtol=0.0, atol=1e-14)
    off = A - np.diag(np.diag(A))
    assert np.all(off <= 0.0)
    assert np.all(np.diag(A) > 0.0)
    # Interior face conductances cancel: column sums are the boundary conductances.
    boundary = np.zeros(system.size)
    for side in system.sides:
        np.add.at(boundary, side.cells, side.conductance)
    np.testing.assert_allclose(A.sum(axis=0), boundary, rtol=1e-12, atol=1e-12)


def test_1d_matrix_is_a_symmetric_m_matrix() -> None:
    check_m_matrix(assemble(problem_1d()))


def test_2d_matrix_is_a_symmetric_m_matrix() -> None:
    system = assemble(problem_2d())
    assert system.matrix.to_sparse().nnz == 5 * 63 - 2 * (7 + 9)
    check_m_matrix(system)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (Dirichlet(1.0), Dirichlet(3.0)),
        (Neumann(-2.0 * 1.5), Robin(0.7, 3.0 + 1.5 * 2.0 / 0.7)),
        (Robin(5.0, 1.0 - 1.5 * 2.0 / 5.0), Neumann(1.5 * 2.0)),
    ],
)
def test_linear_profile_is_exact_on_a_stretched_grid(
    left: Dirichlet | Neumann | Robin, right: Dirichlet | Neumann | Robin
) -> None:
    # T = 1 + 2x with k = 1.5; each pair of conditions is consistent with it.
    grid = Grid1D.stretched(0.0, 1.0, 13, beta=2.5)
    solution = solve_steady(Problem1D(grid, left, right, conductivity=1.5))
    np.testing.assert_allclose(solution.temperature, 1.0 + 2.0 * grid.centres, atol=1e-13)
    left_side, right_side = solution.system.sides
    T = solution.temperature
    assert left_side.surface_temperature(T, 0.0)[0] == pytest.approx(1.0, abs=1e-13)
    assert right_side.surface_temperature(T, 0.0)[0] == pytest.approx(3.0, abs=1e-13)
    assert solution.balance.boundary["left"] == pytest.approx(-3.0, abs=1e-12)


def test_linear_field_is_exact_in_2d() -> None:
    # T = 1 + 2x - 3y with k = 2 on a stretched rectangle; one condition of each kind.
    grid = Grid2D(Grid1D.stretched(0.0, 2.0, 11, beta=2.0), Grid1D.stretched(0.0, 1.0, 8, beta=1.0))
    k, h = 2.0, 3.0

    def exact(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 1.0 + 2.0 * x - 3.0 * y

    problem = Problem2D(
        grid,
        left=Dirichlet(lambda s, t: exact(0.0 * s, s)),
        right=Robin(h, lambda s, t: exact(2.0 + 0.0 * s, s) + k * 2.0 / h),
        bottom=Neumann(k * 3.0),  # k dT/dn with n = -y
        top=Neumann(-k * 3.0),
        conductivity=k,
    )
    solution = solve_steady(problem)
    X, Y = grid.centres()
    np.testing.assert_allclose(solution.temperature, exact(X, Y), atol=1e-12)


def test_harmonic_average_reproduces_a_layered_2d_wall() -> None:
    # Two layers in x, insulated in y: the exact profile is piecewise linear.
    faces = np.concatenate([np.linspace(0.0, 0.4, 5), np.linspace(0.4, 1.0, 9)[1:]])
    grid = Grid2D(Grid1D(faces), Grid1D.uniform(0.0, 1.0, 4))
    k1, k2 = 1.0, 20.0
    problem = Problem2D(
        grid,
        left=Dirichlet(0.0),
        right=Dirichlet(1.0),
        bottom=Neumann(0.0),
        top=Neumann(0.0),
        conductivity=lambda X, Y: np.where(X < 0.4, k1, k2),
    )
    T = solve_steady(problem).temperature
    q = 1.0 / (0.4 / k1 + 0.6 / k2)
    x = grid.x.centres
    exact = np.where(x < 0.4, q * x / k1, q * 0.4 / k1 + q * (x - 0.4) / k2)
    np.testing.assert_allclose(T, np.repeat(exact[:, None], 4, axis=1), atol=1e-13)


def test_stability_bound_is_tight() -> None:
    # Gershgorin bounds the spectrum of M^-1 A from above; on a uniform grid it
    # is within O(1/n^2) of the largest eigenvalue.
    problem = Problem1D(Grid1D.uniform(0.0, 1.0, 50), Dirichlet(0.0), Dirichlet(0.0))
    system = assemble(problem)
    A = system.matrix.to_sparse().toarray()
    lam = np.linalg.eigvals(A / system.mass[:, None]).real.max()
    bound = system.stability_bound()
    assert lam <= bound
    assert lam > 0.99 * bound
    assert bound == pytest.approx(4.0 / (1.0 / 50) ** 2)


def test_time_derivative_vanishes_at_the_steady_state() -> None:
    problem = problem_2d()
    solution = solve_steady(problem)
    rate = solution.system.time_derivative(0.0, solution.temperature.ravel())
    assert np.abs(rate).max() < 1e-10


def test_coefficient_validation() -> None:
    grid = Grid1D.uniform(0.0, 1.0, 4)
    with pytest.raises(ValueError, match="conductivity must be positive"):
        assemble(Problem1D(grid, Dirichlet(0.0), Dirichlet(0.0), conductivity=lambda x: x - 0.5))
    with pytest.raises(ValueError, match="heat_capacity cannot be broadcast"):
        assemble(Problem1D(grid, Dirichlet(0.0), Dirichlet(0.0), heat_capacity=[1.0, 2.0]))
    with pytest.raises(TypeError, match="left must be"):
        Problem1D(grid, 1.0, Dirichlet(0.0))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="face_average"):
        Problem1D(grid, Dirichlet(0.0), Dirichlet(0.0), face_average="geometric")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        Robin(0.0, 1.0)
    with pytest.raises(ValueError, match="not finite"):
        solve_steady(Problem1D(grid, Dirichlet(lambda t: np.nan), Dirichlet(0.0)))
