"""The symbolic derivation of manufactured sources and boundary data."""

import numpy as np
import pytest
import sympy

from verified_diffusion import Dirichlet, Grid1D, Grid2D, Neumann, Robin, solve_steady
from verified_diffusion.boundary import BoundaryData
from verified_diffusion.mms import ManufacturedSolution, t, x, y


def evaluate(data: BoundaryData, *args: object) -> np.ndarray:
    assert callable(data)
    return np.asarray(data(*args), dtype=np.float64)


def test_source_for_a_separable_solution() -> None:
    ms = ManufacturedSolution(sympy.sin(sympy.pi * x) * sympy.exp(-t), heat_capacity=2)
    expected = (sympy.pi**2 - 2) * sympy.sin(sympy.pi * x) * sympy.exp(-t)
    assert sympy.simplify(ms.source_expr - expected) == 0


def test_source_with_variable_conductivity() -> None:
    ms = ManufacturedSolution(x**2, conductivity=1 + x)
    assert sympy.expand(ms.source_expr) == -2 - 4 * x
    q = ms.source()
    np.testing.assert_allclose(q(np.array([0.0, 0.5]), 0.0), [-2.0, -4.0])


def test_source_in_2d() -> None:
    ms = ManufacturedSolution(x**2 + y**2 + t, conductivity=3, dim=2)
    assert ms.source_expr == 1 - 12


def test_boundary_data_follow_the_inflow_convention() -> None:
    ms = ManufacturedSolution(x**2 * (1 + t), conductivity=3)
    right = ms.boundary("right", 1.0, "neumann")
    left = ms.boundary("left", 0.0, "neumann")
    robin = ms.boundary("right", 1.0, "robin", h=2.0)
    dirichlet = ms.boundary("right", 1.0, "dirichlet")
    assert isinstance(right, Neumann)
    assert isinstance(left, Neumann)
    assert isinstance(robin, Robin)
    assert isinstance(dirichlet, Dirichlet)
    # Inflow k dT/dn with n outward: +k dT/dx on the right, -k dT/dx on the left.
    assert evaluate(right.flux, 0.5) == pytest.approx(3 * 2 * 1.5)
    assert evaluate(left.flux, 0.5) == pytest.approx(0.0)
    # Robin: h (T_inf - T) equals the inflow.
    assert evaluate(robin.ambient, 0.5) == pytest.approx(1.5 + 9.0 / 2.0)
    assert evaluate(dirichlet.temperature, 0.5) == pytest.approx(1.5)


def test_2d_boundary_data_take_the_coordinate_along_the_side() -> None:
    ms = ManufacturedSolution(x * y**2 + t, dim=2)
    bottom = ms.boundary("bottom", 0.5, "neumann")
    top = ms.boundary("top", 2.0, "dirichlet")
    assert isinstance(bottom, Neumann)
    assert isinstance(top, Dirichlet)
    s = np.array([1.0, 2.0])
    np.testing.assert_allclose(evaluate(bottom.flux, s, 0.0), -2.0 * s * 0.5)  # -k dT/dy, y = 0.5
    np.testing.assert_allclose(evaluate(top.temperature, s, 1.0), s * 4.0 + 1.0)


@pytest.mark.parametrize(("left", "right"), [("robin", "neumann"), ("neumann", "dirichlet")])
def test_linear_solutions_are_reproduced_to_round_off(left: str, right: str) -> None:
    ms = ManufacturedSolution(2 - 3 * x, conductivity=4)
    grid = Grid1D.stretched(0.0, 2.0, 15, beta=1.5)
    problem = ms.problem_1d(grid, left, right, h=0.5)  # type: ignore[arg-type]
    T = solve_steady(problem).temperature
    np.testing.assert_allclose(T, 2 - 3 * grid.centres, atol=1e-12)


def test_linear_field_in_2d_with_every_kind_of_side() -> None:
    ms = ManufacturedSolution(1 + x - 2 * y, conductivity=2, dim=2)
    grid = Grid2D(Grid1D.uniform(0.0, 1.0, 6), Grid1D.stretched(0.0, 3.0, 9, beta=1.0))
    kinds = {"left": "robin", "right": "neumann", "bottom": "dirichlet", "top": "robin"}
    T = solve_steady(ms.problem_2d(grid, kinds, h=3.0)).temperature  # type: ignore[arg-type]
    X, Y = grid.centres()
    np.testing.assert_allclose(T, 1 + X - 2 * Y, atol=1e-12)


def test_invalid_definitions() -> None:
    with pytest.raises(ValueError, match="dim"):
        ManufacturedSolution(x, dim=3)
    with pytest.raises(ValueError, match="outside"):
        ManufacturedSolution(x * y)
    with pytest.raises(ValueError, match="time"):
        ManufacturedSolution(x, conductivity=1 + t)
    ms = ManufacturedSolution(x)
    with pytest.raises(ValueError, match="unknown side"):
        ms.boundary("top", 1.0, "dirichlet")
    with pytest.raises(ValueError, match="needs the heat transfer"):
        ms.boundary("left", 0.0, "robin")
    with pytest.raises(ValueError, match="unknown kind"):
        ms.boundary("left", 0.0, "periodic")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dim=2"):
        ms.problem_2d(Grid2D.uniform((0, 1), (0, 1), 2, 2), {})
