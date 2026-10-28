"""The reference solutions satisfy their own equations and boundary conditions."""

import math

import numpy as np
import pytest

from verified_diffusion import analytic


def test_semi_infinite_solution_limits_and_equation() -> None:
    alpha = 2.0

    def temp(x: float, t: float) -> float:
        return float(analytic.semi_infinite_step(x, t, alpha=alpha, T_initial=10.0, T_surface=50.0))

    assert temp(0.0, 0.3) == pytest.approx(50.0)
    assert temp(100.0, 0.3) == pytest.approx(10.0)
    # dT/dt = alpha d2T/dx2 by central differences at an interior point.
    x0, t0, d = 0.4, 0.3, 1e-4
    dTdt = (temp(x0, t0 + d) - temp(x0, t0 - d)) / (2 * d)
    d2Tdx2 = (temp(x0 + d, t0) - 2 * temp(x0, t0) + temp(x0 - d, t0)) / d**2
    assert dTdt == pytest.approx(alpha * d2Tdx2, rel=1e-5)
    with pytest.raises(ValueError, match="t > 0"):
        analytic.semi_infinite_step(0.0, 0.0, alpha=1.0, T_initial=0.0, T_surface=1.0)


def test_erfc_study_domain_is_long_enough_to_stand_for_a_half_space() -> None:
    from verified_diffusion import studies

    far = analytic.semi_infinite_step(
        studies.ERFC_LENGTH,
        studies.ERFC_END,
        alpha=studies.ERFC_ALPHA,
        T_initial=0.0,
        T_surface=1.0,
    )
    assert float(far) < 1e-9


def test_slab_eigenvalues_solve_the_characteristic_equation() -> None:
    lam = analytic.slab_convection_eigenvalues(5.0, 50)
    np.testing.assert_allclose(lam * np.tan(lam), 5.0, rtol=1e-10)
    assert np.all(np.diff(lam) > 0.0)
    assert lam[0] < 0.5 * math.pi


def test_slab_series_satisfies_its_boundary_conditions() -> None:
    kw = {
        "length": 0.5,
        "alpha": 1e-5,
        "conductivity": 20.0,
        "h": 200.0,
        "T_initial": 400.0,
        "T_ambient": 300.0,
    }
    t, d = 2000.0, 1e-6

    def temp(x: float) -> float:
        return float(analytic.slab_convection(x, t, **kw))  # type: ignore[arg-type]

    assert (temp(d) - temp(0.0)) / d == pytest.approx(0.0, abs=1e-3)  # insulated
    L = kw["length"]
    flux_out = -kw["conductivity"] * (temp(L) - temp(L - d)) / d
    assert flux_out == pytest.approx(kw["h"] * (temp(L) - kw["T_ambient"]), rel=1e-4)
    early = analytic.slab_convection(np.array([0.0, 0.1]), 1.0, **kw)  # type: ignore[arg-type]
    np.testing.assert_allclose(early, 400.0, atol=1e-6)  # far from the cooled face


def test_plate_series_matches_its_edges_and_is_harmonic() -> None:
    b = analytic.parabola_sine_coefficients(2.0, 2001)
    xs = np.linspace(0.0, 1.0, 21)

    def temp(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return analytic.rectangular_plate(x, y, width=1.0, height=0.5, top_coefficients=b)

    np.testing.assert_allclose(temp(xs, np.full_like(xs, 0.5)), 8.0 * xs * (1 - xs), atol=1e-8)
    np.testing.assert_allclose(temp(xs, np.zeros_like(xs)), 0.0, atol=1e-14)
    np.testing.assert_allclose(temp(np.zeros(3), np.array([0.1, 0.2, 0.3])), 0.0, atol=1e-14)
    x0, y0, d = np.array([0.3]), np.array([0.2]), 1e-3
    lap = (
        temp(x0 + d, y0) + temp(x0 - d, y0) + temp(x0, y0 + d) + temp(x0, y0 - d) - 4 * temp(x0, y0)
    ) / d**2
    assert abs(lap[0]) < 1e-5


def test_parabola_coefficients_reproduce_the_profile() -> None:
    b = analytic.parabola_sine_coefficients(1.0, 4001)
    s = np.linspace(0.0, 1.0, 11)
    series = np.sum(b * np.sin(np.arange(1, b.size + 1) * math.pi * s[:, None]), axis=1)
    np.testing.assert_allclose(series, 4.0 * s * (1.0 - s), atol=1e-8)


def test_composite_wall_profile() -> None:
    xs = np.array([0.0, 0.1, 0.3, 0.6, 1.0])
    T, q = analytic.composite_wall(
        xs, thicknesses=(0.3, 0.7), conductivities=(1.0, 7.0), T_left=100.0, T_right=0.0
    )
    assert q == pytest.approx(100.0 / (0.3 + 0.1))
    assert T[0] == pytest.approx(100.0)
    assert T[-1] == pytest.approx(0.0, abs=1e-12)
    assert T[2] == pytest.approx(100.0 - q * 0.3)  # interface
    assert (T[3] - T[2]) / 0.3 == pytest.approx(-q / 7.0)
    with pytest.raises(ValueError, match="equal length"):
        analytic.composite_wall(
            xs, thicknesses=(1.0,), conductivities=(1.0, 2.0), T_left=0, T_right=1
        )
