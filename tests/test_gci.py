"""Richardson extrapolation and GCI on sequences whose limit and order are known exactly."""

import math

import pytest

from verified_diffusion.gci import grid_convergence_index, observed_order, richardson_extrapolate


def power_law(h: float, f0: float = 3.0, c: float = 0.8, p: float = 2.0) -> float:
    return f0 + c * math.pow(h, p)


@pytest.mark.parametrize("p", [1.0, 2.0, 3.7])
def test_constant_ratio_recovers_order_and_limit(p: float) -> None:
    h = [0.1, 0.2, 0.4]
    f1, f2, f3 = (power_law(hk, p=p) for hk in h)
    result = grid_convergence_index(f1, f2, f3, r21=2.0)
    assert result.order == pytest.approx(p, rel=1e-10)
    assert result.extrapolated == pytest.approx(3.0, rel=1e-12)
    assert result.monotone
    # For an exact power law the fine-grid band is Fs times the true error.
    assert result.uncertainty == pytest.approx(1.25 * abs(f1 - 3.0), rel=1e-10)
    assert result.gci_fine == pytest.approx(result.uncertainty / abs(f1))


def test_non_constant_ratio_recovers_order() -> None:
    h1, h2, h3 = 0.1, 0.13, 0.31
    f1, f2, f3 = (power_law(h, p=2.3) for h in (h1, h2, h3))
    p = observed_order(f1, f2, f3, r21=h2 / h1, r32=h3 / h2)
    assert p == pytest.approx(2.3, rel=1e-9)
    result = grid_convergence_index(f1, f2, f3, r21=h2 / h1, r32=h3 / h2)
    assert result.extrapolated == pytest.approx(3.0, rel=1e-10)


def test_higher_order_terms_fade_under_refinement() -> None:
    def f(h: float) -> float:
        return 1.0 + h**2 + 5.0 * h**3

    near = grid_convergence_index(f(0.01), f(0.02), f(0.04), 2.0)
    far = grid_convergence_index(f(0.1), f(0.2), f(0.4), 2.0)
    assert abs(near.order - 2.0) < 0.3 * abs(far.order - 2.0)
    assert abs(near.extrapolated - 1.0) < abs(far.extrapolated - 1.0)


def test_close_ratios_do_not_break_the_solver() -> None:
    # r21 near 1 and r32 large: a fixed-point iteration on p diverges here.
    h = (0.1, 0.101, 0.5)
    f1, f2, f3 = (power_law(hk, p=1.7) for hk in h)
    assert observed_order(f1, f2, f3, h[1] / h[0], h[2] / h[1]) == pytest.approx(1.7, rel=1e-8)


def test_oscillatory_convergence_is_flagged() -> None:
    result = grid_convergence_index(1.0 + 0.01, 1.0 - 0.04, 1.0 + 0.16, 2.0)
    assert not result.monotone
    assert result.order == pytest.approx(2.0)


def test_richardson_extrapolation_formula() -> None:
    assert richardson_extrapolate(1.01, 1.04, 2.0, 2.0) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="greater than 1"):
        richardson_extrapolate(1.0, 2.0, 1.0, 2.0)


def test_degenerate_inputs() -> None:
    with pytest.raises(ValueError, match="undefined"):
        observed_order(1.0, 1.0, 2.0, 2.0)
    with pytest.raises(ValueError, match="greater than 1"):
        observed_order(1.0, 2.0, 3.0, 0.5)
    with pytest.raises(ValueError, match="do not shrink"):
        observed_order(1.0, 1.1, 1.15, 2.0)
    with pytest.raises(ValueError, match="do not shrink"):
        observed_order(1.0, 1.1, 1.15, 1.5, 2.5)
    result = grid_convergence_index(0.0, 0.04, 0.2, 2.0)
    assert math.isinf(result.gci_fine)
