import numpy as np
import pytest

from verified_diffusion.verification import (
    ConvergenceStudy,
    error_norms,
    fitted_order,
    observed_orders,
)


def test_error_norms_are_volume_weighted() -> None:
    l2, emax = error_norms([1.0, 2.0], [0.0, 0.0], [3.0, 1.0])
    assert l2 == pytest.approx(np.sqrt((3.0 * 1.0 + 1.0 * 4.0) / 4.0))
    assert emax == 2.0


def test_observed_and_fitted_orders_of_a_power_law() -> None:
    h = [0.4, 0.2, 0.1, 0.05]
    e = [3.0 * hk**1.5 for hk in h]
    np.testing.assert_allclose(observed_orders(h, e), 1.5)
    assert fitted_order(h, e) == pytest.approx(1.5)


def test_convergence_study_accessors() -> None:
    study = ConvergenceStudy(
        name="demo",
        description="",
        parameter="h",
        levels=(10, 20, 40),
        sizes=(0.1, 0.05, 0.025),
        errors_l2=(4e-2, 1e-2, 2.5e-3),
        errors_max=(8e-2, 4e-2, 2e-2),
        expected_order=2.0,
    )
    assert study.finest_order() == pytest.approx(2.0)
    assert study.finest_order("max") == pytest.approx(1.0)
    assert study.fitted_order("l2") == pytest.approx(2.0)


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="two levels"):
        observed_orders([0.1], [1.0])
    with pytest.raises(ValueError, match="positive"):
        observed_orders([0.1, 0.05], [1.0, 0.0])
