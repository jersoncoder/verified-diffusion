"""Observed orders of accuracy: the claims the README makes, asserted.

Tolerance: the observed order between the two finest levels must lie within
``ORDER_TOL`` of the theoretical order. The one exception is documented in
docs/verification.md: on the stretched 2D grid the max-norm order rises
towards 2 more slowly (largest error next to a corner where two Dirichlet
sides meet across cells of aspect ratio near 5), so it is held to a lower
bound instead.
"""

from collections.abc import Callable

import numpy as np
import pytest

from verified_diffusion import TransientSolver, studies
from verified_diffusion.verification import ConvergenceStudy, observed_orders

ORDER_TOL = 0.1
STRETCHED_2D_MAX_NORM_FLOOR = 1.75


def assert_order(study: ConvergenceStudy, max_norm_floor: float | None = None) -> None:
    errors = np.array(study.errors_l2)
    assert np.all(np.diff(errors) < 0.0), "errors must fall on every refinement"
    assert abs(study.finest_order("l2") - study.expected_order) <= ORDER_TOL
    if max_norm_floor is None:
        assert abs(study.finest_order("max") - study.expected_order) <= ORDER_TOL
    else:
        assert study.finest_order("max") >= max_norm_floor


@pytest.mark.parametrize(
    "run",
    [
        studies.steady_space_1d_uniform,
        studies.steady_space_1d_stretched,
        studies.transient_space_1d,
        studies.plate_study,
    ],
    ids=lambda f: f.__name__,
)
def test_second_order_in_space(run: Callable[[], ConvergenceStudy]) -> None:
    assert_order(run())


@pytest.mark.parametrize(
    "run", [studies.steady_space_2d, studies.transient_space_2d], ids=lambda f: f.__name__
)
def test_second_order_in_space_on_stretched_2d_grids(run: Callable[[], ConvergenceStudy]) -> None:
    assert_order(run(), max_norm_floor=STRETCHED_2D_MAX_NORM_FLOOR)


@pytest.mark.parametrize(
    ("scheme", "order"),
    [("crank-nicolson", 2.0), ("implicit-euler", 1.0), ("explicit-euler", 1.0)],
)
def test_order_in_time(scheme: str, order: float) -> None:
    study = studies.time_order(scheme)
    assert study.expected_order == order
    assert_order(study)


def test_time_reference_is_far_more_accurate_than_the_errors_it_measures() -> None:
    problem = studies._time_problem()
    system = TransientSolver(problem).system
    T0 = studies.TRANSIENT_1D.exact()(problem.grid.centres, 0.0)
    tight = studies.semi_discrete_reference(system, T0, studies.TIME_STUDY_END, rtol=1e-12)
    loose = studies.semi_discrete_reference(system, T0, studies.TIME_STUDY_END, rtol=1e-10)
    smallest_error = min(studies.time_order("crank-nicolson").errors_max)
    assert np.max(np.abs(tight - loose)) < 1e-3 * smallest_error


def test_erfc_with_start_up_steps_is_second_order() -> None:
    assert_order(studies.erfc_study("crank-nicolson", startup_steps=2))


def test_erfc_implicit_euler_is_first_order_with_dt_proportional_to_h() -> None:
    assert_order(studies.erfc_study("implicit-euler", startup_steps=0))


def test_erfc_plain_crank_nicolson_does_not_converge_in_the_max_norm() -> None:
    # The documented failure mode the start-up steps exist for: a step change
    # at t = 0 leaves an oscillation next to the surface that Crank-Nicolson
    # never damps, so the max-norm error stays O(1) under refinement.
    study = studies.erfc_study("crank-nicolson", startup_steps=0)
    assert min(study.errors_max) > 0.5
    assert abs(study.finest_order("l2") - 0.5) <= ORDER_TOL


def test_slab_with_convection() -> None:
    assert_order(studies.slab_study(startup_steps=2))
    plain = studies.slab_study(startup_steps=0)
    assert plain.finest_order("max") < 1.5  # the same start-up effect, milder


def test_composite_wall_is_exact_with_harmonic_averaging() -> None:
    result = studies.composite_wall_study()
    assert max(result.harmonic) < 1e-12
    assert max(result.harmonic_stretched) < 1e-12
    # Arithmetic averaging is not exact, and only first order.
    sizes = [1.0 / n for n in result.cells_per_layer]
    assert min(result.arithmetic) > 1e-3
    assert abs(observed_orders(sizes, result.arithmetic)[-1] - 1.0) <= ORDER_TOL


def test_gci_study_is_in_the_asymptotic_range() -> None:
    for quantity in studies.gci_study():
        for analysis in (quantity.fine, quantity.coarse):
            assert analysis.monotone
            assert abs(analysis.order - 2.0) <= ORDER_TOL
        # The fine-grid band contains the extrapolated value.
        assert abs(quantity.fine.f1 - quantity.fine.extrapolated) <= quantity.fine.uncertainty


def test_energy_balance_study() -> None:
    rows = studies.energy_balance_study()
    assert len(rows) == 6
    for row in rows:
        assert row.max_relative_residual < 1e-14
        assert row.stored == pytest.approx(row.boundary + row.source, rel=1e-10)
