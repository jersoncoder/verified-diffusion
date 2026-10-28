import numpy as np
import pytest

from verified_diffusion.tridiagonal import TridiagonalLU, solve_tridiagonal, tridiagonal_matvec


def dense(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.diag(diag) + np.diag(lower, -1) + np.diag(upper, 1)


@pytest.mark.parametrize("n", [1, 2, 3, 50])
def test_thomas_matches_dense_solve(n: int) -> None:
    rng = np.random.default_rng(n)
    lower = rng.uniform(-1.0, 1.0, n - 1)
    upper = rng.uniform(-1.0, 1.0, n - 1)
    diag = 2.5 + rng.uniform(0.0, 1.0, n)
    rhs = rng.normal(size=n)
    x = solve_tridiagonal(lower, diag, upper, rhs)
    np.testing.assert_allclose(x, np.linalg.solve(dense(lower, diag, upper), rhs), rtol=1e-12)


def test_factorization_is_reused_for_many_right_hand_sides() -> None:
    rng = np.random.default_rng(7)
    n = 30
    lower = -np.ones(n - 1)
    upper = -np.ones(n - 1)
    diag = np.full(n, 2.0)
    diag[0] = diag[-1] = 3.0  # an M-matrix like a Dirichlet diffusion operator
    lu = TridiagonalLU(lower, diag, upper)
    A = dense(lower, diag, upper)
    for _ in range(5):
        rhs = rng.normal(size=n)
        np.testing.assert_allclose(A @ lu.solve(rhs), rhs, atol=1e-12)


def test_matvec_matches_dense_product() -> None:
    rng = np.random.default_rng(3)
    n = 9
    lower, diag, upper, x = (rng.normal(size=s) for s in (n - 1, n, n - 1, n))
    np.testing.assert_allclose(
        tridiagonal_matvec(lower, diag, upper, x), dense(lower, diag, upper) @ x, rtol=1e-14
    )


def test_zero_pivot_is_reported() -> None:
    with pytest.raises(np.linalg.LinAlgError, match="pivot at row 1"):
        TridiagonalLU([1.0], [1.0, 1.0], [1.0])


def test_shape_errors() -> None:
    with pytest.raises(ValueError, match="length 2"):
        TridiagonalLU([1.0], [1.0, 2.0, 3.0], [1.0, 1.0])
    lu = TridiagonalLU([0.0], [1.0, 1.0], [0.0])
    with pytest.raises(ValueError, match="shape"):
        lu.solve([1.0, 2.0, 3.0])
