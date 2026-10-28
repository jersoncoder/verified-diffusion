"""Tridiagonal systems: the Thomas algorithm with a reusable factorization.

The 1D finite-volume matrices are symmetric, diagonally dominant M-matrices,
for which Gaussian elimination without pivoting is stable. The factorization is
computed once per matrix and reused for every right-hand side, which is what a
constant time step needs.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _bands(
    lower: ArrayLike, diag: ArrayLike, upper: ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray]:
    a = np.asarray(lower, dtype=np.float64)
    d = np.asarray(diag, dtype=np.float64)
    c = np.asarray(upper, dtype=np.float64)
    if d.ndim != 1 or d.size == 0:
        raise ValueError("diag must be a non-empty 1D array")
    if a.shape != (d.size - 1,) or c.shape != (d.size - 1,):
        raise ValueError(
            f"lower and upper must have length {d.size - 1} for a diagonal of length {d.size}"
        )
    return a, d, c


class TridiagonalLU:
    """LU factorization of a tridiagonal matrix without pivoting (Thomas algorithm).

    The matrix has ``diag`` on the main diagonal, ``lower[i] = A[i + 1, i]``
    and ``upper[i] = A[i, i + 1]``. Factorization costs ``O(n)`` and each
    :meth:`solve` costs ``O(n)``.

    Raises
    ------
    numpy.linalg.LinAlgError
        If a pivot is zero or not finite (the matrix needs pivoting or is singular).
    """

    __slots__ = ("_multipliers", "_n", "_pivots", "_upper")

    def __init__(self, lower: ArrayLike, diag: ArrayLike, upper: ArrayLike) -> None:
        a, d, c = _bands(lower, diag, upper)
        n = d.size
        pivots = [0.0] * n
        multipliers = [0.0] * n
        upper_list: list[float] = c.tolist()
        lower_list: list[float] = a.tolist()
        diag_list: list[float] = d.tolist()
        pivot = diag_list[0]
        _check_pivot(pivot, 0)
        pivots[0] = pivot
        for i in range(1, n):
            m = lower_list[i - 1] / pivot
            pivot = diag_list[i] - m * upper_list[i - 1]
            _check_pivot(pivot, i)
            multipliers[i] = m
            pivots[i] = pivot
        self._n = n
        self._pivots = pivots
        self._multipliers = multipliers
        self._upper = upper_list

    @property
    def n(self) -> int:
        """Size of the system."""
        return self._n

    def solve(self, rhs: ArrayLike) -> FloatArray:
        """Solve ``A x = rhs`` for a 1D right-hand side of length :attr:`n`."""
        b = np.asarray(rhs, dtype=np.float64)
        if b.shape != (self._n,):
            raise ValueError(f"rhs must have shape ({self._n},), got {b.shape}")
        y: list[float] = b.tolist()
        m = self._multipliers
        p = self._pivots
        c = self._upper
        n = self._n
        for i in range(1, n):
            y[i] -= m[i] * y[i - 1]
        y[n - 1] /= p[n - 1]
        for i in range(n - 2, -1, -1):
            y[i] = (y[i] - c[i] * y[i + 1]) / p[i]
        return np.array(y, dtype=np.float64)


def solve_tridiagonal(
    lower: ArrayLike, diag: ArrayLike, upper: ArrayLike, rhs: ArrayLike
) -> FloatArray:
    """Solve one tridiagonal system with the Thomas algorithm."""
    return TridiagonalLU(lower, diag, upper).solve(rhs)


def tridiagonal_matvec(
    lower: ArrayLike, diag: ArrayLike, upper: ArrayLike, x: ArrayLike
) -> FloatArray:
    """Return ``A @ x`` for the tridiagonal matrix given by its three bands."""
    a, d, c = _bands(lower, diag, upper)
    v = np.asarray(x, dtype=np.float64)
    if v.shape != d.shape:
        raise ValueError(f"x must have shape {d.shape}, got {v.shape}")
    out = d * v
    out[:-1] += c * v[1:]
    out[1:] += a * v[:-1]
    return out


def _check_pivot(pivot: float, row: int) -> None:
    if pivot == 0.0 or not np.isfinite(pivot):
        raise np.linalg.LinAlgError(
            f"zero or non-finite pivot at row {row}: the matrix is singular or needs pivoting"
        )
