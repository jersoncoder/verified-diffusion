"""Cell-centred grids in one and two dimensions.

A 1D grid is defined by its face positions; cell centres sit at the midpoint of
their two faces. Uniform and smoothly stretched grids are built from a mapping
of a uniform computational coordinate, so that refining a grid by an integer
factor nests the coarse faces inside the fine ones. A 2D grid is the tensor
product of two 1D grids.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

Cluster = Literal["left", "right", "both"]


def _readonly(array: FloatArray) -> FloatArray:
    array.setflags(write=False)
    return array


class Grid1D:
    """A cell-centred grid on the interval ``[faces[0], faces[-1]]``.

    Parameters
    ----------
    faces:
        Strictly increasing face positions; ``len(faces) - 1`` cells.

    Second-order accuracy on a non-uniform grid relies on the cell widths
    varying smoothly, as they do for :meth:`stretched` and :meth:`from_mapping`.
    An arbitrary face array is accepted; jumps in cell width are allowed (and
    are exact for piecewise-linear profiles at material interfaces) but lower
    the local truncation order.
    """

    __slots__ = ("_centres", "_faces", "_widths")

    def __init__(self, faces: ArrayLike) -> None:
        f = np.array(faces, dtype=np.float64)
        if f.ndim != 1 or f.size < 2:
            raise ValueError("faces must be a 1D array with at least two entries")
        if not np.all(np.isfinite(f)):
            raise ValueError("faces must be finite")
        if not np.all(np.diff(f) > 0.0):
            raise ValueError("faces must be strictly increasing")
        self._faces = _readonly(f)
        self._widths = _readonly(np.diff(f))
        self._centres = _readonly(0.5 * (f[:-1] + f[1:]))

    @classmethod
    def uniform(cls, start: float, stop: float, n: int) -> Grid1D:
        """``n`` cells of equal width on ``[start, stop]``."""
        _check_cells(n)
        return cls(np.linspace(start, stop, n + 1))

    @classmethod
    def from_mapping(
        cls, start: float, stop: float, n: int, mapping: Callable[[FloatArray], FloatArray]
    ) -> Grid1D:
        """Faces at ``start + (stop - start) * mapping(xi)`` for ``xi = linspace(0, 1, n + 1)``.

        ``mapping`` must be strictly increasing with ``mapping(0) = 0`` and
        ``mapping(1) = 1``. A smooth mapping gives a grid on which the scheme
        keeps its second-order accuracy under refinement.
        """
        _check_cells(n)
        xi = np.linspace(0.0, 1.0, n + 1)
        phi = np.asarray(mapping(xi), dtype=np.float64)
        if phi.shape != xi.shape:
            raise ValueError("mapping must return an array of the same shape as its input")
        if not (np.isclose(phi[0], 0.0, atol=1e-14) and np.isclose(phi[-1], 1.0, atol=1e-14)):
            raise ValueError("mapping must satisfy mapping(0) = 0 and mapping(1) = 1")
        phi[0], phi[-1] = 0.0, 1.0
        return cls(start + (stop - start) * phi)

    @classmethod
    def stretched(
        cls, start: float, stop: float, n: int, beta: float, cluster: Cluster = "left"
    ) -> Grid1D:
        """A grid clustered by a hyperbolic-tangent mapping.

        ``beta > 0`` sets the strength of the clustering (the grid tends to
        uniform as ``beta`` tends to 0). ``cluster`` selects where the small
        cells are: next to ``start`` ("left"), next to ``stop`` ("right") or
        next to both ends ("both").
        """
        if not beta > 0.0:
            raise ValueError("beta must be positive")
        if cluster not in ("left", "right", "both"):
            raise ValueError(f"cluster must be 'left', 'right' or 'both', not {cluster!r}")
        t = float(np.tanh(beta))

        def mapping(xi: FloatArray) -> FloatArray:
            if cluster == "left":
                return 1.0 - np.tanh(beta * (1.0 - xi)) / t
            if cluster == "right":
                return np.tanh(beta * xi) / t
            return 0.5 * (1.0 + np.tanh(beta * (2.0 * xi - 1.0)) / t)

        return cls.from_mapping(start, stop, n, mapping)

    @property
    def faces(self) -> FloatArray:
        """Face positions, shape ``(n + 1,)``."""
        return self._faces

    @property
    def centres(self) -> FloatArray:
        """Cell-centre positions (face midpoints), shape ``(n,)``."""
        return self._centres

    @property
    def widths(self) -> FloatArray:
        """Cell widths, shape ``(n,)``."""
        return self._widths

    @property
    def n(self) -> int:
        """Number of cells."""
        return int(self._widths.size)

    @property
    def start(self) -> float:
        return float(self._faces[0])

    @property
    def stop(self) -> float:
        return float(self._faces[-1])

    @property
    def length(self) -> float:
        return self.stop - self.start

    @property
    def max_width_ratio(self) -> float:
        """Largest cell width divided by the smallest."""
        return float(np.max(self._widths) / np.min(self._widths))

    def __repr__(self) -> str:
        return f"Grid1D(n={self.n}, start={self.start:g}, stop={self.stop:g})"


@dataclass(frozen=True)
class Grid2D:
    """Tensor product of two 1D grids; cell ``(i, j)`` spans ``x.widths[i]`` by ``y.widths[j]``.

    Cell fields are arrays of shape ``(x.n, y.n)`` (``indexing="ij"``). The
    problem is two-dimensional per unit depth: volumes are areas and face
    areas are lengths.
    """

    x: Grid1D
    y: Grid1D

    @classmethod
    def uniform(
        cls, x_range: tuple[float, float], y_range: tuple[float, float], nx: int, ny: int
    ) -> Grid2D:
        return cls(Grid1D.uniform(*x_range, nx), Grid1D.uniform(*y_range, ny))

    @property
    def shape(self) -> tuple[int, int]:
        return (self.x.n, self.y.n)

    @property
    def size(self) -> int:
        return self.x.n * self.y.n

    def centres(self) -> tuple[FloatArray, FloatArray]:
        """Cell-centre coordinates ``(X, Y)``, each of shape :attr:`shape`."""
        X, Y = np.meshgrid(self.x.centres, self.y.centres, indexing="ij")
        return X, Y

    def volumes(self) -> FloatArray:
        """Cell areas (volumes per unit depth), shape :attr:`shape`."""
        return np.outer(self.x.widths, self.y.widths)

    def __repr__(self) -> str:
        return f"Grid2D(nx={self.x.n}, ny={self.y.n})"


def _check_cells(n: int) -> None:
    if int(n) != n or n < 1:
        raise ValueError(f"the number of cells must be a positive integer, got {n!r}")
