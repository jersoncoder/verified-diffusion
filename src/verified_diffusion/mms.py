"""Method of manufactured solutions, derived symbolically with sympy.

Choose a smooth temperature ``T(x, t)`` (or ``T(x, y, t)``), a conductivity
and a heat capacity; sympy derives the source

    q = rho_c dT/dt - div(k grad T)

and the boundary data for each side and kind of condition, so that the chosen
``T`` is the exact solution of the resulting problem. The expressions are
compiled with ``sympy.lambdify`` into the functions a :class:`Problem1D` or
:class:`Problem2D` expects.

This module needs sympy, which is a development dependency
(``pip install -e ".[dev]"``); the solver itself does not import it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import sympy
from numpy.typing import ArrayLike, NDArray

from .boundary import BoundaryCondition, Dirichlet, Neumann, Robin
from .grid import Grid1D, Grid2D
from .problem import Problem1D, Problem2D

FloatArray = NDArray[np.float64]
Kind = Literal["dirichlet", "neumann", "robin"]

x, y, t = sympy.symbols("x y t", real=True)
"""The symbols manufactured solutions are written in."""


def _compile(symbols: tuple[Any, ...], expr: Any) -> Callable[..., FloatArray]:
    """Lambdify ``expr`` and broadcast its value to the shape of the arguments.

    A constant expression lambdifies to a scalar; broadcasting keeps every
    compiled function returning one value per evaluation point.
    """
    func = sympy.lambdify(symbols, expr, modules="numpy")

    def compiled(*args: ArrayLike) -> FloatArray:
        shape = np.broadcast(*(np.asarray(a) for a in args)).shape
        value = np.asarray(func(*args), dtype=np.float64)
        return np.broadcast_to(value, shape).copy()

    return compiled


@dataclass(frozen=True)
class ManufacturedSolution:
    """A chosen exact solution and the material properties it is paired with.

    ``temperature`` is a sympy expression in ``x``, ``t`` (``dim=1``) or ``x``,
    ``y``, ``t`` (``dim=2``); it may omit ``t`` for a steady problem.
    ``conductivity`` and ``heat_capacity`` are expressions in the space
    variables (or numbers).
    """

    temperature: Any
    conductivity: Any = 1
    heat_capacity: Any = 1
    dim: int = 1

    def __post_init__(self) -> None:
        if self.dim not in (1, 2):
            raise ValueError("dim must be 1 or 2")
        allowed = {x, t} if self.dim == 1 else {x, y, t}
        for name in ("temperature", "conductivity", "heat_capacity"):
            free = sympy.sympify(getattr(self, name)).free_symbols
            if not free <= allowed:
                raise ValueError(f"{name} uses symbols {free - allowed} outside {allowed}")
        for name in ("conductivity", "heat_capacity"):
            if t in sympy.sympify(getattr(self, name)).free_symbols:
                raise ValueError(f"{name} must not depend on time")

    @property
    def space(self) -> tuple[Any, ...]:
        return (x,) if self.dim == 1 else (x, y)

    @property
    def source_expr(self) -> Any:
        """``rho_c dT/dt - div(k grad T)``."""
        T = sympy.sympify(self.temperature)
        k = sympy.sympify(self.conductivity)
        rho_c = sympy.sympify(self.heat_capacity)
        divergence = sum(sympy.diff(k * sympy.diff(T, v), v) for v in self.space)
        return rho_c * sympy.diff(T, t) - divergence

    def exact(self) -> Callable[..., FloatArray]:
        """``T(x, t)`` or ``T(x, y, t)`` as a numpy function."""
        return _compile((*self.space, t), self.temperature)

    def source(self) -> Callable[..., FloatArray]:
        """``q(x, t)`` or ``q(x, y, t)`` as a numpy function."""
        return _compile((*self.space, t), self.source_expr)

    def conductivity_function(self) -> Callable[..., FloatArray]:
        return _compile(self.space, self.conductivity)

    def heat_capacity_function(self) -> Callable[..., FloatArray]:
        return _compile(self.space, self.heat_capacity)

    # Boundary data -----------------------------------------------------------

    def _side(self, side: str) -> tuple[Any, Any, int]:
        """Return the normal variable, the tangential variable (or None) and the normal sign."""
        normals = {"left": (x, -1), "right": (x, 1), "bottom": (y, -1), "top": (y, 1)}
        sides = ("left", "right") if self.dim == 1 else ("left", "right", "bottom", "top")
        if side not in sides:
            raise ValueError(f"unknown side {side!r}; expected one of {sides}")
        normal, sign = normals[side]
        if self.dim == 1:
            return normal, None, sign
        return normal, (y if normal == x else x), sign

    def _side_function(self, expr: Any, tangential: Any) -> Callable[..., FloatArray]:
        symbols = (t,) if tangential is None else (tangential, t)
        return _compile(symbols, expr)

    def boundary(
        self, side: str, position: float, kind: Kind, h: float | None = None
    ) -> BoundaryCondition:
        """The condition of ``kind`` on ``side`` (at ``position``) satisfied by the exact solution.

        Neumann data is the inflow ``k dT/dn`` with ``n`` the outward normal;
        Robin data is the ambient ``T + (k / h) dT/dn``.
        """
        normal, tangential, sign = self._side(side)
        T = sympy.sympify(self.temperature)
        k = sympy.sympify(self.conductivity)
        on_side = {normal: sympy.Float(position)}
        inflow = (sign * k * sympy.diff(T, normal)).subs(on_side)
        if kind == "dirichlet":
            return Dirichlet(self._side_function(T.subs(on_side), tangential))
        if kind == "neumann":
            return Neumann(self._side_function(inflow, tangential))
        if kind == "robin":
            if h is None:
                raise ValueError("a Robin condition needs the heat transfer coefficient h")
            ambient = T.subs(on_side) + inflow / sympy.Float(h)
            return Robin(h, self._side_function(ambient, tangential))
        raise ValueError(f"unknown kind {kind!r}; expected 'dirichlet', 'neumann' or 'robin'")

    # Problems ----------------------------------------------------------------

    def problem_1d(
        self,
        grid: Grid1D,
        left: Kind = "dirichlet",
        right: Kind = "dirichlet",
        h: float | None = None,
    ) -> Problem1D:
        """A :class:`Problem1D` on ``grid`` whose exact solution is :attr:`temperature`."""
        if self.dim != 1:
            raise ValueError("problem_1d needs a manufactured solution with dim=1")
        return Problem1D(
            grid,
            left=self.boundary("left", grid.start, left, h),
            right=self.boundary("right", grid.stop, right, h),
            conductivity=self.conductivity_function(),
            heat_capacity=self.heat_capacity_function(),
            source=self.source(),
        )

    def problem_2d(
        self, grid: Grid2D, kinds: Mapping[str, Kind], h: float | None = None
    ) -> Problem2D:
        """A :class:`Problem2D` on ``grid``; ``kinds`` maps each side to its kind of condition."""
        if self.dim != 2:
            raise ValueError("problem_2d needs a manufactured solution with dim=2")
        positions = {
            "left": grid.x.start,
            "right": grid.x.stop,
            "bottom": grid.y.start,
            "top": grid.y.stop,
        }
        conditions = {
            side: self.boundary(side, positions[side], kinds.get(side, "dirichlet"), h)
            for side in positions
        }
        return Problem2D(
            grid,
            left=conditions["left"],
            right=conditions["right"],
            bottom=conditions["bottom"],
            top=conditions["top"],
            conductivity=self.conductivity_function(),
            heat_capacity=self.heat_capacity_function(),
            source=self.source(),
        )
