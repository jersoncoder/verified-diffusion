"""Finite-volume transient heat conduction in 1D and 2D, with code verification."""

from .boundary import BoundaryCondition, Dirichlet, Neumann, Robin
from .discretization import FiniteVolumeSystem, assemble
from .grid import Grid1D, Grid2D
from .problem import Problem1D, Problem2D
from .solver import (
    SCHEMES,
    StabilityError,
    SteadyBalance,
    SteadySolution,
    StepBalance,
    TransientSolution,
    TransientSolver,
    solve_steady,
)

__version__ = "0.1.0"

__all__ = [
    "SCHEMES",
    "BoundaryCondition",
    "Dirichlet",
    "FiniteVolumeSystem",
    "Grid1D",
    "Grid2D",
    "Neumann",
    "Problem1D",
    "Problem2D",
    "Robin",
    "StabilityError",
    "SteadyBalance",
    "SteadySolution",
    "StepBalance",
    "TransientSolution",
    "TransientSolver",
    "assemble",
    "solve_steady",
]
