"""Richardson extrapolation and the Grid Convergence Index (Roache).

For a quantity ``f`` computed on three grids, fine (``f1``), medium (``f2``)
and coarse (``f3``), with refinement ratios ``r21 = h2 / h1`` and
``r32 = h3 / h2``, the observed order ``p`` solves

    ln|e32 / e21| = p ln(r21) + ln((r32^p - s) / (r21^p - s))

with ``e21 = f2 - f1``, ``e32 = f3 - f2`` and ``s = sign(e32 / e21)``. For a
constant ratio this reduces to ``p = ln|e32 / e21| / ln(r)``. It is the
equation of Celik et al., "Procedure for estimation and reporting of
uncertainty due to discretization in CFD applications", J. Fluids Eng. (2008),
which builds on Roache, "Verification and Validation in Computational Science
and Engineering" (1998). Two deliberate differences: the equation is solved by
bracketing instead of fixed-point iteration (which diverges when ``r21`` is
close to 1 and ``r32`` is not), and a sequence whose differences grow under
refinement is rejected instead of being given ``|p|``.

The Richardson extrapolate is ``f_ext = (r21^p f1 - f2) / (r21^p - 1)`` and the
fine-grid GCI is ``Fs |e21 / f1| / (r21^p - 1)`` with the safety factor
``Fs = 1.25`` for a three-grid study.

With three grids the observed order makes the classic asymptotic-range check
``GCI32 = r^p GCI21`` an identity, so it is not reported; compare the orders
from two overlapping triplets of a four-grid study instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.optimize import brentq


def richardson_extrapolate(f_fine: float, f_coarse: float, ratio: float, order: float) -> float:
    """Estimate of the grid-independent value from two grids of ratio ``ratio > 1``."""
    if not ratio > 1.0:
        raise ValueError("the refinement ratio must be greater than 1")
    factor = math.pow(ratio, order)
    return (factor * f_fine - f_coarse) / (factor - 1.0)


def observed_order(f1: float, f2: float, f3: float, r21: float, r32: float | None = None) -> float:
    """Observed order of accuracy from three grids (fine ``f1`` to coarse ``f3``)."""
    r32 = r21 if r32 is None else r32
    if not (r21 > 1.0 and r32 > 1.0):
        raise ValueError("refinement ratios must be greater than 1")
    e21, e32 = f2 - f1, f3 - f2
    if e21 == 0.0 or e32 == 0.0:
        raise ValueError("the solution does not change between two grids; the order is undefined")
    s = math.copysign(1.0, e32 / e21)
    target = math.log(abs(e32 / e21))
    if r21 == r32:
        p = target / math.log(r21)
    else:

        def residual(p: float) -> float:
            q = math.log((math.pow(r32, p) - s) / (math.pow(r21, p) - s))
            return p * math.log(r21) + q - target

        low, high = 1e-9, 1.0
        if residual(low) >= 0.0:
            p = 0.0
        else:
            while residual(high) < 0.0:
                high *= 2.0
                if high > 1e3:
                    raise ValueError("no finite order matches these values")
            p = float(brentq(residual, low, high, xtol=1e-14, rtol=1e-14))
    if not p > 0.0:
        raise ValueError("the differences do not shrink under refinement; the order is undefined")
    return p


@dataclass(frozen=True)
class GridConvergence:
    """Three-grid Richardson analysis of one quantity of interest.

    ``gci_fine`` and ``gci_medium`` are relative (fractions of ``f1`` and
    ``f2``); ``uncertainty`` is the absolute fine-grid band ``Fs |e21| /
    (r21^p - 1)``. ``monotone`` is False for oscillatory convergence, where
    the estimate is not reliable.
    """

    f1: float
    f2: float
    f3: float
    r21: float
    r32: float
    order: float
    monotone: bool
    extrapolated: float
    gci_fine: float
    gci_medium: float
    uncertainty: float
    safety_factor: float


def grid_convergence_index(
    f1: float,
    f2: float,
    f3: float,
    r21: float,
    r32: float | None = None,
    safety_factor: float = 1.25,
) -> GridConvergence:
    """Observed order, Richardson extrapolate and GCI.

    ``f1`` is the fine-grid value, ``f2`` the medium and ``f3`` the coarse.
    """
    r32 = r21 if r32 is None else r32
    p = observed_order(f1, f2, f3, r21, r32)
    a21, a32 = math.pow(r21, p) - 1.0, math.pow(r32, p) - 1.0
    band21 = safety_factor * abs(f2 - f1) / a21
    band32 = safety_factor * abs(f3 - f2) / a32
    return GridConvergence(
        f1=f1,
        f2=f2,
        f3=f3,
        r21=r21,
        r32=r32,
        order=p,
        monotone=(f3 - f2) / (f2 - f1) > 0.0,
        extrapolated=richardson_extrapolate(f1, f2, r21, p),
        gci_fine=band21 / abs(f1) if f1 != 0.0 else math.inf,
        gci_medium=band32 / abs(f2) if f2 != 0.0 else math.inf,
        uncertainty=band21,
        safety_factor=safety_factor,
    )
