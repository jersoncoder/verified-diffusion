"""Closed-form and series solutions used to validate the solver.

* :func:`semi_infinite_step`: semi-infinite solid whose surface temperature
  jumps at ``t = 0`` (error-function solution).
* :func:`slab_convection`: plane wall cooled by convection on one face and
  insulated on the other (Fourier series over the roots of
  ``lambda tan(lambda) = Bi``).
* :func:`rectangular_plate`: steady temperature in a rectangle held at zero
  on three sides with a prescribed profile on the top (Fourier sine series).
* :func:`composite_wall`: steady conduction through layers of different
  conductivity (piecewise-linear profile).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.special import erfc

FloatArray = NDArray[np.float64]


def semi_infinite_step(
    x: ArrayLike, t: float, *, alpha: float, T_initial: float, T_surface: float
) -> FloatArray:
    """``T = T_i + (T_s - T_i) erfc(x / (2 sqrt(alpha t)))`` for ``x >= 0``, ``t > 0``."""
    if not t > 0.0:
        raise ValueError("the semi-infinite solution is defined for t > 0")
    eta = np.asarray(x, dtype=np.float64) / (2.0 * math.sqrt(alpha * t))
    return np.asarray(T_initial + (T_surface - T_initial) * erfc(eta), dtype=np.float64)


def slab_convection_eigenvalues(biot: float, n: int) -> FloatArray:
    """First ``n`` positive roots of ``lambda tan(lambda) = Bi``.

    The m-th root lies in ``(m pi, m pi + pi/2)``; the equation is solved as
    ``lambda sin(lambda) - Bi cos(lambda) = 0`` to avoid the poles of tan.
    """
    if not biot > 0.0:
        raise ValueError("the Biot number must be positive")

    def f(lam: float) -> float:
        return lam * math.sin(lam) - biot * math.cos(lam)

    roots = [
        float(brentq(f, m * math.pi, m * math.pi + 0.5 * math.pi, xtol=1e-15, rtol=1e-15))
        for m in range(n)
    ]
    return np.array(roots)


def slab_convection(
    x: ArrayLike,
    t: float,
    *,
    length: float,
    alpha: float,
    conductivity: float,
    h: float,
    T_initial: float,
    T_ambient: float,
    n_terms: int = 200,
) -> FloatArray:
    """Plane wall cooled by convection.

    The wall ``0 <= x <= length`` starts at ``T_initial``; the face ``x = 0``
    is insulated and the face ``x = length`` exchanges heat with an ambient
    ``T_ambient`` through the coefficient ``h``:

    ``(T - T_inf) / (T_i - T_inf) = sum_n C_n exp(-lambda_n^2 Fo) cos(lambda_n x / L)``
    with ``Fo = alpha t / L^2``, ``Bi = h L / k`` and
    ``C_n = 4 sin(lambda_n) / (2 lambda_n + sin(2 lambda_n))``.
    """
    if not t > 0.0:
        raise ValueError("evaluate the series at t > 0, where it converges rapidly")
    lam = slab_convection_eigenvalues(h * length / conductivity, n_terms)
    coeff = 4.0 * np.sin(lam) / (2.0 * lam + np.sin(2.0 * lam))
    fourier = alpha * t / length**2
    xi = np.asarray(x, dtype=np.float64)[..., None] / length
    theta = np.sum(coeff * np.exp(-(lam**2) * fourier) * np.cos(lam * xi), axis=-1)
    return np.asarray(T_ambient + (T_initial - T_ambient) * theta, dtype=np.float64)


def parabola_sine_coefficients(T_peak: float, n_terms: int) -> FloatArray:
    """Sine coefficients of ``4 T_peak s (W - s) / W^2`` on ``[0, W]``.

    ``b_n = 32 T_peak / (n pi)^3`` for odd ``n`` and 0 for even ``n``
    (``n = 1 .. n_terms``).
    """
    n = np.arange(1, n_terms + 1)
    return np.where(n % 2 == 1, 32.0 * T_peak / (n * math.pi) ** 3, 0.0)


def rectangular_plate(
    X: ArrayLike, Y: ArrayLike, *, width: float, height: float, top_coefficients: ArrayLike
) -> FloatArray:
    """Steady ``T`` in ``[0, W] x [0, H]``: zero on three sides, ``f(x)`` on ``y = H``.

    ``top_coefficients[n - 1]`` is the n-th sine coefficient of ``f``, and
    ``T = sum_n b_n sin(n pi x / W) sinh(n pi y / W) / sinh(n pi H / W)``. The
    sinh ratio is evaluated as exponentials so large ``n`` do not overflow.
    """
    b = np.asarray(top_coefficients, dtype=np.float64)
    a = np.arange(1, b.size + 1) * math.pi / width
    Xa = np.asarray(X, dtype=np.float64)[..., None]
    Ya = np.asarray(Y, dtype=np.float64)[..., None]
    ratio = np.exp(a * (Ya - height)) * (-np.expm1(-2.0 * a * Ya)) / (-np.expm1(-2.0 * a * height))
    return np.asarray(np.sum(b * np.sin(a * Xa) * ratio, axis=-1), dtype=np.float64)


def composite_wall(
    x: ArrayLike,
    *,
    thicknesses: Sequence[float],
    conductivities: Sequence[float],
    T_left: float,
    T_right: float,
    start: float = 0.0,
) -> tuple[FloatArray, float]:
    """Steady profile through layers in series and the heat flux through them.

    The flux is ``q = (T_left - T_right) / sum(L_i / k_i)`` and the profile is
    linear inside each layer, continuous at the interfaces.
    """
    L = np.asarray(thicknesses, dtype=np.float64)
    k = np.asarray(conductivities, dtype=np.float64)
    if L.shape != k.shape or L.ndim != 1 or L.size == 0:
        raise ValueError("thicknesses and conductivities must be non-empty and of equal length")
    q = (T_left - T_right) / float(np.sum(L / k))
    interfaces = start + np.concatenate([[0.0], np.cumsum(L)])
    T_interfaces = T_left - q * np.concatenate([[0.0], np.cumsum(L / k)])
    xa = np.asarray(x, dtype=np.float64)
    layer = np.clip(np.searchsorted(interfaces, xa, side="right") - 1, 0, L.size - 1)
    T = T_interfaces[layer] - q * (xa - interfaces[layer]) / k[layer]
    return np.asarray(T, dtype=np.float64), q
