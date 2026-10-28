# Boundary conditions: the half-cell flux and why it is second order

This note explains how the solver closes a boundary face, why that closure is
second order even though its local truncation error in a Dirichlet boundary cell
is not, and what the alternative would have cost. The convergence studies in
[verification.md](verification.md) are the evidence; this is the reasoning
behind them.

## The closure

The first cell centre sits a distance $d = \Delta x_1 / 2$ from the boundary face.
The heat flow entering the domain through that face is a two-point flux across
the half cell, with the same series-resistance form for all three conditions:

```
   boundary face                 interior face
        |<---- d ---->|<---- d ---->|<------- dx_2 ------->|
        |             o             |          o           |
       T_b           T_1                      T_2
        |  R = d / k_1 |
```

| condition | heat flow into cell 1 | conductance $G_b$ |
|---|---|---|
| Dirichlet $T = T_b$ | $G_b (T_b - T_1)$ | $A\,k_1 / d$ |
| Robin $k\,\partial T/\partial n = h (T_\infty - T)$ | $G_b (T_\infty - T_1)$ | $A / (d/k_1 + 1/h)$ |
| Neumann $k\,\partial T/\partial n = q_{in}$ | $A\,q_{in}$ | $0$ |

On a uniform grid this is the same as a mirror ghost cell with linear
extrapolation, $T_g = 2T_b - T_1$. The surface temperature of a Robin or Neumann
side follows from the same resistance (`BoundarySide.surface_temperature`).

## Local truncation error versus global error

Take a smooth solution $T(x)$ with constant $k$ near the left boundary. Taylor
expansion of the Dirichlet flux gives

$$
\frac{k\,(T_b - T(d))}{d} = -k\,T'(0) - \frac{k\,d}{2}\,T''(0) + O(d^2),
$$

so the computed boundary flux carries a defect $\tau_b = -\tfrac{k d}{2} T''(0)$ of
order $h$. Divided by the cell volume this is an $O(1)$ local truncation error in
the boundary cell, which is why this closure is sometimes called first order.

The global error is nonetheless $O(h^2)$. In a steady 1D problem the scheme is
conservative, so subtracting the discrete equations satisfied by the exact
solution from those of the numerical one leaves, for every face $f$,

$$
E_f + \tau_f = C + O(h^2),
$$

where $E_f$ is the discrete heat flow (in the $+x$ direction) of the error
$e = T_h - T$ through face $f$, $\tau_f$ the flux defect of that face (order $h^2$
at interior faces) and $C$ a constant. Across each face the error drops by its
resistance times its flow, so the error at a cell is
$e_i = -\sum_{f \le i} R_f E_f$ over the faces between the left boundary and the
cell. With the temperature fixed at both ends (a Dirichlet value, or the ambient
of a Robin side) the error vanishes at both, so the sum over all faces is zero.
That fixes

$$
C \sum_f R_f = \sum_f R_f\,\tau_f + O(h^2) = R_b\,\tau_b + O(h^2).
$$

The boundary defect enters only through the product $R_b \tau_b$, the defect
times the resistance it acts across, and for the Dirichlet closure

$$
R_b\,\tau_b = \frac{d}{k}\cdot\left(-\frac{k d}{2} T''(0)\right) = -\frac{d^2}{2}\,T''(0) = O(h^2).
$$

So $C = O(h^2)$ and every $e_i = O(h^2)$. The Robin closure gives the same
product: its resistance $d/k + 1/h$ is $O(1)$ but its defect,
$-\tfrac{d^2}{2} T''(0) / (d/k + 1/h)$, is already $O(h^2)$. A Neumann face has
no defect at all.

This argument is a 1D steady one. For 2D and transient problems the error flow
is not constant along a line, and the evidence is empirical: the manufactured
solution studies measure second order in the L2 and max norms with every kind
of side, on uniform and stretched grids. On the stretched 2D grid the max-norm
order rises towards 2 more slowly, with its largest error in the row of tall,
narrow cells along the top side, near a corner (the numbers are in
[verification.md](verification.md)).

## Why not a higher-order ghost cell

A quadratic extrapolation through $T_b$, $T_1$ and $T_2$ removes the $O(h)$ flux
defect, but the boundary flux then depends on $T_2$, so the matrix $A$ is no
longer symmetric. Symmetry is what this solver relies on:

* $M^{-1}A$ is similar to a symmetric matrix, so its eigenvalues are real and
  non-negative, the theta method is stable in the energy norm $\sum_i M_i T_i^2$
  for $\theta \ge 1/2$, and the Gershgorin bound used for the explicit stability
  limit is a bound on a real spectrum;
* the same two-point formula serves Dirichlet and Robin sides and variable
  conductivity without special cases;
* the matrix keeps its tridiagonal (1D) and five-point (2D) pattern.

Since the global error is already second order, the quadratic closure would buy
a smaller boundary error constant, not a higher order.

## Conductivity at faces

Interior faces use the same series-resistance idea,
$G_{ij} = A_f / (d_i/k_i + d_j/k_j)$, the distance-weighted harmonic mean of the
two cell conductivities. For a conductivity that jumps on a face this is exact:
a composite wall is reproduced to round-off on any grid whose faces include the
interfaces. Distance-weighted arithmetic averaging (`face_average="arithmetic"`)
is kept only for comparison; it misses the interface flux and converges at
first order.
