"""The quick-start example of the README; its output is quoted there."""

from verified_diffusion import (
    Dirichlet,
    Grid1D,
    Problem1D,
    Robin,
    StabilityError,
    TransientSolver,
)

# A 5 cm steel plate at 20 C: one face is suddenly held at 100 C,
# the other loses heat to 20 C air through h = 25 W/(m^2 K).
problem = Problem1D(
    Grid1D.uniform(0.0, 0.05, 50),
    left=Dirichlet(100.0),
    right=Robin(h=25.0, ambient=20.0),
    conductivity=45.0,  # W/(m K)
    heat_capacity=3.6e6,  # rho * c, J/(m^3 K)
)

# Crank-Nicolson, with two implicit start-up steps for the sudden change at t = 0.
run = TransientSolver(problem, "crank-nicolson").run(20.0, t_end=600.0, dt=5.0, startup_steps=2)
stored, boundary, source = run.total_balance()
print(f"cell next to the cooled face after 600 s: {run.temperature[-1]:.2f} C")
print(f"heat stored {stored:.4e} J/m^2, heat that entered {boundary:.4e} J/m^2")
print(f"every step balances to round-off: {run.max_relative_residual < 1e-14}")

# Explicit Euler refuses a step beyond its stability limit.
try:
    TransientSolver(problem, "explicit-euler").run(20.0, t_end=600.0, dt=5.0)
except StabilityError as error:
    print(error)
