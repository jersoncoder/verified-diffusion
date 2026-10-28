"""Run every verification study and write docs/verification.md.

Usage (from the repository root, with the dev extra installed):

    python scripts/verification_report.py

The report holds the convergence tables of every study, the GCI analysis and
the energy-balance residuals. If matplotlib is installed it also writes PNG
plots to docs/img/. The summary table between the markers in README.md is
rewritten from the same run, so the README always shows numbers this script
produced.
"""

from __future__ import annotations

import argparse
import math
import platform
import re
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import scipy
from scipy.special import erfc

from verified_diffusion import studies
from verified_diffusion.studies import BalanceRow, CompositeWallResult, GCIStudy
from verified_diffusion.verification import ConvergenceStudy, observed_orders

ROOT = Path(__file__).resolve().parent.parent
README_START = "<!-- verification-summary:start -->"
README_END = "<!-- verification-summary:end -->"

# Categorical colours (fixed order) and marker shapes as a second channel.
COLOURS = ("#2a78d6", "#eb6834", "#1baf7a")
MARKERS = ("o", "s", "^")
INK = "#52514e"
GRID = "#e4e3df"


@dataclass(frozen=True)
class Results:
    space: list[ConvergenceStudy]
    time: list[ConvergenceStudy]
    analytic: list[ConvergenceStudy]
    composite: CompositeWallResult
    gci: list[GCIStudy]
    balance: list[BalanceRow]
    seconds: float


def collect() -> Results:
    start = time.perf_counter()
    space = [
        studies.steady_space_1d_uniform(),
        studies.steady_space_1d_stretched(),
        studies.transient_space_1d(),
        studies.steady_space_2d(),
        studies.transient_space_2d(),
    ]
    temporal = [
        studies.time_order("crank-nicolson"),
        studies.time_order("implicit-euler"),
        studies.time_order("explicit-euler"),
    ]
    analytic = [
        studies.erfc_study("crank-nicolson", startup_steps=0),
        studies.erfc_study("crank-nicolson", startup_steps=2),
        studies.erfc_study("implicit-euler", startup_steps=0),
        studies.slab_study(startup_steps=0),
        studies.slab_study(startup_steps=2),
        studies.plate_study(),
    ]
    return Results(
        space=space,
        time=temporal,
        analytic=analytic,
        composite=studies.composite_wall_study(),
        gci=studies.gci_study(),
        balance=studies.energy_balance_study(),
        seconds=time.perf_counter() - start,
    )


# Markdown --------------------------------------------------------------------


def _orders(values: Iterable[float]) -> list[str]:
    return ["", *(f"{p:.2f}" for p in values)]


def study_table(study: ConvergenceStudy) -> str:
    level = "cells per direction" if study.parameter == "h" else "steps"
    size = "h" if study.parameter == "h" else "dt"
    rows = [
        f"| {level} | {size} | L2 error | order | max error | order |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    o2 = _orders(study.orders("l2"))
    om = _orders(study.orders("max"))
    for i, n in enumerate(study.levels):
        rows.append(
            f"| {n} | {study.sizes[i]:.4g} | {study.errors_l2[i]:.3e} | {o2[i]} "
            f"| {study.errors_max[i]:.3e} | {om[i]} |"
        )
    return "\n".join(rows)


NOTES = {
    "steady-2d-stretched": "max norm converges slowly near a corner, see Order in space",
    "transient-2d-cn": "max norm converges slowly near a corner, see Order in space",
    "erfc-crank-nicolson-0": "no start-up steps, on purpose: the known failure mode",
    "slab-0": "no start-up steps, on purpose",
}


def summary_table(results: Results) -> str:
    rows = [
        "| study | expected | observed, L2 | observed, max | note |",
        "|---|---:|---:|---:|---|",
    ]
    for study in [*results.space, *results.time, *results.analytic]:
        rows.append(
            f"| {study.description} | {study.expected_order:.0f} "
            f"| {study.finest_order('l2'):.2f} | {study.finest_order('max'):.2f} "
            f"| {NOTES.get(study.name, '')} |"
        )
    return "\n".join(rows)


def composite_table(result: CompositeWallResult) -> str:
    sizes = [1.0 / n for n in result.cells_per_layer]
    orders = _orders(observed_orders(sizes, result.arithmetic))
    rows = [
        "| cells per layer | harmonic, uniform | harmonic, stretched | arithmetic | order |",
        "|---:|---:|---:|---:|---:|",
    ]
    for i, n in enumerate(result.cells_per_layer):
        rows.append(
            f"| {n} | {result.harmonic[i]:.1e} | {result.harmonic_stretched[i]:.1e} "
            f"| {result.arithmetic[i]:.3e} | {orders[i]} |"
        )
    return "\n".join(rows)


def gci_table(items: list[GCIStudy]) -> str:
    rows = [
        "| quantity | grids (cells) | values | p (finest three) | p (coarsest three) "
        "| extrapolated | GCI fine |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for item in items:
        cells = ", ".join(str(c) for c in item.cells)
        values = ", ".join(f"{v:.6f}" for v in item.values)
        rows.append(
            f"| {item.name} | {cells} | {values} | {item.fine.order:.3f} "
            f"| {item.coarse.order:.3f} | {item.fine.extrapolated:.6f} "
            f"| {100 * item.fine.gci_fine:.4f} % |"
        )
    return "\n".join(rows)


def balance_table(rows: list[BalanceRow]) -> str:
    out = [
        "| case | scheme | steps | stored | boundary | source | worst step residual / scale |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        out.append(
            f"| {row.case} | {row.scheme} | {row.steps} | {row.stored:.6f} "
            f"| {row.boundary:.6f} | {row.source:.6f} | {row.max_relative_residual:.1e} |"
        )
    return "\n".join(out)


def _worst_cell_text(study: ConvergenceStudy) -> tuple[str, float]:
    """The worst cell of a stretched 2D study as text, and its height-to-width ratio."""
    grid = studies._stretched_2d(study.levels[-1])
    x, y = study.worst_cell
    i = int(np.argmin(np.abs(grid.x.centres - x)))
    j = int(np.argmin(np.abs(grid.y.centres - y)))
    return f"$({x:.3f}, {y:.3f})$", float(grid.y.widths[j] / grid.x.widths[i])


def environment() -> str:
    return (
        f"Python {platform.python_version()}, numpy {np.__version__}, "
        f"scipy {scipy.__version__}, {platform.system()} {platform.machine()}"
    )


def render(results: Results, figures: list[str]) -> str:
    space = "\n\n".join(
        f"**{s.description}** (`{s.name}`)\n\n{study_table(s)}" for s in results.space
    )
    temporal = "\n\n".join(
        f"**{s.description}** (`{s.name}`)\n\n{study_table(s)}" for s in results.time
    )
    analytic = "\n\n".join(
        f"**{s.description}** (`{s.name}`)\n\n{study_table(s)}" for s in results.analytic
    )
    figure_md = "\n\n".join(f"![{name}](img/{name}.png)" for name in figures)
    steady, transient = results.space[3], results.space[4]
    steady_cell, steady_aspect = _worst_cell_text(steady)
    transient_cell, transient_aspect = _worst_cell_text(transient)
    steady_max = ", ".join(f"{p:.2f}" for p in steady.orders("max"))
    transient_max = ", ".join(f"{p:.2f}" for p in transient.orders("max"))
    erfc_eta = studies.ERFC_LENGTH / (2.0 * math.sqrt(studies.ERFC_ALPHA * studies.ERFC_END))
    erfc_far = float(erfc(erfc_eta))
    gci_spread = max(abs(g.fine.order - g.coarse.order) for g in results.gci)
    figure_section = (
        f"## Plots\n\n{figure_md}\n"
        if figures
        else "## Plots\n\nmatplotlib was not installed when this report was generated.\n"
    )
    erfc_end, erfc_length = studies.ERFC_END, studies.ERFC_LENGTH
    return f"""# Verification report

Generated by `python scripts/verification_report.py` ({environment()}).
Every number below comes from that run; the test suite
(`tests/test_studies.py`) runs the same study functions and asserts on the same
quantities. Orders are observed between consecutive levels:
$p = \\ln(e_k / e_{{k+1}}) / \\ln(h_k / h_{{k+1}})$. The L2 error is volume weighted,
$\\lVert e \\rVert_2 = \\sqrt{{\\sum_i V_i e_i^2 / \\sum_i V_i}}$, and the max error is
taken over cell centres.

## Summary

Observed order between the two finest levels.

{summary_table(results)}

## Order in space (manufactured solutions)

Sources and boundary data are derived with sympy from the chosen solutions in
`src/verified_diffusion/studies.py`. Transient cases use Crank-Nicolson with
$\\Delta t = h/4$, so time and space errors both fall as $h^2$. Stretched grids use
the tanh mapping of `Grid1D.stretched` ($\\beta = 2$ in 1D, $\\beta = 1.5$ in 2D).

{space}

On the stretched 2D grid the max-norm order approaches 2 more slowly than the L2
order. On the finest level the largest error of the steady case is in the cell
centred at {steady_cell} and that of the transient case in the cell centred at
{transient_cell}. Both are in the top row, where the y-grid has its tallest
cells, close to a top corner, where the x-grid has its narrowest ones; those two
cells are {steady_aspect:.1f} and {transient_aspect:.1f} times taller than they are
wide. The max-norm orders of the steady case, level by level, are {steady_max};
those of the transient case are {transient_max}. The tests hold the 2D max-norm
order to a floor of {studies.STRETCHED_2D_MAX_NORM_FLOOR} and every other order to
$\\pm {studies.ORDER_TOLERANCE}$ of the theory.

## Order in time

The grid is fixed (16 uniform cells) and the error is measured against the
time-exact solution of the same semi-discrete system,
$M\\,\\dot T = -A\\,T + b(t) + s(t)$, integrated by scipy's Radau method at a relative
tolerance of $10^{{-12}}$. The spatial error is common to both and cancels, so what
remains is the error of the time integrator alone, including its treatment of
time-dependent Neumann and Robin data and a time-dependent source. Explicit Euler
starts at the largest equal step within its stability limit.

{temporal}

## Analytic solutions

**Semi-infinite solid with a step surface temperature.** The surface of a solid at
0 is raised to 1 at $t = 0$; $T = \\operatorname{{erfc}}(x / 2\\sqrt{{\\alpha t}})$,
compared at $t = {erfc_end:g}$ on $[0, {erfc_length:g}]$ (the far boundary, held at 0, sits at
$x / 2\\sqrt{{\\alpha t}} = {erfc_eta:.2f}$, where the exact solution is
{erfc_far:.1e}). The time step is proportional to $h$. Plain Crank-Nicolson does not
converge in the max norm: the jump at $t = 0$ excites the stiffest modes, whose
Crank-Nicolson amplification
factor $(1 - \\lambda \\Delta t/2)/(1 + \\lambda \\Delta t/2)$ tends to $-1$, so an
oscillation next to the surface survives every refinement. Replacing the first two
steps by two implicit Euler half steps each (Rannacher start-up,
`run(..., startup_steps=2)`) removes it and restores second order; implicit Euler
alone converges at first order.

**Plane wall with convection.** A wall at 1, insulated at $x = 0$ and cooled by
convection to 0 at $x = 1$ ($Bi = 5$), compared at $Fo = 0.3$ with the Fourier
series over the roots of $\\lambda \\tan \\lambda = Bi$. The initial temperature is
incompatible with the Robin condition, a milder version of the step above.

**Steady plate.** The unit square held at 0 on three sides with $4x(1 - x)$ on
the top, compared with its sine series (1001 terms).

{analytic}

## Composite wall

Two layers, $k = 1$ on $[0, 0.3]$ and $k = 20$ on $[0.3, 1]$, held at 0 and 1,
with the interface on a cell face. The exact profile is piecewise linear. With
harmonic face averaging (resistances in series) the scheme reproduces it to
round-off on uniform and stretched layers; distance-weighted arithmetic averaging
gets the interface flux wrong and converges at first order. Max error at the
cell centres:

{composite_table(results.composite)}

## Grid Convergence Index

A 2 by 1 plate with a Gaussian heat source, a conductivity that varies with
position, the left side at 0, convection on the right ($h = 10$), an insulated
bottom and a prescribed heat loss on the top: no closed-form solution. Four grids
with refinement ratio 2. The observed orders from the three finest and from the
three coarsest grids differ by at most {gci_spread:.3f}, so the grids are in the
asymptotic range. GCI with safety factor 1.25 (Roache).

{gci_table(results.gci)}

## Energy balance

Each step reports the change in stored energy, the heat that entered through each
side and the heat released by the source. Their residual divided by the magnitude
of the terms entering the step (`StepBalance.scale`) is at round-off level for
every scheme. The columns give the totals over a run to $t = 0.1$ (J per unit
area in 1D, J per unit depth in 2D).

{balance_table(results.balance)}

{figure_section}"""


# Plots -----------------------------------------------------------------------


def write_plots(results: Results, directory: Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.axes import Axes
        from matplotlib.ticker import NullFormatter
    except ImportError:
        return []

    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
        }
    )

    def panel(
        ax: Axes,
        series: list[ConvergenceStudy],
        labels: list[str],
        norm: str,
        references: Sequence[tuple[int, float]],
        title: str,
        legend: Literal["lower right", "lower left"] = "lower right",
    ) -> None:
        for i, (study, label) in enumerate(zip(series, labels, strict=True)):
            errors = study.errors_l2 if norm == "l2" else study.errors_max
            ax.loglog(
                study.sizes,
                errors,
                color=COLOURS[i],
                marker=MARKERS[i],
                markersize=5,
                linewidth=2,
                markeredgecolor="white",
                markeredgewidth=1,
                label=label,
            )
        # A reference line of the expected slope under each chosen series.
        for index, slope in references:
            study = series[index]
            errors = study.errors_l2 if norm == "l2" else study.errors_max
            h = np.array(study.sizes)
            ref = 0.4 * errors[-1] * (h / h[-1]) ** slope
            ax.loglog(h, ref, color="#9a9893", linewidth=1)
            ax.annotate(
                f"slope {slope:g}",
                (h[-1], ref[-1]),
                textcoords="offset points",
                xytext=(8, -2),
                ha="left",
                va="top",
                color=INK,
                fontsize=8,
            )
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(title, color="#0b0b0b", fontsize=10, loc="left")
        ax.set_xlabel("h (domain length / cells)" if series[0].parameter == "h" else "dt")
        ax.set_ylabel("L2 error" if norm == "l2" else "max error")
        ax.legend(fontsize=8, loc=legend)

    names = []
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    panel(
        axes[0],
        results.space[:3],
        ["steady, uniform", "steady, stretched, variable k", "transient (CN), stretched"],
        "l2",
        [(2, 2.0)],
        "1D manufactured solutions",
    )
    panel(
        axes[1],
        [*results.space[3:], results.analytic[-1]],
        ["steady, stretched, variable k", "transient (CN), stretched", "plate, series"],
        "l2",
        [(1, 2.0)],
        "2D",
    )
    fig.savefig(directory / "convergence_space.png", dpi=150, facecolor="white")
    plt.close(fig)
    names.append("convergence_space")

    fig, ax = plt.subplots(figsize=(4.8, 3.6), constrained_layout=True)
    panel(
        ax,
        results.time,
        ["Crank-Nicolson", "implicit Euler", "explicit Euler"],
        "l2",
        [(0, 2.0), (1, 1.0), (2, 1.0)],
        "Order in time (16 cells)",
        legend="lower left",
    )
    fig.savefig(directory / "convergence_time.png", dpi=150, facecolor="white")
    plt.close(fig)
    names.append("convergence_time")

    fig, ax = plt.subplots(figsize=(4.8, 3.6), constrained_layout=True)
    panel(
        ax,
        results.analytic[:3],
        ["Crank-Nicolson", "Crank-Nicolson, 2 start-up steps", "implicit Euler"],
        "max",
        [(1, 2.0), (2, 1.0)],
        "Step surface temperature (erfc)",
    )
    fig.savefig(directory / "erfc_startup.png", dpi=150, facecolor="white")
    plt.close(fig)
    names.append("erfc_startup")
    return names


# README ----------------------------------------------------------------------


def update_readme(path: Path, results: Results) -> bool:
    """Replace the summary block between the README markers; False if there are none."""
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(README_START) + ".*?" + re.escape(README_END), re.DOTALL)
    if not pattern.search(text):
        return False
    block = f"{README_START}\n{summary_table(results)}\n{README_END}"
    path.write_text(pattern.sub(lambda _m: block, text), encoding="utf-8")
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "verification.md")
    parser.add_argument("--readme", type=Path, default=ROOT / "README.md")
    parser.add_argument("--no-plots", action="store_true", help="skip the PNG plots")
    args = parser.parse_args(argv)

    results = collect()
    figures = [] if args.no_plots else write_plots(results, args.output.parent / "img")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(results, figures), encoding="utf-8")
    print(f"wrote {args.output} ({results.seconds:.1f} s of studies)")
    for name in figures:
        print(f"wrote {args.output.parent / 'img' / name}.png")
    if args.readme.exists() and update_readme(args.readme, results):
        print(f"updated the summary table in {args.readme}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
