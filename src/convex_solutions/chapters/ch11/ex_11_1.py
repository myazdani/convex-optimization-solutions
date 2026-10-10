"""Exercise 11.1: a one-dimensional central path and barrier plots."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from convex_solutions.plotting import pyplot
from convex_solutions.io import write_json


def central_point(t: float) -> float:
    if t <= 0:
        raise ValueError("The central-path parameter must be positive")
    return float(brentq(lambda x: 2*t*x - 1/(x-2) + 1/(4-x),
                        2 + 1e-13, 4 - 1e-13, xtol=1e-14))


def solve(output_dir: Path | None = None, seed: int = 0) -> dict:
    del seed
    parameters = [.05, .2, 1, 5, 25]
    points = np.array([central_point(t) for t in parameters])
    result = {"exercise": "11.1", "t": parameters, "central_points": points,
              "optimal_point": 2.0, "optimal_value": 5.0,
              "objective_gaps": points**2 - 4,
              "barrier_gap_bounds": 2 / np.array(parameters),
              "stationarity_residuals": [abs(2*t*x-1/(x-2)+1/(4-x))
                                         for t, x in zip(parameters, points)]}
    if output_dir is not None:
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        plt = pyplot()
        grid = np.linspace(2.001, 3.999, 1800)
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
        axes = axes.ravel()
        axes[0].plot(grid, grid**2+1, label=r"$f_0(x)=x^2+1$")
        axes[0].scatter([2], [5], c="black", label=r"$x^\star=2$")
        axes[0].legend(); axes[0].set_title("Original objective")
        for ax, t, point in zip(axes[1:], parameters, points):
            curve = t*(grid**2+1)-np.log(grid-2)-np.log(4-grid)
            value = t*(point**2+1)-np.log(point-2)-np.log(4-point)
            ax.plot(grid, curve)
            ax.scatter([point], [value], c="tab:red")
            ax.annotate(rf"$x^\star({t:g})={point:.4f}$", (point, value),
                        xytext=(.12, .82), textcoords="axes fraction",
                        arrowprops={"arrowstyle": "->"})
            ax.set_title(rf"$t f_0+\phi$, $t={t:g}$")
        for ax in axes:
            ax.set(xlabel="x", ylabel="objective", xlim=(1.95,4.05))
            ax.grid(alpha=.25)
        fig.savefig(output_dir / "barrier_central_path.png", dpi=160)
        plt.close(fig)
        write_json(output_dir / "results.json", result)
    return result
