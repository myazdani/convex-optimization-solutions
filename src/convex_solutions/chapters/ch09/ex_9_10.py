"""Exercise 9.10: two explicit failures of undamped Newton iteration."""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from convex_solutions.io import write_json


def log_cosh_value(x: float | np.ndarray):
    return np.logaddexp(x, -np.asarray(x))


def pure_newton_log_cosh(x0: float, max_iterations: int = 20, tolerance: float = 1e-12) -> dict:
    x = float(x0)
    points = [x]
    status = "maximum_iterations"
    for _ in range(max_iterations):
        if abs(x) <= tolerance:
            status = "converged"
            break
        # Guard the *next evaluation*, rather than producing inf/NaN and
        # misleadingly reporting a completed iteration.
        if abs(2 * x) > 700:
            status = "divergent_overflow_guard"
            break
        x = x - 0.5 * math.sinh(2.0 * x)
        points.append(x)
    return {"x0": x0, "iterates": points, "objectives": log_cosh_value(np.array(points)).tolist(),
            "gradients": np.tanh(points).tolist(), "status": status}


def pure_newton_log_barrier(x0: float = 3.0, max_iterations: int = 20) -> dict:
    if x0 <= 0:
        raise ValueError("The initial point must be in the domain x > 0")
    points = [float(x0)]
    for _ in range(max_iterations):
        x = points[-1]
        if abs(x - 1) < 1e-12:
            return {"iterates": points, "status": "converged", "invalid_candidate": None}
        candidate = 2 * x - x * x
        if candidate <= 0:
            return {"iterates": points, "status": "left_domain", "invalid_candidate": candidate}
        points.append(candidate)
    return {"iterates": points, "status": "maximum_iterations", "invalid_candidate": None}


def solve(output_dir: Path | None = None, seed: int = 0) -> dict:
    del seed  # This exercise specifies every initial point and has no random data.
    threshold = brentq(lambda x: math.sinh(2 * x) - 4 * x, 0.5, 1.5)
    result = {"exercise": "9.10", "part_a": [pure_newton_log_cosh(1.0), pure_newton_log_cosh(1.1)],
              "part_b": pure_newton_log_barrier(3.0), "nonzero_two_cycle_threshold": threshold,
              "source": "https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=529"}
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        from convex_solutions.plotting import pyplot
        plt = pyplot()
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        grid = np.linspace(-3, 3, 500)
        axes[0, 0].plot(grid, log_cosh_value(grid), color="#334155", label="f(x)")
        axes[1, 0].plot(grid, np.tanh(grid), color="#334155", label="f′(x)")
        for run, color in zip(result["part_a"], ["#2563eb", "#db2777"]):
            points = np.array(run["iterates"][:4])
            for axis, values in [(axes[0, 0], log_cosh_value(points)), (axes[1, 0], np.tanh(points))]:
                axis.plot(points, values, "o--", color=color, label=f"x₀={run['x0']}, first 4 states")
                for k, (x, y) in enumerate(zip(points, values)):
                    axis.annotate(str(k), (x, y), xytext=(4, 5), textcoords="offset points", color=color)
        grid = np.linspace(0.03, 4, 500)
        axes[0, 1].plot(grid, -np.log(grid) + grid, label="f(x)", color="#334155")
        axes[1, 1].plot(grid, 1 - 1 / grid, label="f′(x)", color="#334155")
        axes[0, 1].plot([3], [3 - math.log(3)], "o", color="#db2777", label="x₀=3")
        axes[1, 1].plot([3], [2 / 3], "o", color="#db2777", label="x₀=3")
        axes[0, 1].text(0.42, 0.8, "First Newton candidate: x₁=−3\nOutside domain; iteration stops", transform=axes[0, 1].transAxes)
        axes[1, 1].set_ylim(-5, 1)
        for axis in axes.flat:
            axis.set_xlabel("x")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
        axes[0, 0].set_title("(a) log(exp(x)+exp(−x))")
        axes[0, 1].set_title("(b) −log(x)+x, domain x>0")
        fig.suptitle("Exercise 9.10: full Newton steps need not converge")
        fig.savefig(output / "pure_newton_examples.png", dpi=160)
        plt.close(fig)
        write_json(output / "results.json", result)
    return result
