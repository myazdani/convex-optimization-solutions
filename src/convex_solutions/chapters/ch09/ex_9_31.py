"""Exercise 9.31: stale and diagonal Hessians for analytic centering."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from convex_solutions.io import write_json
from .ex_9_30 import AnalyticCenter, backtracking, make_instance, newton_method


def hessian_diagonal(problem: AnalyticCenter, x: np.ndarray) -> np.ndarray:
    if not problem.feasible(x):
        raise ValueError("x must be strictly feasible")
    slack = 1 - problem.A @ x
    return np.sum((problem.A / slack[:, None]) ** 2, axis=0) + 2 * (1 + x * x) / (1 - x * x) ** 2


def minimize(problem: AnalyticCenter, method: str = "reuse", *, period: int = 5,
             tolerance: float = 1e-7, alpha: float = 0.1, beta: float = 0.5,
             max_iterations: int = 30000) -> dict:
    """Compare all directions under the identical gradient-norm stopping rule."""
    if method not in {"gradient", "newton", "reuse", "diagonal"}:
        raise ValueError("Unknown method")
    if not isinstance(period, int) or period < 1:
        raise ValueError("period must be a positive integer")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite")
    if max_iterations < 0:
        raise ValueError("max_iterations must be nonnegative")
    x = np.zeros(problem.n)
    iterates, values, norms, margins, steps = [], [], [], [], []
    factorizations = diagonal_evaluations = total_backtracks = 0
    status = "maximum_iterations"
    factor = None
    for k in range(max_iterations + 1):
        g = problem.gradient(x)
        iterates.append(x.copy())
        values.append(problem.value(x))
        norms.append(float(np.linalg.norm(g)))
        margins.append(problem.margin(x))
        if norms[-1] <= tolerance:
            status = "converged"
            break
        if k == max_iterations:
            break
        if method == "gradient":
            d = -g
        elif method == "diagonal":
            d = -g / hessian_diagonal(problem, x)
            diagonal_evaluations += 1
        else:
            if factor is None or method == "newton" or k % period == 0:
                factor = cho_factor(problem.hessian(x), lower=True, check_finite=False)
                factorizations += 1
            d = cho_solve(factor, -g, check_finite=False)
        step, count = backtracking(problem, x, d, g, alpha=alpha, beta=beta)
        if np.array_equal(x + step * d, x):
            status = "floating_point_stagnation"
            break
        steps.append(step)
        total_backtracks += count
        x = x + step * d
    return {"method": method, "period": period if method == "reuse" else None,
            "x": x.tolist(), "iterates": np.asarray(iterates).tolist(),
            "objectives": values, "gradient_norms": norms, "margins": margins,
            "step_lengths": steps, "iterations": len(steps), "status": status,
            "factorizations": factorizations, "hessian_evaluations": factorizations,
            "diagonal_evaluations": diagonal_evaluations, "total_backtracks": total_backtracks,
            "alpha": alpha, "beta": beta, "gradient_tolerance": tolerance}


def run_experiment(output_dir: Path | None = None, seed: int = 0,
                   dimensions=((30, 10), (100, 30), (300, 100)), periods=(2, 5, 10, 20)) -> dict:
    instances = []
    for index, (m, n) in enumerate(dimensions):
        problem = make_instance(m, n, seed + index)
        reference = newton_method(problem, tolerance=1e-25)
        if not reference.converged:
            raise RuntimeError("Reference did not converge")
        runs = []
        specs = [("gradient", 1), ("newton", 1), ("diagonal", 1)] + [("reuse", p) for p in periods]
        for method, period in specs:
            run = minimize(problem, method, period=period)
            if run["status"] != "converged":
                raise RuntimeError(f"{method} N={period} did not converge for ({m},{n})")
            run["objective_gaps"] = [problem.difference(reference.x, np.asarray(x) - reference.x)
                                      for x in run.pop("iterates")]
            run["label"] = f"reuse N={period}" if method == "reuse" else method
            runs.append(run)
        instances.append({"m": m, "n": n, "seed": seed + index,
                          "p_reference": problem.value(reference.x), "runs": runs})
    result = {"exercise": "9.31", "seed": seed, "instances": instances,
              "source": "https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=533"}
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        from convex_solutions.plotting import pyplot
        plt = pyplot()
        fig, axes = plt.subplots(2, len(instances), figsize=(14, 8), squeeze=False, constrained_layout=True)
        for column, instance in enumerate(instances):
            for run in instance["runs"]:
                axes[0, column].semilogy(np.maximum(run["objective_gaps"], 1e-20), label=run["label"])
            labels = [r["label"] for r in instance["runs"]]
            counts = [r["iterations"] for r in instance["runs"]]
            factors = [r["factorizations"] for r in instance["runs"]]
            positions = np.arange(len(labels))
            axes[1, column].bar(positions - 0.18, counts, width=0.36, label="updates", color="#2563eb")
            axes[1, column].bar(positions + 0.18, factors, width=0.36, label="Cholesky factorizations", color="#db2777")
            axes[1, column].set_xticks(positions, labels, rotation=60, ha="right")
            axes[1, column].set_yscale("symlog", linthresh=1)
            axes[0, column].set(title=f"m={instance['m']}, n={instance['n']}", xlabel="Accepted updates", ylabel="Objective gap (floor 1e−20)")
            axes[1, column].set_ylabel("Count (symlog scale)")
            for row in range(2):
                axes[row, column].grid(alpha=0.2)
                axes[row, column].legend(fontsize=7)
        fig.suptitle("Exercise 9.31: identical ‖gradient‖₂ ≤ 1e−7 criterion")
        fig.savefig(output / "approximate_newton_comparison.png", dpi=160)
        plt.close(fig)
        write_json(output / "results.json", result)
    return result


def solve(output_dir: Path | None = None, seed: int = 0) -> dict:
    return run_experiment(output_dir, seed)
