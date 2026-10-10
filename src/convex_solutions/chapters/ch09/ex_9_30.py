"""Exercise 9.30: reproducible analytic-center gradient/Newton experiments.

Original implementation for Boyd and Vandenberghe, *Convex Optimization*,
Exercise 9.30 (printed p. 519).  No solution-manual material is used.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cho_factor, cho_solve

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class AnalyticCenter:
    """Log barrier for A x < 1 and -1 < x < 1; rows of A are a_i^T."""

    A: FloatArray

    def __post_init__(self) -> None:
        a = np.asarray(self.A, dtype=float)
        if a.ndim != 2 or min(a.shape) == 0 or not np.all(np.isfinite(a)):
            raise ValueError("A must be a nonempty finite matrix")
        object.__setattr__(self, "A", a.copy())

    @property
    def n(self) -> int:
        return self.A.shape[1]

    def _point(self, x: ArrayLike) -> FloatArray:
        point = np.asarray(x, dtype=float)
        if point.shape != (self.n,) or not np.all(np.isfinite(point)):
            raise ValueError(f"x must be a finite vector of length {self.n}")
        return point

    def margin(self, x: ArrayLike) -> float:
        point = self._point(x)
        return float(min(np.min(1.0 - self.A @ point), np.min(1.0 - np.abs(point))))

    def feasible(self, x: ArrayLike) -> bool:
        return self.margin(x) > 0.0

    def value(self, x: ArrayLike) -> float:
        point = self._point(x)
        z = self.A @ point
        if np.any(z >= 1.0) or np.any(np.abs(point) >= 1.0):
            return float("inf")
        return float(-np.sum(np.log1p(-z)) - np.sum(np.log1p(-point * point)))

    def gradient(self, x: ArrayLike) -> FloatArray:
        point = self._point(x)
        slack = 1.0 - self.A @ point
        if np.any(slack <= 0.0) or np.any(np.abs(point) >= 1.0):
            raise ValueError("Derivatives require a strictly feasible point")
        return self.A.T @ (1.0 / slack) + 2.0 * point / (1.0 - point * point)

    def hessian(self, x: ArrayLike) -> FloatArray:
        point = self._point(x)
        slack = 1.0 - self.A @ point
        if np.any(slack <= 0.0) or np.any(np.abs(point) >= 1.0):
            raise ValueError("Derivatives require a strictly feasible point")
        weighted = self.A / slack[:, None]
        diagonal = 2.0 * (1.0 + point * point) / (1.0 - point * point) ** 2
        return weighted.T @ weighted + np.diag(diagonal)

    def difference(self, x: ArrayLike, direction: ArrayLike, step: float = 1.0) -> float:
        """Evaluate f(x + step*d)-f(x) without subtracting large objectives.

        Pairwise log-barrier ratios and extended-precision summation preserve
        useful Armijo comparisons close to the minimizer.  The ordinary
        feasibility check is still performed on the actual candidate point.
        """
        point = self._point(x)
        direction = self._point(direction)
        if not self.feasible(point):
            raise ValueError("The base point must be strictly feasible")
        if not self.feasible(point + step * direction):
            return float("inf")
        dtype = np.longdouble
        delta = dtype(step) * direction.astype(dtype)
        affine_ratio = dtype(step) * (self.A @ direction).astype(dtype) / (
            1.0 - self.A @ point
        ).astype(dtype)
        box_plus_ratio = delta / (1.0 + point).astype(dtype)
        box_minus_ratio = -delta / (1.0 - point).astype(dtype)
        if (np.any(affine_ratio >= 1) or np.any(box_plus_ratio <= -1)
                or np.any(box_minus_ratio <= -1)):
            return float("inf")
        return float(-np.sum(np.log1p(-affine_ratio), dtype=dtype)
                     - np.sum(np.log1p(box_plus_ratio), dtype=dtype)
                     - np.sum(np.log1p(box_minus_ratio), dtype=dtype))


@dataclass
class DescentResult:
    method: str
    x: FloatArray
    iterates: FloatArray
    objectives: FloatArray
    gradient_norms: FloatArray
    margins: FloatArray
    step_lengths: FloatArray
    backtracks: NDArray[np.int64]
    decrement_squared: FloatArray
    converged: bool
    termination: str
    alpha: float
    beta: float
    tolerance: float

    @property
    def iterations(self) -> int:
        """Number of accepted updates (initial point is state zero)."""
        return len(self.step_lengths)

    def as_dict(self) -> dict:
        return {
            "method": self.method, "x": self.x.tolist(),
            "iterates": self.iterates.tolist(),
            "objectives": self.objectives.tolist(),
            "gradient_norms": self.gradient_norms.tolist(),
            "margins": self.margins.tolist(),
            "step_lengths": self.step_lengths.tolist(),
            "backtracks": self.backtracks.tolist(),
            "decrement_squared": self.decrement_squared.tolist(),
            "iterations": self.iterations, "converged": self.converged,
            "termination": self.termination, "alpha": self.alpha,
            "beta": self.beta, "tolerance": self.tolerance,
        }


def make_instance(m: int = 100, n: int = 30, seed: int = 930) -> AnalyticCenter:
    """Generate independent N(0, 1) entries, with zero always strictly feasible."""
    if not isinstance(m, int) or not isinstance(n, int) or min(m, n) <= 0:
        raise ValueError("m and n must be positive integers")
    return AnalyticCenter(np.random.default_rng(seed).standard_normal((m, n)))


def backtracking(
    problem: AnalyticCenter, x: ArrayLike, direction: ArrayLike,
    gradient: ArrayLike | None = None, *, alpha: float = 0.1,
    beta: float = 0.5, max_backtracks: int = 200,
) -> tuple[float, int]:
    """Armijo backtracking from t=1, rejecting non-interior candidates first."""
    if not 0.0 < alpha < 0.5 or not 0.0 < beta < 1.0:
        raise ValueError("Require 0 < alpha < 0.5 and 0 < beta < 1")
    point = problem._point(x)
    d = problem._point(direction)
    g = problem.gradient(point) if gradient is None else problem._point(gradient)
    slope = float(g @ d)
    if slope >= 0.0 or not np.isfinite(slope):
        raise ValueError("The search direction must be a strict descent direction")
    step = 1.0
    for reductions in range(max_backtracks + 1):
        if (problem.feasible(point + step * d)
                and problem.difference(point, d, step) <= alpha * step * slope):
            return step, reductions
        step *= beta
    raise RuntimeError("Backtracking exhausted its iteration limit")


def _minimize(
    problem: AnalyticCenter, method: Literal["gradient", "newton"], *,
    x0: ArrayLike | None = None, alpha: float = 0.1, beta: float = 0.5,
    tolerance: float = 1e-7, max_iterations: int = 20000,
) -> DescentResult:
    if not 0.0 < alpha < 0.5 or not 0.0 < beta < 1.0:
        raise ValueError("Require 0 < alpha < 0.5 and 0 < beta < 1")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite")
    if not isinstance(max_iterations, int) or max_iterations < 0:
        raise ValueError("max_iterations must be a nonnegative integer")
    x = np.zeros(problem.n) if x0 is None else problem._point(x0).copy()
    if not problem.feasible(x):
        raise ValueError("x0 must be strictly feasible")
    iterates, objectives, norms, margins, steps, reductions, decrements = ([] for _ in range(7))
    converged = False
    termination = "maximum_iterations"
    for k in range(max_iterations + 1):
        g = problem.gradient(x)
        norm = float(np.linalg.norm(g))
        iterates.append(x.copy())
        objectives.append(problem.value(x))
        norms.append(norm)
        margins.append(problem.margin(x))
        if method == "newton":
            factor = cho_factor(problem.hessian(x), lower=True, check_finite=False)
            direction = cho_solve(factor, -g, check_finite=False)
            decrement = max(0.0, float(-g @ direction))
            decrements.append(decrement)
            stop = decrement / 2.0 <= tolerance
        else:
            direction = -g
            stop = norm <= tolerance
        if stop:
            converged = True
            termination = "gradient_norm" if method == "gradient" else "newton_decrement"
            break
        if k == max_iterations:
            break
        step, n_backtracks = backtracking(problem, x, direction, g, alpha=alpha, beta=beta)
        candidate = x + step * direction
        if np.array_equal(candidate, x):
            termination = "floating_point_stagnation"
            break
        steps.append(step)
        reductions.append(n_backtracks)
        x = candidate
    return DescentResult(
        method, x, np.asarray(iterates), np.asarray(objectives), np.asarray(norms),
        np.asarray(margins), np.asarray(steps), np.asarray(reductions, dtype=np.int64),
        np.asarray(decrements), converged, termination, alpha, beta, tolerance,
    )


def gradient_descent(problem: AnalyticCenter, *, tolerance: float = 1e-7, **kwargs) -> DescentResult:
    """Stop when the Euclidean gradient norm is at most tolerance."""
    return _minimize(problem, "gradient", tolerance=tolerance, **kwargs)


def newton_method(problem: AnalyticCenter, *, tolerance: float = 1e-12, **kwargs) -> DescentResult:
    """Cholesky Newton method; stop when lambda(x)^2 / 2 <= tolerance."""
    return _minimize(problem, "newton", tolerance=tolerance, **kwargs)


def convergence_diagnostics(result: DescentResult, x_reference: ArrayLike) -> dict:
    """Return errors and unfiltered local rate samples with explicit safeguards."""
    errors = np.linalg.norm(result.iterates - np.asarray(x_reference), axis=1)
    samples = []
    for k in range(len(errors) - 1):
        # Avoid division by rounding-level errors and the large-error damped phase.
        if result.step_lengths[k] == 1.0 and 1e-8 < errors[k] < 0.2 and errors[k + 1] > 1e-13:
            sample = {"iteration": k, "error": float(errors[k]),
                      "next_error": float(errors[k + 1]),
                      "quadratic_ratio": float(errors[k + 1] / errors[k] ** 2)}
            if (k >= 1 and result.step_lengths[k - 1] == 1.0
                    and errors[k - 1] > errors[k] > errors[k + 1]):
                sample["observed_order"] = float(
                    np.log(errors[k + 1] / errors[k]) / np.log(errors[k] / errors[k - 1])
                )
            samples.append(sample)
    return {"distance_to_reference": errors.tolist(), "local_samples": samples}


def run_experiment(output_dir: str | Path | None = None, seed: int = 0, quick: bool = False) -> dict:
    """Run both parts, export full traces/parameter sweeps, and save four figures.

    The quick mode keeps the same experiment structure but reduces the three
    dimensions.  It is suitable for smoke tests, not a substitute for the full run.
    """
    output = Path(output_dir) if output_dir is not None else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
    dimensions = [(30, 10), (100, 30), (300, 100)]
    if quick:
        dimensions = [(15, 5), (30, 10), (60, 20)]
    alphas, betas = [0.01, 0.1, 0.4], [0.2, 0.5, 0.8]
    rows, instances, traces = [], [], []
    for index, (m, n) in enumerate(dimensions):
        instance_seed = seed + index
        problem = make_instance(m, n, instance_seed)
        reference = newton_method(problem, tolerance=1e-25, max_iterations=100)
        if not reference.converged:
            raise RuntimeError(f"Reference failed for m={m}, n={n}: {reference.termination}")
        p_reference = problem.value(reference.x)
        reference_bound = float(np.linalg.norm(problem.gradient(reference.x)) ** 2 / 4.0)
        baseline = {}
        instance_runs = []
        for method, solver, tolerance in [
            ("gradient", gradient_descent, 1e-7), ("newton", newton_method, 1e-20)
        ]:
            for alpha in alphas:
                for beta in betas:
                    result = solver(problem, alpha=alpha, beta=beta, tolerance=tolerance)
                    if not result.converged:
                        raise RuntimeError(f"{method} failed for ({m}, {n}, {alpha}, {beta})")
                    gap = problem.difference(reference.x, result.x - reference.x)
                    row = {
                        "m": m, "n": n, "seed": instance_seed, "method": method,
                        "alpha": alpha, "beta": beta, "iterations": result.iterations,
                        "total_backtracks": int(result.backtracks.sum()),
                        "final_objective": float(result.objectives[-1]),
                        "objective_gap_to_reference": gap,
                        "final_gradient_norm": float(result.gradient_norms[-1]),
                        "final_newton_decrement_squared": (
                            float(result.decrement_squared[-1]) if method == "newton" else None
                        ),
                        "minimum_margin": float(result.margins.min()),
                        "converged": bool(result.converged),
                    }
                    rows.append(row)
                    trace = {**row, **result.as_dict()}
                    trace["objective_gaps"] = [
                        problem.difference(reference.x, x - reference.x) for x in result.iterates
                    ]
                    trace["convergence"] = convergence_diagnostics(result, reference.x)
                    if not (alpha == 0.1 and beta == 0.5):
                        trace.pop("iterates")
                    instance_runs.append(trace)
                    if alpha == 0.1 and beta == 0.5:
                        baseline[method] = trace
        if output is not None:
            np.savez_compressed(output / f"instance_m{m}_n{n}.npz", A=problem.A, x_reference=reference.x)
        info = {
            "m": m, "n": n, "seed": instance_seed, "distribution": "independent standard normal entries",
            "p_reference": p_reference, "reference_gradient_norm": float(reference.gradient_norms[-1]),
            "reference_objective_error_upper_bound": reference_bound,
            "reference_iterations": reference.iterations,
            "hessian_condition_number_at_solution": float(np.linalg.cond(problem.hessian(reference.x))),
            "baseline": {key: {field: value for field, value in trace.items()
                                if field not in {"iterates", "objectives", "gradient_norms", "margins", "step_lengths",
                                                 "backtracks", "decrement_squared", "objective_gaps", "x"}}
                         for key, trace in baseline.items()},
        }
        instances.append(info)
        traces.append({"instance": info, "reference": reference.as_dict(), "runs": instance_runs})

    summary = {
        "exercise": "9.30", "seed": seed, "quick": quick,
        "source": "https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=533",
        "alphas": alphas, "betas": betas,
        "gradient_tolerance": 1e-7, "newton_decrement_half_tolerance": 1e-20,
        "reference_decrement_half_tolerance": 1e-25,
        "instances": instances, "runs": rows,
        "figures": ["objective_and_steps.png", "parameter_sweep.png", "newton_quadratic_convergence.png", "newton_objective_and_steps.png"],
    }
    if output is None:
        return summary
    from convex_solutions.plotting import pyplot
    plt = pyplot()

    # Objective values, accurate gaps, and accepted step sizes at each dimension.
    colors = {"gradient": "#2563eb", "newton": "#db2777"}
    fig, axes = plt.subplots(3, len(dimensions), figsize=(14, 10), constrained_layout=True)
    for column, trace_group in enumerate(traces):
        info = trace_group["instance"]
        for trace in trace_group["runs"]:
            if trace["alpha"] != 0.1 or trace["beta"] != 0.5:
                continue
            method = trace["method"]
            axes[0, column].plot(trace["objectives"], label=method, color=colors[method])
            axes[1, column].semilogy(np.maximum(trace["objective_gaps"], 1e-24), label=method, color=colors[method])
            axes[2, column].semilogy(trace["step_lengths"], label=method, color=colors[method], marker=".", markersize=3)
        axes[0, column].set_title(f"m={info['m']}, n={info['n']}, seed={info['seed']}")
        for row, label in enumerate(["Objective f(x)", "f(x) − p_ref (floor 1e−24)", "Accepted step t"]):
            axes[row, column].set_ylabel(label)
            axes[row, column].set_xlabel("Accepted updates k")
            axes[row, column].grid(alpha=0.2)
        axes[0, column].legend()
    fig.suptitle("Exercise 9.30: gradient and Newton, α=0.1, β=0.5")
    fig.savefig(output / "objective_and_steps.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, len(dimensions), figsize=(13, 7), constrained_layout=True)
    for column, trace_group in enumerate(traces):
        info = trace_group["instance"]
        for row_index, method in enumerate(["gradient", "newton"]):
            counts = np.array([[next(r["iterations"] for r in trace_group["runs"]
                                    if r["method"] == method and r["alpha"] == alpha and r["beta"] == beta)
                                for beta in betas] for alpha in alphas])
            axis = axes[row_index, column]
            im = axis.imshow(counts, cmap="Blues" if method == "gradient" else "RdPu", aspect="auto")
            axis.set_xticks(range(len(betas)), [str(v) for v in betas])
            axis.set_yticks(range(len(alphas)), [str(v) for v in alphas])
            for i in range(len(alphas)):
                for j in range(len(betas)):
                    axis.text(j, i, str(counts[i, j]), ha="center", va="center",
                              color="white" if counts[i, j] > (counts.max() + counts.min()) / 2 else "black")
            axis.set_xlabel("β")
            axis.set_ylabel("α")
            axis.set_title(f"{method}: m={info['m']}, n={info['n']}")
            fig.colorbar(im, ax=axis, label="Accepted updates")
    fig.suptitle("Backtracking parameter sweep (same data and tolerance within each method)")
    fig.savefig(output / "parameter_sweep.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for trace_group in traces:
        info = trace_group["instance"]
        trace = next(r for r in trace_group["runs"] if r["method"] == "newton" and r["alpha"] == 0.1 and r["beta"] == 0.5)
        errors = np.asarray(trace["convergence"]["distance_to_reference"])
        label = f"m={info['m']}, n={info['n']}"
        axes[0].semilogy(np.maximum(errors, 1e-15), marker="o", label=label)
        samples = trace["convergence"]["local_samples"]
        axes[1].loglog([s["error"] for s in samples], [s["next_error"] for s in samples], marker="o", label=label)
    grid = np.logspace(-7, -0.5, 100)
    axes[1].loglog(grid, grid * grid, linestyle="--", color="gray", label="e(k+1) = e(k)² guide")
    axes[0].set(xlabel="Accepted updates k", ylabel="‖x(k) − x_ref‖₂ (floor 1e−15)", title="Newton error history")
    axes[1].set(xlabel="e(k)", ylabel="e(k+1)", title="Local full-step errors: quadratic slope")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.savefig(output / "newton_quadratic_convergence.png", dpi=160)
    plt.close(fig)

    # A dedicated Newton objective/step plot avoids compressing Newton into the
    # first few pixels of the much longer gradient horizontal axis.
    fig, axes = plt.subplots(2, len(dimensions), figsize=(13, 6), constrained_layout=True)
    for column, trace_group in enumerate(traces):
        info = trace_group["instance"]
        trace = next(r for r in trace_group["runs"] if r["method"] == "newton" and r["alpha"] == 0.1 and r["beta"] == 0.5)
        axes[0, column].semilogy(np.maximum(trace["objective_gaps"], 1e-24), "o-", color=colors["newton"])
        axes[1, column].step(range(trace["iterations"]), trace["step_lengths"], where="mid", color=colors["newton"])
        axes[0, column].set_title(f"m={info['m']}, n={info['n']}")
        for row, label in enumerate(["f(x) − p_ref (floor 1e−24)", "Accepted step t"]):
            axes[row, column].set(xlabel="Accepted updates k", ylabel=label)
            axes[row, column].grid(alpha=0.2)
    fig.suptitle("Newton: damped phase followed by full steps")
    fig.savefig(output / "newton_objective_and_steps.png", dpi=160)
    plt.close(fig)

    with (output / "parameter_sweep.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (output / "histories.json").write_text(json.dumps(traces, indent=2, allow_nan=False) + "\n")
    return summary


def solve(output_dir: Path | None = None, seed: int = 0) -> dict:
    """Repository entry point; use seed=0 to reproduce the worked example."""
    return run_experiment(output_dir=output_dir, seed=seed)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/exercise930"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    result = run_experiment(args.output_dir, args.seed, args.quick)
    for instance in result["instances"]:
        print(f"m={instance['m']}, n={instance['n']}: p≈{instance['p_reference']:.12g}; "
              f"gradient {instance['baseline']['gradient']['iterations']} updates, "
              f"Newton {instance['baseline']['newton']['iterations']} updates")
