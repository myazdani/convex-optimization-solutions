"""Exercise 9.32: Gauss–Newton on convex quadratic-residual least squares."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from convex_solutions.io import write_json


@dataclass(frozen=True)
class QuadraticResidualProblem:
    A: np.ndarray  # Shape (m, n, n), with positive-definite slices.
    b: np.ndarray  # Shape (m, n).

    def __post_init__(self):
        A, b = np.asarray(self.A, dtype=float), np.asarray(self.b, dtype=float)
        if (A.ndim != 3 or A.shape[1] != A.shape[2] or b.shape != A.shape[:2]
                or min(A.shape) < 1 or not np.all(np.isfinite(A)) or not np.all(np.isfinite(b))):
            raise ValueError("A and b must have finite compatible nonempty shapes")
        if not np.allclose(A, A.swapaxes(1, 2), atol=1e-13, rtol=1e-13):
            raise ValueError("Each A_i must be symmetric")
        if np.any(np.linalg.eigvalsh(A) <= 0):
            raise ValueError("Each A_i must be positive definite")
        rho = np.einsum("mi,mi->m", b, np.linalg.solve(A, b[..., None])[..., 0])
        if np.any(rho > 2 + 1e-12):
            raise ValueError("Need b_i^T A_i^(-1) b_i <= 2")
        object.__setattr__(self, "A", A.copy())
        object.__setattr__(self, "b", b.copy())

    @property
    def n(self):
        return self.b.shape[1]

    def residuals(self, x):
        return 0.5 * np.einsum("i,mij,j->m", x, self.A, x) + self.b @ x + 1

    def jacobian(self, x):
        return np.einsum("mij,j->mi", self.A, x) + self.b

    def value(self, x):
        r = self.residuals(x)
        return float(r @ r / 2)

    def gradient(self, x):
        return self.jacobian(x).T @ self.residuals(x)

    def hessian(self, x):
        J = self.jacobian(x)
        return J.T @ J + np.einsum("m,mij->ij", self.residuals(x), self.A)

    def difference(self, x, d, step=1.0):
        # Each residual is quadratic, so its finite change is exact in algebra.
        delta = step * (self.jacobian(x) @ d) + 0.5 * step**2 * np.einsum("i,mij,j->m", d, self.A, d)
        r, delta = self.residuals(x).astype(np.longdouble), delta.astype(np.longdouble)
        return float(np.sum(r * delta + 0.5 * delta * delta, dtype=np.longdouble))

    def residual_lower_bounds(self):
        inverse_b = np.linalg.solve(self.A, self.b[..., None])[..., 0]
        return 1 - 0.5 * np.einsum("mi,mi->m", self.b, inverse_b)

    def strong_convexity_bound(self):
        return float(np.linalg.eigvalsh(np.einsum("m,mij->ij", self.residual_lower_bounds(), self.A))[0])


def make_instance(m=60, n=10, seed=932):
    if not isinstance(m, int) or not isinstance(n, int) or m <= n or n <= 0:
        raise ValueError("Use positive dimensions with m > n for these full-rank random instances")
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((m, n, n))
    A = np.einsum("mji,mjk->mik", raw, raw) / n + 0.5 * np.eye(n)
    b = rng.standard_normal((m, n))
    rho = np.einsum("mi,mi->m", b, np.linalg.solve(A, b[..., None])[..., 0])
    targets = rng.uniform(0.1, 1.5, size=m)
    b *= np.sqrt(targets / rho)[:, None]
    return QuadraticResidualProblem(A, b)


def minimize(problem: QuadraticResidualProblem, method="gauss_newton", *, tolerance=1e-7,
             alpha=0.1, beta=0.5, max_iterations=30000, x0=None):
    if method not in {"gauss_newton", "newton", "gradient"}:
        raise ValueError("Unknown method")
    if not 0 < alpha < 0.5 or not 0 < beta < 1 or tolerance <= 0:
        raise ValueError("Invalid line-search parameter or tolerance")
    x = np.zeros(problem.n) if x0 is None else np.asarray(x0, dtype=float).copy()
    if x.shape != (problem.n,) or not np.all(np.isfinite(x)):
        raise ValueError("x0 has the wrong shape or is not finite")
    iterates, objectives, norms, steps, smallest_singular_values, largest_singular_values = [], [], [], [], [], []
    total_backtracks = 0
    status = "maximum_iterations"
    for k in range(max_iterations + 1):
        r, J = problem.residuals(x), problem.jacobian(x)
        g = J.T @ r
        iterates.append(x.copy())
        objectives.append(problem.value(x))
        norms.append(float(np.linalg.norm(g)))
        singular = np.linalg.svd(J, compute_uv=False)
        smallest_singular_values.append(float(singular[-1]))
        largest_singular_values.append(float(singular[0]))
        if norms[-1] <= tolerance:
            status = "converged"
            break
        if k == max_iterations:
            break
        if method == "gauss_newton":
            # Solve min_d ||J d + r||, avoiding squaring the condition number.
            d, _, rank, _ = np.linalg.lstsq(J, -r, rcond=None)
            if rank != problem.n:
                status = "rank_deficient_jacobian"
                break
        elif method == "newton":
            d = cho_solve(cho_factor(problem.hessian(x), lower=True), -g)
        else:
            d = -g
        slope = float(g @ d)
        if slope >= 0:
            status = "non_descent_direction"
            break
        step = 1.0
        for count in range(201):
            if problem.difference(x, d, step) <= alpha * step * slope:
                break
            step *= beta
        else:
            status = "backtracking_limit"
            break
        candidate = x + step * d
        if np.array_equal(candidate, x):
            status = "floating_point_stagnation"
            break
        x = candidate
        steps.append(step)
        total_backtracks += count
    mu = problem.strong_convexity_bound()
    return {"method": method, "x": x.tolist(), "iterates": np.asarray(iterates).tolist(),
            "objectives": objectives, "gradient_norms": norms, "step_lengths": steps,
            "smallest_jacobian_singular_values": smallest_singular_values,
            "largest_jacobian_singular_values": largest_singular_values,
            "iterations": len(steps), "status": status, "alpha": alpha, "beta": beta,
            "gradient_tolerance": tolerance, "total_backtracks": total_backtracks,
            "objective_error_upper_bound": norms[-1] ** 2 / (2 * mu) if mu > 0 else None}


def run_experiment(output_dir: Path | None = None, seed=0, dimensions=((20, 5), (60, 10), (150, 20))):
    instances = []
    output = Path(output_dir) if output_dir is not None else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
    for index, (m, n) in enumerate(dimensions):
        problem = make_instance(m, n, seed + index)
        reference = minimize(problem, "newton", tolerance=1e-12)
        if reference["status"] != "converged":
            raise RuntimeError(f"Reference failed: {reference['status']}")
        x_ref = np.asarray(reference["x"])
        runs = []
        for method in ["gradient", "newton", "gauss_newton"]:
            run = minimize(problem, method)
            if run["status"] != "converged":
                raise RuntimeError(f"{method} failed: {run['status']}")
            run["objective_gaps"] = [problem.difference(x_ref, np.asarray(x) - x_ref)
                                      for x in run.pop("iterates")]
            runs.append(run)
        instances.append({"m": m, "n": n, "seed": seed + index,
                          "minimum_residual_lower_bound": float(problem.residual_lower_bounds().min()),
                          "strong_convexity_lower_bound": problem.strong_convexity_bound(),
                          "p_reference": problem.value(x_ref),
                          "reference_gradient_norm": reference["gradient_norms"][-1],
                          "reference_objective_error_upper_bound": reference["objective_error_upper_bound"],
                          "runs": runs})
        if output is not None:
            np.savez_compressed(output / f"instance_m{m}_n{n}.npz", A=problem.A, b=problem.b, x_reference=x_ref)
    result = {"exercise": "9.32", "seed": seed, "instances": instances,
              "source": "https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=534"}
    if output is not None:
        from convex_solutions.plotting import pyplot
        plt = pyplot()
        fig, axes = plt.subplots(2, len(instances), figsize=(13, 7), squeeze=False, constrained_layout=True)
        for col, instance in enumerate(instances):
            for run in instance["runs"]:
                axes[0, col].semilogy(np.maximum(run["objective_gaps"], 1e-20), label=run["method"])
                axes[1, col].semilogy(run["step_lengths"], label=run["method"])
            axes[0, col].set(title=f"m={instance['m']}, n={instance['n']}", xlabel="Accepted updates", ylabel="Objective gap (floor 1e−20)")
            axes[1, col].set(xlabel="Accepted updates", ylabel="Accepted step")
            for row in range(2):
                axes[row, col].grid(alpha=0.2)
                axes[row, col].legend(fontsize=8)
        fig.suptitle("Exercise 9.32: convex nonlinear least squares, ‖gradient‖₂ ≤ 1e−7")
        fig.savefig(output / "gauss_newton_comparison.png", dpi=160)
        plt.close(fig)
        write_json(output / "results.json", result)
    return result


def solve(output_dir: Path | None = None, seed: int = 0) -> dict:
    return run_experiment(output_dir, seed)
