"""Original numerical solutions to Boyd and Vandenberghe, exercises 10.14–10.16.

All Newton directions are implemented explicitly. CVXPY and SciPy are used only
as independent cross-checks. Histories include the initial and terminal residual;
step_lengths[k] takes iterate k to iterate k+1.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.linalg import solve
from scipy.optimize import root

from convex_solutions.io import write_json
from convex_solutions.numerics import solve_cvxpy
from convex_solutions.plotting import pyplot


@dataclass
class NewtonResult:
    x: np.ndarray
    multiplier: np.ndarray
    objective: float
    residuals: list[float]
    step_lengths: list[float]
    objectives: list[float]
    primal_residuals: list[float]
    dual_residuals: list[float]
    backtracks: list[int]

    @property
    def iterations(self) -> int:
        return len(self.step_lengths)

    def summary(self) -> dict:
        return {
            "x": self.x.tolist(), "multiplier": self.multiplier.tolist(),
            "objective": float(self.objective), "iterations": self.iterations,
            "residuals": self.residuals, "step_lengths": self.step_lengths,
            "objectives": self.objectives, "primal_residuals": self.primal_residuals,
            "dual_residuals": self.dual_residuals, "backtracks": self.backtracks,
        }


def _line_search_parameters(alpha: float, beta: float) -> None:
    if not (0 < alpha < 0.5 and 0 < beta < 1):
        raise ValueError("Require 0 < alpha < 0.5 and 0 < beta < 1.")


def _equality_data(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> None:
    if A.ndim != 2 or A.shape[1] != x.size or b.shape != (A.shape[0],):
        raise ValueError("Incompatible equality-constrained problem dimensions.")
    if A.shape[0] >= A.shape[1] or np.linalg.matrix_rank(A) != A.shape[0]:
        raise ValueError("A must have full row rank and fewer rows than columns.")
    if not (np.isfinite(A).all() and np.isfinite(b).all() and np.isfinite(x).all()):
        raise ValueError("Problem data and starting point must be finite.")


def _armijo_slack(value: float) -> float:
    # A machine-precision allowance avoids rejecting correct final Newton steps
    # when the objective change is smaller than summation roundoff.
    return 8 * np.finfo(float).eps * max(1.0, abs(value))


def _plot_history(results: dict[str, NewtonResult], destination: Path,
                  title: str, objective: bool = False) -> None:
    plt = pyplot()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3 if objective else 2,
                             figsize=(15 if objective else 11, 4.3))
    for label, result in results.items():
        axes[0].semilogy(range(len(result.residuals)), np.maximum(result.residuals, 1e-16),
                         marker=".", label=label)
        axes[1].plot(range(result.iterations), result.step_lengths, marker=".", label=label)
        if objective:
            axes[2].plot(range(len(result.objectives)), result.objectives, marker=".", label=label)
    axes[0].set_ylabel("KKT / stationarity residual norm")
    axes[1].set_ylabel("Accepted Newton step length")
    if objective:
        axes[2].set_ylabel("Primal objective")
    for ax in axes:
        ax.set_xlabel("Iteration")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(destination, dpi=170, bbox_inches="tight")
    plt.close(fig)


def solve_log_investment(alpha: float = 0.01, beta: float = 0.5,
                         tol: float = 1e-11, max_iter: int = 100,
                         x0: np.ndarray | None = None) -> NewtonResult:
    """10.14: Newton on the budget hyperplane; short positions are allowed."""
    _line_search_parameters(alpha, beta)
    P = np.array([[2., 1.3, 1.], [2., .5, 1.], [.5, 1.3, 1.], [.5, .5, 1.]])
    pi = np.array([1/3, 1/6, 1/3, 1/6])
    A = np.ones((1, 3))
    x = np.full(3, 1/3) if x0 is None else np.asarray(x0, dtype=float).copy()
    if x.shape != (3,) or abs(x.sum() - 1) > 1e-10 or np.min(P @ x) <= 0:
        raise ValueError("Starting allocation must sum to one and give positive scenario returns.")
    residuals, steps, objectives, primals, duals, backtracks = [], [], [], [], [], []
    for _ in range(max_iter + 1):
        q = P @ x
        f = float(-pi @ np.log(q))
        grad = -P.T @ (pi / q)
        H = P.T @ ((pi / q**2)[:, None] * P)
        kkt = np.block([[H, A.T], [A, np.zeros((1, 1))]])
        direction = solve(kkt, np.r_[-grad, 0.], assume_a="sym")
        dx, nu = direction[:3], direction[3:]
        rd = float(np.linalg.norm(grad + A.T @ nu))
        rp = float(abs(x.sum() - 1))
        residuals.append(float(np.hypot(rd, rp)))
        objectives.append(f); primals.append(rp); duals.append(rd)
        if residuals[-1] <= tol:
            return NewtonResult(x, nu, f, residuals, steps, objectives, primals, duals, backtracks)
        if len(steps) == max_iter:
            break
        t, count = 1.0, 0
        slope = float(grad @ dx)
        for _ in range(200):
            trial = x + t * dx
            qt = P @ trial
            if np.min(qt) > 0:
                ft = float(-pi @ np.log(qt))
                if ft <= f + alpha * t * slope + _armijo_slack(f):
                    break
            t *= beta; count += 1
        else:
            raise RuntimeError("Portfolio line search failed.")
        x = trial
        steps.append(t); backtracks.append(count)
    raise RuntimeError("Portfolio Newton method did not converge.")


def entropy_data(seed: int = 1015, n: int = 100, p: int = 30) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seeded full-row-rank A and strictly positive feasible xhat, b=A*xhat."""
    if not 0 < p < n:
        raise ValueError("Require 0 < p < n.")
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((p, n)) / np.sqrt(n)
    if np.linalg.matrix_rank(A) != p:
        raise RuntimeError("Generated equality matrix is rank deficient.")
    xhat = rng.uniform(0.05, 1.0, size=n)
    return A, A @ xhat, xhat


def entropy_newton(A: np.ndarray, b: np.ndarray, x0: np.ndarray,
                   method: str = "feasible", nu0: np.ndarray | None = None,
                   alpha: float = 0.01, beta: float = 0.5,
                   tol: float = 1e-10, max_iter: int = 100) -> NewtonResult:
    """10.15(a,b): feasible-objective or infeasible-residual Newton.

    The diagonal Hessian is eliminated analytically. Each step factors only
    A*diag(x)*A.T, not the full (n+p)-dimensional KKT matrix.
    """
    _line_search_parameters(alpha, beta)
    A, b, x = np.asarray(A, float), np.asarray(b, float), np.asarray(x0, float).copy()
    _equality_data(A, b, x)
    if method not in {"feasible", "infeasible"}:
        raise ValueError("method must be 'feasible' or 'infeasible'.")
    if np.min(x) <= 0:
        raise ValueError("Entropy Newton needs a strictly positive initial point.")
    if method == "feasible" and np.linalg.norm(A @ x - b) > 1e-9:
        raise ValueError("The standard Newton starting point must satisfy Ax=b.")
    nu = np.zeros(A.shape[0]) if nu0 is None else np.asarray(nu0, float).copy()
    if nu.shape != b.shape:
        raise ValueError("Wrong multiplier dimension.")
    residuals, steps, objectives, primals, duals, backtracks = [], [], [], [], [], []
    for _ in range(max_iter + 1):
        grad, f = np.log(x) + 1, float(x @ np.log(x))
        M = (A * x) @ A.T
        rp = A @ x - b
        if method == "feasible":
            nu = solve(M, -A @ (x * grad), assume_a="pos")
            rd = grad + A.T @ nu
            dx, dnu = -x * rd, None
        else:
            rd = grad + A.T @ nu
            dnu = solve(M, rp - A @ (x * rd), assume_a="pos")
            dx = -x * (rd + A.T @ dnu)
        residual = float(np.linalg.norm(np.r_[rd, rp]))
        residuals.append(residual); objectives.append(f)
        primals.append(float(np.linalg.norm(rp))); duals.append(float(np.linalg.norm(rd)))
        if residual <= tol:
            return NewtonResult(x, nu, f, residuals, steps, objectives, primals, duals, backtracks)
        if len(steps) == max_iter:
            break
        t, count = 1.0, 0
        slope = float(grad @ dx)
        for _ in range(200):
            trial = x + t * dx
            if np.min(trial) > 0:
                if method == "feasible":
                    accept = (float(trial @ np.log(trial)) <=
                              f + alpha * t * slope + _armijo_slack(f))
                else:
                    rt = np.r_[np.log(trial) + 1 + A.T @ (nu + t * dnu), A @ trial - b]
                    accept = np.linalg.norm(rt) <= (1 - alpha * t) * residual
                if accept:
                    break
            t *= beta; count += 1
        else:
            raise RuntimeError("Entropy line search failed.")
        x = trial
        if dnu is not None:
            nu += t * dnu
        steps.append(t); backtracks.append(count)
    raise RuntimeError(f"{method} entropy Newton did not converge.")


def entropy_dual_newton(A: np.ndarray, b: np.ndarray,
                        nu0: np.ndarray | None = None,
                        alpha: float = 0.01, beta: float = 0.5,
                        tol: float = 1e-10, max_iter: int = 100) -> NewtonResult:
    """10.15(c): minimize sum(exp(-1-A.T*nu)) + b.T*nu."""
    _line_search_parameters(alpha, beta)
    A, b = np.asarray(A, float), np.asarray(b, float)
    nu = np.zeros(A.shape[0]) if nu0 is None else np.asarray(nu0, float).copy()
    _equality_data(A, b, np.ones(A.shape[1]))
    if nu.shape != b.shape:
        raise ValueError("Wrong multiplier dimension.")
    residuals, steps, objectives, primals, duals, backtracks = [], [], [], [], [], []
    for _ in range(max_iter + 1):
        x = np.exp(-1 - A.T @ nu)
        if not np.isfinite(x).all() or np.min(x) <= 0:
            raise ValueError("Dual iterate overflowed or underflowed; use a milder initial multiplier.")
        phi, grad = float(x.sum() + b @ nu), b - A @ x
        H = (A * x) @ A.T
        rd, rp = np.log(x) + 1 + A.T @ nu, A @ x - b
        residual = float(np.linalg.norm(np.r_[rd, rp]))
        residuals.append(residual); objectives.append(float(x @ np.log(x)))
        primals.append(float(np.linalg.norm(rp))); duals.append(float(np.linalg.norm(rd)))
        if residual <= tol:
            return NewtonResult(x, nu, objectives[-1], residuals, steps, objectives, primals, duals, backtracks)
        if len(steps) == max_iter:
            break
        dnu = solve(H, -grad, assume_a="pos")
        slope = float(grad @ dnu)
        t, count = 1.0, 0
        for _ in range(200):
            trial = nu + t * dnu
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                xt = np.exp(-1 - A.T @ trial)
                phit = float(xt.sum() + b @ trial)
            if (np.isfinite(phit) and np.min(xt) > 0 and
                    phit <= phi + alpha * t * slope + _armijo_slack(phi)):
                break
            t *= beta; count += 1
        else:
            raise RuntimeError("Dual entropy line search failed.")
        nu = trial
        steps.append(t); backtracks.append(count)
    raise RuntimeError("Dual entropy Newton did not converge.")


def game_data(seed: int = 1016, p: int = 100, q: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((p, q)) / np.sqrt(q), rng.standard_normal(p), rng.standard_normal(q)


def game_value(A: np.ndarray, b: np.ndarray, c: np.ndarray,
               u: np.ndarray, v: np.ndarray) -> float:
    su, sv = 1 - u @ u, 1 - v @ v
    if min(su, sv) <= 0:
        raise ValueError("Game strategies must lie strictly inside the unit balls.")
    return float(u @ A @ v + b @ u + c @ v - np.log(su) + np.log(sv))


def game_derivatives(A: np.ndarray, b: np.ndarray, c: np.ndarray,
                     u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    su, sv = 1 - u @ u, 1 - v @ v
    if min(su, sv) <= 0:
        raise ValueError("Game strategies must lie strictly inside the unit balls.")
    Hu = 2 * np.eye(u.size) / su + 4 * np.outer(u, u) / su**2
    Hv = 2 * np.eye(v.size) / sv + 4 * np.outer(v, v) / sv**2
    residual = np.r_[A @ v + b + 2*u/su, A.T @ u + c - 2*v/sv]
    jacobian = np.block([[Hu, A], [A.T, -Hv]])
    return residual, jacobian


def game_newton(A: np.ndarray, b: np.ndarray, c: np.ndarray,
                u0: np.ndarray | None = None, v0: np.ndarray | None = None,
                alpha: float = 0.01, beta: float = 0.5,
                tol: float = 1e-10, max_iter: int = 150) -> NewtonResult:
    """10.16: solve grad(f)=0 with residual-norm backtracking."""
    _line_search_parameters(alpha, beta)
    A, b, c = np.asarray(A, float), np.asarray(b, float), np.asarray(c, float)
    if A.shape != (b.size, c.size):
        raise ValueError("Incompatible game dimensions.")
    u = np.zeros(b.size) if u0 is None else np.asarray(u0, float).copy()
    v = np.zeros(c.size) if v0 is None else np.asarray(v0, float).copy()
    if u.shape != b.shape or v.shape != c.shape:
        raise ValueError("Wrong initial game strategy dimensions.")
    game_value(A, b, c, u, v)
    residuals, steps, objectives, primals, duals, backtracks = [], [], [], [], [], []
    for _ in range(max_iter + 1):
        r, J = game_derivatives(A, b, c, u, v)
        residual = float(np.linalg.norm(r))
        residuals.append(residual); objectives.append(game_value(A, b, c, u, v))
        primals.append(float(np.linalg.norm(r[:u.size]))); duals.append(float(np.linalg.norm(r[u.size:])))
        if residual <= tol:
            return NewtonResult(np.r_[u, v], np.array([]), objectives[-1], residuals,
                                steps, objectives, primals, duals, backtracks)
        if len(steps) == max_iter:
            break
        dz = solve(J, -r, assume_a="sym")
        du, dv = dz[:u.size], dz[u.size:]
        t, count = 1., 0
        for _ in range(250):
            ut, vt = u + t * du, v + t * dv
            if max(ut @ ut, vt @ vt) < 1:
                rt, _ = game_derivatives(A, b, c, ut, vt)
                if np.linalg.norm(rt) <= (1 - alpha*t) * residual:
                    break
            t *= beta; count += 1
        else:
            raise RuntimeError("Game residual line search failed.")
        u, v = ut, vt
        steps.append(t); backtracks.append(count)
    raise RuntimeError("Game Newton method did not converge.")


def game_best_responses(A: np.ndarray, b: np.ndarray, c: np.ndarray,
                        u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact one-player best responses, independently useful for saddle checks."""
    s, t = A @ v + b, A.T @ u + c
    return -s / (1 + np.sqrt(1 + s @ s)), t / (1 + np.sqrt(1 + t @ t))


def _independent_game_solution(A: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    def fixed_point_residual(z: np.ndarray) -> np.ndarray:
        u, v = z[:b.size], z[b.size:]
        u_br, v_br = game_best_responses(A, b, c, u, v)
        return np.r_[u-u_br, v-v_br]
    result = root(fixed_point_residual, np.zeros(b.size + c.size), method="hybr", tol=1e-11)
    if np.linalg.norm(fixed_point_residual(result.x)) > 1e-9:
        raise RuntimeError(f"Independent fixed-point root check failed: {result.message}")
    return result.x


def _reference_solve(problem) -> None:
    solve_cvxpy(problem, solver="CLARABEL", tol_gap_abs=1e-10, tol_gap_rel=1e-10,
                tol_feas=1e-10, max_iter=300)


def _save(output_dir: Path | None, exercise: str, summary: dict) -> dict:
    if output_dir is None:
        return summary
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"exercise_{exercise.replace('.', '_')}.json"
    write_json(path, summary)
    return summary


def run_10_14(output_dir: str | Path | None, seed: int = 1014) -> dict:
    import cvxpy as cp
    output_dir = Path(output_dir) if output_dir is not None else None
    result = solve_log_investment()
    P = np.array([[2., 1.3, 1.], [2., .5, 1.], [.5, 1.3, 1.], [.5, .5, 1.]])
    pi = np.array([1/3, 1/6, 1/3, 1/6])
    x = cp.Variable(3)
    reference = cp.Problem(cp.Minimize(-pi @ cp.log(P @ x)), [cp.sum(x) == 1])
    _reference_solve(reference)
    if output_dir is not None:
        _plot_history({"equality-constrained Newton": result}, output_dir / "10_14_newton.png",
                      "10.14: log-optimal portfolio", objective=True)
    summary = {"exercise": "10.14", "seed": seed, "newton": result.summary(),
               "growth_rate": -result.objective, "scenario_returns": (P @ result.x).tolist(),
               "reference": {"solver": "CVXPY/CLARABEL", "status": reference.status,
                             "x": x.value.tolist(), "objective": float(reference.value),
                             "point_error": float(np.linalg.norm(result.x-x.value)),
                             "objective_error": float(abs(result.objective-reference.value))},
               "plots": ["10_14_newton.png"] if output_dir is not None else []}
    return _save(output_dir, "10.14", summary)


def run_10_15(output_dir: str | Path | None, seed: int = 1015) -> dict:
    import cvxpy as cp
    output_dir = Path(output_dir) if output_dir is not None else None
    A, b, xhat = entropy_data(seed)
    results = {
        "(a) feasible Newton": entropy_newton(A, b, xhat, "feasible"),
        "(b) infeasible Newton, xhat": entropy_newton(A, b, xhat, "infeasible"),
        "(b) infeasible Newton, ones": entropy_newton(A, b, np.ones(A.shape[1]), "infeasible"),
        "(c) dual Newton": entropy_dual_newton(A, b),
    }
    x = cp.Variable(A.shape[1])
    equality = A @ x == b
    reference = cp.Problem(cp.Minimize(cp.sum(-cp.entr(x))), [equality])
    _reference_solve(reference)
    baseline = results["(c) dual Newton"]
    comparisons = {}
    for name, result in results.items():
        comparisons[name] = {
            "point_error_vs_dual": float(np.linalg.norm(result.x-baseline.x)),
            "multiplier_error_vs_dual": float(np.linalg.norm(result.multiplier-baseline.multiplier)),
            "point_error_vs_cvxpy": float(np.linalg.norm(result.x-x.value)),
            "multiplier_error_vs_cvxpy": float(np.linalg.norm(result.multiplier-equality.dual_value)),
            "objective_error_vs_cvxpy": float(abs(result.objective-reference.value)),
        }
    if output_dir is not None:
        _plot_history(results, output_dir / "10_15_methods.png", "10.15: entropy Newton methods", objective=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(output_dir / "10_15_data.npz", A=A, b=b, xhat=xhat)
    summary = {"exercise": "10.15", "seed": seed, "n": 100, "p": 30,
               "rank_A": int(np.linalg.matrix_rank(A)), "data": "10_15_data.npz" if output_dir is not None else None,
               "methods": {name: result.summary() for name, result in results.items()},
               "comparisons": comparisons,
               "reference": {"solver": "CVXPY/CLARABEL", "status": reference.status,
                             "objective": float(reference.value), "x": x.value.tolist(),
                             "multiplier": equality.dual_value.tolist()},
               "plots": ["10_15_methods.png"] if output_dir is not None else []}
    return _save(output_dir, "10.15", summary)


def run_10_16(output_dir: str | Path | None, seed: int = 1016) -> dict:
    output_dir = Path(output_dir) if output_dir is not None else None
    A, b, c = game_data(seed)
    rng = np.random.default_rng(seed + 1)
    u_direction, v_direction = rng.standard_normal(b.size), rng.standard_normal(c.size)
    u_direction /= np.linalg.norm(u_direction); v_direction /= np.linalg.norm(v_direction)
    results, experiments = {}, {}
    for radius in (0.0, 0.75, 0.97):
        for alpha, beta in ((0.01, 0.5), (0.1, 0.5), (0.25, 0.8)):
            name = f"radius={radius:g}, alpha={alpha:g}, beta={beta:g}"
            result = game_newton(A, b, c, radius*u_direction, radius*v_direction, alpha, beta)
            results[name] = result
            u, v = result.x[:b.size], result.x[b.size:]
            ubr, vbr = game_best_responses(A, b, c, u, v)
            # For minimizer u and maximizer v, both unilateral improvements must vanish.
            gap = game_value(A, b, c, u, vbr) - game_value(A, b, c, ubr, v)
            experiments[name] = {"radius": radius, "alpha": alpha, "beta": beta,
                                 "newton": result.summary(), "norm_u": float(np.linalg.norm(u)),
                                 "norm_v": float(np.linalg.norm(v)),
                                 "best_response_error": float(np.linalg.norm(np.r_[u-ubr, v-vbr])),
                                 "saddle_gap": float(gap)}
    independent = _independent_game_solution(A, b, c)
    for name in experiments:
        experiments[name]["point_error_vs_scipy"] = float(np.linalg.norm(results[name].x-independent))
    if output_dir is not None:
        _plot_history(results, output_dir / "10_16_game.png", "10.16: convex-concave game parameter/start experiment")
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(output_dir / "10_16_data.npz", A=A, b=b, c=c,
                 u_direction=u_direction, v_direction=v_direction)
    summary = {"exercise": "10.16", "seed": seed, "p": 100, "q": 100,
               "data": "10_16_data.npz" if output_dir is not None else None, "experiments": experiments,
               "reference": {"solver": "SciPy root on exact best-response equations",
                             "x": independent.tolist()}, "plots": ["10_16_game.png"] if output_dir is not None else []}
    return _save(output_dir, "10.16", summary)


def run_exercise(exercise: str, output_dir: str | Path | None, seed: int | None = None) -> dict:
    runners: dict[str, Callable] = {"10.14": run_10_14, "10.15": run_10_15, "10.16": run_10_16}
    if exercise not in runners:
        raise ValueError(f"Unsupported chapter 10 exercise: {exercise}")
    return runners[exercise](output_dir, **({} if seed is None else {"seed": seed}))
