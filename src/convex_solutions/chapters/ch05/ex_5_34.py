"""5.34: LP sensitivity with free primal variables and a quadratic value function."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from convex_solutions.io import write_json
from convex_solutions.plotting import pyplot


A = np.array([[-4., 12., -2., 1.], [-17., 12., 7., 11.], [1., 0., -6., 1.],
              [3., 3., 22., -1.], [-11., 2., -1., -8.]])
B = np.array([8., 13., -4., 27., -18.])
F = np.array([6., 15., -13., 48., 8.])
C = np.array([49., -34., -50., -5.])
D = np.array([3., 8., 21., 25.])
X0 = np.ones(4)
X_DIRECTION = np.array([0., 1., 2., -1.])
Z0 = np.array([0., 2., 1., 2., 2.])
Z_DIRECTION = np.array([0., -1., 2., 0., 2.])
ACTIVE = np.array([1, 2, 3, 4])


def sensitivity_interval() -> tuple[float, float]:
    """All primal slacks and all dual multipliers must be nonnegative."""
    intercepts = np.r_[B - A @ X0, Z0]
    slopes = np.r_[F - A @ X_DIRECTION, Z_DIRECTION]
    lower, upper = -np.inf, np.inf
    for intercept, slope in zip(intercepts, slopes):
        if slope > 0:
            lower = max(lower, -intercept / slope)
        elif slope < 0:
            upper = min(upper, -intercept / slope)
        elif intercept < 0:
            raise ValueError("The proposed basis is never primal-dual feasible.")
    return float(lower), float(upper)


def analytic_solution(epsilon: float) -> dict:
    lower, upper = sensitivity_interval()
    if not lower <= epsilon <= upper:
        raise ValueError(f"The certified quadratic formula holds only on [{lower}, {upper}].")
    x, z = X0 + epsilon * X_DIRECTION, Z0 + epsilon * Z_DIRECTION
    cost, rhs = C + epsilon * D, B + epsilon * F
    slack = rhs - A @ x
    primal, dual = float(cost @ x), float(-rhs @ z)
    return {"epsilon": float(epsilon), "x": x.tolist(), "z": z.tolist(),
            "slack": slack.tolist(), "objective": primal, "dual_objective": dual,
            "formula_value": float(-40 - 72 * epsilon + 25 * epsilon**2),
            "primal_violation": float(max(0., -np.min(slack))),
            "dual_violation": float(max(0., -np.min(z))),
            "stationarity_residual": float(np.linalg.norm(A.T @ z + cost)),
            "complementarity_residual": float(np.linalg.norm(z * slack)),
            "duality_gap": abs(primal - dual),
            "primal_unique": epsilon > lower,
            "dual_unique": epsilon < upper}


def lp_reference(epsilon: float) -> dict:
    """Solve each side independently. SciPy's default x>=0 is explicitly disabled."""
    cost, rhs = C + epsilon * D, B + epsilon * F
    primal = linprog(cost, A_ub=A, b_ub=rhs,
                     bounds=[(None, None)] * A.shape[1], method="highs")
    dual = linprog(rhs, A_eq=A.T, b_eq=-cost,
                   bounds=[(0, None)] * A.shape[0], method="highs")
    if not primal.success or not dual.success:
        raise RuntimeError(f"LP reference failed: primal={primal.message}; dual={dual.message}")
    return {"solver": "SciPy/HiGHS", "primal_status": primal.message, "dual_status": dual.message,
            "x": primal.x.tolist(), "z": dual.x.tolist(), "objective": float(primal.fun),
            "dual_objective": float(-dual.fun), "primal_dual_gap": float(abs(primal.fun + dual.fun)),
            "primal_violation": float(max(0., np.max(A @ primal.x - rhs))),
            "dual_violation": float(max(0., -np.min(dual.x))),
            "stationarity_residual": float(np.linalg.norm(A.T @ dual.x + cost)),
            "complementarity_residual": float(np.linalg.norm(dual.x * (rhs - A @ primal.x)))}


def solve(output_dir: Path | None = None, seed: int = 0) -> dict:
    lower, upper = sensitivity_interval()
    checks = []
    for epsilon in [lower, -.25, 0., .25, .75, upper]:
        analytic, reference = analytic_solution(epsilon), lp_reference(epsilon)
        checks.append({"analytic": analytic, "reference": reference,
                       "objective_error": abs(analytic["objective"] - reference["objective"]),
                       "primal_point_error": float(np.linalg.norm(np.array(analytic["x"]) - reference["x"])),
                       "dual_point_error": float(np.linalg.norm(np.array(analytic["z"]) - reference["z"]))})
    # A separate illustration just outside the certified interval catches the
    # common modeling error x>=0: the actual LP optimum has x4<0 here.
    free_variable_check = {"epsilon": 1.1, "reference": lp_reference(1.1),
                           "purpose": "free-variable regression; outside the affine-basis interval"}
    result = {"exercise": "5.34", "seed": seed, "scope": "optional numerical illustration",
              "active_rows_one_based": (ACTIVE + 1).tolist(),
              "active_determinant": int(round(np.linalg.det(A[ACTIVE]))),
              "x_intercept": X0.tolist(), "x_slope": X_DIRECTION.tolist(),
              "z_intercept": Z0.tolist(), "z_slope": Z_DIRECTION.tolist(),
              "value_coefficients_ascending": [-40, -72, 25],
              "first_order_sensitivity_at_zero": float(D @ X0 - F @ Z0),
              "interval": [lower, upper], "interval_exact": ["-1/2", "1"],
              "slack_intercept": (B - A @ X0).tolist(),
              "slack_slope": (F - A @ X_DIRECTION).tolist(),
              "checks": checks, "free_variable_check": free_variable_check, "plots": []}
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plt = pyplot()
        grid = np.linspace(lower, upper, 101)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].plot(grid, -40 - 72 * grid + 25 * grid**2, label="Certified quadratic value")
        axes[0].scatter([c["analytic"]["epsilon"] for c in checks],
                        [c["reference"]["objective"] for c in checks],
                        marker="x", color="C1", label="Independent LP")
        for i in range(4):
            axes[1].plot(grid, X0[i] + grid * X_DIRECTION[i], label=f"x{i + 1}")
        for i in range(5):
            axes[2].plot(grid, Z0[i] + grid * Z_DIRECTION[i], label=f"z{i + 1}")
        for ax in axes:
            ax.set_xlabel("Perturbation ε")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        axes[0].set_ylabel("Optimal value")
        axes[1].set_ylabel("Primal coordinates")
        axes[2].set_ylabel("Dual coordinates")
        fig.suptitle("5.34: certified sensitivity on [−1/2, 1]")
        fig.tight_layout()
        fig.savefig(output_dir / "5_34_sensitivity.png", dpi=170)
        plt.close(fig)
        result["plots"] = ["5_34_sensitivity.png"]
        write_json(output_dir / "5.34.json", result)
    return result
