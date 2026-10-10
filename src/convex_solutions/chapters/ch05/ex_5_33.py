"""5.33: exact l1 fitting certificate and a maximal affine sensitivity interval."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from convex_solutions.io import write_json
from convex_solutions.plotting import pyplot


A = np.array([[-2., 7., 1.], [-5., -1., 3.], [-7., 3., -5.],
              [-1., 4., -4.], [1., 5., 5.], [2., -5., -1.]])
B = np.array([-4., 3., 9., 0., -11., 5.])
D = np.array([-10., -13., -27., -10., -7., 14.])
X0 = np.ones(3)
X_DIRECTION = np.array([-3., 2., 0.])
DUAL = np.array([1., -.5, .5, -1., 0., 1.])
ZERO_ROWS = np.array([1, 2, 4])


def sensitivity_interval() -> tuple[float, float]:
    """Intersect sign requirements on initially nonzero residuals."""
    r0 = A @ X0 + B
    change = A @ X_DIRECTION + D
    lower, upper = -np.inf, np.inf
    for base, slope in zip(r0, change):
        if base == 0:
            if slope != 0:
                raise ValueError("The zero-residual active set was not preserved.")
            continue
        sign = np.sign(base)
        intercept, derivative = sign * base, sign * slope
        if derivative > 0:
            lower = max(lower, -intercept / derivative)
        elif derivative < 0:
            upper = min(upper, -intercept / derivative)
    return float(lower), float(upper)


def analytic_solution(epsilon: float) -> dict:
    lower, upper = sensitivity_interval()
    if not lower <= epsilon <= upper:
        raise ValueError(f"The certified affine formula holds only on [{lower}, {upper}].")
    x = X0 + epsilon * X_DIRECTION
    residual = A @ x + B + epsilon * D
    primal = float(np.sum(np.abs(residual)))
    dual = float((B + epsilon * D) @ DUAL)
    return {"epsilon": float(epsilon), "x": x.tolist(), "residual": residual.tolist(),
            "objective": primal, "dual_objective": dual,
            "formula_value": float(4 + 7 * epsilon),
            "duality_gap": abs(primal - dual),
            "dual_equality_residual": float(np.linalg.norm(A.T @ DUAL)),
            "dual_bound_violation": float(max(0., np.max(np.abs(DUAL)) - 1)),
            "unique": True}


def lp_reference(epsilon: float) -> dict:
    """Separate primal epigraph and dual LPs, using HiGHS independently of proof."""
    offset = B + epsilon * D
    m, n = A.shape
    primal = linprog(np.r_[np.zeros(n), np.ones(m)],
                     A_ub=np.block([[A, -np.eye(m)], [-A, -np.eye(m)]]),
                     b_ub=np.r_[-offset, offset],
                     bounds=[(None, None)] * n + [(0, None)] * m, method="highs")
    dual = linprog(-offset, A_eq=A.T, b_eq=np.zeros(n),
                   bounds=[(-1, 1)] * m, method="highs")
    if not primal.success or not dual.success:
        raise RuntimeError(f"LP reference failed: primal={primal.message}; dual={dual.message}")
    return {"solver": "SciPy/HiGHS", "primal_status": primal.message, "dual_status": dual.message,
            "x": primal.x[:n].tolist(), "s": dual.x.tolist(),
            "objective": float(primal.fun), "dual_objective": float(-dual.fun),
            "primal_dual_gap": float(abs(primal.fun + dual.fun)),
            "primal_violation": float(max(0., np.max(
                np.abs(A @ primal.x[:n] + offset) - primal.x[n:]))),
            "dual_equality_residual": float(np.linalg.norm(A.T @ dual.x))}


def solve(output_dir: Path | None = None, seed: int = 0) -> dict:
    lower, upper = sensitivity_interval()
    checks = []
    for epsilon in [lower, -.1, 0., .2, .4, upper]:
        analytic = analytic_solution(epsilon)
        reference = lp_reference(epsilon)
        checks.append({"analytic": analytic, "reference": reference,
                       "objective_error": abs(analytic["objective"] - reference["objective"]),
                       "point_error": float(np.linalg.norm(np.array(analytic["x"]) - reference["x"]))})
    result = {"exercise": "5.33", "seed": seed, "scope": "optional numerical illustration",
              "dual_certificate": DUAL.tolist(), "zero_rows_one_based": (ZERO_ROWS + 1).tolist(),
              "active_determinant": int(round(np.linalg.det(A[ZERO_ROWS]))),
              "strict_dual_margin_on_zero_rows": float(1 - np.max(np.abs(DUAL[ZERO_ROWS]))),
              "x_intercept": X0.tolist(), "x_slope": X_DIRECTION.tolist(),
              "value_coefficients_ascending": [4, 7], "interval": [lower, upper],
              "interval_exact": ["-1/5", "1/2"],
              "residual_intercept": (A @ X0 + B).tolist(),
              "residual_slope": (A @ X_DIRECTION + D).tolist(),
              "checks": checks, "plots": []}
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plt = pyplot()
        grid = np.linspace(lower, upper, 101)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(grid, 4 + 7 * grid, label="Certified affine value")
        axes[0].scatter([c["analytic"]["epsilon"] for c in checks],
                        [c["reference"]["objective"] for c in checks],
                        marker="x", color="C1", label="Independent LP")
        for i in range(3):
            axes[1].plot(grid, X0[i] + grid * X_DIRECTION[i], label=f"x{i + 1}")
        for ax in axes:
            ax.set_xlabel("Perturbation ε")
            ax.grid(alpha=.2)
            ax.legend()
        axes[0].set_ylabel("Optimal value")
        axes[1].set_ylabel("Unique optimal coordinates")
        fig.suptitle("5.33: exact sensitivity on [−1/5, 1/2]")
        fig.tight_layout()
        fig.savefig(output_dir / "5_33_sensitivity.png", dpi=170)
        plt.close(fig)
        result["plots"] = ["5_33_sensitivity.png"]
        write_json(output_dir / "5.33.json", result)
    return result
