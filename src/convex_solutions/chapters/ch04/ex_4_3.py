"""4.3: exact box-QP certificate with an independent numerical solve."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from convex_solutions.io import write_json
from convex_solutions.numerics import solve_cvxpy


P = np.array([[13., 12., -2.], [12., 17., 6.], [-2., 6., 12.]])
Q = np.array([-22., -14.5, 13.])
R = 1.
CANDIDATE = np.array([1., .5, -1.])


def objective(x: np.ndarray) -> float:
    return float(.5 * x @ P @ x + Q @ x + R)


def certificate() -> dict:
    """Multipliers use x-1 <= 0 and -x-1 <= 0, respectively."""
    x = CANDIDATE.copy()
    upper = np.array([1., 0., 0.])
    lower = np.array([0., 0., 2.])
    return {
        "x": x.tolist(), "objective": objective(x),
        "gradient": (P @ x + Q).tolist(),
        "upper_multipliers": upper.tolist(),
        "lower_multipliers": lower.tolist(),
        "leading_principal_minors": [13, 77, 100],
        "eigenvalues": np.linalg.eigvalsh(P).tolist(),
        "primal_violation": float(max(0., np.max(np.abs(x) - 1))),
        "stationarity_residual": float(np.linalg.norm(P @ x + Q + upper - lower)),
        "complementarity_residual": float(np.linalg.norm(
            np.r_[upper * (x - 1), lower * (-x - 1)])),
        "unique": True,
    }


def solve(output_dir: Path | None = None, seed: int = 0) -> dict:
    """The analytic proof is primary; CVXPY is only a regression check."""
    import cvxpy as cp

    exact = certificate()
    x = cp.Variable(3)
    constraints = [x <= 1, x >= -1]
    model = cp.Problem(cp.Minimize(.5 * cp.quad_form(x, P) + Q @ x + R), constraints)
    reference = solve_cvxpy(model)
    reference.update({
        "x": x.value.tolist(),
        "point_error": float(np.linalg.norm(x.value - CANDIDATE)),
        "objective_error": abs(reference["objective"] - exact["objective"]),
        "primal_violation": float(max(0., np.max(np.abs(x.value) - 1))),
        "stationarity_residual": float(np.linalg.norm(
            P @ x.value + Q + constraints[0].dual_value - constraints[1].dual_value)),
    })
    result = {"exercise": "4.3", "seed": seed, "scope": "optional numerical illustration",
              "certificate": exact, "reference": reference, "plots": []}
    if output_dir is not None:
        write_json(Path(output_dir) / "4.3.json", result)
    return result
