"""Shared solver selection and numerical quality checks."""
from __future__ import annotations
import numpy as np


def solve_cvxpy(problem, *, solver: str | None = None, allow_inaccurate: bool = False,
                **kwargs) -> dict:
    """Solve with an installed open-source backend; do not hide solver failures.

    A failed solver may be retried using another installed backend. This helper
    does not itself certify residuals: each exercise must check its own model.
    """
    import cvxpy as cp
    available = set(cp.installed_solvers())
    candidates = [solver] if solver else ["CLARABEL", "HIGHS", "OSQP", "SCS", "SCIPY"]
    candidates = [name for name in candidates if name in available]
    if not candidates:
        raise RuntimeError(f"Requested solver unavailable; installed: {sorted(available)}")
    attempts = []
    accepted = {cp.OPTIMAL}
    if allow_inaccurate:
        accepted.add(cp.OPTIMAL_INACCURATE)
    for name in candidates:
        try:
            problem.solve(solver=name, **kwargs)
        except cp.error.SolverError as exc:
            attempts.append({"solver": name, "error": str(exc)})
            continue
        attempts.append({"solver": name, "status": problem.status})
        if problem.status in accepted:
            return {"solver": name, "status": problem.status,
                    "objective": float(problem.value), "attempts": attempts}
        if problem.status in {cp.INFEASIBLE, cp.UNBOUNDED}:
            break
    raise RuntimeError(f"No accepted optimal solution: {attempts}")


def max_violation(problem) -> float:
    """Maximum CVXPY constraint violation after a numerical solve."""
    violations = [float(np.max(np.asarray(c.violation()))) for c in problem.constraints]
    return max([0.0, *violations])


def assert_close(actual, expected, *, atol: float = 1e-6, rtol: float = 1e-5) -> None:
    """A concise numeric check usable from both scripts and tests."""
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
