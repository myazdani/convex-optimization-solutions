"""Dense Newton routines used by the chapter 11 implementations.

The solvers deliberately implement the iterations rather than forwarding the
optimization problem to a modeling package. CVXPY is only an independent check.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
from scipy.linalg import cho_factor, cho_solve, LinAlgError

Array = np.ndarray
Oracle = Callable[[Array, float], tuple[float, Array, Array]]


@dataclass
class BarrierResult:
    x: Array
    t: float
    history: list[dict]
    newton_steps: int


def positive_solve(hessian: Array, rhs: Array) -> Array:
    """Solve a positive definite system; never silently regularize a bad model."""
    hessian = (hessian + hessian.T) / 2
    try:
        return cho_solve(cho_factor(hessian, lower=True, check_finite=False), rhs,
                         check_finite=False)
    except LinAlgError:
        # A convex problem can have flat, redundant directions. Use a minimum-
        # norm Newton step only if the PSD system is actually compatible;
        # do not add a ridge or hide a direction of unbounded decrease.
        eigenvalues, vectors = np.linalg.eigh(hessian)
        threshold = 20*np.finfo(float).eps*len(rhs)*max(1.,float(np.max(np.abs(eigenvalues))))
        if eigenvalues[0] < -threshold:
            raise RuntimeError("Newton Hessian is numerically indefinite")
        positive = eigenvalues > threshold
        coordinates = vectors[:, positive].T @ rhs
        solution = vectors[:, positive] @ (coordinates / eigenvalues[positive])
        if np.linalg.norm(hessian @ solution - rhs) > 1e-7*max(1.,np.linalg.norm(rhs)):
            raise ValueError("Singular Newton system has an incompatible gradient; check unbounded flat directions")
        return solution


def barrier_method(oracle: Oracle, initial: Array, barrier_degree: int, *,
                   tolerance: float = 1e-6, mu: float = 12.0,
                   centering_tolerance: float = 1e-10,
                   max_inner: int = 120, max_outer: int = 60,
                   metrics: Callable[[Array, float], dict] | None = None) -> BarrierResult:
    """Damped Newton centering with domain-aware Armijo backtracking.

    An oracle returns +inf outside its open domain. History contains one entry
    per accepted Newton step and a centered entry for every outer iteration.
    The degree/t value is a nominal central-path bound: it is exact only at
    an exact center. Finite-centering certificates are computed by callers;
    intermediate entries are explicitly labeled. The initial objective must be finite.
    """
    if tolerance <= 0 or mu <= 1 or barrier_degree < 1:
        raise ValueError("Require tolerance > 0, mu > 1, and positive degree")
    x = np.asarray(initial, dtype=float).copy()
    t = 1.0
    history = []
    total = 0
    for outer in range(max_outer):
        for inner in range(max_inner):
            value, gradient, hessian = oracle(x, t)
            if not np.isfinite(value):
                raise ValueError("Initial/iterated point is outside the strict domain")
            direction = positive_solve(hessian, -gradient)
            decrement_squared = float(-gradient @ direction)
            if decrement_squared < -1e-9:
                raise RuntimeError("Newton direction is not descending")
            decrement_squared = max(decrement_squared, 0.0)
            if decrement_squared / 2 <= centering_tolerance:
                break
            step = 1.0
            directional_derivative = float(gradient @ direction)
            for _ in range(100):
                trial = x + step * direction
                trial_value = oracle(trial, t)[0]
                # The roundoff allowance is bounded relative to the actual
                # centering value; it is not a new convergence test.
                allowance = 8 * np.finfo(float).eps * max(1.0, abs(value))
                if np.isfinite(trial_value) and trial_value <= value + .01 * step * directional_derivative + allowance:
                    break
                step *= .5
            else:
                raise RuntimeError("Barrier line search failed")
            if step < 1e-15:
                raise RuntimeError("Barrier line search stagnated")
            x = trial
            total += 1
            history.append({"newton_step": total, "outer_iteration": outer,
                            "t": t, "barrier_gap_bound": barrier_degree / t,
                            "centered": False, "step_length": step,
                            "decrement_squared": decrement_squared,
                            **(metrics(x, t) if metrics else {})})
        else:
            raise RuntimeError(f"Centering did not converge at t={t:g}")
        history.append({"newton_step": total, "outer_iteration": outer, "t": t,
                        "barrier_gap_bound": barrier_degree / t,
                        "centered": True, "decrement_squared": decrement_squared,
                        **(metrics(x, t) if metrics else {})})
        if barrier_degree / t <= tolerance:
            return BarrierResult(x, t, history, total)
        t = min(t * mu, barrier_degree / tolerance)
    raise RuntimeError("Barrier outer iteration limit exceeded")


def infeasible_oracle(n: int) -> tuple[float, Array, Array]:
    return np.inf, np.zeros(n), np.eye(n)


def symmetric_zero_diagonal_weights(n: int, rng: np.random.Generator) -> Array:
    matrix = rng.standard_normal((n, n)) / np.sqrt(n)
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 0)
    return matrix
