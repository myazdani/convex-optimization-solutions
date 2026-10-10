"""5.29: enumerate all sphere-QP KKT pairs, then certify the global optimum."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import brentq

from convex_solutions.io import write_json
from convex_solutions.numerics import solve_cvxpy
from convex_solutions.plotting import pyplot


DIAGONAL = np.array([-3., 1., 2.])
POLES = np.sort(-DIAGONAL)
# Ascending coefficients of the cleared-denominator secular equation.
POLYNOMIAL = np.array([-13., 48., 49., -12., -17., 0., 1.])


def objective(x: np.ndarray) -> float:
    return float(x @ (DIAGONAL * x) + 2 * np.sum(x))


def secular(nu: float) -> float:
    """The sphere constraint after solving stationarity away from the poles."""
    return float(np.sum(1 / (DIAGONAL + nu)**2) - 1)


def secular_derivative(nu: float) -> float:
    return float(-2 * np.sum(1 / (DIAGONAL + nu)**3))


def enumerate_multipliers() -> tuple[np.ndarray, list[dict]]:
    """Bracket every root, including both possible roots in each pole interval.

    F'' = 6 sum_i (d_i+nu)^(-4) > 0 on each finite interval.
    Its unique minimizer determines whether that interval has zero, one,
    or two roots. The two exterior intervals each have exactly one root.
    """
    delta = 1e-8
    left = float(POLES[0] - 1)
    while secular(left) >= 0:
        left = float(POLES[0] - 2 * (POLES[0] - left))
    right = float(POLES[-1] + 1)
    while secular(right) >= 0:
        right = float(POLES[-1] + 2 * (right - POLES[-1]))
    roots = [brentq(secular, left, POLES[0] - delta, xtol=1e-14)]
    intervals = [{"interval": [None, float(POLES[0])], "root_count": 1}]
    for lo, hi in zip(POLES[:-1], POLES[1:]):
        minimum = brentq(secular_derivative, lo + delta, hi - delta, xtol=1e-14)
        value = secular(minimum)
        if value < -1e-12:
            roots.extend([brentq(secular, lo + delta, minimum, xtol=1e-14),
                          brentq(secular, minimum, hi - delta, xtol=1e-14)])
            count = 2
        elif abs(value) <= 1e-12:
            roots.append(minimum)
            count = 1
        else:
            count = 0
        intervals.append({"interval": [float(lo), float(hi)],
                          "minimum_at": float(minimum), "minimum_value": float(value),
                          "root_count": count})
    roots.append(brentq(secular, POLES[-1] + delta, right, xtol=1e-14))
    intervals.append({"interval": [float(POLES[-1]), None], "root_count": 1})
    return np.array(roots), intervals


def stationary_points() -> tuple[list[dict], list[dict]]:
    multipliers, intervals = enumerate_multipliers()
    points = []
    for nu in multipliers:
        shifted = DIAGONAL + nu
        x = -1 / shifted
        tangent = null_space(x[None, :])
        tangent_eigenvalues = np.linalg.eigvalsh(tangent.T @ (2 * shifted[:, None] * tangent))
        if tangent_eigenvalues[0] > 0:
            kind = "strict local minimum"
        elif tangent_eigenvalues[-1] < 0:
            kind = "strict local maximum"
        else:
            kind = "saddle"
        # A finite Lagrange dual value requires D+nu I >= 0. Since the
        # linear vector has no zero coordinates, poles are not feasible KKT pairs.
        global_certificate = bool(np.min(shifted) > 0)
        points.append({
            "nu": float(nu), "x": x.tolist(), "objective": objective(x),
            "classification": kind,
            "sphere_residual": float(abs(x @ x - 1)),
            "stationarity_residual": float(np.linalg.norm(2 * (shifted * x + 1))),
            "tangent_hessian_eigenvalues": tangent_eigenvalues.tolist(),
            "shifted_hessian_eigenvalues": (2 * shifted).tolist(),
            "global_minimum_certificate": global_certificate,
            "dual_value": float(-nu - np.sum(1 / shifted)) if global_certificate else None,
            "dual_status": "finite" if global_certificate else "unbounded below",
        })
    return points, intervals


def _sdp_reference() -> dict:
    """Lifted SDP gives an independent global lower bound, not a local solve."""
    import cvxpy as cp

    x = cp.Variable(3)
    X = cp.Variable((3, 3), symmetric=True)
    lift = cp.bmat([[X, cp.reshape(x, (3, 1), order="F")],
                   [cp.reshape(x, (1, 3), order="F"), np.ones((1, 1))]])
    model = cp.Problem(cp.Minimize(DIAGONAL @ cp.diag(X) + 2 * cp.sum(x)),
                       [lift >> 0, cp.trace(X) == 1])
    reference = solve_cvxpy(model)
    reference.update({"formulation": "lifted semidefinite relaxation", "x": x.value.tolist(),
                      "X": X.value.tolist(),
                      "rank_one_residual": float(np.linalg.norm(X.value - np.outer(x.value, x.value))),
                      "psd_violation": float(max(0., -np.min(np.linalg.eigvalsh(lift.value)))),
                      "trace_residual": float(abs(np.trace(X.value) - 1))})
    return reference


def solve(output_dir: Path | None = None, seed: int = 0) -> dict:
    points, intervals = stationary_points()
    optimum = min(points, key=lambda p: p["objective"])
    maximum = max(points, key=lambda p: p["objective"])
    roots = np.polynomial.polynomial.polyroots(POLYNOMIAL)
    real_roots = sorted(float(z.real) for z in roots if abs(z.imag) < 1e-9)
    reference = _sdp_reference()
    reference["objective_error"] = abs(reference["objective"] - optimum["objective"])
    reference["point_error"] = float(np.linalg.norm(np.array(reference["x"]) - optimum["x"]))
    result = {"exercise": "5.29", "seed": seed, "scope": "optional numerical illustration",
              "points": points, "interval_analysis": intervals,
              "poles": POLES.tolist(), "poles_stationarity_consistent": False,
              "polynomial_coefficients_ascending": POLYNOMIAL.tolist(),
              "independent_polynomial_real_roots": real_roots,
              "polynomial_root_error": float(np.max(np.abs(
                  np.array(real_roots) - np.array([p["nu"] for p in points])))),
              "global_minimum": optimum, "global_maximum": maximum,
              "certified_duality_gap": abs(optimum["objective"] - optimum["dual_value"]),
              "reference": reference, "plots": []}
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plt = pyplot()
        fig, ax = plt.subplots(figsize=(8, 4.8))
        boundaries = [-5., *POLES, 6.]
        for lo, hi in zip(boundaries[:-1], boundaries[1:]):
            grid = np.linspace(lo + 1e-3, hi - 1e-3, 700)
            values = [secular(v) for v in grid]
            ax.plot(grid, values, color="C0")
        for pole in POLES:
            ax.axvline(pole, color="0.65", linestyle="--")
        ax.axhline(0, color="0.25", linewidth=.8)
        for point in points:
            label = point["classification"]
            if point["global_minimum_certificate"]:
                label = "unique global minimum"
            elif point is maximum:
                label = "global maximum"
            elif label == "strict local minimum":
                label = "nonglobal local minimum"
            ax.plot(point["nu"], 0, "o", label=f"{point['nu']:.4f}: {label}")
        ax.set(xlabel="Multiplier ν", ylabel="Secular function F(ν)", ylim=(-1.1, 10),
               title="5.29: every root, with pole-separated intervals")
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(output_dir / "5_29_secular.png", dpi=170)
        plt.close(fig)
        result["plots"] = ["5_29_secular.png"]
        write_json(output_dir / "5.29.json", result)
    return result
