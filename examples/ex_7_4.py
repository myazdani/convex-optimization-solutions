"""Example 7.4 / Figure 7.4: exact-data binary detector design.

Original Python implementation of the problem on textbook pp. 371–372.
The four-row distribution matrix is printed in equation (7.15). No random
or third-party datasets, MATLAB runtime, or copied reference code are needed.
"""
from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from convex_solutions.io import write_json
from convex_solutions.numerics import max_violation, solve_cvxpy
from convex_solutions.plotting import pyplot

P = np.array([[.70, .10], [.20, .10], [.05, .70], [.05, .10]])
BOOK_MINIMAX_POSITIVE = np.array([0., 1 / 3, 1., 1.])
BOOK_PDF = "https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf"
REFERENCE = "https://web.cvxr.com/cvx/examples/cvxbook/Ch07_statistical_estim/html/detector2.html"


def errors(positive: np.ndarray) -> np.ndarray:
    """Return [false positive, false negative] for positive-test probabilities."""
    positive = np.asarray(positive, dtype=float)
    return np.array([P[:, 0] @ positive, P[:, 1] @ (1 - positive)])


def detector(positive: np.ndarray) -> np.ndarray:
    return np.vstack([1 - positive, positive])


def threshold_vertices() -> dict:
    """Exact ROC vertices: add outcomes in descending likelihood ratio q/p."""
    order = np.argsort(-P[:, 1] / P[:, 0], kind="stable")
    positive = np.zeros(4)
    rules, points = [positive.copy()], [errors(positive)]
    for observation in order:
        positive[observation] = 1
        rules.append(positive.copy())
        points.append(errors(positive))
    return {"observation_order_one_based": (order + 1).tolist(),
            "positive_probabilities": np.array(rules), "error_pairs": np.array(points)}


def deterministic_detectors() -> dict:
    rules = np.array(list(product([0., 1.], repeat=4)))
    points = np.array([errors(rule) for rule in rules])
    worst = points.max(axis=1)
    best = float(worst.min())
    return {"count": len(rules), "positive_probabilities": rules,
            "error_pairs": points, "worst_errors": worst,
            "minimax_error": best,
            "minimax_rule_indices": np.flatnonzero(np.isclose(worst, best)).tolist()}


def exact_certificate() -> dict:
    """An attainable rule and a weighted-error lower bound certify optimality."""
    weights = np.array([1 / 3, 2 / 3])
    lower_bound = float(np.minimum(weights[0] * P[:, 0], weights[1] * P[:, 1]).sum())
    point = errors(BOOK_MINIMAX_POSITIVE)
    return {"positive_probabilities": BOOK_MINIMAX_POSITIVE.copy(),
            "detector": detector(BOOK_MINIMAX_POSITIVE), "error_pair": point,
            "worst_error": float(point.max()), "error_weights": weights,
            "weighted_error_lower_bound": lower_bound,
            "gap": float(point.max() - lower_bound)}


def solve_minimax() -> dict:
    import cvxpy as cp

    positive, worst = cp.Variable(4), cp.Variable()
    pfp, pfn = P[:, 0] @ positive, P[:, 1] @ (1 - positive)
    constraints = [positive >= 0, positive <= 1, pfp <= worst, pfn <= worst]
    problem = cp.Problem(cp.Minimize(worst), constraints)
    outcome = solve_cvxpy(problem, solver="CLARABEL", tol_gap_abs=1e-10,
                          tol_gap_rel=1e-10, tol_feas=1e-10)
    lower, upper = (np.asarray(c.dual_value) for c in constraints[:2])
    fp_weight, fn_weight = (float(c.dual_value) for c in constraints[2:])
    positive_value = np.asarray(positive.value)
    dual_objective = fn_weight * P[:, 1].sum() - upper.sum()
    point = errors(positive_value)
    stationarity = fp_weight * P[:, 0] - fn_weight * P[:, 1] - lower + upper
    complementarity = np.r_[lower * positive_value, upper * (1 - positive_value),
                            fp_weight * (worst.value - point[0]),
                            fn_weight * (worst.value - point[1])]
    outcome.update({"positive_probabilities": positive_value,
                    "detector": detector(positive_value), "error_pair": point,
                    "worst_error": float(point.max()), "primal_violation": max_violation(problem),
                    "dual_objective": float(dual_objective),
                    "primal_dual_gap": float(problem.value - dual_objective),
                    "stationarity_residual": float(max(np.max(np.abs(stationarity)),
                                                        abs(1 - fp_weight - fn_weight))),
                    "complementarity_residual": float(np.max(np.abs(complementarity))),
                    "dual_error_weights": [fp_weight, fn_weight],
                    "point_error": float(np.max(np.abs(positive_value - BOOK_MINIMAX_POSITIVE)))})
    return outcome


def scipy_reference() -> dict:
    """Independent LP frontend/backend; variables are (r_1,...,r_4,t)."""
    inequalities = np.vstack([np.r_[P[:, 0], -1], np.r_[-P[:, 1], -1]])
    answer = linprog([0, 0, 0, 0, 1], A_ub=inequalities,
                     b_ub=[0, -P[:, 1].sum()], bounds=[(0, 1)] * 4 + [(None, None)],
                     method="highs")
    if not answer.success:
        raise RuntimeError(f"Independent HiGHS solve failed: {answer.message}")
    return {"solver": "SciPy linprog / HiGHS", "status": answer.message,
            "positive_probabilities": answer.x[:4], "error_pair": errors(answer.x[:4]),
            "objective": float(answer.fun)}


def trace_frontier(budgets: np.ndarray | None = None) -> dict:
    """Solve min Pfn with Pfp <= alpha; compare to exact piecewise-linear ROC."""
    import cvxpy as cp

    vertices = threshold_vertices()["error_pairs"]
    if budgets is None:
        budgets = np.unique(np.r_[np.linspace(0, 1, 101), vertices[:, 0], 1 / 6])
    budgets = np.asarray(budgets, dtype=float)
    if budgets.ndim != 1 or budgets.size == 0 or np.any(~np.isfinite(budgets)) or np.any((budgets < 0) | (budgets > 1)):
        raise ValueError("False-positive budgets must be finite values in [0, 1]")
    positive = cp.Variable(4)
    alpha = cp.Parameter(nonneg=True)
    constraints = [positive >= 0, positive <= 1, P[:, 0] @ positive <= alpha]
    problem = cp.Problem(cp.Minimize(P[:, 1] @ (1 - positive)), constraints)
    measured, violations = [], []
    for budget in budgets:
        alpha.value = float(budget)
        solve_cvxpy(problem, solver="CLARABEL", tol_gap_abs=1e-10,
                    tol_gap_rel=1e-10, tol_feas=1e-10)
        measured.append(errors(positive.value))
        violations.append(max_violation(problem))
    points = np.asarray(measured)
    exact = np.interp(budgets, vertices[:, 0], vertices[:, 1])
    return {"budgets": budgets, "error_pairs": points, "exact_false_negatives": exact,
            "maximum_interpolation_error": float(np.max(np.abs(points[:, 1] - exact))),
            "maximum_primal_violation": float(max(violations, default=0.)),
            "point_count": len(budgets)}


def make_plot(output_dir: Path, vertices: dict, deterministic: dict, minimax: dict) -> str:
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
    points = np.asarray(deterministic["error_pairs"])
    curve = np.asarray(vertices["error_pairs"])
    ax.scatter(points[:, 0], points[:, 1], s=28, color="#a9b4bd", label="All 16 deterministic rules", zorder=2)
    ax.plot(curve[:, 0], curve[:, 1], "o-", color="#075b9a", linewidth=2.2,
            markersize=5, label="Optimal randomized frontier", zorder=3)
    ax.plot([0, 1], [0, 1], "--", color="#69777f", linewidth=1, label="Equal error rates")
    point = np.asarray(minimax["error_pair"])
    ax.scatter(*point, s=90, marker="*", color="#c5531c", label="Minimax: (1/6, 1/6)", zorder=5)
    for index, point in enumerate(curve[1:-1], 1):
        ax.annotate(str(index), point, xytext=(7, 7), textcoords="offset points")
    ax.annotate("4", minimax["error_pair"], xytext=(8, -12), textcoords="offset points", color="#a04013")
    ax.set(xlabel="False-positive probability", ylabel="False-negative probability",
           title="Example 7.4: four-outcome binary detector", xlim=(-.02, 1.02), ylim=(-.02, 1.02))
    ax.grid(alpha=.2)
    ax.legend(loc="upper right", fontsize=8)
    path = output_dir / "receiver_operating_curve.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def solve(output_dir: Path | None = None, seed: int = 0) -> dict:
    vertices, deterministic = threshold_vertices(), deterministic_detectors()
    minimax, exact, independent = solve_minimax(), exact_certificate(), scipy_reference()
    frontier = trace_frontier()
    checks = {"minimax_exact_error": abs(minimax["objective"] - 1 / 6),
              "independent_solver_error": abs(minimax["objective"] - independent["objective"]),
              "randomization_improvement": deterministic["minimax_error"] - minimax["worst_error"]}
    if (checks["minimax_exact_error"] > 1e-7 or checks["independent_solver_error"] > 1e-7
            or minimax["primal_violation"] > 1e-7 or minimax["stationarity_residual"] > 1e-7
            or frontier["maximum_interpolation_error"] > 1e-7
            or frontier["maximum_primal_violation"] > 1e-7):
        raise RuntimeError(f"Binary detector numerical checks failed: {checks}")
    result = {"example": "7.4", "seed": seed, "randomness": "none; seed unused",
              "data": {"kind": "exact printed textbook data", "P": P},
              "source": {"book_pdf": BOOK_PDF,
                         "book_pdf_sha256": "40d976c83c18cce1900eff8c41bd5ad408c102b813af39d05ff85678ccf8d76e", "printed_pages": [371, 372],
                         "pdf_pages": [385, 386], "equation": "7.15", "figure": "7.4",
                         "official_cvx_reference": REFERENCE,
                         "implementation": "original Python; no reference code copied"},
              "vertices": vertices, "deterministic": deterministic, "exact_certificate": exact,
              "minimax": minimax, "independent_reference": independent,
              "frontier": frontier, "checks": checks, "plots": []}
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result["plots"] = [make_plot(output_dir, vertices, deterministic, minimax)]
        write_json(output_dir / "7.4.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    """Run this worked example without changing the historical exercise CLI."""
    import argparse
    from datetime import datetime, timezone
    import hashlib
    import time
    import shlex
    import sys

    from convex_solutions.cli import environment, source_fingerprint
    from convex_solutions.io import json_safe

    example_id = __file__.rsplit("ex_", 1)[1].removesuffix(".py").replace("_", ".")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/examples") / example_id)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    result = solve(output_dir=args.output, seed=args.seed)
    envelope = {"example_id": example_id, "seed": args.seed,
                "executed_at": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": time.perf_counter() - started,
                "environment": environment(), "source_tree_sha256": source_fingerprint(),
                "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "implementation_path": f"examples/ex_{example_id.replace('.', '_')}.py",
                "command": shlex.join([sys.executable, str(Path(__file__)),
                                       "--output", str(args.output), "--seed", str(args.seed)]),
                "working_directory": str(Path.cwd()),
                "results": json_safe(result)}
    write_json(args.output / "results.json", envelope)
    print(f"Example {example_id}: verified and wrote {args.output / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
