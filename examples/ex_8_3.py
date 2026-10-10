"""Example 8.3: exact-data correlation bounds with independent certificates.

Original Python implementation of the SDP printed on textbook p. 408.
The bounds and unit diagonal are printed data; the analytic Gram witnesses
and heatmap are original verification/visualization, not extra input data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from convex_solutions.io import write_json
from convex_solutions.numerics import max_violation, solve_cvxpy
from convex_solutions.plotting import pyplot

# Zero-based matrix indices. Textbook correlation labels use one-based indices.
BOUNDS = [(0, 1, .6, .9), (0, 2, .8, .9), (1, 3, .5, .7), (2, 3, -.8, -.4)]
BOOK_PDF = "https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf"
REFERENCE = "https://web.cvxr.com/cvx/examples/cvxbook/Ch08_geometric_probs/html/ex_8_3.html"


def feasibility(matrix: np.ndarray) -> dict:
    matrix = np.asarray(matrix, dtype=float)
    eigenvalues = np.linalg.eigvalsh(matrix)
    bound_violation = max([0.] + [max(lo - matrix[i, j], matrix[i, j] - hi)
                                 for i, j, lo, hi in BOUNDS])
    return {"eigenvalues": eigenvalues,
            "psd_violation": float(max(0., -eigenvalues[0])),
            "diagonal_residual": float(np.max(np.abs(np.diag(matrix) - 1))),
            "symmetry_residual": float(np.max(np.abs(matrix - matrix.T))),
            "bound_violation": float(bound_violation)}


def analytic_bounds() -> dict:
    """Sharp 3x3 principal-minor bounds plus feasible full 4x4 Gram matrices."""
    lower = .6 * .5 - np.sqrt((1 - .6 ** 2) * (1 - .5 ** 2))
    upper = .8 * (-.4) + np.sqrt((1 - .8 ** 2) * (1 - (-.4) ** 2))
    witnesses = {}
    for label, first, fourth, target, free_index, fixed_index in [
        ("lower", [.6, .8, 0.], [.5, -np.sqrt(.75), 0.], [.85, -.55], 2, 1),
        ("upper", [.8, .6, 0.], [-.4, np.sqrt(.84), 0.], [.72, .6], 1, 2),
    ]:
        vectors = np.zeros((4, 3))
        vectors[0], vectors[3], vectors[fixed_index] = first, fourth, [1., 0., 0.]
        planar = np.linalg.solve(vectors[[0, 3], :2], target)
        residual = 1 - float(planar @ planar)
        if residual < 0:
            raise RuntimeError("Analytic Gram witness is not a unit-vector construction")
        vectors[free_index] = np.r_[planar, np.sqrt(residual)]
        matrix = vectors @ vectors.T
        witnesses[label] = {"unit_vectors_as_rows": vectors, "matrix": matrix,
                            "rho14": float(matrix[0, 3]), **feasibility(matrix)}
    return {"lower": float(lower), "upper": float(upper), "witnesses": witnesses}


def symmetric_entry(i: int, j: int) -> np.ndarray:
    """Matrix H satisfying trace(H C) = C[i,j] for a symmetric C."""
    matrix = np.zeros((4, 4))
    matrix[i, j] += .5
    matrix[j, i] += .5
    return matrix


def solve_bound(direction: str, solver: str = "CLARABEL") -> dict:
    import cvxpy as cp

    if direction not in {"lower", "upper"}:
        raise ValueError("direction must be 'lower' or 'upper'")
    if solver not in {"CLARABEL", "SCS"}:
        raise ValueError("solver must be CLARABEL or SCS")
    sign = 1. if direction == "lower" else -1.
    matrix = cp.Variable((4, 4), symmetric=True)
    constraints = [matrix >> 0, cp.diag(matrix) == 1]
    for i, j, low, high in BOUNDS:
        constraints.extend([matrix[i, j] >= low, matrix[i, j] <= high])
    problem = cp.Problem(cp.Minimize(sign * matrix[0, 3]), constraints)
    options = ({"tol_gap_abs": 1e-10, "tol_gap_rel": 1e-10, "tol_feas": 1e-10}
               if solver == "CLARABEL" else {"eps": 1e-8, "max_iters": 100000})
    outcome = solve_cvxpy(problem, solver=solver, **options)
    value = np.asarray(matrix.value)
    psd_dual = np.asarray(constraints[0].dual_value)
    diagonal_dual = np.asarray(constraints[1].dual_value)
    stationarity = sign * symmetric_entry(0, 3) + np.diag(diagonal_dual) - psd_dual
    dual_objective = -float(diagonal_dual.sum())
    complementarity = [float(np.trace(psd_dual @ value))]
    scalar_duals = []
    for pair, (i, j, low, high) in enumerate(BOUNDS):
        low_dual, high_dual = [float(c.dual_value) for c in constraints[2 + 2 * pair:4 + 2 * pair]]
        scalar_duals.extend([low_dual, high_dual])
        stationarity += (high_dual - low_dual) * symmetric_entry(i, j)
        dual_objective += low * low_dual - high * high_dual
        complementarity.extend([low_dual * (value[i, j] - low),
                                high_dual * (high - value[i, j])])
    dual_violation = max(0., -float(np.linalg.eigvalsh(psd_dual)[0]), -min(scalar_duals))
    outcome.update({"direction": direction, "rho14": float(value[0, 3]), "matrix": value,
                    "minimization_objective": float(problem.value),
                    "dual_objective": dual_objective,
                    "primal_dual_gap": float(problem.value - dual_objective),
                    "primal_violation": max_violation(problem), "dual_violation": dual_violation,
                    "stationarity_residual": float(np.max(np.abs(stationarity))),
                    "complementarity_residual": float(np.max(np.abs(complementarity))),
                    "psd_dual_matrix": psd_dual, "diagonal_dual": diagonal_dual,
                    "bound_duals_lower_upper": np.array(scalar_duals).reshape(-1, 2),
                    **feasibility(value)})
    return outcome


def make_plot(output_dir: Path, lower: dict, upper: dict) -> str:
    plt = pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), constrained_layout=True)
    for ax, label, result in zip(axes, ["Minimum", "Maximum"], [lower, upper]):
        matrix = np.asarray(result["matrix"])
        picture = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{matrix[i, j]:.4f}", ha="center", va="center",
                        fontsize=9, color="white" if abs(matrix[i, j]) > .65 else "black")
        ax.set_xticks(range(4), ["1", "2", "3", "4"])
        ax.set_yticks(range(4), ["1", "2", "3", "4"])
        ax.set(xlabel="Vector index", ylabel="Vector index",
               title=f"{label}: rho14 = {result['rho14']:.8f}")
    fig.colorbar(picture, ax=list(axes), fraction=.045, pad=.025, label="Correlation")
    fig.suptitle("Example 8.3: feasible endpoint correlation matrices", fontsize=13)
    path = output_dir / "correlation_bound_matrices.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def solve(output_dir: Path | None = None, seed: int = 0) -> dict:
    exact = analytic_bounds()
    lower, upper = solve_bound("lower"), solve_bound("upper")
    independent = {direction: solve_bound(direction, solver="SCS") for direction in ["lower", "upper"]}
    checks = {}
    for direction, result in [("lower", lower), ("upper", upper)]:
        checks[f"{direction}_analytic_error"] = abs(result["rho14"] - exact[direction])
        checks[f"{direction}_solver_difference"] = abs(result["rho14"] - independent[direction]["rho14"])
        for computed in [result, independent[direction]]:
            for key in ["primal_violation", "dual_violation", "stationarity_residual",
                        "complementarity_residual"]:
                if computed[key] > 2e-6:
                    raise RuntimeError(f"{direction} {computed['solver']} {key}: {computed[key]}")
            if abs(computed["primal_dual_gap"]) > 2e-6:
                raise RuntimeError(f"{direction} duality gap exceeds tolerance")
        if checks[f"{direction}_analytic_error"] > 1e-7 or checks[f"{direction}_solver_difference"] > 2e-6:
            raise RuntimeError(f"Independent correlation checks failed: {checks}")
    result = {"example": "8.3", "seed": seed, "randomness": "none; seed unused",
              "data": {"kind": "exact printed textbook data", "dimension": 4,
                       "bounds_one_based": [[i + 1, j + 1, lo, hi] for i, j, lo, hi in BOUNDS]},
              "source": {"book_pdf": BOOK_PDF,
                         "book_pdf_sha256": "40d976c83c18cce1900eff8c41bd5ad408c102b813af39d05ff85678ccf8d76e", "printed_pages": [408], "pdf_pages": [422],
                         "equation": "8.7", "official_cvx_reference": REFERENCE,
                         "implementation": "original Python; no reference code copied",
                         "figure": "original heatmap; no textbook figure for this example"},
              "lower": lower, "upper": upper, "independent_reference": independent,
              "analytic_certificate": exact, "checks": checks, "plots": []}
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result["plots"] = [make_plot(output_dir, lower, upper)]
        write_json(output_dir / "8.3.json", result)
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
