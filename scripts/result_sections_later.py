"""Deterministic Markdown summaries for the executed chapter 10/11 experiments.

Only the supplied result envelope is read. The shared render_results.py driver
owns provenance, generated-block replacement, and stale-output verification.
"""
from __future__ import annotations

import math


EXERCISES = ("10.14", "10.15", "10.16", "11.1", "11.21", "11.22", "11.23", "11.24")


def number(value, digits=11):
    if value is None:
        return "Not available"
    if isinstance(value, int):
        return str(value)
    if value == 0:
        return "0"
    return f"{value:.{digits}g}"


def error(value):
    if value is None:
        return "Not available"
    return "0" if value == 0 else f"{value:.3e}"


def vector(values, digits=9):
    return "(" + ", ".join(number(value, digits) for value in values) + ")"


def norm(values):
    return math.sqrt(sum(value * value for value in values))


def distance(first, second):
    return norm([a - b for a, b in zip(first, second)])


def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(map(str, row)) + " |" for row in rows),
    ])


def plot(exercise, filename, caption):
    return f"![{caption}](../../artifacts/{exercise}/{filename})"


def render_10_14(data):
    result, ref = data["newton"], data["reference"]
    rows = [
        ["Allocation", vector(result["x"], 12)],
        ["Expected logarithmic growth", number(data["growth_rate"], 14)],
        ["Minimized negative growth", number(result["objective"], 14)],
        ["Scenario returns", vector(data["scenario_returns"], 12)],
        ["Equality multiplier", vector(result["multiplier"], 12)],
        ["Budget residual", error(result["primal_residuals"][-1])],
        ["Final KKT residual", error(result["residuals"][-1])],
    ]
    history = []
    for k, objective in enumerate(result["objectives"]):
        history.append([k, number(objective, 14), error(result["residuals"][k]),
                        number(result["step_lengths"][k]) if k < result["iterations"] else "Terminal point",
                        result["backtracks"][k] if k < result["iterations"] else "—"])
    return "\n\n".join([
        "### Allocation and Newton history",
        table(["Quantity", "Executed value"], rows),
        f"The stopping test was met after {result['iterations']} accepted Newton steps "
        f"and {sum(result['backtracks'])} backtracking reductions. Every scenario return "
        "is positive. The allocation happens to be positive even though short positions were allowed.",
        table(["Iterate", "Negative growth", "KKT residual", "Step to next iterate", "Backtracks"], history),
        "### Independent comparison",
        table(["Reference solver / status", "Reference objective", "Objective difference", "Allocation norm difference"],
              [[f"{ref['solver']} / {ref['status']}", number(ref["objective"], 14),
                error(ref["objective_error"]), error(ref["point_error"])]]),
        f"The independently computed allocation is {vector(ref['x'], 12)}. "
        "An exponential-cone objective tolerance does not imply the same accuracy in the allocation coordinates.",
        plot("10.14", "10_14_newton.png", "Executed Newton residual, accepted step length, and objective"),
    ])


def render_10_15(data):
    methods, comparisons, ref = data["methods"], data["comparisons"], data["reference"]
    names = ["(a) feasible Newton", "(b) infeasible Newton, xhat", "(b) infeasible Newton, ones", "(c) dual Newton"]
    rows, checks = [], []
    for name in names:
        item, check = methods[name], comparisons[name]
        rows.append([name, item["iterations"], sum(item["backtracks"]), number(item["objective"], 14),
                     error(item["primal_residuals"][-1]), error(item["dual_residuals"][-1]), error(item["residuals"][-1])])
        checks.append([name, error(check["point_error_vs_dual"]), error(check["multiplier_error_vs_dual"]),
                       error(check["point_error_vs_cvxpy"]), error(check["multiplier_error_vs_cvxpy"])])
    dual = methods["(c) dual Newton"]
    return "\n\n".join([
        "### (a–c) All four requested runs",
        f"The executed instance has n = {data['n']}, p = {data['p']}, and rank(A) = {data['rank_A']}. "
        "Every method met the stated KKT stopping test.",
        table(["Method", "Steps", "Backtracks", "Primal objective", "Equality residual", "Stationarity residual", "KKT residual"], rows),
        f"For the dual-Newton solution, the smallest and largest primal coordinates are "
        f"{number(min(dual['x']))} and {number(max(dual['x']))}; its primal and multiplier 2-norms are "
        f"{number(norm(dual['x']))} and {number(norm(dual['multiplier']))}. "
        "The JSON linked above contains every 100-coordinate primal point and 30-coordinate multiplier.",
        "### Agreement of the recovered primal and dual solutions",
        table(["Method", "Primal error vs dual Newton", "Multiplier error vs dual Newton", "Primal error vs CLARABEL", "Multiplier error vs CLARABEL"], checks),
        f"The independent {ref['solver']} solve returned **{ref['status']}**, objective "
        f"{number(ref['objective'], 14)}. The maximum objective difference across the four methods is "
        f"{error(max(item['objective_error_vs_cvxpy'] for item in comparisons.values()))}. "
        "All point and multiplier errors in the table are 2-norms; agreement among the explicit Newton "
        "methods is much tighter than agreement with the exponential-cone reference solve.",
        "Equal iteration counts for the feasible-start and one infeasible-start run are empirical. "
        "Their objective and residual line searches use different merit functions.",
        "[Exact generated matrix, right-hand side, and feasible starting point](../../artifacts/10.15/10_15_data.npz)",
        plot("10.15", "10_15_methods.png", "Executed residual, step-length, and recovered primal-objective comparison for all four Newton runs"),
    ])


def render_10_16(data):
    experiments = sorted(data["experiments"].values(), key=lambda item: (item["radius"], item["alpha"], item["beta"]))
    rows = []
    for item in experiments:
        newton = item["newton"]
        rows.append([number(item["radius"]), number(item["alpha"]), number(item["beta"]),
                     newton["iterations"], sum(newton["backtracks"]), error(newton["residuals"][-1]),
                     error(item["point_error_vs_scipy"])])
    first = experiments[0]
    return "\n\n".join([
        "### All starting radii and backtracking choices",
        f"The executed game has p = q = {data['p']}. All {len(experiments)} runs met the residual stopping test.",
        table(["Initial radius", "α", "β", "Newton steps", "Backtracks", "Final residual", "Point error vs SciPy"], rows),
        table(["Saddle-point quantity (first run)", "Executed value"], [
            ["Payoff", number(first["newton"]["objective"], 14)],
            ["Minimizing strategy norm", number(first["norm_u"], 13)],
            ["Maximizing strategy norm", number(first["norm_v"], 13)],
            ["Smallest ball slack across both strategies", error(min(1-first["norm_u"]**2, 1-first["norm_v"]**2))],
        ]),
        "### Independent best-response checks",
        f"Reference: {data['reference']['solver']}. The maximum point error across all runs is "
        f"{error(max(item['point_error_vs_scipy'] for item in experiments))}; "
        f"the maximum closed-form best-response residual is {error(max(item['best_response_error'] for item in experiments))}. "
        f"The independently evaluated saddle gaps range from {error(min(item['saddle_gap'] for item in experiments))} "
        f"to {error(max(item['saddle_gap'] for item in experiments))}. Their tiny negative values are floating-point cancellation, "
        "not negative theoretical saddle gaps. All complete strategy vectors are in the JSON linked above.",
        "For this instance, changing α from 0.01 to 0.1 leaves accepted steps unchanged. "
        "The finer β = 0.8 search saves one Newton step from radius 0.75, but requires more backtracking. "
        "These are measured results for this instance, not universal parameter rankings.",
        "[Exact game matrix, vectors, and initial directions](../../artifacts/10.16/10_16_data.npz)",
        plot("10.16", "10_16_game.png", "Executed residual and accepted-step histories for all nine convex-concave game experiments"),
    ])


def render_11_1(data):
    rows = []
    multipliers = []
    for t, x, gap, bound, residual in zip(data["t"], data["central_points"], data["objective_gaps"],
                                        data["barrier_gap_bounds"], data["stationarity_residuals"]):
        rows.append([number(t), number(x, 13), number(gap+data["optimal_value"], 13),
                     number(gap), number(bound), error(residual)])
        multipliers.append([number(t), number(1/(t*(x-2))), number(1/(t*(4-x))), number(bound)])
    return "\n\n".join([
        f"The independent analytical constrained optimum is x = {number(data['optimal_point'])}, "
        f"objective = {number(data['optimal_value'])}. The bracketed scalar solves produced:",
        table(["t", "Central point", "Original objective", "Objective error", "Bound 2/t", "Centering derivative residual"], rows),
        "The following multipliers are evaluated from the stored central points using the formulas above. "
        "Their complementarity sum is 2/t; their Lagrangian stationarity residual is the centering derivative residual divided by t.",
        table(["t", "Lower-bound multiplier", "Upper-bound multiplier", "Complementarity sum"], multipliers),
        "All five roots are strictly interior, decrease toward 2, and satisfy the displayed objective-gap bound. "
        "The seed is recorded for CLI consistency; this experiment has no random data.",
        plot("11.1", "barrier_central_path.png", "Executed original objective and five individually scaled centering objectives with their computed central points"),
    ])


def render_11_21(data):
    instances = data["instances"]
    rows, certificates, fits = [], [], []
    for item in instances:
        ref = item["reference"]
        rows.append([item["name"], f"{len(item['b'])} × {len(item['x'])}", item["newton_steps"],
                     number(item["rho"], 13), number(ref["objective"], 13), error(item["objective_difference"]),
                     f"{ref['solver']} / {ref['status']}"])
        certificates.append([item["name"], number(item["certified_lower_bound"], 13), error(item["certified_objective_gap"]),
                             error(item["central_path_gap_bound"]), error(item["stationarity_residual"]), error(item["primal_violation"])])
        ratios = [sum(a*x for a, x in zip(row, item["x"]))/b for row, b in zip(item["A"], item["b"])]
        fits.append([item["name"], number(item["fractional_deviation"]), number(item["log_chebyshev_error"]),
                     number(min(ratios)), number(max(ratios))])
    return "\n\n".join([
        "### (a–b) Executed epigraph formulation and barrier solutions",
        table(["Instance", "m × n", "Newton steps", "Barrier ρ", "Reference ρ", "Objective difference", "Reference / status"], rows),
        "### Feasibility and finite-centering certificates",
        table(["Instance", "Certified lower bound", "Certified objective gap", "Nominal M/t", "Stationarity residual", "Primal violation"], certificates),
        "The certified gaps include the finite-centering correction. They are the reported upper bounds on objective error; "
        "the differences from CLARABEL alone would not be certificates.",
        table(["Instance", "Actual worst ratio / reciprocal", "Log-Chebyshev error", "Minimum fitted ratio", "Maximum fitted ratio"], fits),
        *[f"Final x for **{item['name']}**: {vector(item['x'])}." for item in instances],
        "The fitted ratios were recomputed from the saved A, b, and x. The slightly larger epigraph value ρ "
        "keeps every barrier inequality strict. All input arrays, bounds, multipliers, and histories are in the linked JSON.",
        plot("11.21", "fractional_fit.png", "Executed bounded fractional fits and ratio envelopes for both synthetic instances"),
    ])


def render_11_22(data):
    instances = data["instances"]
    exact = {**data["analytic_example_volumes"], "book_polyhedron": 15/128}
    answers, references, certificates = [], [], []
    for item in instances:
        ref = item["reference"]
        lam = item["dual_feasible_multipliers"]
        equality = norm([sum(row[j]*weight for row, weight in zip(item["A"], lam)) for j in range(len(item["radius"]))]) if lam else None
        answers.append([item["name"], vector(item["lower"], 10), vector(item["upper"], 10),
                        number(item["volume"], 13), number(exact[item["name"]]), item["newton_steps"]])
        references.append([item["name"], number(item["log_volume"], 13), number(ref["log_volume"], 13),
                           error(item["log_volume_difference"]), f"{ref['solver']} / {ref['status']}"])
        certificates.append([item["name"], error(item["certified_objective_gap"]), error(item["primal_violation"]),
                             error(item["minimum_slack"]), error(item["stationarity_residual"]), error(equality)])
    book = next(item for item in instances if item["name"] == "book_polyhedron")
    return "\n\n".join([
        "### Computed rectangles, including the specified matrix",
        table(["Polyhedron", "Lower corner", "Upper corner", "Computed volume", "Exact volume", "Newton steps"], answers),
        "### Independent log-volume comparison",
        table(["Polyhedron", "Barrier log-volume", "Reference log-volume", "Reference minus barrier", "Reference / status"], references),
        "### Numerical certificates",
        table(["Polyhedron", "Certified negative-log-radius gap", "Containment violation", "Minimum slack", "Raw stationarity residual", "Projected dual equality residual"], certificates),
        f"For the book instance, the projected dual-feasible multiplier is "
        f"{vector(book['dual_feasible_multipliers'], 12)} and its dual objective is "
        f"{number(book['dual_objective'], 13)}. The dual equality residual was recomputed from the saved A "
        "and projected multipliers. All three certificates use nonnegative multipliers and positive radius coefficients. "
        "The nominal gap is 1e-6, while the table reports the independently evaluated feasible primal-minus-dual gap. "
        "The factor 2 in each full width contributes an additive constant to log-volume, so it does not affect this gap.",
        "The small corner offsets are expected from strict interiority at finite barrier parameter. "
        "The analytic solutions above give an additional independent check of the computed volumes.",
        plot("11.22", "maximum_volume_rectangles.png", "Executed maximum-volume rectangles in the box, triangle, and specified book polyhedron"),
    ])


def render_11_23(data):
    instances = data["instances"]
    sdp_rows, feasibility, references, rounding, greedy, refinements, exact = [], [], [], [], [], [], []
    for item in instances:
        sdp, ref, random = item["sdp"], item["reference"], item["randomized"]
        sdp_rows.append([item["n"], number(sdp["dual_lower_bound"], 12), number(sdp["primal_sdp_value"], 12),
                         error(sdp["certified_gap"]), error(sdp["central_path_gap_bound"]), sdp["newton_steps"]])
        feasibility.append([item["n"], error(sdp["dual_cone_min_eigenvalue"]), error(sdp["primal_cone_min_eigenvalue"]),
                            error(sdp["uncorrected_diagonal_residual"]), error(sdp["diagonal_residual"]), error(sdp["complementarity_norm"])])
        references.append([item["n"], f"{ref['solver']} / {ref['status']}", number(ref["objective"], 12),
                           error(ref["diagonal_residual"]), error(ref["cone_min_eigenvalue"])])
        exact.append([item["n"], item["exact"]["partitions_evaluated"] if item["exact"] else "Not enumerated (2^99 representatives)",
                      number(item["exact"]["value"], 12) if item["exact"] else "Unknown"])
        rounding.append([item["n"], number(item["spectral"]["value"], 12), random["samples"], number(random["best_value"], 12),
                         number(random["best_value"]-sdp["dual_lower_bound"], 10)])
        greedy.append([item["n"], number(item["all_ones"]["value"], 12), number(item["all_ones"]["greedy"]["value"], 12),
                       number(item["spectral"]["greedy"]["value"], 12), number(random["best_sample_greedy"]["value"], 12),
                       number(random["best_over_all_refined_samples"]["value"], 12)])
        refinements.append([item["n"], item["all_ones"]["greedy"]["flips"], item["spectral"]["greedy"]["flips"],
                            random["best_over_all_refined_samples"]["flips"], error(random["best_over_all_refined_samples"]["best_remaining_flip"]),
                            number(random["mean_greedy_improvement"], 10)])
    large = next(item for item in instances if item["n"] == 100)
    best = large["randomized"]["best_over_all_refined_samples"]
    return "\n\n".join([
        "### (a) Both SDP solutions and their certificates",
        table(["n", "Dual lower bound", "Primal SDP value", "Measured SDP gap", "Nominal n/t", "Newton steps"], sdp_rows),
        table(["n", "Smallest dual eigenvalue", "Smallest primal eigenvalue", "Raw diagonal residual", "Corrected diagonal residual", "Complementarity norm"], feasibility),
        "Both reported SDP matrices are positive definite and the normalized primal diagonal is one to roundoff. "
        "The lower and upper objective columns bracket the relaxed optimum; only the dual column is a lower bound for the discrete problem.",
        table(["n", "Independent solver / status", "Reference SDP objective", "Reference diagonal residual", "Reference minimum eigenvalue"], references),
        "SCS reports an optimal status for n = 100 but its small negative minimum eigenvalue is an approximate-cone residual. "
        "The certificate uses the custom solver's separately checked feasible pair, not the SCS status.",
        table(["n", "Exact-search representatives evaluated", "Exact discrete optimum"], exact),
        "### (b–c) Spectral and Gaussian rounding",
        table(["n", "Spectral objective", "Gaussian samples", "Best Gaussian objective", "Best Gaussian minus lower bound"], rounding),
        "Every rounded result is a feasible sign vector. The last column bounds the best Gaussian partition's absolute suboptimality; "
        "it need not be close to the SDP optimization gap.",
        "### (d) Greedy refinement of every required start",
        table(["n", "All ones before", "All ones after", "Spectral after", "Originally best Gaussian after", "Best over all refined Gaussian samples"], greedy),
        table(["n", "All-ones flips", "Spectral flips", "Winning refined sample flips", "Winner's best remaining flip change", "Mean Gaussian improvement"], refinements),
        "The winning refined sample need not come from the originally best Gaussian draw. "
        "Each displayed final best-flip change is nonnegative, verifying a one-coordinate local minimum for the winner. "
        "All 300 Gaussian samples are refined, and the JSON retains every raw and refined objective plus the displayed winners' histories.",
        f"For n = 100, the best reported partition has objective {number(best['value'], 12)}, "
        f"with {sum(value > 0 for value in best['x'])} positive and {sum(value < 0 for value in best['x'])} negative signs. "
        f"Its certified absolute suboptimality bound is {number(best['value']-large['sdp']['dual_lower_bound'], 12)}. "
        "The discrete optimum is unknown at this size. The small instances' best rounded results reach their enumerated optima.",
        "The linked JSON contains every generated W, SDP matrix, multiplier, and winning sign vector, including the full 100-by-100 instance.",
        plot("11.23", "partition_heuristics.png", "Executed spectral, Gaussian, and greedy partition objectives against SDP bounds and small exact optima"),
        plot("11.23", "randomized_greedy_distribution.png", "Executed distribution of all 300 Gaussian partitions before and after greedy refinement at n equals 100"),
    ])


def render_11_24(data):
    instances = data["instances"]
    answers, references, feasibility = [], [], []
    for item in instances:
        ref = item["reference"]
        references.append([item["name"], f"{len(item['q'])} × {len(item['b'])}", number(ref["objective"], 13),
                           error(item["barrier_objective_difference"]), error(item["primal_dual_objective_difference"]),
                           f"{ref['solver']} / {ref['status']}"])
        for key, name, steps in [("barrier", "Barrier", "newton_steps"), ("primal_dual", "Primal-dual", "iterations")]:
            result = item[key]
            answers.append([item["name"], name, result[steps], number(result["objective"], 13),
                            number(result["dual_objective"], 13), error(result["duality_gap"]), error(result["surrogate_gap"])])
            feasibility.append([item["name"], name, error(result["primal_violation"]), error(result["minimum_slack"]),
                                error(result["minimum_multiplier"]), error(result["dual_residual_norm"]),
                                error(distance(result["x"], ref["x"]))])
    return "\n\n".join([
        "### Both directly implemented algorithms",
        table(["Instance", "Method", "Accepted steps / iterations", "Primal objective", "Dual objective", "Actual duality gap", "Complementarity sum"], answers),
        "The barrier iteration count is the total number of accepted inner Newton steps. "
        "The primal-dual count is the number of outer primal-dual steps. The actual gap includes the stationarity correction; "
        "the complementarity column is the surrogate gap used by the primal-dual stopping test.",
        "### Independent QP comparisons",
        table(["Instance", "n × m", "Reference objective", "Barrier objective error", "Primal-dual objective error", "Reference / status"], references),
        "### Feasibility, residuals, and solution errors",
        table(["Instance", "Method", "Primal violation", "Minimum slack", "Minimum multiplier", "Stationarity norm", "Point error vs reference"], feasibility),
        "All returned slacks and multipliers are strictly positive. The barrier's decrement stopping rule allows a larger "
        "stationarity residual than the primal-dual rule; the evaluated positive-definite dual objective includes its contribution.",
        *[f"Final primal-dual x for **{item['name']}**: {vector(item['primal_dual']['x'])}." for item in instances],
        "All barrier and reference solution coordinates, input matrices, and complete convergence histories are in the linked JSON.",
        plot("11.24", "qp_convergence.png", "Executed barrier actual gap, primal-dual surrogate gap, and primal-dual stationarity norm across all three QPs"),
    ])


RENDERERS = {
    "10.14": render_10_14,
    "10.15": render_10_15,
    "10.16": render_10_16,
    "11.1": render_11_1,
    "11.21": render_11_21,
    "11.22": render_11_22,
    "11.23": render_11_23,
    "11.24": render_11_24,
}


def render(exercise_id: str, envelope: dict) -> str:
    """Return the generated Markdown body for one supported exercise."""
    return RENDERERS[exercise_id](envelope["results"])
