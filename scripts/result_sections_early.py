"""Render early-chapter executed results directly from CLI JSON envelopes.

No solving, file reads, timestamps, or synthetic measurements belong here.
The shared renderer owns provenance and insertion into the solution notes.
"""
from __future__ import annotations


def number(value: float, precision: int = 12) -> str:
    if value == 0:
        return "0"
    return f"{value:.{precision}g}"


def residual(value: float) -> str:
    return "0" if value == 0 else f"{value:.3e}"


def vector(values: list[float], precision: int = 9) -> str:
    return "(" + ", ".join(number(v, precision) for v in values) + ")"


def table(headers: list[str], rows: list[list[object]]) -> str:
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows),
    ])


def plot(eid: str, filename: str, caption: str) -> str:
    return f"![{caption}](../../artifacts/{eid}/{filename})"


def render_4_3(result: dict) -> str:
    cert, ref = result["certificate"], result["reference"]
    comparison = table(
        ["Quantity", "Exact candidate evaluated by code", f"Independent {ref['solver']} solve"],
        [
            ["Optimizer", vector(cert["x"], 12), vector(ref["x"], 12)],
            ["Objective", number(cert["objective"], 16), number(ref["objective"], 16)],
            ["Status", "Unique optimum certified", ref["status"]],
            ["Box-constraint violation", residual(cert["primal_violation"]), residual(ref["primal_violation"])],
            ["Stationarity residual", residual(cert["stationarity_residual"]), residual(ref["stationarity_residual"])],
            ["Complementarity residual", residual(cert["complementarity_residual"]), "Not separately recorded"],
        ],
    )
    return f"""{comparison}

The computed eigenvalues of $`P`$ are **{vector(cert['eigenvalues'], 12)}**;
all are positive. The evaluated gradient is {vector(cert['gradient'])}, with
upper multipliers {vector(cert['upper_multipliers'])} and lower multipliers
{vector(cert['lower_multipliers'])}. The exact candidate's recorded primal,
stationarity, and complementarity residuals are all zero.

The independent solver returned objective error **{residual(ref['objective_error'])}**
and Euclidean point error **{residual(ref['point_error'])}** relative to the
certificate. These are measured errors from this execution, not solver tolerances.
The full-precision optimizer and solver-attempt record are in the linked JSON.
"""


def render_5_29(result: dict) -> str:
    points, ref = result["points"], result["reference"]
    labels = ["Global maximum", "Saddle", "Nonglobal strict local minimum", "Unique global minimum"]
    pairs = table(
        ["Multiplier ν", "Stationary point x", "Objective", "Classification", "Tangent-Hessian eigenvalues"],
        [[number(p["nu"]), vector(p["x"]), number(p["objective"]), label,
          vector(p["tangent_hessian_eigenvalues"], 8)] for p, label in zip(points, labels)],
    )
    intervals = table(
        ["Pole-separated interval", "Computed minimum of F", "Root count"],
        [[f"({number(i['interval'][0]) if i['interval'][0] is not None else '−∞'}, "
          f"{number(i['interval'][1]) if i['interval'][1] is not None else '+∞'})",
          number(i["minimum_value"]) if "minimum_value" in i else "Monotone exterior interval",
          i["root_count"]] for i in result["interval_analysis"]],
    )
    checks = table(
        ["Independent lifted SDP check", "Actual result"],
        [["Solver / status", f"{ref['solver']} / {ref['status']}"],
         ["Optimizer x", vector(ref["x"], 12)],
         ["Objective", number(ref["objective"], 15)],
         ["Objective error against certified minimum", residual(ref["objective_error"])],
         ["Euclidean point error", residual(ref["point_error"])],
         ["Trace residual", residual(ref["trace_residual"])],
         ["PSD violation", residual(ref["psd_violation"])],
         ["Rank-one residual", residual(ref["rank_one_residual"])]],
    )
    return f"""### All stationary pairs and their classification

{pairs}

The maximum sphere-feasibility residual is
**{residual(max(p['sphere_residual'] for p in points))}**; the maximum stationarity
residual is **{residual(max(p['stationarity_residual'] for p in points))}**.

{intervals}

The independently computed polynomial roots disagree with the bracketed secular
roots by at most **{residual(result['polynomial_root_error'])}**. At the global
minimum the shifted Lagrangian-Hessian eigenvalues are
{vector(result['global_minimum']['shifted_hessian_eigenvalues'], 10)}, all positive;
the primal objective and dual value are both
**{number(result['global_minimum']['dual_value'], 15)}**, with recorded certified
duality gap **{residual(result['certified_duality_gap'])}**. The other three
multipliers give an unbounded-below Lagrangian.

### Independent semidefinite cross-check

{checks}

The SDP accuracy is lower than that of the exact secular certificate, especially
in the optimizer coordinates; the objective and PSD/rank-one residuals above
make that distinction explicit. Full matrices and all KKT residuals are retained
in the JSON.

{plot('5.29', '5_29_secular.png', 'Secular function, poles, and every stationary multiplier')}
"""


def render_5_33(result: dict) -> str:
    checks = result["checks"]
    values = table(
        ["ε", "Analytic optimizer", "Affine value", "Primal LP value", "Dual LP value", "Primal–dual gap"],
        [[number(c["analytic"]["epsilon"]), vector(c["analytic"]["x"]),
          number(c["analytic"]["formula_value"]), number(c["reference"]["objective"]),
          number(c["reference"]["dual_objective"]), residual(c["reference"]["primal_dual_gap"])]
         for c in checks],
    )
    diagnostics = table(
        ["Diagnostic over all displayed perturbations", "Maximum measured value"],
        [["Analytic-versus-LP objective error", residual(max(c["objective_error"] for c in checks))],
         ["Euclidean optimizer error", residual(max(c["point_error"] for c in checks))],
         ["Primal LP feasibility violation", residual(max(c["reference"]["primal_violation"] for c in checks))],
         ["Dual equality residual", residual(max(c["reference"]["dual_equality_residual"] for c in checks))]],
    )
    statuses = sorted({c["reference"][key] for c in checks for key in ("primal_status", "dual_status")})
    return f"""### Sensitivity across the entire certified interval

{values}

Every primal and dual SciPy/HiGHS run returned:
**{' / '.join(statuses)}**. At zero the executed residual is
{vector(next(c['analytic']['residual'] for c in checks if c['analytic']['epsilon'] == 0))},
with objective 4. The computed active determinant is **{result['active_determinant']}**
and the strict dual margin on the zero-residual rows is
**{number(result['strict_dual_margin_on_zero_rows'])}**, supporting the uniqueness
argument, including at both endpoints.

{diagnostics}

Printed objectives are rounded; the small nonzero gaps are computed from the
full-precision values. The LP optimizers, independent dual vectors, and all
residuals for each perturbation are in the JSON.

{plot('5.33', '5_33_sensitivity.png', 'Executed primal and dual LP values versus the exact affine sensitivity law')}
"""


def render_5_34(result: dict) -> str:
    checks = result["checks"]
    optima = table(
        ["ε", "Displayed analytic optimizer x", "Displayed analytic dual z", "Exact value formula"],
        [[number(c["analytic"]["epsilon"]), vector(c["analytic"]["x"]),
          vector(c["analytic"]["z"]), number(c["analytic"]["formula_value"])] for c in checks],
    )
    comparisons = table(
        ["ε", "Primal LP value", "Dual LP value", "Primal–dual gap", "Stationarity residual"],
        [[number(c["analytic"]["epsilon"]), number(c["reference"]["objective"]),
          number(c["reference"]["dual_objective"]), residual(c["reference"]["primal_dual_gap"]),
          residual(c["reference"]["stationarity_residual"])] for c in checks],
    )
    endpoint = checks[-1]
    free = result["free_variable_check"]; ref = free["reference"]
    statuses = sorted({c["reference"][key] for c in checks for key in ("primal_status", "dual_status")})
    return f"""### Primal and dual sensitivity checks

{optima}

{comparisons}

Every listed primal and dual SciPy/HiGHS run returned:
**{' / '.join(statuses)}**. Across those runs, maximum primal violation is
**{residual(max(c['reference']['primal_violation'] for c in checks))}**,
maximum dual nonnegativity violation is
**{residual(max(c['reference']['dual_violation'] for c in checks))}**, and maximum
complementarity residual is
**{residual(max(c['reference']['complementarity_residual'] for c in checks))}**.
The maximum error against the quadratic value formula is
**{residual(max(c['objective_error'] for c in checks))}**.

At ε = −0.5 the primal is nonunique; at ε = 1 the dual is nonunique.
At the latter endpoint HiGHS actually returned
$`z={vector(endpoint['reference']['z'], 10)}`$, a distance
**{number(endpoint['dual_point_error'])}** from the displayed analytic dual,
while attaining the same optimal value. This coordinate difference is expected
from nonuniqueness, not a failed optimality check.

### Free-variable regression outside the basis interval

At ε = **{number(free['epsilon'])}**, the independent LP returned
$`x={vector(ref['x'], 12)}`$ and objective
**{number(ref['objective'], 15)}**. The fourth coordinate is negative, as allowed
by the original problem. Both LP statuses are optimal; the measured primal–dual
gap is **{residual(ref['primal_dual_gap'])}**, primal violation
**{residual(ref['primal_violation'])}**, and stationarity residual
**{residual(ref['stationarity_residual'])}**. This point is outside the proven
sensitivity interval and is not evaluated using its local formula.

{plot('5.34', '5_34_sensitivity.png', 'Executed primal and dual LP values versus the exact quadratic sensitivity law')}
"""


def render_9_10(result: dict) -> str:
    left, right = result["part_a"]
    sequence = table(
        ["k", "x from start 1", "f(x) from start 1", "x from start 1.1", "f(x) from start 1.1"],
        [[k, number(left["iterates"][k]), number(left["objectives"][k]),
          number(right["iterates"][k]), number(right["objectives"][k])]
         for k in range(len(left["iterates"]))],
    )
    barrier = result["part_b"]
    return f"""### (a) Full Newton on log-sum-exp

{sequence}

The first run returned **{left['status']}** after
**{len(left['iterates']) - 1}** accepted updates, at
$`x={number(left['iterates'][-1], 15)}`$ with objective
**{number(left['objectives'][-1], 16)}** and absolute gradient
**{residual(abs(left['gradients'][-1]))}**.
The second returned **{right['status']}** at
$`x={number(right['iterates'][-1], 15)}`$; its absolute gradient is
**{number(abs(right['gradients'][-1]))}**. The code stops before the next
hyperbolic-sine evaluation would overflow. The numerically computed two-cycle
threshold is **{number(result['nonzero_two_cycle_threshold'], 15)}**.

### (b) Full Newton leaves the barrier domain

Starting at **{number(barrier['iterates'][0])}**, the recorded first candidate is
**{number(barrier['invalid_candidate'])}** and status is **{barrier['status']}**.
Only the initial point is retained as a valid iterate; no objective is evaluated
at the invalid candidate. This is the intended failure demonstration.

{plot('9.10', 'pure_newton_examples.png', 'Functions, derivatives, Newton iterates, and invalid barrier update')}
"""


def render_9_30(result: dict) -> str:
    instances = result["instances"]
    refs = table(
        ["(m,n); instance seed", "Reference objective", "Reference gradient norm", "Objective-error bound", "Hessian condition"],
        [[f"({i['m']},{i['n']}); {i['seed']}", number(i["p_reference"], 15),
          residual(i["reference_gradient_norm"]), residual(i["reference_objective_error_upper_bound"]),
          number(i["hessian_condition_number_at_solution"], 8)] for i in instances],
    )
    baselines = table(
        ["(m,n)", "Method", "Updates", "Final objective", "Final gradient norm", "Final λ²/2", "Minimum domain margin", "Status / stopping test"],
        [[f"({i['m']},{i['n']})", method, run["iterations"], number(run["final_objective"], 15),
          residual(run["final_gradient_norm"]),
          residual(run["final_newton_decrement_squared"] / 2) if run["final_newton_decrement_squared"] is not None else "—",
          number(run["minimum_margin"], 7),
          ("converged" if run["converged"] else "not converged") + " / " + run["termination"]]
         for i in instances for method, run in i["baseline"].items()],
    )
    sweep_rows = []
    for alpha in result["alphas"]:
        for beta in result["betas"]:
            row = [number(alpha), number(beta)]
            for i in instances:
                counts = {r["method"]: r["iterations"] for r in result["runs"]
                          if (r["m"], r["n"], r["alpha"], r["beta"]) == (i["m"], i["n"], alpha, beta)}
                row.append(f"{counts['gradient']} / {counts['newton']}")
            sweep_rows.append(row)
    sweep = table(["α", "β", *[f"({i['m']},{i['n']}) gradient / Newton" for i in instances]], sweep_rows)
    local = table(
        ["(m,n)", "Last two measurable quadratic ratios", "Last two measured orders"],
        [[f"({i['m']},{i['n']})",
          ", ".join(number(s["quadratic_ratio"], 7) for s in i["baseline"]["newton"]["convergence"]["local_samples"][-2:]),
          ", ".join(number(s["observed_order"], 7) for s in i["baseline"]["newton"]["convergence"]["local_samples"][-2:] if s["observed_order"] is not None)]
         for i in instances],
    )
    gradients = [r for r in result["runs"] if r["method"] == "gradient"]
    newtons = [r for r in result["runs"] if r["method"] == "newton"]
    links = ", ".join(f"[({i['m']},{i['n']}) arrays](../../artifacts/9.30/instance_m{i['m']}_n{i['n']}.npz)" for i in instances)
    return f"""### Reference solutions and baseline methods

{refs}

The following baseline uses α = 0.1 and β = 0.5. A domain margin is the minimum
of all linear and box slacks over the recorded trajectory, so a positive value
also checks feasibility of the intermediate iterates.

{baselines}

### Full backtracking-parameter sweep

Every cell is **gradient updates / Newton updates**, from a separate start at zero.

{sweep}

**{sum(r['converged'] for r in result['runs'])}/{len(result['runs'])} runs converged**.
The largest final gradient norm across gradient runs is
**{residual(max(r['final_gradient_norm'] for r in gradients))}**; the largest
final half-decrement across Newton runs is
**{residual(max(r['final_newton_decrement_squared'] / 2 for r in newtons))}**.
The smallest domain margin over the complete sweep is
**{number(min(r['minimum_margin'] for r in result['runs']), 9)}**, strictly positive.

### Local Newton convergence

{local}

These full-step measurements use the roundoff exclusions described above.
The larger two instances show successive observed orders approaching two.

{plot('9.30', 'objective_and_steps.png', 'Gradient and Newton objective values, gaps, and step lengths')}

{plot('9.30', 'parameter_sweep.png', 'Accepted-update counts for every backtracking-parameter pair')}

{plot('9.30', 'newton_quadratic_convergence.png', 'Newton iterate error and error-to-error quadratic convergence')}

{plot('9.30', 'newton_objective_and_steps.png', 'Newton objective gaps and accepted steps on an iteration scale')}

Full scalar traces and baseline optimizer coordinates are in
[histories.json](../../artifacts/9.30/histories.json); all parameter counts and
residuals are also in [parameter_sweep.csv](../../artifacts/9.30/parameter_sweep.csv).
Generated matrices and reference optimizers: {links}.
"""


def render_9_31(result: dict) -> str:
    instances = result["instances"]
    labels = [r["label"] for r in instances[0]["runs"]]
    by_label = [{r["label"]: r for r in i["runs"]} for i in instances]
    counts = table(
        ["Method", *[f"({i['m']},{i['n']}) updates / factorizations" for i in instances]],
        [[label, *[f"{runs[label]['iterations']} / {runs[label]['factorizations']}" for runs in by_label]]
         for label in labels],
    )
    gradients = table(
        ["Method", *[f"({i['m']},{i['n']}) final gradient norm" for i in instances]],
        [[label, *[residual(runs[label]["gradient_norms"][-1]) for runs in by_label]] for label in labels],
    )
    diagnostics = table(
        ["(m,n); instance seed", "Reference objective", "Largest final absolute gap", "Largest gradient-based error bound", "Smallest trajectory margin"],
        [[f"({i['m']},{i['n']}); {i['seed']}", number(i["p_reference"], 15),
          residual(max(abs(r["objective_gaps"][-1]) for r in i["runs"])),
          residual(max(r["gradient_norms"][-1] ** 2 / 4 for r in i["runs"])),
          number(min(min(r["margins"]) for r in i["runs"]), 9)] for i in instances],
    )
    statuses = sorted({r["status"] for i in instances for r in i["runs"]})
    return f"""### (a) Hessian reuse and (b) diagonal approximation

{counts}

All **{sum(len(i['runs']) for i in instances)}** recorded statuses are
**{' / '.join(statuses)}**. The shared stopping tolerance is checked by the
actual final norms below, rather than inferred from update counts.

{gradients}

{diagnostics}

The gradient-based bound is computed from each recorded final norm as
$`\\|\\nabla f\\|_2^2/4`$. Positive trajectory margins confirm strict feasibility.
Full final optimizers (including all 100 coordinates of the largest instance),
objective histories, step lengths, backtracks, and actual Hessian/diagonal
work counts are preserved in the JSON.

Hessian reuse reduces factorization counts while increasing accepted updates.
The diagonal approximation is faster in update count than gradient descent
on the smallest and largest instances, but slower on the middle one; no
universal wall-clock speed ranking follows from these counts.

{plot('9.31', 'approximate_newton_comparison.png', 'All methods: convergence histories and factorization work')}
"""


def render_9_32(result: dict) -> str:
    instances = result["instances"]
    refs = table(
        ["(m,n); instance seed", "Reference objective", "Reference gradient norm", "Strong-convexity lower bound μ", "Minimum residual lower bound"],
        [[f"({i['m']},{i['n']}); {i['seed']}", number(i["p_reference"], 15),
          residual(i["reference_gradient_norm"]), number(i["strong_convexity_lower_bound"], 10),
          number(i["minimum_residual_lower_bound"], 10)] for i in instances],
    )
    runs = table(
        ["(m,n)", "Method", "Status", "Updates", "Final objective", "Final gradient norm", "Objective-error bound"],
        [[f"({i['m']},{i['n']})", r["method"].replace("_", "-"), r["status"], r["iterations"],
          number(r["objectives"][-1], 15), residual(r["gradient_norms"][-1]),
          residual(r["objective_error_upper_bound"])] for i in instances for r in i["runs"]],
    )
    gn = table(
        ["(m,n)", "Smallest recorded singular value of J", "Accepted Gauss–Newton step range", "Total backtracks"],
        [[f"({i['m']},{i['n']})", number(min(r["smallest_jacobian_singular_values"]), 9),
          f"{number(min(r['step_lengths']))}–{number(max(r['step_lengths']))}", r["total_backtracks"]]
         for i in instances for r in i["runs"] if r["method"] == "gauss_newton"],
    )
    links = ", ".join(f"[({i['m']},{i['n']}) arrays](../../artifacts/9.32/instance_m{i['m']}_n{i['n']}.npz)" for i in instances)
    return f"""### Generated-instance checks and numerical references

{refs}

All minimum residual bounds and strong-convexity bounds are positive, verifying
the generated instances' required convexity assumptions numerically.

### Gradient, exact Newton, and Gauss–Newton

{runs}

The objective-error bounds use the recorded gradient norms and μ, so they do
not depend on subtracting nearly equal objective values. Every recorded method
meets its gradient tolerance.

{gn}

The positive minimum singular values verify full column rank throughout the
Gauss–Newton trajectories. Short accepted steps and repeated backtracking show
why these runs need more updates than exact Newton despite converging.
Full optimizers, singular-value traces, and step histories are in the JSON;
generated arrays and reference points are available as {links}.

{plot('9.32', 'gauss_newton_comparison.png', 'Gradient, exact Newton, and Gauss–Newton convergence and step lengths')}
"""


RENDERERS = {
    "4.3": render_4_3,
    "5.29": render_5_29,
    "5.33": render_5_33,
    "5.34": render_5_34,
    "9.10": render_9_10,
    "9.30": render_9_30,
    "9.31": render_9_31,
    "9.32": render_9_32,
}


def render(exercise_id: str, envelope: dict) -> str:
    """Return only the generated section body; shared code adds provenance."""
    return RENDERERS[exercise_id](envelope["results"]).strip() + "\n"
