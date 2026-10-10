"""Render measured worked-example outputs without modifying exercise result pages."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
IDS = ("7.4", "8.3")
BEGIN = "<!-- BEGIN GENERATED EXAMPLE RESULTS -->"
END = "<!-- END GENERATED EXAMPLE RESULTS -->"


def reject_constant(value: str):
    raise ValueError(f"Nonfinite JSON value: {value}")


def load_envelope(example_id: str, artifacts: Path = ROOT / "artifacts" / "examples") -> dict:
    path = artifacts / example_id / "results.json"
    envelope = json.loads(path.read_text(), parse_constant=reject_constant)
    if envelope["example_id"] != example_id:
        raise ValueError(f"{path}: example ID mismatch")
    code = ROOT / "examples" / f"ex_{example_id.replace('.', '_')}.py"
    if envelope["implementation_sha256"] != hashlib.sha256(code.read_bytes()).hexdigest():
        raise ValueError(f"{path}: implementation changed; rerun the example first")
    for key in ["seed", "executed_at", "environment", "duration_seconds", "command", "results"]:
        if key not in envelope:
            raise ValueError(f"{path}: missing execution provenance {key}")
    return envelope


def table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
                      *["| " + " | ".join(row) + " |" for row in rows]])


def matrix_table(matrix: list[list[float]]) -> str:
    return table(["Row", "Column 1", "Column 2", "Column 3", "Column 4"],
                 [[str(index), *[f"{value:.9f}" for value in row]] for index, row in enumerate(matrix, 1)])


def render_block(example_id: str, envelope: dict) -> str:
    r = envelope["results"]
    executed = envelope["executed_at"].replace("+00:00", "Z")
    py = envelope["environment"]["python"].split()[0]
    body = [BEGIN, "## Executed results", "",
            f"Executed **{executed}**, Python **{py}**, seed **{envelope['seed']}** "
            f"(unused), elapsed **{envelope['duration_seconds']:.3f} s**. "
            f"These tables are generated from the [actual recorded run](../../artifacts/examples/{example_id}/results.json). "
            "They are not substituted expected values.", ""]
    if example_id == "7.4":
        m = r["minimax"]
        rows = [["CLARABEL minimax", f"{m['error_pair'][0]:.12f}", f"{m['error_pair'][1]:.12f}", f"{m['worst_error']:.12f}"],
                ["SciPy/HiGHS minimax", *[f"{x:.12f}" for x in r["independent_reference"]["error_pair"]],
                 f"{r['independent_reference']['objective']:.12f}"],
                ["Best deterministic", *[f"{x:.12f}" for x in r["deterministic"]["error_pairs"][r["deterministic"]["minimax_rule_indices"][0]]],
                 f"{r['deterministic']['minimax_error']:.12f}"]]
        body += [table(["Method", "False positive", "False negative", "Worst error"], rows), "",
                 "Measured positive-declaration probabilities:", "",
                 table(["Observation", "Probability"], [[str(i), f"{x:.12f}"] for i, x in enumerate(m["positive_probabilities"], 1)]), "",
                 table(["Check", "Measured value"], [
                     ["Primal constraint violation", f"{m['primal_violation']:.3e}"],
                     ["Primal–dual gap", f"{m['primal_dual_gap']:.3e}"],
                     ["Stationarity residual", f"{m['stationarity_residual']:.3e}"],
                     ["Complementarity residual", f"{m['complementarity_residual']:.3e}"],
                     ["Absolute error versus exact 1/6", f"{r['checks']['minimax_exact_error']:.3e}"],
                     ["Independent-solver difference", f"{r['checks']['independent_solver_error']:.3e}"],
                     ["Frontier LPs solved", str(r["frontier"]["point_count"])],
                     ["Maximum frontier interpolation error", f"{r['frontier']['maximum_interpolation_error']:.3e}"],
                     ["Maximum frontier primal violation", f"{r['frontier']['maximum_primal_violation']:.3e}"]]), ""]
    elif example_id == "8.3":
        body += [table(["Endpoint", "CLARABEL correlation", "SCS correlation", "Closed-form bound"],
                       [[label.capitalize(), f"{r[label]['rho14']:.12f}",
                         f"{r['independent_reference'][label]['rho14']:.12f}",
                         f"{r['analytic_certificate'][label]:.12f}"] for label in ["lower", "upper"]]), ""]
        rows = []
        for label in ["lower", "upper"]:
            z = r[label]
            rows.append([label.capitalize(), f"{z['primal_violation']:.3e}",
                         f"{z['stationarity_residual']:.3e}", f"{z['primal_dual_gap']:.3e}",
                         f"{z['eigenvalues'][0]:.3e}"])
        body += [table(["CLARABEL endpoint", "Primal violation", "Stationarity", "Primal–dual gap", "Minimum eigenvalue"], rows), "",
                 "Lower-endpoint matrix returned by CLARABEL:", "", matrix_table(r["lower"]["matrix"]), "",
                 "Upper-endpoint matrix returned by CLARABEL:", "", matrix_table(r["upper"]["matrix"]), "",
                 "Original Gram-certificate checks (independent of the numerical solvers):", "",
                 table(["Certificate", "Achieved correlation", "PSD violation", "Interval violation", "Diagonal residual"],
                       [[label.capitalize(), f"{z['rho14']:.12f}", f"{z['psd_violation']:.3e}",
                         f"{z['bound_violation']:.3e}", f"{z['diagonal_residual']:.3e}"]
                        for label, z in r["analytic_certificate"]["witnesses"].items()]), ""]
    else:
        raise ValueError(f"Unsupported example: {example_id}")
    body.append(END)
    return "\n".join(body)


def replace_block(document: str, block: str) -> str:
    if document.count(BEGIN) != 1 or document.count(END) != 1:
        raise ValueError("Expected exactly one generated-example-results marker pair")
    start, end = document.index(BEGIN), document.index(END) + len(END)
    return document[:start] + block + document[end:]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    stale = []
    for example_id in IDS:
        path = ROOT / "docs" / "examples" / f"{example_id}.md"
        current = path.read_text()
        changed = replace_block(current, render_block(example_id, load_envelope(example_id)))
        if current != changed:
            stale.append(example_id)
            if not args.check:
                path.write_text(changed)
    if args.check and stale:
        print("Stale example results: " + ", ".join(stale), file=sys.stderr)
        return 1
    print(f"{'Verified' if args.check else 'Rendered'} results for {len(IDS)} worked examples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
