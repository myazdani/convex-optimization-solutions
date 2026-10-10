"""Render checked-in Markdown result tables directly from executed JSON outputs.

Run the experiments first. This command never runs solvers or invents fallback
values: missing, mismatched, or nonfinite outputs are errors. The --check mode
compares every generated section without changing a file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BEGIN = "<!-- BEGIN GENERATED RESULTS -->"
END = "<!-- END GENERATED RESULTS -->"


def reject_constant(value: str):
    raise ValueError(f"Nonfinite JSON value: {value}")


def load_envelope(exercise_id: str, artifacts: Path) -> dict:
    path = artifacts / exercise_id / "results.json"
    envelope = json.loads(path.read_text(), parse_constant=reject_constant)
    if envelope["exercise_id"] != exercise_id:
        raise ValueError(f"{path}: exercise ID mismatch")
    for key in ("seed", "executed_at", "environment", "source_tree_sha256", "results"):
        if key not in envelope:
            raise ValueError(f"{path}: missing execution provenance {key}")
    return envelope


def render_block(exercise_id: str, envelope: dict) -> str:
    from result_sections_early import render as render_early
    from result_sections_later import render as render_later

    render = render_early if int(exercise_id.split(".")[0]) < 10 else render_later
    body = render(exercise_id, envelope).strip()
    if "|" not in body:
        raise ValueError(f"{exercise_id}: result section must include measured tables")
    executed = envelope["executed_at"].replace("+00:00", "Z")
    python = envelope["environment"]["python"].split()[0]
    header = (
        f"## Executed results\n\n"
        f"The full-size experiment was executed at **{executed}**, with "
        f"seed **{envelope['seed']}** and Python **{python}**. "
        "The tables below are generated directly from this run's "
        f"[recorded output](../../artifacts/{exercise_id}/results.json), "
        "rather than illustrative or expected values. "
        "See the [execution record](../../coverage/results-refresh.json) for "
        "the exact command, package versions, source fingerprint, and checks.\n\n"
    )
    return f"{BEGIN}\n{header}{body}\n{END}"


def replace_block(document: str, block: str) -> str:
    if document.count(BEGIN) != 1 or document.count(END) != 1:
        raise ValueError("Expected exactly one generated-results marker pair")
    start = document.index(BEGIN)
    finish = document.index(END, start) + len(END)
    return document[:start] + block + document[finish:]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail on stale Markdown")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--docs", type=Path, default=ROOT / "docs" / "solutions")
    args = parser.parse_args(argv)
    from convex_solutions.registry import audit_inventory, load_inventory

    stale = []
    ids = audit_inventory(load_inventory())["implemented_ids"]
    for eid in ids:
        path = args.docs / f"{eid}.md"
        document = path.read_text()
        generated = render_block(eid, load_envelope(eid, args.artifacts))
        updated = replace_block(document, generated)
        if updated != document:
            stale.append(eid)
            if not args.check:
                path.write_text(updated)
    if args.check and stale:
        print("Stale generated results: " + ", ".join(stale), file=sys.stderr)
        return 1
    verb = "Verified" if args.check else "Rendered"
    print(f"{verb} results for {len(ids)} exercises ({len(stale)} changed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
