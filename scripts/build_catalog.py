"""Build the three-category catalog and indexes without creating solution stubs.

The historical computational inventory remains an input for implemented exercise
metadata and provenance. It does not define the scope of the full book catalog.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATEGORY_PATHS = {
    "examples": "docs/examples/README.md",
    "computational_exercises": "docs/computational-exercises/README.md",
    "other_exercises": "docs/other-exercises/README.md",
}
CHAPTER_TITLES = {
    2: "Convex sets", 3: "Convex functions", 4: "Convex optimization problems",
    5: "Duality", 6: "Approximation and fitting", 7: "Statistical estimation",
    8: "Geometric problems", 9: "Unconstrained minimization",
    10: "Equality constrained minimization", 11: "Interior-point methods",
}


def read_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


def build_catalog() -> dict:
    audit = read_json("coverage/source_audit.json")
    inventory = read_json("coverage/inventory.json")
    worked = read_json("coverage/worked_examples.json")
    reproduced = {e["id"]: e for e in read_json("coverage/example_reproductions.json")["entries"]} if (ROOT / "coverage/example_reproductions.json").exists() else {}
    solved = {e["id"]: e for e in read_json("coverage/exercise_solutions.json")["entries"]}
    implemented = {e["id"]: e for e in inventory["exercises"]
                   if e.get("selected_for_implementation")}
    metadata = {e["id"]: e for e in inventory["exercises"]}
    exercises = []
    for source in audit["exercises"]:
        eid = source["id"]
        delivered = eid in implemented
        detail = metadata.get(eid, {})
        record = {
            "key": f"exercise:{eid}", "id": eid, "kind": "exercise",
            "chapter": int(eid.split(".")[0]),
            "title": detail.get("title", f"Exercise {eid}"),
            "category": "computational_exercises" if delivered else "other_exercises",
            "status": "verified" if delivered else "pending",
            "source_page": source["source_page"],
            "source_pdf_page": int(source["source_url"].split("#page=")[1]),
            "source_url": source["source_url"],
        }
        if delivered:
            record.update({
                "textbook_role": "explicit_computation" if detail["scope"] == "required" else "numerical_extension",
                "solution_path": f"docs/solutions/{eid}.md",
                "results_path": f"artifacts/{eid}/results.json",
                "implementation_module": f"convex_solutions.chapters.ch{record['chapter']:02d}.ex_{eid.replace('.', '_')}",
                "verification_path": "coverage/verification.json",
            })
        elif eid in solved:
            update = solved[eid]
            forbidden = {"key", "kind", "chapter", "category", "source_page", "source_pdf_page", "source_url"}
            if forbidden.intersection(update):
                raise ValueError(f"Solution overlay cannot change source identity: {eid}")
            record.update(update)
        else:
            record["status_note"] = "A complete, independently reviewed solution is not delivered yet."
        exercises.append(record)
    unknown_solutions = set(solved) - {e["id"] for e in exercises if e["category"] == "other_exercises"}
    if unknown_solutions:
        raise ValueError(f"Unexpected solution overlay IDs: {sorted(unknown_solutions)}")
    examples = []
    for original in worked["entries"]:
        record = dict(original)
        record.update({"key": f"example:{original['id']}", "kind": "worked_example",
                       "category": "examples"})
        if record.get("status") == "not_implemented":
            record["status"] = "pending"
        record.update(reproduced.get(original["id"], {}))
        examples.append(record)
    unknown = set(reproduced) - {e["id"] for e in examples}
    if unknown:
        raise ValueError(f"Reproductions missing from source inventory: {sorted(unknown)}")
    counts = {
        "numbered_exercises": len(exercises),
        "computational_exercises_verified": len(implemented),
        "numbered_exercises_verified": sum(e["status"] == "verified" for e in exercises),
        "other_exercises_total": sum(e["category"] == "other_exercises" for e in exercises),
        "other_exercises_verified": sum(e["category"] == "other_exercises" and e["status"] == "verified" for e in exercises),
        "other_exercises_pending": sum(e["category"] == "other_exercises" and e["status"] != "verified" for e in exercises),
        "worked_examples_catalogued": len(examples),
        "worked_examples_reproduced": sum(e["status"] == "verified" for e in examples),
        "worked_examples_exact": sum(e["status"] == "verified" and e.get("reproduction_kind", "exact") == "exact" for e in examples),
        "worked_examples_method": sum(e["status"] == "verified" and e.get("reproduction_kind") == "method" for e in examples),
    }
    categories = [
        {"id": "examples", "number": 1, "title": "Examples",
         "description": "Numerical worked examples in the text with supplied results to reproduce.",
         "index_path": CATEGORY_PATHS["examples"], "count": len(examples)},
        {"id": "computational_exercises", "number": 2, "title": "Computational exercises",
         "description": "The 16 existing numerical exercise solutions, including five numerical extensions.",
         "index_path": CATEGORY_PATHS["computational_exercises"], "count": len(implemented)},
        {"id": "other_exercises", "number": 3, "title": "All other exercises",
         "description": "Every remaining numbered textbook exercise, with completion based on independent review.",
         "index_path": CATEGORY_PATHS["other_exercises"], "count": counts["other_exercises_total"]},
    ]
    inputs = ["coverage/source_audit.json", "coverage/inventory.json", "coverage/worked_examples.json", "coverage/exercise_solutions.json"]
    if (ROOT / "coverage/example_reproductions.json").exists():
        inputs.append("coverage/example_reproductions.json")
    return {
        "schema_version": 1, "catalog_date": "2026-10-10",
        "source": {k: v for k, v in audit["source"].items()
                   if k not in {"worked_examples_included", "supplemental_exercises_included"}},
        "scope": {
            "numbered_exercise_inventory": "complete",
            "numbered_exercise_solution_set": "complete" if counts["numbered_exercises_verified"] == len(exercises) else "incomplete",
            "worked_example_inventory": worked.get("inventory_status", worked.get("audit_status", "See worked_examples.json for audit scope and limitations.")),
            "worked_example_inventory_note": worked.get("inventory_note", "See the worked-example source inventory for its screening coverage and limits."),
            "worked_example_inventory_completeness": worked.get("inventory_completeness", {}),
            "worked_example_policy": worked.get("policy", {}),
            "separate_supplementary_exercise_collection_included": False,
            "note": "Category membership is organizational, not a claim of completion. The historical computational exclusions do not exclude theoretical exercises from this catalog.",
        },
        "summary": counts, "categories": categories,
        "inputs": [{"path": p, "sha256": hashlib.sha256((ROOT / p).read_bytes()).hexdigest()} for p in inputs],
        "chapters": [{"chapter": int(ch), "title": CHAPTER_TITLES[int(ch)],
                      "numbered_exercises": info["exercise_count"],
                      "computational_exercises": sum(e["chapter"] == int(ch) and e["category"] == "computational_exercises" for e in exercises),
                      "other_exercises": sum(e["chapter"] == int(ch) and e["category"] == "other_exercises" for e in exercises),
                      "other_exercises_verified": sum(e["chapter"] == int(ch) and e["category"] == "other_exercises" and e["status"] == "verified" for e in exercises),
                      "other_exercises_pending": sum(e["chapter"] == int(ch) and e["category"] == "other_exercises" and e["status"] != "verified" for e in exercises)}
                     for ch, info in audit["chapters"].items()],
        "entries": examples + exercises,
    }


def nav() -> str:
    return ("[Book index](../README.md) · [1. Examples](../examples/README.md) · "
            "[2. Computational exercises](../computational-exercises/README.md) · "
            "[3. All other exercises](../other-exercises/README.md)\n")


def cell(value) -> str:
    if isinstance(value, list):
        value = "; ".join(str(v) for v in value)
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_computational(catalog: dict) -> str:
    lines = ["# 2. Computational exercises", "", nav(),
             "These are the **16 already-delivered numerical exercise solutions**. All have "
             "original derivations, runnable Python code, executed result tables, and verified output.", "",
             "Eleven exercises explicitly request numerical execution or implementation. The other "
             "five are numerical extensions of end-of-chapter exercises. All sixteen belong here; "
             "none is a worked example from the textbook narrative.", "",
             "| Exercise | Topic | Textbook role | Status | Executed output |",
             "|---|---|---|---|---|"]
    for e in catalog["entries"]:
        if e["category"] != "computational_exercises":
            continue
        role = "Explicit computation" if e["textbook_role"] == "explicit_computation" else "Numerical extension"
        lines.append(f"| [{e['id']}](../solutions/{e['id']}.md) | {cell(e['title'])} | {role} | Verified | "
                     f"[Tables](../solutions/{e['id']}.md#executed-results) · [JSON](../../{e['results_path']}) · [Book p. {e['source_page']}]({e['source_url']}) |")
    lines += ["", "The original `docs/solutions/` URLs remain unchanged. The numerical extensions "
              "are 4.3, 5.29, 5.33, 5.34, and 11.1.", "", "## Reproduce", "", "```sh",
              "convex-solutions run all --output artifacts --seed 0", "python scripts/render_results.py --check", "```", "",
              "See the [execution record](../../coverage/results-refresh.json), "
              "[verification report](../../coverage/verification.json), and "
              "[complete category manifest](../../coverage/catalog.json).", ""]
    return "\n".join(lines)


def render_other(catalog: dict) -> str:
    summary = catalog["summary"]
    lines = ["# 3. All other exercises", "", nav(),
             f"**{summary['other_exercises_total']} numbered exercises; "
             f"{summary['other_exercises_verified']} solutions verified, "
             f"{summary['other_exercises_pending']} still open.** This is the complete complement "
             "of the 16 delivered computational solutions within the book's 337 numbered exercises.", "",
             "Theoretical proofs, symbolic problems, modeling exercises, and additional numerical "
             "possibilities all belong here. Verified entries link to complete original solutions "
             "and independent, source-checked review evidence. Category membership does not change "
             "when a solution is completed.", "",
             "## Chapters", "", "| Chapter | Verified | Open | Numbered exercises in chapter |",
             "|---|---|---|---|"]
    for ch in catalog["chapters"]:
        lines.append(f"| [Chapter {ch['chapter']}: {ch['title']}](#chapter-{ch['chapter']}) | "
                     f"{ch['other_exercises_verified']} | {ch['other_exercises_pending']} | {ch['numbered_exercises']} |")
    lines += ["", "Chapter 1 and the appendices contain no numbered end-of-chapter exercises. "
              "The separate supplementary exercise collection is not included in this book catalog.", ""]
    for ch in catalog["chapters"]:
        lines += [f"## Chapter {ch['chapter']}", "", f"{ch['title']} · "
                  f"**{ch['other_exercises_verified']} verified, {ch['other_exercises_pending']} open**", "",
                  "| Exercise | Source in official book | Status |", "|---|---|---|"]
        for e in catalog["entries"]:
            if e["category"] == "other_exercises" and e["chapter"] == ch["chapter"]:
                status = f"[Verified](../../{e['solution_path']}) · [Review](../../{e['verification_path']})" if e["status"] == "verified" else e["status"].capitalize()
                lines.append(f"| {e['id']} | [Printed p. {e['source_page']} / PDF p. {e['source_pdf_page']}]({e['source_url']}) | {status} |")
        lines.append("")
    lines += ["## Coverage", "", "The [category manifest](../../coverage/catalog.json) lists each numbered "
              "exercise exactly once. The [work ledger](../../coverage/solution_progress.json) distinguishes "
              "active work, submitted solutions, accepted solutions, and exact blockers. The "
              "[historical source audit](../../coverage/source_audit.json) supplies source-page checks; "
              "its former exclusions are not current scope exclusions.", ""]
    return "\n".join(lines)


def render_examples(catalog: dict) -> str:
    entries = [e for e in catalog["entries"] if e["category"] == "examples"]
    available = [e for e in entries if e["status"] == "verified"]
    available_lines = []
    if available:
        available_lines = ["## Available reproductions", ""]
        for e in available:
            label = "exact-data reproduction" if e.get("reproduction_kind", "exact") == "exact" else "method reproduction on documented data"
            available_lines.append(f"- [{e['title']}]({Path(e['solution_path']).name}) ({label}): "
                                   f"original explanation and [executed output](../../{e['results_path']}).")
        available_lines.append("")
    lines = ["# 1. Examples", "", nav(),
             "Numerical worked examples in the textbook narrative, where the book supplies a "
             "numerical result or plot to reproduce. These are separate from end-of-chapter exercises.", "",
             f"**{len(entries)} catalogued; {sum(e['status'] == 'verified' for e in entries)} reproduced in this repository.** "
             f"{catalog['summary']['worked_examples_exact']} exact-data and {catalog['summary']['worked_examples_method']} method reproductions. "
             "Method reproductions use explicitly documented inputs when the original inputs are unavailable; they do not claim an exact figure match. "
             "Verified entries link to original solution notes and executed output; every other entry "
             "is pending. A source's solution or figure alone is not evidence of a completed implementation here.", "",
             *available_lines,
             "## Inventory and reproducibility", "",
             catalog["scope"]["worked_example_inventory_note"], "",
             "The [worked-example source inventory](../../coverage/worked_examples.json) records "
             "the screening policy, audit coverage, source references, expected outputs, and data limits. "
             "Consult its inventory status before treating this list as exhaustive.", "",
             "- Exact data available: source data/code can support a matched reproduction.",
             "- Self-contained: the text supplies enough information to construct the stated example.",
             "- Official seeded generator: source code fixes the original random generator state; "
             "using the same integer seed in a different generator does not recreate its arrays.",
             "- Method only / unspecified data: reproduce the method on clearly labeled data; "
             "do not claim an exact match to the book's plotted instance.", ""]
    for chapter in sorted({e["chapter"] for e in entries}):
        lines += [f"## Chapter {chapter}", "", "| Example / figure | Topic | Reproduction basis | Status |",
                  "|---|---|---|---|"]
        for e in entries:
            if e["chapter"] != chapter:
                continue
            refs = e.get("textbook_refs", [])
            label = "; ".join(f"{r.get('kind', 'Reference')} {r.get('id', '')}".strip() for r in refs) or e["id"]
            basis = e.get("reproducibility", "See source inventory").replace("_", " ")
            status = f"[Verified {e.get('reproduction_kind', 'exact')}]({Path(e['solution_path']).name}) · [Output](../../{e['results_path']})" if e["status"] == "verified" else "Pending"
            lines.append(f"| [{cell(label)}]({e['source_url']}) | {cell(e['title'])} | {cell(basis)} | {status} |")
        lines.append("")
        lines += ["### Source and data notes", ""]
        for e in entries:
            if e["chapter"] != chapter:
                continue
            notes = e.get("notes", [])
            if isinstance(notes, str):
                notes = [notes]
            detail = " ".join([e.get("data_availability", "See source inventory."), *notes])
            lines.append(f"- **{e['title']}:** {detail}")
        lines.append("")
    lines += ["## Next steps", "", "For each reproduction, use the identified source and available "
              "data, write original notes, run the implementation, and compare the output with the "
              "book's reported result. Record discrepancies and unspecified inputs explicitly. "
              "Only then add solution/output links and update its status.", "",
              "The [complete category manifest](../../coverage/catalog.json) keeps this catalog "
              "separate from the 337 numbered exercises.", ""]
    return "\n".join(lines)


def replace_section(text: str, start: str, end: str, body: str) -> str:
    a, b = text.index(start), text.index(end)
    return text[:a] + body.rstrip() + "\n\n" + text[b:]


def render_summary(catalog: dict, prefix: str = "docs/") -> str:
    s = catalog["summary"]
    return (f"1. **[Examples]({prefix}examples/README.md)**: "
            f"**{s['worked_examples_reproduced']} of {s['worked_examples_catalogued']}** "
            "numerical worked-example studies reproduced and verified.\n"
            f"2. **[Computational exercises]({prefix}computational-exercises/README.md)**: "
            "the **16 already-delivered numerical solutions**, with original derivations, "
            "executable code, and actual result tables.\n"
            f"3. **[All other exercises]({prefix}other-exercises/README.md)**: "
            f"**{s['other_exercises_verified']} of {s['other_exercises_total']}** remaining "
            "numbered exercises have complete, independently verified solutions; "
            f"**{s['other_exercises_pending']} remain open**.")


def generated_files() -> dict[str, str]:
    catalog = build_catalog()
    return {
        "coverage/catalog.json": json.dumps(catalog, indent=2, ensure_ascii=False) + "\n",
        "README.md": replace_section((ROOT / "README.md").read_text(), "## Start here", "## Verified scope",
            "## Start here\n\n" + render_summary(catalog) + "\n\nThe [canonical category manifest](coverage/catalog.json) "
            "accounts for all **337 numbered exercises exactly once** across categories 2 and 3. "
            "Worked examples are tracked separately. The [work ledger](coverage/solution_progress.json) "
            "records all 399 items and distinguishes active work, independent review, completion, "
            "and exact blockers. Inventory coverage is not solution completion."),
        "docs/README.md": replace_section((ROOT / "docs/README.md").read_text(), "## Three categories", "## Existing solution links",
            "## Three categories\n\n" + render_summary(catalog, "") + "\n\nAll 337 numbered exercises appear "
            "exactly once in categories 2 and 3. Worked examples are separate and do not count toward "
            "those 337. The [canonical manifest](../coverage/catalog.json) records categories, "
            "status, sources, and delivered files. The [work ledger](../coverage/solution_progress.json) "
            "tracks completion and blockers across the full request."),
        CATEGORY_PATHS["examples"]: render_examples(catalog),
        CATEGORY_PATHS["computational_exercises"]: render_computational(catalog),
        CATEGORY_PATHS["other_exercises"]: render_other(catalog),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if the manifest or indexes are stale.")
    args = parser.parse_args()
    for relative, content in generated_files().items():
        path = ROOT / relative
        if args.check:
            if not path.is_file() or path.read_text() != content:
                raise SystemExit(f"Stale or missing catalog output: {relative}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    print("Three-category catalog and indexes are current." if args.check else "Built three-category catalog and indexes.")


if __name__ == "__main__":
    main()
