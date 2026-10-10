"""Write the durable full-scope work ledger without promoting unreviewed work."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from build_catalog import ROOT, build_catalog


def refresh_queue() -> dict:
    """Capture draft status without publishing or claiming completion of drafts."""
    submitted = {}
    blockers = {}
    for path in sorted((ROOT / "coverage/submissions").glob("*.json")):
        record = json.loads(path.read_text())
        for entry in record.get("entries", []):
            kind = "example" if entry["id"].startswith(("example_", "figure_", "section_")) or path.name.startswith("examples_") else "exercise"
            key = f"{kind}:{entry['id']}"
            if path.stem.endswith("_blockers"):
                blockers[key] = entry
            else:
                submitted[key] = entry
    queue = {"schema_version": 1, "submitted_keys": sorted(submitted), "blockers": blockers}
    (ROOT / "coverage/work_queue.json").write_text(json.dumps(queue, indent=2, ensure_ascii=False) + "\n")
    return queue


def generated_progress() -> dict:
    catalog = build_catalog()
    queue_path = ROOT / "coverage/work_queue.json"
    queue = json.loads(queue_path.read_text()) if queue_path.exists() else {"submitted_keys": [], "blockers": {}}
    submitted, blockers = set(queue["submitted_keys"]), queue["blockers"]
    entries = []
    for item in catalog["entries"]:
        state = "verified" if item["status"] == "verified" else "in_progress"
        e = {"key": item["key"], "id": item["id"], "chapter": item["chapter"], "category": item["category"],
             "title": item["title"], "state": state, "source_url": item["source_url"]}
        if state == "verified":
            e.update(solution_path=item["solution_path"], verification_path=item["verification_path"])
        elif item["key"] in blockers:
            e.update(state="blocked", blocker=blockers[item["key"]])
        elif item["key"] in submitted:
            e["state"] = "in_review"
        entries.append(e)
    states = ("verified", "in_progress", "in_review", "blocked")
    summary = {s: sum(e["state"] == s for e in entries) for s in states}
    return {"schema_version": 1, "request_date": "2026-10-10", "scope": {
        "numbered_exercises": 337, "worked_example_studies": 62, "total_items": 399,
        "previously_verified_items": 18, "newly_requested_items": 381,
        "completion_rule": "All requested subparts complete, source checked, independently reviewed against the exact saved bytes, and numerical results executed where required.",
        "exact_reproduction_rule": "User approved documented method reproductions when exact textbook inputs are unavailable (2026-10-10). They may complete the task but never claim an exact output match."},
        "summary": summary, "entries": entries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--refresh-submissions", action="store_true", help="Refresh staged work status before generating")
    args = parser.parse_args()
    if args.refresh_submissions:
        refresh_queue()
    content = json.dumps(generated_progress(), indent=2, ensure_ascii=False) + "\n"
    path = ROOT / "coverage/solution_progress.json"
    if args.check:
        assert path.read_text() == content, "Work ledger is stale"
    else:
        path.write_text(content)
    print(json.dumps(json.loads(content)["summary"]))


if __name__ == "__main__":
    main()
