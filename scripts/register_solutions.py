"""Accept a submitted solution batch only after independent, byte-bound review."""
from __future__ import annotations
import argparse
import json
from solution_evidence import ROOT, validate_solution_evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", help="Repository-relative submission JSON")
    parser.add_argument("review", help="Repository-relative independent review JSON")
    parser.add_argument("--ids", nargs="+", help="Only accept these exercise IDs")
    args = parser.parse_args()
    submission = json.loads((ROOT / args.submission).read_text())
    review = json.loads((ROOT / args.review).read_text())
    accepted_ids = {r["id"] for r in review["entries"] if r.get("accepted") is True}
    if args.ids:
        assert set(args.ids) <= accepted_ids, "Requested exercise lacks accepted review"
        accepted_ids = set(args.ids)
    overlay_path = ROOT / "coverage/exercise_solutions.json"
    overlay = json.loads(overlay_path.read_text())
    indexed = {e["id"]: e for e in overlay["entries"]}
    changes = []
    for submitted in submission["entries"]:
        if submitted["id"] not in accepted_ids:
            continue
        entry = dict(submitted)
        entry.pop("source_url", None)  # immutable source identity comes from the source audit
        entry.update(status="verified", author=submission["author"], verification_path=args.review)
        validate_solution_evidence(entry)
        indexed[entry["id"]] = entry
        changes.append(entry["id"])
    assert set(changes) == accepted_ids, "Review references an exercise absent from this submission"
    overlay["entries"] = sorted(indexed.values(), key=lambda e: tuple(map(int, e["id"].split("."))))
    overlay_path.write_text(json.dumps(overlay, indent=2, ensure_ascii=False) + "\n")
    print("Registered independently verified solutions: " + ", ".join(changes))


if __name__ == "__main__":
    main()
