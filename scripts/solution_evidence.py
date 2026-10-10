"""Bind accepted original proofs to independent reviews of their exact bytes."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def validate_solution_evidence(entry: dict, root: Path = ROOT) -> None:
    """Reject unsupported completion claims; numerical output is optional for proofs."""
    eid = entry["id"]
    assert entry.get("status") == "verified", f"Only verified solutions belong in the accepted overlay: {eid}"
    assert entry.get("source_checked") is True, f"Source not checked: {eid}"
    assert entry.get("all_subparts_completed") is True, f"Incomplete subparts: {eid}"
    assert entry.get("subparts") and all(isinstance(x, str) and x for x in entry["subparts"]), f"Missing subpart inventory: {eid}"
    assert len(entry["subparts"]) == len(set(entry["subparts"])), f"Duplicate subpart inventory: {eid}"
    assert entry.get("author"), f"Missing authorship identifier: {eid}"
    for key in ("solution_path", "verification_path"):
        assert entry.get(key), f"Missing {key}: {eid}"
        path = (root / entry[key]).resolve()
        assert path.is_relative_to(root.resolve()) and path.is_file(), f"Invalid evidence path: {eid}: {key}"
    solution = root / entry["solution_path"]
    actual_hash = hashlib.sha256(solution.read_bytes()).hexdigest()
    assert entry.get("solution_sha256") == actual_hash, f"Changed solution after review: {eid}"
    review_document = json.loads((root / entry["verification_path"]).read_text())
    reviews = [r for r in review_document["entries"] if r["id"] == eid]
    assert len(reviews) == 1, f"Missing/duplicate independent review: {eid}"
    review = reviews[0]
    assert review.get("reviewer") and review["reviewer"] != entry["author"], f"Review must be independent: {eid}"
    assert review.get("solution_sha256") == actual_hash, f"Review does not cover current solution: {eid}"
    for key in ("accepted", "source_checked", "all_subparts_checked", "correctness_checked", "assumptions_checked"):
        assert review.get(key) is True, f"Review has not accepted {key}: {eid}"
    assert set(review.get("subparts", [])) == set(entry["subparts"]), f"Review subparts mismatch: {eid}"
    assert review.get("notes"), f"Review must summarize mathematical checks: {eid}"
    for key in ("results_path", "code_path"):
        if key in entry:
            assert (root / entry[key]).is_file(), f"Missing computational evidence: {eid}: {key}"
