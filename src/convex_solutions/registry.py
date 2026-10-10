"""Source-backed exercise inventory and lazy implementation discovery."""
from __future__ import annotations
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re

EXERCISE_ID = re.compile(r"^([1-9][0-9]*)\.([1-9][0-9]*)([a-z]?)$")
STATUSES = {"pending", "implemented", "verified", "blocked", "not_applicable"}


def inventory_path() -> Path:
    explicit = os.environ.get("CONVEX_SOLUTIONS_INVENTORY")
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"Configured inventory does not exist: {path}")
        return path
    root = Path(__file__).resolve().parents[2]
    candidates = [root / "coverage" / "inventory.json",
                  Path(__file__).parent / "resources" / "inventory.json"]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("No exercise inventory found. Run from the source checkout or reinstall package.")


def load_inventory(path: Path | None = None) -> dict:
    value = json.loads((path or inventory_path()).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("exercises"), list):
        raise ValueError("Inventory must be an object containing an exercises array")
    return value


def module_name(exercise_id: str) -> str:
    match = EXERCISE_ID.fullmatch(exercise_id)
    if not match:
        raise ValueError(f"Invalid exercise ID: {exercise_id!r}")
    chapter, number, suffix = match.groups()
    return f"convex_solutions.chapters.ch{int(chapter):02d}.ex_{chapter}_{number}{suffix}"


def implemented(exercise_id: str) -> bool:
    try:
        return importlib.util.find_spec(module_name(exercise_id)) is not None
    except ModuleNotFoundError:
        return False


def load_solution(exercise_id: str):
    if not implemented(exercise_id):
        raise ValueError(f"Exercise {exercise_id} does not have an implementation yet")
    module = importlib.import_module(module_name(exercise_id))
    if not callable(getattr(module, "solve", None)):
        raise ValueError(f"Exercise {exercise_id} module does not expose solve()")
    return module.solve


def audit_inventory(value: dict) -> dict:
    issues = []
    ids = set()
    counts = {status: 0 for status in sorted(STATUSES)}
    missing = []
    implemented_ids = []
    required = []
    selected = []
    for entry in value["exercises"]:
        eid = entry.get("id", "")
        if not EXERCISE_ID.fullmatch(eid):
            issues.append(f"Invalid exercise ID: {eid!r}")
            continue
        if eid in ids:
            issues.append(f"Duplicate exercise ID: {eid}")
        ids.add(eid)
        if entry.get("chapter") != int(eid.split(".")[0]):
            issues.append(f"Chapter mismatch: {eid}")
        status = entry.get("status", "pending")
        if status not in STATUSES:
            issues.append(f"Invalid status for {eid}: {status}")
        else:
            counts[status] += 1
        has_code = implemented(eid)
        if has_code:
            implemented_ids.append(eid)
        if status in {"implemented", "verified"} and not has_code:
            issues.append(f"{eid} claims {status} but its module is missing")
        if entry.get("selected_for_implementation", entry.get("scope") == "required"):
            selected.append(eid)
        if entry.get("scope") == "required" and status != "not_applicable":
            required.append(eid)
            if not has_code:
                missing.append(eid)
    audit_complete = value.get("audit_status") == "complete"
    required_complete = bool(required) and audit_complete and not issues and all(
        entry.get("status") == "verified" for entry in value["exercises"]
        if entry.get("id") in required)
    selected_complete = bool(selected) and audit_complete and not issues and all(
        entry.get("status") == "verified" for entry in value["exercises"]
        if entry.get("id") in selected)
    return {"audit_status": value.get("audit_status", "in_progress"),
            "required_count": len(required), "status_counts": counts,
            "implemented_ids": implemented_ids, "missing_required": missing,
            "selected_count": len(selected),
            "missing_selected": [eid for eid in selected if not implemented(eid)],
            "required_complete": required_complete, "selected_complete": selected_complete,
            "issues": issues, "complete": required_complete and selected_complete}
