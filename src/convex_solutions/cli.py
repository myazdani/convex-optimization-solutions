"""Command-line entry point and reproducibility envelopes."""
from __future__ import annotations
import argparse
import hashlib
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
import json
import os
from pathlib import Path
import platform
import sys
import time

from .io import json_safe, write_json
from .registry import audit_inventory, implemented, inventory_path, load_inventory, load_solution


def source_fingerprint() -> str:
    """Platform-independent digest of the Python source tree for this run."""
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def environment() -> dict:
    versions = {}
    for package in ("numpy", "scipy", "matplotlib", "cvxpy", "clarabel", "osqp", "scs"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    try:
        import cvxpy as cp
        solvers = cp.installed_solvers()
    except ImportError:
        solvers = []
    return {"python": sys.version, "platform": platform.platform(),
            "packages": versions, "available_solvers": solvers,
            "thread_environment": {key: os.environ.get(key) for key in
                                   ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}


def run_exercise(exercise_id: str, output_dir: Path, seed: int) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    solve = load_solution(exercise_id)
    started = time.perf_counter()
    result = solve(output_dir=output_dir, seed=seed)
    if not isinstance(result, dict):
        raise TypeError(f"{exercise_id}: solve() must return a dict")
    envelope = {"exercise_id": exercise_id, "seed": seed,
                "executed_at": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": time.perf_counter() - started,
                "environment": environment(),
                "source_tree_sha256": source_fingerprint(),
                "inventory_sha256": hashlib.sha256(inventory_path().read_bytes()).hexdigest(),
                "results": json_safe(result)}
    write_json(output_dir / "results.json", envelope)
    return envelope


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ls = sub.add_parser("list", help="list the source-audited inventory")
    ls.add_argument("--json", action="store_true")
    au = sub.add_parser("audit", help="report coverage without inferring completeness")
    au.add_argument("--require-complete", action="store_true")
    run = sub.add_parser("run", help="run one exercise or every implemented exercise")
    run.add_argument("exercise_id")
    run.add_argument("--output", type=Path, default=Path("artifacts"))
    run.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        inventory = load_inventory()
        audit = audit_inventory(inventory)
        if args.command == "list":
            if args.json:
                print(json.dumps(inventory, indent=2))
            else:
                print(f"Source audit: {audit['audit_status']}")
                for entry in inventory["exercises"]:
                    code = "code" if implemented(entry["id"]) else "no code"
                    print(f"{entry['id']:>5}  {entry.get('scope', '?'):8} "
                          f"{entry.get('status', 'pending'):14} {code:7} {entry['title']}")
            return 0
        if args.command == "audit":
            print(json.dumps(audit, indent=2))
            return int(bool(audit["issues"]) or (args.require_complete and not audit["complete"]))
        all_exercises = args.exercise_id == "all"
        ids = audit["implemented_ids"] if all_exercises else [args.exercise_id]
        inventory_ids = {entry["id"] for entry in inventory["exercises"]}
        if not all_exercises and args.exercise_id not in inventory_ids:
            raise ValueError(f"Exercise {args.exercise_id} is not in the audited inventory")
        if not ids:
            raise ValueError("No implemented exercises are available yet")
        outcomes = []
        for eid in ids:
            path = args.output / eid if all_exercises else args.output
            try:
                envelope = run_exercise(eid, path, args.seed)
                outcomes.append({"id": eid, "status": "success",
                                 "results": str(path / "results.json"),
                                 "duration_seconds": envelope["duration_seconds"]})
                print(f"{eid}: wrote {path / 'results.json'}")
            except Exception as exc:
                outcomes.append({"id": eid, "status": "failed", "error": str(exc)})
                print(f"{eid}: {exc}", file=sys.stderr)
        if all_exercises:
            write_json(args.output / "run_summary.json", {"runs": outcomes, "coverage": audit})
            print(f"Required exercises without code: {len(audit['missing_required'])}. "
                  f"Required computational scope verified: {audit['required_complete']}.")
        return int(any(item["status"] == "failed" for item in outcomes))
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.error(str(exc))
    return 2
