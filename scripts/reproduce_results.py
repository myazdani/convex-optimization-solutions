"""Execute the complete profile, record provenance, and refresh Markdown tables."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    environment = os.environ.copy()
    thread_keys = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    environment.update({key: "1" for key in thread_keys})
    command = [sys.executable, "-u", "-m", "convex_solutions", "run", "all",
               "--output", "artifacts", "--seed", "0"]
    started = datetime.now(timezone.utc).isoformat()
    log = ROOT / "coverage" / "results-rerun.log"
    with log.open("w") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
        code = process.wait()
    if code:
        print(f"Execution failed ({code}); Markdown was not refreshed.", file=sys.stderr)
        return code
    summary = json.loads((ROOT / "artifacts" / "run_summary.json").read_text())
    ids = [entry["id"] for entry in summary["runs"] if entry["status"] == "success"]
    if len(ids) != len(summary["runs"]):
        raise RuntimeError("Not every exercise succeeded; Markdown was not refreshed")
    outputs = []
    envelopes = []
    for eid in ids:
        path = ROOT / "artifacts" / eid / "results.json"
        envelope = json.loads(path.read_text())
        envelopes.append(envelope)
        outputs.append({"exercise_id": eid, "path": path.relative_to(ROOT).as_posix(),
                        "executed_at": envelope["executed_at"],
                        "duration_seconds": envelope["duration_seconds"],
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if any(item["executed_at"] < started for item in envelopes):
        raise RuntimeError("A result predates the current execution")
    first = envelopes[0]
    record = {
        "description": "Fresh full execution used to generate numerical tables in every solution Markdown page. Earlier coverage reports are historical validation evidence.",
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(f"{key}=1" for key in thread_keys) + " " + shlex.join(command),
        "reproduce_command": "python scripts/reproduce_results.py",
        "exit_code": code, "mode": "full", "seed": 0, "exercise_ids": ids,
        "successful_exercises": len(ids), "failed_exercises": 0,
        "source_tree_sha256": first["source_tree_sha256"],
        "inventory_sha256": first["inventory_sha256"],
        "environment": first["environment"],
        "total_solver_plot_seconds": sum(item["duration_seconds"] for item in outputs),
        "log": log.relative_to(ROOT).as_posix(), "outputs": outputs, "checks": [],
        "limitations": [
            "Runtime and final floating-point digits vary by machine and solver versions.",
            "Embedded results are numerical checks, not interval-arithmetic proofs.",
            "The 16 selected solutions cover 11 required computational exercises plus five selected extras, not every analytical textbook exercise.",
        ],
    }
    (ROOT / "coverage" / "results-refresh.json").write_text(json.dumps(record, indent=2) + "\n")
    return subprocess.call([sys.executable, "scripts/render_results.py"], cwd=ROOT, env=environment)


if __name__ == "__main__":
    raise SystemExit(main())
