"""The displayed numerical tables must be reproducible from real run outputs."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from render_results import BEGIN, END, load_envelope, render_block, replace_block
from convex_solutions.cli import source_fingerprint
from convex_solutions.registry import audit_inventory, inventory_path, load_inventory

IDS = audit_inventory(load_inventory())["implemented_ids"]


@pytest.mark.parametrize("exercise_id", IDS)
def test_embedded_results_match_executed_outputs(exercise_id):
    envelope = load_envelope(exercise_id, ROOT / "artifacts")
    datetime.fromisoformat(envelope["executed_at"])
    assert envelope["source_tree_sha256"] == source_fingerprint()
    assert envelope["inventory_sha256"] == hashlib.sha256(inventory_path().read_bytes()).hexdigest()
    path = ROOT / "docs" / "solutions" / f"{exercise_id}.md"
    document = path.read_text()
    block = render_block(exercise_id, envelope)
    assert replace_block(document, block) == document, (
        f"{exercise_id}: rerun python scripts/render_results.py after changing results or renderer"
    )
    assert document.count("## Executed results") == 1
    assert envelope["executed_at"].replace("+00:00", "Z") in block
    # Each displayed local figure/data link must resolve within the repository.
    for target in re.findall(r"\]\((\.\./[^)#]+)(?:#[^)]*)?\)", document):
        assert (path.parent / target).resolve().exists(), f"Broken link: {target}"


def test_generated_results_reject_missing_or_duplicate_markers():
    for document in ("No markers", f"{BEGIN}\n{BEGIN}\n{END}", f"{BEGIN}\n{END}\n{END}"):
        with pytest.raises(ValueError, match="exactly one"):
            replace_block(document, "replacement")


def test_execution_record_covers_delivered_outputs():
    record = json.loads((ROOT / "coverage" / "results-refresh.json").read_text())
    assert sorted(record["exercise_ids"]) == sorted(IDS)
    assert record["seed"] == 0
    assert record["mode"] == "full"
    for entry in record["outputs"]:
        path = ROOT / entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
