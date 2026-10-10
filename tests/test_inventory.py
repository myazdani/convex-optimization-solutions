from pathlib import Path
import inspect
import json
import jsonschema
from convex_solutions.registry import load_inventory, implemented, load_solution, inventory_path


def test_manifest_schema_and_unique_ids():
    inventory = load_inventory()
    root = Path(__file__).resolve().parents[1]
    schema = json.loads((root / "coverage/inventory.schema.json").read_text())
    jsonschema.validate(inventory, schema)
    ids = [entry["id"] for entry in inventory["exercises"]]
    assert len(ids) == len(set(ids))


def test_implemented_solution_contracts():
    for entry in load_inventory()["exercises"]:
        if implemented(entry["id"]):
            signature = inspect.signature(load_solution(entry["id"]))
            assert "output_dir" in signature.parameters
            assert "seed" in signature.parameters


def test_verified_entries_have_explanations():
    root = Path(__file__).resolve().parents[1]
    for entry in load_inventory()["exercises"]:
        if entry.get("status") == "verified":
            assert implemented(entry["id"])
            assert (root / "docs/solutions" / f"{entry['id']}.md").is_file()


def test_full_book_audit_has_no_numbering_gaps():
    root = Path(__file__).resolve().parents[1]
    audit = json.loads((root / "coverage/source_audit.json").read_text())
    entries = audit["exercises"]
    ids = {entry["id"] for entry in entries}
    assert len(ids) == len(entries) == audit["summary"]["numbered_exercises_reviewed"]
    expected = set()
    for chapter, record in audit["chapters"].items():
        expected.update(f"{chapter}.{number}" for number in range(1, record["exercise_count"] + 1))
    assert ids == expected
    manifest = load_inventory()
    assert {entry["id"] for entry in manifest["exercises"]}.issubset(ids)
    selected = {entry["id"] for entry in entries if entry["selected_for_implementation"]}
    assert selected == {entry["id"] for entry in manifest["exercises"]
                        if entry.get("selected_for_implementation")}
