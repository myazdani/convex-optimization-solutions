"""Full book categorization must never imply unimplemented work is solved."""
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_catalog import generated_files
from validate_catalog import validate, check_local_links


def test_catalog_schema_coverage_statuses_and_evidence():
    summary = validate()
    assert summary["numbered_exercises"] == 337
    assert summary["computational_exercises_verified"] == 16
    assert summary["other_exercises_pending"] == 321


def test_generated_category_pages_are_current():
    for relative, expected in generated_files().items():
        assert (ROOT / relative).read_text() == expected


def test_all_documentation_local_links_and_anchors():
    check_local_links()


def test_examples_do_not_change_existing_exercise_provenance():
    from convex_solutions.cli import source_fingerprint
    current = source_fingerprint()
    for result in (ROOT / "artifacts").glob("*/results.json"):
        assert json.loads(result.read_text())["source_tree_sha256"] == current


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "unsubstantiated_solution", "unverified_computational", "unverified_example_output"])
def test_validation_rejects_misleading_catalog(mutation):
    catalog = json.loads((ROOT / "coverage/catalog.json").read_text())
    if mutation == "duplicate":
        catalog["entries"].append(copy.deepcopy(catalog["entries"][-1]))
    elif mutation == "missing":
        catalog["entries"].pop()
    elif mutation == "unsubstantiated_solution":
        pending = next(e for e in catalog["entries"] if e["category"] == "other_exercises")
        pending["status"] = "verified"
    elif mutation == "unverified_computational":
        next(e for e in catalog["entries"] if e["category"] == "computational_exercises")["status"] = "implemented"
    else:
        next(e for e in catalog["entries"] if e["category"] == "examples" and e["status"] == "verified")["output_match_verified"] = False
    with pytest.raises(AssertionError):
        validate(catalog)
