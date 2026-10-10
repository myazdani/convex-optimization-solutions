import json
import sys
from types import SimpleNamespace
import pytest
from convex_solutions.cli import main, run_exercise
from convex_solutions.io import write_json


@pytest.fixture
def inventory(tmp_path, monkeypatch):
    p = tmp_path / "inventory.json"
    write_json(p, {"schema_version": 1, "audit_status": "in_progress", "source": "test",
                   "exercises": [{"id": "99.1", "chapter": 99, "title": "Test fixture",
                                  "scope": "required", "status": "pending"}]})
    monkeypatch.setenv("CONVEX_SOLUTIONS_INVENTORY", str(p))
    return p


def test_list_and_audit(inventory, capsys):
    assert main(["list"]) == 0
    assert "99.1" in capsys.readouterr().out
    assert main(["audit", "--require-complete"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert not result["complete"]
    assert result["missing_required"] == ["99.1"]


def test_unimplemented_run_is_not_success(inventory, tmp_path, capsys):
    assert main(["run", "99.1", "--output", str(tmp_path / "out")]) == 1
    assert not (tmp_path / "out" / "results.json").exists()


def test_unknown_exercise_rejected(inventory):
    with pytest.raises(SystemExit) as error:
        main(["run", "98.1"])
    assert error.value.code == 2


def test_reproducibility_envelope(tmp_path, monkeypatch):
    monkeypatch.setattr("convex_solutions.cli.load_solution",
                        lambda eid: lambda output_dir, seed: {"seed_copy": seed, "answer": 42})
    result = run_exercise("99.1", tmp_path, 7)
    assert result["seed"] == 7
    assert result["results"]["seed_copy"] == 7
    assert result["environment"]["packages"]["cvxpy"]
    assert (tmp_path / "results.json").exists()
