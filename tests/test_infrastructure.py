import json
from pathlib import Path
import numpy as np
import pytest
from convex_solutions.io import json_safe, write_json
from convex_solutions.registry import module_name, audit_inventory
from convex_solutions.provenance import file_sha256, verify_data


def test_module_naming():
    assert module_name("9.30") == "convex_solutions.chapters.ch09.ex_9_30"
    with pytest.raises(ValueError):
        module_name("../../etc/passwd")


def test_json_numpy_and_no_nonfinite(tmp_path):
    target = tmp_path / "out.json"
    write_json(target, {"a": np.array([1, 2]), "b": np.float64(2.5)})
    assert json.loads(target.read_text()) == {"a": [1, 2], "b": 2.5}
    with pytest.raises(ValueError, match="Nonfinite"):
        json_safe(np.array([np.nan]))


def test_incomplete_inventory_never_claims_complete():
    audit = audit_inventory({"audit_status": "in_progress", "exercises": []})
    assert audit["complete"] is False


def test_duplicate_inventory_is_flagged():
    entry = {"id": "99.1", "chapter": 99, "status": "pending", "scope": "required"}
    audit = audit_inventory({"audit_status": "complete", "exercises": [entry, entry]})
    assert any("Duplicate" in issue for issue in audit["issues"])
    assert not audit["complete"]


def test_provenance_checksum(tmp_path):
    source = tmp_path / "data.txt"
    source.write_text("1 2 3\n")
    record = {"kind": "upstream", "id": "test", "source_url": "https://example.org",
              "source_title": "Test", "retrieved_at": "2026-10-10T00:00:00Z",
              "sha256": file_sha256(source), "rights": "Test fixture"}
    source.with_name("data.txt.provenance.json").write_text(json.dumps(record))
    assert verify_data(source) == record
    source.write_text("changed")
    with pytest.raises(ValueError, match="Checksum"):
        verify_data(source)


def test_solver_and_violation():
    import cvxpy as cp
    from convex_solutions.numerics import solve_cvxpy, max_violation
    x = cp.Variable(2)
    p = cp.Problem(cp.Minimize(cp.sum_squares(x - [1, 2])), [x >= 0])
    meta = solve_cvxpy(p)
    assert meta["status"] == "optimal"
    np.testing.assert_allclose(x.value, [1, 2], atol=1e-5)
    assert max_violation(p) < 1e-7
