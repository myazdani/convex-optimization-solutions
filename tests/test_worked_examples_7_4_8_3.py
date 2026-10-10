"""Exact-data worked examples, independent certificates, and rendered real results."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_file(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


binary = load_file("worked_7_4", "examples/ex_7_4.py")
correlation = load_file("worked_8_3", "examples/ex_8_3.py")
renderer = load_file("worked_results", "scripts/render_example_results.py")


@pytest.fixture(scope="module")
def binary_result():
    return binary.solve()


@pytest.fixture(scope="module")
def correlation_result():
    return correlation.solve()


def test_binary_printed_data_and_probability_convention():
    np.testing.assert_array_equal(binary.P, [[.70, .10], [.20, .10], [.05, .70], [.05, .10]])
    np.testing.assert_allclose(binary.P.sum(axis=0), 1, atol=1e-15)
    for rule in binary.deterministic_detectors()["positive_probabilities"]:
        detection = binary.detector(rule) @ binary.P
        np.testing.assert_allclose(detection.sum(axis=0), 1, atol=1e-15)
        np.testing.assert_allclose(binary.errors(rule), [detection[1, 0], detection[0, 1]], atol=1e-15)


def test_binary_threshold_vertices_and_convex_slopes():
    v = binary.threshold_vertices()
    assert v["observation_order_one_based"] == [3, 4, 2, 1]
    np.testing.assert_allclose(v["error_pairs"], [[0, 1], [.05, .30], [.10, .20], [.30, .10], [1, 0]], atol=1e-15)
    slopes = np.diff(v["error_pairs"][:, 1]) / np.diff(v["error_pairs"][:, 0])
    assert np.all(np.diff(slopes) > 0)
    assert np.all(np.diff(v["positive_probabilities"], axis=0).sum(axis=1) == 1)


def test_binary_exact_primal_and_weighted_lower_bound():
    exact = binary.exact_certificate()
    np.testing.assert_allclose(exact["error_pair"], [1 / 6, 1 / 6], atol=1e-15)
    assert exact["weighted_error_lower_bound"] == pytest.approx(1 / 6, abs=1e-15)
    assert abs(exact["gap"]) < 1e-15
    points = binary.deterministic_detectors()["error_pairs"]
    assert np.min(points @ exact["error_weights"]) == pytest.approx(1 / 6, abs=1e-15)
    assert np.all(points.max(axis=1) >= points @ exact["error_weights"] - 1e-15)


def test_binary_exhaustive_deterministic_minimax():
    enumeration = binary.deterministic_detectors()
    assert enumeration["count"] == 16
    assert len(np.unique(enumeration["positive_probabilities"], axis=0)) == 16
    assert enumeration["minimax_error"] == pytest.approx(.2, abs=1e-15)
    assert len(enumeration["minimax_rule_indices"]) == 1
    np.testing.assert_array_equal(enumeration["positive_probabilities"][enumeration["minimax_rule_indices"][0]], [0, 0, 1, 1])
    assert enumeration["minimax_error"] - 1 / 6 == pytest.approx(1 / 30, abs=1e-15)


def test_binary_numerical_minimax_and_dual(binary_result):
    z = binary_result["minimax"]
    assert z["status"] == "optimal"
    assert abs(z["objective"] - 1 / 6) < 1e-8
    for key in ["primal_violation", "stationarity_residual", "complementarity_residual", "point_error"]:
        assert z[key] < 1e-8
    assert abs(z["primal_dual_gap"]) < 1e-8
    np.testing.assert_allclose(z["dual_error_weights"], [1 / 3, 2 / 3], atol=1e-8)
    np.testing.assert_allclose(z["positive_probabilities"], binary.BOOK_MINIMAX_POSITIVE, atol=1e-8)


def test_binary_independent_solver_and_full_frontier(binary_result):
    assert binary_result["checks"]["independent_solver_error"] < 1e-8
    f = binary_result["frontier"]
    assert f["point_count"] >= 101
    assert f["maximum_interpolation_error"] < 1e-8
    assert f["maximum_primal_violation"] < 1e-8
    assert np.all(np.asarray(f["error_pairs"])[:, 0] <= np.asarray(f["budgets"]) + 1e-8)
    for vertex in binary_result["vertices"]["error_pairs"]:
        assert np.min(np.abs(np.asarray(f["budgets"]) - vertex[0])) < 1e-15


@pytest.mark.parametrize("budgets", [np.array([]), np.array([-.1]), np.array([1.1]), np.array([np.nan]), np.array([[.5]])])
def test_binary_rejects_invalid_frontier_budgets(budgets):
    with pytest.raises(ValueError, match="budgets"):
        binary.trace_frontier(budgets)


def test_correlation_printed_data_and_analytic_endpoints():
    assert correlation.BOUNDS == [(0, 1, .6, .9), (0, 2, .8, .9), (1, 3, .5, .7), (2, 3, -.8, -.4)]
    exact = correlation.analytic_bounds()
    assert exact["lower"] == pytest.approx(.3 - np.sqrt(.48), abs=1e-15)
    assert exact["upper"] == pytest.approx(-.32 + np.sqrt(.3024), abs=1e-15)
    assert round(exact["lower"], 2) == -.39
    assert round(exact["upper"], 2) == .23


@pytest.mark.parametrize("label", ["lower", "upper"])
def test_correlation_gram_witness_attains_sharp_bound(label):
    exact = correlation.analytic_bounds()
    w = exact["witnesses"][label]
    vectors, matrix = np.asarray(w["unit_vectors_as_rows"]), np.asarray(w["matrix"])
    np.testing.assert_allclose(np.sum(vectors ** 2, axis=1), 1, atol=5e-15)
    np.testing.assert_allclose(matrix, vectors @ vectors.T, atol=5e-15)
    assert matrix[0, 3] == pytest.approx(exact[label], abs=5e-15)
    for key in ["psd_violation", "diagonal_residual", "symmetry_residual", "bound_violation"]:
        assert w[key] < 5e-15


@pytest.mark.parametrize("label", ["lower", "upper"])
def test_correlation_primal_dual_and_independent_backend(label, correlation_result):
    r = correlation_result
    for z in [r[label], r["independent_reference"][label]]:
        assert z["status"] == "optimal"
        assert abs(z["rho14"] - r["analytic_certificate"][label]) < 2e-6
        for key in ["primal_violation", "dual_violation", "stationarity_residual", "complementarity_residual",
                    "psd_violation", "bound_violation", "diagonal_residual", "symmetry_residual"]:
            assert z[key] < 2e-6
        assert abs(z["primal_dual_gap"]) < 2e-6
    assert r["checks"][f"{label}_solver_difference"] < 2e-6
    assert r["checks"][f"{label}_analytic_error"] < 1e-7
    sign = 1 if label == "lower" else -1
    assert r[label]["objective"] == pytest.approx(sign * r[label]["rho14"], abs=1e-15)


@pytest.mark.parametrize("label", ["lower", "upper"])
def test_correlation_optimizer_is_not_unique(label, correlation_result):
    numerical = np.asarray(correlation_result[label]["matrix"])
    witness = np.asarray(correlation_result["analytic_certificate"]["witnesses"][label]["matrix"])
    assert np.linalg.norm(numerical - witness) > .01
    midpoint = (numerical + witness) / 2
    assert midpoint[0, 3] == pytest.approx(correlation_result["analytic_certificate"][label], abs=1e-7)
    checks = correlation.feasibility(midpoint)
    for key in ["psd_violation", "diagonal_residual", "symmetry_residual", "bound_violation"]:
        assert checks[key] < 1e-7


def test_correlation_solver_argument_validation():
    with pytest.raises(ValueError, match="direction"):
        correlation.solve_bound("sideways")
    with pytest.raises(ValueError, match="solver"):
        correlation.solve_bound("lower", "unknown")


@pytest.mark.parametrize("example_id", ["7.4", "8.3"])
def test_saved_output_code_fingerprint_and_generated_markdown(example_id):
    envelope = renderer.load_envelope(example_id)
    assert envelope["results"]["data"]["kind"] == "exact printed textbook data"
    assert envelope["results"]["randomness"] == "none; seed unused"
    assert envelope["duration_seconds"] > 0
    assert envelope["environment"]["packages"]["cvxpy"]
    document = (ROOT / "docs" / "examples" / f"{example_id}.md").read_text()
    block = renderer.render_block(example_id, envelope)
    assert renderer.replace_block(document, block) == document
    code = ROOT / envelope["implementation_path"]
    assert hashlib.sha256(code.read_bytes()).hexdigest() == envelope["implementation_sha256"]
    for path in envelope["results"]["plots"]:
        image_path = ROOT / path
        assert image_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert image_path.stat().st_size > 10000


@pytest.mark.parametrize("example_id", ["7.4", "8.3"])
def test_standalone_example_cli(example_id, tmp_path):
    out = tmp_path / example_id
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1")
    outcome = subprocess.run([sys.executable, str(ROOT / "examples" / f"ex_{example_id.replace('.', '_')}.py"),
                              "--output", str(out), "--seed", "19"], capture_output=True, text=True,
                             env=env, cwd=ROOT, timeout=45)
    assert outcome.returncode == 0, outcome.stderr
    envelope = json.loads((out / "results.json").read_text())
    assert envelope["example_id"] == example_id
    assert envelope["seed"] == 19
    assert envelope["results"]["seed"] == 19
    assert envelope["results"]["plots"]
    assert (out / f"{example_id}.json").is_file()
    assert "verified and wrote" in outcome.stdout
