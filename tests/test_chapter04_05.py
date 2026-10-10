"""Exact certificates and independent regression checks for the four extras."""
import json

import numpy as np
import pytest

from convex_solutions.chapters.ch04 import ex_4_3 as box
from convex_solutions.chapters.ch05 import ex_5_29 as sphere
from convex_solutions.chapters.ch05 import ex_5_33 as l1
from convex_solutions.chapters.ch05 import ex_5_34 as lp
from convex_solutions.io import json_safe


def test_box_exact_certificate_and_uniqueness():
    result = box.certificate()
    assert result["objective"] == -173 / 8
    assert result["stationarity_residual"] == 0
    assert result["complementarity_residual"] == 0
    assert result["primal_violation"] == 0
    for k, determinant in enumerate([13, 77, 100], start=1):
        assert np.linalg.det(box.P[:k, :k]) == pytest.approx(determinant)
    assert np.linalg.eigvalsh(box.P)[0] > 0
    rng = np.random.default_rng(43)
    for x in rng.uniform(-1, 1, (25, 3)):
        delta = x - box.CANDIDATE
        expansion = (box.P @ box.CANDIDATE + box.Q) @ delta + .5 * delta @ box.P @ delta
        assert expansion > 0
        assert box.objective(x) - result["objective"] == pytest.approx(expansion)


def test_box_independent_qp():
    result = box.solve()
    assert result["reference"]["point_error"] < 2e-5
    assert result["reference"]["objective_error"] < 1e-6
    assert result["reference"]["stationarity_residual"] < 1e-6


def test_sphere_enumeration_is_complete():
    points, intervals = sphere.stationary_points()
    assert [i["root_count"] for i in intervals] == [1, 0, 2, 1]
    assert intervals[1]["minimum_value"] > 7
    assert intervals[2]["minimum_value"] < -.39
    assert sphere.secular(1.) == pytest.approx(-7 / 18)
    assert len(points) == 4
    assert [p["classification"] for p in points] == [
        "strict local maximum", "saddle", "strict local minimum", "strict local minimum"]
    expected = [-3.14928980661826, .223508996482819, 1.89189898417542, 4.03522594618260]
    np.testing.assert_allclose([p["nu"] for p in points], expected, atol=2e-12)
    for p in points:
        assert p["sphere_residual"] < 1e-11
        assert p["stationarity_residual"] < 1e-12
    # At each pole the corresponding equation would be 0*x_i = -1.
    for pole in sphere.POLES:
        assert np.any(sphere.DIAGONAL + pole == 0)


def test_sphere_polynomial_and_global_certificate():
    points, _ = sphere.stationary_points()
    roots = np.polynomial.polynomial.polyroots(sphere.POLYNOMIAL)
    real = sorted(r.real for r in roots if abs(r.imag) < 1e-9)
    np.testing.assert_allclose(real, [p["nu"] for p in points], atol=1e-10)
    assert [p["global_minimum_certificate"] for p in points] == [False, False, False, True]
    optimum = points[-1]
    assert optimum["objective"] == pytest.approx(-5.365493342948376, abs=1e-11)
    assert abs(optimum["objective"] - optimum["dual_value"]) < 1e-11
    assert min(optimum["shifted_hessian_eigenvalues"]) > 0
    assert points[2]["dual_value"] is None
    assert min(points[2]["tangent_hessian_eigenvalues"]) > 0
    assert min(points[2]["shifted_hessian_eigenvalues"]) < 0


def test_sphere_independent_sdp():
    result = sphere.solve()
    assert result["reference"]["objective_error"] < 2e-6
    assert result["reference"]["rank_one_residual"] < 2e-5
    assert result["reference"]["psd_violation"] < 2e-6
    assert result["reference"]["trace_residual"] < 1e-7
    assert result["reference"]["point_error"] < 1e-3


def test_l1_exact_dual_certificate_and_active_system():
    np.testing.assert_array_equal(l1.A.T @ l1.DUAL, np.zeros(3))
    np.testing.assert_array_equal(l1.A @ l1.X0 + l1.B, [2, 0, 0, -1, 0, 1])
    np.testing.assert_array_equal(l1.A @ l1.X_DIRECTION + l1.D, [10, 0, 0, 1, 0, -2])
    assert np.max(np.abs(l1.DUAL[l1.ZERO_ROWS])) == .5
    assert np.linalg.det(l1.A[l1.ZERO_ROWS]) == pytest.approx(-344)
    assert l1.DUAL @ l1.B == 4
    assert l1.DUAL @ l1.D == 7
    assert l1.sensitivity_interval() == (-.2, .5)


@pytest.mark.parametrize("epsilon", [-.2, -.1, 0, .2, .4, .5])
def test_l1_sensitivity_against_independent_lps(epsilon):
    exact, reference = l1.analytic_solution(epsilon), l1.lp_reference(epsilon)
    assert exact["objective"] == pytest.approx(4 + 7 * epsilon)
    assert exact["duality_gap"] < 1e-12
    assert reference["primal_dual_gap"] < 1e-8
    assert reference["primal_violation"] < 1e-8
    assert reference["dual_equality_residual"] < 1e-8
    assert exact["objective"] == pytest.approx(reference["objective"], abs=1e-8)
    np.testing.assert_allclose(exact["x"], reference["x"], atol=1e-8)
    assert exact["unique"]


def test_l1_formula_interval_is_maximal():
    for epsilon in [-.21, .51]:
        with pytest.raises(ValueError):
            l1.analytic_solution(epsilon)
        reference = l1.lp_reference(epsilon)
        assert reference["objective"] > 4 + 7 * epsilon + 1e-4


def test_lp_exact_primal_dual_identities():
    np.testing.assert_array_equal(lp.A.T @ lp.Z0 + lp.C, np.zeros(4))
    np.testing.assert_array_equal(lp.A.T @ lp.Z_DIRECTION + lp.D, np.zeros(4))
    np.testing.assert_array_equal(lp.B - lp.A @ lp.X0, [1, 0, 0, 0, 0])
    np.testing.assert_array_equal(lp.F - lp.A @ lp.X_DIRECTION, [-1, 0, 0, 0, 0])
    assert np.linalg.det(lp.A[lp.ACTIVE]) == pytest.approx(8029)
    assert lp.sensitivity_interval() == (-.5, 1.)
    assert lp.C @ lp.X0 == -40
    assert lp.D @ lp.X0 + lp.C @ lp.X_DIRECTION == -72
    assert lp.D @ lp.X_DIRECTION == 25
    assert lp.D @ lp.X0 - lp.F @ lp.Z0 == -72


@pytest.mark.parametrize("epsilon", [-.5, -.25, 0, .25, .75, 1.])
def test_lp_sensitivity_against_independent_lps(epsilon):
    exact, reference = lp.analytic_solution(epsilon), lp.lp_reference(epsilon)
    assert exact["objective"] == pytest.approx(-40 - 72 * epsilon + 25 * epsilon**2)
    for key in ["primal_violation", "dual_violation", "stationarity_residual",
                "complementarity_residual", "duality_gap"]:
        assert exact[key] < 1e-11
    assert reference["primal_dual_gap"] < 1e-8
    assert reference["primal_violation"] < 1e-8
    assert reference["stationarity_residual"] < 1e-8
    assert reference["complementarity_residual"] < 1e-8
    assert exact["objective"] == pytest.approx(reference["objective"], abs=1e-8)
    if exact["primal_unique"]:
        np.testing.assert_allclose(exact["x"], reference["x"], atol=1e-8)
    if exact["dual_unique"]:
        np.testing.assert_allclose(exact["z"], reference["z"], atol=1e-8)


def test_lp_endpoint_nonuniqueness_is_real():
    lower = lp.analytic_solution(-.5)
    x = np.array(lower["x"]) + 1e-6 * np.array([-2487, -5883, 1223, 1796])
    assert np.max(lp.A @ x - (lp.B - .5 * lp.F)) < 1e-10
    assert (lp.C - .5 * lp.D) @ x == pytest.approx(lower["objective"])
    upper = lp.analytic_solution(1.)
    z = np.array(upper["z"]) + 1e-6 * np.array([8029, -2739, -61298, -15502, -8487])
    assert np.min(z) >= 0
    np.testing.assert_allclose(lp.A.T @ z + lp.C + lp.D, np.zeros(4), atol=1e-11)
    assert -(lp.B + lp.F) @ z == pytest.approx(upper["objective"])
    assert not lower["primal_unique"] and lower["dual_unique"]
    assert upper["primal_unique"] and not upper["dual_unique"]


def test_lp_free_variables_outside_basis_interval():
    reference = lp.lp_reference(1.1)
    assert reference["x"][3] < -.09
    assert reference["objective"] == pytest.approx(-88.9080854187739, abs=1e-8)
    with pytest.raises(ValueError):
        lp.analytic_solution(1.1)


@pytest.mark.parametrize("module", [box, sphere, l1, lp])
def test_extra_artifacts_and_json_are_finite(module, tmp_path):
    result = module.solve(tmp_path, seed=3)
    encoded = json.dumps(json_safe(result), allow_nan=False)
    assert encoded
    saved = tmp_path / f"{result['exercise']}.json"
    assert saved.is_file()
    for plot in result["plots"]:
        assert (tmp_path / plot).stat().st_size > 1000
