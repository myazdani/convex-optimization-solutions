"""Independent correctness checks for every chapter 10 numerical exercise."""
import numpy as np
import pytest
from scipy.linalg import solve
from scipy.optimize._numdiff import approx_derivative

from convex_solutions.chapters.ch10.common import (
    entropy_data, entropy_newton, entropy_dual_newton,
    game_data, game_derivatives, game_value, game_newton, game_best_responses,
    solve_log_investment,
)
from convex_solutions.chapters.ch10.ex_10_14 import solve as exercise_10_14
from convex_solutions.chapters.ch10.ex_10_15 import solve as exercise_10_15
from convex_solutions.chapters.ch10.ex_10_16 import solve as exercise_10_16


def test_10_14_budget_domain_stationarity_and_reference():
    result = exercise_10_14()
    x = np.array(result["newton"]["x"])
    np.testing.assert_allclose(x, [.49725104854, .19936802214, .30338092932], atol=1e-9)
    assert abs(x.sum() - 1) < 1e-12
    assert min(result["scenario_returns"]) > 0
    assert result["newton"]["residuals"][-1] < 1e-10
    assert abs(result["newton"]["multiplier"][0] - 1) < 1e-10
    assert result["reference"]["objective_error"] < 1e-8
    assert result["reference"]["point_error"] < 1e-4
    assert result["growth_rate"] == pytest.approx(.0622715794965, abs=1e-11)


def test_10_14_short_start_is_allowed_and_objective_descends():
    result = solve_log_investment(x0=np.array([-.1, .1, 1.]))
    assert result.residuals[-1] < 1e-10
    assert np.max(np.diff(result.objectives)) < 1e-12
    np.testing.assert_allclose(result.x, solve_log_investment().x, atol=1e-9)


def test_10_15_full_size_all_subparts_and_cvxpy_comparison():
    result = exercise_10_15(seed=0)
    assert (result["n"], result["p"], result["rank_A"]) == (100, 30, 30)
    assert len(result["methods"]) == 4
    for method in result["methods"].values():
        assert method["residuals"][-1] < 1e-9
        assert min(method["x"]) > 0
        assert len(method["residuals"]) == method["iterations"] + 1
    for comparison in result["comparisons"].values():
        assert comparison["point_error_vs_dual"] < 1e-8
        assert comparison["multiplier_error_vs_dual"] < 1e-8
        assert comparison["objective_error_vs_cvxpy"] < 1e-7
        assert comparison["point_error_vs_cvxpy"] < 1e-3
        assert comparison["multiplier_error_vs_cvxpy"] < 1e-3


@pytest.mark.parametrize("seed", [3, 1015, 2026])
def test_10_15_multiple_random_instances_and_line_search(seed):
    A, b, xhat = entropy_data(seed, n=40, p=12)
    a = entropy_newton(A, b, xhat, "feasible")
    bf = entropy_newton(A, b, xhat, "infeasible")
    bi = entropy_newton(A, b, np.ones(40), "infeasible")
    c = entropy_dual_newton(A, b)
    assert np.max(np.diff(a.objectives)) < 1e-12
    for result in (a, bf, bi, c):
        np.testing.assert_allclose(result.x, c.x, atol=2e-9)
        np.testing.assert_allclose(result.multiplier, c.multiplier, atol=2e-9)
        assert np.linalg.norm(A @ result.x - b) < 1e-9
        assert np.linalg.norm(np.log(result.x) + 1 + A.T @ result.multiplier) < 1e-9
    for result in (bf, bi):
        old, new, steps = np.array(result.residuals[:-1]), np.array(result.residuals[1:]), np.array(result.step_lengths)
        assert np.all(new <= (1-.01*steps)*old + 1e-13)


def test_10_15_structured_kkt_direction_matches_full_solve():
    A, b, _ = entropy_data(8, n=20, p=6)
    x, nu = np.linspace(.3, 1.2, 20), np.linspace(-.1, .2, 6)
    rd, rp = np.log(x)+1+A.T@nu, A@x-b
    M = (A*x)@A.T
    dnu = solve(M, rp-A@(x*rd), assume_a="pos")
    dx = -x*(rd+A.T@dnu)
    K = np.block([[np.diag(1/x), A.T], [A, np.zeros((6, 6))]])
    full = solve(K, -np.r_[rd, rp], assume_a="sym")
    np.testing.assert_allclose(np.r_[dx, dnu], full, atol=1e-12)


def test_10_15_dual_gradient_and_hessian_by_finite_differences():
    A, b, _ = entropy_data(2, n=20, p=6)
    nu = np.linspace(-.1, .1, 6)
    objective = lambda n: np.exp(-1-A.T@n).sum()+b@n
    gradient = lambda n: b-A@np.exp(-1-A.T@n)
    x = np.exp(-1-A.T@nu)
    np.testing.assert_allclose(approx_derivative(objective, nu).ravel(), gradient(nu), atol=2e-9)
    np.testing.assert_allclose(approx_derivative(gradient, nu), (A*x)@A.T, atol=2e-9)


def test_10_16_gradient_and_hessian_by_finite_differences():
    A, b, c = game_data(seed=9, p=5, q=4)
    z = np.linspace(-.13, .17, 9)
    objective = lambda w: game_value(A, b, c, w[:5], w[5:])
    residual = lambda w: game_derivatives(A, b, c, w[:5], w[5:])[0]
    r, J = game_derivatives(A, b, c, z[:5], z[5:])
    np.testing.assert_allclose(approx_derivative(objective, z).ravel(), r, atol=2e-8)
    np.testing.assert_allclose(approx_derivative(residual, z), J, atol=2e-8)
    assert np.min(np.linalg.eigvalsh(J[:5, :5])) >= 2
    assert np.max(np.linalg.eigvalsh(J[5:, 5:])) <= -2


def test_10_16_full_size_all_nine_experiments_and_independent_reference():
    result = exercise_10_16(seed=0)
    assert (result["p"], result["q"]) == (100, 100)
    assert len(result["experiments"]) == 9
    for experiment in result["experiments"].values():
        run = experiment["newton"]
        assert run["residuals"][-1] < 1e-9
        assert max(experiment["norm_u"], experiment["norm_v"]) < 1
        assert experiment["point_error_vs_scipy"] < 1e-8
        assert experiment["best_response_error"] < 1e-9
        assert abs(experiment["saddle_gap"]) < 1e-9
        old, new, steps = np.array(run["residuals"][:-1]), np.array(run["residuals"][1:]), np.array(run["step_lengths"])
        assert np.all(new <= (1-experiment["alpha"]*steps)*old + 1e-12)
        assert np.all((steps > 0) & (steps <= 1))


def test_10_16_saddle_property_under_independent_strategy_perturbations():
    A, b, c = game_data(seed=15, p=8, q=6)
    result = game_newton(A, b, c)
    u, v = result.x[:8], result.x[8:]
    ubr, vbr = game_best_responses(A, b, c, u, v)
    np.testing.assert_allclose(u, ubr, atol=1e-10)
    np.testing.assert_allclose(v, vbr, atol=1e-10)
    rng = np.random.default_rng(33)
    for _ in range(20):
        ut, vt = rng.normal(size=8), rng.normal(size=6)
        ut *= .9/np.linalg.norm(ut); vt *= .9/np.linalg.norm(vt)
        assert game_value(A, b, c, u, vt) <= result.objective + 1e-12
        assert result.objective <= game_value(A, b, c, ut, v) + 1e-12


def test_invalid_starting_points_and_parameters_raise():
    A, b, xhat = entropy_data(0, n=10, p=3)
    with pytest.raises(ValueError, match="positive"):
        entropy_newton(A, b, -xhat)
    with pytest.raises(ValueError, match="satisfy"):
        entropy_newton(A, b, np.ones(10), "feasible")
    with pytest.raises(ValueError, match="alpha"):
        solve_log_investment(alpha=.7)
    Ag, bg, cg = game_data(0, 4, 3)
    with pytest.raises(ValueError, match="unit balls"):
        game_newton(Ag, bg, cg, np.ones(4), np.zeros(3))


def test_artifact_generation(tmp_path):
    result = exercise_10_14(tmp_path)
    assert (tmp_path/"10_14_newton.png").stat().st_size > 1000
    assert (tmp_path/"exercise_10_14.json").stat().st_size > 1000
    assert result["plots"] == ["10_14_newton.png"]
