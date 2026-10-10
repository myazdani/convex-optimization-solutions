import numpy as np
import pytest
from convex_solutions.chapters.ch09.ex_9_30 import (
    AnalyticCenter, backtracking, convergence_diagnostics, gradient_descent,
    make_instance, newton_method,
)


def finite_gradient(fun, x, step=1e-6):
    eye = np.eye(len(x)) * step
    return np.array([(fun(x + e) - fun(x - e)) / (2 * step) for e in eye])


def test_gradient_hessian_and_strong_convexity():
    p = make_instance(20, 5, seed=1)
    x = np.full(5, 0.02)
    assert np.allclose(p.gradient(x), finite_gradient(p.value, x), rtol=2e-8, atol=2e-8)
    numerical_hessian = np.column_stack([finite_gradient(lambda z: p.gradient(z)[j], x) for j in range(5)])
    assert np.allclose(p.hessian(x), numerical_hessian, rtol=2e-8, atol=2e-8)
    assert np.linalg.eigvalsh(p.hessian(x)).min() >= 2
    d = np.full(5, -0.01)
    assert p.difference(x, d) == pytest.approx(p.value(x+d)-p.value(x), abs=1e-13)


@pytest.mark.parametrize('solver', [gradient_descent, newton_method])
def test_descent_domain_and_stopping(solver):
    p = make_instance(30, 10, seed=930)
    r = solver(p)
    assert r.converged
    assert r.iterations + 1 == len(r.iterates)
    assert np.all(r.margins > 0)
    assert np.all(np.diff(r.objectives) <= 2e-13)
    assert np.all((0 < r.step_lengths) & (r.step_lengths <= 1))
    assert np.allclose(r.step_lengths, r.beta ** r.backtracks)
    if r.method == 'gradient':
        assert r.gradient_norms[-1] <= r.tolerance
    else:
        assert r.decrement_squared[-1] / 2 <= r.tolerance
    reference = newton_method(p, tolerance=1e-25)
    gap = p.difference(reference.x, r.x-reference.x)
    assert gap >= -1e-25
    assert gap <= r.gradient_norms[-1] ** 2 / 4 + 1e-24


def test_quadratic_convergence_before_roundoff():
    p = make_instance(100, 30, seed=931)
    reference = newton_method(p, tolerance=1e-25)
    r = newton_method(p, tolerance=1e-20)
    samples = convergence_diagnostics(r, reference.x)['local_samples']
    orders = [s['observed_order'] for s in samples if 'observed_order' in s]
    assert len(orders) >= 2
    assert 1.9 < orders[-1] < 2.1
    assert r.step_lengths[-1] == 1


def test_independent_conic_solver_agreement():
    import cvxpy as cp
    p = make_instance(20, 5, seed=71)
    x = cp.Variable(p.n)
    objective = -cp.sum(cp.log(1-p.A@x))-cp.sum(cp.log(1-x))-cp.sum(cp.log(1+x))
    model = cp.Problem(cp.Minimize(objective))
    model.solve(solver='CLARABEL', tol_gap_abs=1e-10, tol_feas=1e-10, tol_gap_rel=1e-10)
    own = newton_method(p, tolerance=1e-22)
    assert model.status == cp.OPTIMAL
    assert model.value == pytest.approx(p.value(own.x), abs=2e-8)
    assert np.linalg.norm(x.value-own.x) < 2e-5


def test_validation_and_nonconvergence_status():
    p = make_instance(12, 4, seed=1)
    assert p.value(np.ones(4)) == float('inf')
    with pytest.raises(ValueError):
        p.gradient(np.ones(4))
    with pytest.raises(ValueError):
        newton_method(p, x0=np.ones(4))
    with pytest.raises(ValueError):
        gradient_descent(p, alpha=0.8)
    with pytest.raises(ValueError):
        backtracking(p, np.zeros(4), p.gradient(np.zeros(4)))
    result = gradient_descent(p, max_iterations=0)
    assert not result.converged and result.termination == 'maximum_iterations'
    assert result.iterations == 0
    assert np.array_equal(make_instance(12,4,1).A, p.A)
    with pytest.raises(ValueError):
        AnalyticCenter(np.array([[np.nan]]))
