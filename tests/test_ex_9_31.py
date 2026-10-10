import numpy as np
import pytest
from convex_solutions.chapters.ch09.ex_9_30 import make_instance
from convex_solutions.chapters.ch09.ex_9_31 import hessian_diagonal, minimize, run_experiment


def test_diagonal_formula():
    p = make_instance(30, 10, 930)
    x = np.full(10, 0.01)
    assert np.allclose(hessian_diagonal(p, x), np.diag(p.hessian(x)))


@pytest.mark.parametrize('method,period', [('gradient',1),('newton',1),('diagonal',1),('reuse',2),('reuse',5),('reuse',10)])
def test_approximate_methods_converge(method, period):
    p = make_instance(30, 10, 930)
    run = minimize(p, method, period=period)
    assert run['status'] == 'converged'
    assert run['gradient_norms'][-1] <= 1e-7
    assert min(run['margins']) > 0
    assert np.max(np.diff(run['objectives'])) <= 2e-13
    if method == 'reuse':
        assert run['factorizations'] == (run['iterations'] + period - 1) // period
    if method == 'diagonal':
        assert run['factorizations'] == 0
        assert run['diagonal_evaluations'] == run['iterations']


def test_reuse_period_one_equals_newton():
    p = make_instance(30, 10, 930)
    exact = minimize(p, 'newton')
    reuse = minimize(p, 'reuse', period=1)
    assert np.array_equal(exact['iterates'], reuse['iterates'])


def test_experiment_has_both_parts_and_all_periods():
    result = run_experiment(seed=930, dimensions=((20,5),), periods=(2,5,10,20))
    assert len(result['instances'][0]['runs']) == 7
