import numpy as np
import pytest
from convex_solutions.chapters.ch09.ex_9_32 import QuadraticResidualProblem, make_instance, minimize


def test_convexity_and_derivatives():
    p = make_instance(20, 5, 932)
    x = np.linspace(-0.15, 0.15, p.n)
    h = 1e-6
    eye = h*np.eye(p.n)
    grad = np.array([(p.value(x+e)-p.value(x-e))/(2*h) for e in eye])
    hessian = np.column_stack([(p.gradient(x+e)-p.gradient(x-e))/(2*h) for e in eye])
    assert np.allclose(grad, p.gradient(x), atol=1e-7, rtol=1e-7)
    assert np.allclose(hessian, p.hessian(x), atol=1e-7, rtol=1e-7)
    assert np.min(p.residual_lower_bounds()) >= 0.25-1e-12
    mu = p.strong_convexity_bound()
    assert mu > 0
    assert np.linalg.eigvalsh(p.hessian(x)).min() >= mu
    d = -x/3
    assert p.difference(x,d) == pytest.approx(p.value(x+d)-p.value(x), abs=2e-14)


@pytest.mark.parametrize('method', ['gauss_newton','newton','gradient'])
def test_methods_agree_and_jacobian_stays_full_rank(method):
    p = make_instance(20, 5, 932)
    run = minimize(p, method)
    reference = minimize(p, 'newton', tolerance=1e-12)
    assert run['status'] == 'converged'
    assert run['gradient_norms'][-1] <= 1e-7
    assert min(run['smallest_jacobian_singular_values']) > 1e-6
    assert np.max(np.diff(run['objectives'])) <= 2e-13
    assert np.linalg.norm(np.array(run['x'])-reference['x']) < 1e-7
    gap = p.difference(np.array(reference['x']), np.array(run['x'])-reference['x'])
    assert gap >= -1e-26
    assert gap <= run['objective_error_upper_bound'] + 1e-25


def test_gauss_newton_linear_least_squares_equations():
    p = make_instance(20, 5, 11)
    x = np.full(5, 0.1)
    J, r = p.jacobian(x), p.residuals(x)
    d = np.linalg.lstsq(J,-r,rcond=None)[0]
    assert np.linalg.norm(J.T@(J@d+r)) < 1e-12
    assert p.gradient(x)@d < 0


def test_nonconvex_residual_instance_rejected():
    with pytest.raises(ValueError):
        QuadraticResidualProblem(np.eye(2)[None],np.array([[4.,0.]]))


def test_independent_conic_model():
    import cvxpy as cp
    p = make_instance(10, 3, 19)
    x = cp.Variable(3)
    centers = np.linalg.solve(p.A,p.b[...,None])[...,0]
    residual = cp.hstack([0.5*cp.quad_form(x+centers[i],p.A[i])+p.residual_lower_bounds()[i]
                          for i in range(len(p.b))])
    model = cp.Problem(cp.Minimize(0.5*cp.sum_squares(residual)))
    model.solve(solver='CLARABEL',tol_gap_abs=1e-8,tol_feas=1e-8,tol_gap_rel=1e-8)
    own = minimize(p, 'newton', tolerance=1e-11)
    assert model.status == cp.OPTIMAL
    assert model.value == pytest.approx(p.value(np.array(own['x'])),abs=1e-8)
    assert np.linalg.norm(x.value-own['x']) < 2e-5
    assert np.linalg.norm(p.gradient(x.value))**2 / (2*p.strong_convexity_bound()) < 1e-8
