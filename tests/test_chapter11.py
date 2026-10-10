"""Independent mathematical and numerical checks for every chapter 11 part."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from convex_solutions.chapters.ch11 import ex_11_1 as e1
from convex_solutions.chapters.ch11 import ex_11_21 as e21
from convex_solutions.chapters.ch11 import ex_11_22 as e22
from convex_solutions.chapters.ch11 import ex_11_23 as e23
from convex_solutions.chapters.ch11 import ex_11_24 as e24
from convex_solutions.chapters.ch11.common import symmetric_zero_diagonal_weights


def check_derivatives(oracle,x,t=2.3,step=1e-5):
    value,gradient,hessian=oracle(x,t)
    eye=np.eye(len(x));gfd=[];hfd=[]
    for direction in eye:
        plus=oracle(x+step*direction,t);minus=oracle(x-step*direction,t)
        gfd.append((plus[0]-minus[0])/(2*step))
        hfd.append((plus[1]-minus[1])/(2*step))
    assert np.isfinite(value)
    assert_allclose(gradient,gfd,rtol=2e-6,atol=2e-6)
    assert_allclose(hessian,np.array(hfd).T,rtol=2e-6,atol=2e-6)
    assert np.linalg.eigvalsh(hessian)[0]>0


def test_11_1_central_path():
    result=e1.solve()
    points=result['central_points']
    assert np.all((points>2)&(points<3))
    assert np.all(np.diff(points)<0)
    assert np.max(result['stationarity_residuals'])<1e-8
    assert np.all(result['objective_gaps']<=result['barrier_gap_bounds'])


def test_11_1_rejects_bad_parameter():
    with pytest.raises(ValueError):e1.central_point(0)


def test_11_21_oracle_derivatives():
    rng=np.random.default_rng(9);A=rng.uniform(.1,1,(7,3));x=np.ones(3)
    b=A@x;lower=np.full(3,.1);upper=np.full(3,2.)
    check_derivatives(e21.make_oracle(A,b,lower,upper),np.r_[x,2.])


@pytest.mark.parametrize('seed',[0,17])
def test_11_21_barrier_matches_independent_solver(seed):
    for a in e21.solve(seed=seed)['instances']:
        assert a['primal_violation']<1e-12
        assert abs(a['objective_difference'])<2e-6
        assert a['stationarity_residual']<1e-4
        assert a['certified_lower_bound']<=a['reference']['rho']+2e-8
        assert 0<a['certified_objective_gap']<5e-6
        assert a['fractional_deviation']<=a['rho']
        assert_allclose(a['complementarity_gap'],a['central_path_gap_bound'])


def test_11_21_requires_strict_initial_point():
    with pytest.raises(ValueError):
        e21.solve_fractional_chebyshev(np.eye(2),np.ones(2),np.zeros(2),np.ones(2),np.zeros(2))


def test_11_22_oracle_derivatives():
    check_derivatives(e22.make_oracle(e22.BOOK_A,e22.BOOK_B),np.array([0.,0.,.03,.04]))


def test_11_22_analytic_rectangles_and_independent_solver():
    results=e22.solve()['instances']
    for a in results:
        assert a['primal_violation']<1e-12
        assert abs(a['log_volume_difference'])<2e-6
        assert a['stationarity_residual']<1e-4
        assert 0<a['certified_objective_gap']<2e-6
        lam=a['dual_feasible_multipliers']
        assert np.min(lam)>=0
        assert_allclose(a['A'].T@lam,0,atol=1e-12)
    assert_allclose(results[0]['volume'],24,rtol=2e-6)
    assert_allclose(results[1]['volume'],2.25,rtol=2e-6)
    assert_allclose(results[2]['lower'],[0,-1/16],atol=2e-6)
    assert_allclose(results[2]['upper'],[3/8,1/4],atol=2e-6)
    assert_allclose(results[2]['volume'],15/128,rtol=2e-6)
    assert e22.polygon_vertices(e22.BOOK_A,e22.BOOK_B).shape==(4,2)


def test_11_23_oracle_derivatives():
    rng=np.random.default_rng(2);W=symmetric_zero_diagonal_weights(5,rng)
    check_derivatives(e23.make_oracle(W),np.full(5,3.))


def test_11_23_all_subparts_and_large_instance():
    result=e23.solve(seed=0)
    assert [a['n'] for a in result['instances']]==[6,8,10,100]
    for a in result['instances']:
        s=a['sdp']
        assert s['dual_cone_min_eigenvalue']>0
        assert s['primal_cone_min_eigenvalue']>-1e-10
        assert s['diagonal_residual']<1e-12
        assert 0<s['certified_gap']<=1.001*s['central_path_gap_bound']
        assert abs(s['dual_lower_bound']-a['reference']['objective'])<2e-3
        if a['exact'] is not None:
            assert s['dual_lower_bound']<=a['exact']['value']+1e-9
            assert a['exact']['value']<=a['randomized']['best_value']+1e-9
        assert s['dual_lower_bound']<=a['spectral']['value']+1e-8
        assert np.all(a['randomized']['refined_values']<=a['randomized']['raw_values']+1e-9)
        for greedy in [a['all_ones']['greedy'],a['spectral']['greedy'],
                       a['randomized']['best_over_all_refined_samples']]:
            assert np.all(np.diff(greedy['history'])<=1e-10)
            assert greedy['best_remaining_flip']>=-1e-10
            assert np.all(np.abs(greedy['x'])==1)


def test_11_23_greedy_flip_formula():
    rng=np.random.default_rng(20);W=symmetric_zero_diagonal_weights(9,rng)
    x=e23.sign_partition(rng.standard_normal(9));deltas=-4*x*(W@x)
    for i in range(len(x)):
        y=x.copy();y[i]*=-1
        assert_allclose(e23.partition_cost(W,y)-e23.partition_cost(W,x),deltas[i],atol=1e-12)


def test_11_23_exhaustive_symmetry_and_covariance():
    W=np.array([[0.,1],[1,0.]])
    exact=e23.exhaustive_partition(W)
    assert exact['value']==-2
    assert exact['partitions_evaluated']==2
    samples=e23.randomized_partitions(np.ones((4,4)),200,np.random.default_rng(0))
    assert np.all(samples==samples[:,[0]])


def test_11_24_oracle_derivatives():
    args=e24.generate_instance(4,14,np.random.default_rng(31),conditioning=20)
    check_derivatives(e24.make_oracle(*args[:4]),args[4])


@pytest.mark.parametrize('seed',[0,21])
def test_11_24_both_methods_and_independent_solver(seed):
    for a in e24.solve(seed=seed)['instances']:
        for method in ['barrier','primal_dual']:
            r=a[method]
            assert r['primal_violation']<1e-12
            assert r['minimum_multiplier']>0
            assert r['duality_gap']>=-1e-10
            assert r['duality_gap']<2e-7
            assert r['dual_residual_norm']<1e-4
            assert abs(a[method+'_objective_difference'])<2e-7
            assert r['dual_objective']<=a['reference']['objective']+2e-8
        assert a['primal_dual']['dual_residual_norm']<1e-7


def test_11_24_singular_hessian_qp():
    P=np.diag([1.,0]);q=np.array([-2.,.5]);A=np.vstack([np.eye(2),-np.eye(2)])
    b=np.ones(4);x0=np.zeros(2)
    for solver in [e24.solve_barrier_qp,e24.solve_primal_dual_qp]:
        answer=solver(P,q,A,b,x0)
        assert_allclose(answer['x'],[1,-1],atol=2e-6)
        assert answer['duality_gap'] is None  # No false finite dual for singular P.
        assert answer['dual_residual_norm']<1e-5
        assert answer['surrogate_gap']<2e-7


def test_11_24_rejects_indefinite_or_infeasible_inputs():
    P=np.eye(2);q=np.ones(2);A=np.vstack([np.eye(2),-np.eye(2)]);b=np.ones(4)
    with pytest.raises(ValueError):e24.solve_barrier_qp(-P,q,A,b,np.zeros(2))
    with pytest.raises(ValueError):e24.solve_primal_dual_qp(P,q,A,b,np.ones(2))


def test_11_24_redundant_flat_direction_uses_minimum_norm_newton():
    P=np.diag([1.,0]);q=np.array([-2.,0]);A=np.array([[1.,0],[-1.,0]])
    for solver in [e24.solve_barrier_qp,e24.solve_primal_dual_qp]:
        answer=solver(P,q,A,np.ones(2),np.array([0.,7.]))
        assert_allclose(answer['x'],[1,7],atol=2e-6)
        assert answer['dual_residual_norm']<1e-5
