import numpy as np
import pytest
from convex_solutions.chapters.ch09.ex_9_10 import solve, log_cosh_value, pure_newton_log_barrier


def test_prescribed_pure_newton_runs():
    result = solve()
    convergent, divergent = result['part_a']
    assert convergent['status'] == 'converged'
    assert abs(convergent['iterates'][-1]) < 1e-12
    assert convergent['objectives'][-1] == pytest.approx(np.log(2))
    assert divergent['status'] == 'divergent_overflow_guard'
    assert np.all(np.diff(np.abs(divergent['iterates'])) > 0)
    assert abs(divergent['iterates'][-1]) > 20000
    assert result['part_b']['invalid_candidate'] == -3
    assert result['part_b']['status'] == 'left_domain'
    assert result['nonzero_two_cycle_threshold'] == pytest.approx(1.08865949248)


def test_stable_function_and_invalid_domain():
    assert np.isfinite(log_cosh_value(1000))
    assert log_cosh_value(1000) == 1000
    with pytest.raises(ValueError):
        pure_newton_log_barrier(0)
