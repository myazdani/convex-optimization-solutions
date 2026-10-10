# example 6 7: Degree-five polynomial fitting

**Exact-input reproduction.** Uses the printed instance or the named official numerical data. Numerical precision and drawing style can differ from the source.

[Textbook source](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=345) · [Original Python](../../examples/application_fitting.py) · [Executed JSON](../../artifacts/examples/example_6_7/results.json) · [Tests](../../tests/test_worked_applications.py)

## Model and verification

Build a degree-five Vandermonde design matrix on the 40 published pairs. Solve the least-squares fit by QR/SVD and the maximum-residual fit as a linear program. A separately assembled SciPy/HiGHS epigraph LP checks the latter; normal equations check the former. Both fitted curves and the original observations appear below.

## Inputs and source comparison

Forty data pairs are in polapprox.bin; CVX supplies a seeded generator.

- [Official reference](https://cvxopt.org/examples/book/polapprox.html)
- [Official reference](https://cvxopt.org/_downloads/ad5823961411da4165f3d7cb85b21f2c/polapprox.py)
- [Official reference](https://cvxr.com/cvx/examples/cvxbook/Ch06_approx_fitting/fig6_19.m)
- [polapprox.bin](https://cvxopt.org/_downloads/7e1f992be338a8c8dea5f5e165b529df/polapprox.bin), SHA-256 `bd181e2d598f6cb03bd8d1364fce8075e0dc18e3347cfaa820948ca35a345d69`

The implementation is original Python. Reused numeric data are attributed separately in `data/worked_examples`, including conversion hashes and the upstream GPL-3.0-or-later notice. No textbook scans or reference program bodies are redistributed.

## Executed results

Run at 2026-10-10T05:03:20.599061+00:00; Python 3.12.14; NumPy 2.3.5; CVXPY 1.9.3; elapsed 1.397 s.

| Quantity | Actual value |
| --- | --- |
| basis | degree-five monomials |
| least_squares_residual_l2 | 0.111743155085 |
| least_squares_residual_linf | 0.0375189131691 |
| minimax_residual_l2 | 0.119169517742 |
| minimax_residual_linf | 0.030156059105 |

## Numerical checks

| Check | Measured error | Acceptance limit |
| --- | --- | --- |
| minimax objective versus independent HiGHS LP | 2.81242e-12 | 1e-08 |
| LS normal equations | 2.94209e-15 | 1e-08 |

## Recomputed figures

![Degree-five polynomial fitting: fits.png](../../artifacts/examples/example_6_7/fits.png)

## Reproduce

```sh
OPENBLAS_NUM_THREADS=1 python examples/application_fitting.py example_6_7
python -m pytest tests/test_worked_applications.py -q
```
