# 2. Computational exercises

[Book index](../README.md) · [1. Examples](../examples/README.md) · [2. Computational exercises](../computational-exercises/README.md) · [3. All other exercises](../other-exercises/README.md)

These are the **16 already-delivered numerical exercise solutions**. All have original derivations, runnable Python code, executed result tables, and verified output.

Eleven exercises explicitly request numerical execution or implementation. The other five are numerical extensions of end-of-chapter exercises. All sixteen belong here; none is a worked example from the textbook narrative.

| Exercise | Topic | Textbook role | Status | Executed output |
|---|---|---|---|---|
| [4.3](../solutions/4.3.md) | Box-QP optimality certificate | Numerical extension | Verified | [Tables](../solutions/4.3.md#executed-results) · [JSON](../../artifacts/4.3/results.json) · [Book p. 189](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=203) |
| [5.29](../solutions/5.29.md) | Nonconvex sphere-QP stationary points | Numerical extension | Verified | [Tables](../solutions/5.29.md#executed-results) · [JSON](../../artifacts/5.29/results.json) · [Book p. 282](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=296) |
| [5.33](../solutions/5.33.md) | L1 fitting sensitivity certificate | Numerical extension | Verified | [Tables](../solutions/5.33.md#executed-results) · [JSON](../../artifacts/5.33/results.json) · [Book p. 283](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=297) |
| [5.34](../solutions/5.34.md) | LP primal-dual sensitivity | Numerical extension | Verified | [Tables](../solutions/5.34.md#executed-results) · [JSON](../../artifacts/5.34/results.json) · [Book p. 284](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=298) |
| [9.10](../solutions/9.10.md) | Undamped Newton behavior | Explicit computation | Verified | [Tables](../solutions/9.10.md#executed-results) · [JSON](../../artifacts/9.10/results.json) · [Book p. 515](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=529) |
| [9.30](../solutions/9.30.md) | Gradient versus Newton centering | Explicit computation | Verified | [Tables](../solutions/9.30.md#executed-results) · [JSON](../../artifacts/9.30/results.json) · [Book p. 519](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=533) |
| [9.31](../solutions/9.31.md) | Approximate Newton centering | Explicit computation | Verified | [Tables](../solutions/9.31.md#executed-results) · [JSON](../../artifacts/9.31/results.json) · [Book p. 519](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=533) |
| [9.32](../solutions/9.32.md) | Gauss–Newton for convex residuals | Explicit computation | Verified | [Tables](../solutions/9.32.md#executed-results) · [JSON](../../artifacts/9.32/results.json) · [Book p. 520](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=534) |
| [10.14](../solutions/10.14.md) | Log-growth portfolio Newton solve | Explicit computation | Verified | [Tables](../solutions/10.14.md#executed-results) · [JSON](../../artifacts/10.14/results.json) · [Book p. 559](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=573) |
| [10.15](../solutions/10.15.md) | Three entropy Newton formulations | Explicit computation | Verified | [Tables](../solutions/10.15.md#executed-results) · [JSON](../../artifacts/10.15/results.json) · [Book p. 560](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=574) |
| [10.16](../solutions/10.16.md) | Newton saddle-point game | Explicit computation | Verified | [Tables](../solutions/10.16.md#executed-results) · [JSON](../../artifacts/10.16/results.json) · [Book p. 560](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=574) |
| [11.1](../solutions/11.1.md) | Scalar barrier path visualization | Numerical extension | Verified | [Tables](../solutions/11.1.md#executed-results) · [JSON](../../artifacts/11.1/results.json) · [Book p. 623](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=637) |
| [11.21](../solutions/11.21.md) | Fractional Chebyshev barrier solver | Explicit computation | Verified | [Tables](../solutions/11.21.md#executed-results) · [JSON](../../artifacts/11.21/results.json) · [Book p. 629](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=643) |
| [11.22](../solutions/11.22.md) | Inscribed rectangle barrier solver | Explicit computation | Verified | [Tables](../solutions/11.22.md#executed-results) · [JSON](../../artifacts/11.22/results.json) · [Book p. 629](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=643) |
| [11.23](../solutions/11.23.md) | Partitioning SDP and rounding | Explicit computation | Verified | [Tables](../solutions/11.23.md#executed-results) · [JSON](../../artifacts/11.23/results.json) · [Book p. 629](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=643) |
| [11.24](../solutions/11.24.md) | QP barrier and primal-dual methods | Explicit computation | Verified | [Tables](../solutions/11.24.md#executed-results) · [JSON](../../artifacts/11.24/results.json) · [Book p. 630](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=644) |

The original `docs/solutions/` URLs remain unchanged. The numerical extensions are 4.3, 5.29, 5.33, 5.34, and 11.1.

## Reproduce

```sh
convex-solutions run all --output artifacts --seed 0
python scripts/render_results.py --check
```

See the [execution record](../../coverage/results-refresh.json), [verification report](../../coverage/verification.json), and [complete category manifest](../../coverage/catalog.json).
