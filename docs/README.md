# Solution index

## Three categories

1. **[Examples](examples/README.md)**: numerical worked examples in the text with
   supplied results or figures to reproduce: **2 verified, 60 pending** across 62 studies.
   Start with [Example 7.4](examples/7.4.md) and [Example 8.3](examples/8.3.md).
2. **[Computational exercises](computational-exercises/README.md)**: the 16 existing
   numerical exercise solutions, including five numerical extensions, all verified.
3. **[All other exercises](other-exercises/README.md)**: the other 321 numbered
   exercises, including theoretical work, with chapter/source links and pending status.

All 337 numbered exercises appear exactly once in categories 2 and 3. The worked
examples in category 1 are a different kind of book content and do not count toward
those 337 exercises. The [canonical manifest](../coverage/catalog.json) records
categories, status, source pages, and delivered files.

## Existing solution links

The 16 solution URLs below are preserved. "Required" and "Extra" describe the
historical numerical selection, not the current three-category scope. In particular,
the five extras are numbered exercises, not worked examples from the text.

| Exercise | Historical numerical selection | Topic | Results |
|---|---|---|---|
| [4.3](solutions/4.3.md) | Extra | Box-QP optimality certificate | [Tables](solutions/4.3.md#executed-results) · [JSON](../artifacts/4.3/results.json) |
| [5.29](solutions/5.29.md) | Extra | Nonconvex sphere-QP stationary points | [Tables](solutions/5.29.md#executed-results) · [JSON](../artifacts/5.29/results.json) |
| [5.33](solutions/5.33.md) | Extra | L1 fitting sensitivity certificate | [Tables](solutions/5.33.md#executed-results) · [JSON](../artifacts/5.33/results.json) |
| [5.34](solutions/5.34.md) | Extra | LP primal-dual sensitivity | [Tables](solutions/5.34.md#executed-results) · [JSON](../artifacts/5.34/results.json) |
| [9.10](solutions/9.10.md) | Required | Undamped Newton behavior | [Tables](solutions/9.10.md#executed-results) · [JSON](../artifacts/9.10/results.json) |
| [9.30](solutions/9.30.md) | Required | Gradient versus Newton centering | [Tables](solutions/9.30.md#executed-results) · [JSON](../artifacts/9.30/results.json) |
| [9.31](solutions/9.31.md) | Required | Approximate Newton centering | [Tables](solutions/9.31.md#executed-results) · [JSON](../artifacts/9.31/results.json) |
| [9.32](solutions/9.32.md) | Required | Gauss–Newton for convex residuals | [Tables](solutions/9.32.md#executed-results) · [JSON](../artifacts/9.32/results.json) |
| [10.14](solutions/10.14.md) | Required | Log-growth portfolio Newton solve | [Tables](solutions/10.14.md#executed-results) · [JSON](../artifacts/10.14/results.json) |
| [10.15](solutions/10.15.md) | Required | Three entropy Newton formulations | [Tables](solutions/10.15.md#executed-results) · [JSON](../artifacts/10.15/results.json) |
| [10.16](solutions/10.16.md) | Required | Newton saddle-point game | [Tables](solutions/10.16.md#executed-results) · [JSON](../artifacts/10.16/results.json) |
| [11.1](solutions/11.1.md) | Extra | Scalar barrier path visualization | [Tables](solutions/11.1.md#executed-results) · [JSON](../artifacts/11.1/results.json) |
| [11.21](solutions/11.21.md) | Required | Fractional Chebyshev barrier solver | [Tables](solutions/11.21.md#executed-results) · [JSON](../artifacts/11.21/results.json) |
| [11.22](solutions/11.22.md) | Required | Inscribed rectangle barrier solver | [Tables](solutions/11.22.md#executed-results) · [JSON](../artifacts/11.22/results.json) |
| [11.23](solutions/11.23.md) | Required | Partitioning SDP and rounding | [Tables](solutions/11.23.md#executed-results) · [JSON](../artifacts/11.23/results.json) |
| [11.24](solutions/11.24.md) | Required | QP barrier and primal-dual methods | [Tables](solutions/11.24.md#executed-results) · [JSON](../artifacts/11.24/results.json) |

## Coverage and verification

- [Canonical three-category manifest](../coverage/catalog.json)
- [Worked-example source inventory](../coverage/worked_examples.json)
- [Coverage definitions and validation](../coverage/README.md)
- [Historical book source audit](../coverage/source_audit.json): source pages for 337 numbered exercises
- [Historical computational inventory](../coverage/inventory.json): the numerical delivery's selection
- [Verification report](../coverage/verification.json): existing exercise checks, environment, and limits
- [Exercise execution record](../coverage/results-refresh.json), [test log](../coverage/results-tests.log), and [run log](../coverage/results-rerun.log)
- [Check or refresh the exercise result tables](../scripts/render_results.py) from executed output
- [Generated figures and data](../artifacts/)

An entry marked pending has no delivered solution. Inventory completeness and
solution completeness are separate claims. See [contributing](CONTRIBUTING.md)
for the evidence needed to add a solution or example reproduction.
