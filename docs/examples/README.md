# 1. Examples

[Book index](../README.md) · [1. Examples](../examples/README.md) · [2. Computational exercises](../computational-exercises/README.md) · [3. All other exercises](../other-exercises/README.md)

Numerical worked examples in the textbook narrative, where the book supplies a numerical result or plot to reproduce. These are separate from end-of-chapter exercises.

**62 catalogued; 2 reproduced in this repository.** 2 exact-data and 0 method reproductions. Method reproductions use explicitly documented inputs when the original inputs are unavailable; they do not claim an exact figure match. Verified entries link to original solution notes and executed output; every other entry is pending. A source's solution or figure alone is not evidence of a completed implementation here.

## Available reproductions

- [Binary hypothesis testing](7.4.md) (exact-data reproduction): original explanation and [executed output](../../artifacts/examples/7.4/results.json).
- [Bounds on correlation coefficients](8.3.md) (exact-data reproduction): original explanation and [executed output](../../artifacts/examples/8.3/results.json).

## Inventory and reproducibility

All 156 numbered Examples and 175 Figures, plus unnumbered numerical studies, were screened across Chapters 1–11 and Appendices A–C; this catalog contains 62 numerical studies under the stated inclusion rules, with input-data limitations recorded and reproduction status tracked separately.

The [worked-example source inventory](../../coverage/worked_examples.json) records the screening policy, audit coverage, source references, expected outputs, and data limits. Consult its inventory status before treating this list as exhaustive.

- Exact data available: source data/code can support a matched reproduction.
- Self-contained: the text supplies enough information to construct the stated example.
- Official seeded generator: source code fixes the original random generator state; using the same integer seed in a different generator does not recreate its arrays.
- Method only / unspecified data: reproduce the method on clearly labeled data; do not claim an exact match to the book's plotted instance.

## Chapter 4

| Example / figure | Topic | Reproduction basis | Status |
|---|---|---|---|
| [example 4.1](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=142) | Scalar optimal values, attainment, and unboundedness | self contained | Pending |
| [example 4.10; figure 4.10](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=194) | Minimal matrix upper bounds via weighted SDP | method only unspecified data | Pending |
| [section 4.7.6; figure 4.11](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=198) | Regularized least-squares achievable set and Pareto curve | exact data available | Pending |
| [section 4.7.6; figure 4.12](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=200) | Four-asset portfolio risk-return trade-off | self contained | Pending |

### Source and data notes

- **Scalar optimal values, attainment, and unboundedness:** All three scalar objective functions and their common domain x>0 are explicitly supplied. Small analytically solved numerical instances, retained under the inventory rule that a concrete instance with an explicit numeric result qualifies even without a numerical algorithm. The first two have no finite attained optimizer; a bounded-domain numerical approximation must not be mistaken for an optimizer.
- **Minimal matrix upper bounds via weighted SDP:** The book specifies n=2 and three input matrices geometrically, but supplies neither their numerical entries nor the two weight matrices. No matching exact-data source was located in the official CVX/CVXOPT example indexes. Included despite its geometric presentation because the caption explicitly identifies the two enclosing ellipsoids as results of solving SDP (4.64). A new instance can reproduce the method, but not certify exact Figure 4.10 reproduction.
- **Regularized least-squares achievable set and Pareto curve:** The textbook gives A in R^(100x10), b in R^100 but not their entries. The author-endorsed CVXOPT book example supplies A and b in rls.bin, plus a figure-specific script. Linked data file was HTTP-fetched and hashed but not deserialized or numerically checked. The companion computes both boundary curves with small SDPs; a regularization sweep alone omits part of the shaded achievable set.
- **Four-asset portfolio risk-return trade-off:** Book supplies means [0.12,0.10,0.07,0.03], standard deviations [0.20,0.10,0.05,0], and correlations rho12=0.30, rho13=-0.40, rho23=0. The implied covariance matrix is [[0.04,0.006,-0.004,0],[0.006,0.01,0,0],[-0.004,0,0.0025,0],[0,0,0,0]]. Long-only, fully invested portfolio: x>=0 and sum(x)=1. Sample the QP scalarization and include the two limiting portfolios. Companion QP convention includes a 1/2 quadratic factor, changing the meaning of its mu but not the frontier.

## Chapter 5

| Example / figure | Topic | Reproduction basis | Status |
|---|---|---|---|
| [section 5.1.3; figure 5.1; figure 5.2](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=230) | Nonconvex scalar primal and Lagrangian dual numerical illustration | method only unspecified data | Pending |

### Source and data notes

- **Nonconvex scalar primal and Lagrangian dual numerical illustration:** Numerical results and plots appear in the captions, but formulas or sampled data for f0 and f1 are not provided. No matching exact-data companion was located in the official example indexes. Both figures are one numerical instance and should be reproduced together. Do not invent f0 or f1 and label the outcome an exact reproduction; a newly specified pair is only a methodological illustration.

## Chapter 6

| Example / figure | Topic | Reproduction basis | Status |
|---|---|---|---|
| [figure 6.2; section 6.1.2](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=310) | Comparing four residual penalties | official seeded generator | Pending |
| [example 6.2; figure 6.5](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=313) | Huber regression with two outliers | exact data available | Pending |
| [example 6.3; figure 6.6](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=321) | Regularized input design | exact data available | Pending |
| [example 6.4; figure 6.7](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=324) | Sparse regressor selection | exact data available | Pending |
| [figure 6.8; figure 6.9; figure 6.10; section 6.3.3](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=326) | Quadratic signal smoothing | official seeded generator | Pending |
| [figure 6.11; figure 6.12; figure 6.13; figure 6.14; section 6.3.3](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=328) | Total-variation signal reconstruction | method only unspecified data | Pending |
| [example 6.5; figure 6.15](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=333) | Nominal, stochastic and worst-case least squares | exact data available | Pending |
| [example 6.6; figure 6.16](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=338) | Robust least squares with disk uncertainty | exact data available | Pending |
| [example 6.7; figure 6.19](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=345) | Degree-five polynomial fitting | exact data available | Pending |
| [example 6.8; figure 6.20](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=346) | Cubic spline fitting | exact data available | Pending |
| [figure 6.21; figure 6.22; figure 6.23; section 6.5.4](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=348) | Gabor basis pursuit and time-frequency recovery | self contained with source discrepancy | Pending |
| [figure 6.24; section 6.5.5](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=352) | Convex least-squares function fitting | exact data available | Pending |
| [example 6.9; figure 6.25; figure 6.26](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=355) | Consumer preference bounds | exact data available | Pending |

### Source and data notes

- **Comparing four residual penalties:** Text gives a random 100-by-30 problem; CVX fixes the legacy MATLAB randn state; CVXOPT generates fresh random arrays. Example 6.1 itself describes penalty families; its later unnumbered numerical study is cataloged here.
- **Huber regression with two outliers:** CVXOPT publishes the 42 input points in huber.bin; CVX provides the original seeded generator.
- **Regularized input design:** Impulse response and weights are in the text; the exact piecewise-constant desired output is specified in both companion scripts. The CVX script preserves the text normalization of the derivative penalty; the CVXOPT least-squares stacking omits the n/(n-1) relative normalization. Choose and document the convention before numerical comparison.
- **Sparse regressor selection:** The 10-by-20 matrix and response are published in regsel.bin. Do not replace exhaustive enumeration with a heuristic while claiming to reproduce the globally optimal curve.
- **Quadratic signal smoothing:** Signal formula is explicit in companion code; noise realization is fixed by legacy MATLAB randn state 0 in CVX. CVXOPT draws fresh noise.
- **Total-variation signal reconstruction:** Companion code defines the piecewise signal but draws unseeded Gaussian noise. Original noise samples are not published in these sources. Fix and publish a new seed for a method reproduction; do not assert pointwise equality with the original noisy signal.
- **Nominal, stochastic and worst-case least squares:** robls.bin contains the 20-by-10 instance indexed as 6.15.
- **Robust least squares with disk uncertainty:** robls.bin contains the 50-by-20 instance indexed as 6.16. CVX supplies a seeded generator and Monte Carlo sampler. Book p. 324 states delta=1; both official CVX and CVXOPT scripts use 0.1. This discrepancy must be resolved or both variants reported. The CVXOPT Monte Carlo draws are unseeded; exact histogram bins are not fixed by its data file. Matrix-data availability does not imply exact histogram reproduction.
- **Degree-five polynomial fitting:** Forty data pairs are in polapprox.bin; CVX supplies a seeded generator.
- **Cubic spline fitting:** Same 40 input pairs as Example 6.7, published in polapprox.bin.
- **Gabor basis pursuit and time-frequency recovery:** Signal and dictionary are explicit. The sampling-grid statements in the text and companion implementations need reconciliation. Book says 501 equally spaced samples over [0,1] but prints t_k=0.005k for k=0,...,500. CVX uses k/500 and truncates small Gaussian tails at 0.001; CVXOPT uses k/501 and an untruncated dictionary. Reconcile and state the target before comparing atom counts or error. Selected atom counts depend on solution precision and threshold; 42 is a benchmark, not a guarantee from changing the sampling grid.
- **Convex least-squares function fitting:** The data pairs for the figure are published in cvxfit.bin.
- **Consumer preference bounds:** Companion source contains the forty goods baskets as literal constants.

## Chapter 7

| Example / figure | Topic | Reproduction basis | Status |
|---|---|---|---|
| [figure 7.1; section 7.1.1](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=368) | Logistic maximum-likelihood regression | exact data available | Pending |
| [example 7.2; figure 7.2; figure 7.3](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=376) | Maximum entropy and distribution bounds | self contained | Pending |
| [example 7.4; figure 7.4](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=385) | Binary hypothesis testing | self contained | [Verified exact](7.4.md) · [Output](../../artifacts/examples/7.4/results.json) |
| [figure 7.5; figure 7.6; figure 7.7; section 7.4.3](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=395) | Chebyshev bounds for symbol detection | exact data available | Pending |
| [figure 7.8; section 7.4.3](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=396) | Chernoff and Monte Carlo detection bounds | exact data available | Pending |
| [figure 7.9; figure 7.10; figure 7.11; figure 7.12; section 7.5.2](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=402) | D-, E- and A-optimal experiment design | exact data available | Pending |

### Source and data notes

- **Logistic maximum-likelihood regression:** Fifty observations are published in logreg.bin; CVX also provides seeded generation.
- **Maximum entropy and distribution bounds:** 100 equally spaced support points and all four prior constraints are stated in the text.
- **Binary hypothesis testing:** The complete 4-by-2 probability matrix and four comparison detectors are printed on pp. 371–372.
- **Chebyshev bounds for symbol detection:** Official source fixes the seven-symbol constellation via literal Voronoi vertices.
- **Chernoff and Monte Carlo detection bounds:** Same fixed constellation as Figures 7.5–7.7; Monte Carlo noise samples are generated afresh. Exact bound can be reproduced; the Monte Carlo curve requires a published new seed and statistical error bars, not exact equality to book samples.
- **D-, E- and A-optimal experiment design:** Twenty 2D candidate measurement vectors are literal constants in companion source. Companion vector coordinates are rounded constants, so match printed weights within their stated precision.

## Chapter 8

| Example / figure | Topic | Reproduction basis | Status |
|---|---|---|---|
| [example 8.1](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=411) | Projection of the origin on the unit-square boundary | self contained | Pending |
| [example 8.3](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=422) | Bounds on correlation coefficients | self contained | [Verified exact](8.3.md) · [Output](../../artifacts/examples/8.3/results.json) |
| [figure 8.3; figure 8.4; section 8.4](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=426) | Minimum-volume enclosing and maximum-volume inscribed ellipsoids | exact data available | Pending |
| [figure 8.5; figure 8.6; figure 8.7; section 8.5](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=431) | Three centers of a polyhedron | exact data available | Pending |
| [figure 8.8; figure 8.9; section 8.6.1](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=437) | Linear and maximum-margin discrimination | official seeded generator | Pending |
| [figure 8.10; figure 8.11; figure 8.12; section 8.6.1](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=440) | Approximate linear classification comparison | exact data available | Pending |
| [figure 8.13; section 8.6.2](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=444) | Quadratic ellipsoidal discrimination | official seeded generator | Pending |
| [figure 8.14; section 8.6.2](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=444) | Minimum-degree polynomial discrimination | official seeded generator | Pending |
| [example 8.6; figure 8.15; figure 8.16; figure 8.17](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=448) | Linear, quadratic and fourth-order placement | exact data available | Pending |
| [example 8.7; figure 8.19; figure 8.20](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=455) | Five-cell floor planning | exact data available | Pending |

### Source and data notes

- **Projection of the origin on the unit-square boundary:** Set and point are explicitly given; no external data. Small analytic numerical check. The boundary is nonconvex; do not feed equality norm=1 to a convex solver as a DCP constraint.
- **Bounds on correlation coefficients:** All bounds on four correlations and the unit diagonal are printed in the book. Printed witness matrices are rounded and optimal witnesses need not be unique. Compare objective and feasibility rather than requiring entrywise equality.
- **Minimum-volume enclosing and maximum-volume inscribed ellipsoids:** The common six-vertex polygon is explicitly given in official companion source.
- **Three centers of a polyhedron:** The six-vertex polygon and its inequalities are constructed from literal constants in companion source.
- **Linear and maximum-margin discrimination:** Both official scripts regenerate the same point clouds with legacy MATLAB randn state 3.
- **Approximate linear classification comparison:** The common two 50-point classes are published in linsep.bin.
- **Quadratic ellipsoidal discrimination:** Official section-specific CVX script provides a seeded 2D point-cloud generator. CVX script is linked by section rather than explicitly by figure; verify point-cloud correspondence before claiming exact Figure 8.13 recreation.
- **Minimum-degree polynomial discrimination:** Official section-specific scripts share a seeded generator with an exchanged wedge between two point classes. Verify generated points and feasibility formulation against the textbook before claiming exact Figure 8.14 correspondence; the scripts are section-specific.
- **Linear, quadratic and fourth-order placement:** Eight fixed points, six free points and 27 graph links are published in placement.bin.
- **Five-cell floor planning:** Companion source supplies spacing 1, aspect-ratio bound 5, relative-position constraints and all four minimum-area vectors. Figure 8.19 is input geometry; Figure 8.20 is the numerical output. CVX test_floorplan.m uses a different 60-cell example and is not the exact five-cell target.

## Chapter 9

| Example / figure | Topic | Reproduction basis | Status |
|---|---|---|---|
| [figure 9.2; section 9.3.2](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=483) | Gradient descent on an ill-conditioned quadratic | self contained | Pending |
| [figure 9.3; figure 9.4; figure 9.5; section 9.3.2](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=484) | Gradient descent on a two-variable exponential objective | self contained | Pending |
| [figure 9.6; section 9.3.2](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=486) | Gradient descent for a 100-variable log-barrier objective | official seeded generator | Pending |
| [figure 9.7; figure 9.8; section 9.3.2](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=487) | Conditioning and gradient-method convergence | official seeded generator | Pending |
| [figure 9.11; figure 9.12; figure 9.13; figure 9.14; figure 9.15; section 9.4.4](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=495) | Steepest descent under two quadratic norms | self contained | Pending |
| [figure 9.19; figure 9.20; section 9.5.4](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=506) | Newton descent on the two-variable exponential objective | self contained | Pending |
| [figure 9.21; figure 9.22; section 9.5.4](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=506) | Newton convergence and step lengths in 100 variables | official seeded generator | Pending |
| [figure 9.23; section 9.5.4](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=508) | Sparse Newton solve with 10000 variables | official seeded generator | Pending |
| [figure 9.25; section 9.6.5](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=520) | Newton iteration counts for self-concordant functions | official seeded generator | Pending |

### Source and data notes

- **Gradient descent on an ill-conditioned quadratic:** f=(x1^2+10*x2^2)/2 and initial point (10,1) are fully specified. Companion plot code omits the harmless objective factor 1/2 in its comment/contour values; iterates are unaffected.
- **Gradient descent on a two-variable exponential objective:** The three-exponential objective is explicit; companion source supplies start(-1,1), backtracking and exact-search settings.
- **Gradient descent for a 100-variable log-barrier objective:** Companion source generates A(500,100), b and c using legacy MATLAB random states 0.
- **Conditioning and gradient-method convergence:** Same seeded 100-variable problem, rescaled across diagonal coordinate transformations. Related text on p. 495 reports Newton affine-invariance: nine iterations across gamma from 1e-10 to 1e10. That claim is an additional comparison target, not a separate supplied script.
- **Steepest descent under two quadratic norms:** Same explicit exponential objective with the two prescribed quadratic norms; matrices and start are in book/source.
- **Newton descent on the two-variable exponential objective:** Same explicit exponential objective and initial point as gradient demonstration.
- **Newton convergence and step lengths in 100 variables:** Same A,b,c seeded log-barrier instance as Figure 9.6.
- **Sparse Newton solve with 10000 variables:** Legacy MATLAB seeded sparse random generator specifies the instance; no stored arrays. Book states m=100000 additional log terms plus n box barriers; source constructs 100000 total rows including 20000 box rows. This dimensional discrepancy requires an explicit target choice. Use sparse linear algebra; do not silently scale down and label as the original experiment.
- **Newton iteration counts for self-concordant functions:** Companion source fixes legacy MATLAB seeds and generates three dimension families, five problems and ten starts per family. Exact dots require the original RNG semantics, problem rejection logic and start construction.

## Chapter 10

| Example / figure | Topic | Reproduction basis | Status |
|---|---|---|---|
| [figure 10.1; figure 10.2; section 10.3.5](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=555) | Infeasible-start Newton on a feasible analytic-center problem | official seeded generator | Pending |
| [figure 10.3; figure 10.4; section 10.3.5](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=556) | Infeasible-start Newton on an infeasible problem | official seeded generator | Pending |
| [figure 10.5; section 10.3.5](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=556) | Newton method for a convex-concave game | official seeded generator | Pending |
| [figure 10.6; figure 10.7; figure 10.8; section 10.4.3](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=562) | Three Newton formulations for analytic centering | official seeded generator | Pending |

### Source and data notes

- **Infeasible-start Newton on a feasible analytic-center problem:** 100 variables/50 equalities; legacy MATLAB rand state 3 and randn state 4.
- **Infeasible-start Newton on an infeasible problem:** Second instance generated in same official script with fixed states; affine constraints fail to intersect positive orthant.
- **Newton method for a convex-concave game:** 100-by-100 game matrix and linear terms from legacy MATLAB randn state 0. Book initializes u=v=0; companion source initializes x=y=0.01*ones. Report the chosen initialization before asserting a figure match.
- **Three Newton formulations for analytic centering:** 500 variables/100 equalities, seeded primal/dual feasible data and four starting points for each method.

## Chapter 11

| Example / figure | Topic | Reproduction basis | Status |
|---|---|---|---|
| [example 11.1; figure 11.2](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=579) | Central path of a two-variable linear program | method only unspecified data | Pending |
| [example 11.3; figure 11.3](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=581) | Central path as a force equilibrium | method only unspecified data | Pending |
| [figure 11.4; figure 11.5; section 11.3.2](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=585) | Barrier method on a small inequality-form LP | official seeded generator | Pending |
| [figure 11.6; section 11.3.2](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=587) | Barrier method on a small geometric program | official seeded generator | Pending |
| [figure 11.7; section 11.3.2](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=588) | Barrier method on three standard-form LP sizes | method only unspecified data | Pending |
| [figure 11.8; section 11.3.2](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=589) | Barrier-method scaling across 2000 LPs | method only unspecified data | Pending |
| [example 11.4; figure 11.9](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=594) | Comparing maximum and total phase-I infeasibility | method only unspecified data | Pending |
| [figure 11.10; figure 11.11; figure 11.12; section 11.4.3](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=596) | Feasibility difficulty near the boundary | method only unspecified data | Pending |
| [figure 11.15; figure 11.16; section 11.6.3](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=615) | Barrier method on a small SOCP | official seeded generator | Pending |
| [figure 11.17; figure 11.18; section 11.6.3](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=616) | Barrier method on a small SDP | official seeded generator | Pending |
| [figure 11.19; section 11.6.3](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=617) | Barrier method on three SDP sizes | official seeded generator | Pending |
| [figure 11.20; section 11.6.3](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=619) | Barrier-method scaling across 2000 SDPs | method only unspecified data | Pending |
| [figure 11.21; section 11.7.4](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=627) | Primal-dual method on a small LP | official seeded generator | Pending |
| [figure 11.22; section 11.7.4](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=627) | Primal-dual method on a small GP | official seeded generator | Pending |
| [figure 11.23; section 11.7.4](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=629) | Primal-dual scaling across 2000 LPs | method only unspecified data | Pending |

### Source and data notes

- **Central path of a two-variable linear program:** Text supplies dimensions n=2,m=6 and t=10 but not A,b,c; no figure-specific source found in the official chapter directory. A newly chosen polygon demonstrates the method, not the exact textbook instance.
- **Central path as a force equilibrium:** Five-constraint two-variable LP is drawn but A,b,c are not supplied in the checked sources.
- **Barrier method on a small inequality-form LP:** 100 inequalities/50 variables generated with fixed MATLAB states and shifted to optimum 1.
- **Barrier method on a small geometric program:** 50 variables/100 constraints with five exponential terms per function, generated with fixed legacy states.
- **Barrier method on three standard-form LP sizes:** Official script generates n=2m instances at three sizes but contains no RNG seed or stored arrays. Reproduce the algorithm/instance distribution with a recorded new seed; exact original data are not recoverable from this script alone.
- **Barrier-method scaling across 2000 LPs:** Text specifies 100 instances for each of 20 sizes and a random-data construction; no exact dataset/seed or matching full sweep script is linked. figure_11_7.m covers three illustrative instances, not the full 2000-instance sweep.
- **Comparing maximum and total phase-I infeasibility:** Text gives dimensions 100-by-50 but not the actual infeasible A,b; no matching source found in the official chapter directory.
- **Feasibility difficulty near the boundary:** Text gives a 50-by-20 parametric family A x <= b+gamma*delta_b but not A,b,delta_b or original generator.
- **Barrier method on a small SOCP:** 50 variables/50 cone constraints, with fixed legacy MATLAB states.
- **Barrier method on a small SDP:** 100 variables and 100-by-100 semidefinite inequality from a fixed-state generator.
- **Barrier method on three SDP sizes:** Seeded symmetric matrices at n=50,500,1000 for minimizing sum(x) with A+diag(x) PSD.
- **Barrier-method scaling across 2000 SDPs:** Text specifies 100 random symmetric matrices for each of 20 dimensions; no full sweep dataset or seed published in checked sources. figure_11_19.m supplies three illustrative sizes, not the full 2000-instance sweep.
- **Primal-dual method on a small LP:** Fixed-state generator supplies the same small LP family as the barrier-method comparison.
- **Primal-dual method on a small GP:** Fixed-state generator supplies the GP comparison instance.
- **Primal-dual scaling across 2000 LPs:** Text reuses the 2000 LP instances of Figure 11.8; no exact arrays, seed or matching sweep source found.

## Next steps

For each reproduction, use the identified source and available data, write original notes, run the implementation, and compare the output with the book's reported result. Record discrepancies and unspecified inputs explicitly. Only then add solution/output links and update its status.

The [complete category manifest](../../coverage/catalog.json) keeps this catalog separate from the 337 numbered exercises.
