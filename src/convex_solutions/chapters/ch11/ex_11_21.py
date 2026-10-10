"""Exercise 11.21: bounded fractional Chebyshev approximation."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import cvxpy as cp
from convex_solutions.io import write_json
from convex_solutions.numerics import solve_cvxpy
from convex_solutions.plotting import pyplot
from .common import barrier_method, infeasible_oracle


def make_oracle(A, b, lower, upper):
    """Return t*rho - sum(log(-g_i)); all g_i are convex and C2."""
    A, b = np.asarray(A, float), np.asarray(b, float)
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    m, n = A.shape
    if b.shape != (m,) or lower.shape != (n,) or upper.shape != (n,):
        raise ValueError("Inconsistent dimensions")
    if np.any(b <= 0) or np.any(lower >= upper):
        raise ValueError("Require positive b and strict lower < upper")
    def oracle(z, t):
        x, rho = z[:-1], z[-1]
        y = A @ x
        if np.any(y <= 0):
            return infeasible_oracle(n+1)
        # g = [y/b-rho, b/y-rho, lower-x, x-upper].
        g = np.r_[y/b-rho, b/y-rho, lower-x, x-upper]
        if np.any(g >= 0):
            return infeasible_oracle(n+1)
        jacobian = np.vstack([np.c_[A/b[:,None], -np.ones(m)],
                              np.c_[-b[:,None]*A/y[:,None]**2, -np.ones(m)],
                              np.c_[-np.eye(n), np.zeros(n)],
                              np.c_[np.eye(n), np.zeros(n)]])
        reciprocal_slack = -1/g
        value = t*rho - np.log(-g).sum()
        gradient = jacobian.T @ reciprocal_slack
        gradient[-1] += t
        hessian = jacobian.T @ (reciprocal_slack[:,None]**2 * jacobian)
        weights = reciprocal_slack[m:2*m] * 2*b/y**3
        hessian[:n,:n] += A.T @ (weights[:,None]*A)
        return float(value), gradient, hessian
    return oracle


def solve_fractional_chebyshev(A, b, lower, upper, x0, *, tolerance=1e-6):
    A, b, lower, upper, x0 = map(lambda x: np.asarray(x, float),
                                (A,b,lower,upper,x0))
    y = A @ x0
    if np.any(x0 <= lower) or np.any(x0 >= upper) or np.any(y <= 0):
        raise ValueError("The initial point must satisfy strict bounds and Ax0 > 0")
    rho0 = max(float(np.max(y/b)), float(np.max(b/y))) + 1
    m, n = A.shape
    result = barrier_method(make_oracle(A,b,lower,upper), np.r_[x0,rho0],
                            2*m+2*n, tolerance=tolerance)
    x, rho = result.x[:-1], float(result.x[-1])
    y = A@x
    deviations = np.maximum(y/b,b/y)
    constraints = np.r_[y/b-rho,b/y-rho,lower-x,x-upper]
    multipliers = 1/(-result.t*constraints)
    _, gradient, _ = make_oracle(A,b,lower,upper)(result.x,result.t)
    residual = gradient/result.t
    # The optimum lies in [lower,upper] x [1,rho]. Minimize the affine
    # supporting function of the convex Lagrangian over that box to account
    # conservatively for finite Newton centering accuracy.
    correction = float(np.minimum(residual[:-1]*(lower-x),
                                  residual[:-1]*(upper-x)).sum()
                       + min(residual[-1]*(1-rho),0.0))
    certified_lower_bound = rho-(2*m+2*n)/result.t+correction
    return {"x": x, "rho": rho, "fractional_deviation": float(max(deviations)),
            "log_chebyshev_error": float(np.log(max(deviations))),
            "multipliers": multipliers, "central_path_gap_bound": (2*m+2*n)/result.t,
            "stationarity_residual": float(np.linalg.norm(residual)),
            "certified_lower_bound": certified_lower_bound,
            "certified_objective_gap": rho-certified_lower_bound,
            "primal_violation": float(max(0, max(constraints))),
            "complementarity_gap": float(-constraints@multipliers),
            "history": result.history, "newton_steps": result.newton_steps,
            "t": result.t}


def reference_solution(A,b,lower,upper):
    x = cp.Variable(A.shape[1]); rho = cp.Variable()
    constraints = [x >= lower, x <= upper, A@x <= cp.multiply(b,rho),
                   cp.multiply(b,cp.inv_pos(A@x)) <= rho]
    problem = cp.Problem(cp.Minimize(rho),constraints)
    solver = solve_cvxpy(problem, solver="CLARABEL", tol_gap_abs=1e-9,
                         tol_feas=1e-9, tol_gap_rel=1e-9)
    return {"x": x.value, "rho": float(rho.value), **solver}


def solve(output_dir: Path | None = None, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    problems = []
    for name, m, n, noise in [("moderate_noise",20,6,.12),
                             ("tight_bounds",30,8,.45)]:
        A = rng.uniform(.05,1,(m,n))
        lower = rng.uniform(.05,.3,n)
        upper = lower + rng.uniform(.4,1,n)
        x0 = (lower+upper)/2
        b = (A@rng.uniform(lower,upper))*np.exp(rng.normal(0,noise,m))
        answer = solve_fractional_chebyshev(A,b,lower,upper,x0)
        reference = reference_solution(A,b,lower,upper)
        answer.update({"name":name,"A":A,"b":b,"lower":lower,"upper":upper,
                       "initial_point":x0,"reference":reference,
                       "objective_difference":answer["rho"]-reference["rho"]})
        problems.append(answer)
    result = {"exercise":"11.21","seed":seed,"instances":problems,
              "formulation":"min rho; (Ax)/b <= rho, b/(Ax) <= rho, lower <= x <= upper"}
    if output_dir is not None:
        output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
        plt=pyplot(); fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
        for ax,answer in zip(axes,problems):
            y=answer["A"]@answer["x"]
            ax.plot(y/answer["b"],"o",label=r"$(Ax)_i/b_i$")
            ax.axhline(answer["rho"],color="tab:red",linestyle="--",label=r"$\rho$")
            ax.axhline(1/answer["rho"],color="tab:red",linestyle=":",label=r"$1/\rho$")
            ax.axhline(1,color="gray",linewidth=.8)
            ax.set(xlabel="row i",ylabel="fit / target",title=answer["name"])
            ax.legend();ax.grid(alpha=.25)
        fig.savefig(output_dir/"fractional_fit.png",dpi=160);plt.close(fig)
        write_json(output_dir/"results.json",result)
    return result
