"""Exercise 11.24: barrier and primal-dual algorithms for inequality QPs."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import cvxpy as cp
from convex_solutions.io import write_json
from convex_solutions.numerics import solve_cvxpy
from convex_solutions.plotting import pyplot
from .common import barrier_method, infeasible_oracle, positive_solve


def validate_qp(P,q,A,b,x0):
    P,q,A,b,x0=map(lambda z:np.asarray(z,float),(P,q,A,b,x0))
    n=len(q);m=len(b)
    if P.shape!=(n,n) or A.shape!=(m,n) or x0.shape!=(n,):
        raise ValueError("Inconsistent QP dimensions")
    if not np.allclose(P,P.T) or np.linalg.eigvalsh(P)[0]<-1e-10:
        raise ValueError("P must be symmetric positive semidefinite")
    if np.any(b-A@x0<=0):
        raise ValueError("Require a strictly feasible initial point")
    return P,q,A,b,x0


def objective(P,q,x):
    return float(.5*x@P@x+q@x)


def make_oracle(P,q,A,b):
    n=len(q)
    def oracle(x,t):
        slack=b-A@x
        if np.any(slack<=0):
            return infeasible_oracle(n)
        reciprocal=1/slack
        value=t*objective(P,q,x)-np.log(slack).sum()
        gradient=t*(P@x+q)+A.T@reciprocal
        hessian=t*P+A.T@(reciprocal[:,None]**2*A)
        return float(value),gradient,hessian
    return oracle


def qp_metrics(P,q,A,b,x,multipliers):
    slack=b-A@x
    linear=q+A.T@multipliers
    dual_residual=P@x+linear
    primal=objective(P,q,x)
    gap=None;dual=None
    # Every nonnegative lambda is dual feasible when P is positive definite.
    # A singular P additionally requires linear in range(P); near-feasibility
    # alone cannot justify a finite exact dual objective, so do not claim one.
    if np.linalg.eigvalsh(P)[0]>1e-11:
        dual=float(-.5*linear@positive_solve(P,linear)-b@multipliers)
        gap=primal-dual
    return {"objective":primal,"dual_objective":dual,"duality_gap":gap,
            "surrogate_gap":float(slack@multipliers),
            "dual_residual_norm":float(np.linalg.norm(dual_residual)),
            "primal_violation":float(max(0,-min(slack))),
            "minimum_slack":float(min(slack)),
            "minimum_multiplier":float(min(multipliers))}


def solve_barrier_qp(P,q,A,b,x0, *, tolerance=1e-7):
    P,q,A,b,x0=validate_qp(P,q,A,b,x0)
    def metrics(x,t):
        return qp_metrics(P,q,A,b,x,1/(t*(b-A@x)))
    answer=barrier_method(make_oracle(P,q,A,b),x0,len(b),tolerance=tolerance,
                          metrics=metrics)
    multipliers=1/(answer.t*(b-A@answer.x))
    return {"x":answer.x,"multipliers":multipliers,
            **qp_metrics(P,q,A,b,answer.x,multipliers),
            "history":answer.history,"newton_steps":answer.newton_steps,
            "central_path_gap_bound":len(b)/answer.t}


def solve_primal_dual_qp(P,q,A,b,x0, *, tolerance=1e-7, mu=10., max_iterations=200):
    """Strictly primal-feasible primal-dual Newton method (Algorithm 11.2).

    The Newton system is reduced by eliminating d_lambda. Backtracking keeps
    both slacks and multipliers positive and decreases the residual for a
    fixed target complementarity parameter during each line search.
    """
    P,q,A,b,x=validate_qp(P,q,A,b,x0)
    if tolerance<=0 or mu<=1:
        raise ValueError("Require positive tolerance and mu > 1")
    x=x.copy();m=len(b);multipliers=np.ones(m)
    history=[]
    for iteration in range(max_iterations):
        current=qp_metrics(P,q,A,b,x,multipliers)
        history.append({"iteration":iteration,**current})
        if current["surrogate_gap"]<=tolerance and current["dual_residual_norm"]<=tolerance:
            return {"x":x,"multipliers":multipliers,**current,"history":history,
                    "iterations":iteration}
        slack=b-A@x
        target=current["surrogate_gap"]/(mu*m)
        dual_residual=P@x+q+A.T@multipliers
        center_residual=slack*multipliers-target
        hessian=P+A.T@((multipliers/slack)[:,None]*A)
        dx=positive_solve(hessian,-dual_residual+A.T@(center_residual/slack))
        dlambda=(-center_residual+multipliers*(A@dx))/slack
        ds=-A@dx
        step=1.
        negative=dlambda<0
        if np.any(negative):
            step=min(step,.99*float(np.min(-multipliers[negative]/dlambda[negative])))
        negative=ds<0
        if np.any(negative):
            step=min(step,.99*float(np.min(-slack[negative]/ds[negative])))
        residual_norm=np.linalg.norm(np.r_[dual_residual,center_residual])
        for _ in range(100):
            trial_x=x+step*dx;trial_lambda=multipliers+step*dlambda
            trial_slack=b-A@trial_x
            trial_residual=np.r_[P@trial_x+q+A.T@trial_lambda,
                                 trial_slack*trial_lambda-target]
            if (np.all(trial_slack>0) and np.all(trial_lambda>0)
                and np.linalg.norm(trial_residual)<=(1-.01*step)*residual_norm):
                break
            step*=.5
        else:
            raise RuntimeError("Primal-dual residual line search failed")
        history[-1]["step_length"]=step
        x,multipliers=trial_x,trial_lambda
    raise RuntimeError("Primal-dual iteration limit exceeded")


def reference_solution(P,q,A,b):
    x=cp.Variable(len(q))
    problem=cp.Problem(cp.Minimize(.5*cp.quad_form(x,P)+q@x),[A@x<=b])
    meta=solve_cvxpy(problem,solver="CLARABEL",tol_gap_abs=1e-10,
                      tol_gap_rel=1e-10,tol_feas=1e-10)
    return {"x":x.value,**meta}


def generate_instance(n,m,rng, *, conditioning=1.):
    Q,_=np.linalg.qr(rng.standard_normal((n,n)))
    P=Q@np.diag(np.geomspace(1,conditioning,n))@Q.T
    q=rng.standard_normal(n)*3
    random_A=rng.standard_normal((m-2*n,n))/np.sqrt(n)
    A=np.vstack([random_A,np.eye(n),-np.eye(n)])
    b=rng.uniform(.3,1.3,m)
    return P,q,A,b,np.zeros(n)


def solve(output_dir: Path | None = None, seed: int = 0) -> dict:
    rng=np.random.default_rng(seed);instances=[]
    for name,n,m,condition in [("small",5,20,3),("medium",20,70,20),("ill_conditioned",12,50,1000)]:
        P,q,A,b,x0=generate_instance(n,m,rng,conditioning=condition)
        barrier=solve_barrier_qp(P,q,A,b,x0)
        primal_dual=solve_primal_dual_qp(P,q,A,b,x0)
        reference=reference_solution(P,q,A,b)
        instances.append({"name":name,"P":P,"q":q,"A":A,"b":b,"initial_point":x0,
                          "barrier":barrier,"primal_dual":primal_dual,"reference":reference,
                          "barrier_objective_difference":barrier["objective"]-reference["objective"],
                          "primal_dual_objective_difference":primal_dual["objective"]-reference["objective"]})
    result={"exercise":"11.24","seed":seed,"instances":instances}
    if output_dir is not None:
        output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
        plt=pyplot();fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
        for item in instances:
            history=item["barrier"]["history"]
            centered=[h for h in history if h["centered"]]
            axes[0].semilogy([h["newton_step"] for h in history],
                             [max(h["duality_gap"],1e-16) for h in history],
                             alpha=.8,label=item["name"])
            axes[0].scatter([h["newton_step"] for h in centered],
                            [h["duality_gap"] for h in centered],s=10)
            history=item["primal_dual"]["history"]
            axes[1].semilogy([h["iteration"] for h in history],
                             [h["surrogate_gap"] for h in history],label=item["name"])
            axes[2].semilogy([h["iteration"] for h in history],
                             [max(h["dual_residual_norm"],1e-16) for h in history],label=item["name"])
        axes[0].set(title="Barrier method",xlabel="accepted Newton steps",ylabel="actual primal-dual gap")
        axes[1].set(title="Primal-dual method",xlabel="iteration",ylabel="surrogate duality gap")
        axes[2].set(title="Primal-dual method",xlabel="iteration",ylabel="dual residual norm")
        for ax in axes:ax.grid(alpha=.25);ax.legend(fontsize=8)
        fig.savefig(output_dir/"qp_convergence.png",dpi=170);plt.close(fig)
        write_json(output_dir/"results.json",result)
    return result
