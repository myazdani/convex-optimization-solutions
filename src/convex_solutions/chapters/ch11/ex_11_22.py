"""Exercise 11.22: maximum-volume axis-aligned inscribed rectangles."""
from __future__ import annotations
from itertools import combinations
from pathlib import Path
import numpy as np
import cvxpy as cp
from scipy.spatial import ConvexHull
from convex_solutions.io import write_json
from convex_solutions.numerics import solve_cvxpy
from convex_solutions.plotting import pyplot
from .common import barrier_method, infeasible_oracle

BOOK_A = np.array([[0,-1],[2,-4],[2,1],[-4,4],[-4,0]],dtype=float)
BOOK_B = np.ones(5)


def make_oracle(A,b):
    A,b=np.asarray(A,float),np.asarray(b,float)
    m,n=A.shape
    G=np.c_[A,np.abs(A)]
    def oracle(z,t):
        radius=z[n:]
        slack=b-G@z
        if np.any(radius<=0) or np.any(slack<=0):
            return infeasible_oracle(2*n)
        inverse=1/slack
        gradient=G.T@inverse
        gradient[n:]-=t/radius
        hessian=G.T@(inverse[:,None]**2*G)
        hessian[n:,n:]+=np.diag(t/radius**2)
        return float(-t*np.log(radius).sum()-np.log(slack).sum()),gradient,hessian
    return oracle


def solve_rectangle(A,b,*,tolerance=1e-6):
    A,b=np.asarray(A,float),np.asarray(b,float)
    m,n=A.shape
    if b.shape!=(m,) or np.any(b<=0):
        raise ValueError("Require b > 0 and consistent dimensions")
    row_norm=np.abs(A).sum(axis=1)
    nonzero=row_norm>0
    if not np.any(nonzero):
        raise ValueError("The polyhedron is unbounded")
    radius0=.25*np.min(b[nonzero]/row_norm[nonzero])
    z0=np.r_[np.zeros(n),np.full(n,radius0)]
    result=barrier_method(make_oracle(A,b),z0,m,tolerance=tolerance)
    center,radius=result.x[:n],result.x[n:]
    G=np.c_[A,np.abs(A)];slack=b-G@result.x
    multipliers=1/(result.t*slack)
    stationarity=np.r_[A.T@multipliers,np.abs(A).T@multipliers-1/radius]
    lower,upper=center-radius,center+radius
    # Enforce the dual equality A.T lambda=0 for a genuine objective bound.
    # Drop asymptotically inactive constraints and orthogonally project the
    # remaining multipliers; verify rather than assume nonnegativity.
    active=multipliers>max(1e-5,1e-4*np.max(multipliers))
    dual_multipliers=np.zeros(m)
    active_matrix=A[active].T
    lam=multipliers[active]
    lam-=np.linalg.lstsq(active_matrix,active_matrix@lam,rcond=None)[0]
    dual_multipliers[active]=lam
    dual_d=np.abs(A).T@dual_multipliers
    dual_feasible=(np.min(dual_multipliers)>=0 and np.min(dual_d)>0
                   and np.linalg.norm(A.T@dual_multipliers)<1e-9)
    dual_objective=(float(np.sum(1+np.log(dual_d))-b@dual_multipliers)
                    if dual_feasible else None)
    certified_gap=(float(-np.log(radius).sum()-dual_objective)
                   if dual_feasible else None)
    return {"center":center,"radius":radius,"lower":lower,"upper":upper,
            "volume":float(np.prod(upper-lower)),
            "log_volume":float(np.log(upper-lower).sum()),
            "multipliers":multipliers,"central_path_gap_bound":m/result.t,
            "dual_feasible_multipliers":dual_multipliers if dual_feasible else None,
            "dual_objective":dual_objective,"certified_objective_gap":certified_gap,
            "stationarity_residual":float(np.linalg.norm(stationarity)),
            "primal_violation":float(max(0,-min(slack))),
            "minimum_slack":float(min(slack)),"history":result.history,
            "newton_steps":result.newton_steps}


def reference_solution(A,b):
    n=A.shape[1];center=cp.Variable(n);radius=cp.Variable(n,pos=True)
    problem=cp.Problem(cp.Maximize(cp.sum(cp.log(2*radius))),
                       [A@center+np.abs(A)@radius<=b])
    solver=solve_cvxpy(problem,solver="CLARABEL",tol_gap_abs=1e-9,
                        tol_feas=1e-9,tol_gap_rel=1e-9)
    return {"lower":center.value-radius.value,"upper":center.value+radius.value,
            "log_volume":float(problem.value),**solver}


def polygon_vertices(A,b):
    vertices=[]
    for i,j in combinations(range(len(b)),2):
        active=A[[i,j]]
        if abs(np.linalg.det(active))<1e-12:
            continue
        vertex=np.linalg.solve(active,b[[i,j]])
        if np.all(A@vertex<=b+1e-10):
            vertices.append(vertex)
    vertices=np.unique(np.round(vertices,12),axis=0)
    if len(vertices)<3:
        raise ValueError("No bounded two-dimensional polygon found")
    return vertices[ConvexHull(vertices).vertices]


def solve(output_dir: Path | None = None, seed: int = 0) -> dict:
    del seed
    examples=[("box",np.vstack([np.eye(2),-np.eye(2)]),np.array([3,4,1,2.])),
              ("triangle",np.array([[1,1],[-1,0],[0,-1.]]),np.ones(3)),
              ("book_polyhedron",BOOK_A,BOOK_B)]
    answers=[]
    for name,A,b in examples:
        answer=solve_rectangle(A,b)
        reference=reference_solution(A,b)
        answer.update({"name":name,"A":A,"b":b,"reference":reference,
                       "log_volume_difference":reference["log_volume"]-answer["log_volume"]})
        answers.append(answer)
    result={"exercise":"11.22","instances":answers,
            "analytic_example_volumes":{"box":24.0,"triangle":2.25}}
    if output_dir is not None:
        output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
        plt=pyplot();fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
        for ax,answer in zip(axes,answers):
            vertices=polygon_vertices(answer["A"],answer["b"])
            ax.fill(vertices[:,0],vertices[:,1],color="tab:blue",alpha=.15,label="polyhedron")
            closed=np.vstack([vertices,vertices[0]])
            ax.plot(closed[:,0],closed[:,1],color="tab:blue")
            l,u=answer["lower"],answer["upper"]
            rectangle=np.array([[l[0],l[1]],[u[0],l[1]],[u[0],u[1]],[l[0],u[1]],[l[0],l[1]]])
            ax.plot(rectangle[:,0],rectangle[:,1],color="tab:red",lw=2,label="optimal rectangle")
            ax.scatter(*answer["center"],color="tab:red",s=15)
            ax.set(title=f'{answer["name"]}\nvolume={answer["volume"]:.6f}',xlabel="x1",ylabel="x2")
            ax.set_aspect("equal");ax.legend(fontsize=8);ax.grid(alpha=.2)
        fig.savefig(output_dir/"maximum_volume_rectangles.png",dpi=170);plt.close(fig)
        write_json(output_dir/"results.json",result)
    return result
