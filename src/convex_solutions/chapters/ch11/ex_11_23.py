"""Exercise 11.23: an SDP barrier solver and three partition heuristics."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import cvxpy as cp
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from convex_solutions.io import write_json
from convex_solutions.numerics import solve_cvxpy
from convex_solutions.plotting import pyplot
from .common import barrier_method, infeasible_oracle, symmetric_zero_diagonal_weights


def make_oracle(W):
    W=np.asarray(W,float);n=len(W)
    identity=np.eye(n)
    def oracle(nu,t):
        S=W+np.diag(nu)
        try:
            factor=cho_factor(S,lower=True,check_finite=False)
        except LinAlgError:
            return infeasible_oracle(n)
        inverse=cho_solve(factor,identity,check_finite=False)
        value=t*nu.sum()-2*np.log(np.diag(factor[0])).sum()
        gradient=t-np.diag(inverse)
        hessian=inverse*inverse
        return float(value),gradient,hessian
    return oracle


def solve_sdp(W, *, tolerance=1e-5):
    """Solve both SDPs using the diagonal dual and recover a feasible primal.

    For an exactly centered point K/t has unit diagonal. Numerical centering
    is finite, so congruence scaling enforces diag(X)=1 before reporting the
    primal feasible upper bound. The returned gap is measured, not inferred.
    """
    W=np.asarray(W,float);n=len(W)
    if W.shape!=(n,n) or not np.allclose(W,W.T,atol=1e-12):
        raise ValueError("W must be symmetric")
    if np.max(np.abs(np.diag(W)))>1e-12:
        raise ValueError("This exercise assumes a zero diagonal")
    shift=max(1.,1.-np.linalg.eigvalsh(W)[0])
    initial=np.full(n,shift)
    answer=barrier_method(make_oracle(W),initial,n,tolerance=tolerance,
                          centering_tolerance=2e-9,mu=10)
    nu=answer.x;S=W+np.diag(nu)
    raw_X=cho_solve(cho_factor(S,lower=True),np.eye(n))/answer.t
    scaling=1/np.sqrt(np.diag(raw_X))
    X=raw_X*scaling[:,None]*scaling[None,:]
    X=(X+X.T)/2
    lower_bound=float(-nu.sum());primal_value=float(np.sum(W*X))
    return {"nu":nu,"X":X,"dual_lower_bound":lower_bound,
            "primal_sdp_value":primal_value,"certified_gap":primal_value-lower_bound,
            "central_path_gap_bound":n/answer.t,
            "dual_cone_min_eigenvalue":float(np.linalg.eigvalsh(S)[0]),
            "primal_cone_min_eigenvalue":float(np.linalg.eigvalsh(X)[0]),
            "diagonal_residual":float(np.max(np.abs(np.diag(X)-1))),
            "uncorrected_diagonal_residual":float(np.max(np.abs(np.diag(raw_X)-1))),
            "complementarity_norm":float(np.linalg.norm(S@X,"fro")),
            "newton_steps":answer.newton_steps,"history":answer.history}


def partition_cost(W,x):
    return float(x@W@x)


def exhaustive_partition(W):
    """Enumerate 2^(n-1) representatives using the global-sign symmetry."""
    n=len(W)
    if n>22:
        raise ValueError("Exhaustive search is deliberately limited to n <= 22")
    codes=np.arange(2**(n-1),dtype=np.uint64)
    x=np.ones((len(codes),n))
    x[:,1:]=2*((codes[:,None]>>np.arange(n-1,dtype=np.uint64))&1).astype(float)-1
    costs=np.einsum("bi,ij,bj->b",x,W,x)
    index=int(np.argmin(costs))
    return {"x":x[index],"value":float(costs[index]),"partitions_evaluated":len(codes)}


def sign_partition(x):
    return np.where(x>=0,1.,-1.)


def spectral_partition(X):
    _,vectors=np.linalg.eigh(X)
    vector=vectors[:,-1]
    # Fix the arbitrary eigenvector sign, including when some entries are 0.
    if vector[np.argmax(np.abs(vector))]<0:
        vector=-vector
    return sign_partition(vector)


def randomized_partitions(X, count, rng):
    """Draw independent N(0,X) samples by an eigendecomposition square root."""
    eigenvalues,vectors=np.linalg.eigh((X+X.T)/2)
    if eigenvalues[0]<-1e-8:
        raise ValueError("Covariance matrix is not positive semidefinite")
    factor=vectors*np.sqrt(np.maximum(eigenvalues,0))[None,:]
    samples=rng.standard_normal((count,len(X)))@factor.T
    return sign_partition(samples)


def greedy_refine(W,x, *, tolerance=1e-12):
    """Best-improvement one-bit descent; delta_i = -4*x_i*(W*x)_i."""
    W=np.asarray(W,float);x=np.asarray(x,float).copy()
    if not np.all(np.abs(x)==1) or np.max(np.abs(np.diag(W)))>1e-12:
        raise ValueError("Require a sign partition and zero-diagonal W")
    product=W@x
    history=[partition_cost(W,x)]
    for _ in range(100000):
        changes=-4*x*product
        index=int(np.argmin(changes))
        if changes[index]>=-tolerance:
            return {"x":x,"value":partition_cost(W,x),"history":history,
                    "flips":len(history)-1,"best_remaining_flip":float(min(changes))}
        old=x[index];x[index]=-old
        product-=2*old*W[:,index]
        history.append(partition_cost(W,x))
    raise RuntimeError("Greedy descent exceeded its safety limit")


def reference_sdp(W):
    n=len(W);X=cp.Variable((n,n),symmetric=True)
    problem=cp.Problem(cp.Minimize(cp.sum(cp.multiply(W,X))),[X>>0,cp.diag(X)==1])
    if n<=20:
        meta=solve_cvxpy(problem,solver="CLARABEL",tol_gap_abs=1e-8,
                         tol_gap_rel=1e-8,tol_feas=1e-8)
    else:
        meta=solve_cvxpy(problem,solver="SCS",eps=1e-5,max_iters=20000)
    return {**meta,"diagonal_residual":float(np.max(np.abs(np.diag(X.value)-1))),
            "cone_min_eigenvalue":float(np.linalg.eigvalsh(X.value)[0])}


def run_instance(W,rng, *, samples=300):
    n=len(W)
    sdp=solve_sdp(W,tolerance=1e-5 if n<=20 else 1e-3)
    spectral=spectral_partition(sdp["X"])
    random=randomized_partitions(sdp["X"],samples,rng)
    random_costs=np.einsum("bi,ij,bj->b",random,W,random)
    index=int(np.argmin(random_costs))
    refined=[greedy_refine(W,x) for x in random]
    refined_costs=np.array([r["value"] for r in refined])
    best_refined=int(np.argmin(refined_costs))
    exact=exhaustive_partition(W) if n<=22 else None
    return {"n":n,"W":W,"sdp":sdp,"reference":reference_sdp(W),"exact":exact,
            "exhaustive_search_note":("Used global-sign symmetry" if exact is not None
                                       else "Not run: 2^99 representatives at n=100"),
            "spectral":{"x":spectral,"value":partition_cost(W,spectral),
                         "greedy":greedy_refine(W,spectral)},
            "all_ones":{"value":partition_cost(W,np.ones(n)),
                        "greedy":greedy_refine(W,np.ones(n))},
            "randomized":{"samples":samples,"best_x":random[index],
                          "best_value":float(random_costs[index]),
                          "raw_values":random_costs,"refined_values":refined_costs,
                          "best_sample_greedy":refined[index],
                          "best_over_all_refined_samples":refined[best_refined],
                          "mean_greedy_improvement":float(np.mean(random_costs-refined_costs))}}


def solve(output_dir: Path | None = None, seed: int = 0) -> dict:
    rng=np.random.default_rng(seed)
    instances=[]
    for n in [6,8,10,100]:
        W=symmetric_zero_diagonal_weights(n,rng)
        instances.append(run_instance(W,rng))
    result={"exercise":"11.23","seed":seed,"instances":instances,
            "greedy_flip_formula":"delta_i = -4*x_i*(W@x)_i (W_ii=0)"}
    if output_dir is not None:
        output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
        plt=pyplot();fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
        for ax,item in zip(axes.ravel(),instances):
            labels=["spectral","random best","ones + greedy","spectral + greedy","random + greedy"]
            values=[item["spectral"]["value"],item["randomized"]["best_value"],
                    item["all_ones"]["greedy"]["value"],item["spectral"]["greedy"]["value"],
                    item["randomized"]["best_over_all_refined_samples"]["value"]]
            ax.bar(np.arange(5),values,color="tab:blue",alpha=.7)
            ax.axhline(item["sdp"]["dual_lower_bound"],color="tab:red",ls="--",label="SDP lower bound")
            if item["exact"] is not None:
                ax.axhline(item["exact"]["value"],color="black",ls=":",label="exact optimum")
            ax.set_xticks(np.arange(5),labels,rotation=30,ha="right",fontsize=8)
            ax.set(title=f'n={item["n"]}',ylabel="partition objective (lower is better)")
            ax.legend(fontsize=8);ax.grid(axis="y",alpha=.2)
        fig.savefig(output_dir/"partition_heuristics.png",dpi=160);plt.close(fig)
        large=instances[-1];fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
        ax.hist(large["randomized"]["raw_values"],bins=24,alpha=.6,label="Gaussian rounding")
        ax.hist(large["randomized"]["refined_values"],bins=24,alpha=.6,label="after greedy refinement")
        ax.axvline(large["sdp"]["dual_lower_bound"],color="black",ls="--",label="SDP lower bound")
        ax.set(xlabel="partition objective",ylabel="number of samples",title="n=100, 300 independent Gaussian samples")
        ax.legend();fig.savefig(output_dir/"randomized_greedy_distribution.png",dpi=160);plt.close(fig)
        write_json(output_dir/"results.json",result)
    return result
