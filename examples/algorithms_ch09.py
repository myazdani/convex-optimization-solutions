"""Original implementations of chapter 9 worked algorithm studies."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import spsolve
from scipy.optimize import brentq
from examples.algorithm_common import ROOT,begin_run, save_result, convergence_plot, pyplot

EXP_A=np.array([[1.,3.],[1.,-3.],[-1.,0.]])
EXP_XSTAR=np.array([-np.log(2)/2,0.])
EXP_FSTAR=2*np.sqrt(2)*np.exp(-.1)


def exponential(x):
    z=np.exp(np.clip(EXP_A@x-.1,-745,700))
    return z.sum(),EXP_A.T@z,(EXP_A.T*z)@EXP_A


def exp_descent(method='gradient', exact=False, metric=None):
    x=np.array([-1.,1.]); xs=[]; fs=[]; steps=[]
    for k in range(1001):
        f,g,H=exponential(x); xs.append(x.copy()); fs.append(f)
        if f-EXP_FSTAR<1e-12: break
        v=-linalg.solve(H,g,assume_a='pos') if method=='newton' else -g if metric is None else -linalg.solve(metric,g)
        s=1.
        if exact:
            derivative=lambda a:float(v@exponential(x+a*v)[1])
            while derivative(s)<0: s*=2
            s=brentq(derivative,0,s,xtol=1e-14)
        else:
            while exponential(x+s*v)[0]>f+.1*s*(g@v): s*=.7
        steps.append(s); x+=s*v
    return {'x':x,'trajectory':xs,'values':fs,'steps':steps,'iterations':len(steps),
            'objective_error':abs(fs[-1]-EXP_FSTAR),'point_error':float(np.linalg.norm(x-EXP_XSTAR)),
            'gradient_norm':float(np.linalg.norm(exponential(x)[1]))}


def _trajectories(out, runs, transformed=False):
    plt=pyplot(); fig,axes=plt.subplots(1,len(runs),figsize=(6*len(runs),4.5),squeeze=False)
    xx,yy=np.meshgrid(np.linspace(-1.7,1.,220),np.linspace(-1.25,1.2,200))
    vals=np.exp(xx+3*yy-.1)+np.exp(xx-3*yy-.1)+np.exp(-xx-.1)
    levels=np.linspace(EXP_FSTAR,exponential(np.array([-1.,1.]))[0]*1.3,7)[1:]
    for ax,(label,r) in zip(axes[0],runs.items()):
        T=np.sqrt(np.array([2,8]) if 'P1' in label else np.array([8,2])) if transformed else np.ones(2)
        ax.contour(T[0]*xx,T[1]*yy,vals,levels=levels,colors='.65',linewidths=.7)
        xy=np.asarray(r['trajectory'])*T
        ax.plot(xy[:,0],xy[:,1],'-o',markersize=3); ax.set(title=label,xlabel='x1' if not transformed else 'y1',ylabel='x2' if not transformed else 'y2')
        ax.set_aspect('equal'); ax.grid(alpha=.15)
    fig.tight_layout(); name='transformed.png' if transformed else 'trajectories.png'; fig.savefig(out/name,dpi=160); plt.close(fig)
    return name


def run_small(eid,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if eid=='fig_9_2':
        Q=np.diag([1.,10.]);x=np.array([10.,1.]); xs=[x.copy()]; steps=[]
        for _ in range(40):
            g=Q@x;s=g@g/(g@Q@g);x=x-s*g;xs.append(x.copy());steps.append(s)
        xs=np.array(xs);fs=np.einsum('ij,ij->i',xs@Q,xs)/2; q=(9/11)**2
        predicted=np.c_[10*(9/11)**np.arange(41),(-9/11)**np.arange(41)]
        result={'trajectory':xs,'values':fs,'steps':steps,'contraction_factor':q,'max_formula_error':np.max(abs(xs-predicted)),
                'max_ratio_error':np.max(abs(fs[1:]/fs[:-1]-q)),'final_objective':fs[-1]}
        plt=pyplot();fig,ax=plt.subplots(figsize=(7,4));t=np.linspace(0,2*np.pi,300)
        for level in np.linspace(10,55,5):ax.plot(np.sqrt(2*level)*np.cos(t),np.sqrt(level/5)*np.sin(t),'--',color='.7')
        ax.plot(xs[:,0],xs[:,1],'-o',markersize=3);ax.set(xlabel='x1',ylabel='x2',title='Exact-line-search gradient descent');fig.tight_layout();fig.savefig(out/'trajectories.png',dpi=160);plt.close(fig)
        convergence_plot(out,{'computed':fs})
        return save_result(eid,out,result,r'f(x)=\frac12(x_1^2+10x_2^2),\quad x^{(0)}=(10,1),\quad t_k=\frac{g_k^Tg_k}{g_k^TQg_k}.',
             'The exact directional quadratic minimizer is computed at every iteration. Its closed-form trajectory is checked independently, along with the ratio of successive objective values.',
             [('Updates',40),('Objective contraction',f'{q:.12g}'),('Maximum trajectory error',f'{result["max_formula_error"]:.3g}'),('Maximum ratio error',f'{result["max_ratio_error"]:.3g}'),('Final objective',f'{fs[-1]:.12g}')],
             [('trajectories.png','Gradient trajectory'),('convergence.png','Objective history')],
             ['The companion contour code omits the objective factor 1/2. The book normalization is used here; this does not change the exact-search iterates.'])
    if eid=='fig_9_3':runs={'backtracking':exp_descent(),'exact':exp_descent(exact=True)}
    elif eid=='fig_9_11':runs={'P1 = diag(2,8)':exp_descent(metric=np.diag([2.,8.])),'P2 = diag(8,2)':exp_descent(metric=np.diag([8.,2.]))}
    else:runs={'Newton':exp_descent(method='newton')}
    trajectory=_trajectories(out,runs);plots=[(trajectory,'Iterates and objective contours')]
    if eid=='fig_9_11':plots.append((_trajectories(out,runs,True),'Trajectories in transformed coordinates'))
    convergence_plot(out,{k:r['values'] for k,r in runs.items()},EXP_FSTAR);plots.append(('convergence.png','Objective errors'))
    rows=[('Analytic optimal x1',f'{EXP_XSTAR[0]:.12g}'),('Analytic optimal value',f'{EXP_FSTAR:.12g}')]
    for label,r in runs.items():rows.extend([(f'{label}: updates',r['iterations']),(f'{label}: objective error',f'{r["objective_error"]:.3g}'),(f'{label}: point error',f'{r["point_error"]:.3g}')])
    notes=['The exact optimum is derived by symmetry: x2 = 0 and exp(2 x1) = 1/2. It provides an independent benchmark, rather than taking the last iterate as the optimum.',
           'The companion approximates exact line search by bisection with a 0.001 interval width and caps backtracking at ten reductions. This implementation solves the directional derivative to numerical precision and requires the Armijo inequality before every accepted step. Thus the same instance and methods are reproduced, without claiming pixel-identical paths under the companion stopping approximations.']
    return save_result(eid,out,{'runs':runs,'analytic_minimizer':EXP_XSTAR,'analytic_optimum':EXP_FSTAR},
            r'f(x)=e^{x_1+3x_2-0.1}+e^{x_1-3x_2-0.1}+e^{-x_1-0.1},\quad x^{(0)}=(-1,1).',
            'Directions are negative gradients, negative metric-inverse gradients, or negative Hessian-inverse gradients as appropriate. Backtracking uses alpha = 0.1 and beta = 0.7. The quadratic-norm plots also show the coordinate transformation y = square-root(P) x.',rows,plots,notes)


class LogBarrier:
    def __init__(self,A,b,c=None):self.A=A;self.b=b;self.c=np.zeros(A.shape[1]) if c is None else c
    def eval(self,x,hessian=True):
        y=self.b-self.A@x
        if np.min(y)<=0:return np.inf,None,None
        inv=1/y;f=float(self.c@x-np.log(y).sum());g=self.c+self.A.T@inv
        H=(self.A.T@sparse.diags(inv**2)@self.A).tocsc() if sparse.issparse(self.A) and hessian else (self.A.T*inv**2)@self.A if hessian else None
        return f,g,H


def barrier_descent(problem,x0=None,method='newton',exact=False,alpha=.01,beta=.5,tol=1e-10,maxiter=100,fstar=None,equilibrate=False,trace=False):
    x=np.zeros(problem.A.shape[1]) if x0 is None else x0.copy(); vals=[];steps=[];decrements=[];gnorm=[];trajectory=[]
    for k in range(maxiter+1):
        f,g,H=problem.eval(x,method=='newton');vals.append(f);gnorm.append(np.linalg.norm(g))
        if trace:trajectory.append(x.copy())
        if method=='gradient' and fstar is not None and f-fstar<tol:break
        if method=='newton' and equilibrate:
            scale=np.sqrt(np.diag(H));v=-linalg.solve(H/scale[:,None]/scale[None,:],g/scale,assume_a='pos')/scale
        else:v=-(spsolve(H,g) if sparse.issparse(H) else linalg.solve(H,g,assume_a='pos')) if method=='newton' else -g
        dec=-g@v;decrements.append(dec)
        if method=='newton' and dec/2<tol:break
        if k==maxiter:break
        y=problem.b-problem.A@x;d=-(problem.A@v)/y;dc=problem.c@v
        neg=d<0;bound=np.min(-1/d[neg]) if neg.any() else np.inf
        if exact:
            hi=.999999*bound
            deriv=lambda s:float(dc-np.sum(d/(1+s*d)))
            if np.isinf(hi):
                hi=1.
                while deriv(hi)<0:hi*=2
            s=brentq(deriv,0,hi,xtol=1e-13)
        else:
            s=1.
            while s>=bound or s*dc-np.log1p(s*d).sum()>-alpha*s*dec:s*=beta
        steps.append(s);x+=s*v
    f,g,H=problem.eval(x)
    return {'x':x,'values':vals,'steps':steps,'decrements':decrements,'gradient_norms':gnorm,'iterations':len(steps),'objective':f,
            'gradient_norm':np.linalg.norm(g),'minimum_slack':np.min(problem.b-problem.A@x),'newton_decrement_squared':decrements[-1] if method=='newton' else None,'trajectory':trajectory}


def dense_instance():
    r=np.random.default_rng(906);A=r.standard_normal((500,100));b=r.uniform(.01,1.,500);c=20*r.standard_normal(100)
    return LogBarrier(A,b,c)


def run_dense(eid,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);p=dense_instance();np.savez_compressed(out/'instance.npz',A=p.A,b=p.b,c=p.c,seed=906)
    optimum=barrier_descent(p,tol=1e-18);fstar=optimum['objective']
    if eid=='fig_9_7':
        gammas=np.logspace(-1.5,1.5,11);sweep=[]
        for gamma in gammas:
            t=gamma**(np.arange(100)/100);q=LogBarrier(p.A*t,p.b,p.c*t)
            r=barrier_descent(q,method='gradient',alpha=.3,beta=.7,tol=1e-5,maxiter=5000,fstar=fstar)
            sweep.append({'gamma':gamma,'iterations':r['iterations'],'final_gap':r['objective']-fstar,'converged':r['objective']-fstar<1e-5})
        condition_gammas=np.logspace(-1.4,1.5,1000);H=p.eval(optimum['x'])[2]
        conditions=[np.linalg.cond((g**(np.arange(100)/100))[:,None]*H*(g**(np.arange(100)/100))[None,:]) for g in condition_gammas]
        affine=[];base=barrier_descent(p,tol=1e-10,equilibrate=True,trace=True)
        for gamma in np.logspace(-10,10,9):
            t=gamma**(np.arange(100)/100)
            q=LogBarrier(p.A*t,p.b,p.c*t)
            r=barrier_descent(q,tol=1e-10,equilibrate=True,trace=True)
            mapped=np.array(r['trajectory'])*t
            affine.append({'gamma':gamma,'iterations':r['iterations'],'mapped_point_error':np.linalg.norm(r['x']*t-base['x']),'trajectory_error':np.max(np.abs(mapped-np.array(base['trajectory']))),'transformed_trajectory':r['trajectory'],'mapped_trajectory':mapped,'objective_values':r['values']})
        result={'optimum':optimum,'sweep':sweep,'condition_gammas':condition_gammas,'condition_numbers':conditions,'affine_invariance':affine,'unscaled_newton':base}
        plt=pyplot();fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].loglog(gammas,[r['iterations'] for r in sweep],'-o');censored=[r for r in sweep if not r['converged']];axes[0].scatter([r['gamma'] for r in censored],[r['iterations'] for r in censored],marker='x',s=80,color='red',label='censored at 5000');axes[0].legend();axes[0].set(xlabel='gamma',ylabel='Gradient updates');axes[1].loglog(condition_gammas,conditions);axes[1].set(xlabel='gamma',ylabel='Hessian condition number');fig.tight_layout();fig.savefig(out/'conditioning.png',dpi=160);plt.close(fig)
        return save_result(eid,out,result,r'f(x)=c^Tx-\sum_{i=1}^{500}\log(b_i-a_i^Tx),\quad x=Tz,\quad T_{jj}=\gamma^{j/100}.',
            'For each prescribed scaling, gradient descent uses the transformed gradient and backtracking. Hessian conditioning is computed at the common optimum. The extreme-scale Newton comparison uses diagonal equilibration, avoiding artificial floating-point failure from condition numbers near 1e40.',
            [(f'gamma {r["gamma"]:.5g}: iterations',r['iterations']) for r in sweep]+[('Reference gradient norm',f'{optimum["gradient_norm"]:.3g}')], [('conditioning.png','Conditioning and iterations')],
            ['The source uses a uniform (0,1) right-hand side. This reproducible replacement uses uniform (0.01,1) to avoid accidental near-zero initialization slacks. Unconverged gradient runs are explicitly censored at the published 5000-update cap.'])
    method='gradient' if eid=='fig_9_6' else 'newton';tol=1e-4 if method=='gradient' else 1e-16
    runs={label:barrier_descent(p,method=method,exact=exact,alpha=.1 if method=='gradient' else .01,tol=tol,maxiter=2000 if method=='gradient' else 100,fstar=fstar) for label,exact in [('backtracking',False),('exact',True)]}
    sensitivity=[]
    for alpha in [.01,.1,.3]:
        for beta in [.1,.5,.8]:
            r=barrier_descent(p,method=method,alpha=alpha,beta=beta,tol=tol,maxiter=2000 if method=='gradient' else 100,fstar=fstar)
            sensitivity.append({'alpha':alpha,'beta':beta,'iterations':r['iterations'],'objective_gap':r['objective']-fstar})
    convergence_plot(out,{label:r['values'] for label,r in runs.items()},fstar)
    plt=pyplot();fig,ax=plt.subplots(figsize=(7,4))
    for label,r in runs.items():ax.plot(r['steps'],label=label)
    ax.set(xlabel='Iteration',ylabel='Step length');ax.legend();fig.tight_layout();fig.savefig(out/'steps.png',dpi=160);plt.close(fig)
    return save_result(eid,out,{'runs':runs,'reference':optimum,'parameter_sweep':sensitivity},r'f(x)=c^Tx-\sum_{i=1}^{500}\log(b_i-a_i^Tx),\quad n=100,\quad x^{(0)}=0.',
        'Gradients and Hessians are formed directly. Exact search roots the directional derivative within the strictly feasible interval. The independent reference is a high-accuracy Newton solution; saved slacks, derivatives and histories permit direct checks. The nine alpha/beta combinations quantify parameter sensitivity.',
        [('Reference objective',f'{fstar:.12g}')]+[(label+' updates',r['iterations']) for label,r in runs.items()]+[(label+' final objective gap',f'{r["objective"]-fstar:.3g}') for label,r in runs.items()],
        [('convergence.png','Objective gap histories'),('steps.png','Accepted step lengths')],
        ['Reference-data substitution: b is uniform (0.01,1), rather than the companion uniform (0,1). Domain tests require strictly positive slacks; the companion permits a small negative slack in an intermediate test, which is unsafe for real logarithms.'])


def run_sparse(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);r=np.random.default_rng(923);n=10000;m=100000;N=200
    A1=sparse.random(400,50,density=.05,random_state=r,data_rvs=r.standard_normal,format='csr')
    A=sparse.vstack([sparse.eye(n),-sparse.eye(n),sparse.random(m-2*n,n,density=1e-6,random_state=r,data_rvs=r.standard_normal,format='csr')+sparse.kron(sparse.eye(N),A1)],format='csr')
    b=np.r_[np.ones(2*n),r.uniform(.001,1,m-2*n)];p=LogBarrier(A,b)
    sparse.save_npz(out/'A.npz',A);np.savez_compressed(out/'instance.npz',b=b,seed=923)
    res=barrier_descent(p,tol=5e-11,maxiter=60);convergence_plot(out,{'sparse Newton':res['values']},res['objective'])
    return save_result('fig_9_23',out,{'run':res,'dimensions':[m,n],'nonzeros':A.nnz},r'f(x)=-\sum_{i=1}^{100000}\log(b_i-a_i^Tx),\quad n=10000.',
        'The full companion dimension is retained. Sparse block-diagonal plus sparse global couplings are used, and every Newton system is solved by sparse direct factorization. No generic convex solver replaces Newton.',
        [('Variables',n),('Total inequalities',m),('Nonzeros',A.nnz),('Newton updates',res['iterations']),('Final squared decrement',f'{res["newton_decrement_squared"]:.4g}'),('Minimum slack',f'{res["minimum_slack"]:.4g}')], [('convergence.png','Full-size sparse Newton convergence')],
        ['Target choice: the companion has 100000 total rows including 20000 box rows; the book states 100000 additional log terms. This implementation follows the source dimensions and explicitly does not certify the larger book interpretation. The replacement b distribution is uniform (0.001,1).'])


def run_self_concordant(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rg=np.random.default_rng(925);records=[];accepted=[]
    for family,(m,n) in enumerate([(100,50),(1000,500),(1000,50)]):
        found=0;attempts=0
        while found<5:
            attempts+=1
            A=rg.standard_normal((m,n));b=rg.uniform(.001,1,m);p=LogBarrier(A,b)
            try: center=barrier_descent(p,alpha=.1,beta=.8,tol=1e-14,maxiter=20)
            except (linalg.LinAlgError,ValueError):continue
            if center['newton_decrement_squared']>1e-12:continue
            found+=1;xstar=center['x'];fstar=center['objective'];y=b-A@xstar
            starts=[]
            for k in range(1,11):
                v=rg.standard_normal(n);dy=-A@v;smax=np.min(-y[dy<0]/dy[dy<0]);relative=dy/y
                maxgap=-np.log1p((1-1e-6)*smax*relative).sum();target=k/10*maxgap
                s=brentq(lambda s:-np.log1p(s*relative).sum()-target,0,(1-1e-6)*smax,xtol=1e-14)
                start=xstar+s*v;starts.append(start);r=barrier_descent(p,start,alpha=.1,beta=.8,tol=5e-11,maxiter=100)
                records.append({'family':[m,n],'problem':found,'start':k,'initial_gap':r['values'][0]-fstar,'iterations':r['iterations'],
                                'final_gap':r['objective']-fstar,'final_decrement_squared':r['newton_decrement_squared'],'minimum_slack':r['minimum_slack']})
            np.savez_compressed(out/f'family{family+1}_problem{found}.npz',A=A,b=b,starts=np.array(starts),xstar=xstar)
        accepted.append({'dimensions':[m,n],'accepted':found,'generated':attempts})
    plt=pyplot();fig,ax=plt.subplots(figsize=(7,4.5))
    for family in [(100,50),(1000,500),(1000,50)]:
        subset=[r for r in records if r['family']==list(family)];ax.scatter([r['initial_gap'] for r in subset],[r['iterations'] for r in subset],label=f'm={family[0]}, n={family[1]}',s=20)
    ax.set(xlabel='Initial objective gap',ylabel='Newton updates');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'self_concordant.png',dpi=160);plt.close(fig)
    return save_result('fig_9_25',out,{'records':records,'accepted_instances':accepted,'seed':925},r'f(x)=-\sum_i\log(b_i-a_i^Tx),\quad(m,n)\in\{(100,50),(1000,500),(1000,50)\}.',
        'Each of the three full-size families contains five independently generated bounded analytic-center problems, with ten prescribed objective-gap starts per problem. A start lies along a random ray from the optimum, selected to attain a chosen fraction of the near-boundary gap. All 150 Newton runs use alpha = 0.1 and beta = 0.8.',
        [('Total runs',len(records)),('Smallest iteration count',min(r['iterations'] for r in records)),('Largest iteration count',max(r['iterations'] for r in records)),('Worst final objective gap',f"{max(r['final_gap'] for r in records):.5g}"),('Worst squared decrement',f"{max(r['final_decrement_squared'] for r in records):.5g}")],
        [('self_concordant.png','All 150 original-size method-reproduction runs')],
        ['The source uses MATLAB v4 seed semantics, unlike the state-based figures elsewhere. New PCG64 arrays are saved explicitly. The center rejection threshold is squared decrement 1e-12 after at most 20 steps; the source uses decrement 1e-8. The replacement right-hand sides use uniform (0.001,1). Rejection is recorded, rather than conflating nonconvergence in 20 steps with a mathematical proof of unboundedness.'])


def run(eid,output_dir=None):
    begin_run(eid)
    out=Path(output_dir or ROOT/'artifacts/examples'/eid)
    if eid in ['fig_9_2','fig_9_3','fig_9_11','fig_9_19']:return run_small(eid,out)
    if eid in ['fig_9_6','fig_9_7','fig_9_21']:return run_dense(eid,out)
    if eid=='fig_9_23':return run_sparse(out)
    if eid=='fig_9_25':return run_self_concordant(out)
    raise ValueError(eid)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('id');a=ap.parse_args();r=run(a.id);print(r['id'],'completed')
