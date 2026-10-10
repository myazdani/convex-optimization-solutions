"""Original KKT/Newton implementations for chapter 10 worked studies."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from scipy import linalg
from scipy.optimize import linprog
from examples.algorithm_common import ROOT,begin_run,save_result,convergence_plot,pyplot
from examples.algorithms_ch09 import LogBarrier,barrier_descent


def infeasible_newton(A,b,x=None,w=None,tol=1e-10,maxiter=100,alpha=.011):
    n=A.shape[1];m=A.shape[0];x=np.ones(n) if x is None else x.copy();w=np.zeros(m) if w is None else w.copy()
    primal=[];dual=[];steps=[];norms=[]
    for k in range(maxiter+1):
        rp=A@x-b;rd=-1/x+A.T@w;nr=np.linalg.norm(np.r_[rd,rp]);primal.append(np.linalg.norm(rp));dual.append(np.linalg.norm(rd));norms.append(nr)
        if nr<tol or k==maxiter:break
        d=x*x;dw=linalg.solve((A*d)@A.T,rp-A@(d*rd),assume_a='pos');dx=-d*(rd+A.T@dw)
        s=1.
        while np.min(x+s*dx)<=0:s*=.5
        while np.linalg.norm(np.r_[-1/(x+s*dx)+A.T@(w+s*dw),A@(x+s*dx)-b])>(1-alpha*s)*nr:
            s*=.5
            if s<1e-16:break
        if s<1e-16:break
        steps.append(s);x+=s*dx;w+=s*dw
    return {'x':x,'w':w,'primal_residuals':primal,'dual_residuals':dual,'residuals':norms,'steps':steps,'iterations':len(steps),
            'primal_residual':primal[-1],'dual_residual':dual[-1],'minimum_x':float(x.min()),'objective':float(-np.log(x).sum())}


def center_instance(infeasible=False):
    # Rejection validates the property on which the book illustration depends.
    for seed in range(1004,1100):
        r=np.random.default_rng(seed);A=r.standard_normal((50,100));x0=r.standard_normal(100) if infeasible else r.uniform(.01,1,100);b=A@x0
        if infeasible:
            check=linprog(np.zeros(100),A_eq=A,b_eq=b,bounds=(0,None),method='highs')
            if check.status==2:
                cert=linprog(np.zeros(50),A_ub=-A.T,b_ub=np.zeros(100),A_eq=b[None,:],b_eq=[-1.],bounds=(None,None),method='highs')
                if cert.success:return A,b,x0,seed,cert.x
        else:
            cert=linprog(np.zeros(50),A_ub=-A.T,b_ub=-np.ones(100),bounds=(None,None),method='highs')
            if cert.success:return A,b,x0,seed,cert.x
    raise RuntimeError('No required random instance found')


def feasible_center(A,b,x):
    x=x.copy();values=[];steps=[]
    for k in range(100):
        g=-1/x;d=x*x;w=linalg.solve((A*d)@A.T,-A@(d*g),assume_a='pos');dx=-d*(g+A.T@w)
        dec=-g@dx;f=-np.log(x).sum();values.append(f)
        if dec/2<1e-12:break
        s=1.
        while np.min(x+s*dx)<=0:s*=.5
        while -np.log1p(s*dx/x).sum()>-.1*s*dec:s*=.5
        x+=s*dx;steps.append(s)
    w=linalg.lstsq(A.T,1/x)[0]
    return {'x':x,'w':w,'values':values,'steps':steps,'iterations':len(steps),'primal_residual':np.linalg.norm(A@x-b),
            'dual_residual':np.linalg.norm(-1/x+A.T@w),'objective':-np.log(x).sum()}


def game_residual(A,c,d,x,y):return np.r_[A@y+c+2*x/(1-x@x),A.T@x+d-2*y/(1-y@y)]

def game_newton(A,c,d,start):
    n=len(c);x=np.full(n,start);y=np.full(n,start);res=[];steps=[]
    for k in range(100):
        r=game_residual(A,c,d,x,y);res.append(np.linalg.norm(r))
        if res[-1]<1e-10:break
        hx=2*np.eye(n)/(1-x@x)+4*np.outer(x,x)/(1-x@x)**2
        hy=-2*np.eye(n)/(1-y@y)-4*np.outer(y,y)/(1-y@y)**2
        delta=linalg.solve(np.block([[hx,A],[A.T,hy]]),-r,assume_a='sym');dx,dy=delta[:n],delta[n:];s=1.
        while np.linalg.norm(x+s*dx)>=1 or np.linalg.norm(y+s*dy)>=1:s*=.5
        while np.linalg.norm(game_residual(A,c,d,x+s*dx,y+s*dy))>(1-.01*s)*res[-1]:s*=.5
        x+=s*dx;y+=s*dy;steps.append(s)
    return {'x':x,'y':y,'residuals':res,'steps':steps,'iterations':len(steps),'residual_norm':res[-1],
            'x_norm':np.linalg.norm(x),'y_norm':np.linalg.norm(y),'value':float(x@A@y+c@x+d@y-np.log1p(-x@x)+np.log1p(-y@y))}


def run(eid,output_dir=None):
    begin_run(eid)
    out=Path(output_dir or ROOT/'artifacts/examples'/eid);out.mkdir(parents=True,exist_ok=True)
    if eid in ['fig_10_1','fig_10_3']:
        infeasible=eid=='fig_10_3';A,b,x0,seed,cert=center_instance(infeasible)
        np.savez_compressed(out/'instance.npz',A=A,b=b,x0=x0,seed=seed,certificate=cert)
        r=infeasible_newton(A,b,maxiter=21 if infeasible else 100)
        convergence_plot(out,{'primal':r['primal_residuals'],'dual':r['dual_residuals']},ylabel='KKT residual norm')
        plt=pyplot();fig,ax=plt.subplots(figsize=(7,4));ax.plot(r['steps'],'-o');ax.set(xlabel='Iteration',ylabel='Step length');fig.tight_layout();fig.savefig(out/'steps.png',dpi=160);plt.close(fig)
        check={'min_A_transpose_certificate':float((A.T@cert).min()),'b_dot_certificate':float(b@cert)}
        return save_result(eid,out,{'run':r,'certificate':check,'seed':seed,'dimensions':[50,100]},r'\min_x -\sum_{j=1}^{100}\log x_j\quad\mathrm{subject\ to}\quad Ax=b,\quad x>0.',
            'Newton solves the primal-dual KKT residual by eliminating the diagonal Hessian. The line search decreases the combined residual while retaining positive x. A separate linear program checks the required feasibility property of each generated instance.',
            [('Accepted random seed',seed),('Updates',r['iterations']),('Primal residual',f'{r["primal_residual"]:.6g}'),('Dual residual',f'{r["dual_residual"]:.6g}'),('Smallest x',f'{r["minimum_x"]:.6g}'),('Last step',f'{r["steps"][-1]:.6g}'),('Certificate minimum A-transpose y',f'{check["min_A_transpose_certificate"]:.6g}'),('Certificate b-transpose y',f'{check["b_dot_certificate"]:.6g}')],
            [('convergence.png','Primal and dual residuals'),('steps.png','Step lengths')],
            ['The substituted random family is conditioned by rejection: feasible experiments require a positive primal point and a strictly positive dual image; infeasible experiments require a Farkas certificate with A-transpose y nonnegative and b-transpose y = -1. This prevents assuming that a new random draw has the same feasibility outcome as the original MATLAB draw.'] )
    if eid=='fig_10_5':
        rg=np.random.default_rng(1005);A=rg.standard_normal((100,100));c=rg.standard_normal(100);d=rg.standard_normal(100)
        runs={'book zero start':game_newton(A,c,d,0.),'source 0.01 start':game_newton(A,c,d,.01)}
        np.savez_compressed(out/'instance.npz',A=A,c=c,d=d,seed=1005)
        convergence_plot(out,{k:v['residuals'] for k,v in runs.items()},ylabel='Saddle KKT residual')
        return save_result(eid,out,{'runs':runs},r'f(x,y)=x^TAy+c^Tx+d^Ty-\log(1-x^Tx)+\log(1-y^Ty),\quad \min_x\max_y f(x,y).',
            'The root of the convex-concave saddle gradient is found with explicit block Newton steps and residual backtracking. Both players remain inside their open unit balls. Convexity in x and concavity in y make a stationary point a global saddle.',
            [(label+' updates',r['iterations']) for label,r in runs.items()]+[(label+' residual',f'{r["residual_norm"]:.4g}') for label,r in runs.items()], [('convergence.png','Residual convergence from both documented starts')],
            ['The book starts at zero and the companion starts at 0.01 in every coordinate. Both are evaluated on the same replacement instance and reported separately.'])
    if eid!='fig_10_6':raise ValueError(eid)
    rg=np.random.default_rng(1006);m,n=100,500;A=rg.standard_normal((m,n));z0=rg.standard_normal(m)
    A+=np.outer(z0,1-A.T@z0)/(z0@z0);x0=rg.uniform(.01,1,n);b=A@x0
    np.savez_compressed(out/'instance.npz',A=A,b=b,x0=x0,z0=z0,seed=1006)
    runs={'feasible primal':[],'dual':[],'infeasible KKT':[]}
    for _ in range(4):
        v=rg.standard_normal(n);v-=A.T@linalg.solve(A@A.T,A@v,assume_a='pos');x=x0+.99/np.max(-v/x0)*v
        runs['feasible primal'].append(feasible_center(A,b,x))
        v=rg.standard_normal(m);z=z0+.99/np.max(-(A.T@v)/(A.T@z0))*v
        q=LogBarrier(-A.T,np.zeros(n),b);r=barrier_descent(q,x0=z,tol=1e-12,alpha=.1)
        r['primal_recovered']=1/(A.T@r['x']);r['primal_residual']=np.linalg.norm(A@r['primal_recovered']-b);runs['dual'].append(r)
        runs['infeasible KKT'].append(infeasible_newton(A,b,rg.uniform(.01,1,n),rg.standard_normal(m),tol=1e-8,alpha=.1))
    fstar=min(r['objective'] for r in runs['feasible primal']);plots=[]
    for label,group in runs.items():
        fn=label.replace(' ','_')+'.png'
        vals={f'start {j+1}':r['residuals'] if label=='infeasible KKT' else np.array(r['values'])-(r['objective'] if label=='dual' else fstar) for j,r in enumerate(group)}
        convergence_plot(out,vals,filename=fn,ylabel='KKT residual' if label=='infeasible KKT' else 'Objective error');plots.append((fn,label+' convergence'))
    rows=[]
    for label,group in runs.items():
        for j,r in enumerate(group):rows.extend([(f'{label} start {j+1}: updates',r['iterations']),(f'{label} start {j+1}: primal residual',f'{r["primal_residual"]:.4g}')])
    return save_result(eid,out,{'runs':runs,'reference_objective':fstar,'dimensions':[100,500]},r'\min_{x>0,\ Ax=b}-\sum_j\log x_j,\qquad\min_{A^Tz>0}b^Tz-\sum_j\log(A^Tz)_j.',
        'Four near-boundary starts are used for each feasible primal and dual formulation, and four positive infeasible starts for the KKT formulation. Feasible primal steps lie in the equality nullspace. Dual minimizers recover primal points by x = 1/(A-transpose z).',rows,plots,
        ['Corrections to the published MATLAB script: the feasible-primal Armijo expression uses log(x+s dx), not log(1+x+s dx); its BETA*S typo is corrected; both primal and dual Armijo right-hand sides include the step length. These are implementation errors in the source, not changes to the textbook algorithms.'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('id');a=p.parse_args();r=run(a.id);print(r['id'],'completed')
