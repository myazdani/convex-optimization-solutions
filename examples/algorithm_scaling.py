"""Full-size, resumable Monte Carlo studies for chapter 11."""
from __future__ import annotations
import json
import hashlib,datetime,sys,platform,scipy
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy import linalg
from examples.algorithm_common import ROOT,begin_run,save_result,pyplot,plain


def standard_instance(m,seed):
    r=np.random.default_rng(seed);n=2*m;A=r.standard_normal((m,n));x=r.random(n);b=A@x;y=r.standard_normal(m);s=r.random(n);c=A.T@y+s
    return A,b,c,x,y,s


def standard_barrier(A,b,c,x0,mu=100,stop_t=10000,centering_tol=1e-5):
    x=x0.copy();t=1.;steps=0;history=[]
    cost_shift=linalg.solve(A@A.T,A@c,assume_a='pos');cost=c-A.T@cost_shift
    for outer in range(20):
        before=steps
        for _ in range(100):
            d=x*x;g=t*cost-1/x
            z=linalg.solve((A*d)@A.T,-A@(d*g),assume_a='pos');dx=-d*(g+A.T@z);dec=np.sum(dx*dx/d)
            if dec/2<centering_tol:break
            a=1.
            while (x+a*dx).min()<=0:a*=.5
            gd=g@dx
            while a*((t*cost+A.T@z)@dx)-np.log1p(a*dx/x).sum()>.01*a*gd:a*=.5
            x+=a*dx;steps+=1
        else:raise RuntimeError('Standard LP centering did not converge')
        y=-z/t+cost_shift;s=c-A.T@y;gap=x@s
        history.append({'t':t,'iterations':steps,'centering_steps':steps-before,'gap':gap,'objective':c@x,
                        'primal_residual':np.linalg.norm(A@x-b),'minimum_dual':s.min()})
        if t>=stop_t:break
        t=min(t*mu,stop_t)
    return {'x':x,'y':y,'s':s,'gap':gap,'objective':c@x,'primal_residual':np.linalg.norm(A@x-b),
            'dual_residual':np.linalg.norm(A.T@y+s-c),'minimum_x':x.min(),'minimum_dual':s.min(),
            'iterations':steps,'path_iterations':steps-history[0]['centering_steps'],'history':history,
            'gap_reduction':history[0]['gap']/gap}


def standard_primal_dual(A,b,c,x0,y0,s0,tol=1e-8):
    x=x0.copy();y=y0.copy();s=s0.copy();history=[]
    for k in range(500):
        rp=A@x-b;rd=A.T@y+s-c;gap=x@s;nr=max(np.linalg.norm(rp),np.linalg.norm(rd))
        history.append({'iteration':k,'gap':gap,'primal_residual':np.linalg.norm(rp),'dual_residual':np.linalg.norm(rd),'objective':c@x})
        if gap<tol and nr<tol:break
        tau=gap/(len(x)*10);rc=x*s-tau;d=x/s
        rhs=-rp+A@(rc/s-d*rd)
        if gap<1e-4:
            # The scaled symmetric augmented system avoids squared conditioning
            # and preserves all three Newton equations near complementarity.
            D=np.sqrt(d);AD=A*D
            K=np.block([[np.eye(len(x)),-AD.T],[-AD,np.zeros((len(y),len(y)))]])
            sol=linalg.solve(K,np.r_[D*(-rc/x+rd),rp],assume_a='sym')
            dx=D*sol[:len(x)];dy=sol[len(x):];ds=-rd-A.T@dy
        else:
            dy=linalg.solve((A*d)@A.T,rhs,assume_a='pos');ds=-rd-A.T@dy;dx=(-rc-x*ds)/s
        a=1.
        for v,dv in [(x,dx),(s,ds)]:
            inds=dv<0
            if inds.any():a=min(a,.99*np.min(-v[inds]/dv[inds]))
        normr=np.linalg.norm(np.r_[rp,rd,rc])
        while True:
            xn=x+a*dx;yn=y+a*dy;sn=s+a*ds
            rn=np.r_[A@xn-b,A.T@yn+sn-c,xn*sn-tau]
            if np.linalg.norm(rn)<=(1-.01*a)*normr:break
            a*=.5
            if a<1e-16:raise RuntimeError('Standard primal-dual stalled')
        x=xn;y=yn;s=sn
    else:raise RuntimeError('Standard primal-dual did not converge')
    return {'x':x,'y':y,'s':s,'iterations':len(history)-1,'gap':gap,'primal_residual':np.linalg.norm(A@x-b),
            'dual_residual':np.linalg.norm(A.T@y+s-c),'minimum_x':x.min(),'minimum_dual':s.min(),'objective':c@x,'history':history}


def sdp_matrix(n,seed):
    rg=np.random.default_rng(seed);lower=np.tril(rg.standard_normal((n,n)));B=lower+lower.T-np.diag(np.diag(lower))
    B/=np.max(abs(linalg.eigvalsh(B)));return B


def diagonal_sdp(B,mu=20,stop_t=8000,centering_tol=1e-5):
    n=len(B);x=np.full(n,1.1*(-linalg.eigvalsh(B,subset_by_index=[0,0])[0]+.1));t=1.;steps=0;history=[];I=np.eye(n)
    for outer in range(10):
        before=steps
        for _ in range(100):
            X=B+np.diag(x);L=linalg.cholesky(X,lower=True);invX=linalg.cho_solve((L,True),I);g=t-np.diag(invX);H=invX*invX
            dx=linalg.solve(H,-g,assume_a='pos');dec=-g@dx
            if dec/2<centering_tol:break
            a=1.;logdet=2*np.log(np.diag(L)).sum()
            while True:
                try:
                    Ln=linalg.cholesky(X+np.diag(a*dx),lower=True)
                    diff=t*a*dx.sum()-2*np.log(np.diag(Ln)).sum()+logdet
                    if diff<=-.01*a*dec:break
                except linalg.LinAlgError:pass
                a*=.5
                if a<1e-16:raise RuntimeError('Diagonal SDP line search stalled')
            x+=a*dx;steps+=1
        else:raise RuntimeError('Diagonal SDP centering did not converge')
        Z=(invX-(invX*dx)@invX)/t;gap=np.sum((B+np.diag(x))*Z);dual=-np.sum(B*Z)
        history.append({'t':t,'iterations':steps,'centering_steps':steps-before,'gap':gap,'objective':x.sum(),
                        'dual_residual':np.max(abs(np.diag(Z)-1))})
        if t>=stop_t:break
        t=min(t*mu,stop_t)
    # Positive definiteness of Z follows from I - X^-1/2 diag(dx) X^-1/2 positive definite.
    return {'x':x,'dual':Z,'iterations':steps,'path_iterations':steps-history[0]['centering_steps'],'history':history,
            'objective':x.sum(),'dual_objective':dual,'gap':gap,'gap_reduction':history[0]['gap']/gap,
            'dual_residual':np.max(abs(np.diag(Z)-1)),
            'minimum_primal_eigenvalue':linalg.eigvalsh(B+np.diag(x),subset_by_index=[0,0])[0],
            'minimum_dual_eigenvalue':linalg.eigvalsh(Z,subset_by_index=[0,0])[0]}


def trial(job):
    kind,n,index,seed=job;t0=time.monotonic();started=datetime.datetime.now(datetime.timezone.utc).isoformat();code_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if kind=='lp':
        A,b,c,x,y,s=standard_instance(n,seed);br=standard_barrier(A,b,c,x);pd=standard_primal_dual(A,b,c,np.ones_like(x),np.zeros_like(y),np.ones_like(s))
        result={'barrier':{k:v for k,v in br.items() if k not in ['x','y','s']},'primal_dual':{k:v for k,v in pd.items() if k not in ['x','y','s']}}
    else:
        B=sdp_matrix(n,seed);sd=diagonal_sdp(B);result={'barrier':{k:v for k,v in sd.items() if k not in ['x','dual']}}
    return plain({'kind':kind,'dimension':n,'replicate':index,'seed':seed,'seconds':time.monotonic()-t0,'executed_at':started,'code_sha256':code_hash,'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,**result})


def run_sweep(kind,workers=2):
    for eid in (['fig_11_8','fig_11_23'] if kind=='lp' else ['fig_11_20']):begin_run(eid)
    folder=ROOT/'artifacts/examples'/('_lp_scaling_trials' if kind=='lp' else '_sdp_scaling_trials');folder.mkdir(parents=True,exist_ok=True)
    sizes=np.rint(np.geomspace(10,1000,20)).astype(int).tolist();jobs=[]
    for n in sizes:
        for j in range(100):
            seed=(11800000 if kind=='lp' else 11200000)+n*100+j
            existing=folder/f'{n}_{j}.json'
            if not existing.exists() or json.loads(existing.read_text()).get('code_sha256')!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():jobs.append((kind,n,j,seed))
    if jobs:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(trial,job):job for job in jobs}
            for i,future in enumerate(as_completed(futures)):
                r=future.result();(folder/f'{r["dimension"]}_{r["replicate"]}.json').write_text(json.dumps(r,indent=2)+'\n')
                if i%20==0:print(kind,i+1,'/',len(jobs),r['dimension'],flush=True)
    records=[json.loads((folder/f'{n}_{j}.json').read_text()) for n in sizes for j in range(100)]
    for eid,method in ([('fig_11_8','barrier'),('fig_11_23','primal_dual')] if kind=='lp' else [('fig_11_20','barrier')]):
        out=ROOT/'artifacts/examples'/eid;out.mkdir(parents=True,exist_ok=True);summary=[]
        for n in sizes:
            group=[r[method] for r in records if r['dimension']==n];counts=[r['path_iterations'] if method=='barrier' else r['iterations'] for r in group]
            summary.append({'dimension':n,'count':len(group),'mean_iterations':np.mean(counts),'std_iterations':np.std(counts,ddof=1),'min_iterations':min(counts),'max_iterations':max(counts),
                            'maximum_gap':max(r['gap'] for r in group),'maximum_dual_residual':max(r['dual_residual'] for r in group)})
        plt=pyplot();fig,ax=plt.subplots(figsize=(7,4.5));ax.errorbar(sizes,[r['mean_iterations'] for r in summary],yerr=[r['std_iterations'] for r in summary],fmt='-o',capsize=3);ax.set_xscale('log');ax.set(xlabel='Equality constraints m' if kind=='lp' else 'Matrix order n',ylabel='Mean Newton updates ± sample SD');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'scaling.png',dpi=160);plt.close(fig)
        formula=r'\min c^Tx\quad\mathrm{subject\ to}\quad Ax=b,\ x\geq0,\quad n=2m.' if kind=='lp' else r'\min\mathbf1^Tx\quad\mathrm{subject\ to}\quad B+\mathrm{diag}(x)\succeq0.'
        description=('All 2000 Gaussian standard-form LPs are generated once and reused by both the barrier and primal-dual sweeps. Primal and dual strictly feasible points are constructed. '
                     if kind=='lp' else 'All 2000 dense symmetric matrices have independent Gaussian lower triangles, reflected upper triangles and spectral norm one. ')
        description+=('The barrier runs start on the t = 1 central path; reported counts exclude initial centering. The original book stopping rule is a gap reduction of 10000 for LP and 8000 for SDP, not a fixed absolute tolerance.'
                      if method=='barrier' else 'Primal-dual runs start from x = 1, inequality multipliers = 1 and equality multipliers = 0, as specified for Figure11.23, and require both absolute gap and feasibility residuals below 1e-8, using mu = 10 and positivity/residual backtracking. Below gap 1e-4 the same Newton equations are solved as a scaled symmetric augmented system, avoiding the squared conditioning of normal equations and preserving all residual equations at high accuracy.')
        save_result(eid,out,{'summary':summary,'records':records,'trial_count':len(records),'dimension_count':len(sizes),'method':method},formula,description,
            [(f'n={r["dimension"]}: mean ± SD',f'{r["mean_iterations"]:.3f} ± {r["std_iterations"]:.3f}') for r in summary], [('scaling.png','Full 2000-instance scaling experiment')],
            ['The original 20 dimension values and seeds are not published; the replacement uses 20 rounded geometric spacings from 10 through 1000. Every trial seed and independently recomputed feasibility/certificate check is saved. No sample count or maximum dimension is reduced.',
             'The SDP companion zeros the diagonal of its random matrix, whereas the textbook samples the diagonal normally. These scaling experiments follow the textbook distribution.'] if kind=='sdp' else
            ['The original 20 dimension values and seeds are not published; the replacement uses 20 rounded geometric spacings from 10 through 1000. Every trial seed and independently recomputed feasibility/certificate check is saved. No sample count or maximum dimension is reduced.'])
    return records


def run_three(eid,out):
    begin_run(eid)
    out=Path(out);out.mkdir(parents=True,exist_ok=True);runs={};kind='lp' if eid=='fig_11_7' else 'sdp'
    for n in [50,500,1000]:
        seed=(117000 if kind=='lp' else 111900)+n
        if kind=='lp':
            A,b,c,x,y,s=standard_instance(n,seed);r=standard_barrier(A,b,c,x);np.savez_compressed(out/f'instance_{n}.npz',A=A,b=b,c=c,x0=x,y0=y,s0=s,seed=seed)
        else:
            B=sdp_matrix(n,seed);r=diagonal_sdp(B);np.savez_compressed(out/f'instance_{n}.npz',B=B,seed=seed)
        runs[str(n)]=r
    plt=pyplot();fig,ax=plt.subplots(figsize=(7,4.3))
    for n,r in runs.items():
        h=r['history'];initial=h[0]['iterations'];ax.step([v['iterations']-initial for v in h],[v['gap'] for v in h],where='post',label=('m=' if kind=='lp' else 'n=')+n)
    ax.set_yscale('log');ax.set(xlabel='Newton updates after initial centering',ylabel='Certified duality gap');ax.legend();fig.tight_layout();fig.savefig(out/'progress.png',dpi=160);plt.close(fig)
    return save_result(eid,out,{'runs':runs},r'\min c^Tx\quad\mathrm{subject\ to}\quad Ax=b,\ x\geq0,\quad n=2m.' if kind=='lp' else r'\min\mathbf1^Tx\quad\mathrm{subject\ to}\quad B+\mathrm{diag}(x)\succeq0.',
        'The three full-size runs preserve the textbook dimensions and use custom Newton centering with backtracking. Initial centering is excluded from plotted update counts. Explicit feasible dual certificates give the reported objective gaps.',
        [(n+': updates after centering',r['path_iterations']) for n,r in runs.items()]+[(n+': gap reduction',f'{r["gap_reduction"]:.7g}') for n,r in runs.items()], [('progress.png','Full-size three-instance comparison')],
        ['The textbook stops after relative gap reduction 10000 (LP) or 8000 (SDP); the companion instead requests absolute gaps 0.001 or 0.0001. This implementation follows the book relative-reduction experiment. For SDP it also retains the textbook random diagonal, unlike the companion zero-diagonal construction.'])

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('kind',choices=['lp','sdp']);ap.add_argument('--workers',type=int,default=2);a=ap.parse_args();run_sweep(a.kind,a.workers)
