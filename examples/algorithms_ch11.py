"""Explicit barrier and primal-dual methods for chapter 11 numerical studies."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from scipy import linalg
from scipy.optimize import linprog
from scipy.special import logsumexp,softmax
from examples.algorithm_common import ROOT,begin_run,save_result,convergence_plot,pyplot

class LinearProgram:
    def __init__(self,A,b,c):self.A=A;self.b=b;self.c=c;self.degree=len(b)
    def evaluate(self,x,t,derivatives=True):
        s=self.b-self.A@x
        if s.min()<=0:return np.inf,None,None
        val=t*self.c@x-np.log(s).sum()
        if not derivatives:return val,None,None
        return val,t*self.c+self.A.T@(1/s),(self.A.T/s**2)@self.A
    def certificate(self,x,t,dx):
        s=self.b-self.A@x;z=(1+self.A@dx/s)/(t*s)
        return {'dual':z,'objective':self.c@x,'gap':s@z,'stationarity':np.linalg.norm(self.c+self.A.T@z),
                'minimum_slack':s.min(),'minimum_dual':z.min(),'dual_objective':-self.b@z}

class GeometricProgram:
    def __init__(self,A,b):self.A=A;self.b=b;self.degree=len(b)-1
    def values(self,x,derivatives=True):
        y=np.einsum('ikj,j->ik',self.A,x)+self.b;f=logsumexp(y,axis=1)
        if not derivatives:return f,None,None
        w=softmax(y,axis=1);g=np.einsum('ik,ikj->ij',w,self.A)
        H=np.einsum('ik,ikj,ikl->ijl',w,self.A,self.A)-np.einsum('ij,ik->ijk',g,g)
        return f,g,H
    def evaluate(self,x,t,derivatives=True):
        f,g,H=self.values(x,derivatives)
        if f[1:].max()>=0:return np.inf,None,None
        val=t*f[0]-np.log(-f[1:]).sum()
        if not derivatives:return val,None,None
        weights=np.r_[t,-1/f[1:]]
        return val,weights@g,np.einsum('i,ijk->jk',weights,H)+(g[1:].T/f[1:]**2)@g[1:]
    def certificate(self,x,t,dx):
        f,g,H=self.values(x);lam=-1/(t*f[1:]);y=np.einsum('ikj,j->ik',self.A,x)+self.b;w=softmax(y,axis=1)
        dy=np.einsum('ikj,j->ik',self.A,dx);mean=np.sum(w*dy,axis=1);dw=w*(dy-mean[:,None]);nu=np.empty_like(w)
        nu[0]=w[0]+dw[0];nu[1:]=lam[:,None]*(w[1:]+dw[1:])+(mean[1:]/(t*f[1:]**2))[:,None]*w[1:]
        masses=nu.sum(axis=1);dualobj=np.sum(self.b*nu)-np.sum(nu*np.log(nu/masses[:,None]))
        return {'objective':f[0],'nominal_gap':self.degree/t,'gap':f[0]-dualobj,'dual_objective':dualobj,
                'stationarity':np.linalg.norm(np.einsum('ik,ikj->j',nu,self.A)),
                'minimum_slack':-f[1:].max(),'dual':masses[1:],'nu':nu,'minimum_dual':nu.min()}

class ConeProgram:
    def __init__(self,A,b,c):self.A=A;self.b=b;self.c=c;self.degree=2*len(b)
    def evaluate(self,x,t,derivatives=True):
        y=np.einsum('ikj,j->ik',self.A,x)+self.b;v=y[:,-1];s=v*v-np.sum(y[:,:-1]**2,axis=1)
        if v.min()<=0 or s.min()<=0:return np.inf,None,None
        val=t*self.c@x-np.log(s).sum()
        if not derivatives:return val,None,None
        signs=np.r_[np.ones(self.A.shape[1]-1),-1.];d=2*y*signs/s[:,None];blockg=np.einsum('ik,ikj->ij',d,self.A)
        H=np.einsum('ik,ikj,ikl->jl',2*signs[None,:]/s[:,None],self.A,self.A)+blockg.T@blockg
        return val,t*self.c+blockg.sum(axis=0),H
    def certificate(self,x,t,dx):
        y=np.einsum('ikj,j->ik',self.A,x)+self.b;s=y[:,-1]**2-np.sum(y[:,:-1]**2,axis=1)
        signs=np.r_[np.ones(self.A.shape[1]-1),-1.];jy=y*signs;dy=np.einsum('ikj,j->ik',self.A,dx)
        d=2*jy/s[:,None];dd=2*dy*signs/s[:,None]+4*jy*(np.sum(jy*dy,axis=1)/s**2)[:,None];z=-(d+dd)/t
        return {'objective':self.c@x,'nominal_gap':self.degree/t,'gap':np.sum(y*z),'dual':z,
                'stationarity':np.linalg.norm(self.c-np.einsum('ik,ikj->j',z,self.A)),
                'minimum_slack':np.min(y[:,-1]-np.linalg.norm(y[:,:-1],axis=1)),
                'minimum_dual':np.min(z[:,-1]-np.linalg.norm(z[:,:-1],axis=1)),
                'dual_objective':-np.sum(self.b*z)}

class SemidefiniteProgram:
    def __init__(self,A,B,c):self.A=A;self.B=B;self.c=c;self.degree=len(B)
    def evaluate(self,x,t,derivatives=True):
        X=self.B-np.einsum('i,ijk->jk',x,self.A)
        try:L=linalg.cholesky(X,lower=True)
        except linalg.LinAlgError:return np.inf,None,None
        val=t*self.c@x-2*np.log(np.diag(L)).sum()
        if not derivatives:return val,None,None
        # vec(L^-1 Ai L^-T) Gram matrix is the exact logdet Hessian.
        invL=linalg.solve_triangular(L,np.eye(len(L)),lower=True)
        scaled=invL[None,:,:]@self.A@invL.T[None,:,:]
        g=t*self.c+np.trace(scaled,axis1=1,axis2=2);F=scaled.reshape(len(self.A),-1)
        return val,g,F@F.T
    def certificate(self,x,t,dx):
        X=self.B-np.einsum('i,ijk->jk',x,self.A);invX=linalg.cho_solve(linalg.cho_factor(X),np.eye(len(X)))
        # Newton correction makes linear dual equations exact up to solve accuracy.
        Z=(invX+invX@np.einsum('i,ijk->jk',dx,self.A)@invX)/t
        return {'objective':self.c@x,'dual':Z,'gap':np.sum(X*Z),'stationarity':np.linalg.norm(self.c+np.einsum('ijk,jk->i',self.A,Z)),
                'minimum_slack':linalg.eigvalsh(X,subset_by_index=[0,0])[0],
                'minimum_dual':linalg.eigvalsh(Z,subset_by_index=[0,0])[0],'dual_objective':-np.sum(self.B*Z)}


def barrier(problem,x0,mu=10,tol=1e-6,stop_t=None,initial_t=1.,centering_tol=1e-5,max_updates=500):
    x=x0.copy();t=initial_t;iterations=0;history=[];steps=[]
    while iterations<=max_updates:
        count=0
        for _ in range(100):
            val,g,H=problem.evaluate(x,t);dx=linalg.solve(H,-g,assume_a='pos');dec=-g@dx
            if dec/2<centering_tol:break
            s=1.
            while problem.evaluate(x+s*dx,t,False)[0]>val+.01*s*(g@dx):
                s*=.5
                if s<1e-16:raise RuntimeError('Barrier line search stalled')
            x+=s*dx;steps.append(s);iterations+=1;count+=1
        else:raise RuntimeError('Centering did not converge')
        cert=problem.certificate(x,t,dx);gap=float(cert.get('gap',cert.get('nominal_gap',problem.degree/t)))
        history.append({'t':t,'iterations':iterations,'centering_steps':count,'nominal_gap':problem.degree/t,'gap':gap,
                        'stationarity':float(cert['stationarity']),'objective':float(cert['objective']),'decrement_squared':float(dec)})
        if gap<=tol or (stop_t is not None and t>=stop_t):break
        t=min(t*mu,(problem.degree+1)/tol) if stop_t is None else min(t*mu,stop_t)
    if iterations>max_updates:raise RuntimeError('Barrier maximum iterations reached')
    cert.update({'x':x,'t':t,'iterations':iterations,'history':history,'steps':steps,'excluded_initial_steps':history[0]['centering_steps'],
                 'path_iterations':iterations-history[0]['centering_steps']})
    return cert


def inequality_instance(seed=1104):
    r=np.random.default_rng(seed);A=r.standard_normal((100,50));b=r.uniform(.01,1,100);z=r.uniform(.01,1,100);c=-A.T@z
    ref=linprog(c,A_ub=A,b_ub=b,bounds=(None,None),method='highs');assert ref.success
    shift=(1-ref.fun)/(c@c);b=b+shift*(A@c);x0=shift*c
    return LinearProgram(A,b,c),x0


def geometric_instance(seed=1106):
    r=np.random.default_rng(seed);nu=r.uniform(.01,1,505);nu[:5]/=nu[:5].sum();A=r.standard_normal((505,50));A-=np.outer(nu,nu@A)/(nu@nu)
    A=A.reshape(101,5,50);x0=r.standard_normal(50);b=np.empty((101,5));b[0]=r.standard_normal(5);b[1:]=-np.einsum('ikj,j->ik',A[1:],x0)-r.uniform(.01,1,100)[:,None]-np.log(5)
    p=GeometricProgram(A,b);reference=barrier(p,x0,mu=10,tol=1e-7);b[0]+=1-reference['objective'];return p,x0


def cone_instance(seed=1115):
    r=np.random.default_rng(seed);A=r.standard_normal((50,6,50));x0=r.standard_normal(50);z=r.standard_normal((50,6));z[:,-1]=np.linalg.norm(z[:,:-1],axis=1)+r.uniform(.01,1,50)
    c=np.einsum('ik,ikj->j',z,A);b=r.standard_normal((50,6));b[:,-1]=0;y=np.einsum('ikj,j->ik',A,x0)+b;b[:,-1]=np.linalg.norm(y[:,:-1],axis=1)-y[:,-1]+r.uniform(.01,1,50)
    p=ConeProgram(A,b,c);ref=barrier(p,x0,tol=1e-7);shift=(1-ref['objective'])/(c@c);p.b-=shift*np.einsum('ikj,j->ik',A,c);x0+=shift*c
    return p,x0


def sdp_instance(seed=1117,n=100,m=100):
    r=np.random.default_rng(seed);A=r.standard_normal((n,m,m));A=(A+A.transpose(0,2,1))/2
    Q=linalg.qr(r.standard_normal((m,m)))[0];B=(Q*r.uniform(.01,1,m))@Q.T
    Q=linalg.qr(r.standard_normal((m,m)))[0];Z=(Q*r.uniform(.01,1,m))@Q.T;c=-np.einsum('ijk,jk->i',A,Z)
    p=SemidefiniteProgram(A,B,c);x0=np.zeros(n);ref=barrier(p,x0,tol=1e-7);shift=(1-ref['objective'])/(c@c);p.B+=shift*np.einsum('i,ijk->jk',c,A);x0=shift*c
    return p,x0


def primal_dual(problem,x0,mu=10,tol=1e-8):
    x=x0.copy()
    if isinstance(problem,LinearProgram):
        evaluate=lambda x:(np.r_[problem.c@x,problem.A@x-problem.b],np.vstack([problem.c,problem.A]),np.zeros((len(problem.b)+1,len(x),len(x))))
    else:evaluate=problem.values
    f,g,H=evaluate(x);z=-1/f[1:];hist=[];steps=[]
    for k in range(500):
        f,g,H=evaluate(x);s=-f[1:];rd=g[0]+z@g[1:];gap=s@z;hist.append({'iteration':k,'gap':gap,'residual':np.linalg.norm(rd),'objective':f[0]})
        if gap<tol and np.linalg.norm(rd)<tol:break
        tau=gap/(problem.degree*mu);rc=s*z-tau;J=g[1:]
        Hess=H[0]+np.einsum('i,ijk->jk',z,H[1:]);M=Hess+(J.T*(z/s))@J
        dx=linalg.solve(M,-rd+J.T@(rc/s),assume_a='pos');dz=(-rc+z*(J@dx))/s
        a=1.;neg=dz<0
        if neg.any():a=min(a,.99*np.min(-z[neg]/dz[neg]))
        normr=np.linalg.norm(np.r_[rd,rc])
        while True:
            fn,gn,_=evaluate(x+a*dx);zn=z+a*dz;sn=-fn[1:]
            if sn.min()>0:
                rn=np.r_[gn[0]+zn@gn[1:],sn*zn-tau]
                if np.linalg.norm(rn)<=(1-.01*a)*normr:break
            a*=.5
            if a<1e-16:raise RuntimeError('Primal-dual line search stalled')
        x+=a*dx;z+=a*dz;steps.append(a)
    else:raise RuntimeError('Primal-dual maximum iterations')
    return {'x':x,'dual':z,'history':hist,'steps':steps,'iterations':len(steps),'objective':f[0],'gap':gap,
            'stationarity':np.linalg.norm(rd),'minimum_slack':s.min(),'minimum_dual':z.min()}


def _progress(out,runs,filename='progress.png'):
    plt=pyplot();fig,ax=plt.subplots(figsize=(7,4.4))
    for label,r in runs.items():
        h=r['history'];x=[v.get('iterations',v.get('iteration')) for v in h];g=[v['gap'] for v in h]
        ax.step(x,np.maximum(g,1e-15),where='post',label=label);ax.scatter(x,g,s=10)
    ax.set_yscale('log');ax.set(xlabel='Newton updates',ylabel='Duality gap');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/filename,dpi=160);plt.close(fig)


def run_small(eid,out):
    if eid in ['fig_11_4','fig_11_21']:p,x0=inequality_instance();formula=r'\min_x c^Tx\quad\mathrm{subject\ to}\quad Ax\leq b,\quad A\in\mathbb R^{100\times50}.'
    elif eid in ['fig_11_6','fig_11_22']:p,x0=geometric_instance();formula=r'\min_x\log\sum_{k=1}^5 e^{a_{0k}^Tx+b_{0k}}\quad\mathrm{subject\ to}\quad\log\sum_{k=1}^5 e^{a_{ik}^Tx+b_{ik}}\leq0,\quad i=1,\ldots,100.'
    elif eid=='fig_11_15':p,x0=cone_instance();formula=r'\min_x c^Tx\quad\mathrm{subject\ to}\quad\|A_i x+b_i\|_2\leq c_i^Tx+d_i,\quad i=1,\ldots,50,\quad x\in\mathbb R^{50}.'
    else:p,x0=sdp_instance();formula=r'\min_x c^Tx\quad\mathrm{subject\ to}\quad\sum_{i=1}^{100}x_i A_i\preceq B,\quad A_i,B\in\mathbb S^{100}.'
    data={'A':p.A,'b':p.b if hasattr(p,'b') else p.B,'x0':x0}
    if hasattr(p,'c'):data['c']=p.c
    np.savez_compressed(out/'instance.npz',**data)
    notes=[];sweep=[]
    if eid in ['fig_11_21','fig_11_22']:
        r=primal_dual(p,x0);runs={'primal-dual':r};_progress(out,runs)
        convergence_plot(out,{'dual residual':[v['residual'] for v in r['history']]},filename='residual.png',ylabel='Dual feasibility residual')
        plots=[('progress.png','Surrogate duality gap'),('residual.png','Dual feasibility residual')]
        explanation='The primal-dual direction solves the linearized KKT equations by Schur elimination. Each update preserves strictly positive inequality slacks and multipliers and decreases the combined centering and dual residual. The barrier parameter is recomputed from the current surrogate gap with mu = 10.'
    else:
        xc=barrier(p,x0,tol=p.degree+1)['x'];muvalues=[2,50,200 if eid=='fig_11_15' else 150]
        runs={f'mu={mu}':barrier(p,xc,mu=mu,tol=1e-6) for mu in muvalues};_progress(out,runs)
        if True:
            muvals=[1.5,2,4,6,8]+list(range(10,121 if eid=='fig_11_17' else 201,10))
            for mu in muvals:
                r=barrier(p,xc,mu=mu,tol=1e-3);sweep.append({'mu':mu,'iterations':r['iterations'],'final_gap':r.get('gap',r.get('nominal_gap'))})
            plt=pyplot();fig,ax=plt.subplots(figsize=(7,4));ax.plot([r['mu'] for r in sweep],[r['iterations'] for r in sweep],'-o');ax.set(xlabel='mu',ylabel='Newton updates to gap 0.001');fig.tight_layout();fig.savefig(out/'mu_sweep.png',dpi=160);plt.close(fig)
        plots=[('progress.png','Barrier gap histories')]+([('mu_sweep.png','Complete companion parameter sweep')] if sweep else [])
        explanation='Explicit Newton centering uses alpha = 0.01, beta = 0.5 and squared decrement divided by two at most 1e-5. A t = 1 central point initializes each comparison. A log-sum-exp barrier, Lorentz-cone logarithm, or log determinant is differentiated directly as required. The optimal value is shifted to approximately one as in the book.'
    if eid=='fig_11_17':notes.append('The textbook uses 100 variables and order-100 matrices; the published MATLAB source sets n = 50, m = 100. This implementation deliberately follows the textbook 100-variable dimension. The source comment calling its parameter sweep Figure 11.19 is a typo; it corresponds to Figure 11.18.')
    if eid=='fig_11_4':notes.append('The companion generator uses c = A-transpose z but labels z a strictly feasible dual multiplier, which would require c = -A-transpose z. The replacement enforces that negative sign, guaranteeing the declared strict dual feasibility. Its mu = 150 curve is mislabeled mu = 100 in one source annotation; the numeric parameter 150 is retained.')
    if eid!='fig_11_22' and isinstance(p,(GeometricProgram,ConeProgram)):notes.append('A Newton correction is applied to the central-path dual variables, enforcing dual linear equations despite approximate centering. The corrected feasible-dual objective gap is plotted; nominal degree/t is retained separately in the JSON. Cone or entropy-dual feasibility is checked independently.')
    if eid=='fig_11_22':notes.append('The primal-dual method plots the surrogate gap minus f(x)-transpose lambda and the combined KKT residual. It stops only when both are below 1e-8. Its lambda multipliers and saved primal point permit independent positivity, feasibility, stationarity and complementarity checks; no entropy-dual correction or nominal degree/t quantity is used in this experiment.')
    rows=[]
    for label,r in runs.items():rows.extend([(label+' updates',r['iterations']),(label+' objective',f'{r["objective"]:.10g}'),(label+' final gap',f'{r.get("gap",r.get("nominal_gap")):.4g}'),(label+' stationarity',f'{r["stationarity"]:.4g}')])
    return save_result(eid,out,{'runs':runs,'parameter_sweep':sweep},formula,explanation,rows,plots,notes)


def run(eid,output_dir=None):
    begin_run(eid)
    out=Path(output_dir or ROOT/'artifacts/examples'/eid);out.mkdir(parents=True,exist_ok=True)
    if eid in ['example_11_1','example_11_3','example_11_4','fig_11_10']:
        from examples.algorithm_phase import path_experiment,phase_residual_experiment,boundary_experiment
        if eid in ['example_11_1','example_11_3']:return path_experiment(eid,out)
        return phase_residual_experiment(out) if eid=='example_11_4' else boundary_experiment(out)
    if eid in ['fig_11_7','fig_11_19']:
        from examples.algorithm_scaling import run_three
        return run_three(eid,out)
    if eid in ['fig_11_8','fig_11_20','fig_11_23']:
        from examples.algorithm_scaling import run_sweep
        run_sweep('sdp' if eid=='fig_11_20' else 'lp')
        import json
        return json.loads((out/'results.json').read_text())
    return run_small(eid,out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('id');a=p.parse_args();r=run(a.id);print(r['id'],'completed')
