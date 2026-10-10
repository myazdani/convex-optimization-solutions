"""Original Python reproductions of six self-contained or exact-input studies.
Run: python examples/application_basic.py all
"""
from pathlib import Path
import numpy as np
import cvxpy as cp
from scipy.linalg import toeplitz
from scipy.optimize import brentq, minimize, minimize_scalar
from application_support import data,optimize,check,figsave,run,pyplot

def scalar(out):
    answer=minimize_scalar(lambda x:x*np.log(x),bounds=(1e-12,3),method='bounded',options={'xatol':1e-14})
    xs=np.logspace(-2,6,200)
    fig,ax=pyplot().subplots(1,2,figsize=(9,3));ax[0].semilogx(xs,1/xs,label='1/x');ax[0].semilogx(xs,-np.log(xs),label='-log x');ax[0].legend();z=np.linspace(.001,2,300);ax[1].plot(z,z*np.log(z));ax[1].scatter([1/np.e],[-1/np.e]);ax[1].set_title('Unique attained minimum')
    return {'summary':{'minimizer':answer.x,'minimum':answer.fun,'reciprocal_infimum':0,'reciprocal_attained':False,'negative_log_infimum':'-infinity'},'checks':[check('numerical minimizer versus 1/e',abs(answer.x-1/np.e)),check('objective versus -1/e',abs(answer.fun+1/np.e)),check('first derivative at 1/e',abs(1+np.log(1/np.e)))],'sequences':{'x':xs,'reciprocal':1/xs,'negative_log':-np.log(xs)},'plots':[figsave(fig,out,'objectives')]}

def square(out):
    t=np.linspace(-1,1,1001);boundary=np.vstack([np.c_[t,np.ones_like(t)],np.c_[t,-np.ones_like(t)],np.c_[np.ones_like(t),t],np.c_[-np.ones_like(t),t]])
    norms={str(p):np.linalg.norm(boundary,ord=p,axis=1) for p in (1,2,np.inf)}
    fig,ax=pyplot().subplots(figsize=(4,4));ax.scatter(*boundary.T,s=2,label='Every point minimizes infinity norm');axes=np.array([[1,0],[-1,0],[0,1],[0,-1]]);ax.scatter(*axes.T,c='r',label='1- and 2-norm minimizers');ax.scatter(0,0,c='k');ax.set_aspect('equal');ax.legend(fontsize=7)
    return {'summary':{'l1_distance':min(norms['1']),'l2_distance':min(norms['2']),'linfinity_distance':min(norms['inf']),'l1_l2_minimizers':axes},'checks':[check('all infinity-norm values equal one',max(abs(norms['inf']-1))),check('l1 and l2 exact minimum',max(abs(min(norms[k])-1) for k in ['1','2']))],'plots':[figsave(fig,out,'boundary')]}

def rls(out):
    d=data('rls');A=d['A'];b=d['b'].ravel();H=A.T@A;g=A.T@b;ev,V=np.linalg.eigh(H);q=V.T@g;offset=b@b
    lower_grid=np.r_[np.linspace(0,2,20),np.linspace(2,15,21)[1:]];upper_grid=np.linspace(0,1,10)
    def boundary(alpha,upper=False):
        if alpha==0:return offset,np.zeros(A.shape[1]),0.
        edge=max(ev) if upper else -min(ev)
        def norm2(l):return np.sum((q/(l-ev if upper else l+ev))**2)
        lo=edge+1e-12;hi=max(1.,lo+1)
        while norm2(hi)>alpha:hi=2*hi+abs(edge)+1
        lam=brentq(lambda l:norm2(l)-alpha,lo,hi,xtol=1e-13)
        x=V@(-q/(lam-ev) if upper else q/(lam+ev))
        return np.linalg.norm(A@x-b)**2,x,lam
    lower=[boundary(a) for a in lower_grid];upper=[boundary(a,True) for a in upper_grid]
    # Independently certify every sphere extremum with the two-variable SDP dual.
    gaps=[];viol=[]
    for grid,records,isupper in [(lower_grid,lower,False),(upper_grid,upper,True)]:
        for a,(value,x,lam) in zip(grid[1:],records[1:]):
            t=cp.Variable();u=cp.Variable();K=np.block([[H,g[:,None]],[g[None,:],np.array([[offset]])]])
            D=cp.bmat([[u*np.eye(10),np.zeros((10,1))],[np.zeros((1,10)),cp.reshape(t,(1,1),order='C')]])
            p=cp.Problem(cp.Minimize(t+u*a),[D-K>>0] if isupper else [D+K>>0]);r=optimize(p);dual=p.value if isupper else -p.value
            gaps.append(abs(dual-value));viol.append(r['primal_violation'])
    xls=np.linalg.lstsq(A,b,rcond=None)[0];endpoint=[np.linalg.norm(A@xls-b)**2,xls@xls]
    ls=np.array([r[0] for r in lower]);us=np.array([r[0] for r in upper]);fig,ax=pyplot().subplots(figsize=(6,4));ax.plot(ls,lower_grid);ax.plot(us,upper_grid);ax.fill(np.r_[ls[::-1],us,us[-1]],np.r_[lower_grid[::-1],upper_grid,lower_grid[-1]],alpha=.15)
    pareto_alpha=np.linspace(0,xls@xls,100);pareto_res=np.array([boundary(a)[0] for a in pareto_alpha]);ax.plot(pareto_res,pareto_alpha,lw=3,label='Pareto frontier');ax.scatter([offset,endpoint[0]],[0,endpoint[1]]);ax.set(xlim=(0,15),ylim=(0,15),xlabel='Squared residual',ylabel='Squared solution norm');ax.legend()
    return {'summary':{'least_squares_residual_squared':endpoint[0],'least_squares_solution_squared':endpoint[1],'zero_solution_residual_squared':offset,'boundary_sdps':len(gaps)},'lower':{'alpha':lower_grid,'residual_squared':ls},'upper':{'alpha':upper_grid,'residual_squared':us},'pareto':{'alpha':pareto_alpha,'residual_squared':pareto_res},'checks':[check('all secular extrema versus SDP dual bounds',max(gaps),2e-6),check('SDP feasibility',max(viol),2e-6),check('sphere norm constraints',max(abs(x@x-a) for grid,recs in [(lower_grid,lower),(upper_grid,upper)] for a,(_,x,_) in zip(grid,recs)),1e-7)],'plots':[figsave(fig,out,'achievable_region')]}

def portfolio(out):
    mean=np.array([.12,.1,.07,.03]);S=np.array([[.04,.006,-.004,0],[.006,.01,0,0],[-.004,0,.0025,0],[0,0,0,0]])
    mus=np.logspace(-1,4,101);w=cp.Variable(4);mu=cp.Parameter(nonneg=True);p=cp.Problem(cp.Minimize(.5*mu*cp.quad_form(w,S)-mean@w),[w>=0,cp.sum(w)==1]);weights=[];gaps=[];viol=[]
    for val in mus:
        mu.value=val;r=optimize(p);weights.append(w.value.copy());viol.append(r['primal_violation'])
        ref=minimize(lambda x:.5*val*x@S@x-mean@x,np.ones(4)/4,jac=lambda x:val*S@x-mean,bounds=[(0,1)]*4,constraints=[{'type':'eq','fun':lambda x:x.sum()-1,'jac':lambda x:np.ones(4)}],method='SLSQP',options={'ftol':1e-13,'maxiter':500})
        gaps.append(abs(p.value-ref.fun))
    ws=np.vstack([[0,0,0,1],np.array(weights)[::-1],[1,0,0,0]]);risks=np.sqrt(np.maximum(0,np.einsum('ni,ij,nj->n',ws,S,ws)));returns=ws@mean
    fig,ax=pyplot().subplots(1,2,figsize=(10,4));ax[0].plot(risks,returns);ax[0].set(xlabel='Standard deviation',ylabel='Expected return');ax[1].stackplot(risks,ws.T,labels=['Asset 1','Asset 2','Asset 3','Risk-free']);ax[1].legend(fontsize=8);ax[1].set(xlabel='Standard deviation',ylabel='Allocation')
    return {'summary':{'portfolios':len(ws),'risk_free_return':returns[0],'maximum_return':returns[-1],'maximum_risk':risks[-1]},'data':{'means':mean,'covariance':S},'frontier':{'weights':ws,'risk':risks,'return':returns},'checks':[check('independent SLSQP objective agreement',max(gaps)),check('simplex violation',max(viol)),check('frontier return monotonicity',max(0,-np.diff(returns).min()),1e-7)],'plots':[figsave(fig,out,'portfolio')]}

def huber(out):
    d=data('huber');u=d['u'].ravel();b=d['v'].ravel();A=np.c_[np.ones(len(u)),u];ls=np.linalg.lstsq(A,b,rcond=None)[0];x=cp.Variable(2);p=cp.Problem(cp.Minimize(cp.sum(cp.huber(A@x-b,1))));r=optimize(p);xh=x.value
    f=lambda z:np.where(abs(A@z-b)<=1,(A@z-b)**2,2*abs(A@z-b)-1).sum()
    ref=minimize(f,ls,jac=lambda z:2*A.T@np.clip(A@z-b,-1,1),method='BFGS',options={'gtol':1e-10})
    fig,ax=pyplot().subplots(figsize=(6,4));ax.scatter(u,b);t=np.array([-11,11]);ax.plot(t,np.c_[np.ones(2),t]@ls,'--',label='Least squares');ax.plot(t,np.c_[np.ones(2),t]@xh,label='Huber M=1');ax.legend();ax.set(xlabel='t',ylabel='f(t)')
    return {'summary':{'least_squares_intercept':ls[0],'least_squares_slope':ls[1],'huber_intercept':xh[0],'huber_slope':xh[1],'huber_objective':r['objective']},'coefficients':{'least_squares':ls,'huber':xh},'checks':[check('Huber gradient',np.max(abs(2*A.T@np.clip(A@xh-b,-1,1))),1e-5),check('Huber independent BFGS objective',abs(f(xh)-ref.fun),1e-7),check('least-squares normal equations',np.max(abs(A.T@(A@ls-b))),1e-8)],'plots':[figsave(fig,out,'regression')]}

def inputdesign(out):
    n=201;t=np.arange(n);h=(.9**t)*(1-.4*np.cos(2*t))/9;H=toeplitz(h,np.r_[h[0],np.zeros(n-1)]);D=np.diff(np.eye(n),axis=0);yd=np.r_[np.zeros(40),np.ones(50),-np.ones(50),np.zeros(61)];records=[];checks=[]
    fig,axs=pyplot().subplots(3,2,figsize=(10,8))
    for j,(delta,eta) in enumerate([(0,.005),(0,.05),(.3,.05)]):
        stack=np.vstack([H/np.sqrt(n),np.sqrt(eta/n)*np.eye(n),np.sqrt(delta/(n-1))*D]);rhs=np.r_[yd/np.sqrt(n),np.zeros(2*n-1)];u=np.linalg.lstsq(stack,rhs,rcond=None)[0];x=cp.Variable(n);p=cp.Problem(cp.Minimize(cp.sum_squares(H@x-yd)/n+eta*cp.sum_squares(x)/n+delta*cp.sum_squares(D@x)/(n-1)));r=optimize(p)
        terms={'tracking':np.linalg.norm(H@u-yd)**2/n,'magnitude':u@u/n,'derivative':np.linalg.norm(D@u)**2/(n-1)}
        records.append({'delta':delta,'eta':eta,'u':u,'output':H@u,'terms':terms,'objective':r['objective']});checks.append(check(f'case {j+1} QR versus convex QP',np.max(abs(u-x.value)),1e-7));axs[j,0].plot(t,u);axs[j,0].set_ylabel(f'delta={delta}, eta={eta}');axs[j,1].plot(t,H@u);axs[j,1].plot(t,yd,'--')
    return {'summary':{'cases':3,'normalization':'Book: Jtrack and Jmag divide by n=201; Jder divides by n-1=200.'},'desired_output':yd,'impulse_response':h,'cases':records,'checks':checks,'plots':[figsave(fig,out,'input_output')],'caveats':['CVXOPT stacking omits the relative n/(n-1) derivative normalization. This implementation follows the book and CVX convention.']}
FUNCTIONS={'example_4_1':scalar,'fig_4_11':rls,'fig_4_12':portfolio,'example_6_2':huber,'example_6_3':inputdesign,'example_8_1':square}
if __name__=='__main__':run(FUNCTIONS)
