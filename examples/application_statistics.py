"""Original statistical and preference-analysis numerical studies."""
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize, linprog
from scipy.special import expit
from scipy.stats import chi2
from application_support import data,optimize,check,figsave,run,pyplot

def logistic(out):
    d=data('logreg');u=d['u'].ravel();y=d['y'].ravel();A=np.c_[u,np.ones(len(u))];x=cp.Variable(2);p=cp.Problem(cp.Minimize(cp.sum(cp.logistic(A@x))-y@(A@x)));r=optimize(p)
    f=lambda z:np.logaddexp(0,A@z).sum()-y@(A@z);g=lambda z:A.T@(expit(A@z)-y);ref=minimize(f,[0,0],jac=g,hess=lambda z:A.T@((expit(A@z)*(1-expit(A@z)))[:,None]*A),method='trust-exact',options={'gtol':1e-10});coef=ref.x
    t=np.linspace(-1,11,1000);fig,ax=pyplot().subplots(figsize=(6,4));ax.scatter(u,y);ax.plot(t,expit(coef[0]*t+coef[1]));ax.set(xlabel='u',ylabel='Probability y=1')
    return {'summary':{'observations':len(u),'slope':coef[0],'intercept':coef[1],'negative_log_likelihood':f(coef)},'coefficients':coef,'checks':[check('independent Newton versus exponential-cone objective',abs(f(coef)-r['objective']),1e-7),check('likelihood gradient',max(abs(g(coef))),1e-7)],'plots':[figsave(fig,out,'logistic')],'caveats':['The data file and both official scripts contain 100 observations. The earlier catalog count of 50 was incorrect.']}

def maxent(out):
    a=np.linspace(-1,1,100);F=np.vstack([a,a*a,3*a**3-2*a,(a<0).astype(float)]);lo=np.array([-.1,.5,-.3,.3]);hi=np.array([.1,.6,-.2,.4]);p=cp.Variable(100);cons=[p>=0,cp.sum(p)==1,F@p>=lo,F@p<=hi];pr=cp.Problem(cp.Maximize(cp.sum(cp.entr(p))),cons);r=optimize(pr);pv=p.value
    lows=[];ups=[];maxdual=0
    for i in range(100):
        c=(np.arange(100)<=i).astype(float)
        for sign,dest in [(1,lows),(-1,ups)]:
            ans=linprog(sign*c,A_ub=np.vstack([F,-F]),b_ub=np.r_[hi,-lo],A_eq=np.ones((1,100)),b_eq=[1],bounds=(0,None),method='highs')
            if not ans.success:raise RuntimeError(ans.message)
            dest.append(sign*ans.fun)
            dual=np.r_[hi,-lo]@ans.ineqlin.marginals+ans.eqlin.marginals[0]
            maxdual=max(maxdual,abs(ans.fun-dual))
    # Independent exponential-family dual, parameterized by nonnegative multipliers.
    G=np.vstack([F,-F]);h=np.r_[hi,-lo]
    from scipy.special import logsumexp
    f=lambda z:logsumexp(-G.T@z)+h@z
    def grad(z):return h-G@np.exp(-G.T@z-logsumexp(-G.T@z))
    ref=minimize(f,np.zeros(8),jac=grad,bounds=[(0,None)]*8,method='L-BFGS-B',options={'ftol':1e-15,'gtol':1e-10,'maxiter':2000})
    fig,ax=pyplot().subplots(1,2,figsize=(10,4));ax[0].step(a,pv,where='mid');ax[0].set(xlabel='Support',ylabel='Probability');ax[1].step(a,np.cumsum(pv),label='Maximum entropy');ax[1].step(a,lows,label='Lower');ax[1].step(a,ups,label='Upper');ax[1].legend();ax[1].set(xlabel='Support',ylabel='CDF')
    return {'summary':{'entropy':r['objective'],'mean':F[0]@pv,'second_moment':F[1]@pv,'cubic_moment':F[2]@pv,'negative_probability':F[3]@pv,'cdf_lps':200},'support':a,'probabilities':pv,'cdf_lower':lows,'cdf_upper':ups,'checks':[check('moment feasibility',r['primal_violation'],1e-7),check('entropy primal versus independent exponential-family dual',abs(r['objective']-ref.fun),1e-7),check('all 200 HiGHS primal-dual gaps',maxdual,1e-8),check('CDF contains maximum entropy distribution',max(np.max(np.array(lows)-np.cumsum(pv)),np.max(np.cumsum(pv)-np.array(ups)),0),1e-7)],'plots':[figsave(fig,out,'entropy_cdf')]}

def consumer(out):
    d=data('consumerpref');B=d['values'].reshape(tuple(d['shape']),order='F').T;C=np.vstack([B,[.5,.5]]);m=len(C);util=(1.1*np.sqrt(B[:,0])+.8*np.sqrt(B[:,1]))/1.9;order=np.argsort(util)
    # Variables: 41 utility values followed by 41 two-dimensional nonnegative supergradients.
    rows=[];rhs=[]
    for i in range(m):
        for j in range(m):
            if i==j:continue
            row=np.zeros(3*m);row[j]=1;row[i]-=1;row[m+2*i:m+2*i+2]=-(C[j]-C[i]);rows.append(row);rhs.append(0.)
    for i,j in zip(order[:-1],order[1:]):
        row=np.zeros(3*m);row[i]=1;row[j]=-1;rows.append(row);rhs.append(-1.)
    A=np.array(rows);rhs=np.array(rhs);eq=np.zeros((1,3*m));eq[0,m-1]=1;bounds=[(None,None)]*m+[(0,None)]*(2*m);records=[];checks=[]
    for k in range(40):
        vals=[]
        for sign in (1,-1):
            c=np.zeros(3*m);c[k]=sign;ans=linprog(c,A_ub=A,b_ub=rhs,A_eq=eq,b_eq=[0],bounds=bounds,method='highs')
            if ans.status not in (0,3):raise RuntimeError(ans.message)
            vals.append(sign*ans.fun if ans.success else ('-infinity' if sign==1 else 'infinity'))
            if ans.success:
                dual=rhs@ans.ineqlin.marginals;checks.append(check(f'basket {k+1}, sign {sign}: LP dual gap',abs(ans.fun-dual),1e-6))
        lower,upper=vals;label='preferred' if isinstance(lower,float) and lower>1e-7 else 'rejected' if isinstance(upper,float) and upper< -1e-7 else 'unresolved';records.append({'basket':k+1,'lower':lower,'upper':upper,'classification':label})
    counts={s:sum(r['classification']==s for r in records) for s in ('preferred','rejected','unresolved')};fig,ax=pyplot().subplots(1,2,figsize=(10,4));xx,yy=np.meshgrid(np.linspace(0,1,201),np.linspace(0,1,201));U=(1.1*np.sqrt(xx)+.8*np.sqrt(yy))/1.9;ax[0].contour(xx,yy,U,levels=np.arange(.1,1,.1),colors='gray');ax[0].scatter(*B.T)
    for label,color in [('preferred','green'),('rejected','red'),('unresolved','orange')]:
        pts=B[[i for i,r in enumerate(records) if r['classification']==label]];ax[1].scatter(*pts.T,label=label,color=color)
    ax[1].scatter(.5,.5,marker='+',color='k');ax[1].contour(xx,yy,U,levels=[np.sqrt(.5)],colors='gray');ax[1].legend()
    checks.append(check('printed classification counts',sum(abs(counts[k]-v) for k,v in {'preferred':14,'rejected':21,'unresolved':5}.items()),0))
    return {'summary':counts,'baskets':B,'utility_order_zero_based':order,'bounds':records,'checks':checks,'plots':[figsave(fig,out,'preferences')]}

def experiment(out):
    d=data('expdesign');V=d['values'].reshape(tuple(d['shape']),order='F');n=V.shape[1];records={};checks=[];theta=np.linspace(0,2*np.pi,361);circle=np.array([np.cos(theta),np.sin(theta)]);fig,ax=pyplot().subplots(1,3,figsize=(12,4));conf,ca=pyplot().subplots(figsize=(5,5))
    for j,name in enumerate(['D','E','A']):
        w=cp.Variable(n);M=V@cp.diag(w)@V.T;constraints=[w>=0,cp.sum(w)==1]
        if name=='D':p=cp.Problem(cp.Minimize(-cp.log_det(M)),constraints)
        elif name=='E':t=cp.Variable();psd=M-t*np.eye(2);con=psd>>0;p=cp.Problem(cp.Maximize(t),constraints+[con])
        else:Y=cp.Variable((2,2),symmetric=True);con=cp.bmat([[M,np.eye(2)],[np.eye(2),Y]])>>0;p=cp.Problem(cp.Minimize(cp.trace(Y)),constraints+[con])
        r=optimize(p);wv=w.value;Mv=V@np.diag(wv)@V.T;inv=np.linalg.inv(Mv)
        if name=='D':Z=inv;bound=2.;fun=lambda a:-np.linalg.slogdet(V@np.diag(a)@V.T)[1]
        elif name=='E':Z=con.dual_value;bound=r['objective'];fun=lambda a:-np.linalg.eigvalsh(V@np.diag(a)@V.T)[0]
        else:Z=inv@inv;bound=np.trace(inv);fun=lambda a:np.trace(np.linalg.inv(V@np.diag(a)@V.T))
        ref=minimize(fun,np.ones(n)/n,bounds=[(0,1)]*n,constraints={'type':'eq','fun':lambda a:a.sum()-1},method='SLSQP',options={'ftol':1e-12,'maxiter':1000});target=-r['objective'] if name=='E' else r['objective'];checks.append(check(name+' versus SLSQP objective',abs(target-ref.fun),2e-6));checks.append(check(name+' dual ellipsoid containment',max(0,np.max(np.einsum('in,ij,jn->n',V,Z,V))-bound),2e-5))
        ev,Q=np.linalg.eigh(Z);ellipse=Q@np.diag(np.sqrt(bound/ev))@circle;ax[j].scatter(*V,s=18);ax[j].scatter(*V[:,wv>1e-4],s=60,facecolors='none',edgecolors='r');ax[j].plot(*ellipse,'--');ax[j].set_title(name+'-optimal');ax[j].set_aspect('equal')
        e,Q=np.linalg.eigh(inv);ellipse90=Q@np.diag(np.sqrt(chi2.ppf(.9,2)*e))@circle;ca.plot(*ellipse90,label=name)
        records[name]={'weights':wv,'information':Mv,'covariance':inv,'objective':r['objective'],'dual_ellipsoid_matrix':Z,'dual_ellipsoid_bound':bound,'selected_indices_one_based':(np.flatnonzero(wv>1e-4)+1)}
    inv=np.linalg.inv(V@V.T/n);e,Q=np.linalg.eigh(inv);ca.plot(*(Q@np.diag(np.sqrt(chi2.ppf(.9,2)*e))@circle),'--',label='Uniform');ca.legend();ca.set_aspect('equal');ca.set_title('90% confidence ellipsoids, unit noise and one observation')
    return {'summary':{name:{'selected_indices_one_based':v['selected_indices_one_based'],'selected_weights':v['weights'][v['weights']>1e-4],'objective':v['objective']} for name,v in records.items()},'candidate_vectors':V,'designs':records,'checks':checks,'plots':[figsave(fig,out,'design_dual_ellipsoids'),figsave(conf,out,'confidence_ellipsoids')],'caveats':['Official candidate coordinates are rounded to four decimal places; optimal weights should be compared at book precision.']}
FUNCTIONS={'fig_7_1':logistic,'example_7_2':maxent,'example_6_9':consumer,'fig_7_9':experiment}
if __name__=='__main__':run(FUNCTIONS)
