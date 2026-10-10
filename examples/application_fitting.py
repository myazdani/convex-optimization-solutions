"""Original chapter-six fitting and robust-approximation computations."""
import itertools, math
import numpy as np
import cvxpy as cp
from scipy.linalg import null_space
from scipy.optimize import linprog, minimize_scalar
from application_support import data,optimize,check,figsave,run,pyplot

def fit(out,spline=False):
    d=data('polapprox');t=d['t'].ravel();y=d['y'].ravel()
    if spline:
        # Truncated-power basis automatically enforces C2 continuity at both knots.
        design=lambda z:np.c_[np.vander(z,4,increasing=True),np.maximum(z+1/3,0)**3,np.maximum(z-1/3,0)**3]
    else:design=lambda z:np.vander(z,6,increasing=True)
    A=design(t);xls=np.linalg.lstsq(A,y,rcond=None)[0];x=cp.Variable(6);p=cp.Problem(cp.Minimize(cp.norm_inf(A@x-y)));r=optimize(p);xc=x.value
    ref=linprog(np.r_[np.zeros(6),1],A_ub=np.vstack([np.c_[A,-np.ones(len(t))],np.c_[-A,-np.ones(len(t))]]),b_ub=np.r_[y,-y],bounds=[(None,None)]*6+[(0,None)],method='highs')
    tt=np.linspace(-1.1,1.1,1000);fig,ax=pyplot().subplots(figsize=(6,4));ax.scatter(t,y,facecolors='none',edgecolors='k');ax.plot(tt,design(tt)@xls,label='Least squares');ax.plot(tt,design(tt)@xc,'--',label='Minimax');ax.legend();ax.set(xlabel='u',ylabel='p(u)')
    if spline:
        for knot in (-1/3,1/3):ax.axvline(knot,color='gray',alpha=.3)
    return {'summary':{'basis':'C2 cubic truncated powers' if spline else 'degree-five monomials','least_squares_residual_l2':np.linalg.norm(A@xls-y),'least_squares_residual_linf':max(abs(A@xls-y)),'minimax_residual_l2':np.linalg.norm(A@xc-y),'minimax_residual_linf':max(abs(A@xc-y))},'coefficients':{'least_squares':xls,'minimax':xc},'checks':[check('minimax objective versus independent HiGHS LP',abs(r['objective']-ref.fun),1e-8),check('LS normal equations',max(abs(A.T@(A@xls-y))),1e-8)],'plots':[figsave(fig,out,'fits')]}

def convexfit(out):
    d=data('cvxfit');u=d['u'].ravel();y=d['y'].ravel();m=len(u);z=cp.Variable(m);g=cp.Variable(m);cons=[z[None,:]>=z[:,None]+cp.multiply(g[:,None],u[None,:]-u[:,None])];p=cp.Problem(cp.Minimize(cp.sum_squares(z-y)),cons);r=optimize(p);zv=z.value;gv=g.value
    order=np.argsort(u);us=u[order];ref=cp.Variable(m);slope=cp.multiply(1/np.diff(us),ref[1:]-ref[:-1]);p2=cp.Problem(cp.Minimize(cp.sum_squares(ref-y[order])),[slope[1:]>=slope[:-1]]);r2=optimize(p2)
    tt=np.linspace(0,2.2,1000);ff=np.max(zv[None,:]+gv[None,:]*(tt[:,None]-u[None,:]),axis=1);fig,ax=pyplot().subplots(figsize=(6,4));ax.scatter(u,y,facecolors='none',edgecolors='k');ax.plot(tt,ff);ax.set(xlabel='u',ylabel='Convex fitted function')
    return {'summary':{'samples':m,'squared_error':r['objective'],'support_inequalities':m*m},'fitted_values':zv,'subgradients':gv,'grid':tt,'function':ff,'checks':[check('support feasibility',r['primal_violation'],1e-6),check('objective versus independent monotone-slope formulation',abs(r['objective']-r2['objective']),1e-6),check('fit versus monotone-slope formulation',max(abs(zv[order]-ref.value)),2e-5)],'plots':[figsave(fig,out,'convex_fit')]}

def robust_interval(out):
    d=data('robls')['6.15'];A=d['A'];b=d['b'].ravel();B=d['B'];nom=np.linalg.lstsq(A,b,rcond=None)[0];sto=np.linalg.solve(A.T@A+B.T@B/3,A.T@b);x=cp.Variable(A.shape[1]);t=cp.Variable();p=cp.Problem(cp.Minimize(t),[cp.norm((A+B)@x-b)<=t,cp.norm((A-B)@x-b)<=t]);r=optimize(p);worst=x.value
    # Independent worst-case SDP from the S-lemma.
    z=cp.Variable(A.shape[1]);v=cp.Variable(nonneg=True);s=cp.Variable();q=cp.reshape(A@z-b,(-1,1),order='C');P=cp.reshape(B@z,(-1,1),order='C');L=cp.bmat([[np.eye(len(b)),P,q],[P.T,cp.reshape(v,(1,1),order='C'),np.zeros((1,1))],[q.T,np.zeros((1,1)),cp.reshape(s,(1,1),order='C')]]);pr=cp.Problem(cp.Minimize(s+v),[L>>0]);rr=optimize(pr)
    us=np.linspace(-2,2,500);sols={'nominal':nom,'stochastic':sto,'worst_case':worst};curves={k:np.array([np.linalg.norm((A+u*B)@z-b) for u in us]) for k,z in sols.items()};fig,ax=pyplot().subplots(figsize=(6,4))
    for k,vv in curves.items():ax.plot(us,vv,label=k)
    ax.axvline(-1,color='k',ls=':');ax.axvline(1,color='k',ls=':');ax.legend();ax.set(xlabel='Uncertainty u',ylabel='Residual norm')
    return {'summary':{'worst_case_residual':r['objective'],'nominal_expected_squared_residual':np.linalg.norm(A@nom-b)**2+np.linalg.norm(B@nom)**2/3,'stochastic_expected_squared_residual':np.linalg.norm(A@sto-b)**2+np.linalg.norm(B@sto)**2/3},'solutions':sols,'uncertainty_grid':us,'residual_curves':curves,'checks':[check('SOCP versus S-lemma SDP',abs(r['objective']**2-rr['objective']),2e-6),check('stochastic normal equations',max(abs((A.T@A+B.T@B/3)@sto-A.T@b)),1e-8)],'plots':[figsave(fig,out,'robust_interval')]}

def disk_worst(A,Ap,b):
    n=A.shape[1];m=len(b);x=cp.Variable(n);v=cp.Variable(nonneg=True);t=cp.Variable();P=cp.hstack([cp.reshape(a@x,(m,1),order='C') for a in Ap]);q=cp.reshape(A@x-b,(m,1),order='C');M=cp.bmat([[np.eye(m),P,q],[P.T,v*np.eye(2),np.zeros((2,1))],[q.T,np.zeros((1,2)),cp.reshape(t,(1,1),order='C')]]);p=cp.Problem(cp.Minimize(t+v),[M>>0]);r=optimize(p);return x.value,r

def disk_max(A,Ap,b,x):
    q=A@x-b;P=np.column_stack([a@x for a in Ap]);ts=np.linspace(0,2*np.pi,2001);vals=np.linalg.norm(q[:,None]+P@np.array([np.cos(ts),np.sin(ts)]),axis=0);k=np.argmax(vals);h=ts[1]-ts[0];r=minimize_scalar(lambda z:-np.linalg.norm(q+P@np.array([np.cos(z),np.sin(z)])),bounds=(ts[k]-h,ts[k]+h),method='bounded');return -r.fun

def robust_disk(out):
    d=data('robls')['6.16'];A=d['A0'];Ap=[d['A1'],d['A2']];b=d['b'].ravel();nom=np.linalg.lstsq(A,b,rcond=None)[0];wc,r=disk_worst(A,Ap,b);sols={'nominal':nom,'Tikhonov_delta_1_book':np.linalg.solve(A.T@A+np.eye(A.shape[1]),A.T@b),'Tikhonov_delta_0.1_companion':np.linalg.solve(A.T@A+.1*np.eye(A.shape[1]),A.T@b),'robust':wc}
    rng=np.random.default_rng(606);rad=np.sqrt(rng.random(100000));theta=2*np.pi*rng.random(100000);U=np.array([rad*np.cos(theta),rad*np.sin(theta)]);res={};summary={};fig,ax=pyplot().subplots(figsize=(7,4))
    for name,x in sols.items():
        rv=np.linalg.norm((A@x-b)[:,None]+np.column_stack([a@x for a in Ap])@U,axis=0);res[name]=rv;summary[name]={'nominal_residual':np.linalg.norm(A@x-b),'disk_maximum':disk_max(A,Ap,b,x),'sample_mean':rv.mean(),'sample_maximum':rv.max()};ax.hist(rv,bins=np.linspace(0,5,51),density=True,histtype='step',label=name)
    ax.legend(fontsize=7);ax.set(xlabel='Residual norm',ylabel='Density')
    bins=np.linspace(0,5,51)
    return {'summary':summary,'solutions':sols,'sample_count':100000,'monte_carlo_seed':606,'histograms':{k:np.histogram(v,bins=bins)[0] for k,v in res.items()},'bin_edges':bins,'checks':[check('SDP objective versus independently maximized disk boundary',abs(r['objective']-summary['robust']['disk_maximum']**2),2e-5),check('all disk draws feasible',max(0,np.max(np.linalg.norm(U,axis=0))-1))],'plots':[figsave(fig,out,'disk_histograms')],'caveats':['Both book delta=1 and companion delta=0.1 are executed. The seed-606 Monte Carlo sample is newly generated and is not the unpublished original histogram sample.']}

def regsel(out):
    d=data('regsel');A=d['A'];b=d['b'].ravel();m,n=A.shape;best=[np.linalg.norm(b)];subsets=[[]];counts=[1]
    for k in range(1,m):
        record=(np.inf,None)
        for idx in itertools.combinations(range(n),k):
            z=np.linalg.lstsq(A[:,idx],b,rcond=None)[0];r=np.linalg.norm(A[:,idx]@z-b)
            if r<record[0]:record=(r,list(idx))
        best.append(record[0]);subsets.append(record[1]);counts.append(math.comb(n,k));print('cardinality',k,'subsets',counts[-1],flush=True)
    best.append(0.);subsets.append(list(range(m)));counts.append(1)
    x=cp.Variable(n);alpha=cp.Parameter(nonneg=True);p=cp.Problem(cp.Minimize(cp.sum_squares(A@x-b)),[cp.norm1(x)<=alpha]);xln=np.linalg.lstsq(A,b,rcond=None)[0];heuristic={0:np.linalg.norm(b)}
    for val in np.linspace(0,np.linalg.norm(xln,1),100)[1:]:
        alpha.value=val;optimize(p);idx=np.flatnonzero(abs(x.value)>.1*max(abs(x.value)))
        if len(idx)<=m:
            z=np.linalg.lstsq(A[:,idx],b,rcond=None)[0];r=np.linalg.norm(A[:,idx]@z-b);heuristic[len(idx)]=min(heuristic.get(len(idx),np.inf),r)
    fig,ax=pyplot().subplots(figsize=(6,4));ax.step(best,range(m+1),label='Exhaustive optimum');ax.plot(list(heuristic.values()),list(heuristic),'o--',label='L1 heuristic + refit');ax.legend();ax.set(xlabel='Residual norm',ylabel='Cardinality')
    checks=[check('heuristic never beats exhaustive optimum',max([0]+[best[k]-v for k,v in heuristic.items()]),1e-8),check('enumerated all supports below rank',abs(sum(counts[:-1])-sum(math.comb(n,k) for k in range(m))),0),check('rank-ten support achieves zero residual',np.linalg.norm(A[:,:m]@np.linalg.solve(A[:,:m],b)-b),1e-8)]
    return {'summary':{'matrix_shape':[m,n],'subsets_enumerated':sum(counts[:-1]),'reproduction':'official companion data; explicitly different from book'},'optimal_residuals':best,'optimal_subsets_zero_based':subsets,'subsets_per_cardinality':counts,'heuristic_residuals':heuristic,'checks':checks,'plots':[figsave(fig,out,'sparse_selection')],'caveats':['Official regsel.py explicitly states its problem data differ from the book. This is a complete method reproduction on the published companion instance.']}
FUNCTIONS={'example_6_7':lambda out:fit(out,False),'example_6_8':lambda out:fit(out,True),'fig_6_24':convexfit,'example_6_5':robust_interval,'example_6_6':robust_disk,'example_6_4':regsel}
if __name__=='__main__':run(FUNCTIONS)
