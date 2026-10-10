"""Signal smoothing and penalty studies with explicitly new reproducible noise."""
import numpy as np
import cvxpy as cp
from scipy.linalg import solveh_banded
from scipy.optimize import linprog, minimize, brentq, minimize_scalar
from scipy import sparse
from application_support import optimize,check,figsave,run,pyplot

def penalties(out):
    rng=np.random.default_rng(602);A=rng.normal(size=(100,30));b=rng.normal(size=100);b/=1.1*max(abs(b));records={};checks=[];fig,axs=pyplot().subplots(4,1,figsize=(7,9));bins=np.linspace(-1.5,1.5,61)
    for j,name in enumerate(['absolute','quadratic','deadzone','log_barrier']):
        x=cp.Variable(30);res=A@x+b
        obj=cp.norm1(res) if name=='absolute' else cp.sum_squares(res) if name=='quadratic' else cp.sum(cp.pos(cp.abs(res)-.5)) if name=='deadzone' else -cp.sum(cp.log(1-res)+cp.log(1+res))
        p=cp.Problem(cp.Minimize(obj));r=optimize(p);xv=x.value;rv=A@xv+b
        if name=='quadratic':checks.append(check('quadratic normal equations',max(abs(A.T@rv)),1e-7));checks.append(check('quadratic least-squares coefficients',max(abs(xv-np.linalg.lstsq(A,-b,rcond=None)[0])),1e-7))
        elif name in ('absolute','deadzone'):
            threshold=0 if name=='absolute' else .5;ref=linprog(np.r_[np.zeros(30),np.ones(100)],A_ub=np.vstack([np.c_[A,-np.eye(100)],np.c_[-A,-np.eye(100)]]),b_ub=np.r_[-b+threshold,b+threshold],bounds=[(None,None)]*30+[(0,None)]*100,method='highs');checks.append(check(name+' independent HiGHS objective',abs(ref.fun-r['objective']),1e-7))
        else:
            f=lambda x:-np.log1p(-(A@x+b)**2).sum() if max(abs(A@x+b))<1 else np.inf
            g=lambda x:2*A.T@((A@x+b)/(1-(A@x+b)**2));ref=minimize(f,np.zeros(30),jac=g,method='BFGS',options={'gtol':1e-9});checks.append(check('barrier versus independent BFGS',abs(f(ref.x)-r['objective']),1e-7));checks.append(check('barrier residual domain',max(0,max(abs(rv))-1),0))
        records[name]={'coefficients':xv,'residuals':rv,'objective':r['objective'],'histogram':np.histogram(rv,bins)[0]};axs[j].hist(rv,bins);axs[j].set_ylabel(name)
    return {'summary':{k:v['objective'] for k,v in records.items()},'seed':602,'A':A,'b':b,'solutions':records,'bin_edges':bins,'checks':checks,'plots':[figsave(fig,out,'penalty_histograms')],'caveats':['New PCG64 seed602 100-by-30 instance. Common b is scaled by 1/(1.1 max|b|), making x=0 strictly feasible for the logarithmic loss. This follows the CVXOPT input convention; its source explicitly says its own data differ from the book. Unlike the old CVXOPT least-squares branch, the right-hand side is actually -b rather than a zero vector.']}

def quadratic_solution(y,weight):
    n=len(y);diag=np.r_[1.,np.full(n-2,2.),1.];band=np.zeros((2,n));band[0]=1+weight*diag;band[1,:-1]=-weight;return solveh_banded(band,y,lower=True)
def quadratic_sweep(y,count,targets):
    weights=np.logspace(-10,10,count);xs=[quadratic_solution(y,w) for w in weights];cost=np.array([[np.linalg.norm(x-y),np.linalg.norm(np.diff(x))] for x in xs]);chosen=[]
    for target in targets:
        logw=brentq(lambda l:np.linalg.norm(quadratic_solution(y,10**l)-y)-target,-10,10);x=quadratic_solution(y,10**logw);chosen.append({'weight':10**logw,'x':x,'residual_l2':np.linalg.norm(x-y),'roughness_l2':np.linalg.norm(np.diff(x))})
    return weights,cost,chosen

def smooth(out):
    n=4000;t=np.arange(n);truth=.5*np.sin(2*np.pi*t/n)*np.sin(.01*t);noise=np.random.default_rng(608).normal(scale=.05,size=n);y=truth+noise;weights,cost,chosen=quadratic_sweep(y,50,[8,3,1]);checks=[];D=sparse.diags([-np.ones(n-1),np.ones(n-1)],[0,1],shape=(n-1,n))
    for i,c in enumerate(chosen):
        x=cp.Variable(n);p=cp.Problem(cp.Minimize(cp.sum_squares(x-y)+c['weight']*cp.sum_squares(D@x)));r=optimize(p);checks.append(check(f'reconstruction{i+1} tridiagonal versus QP',max(abs(x.value-c['x'])),2e-6))
    fig,axs=pyplot().subplots(3,2,figsize=(11,8),constrained_layout=True);axs[0,0].plot(t,truth,label='True');axs[0,0].plot(t,y,alpha=.5,label='Noisy');axs[0,0].legend();axs[0,1].plot(*cost.T);axs[0,1].set(xlabel='Residual l2',ylabel='Difference l2')
    for ax,c in zip(axs.ravel()[2:],chosen):ax.plot(t,c['x']);ax.set_title(f'Residual {c["residual_l2"]:.2f}')
    axs.ravel()[-1].axis('off')
    return {'summary':{'length':n,'seed':608,'tradeoff_points':50,'reconstruction_residuals':[c['residual_l2'] for c in chosen]},'truth':truth,'observed':y,'weights':weights,'tradeoff':cost,'reconstructions':chosen,'checks':checks,'plots':[figsave(fig,out,'quadratic_smoothing')],'caveats':['New PCG64 seed608 noise replaces the unpublished MATLAB realization. Formula, length4000 and noise standard deviation0.05 match the source. Exact target residuals8,3,1 are found by a scalar root instead of picking the closest point on the coarse trade-off grid.']}

def tv(out):
    n=2000;t=np.arange(n);truth=np.tile(np.r_[np.ones(500),-np.ones(500)],2)+.5*np.sin(2*np.pi*t/n);y=truth+np.random.default_rng(611).normal(scale=.1,size=n);D=sparse.diags([-np.ones(n-1),np.ones(n-1)],[0,1],shape=(n-1,n));qw,qcost,qchosen=quadratic_sweep(y,100,[10,7,4]);x=cp.Variable(n);z=cp.Variable(n-1);w=cp.Parameter(nonneg=True);cons=[D@x<=z,-D@x<=z];p=cp.Problem(cp.Minimize(.5*cp.sum_squares(x-y)+w*cp.sum(z)),cons);weights=np.logspace(-3,3,15);records=[];checks=[]
    for val in weights:
        w.value=val;r=optimize(p);xv=x.value.copy();dual=np.clip(np.asarray(cons[0].dual_value)-np.asarray(cons[1].dual_value),-val,val);dx=D.T@dual;dualvalue=y@dx-.5*dx@dx;objective=.5*np.linalg.norm(xv-y)**2+val*np.linalg.norm(np.diff(xv),1);checks.append(check(f'weight{val:.6g} TV relative primal-dual gap',abs(objective-dualvalue)/(1+abs(objective)),1e-6));checks.append(check(f'weight{val:.6g} TV stationarity',max(abs(xv-y+dx)),1e-6));records.append({'weight':val,'x':xv,'residual_l2':np.linalg.norm(xv-y),'total_variation':np.linalg.norm(np.diff(xv),1),'objective':objective,'absolute_primal_dual_gap':abs(objective-dualvalue)})
    companion_chosen=[min(records,key=lambda r:abs(r['total_variation']-target)) for target in [20,8,5]];chosen=[]
    for budget in [10.,8.,5.]:
        xx=cp.Variable(n);zz=cp.Variable(n-1);cc=[D@xx<=zz,-D@xx<=zz,cp.sum(zz)<=budget];pp=cp.Problem(cp.Minimize(.5*cp.sum_squares(xx-y)),cc);rr=optimize(pp);xv=xx.value.copy();weight=float(cc[-1].dual_value);u=np.clip(np.asarray(cc[0].dual_value)-np.asarray(cc[1].dual_value),-weight,weight);dx=D.T@u;dualvalue=y@dx-.5*dx@dx-weight*budget;gap=rr['objective']-dualvalue;variation=np.linalg.norm(np.diff(xv),1);checks.append(check(f'book TV budget{budget} primal-dual gap',abs(gap),1e-5));checks.append(check(f'book TV budget{budget} attained target',abs(variation-budget),1e-5));chosen.append({'target_total_variation':budget,'weight':weight,'x':xv,'residual_l2':np.linalg.norm(xv-y),'total_variation':variation,'objective':rr['objective'],'dual_vector':u,'primal_dual_gap':gap})
    fig,ax=pyplot().subplots(4,2,figsize=(11,11),constrained_layout=True);ax[0,0].plot(t,truth,label='True');ax[0,0].plot(t,y,alpha=.5,label='Noisy');ax[0,0].legend();ax[0,1].plot([r['residual_l2'] for r in records],[r['total_variation'] for r in records],label='TV');ax[0,1].set(xlabel='Residual l2',ylabel='Total variation')
    for j,(q,c) in enumerate(zip(qchosen,chosen),1):ax[j,0].plot(t,q['x']);ax[j,0].set_title(f'Quadratic, residual {q["residual_l2"]:.1f}');ax[j,1].plot(t,c['x']);ax[j,1].set_title(f'TV {c["total_variation"]:.2f}, residual {c["residual_l2"]:.2f}')
    trade,ta=pyplot().subplots(figsize=(5,4));ta.plot(*qcost.T);ta.set(xlabel='Residual l2',ylabel='Difference l2',title='Quadratic comparison trade-off')
    return {'summary':{'length':n,'seed':611,'TV_tradeoff_points':15,'quadratic_tradeoff_points':100,'selected_TV':[r['total_variation'] for r in chosen]},'truth':truth,'observed':y,'quadratic_weights':qw,'quadratic_tradeoff':qcost,'quadratic_reconstructions':qchosen,'tv_sweep':records,'tv_reconstructions':chosen,'companion_nearest_grid_reconstructions':companion_chosen,'checks':checks,'plots':[figsave(fig,out,'tv_reconstructions'),figsave(trade,out,'quadratic_tradeoff')],'caveats':['New PCG64 seed611 replaces original unseeded noise; the 2000-point signal and noise standard deviation0.1 follow the companion. The book reconstructions are separately solved at exact TV budgets10,8,5. The companion15-point weight sweep and nearest-grid20,8,5 comparisons are also retained, clearly separate from the book targets.']}

def nonconvex(out):
    a=.46;c=.1
    f0=lambda x:(x*x-1)**2+.2*x
    f1=lambda x:(x*x-a*a)*(x*x+c)
    def global_min(lam):
        # Exact real stationary roots of the coercive quartic Lagrangian.
        coef=[4*(1+lam),0,2*(-2+lam*(c-a*a)),.2];roots=np.roots(coef);real=roots.real[abs(roots.imag)<1e-9];values=f0(real)+lam*f1(real);j=np.argmin(values);return float(values[j]),float(real[j])
    primal=f0(-a);lams=np.linspace(0,1,501);g=np.array([global_min(l)[0] for l in lams]);maxdual=minimize_scalar(lambda l:-global_min(l)[0],bounds=(0,20),method='bounded');checks=[check('weak duality over 501 multipliers',max(0,max(g)-primal),1e-9),check('dual discrete concavity',max(0,np.max(np.diff(g,2))),1e-8)]
    for l in np.arange(.1,1.01,.1):
        val,x=global_min(l);checks.append(check(f'lambda{l:.1f} stationarity',abs(4*(1+l)*x**3+2*(-2+l*(c-a*a))*x+.2),1e-8))
    xx=np.linspace(-1.3,1.3,700);fig,ax=pyplot().subplots(1,2,figsize=(10,4));ax[0].plot(xx,f0(xx),label='New objective');ax[0].plot(xx,f1(xx),'--',label='New constraint')
    for l in np.arange(.1,1.01,.1):ax[0].plot(xx,f0(xx)+l*f1(xx),':',color='gray',lw=.7)
    ax[0].axvline(-a,color='gray');ax[0].axvline(a,color='gray');ax[0].scatter(-a,primal);ax[0].legend();ax[1].plot(lams,g,label='Dual g(lambda)');ax[1].axhline(primal,ls='--',label='Primal optimum');ax[1].legend();ax[1].set_xlabel('lambda')
    return {'summary':{'new_primal_minimizer':-a,'new_primal_value':primal,'dual_optimal_lambda':maxdual.x,'dual_optimal_value':-maxdual.fun,'gap':primal+maxdual.fun},'new_functions':{'f0':'(x^2-1)^2+0.2x','f1':'(x^2-0.46^2)(x^2+0.1)'},'lambda_grid':lams,'dual_values':g,'checks':checks,'plots':[figsave(fig,out,'nonconvex_duality')],'caveats':['The original formulas for Figures5.1–5.2 were not published or located in official companion sources. Both functions here are newly specified nonconvex quartics. Their feasible interval is[-0.46,0.46], but the new objective value is not the book’s1.54. All ten Lagrangian curves and the concave dual function belong to this explicitly different instance.']}
FUNCTIONS={'fig_6_2':penalties,'fig_6_8':smooth,'fig_6_11':tv,'fig_5_1_5_2':nonconvex}
if __name__=='__main__':run(FUNCTIONS)
