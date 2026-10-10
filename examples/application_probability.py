"""Chebyshev/Chernoff symbol detection, explicit moment certificates and Monte Carlo."""
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.spatial import Voronoi, voronoi_plot_2d
from application_support import optimize,check,figsave,run,pyplot
VERTICES=np.array([[1,1],[-1,2],[-2,1],[-2,-1],[0,-2],[1.5,-1]])
def constellation():
    E=np.roll(VERTICES,-1,axis=0)-VERTICES;A=np.c_[E[:,1],-E[:,0]];b=np.einsum('ij,ij->i',A,VERTICES);C=np.vstack([np.zeros(2),2*b[:,None]*A/np.sum(A*A,axis=1)[:,None]]);regions=[]
    for k,c in enumerate(C):
        other=np.delete(C,k,axis=0);G=other-c;h=(np.sum(other**2,axis=1)-c@c)/2;regions.append((G,h))
    return C,regions

def cheb_problem(A,b):
    P=cp.Variable((2,2),symmetric=True);q=cp.Variable(2);r=cp.Variable();tau=cp.Variable(len(b),nonneg=True);sig2=cp.Parameter(nonneg=True);Q=cp.bmat([[P,cp.reshape(q,(2,1),order='C')],[cp.reshape(q,(1,2),order='C'),cp.reshape(r,(1,1),order='C')]]);cons=[]
    for i in range(len(b)):
        cross=q-tau[i]*A[i]/2;cons.append(cp.bmat([[P,cp.reshape(cross,(2,1),order='C')],[cp.reshape(cross,(1,2),order='C'),cp.reshape(r-1+tau[i]*b[i],(1,1),order='C')]])>>0)
    cons.append(Q>>0);p=cp.Problem(cp.Minimize(sig2*cp.trace(P)+r),cons);return p,sig2,(P,q,r,tau),cons

def chebyshev(out):
    C,regions=constellation();sigmas=np.linspace(.001,2.5,151);curves=[];viol=[];special=None
    for k in range(3):
        A,b=regions[k];b=b-A@C[k];p,sig2,vars,cons=cheb_problem(A,b);values=[]
        for sigma in sigmas:
            sig2.value=sigma*sigma;r=optimize(p);values.append(max(0,1-r['objective']));viol.append(r['primal_violation'])
        curves.append(values)
        if k==0:
            sig2.value=1.;r=optimize(p);P,q,rr,tau=vars;duals=[np.asarray(c.dual_value) for c in cons];weights=np.array([Z[2,2] for Z in duals]);keep=weights>1e-7;atoms=np.array([Z[:2,2]/Z[2,2] for Z in duals if Z[2,2]>1e-7]);w=weights[keep];moment=sum(duals);special={'correct_detection_lower_bound':1-r['objective'],'P':P.value,'q':q.value,'r':float(rr.value),'multipliers':tau.value,'atoms':atoms,'weights':w,'dual_moment_matrices':duals,'moment_matrix':moment}
    S=special;checks=[check('printed sigma=1 benchmark',abs(S['correct_detection_lower_bound']-.2048),5e-5),check('all SDP feasibility',max(viol),2e-6),check('dual mass/mean/covariance matrix',np.max(abs(S['moment_matrix']-np.eye(3))),2e-6),check('atomic mean',max(abs(S['weights']@S['atoms'])),2e-6),check('atomic covariance',np.max(abs(S['atoms'].T@np.diag(S['weights'])@S['atoms']-np.eye(2))),2e-5)]
    fig,ax=pyplot().subplots(1,2,figsize=(11,5));vv=np.vstack([VERTICES,VERTICES[0]]);voronoi_plot_2d(Voronoi(C),ax=ax[0],show_vertices=False,show_points=False,line_colors='gray',line_width=1);ax[0].plot(*vv.T,'k-');ax[0].scatter(*C.T,c='b',label='Symbols');ax[0].scatter(*S['atoms'].T,s=500*S['weights']+10,c='r',alpha=.6,label='Extremal atoms')
    center=-np.linalg.solve(S['P'],S['q']);level=1-S['r']+S['q']@np.linalg.solve(S['P'],S['q']);ev,Q=np.linalg.eigh(S['P']);theta=np.linspace(0,2*np.pi,500);ell=center[:,None]+Q@np.diag(np.sqrt(level/ev))@np.array([np.cos(theta),np.sin(theta)]);ax[0].plot(*ell,'r--');ax[0].legend(fontsize=7);ax[0].set_aspect('equal')
    for k,values in enumerate(curves):ax[1].plot(sigmas,values,label=f'Symbol {k}')
    ax[1].legend();ax[1].set(xlabel='Noise standard deviation',ylabel='Correct-detection lower bound')
    return {'summary':{'sigma1_correct_detection_lower_bound':S['correct_detection_lower_bound'],'sweep_sdps':3*len(sigmas),'extremal_atoms':len(S['weights'])},'symbols':C,'noise_sigma':sigmas,'bound_curves':curves,'sigma1_certificate':S,'checks':checks,'plots':[figsave(fig,out,'chebyshev_bounds')],'caveats':['Extremal outside atoms can lie exactly on Voronoi boundaries. The strict/open-cell worst-case probability is the limiting distribution under arbitrarily small outward perturbations; a tie-breaking convention should not be mistaken for a discrepancy in the moment bound.']}

def chernoff(out):
    C,regions=constellation();opt=[];checks=[]
    for k,(A,b) in enumerate(regions[1:],1):
        x=cp.Variable(2);p=cp.Problem(cp.Minimize(cp.sum_squares(x)),[A@x<=b]);r=optimize(p);opt.append(r['objective']);ref=minimize(lambda z:z@z,C[k],jac=lambda z:2*z,constraints={'type':'ineq','fun':lambda z:b-A@z,'jac':lambda z:-A},method='SLSQP',options={'ftol':1e-13});checks.append(check(f'region{k} projection versus SLSQP',abs(ref.fun-r['objective']),1e-7))
    sigmas=np.linspace(.2,.5,201);bounds=1-np.exp(-np.array(opt)[None,:]/(2*sigmas[:,None]**2)).sum(axis=1);rng=np.random.default_rng(708);samples=rng.normal(size=(100000,2));A,b=regions[0];v=samples@A.T;threshold=np.min(np.where(v>0,b[None,:]/np.maximum(v,1e-300),np.inf),axis=1);counts=np.array([(s<=threshold).sum() for s in sigmas]);estimate=counts/len(samples);z=1.959963984540054;den=1+z*z/len(samples);center=(estimate+z*z/(2*len(samples)))/den;half=z*np.sqrt(estimate*(1-estimate)/len(samples)+z*z/(4*len(samples)**2))/den
    # Independent Gaussian quadrature at 11 points for a statistical check.
    exact=[];idx=np.arange(0,len(sigmas),20)
    for i in idx:
        sigma=sigmas[i]
        def integrand(x):
            lo=-np.inf;hi=np.inf
            for (aa,bb),rhs in zip(A,b):
                if abs(bb)<1e-14:
                    if aa*x>rhs:return 0.
                elif bb>0:hi=min(hi,(rhs-aa*x)/bb)
                else:lo=max(lo,(rhs-aa*x)/bb)
            return max(0,ndtr(hi/sigma)-ndtr(lo/sigma))*np.exp(-.5*(x/sigma)**2)/(sigma*np.sqrt(2*np.pi))
        val=quad(integrand,-5,5,epsabs=1e-10,points=list(VERTICES[:,0]))[0];exact.append(val)
    exact=np.array(exact);error=np.abs(estimate[idx]-exact);se=np.sqrt(np.maximum(exact*(1-exact),1e-8)/len(samples));checks.append(check('Monte Carlo versus independent Gaussian quadrature (six sigma)',max(0,np.max(error-6*se)),1e-8));checks.append(check('Chernoff lower bounds below exact Gaussian probability',max(0,np.max(bounds[idx]-exact)),1e-9))
    fig,ax=pyplot().subplots(figsize=(7,4));ax.plot(sigmas,bounds,label='Chernoff lower bound');ax.plot(sigmas,estimate,'--',label='Monte Carlo (100000)');ax.fill_between(sigmas,center-half,center+half,alpha=.2,label='95% pointwise Wilson interval');ax.scatter(sigmas[idx],exact,s=14,label='Independent quadrature');ax.legend(fontsize=8);ax.set(xlabel='Noise standard deviation',ylabel='Correct-detection probability',ylim=(.89,1.005))
    return {'summary':{'minimum_squared_distances':opt,'sigma0.5_bound':bounds[-1],'sigma0.5_estimate':estimate[-1],'samples_per_sigma':len(samples),'seed':708,'standardized_maximum_MC_error':max(error/se)},'sigma':sigmas,'chernoff_lower':bounds,'monte_carlo_estimate':estimate,'wilson_lower':center-half,'wilson_upper':center+half,'quadrature_sigma':sigmas[idx],'quadrature_probability':exact,'checks':checks,'plots':[figsave(fig,out,'chernoff_monte_carlo')],'caveats':['The deterministic bounds use the published constellation. A new PCG64 seed708 Gaussian sample is used, with common random numbers across noise levels and pointwise uncertainty intervals. Original Monte Carlo samples were not published.']}
FUNCTIONS={'fig_7_5':chebyshev,'fig_7_8':chernoff}
if __name__=='__main__':run(FUNCTIONS)
