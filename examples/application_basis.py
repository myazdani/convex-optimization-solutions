"""Gabor sparse approximation with all source sampling conventions disclosed."""
import numpy as np
import cvxpy as cp
from application_support import optimize,check,figsave,run,pyplot

def dictionary(variant):
    t=np.arange(501)/(500 if variant=='CVX_endpoint_grid' else 501) if variant!='book_literal_0.005_grid' else .005*np.arange(501)
    centers=.002*np.arange(501);B=np.exp(-((t[:,None]-centers[None,:])/.05)**2)
    if variant=='CVX_endpoint_grid':B[B<.001]=0
    A=np.column_stack([B]+[B*np.cos(5*l*t)[:,None] for l in range(1,31)]+[B*np.sin(5*l*t)[:,None] for l in range(1,31)])
    frequencies=np.repeat(np.r_[0,5*np.arange(1,31),5*np.arange(1,31)],501);times=np.tile(centers,61);y=(1+.5*np.sin(11*t))*np.sin(30*np.sin(5*t));return t,A,y,times,frequencies

def sparse_fit(A,y):
    n=A.shape[1];active=np.argsort(-abs(A.T@y))[:60];history=[];full=np.zeros(n)
    for it in range(80):
        x=cp.Variable(len(active));p=cp.Problem(cp.Minimize(cp.sum_squares(A[:,active]@x-y)+cp.norm1(x)));r=optimize(p);full[:]=0;full[active]=x.value;res=y-A@full;corr=2*A.T@res;scale=max(1,np.max(abs(corr)));nu=2*res/scale;primal=res@res+np.linalg.norm(full,1);dual=y@nu-nu@nu/4;gap=primal-dual;history.append({'iteration':it,'active_columns':len(active),'primal':primal,'dual':dual,'gap':gap,'maximum_dual_violation':max(0,max(abs(corr))-1)});print('active',it,len(active),'gap',gap,flush=True)
        if gap<2e-6:return full,history,nu
        candidate=np.argsort(-abs(corr))[:100];keep=active[abs(x.value)>1e-8];new=np.unique(np.r_[keep,candidate]);
        if np.array_equal(new,active):
            inactive=np.setdiff1d(np.arange(n),active);new=np.unique(np.r_[active,inactive[np.argsort(-abs(corr[inactive]))[:100]]])
        active=new
    raise RuntimeError('Basis pursuit did not reach the certified gap; do not label converged')

def basis(out):
    records={};checks=[];plots=[]
    for variant in ['CVX_endpoint_grid','CVXOPT_k_over_501','book_literal_0.005_grid']:
        t,A,y,times,freq=dictionary(variant);x,hist,dual=sparse_fit(A,y);support=np.flatnonzero(abs(x)>1e-5);coeff=np.linalg.lstsq(A[:,support],y,rcond=None)[0];prediction=A[:,support]@coeff;rel=np.linalg.norm(prediction-y)**2/(y@y);checks.append(check(variant+' full-dictionary primal-dual gap',hist[-1]['gap'],2e-6));checks.append(check(variant+' refit normal equations',np.max(abs(A[:,support].T@(prediction-y))),1e-7))
        records[variant]={'grid':t,'signal':y,'selected_indices_zero_based':support,'l1_coefficients':x[support],'refit_coefficients':coeff,'selected_time':times[support],'selected_frequency':freq[support],'selected_atom_count':len(support),'relative_squared_refit_error':rel,'refit':prediction,'history':hist,'coefficient_threshold':1e-5,'primal_certificate_indices_zero_based':np.flatnonzero(x),'primal_certificate_coefficients':x[x!=0],'dual_certificate':dual,'certified_primal_objective':np.linalg.norm(A@x-y)**2+np.linalg.norm(x,1),'certified_dual_objective':y@dual-dual@dual/4}
        fig,ax=pyplot().subplots(2,2,figsize=(11,7));
        for l in [0,15,30]:
            col=250 if l==0 else l*501+250;ax[0,0].plot(t,A[:,col],label=f'frequency {5*l}')
        ax[0,0].legend(fontsize=7);ax[0,0].set_title('Selected dictionary examples');ax[0,1].plot(t,y,label='Signal');ax[0,1].plot(t,prediction,'--',label='Refit');ax[0,1].legend();ax[1,0].plot(t,prediction-y);ax[1,0].set_title('Refit residual');ax[1,1].scatter(times[support],freq[support],s=15+30*abs(coeff));ax[1,1].set(xlabel='Pulse center',ylabel='Frequency',title=f'{len(support)} selected atoms');plots.append(figsave(fig,out,variant))
    return {'summary':{k:{'selected_atoms':v['selected_atom_count'],'relative_squared_refit_error':v['relative_squared_refit_error'],'primal_dual_gap':v['history'][-1]['gap']} for k,v in records.items()},'variants':records,'checks':checks,'plots':plots,'caveats':['The text says501 equally spaced samples on[0,1], but prints0.005k, whose endpoint is2.5. CVX usesk/500 and removes Gaussian tails below0.001. CVXOPT usesk/501 without truncation. All three conventions are executed and reported separately.','The full dictionary has30561 columns. Active-column optimization is accepted only after a full-dictionary dual certificate; this is not a reduced-dictionary substitute. The l1 optimum is followed by least-squares refitting on coefficients exceeding1e-5, matching the source workflow.','The book reports42 selected atoms and relative squared error2.6e-4. Those are comparison benchmarks; source ambiguities and solver-sensitive support thresholds preclude silently asserting that every convention must recover them.']}
FUNCTIONS={'fig_6_21':basis}
if __name__=='__main__':run(FUNCTIONS)
