"""Disclosed PCG64 method reproductions of legacy-MATLAB classification studies."""
import numpy as np
import cvxpy as cp
from scipy.optimize import linprog
from application_support import optimize,check,figsave,run,pyplot

def linear(out):
    rng=np.random.default_rng(0);Y=np.vstack([1.5+rng.normal(size=6),2*rng.normal(size=6)]);X=np.vstack([-1.5+rng.normal(size=10),2*rng.normal(size=10)]);T=np.array([[-1,1],[1,1]]);X=(T@X).T;Y=(T@Y).T;P=np.vstack([X,Y]);labels=np.r_[np.ones(len(X)),-np.ones(len(Y))]
    a=cp.Variable(2);b=cp.Variable();p=cp.Problem(cp.Minimize(cp.sum_squares(a)),[cp.multiply(labels,P@a-b)>=1]);r=optimize(p);av=a.value.copy();bv=float(b.value);margin=1/np.linalg.norm(av)
    aa=cp.Variable(2);bb=cp.Variable();t=cp.Variable();q=cp.Problem(cp.Maximize(t),[cp.multiply(labels,P@aa-bb)>=t,cp.norm(aa)<=1]);rr=optimize(q)
    # Distance between the two convex hulls is twice the maximum slab half-width.
    u=cp.Variable(len(X));v=cp.Variable(len(Y));dual=cp.Problem(cp.Minimize(cp.norm(X.T@u-Y.T@v)),[u>=0,v>=0,cp.sum(u)==1,cp.sum(v)==1]);rd=optimize(dual)
    fig,axs=pyplot().subplots(1,2,figsize=(10,4))
    for ax,coef,intercept,width in [(axs[0],av,bv,1),(axs[1],aa.value,bb.value,t.value)]:
        ax.scatter(*X.T,facecolors='none',edgecolors='b');ax.scatter(*Y.T,c='orange');xgrid=np.linspace(P[:,0].min()-1,P[:,0].max()+1,100)
        for offset in (-width,0,width):ax.plot(xgrid,(intercept+offset-coef[0]*xgrid)/coef[1],'-' if offset==0 else '--',color='gray')
        ax.set_aspect('equal')
    axs[0].set_title('Feasible separator');axs[1].set_title('Maximum-width separating slab')
    return {'summary':{'class_sizes':[len(X),len(Y)],'maximum_slab_width':2*rr['objective'],'seed':0},'points':{'X':X,'Y':Y},'feasible_separator':{'a':av,'b':bv},'maximum_margin':{'a':aa.value,'b':bb.value,'half_width':t.value},'checks':[check('minimum-norm versus max-margin formulation',abs(margin-rr['objective']),1e-6),check('convex-hull distance duality',abs(rd['objective']-2*rr['objective']),1e-6),check('margin feasibility',rr['primal_violation'])],'plots':[figsave(fig,out,'linear_separators')],'caveats':['Uses NumPy PCG64 seed 0 with the official point-cloud dimensions, Gaussian construction and linear transform. This is not MATLAB randn(state,3), and the exact original points/figure are not claimed. Seeds3,808,8 were checked and gave nonseparable replacement instances; seed0 was selected to demonstrate the prescribed separable case.']}

def quadratic(out):
    rng=np.random.default_rng(813);N=50;X=rng.normal(size=(2,N));X*=.99*rng.random(N)/np.linalg.norm(X,axis=0);Y=rng.normal(size=(2,N));Y*=(1.02+rng.random(N))/np.linalg.norm(Y,axis=0);T=np.array([[1,-1],[2,1]]);X=(T@X).T;Y=(T@Y).T
    P=cp.Variable((2,2),symmetric=True);q=cp.Variable(2);r=cp.Variable();fx=cp.sum(cp.multiply(X@P,X),axis=1)+X@q+r;fy=cp.sum(cp.multiply(Y@P,Y),axis=1)+Y@q+r;con=[P>>np.eye(2),fx<=-1,fy>=1];pr=cp.Problem(cp.Minimize(cp.trace(P)),con);res=optimize(pr);Pv=P.value;qv=q.value;rv=float(r.value);center=-.5*np.linalg.solve(Pv,qv);level=center@Pv@center-rv;ev,Q=np.linalg.eigh(Pv);tt=np.linspace(0,2*np.pi,501);ellipse=center[:,None]+Q@np.diag(np.sqrt(level/ev))@np.array([np.cos(tt),np.sin(tt)]);fig,ax=pyplot().subplots(figsize=(6,5));ax.scatter(*X.T,facecolors='none',edgecolors='b');ax.scatter(*Y.T,c='orange');ax.plot(*ellipse);ax.set_aspect('equal')
    valuesX=np.einsum('ni,ij,nj->n',X,Pv,X)+X@qv+rv;valuesY=np.einsum('ni,ij,nj->n',Y,Pv,Y)+Y@qv+rv
    # A separate analytic separator follows from known inner/outer radial bounds.
    base=np.linalg.inv(T).T@np.linalg.inv(T);radialX=np.einsum('ni,ij,nj->n',X,base,X);radialY=np.einsum('ni,ij,nj->n',Y,base,Y);cut=(radialX.max()+radialY.min())/2
    return {'summary':{'class_sizes':[50,50],'center':center,'maximum_inner_polynomial':valuesX.max(),'minimum_outer_polynomial':valuesY.min(),'seed':813},'points':{'X':X,'Y':Y},'polynomial':{'P':Pv,'q':qv,'r':rv},'checks':[check('SDP feasibility',res['primal_violation'],1e-7),check('independent radial separator inner inequality',max(0,radialX.max()-cut)),check('independent radial separator outer inequality',max(0,cut-radialY.min()))],'plots':[figsave(fig,out,'quadratic_separator')],'caveats':['New PCG64 seed813 replaces the legacy MATLAB random streams. The 50+50 points retain the stated inside/outside radial construction and transform. A trace objective selects one valid ellipse from the feasible set.']}

def monomials(P,degree):
    powers=[(i,k-i) for k in range(degree+1) for i in range(k+1)];return np.column_stack([P[:,0]**i*P[:,1]**j for i,j in powers]),powers

def polynomial(out):
    rng=np.random.default_rng(0);X=rng.uniform(-1,1,(2,100));X*=.9*rng.random(100)/np.linalg.norm(X,axis=0);Y=rng.uniform(-1,1,(2,120));Y*=(1.1+rng.random(120))/np.linalg.norm(Y,axis=0);dist=np.linalg.norm(X-np.array([[1.1],[0]]),axis=0);Y=np.c_[Y,X[:,dist<.9]];X=X[:,dist>1];X=X.T;Y=Y.T;P=np.vstack([X,Y]);labels=np.r_[-np.ones(len(X)),np.ones(len(Y))];records={};checks=[];coef=None
    for degree in [1,2,3,4]:
        A,powers=monomials(P,degree);signed=labels[:,None]*A;ref=linprog(np.zeros(A.shape[1]),A_ub=-signed,b_ub=-np.ones(len(P)),bounds=[(None,None)]*A.shape[1],method='highs');records[str(degree)]={'status':'feasible' if ref.success else 'infeasible' if ref.status==2 else ref.message,'monomial_powers':powers}
        if ref.success:
            a=cp.Variable(A.shape[1]);p=cp.Problem(cp.Minimize(cp.sum_squares(a)),[signed@a>=1]);r=optimize(p);records[str(degree)]['coefficients']=a.value;records[str(degree)]['minimum_signed_value']=min(signed@a.value);checks.append(check(f'degree {degree} separator feasibility',r['primal_violation'],1e-6));coef=a.value;pows=powers
        elif ref.status==2:
            # Farkas certificate: convex combination of signed feature rows equals zero.
            w=cp.Variable(len(P));pr=cp.Problem(cp.Minimize(0),[w>=0,cp.sum(w)==1,signed.T@w==0]);r=optimize(pr);records[str(degree)]['farkas_weights']=w.value;checks.append(check(f'degree {degree} independent Farkas certificate',max(abs(signed.T@w.value)),1e-7))
        else:raise RuntimeError(ref.message)
    if records['3']['status']!='infeasible' or records['4']['status']!='feasible':raise AssertionError('Replacement instance does not demonstrate cubic failure/quartic success; report rather than silently relabel')
    xx,yy=np.meshgrid(np.linspace(-2.2,2.2,400),np.linspace(-2.2,2.2,400));AA,_=monomials(np.c_[xx.ravel(),yy.ravel()],4);fig,ax=pyplot().subplots(figsize=(6,5));ax.scatter(*X.T,facecolors='none',edgecolors='b');ax.scatter(*Y.T,c='orange');ax.contour(xx,yy,(AA@coef).reshape(xx.shape),levels=[0],colors='k');ax.set_aspect('equal')
    return {'summary':{'class_sizes':[len(X),len(Y)],'degree_status':{k:v['status'] for k,v in records.items()},'seed':0},'points':{'X':X,'Y':Y},'degree_results':records,'checks':checks,'plots':[figsave(fig,out,'quartic_separator')],'caveats':['PCG64 seed0 replaces the original MATLAB uniform generator. The same 100/120 initial sizes, radial construction and exchanged wedge are retained. Infeasibility of degrees1–3 is certified by Farkas witnesses, not inferred from a failed numerical solve. Initial trial seed814 admitted a cubic separator; seed0 was selected from a disclosed generator check to demonstrate the stated cubic-versus-quartic phenomenon.']}
FUNCTIONS={'fig_8_8':linear,'fig_8_13':quadratic,'fig_8_14':polynomial}
if __name__=='__main__':run(FUNCTIONS)
