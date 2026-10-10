"""Original geometric optimization models, exact inputs and disclosed substitutes."""
import numpy as np
import cvxpy as cp
from scipy.optimize import linprog,minimize
from scipy.special import expit
from application_support import data,optimize,check,figsave,run,pyplot
POLYGON=np.array([[.55,0],[.25,.35],[-.2,.2],[-.25,-.1],[0,-.3],[.4,-.2]])
def polygon():
    V=POLYGON;edges=np.roll(V,-1,axis=0)-V;A=np.c_[edges[:,1],-edges[:,0]];b=np.einsum('ij,ij->i',A,V);return V,A/b[:,None],np.ones(len(V))
def circle():
    t=np.linspace(0,2*np.pi,501);return np.array([np.cos(t),np.sin(t)])
def extremal_ellipsoids():
    V,G,h=polygon();A=cp.Variable((2,2),symmetric=True);b=cp.Variable(2);cons=[cp.norm(A@v+b)<=1 for v in V];p=cp.Problem(cp.Minimize(-cp.log_det(A)),cons);ro=optimize(p);Av=A.value;bv=b.value;co=-np.linalg.solve(Av,bv);Bo=np.linalg.inv(Av)
    B=cp.Variable((2,2),symmetric=True);c=cp.Variable(2);con=[cp.norm(B@g)+g@c<=z for g,z in zip(G,h)];q=cp.Problem(cp.Maximize(cp.log_det(B)),con);ri=optimize(q);Bi=B.value;ci=c.value
    # Explicit KKT equations, using scalar cone multipliers.
    lam=np.array([k.dual_value for k in cons]);Y=V@Av.T+bv;directions=Y/np.linalg.norm(Y,axis=1)[:,None];gradA=-np.linalg.inv(Av)+sum(l*(np.outer(d,v)+np.outer(v,d))/2 for l,d,v in zip(lam,directions,V));gradb=lam@directions
    mu=np.array([k.dual_value for k in con]);gradB=-np.linalg.inv(Bi)+sum(l*(np.outer(Bi@g,g)+np.outer(g,Bi@g))/(2*np.linalg.norm(Bi@g)) for l,g in zip(mu,G));gradc=mu@G
    checks=[check('enclosing KKT stationarity',max(np.max(abs(gradA)),max(abs(gradb))),2e-4),check('inscribed KKT stationarity',max(np.max(abs(gradB)),max(abs(gradc))),2e-4),check('enclosing feasibility',ro['primal_violation']),check('inscribed feasibility',ri['primal_violation']),check('half-sized enclosing ellipse inside polygon',max(0,np.max(G@co+.5*np.linalg.norm(G@Bo,axis=1)-h)),1e-6),check('twice-sized inscribed ellipse contains vertices',max(0,np.max(np.linalg.norm(np.linalg.solve(Bi,(V-ci).T),axis=0))-2),1e-6)]
    return {'enclosing':{'center':co,'shape':Bo,'area':np.pi*np.linalg.det(Bo)},'inscribed':{'center':ci,'shape':Bi,'area':np.pi*np.linalg.det(Bi)}},checks

def ellipsoids(out):
    rec,checks=extremal_ellipsoids();fig,ax=pyplot().subplots(1,2,figsize=(10,4));vv=np.vstack([POLYGON,POLYGON[0]])
    for a,(label,scale) in zip(ax,[('enclosing',.5),('inscribed',2)]):
        e=rec[label];a.plot(*vv.T,'k-');a.plot(*(e['center'][:,None]+e['shape']@circle()),label=label);a.plot(*(e['center'][:,None]+scale*e['shape']@circle()),'--',label=f'scale {scale}');a.legend();a.set_aspect('equal')
    return {'summary':{k:{'area':e['area'],'center':e['center']} for k,e in rec.items()},'ellipsoids':rec,'checks':checks,'plots':[figsave(fig,out,'extremal_ellipsoids')]}

def centers(out):
    V,G,h=polygon();rec,checks=extremal_ellipsoids();sol=linprog([0,0,-1],A_ub=np.c_[G,np.linalg.norm(G,axis=1)],b_ub=h,bounds=[(None,None),(None,None),(0,None)],method='highs');cc=sol.x[:2];radius=sol.x[2]
    f=lambda x:-np.log(h-G@x).sum() if np.min(h-G@x)>0 else np.inf;g=lambda x:G.T@(1/(h-G@x));H=lambda x:G.T@((1/(h-G@x)**2)[:,None]*G);ref=minimize(f,np.zeros(2),jac=g,hess=H,method='trust-exact',options={'gtol':1e-11});ac=ref.x
    x=cp.Variable(2);p=cp.Problem(cp.Minimize(-cp.sum(cp.log(h-G@x))));r=optimize(p);checks+=[check('analytic center convex versus Newton objective',abs(f(ac)-r['objective']),1e-8),check('analytic center gradient',max(abs(g(ac))),1e-6),check('Chebyshev LP dual gap',abs(sol.fun-h@sol.ineqlin.marginals),1e-8)]
    fig,ax=pyplot().subplots(1,3,figsize=(12,4));vv=np.vstack([V,V[0]])
    for a in ax:a.plot(*vv.T,'k-');a.set_aspect('equal')
    ax[0].plot(*(cc[:,None]+radius*circle()));ax[0].scatter(*cc);ax[0].set_title('Chebyshev')
    e=rec['inscribed'];ax[1].plot(*(e['center'][:,None]+e['shape']@circle()));ax[1].scatter(*e['center']);ax[1].set_title('Maximum-volume center')
    xx,yy=np.meshgrid(np.linspace(-.3,.6,200),np.linspace(-.35,.4,200));pts=np.c_[xx.ravel(),yy.ravel()];slacks=h-pts@G.T;vals=np.full(len(pts),np.nan);valid=np.all(slacks>0,axis=1);vals[valid]=-np.log(slacks[valid]).sum(axis=1);ax[2].contour(xx,yy,vals.reshape(xx.shape),levels=np.linspace(f(ac)+.1,f(ac)+4,7));eig,Q=np.linalg.eigh(H(ac));ax[2].plot(*(ac[:,None]+Q@np.diag(1/np.sqrt(eig))@circle()),'--');ax[2].scatter(*ac);ax[2].set_title('Analytic center and Hessian ellipse')
    return {'summary':{'Chebyshev_center':cc,'Chebyshev_radius':radius,'maximum_volume_center':e['center'],'analytic_center':ac},'analytic_hessian':H(ac),'checks':checks,'plots':[figsave(fig,out,'centers')]}

def upperbounds(out):
    # New, explicit instance: the original three matrices and weights were not published.
    As=np.array([[[1.,.3],[.3,.5]],[[.5,-.2],[-.2,1.1]],[[.8,.15],[.15,.8]]]);Ws=[np.eye(2),np.diag([4.,1.])];records=[];checks=[];fig,ax=pyplot().subplots(figsize=(6,5))
    for i,A in enumerate(As):ev,Q=np.linalg.eigh(A);ax.plot(*(Q@np.diag(np.sqrt(ev))@circle()),label=f'Input {i+1}')
    for j,W in enumerate(Ws):
        X=cp.Variable((2,2),symmetric=True);cons=[X-A>>0 for A in As];p=cp.Problem(cp.Minimize(cp.trace(W@X)),cons);r=optimize(p);Z=[c.dual_value for c in cons];dual=sum(np.trace(z@a) for z,a in zip(Z,As));checks+=[check(f'weight {j+1} dual stationarity',np.max(abs(sum(Z)-W)),1e-7),check(f'weight {j+1} primal-dual gap',abs(p.value-dual),1e-7)];ev,Q=np.linalg.eigh(X.value);ax.plot(*(Q@np.diag(np.sqrt(ev))@circle()),'--',lw=2,label=f'Upper bound {j+1}');records.append({'weight':W,'matrix':X.value,'objective':r['objective']})
    ax.legend(fontsize=8);ax.set_aspect('equal')
    return {'summary':{'instance':'new explicit 2x2 matrices and positive-definite objective weights','weighted_objectives':[r['objective'] for r in records]},'input_matrices':As,'solutions':records,'checks':checks,'plots':[figsave(fig,out,'matrix_upper_bounds')],'caveats':['Original Figure 4.10 numerical matrices and weights were not supplied in the textbook or located in the official indexes. This new instance demonstrates the same weighted-SDP minimal-upper-bound construction. Positive-definite W ensures no strictly smaller PSD-feasible upper bound can dominate the computed solution.']}

def classification(out):
    d=data('linsep');X=d['X'].T;Y=d['Y'].T;P=np.vstack([X,Y]);labels=np.r_[np.ones(len(X)),-np.ones(len(Y))];records={};checks=[];fig,axs=pyplot().subplots(1,3,figsize=(12,4))
    for j,name in enumerate(['LP_slack','SVM','logistic']):
        a=cp.Variable(2);b=cp.Variable();margin=cp.multiply(labels,P@a-b)
        if name=='LP_slack':objective=cp.sum(cp.pos(1-margin))
        elif name=='SVM':objective=cp.norm(a)+.1*cp.sum(cp.pos(1-margin))
        else:objective=cp.sum(cp.logistic(-margin))
        p=cp.Problem(cp.Minimize(objective));r=optimize(p);av=a.value;bv=float(b.value);marg=labels*(P@av-bv);rec={'a':av,'b':bv,'objective':r['objective'],'misclassified':int((marg<0).sum()),'correct_inside_slab':int(((marg>0)&(marg<1-1e-6)).sum())}
        # Independently solve the dual of each hinge-loss classifier.
        if name!='logistic':
            l=cp.Variable(len(P));dualcons=[l>=0,l<=(1 if name=='LP_slack' else .1),labels@l==0];normal=P.T@cp.multiply(labels,l)
            dualcons += [normal==0] if name=='LP_slack' else [cp.norm(normal)<=1]
            dual=cp.Problem(cp.Maximize(cp.sum(l)),dualcons);dr=optimize(dual);checks.append(check(name+' independent dual gap',abs(r['objective']-dr['objective']),1e-6))
        # The smooth logistic likelihood has a direct stationarity certificate.
        if name=='logistic':grad=np.c_[P,-np.ones(len(P))].T@(-labels*expit(-marg));checks.append(check('logistic gradient',max(abs(grad)),1e-4))
        # Independently evaluate objective from numeric margins.
        obj=np.maximum(1-marg,0).sum() if name=='LP_slack' else np.linalg.norm(av)+.1*np.maximum(1-marg,0).sum() if name=='SVM' else np.logaddexp(0,-marg).sum();checks.append(check(name+' objective re-evaluation',abs(obj-r['objective']),1e-6));records[name]=rec;ax=axs[j];ax.scatter(*X.T,facecolors='none',edgecolors='b');ax.scatter(*Y.T,c='orange');t=np.linspace(-10,10,100)
        for v in (-1,0,1):ax.plot(t,(bv+v-av[0]*t)/av[1],'-' if v==0 else '--',color='k')
        ax.set(xlim=(-10,10),ylim=(-10,10),title=name);ax.set_aspect('equal')
    checks.append(check('printed misclassification counts',sum(abs(records[k]['misclassified']-v) for k,v in zip(records,[1,3,2])),0));checks.append(check('verified companion correctly-classified slab counts',sum(abs(records[k]['correct_inside_slab']-v) for k,v in zip(records,[4,14,3])),0))
    return {'summary':records,'checks':checks,'plots':[figsave(fig,out,'classifiers')],'caveats':['Book reports correctly classified strict-slab counts 4/15/3; this published companion instance gives 4/14/3 at 1e-6 boundary tolerance. Misclassification counts match 1/3/2. Boundary points are reported separately from strict interior points; no count is silently forced to the book.']}

def placement(out):
    d=data('placement');fixed=d['X'];edges=d['E'].astype(int);N=int(edges.max()+1-len(fixed));inc=np.zeros((len(edges),N+len(fixed)))
    for i,(a,b) in enumerate(edges):inc[i,a]=1;inc[i,b]=-1
    A=inc[:,:N];B=inc[:,N:]@fixed;records={};checks=[];fig,ax=pyplot().subplots(2,3,figsize=(12,7))
    for j,power in enumerate([1,2,4]):
        X=cp.Variable((N,2));length=cp.norm(A@X+B,axis=1);p=cp.Problem(cp.Minimize(cp.sum(cp.power(length,power))));r=optimize(p);xx=X.value;diff=A@xx+B;norm=np.linalg.norm(diff,axis=1);records[str(power)]={'free_locations':xx,'edge_lengths':norm,'objective':r['objective']}
        if power==2:checks.append(check('quadratic placement versus independent least squares',np.max(abs(xx-np.linalg.lstsq(A,-B,rcond=None)[0])),1e-6))
        if power==4:checks.append(check('quartic placement stationarity',np.max(abs(4*A.T@((norm**2)[:,None]*diff))),1e-4))
        if power==1:
            Z=cp.Variable(diff.shape);dual=cp.Problem(cp.Maximize(cp.sum(cp.multiply(B,Z))),[A.T@Z==0,cp.norm(Z,axis=1)<=1]);rd=optimize(dual);checks.append(check('linear placement independent dual gap',abs(r['objective']-rd['objective']),1e-6))
        allx=np.vstack([xx,fixed]);ax[0,j].scatter(*fixed.T,c='r',marker='s');ax[0,j].scatter(*xx.T,c='b')
        for a,b in edges:ax[0,j].plot(*allx[[a,b]].T,color='gray',lw=.7)
        ax[0,j].set_title(f'Power {power}');ax[0,j].set_aspect('equal');ax[1,j].hist(norm,bins=10);ax[1,j].set_xlabel('Edge length')
    return {'summary':{'fixed_nodes':len(fixed),'free_nodes':N,'edges':len(edges),'objectives':{k:v['objective'] for k,v in records.items()}},'solutions':records,'checks':checks,'plots':[figsave(fig,out,'placements')],'caveats':['Official placement.py explicitly states its data differ from the book. Published placement.bin has 28 edges, not the inventory’s earlier count of 27. This is a complete method reproduction on that companion instance.']}

def floor_constraints(x,y,w,h,W,H):
    return [x[[0,1,3]]>=0,y[[1,2,4]]>=0,x[0]+w[0]+1<=x[2],x[1]+w[1]+1<=x[2],x[2]+w[2]+1<=x[4],x[3]+w[3]+1<=x[4],x[4]+w[4]<=W,y[1]+h[1]+1<=y[0],y[0]+h[0]+1<=y[3],y[2]+h[2]+1<=y[3],y[3]+h[3]<=H,y[4]+h[4]<=H,w>=h/5,w<=5*h]
def floorplan(out):
    areas=[[100]*5,[20,50,80,150,200],[180,80,80,80,80],[20,150,20,200,110]];records=[];checks=[];fig,ax=pyplot().subplots(2,2,figsize=(9,9))
    for j,amin in enumerate(areas):
        x=cp.Variable(5);y=cp.Variable(5);w=cp.Variable(5,nonneg=True);h=cp.Variable(5,nonneg=True);W=cp.Variable();H=cp.Variable();con=floor_constraints(x,y,w,h,W,H);p=cp.Problem(cp.Minimize(W+H),con+[cp.multiply(amin,cp.inv_pos(h))<=w]);r=optimize(p);record={'minimum_areas':amin,'x':x.value.copy(),'y':y.value.copy(),'width':w.value.copy(),'height':h.value.copy(),'W':float(W.value),'H':float(H.value),'half_perimeter':r['objective']};records.append(record)
        # Equivalent rotated quadratic cone formulation checks the reciprocal model.
        q=cp.Problem(cp.Minimize(W+H),con+[cp.SOC(w[i]+h[i],cp.hstack([2*np.sqrt(amin[i]),w[i]-h[i]])) for i in range(5)]);rr=optimize(q);checks.append(check(f'layout {j+1} reciprocal versus rotated-cone objective',abs(r['objective']-rr['objective']),2e-5));checks.append(check(f'layout {j+1} min-area satisfaction',max(0,np.max(np.array(amin)-record['width']*record['height'])),2e-5));a=ax.ravel()[j]
        for k in range(5):a.add_patch(pyplot().Rectangle((record['x'][k],record['y'][k]),record['width'][k],record['height'][k],facecolor=f'C{k}',alpha=.4));a.text(record['x'][k]+record['width'][k]/2,record['y'][k]+record['height'][k]/2,str(k+1))
        a.set(xlim=(-1,26),ylim=(-1,26),title=str(amin));a.set_aspect('equal')
    return {'summary':{'half_perimeters':[r['half_perimeter'] for r in records],'widths':[r['W'] for r in records],'heights':[r['H'] for r in records]},'layouts':records,'checks':checks,'plots':[figsave(fig,out,'floorplans')]}
FUNCTIONS={'fig_8_3':ellipsoids,'fig_8_5':centers,'example_4_10':upperbounds,'fig_8_10':classification,'example_8_6':placement,'example_8_7':floorplan}
if __name__=='__main__':run(FUNCTIONS)
