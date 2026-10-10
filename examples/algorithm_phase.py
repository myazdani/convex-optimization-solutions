"""Central paths and phase-I experiments on fully disclosed replacement inputs."""
from pathlib import Path
import numpy as np
from scipy import linalg
from scipy.optimize import linprog
from examples.algorithm_common import ROOT,begin_run,save_result,pyplot
from examples.algorithms_ch11 import LinearProgram,barrier


def polygon(m):
    theta=np.linspace(0,2*np.pi,m,endpoint=False)+.17;A=np.c_[np.cos(theta),np.sin(theta)];b=1+.15*np.sin(2*theta);c=np.array([1.,.65]);return A,b,c


def path_experiment(eid,out):
    begin_run(eid);out=Path(out);out.mkdir(parents=True,exist_ok=True);m=6 if eid=='example_11_1' else 5;A,b,c=polygon(m);lp=linprog(c,A_ub=A,b_ub=b,bounds=(None,None),method='highs');assert lp.success
    ts=np.geomspace(.01,10000,60);ts=np.unique(np.r_[ts,1,3,10]);points=[];rows=[];x=np.zeros(2)
    for t in ts:
        r=barrier(LinearProgram(A,b,c),x,initial_t=t,tol=m/t+1,centering_tol=1e-14);x=r['x'];s=b-A@x;z=1/(t*s)
        points.append(x.copy());rows.append({'t':t,'x':x.copy(),'slacks':s,'multipliers':z,'objective':c@x,'duality_gap':s@z,'stationarity':np.linalg.norm(c+A.T@z),'force_residual':np.linalg.norm(t*c+A.T@(1/s))})
    np.savez_compressed(out/'instance.npz',A=A,b=b,c=c);vertices=[]
    for i in range(m):
        for j in range(i):
            if abs(np.linalg.det(A[[i,j]]))<1e-10:continue
            v=np.linalg.solve(A[[i,j]],b[[i,j]])
            if np.max(A@v-b)<1e-9:vertices.append(v)
    vertices=np.array(vertices);vertices=vertices[np.argsort(np.arctan2(vertices[:,1],vertices[:,0]))];vertices=np.vstack([vertices,vertices[0]])
    plt=pyplot();fig,axes=plt.subplots(1,1 if eid=='example_11_1' else 2,figsize=(7 if eid=='example_11_1' else 10,4.5));axes=np.atleast_1d(axes)
    xy=np.array(points)
    for ax,t in zip(axes,[10] if len(axes)==1 else [1,3]):
        ax.fill(vertices[:,0],vertices[:,1],alpha=.1);ax.plot(vertices[:,0],vertices[:,1],color='.6');ax.plot(xy[:,0],xy[:,1],'--');ax.scatter(*lp.x,marker='*',s=130,color='red',label='LP optimum');ax.legend(loc='upper right');rec=next(v for v in rows if v['t']==t);q=rec['x'];ax.scatter(*q)
        if eid=='example_11_3':
            forces=-A/rec['slacks'][:,None];forces=np.vstack([forces,-t*c]);ax.quiver(np.full(m+1,q[0]),np.full(m+1,q[1]),forces[:,0],forces[:,1],angles='xy',scale_units='xy',scale=5,width=.005)
        else:
            gx,gy=np.meshgrid(np.linspace(-1.4,1.4,200),np.linspace(-1.4,1.4,200));grid=np.c_[gx.ravel(),gy.ravel()];slacks=b-grid@A.T;valid=slacks.min(axis=1)>0;val=np.full(len(grid),np.nan);val[valid]=-np.log(slacks[valid]).sum(axis=1);ax.contour(gx,gy,val.reshape(gx.shape),levels=sorted(set([.1,1,2,4,7,float(-np.log(rec['slacks']).sum())])),colors='.6',linewidths=.6);tangent=np.array([-c[1],c[0]])/np.linalg.norm(c);segment=q[None,:]+np.linspace(-1.3,1.3,2)[:,None]*tangent;ax.plot(segment[:,0],segment[:,1],color='purple',label='Objective tangent at t=10');ax.legend(loc='upper right')
        ax.set_aspect('equal');ax.set(title=f't = {t}',xlabel='x1',ylabel='x2')
    fig.tight_layout();fig.savefig(out/'central_path.png',dpi=160);plt.close(fig)
    special=[r for r in rows if r['t'] in [1,3,10]]
    return save_result(eid,out,{'points':rows,'constraint_count':m,'lp_optimum':{'x':lp.x,'objective':lp.fun,'dual':-lp.ineqlin.marginals,'stationarity':np.linalg.norm(c-A.T@lp.ineqlin.marginals),'duality_gap':lp.fun-b@lp.ineqlin.marginals}},r'x^\star(t)=\arg\min_x\left(tc^Tx-\sum_i\log(b_i-a_i^Tx)\right),\quad\lambda_i(t)=\frac1{t(b_i-a_i^Tx)}.',
        'Newton centering traces 63 parameter values, including t = 1, t = 3 and t = 10. The six-inequality plot marks the t = 10 point, its barrier contour and the tangent objective hyperplane; the force plots retain t = 1 and t = 3. Each inequality normal has unit length, so each repulsive force has the required inverse-distance magnitude. Primal feasibility, dual stationarity and the identity gap = m/t are checked numerically.',
        [('Inequalities',m),('Central points',len(rows)),('Independent LP optimum',f'{lp.fun:.12g}'),('Final central-point objective error',f'{rows[-1]["objective"]-lp.fun:.5g}'),('Maximum dual-stationarity residual',f'{max(r["stationarity"] for r in rows):.5g}')]+[(f't={r["t"]}: gap',f'{r["duality_gap"]:.8g}') for r in special], [('central_path.png','Computed central path and constraint geometry')],
        ['The book does not publish the pictured polytope coordinates. This replacement uses unit normals at angles 2 pi j/m + 0.17, offsets 1 + 0.15 sin(2 angle), and c = (1,0.65). The dimension and number of inequalities follow the book separately: six for Example 11.1 and five for Example 11.3.'])


def phase_residual_experiment(out):
    eid='example_11_4';begin_run(eid);out=Path(out);out.mkdir(parents=True,exist_ok=True);rg=np.random.default_rng(1114);A=rg.standard_normal((100,50));b=rg.standard_normal(100);A[1]=-A[0];b[:2]=-.4
    n=50;m=100
    mx=linprog(np.r_[np.zeros(n),1],A_ub=np.c_[A,-np.ones(m)],b_ub=b,bounds=[(None,None)]*(n+1),method='highs')
    sm=linprog(np.r_[np.zeros(n),np.ones(m)],A_ub=np.c_[A,-np.eye(m)],b_ub=b,bounds=[(None,None)]*n+[(0,None)]*m,method='highs');assert mx.success and sm.success
    runs={}
    for label,r in [('maximum',mx),('sum',sm)]:
        x=r.x[:n];res=A@x-b;dual=-r.ineqlin.marginals;station=A.T@dual
        runs[label]={'x':x,'residuals':res,'objective':r.fun,'satisfied_count':int(np.sum(res<=1e-8)),'dual':dual,'stationarity':np.linalg.norm(station),'duality_gap':r.fun+b@dual}
    np.savez_compressed(out/'instance.npz',A=A,b=b,seed=1114)
    plt=pyplot();fig,axes=plt.subplots(1,2,figsize=(10,4));bins=np.linspace(min(min(v['residuals']) for v in runs.values()),max(max(v['residuals']) for v in runs.values()),25)
    for ax,(label,r) in zip(axes,runs.items()):ax.hist(r['residuals'],bins=bins);ax.axvline(0,color='k');ax.set(title=f'{label}: {r["satisfied_count"]}/100 satisfied',xlabel='Inequality residual',ylabel='Count')
    fig.tight_layout();fig.savefig(out/'residuals.png',dpi=160);plt.close(fig)
    return save_result(eid,out,{'runs':runs},r'\min_x\max_i(a_i^Tx-b_i),\qquad\min_x\sum_i\max\{a_i^Tx-b_i,0\}.',
        'Both phase-I objectives are formulated as explicit linear programs. The maximum violation uses one unrestricted epigraph scalar; the sum uses 100 nonnegative individual slacks. Independent primal-dual LP certificates verify each optimum and histogram.',
        [(label+' satisfied inequalities',r['satisfied_count']) for label,r in runs.items()]+[(label+' objective',f'{r["objective"]:.8g}') for label,r in runs.items()], [('residuals.png','Maximum versus total infeasibility')],
        ['The original 100-by-50 matrix is unpublished. The replacement seed is 1114; its first two rows are opposite normals with offsets -0.4, supplying an explicit infeasibility certificate. The book counts 39 and 79 apply only to its missing instance; the actual replacement counts are reported above.'])


def phase_one(A,b,gamma):
    m,n=A.shape;ext=np.c_[A,-np.ones(m)];rhs=b+gamma;p=LinearProgram(ext,rhs,np.r_[np.zeros(n),1]);v=np.r_[np.zeros(n),1-rhs.min()];t=1.;steps=0
    for outer in range(30):
        for _ in range(100):
            val,g,H=p.evaluate(v,t);dx=linalg.solve(H,-g,assume_a='pos');dec=-g@dx
            if v[-1]<0:return {'gamma':gamma,'iterations':steps,'outcome':'feasible','margin':float(np.min(rhs-A@v[:-1])),'x':v[:-1].copy(),'auxiliary_s':float(v[-1])}
            if dec/2<1e-5:
                cert=p.certificate(v,t,dx);z=cert['dual'];margin=-rhs@z
                if z.min()>0 and margin>0 and np.linalg.norm(A.T@z)<1e-6:return {'gamma':gamma,'iterations':steps,'outcome':'infeasible','margin':float(margin),'certificate_residual':float(np.linalg.norm(A.T@z)),'dual':z.copy(),'dual_normalization':float(z.sum()),'auxiliary_s':float(v[-1])}
                break
            s=1.
            while p.evaluate(v+s*dx,t,False)[0]>val-.01*s*dec:s*=.5
            v+=s*dx;steps+=1
        t*=10
    raise RuntimeError('Phase I failed')


def infeasible_feasibility(A,b,gamma):
    m,n=A.shape;rhs=b+gamma;x=np.zeros(n);s=np.ones(m);nu=np.zeros(m);steps=0
    for k in range(200000):
        margin=np.min(rhs-A@x)
        if margin>0:return {'gamma':gamma,'iterations':steps,'margin':margin}
        rp=A@x+s-rhs;rdx=A.T@nu;rds=-1/s+nu;nr=np.linalg.norm(np.r_[rdx,rds,rp]);D=s*s
        # Eliminate diagonal slack Hessian and multiplier block.
        dx=linalg.solve((A.T/D)@A,A.T@(rds-rp/D)-rdx,assume_a='pos')
        ds=-rp-A@dx;dnu=-rds-ds/D;a=1.
        while np.min(s+a*ds)<=0:a*=.9
        while np.linalg.norm(np.r_[A.T@(nu+a*dnu),-1/(s+a*ds)+nu+a*dnu,A@(x+a*dx)+s+a*ds-rhs])>(1-.01*a)*nr:
            a*=.9
            if a<1e-16:raise RuntimeError('Feasibility Newton stalled')
        x+=a*dx;s+=a*ds;nu+=a*dnu;steps+=1
    raise RuntimeError('Feasibility Newton cap')


def boundary_experiment(out):
    eid='fig_11_10';begin_run(eid);out=Path(out);out.mkdir(parents=True,exist_ok=True);rg=np.random.default_rng(1110);A=rg.standard_normal((50,20));b0=rg.standard_normal(50);boundary=linprog(np.r_[np.zeros(20),1.],A_ub=np.c_[A,-np.ones(50)],b_ub=b0,bounds=(None,None),method='highs');assert boundary.success;xstar=boundary.x[:20];b=b0+boundary.fun
    np.savez_compressed(out/'instance.npz',A=A,b=b,delta_b=np.ones(50),xstar=xstar,seed=1110)
    ordinary=[phase_one(A,b,g) for g in np.linspace(-1,1,40)];near=[phase_one(A,b,g) for g in np.r_[-np.geomspace(1,1e-6,30),np.geomspace(1e-6,1,30)]]
    infeasible=[infeasible_feasibility(A,b,g) for g in np.geomspace(.01,10,25)]
    plt=pyplot();fig,axes=plt.subplots(1,3,figsize=(14,4));axes[0].plot([r['gamma'] for r in ordinary],[r['iterations'] for r in ordinary],'-o');axes[0].set(xlabel='gamma',ylabel='Phase-I updates')
    for sign in [-1,1]:
        group=[r for r in near if np.sign(r['gamma'])==sign];axes[1].semilogx([abs(r['gamma']) for r in group],[r['iterations'] for r in group],label='negative' if sign<0 else 'positive')
    axes[1].set(xlabel='absolute gamma',ylabel='Phase-I updates');axes[1].legend();axes[2].loglog([r['gamma'] for r in infeasible],[max(1,r['iterations']) for r in infeasible],'-o');axes[2].set(xlabel='positive gamma',ylabel='Infeasible-start updates (zero displayed at one)');fig.tight_layout();fig.savefig(out/'boundary.png',dpi=160);plt.close(fig)
    return save_result(eid,out,{'ordinary':ordinary,'near_boundary':near,'infeasible_start':infeasible},r'Ax\leq b+\gamma\mathbf1,\quad A\in\mathbb R^{50\times20},\quad b=b_0+s_\star\mathbf1.',
        'A full Gaussian 50-by-20 matrix and random right-hand side are shifted by the optimal common-slack LP value. Thus gamma positive has strictly feasible x-star, gamma zero is feasible without an interior, and gamma negative is infeasible. The boundary LP supplies an independent certificate of this classification. Phase I explicitly centers the common-slack LP with mu = 10 and stops when the auxiliary scalar s is negative or a corrected dual certificate proves infeasibility. Saved feasible points, auxiliary scalars, dual vectors and their normalizations permit independent checks of both stopping conditions. Infeasible-start Newton solves the slack analytic-center KKT equations with alpha = 0.01, beta = 0.9.',
        [('Linear gamma values',len(ordinary)),('Near-boundary values',len(near)),('Infeasible-start values',len(infeasible)),('Largest phase-I count',max(r['iterations'] for r in near)),('Largest infeasible-start count',max(r['iterations'] for r in infeasible))], [('boundary.png','All three feasibility-boundary experiments')],
        ['The unpublished original A, b and perturbation are replaced by the explicit seed-1110 family above. The 40-value main sweep, positive/negative near-boundary range down to 1e-6 and 25 positive infeasible-start tests are executed. Exact counts and slopes depend on the instance, so no claim is made to match the original curves.'])
