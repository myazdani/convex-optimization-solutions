"""Generate Exercise 5.1's original plots and numerical formula checks."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(__file__).parent
x=np.linspace(-1,5,601)
fig,ax=plt.subplots(1,3,figsize=(14,4),layout='constrained')
ax[0].axvspan(2,4,color='lightblue',alpha=.5,label='Feasible x')
ax[0].plot(x,x*x+1,'k',lw=2,label='Objective')
for l in [.25,1,2,4]:ax[0].plot(x,(1+l)*x*x-6*l*x+1+8*l,label=f'L, λ={l:g}')
ax[0].plot([2],[5],'ro');ax[0].annotate('(2,5)',(2,5),xytext=(2.3,7))
ax[0].set(xlim=(-1,5),ylim=(-3,30),xlabel='x',title='Primal and Lagrangians');ax[0].legend(fontsize=7)
l=np.linspace(0,8,801);g=10-l-9/(1+l)
ax[1].plot(l,g);ax[1].plot([2],[5],'ro');ax[1].axhline(5,color='gray',ls='--');ax[1].set(xlabel='λ ≥ 0',ylabel='g(λ)',title='Dual lower bounds')
u=np.linspace(-1,12,1001);p=1+np.maximum(3-np.sqrt(1+u),0)**2
ax[2].plot(u,p);ax[2].plot([0],[5],'ro');ax[2].axvspan(-2,-1,color='lightgray');ax[2].text(-1.95,7,'Infeasible',rotation=90);ax[2].set(xlim=(-2,12),xlabel='u',ylabel='p*(u)',title='Perturbation value')
fig.savefig(out/'duality.svg');plt.close(fig)
h=1e-5
value=lambda u:1+max(3-np.sqrt(1+u),0)**2
results={'primal_x':2,'primal_value':5,'dual_lambda':2,'dual_value':5,'gap_identity_max_abs_error':float(np.max(np.abs((5-g)-(l-2)**2/(1+l)))),'sensitivity_central_difference':float((value(h)-value(-h))/(2*h))}
assert results['gap_identity_max_abs_error']<1e-12
assert abs(results['sensitivity_central_difference']+2)<1e-8
(out/'checks.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
