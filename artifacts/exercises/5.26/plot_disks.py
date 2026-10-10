"""Draw tangent constraints and exact objective levels for Exercise 5.26."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(__file__).parent
t=np.linspace(0,2*np.pi,801)
fig,ax=plt.subplots(figsize=(5,5),layout='constrained')
for c,color in [(1,'#3079b5'),(-1,'#cd7031')]:
 ax.fill(1+np.cos(t),c+np.sin(t),color=color,alpha=.15)
 ax.plot(1+np.cos(t),c+np.sin(t),color=color,label=f'Disk center (1,{c})')
for r in [.5,1,1.5]:ax.plot(r*np.cos(t),r*np.sin(t),'k--',lw=.7)
ax.plot([1],[0],'ro');ax.annotate('Only feasible point (1,0)',(1,0),xytext=(-.6,.25),fontsize=9)
ax.set(xlim=(-1,2.5),ylim=(-2.3,2.3),xlabel='x₁',ylabel='x₂',title='Objective circles: values 0.25, 1, 2.25',aspect='equal')
ax.legend(loc='upper left',fontsize=8)
fig.savefig(out/'tangent_disks.svg');plt.close(fig)
print('Saved tangent_disks.svg; exact feasible point (1,0), objective 1.')
