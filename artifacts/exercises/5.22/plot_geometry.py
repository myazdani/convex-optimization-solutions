"""Original geometric sketches for all six Exercise 5.22 instances."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(__file__).parent
fig,axs=plt.subplots(2,3,figsize=(12,8),layout='constrained')
u=np.linspace(-3,4,1501)
lows=[np.where(u>=-1,-np.sqrt(np.maximum(u+1,0)),np.nan),np.where(u>=0,-np.sqrt(np.maximum(u,0)),np.nan),np.where(u>=0,-u,np.nan),np.where(u<-1,2-u,-2-u),(1-u)**3,np.maximum(1-u,0)**3]
t=np.linspace(-4,5,1500)
fns=[lambda t:t*t-1,lambda t:t*t,lambda t:abs(t),lambda t:np.where(t<=-1,-t-2,np.where(t<=1,t,2-t))]
for j,ax in enumerate(axs.flat):
 ax.fill_between(u,lows[j],6,color='lightblue',alpha=.7,label='A')
 if j<4:ax.plot(fns[j](t),t,'k',lw=1.5,label='G')
 else:
  xx=np.linspace(-2,3,1500) if j==4 else np.linspace(0,3,1500)
  ax.plot(1-xx,xx**3,'k',lw=1.5,label='G')
 ax.axvline(0,color='gray',lw=.6);ax.axhline(0,color='gray',lw=.6)
 ax.plot([0],[[-1,0,0,-2,1,1][j]],'ro',ms=4)
 ax.set(xlim=(-3,4),ylim=(-4,6),xlabel='u (constraint value)',ylabel='t (objective value)',title=f'({chr(97+j)})')
 ax.legend(loc='upper right',fontsize=8)
fig.savefig(out/'geometry.svg');plt.close(fig)
print('Saved six-panel geometry.svg from exact boundary formulas.')
