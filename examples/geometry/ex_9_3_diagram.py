"""Exact geometry and backtracking trajectory for Boyd--Vandenberghe 9.3."""
from pathlib import Path
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/convex-mpl-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/convex-cache")
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    out=Path(__file__).resolve().parents[2]/'docs/figures/ex_9_3.svg'
    fig,ax=plt.subplots(figsize=(5.6,5.2))
    z=np.linspace(-np.sqrt(7),np.sqrt(7),400)
    ax.fill_betweenx(z,1,np.sqrt(8-z*z),color='#cbe4f0',label='S: disk cap, x₁ > 1')
    th=np.linspace(-np.arccos(1/np.sqrt(8)),np.arccos(1/np.sqrt(8)),400)
    ax.plot(np.sqrt(8)*np.cos(th),np.sqrt(8)*np.sin(th),color='#2c6887')
    ax.plot([1,1],[-np.sqrt(7),np.sqrt(7)],'--',color='#b94d38',label='excluded boundary')
    s=2.;pts=[s]
    for _ in range(8):
        t=1.
        while s*(1-2*t)<=1:t*=.5
        s*=1-2*t;pts.append(s)
    ax.plot(pts,pts,'o-',color='#204f48',markersize=4,label='gradient iterates, β = 1/2')
    ax.plot([1],[1],'o',mfc='white',mec='#204f48',ms=7)
    ax.plot([1],[0],'o',mfc='white',mec='#b94d38',ms=7)
    ax.annotate('limit: f → 2',(1,1),xytext=(1.3,.65),arrowprops={'arrowstyle':'->'})
    ax.annotate('infimum: f → 1',(1,0),xytext=(1.4,-.6),arrowprops={'arrowstyle':'->'})
    ax.set(xlim=(.6,3.1),ylim=(-2.9,2.9),xlabel='x₁',ylabel='x₂')
    ax.set_aspect('equal');ax.legend(loc='upper left',bbox_to_anchor=(-.12,1.24),fontsize=8)
    fig.tight_layout();fig.savefig(out);fig.savefig('/tmp/ex_9_3.png');plt.close(fig)
if __name__=='__main__':main()
