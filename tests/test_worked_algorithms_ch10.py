import json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from examples.algorithms_ch10 import game_residual

def result(eid):return json.loads((ROOT/'artifacts/examples'/eid/'results.json').read_text())
def data(eid):
 from examples.algorithm_evidence import load_data
 return load_data(eid)

def test_feasible_center_kkt():
 d=data('fig_10_1');r=result('fig_10_1')['run'];x=np.array(r['x']);w=np.array(r['w'])
 assert x.min()>0;assert np.linalg.norm(d['A']@x-d['b'])<1e-8;assert np.linalg.norm(-1/x+d['A'].T@w)<1e-8

def test_infeasible_farkas_certificate():
 d=data('fig_10_3');y=d['certificate'];assert (d['A'].T@y).min()>-1e-8;assert abs(d['b']@y+1)<1e-8
 r=result('fig_10_3')['run'];assert r['primal_residual']>1e-2;assert r['steps'][-1]<r['steps'][0]

def test_game_saddle_stationarity():
 d=data('fig_10_5');runs=result('fig_10_5')['runs'];xs=[]
 for r in runs.values():
  x=np.array(r['x']);y=np.array(r['y']);xs.append(np.r_[x,y]);assert max(np.linalg.norm(x),np.linalg.norm(y))<1
  assert np.linalg.norm(game_residual(d['A'],d['c'],d['d'],x,y))<1e-8
 np.testing.assert_allclose(xs[0],xs[1],atol=1e-9)

def test_three_methods_agree():
 d=data('fig_10_6');A,b=d['A'],d['b'];r=result('fig_10_6');values=[]
 for group in r['runs'].values():
  assert len(group)==4
 for q in r['runs']['feasible primal']+r['runs']['infeasible KKT']:
  x=np.array(q['x']);assert x.min()>0;assert np.linalg.norm(A@x-b)<1e-7;values.append(-np.log(x).sum())
 for q in r['runs']['dual']:
  z=np.array(q['x']);x=1/(A.T@z);assert x.min()>0;assert np.linalg.norm(A@x-b)<2e-4;values.append(-np.log(x).sum())
 assert np.ptp(values)<1e-4
