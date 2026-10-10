"""Independent checks of worked chapter 9 experiments."""
import json
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from examples.algorithm_evidence import load_data,load_sparse
from examples.algorithms_ch09 import EXP_A,EXP_FSTAR,EXP_XSTAR,exponential,exp_descent,LogBarrier,barrier_descent
ROOT=Path(__file__).resolve().parents[1]

def load(eid):return json.loads((ROOT/'artifacts/examples'/eid/'results.json').read_text())

def test_quadratic_closed_form():
 r=load('fig_9_2');x=np.array(r['trajectory']);Q=np.diag([1.,10.]);g=x[:-1]@Q
 np.testing.assert_allclose(x[1:],x[:-1]-np.array(r['steps'])[:,None]*g,atol=2e-14)
 np.testing.assert_allclose(r['values'],np.sum(x*(x@Q),axis=1)/2,atol=1e-13)
 np.testing.assert_allclose(np.array(r['values'])[1:]/np.array(r['values'])[:-1],(9/11)**2,atol=1e-13)

def test_exponential_analytic_benchmark():
 assert np.linalg.norm(exponential(EXP_XSTAR)[1])<1e-14
 assert abs(exponential(EXP_XSTAR)[0]-EXP_FSTAR)<1e-14
 for eid in ['fig_9_3','fig_9_11','fig_9_19']:
  r=load(eid)
  for name,run in r['runs'].items():
   xy=np.array(run['trajectory']);fs=np.exp(xy@EXP_A.T-.1).sum(axis=1)
   np.testing.assert_allclose(fs,run['values'],rtol=1e-14)
   assert np.max(np.diff(fs))<1e-12
   assert np.linalg.norm(xy[-1]-EXP_XSTAR)<2e-6
   assert abs(fs[-1]-EXP_FSTAR)<1e-11

def test_exponential_derivatives():
 x=np.array([.2,-.1]);f,g,H=exponential(x);eps=1e-5
 for j in range(2):
  e=np.eye(2)[j]*eps
  np.testing.assert_allclose((exponential(x+e)[1]-exponential(x-e)[1])/(2*eps),H[:,j],rtol=1e-8)
  assert abs((exponential(x+e)[0]-exponential(x-e)[0])/(2*eps)-g[j])<1e-8

def test_dense_recomputed_conditions():
 for eid in ['fig_9_6','fig_9_21']:
  data=load_data(eid);r=load(eid)
  A,b,c=(data[k] for k in ['A','b','c']);ref=r['reference'];x=np.array(ref['x']);slack=b-A@x
  assert slack.min()>0
  assert np.linalg.norm(c+A.T@(1/slack))<1e-5
  for run in r['runs'].values():
   x=np.array(run['x']);s=b-A@x
   assert s.min()>0
   assert abs(c@x-np.log(s).sum()-run['objective'])<1e-8
   assert run['objective']-ref['objective']<1.01e-4
   assert np.max(np.diff(run['values']))<1e-9

def test_conditioning_dimensions_and_censoring():
 r=load('fig_9_7')
 assert len(r['condition_numbers'])==1000 and len(r['sweep'])==11
 for e in r['sweep']:assert e['converged'] or e['iterations']==5000
 assert max(e['mapped_point_error'] for e in r['affine_invariance'])<1e-10
 assert max(e['trajectory_error'] for e in r['affine_invariance'])<1e-10
 assert len({e['iterations'] for e in r['affine_invariance']})==1
 for e in r['affine_invariance']:
  t=e['gamma']**(np.arange(100)/100)
  np.testing.assert_allclose(np.array(e['transformed_trajectory'])*t,e['mapped_trajectory'],rtol=1e-13,atol=1e-13)

def test_full_sparse_instance():
 from scipy import sparse
 r=load('fig_9_23');out=ROOT/'artifacts/examples/fig_9_23';A=load_sparse();b=load_data('fig_9_23')['b'];x=np.array(r['run']['x']);s=b-A@x
 assert A.shape==(100000,10000);assert s.min()>0
 assert abs(-np.log(s).sum()-r['run']['objective'])<1e-7
 assert r['run']['newton_decrement_squared']<1.01e-10

def test_all_self_concordant_trials():
 r=load('fig_9_25');assert len(r['records'])==150;assert len(r['accepted_instances'])==3
 assert all(q['minimum_slack']>0 for q in r['records'])
 assert max(q['final_decrement_squared'] for q in r['records'])<1.01e-10
 assert max(abs(q['final_gap']) for q in r['records'])<1e-6
