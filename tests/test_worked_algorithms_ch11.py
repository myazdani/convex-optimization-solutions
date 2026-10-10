import json,sys,hashlib
from pathlib import Path
import numpy as np
import pytest
from scipy import linalg
from scipy.special import logsumexp
from scipy.optimize import linprog
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from examples.algorithms_ch11 import LinearProgram,GeometricProgram,ConeProgram,SemidefiniteProgram

def result(eid):return json.loads((ROOT/'artifacts/examples'/eid/'results.json').read_text())
def data(eid):
 from examples.algorithm_evidence import load_data
 return load_data(eid)

def test_lp_certificates_against_highs():
 for eid in ['fig_11_4','fig_11_21']:
  d=data(eid);A,b,c=d['A'],d['b'],d['c'];ref=linprog(c,A_ub=A,b_ub=b,bounds=(None,None),method='highs');assert ref.success
  for r in result(eid)['runs'].values():
   x=np.array(r['x']);z=np.array(r['dual']);assert min(np.min(b-A@x),z.min())>0
   assert np.linalg.norm(c+A.T@z)<1e-7;assert abs(c@x-ref.fun)<1.1e-6
   assert abs((b-A@x)@z-r['gap'])<1e-8

def test_gp_entropy_duals():
 for eid in ['fig_11_6','fig_11_22']:
  d=data(eid);A,b=d['A'],d['b']
  for r in result(eid)['runs'].values():
   x=np.array(r['x']);y=np.einsum('ikj,j->ik',A,x)+b;f=logsumexp(y,axis=1);assert f[1:].max()<0
   assert abs(f[0]-r['objective'])<1e-12
   if 'nu' in r:
    nu=np.array(r['nu']);mass=nu.sum(axis=1);assert nu.min()>0;assert abs(mass[0]-1)<1e-12
    assert np.linalg.norm(np.einsum('ik,ikj->j',nu,A))<1e-6
    dual=np.sum(b*nu)-np.sum(nu*np.log(nu/mass[:,None]));assert 0<=f[0]-dual<1.1e-6
   else:assert r['gap']<1e-8 and r['stationarity']<1e-8


def test_socp_conic_dual():
 d=data('fig_11_15');A,b,c=d['A'],d['b'],d['c']
 for r in result('fig_11_15')['runs'].values():
  x=np.array(r['x']);z=np.array(r['dual']);y=np.einsum('ikj,j->ik',A,x)+b
  assert (y[:,-1]-np.linalg.norm(y[:,:-1],axis=1)).min()>0
  assert (z[:,-1]-np.linalg.norm(z[:,:-1],axis=1)).min()>0
  assert np.linalg.norm(c-np.einsum('ik,ikj->j',z,A))<1e-5
  assert abs(c@x+np.sum(b*z)-r['gap'])<1e-6
  assert r['gap']<1e-6


def test_sdp_primal_dual():
 d=data('fig_11_17');A,B,c=d['A'],d['b'],d['c'];assert A.shape==(100,100,100)
 for r in result('fig_11_17')['runs'].values():
  x=np.array(r['x']);Z=np.array(r['dual']);X=B-np.einsum('i,ijk->jk',x,A)
  assert linalg.eigvalsh(X)[0]>0;assert linalg.eigvalsh(Z)[0]>0
  assert np.linalg.norm(c+np.einsum('ijk,jk->i',A,Z))<1e-5
  assert abs(np.sum(X*Z)-r['gap'])<1e-9;assert r['gap']<1e-6


def test_barrier_derivatives():
 for eid,cls in [('fig_11_4',LinearProgram),('fig_11_6',GeometricProgram),('fig_11_15',ConeProgram),('fig_11_17',SemidefiniteProgram)]:
  d=data(eid);p=cls(d['A'],d['b'],d['c']) if 'c' in d else cls(d['A'],d['b']);x=d['x0'];_,g,H=p.evaluate(x,1);v=np.random.default_rng(22).standard_normal(len(x));v/=np.linalg.norm(v);eps=1e-6
  fp,gp,_=p.evaluate(x+eps*v,1);fm,gm,_=p.evaluate(x-eps*v,1)
  np.testing.assert_allclose((gp-gm)/(2*eps),H@v,rtol=2e-5,atol=1e-4)
  assert abs((fp-fm)/(2*eps)-g@v)<1e-4


def test_geometric_paths_and_phase_one():
 for eid in ['example_11_1','example_11_3']:
  d=data(eid);A,b,c=d['A'],d['b'],d['c'];r=result(eid)
  for q in r['points']:
   x=np.array(q['x']);s=b-A@x;z=1/(q['t']*s);assert s.min()>0
   assert np.linalg.norm(c+A.T@z)<1e-3;assert abs(s@z-len(b)/q['t'])<1e-9
 r=result('example_11_4');d=data('example_11_4')
 for label,q in r['runs'].items():
  z=np.array(q['dual']);assert z.min()>-1e-9;assert np.linalg.norm(d['A'].T@z)<1e-7
  assert abs(q['objective']+d['b']@z)<1e-7
  assert abs(z.sum()-1)<1e-8 if label=='maximum' else z.max()<1+1e-8
 d=data('fig_11_10');A,b=d['A'],d['b'];r=result('fig_11_10');assert len(r['ordinary'])==40;assert len(r['near_boundary'])==60
 for q in r['ordinary']+r['near_boundary']:
  assert q['margin']>0;assert (q['gamma']>0)==(q['outcome']=='feasible')
  rhs=b+q['gamma']
  if q['outcome']=='feasible':
   x=np.array(q['x']);assert q['auxiliary_s']<0
   assert np.min(rhs-A@x)>0;assert abs(np.min(rhs-A@x)-q['margin'])<1e-12
  else:
   z=np.array(q['dual']);assert z.min()>0;assert np.linalg.norm(A.T@z)<1e-6
   assert abs(z.sum()-1)<1e-6;assert abs(z.sum()-q['dual_normalization'])<1e-12
   assert -rhs@z>0;assert abs(-rhs@z-q['margin'])<1e-12
 assert all(q['margin']>0 for q in r['infeasible_start'])


def test_original_size_scaling_examples():
 for eid in ['fig_11_7','fig_11_19']:
  r=result(eid);assert set(r['runs'])=={'50','500','1000'}
  for size,q in r['runs'].items():
   assert q['gap']>0;assert q['dual_residual']<1e-5
   assert q['gap_reduction']>9900 if eid=='fig_11_7' else q['gap_reduction']>7900

@pytest.mark.parametrize('eid',['fig_11_8','fig_11_20','fig_11_23'])
def test_full_monte_carlo_sweeps(eid):
 for eid in [eid]:
  path=ROOT/'artifacts/examples'/eid/'results.json'
  if not path.exists():pytest.skip('This Monte Carlo artifact has not yet been delivered; it is not verified.')
  r=json.loads(path.read_text());records=r.get('records',[])
  for shard in r.get('trial_shards',[]):
   p=path.parent/shard['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==shard['sha256'];records+=json.loads(p.read_text())['records']
  assert len(records)==2000 and len({q['dimension'] for q in records})==20
  assert min(q['dimension'] for q in records)==10 and max(q['dimension'] for q in records)==1000
  assert len({(q['dimension'],q['replicate']) for q in records})==2000
  for q in records:
   assert q['code_sha256']==hashlib.sha256((ROOT/'examples/algorithm_scaling.py').read_bytes()).hexdigest()
   a=q[r['method']];assert a['gap']>0;assert a['dual_residual']<1e-5
   if eid=='fig_11_23':
    assert a['gap']<1e-8 and a['primal_residual']<1e-8
    assert a['history'][0]['gap']==2*q['dimension']
   elif eid=='fig_11_8':assert a['gap_reduction']>9900 and a['minimum_x']>0 and a['minimum_dual']>0
   else:assert a['gap_reduction']>7900 and a['minimum_primal_eigenvalue']>0 and a['minimum_dual_eigenvalue']>0
  for summary in r['summary']:
   group=[q[r['method']] for q in records if q['dimension']==summary['dimension']]
   counts=[q['iterations'] if eid=='fig_11_23' else q['path_iterations'] for q in group]
   assert len(group)==100
   assert abs(np.mean(counts)-summary['mean_iterations'])<1e-12
   assert abs(np.std(counts,ddof=1)-summary['std_iterations'])<1e-12

@pytest.mark.parametrize('eid',['fig_11_8','fig_11_20','fig_11_23'])
def test_monte_carlo_regeneration_and_independent_certificates(eid):
 path=ROOT/'artifacts/examples'/eid/'results.json'
 if not path.exists():pytest.skip('This Monte Carlo artifact has not yet been delivered; it is not verified.')
 r=json.loads(path.read_text());records=r.get('records',[])
 for shard in r.get('trial_shards',[]):records+=json.loads((path.parent/shard['path']).read_text())['records']
 from examples.algorithm_scaling import standard_instance,standard_barrier,standard_primal_dual,sdp_matrix,diagonal_sdp
 for n in [10,1000]:
  q=next(q for q in records if q['dimension']==n and q['replicate']==0);saved=q[r['method']]
  if eid=='fig_11_20':
   B=sdp_matrix(n,q['seed']);run=diagonal_sdp(B);x,Z=run['x'],run['dual'];X=B+np.diag(x)
   assert linalg.eigvalsh(X,subset_by_index=[0,0])[0]>0
   assert linalg.eigvalsh(Z,subset_by_index=[0,0])[0]>0
   assert np.max(abs(np.diag(Z)-1))<1e-7
   assert abs(x.sum()+np.sum(B*Z)-np.sum(X*Z))<1e-7
   np.testing.assert_allclose(np.sum(X*Z),saved['gap'],rtol=1e-7,atol=1e-9)
  else:
   A,b,c,x0,y0,s0=standard_instance(n,q['seed'])
   run=standard_barrier(A,b,c,x0) if eid=='fig_11_8' else standard_primal_dual(A,b,c,np.ones(2*n),np.zeros(n),np.ones(2*n))
   x,y,s=run['x'],run['y'],run['s'];assert min(x.min(),s.min())>0
   assert np.linalg.norm(A@x-b)<1e-5;assert np.linalg.norm(A.T@y+s-c)<1e-7
   assert abs(c@x-b@y-x@s)<1e-5
   np.testing.assert_allclose(x@s,saved['gap'],rtol=1e-5,atol=1e-8)
   if n==10:
    ref=linprog(c,A_eq=A,b_eq=b,bounds=(0,None),method='highs');assert ref.success
    assert -1e-8<=c@x-ref.fun<=saved['gap']+1e-7
  assert run['iterations']==saved['iterations']
  np.testing.assert_allclose(run['objective'],saved['objective'],rtol=1e-9,atol=1e-8)
