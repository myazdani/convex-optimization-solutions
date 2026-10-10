"""Validate executed application-study evidence and independent exact benchmarks."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples'))
FILES=sorted((ROOT/'artifacts/examples').glob('*/results.json'))
FILES=[p for p in FILES if json.loads(p.read_text()).get('source',{}).get('implementation')=='Original Python formulation and implementation; upstream numeric data attributed separately.']
@pytest.mark.parametrize('path',FILES,ids=lambda p:p.parent.name)
def test_executed_checks(path):
    r=json.loads(path.read_text());assert r['checks'];assert r['source']['book_pdf_sha256']=='40d976c83c18cce1900eff8c41bd5ad408c102b813af39d05ff85678ccf8d76e'
    for c in r['checks']:assert c['passed'] and np.isfinite(c['measured']) and c['measured']<=c['tolerance']
    for p in r['plots']:assert (path.parent/p).stat().st_size>1000

def test_published_input_hashes():
    for p in (ROOT/'data/worked_examples').glob('*.provenance.json'):
        d=json.loads(p.read_text());f=p.with_name(p.name.replace('.provenance',''));assert hashlib.sha256(f.read_bytes()).hexdigest()==d['sha256']

def test_huber_and_polynomial_independent_normal_equations():
    from application_support import data
    d=data('huber');A=np.c_[np.ones(42),d['u'].ravel()];b=d['v'].ravel();x=np.linalg.lstsq(A,b,rcond=None)[0];assert np.linalg.norm(A.T@(A@x-b),np.inf)<1e-8
    d=data('polapprox');A=np.vander(d['t'].ravel(),6,increasing=True);x=np.linalg.lstsq(A,d['y'].ravel(),rcond=None)[0];assert np.linalg.norm(A.T@(A@x-d['y'].ravel()),np.inf)<1e-8

def test_scalar_and_square_exact_claims():
    x=1/np.e;assert abs(1+np.log(x))<1e-15;assert abs(x*np.log(x)+1/np.e)<1e-15
    t=np.linspace(-1,1,10001);assert np.min(1+abs(t))==1;assert np.min(np.sqrt(1+t*t))==1;assert np.all(np.maximum(1,abs(t))==1)

def read_result(id):return json.loads((ROOT/'artifacts/examples'/id/'results.json').read_text())
def test_execution_closure_hashes():
    for path in FILES:
        r=json.loads(path.read_text());assert r['runtime_source_sha256'];assert r['execution_command'];assert r['elapsed_seconds']>0
        for key,digest in {**r['runtime_source_sha256'],**r['input_file_sha256']}.items():assert hashlib.sha256((ROOT/key).read_bytes()).hexdigest()==digest

def test_symbol_moment_and_maximum_entropy_certificates():
    if 'fig_7_5' not in [p.parent.name for p in FILES]:return
    r=read_result('fig_7_5')['sigma1_certificate'];x=np.array(r['atoms']);w=np.array(r['weights']);np.testing.assert_allclose(w.sum(),1,atol=1e-7);np.testing.assert_allclose(w@x,0,atol=1e-6);np.testing.assert_allclose(x.T@np.diag(w)@x,np.eye(2),atol=2e-5)
    r=read_result('example_7_2');a=np.array(r['support']);p=np.array(r['probabilities']);assert p.min()>=-1e-10;assert abs(p.sum()-1)<1e-8;assert .5-1e-8<=a*a@p<=.6+1e-8;assert -.3-1e-8<=(3*a**3-2*a)@p<=-.2+1e-8

def test_polynomial_farkas_and_quartic_witnesses():
    if 'fig_8_14' not in [p.parent.name for p in FILES]:return
    r=read_result('fig_8_14');X=np.array(r['points']['X']);Y=np.array(r['points']['Y']);P=np.vstack([X,Y]);labels=np.r_[-np.ones(len(X)),np.ones(len(Y))]
    for degree,item in r['degree_results'].items():
        A=np.column_stack([P[:,0]**i*P[:,1]**j for i,j in item['monomial_powers']]);signed=labels[:,None]*A
        if item['status']=='infeasible':
            w=np.array(item['farkas_weights']);assert w.min()>-1e-7;assert abs(w.sum()-1)<1e-7;assert np.linalg.norm(signed.T@w,np.inf)<1e-7
        else:assert np.min(signed@np.array(item['coefficients']))>1-1e-6

def test_basis_full_dimension_and_three_conventions():
    if 'fig_6_21' not in [p.parent.name for p in FILES]:return
    r=read_result('fig_6_21');assert len(r['variants'])==3
    for name,v in r['variants'].items():
        assert len(v['grid'])==501;assert max(v['selected_indices_zero_based'])<30561;assert v['history'][-1]['gap']<2e-6
        y=np.array(v['signal']);yh=np.array(v['refit']);assert abs(np.sum((y-yh)**2)/np.sum(y*y)-v['relative_squared_refit_error'])<1e-12
    assert r['variants']['CVX_endpoint_grid']['grid'][-1]==1
    assert r['variants']['book_literal_0.005_grid']['grid'][-1]==2.5

def test_saved_full_gabor_primal_dual_certificate():
    if 'fig_6_21' not in [p.parent.name for p in FILES]:return
    for name,v in read_result('fig_6_21')['variants'].items():
        t=np.array(v['grid']);y=np.array(v['signal']);B=np.exp(-((t[:,None]-.002*np.arange(501)[None,:])/.05)**2)
        if name=='CVX_endpoint_grid':B[B<.001]=0
        A=np.column_stack([B]+[B*np.cos(5*k*t)[:,None] for k in range(1,31)]+[B*np.sin(5*k*t)[:,None] for k in range(1,31)])
        index=np.array(v['primal_certificate_indices_zero_based']);coef=np.array(v['primal_certificate_coefficients']);dual=np.array(v['dual_certificate']);assert np.max(abs(A.T@dual))<=1+1e-9
        primal=np.linalg.norm(A[:,index]@coef-y)**2+np.linalg.norm(coef,1);lower=y@dual-dual@dual/4
        assert -1e-8<=primal-lower<=2e-6;assert abs(primal-v['certified_primal_objective'])<1e-8;assert abs(lower-v['certified_dual_objective'])<1e-8

def test_book_signal_reconstruction_targets():
    if 'fig_6_8' not in [p.parent.name for p in FILES]:return
    r=read_result('fig_6_8');y=np.array(r['observed'])
    np.testing.assert_allclose([np.linalg.norm(np.array(x['x'])-y) for x in r['reconstructions']],[8,3,1],atol=1e-7)
    r=read_result('fig_6_11')
    np.testing.assert_allclose([np.linalg.norm(np.diff(x['x']),1) for x in r['tv_reconstructions']],[10,8,5],atol=1e-5)
