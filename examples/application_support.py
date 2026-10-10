"""Shared artifact/provenance support for original chapter 4–8 implementations."""
from __future__ import annotations
import argparse, hashlib, json, platform, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import cvxpy as cp
from convex_solutions.io import json_safe
from convex_solutions.numerics import max_violation
from convex_solutions.plotting import pyplot
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/worked_examples'
INPUTS = {'fig_4_11':['rls'],'example_6_2':['huber'],'example_6_4':['regsel'],'example_6_5':['robls'],'example_6_6':['robls'],'example_6_7':['polapprox'],'example_6_8':['polapprox'],'fig_6_24':['cvxfit'],'example_6_9':['consumerpref'],'fig_7_1':['logreg'],'fig_7_9':['expdesign'],'fig_8_10':['linsep'],'example_8_6':['placement']}
BOOK_SHA = '40d976c83c18cce1900eff8c41bd5ad408c102b813af39d05ff85678ccf8d76e'

def data(name):
    def convert(x):
        if isinstance(x, dict): return {k: convert(v) for k,v in x.items()}
        return np.asarray(x) if isinstance(x,list) else x
    return convert(json.loads((DATA / (name+'.json')).read_text()))

def optimize(problem, solver='CLARABEL', **kwargs):
    opts = dict(tol_gap_abs=1e-9,tol_gap_rel=1e-9,tol_feas=1e-9) if solver=='CLARABEL' else {}
    opts.update(kwargs)
    problem.solve(solver=solver, **opts)
    if problem.status != cp.OPTIMAL:
        raise RuntimeError(f'{solver}: {problem.status}')
    v = max_violation(problem)
    if v > 2e-5: raise RuntimeError(f'Infeasible numeric result: {v}')
    return {'objective':float(problem.value),'status':problem.status,'solver':solver,'primal_violation':v}

def figsave(fig, out, name):
    path = out/(name+'.png');fig.savefig(path,dpi=150,bbox_inches='tight');pyplot().close(fig);return path.name

def check(name, value, tolerance=1e-6):
    value=float(value)
    if not np.isfinite(value) or value>tolerance:
        raise AssertionError(f'{name}: {value} > {tolerance}')
    return {'name':name,'measured':value,'tolerance':tolerance,'passed':True}

def run(functions):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('id',choices=list(functions)+['all']);p.add_argument('--output',type=Path);a=p.parse_args()
    for id in functions if a.id=='all' else [a.id]:
        out=a.output or ROOT/'artifacts/examples'/id;out.mkdir(parents=True,exist_ok=True)
        start=time.perf_counter();r=functions[id](out);r.update(id=id,executed_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start,python=platform.python_version(),numpy=np.__version__,cvxpy=cp.__version__)
        e=next(e for e in json.loads((ROOT/'coverage/worked_examples.json').read_text())['entries'] if e['id']==id)
        r['source']={'book_pdf_sha256':BOOK_SHA,'source_url':e['source_url'],'official_code_urls':e['official_code_urls'],'official_data_files':e['official_data_files'],'implementation':'Original Python formulation and implementation; upstream numeric data attributed separately.'}
        r['code_sha256']=hashlib.sha256(Path(sys.argv[0]).read_bytes()).hexdigest()
        sources=[Path(sys.argv[0]).resolve(),Path(__file__).resolve()]+[ROOT/'src/convex_solutions'/n for n in ['io.py','numerics.py','plotting.py']]
        r['runtime_source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        r['input_file_sha256']={str((DATA/(name+'.json')).relative_to(ROOT)):hashlib.sha256((DATA/(name+'.json')).read_bytes()).hexdigest() for name in INPUTS.get(id,[])}
        r['execution_command']='OPENBLAS_NUM_THREADS=1 python '+str(Path(sys.argv[0]).resolve().relative_to(ROOT))+' '+id
        import scipy
        r['scipy']=scipy.__version__
        r['installed_solvers']=cp.installed_solvers()
        (out/'results.json').write_text(json.dumps(json_safe(r),indent=2,allow_nan=False)+'\n')
        print(id,'PASS',round(r['elapsed_seconds'],3),json_safe(r.get('summary',{})),flush=True)
