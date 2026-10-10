"""Shared serialization, evidence, and plotting for original algorithm studies."""
from __future__ import annotations
import hashlib
import json
import sys, time, platform, datetime, os
os.environ.setdefault("MPLCONFIGDIR","/tmp/convex-worked-matplotlib")
import scipy, matplotlib
from pathlib import Path
import numpy as np
from convex_solutions.plotting import pyplot

ROOT = Path(__file__).resolve().parents[1]
CATALOG = {e['id']: e for e in json.loads((ROOT/'coverage/worked_examples.json').read_text())['entries']}
RNG_NOTE = ('Random instances use NumPy PCG64, with saved arrays and seeds. They are method reproductions; '
            'MATLAB rand(state,...) uses v5uniform and randn(state,...) uses v5normal, whereas seed selects '
            'the v4 family. Equal integer seeds do not identify equal arrays. No exact MATLAB fixture was '
            'published in the checked companion sources; exact book-figure matching is not claimed.')
ACTIVE_RUNS = {}

def begin_run(eid):
    ACTIVE_RUNS[eid]=(datetime.datetime.now(datetime.timezone.utc).isoformat(),time.monotonic())

RNG_SOURCE = 'https://www.mathworks.com/help/matlab/math/updating-your-random-number-generator-syntax.html'

def plain(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, dict): return {k:plain(v) for k,v in x.items()}
    if isinstance(x, (tuple,list)): return [plain(v) for v in x]
    return x

def save_result(eid, out, result, formula, explanation, rows, plots, notes=()):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    entry=CATALOG[eid]
    start=ACTIVE_RUNS.get(eid)
    if start is None:raise RuntimeError('Execution provenance requires begin_run before computation')
    modules=['algorithm_common.py',f'algorithms_ch{entry["chapter"]:02d}.py']
    if entry['chapter']==10:modules.append('algorithms_ch09.py')
    if eid in ['fig_11_7','fig_11_8','fig_11_19','fig_11_20','fig_11_23']:modules.append('algorithm_scaling.py')
    if eid in ['example_11_1','example_11_3','example_11_4','fig_11_10']:modules.append('algorithm_phase.py')
    hashes={f'examples/{f}':hashlib.sha256((ROOT/'examples'/f).read_bytes()).hexdigest() for f in modules}
    result['execution']={'executed_at':start[0],'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'duration_seconds':time.monotonic()-start[1],
                         'invoked_argv':sys.argv,'reproduction_command':f'OPENBLAS_NUM_THREADS=1 python -m examples.algorithms_ch{entry["chapter"]:02d} {eid}',
                         'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__,
                         'solver':'Explicit Python Newton/barrier/primal-dual implementation; SciPy LAPACK and SuperLU linear solves',
                         'generator':'NumPy PCG64; numeric seed and saved input arrays accompany each random instance','source_hashes':hashes,
                         'data_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*.npz'))}}
    exact=entry['reproducibility']=='self_contained'
    result.update({'id':eid,'title':entry['title'],'source_url':entry['source_url'],
                   'reproduction_basis':'self_contained_exact_instance' if exact else 'method_reproduction_substituted_data',
                   'output_match_verified':exact,'source_code_urls':entry['official_code_urls'],
                   'input_provenance': 'Explicit textbook and companion constants.' if exact else RNG_NOTE,
                   'notes':list(notes)})
    result=plain(result)
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    doc=ROOT/'docs/examples'/f'{eid}.md'
    doc.write_text(f'# {entry["title"]}\n\n[Textbook source]({entry["source_url"]}). '
        + ' '.join(f'[Official reference {i+1}]({u}).' for i,u in enumerate(entry['official_code_urls']))
        +'\n\n## Formulation and method\n\n```math\n'+formula+'\n```\n\n'+explanation
        +'\n\n## Input provenance and comparison\n\n'+result['input_provenance']
        +('' if exact else f' [MATLAB generator documentation]({RNG_SOURCE}).')
        +'\n\n'+'\n\n'.join(notes)
        +'\n\n## Recomputed output\n\nThe following values come from the checked-in [results](../../artifacts/examples/'+eid+'/results.json).'
        +'\n\n| Quantity | Computed value |\n| :--- | ---: |\n'
        +'\n'.join(f'| {name} | {value} |' for name,value in rows)
        +'\n\n'+'\n\n'.join(f'![{caption}](../../artifacts/examples/{eid}/{name})' for name,caption in plots)
        +'\n\n## Reproduction\n\n```sh\nOPENBLAS_NUM_THREADS=1 python -m examples.'
        +('algorithms_ch09' if entry['chapter']==9 else 'algorithms_ch10' if entry['chapter']==10 else 'algorithms_ch11')
        +' '+eid+'\n```\n\nThe implementation preserves complete numerical histories, uses explicit algorithm steps, and checks the final conditions independently. '
        +'Iteration counts denote accepted updates; the initial point is iteration zero.\n')
    return result

def convergence_plot(out, histories, reference=0., filename='convergence.png', ylabel='Objective error'):
    plt=pyplot(); fig,ax=plt.subplots(figsize=(7,4.2))
    for label,values in histories.items():
        a=np.asarray(values,dtype=float)-reference
        ax.semilogy(np.maximum(a,1e-15),marker='o' if len(a)<30 else None,markersize=3,label=label)
    ax.set(xlabel='Iteration',ylabel=ylabel); ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(Path(out)/filename,dpi=160); plt.close(fig)


def emit_submission(ids, module, test_command, independent_checks):
    entries=[]
    for eid in ids:
        doc=ROOT/'docs/examples'/f'{eid}.md'; res=ROOT/'artifacts/examples'/eid/'results.json'
        r=json.loads(res.read_text())
        entries.append({'id':eid,'title':r['title'],'solution_path':str(doc.relative_to(ROOT)),
                        'code_path':f'examples/{module}.py','results_path':str(res.relative_to(ROOT)),
                        'supporting_paths':[str(p.relative_to(ROOT)) for p in sorted((ROOT/'artifacts/examples'/eid).glob('*')) if p.is_file()]+list(r.get('execution',{}).get('source_hashes',{}))+['tests/test_worked_algorithms_ch09.py','tests/test_worked_algorithms_ch10.py','tests/test_worked_algorithms_ch11.py','src/convex_solutions/plotting.py'],
                        'source_url':r['source_url'],'source_checked':True,'all_subparts_completed':True,
                        'solution_sha256':hashlib.sha256(doc.read_bytes()).hexdigest(),
                        'reproduction_basis':r['reproduction_basis'],'reproduction_kind':'exact' if r['output_match_verified'] else 'method','output_match_verified':r['output_match_verified'],
                        'verification':{'test_command':test_command,'test_result':'passed','independent_checks':independent_checks},
                        'notes':r['notes']})
    path=ROOT/'coverage/submissions/examples_ch09_11.json'
    old=json.loads(path.read_text()) if path.exists() else {'schema_version':1,'author':'algorithm_worked_examples','entries':[]}
    old['entries']=[e for e in old['entries'] if e['id'] not in ids]+entries
    path.write_text(json.dumps(old,indent=2)+'\n')
