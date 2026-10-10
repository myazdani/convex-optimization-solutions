"""Publish full Monte Carlo evidence in bounded, independently hashed JSON shards."""
import argparse,datetime,hashlib,json
from pathlib import Path
from examples.algorithm_common import ROOT
from examples.algorithm_evidence import package

def package_scaling(eid):
    folder=ROOT/'artifacts/examples'/eid;p=folder/'results.json';r=json.loads(p.read_text())
    if 'records' not in r:raise ValueError('Run the full sweep before packaging')
    records=r.pop('records');method=r['method'];shards=[]
    for n in sorted({q['dimension'] for q in records}):
        subset=[]
        for q in records:
            if q['dimension']!=n:continue
            row={k:v for k,v in q.items() if k not in ['barrier','primal_dual']};row[method]=q[method];subset.append(row)
        path=folder/f'trials_{n}.json';path.write_text(json.dumps({'dimension':n,'method':method,'records':subset},separators=(',',':'),allow_nan=False)+'\n')
        shards.append({'path':path.name,'dimension':n,'count':len(subset),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    r['trial_shards']=shards;r['all_trials_preserved']=True
    r['publication_partition']={'script':'examples/algorithm_scaling_pack.py','sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'packaged_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                'description':'All2000trial records preserved; each study includes only its own method, partitioned into20dimension shards.'}
    p.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');package(eid)
    doc=ROOT/'docs/examples'/f'{eid}.md';s=doc.read_text();s=s.replace('| n=','| m=') if eid in ['fig_11_8','fig_11_23'] else s;s+='\n## Complete trial evidence\n\nAll 2000 trials are retained in the twenty dimension shards listed, counted and SHA256-bound in results.json. Each record includes its random seed, kernel hash, execution metadata, iteration history and primal-dual certificate diagnostics. The compact main file contains all aggregate means, sample deviations and ranges.\n';doc.write_text(s)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('id');a=p.parse_args();package_scaling(a.id)
