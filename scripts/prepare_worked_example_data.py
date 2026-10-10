"""Convert hash-pinned official CVXOPT numeric inputs without executable pickle globals.

Usage: python scripts/prepare_worked_example_data.py /path/to/official-downloads
No network access and no arbitrary pickle imports are used.
"""
import hashlib,json,pickle,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
class NumericOnly(pickle.Unpickler):
    def find_class(self,module,name):
        if (module,name)!=('cvxopt.base','matrix'):raise pickle.UnpicklingError(f'Forbidden global {module}.{name}')
        def matrix(values,shape,typecode):
            if typecode not in ('d','i'):raise ValueError('Non-real matrix')
            a=np.asarray(values,dtype=float)
            if a.size!=int(np.prod(shape)) or a.size>1000000:raise ValueError('Invalid shape')
            return a.reshape(shape,order='F').tolist()
        return matrix
    def persistent_load(self,pid):raise pickle.UnpicklingError('Persistent IDs forbidden')

def main(source):
    files={d['url'].split('/')[-1]:d for e in json.loads((ROOT/'coverage/worked_examples.json').read_text())['entries'] for d in e['official_data_files']}
    dest=ROOT/'data/worked_examples';dest.mkdir(exist_ok=True)
    for name,d in files.items():
        p=Path(source)/name
        if hashlib.sha256(p.read_bytes()).hexdigest()!=d['sha256']:raise ValueError(f'Hash mismatch: {name}')
        with p.open('rb') as f:obj=NumericOnly(f,encoding='latin1').load()
        output=dest/(p.stem+'.json');output.write_text(json.dumps(obj,indent=2)+'\n')
        provenance={'kind':'upstream_numeric_data','source_url':d['url'],'original_filename':name,'original_sha256':d['sha256'],'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'rights':'CVXOPT data: GPL-3.0-or-later; see LICENSE-CVXOPT-DATA.txt and https://cvxopt.org/copyright.html','transformations':['Restricted pickle decoder only allows real cvxopt.base.matrix construction; preserve column-major shape; serialize numeric arrays as JSON.']}
        (dest/(p.stem+'.provenance.json')).write_text(json.dumps(provenance,indent=2)+'\n')
        print(name,[(k,np.shape(v)) for k,v in obj.items()])
if __name__=='__main__':main(sys.argv[1])
