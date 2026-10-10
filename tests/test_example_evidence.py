"""Actual numerical runs must record and preserve their implementation provenance."""
import hashlib
import json
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from example_evidence import validate_execution_envelope


@pytest.fixture
def executed(tmp_path):
    (tmp_path/'study.py').write_text('print(2 + 2)\n')
    execution={'executed_at':'2026-10-10T00:00:00Z','duration_seconds':.1,
               'reproduction_command':'python study.py','python':'3.12','numpy':'2.3',
               'source_hashes':{'study.py':hashlib.sha256((tmp_path/'study.py').read_bytes()).hexdigest()}}
    result={'execution':execution,'value':4}
    (tmp_path/'results.json').write_text(json.dumps(result))
    entry={'id':'fixture','code_path':'study.py','results_path':'results.json'}
    return tmp_path,entry,result


def test_complete_execution_provenance(executed):
    root,entry,_=executed
    validate_execution_envelope(entry,root)


@pytest.mark.parametrize('field',['executed_at','duration_seconds','reproduction_command','python','numpy','source_hashes'])
def test_incomplete_execution_provenance_rejected(executed,field):
    root,entry,result=executed
    result['execution'].pop(field)
    (root/'results.json').write_text(json.dumps(result))
    with pytest.raises(AssertionError):validate_execution_envelope(entry,root)


def test_changed_helper_or_kernel_rejected(executed):
    root,entry,_=executed
    (root/'study.py').write_text('print(3 + 3)\n')
    with pytest.raises(AssertionError,match='Changed runtime'):validate_execution_envelope(entry,root)
