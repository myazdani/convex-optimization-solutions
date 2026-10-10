import hashlib
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from markdown_math import normalize_fenced_less_than, validate_baseline_markdown_exception


def test_normalization_preserves_prose_code_and_inline_math():
    before = 'Prose <x and $`x<y`$.\n```python\nx<y\n```\n```math\nx<y\n```\n'
    assert normalize_fenced_less_than(before) == before.replace('```math\nx<y', '```math\nx\\lt y')


@pytest.mark.parametrize('mutation', ['unaccepted', 'wrong_reviewer', 'original_hash', 'nonmath_edit', 'nonpermitted_path'])
def test_baseline_exception_rejects_expanded_or_unreviewed_change(mutation):
    original = '# Note\n```math\nx<y\n```\n'
    result = normalize_fenced_less_than(original).encode()
    expected = hashlib.sha256(original.encode()).hexdigest()
    evidence = dict(accepted=True, reviewer='root_independent_format_review', original_text=original,
                    original_sha256=expected, corrected_sha256=hashlib.sha256(result).hexdigest())
    relative='docs/solutions/9.31.md'
    if mutation == 'unaccepted': evidence['accepted']=False
    elif mutation == 'wrong_reviewer': evidence['reviewer']='author'
    elif mutation == 'original_hash': evidence['original_text']='changed'
    elif mutation == 'nonmath_edit':
        result=result.replace(b'Note',b'Changed'); evidence['corrected_sha256']=hashlib.sha256(result).hexdigest()
    else: relative='src/numerical_code.py'
    with pytest.raises(AssertionError): validate_baseline_markdown_exception(relative, expected, result, evidence)
