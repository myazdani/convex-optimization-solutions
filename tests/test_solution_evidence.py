"""Completion requires independent mathematical review of exact saved bytes."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from solution_evidence import validate_solution_evidence
from markdown_math import validate_baseline_markdown_exception


@pytest.fixture
def accepted(tmp_path):
    solution = tmp_path / 'solution.md'
    solution.write_text('# Original proof\n\nA complete proof goes here for this validator fixture.\n')
    sha = hashlib.sha256(solution.read_bytes()).hexdigest()
    entry = dict(id='2.1', status='verified', source_checked=True, all_subparts_completed=True,
                 subparts=['main'], author='author', solution_path='solution.md',
                 solution_sha256=sha, verification_path='review.json')
    review = dict(id='2.1', reviewer='independent-reviewer', solution_sha256=sha,
                  accepted=True, source_checked=True, all_subparts_checked=True,
                  correctness_checked=True, assumptions_checked=True, subparts=['main'],
                  notes='Checked the argument and its hypotheses against the source.')
    (tmp_path / 'review.json').write_text(json.dumps({'entries':[review]}))
    return tmp_path, entry, review


def test_proof_needs_no_fabricated_numerical_results(accepted):
    root, entry, _ = accepted
    validate_solution_evidence(entry, root)


@pytest.mark.parametrize('field', ['source_checked', 'all_subparts_completed', 'solution_sha256', 'author', 'subparts'])
def test_incomplete_or_unbound_solution_is_rejected(accepted, field):
    root, entry, _ = accepted
    entry.pop(field)
    with pytest.raises(AssertionError):
        validate_solution_evidence(entry, root)


@pytest.mark.parametrize('mutation', ['same_author', 'stale_hash', 'not_accepted', 'unchecked_source', 'missing_part', 'no_mathematical_notes'])
def test_independent_review_gate(accepted, mutation):
    root, entry, review = accepted
    if mutation == 'same_author': review['reviewer'] = 'author'
    elif mutation == 'stale_hash': review['solution_sha256'] = '0' * 64
    elif mutation == 'not_accepted': review['accepted'] = False
    elif mutation == 'unchecked_source': review['source_checked'] = False
    elif mutation == 'missing_part': review['subparts'] = []
    else: review['notes'] = ''
    (root / 'review.json').write_text(json.dumps({'entries':[review]}))
    with pytest.raises(AssertionError):
        validate_solution_evidence(entry, root)


def test_original_numerical_deliverables_are_unchanged():
    baseline = json.loads((ROOT / 'coverage/preserved_numerical_baseline.json').read_text())
    exceptions = json.loads((ROOT / 'coverage/markdown_rendering_exceptions.json').read_text())['entries']
    assert {x['path'] for x in exceptions} == {'docs/solutions/9.31.md', 'docs/solutions/10.16.md', 'docs/solutions/11.1.md'}
    by_path = {x['path']: x for x in exceptions}
    for relative, expected in baseline['sha256'].items():
        actual = (ROOT / relative).read_bytes()
        if relative in by_path:
            validate_baseline_markdown_exception(relative, expected, actual, by_path[relative])
        else:
            assert hashlib.sha256(actual).hexdigest() == expected, relative
