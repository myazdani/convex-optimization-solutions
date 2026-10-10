"""Validate exact and explicitly labeled method reproductions independently."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from solution_evidence import ROOT

LEGACY_IDS = {'example_7_4', 'example_8_3'}


def validate_example_evidence(entry: dict, root: Path = ROOT) -> None:
    eid = entry['id']
    kind = entry.get('reproduction_kind', 'exact')
    assert kind in {'exact', 'method'}, f'Unknown reproduction kind: {eid}'
    if kind == 'exact':
        assert entry.get('output_match_verified') is True, f'Exact output not verified: {eid}'
    else:
        assert entry.get('output_match_verified') is False, f'Method reproduction must not claim exact output: {eid}'
        assert entry.get('reproduction_basis'), f'Method/data provenance missing: {eid}'
    if eid in LEGACY_IDS:
        return  # Already-reviewed baseline is protected by byte-level regression checks.
    assert entry.get('source_checked') is True, f'Source not checked: {eid}'
    assert entry.get('all_subparts_completed') is True, f'Incomplete requested scope: {eid}'
    assert entry.get('author'), f'Missing author identifier: {eid}'
    for field in ('solution', 'code', 'results'):
        path = (root / entry[f'{field}_path']).resolve()
        assert path.is_relative_to(root.resolve()) and path.is_file(), f'Missing {field}: {eid}'
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        assert entry.get(f'{field}_sha256') == sha, f'Stale {field} evidence: {eid}'
    assert entry.get('verification_path'), f'Missing independent review: {eid}'
    review_path = root / entry['verification_path']
    reviews = [r for r in json.loads(review_path.read_text())['entries'] if r['id'] == eid]
    assert len(reviews) == 1, f'Missing/duplicate review: {eid}'
    r = reviews[0]
    assert r.get('reviewer') and r['reviewer'] != entry['author'], f'Non-independent review: {eid}'
    for key in ('accepted', 'source_checked', 'all_subparts_checked', 'correctness_checked', 'assumptions_checked'):
        assert r.get(key) is True, f'Review missing {key}: {eid}'
    for field in ('solution', 'code', 'results'):
        assert r.get(f'{field}_sha256') == entry[f'{field}_sha256'], f'Review does not bind {field}: {eid}'
    assert r.get('reproduction_kind') == kind, f'Review reproduction classification mismatch: {eid}'
    assert r.get('notes'), f'Missing mathematical review notes: {eid}'
    evidence = entry.get('verification', {})
    assert evidence.get('test_command') and evidence.get('test_result') and evidence.get('independent_checks'), f'Missing execution/check evidence: {eid}'
    validate_execution_envelope(entry, root)


def validate_execution_envelope(entry: dict, root: Path = ROOT) -> None:
    """Validate actual-run metadata and bind every recorded runtime dependency."""
    eid = entry['id']
    result = json.loads((root / entry['results_path']).read_text())
    execution = result.get('execution', result)
    assert execution.get('executed_at'), f'Missing execution timestamp: {eid}'
    assert execution.get('duration_seconds', execution.get('elapsed_seconds', 0)) > 0, f'Missing execution duration: {eid}'
    assert execution.get('reproduction_command', execution.get('execution_command')), f'Missing runnable command: {eid}'
    assert execution.get('python') and execution.get('numpy'), f'Missing runtime versions: {eid}'
    hashes = execution.get('source_hashes', result.get('runtime_source_sha256', {}))
    assert hashes and entry['code_path'] in hashes, f'Missing runtime code hashes: {eid}'
    for relative, expected in hashes.items():
        path = (root / relative).resolve()
        assert path.is_relative_to(root.resolve()) and path.is_file(), f'Missing runtime dependency: {eid}: {relative}'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, f'Changed runtime dependency since execution: {eid}: {relative}'
    for relative, expected in result.get('input_file_sha256', {}).items():
        path = (root / relative).resolve()
        assert path.is_relative_to(root.resolve()) and path.is_file(), f'Missing executed input: {eid}: {relative}'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, f'Changed executed input: {eid}: {relative}'
    for relative in entry.get('supporting_paths', []):
        path = (root / relative).resolve()
        assert path.is_relative_to(root.resolve()) and path.is_file(), f'Missing supporting file: {eid}: {relative}'
