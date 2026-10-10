"""The portable package must preserve canonical evidence, not staging debris."""
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from package_repository import repository_files


@pytest.fixture
def repository(tmp_path):
    entry = {
        'id': 'fig_11_19',
        'solution_path': 'docs/examples/fig_11_19.md',
        'code_path': 'examples/algorithm_evidence.py',
        'results_path': 'artifacts/examples/fig_11_19/results.json',
        'supporting_paths': [
            'artifacts/examples/fig_11_19/input_manifest.json',
            'artifacts/examples/fig_11_19/instance_50.npz',
            'artifacts/examples/fig_11_19/trials_1000.json',
        ],
    }
    required = [entry[key] for key in ('solution_path', 'code_path', 'results_path')]
    required += entry['supporting_paths']
    unrelated = ['artifacts/examples/application/large.npz', 'artifacts/9.30/results.json']
    for relative in required + unrelated:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture\n')
    submission = tmp_path / 'coverage/submissions/examples_ch09_11.json'
    submission.parent.mkdir(parents=True)
    submission.write_text(json.dumps({'entries': [entry]}))
    return tmp_path, required, unrelated


@pytest.mark.parametrize('relative', [
    'artifacts/examples/fig_11_19/results.full.json',
    'artifacts/examples/fig_11_19/instance_1000.npz',
    'artifacts/examples/_lp_scaling_trials/1000_0.json',
    'artifacts/examples/_sdp_scaling_trials/1000_0.json',
])
def test_local_regenerated_arrays_and_staging_are_excluded(repository, relative):
    root, _, _ = repository
    extra = root / relative
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_text('local materialization\n')
    assert extra not in repository_files(root, root.parent / 'package.zip')
    assert extra.is_file()  # Packaging never deletes local evidence.


def test_canonical_shards_and_unrelated_numerical_artifacts_are_retained(repository):
    root, required, unrelated = repository
    selected = {str(path.relative_to(root)) for path in repository_files(root, root.parent / 'package.zip')}
    assert set(required + unrelated) <= selected


def test_missing_canonical_supporting_file_blocks_packaging(repository):
    root, _, _ = repository
    (root / 'artifacts/examples/fig_11_19/trials_1000.json').unlink()
    with pytest.raises(ValueError, match='Missing or invalid canonical'):
        repository_files(root, root.parent / 'package.zip')
