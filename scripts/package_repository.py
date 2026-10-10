"""Make a portable source-and-results ZIP from the canonical delivery inventory.

Algorithm studies explicitly distinguish published artifacts from regenerable raw
inputs and resumable staging files. Respect their accepted submission allowlist
even after a user has rerun studies and materialized larger local intermediates.
"""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

EXCLUDED_PARTS = {'.venv', '__pycache__', '.pytest_cache', '.git', 'build', 'dist', '.mypy_cache', '.ruff_cache'}
STAGING_DIRECTORIES = {
    Path('artifacts/examples/_lp_scaling_trials'),
    Path('artifacts/examples/_sdp_scaling_trials'),
}


def algorithm_delivery_contract(root: Path) -> tuple[set[Path], set[Path]]:
    """Return current canonical payload and governed artifact directories."""
    submission = root / 'coverage/submissions/examples_ch09_11.json'
    if not submission.is_file():
        return set(), set()
    payload, directories = set(), set()
    for entry in json.loads(submission.read_text())['entries']:
        paths = [entry[key] for key in ('solution_path', 'code_path', 'results_path')]
        paths.extend(entry.get('supporting_paths', []))
        for relative in paths:
            path = Path(relative)
            resolved = (root / path).resolve()
            if path.is_absolute() or not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
                raise ValueError(f'Missing or invalid canonical delivery file: {relative}')
            payload.add(path)
        directory = Path(entry['results_path']).parent
        if not directory.is_relative_to(Path('artifacts/examples')):
            raise ValueError(f'Unexpected algorithm artifact directory: {directory}')
        directories.add(directory)
    return payload, directories


def repository_files(root: Path, output: Path) -> list[Path]:
    payload, directories = algorithm_delivery_contract(root)
    files = []
    for path in root.rglob('*'):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in EXCLUDED_PARTS or part.endswith('.egg-info') for part in relative.parts):
            continue
        if path.name in {'bv_cvxbook.pdf', 'bv_cvxbook.txt'} or path.name.endswith(('.pyc', '.pyo', '.tmp')):
            continue
        if path.resolve() == output.resolve():
            continue
        if any(relative.is_relative_to(directory) for directory in STAGING_DIRECTORIES):
            continue
        if any(relative.is_relative_to(directory) for directory in directories) and relative not in payload:
            continue
        files.append(path)
    return sorted(files)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=root.parent / 'convex-optimization-solutions.zip')
    args = parser.parse_args()
    files = repository_files(root, args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            archive.write(path, arcname=Path(root.name) / path.relative_to(root))
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Archive CRC verification failed')
    sha = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(args.output.suffix + '.sha256').write_text(f'{sha}  {args.output.name}\n')
    print(f'{args.output}: {len(files)} files, {args.output.stat().st_size:,} bytes; sha256 {sha}')


if __name__ == '__main__':
    main()
