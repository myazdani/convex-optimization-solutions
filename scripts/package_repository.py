"""Make a portable source-and-results ZIP without caches or third-party sources."""
from pathlib import Path
import argparse
import hashlib
import zipfile

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=root.parent / "convex-optimization-solutions.zip")
args = parser.parse_args()
excluded = {".venv", "__pycache__", ".pytest_cache", ".git", "build", "dist", ".mypy_cache", ".ruff_cache"}
files = [path for path in root.rglob("*") if path.is_file()
         and not any(part in excluded or part.endswith(".egg-info") for part in path.relative_to(root).parts)
         and path.name not in {"bv_cvxbook.pdf", "bv_cvxbook.txt"}
         and not path.name.endswith((".pyc", ".pyo", ".tmp"))
         and path.resolve() != args.output.resolve()]
args.output.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    for path in sorted(files):
        archive.write(path, arcname=Path(root.name) / path.relative_to(root))
with zipfile.ZipFile(args.output) as archive:
    if archive.testzip() is not None:
        raise RuntimeError("Archive CRC verification failed")
sha = hashlib.sha256(args.output.read_bytes()).hexdigest()
args.output.with_suffix(args.output.suffix + ".sha256").write_text(f"{sha}  {args.output.name}\n")
print(f"{args.output}: {len(files)} files, {args.output.stat().st_size:,} bytes; sha256 {sha}")
