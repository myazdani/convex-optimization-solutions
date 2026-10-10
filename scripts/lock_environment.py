"""Record relevant installed runtime/test dependencies, excluding unrelated tools.

The snapshot is platform-specific. Refresh deliberately after validating all
experiments; do not mistake this file for a universal cross-platform lock.
"""
from importlib.metadata import distribution
from pathlib import Path
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
import platform

pending = ["numpy", "scipy", "matplotlib", "cvxpy", "pytest", "jsonschema", "setuptools"]
versions = {}
while pending:
    name = canonicalize_name(pending.pop())
    if name in versions:
        continue
    package = distribution(name)
    versions[name] = package.version
    for raw in package.requires or []:
        requirement = Requirement(raw)
        if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
            pending.append(requirement.name)
header = (f"# Validated environment snapshot: Python {platform.python_version()}, "
          f"{platform.system()} {platform.machine()}\n"
          "# Install in a fresh virtual environment; then pip install -e . --no-deps\n")
root = Path(__file__).resolve().parents[1]
(root / "requirements-lock.txt").write_text(header + "".join(
    f"{name}=={version}\n" for name, version in sorted(versions.items())))
print(f"Pinned {len(versions)} relevant distributions")
