"""Boyd and Vandenberghe, exercise 10.16. See docs/solutions/10.16.md."""
from pathlib import Path
from .common import run_10_16


def solve(output_dir: Path | None = None, seed: int = 0) -> dict:
    """Run all requested subparts and optional artifact generation."""
    return run_10_16(output_dir=output_dir, seed=seed)
