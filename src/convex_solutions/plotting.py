"""Headless plotting with writable temporary caches when no writable cache is configured."""
from __future__ import annotations
import os
from pathlib import Path
from tempfile import TemporaryDirectory

_runtime_cache: TemporaryDirectory | None = None


def _usable_cache(value: str | None) -> bool:
    if not value:
        return False
    path = Path(value).expanduser()
    return path.is_dir() and os.access(path, os.W_OK | os.X_OK)


def pyplot():
    global _runtime_cache
    if not all(_usable_cache(os.environ.get(key)) for key in ("MPLCONFIGDIR", "XDG_CACHE_HOME")):
        if _runtime_cache is None:
            _runtime_cache = TemporaryDirectory(prefix="convex-solutions-plot-cache-")
        root = Path(_runtime_cache.name)
        for key, folder in (("MPLCONFIGDIR", "matplotlib"), ("XDG_CACHE_HOME", "cache")):
            if not _usable_cache(os.environ.get(key)):
                path = root / folder
                path.mkdir(exist_ok=True)
                os.environ[key] = str(path)
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    return plt
