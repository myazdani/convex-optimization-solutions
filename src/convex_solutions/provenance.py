"""Offline provenance checks; never silently retrieve or replace inputs."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_data(path: Path, provenance: Path | None = None) -> dict:
    path = Path(path)
    provenance = provenance or path.with_name(path.name + ".provenance.json")
    record = json.loads(Path(provenance).read_text(encoding="utf-8"))
    if record.get("kind") != "upstream":
        raise ValueError("Stored external files must have upstream provenance")
    for field in ("id", "source_url", "source_title", "retrieved_at", "sha256", "rights"):
        if not record.get(field):
            raise ValueError(f"Missing provenance field: {field}")
    actual = file_sha256(path)
    if actual != record["sha256"]:
        raise ValueError(f"Checksum mismatch for {path.name}")
    return record
