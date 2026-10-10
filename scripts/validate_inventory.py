"""Validate the canonical source manifest with its JSON Schema."""
import json
from pathlib import Path
import jsonschema

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "coverage/inventory.json").read_text())
schema = json.loads((root / "coverage/inventory.schema.json").read_text())
jsonschema.validate(manifest, schema)
print(f"Validated {len(manifest['exercises'])} exercise records")
