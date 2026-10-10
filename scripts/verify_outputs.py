"""Check complete CLI artifacts for every implemented inventory entry."""
from pathlib import Path
import argparse
import json
from PIL import Image
from convex_solutions.registry import load_inventory, implemented

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("artifacts"))
args = parser.parse_args()
checked = []


def reject_constant(value):
    raise ValueError(f"Nonstandard/nonfinite JSON constant: {value}")


for exercise in load_inventory()["exercises"]:
    eid = exercise["id"]
    if not implemented(eid):
        continue
    directory = args.output / eid
    result = json.loads((directory / "results.json").read_text(), parse_constant=reject_constant)
    assert result["exercise_id"] == eid
    assert result["environment"]["packages"]["numpy"]
    assert result["duration_seconds"] >= 0
    assert isinstance(result["results"], dict)
    figures = list(directory.glob("*.png"))
    for figure in figures:
        with Image.open(figure) as img:
            assert min(img.size) >= 100, f"Unexpectedly small figure: {figure}"
            img.verify()
    checked.append({"id": eid, "png_count": len(figures)})
print(json.dumps({"verified_artifact_structure": checked, "count": len(checked)}, indent=2))
