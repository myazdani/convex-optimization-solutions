"""Bundle a snapshot of the canonical inventory for wheel installations."""
from pathlib import Path
import shutil
root = Path(__file__).resolve().parents[1]
target = root / "src/convex_solutions/resources"
target.mkdir(parents=True, exist_ok=True)
shutil.copy2(root / "coverage/inventory.json", target / "inventory.json")
print(target / "inventory.json")
