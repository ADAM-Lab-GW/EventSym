import os
import shutil
from pathlib import Path

# Original root directory
SRC_ROOT = Path("path/to/your/root")   # <-- change this
# Destination root where "unrecorded" contents will be copied
DST_ROOT = Path("path/to/destination")  # <-- change this

for path in SRC_ROOT.rglob("*"):
    if path.is_dir() and path.name == "unrecorded":
        # Parent folder (e.g., hazard_symbols/flammable)
        parent = path.parent.relative_to(SRC_ROOT)

        for file in path.glob("*"):
            if file.is_file():
                # Preserve structure under DST_ROOT
                target_dir = DST_ROOT / parent
                target_dir.mkdir(parents=True, exist_ok=True)

                shutil.copy2(file, target_dir / file.name)
                print(f"Copied {file} -> {target_dir/file.name}")
