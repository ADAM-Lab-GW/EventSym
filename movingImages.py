import shutil
from pathlib import Path

# Root directory containing "Kate", "Phuoc", etc.
SRC_ROOT = Path("/home/adamlab/Downloads/subset_symbols")
DST_ROOT = Path("recordings/input_files")

# Loop through each person folder (Kate, Phuoc, ...)
for person_folder in SRC_ROOT.iterdir():
    if person_folder.is_dir():  # e.g., Kate, Phuoc
        print(f"Processing folder: {person_folder}")  # Debug: Show folder being processed
        
        # Search inside this person's folder for "unrecorded" (case-insensitive)
        for path in person_folder.rglob("*"):
            if path.name.lower() == "unrecorded":  # Case-insensitive check
                print(f"Found 'unrecorded' folder: {path}")  # Debug: Show path found
                parent = path.parent.relative_to(person_folder)

                for file in path.glob("*"):
                    if file.is_file():
                        target_dir = DST_ROOT / parent
                        target_dir.mkdir(parents=True, exist_ok=True)

                        shutil.copy2(file, target_dir / file.name)
                        print(f"Copied {file} -> {target_dir / file.name}")
