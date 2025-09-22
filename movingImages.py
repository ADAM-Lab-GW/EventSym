import shutil
from pathlib import Path

# Root directory containing "Kate", "Phuoc", etc.
SRC_ROOT = Path("C:/Users/zahin/Downloads/subset_symbols-20250922T140954Z-1-001/subset_symbols")
DST_ROOT = Path("recordings/input_files")

# Loop through each person folder (Kate, Phuoc, ...)
for person_folder in SRC_ROOT.iterdir():
    if person_folder.is_dir():  # e.g., Kate, Phuoc
        # Search inside this person's folder for "unrecorded"
        for path in person_folder.rglob("unrecorded"):
            # Relative path excluding the top-level (Kate/Phuoc)
            parent = path.parent.relative_to(person_folder)

            for file in path.glob("*"):
                if file.is_file():
                    target_dir = DST_ROOT / parent
                    target_dir.mkdir(parents=True, exist_ok=True)

                    shutil.copy2(file, target_dir / file.name)
                    print(f"Copied {file} -> {target_dir / file.name}")
