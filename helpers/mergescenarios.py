import os
import re
import shutil
from pathlib import Path


def natural_sort_key(s):
    """Key for natural alphanumeric sorting (e.g., scenario_2 before scenario_10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]


def merge_scenarios():
    base_dir = Path.cwd()
    dest_dir = base_dir / "all_scenerios"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Find all folders matching generated_scen* or generated_scenerios* / generated_scenarios*
    source_dirs = [
        d for d in base_dir.iterdir()
        if d.is_dir() and re.match(r'^generated_scen', d.name, re.IGNORECASE)
    ]

    # Sort source folders naturally (e.g., generated_scenarios, generated_scenarios2, generated_scenarios3...)
    source_dirs.sort(key=lambda d: natural_sort_key(d.name))

    if not source_dirs:
        print("No matching 'generated_scenarios' or 'generated_scenerios' folders found.")
        return

    print(f"Found {len(source_dirs)} source folders:")
    for d in source_dirs:
        print(f" - {d.name}")

    total_moved = 0

    # Determine starting index from the first file found (e.g., 0 or 1)
    start_index = None

    for d in source_dirs:
        # Collect XML files (or all files)
        files = [f for f in d.iterdir() if f.is_file() and f.suffix.lower() == ".xml"]
        files.sort(key=lambda f: natural_sort_key(f.name))

        if not files:
            continue

        if start_index is None:
            # Check starting number from first file name if possible
            match = re.search(r'(\d+)\.xml$', files[0].name, re.IGNORECASE)
            start_index = int(match.group(1)) if match else 1

        for f in files:
            # Extract prefix before digits, if any (e.g. "scenario_" from "scenario_0001.xml")
            match = re.match(r'^(.*?)(\d+)(\.[^.]+)$', f.name)
            if match:
                prefix, _, ext = match.groups()
            else:
                prefix, ext = "scenario_", f.suffix

            current_num = start_index + total_moved
            # Format number with 4-digit zero padding (or more if total > 9999)
            new_name = f"{prefix}{current_num:04d}{ext}"
            dest_file = dest_dir / new_name

            # MOVE the file instead of copying it to save space
            shutil.move(f, dest_file)
            total_moved += 1
            
        # Optional Cleanup: Remove the source directory if it's now empty
        try:
            d.rmdir()
            print(f"Cleaned up empty directory: {d.name}")
        except OSError:
            # Directory might not be empty if there were non-XML files inside
            pass

    print(f"\nSuccessfully moved {total_moved} files to '{dest_dir.name}/'.")


if __name__ == "__main__":
    merge_scenarios()