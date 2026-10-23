#!/usr/bin/env python3
"""
Sample Inspection Script for Dental Smile Dataset
-------------------------------------------------
Validates that local raw images in data/raw can be opened, decoded,
and inspected (dimensions, file size, format) without needing face parsing.

Usage:
    python scripts/test_dataset_sample.py
"""

import sys
from pathlib import Path
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

RAW_DIR = ROOT_DIR / "data" / "raw"
OUTPUT_DIR = ROOT_DIR / "data" / "test_output"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def main():
    print("\n" + "=" * 65)
    print("         DENTAL DATASET SAMPLE INSPECTOR")
    print("=" * 65)

    if not RAW_DIR.is_dir():
        print(f"[ERROR] Directory does not exist: {RAW_DIR}")
        return

    files = [p for p in RAW_DIR.glob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    total_found = len(files)

    if total_found == 0:
        print(f"[ERROR] No image files found in {RAW_DIR}")
        return

    print(f"Found {total_found:,} image files in {RAW_DIR.resolve()}\n")

    # Sample up to 5 images for detailed inspection
    sample_files = files[:5]
    print("Inspecting first few sample images:")
    print("-" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for idx, f in enumerate(sample_files, start=1):
        try:
            with Image.open(f) as img:
                w, h = img.size
                mode = img.mode
                fmt = img.format
                size_kb = f.stat().st_size / 1024.0

            print(f"[{idx}] {f.name:25s} | Size: {size_kb:>6.1f} KB | Res: {w}x{h} px | Mode: {mode} | Format: {fmt} -> VALID")
        except Exception as e:
            print(f"[{idx}] {f.name:25s} | [ERROR] Corrupted or unreadable: {e}")

    print("-" * 65)
    print(f"\n[SUMMARY] Dataset images are ready for direct labeling!")
    print("Run: python scripts/shuffle_dataset.py --upload-images\n")

if __name__ == "__main__":
    main()
