#!/usr/bin/env python3
"""
CLI wrapper for MouthCropper matching images_processing_model_prompt.txt:
Usage: python scripts/crop_mouths.py --input_folder PATH --output_folder PATH
"""
import sys
from pathlib import Path

# Ensure scripts directory and project root are in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from crop_engine import main

if __name__ == "__main__":
    main()
