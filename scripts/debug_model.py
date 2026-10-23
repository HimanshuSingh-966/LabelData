#!/usr/bin/env python3
"""Debug script: dumps the raw model output to see what classes are detected."""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
from PIL import Image
from crop_engine import MouthCropper, MODEL_INPUT_SIZE

image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT_DIR / "data" / "raw" / "1.jpg"
print(f"\n[DEBUG] Image: {image_path}")

cropper = MouthCropper()

# Load and preprocess
img = Image.open(image_path).convert("RGB")
print(f"[DEBUG] Image size: {img.size}")

tensor, w, h = cropper.preprocess(img)
print(f"[DEBUG] Tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
print(f"[DEBUG] Tensor min={tensor.min():.3f}, max={tensor.max():.3f}, mean={tensor.mean():.3f}")

# Raw model output
outputs = cropper.session.run([cropper.output_name], {cropper.input_name: tensor})
logits = outputs[0]
print(f"\n[DEBUG] Raw logits shape: {logits.shape}")
print(f"[DEBUG] Logits min={logits.min():.3f}, max={logits.max():.3f}")

# Argmax mask
mask = np.argmax(logits, axis=1)[0]
print(f"[DEBUG] Mask shape: {mask.shape}")

# Count ALL classes
unique, counts = np.unique(mask, return_counts=True)
print(f"\n[DEBUG] === ALL DETECTED CLASSES ===")
CLASS_NAMES = {
    0: "background", 1: "skin", 2: "left_brow", 3: "right_brow",
    4: "left_eye", 5: "right_eye", 6: "lips", 7: "teeth", 8: "person/other"
}
for cls_id, cnt in zip(unique, counts):
    pct = cnt / mask.size * 100
    name = CLASS_NAMES.get(int(cls_id), f"unknown_{cls_id}")
    print(f"  Class {cls_id}: {name:15s} → {cnt:>7,} pixels ({pct:.1f}%)")

# Also check softmax confidence for lip/teeth channels
print(f"\n[DEBUG] === LOGIT VALUES FOR LIP/TEETH CHANNELS ===")
for ch in range(logits.shape[1]):
    ch_vals = logits[0, ch]
    name = CLASS_NAMES.get(ch, f"channel_{ch}")
    print(f"  Channel {ch} ({name:15s}): min={ch_vals.min():.3f}, max={ch_vals.max():.3f}, mean={ch_vals.mean():.3f}")
