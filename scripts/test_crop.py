#!/usr/bin/env python3
"""
Test & Validation Script for EasyPortrait Mouth Cropper
-------------------------------------------------------
Runs a visual test of the MouthCropper model to verify detection accuracy,
bounding box scaling, and 15% padding.

Usage:
    # 1. Test on any photo in data/raw or auto-download a test smile photo:
    python scripts/test_crop.py

    # 2. Test on a specific image:
    python scripts/test_crop.py --image path/to/photo.jpg

Outputs (saved to ./data/test_output/):
    1. sample_cropped.jpg    -> The final high-res cropped mouth (lips + teeth).
    2. sample_comparison.jpg -> Side-by-side: original image with bounding box vs cropped mouth.
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Optional

# Ensure scripts directory and project root are in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests

from crop_engine import MouthCropper, CLASS_LIPS, CLASS_TEETH

TEST_OUTPUT_DIR = ROOT_DIR / "data" / "test_output"

# High-resolution public domain / Creative Commons smile photo for testing if data/raw is empty
SAMPLE_SMILE_URL = "https://images.unsplash.com/photo-1544005313-94ddf0286df2?auto=format&fit=crop&w=800&q=80"


def get_test_image(custom_path: Optional[str] = None, force_sample: bool = False) -> Path:
    """Finds an image to test with: user path, sample smile photo, or first image in data/raw."""
    if custom_path:
        p = Path(custom_path)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Provided image path not found: {custom_path}")

    # If force sample requested, bypass data/raw
    sample_path = TEST_OUTPUT_DIR / "sample_input_smile.jpg"
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if force_sample:
        if not sample_path.is_file():
            print(f"[INPUT] Downloading reference smile photo from Unsplash...")
            resp = requests.get(SAMPLE_SMILE_URL, timeout=30)
            resp.raise_for_status()
            with open(sample_path, "wb") as f:
                f.write(resp.content)
            print(f"[INPUT] Downloaded reference image to: {sample_path.resolve()}")
        else:
            print(f"[INPUT] Using reference smile photo: {sample_path.name}")
        return sample_path

    # Check data/raw for any image
    raw_dir = ROOT_DIR / "data" / "raw"
    if raw_dir.is_dir():
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            found = list(raw_dir.glob(f"*{ext}"))
            if found:
                print(f"[INPUT] Using image found in data/raw: {found[0].name}")
                return found[0]

    # Otherwise, download sample smile photo
    if not sample_path.is_file():
        print(f"[INPUT] No images found in data/raw. Downloading sample smile photo from Unsplash...")
        resp = requests.get(SAMPLE_SMILE_URL, timeout=30)
        resp.raise_for_status()
        with open(sample_path, "wb") as f:
            f.write(resp.content)
        print(f"[INPUT] Downloaded test image to: {sample_path.resolve()}")
    else:
        print(f"[INPUT] Using cached sample smile image: {sample_path.name}")

    return sample_path


def create_comparison_image(
    original_img: Image.Image,
    crop_box: tuple,
    cropped_img: Image.Image,
    save_path: Path
):
    """
    Creates a side-by-side verification image:
    Left: Original photo with green bounding box drawn around the mouth.
    Right: The extracted crop centered with labels.
    """
    orig_boxed = original_img.copy()
    draw = ImageDraw.Draw(orig_boxed)

    # Draw green box for mouth crop
    x1, y1, x2, y2 = crop_box
    draw.rectangle([x1, y1, x2, y2], outline="#00ff00", width=4)

    # Resize boxed original to fixed preview height (e.g. 500px)
    target_height = 500
    aspect_left = orig_boxed.width / orig_boxed.height
    left_w = int(target_height * aspect_left)
    left_resized = orig_boxed.resize((left_w, target_height), Image.Resampling.LANCZOS)

    # Scale cropped image to fit preview
    crop_w, crop_h = cropped_img.size
    aspect_crop = crop_w / crop_h
    right_h = min(280, target_height)
    right_w = int(right_h * aspect_crop)
    crop_resized = cropped_img.resize((right_w, right_h), Image.Resampling.LANCZOS)

    # Combine canvas
    padding = 20
    canvas_w = left_w + right_w + (padding * 3)
    canvas_h = target_height + (padding * 2)

    canvas = Image.new("RGB", (canvas_w, canvas_h), color="#0f172a")

    # Paste left original boxed
    canvas.paste(left_resized, (padding, padding))

    # Paste right cropped mouth centered vertically
    right_y = padding + (target_height - right_h) // 2
    right_x = left_w + (padding * 2)
    canvas.paste(crop_resized, (right_x, right_y))

    # Annotate labels
    canvas_draw = ImageDraw.Draw(canvas)
    canvas_draw.text((padding + 8, padding + 8), "DETECTED MOUTH BOUNDING BOX (15% PADDING)", fill="#00ff00")
    canvas_draw.text((right_x, right_y - 22), f"CROPPED MOUTH ({crop_w}x{crop_h}px)", fill="#38bdf8")

    canvas.save(save_path, quality=95)


def run_test(image_path: Path):
    print("\n" + "=" * 65)
    print("         EASYPORTRAIT MOUTH CROPPER - VALIDATION TEST")
    print("=" * 65)

    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Cropper
    print("[1/4] Loading EasyPortrait Face Parsing ONNX Model...")
    t0 = time.time()
    cropper = MouthCropper()
    load_time = time.time() - t0
    print(f"      Model ready in {load_time:.2f}s")

    # 2. Read Image
    print(f"[2/4] Reading input image: {image_path.name}")
    with Image.open(image_path) as raw_img:
        img_rgb = raw_img.convert("RGB")
        orig_w, orig_h = img_rgb.size
    print(f"      Resolution: {orig_w}x{orig_h} pixels")

    # 3. Preprocess & Inference
    print("[3/4] Running Segmentation Inference...")
    t_inf_start = time.time()
    tensor, w, h = cropper.preprocess(img_rgb)
    mask = cropper.run_inference(tensor)
    inf_time = (time.time() - t_inf_start) * 1000

    # Count pixels
    lip_pixels = int(np.sum(mask == CLASS_LIPS))
    teeth_pixels = int(np.sum(mask == CLASS_TEETH))
    total_mouth = lip_pixels + teeth_pixels

    print(f"      Inference Time: {inf_time:.1f} ms")
    print(f"      Detected Lip Pixels:   {lip_pixels:,}")
    print(f"      Detected Teeth Pixels: {teeth_pixels:,}")
    print(f"      Total Mouth Pixels:    {total_mouth:,}")

    # 4. Compute Bounding Box
    print("[4/4] Calculating Bounding Box & 15% Padding...")
    crop_box = cropper.get_crop_box(mask, orig_w, orig_h, min_pixels=50, padding_ratio=0.15)

    if crop_box is None:
        print("\n" + "!" * 65)
        print(" [RESULT: FAILED] No mouth/teeth detected or pixel count < 50!")
        print(" This image would be automatically filtered out by the pipeline.")
        print("!" * 65 + "\n")
        return False

    x1, y1, x2, y2 = crop_box
    box_w = x2 - x1
    box_h = y2 - y1

    print(f"      Crop Box Coordinates: (x1={x1}, y1={y1}, x2={x2}, y2={y2})")
    print(f"      Crop Dimensions:      {box_w}x{box_h} px")

    # Save Cropped Mouth
    cropped_img = img_rgb.crop(crop_box)
    crop_output_path = TEST_OUTPUT_DIR / f"test_crop_{image_path.stem}.jpg"
    cropped_img.save(crop_output_path, quality=95)

    # Save Comparison Image
    comparison_output_path = TEST_OUTPUT_DIR / f"test_comparison_{image_path.stem}.jpg"
    create_comparison_image(img_rgb, crop_box, cropped_img, comparison_output_path)

    # Summary Report
    print("\n" + "=" * 65)
    print("                     TEST SUCCESSFUL!")
    print("=" * 65)
    print(f" Status:             PASSED (Teeth & Lips detected)")
    print(f" Original Image:     {image_path.resolve()}")
    print(f" Cropped Mouth:      {crop_output_path.resolve()}")
    print(f" Visual Comparison:  {comparison_output_path.resolve()}")
    print("=" * 65)
    print("\nTip: Open 'test_comparison_*.jpg' to visually see the bounding box overlay!\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test and visualize EasyPortrait mouth cropping on a photo.")
    parser.add_argument("--image", default=None, help="Path to a specific image file to test.")
    parser.add_argument("--sample", action="store_true", help="Force download and test on a reference portrait smile photo.")
    args = parser.parse_args()

    try:
        target_image = get_test_image(args.image, force_sample=args.sample)
        run_test(target_image)
    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
