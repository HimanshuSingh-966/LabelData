#!/usr/bin/env python3
"""
EasyPortrait Face-Parsing Mouth & Smile Cropper
-----------------------------------------------
Pretrained face-parsing inference using SegFormer-B0 ONNX export from
huggingface.co/sadzip/EasyPortrait-ONNX.

Extracts Class 6 (LIPS) and Class 7 (TEETH), adds 15% padding,
and crops the original high-resolution photo.

Can be run standalone:
    python crop_engine.py --input_folder PATH --output_folder PATH

Or imported into shuffle_dataset.py and backend:
    from crop_engine import MouthCropper
    cropper = MouthCropper()
    result = cropper.crop_image(image_path, output_path)
"""

import os
import csv
import logging
import argparse
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
from PIL import Image

# ONNX Runtime
try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("crop-engine")

# EasyPortrait Model Configuration (SegFormer-B0 Face Parsing from Hugging Face)
DEFAULT_MODEL_FILENAME = "segformer_b0_fp_512.onnx"
DEFAULT_MODEL_DIR = Path("./models")
HF_BASE_URL = "https://huggingface.co/sadzip/EasyPortrait-ONNX/resolve/main"

DEFAULT_INPUT_SIZE = 512

# Class definitions for EasyPortrait face parsing (official 9-class mapping)
# 0=background, 1=skin, 2=left_brow, 3=right_brow, 4=left_eye, 5=right_eye, 6=lips, 7=teeth
CLASS_LIPS = 6
CLASS_TEETH = 7

# EasyPortrait BGR normalization parameters
NORM_MEAN = np.array([143.55267075, 132.96705975, 126.94924335], dtype=np.float32)
NORM_STD = np.array([60.2625333, 60.32740275, 59.30988645], dtype=np.float32)

DEFAULT_MIN_PIXELS = 50
DEFAULT_PADDING_RATIO = 0.15


class MouthCropper:
    """
    Inference wrapper for EasyPortrait face parsing to detect and crop lips and teeth.
    """

    def __init__(self, model_path: Optional[str] = None):
        if not ORT_AVAILABLE:
            raise ImportError(
                "onnxruntime is not installed. Please install it with: pip install onnxruntime"
            )

        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_DIR / DEFAULT_MODEL_FILENAME
        self._ensure_model_exists()

        # Initialize ONNX Runtime session
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"])

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Determine model input resolution dynamically from ONNX graph
        inp_shape = self.session.get_inputs()[0].shape
        if len(inp_shape) >= 4 and isinstance(inp_shape[2], int) and isinstance(inp_shape[3], int) and inp_shape[2] > 0 and inp_shape[3] > 0:
            self.input_h = inp_shape[2]
            self.input_w = inp_shape[3]
        else:
            self.input_h = DEFAULT_INPUT_SIZE
            self.input_w = DEFAULT_INPUT_SIZE

        # Print notice regarding weights
        model_name = self.model_path.stem
        print("\n" + "-" * 70)
        print(f"[MODEL]  Architecture: {model_name} (EasyPortrait Face Parsing)")
        print(f"[INPUT]  Resolution:   {self.input_w}x{self.input_h}")
        print(f"[PATH]   {self.model_path.resolve()}")
        print("-" * 70 + "\n")

    def _ensure_model_exists(self):
        """Downloads the ONNX model from Hugging Face if not found locally."""
        if not self.model_path.is_file():
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            download_url = f"{HF_BASE_URL}/{self.model_path.name}"
            logger.info(f"Downloading {self.model_path.name} weights from {download_url}...")
            try:
                import requests
                response = requests.get(download_url, stream=True, timeout=120)
                response.raise_for_status()
                with open(self.model_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                logger.info(f"Model saved to {self.model_path.resolve()}")
            except Exception as e:
                logger.error(f"Failed to download model weights: {e}")
                raise RuntimeError(
                    f"Could not download model from {download_url}. "
                    f"Please download '{self.model_path.name}' manually and place it in '{self.model_path.parent}'."
                )

    def preprocess(self, pil_image: Image.Image) -> Tuple[np.ndarray, int, int]:
        """
        Prepares PIL image for EasyPortrait ONNX model.
        Returns: (input_tensor, original_width, original_height)
        """
        orig_w, orig_h = pil_image.size

        # Resize to expected model resolution
        resized = pil_image.resize((self.input_w, self.input_h), Image.Resampling.BILINEAR)

        # Ensure RGB
        if resized.mode != "RGB":
            resized = resized.convert("RGB")

        # Convert to numpy array (H, W, 3) in RGB
        img_np = np.array(resized, dtype=np.float32)

        # Reverse RGB to BGR as specified by EasyPortrait manifest
        img_bgr = img_np[:, :, ::-1]

        # Normalize with dataset mean and std
        normalized = (img_bgr - NORM_MEAN) / NORM_STD

        # Transpose to (1, 3, H, W)
        tensor = np.transpose(normalized, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0).astype(np.float32)

        return tensor, orig_w, orig_h

    def run_inference(self, tensor: np.ndarray) -> np.ndarray:
        """
        Runs model inference and returns the 2D argmax segmentation mask (H, W).
        """
        outputs = self.session.run([self.output_name], {self.input_name: tensor})
        logits = outputs[0]  # Shape: (1, classes, H, W)
        mask = np.argmax(logits, axis=1)[0]  # Shape: (H, W)
        return mask

    def get_crop_box(
        self,
        mask: np.ndarray,
        orig_w: int,
        orig_h: int,
        min_pixels: int = DEFAULT_MIN_PIXELS,
        padding_ratio: float = DEFAULT_PADDING_RATIO
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Finds pixels labeled LIPS (6) or TEETH (7), computes bounding box in original
        resolution coordinates, and applies padding.
        Returns: (x1, y1, x2, y2) or None if detection fails.
        """
        # Find pixels matching Lips or Teeth
        mouth_pixels = (mask == CLASS_LIPS) | (mask == CLASS_TEETH)
        pixel_count = int(np.sum(mouth_pixels))

        if pixel_count < min_pixels:
            return None

        # Bounding box in mask coordinate space
        ys, xs = np.where(mouth_pixels)
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())

        # Scale coordinates to original full-resolution image space
        scale_x = orig_w / float(self.input_w)
        scale_y = orig_h / float(self.input_h)

        orig_x1 = x_min * scale_x
        orig_y1 = y_min * scale_y
        orig_x2 = x_max * scale_x
        orig_y2 = y_max * scale_y

        box_w = orig_x2 - orig_x1
        box_h = orig_y2 - orig_y1

        # Add 15% padding on each side
        pad_x = box_w * padding_ratio
        pad_y = box_h * padding_ratio

        final_x1 = max(0, int(round(orig_x1 - pad_x)))
        final_y1 = max(0, int(round(orig_y1 - pad_y)))
        final_x2 = min(orig_w, int(round(orig_x2 + pad_x)))
        final_y2 = min(orig_h, int(round(orig_y2 + pad_y)))

        # Sanity check
        if final_x2 <= final_x1 or final_y2 <= final_y1:
            return None

        return (final_x1, final_y1, final_x2, final_y2)

    def crop_image(
        self,
        image_path: Path,
        output_path: Optional[Path] = None,
        min_pixels: int = DEFAULT_MIN_PIXELS,
        padding_ratio: float = DEFAULT_PADDING_RATIO
    ) -> Dict[str, Any]:
        """
        Processes a single image file: loads, infers, crops to mouth, and saves.
        Returns dict with keys: 'status' ('success' or 'failed'), 'crop_box', 'error'.
        """
        try:
            with Image.open(image_path) as img:
                # Load full image into memory in RGB
                img_rgb = img.convert("RGB")
                orig_w, orig_h = img_rgb.size

                # Preprocess & Infer
                tensor, w, h = self.preprocess(img_rgb)
                mask = self.run_inference(tensor)

                # Get Bounding Box with Padding
                crop_box = self.get_crop_box(mask, orig_w, orig_h, min_pixels, padding_ratio)

                if crop_box is None:
                    return {
                        "filename": image_path.name,
                        "status": "failed",
                        "crop_box": None,
                        "error": "No teeth/lips detected or below threshold."
                    }

                # Crop original high-resolution image
                cropped_img = img_rgb.crop(crop_box)

                if output_path:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    cropped_img.save(output_path, quality=95)

                return {
                    "filename": image_path.name,
                    "status": "success",
                    "crop_box": crop_box,
                    "error": None
                }

        except Exception as e:
            return {
                "filename": image_path.name,
                "status": "failed",
                "crop_box": None,
                "error": str(e)
            }


def process_folder(
    input_folder: str,
    output_folder: str,
    log_csv: Optional[str] = None,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    padding: float = DEFAULT_PADDING_RATIO
):
    """Batch processes all images in input_folder and saves crops to output_folder."""
    in_dir = Path(input_folder)
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    cropper = MouthCropper()

    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_files = [p for p in in_dir.glob("*") if p.is_file() and p.suffix.lower() in image_extensions]

    total = len(image_files)
    logger.info(f"Processing {total} images from {in_dir.resolve()}...")

    results = []
    succeeded = 0
    failed = 0

    for idx, img_path in enumerate(image_files, start=1):
        target_path = out_dir / img_path.name
        res = cropper.crop_image(img_path, target_path, min_pixels=min_pixels, padding_ratio=padding)
        results.append(res)

        if res["status"] == "success":
            succeeded += 1
        else:
            failed += 1

        if idx % 50 == 0 or idx == total:
            logger.info(f"Progress: [{idx}/{total}] (Success: {succeeded}, Failed: {failed})")

    # Save log CSV
    if not log_csv:
        log_csv = str(out_dir / "crop_log.csv")

    log_path = Path(log_csv)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "status", "crop_box", "error"])
        for r in results:
            box_str = f"({r['crop_box'][0]},{r['crop_box'][1]},{r['crop_box'][2]},{r['crop_box'][3]})" if r["crop_box"] else ""
            writer.writerow([r["filename"], r["status"], box_str, r.get("error", "")])

    success_rate = (succeeded / total * 100) if total > 0 else 0
    print("\n" + "=" * 60)
    print("              MOUTH CROP BATCH SUMMARY")
    print("=" * 60)
    print(f" Total Processed:    {total:,}")
    print(f" Successfully Cropped: {succeeded:,} ({success_rate:.1f}%)")
    print(f" Failed / Excluded:  {failed:,} ({100 - success_rate:.1f}%)")
    print(f" Cropped Output Dir: {out_dir.resolve()}")
    print(f" Log CSV:            {log_path.resolve()}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Crop smile photos to lips and teeth using EasyPortrait ONNX model.")
    parser.add_argument("--input_folder", required=True, help="Folder containing raw smile photos.")
    parser.add_argument("--output_folder", required=True, help="Folder to save cropped mouth photos.")
    parser.add_argument("--log_csv", default=None, help="Path for crop log CSV.")
    parser.add_argument("--min_pixels", type=int, default=DEFAULT_MIN_PIXELS, help="Minimum teeth/lips pixel count.")
    parser.add_argument("--padding", type=float, default=DEFAULT_PADDING_RATIO, help="Padding ratio around mouth.")

    args = parser.parse_args()
    process_folder(args.input_folder, args.output_folder, args.log_csv, args.min_pixels, args.padding)


if __name__ == "__main__":
    main()
