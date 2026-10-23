#!/usr/bin/env python3
"""
Dataset Shuffling, Mouth-Cropping & Staging Pipeline
----------------------------------------------------
Prepares the target labeling image manifest for the SmileLabel application.

Integrated Workflow:
1. Candidate Pull:
   - Reads raw master dataset from local folder (LOCAL_RAW_DATASET_DIR).
   - Excludes previously pulled images using local history (LOCAL_PULL_HISTORY_PATH).
   - Randomly shuffles candidates using a fixed reproducible seed (SHUFFLE_SEED=42).
2. Auto-Filter & Mouth Crop (EasyPortrait SegFormer-B0 ONNX):
   - Crops each photo to lips + teeth with 15% margin padding.
   - Images without visible teeth (< 50 pixels) are marked FAILED and discarded.
   - Continues until TARGET_SAMPLE_SIZE (~2,000–2,400) successfully cropped smile photos are gathered.
   - Saves cropped smile photos to local staging directory (LOCAL_PULLED_BATCH_DIR).
   - Writes detailed crop results to CROP_LOG_PATH (crop_log.csv).
3. Cloud Synchronization:
   - Updates local pull history so raw photos are never re-pulled.
   - Generates batch manifest (B2_IMAGE_LIST_PATH).
   - If --upload-images: uploads only the successfully cropped mouth photos to B2.
   - If --upload or --upload-images: uploads the manifest to B2.
"""

import os
import io
import sys
import csv
import json
import random
import shutil
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Set, Optional, Dict, Any, Tuple
from PIL import Image

# Ensure script directory and project root are in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Load environment configuration from .env
try:
    from dotenv import load_dotenv
    env_paths = [ROOT_DIR / ".env", Path(".env"), Path("../.env"), ROOT_DIR / "backend" / ".env"]
    for p in env_paths:
        if p.exists():
            load_dotenv(dotenv_path=p)
            break
except ImportError:
    pass

# Try importing B2 SDK for direct cloud operations
try:
    from b2sdk.v2 import B2Api, InMemoryAccountInfo
    from b2sdk.v2.exception import FileNotPresent
    try:
        from b2sdk.download_dest import DownloadDestBytes
    except ImportError:
        try:
            from b2sdk.v2 import DownloadDestBytes
        except ImportError:
            from b2sdk.v1 import DownloadDestBytes
    B2_AVAILABLE = True
except ImportError:
    B2_AVAILABLE = False

# Try importing MouthCropper engine
try:
    from crop_engine import MouthCropper, DEFAULT_MIN_PIXELS, DEFAULT_PADDING_RATIO
    CROPPER_AVAILABLE = True
except ImportError:
    CROPPER_AVAILABLE = False
    MouthCropper = None
    DEFAULT_MIN_PIXELS = 50
    DEFAULT_PADDING_RATIO = 0.15

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("shuffle-dataset")

# Environment defaults (no hardcoded static paths)
DEFAULT_RAW_DIR = os.getenv("LOCAL_RAW_DATASET_DIR", "./data/raw")
DEFAULT_BATCH_DIR = os.getenv("LOCAL_PULLED_BATCH_DIR", "./data/pulled_batch")
DEFAULT_HISTORY_PATH = os.getenv("LOCAL_PULL_HISTORY_PATH", "./data/pulled_history.csv")
DEFAULT_CROP_LOG_PATH = os.getenv("CROP_LOG_PATH", "./data/crop_log.csv")
DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "./models/segformer_b0_fp_512.onnx")

DEFAULT_IMAGE_LIST_PATH = os.getenv("B2_IMAGE_LIST_PATH") or os.getenv("IMAGE_LIST_PATH") or "image_list.json"
DEFAULT_LABELS_CSV_PATH = os.getenv("B2_LABELS_CSV_PATH") or os.getenv("LABELS_CSV_PATH") or "labels.csv"

DEFAULT_PULL_LIMIT = int(os.getenv("PULL_LIMIT", "3000"))
DEFAULT_TARGET_SAMPLE = int(os.getenv("TARGET_SAMPLE_SIZE", "2200"))
DEFAULT_SHUFFLE_SEED = int(os.getenv("SHUFFLE_SEED", "42"))

B2_KEY_ID = os.getenv("B2_KEY_ID", "").strip()
B2_APP_KEY = os.getenv("B2_APP_KEY", "").strip()
B2_BUCKET_NAME = os.getenv("B2_BUCKET_NAME", "").strip()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TARGET_MIN = 1500
TARGET_MAX = 2500


def get_b2_bucket():
    """Initializes and returns B2 bucket client if credentials exist."""
    if not B2_AVAILABLE:
        logger.warning("b2sdk not installed. Direct B2 operations unavailable.")
        return None
    if not (B2_KEY_ID and B2_APP_KEY and B2_BUCKET_NAME):
        logger.warning("B2 credentials (B2_KEY_ID, B2_APP_KEY, B2_BUCKET_NAME) are incomplete in .env.")
        return None
    try:
        info = InMemoryAccountInfo()
        b2_api = B2Api(info)
        b2_api.authorize_account("production", B2_KEY_ID, B2_APP_KEY)
        bucket = b2_api.get_bucket_by_name(B2_BUCKET_NAME)
        return bucket
    except Exception as e:
        logger.error(f"Failed to authenticate with Backblaze B2: {e}")
        return None


def load_local_pull_history(history_csv_path: str) -> Set[str]:
    """Reads previously pulled image filenames from local history CSV to avoid pulling them again."""
    p = Path(history_csv_path)
    if not p.is_file():
        return set()
    pulled = set()
    try:
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fn = row.get("filename", "").strip()
                if fn:
                    pulled.add(fn)
        logger.info(f"Loaded {len(pulled)} previously pulled images from local history ({p})")
    except Exception as e:
        logger.warning(f"Could not read local pull history {p}: {e}")
    return pulled


def append_to_local_pull_history(history_csv_path: str, new_filenames: List[str]):
    """Logs newly pulled image filenames into the local history CSV."""
    p = Path(history_csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    file_exists = p.is_file()
    now_str = datetime.now(timezone.utc).isoformat()
    try:
        with open(p, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["filename", "pulled_at"])
            for fn in new_filenames:
                writer.writerow([fn, now_str])
        logger.info(f"Recorded {len(new_filenames)} newly pulled images to local history: {p}")
    except Exception as e:
        logger.error(f"Failed to update local pull history {p}: {e}")


def collect_images_from_local(directory_path: str, pull_limit: int) -> List[Path]:
    """Collects image file paths from a local directory up to pull limit."""
    p = Path(directory_path)
    if not p.is_dir():
        logger.warning(f"Local directory '{directory_path}' does not exist.")
        return []

    collected = []
    for file_path in p.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            collected.append(file_path)
            if len(collected) >= pull_limit:
                break
    return collected


def process_candidates_with_crop(
    candidates: List[Path],
    batch_dir: Path,
    target_count: int,
    cropper: Any,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    padding: float = DEFAULT_PADDING_RATIO
) -> Tuple[List[str], List[Dict[str, Any]], int, int]:
    """
    Runs face-parsing mouth detection & crop on candidate photos.
    Only successful crops (teeth/lips detected) are kept and saved to batch_dir.
    Stops once target_count is reached or candidates exhausted.
    """
    batch_dir.mkdir(parents=True, exist_ok=True)
    succeeded_files: List[str] = []
    crop_logs: List[Dict[str, Any]] = []

    succeeded = 0
    failed = 0
    total_candidates = len(candidates)

    logger.info(f"Starting EasyPortrait mouth crop auto-filter on up to {total_candidates} candidates...")
    logger.info(f"Target sample size: {target_count} successfully cropped smile photos.")

    for idx, img_path in enumerate(candidates, start=1):
        target_crop_path = batch_dir / img_path.name
        res = cropper.crop_image(img_path, target_crop_path, min_pixels=min_pixels, padding_ratio=padding)
        crop_logs.append(res)

        if res["status"] == "success":
            succeeded += 1
            succeeded_files.append(img_path.name)
        else:
            failed += 1

        if idx % 50 == 0 or idx == total_candidates or len(succeeded_files) == target_count:
            logger.info(f"Progress: Processed {idx}/{total_candidates} | Cropped: {succeeded}/{target_count} | Discarded: {failed}")

        # Stop when we reached our target sample size
        if len(succeeded_files) >= target_count:
            logger.info(f"Reached target sample count of {target_count} successfully cropped photos!")
            break

    return succeeded_files, crop_logs, succeeded, failed


def save_crop_log(log_path: str, logs: List[Dict[str, Any]]):
    """Saves crop execution details to CSV."""
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "status", "crop_box", "error"])
        for r in logs:
            box_str = f"({r['crop_box'][0]},{r['crop_box'][1]},{r['crop_box'][2]},{r['crop_box'][3]})" if r.get("crop_box") else ""
            writer.writerow([r["filename"], r["status"], box_str, r.get("error", "")])
    logger.info(f"Saved crop log to {p.resolve()}")


def validate_and_stage_images(
    candidates: List[Path],
    batch_dir: Path,
    target_count: int
) -> Tuple[List[str], int, int]:
    """
    Validates candidate dental images (checks readability, format, non-zero dimensions)
    and stages them into batch_dir.
    Stops once target_count valid images are gathered.
    """
    batch_dir.mkdir(parents=True, exist_ok=True)
    staged_filenames: List[str] = []
    valid_count = 0
    corrupt_count = 0

    logger.info(f"Validating and staging candidates (Target: {target_count})...")

    for idx, img_path in enumerate(candidates, start=1):
        # Quick validation: file must have content and be readable by PIL
        try:
            if img_path.stat().st_size < 100:  # Less than 100 bytes is invalid/empty
                corrupt_count += 1
                continue

            with Image.open(img_path) as img:
                w, h = img.size
                if w < 50 or h < 30:  # Too small to be a usable dental image
                    corrupt_count += 1
                    continue
                # Verify image can be decoded
                img.verify()

            # Copy to batch_dir
            dst = batch_dir / img_path.name
            if not dst.exists():
                shutil.copy2(img_path, dst)

            staged_filenames.append(img_path.name)
            valid_count += 1

            if valid_count % 200 == 0 or valid_count == target_count:
                logger.info(f"Progress: [{valid_count}/{target_count}] valid images staged...")

            if valid_count >= target_count:
                logger.info(f"Successfully reached target count of {target_count} images!")
                break

        except Exception as e:
            logger.warning(f"Skipping corrupted/unreadable image {img_path.name}: {e}")
            corrupt_count += 1

    return staged_filenames, valid_count, corrupt_count


def main():
    parser = argparse.ArgumentParser(description="Shuffle, mouth-crop, and stage dataset for image labeling.")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR,
                        help=f"Path to local raw dataset folder (default: {DEFAULT_RAW_DIR}).")
    parser.add_argument("--batch-dir", default=DEFAULT_BATCH_DIR,
                        help=f"Path to local pulled batch staging folder (default: {DEFAULT_BATCH_DIR}).")
    parser.add_argument("--history-csv", default=DEFAULT_HISTORY_PATH,
                        help=f"Path to local pull history CSV (default: {DEFAULT_HISTORY_PATH}).")
    parser.add_argument("--crop-log", default=DEFAULT_CROP_LOG_PATH,
                        help=f"Path to crop log CSV (default: {DEFAULT_CROP_LOG_PATH}).")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH,
                        help=f"Path to SegFormer-B0 ONNX model (default: {DEFAULT_MODEL_PATH}).")
    parser.add_argument("--output", default=DEFAULT_IMAGE_LIST_PATH,
                        help=f"Output manifest JSON path (default: {DEFAULT_IMAGE_LIST_PATH}).")
    parser.add_argument("--pull-limit", type=int, default=DEFAULT_PULL_LIMIT,
                        help=f"Max raw images to inspect (default: {DEFAULT_PULL_LIMIT}).")
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET_SAMPLE,
                        help=f"Target sample size of cropped smile photos (default: {DEFAULT_TARGET_SAMPLE}, expect ~2,000–2,400).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SHUFFLE_SEED,
                        help=f"Random seed for reproducible shuffling (default: {DEFAULT_SHUFFLE_SEED}).")
    parser.add_argument("--crop", action="store_true",
                        help="Enable EasyPortrait mouth cropping (only needed if raw photos are uncropped full-face portraits).")
    parser.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS,
                        help=f"Minimum teeth/lips pixel count when --crop is enabled (default: {DEFAULT_MIN_PIXELS}).")
    parser.add_argument("--padding", type=float, default=DEFAULT_PADDING_RATIO,
                        help=f"Padding ratio around mouth crop box when --crop is enabled (default: {DEFAULT_PADDING_RATIO}).")
    parser.add_argument("--upload", action="store_true",
                        help="Upload output manifest JSON directly to Backblaze B2 bucket.")
    parser.add_argument("--upload-images", action="store_true",
                        help="Upload the actual staged dental image files to Backblaze B2.")

    args = parser.parse_args()

    bucket = get_b2_bucket()

    # Step 1: Load local pull history to prevent duplicate pulls from PC
    local_history = load_local_pull_history(args.history_csv)

    # Step 2: Collect candidate images from raw folder
    raw_path = Path(args.raw_dir)
    images = collect_images_from_local(args.raw_dir, args.pull_limit)

    if not images:
        logger.error(f"No images found in '{args.raw_dir}'.")
        logger.info(f"Please place your downloaded raw images into: {raw_path.resolve()}")
        sys.exit(1)

    logger.info(f"Discovered {len(images)} raw images in {raw_path.resolve()}.")

    # Step 3: Deduplicate against history
    candidates = [img for img in images if img.name not in local_history]
    logger.info(f"After excluding {len(images) - len(candidates)} previously pulled images: {len(candidates)} eligible candidates.")

    if not candidates:
        logger.warning("All candidate images have already been pulled according to history log!")
        sys.exit(0)

    # Step 4: Deterministic random shuffle
    rng = random.Random(args.seed)
    rng.shuffle(candidates)

    batch_dir = Path(args.batch_dir)
    do_crop = args.crop

    # Step 5: Process candidates
    final_list: List[str] = []
    crop_logs: List[Dict[str, Any]] = []
    succeeded_count = 0
    failed_count = 0

    if do_crop:
        logger.info("Face-parsing mouth cropping enabled (--crop)...")
        if not CROPPER_AVAILABLE:
            logger.error("crop_engine module or onnxruntime not found. Please install: pip install onnxruntime Pillow numpy")
            sys.exit(1)

        try:
            cropper = MouthCropper(model_path=args.model_path)
        except Exception as e:
            logger.error(f"Failed to initialize MouthCropper: {e}")
            sys.exit(1)

        final_list, crop_logs, succeeded_count, failed_count = process_candidates_with_crop(
            candidates=candidates,
            batch_dir=batch_dir,
            target_count=args.target,
            cropper=cropper,
            min_pixels=args.min_pixels,
            padding=args.padding
        )

        # Save crop log
        save_crop_log(args.crop_log, crop_logs)
    else:
        logger.info("Validating and staging dental smile photos directly (images are already cropped)...")
        final_list, succeeded_count, failed_count = validate_and_stage_images(
            candidates=candidates,
            batch_dir=batch_dir,
            target_count=args.target
        )

    # Step 6: Update local pull history CSV (records newly pulled & staged photos)
    append_to_local_pull_history(args.history_csv, final_list)

    # Step 7: Save manifest JSON locally
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_list, f, indent=2)

    logger.info(f"Saved {len(final_list)} images to manifest: {output_path.resolve()}")

    # Step 8: Upload images to B2 if requested
    uploaded_images_count = 0
    if args.upload_images and bucket:
        total_to_upload = len(final_list)
        logger.info(f"Uploading {total_to_upload} dental smile photos to B2 bucket '{B2_BUCKET_NAME}'...")

        for idx, filename in enumerate(final_list, start=1):
            local_file = batch_dir / filename
            if not local_file.is_file():
                logger.warning(f"File not found for upload: {filename}")
                continue

            content_type = "image/jpeg" if local_file.suffix.lower() in [".jpg", ".jpeg"] else "image/png"
            try:
                bucket.upload_local_file(
                    local_file=str(local_file),
                    file_name=filename,
                    content_type=content_type
                )
                uploaded_images_count += 1
                if idx % 100 == 0 or idx == total_to_upload:
                    logger.info(f"Progress: [{idx}/{total_to_upload}] photos uploaded to B2...")
            except Exception as e:
                logger.error(f"Failed to upload {filename} to B2: {e}")

    # Step 9: Upload manifest JSON to B2 if requested
    uploaded_to_b2 = False
    if (args.upload or args.upload_images) and bucket:
        try:
            logger.info(f"Uploading manifest {args.output} to Backblaze B2 '{B2_BUCKET_NAME}'...")
            with open(output_path, "rb") as f:
                data_bytes = f.read()
            bucket.upload_bytes(data_bytes, file_name=args.output, content_type="application/json")
            uploaded_to_b2 = True
            logger.info("Manifest successfully uploaded to B2.")
        except Exception as e:
            logger.error(f"Failed to upload manifest to B2: {e}")

    # Step 10: Final Report
    in_range = TARGET_MIN <= len(final_list) <= TARGET_MAX
    b2_status = []
    if uploaded_to_b2:
        b2_status.append(f"Manifest ({args.output})")
    if uploaded_images_count > 0:
        b2_status.append(f"{uploaded_images_count:,} Photos")
    b2_str = " & ".join(b2_status) + f" uploaded to {B2_BUCKET_NAME}" if b2_status else "Local only (use --upload-images to sync to B2)"

    print("\n" + "=" * 64)
    print("           DATASET PULL & STAGING SUMMARY REPORT")
    print("=" * 64)
    print(f" Candidates Examined:     {succeeded_count + failed_count:,}")
    print(f" Valid Images Staged:     {succeeded_count:,}")
    if failed_count > 0:
        print(f" Corrupted/Too Small:     {failed_count:,} (skipped)")
    print(f" Prev. Pulled History:    {len(local_history):,} (excluded)")
    print(f" Target Sample Set:       {args.target:,} (Expected ~2,000–2,400)")
    print(f" Final Batch Count:       {len(final_list):,}")
    print(f" Target Window Check:     {len(final_list):,} images -> {'[PASS] Inside 1,500–2,500 target' if in_range else '[NOTICE] Outside standard 1,500–2,500 window'}")
    print(f" Staged Batch Directory:  {batch_dir.resolve()}")
    if do_crop:
        print(f" Crop Log CSV:            {Path(args.crop_log).resolve()}")
    print(f" Output Manifest JSON:    {output_path.resolve()}")
    print(f" B2 Cloud Upload:         {b2_str}")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
