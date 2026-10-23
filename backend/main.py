import os
import io
import csv
import json
import random
import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import List, Set, Dict, Any, Optional
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

# B2 SDK
# B2 SDK
try:
    from b2sdk.v2 import B2Api, InMemoryAccountInfo
    from b2sdk.v2.exception import B2Error, FileNotPresent
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
    B2Error = Exception
    FileNotPresent = Exception
    DownloadDestBytes = None

# Try to load environment variables from local or root .env file
try:
    from dotenv import load_dotenv
    from pathlib import Path
    # Search in current directory, backend directory, and parent project root
    env_paths = [Path(".env"), Path("backend/.env"), Path("../.env")]
    for p in env_paths:
        if p.exists():
            load_dotenv(dotenv_path=p)
            break
except ImportError:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("labeldata-backend")

# Environment Configuration & File Paths (referenced from .env, never hardcoded)
B2_KEY_ID = os.getenv("B2_KEY_ID", "").strip()
B2_APP_KEY = os.getenv("B2_APP_KEY", "").strip()
B2_BUCKET_NAME = os.getenv("B2_BUCKET_NAME", "").strip()
FRONTEND_URL = os.getenv("FRONTEND_URL", "").strip()

# Storage File Paths (from .env)
LABELS_CSV_FILENAME = (os.getenv("B2_LABELS_CSV_PATH") or os.getenv("LABELS_CSV_PATH") or "labels.csv").strip()
IMAGE_LIST_FILENAME = (os.getenv("B2_IMAGE_LIST_PATH") or os.getenv("IMAGE_LIST_PATH") or "image_list.json").strip()

# Constants
CSV_HEADER = ["filename", "crowding", "gaps", "discoloration", "gum_line", "symmetry", "skipped", "labeled_by", "timestamp"]
ALLOWED_SEVERITIES = {"mild", "moderate", "severe", "none"}

# How long an image is reserved for a user before it expires and goes back to the pool
RESERVATION_TIMEOUT_SECONDS = int(os.getenv("RESERVATION_TIMEOUT_SECONDS", "300"))  # 5 minutes default

app = FastAPI(
    title="LabelData Backend",
    description="FastAPI service for Backblaze B2 image labeling platform",
    version="1.0.0"
)

# CORS Setup
origins: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
if FRONTEND_URL:
    for url in FRONTEND_URL.split(","):
        cleaned = url.strip().rstrip("/")
        if cleaned and cleaned not in origins:
            origins.append(cleaned)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-Memory State
class State:
    b2_api: Optional[Any] = None
    bucket: Optional[Any] = None
    image_list: List[str] = []
    labeled_set: Set[str] = set()
    skipped_set: Set[str] = set()
    b2_ready: bool = False
    init_error: Optional[str] = None
    # Reservation map: {filename: {"labeler": str, "expires": datetime}}
    reservations: Dict[str, Dict[str, Any]] = {}

state = State()
csv_lock = asyncio.Lock()
reservation_lock = asyncio.Lock()


def cleanup_expired_reservations():
    """Removes expired reservations so images go back to the available pool."""
    now = datetime.now(timezone.utc)
    expired = [fname for fname, info in state.reservations.items() if now >= info["expires"]]
    for fname in expired:
        labeler = state.reservations[fname].get("labeler", "unknown")
        del state.reservations[fname]
        logger.info(f"Reservation expired for '{fname}' (was held by '{labeler}')")
    return len(expired)


# Models
class SubmitLabelRequest(BaseModel):
    filename: str = Field(..., description="Image filename being labeled")
    labeler: str = Field(..., description="Reviewer name or ID")
    crowding: str = Field("none", description="Mild, moderate, severe, or none")
    gaps: str = Field("none", description="Mild, moderate, severe, or none")
    discoloration: str = Field("none", description="Mild, moderate, severe, or none")
    gum_line: str = Field("none", description="Mild, moderate, severe, or none")
    symmetry: str = Field("none", description="Mild, moderate, severe, or none")
    skipped: bool = Field(False, description="Flag if photo is blurry or unusable")


# Helper Functions
def init_b2():
    """Initializes connection to Backblaze B2 and loads initial CSV state."""
    if not B2_KEY_ID or not B2_APP_KEY or not B2_BUCKET_NAME:
        state.b2_ready = False
        state.init_error = "Missing B2_KEY_ID, B2_APP_KEY, or B2_BUCKET_NAME environment variables."
        logger.warning(state.init_error)
        return False

    try:
        info = InMemoryAccountInfo()
        b2_api = B2Api(info)
        b2_api.authorize_account("production", B2_KEY_ID, B2_APP_KEY)
        bucket = b2_api.get_bucket_by_name(B2_BUCKET_NAME)

        state.b2_api = b2_api
        state.bucket = bucket
        state.b2_ready = True
        state.init_error = None
        logger.info(f"Successfully authenticated with B2 bucket: {B2_BUCKET_NAME}")

        # Load or initialize labels.csv
        load_or_create_labels_csv()

        # Load image_list.json
        load_image_list()
        return True
    except Exception as e:
        state.b2_ready = False
        state.init_error = f"Failed to connect to B2: {str(e)}"
        logger.error(state.init_error, exc_info=True)
        return False


def load_or_create_labels_csv():
    """Fetches labels.csv from B2, creates it if missing, and syncs in-memory set."""
    if not state.bucket:
        return

    try:
        dest = DownloadDestBytes()
        state.bucket.download_file_by_name(LABELS_CSV_FILENAME, dest)
        csv_bytes = dest.get_bytes_written()
        csv_text = csv_bytes.decode("utf-8")
        sync_csv_content(csv_text)
        logger.info(f"Loaded existing labels.csv ({len(state.labeled_set)} labeled files).")
    except (FileNotPresent, Exception):
        # File doesn't exist, create it with header
        logger.info("labels.csv not found in B2 bucket. Initializing new labels.csv...")
        header_content = ",".join(CSV_HEADER) + "\n"
        state.bucket.upload_bytes(
            header_content.encode("utf-8"),
            file_name=LABELS_CSV_FILENAME,
            content_type="text/csv"
        )
        state.labeled_set = set()
        state.skipped_set = set()


def sync_csv_content(csv_text: str):
    """Parses labels.csv text and updates labeled_set and skipped_set."""
    reader = csv.DictReader(io.StringIO(csv_text))
    labeled = set()
    skipped = set()

    for row in reader:
        filename = row.get("filename", "").strip()
        if filename:
            labeled.add(filename)
            is_skip = str(row.get("skipped", "")).strip().lower() in ("true", "1", "yes")
            if is_skip:
                skipped.add(filename)

    state.labeled_set = labeled
    state.skipped_set = skipped


def load_image_list():
    """Loads image_list.json from B2, or lists bucket photos as fallback."""
    if not state.bucket:
        return

    try:
        dest = DownloadDestBytes()
        state.bucket.download_file_by_name(IMAGE_LIST_FILENAME, dest)
        content = json.loads(dest.get_bytes_written().decode("utf-8"))
        if isinstance(content, list):
            state.image_list = [str(item).strip() for item in content if str(item).strip()]
        elif isinstance(content, dict) and "images" in content:
            state.image_list = [str(item).strip() for item in content["images"] if str(item).strip()]
        logger.info(f"Loaded {len(state.image_list)} images from {IMAGE_LIST_FILENAME}.")
    except Exception as e:
        logger.warning(f"Could not load {IMAGE_LIST_FILENAME} ({e}). Scanning bucket files as fallback...")
        try:
            image_extensions = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
            found_images = []
            for file_version, _ in state.bucket.ls(recursive=True):
                name = file_version.file_name
                if name.lower().endswith(image_extensions) and not name.startswith("."):
                    found_images.append(name)
            state.image_list = found_images
            logger.info(f"Fallback bucket scan found {len(found_images)} image files.")
        except Exception as scan_err:
            logger.error(f"Failed to scan bucket files: {scan_err}")
            state.image_list = []


def generate_signed_url(filename: str, valid_duration: int = 3600) -> str:
    """Generates a temporary authorized download URL for private B2 bucket access."""
    if not state.bucket or not state.b2_api:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="B2 storage not available")

    try:
        auth_token = state.bucket.get_download_authorization(
            file_name_prefix=filename,
            valid_duration_in_seconds=valid_duration
        )
        download_url = state.b2_api.account_info.get_download_url()
        encoded_name = quote(filename, safe="/")
        return f"{download_url}/file/{B2_BUCKET_NAME}/{encoded_name}?Authorization={auth_token}"
    except Exception as e:
        logger.error(f"Failed to generate download token for {filename}: {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Failed to authorize image download: {str(e)}")


# Startup Event
@app.on_event("startup")
async def on_startup():
    init_b2()


# Endpoints
@app.get("/health", tags=["Health"])
@app.get("/", tags=["Health"])
def health_check():
    """Health check endpoint indicating service and B2 connection status."""
    return {
        "status": "online",
        "b2_connected": state.b2_ready,
        "bucket": B2_BUCKET_NAME or None,
        "total_images": len(state.image_list),
        "total_labeled": len(state.labeled_set),
        "error": state.init_error
    }


@app.get("/reload", tags=["Management"])
def reload_from_storage():
    """Manually reloads image_list.json and labels.csv from B2 storage."""
    if not state.b2_ready:
        init_b2()
    else:
        load_image_list()
        load_or_create_labels_csv()
    return {
        "status": "reloaded",
        "total_images": len(state.image_list),
        "total_labeled": len(state.labeled_set)
    }


@app.get("/next-image", tags=["Labeling"])
async def get_next_image(labeler: str = Query(..., description="Name of the labeler")):
    """
    Returns a random unlabeled & unreserved image for this labeler.
    Reserves the image for RESERVATION_TIMEOUT_SECONDS so no other user gets the same one.
    If the labeler already has a valid reservation, returns that same image.
    Returns all_done: True if no unlabeled images remain.
    """
    if not state.b2_ready:
        if not init_b2():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=state.init_error or "Storage service (Backblaze B2) is unreachable."
            )

    # Auto-reload if list is empty (e.g. images were uploaded after backend started)
    if len(state.image_list) == 0:
        load_image_list()

    async with reservation_lock:
        # Clean up any expired reservations first
        cleanup_expired_reservations()

        total_images = len(state.image_list)
        done_count = len(state.labeled_set)

        # Check if this labeler already has a valid reservation — return it instead of a new one
        for fname, info in state.reservations.items():
            if info["labeler"] == labeler and fname not in state.labeled_set:
                signed_url = generate_signed_url(fname)
                return {
                    "all_done": False,
                    "filename": fname,
                    "url": signed_url,
                    "done": done_count,
                    "total": total_images
                }

        # Build pool: unlabeled AND not currently reserved by anyone
        reserved_filenames = set(state.reservations.keys())
        available = [
            name for name in state.image_list
            if name not in state.labeled_set and name not in reserved_filenames
        ]

        if not available:
            # Check if there are truly no unlabeled images left (ignoring reservations)
            truly_remaining = [n for n in state.image_list if n not in state.labeled_set]
            if not truly_remaining:
                return {
                    "all_done": True,
                    "filename": None,
                    "url": None,
                    "done": total_images,
                    "total": total_images,
                    "message": "All images have been labeled."
                }
            else:
                # All remaining images are reserved by other users — ask to wait
                return {
                    "all_done": False,
                    "filename": None,
                    "url": None,
                    "done": done_count,
                    "total": total_images,
                    "message": f"All remaining images are being reviewed by other labelers. Please wait and retry in a moment."
                }

        # Pick a random image from the available pool
        next_filename = random.choice(available)

        # Reserve it for this labeler
        state.reservations[next_filename] = {
            "labeler": labeler,
            "expires": datetime.now(timezone.utc) + timedelta(seconds=RESERVATION_TIMEOUT_SECONDS)
        }
        logger.info(f"Reserved '{next_filename}' for labeler '{labeler}' ({RESERVATION_TIMEOUT_SECONDS}s)")

    # Generate signed URL (outside the lock for speed)
    signed_url = generate_signed_url(next_filename)

    return {
        "all_done": False,
        "filename": next_filename,
        "url": signed_url,
        "done": done_count,
        "total": total_images
    }


@app.post("/submit-label", tags=["Labeling"])
async def submit_label(payload: SubmitLabelRequest):
    """
    Appends a new label row to labels.csv in B2.
    Safe against concurrent submissions by downloading latest CSV and locking writes.
    """
    if not state.b2_ready:
        if not init_b2():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=state.init_error or "Storage service (Backblaze B2) is unreachable."
            )

    # Validate severities if not skipped
    crowding = payload.crowding.lower().strip()
    gaps = payload.gaps.lower().strip()
    discoloration = payload.discoloration.lower().strip()
    gum_line = payload.gum_line.lower().strip()
    symmetry = payload.symmetry.lower().strip()

    if not payload.skipped:
        for field_name, val in [("crowding", crowding), ("gaps", gaps), ("discoloration", discoloration), ("gum_line", gum_line), ("symmetry", symmetry)]:
            if val not in ALLOWED_SEVERITIES:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Field '{field_name}' must be one of: {', '.join(ALLOWED_SEVERITIES)}"
                )
    else:
        crowding = gaps = discoloration = gum_line = symmetry = "none"

    # Acquire lock for thread/coroutine-safe CSV update
    async with csv_lock:
        # Download latest labels.csv to prevent double-labeling and race conditions
        try:
            dest = DownloadDestBytes()
            state.bucket.download_file_by_name(LABELS_CSV_FILENAME, dest)
            current_csv_text = dest.get_bytes_written().decode("utf-8")
        except Exception:
            current_csv_text = ",".join(CSV_HEADER) + "\n"

        # Check if already labeled
        reader = csv.DictReader(io.StringIO(current_csv_text))
        existing_filenames = {row.get("filename", "").strip() for row in reader if row.get("filename")}

        if payload.filename.strip() in existing_filenames:
            # Refresh in-memory set
            sync_csv_content(current_csv_text)
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Image '{payload.filename}' has already been labeled."
            )

        # Prepare new row
        timestamp = datetime.now(timezone.utc).isoformat()
        new_row = [
            payload.filename.strip(),
            crowding,
            gaps,
            discoloration,
            gum_line,
            symmetry,
            str(payload.skipped),
            payload.labeler.strip(),
            timestamp
        ]

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(new_row)
        new_row_csv = output.getvalue()

        # Append to existing CSV string ensuring newline
        if not current_csv_text.endswith("\n"):
            current_csv_text += "\n"
        updated_csv = current_csv_text + new_row_csv

        # Upload back to B2
        try:
            state.bucket.upload_bytes(
                updated_csv.encode("utf-8"),
                file_name=LABELS_CSV_FILENAME,
                content_type="text/csv"
            )
        except Exception as e:
            logger.error(f"Failed to upload updated labels.csv to B2: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Failed to save label to storage. Please retry."
            )

        # Update in-memory state
        state.labeled_set.add(payload.filename.strip())
        if payload.skipped:
            state.skipped_set.add(payload.filename.strip())

        # Clear the reservation (image is now labeled, no need to hold it)
        state.reservations.pop(payload.filename.strip(), None)

    return {
        "status": "success",
        "filename": payload.filename,
        "labeled_by": payload.labeler,
        "timestamp": timestamp
    }


@app.get("/progress", tags=["Metrics"])
def get_progress():
    """Returns progress counts: total, labeled, skipped, and remaining."""
    total = len(state.image_list)
    labeled = len(state.labeled_set)
    skipped = len(state.skipped_set)
    remaining = max(0, total - labeled)

    return {
        "total": total,
        "labeled": labeled,
        "skipped": skipped,
        "remaining": remaining
    }


@app.get("/export-csv", tags=["Export"])
def export_csv():
    """Streams the current labels.csv file from B2 as a downloadable file."""
    if not state.b2_ready:
        if not init_b2():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Storage service (Backblaze B2) is unreachable."
            )

    try:
        dest = DownloadDestBytes()
        state.bucket.download_file_by_name(LABELS_CSV_FILENAME, dest)
        csv_bytes = dest.get_bytes_written()
    except (FileNotPresent, Exception):
        # Empty CSV fallback
        header_content = ",".join(CSV_HEADER) + "\n"
        csv_bytes = header_content.encode("utf-8")

    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={LABELS_CSV_FILENAME}"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
