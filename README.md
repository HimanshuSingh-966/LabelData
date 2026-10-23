# LabelData - Dental Image Labeling System

A distributed image labeling platform for dental specialists to review and rate smile photos stored in a private Backblaze B2 bucket.

- **Backend**: FastAPI (Deployed to **Render**)
- **Frontend**: Next.js (Deployed to **Vercel**)
- **Storage**: Private Backblaze B2 Bucket

---

## Repository Structure

```
LabelData/
├── data/
│   ├── raw/                  # Place downloaded master dataset photos here (.gitkept)
│   ├── pulled_batch/         # Staging directory where pulled/sampled batches are stored (.gitkept)
│   └── pulled_history.csv    # Local CSV tracking all photos ever pulled/uploaded
│
├── backend/                  # FastAPI backend application
│   ├── main.py               # API endpoints, B2 signed URLs, CSV sync & concurrency locks
│   └── requirements.txt      # Python dependencies (fastapi, uvicorn, b2sdk, python-dotenv)
│
├── frontend/                 # Next.js frontend application
│   ├── app/
│   │   ├── page.tsx          # Reviewer interface, state management, assessment controls
│   │   ├── globals.css       # Clean medical UI styling & touch targets
│   │   └── layout.tsx        # Next.js root layout & metadata
│   ├── public/               # Static assets & public files (.gitkept)
│   ├── package.json          # Node dependencies
│   ├── tsconfig.json         # TypeScript configuration
│   └── next.config.mjs       # Next.js configuration
│
├── scripts/
│   ├── shuffle_dataset.py    # Integrated dataset pull, mouth-crop, deduplication & B2 sync
│   ├── crop_engine.py        # EasyPortrait SegFormer-B0 ONNX mouth cropping engine
│   └── crop_mouths.py        # Standalone CLI wrapper for mouth cropping
│
├── docs/
│   ├── backend_prompt.txt    # Backend architecture specification
│   ├── frontend_prompt.txt   # Frontend UI/UX specification
│   └── images_processing_model_prompt.txt # Face-parsing crop model specification
│
├── requirements.txt          # Python dependencies for pipeline and model inference
├── vercel.json               # Vercel monorepo deployment configuration
├── .env.example              # Single unified configuration template for all services
├── .gitignore                # Global gitignore for Python, Node, data, models, and envs
└── README.md                 # Single source of truth for setup and deployment
```

---

## Environment Variables

All configuration is consolidated in [`.env.example`](file:///.env.example). Create a `.env` file before running:

| Variable | Scope | Description | Default / Example |
|---|---|---|---|
| `LOCAL_RAW_DATASET_DIR` | Local Dataset | Path where raw master dataset is stored on PC | `./data/raw` |
| `LOCAL_PULLED_BATCH_DIR`| Local Dataset | Path where sampled batch images are saved on PC | `./data/pulled_batch` |
| `LOCAL_PULL_HISTORY_PATH`| Local Dataset | Local CSV tracking all pulled photos across batches | `./data/pulled_history.csv` |
| `MODEL_PATH` | Face Parsing | Local path for SegFormer-B0 ONNX weights | `./models/segformer_b0_fp_512.onnx` |
| `CROP_LOG_PATH` | Face Parsing | CSV log recording success/fail of mouth crops | `./data/crop_log.csv` |
| `CROP_MIN_PIXELS` | Face Parsing | Minimum teeth/lips pixel count threshold | `50` |
| `CROP_PADDING` | Face Parsing | Margin padding ratio around mouth (15% per side) | `0.15` |
| `PULL_LIMIT` | Sampling | Initial pull pool cap | `3000` |
| `TARGET_SAMPLE_SIZE` | Sampling | Target successfully cropped smile photos | `2200` (~2,000–2,400) |
| `SHUFFLE_SEED` | Sampling | Seed for deterministic, reproducible shuffling | `42` |
| `B2_KEY_ID` | Cloud / B2 | Backblaze B2 Application Key ID | `005...` |
| `B2_APP_KEY` | Cloud / B2 | Backblaze B2 Application Key | `K005...` |
| `B2_BUCKET_NAME` | Cloud / B2 | Name of your private B2 bucket | `dental-smiles-bucket` |
| `B2_IMAGE_LIST_PATH` | Cloud / B2 | Name of batch review manifest inside B2 | `image_list.json` |
| `B2_LABELS_CSV_PATH` | Cloud / B2 | Name of progress labels CSV inside B2 | `labels.csv` |
| `FRONTEND_URL` | Backend | Allowed CORS origin (Vercel domain or localhost) | `https://your-app.vercel.app` |
| `NEXT_PUBLIC_API_URL` | Frontend | Base URL of deployed backend on Render | `https://your-backend.onrender.com` |

---

## Dataset Pipeline: Shuffling, Deduplication & Cloud Sync

Since the dental dataset (`smile_unprocessed`) already consists of close-up cropped mouth photos, [`scripts/shuffle_dataset.py`](file:///scripts/shuffle_dataset.py) directly validates and stages the images (checking image readability and non-zero dimensions) without running face-parsing models:

```bash
# Install local dependencies
pip install -r requirements.txt

# Run integrated pipeline (pulls from data/raw, validates, and uploads to B2):
python scripts/shuffle_dataset.py --upload-images

# Local staging only (without uploading to B2):
python scripts/shuffle_dataset.py
```

### What this command does:
1. **Deduplicates:** Checks `data/pulled_history.csv` to ensure no image is ever pulled twice across multiple batches.
2. **Deterministic Shuffling:** Uses fixed seed (`SHUFFLE_SEED=42`) for reproducible random sampling.
3. **Validates & Stages:** Validates that each image is non-corrupt and has valid dimensions, staging valid photos into `data/pulled_batch/`.
4. **Cloud Upload:** Automatically uploads the staged photos and `image_list.json` directly to your Backblaze B2 bucket.
5. **Updates Pull History:** Records new image filenames to `data/pulled_history.csv`.

*(Optional: If you ever have uncropped, full-face portraits in the future, pass `--crop` to run the EasyPortrait mouth cropper).*

---

## Deployment Guide

### 1. Backend on Render
1. Create a **New Web Service** on [Render](https://render.com/).
2. Connect your Git repository.
3. Configure service settings:
   - **Root Directory**: `backend`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Add the following **Environment Variables**:
   - `B2_KEY_ID`
   - `B2_APP_KEY`
   - `B2_BUCKET_NAME`
   - `FRONTEND_URL` (set to your Vercel URL once deployed, or `*` temporarily)
   - `LABELS_CSV_PATH` (optional, defaults to `labels.csv`)
   - `IMAGE_LIST_PATH` (optional, defaults to `image_list.json`)
5. Deploy the service and copy the provided service URL (e.g., `https://labeldata-backend.onrender.com`).

### 2. Frontend on Vercel
Thanks to [`vercel.json`](file:///vercel.json) in the repository root, deployment to Vercel requires zero manual directory tweaking:
1. Import the Git repository into [Vercel](https://vercel.com/).
2. Add the following **Environment Variable**:
   - `NEXT_PUBLIC_API_URL`: Your Render backend URL (e.g. `https://labeldata-backend.onrender.com`).
3. Click **Deploy**.

---

## Local Development

### Running the Backend
```bash
cd backend
python -m venv venv

# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

pip install -r requirements.txt

# Create .env in root or backend/ and fill in B2 credentials
uvicorn main:app --reload --port 8000
```
Backend will run at `http://localhost:8000`. Swagger API docs available at `http://localhost:8000/docs`.

### Running the Frontend
```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.
