# Lacrosse Jersey Detector - Backend

Backend API for analyzing lacrosse game footage to detect jersey numbers.

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure ffmpeg is installed on your system:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg
```

3. Run the server:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

API documentation (Swagger UI) is available at `http://localhost:8000/docs`

## Deploy on AWS Elastic Beanstalk

1. From the **backend** directory, create the deployment zip (output: `lacrosse-backend.zip` in the repo root). Include `.platform` so ffmpeg is installed on the instance:
   ```bash
   cd backend
   zip -r ../lacrosse-backend.zip . -x "*.pyc" -x "__pycache__/*" -x "venv/*" -x "uploads/*" -x "results/*"
   ```
   The zip must include **`.platform`** (ffmpeg) and **`.ebextensions`** (larger root volume so pip doesn’t run out of space).
2. In [Elastic Beanstalk](https://console.aws.amazon.com/elasticbeanstalk), create your environment with **Python 3.11** or **Python 3.12** (e.g. "Python 3.11 running on 64bit Amazon Linux 2"). Do not use Python 3.14—EasyOCR’s dependencies do not have prebuilt wheels for 3.14 and the install will fail. Upload **lacrosse-backend.zip** and deploy.
3. Set **CORS_ORIGINS** in Configuration → Software → Environment properties to your frontend URL.
4. The Procfile runs `uvicorn` on port 5000 for the default Python platform.

#### Troubleshooting EB: "Instance deployment failed to install application dependencies"

**If the log shows `python-bidi` and Rust/Cargo:** Your environment is using **Python 3.14**. EasyOCR depends on `python-bidi`, which has no prebuilt wheel for 3.14, so pip tries to build it from source and needs Rust—which fails on EB. **Fix:** Create (or recreate) your EB environment with **Python 3.11** or **Python 3.12**, not 3.14. In the EB console, choose a platform branch like **"Python 3.11 running on 64bit Amazon Linux 2"**. You cannot change Python version via a file; it is set by the platform you select.

**If the log shows `OSError: [Errno 28] No space left on device`:** The instance root disk is full during `pip install`. Do this in order: (1) Wait until environment **Status** is **Ready**. (2) **Set root volume in the EB Console**: **Configuration** → **Instances** → **Edit** → **Root volume (boot device)** → set **Root volume size** to **40** GiB (and type **gp3** if available) → **Apply**. (3) Wait for the config update to finish (Status **Ready** again). (4) Rebuild your zip so it includes **`.ebextensions`** and **`.platform`** (this repo adds a **prebuild** hook to free disk and **PIP_NO_CACHE_DIR**; it also uses **opencv-python-headless** to reduce install size). (5) Deploy the new zip. If it still fails, set root volume to **50** GiB in the console and redeploy.

The stack (PyTorch, Ultralytics, OpenCV, EasyOCR) is also heavy; install often fails on default settings (time/memory).

1. **Get the real error**
   - In EB Console → your environment → **Logs** (left) → **Request Logs** → **Full Logs**.
   - Download the zip, then open **`/var/log/eb-engine.log`** (or **eb-engine.log** in the zip).
   - Search for `ERROR` or the first `pip` failure; that’s the cause (e.g. out of memory, timeout, or a package that didn’t build).

2. **Give the deploy more time and memory**
   - **Configuration** → **Capacity** → set **Instance types** to at least **t3.small** (1 GB RAM). Default t2.micro (1 GB) often OOMs during `pip install`.
   - **Configuration** → **Deployment** → **Command timeout**: set to **1800** (30 minutes). Pip for this stack can take 15–20+ minutes.

3. **Ensure ffmpeg is on the instance**
   - Include the repo’s **`.platform`** folder in your zip so EB runs the install script and ffmpeg is available at runtime.

4. **If eb-engine.log shows out-of-memory (OOM) or timeout**
   - Use **t3.small** or **t3.medium** and the 30-minute command timeout above.
   - Optionally add a **swap file** via `.platform/hooks/predeploy` (e.g. 1 GB swap) so pip has more effective memory.

5. **If a specific package fails to build**
   - Note the package name from eb-engine.log. Common fixes: pin a version that has a prebuilt wheel (e.g. `torch` CPU on Linux), or add build dependencies in a `.platform` hook.

After changing **Capacity** or **Deployment** timeout, deploy again (same **lacrosse-backend.zip** or a new one that includes `.platform`).

## Deploy on Render.com

1. In [Render](https://render.com), create a **Web Service** and connect this repo.
2. Set **Root Directory** to `backend` (or use the repo-root `render.yaml` blueprint).
3. **Build**: `pip install -r requirements.txt`  
   **Start**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. In the service **Environment** tab, add **CORS_ORIGINS** = your frontend URL (e.g. `https://your-app.vercel.app`). Use a comma-separated list for multiple origins. Without this, the browser will block requests from your frontend.

## API Endpoints

- `GET /api/health` - Health check
- `POST /api/upload` - Upload video file (MP4)
- `POST /api/analyze` - Start analysis job
- `GET /api/results/{job_id}` - Get analysis results

## Environment Variables

- **`CORS_ORIGINS`** – Comma-separated allowed origins for CORS (required on Render when using a separate frontend, e.g. Vercel). Example: `https://your-app.vercel.app`
- `FRAME_EXTRACTION_INTERVAL` - Seconds between frame extraction (default: 0.3)
- `MAX_VIDEO_SIZE_MB` - Maximum video file size in MB (default: 500)
- `YOLO_MODEL_PATH` - Path to YOLOv8 model (default: yolov8m.pt)
- `OCR_ENGINE` - OCR engine: "easyocr" or "paddleocr" (default: easyocr)
- `TIMESTAMP_GROUPING_THRESHOLD` - Max gap in seconds to group timestamps (default: 2.0)

## Architecture

- **API Layer** (`app/api/`): FastAPI routes and request/response models
- **Service Layer** (`app/services/`): Business logic for video handling and analysis orchestration
- **ML Layer** (`app/ml/`): Computer vision pipeline components

## Notes

- First run will download YOLOv8 model and EasyOCR models (may take a few minutes)
- Videos are stored in `uploads/` directory
- Analysis results are stored in `results/` directory as JSON files
