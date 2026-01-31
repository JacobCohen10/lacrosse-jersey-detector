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
