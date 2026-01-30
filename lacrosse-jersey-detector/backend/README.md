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

## API Endpoints

- `GET /api/health` - Health check
- `POST /api/upload` - Upload video file (MP4)
- `POST /api/analyze` - Start analysis job
- `GET /api/results/{job_id}` - Get analysis results

## Environment Variables

- `FRAME_EXTRACTION_INTERVAL` - Seconds between frame extraction (default: 1.0)
- `MAX_VIDEO_SIZE_MB` - Maximum video file size in MB (default: 500)
- `YOLO_MODEL_PATH` - Path to YOLOv8 model (default: yolov8n.pt)
- `OCR_ENGINE` - OCR engine to use: "easyocr" or "tesseract" (default: easyocr)
- `TIMESTAMP_GROUPING_THRESHOLD` - Max gap in seconds to group timestamps (default: 2.0)

## Architecture

- **API Layer** (`app/api/`): FastAPI routes and request/response models
- **Service Layer** (`app/services/`): Business logic for video handling and analysis orchestration
- **ML Layer** (`app/ml/`): Computer vision pipeline components

## Notes

- First run will download YOLOv8 model and EasyOCR models (may take a few minutes)
- Videos are stored in `uploads/` directory
- Analysis results are stored in `results/` directory as JSON files
