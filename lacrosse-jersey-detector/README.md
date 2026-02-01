# Lacrosse Jersey Detector MVP

A web application that analyzes lacrosse game footage to detect and timestamp plays where a specific jersey number appears.

## Problem

Coaches and analysts need to quickly find specific player moments in game footage. Manually scrubbing through hours of video is time-consuming. This MVP automates the process by using computer vision to detect jersey numbers and return timestamp intervals.

## Approach

The application uses a multi-stage pipeline:

1. **Video Upload**: User uploads MP4 video file
2. **Frame Extraction**: Extract frames at fixed intervals (1 second default)
3. **Player Detection**: Use YOLOv8 to detect players (persons) in each frame
4. **Jersey Cropping**: Crop jersey regions from detected player bounding boxes
5. **OCR**: Use EasyOCR to read jersey numbers from cropped regions
6. **Timestamp Aggregation**: Group consecutive detections into play intervals
7. **Results Display**: Show timestamp intervals where the jersey number was detected

## Tech Stack

### Frontend
- React 18 with TypeScript
- Tailwind CSS for styling
- Vite for build tooling
- Axios for API calls

### Backend
- Python 3.9+
- FastAPI for REST API
- YOLOv8 (ultralytics) for player detection
- EasyOCR for jersey number recognition
- OpenCV (cv2) for video processing
- ffmpeg for video operations

## Project Structure

```
lacrosse-jersey-detector/
├── frontend/          # React + TypeScript frontend
├── backend/           # Python + FastAPI backend
├── README.md          # This file
└── .gitignore
```

## Setup Instructions

### Prerequisites

1. **Python 3.9+** installed
2. **Node.js 18+** and npm installed
3. **ffmpeg** installed:
   - macOS: `brew install ffmpeg`
   - Ubuntu/Debian: `sudo apt-get install ffmpeg`
   - Windows: Download from [ffmpeg.org](https://ffmpeg.org/)

### Backend Setup

1. Navigate to backend directory:
```bash
cd backend
```

2. Create virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the server:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`
API documentation: `http://localhost:8000/docs`

**Note**: First run will download YOLOv8 and EasyOCR models (may take a few minutes).

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`

## Usage

1. Open the frontend in your browser (`http://localhost:5173`)
2. Upload an MP4 video file
3. Enter the jersey number you want to detect (e.g., "7")
4. Click "Analyze"
5. Wait for processing (may take several minutes for long videos)
6. View the timestamp intervals where the jersey number was detected

## Assumptions & Limitations

### Assumptions

1. **Video Format**: Assumes MP4 format (other formats may need conversion)
2. **Jersey Visibility**: Assumes jersey numbers are visible and readable in the video
3. **Single Jersey**: Analyzes one jersey number per request
4. **Video Quality**: Assumes reasonable video quality for OCR to work

### Limitations

1. **Accuracy**: OCR accuracy depends on:
   - Video quality and resolution
   - Jersey number visibility
   - Lighting conditions
   - Camera angle

2. **Processing Time**: Analysis can take several minutes for long videos:
   - Frame extraction time depends on video length
   - Player detection and OCR are computationally intensive

3. **False Positives**: YOLOv8 may detect:
   - Non-players (coaches, referees, spectators)
   - Players from other teams with similar numbers

4. **Jersey Region**: The jersey cropping assumes jerseys are in the upper-middle region of player bounding boxes. This may not work for all camera angles.

5. **No Authentication**: MVP assumes single-user or local use

6. **File Storage**: Videos and results are stored locally on the server

## Deploy: Vercel (frontend + serverless API) + Render (analysis)

You can run the **frontend and upload/results API** on Vercel and the **heavy analysis** on Render:

1. **Render**: Deploy the backend (see `backend/README.md`). Set **CORS_ORIGINS** to your Vercel URL (e.g. `https://your-app.vercel.app`).
2. **Vercel**: Connect this repo. Set **Root Directory** to the repo root (so `api/` and `frontend/` are used).
3. **Vercel env**: Add **RENDER_API_URL** = your Render URL (e.g. `https://lacrosse-jersey-api.onrender.com`). Create **Vercel Blob** and **Vercel KV** in the dashboard and link them (env vars are added automatically).
4. **Frontend**: Leave **VITE_API_URL** unset in production so the app uses the same origin (`/api/*` on Vercel).

Flow: Upload → Vercel Blob + KV. Analyze → Vercel calls Render with the video URL; Render downloads the video, runs YOLO+OCR, stores results on Render. Get results → Vercel proxies to Render. See `api/README.md` for API env details.

**Upload size**: Vercel request body limit applies (e.g. 4.5 MB Hobby). For larger videos use a Pro plan or client-side direct upload to Blob.

## Configuration

### Backend Environment Variables

Create a `.env` file in the `backend/` directory (optional):

```env
FRAME_EXTRACTION_INTERVAL=1.0          # Seconds between frame extraction
MAX_VIDEO_SIZE_MB=500                  # Maximum video file size
YOLO_MODEL_PATH=yolov8n.pt            # YOLOv8 model file
OCR_ENGINE=easyocr                     # OCR engine: easyocr or tesseract
TIMESTAMP_GROUPING_THRESHOLD=2.0       # Max gap in seconds to group timestamps
```

### Frontend Environment Variables

Create a `.env` file in the `frontend/` directory (optional):

```env
VITE_API_URL=http://localhost:8000     # Backend API URL
```

## API Endpoints

- `GET /api/health` - Health check
- `POST /api/upload` - Upload video file (multipart/form-data)
- `POST /api/analyze` - Start analysis job (JSON body: `{video_id, jersey_number}`)
- `GET /api/results/{job_id}` - Get analysis results

## Possible Future Improvements

1. **Real-time Progress**: WebSocket updates for analysis progress
2. **Multiple Jersey Numbers**: Analyze multiple numbers in one request
3. **Video Preview**: Show video with timestamp markers
4. **Better Filtering**: Filter out non-players (refs, coaches) using additional heuristics
5. **Model Fine-tuning**: Fine-tune YOLOv8 on lacrosse-specific data
6. **Database**: Store results in a database instead of JSON files
7. **User Authentication**: Add user accounts and session management
8. **Cloud Storage**: Store videos in cloud storage (S3, etc.)
9. **Batch Processing**: Process multiple videos at once
10. **Video Player Integration**: Click timestamp to jump to that moment in video
11. **Export Results**: Export results as CSV or JSON
12. **Better OCR**: Fine-tune OCR or use specialized number recognition models

## Troubleshooting

### Backend Issues

- **Import errors**: Ensure virtual environment is activated and dependencies are installed
- **ffmpeg not found**: Install ffmpeg and ensure it's in your PATH
- **Model download fails**: Check internet connection; models download on first run
- **Out of memory**: Reduce `FRAME_EXTRACTION_INTERVAL` or use smaller videos

### Frontend Issues

- **API connection errors**: Ensure backend is running on port 8000
- **CORS errors**: Check that backend CORS settings include frontend URL
- **Build errors**: Clear `node_modules` and reinstall: `rm -rf node_modules && npm install`

## License

This is an MVP project for demonstration purposes.
