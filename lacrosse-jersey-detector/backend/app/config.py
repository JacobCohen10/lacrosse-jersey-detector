"""Configuration settings for the application."""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Directories
UPLOADS_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

# Create directories if they don't exist
UPLOADS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Video processing settings (denser sampling to catch more moments)
FRAME_EXTRACTION_INTERVAL = float(os.getenv("FRAME_EXTRACTION_INTERVAL", "0.3"))  # seconds
# Fast mode: fewer frames, fewer crops, fewer OCR strategies (5–10x faster, slightly lower recall)
FAST_MODE_FRAME_INTERVAL = float(os.getenv("FAST_MODE_FRAME_INTERVAL", "0.5"))  # seconds between frames
FAST_MODE_YOLO_CONF = float(os.getenv("FAST_MODE_YOLO_CONF", "0.45"))
FAST_MODE_YOLO_IMGSZ = int(os.getenv("FAST_MODE_YOLO_IMGSZ", "640"))
MAX_VIDEO_SIZE_MB = int(os.getenv("MAX_VIDEO_SIZE_MB", "500"))  # 500MB default

# ML settings — STRONGEST: large model + high res for small/distant players
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8m.pt")  # m = strong small-object; use yolov8l.pt for max
YOLO_CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", "0.2"))
MIN_PLAYER_HEIGHT = int(os.getenv("MIN_PLAYER_HEIGHT", "36"))  # Include very small/far players
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "1280"))  # High res for small numbers
OCR_ENGINE = os.getenv("OCR_ENGINE", "easyocr")  # easyocr = default; set paddleocr if paddlepaddle+paddleocr installed

# OCR settings — STRONGEST: high res upscale + all strategies
MIN_STRATEGY_CONFIDENCE = float(os.getenv("MIN_STRATEGY_CONFIDENCE", "0.14"))
OCR_FRAME_MIN_CONFIDENCE = float(os.getenv("OCR_FRAME_MIN_CONFIDENCE", "0.26"))
OCR_MIN_RESOLUTION = int(os.getenv("OCR_MIN_RESOLUTION", "320"))  # Strong upscale for small numbers
ENSEMBLE_MIN_VOTES = int(os.getenv("ENSEMBLE_MIN_VOTES", "1"))
ENSEMBLE_HIGH_CONFIDENCE_THRESHOLD = float(os.getenv("ENSEMBLE_HIGH_CONFIDENCE_THRESHOLD", "0.48"))
OCR_EARLY_EXIT_CONFIDENCE = float(os.getenv("OCR_EARLY_EXIT_CONFIDENCE", "0.88"))
USE_ROTATION_STRATEGIES = os.getenv("USE_ROTATION_STRATEGIES", "true").lower() == "true"
MERGE_DIGITS_MIN_CONFIDENCE = float(os.getenv("MERGE_DIGITS_MIN_CONFIDENCE", "0.11"))

# API settings
API_PREFIX = "/api"
CORS_ORIGINS = ["http://localhost:3000", "http://localhost:5173", "http://localhost:5174"]

# Timestamp aggregation settings
TIMESTAMP_GROUPING_THRESHOLD = float(os.getenv("TIMESTAMP_GROUPING_THRESHOLD", "2.0"))  # seconds
TEMPORAL_CONSENSUS_WINDOW = float(os.getenv("TEMPORAL_CONSENSUS_WINDOW", "0.5"))  # Time window in seconds for consensus (relaxed for movement)
TEMPORAL_CONSENSUS_MIN_DETECTIONS = int(os.getenv("TEMPORAL_CONSENSUS_MIN_DETECTIONS", "1"))  # Min detections in window (1 = accept single detections)
MIN_DETECTIONS_PER_INTERVAL = int(os.getenv("MIN_DETECTIONS_PER_INTERVAL", "1"))  # Minimum detections per interval (relaxed)
