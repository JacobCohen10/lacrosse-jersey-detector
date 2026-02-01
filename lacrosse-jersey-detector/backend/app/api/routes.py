"""API routes for the application."""
import uuid
import numpy as np
import cv2
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional, Union

from app.api.models import (
    UploadResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    AnalysisResult,
    ImageDetectionResponse,
    PlayerDetection,
)
from app.services.video_service import VideoService
from app.services.analysis_service import AnalysisService
from app.services.storage_service import StorageService
from app.ml.player_detector import PlayerDetector
from app.ml.jersey_cropper import JerseyCropper
from app.ml.ocr_engine import OCREngine

router = APIRouter()

video_service = VideoService()
analysis_service = AnalysisService()
storage_service = StorageService()
player_detector = PlayerDetector()
jersey_cropper = JerseyCropper()
ocr_engine = OCREngine()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "lacrosse-jersey-detector"}


@router.post("/upload", response_model=UploadResponse)
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a video file.
    
    Accepts MP4 and MOV video files and stores them for analysis.
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(('.mp4', '.mov')):
        raise HTTPException(status_code=400, detail="Only MP4 and MOV video files are supported")
    
    # Generate video ID
    video_id = str(uuid.uuid4())
    
    # Save video file
    try:
        filename = await video_service.save_video(file, video_id)
        return UploadResponse(
            video_id=video_id,
            filename=filename,
            message="Video uploaded successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload video: {str(e)}")


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_video(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """
    Start analysis of a video for a specific jersey number.
    Supports (1) video_id (local upload) or (2) video_url + job_id (Vercel Blob).
    """
    if request.video_url and request.job_id:
        # Vercel hybrid: download from Blob URL, then analyze with video_id=job_id
        try:
            await video_service.save_video_from_url(request.video_url, request.job_id)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch video: {e}")
        job_id = request.job_id
        video_id = request.job_id
    elif request.video_id:
        # Local/Render upload: video already on disk
        if not storage_service.video_exists(request.video_id):
            raise HTTPException(status_code=404, detail="Video not found")
        job_id = str(uuid.uuid4())
        video_id = request.video_id
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either video_id (local) or both video_url and job_id (Vercel Blob)",
        )
    
    background_tasks.add_task(
        analysis_service.analyze_video,
        job_id=job_id,
        video_id=video_id,
        jersey_number=request.jersey_number,
        fast_mode=request.fast_mode,
    )
    
    return AnalyzeResponse(
        job_id=job_id,
        message="Analysis started",
        status="processing"
    )


@router.get("/results/{job_id}", response_model=AnalysisResult)
async def get_results(job_id: str):
    """
    Get analysis results for a job.
    
    Returns the current status and results (if completed).
    """
    result = storage_service.get_analysis_result(job_id)
    
    if result is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return result


@router.post("/detect-image", response_model=ImageDetectionResponse)
async def detect_jersey_numbers_in_image(file: UploadFile = File(...)):
    """
    Detect all players in a single image and assign at most one jersey number per player.
    Output: list of { player_bbox, jersey_number (0-99 or "unknown"), confidence }.
    """
    # Decode image
    raw = await file.read()
    buf = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid or unsupported image")
    # Detect players (prioritize recall)
    boxes = player_detector.detect_players(image)
    detections = []
    wider_cropper = JerseyCropper(jersey_region_ratio=(0.15, 0.05, 0.85, 0.5))
    for box in boxes:
        x1, y1, x2, y2 = box
        player_bbox = [int(x1), int(y1), int(x2), int(y2)]
        # Get jersey crops (same logic as video pipeline)
        crops = []
        c1 = jersey_cropper.crop_jersey_region(image, box)
        if c1 is not None:
            crops.append(c1)
        c2 = wider_cropper.crop_jersey_region(image, box)
        if c2 is not None:
            crops.append(c2)
        # One number per player: collect (number, conf) from all crops, pick best
        best_number: Optional[str] = None
        best_conf = 0.0
        for crop in crops:
            number, conf = ocr_engine.read_jersey_number(crop)
            if number and conf > best_conf:
                best_number = number
                best_conf = conf
        # Output: 0-99 as int, or "unknown"
        if best_number is not None and best_conf > 0.0:
            try:
                num_int = int(best_number)
                if 0 <= num_int <= 99:
                    jersey_number: Union[int, str] = num_int
                else:
                    jersey_number = "unknown"
                    best_conf = 0.0
            except ValueError:
                jersey_number = "unknown"
                best_conf = 0.0
        else:
            jersey_number = "unknown"
            best_conf = 0.0
        detections.append(
            PlayerDetection(
                player_bbox=player_bbox,
                jersey_number=jersey_number,
                confidence=round(best_conf, 4),
            )
        )
    return ImageDetectionResponse(detections=detections)
