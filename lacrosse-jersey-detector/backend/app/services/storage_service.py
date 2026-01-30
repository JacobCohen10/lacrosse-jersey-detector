"""Service for managing file storage and retrieval."""
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from app.config import UPLOADS_DIR, RESULTS_DIR
from app.api.models import AnalysisResult


# Video file extensions we accept (must match upload validation)
VIDEO_EXTENSIONS = (".mp4", ".mov")


class StorageService:
    """Handles file system operations for videos and results."""
    
    def __init__(self):
        self.uploads_dir = UPLOADS_DIR
        self.results_dir = RESULTS_DIR
    
    def video_exists(self, video_id: str) -> bool:
        """Check if a video file exists (any supported extension)."""
        for ext in VIDEO_EXTENSIONS:
            if (self.uploads_dir / f"{video_id}{ext}").exists():
                return True
        return False
    
    def get_video_path(self, video_id: str) -> Path:
        """Get the path to a video file (any supported extension)."""
        for ext in VIDEO_EXTENSIONS:
            path = self.uploads_dir / f"{video_id}{ext}"
            if path.exists():
                return path
        return self.uploads_dir / f"{video_id}.mp4"
    
    def save_analysis_result(self, result: AnalysisResult):
        """Save analysis result to disk."""
        result_path = self.results_dir / f"{result.job_id}.json"
        with open(result_path, 'w') as f:
            json.dump(result.model_dump(mode='json'), f, indent=2, default=str)
    
    def get_analysis_result(self, job_id: str) -> Optional[AnalysisResult]:
        """Retrieve analysis result from disk."""
        result_path = self.results_dir / f"{job_id}.json"
        
        if not result_path.exists():
            return None
        
        try:
            with open(result_path, 'r') as f:
                data = json.load(f)
            return AnalysisResult(**data)
        except Exception as e:
            print(f"Error loading result: {e}")
            return None
    
    def update_analysis_status(self, job_id: str, status: str, error: Optional[str] = None):
        """Update the status of an analysis job."""
        result = self.get_analysis_result(job_id)
        if result:
            result.status = status
            if error:
                result.error = error
            if status == "completed":
                result.completed_at = datetime.now()
            self.save_analysis_result(result)
