"""Service for handling video uploads and validation."""
import aiofiles
from pathlib import Path
from app.config import UPLOADS_DIR, MAX_VIDEO_SIZE_MB
from app.services.storage_service import StorageService


class VideoService:
    """Handles video file operations."""
    
    def __init__(self):
        self.uploads_dir = UPLOADS_DIR
        self.storage_service = StorageService()
        self.max_size_bytes = MAX_VIDEO_SIZE_MB * 1024 * 1024
    
    async def save_video(self, file, video_id: str) -> str:
        """
        Save uploaded video file.
        
        Args:
            file: Uploaded file object
            video_id: Unique identifier for the video
            
        Returns:
            Filename of saved video
        """
        ext = Path(file.filename).suffix.lower() if file.filename else ".mp4"
        if ext not in (".mp4", ".mov"):
            ext = ".mp4"
        filename = f"{video_id}{ext}"
        file_path = self.uploads_dir / filename
        
        # Check file size
        file_content = await file.read()
        if len(file_content) > self.max_size_bytes:
            raise ValueError(f"File size exceeds maximum of {MAX_VIDEO_SIZE_MB}MB")
        
        # Save file
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(file_content)
        
        return filename
    
    def get_video_path(self, video_id: str) -> Path:
        """Get the path to a video file."""
        return self.storage_service.get_video_path(video_id)
