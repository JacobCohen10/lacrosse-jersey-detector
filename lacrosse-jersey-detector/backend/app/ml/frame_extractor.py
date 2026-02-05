"""Extract frames from video at fixed intervals."""
import cv2
from pathlib import Path
from typing import List, Dict, Iterator
from app.config import FRAME_EXTRACTION_INTERVAL


class FrameExtractor:
    """Extracts frames from video files at fixed time intervals."""

    def __init__(self, interval: float = None):
        """
        Initialize frame extractor.

        Args:
            interval: Time interval in seconds between frames (default from config)
        """
        self.interval = interval or FRAME_EXTRACTION_INTERVAL

    def iter_frames(self, video_path: Path, interval: float = None) -> Iterator[Dict]:
        """
        Stream frames one at a time (low memory). Yields dicts with 'timestamp' and 'image'.
        Use this for long videos or limited memory (e.g. 512MB on Render).
        """
        use_interval = interval if interval is not None else self.interval
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = max(1, int(fps * use_interval))
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % frame_interval == 0:
                    timestamp = frame_count / fps
                    yield {"timestamp": timestamp, "image": frame}
                frame_count += 1
        finally:
            cap.release()

    def extract_frames(self, video_path: Path, interval: float = None) -> List[Dict]:
        """
        Extract all frames into a list (higher memory). Prefer iter_frames for long videos.
        """
        return list(self.iter_frames(video_path, interval))
