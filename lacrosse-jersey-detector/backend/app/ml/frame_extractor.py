"""Extract frames from video at fixed intervals using ffmpeg."""
import subprocess
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict
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
    
    def extract_frames(self, video_path: Path, interval: float = None) -> List[Dict]:
        """
        Extract frames from video at fixed intervals.
        
        Args:
            video_path: Path to video file
            interval: Override interval in seconds (e.g. for fast mode). None = use self.interval.
            
        Returns:
            List of dictionaries with 'timestamp' and 'image' (numpy array) keys
        """
        use_interval = interval if interval is not None else self.interval
        frames = []
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(fps * use_interval))
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frame at intervals
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                frames.append({
                    'timestamp': timestamp,
                    'image': frame
                })
            
            frame_count += 1
        
        cap.release()
        return frames
