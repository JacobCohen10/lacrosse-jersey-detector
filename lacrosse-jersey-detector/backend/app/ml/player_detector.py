"""Detect players in video frames using YOLOv8."""
import numpy as np
from typing import List, Tuple
from ultralytics import YOLO
import torch
from app.config import YOLO_MODEL_PATH, MODELS_DIR, YOLO_CONFIDENCE_THRESHOLD, MIN_PLAYER_HEIGHT, YOLO_IMGSZ


class PlayerDetector:
    """Detects players (persons) in video frames using YOLOv8."""
    
    def __init__(self):
        """Initialize YOLOv8 model for person detection."""
        model_path = MODELS_DIR / YOLO_MODEL_PATH
        # If model doesn't exist locally, YOLO will download it
        if not model_path.exists():
            self.model = YOLO(YOLO_MODEL_PATH)  # Will download if needed
        else:
            self.model = YOLO(str(model_path))

        # Select best available device: CUDA GPU > MPS (Apple) > CPU
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        try:
            self.model.to(self.device)
            print(f"YOLO player detector using device: {self.device}")
        except Exception as e:
            # Fallback to CPU if moving to device fails
            print(f"Failed to move YOLO model to {self.device}, falling back to CPU: {e}")
            self.device = "cpu"

        self.confidence_threshold = YOLO_CONFIDENCE_THRESHOLD
        self.min_player_height = MIN_PLAYER_HEIGHT
        self.imgsz = YOLO_IMGSZ  # Larger = better small-person detection (slower)
    
    def detect_players(
        self,
        frame: np.ndarray,
        confidence_threshold: float = None,
        imgsz: int = None,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect players (persons) in a frame with confidence and size filtering.
        
        Args:
            frame: Image frame as numpy array (BGR format from OpenCV)
            confidence_threshold: Override conf threshold (e.g. for fast mode). None = use self.confidence_threshold.
            imgsz: Override input size (e.g. 640 for fast mode). None = use self.imgsz.
            
        Returns:
            List of bounding boxes as (x1, y1, x2, y2) tuples
        """
        conf = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        sz = imgsz if imgsz is not None else self.imgsz
        results = self.model(
            frame,
            classes=[0],
            conf=conf,
            verbose=False,
            device=self.device,
            imgsz=sz,
        )
        
        boxes = []
        for result in results:
            for box in result.boxes:
                # Get confidence score
                confidence = float(box.conf[0].cpu().numpy())
                
                # Filter by confidence threshold
                if confidence < conf:
                    continue
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                box_height = y2 - y1
                
                # Filter by minimum player height (reduces false positives from tiny detections)
                if box_height < self.min_player_height:
                    continue
                
                boxes.append((int(x1), int(y1), int(x2), int(y2)))
        
        return boxes
