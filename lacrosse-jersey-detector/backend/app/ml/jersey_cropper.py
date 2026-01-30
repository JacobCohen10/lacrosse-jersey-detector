"""Crop jersey regions from player bounding boxes."""
import numpy as np
from typing import Optional, Tuple
from PIL import Image


class JerseyCropper:
    """Crops jersey regions from detected player bounding boxes."""
    
    def __init__(self, jersey_region_ratio: Tuple[float, float, float, float] = (0.15, 0.05, 0.85, 0.5)):
        """
        Initialize jersey cropper.
        
        Args:
            jersey_region_ratio: (x1, y1, x2, y2) as ratios of bounding box
                                Default: wider torso (0.05-0.5 vertical) to catch numbers that sit lower
        """
        self.jersey_region_ratio = jersey_region_ratio
    
    def crop_jersey_region(self, frame: np.ndarray, player_box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Crop the jersey region from a player bounding box.
        
        Args:
            frame: Full frame image
            player_box: Bounding box (x1, y1, x2, y2) of detected player
            
        Returns:
            Cropped jersey region as numpy array, or None if invalid
        """
        x1, y1, x2, y2 = player_box
        
        # Calculate jersey region within bounding box
        box_width = x2 - x1
        box_height = y2 - y1
        
        # Jersey region is typically in the upper-middle portion
        # Adjusted ratios for better number visibility
        jersey_x1 = int(x1 + box_width * self.jersey_region_ratio[0])
        jersey_y1 = int(y1 + box_height * self.jersey_region_ratio[1])
        jersey_x2 = int(x1 + box_width * self.jersey_region_ratio[2])
        jersey_y2 = int(y1 + box_height * self.jersey_region_ratio[3])
        
        # Ensure coordinates are within frame bounds
        frame_height, frame_width = frame.shape[:2]
        jersey_x1 = max(0, min(jersey_x1, frame_width))
        jersey_y1 = max(0, min(jersey_y1, frame_height))
        jersey_x2 = max(0, min(jersey_x2, frame_width))
        jersey_y2 = max(0, min(jersey_y2, frame_height))
        
        # Validate crop region
        if jersey_x2 <= jersey_x1 or jersey_y2 <= jersey_y1:
            return None
        
        # Crop the region
        jersey_crop = frame[jersey_y1:jersey_y2, jersey_x1:jersey_x2]
        
        # Allow smaller crops (OCR will upscale); reject only very tiny
        if jersey_crop.shape[0] < 24 or jersey_crop.shape[1] < 24:
            return None
        
        return jersey_crop