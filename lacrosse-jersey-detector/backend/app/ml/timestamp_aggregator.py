"""Aggregate consecutive timestamps into play intervals."""
from typing import List
from app.api.models import TimestampInterval
from app.config import TIMESTAMP_GROUPING_THRESHOLD, MIN_DETECTIONS_PER_INTERVAL


class TimestampAggregator:
    """Groups consecutive timestamp detections into play intervals."""
    
    def __init__(self, threshold: float = None, min_detections: int = None):
        """
        Initialize timestamp aggregator.
        
        Args:
            threshold: Maximum gap in seconds between detections to group them
                      (default from config)
            min_detections: Minimum number of detections per interval (default from config)
        """
        self.threshold = threshold or TIMESTAMP_GROUPING_THRESHOLD
        self.min_detections = min_detections or MIN_DETECTIONS_PER_INTERVAL
    
    def aggregate(self, timestamps: List[float]) -> List[TimestampInterval]:
        """
        Group consecutive timestamps into intervals, filtering sparse intervals.
        
        Args:
            timestamps: List of timestamps (in seconds) where jersey was detected
            
        Returns:
            List of TimestampInterval objects representing play intervals
        """
        if not timestamps:
            return []
        
        # Sort timestamps
        sorted_timestamps = sorted(set(timestamps))
        
        intervals = []
        current_start = sorted_timestamps[0]
        current_end = sorted_timestamps[0]
        current_detections = [sorted_timestamps[0]]
        
        for i in range(1, len(sorted_timestamps)):
            timestamp = sorted_timestamps[i]
            
            # If gap is within threshold, extend current interval
            if timestamp - current_end <= self.threshold:
                current_end = timestamp
                current_detections.append(timestamp)
            else:
                # Save current interval if it has enough detections
                if len(current_detections) >= self.min_detections:
                    intervals.append(TimestampInterval(
                        start_time=current_start,
                        end_time=current_end,
                        duration=current_end - current_start
                    ))
                
                # Start new interval
                current_start = timestamp
                current_end = timestamp
                current_detections = [timestamp]
        
        # Add final interval if it has enough detections
        if len(current_detections) >= self.min_detections:
            intervals.append(TimestampInterval(
                start_time=current_start,
                end_time=current_end,
                duration=current_end - current_start
            ))
        
        return intervals
