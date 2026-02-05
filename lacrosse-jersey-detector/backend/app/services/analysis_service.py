"""Service for orchestrating video analysis."""
from datetime import datetime
from app.api.models import AnalysisResult, TimestampInterval
from app.services.storage_service import StorageService
from app.ml.frame_extractor import FrameExtractor
from app.ml.player_detector import PlayerDetector
from app.ml.jersey_cropper import JerseyCropper
from app.ml.ocr_engine import OCREngine
from app.ml.timestamp_aggregator import TimestampAggregator
from app.config import (
    TEMPORAL_CONSENSUS_WINDOW, TEMPORAL_CONSENSUS_MIN_DETECTIONS,
    OCR_FRAME_MIN_CONFIDENCE, OCR_EARLY_EXIT_CONFIDENCE,
    FAST_MODE_FRAME_INTERVAL, FAST_MODE_YOLO_CONF, FAST_MODE_YOLO_IMGSZ,
)


class AnalysisService:
    """Orchestrates the video analysis pipeline.
    Heavy ML components (PlayerDetector, OCREngine) are lazy-loaded so the app can bind to PORT quickly on Render.
    """
    
    def __init__(self):
        self.storage_service = StorageService()
        self.frame_extractor = FrameExtractor()
        self._player_detector = None
        self.jersey_cropper = JerseyCropper()
        self._ocr_engine = None
        self.timestamp_aggregator = TimestampAggregator()
    
    @property
    def player_detector(self):
        if self._player_detector is None:
            self._player_detector = PlayerDetector()
        return self._player_detector
    
    @property
    def ocr_engine(self):
        if self._ocr_engine is None:
            self._ocr_engine = OCREngine()
        return self._ocr_engine
    
    def _normalize_number(self, number: str) -> str:
        """Normalize jersey number for comparison (remove leading zeros, whitespace)."""
        return number.strip().lstrip('0') or '0'
    
    def _get_multiple_crop_regions(self, frame_image, box, fast_mode: bool = False):
        """
        Get multiple crop regions for a player to handle different angles.
        When fast_mode=True, returns only the default crop (1 region).
        
        Returns:
            List of cropped jersey regions
        """
        crops = []
        x1, y1, x2, y2 = box
        box_height = y2 - y1
        
        # Small players: one crop only
        if box_height < 35:
            crop = self.jersey_cropper.crop_jersey_region(frame_image, box)
            if crop is not None:
                crops.append(crop)
            return crops
        
        # Strategy 1: Default torso (wider vertical 0.05-0.5)
        crop1 = self.jersey_cropper.crop_jersey_region(frame_image, box)
        if crop1 is not None:
            crops.append(crop1)
        
        if fast_mode:
            return crops
        
        # Strategy 2: Wider horizontal (side angles)
        wider_cropper = JerseyCropper(jersey_region_ratio=(0.15, 0.05, 0.85, 0.5))
        crop2 = wider_cropper.crop_jersey_region(frame_image, box)
        if crop2 is not None:
            crops.append(crop2)
        
        # Strategy 3: Slightly lower (bent posture / number lower on chest)
        lower_cropper = JerseyCropper(jersey_region_ratio=(0.2, 0.12, 0.8, 0.55))
        crop3 = lower_cropper.crop_jersey_region(frame_image, box)
        if crop3 is not None:
            crops.append(crop3)
        
        return crops
    
    def analyze_video(self, job_id: str, video_id: str, jersey_number: str, fast_mode: bool = False):
        """
        Analyze video for jersey number detection with ensemble predictions and confidence filtering.
        When fast_mode=True: fewer frames (0.5s interval), higher YOLO conf, 1 crop/player, 2 OCR strategies.
        
        Pipeline:
        1. Extract frames
        2. Detect players
        3. Crop jersey regions (1 or 3 per player)
        4. Run OCR (2 or 7 strategies per crop)
        5. Filter low-confidence frames
        6. Aggregate timestamps
        """
        # Initialize result
        result = AnalysisResult(
            job_id=job_id,
            status="processing",
            jersey_number=jersey_number,
            video_id=video_id,
            created_at=datetime.now()
        )
        self.storage_service.save_analysis_result(result)
        
        try:
            # Get video path
            video_path = self.storage_service.get_video_path(video_id)
            
            # Normalize target jersey number
            target_number = self._normalize_number(jersey_number)
            
            # Step 1: Stream frames one at a time (low memory; no full frame list in RAM)
            extract_interval = FAST_MODE_FRAME_INTERVAL if fast_mode else None
            if fast_mode:
                print(f"[{job_id}] Fast mode: streaming frames (interval={FAST_MODE_FRAME_INTERVAL}s)...")
            else:
                print(f"[{job_id}] Streaming frames from video...")

            # Step 2-5: Process each frame as it is yielded (only one frame in memory at a time)
            frame_detections = []  # List of (frame_idx, timestamp, confidence) for detections
            total_players = 0
            total_ocr_attempts = 0
            detected_numbers = {}  # Track what numbers we're seeing
            rejected_frames = 0  # Track rejected low-confidence frames
            frame_count = 0

            for frame_idx, frame_data in enumerate(self.frame_extractor.iter_frames(video_path, interval=extract_interval)):
                frame_count = frame_idx + 1
                timestamp = frame_data["timestamp"]
                frame_image = frame_data["image"]

                # Detect players (fast mode: higher conf, smaller imgsz)
                if fast_mode:
                    player_boxes = self.player_detector.detect_players(
                        frame_image,
                        confidence_threshold=FAST_MODE_YOLO_CONF,
                        imgsz=FAST_MODE_YOLO_IMGSZ,
                    )
                else:
                    player_boxes = self.player_detector.detect_players(frame_image)
                total_players += len(player_boxes)

                # Collect all detections for this frame across all players and crops
                frame_target_detections = []  # List of (number, confidence) for target number
                frame_all_detections = []  # All detections for debugging
                found_high_confidence = False  # Early exit flag

                # For each detected player, try multiple crop regions and OCR strategies
                for box in player_boxes:
                    if found_high_confidence:
                        break
                    crop_regions = self._get_multiple_crop_regions(frame_image, box, fast_mode=fast_mode)
                    for jersey_crop in crop_regions:
                        total_ocr_attempts += 1
                        detected_number, confidence = self.ocr_engine.read_jersey_number(
                            jersey_crop, fast_mode=fast_mode
                        )
                        if detected_number:
                            normalized = self._normalize_number(detected_number)
                            detected_numbers[normalized] = detected_numbers.get(normalized, 0) + 1
                            frame_all_detections.append((normalized, confidence))
                            if normalized == target_number:
                                frame_target_detections.append((normalized, confidence))
                                if confidence >= OCR_EARLY_EXIT_CONFIDENCE:
                                    found_high_confidence = True
                                    break
                    if found_high_confidence:
                        break

                # Frame-level ensemble: compute frame confidence for target number
                if frame_target_detections:
                    confidences = [conf for _, conf in frame_target_detections]
                    frame_confidence = max(confidences)
                    if frame_confidence >= OCR_FRAME_MIN_CONFIDENCE:
                        frame_detections.append((frame_idx, timestamp, frame_confidence))
                    else:
                        rejected_frames += 1

                # Progress update every 10 frames
                if frame_count % 10 == 0:
                    print(f"[{job_id}] Processed {frame_count} frames, found {len(frame_detections)} matches, rejected {rejected_frames} low-confidence")

            print(f"[{job_id}] Finished streaming. Total frames: {frame_count}")

            # Apply temporal consensus: use sliding window approach (more flexible for movement)
            detections = []
            consensus_window = TEMPORAL_CONSENSUS_WINDOW
            min_detections_in_window = TEMPORAL_CONSENSUS_MIN_DETECTIONS
            processed_indices = set()

            for i in range(len(frame_detections)):
                if i in processed_indices:
                    continue
                frame_idx, timestamp, confidence = frame_detections[i]
                window_detections = [timestamp]
                window_confidences = [confidence]
                window_indices = [i]
                for j in range(i + 1, len(frame_detections)):
                    next_frame_idx, next_timestamp, next_confidence = frame_detections[j]
                    time_diff = next_timestamp - timestamp
                    if time_diff <= consensus_window:
                        window_detections.append(next_timestamp)
                        window_confidences.append(next_confidence)
                        window_indices.append(j)
                    else:
                        break
                if len(window_detections) >= min_detections_in_window:
                    detections.extend(window_detections)
                    for idx in window_indices:
                        processed_indices.add(idx)

            # Debug output
            avg_players_per_frame = total_players / frame_count if frame_count else 0
            print(f"[{job_id}] Detection summary:")
            print(f"  - Frames: {frame_count}, avg players/frame: {avg_players_per_frame:.1f}")
            print(f"  - Total players detected: {total_players}")
            print(f"  - OCR attempts: {total_ocr_attempts}")
            print(f"  - Target number: {target_number}")
            print(f"  - Numbers detected: {dict(sorted(detected_numbers.items(), key=lambda x: x[1], reverse=True)[:10])}")
            print(f"  - Frame detections (before filtering): {len(frame_detections) + rejected_frames}")
            print(f"  - Rejected low-confidence frames: {rejected_frames}")
            print(f"  - Final matches (after temporal consensus): {len(detections)}")
            
            # Step 6: Aggregate timestamps into intervals
            print(f"[{job_id}] Aggregating timestamps...")
            intervals = self.timestamp_aggregator.aggregate(detections)
            
            # Update result
            result.status = "completed"
            result.intervals = intervals
            result.total_detections = len(detections)
            result.completed_at = datetime.now()
            
            self.storage_service.save_analysis_result(result)
            print(f"[{job_id}] Analysis completed. Found {len(intervals)} intervals.")
            
        except Exception as e:
            print(f"[{job_id}] Analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()
            self.storage_service.update_analysis_status(
                job_id,
                status="failed",
                error=str(e)
            )
