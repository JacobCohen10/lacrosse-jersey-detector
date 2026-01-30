"""OCR engine for reading jersey numbers from cropped images."""
import numpy as np
from typing import Optional, List, Tuple, Any
import cv2
from app.config import (
    OCR_ENGINE, MIN_STRATEGY_CONFIDENCE, OCR_MIN_RESOLUTION,
    ENSEMBLE_MIN_VOTES, ENSEMBLE_HIGH_CONFIDENCE_THRESHOLD,
    OCR_EARLY_EXIT_CONFIDENCE, USE_ROTATION_STRATEGIES,
    MERGE_DIGITS_MIN_CONFIDENCE,
)

# Optional: EasyOCR (lighter) and PaddleOCR (strongest for low-quality/small text)
try:
    import easyocr
except ImportError:
    easyocr = None
try:
    from paddleocr import PaddleOCR
    import torch
except ImportError:
    PaddleOCR = None
    torch = None


class OCREngine:
    """Reads jersey numbers from cropped jersey region images."""
    
    def __init__(self):
        """Initialize OCR engine (lazy initialization)."""
        self.engine_type = OCR_ENGINE
        self.reader = None  # Will be initialized on first use
    
    def _ensure_reader_initialized(self):
        """Initialize OCR reader on first use (lazy loading). PaddleOCR = strongest for small/low-quality numbers."""
        if self.reader is not None:
            return
        if self.engine_type == "paddleocr":
            if PaddleOCR is None:
                raise ImportError(
                    "PaddleOCR is not installed. Install with: pip install paddlepaddle paddleocr. "
                    "Or set OCR_ENGINE=easyocr to use EasyOCR."
                )
            use_gpu = False
            try:
                if torch is not None:
                    use_gpu = torch.cuda.is_available()
            except Exception:
                use_gpu = False
            print(
                f"Initializing PaddleOCR (strongest for small numbers) on "
                f"{'GPU' if use_gpu else 'CPU'} (first run may download models)..."
            )
            self.reader = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                use_gpu=use_gpu,
                show_log=False,
            )
            return
        if self.engine_type == "easyocr":
            if easyocr is None:
                raise ImportError("EasyOCR is not installed. pip install easyocr")
            use_gpu = False
            try:
                if torch is not None:
                    use_gpu = torch.cuda.is_available()
            except Exception:
                use_gpu = False
            print(
                f"Initializing EasyOCR on {'GPU' if use_gpu else 'CPU'} "
                f"(this may take a few minutes on first run)..."
            )
            self.reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
            return
        raise ValueError(f"Unsupported OCR engine: {self.engine_type}. Use 'paddleocr' or 'easyocr'.")
    
    def _is_valid_jersey_number(self, number: str) -> bool:
        """
        Validate if detected number is a valid jersey number (0-99 only).
        Never split numbers; never accept individual digits as separate numbers.
        """
        if not number or len(number) == 0:
            return False
        # Jersey numbers: 0-99 only (1-2 digits)
        if len(number) > 2:
            return False
        try:
            num_int = int(number)
            if num_int < 0 or num_int > 99:
                return False
        except ValueError:
            return False
        return True
    
    def _merge_adjacent_digits(
        self,
        results: List[Tuple],
        crop_height: int,
        crop_width: int,
    ) -> List[Tuple[str, float]]:
        """
        Merge adjacent digit detections into single jersey numbers (never split).
        EasyOCR may return "4" and "2" separately; merge into "42" when on same line and close.
        Returns list of (merged_number_str, confidence).
        """
        # Build (center_x, center_y, digits, confidence) for each result
        # Use MERGE_DIGITS_MIN_CONFIDENCE (lower than MIN_STRATEGY) so weak digits can combine
        items = []
        for (bbox, text, confidence) in results:
            digits = "".join(c for c in text if c.isdigit())
            if not digits or confidence < MERGE_DIGITS_MIN_CONFIDENCE:
                continue
            bbox_arr = np.array(bbox)
            cx = float(np.mean(bbox_arr[:, 0]))
            cy = float(np.mean(bbox_arr[:, 1]))
            items.append((cx, cy, digits, confidence))
        if not items:
            return []
        # Sort by row (y) then column (x) so we read left-to-right, top-to-bottom
        items.sort(key=lambda x: (round(x[1] / max(crop_height * 0.15, 1)), x[0]))
        merged = []
        y_tol = max(crop_height * 0.25, 8)
        x_gap_max = max(crop_width * 0.35, 20)
        i = 0
        while i < len(items):
            cx, cy, digits, conf = items[i]
            group = [(cx, cy, digits, conf)]
            j = i + 1
            while j < len(items):
                nx, ny, nd, nc = items[j]
                if abs(ny - cy) > y_tol:
                    break
                if nx - group[-1][0] > x_gap_max:
                    break
                group.append((nx, ny, nd, nc))
                j += 1
            # Merge digits left-to-right (group already sorted by x within line)
            group.sort(key=lambda x: x[0])
            combined = "".join(g[2] for g in group)
            avg_conf = sum(g[3] for g in group) / len(group)
            if combined and self._is_valid_jersey_number(combined):
                merged.append((combined, avg_conf))
            i = j
        return merged
    
    def _normalize_resolution(self, image: np.ndarray, min_size: int = None) -> np.ndarray:
        """
        Normalize image resolution - upscale if too small for OCR.
        
        Args:
            image: Input image as numpy array
            min_size: Minimum size for smallest dimension (default from config)
            
        Returns:
            Resized image
        """
        if min_size is None:
            min_size = OCR_MIN_RESOLUTION
        
        height, width = image.shape[:2]
        if height < min_size or width < min_size:
            scale = max(min_size / height, min_size / width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            # Use CUBIC for faster upscaling (good balance)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        return image
    
    def _preprocess_image_aggressive(self, image: np.ndarray) -> np.ndarray:
        """
        Aggressive preprocessing for blurry/poor quality images.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        # Normalize resolution first
        image = self._normalize_resolution(image)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply unsharp masking to sharpen the image (helps with blur)
        gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
        sharpened = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
        
        # Enhance contrast using CLAHE (very aggressive for poor quality)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(sharpened)
        
        # Apply bilateral filter to reduce noise while keeping edges sharp
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Convert back to BGR for EasyOCR
        if len(denoised.shape) == 2:
            denoised = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        
        return denoised
    
    def _preprocess_image_threshold(self, image: np.ndarray) -> np.ndarray:
        """
        Alternative preprocessing using thresholding.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        # Normalize resolution first
        image = self._normalize_resolution(image)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Convert back to BGR
        if len(thresh.shape) == 2:
            thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        
        return thresh
    
    def _preprocess_image_brightness(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocessing with brightness adjustment for different lighting conditions.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        # Normalize resolution first
        image = self._normalize_resolution(image)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Adjust brightness (increase for dark images)
        mean_brightness = np.mean(gray)
        if mean_brightness < 100:  # Dark image
            brightness_factor = 1.3
            brightened = cv2.convertScaleAbs(gray, alpha=1, beta=int(30 * brightness_factor))
        else:
            brightened = gray.copy()
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(brightened)
        
        # Apply unsharp masking
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 1.5)
        sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        
        # Convert back to BGR
        if len(sharpened.shape) == 2:
            sharpened = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
        
        return sharpened
    
    def _preprocess_image_motion_blur(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocessing specifically for motion blur reduction.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        # Normalize resolution first
        image = self._normalize_resolution(image)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply deconvolution-based sharpening (Wiener filter approximation)
        # Use Laplacian for edge enhancement
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpened = gray.astype(np.float64) - 0.3 * laplacian
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        # Apply strong contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(sharpened)
        
        # Apply morphological operations to enhance text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        
        # Convert back to BGR
        if len(enhanced.shape) == 2:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        return enhanced
    
    def _preprocess_image_partial_occlusion(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocessing optimized for partial occlusion (smaller CLAHE tiles for local enhancement).
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        # Normalize resolution first
        image = self._normalize_resolution(image)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Use smaller tile size for CLAHE to handle partial occlusion better
        # Smaller tiles = more local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)
        
        # Apply unsharp masking
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 1.5)
        sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        
        # Convert back to BGR
        if len(sharpened.shape) == 2:
            sharpened = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
        
        return sharpened
    
    def _preprocess_image_rotation(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Preprocessing with rotation for curved fabric (numbers on curved jerseys).
        
        Args:
            image: Input image as numpy array
            angle: Rotation angle in degrees
            
        Returns:
            Rotated and preprocessed image
        """
        # Normalize resolution first
        image = self._normalize_resolution(image)
        
        # Get image center
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Rotate image
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        # Convert to grayscale if needed
        if len(rotated.shape) == 3:
            gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        else:
            gray = rotated.copy()
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Convert back to BGR
        if len(enhanced.shape) == 2:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        return enhanced
    
    def _preprocess_image(self, image: np.ndarray, fast_mode: bool = False) -> List[Tuple[np.ndarray, str]]:
        """
        Preprocess image with full strategies for strongest small-number detection.
        Motion-blur first, then aggressive/threshold/brightness/occlusion, then rotation.
        When fast_mode=True, only motion_blur and aggressive (2 strategies).
        """
        if fast_mode:
            return [
                (self._preprocess_image_motion_blur(image), "motion_blur"),
                (self._preprocess_image_aggressive(image), "aggressive"),
            ]
        strategies = [
            (self._preprocess_image_motion_blur(image), "motion_blur"),
            (self._preprocess_image_aggressive(image), "aggressive"),
            (self._preprocess_image_threshold(image), "threshold"),
            (self._preprocess_image_brightness(image), "brightness"),
            (self._preprocess_image_partial_occlusion(image), "partial_occlusion"),
        ]
        if USE_ROTATION_STRATEGIES:
            strategies.extend([
                (self._preprocess_image_rotation(image, 5.0), "rotation_5deg"),
                (self._preprocess_image_rotation(image, -5.0), "rotation_-5deg"),
            ])
        return strategies
    
    def read_jersey_number(self, jersey_crop: np.ndarray, fast_mode: bool = False) -> Tuple[Optional[str], float]:
        """
        Read jersey number from cropped jersey region with ensemble predictions.
        Returns both the number and ensemble confidence score.
        Includes early exit optimization for high-confidence detections.
        
        Args:
            jersey_crop: Cropped jersey region image as numpy array
            
        Returns:
            Tuple of (detected_number, ensemble_confidence)
            Returns (None, 0.0) if no valid detection
        """
        try:
            self._ensure_reader_initialized()
            
            # Check if crop is too small (likely partial occlusion or low resolution)
            height, width = jersey_crop.shape[:2]
            if height < 30 or width < 30:
                return None, 0.0
            
            # Try multiple preprocessing strategies with early exit
            preprocessed_images = self._preprocess_image(jersey_crop, fast_mode=fast_mode)
            
            all_detections = []  # Collect all detections from all strategies
            high_confidence_detections = []  # Track very high confidence detections
            fast_mode_early_exit_conf = 0.85  # Return after first strategy if conf >= this in fast mode
            
            for processed_image, strategy_name in preprocessed_images:
                h, w = processed_image.shape[:2]
                if self.engine_type == "paddleocr":
                    # PaddleOCR 3.x: rec_only=True for recognition on crop (no detection)
                    try:
                        img_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                        raw = self.reader.ocr(img_rgb, rec_only=True)
                    except TypeError:
                        try:
                            raw = self.reader.ocr(processed_image, rec_only=True)
                        except TypeError:
                            raw = self.reader.ocr(processed_image, rec=True, det=False)
                    # Parse: Paddle returns [[(box, (text, conf)), ...]] or similar
                    paddle_results: List[Tuple[str, float]] = []
                    def _extract_text_conf(item: Any) -> Optional[Tuple[str, float]]:
                        if not isinstance(item, (list, tuple)) or len(item) < 2:
                            return None
                        # item can be (box, (text, conf)) or (text, conf)
                        part = item[1] if isinstance(item[1], (list, tuple)) else item
                        if not isinstance(part, (list, tuple)) or len(part) < 2:
                            return None
                        text, conf = str(part[0]), float(part[1])
                        digits = "".join(c for c in text if c.isdigit())
                        if digits and conf >= MIN_STRATEGY_CONFIDENCE and self._is_valid_jersey_number(digits):
                            return (digits, conf)
                        return None
                    if raw and isinstance(raw, list):
                        for line in raw:
                            if isinstance(line, (list, tuple)):
                                for item in line:
                                    res = _extract_text_conf(item)
                                    if res:
                                        paddle_results.append(res)
                            else:
                                res = _extract_text_conf(line)
                                if res:
                                    paddle_results.append(res)
                    for (number, conf) in paddle_results:
                        all_detections.append((number, conf, strategy_name))
                        if conf >= OCR_EARLY_EXIT_CONFIDENCE:
                            high_confidence_detections.append((number, conf))
                        if fast_mode and conf >= fast_mode_early_exit_conf:
                            return number, conf
                else:
                    # EasyOCR: readtext returns (bbox, text, conf); merge adjacent digits
                    results = self.reader.readtext(processed_image)
                    merged_list = self._merge_adjacent_digits(results, h, w)
                    for (number, conf) in merged_list:
                        all_detections.append((number, conf, strategy_name))
                        if conf >= OCR_EARLY_EXIT_CONFIDENCE:
                            high_confidence_detections.append((number, conf))
                        if fast_mode and conf >= fast_mode_early_exit_conf:
                            return number, conf
            
            # Early exit: if we have very high confidence detection, return immediately
            if high_confidence_detections:
                # Use the highest confidence detection
                best_detection = max(high_confidence_detections, key=lambda x: x[1])
                return best_detection[0], best_detection[1]
            
            if not all_detections:
                return None, 0.0
            
            # Ensemble voting: count how many times each number appears
            number_votes = {}
            number_confidence_sum = {}
            number_strategies = {}  # Track which strategies detected each number
            
            for number, confidence, strategy in all_detections:
                if number not in number_votes:
                    number_votes[number] = 0
                    number_confidence_sum[number] = 0.0
                    number_strategies[number] = set()
                number_votes[number] += 1
                number_confidence_sum[number] += confidence
                number_strategies[number].add(strategy)
            
            # Calculate average confidence for each number
            number_avg_confidence = {
                num: number_confidence_sum[num] / number_votes[num]
                for num in number_votes
            }
            
            # Ensemble decision: require multiple votes OR very high confidence
            valid_numbers = []
            for num in number_votes:
                votes = number_votes[num]
                avg_conf = number_avg_confidence[num]
                
                # Accept if:
                # 1. Multiple strategies agree (ENSEMBLE_MIN_VOTES or more)
                # 2. OR single strategy with very high confidence
                if votes >= ENSEMBLE_MIN_VOTES or avg_conf >= ENSEMBLE_HIGH_CONFIDENCE_THRESHOLD:
                    valid_numbers.append(num)
            
            if not valid_numbers:
                return None, 0.0
            
            # Among valid numbers, pick the one with highest ensemble score
            # Ensemble score = (average_confidence * 0.7) + (vote_count_normalized * 0.3)
            max_votes = max(number_votes.values())
            best_number = max(
                valid_numbers,
                key=lambda n: (
                    number_avg_confidence[n] * 0.7 + 
                    (number_votes[n] / max_votes) * 0.3
                )
            )
            
            # Return number and ensemble confidence
            ensemble_confidence = number_avg_confidence[best_number]
            return best_number, ensemble_confidence
                
        except Exception as e:
            print(f"OCR error: {e}")
            return None, 0.0
