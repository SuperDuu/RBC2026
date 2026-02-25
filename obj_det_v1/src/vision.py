"""
Vision module for RBC2026 Robocon Vision System.

This module provides YOLO-based object detection and Kalman filter tracking.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from openvino.runtime import Core, CompiledModel

logger = logging.getLogger(__name__)

# Constants
DEFAULT_CLASS_ID = 0
DEFAULT_INPUT_SIZE = 512
DEFAULT_CONF_THRESHOLD = 0.5
DEFAULT_NMS_THRESHOLD = 0.45
BACKGROUND_VALUE = 128
KALMAN_PROCESS_NOISE = 0.03


class DetectedObject:
    """Represents a detected object with bounding box and confidence."""
    
    def __init__(self, x1: int, y1: int, x2: int, y2: int, conf: float):
        """
        Initialize detected object.
        
        Args:
            x1: Top-left x coordinate
            y1: Top-left y coordinate
            x2: Bottom-right x coordinate
            y2: Bottom-right y coordinate
            conf: Confidence score
        """
        self.xyxy = [np.array([x1, y1, x2, y2])]
        self.conf = conf


class RobotVision:
    """
    Robot vision system using YOLO for detection and Kalman filter for tracking.
    
    Uses OpenVINO for optimized inference.
    """
    
    def __init__(self, model_path: str, class_id: int = DEFAULT_CLASS_ID, device: str = "CPU"):
        """
        Initialize RobotVision system.
        
        Args:
            model_path: Path to YOLO model XML file or directory containing best.xml
            class_id: Class ID to detect (default: 0)
            device: Device to run inference on (default: "CPU")
        
        Raises:
            FileNotFoundError: If model file not found
            RuntimeError: If model loading fails
        """
        self.class_id = class_id
        self.device = device
        self.logger = logging.getLogger(f"{__name__}.RobotVision")
        
        # Cache for preprocessing optimization
        self._cached_frame_size = None
        self._cached_canvas = None
        self.last_boxes = {} # Lưu box cũ để làm mượt
        self.alpha = 0.7
        try:
            # Initialize OpenVINO Core
            self.ie = Core()
            
            # Resolve model path
            xml_path = model_path if model_path.endswith('.xml') else f"{model_path}/best.xml"
            xml_path_obj = Path(xml_path)
            
            if not xml_path_obj.exists():
                raise FileNotFoundError(f"YOLO model not found: {xml_path}")
            
            # Load and compile model
            self.logger.info(f"Loading YOLO model from {xml_path} on {device}")
            self.model_ir = self.ie.read_model(model=str(xml_path_obj))
            
            # Compile with performance optimization
            # compile_config = {}
            compile_config = {"PERFORMANCE_HINT": "LATENCY"}
            self.ie.set_property("GPU", {"CACHE_DIR": "model_cache"})
            # if device == "GPU":
            #     # Use throughput hint for GPU (better for batch processing)
            #     compile_config["PERFORMANCE_HINT"] = "THROUGHPUT"
            # elif device == "CPU":
            #     # Use latency hint for CPU (better for real-time)
            #     compile_config["PERFORMANCE_HINT"] = "LATENCY"
            
            self.compiled_model = self.ie.compile_model(
                model=self.model_ir, 
                device_name=device,
                config=compile_config
            )
            self.output_layer = self.compiled_model.output(0)
            
            # Initialize Kalman filter for trajectory smoothing
            self._init_kalman_filter()
            
            self.logger.info("RobotVision initialized successfully")
        
        except Exception as e:
            self.logger.error(f"Failed to initialize RobotVision: {e}")
            raise RuntimeError(f"RobotVision initialization failed: {e}") from e
    
    def _init_kalman_filter(self) -> None:
        """Initialize Kalman filter for object tracking."""
        self.kalman = cv2.KalmanFilter(4, 2)
        
        # Measurement matrix: we measure x, y position
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Transition matrix: constant velocity model
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Process noise covariance
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * KALMAN_PROCESS_NOISE
    
    def predict(
        self, 
        frame: np.ndarray, 
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        imgsz: int = DEFAULT_INPUT_SIZE
    ) -> List[DetectedObject]:
        """
        Detect objects in frame using YOLO model.
        
        Uses letterbox preprocessing to preserve aspect ratio.
        
        Args:
            frame: Input frame (BGR format)
            conf_threshold: Confidence threshold for detections
            imgsz: Input image size for YOLO (default: 512)
        
        Returns:
            List of DetectedObject instances
        
        Raises:
            ValueError: If frame is invalid
            RuntimeError: If inference fails
        """
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame: empty or None")
        
        try:
            h_orig, w_orig = frame.shape[:2]
            
            # Letterbox preprocessing: resize with aspect ratio preservation
            scale = min(imgsz / h_orig, imgsz / w_orig)
            nh, nw = int(h_orig * scale), int(w_orig * scale)
            input_img = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
            
            # Cache canvas if frame size hasn't changed (optimization)
            current_frame_size = (h_orig, w_orig, nh, nw)
            if self._cached_frame_size != current_frame_size:
                # Create new canvas with gray background
                canvas = np.full((imgsz, imgsz, 3), BACKGROUND_VALUE, dtype=np.uint8)
                self._cached_frame_size = current_frame_size
                self._cached_canvas = canvas
            else:
                # Reuse cached canvas (faster)
                canvas = self._cached_canvas.copy()
            
            y_offset = (imgsz - nh) // 2
            x_offset = (imgsz - nw) // 2
            canvas[y_offset:y_offset+nh, x_offset:x_offset+nw] = input_img
            
            # Preprocess for OpenVINO: (H, W, C) -> (1, C, H, W), normalize to [0, 1]
            input_data = canvas.transpose((2, 0, 1)).reshape((1, 3, imgsz, imgsz)).astype(np.float32) / 255.0
            
            # Run inference
            results = self.compiled_model([input_data])[self.output_layer]
            predictions = np.squeeze(results)
            
            # Handle different output shapes
            if predictions.shape[0] < predictions.shape[1]:
                predictions = predictions.T
            
            # Parse predictions (vectorized for better performance)
            pad_x, pad_y = (imgsz - nw) // 2, (imgsz - nh) // 2
            
            # Vectorized operations for better performance
            if len(predictions) > 0:
                # Extract scores and find max scores and class IDs
                scores_all = predictions[:, 4:]  # All scores
                max_scores = np.max(scores_all, axis=1)
                cls_ids = np.argmax(scores_all, axis=1)
                
                # Filter by confidence and class ID
                valid_mask = (max_scores > conf_threshold) & (cls_ids == self.class_id)
                valid_preds = predictions[valid_mask]
                valid_confidences = max_scores[valid_mask]
                
                if len(valid_preds) > 0:
                    # Extract coordinates
                    xc, yc, w, h = valid_preds[:, 0], valid_preds[:, 1], valid_preds[:, 2], valid_preds[:, 3]
                    
                    # Convert to pixel coordinates (vectorized)
                    x1 = ((xc - w/2 - pad_x) / scale).astype(np.int32)
                    y1 = ((yc - h/2 - pad_y) / scale).astype(np.int32)
                    width = (w / scale).astype(np.int32)
                    height = (h / scale).astype(np.int32)
                    
                    # Prepare for NMS
                    raw_boxes = np.column_stack([x1, y1, width, height]).tolist()
                    confidences = valid_confidences.tolist()
                else:
                    raw_boxes, confidences = [], []
            else:
                raw_boxes, confidences = [], []
            
            # Apply Non-Maximum Suppression
            final_boxes = []
            if len(raw_boxes) > 0:
                indices = cv2.dnn.NMSBoxes(raw_boxes, confidences, conf_threshold, DEFAULT_NMS_THRESHOLD)
                
                # Dùng một set để theo dõi các ID đã được cập nhật trong frame này
                current_frame_ids = set()

                if len(indices) > 0:
                    for i in indices.flatten():
                        rx, ry, rw, rh = raw_boxes[i]
                        rx2, ry2 = rx + rw, ry + rh
                        conf = confidences[i]

                        # --- 4. EMA Smoothing (Làm mượt Jitter) ---
                        cx, cy = (rx + rx2) // 2, (ry + ry2) // 2
                        # Tạo ID dựa trên lưới (grid) để nhận diện cùng một vật thể qua các frame
                        grid_id = f"{cx//30}_{cy//30}" 
                        
                        if grid_id in self.last_boxes:
                            prev_box = self.last_boxes[grid_id]
                            # Công thức EMA: New = Alpha * Current + (1 - Alpha) * Previous
                            # Alpha = 0.7 giúp cân bằng giữa độ mượt và độ trễ
                            fx1 = int(self.alpha * rx + (1 - self.alpha) * prev_box[0])
                            fy1 = int(self.alpha * ry + (1 - self.alpha) * prev_box[1])
                            fx2 = int(self.alpha * rx2 + (1 - self.alpha) * prev_box[2])
                            fy2 = int(self.alpha * ry2 + (1 - self.alpha) * prev_box[3])
                        else:
                            fx1, fy1, fx2, fy2 = rx, ry, rx2, ry2
                        
                        self.last_boxes[grid_id] = [fx1, fy1, fx2, fy2]
                        current_frame_ids.add(grid_id)
                        final_boxes.append(DetectedObject(fx1, fy1, fx2, fy2, conf))

                # Dọn dẹp các box cũ không còn xuất hiện để tránh rò rỉ bộ nhớ
                self.last_boxes = {k: v for k, v in self.last_boxes.items() if k in current_frame_ids}
                
                return final_boxes
            
            return []
        
        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            raise RuntimeError(f"Prediction failed: {e}") from e
    
    # def update_kalman(self, x: float, y: float) -> Tuple[int, int]:
    #     """
    #     Update Kalman filter with new measurement and return predicted position.
        
    #     Args:
    #         x: Measured x coordinate
    #         y: Measured y coordinate
        
    #     Returns:
    #         Tuple of (predicted_x, predicted_y) as integers
    #     """
    #     try:
    #         measured = np.array([[np.float32(x)], [np.float32(y)]])
    #         self.kalman.correct(measured)
    #         pred = self.kalman.predict()
            
    #         # Extract predicted position (first 2 elements of state vector)
    #         px, py = pred.flatten()[:2]
    #         return int(px), int(py)
        
    #     except Exception as e:
    #         self.logger.error(f"Error updating Kalman filter: {e}")
    #         # Return measured values if Kalman fails
    #         return int(x), int(y)
    def update_kalman(self, x: Optional[float] = None, y: Optional[float] = None) -> Tuple[int, int]:
        """
        Bản nâng cấp: Tách biệt Predict và Correct để trị dứt điểm lỗi nháy Box.
        """
        try:
            if x is not None and y is not None:
                # PHA CORRECT: Chỉ chạy khi YOLO có kết quả thực
                measured = np.array([[np.float32(x)], [np.float32(y)]])
                self.kalman.correct(measured)
            
            # PHA PREDICT: Luôn chạy ở mọi frame để giữ quỹ đạo mượt
            pred = self.kalman.predict()
            px, py = pred.flatten()[:2]
            
            return int(px), int(py)
        
        except Exception as e:
            self.logger.error(f"Error updating Kalman filter: {e}")
            return int(x) if x else 0, int(y) if y else 0
