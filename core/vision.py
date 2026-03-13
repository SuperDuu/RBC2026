"""
Unified Vision module for RBC2026.
Provides YOLO detection and advanced Kalman/EMA tracking.
"""

import cv2
import numpy as np
import logging
import time
from pathlib import Path
from typing import List, Tuple, Optional
from openvino.runtime import Core
from .utils import letterbox

logger = logging.getLogger(__name__)

# Constants
DEFAULT_CLASS_ID = 0
DEFAULT_INPUT_SIZE = 512
DEFAULT_CONF_THRESHOLD = 0.5
DEFAULT_NMS_THRESHOLD = 0.45
BACKGROUND_VALUE = 128
KALMAN_PROCESS_NOISE = 0.03

class DetectedObject:
    def __init__(self, x1: int, y1: int, x2: int, y2: int, conf: float):
        self.xyxy = [np.array([x1, y1, x2, y2])]
        self.conf = conf

class RobotVision:
    def __init__(self, model_path: str, class_id: int = DEFAULT_CLASS_ID, device: str = "CPU", ie_core: Optional[Core] = None):
        self.class_id = class_id
        self.device = device
        self.logger = logging.getLogger(f"{__name__}.RobotVision")
        self.last_boxes = {} 
        self.alpha = 0.7
        try:
            self.ie = ie_core if ie_core is not None else Core()
            xml_path = model_path if model_path.endswith('.xml') else f"{model_path}/best.xml"
            if not Path(xml_path).exists():
                raise FileNotFoundError(f"YOLO model not found: {xml_path}")
            
            self.model_ir = self.ie.read_model(model=xml_path)
            self.compiled_model = self.ie.compile_model(model=self.model_ir, device_name=device, config={"PERFORMANCE_HINT": "LATENCY"})
            self.output_layer = self.compiled_model.output(0)
            self._init_kalman_filter()
        except Exception as e:
            self.logger.error(f"Failed to initialize RobotVision: {e}")
            raise

    def _init_kalman_filter(self) -> None:
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * KALMAN_PROCESS_NOISE

    def predict(self, frame: np.ndarray, conf_threshold: float = DEFAULT_CONF_THRESHOLD, imgsz: int = DEFAULT_INPUT_SIZE) -> List[DetectedObject]:
        if frame is None: return []
        h_orig, w_orig = frame.shape[:2]
        canvas, scale, (pad_x, pad_y) = letterbox(frame, (imgsz, imgsz))
        nw, nh = int(round(w_orig * scale)), int(round(h_orig * scale))
        input_data = canvas.transpose((2, 0, 1)).reshape((1, 3, imgsz, imgsz)).astype(np.float32) / 255.0
        
        results = self.compiled_model([input_data])[self.output_layer]
        predictions = np.squeeze(results)
        if predictions.shape[0] < predictions.shape[1]: predictions = predictions.T
        
        if len(predictions) > 0:
            scores_all = predictions[:, 4:]
            max_scores = np.max(scores_all, axis=1)
            cls_ids = np.argmax(scores_all, axis=1)
            valid_mask = (max_scores > conf_threshold) & (cls_ids == self.class_id)
            valid_preds = predictions[valid_mask]
            if len(valid_preds) > 0:
                xc, yc, w, h = valid_preds[:, 0], valid_preds[:, 1], valid_preds[:, 2], valid_preds[:, 3]
                x1 = ((xc - w/2 - pad_x) / scale).astype(np.int32)
                y1 = ((yc - h/2 - pad_y) / scale).astype(np.int32)
                raw_boxes = np.column_stack([x1, y1, (w / scale).astype(np.int32), (h / scale).astype(np.int32)]).tolist()
                confidences = max_scores[valid_mask].tolist()
                indices = cv2.dnn.NMSBoxes(raw_boxes, confidences, conf_threshold, DEFAULT_NMS_THRESHOLD)
                
                final_boxes, new_last_boxes = [], {}
                if len(indices) > 0:
                    for i in indices.flatten():
                        rx, ry, rw, rh = raw_boxes[i]
                        rx2, ry2 = rx + rw, ry + rh
                        new_box = [rx, ry, rx2, ry2]
                        best_iou, best_match_id = 0.0, None
                        for prev_id, prev_box in self.last_boxes.items():
                            xA, yA, xB, yB = max(new_box[0], prev_box[0]), max(new_box[1], prev_box[1]), min(new_box[2], prev_box[2]), min(new_box[3], prev_box[3])
                            inter = max(0, xB - xA) * max(0, yB - yA)
                            areaA, areaB = (rx2-rx)*(ry2-ry), (prev_box[2]-prev_box[0])*(prev_box[3]-prev_box[1])
                            iou = inter / float(areaA + areaB - inter) if (areaA + areaB - inter) > 0 else 0
                            if iou > best_iou: best_iou, best_match_id = iou, prev_id
                        
                        if best_iou > 0.3 and best_match_id:
                            prev_box = self.last_boxes[best_match_id]
                            fx1, fy1, fx2, fy2 = [int(self.alpha * c + (1-self.alpha) * p) for c, p in zip(new_box, prev_box)]
                            new_box_id = best_match_id
                        else:
                            fx1, fy1, fx2, fy2 = rx, ry, rx2, ry2
                            new_box_id = f"{rx}_{ry}_{time.time()}"
                        
                        new_last_boxes[new_box_id] = [fx1, fy1, fx2, fy2]
                        final_boxes.append(DetectedObject(fx1, fy1, fx2, fy2, confidences[i]))
                self.last_boxes = new_last_boxes
                return final_boxes
        return []

    def update_kalman(self, x: Optional[float] = None, y: Optional[float] = None) -> Tuple[int, int]:
        try:
            if x is not None and y is not None:
                self.kalman.correct(np.array([[np.float32(x)], [np.float32(y)]]))
            pred = self.kalman.predict()
            px, py = pred.flatten()[:2]
            return int(px), int(py)
        except Exception as e:
            self.logger.error(f"Kalman error: {e}")
            return int(x) if x else 0, int(y) if y else 0
