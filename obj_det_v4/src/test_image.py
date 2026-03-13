"""
Test image processing module for RBC2026 Robocon Vision System.
Features:
- YOLO-only logic matching main.py.
- Letterbox coordinate transformation for consistent visualization.
- Selective Leftmost Priority matching production strategy.
- Production-style UI (Status bar, ErrX display, Kalman visualization).
Author: Vu Duc Du + Gemini
"""

import cv2
import logging
import numpy as np
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

from vision import RobotVision
from label_smoother import LabelSmoother
from config_manager import ConfigManager
from utils import letterbox


class ImageTester:
    """
    Test detection on single images with logic aligned to main.py.
    """
    
    def __init__(self, config_name: str = "config.yaml"):
        """Initialize image tester similarly to RoboconSystem."""
        self._setup_logging()
        self.logger = logging.getLogger("RBC2026_TEST")
        
        # --- TỰ ĐỘNG XÁC ĐỊNH ĐƯỜNG DẪN CONFIG ---
        possible_paths = [Path(config_name), Path(__file__).parent.parent / config_name]
        config_path = next((str(p) for p in possible_paths if p.exists()), None)
        if not config_path: raise FileNotFoundError(f"Config file {config_name} not found.")

        self.config = ConfigManager(config_path)
        
        # --- CẤU HÌNH CHIẾN THUẬT ---
        self.target_name = self.config.get("detection.target_name", "Target").upper()
        self.force_square = self.config.get("display.force_square", True)
        self.imgsz = self.config.get("models.yolo.input_size", 512)
        self.conf_yolo = self.config.get("models.yolo.conf_threshold", 0.6)
        
        try:
            self._init_models()
            self._init_tracking()
            self.logger.info(f"--- TESTER READY | TARGET: {self.target_name} ---")
        except Exception as e:
            self.logger.error(f"Init Failed: {e}")
            raise

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    def _init_models(self):
        yolo_device = self.config.get("models.yolo.device", "GPU")
        self.vision = RobotVision(self.config.get_path("yolo_xml"), device=yolo_device)

    def _init_tracking(self):
        self.smoother = LabelSmoother(window_size=self.config.get("detection.label_smoothing.window_size", 7))

    def _reset_kalman_at_pos(self, x: float, y: float):
        """Mồi vị trí tức thì để tránh chấm tròn bay từ (0,0)."""
        self.vision._init_kalman_filter()
        self.vision.kalman.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.vision.kalman.statePre = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.smoother.history.clear()

    def _process_selective_leftmost(self, detections):
        # Ưu tiên vật thể bên trái nhất (Match main.py)
        sorted_dets = sorted(detections, key=lambda d: (d.xyxy[0][0] + d.xyxy[0][2]) / 2)

        if sorted_dets:
            det = sorted_dets[0]
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            label_raw = self.target_name
            
            # Reset Kalman for the single image test to center it on first det
            self._reset_kalman_at_pos(cx, cy)
            
            smoothed_label, _ = self.smoother.smooth("target", label_raw, det.conf)
            return (cx, cy), smoothed_label

        return None, "NONE"

    def process_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Process single image using the same logic as main.py.
        """
        try:
            frame = cv2.imread(image_path)
            if frame is None:
                self.logger.error(f"Failed to load image: {image_path}")
                return None
            
            h_f, w_f = frame.shape[:2]
            
            # --- PREPARE DISPLAY FRAME (LETTERBOX IF SQUARE) ---
            if self.force_square:
                display_frame, display_scale, (pad_w, pad_h) = letterbox(frame, (self.imgsz, self.imgsz))
                screen_center_x = self.imgsz // 2
                curr_h, curr_w = self.imgsz, self.imgsz
            else:
                display_frame = frame.copy()
                screen_center_x = w_f // 2
                curr_h, curr_w = h_f, w_f

            # --- AI LOGIC ---
            detections = self.vision.predict(frame, conf_threshold=self.conf_yolo, imgsz=self.imgsz)
            
            latest_target_point = None
            latest_label = "NONE"
            status_text = "SEARCHING"
            
            # --- DRAW ALL BOUNDING BOXES > 0.6 ---
            for det in detections:
                if det.conf >= 0.6:
                    x1, y1, x2, y2 = map(int, det.xyxy[0])
                    if self.force_square:
                        dx1 = int(x1 * display_scale + pad_w)
                        dy1 = int(y1 * display_scale + pad_h)
                        dx2 = int(x2 * display_scale + pad_w)
                        dy2 = int(y2 * display_scale + pad_h)
                    else:
                        dx1, dy1, dx2, dy2 = x1, y1, x2, y2
                    
                    cv2.rectangle(display_frame, (dx1, dy1), (dx2, dy2), (0, 255, 255), 2)
                    cv2.putText(display_frame, f"{det.conf:.2f}", (dx1, dy1 - 5), 0, 0.5, (0, 255, 255), 1)

            if detections:
                new_pt, label = self._process_selective_leftmost(detections)
                if new_pt:
                    tx, ty = self.vision.update_kalman(new_pt[0], new_pt[1])
                    
                    # Convert to Display Coordinates if using Letterbox
                    if self.force_square:
                        dtx = int(tx * display_scale + pad_w)
                        dty = int(ty * display_scale + pad_h)
                        latest_target_point = (dtx, dty)
                    else:
                        latest_target_point = (int(tx), int(ty))
                        
                    latest_label = label
                    status_text = "LOCKED"
            
            if not latest_target_point and status_text == "SEARCHING":
                 latest_target_point = (screen_center_x, curr_h // 2)
                 status_text = "LOST"

            # --- CALCULATE ERROR X ---
            error_x = int(latest_target_point[0] - screen_center_x) if latest_target_point else 0

            # --- DRAW UI (MATCH main.py) ---
            cv2.rectangle(display_frame, (0, 0), (curr_w, 40), (40, 40, 40), -1)
            cv2.putText(display_frame, f"T:{self.target_name} | {status_text} | {latest_label} | ErrX:{error_x}", (10, 27), 0, 0.6, (255, 255, 255), 1)
            
            if latest_target_point:
                color = (0, 255, 0) if status_text == "LOCKED" else (0, 0, 255)
                cv2.circle(display_frame, latest_target_point, 10, color, -1)
                cv2.line(display_frame, (screen_center_x, curr_h), latest_target_point, color, 2)
                cv2.drawMarker(display_frame, (screen_center_x, curr_h // 2), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)

            return display_frame

        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            return None


def main():
    import argparse
    from pathlib import Path
    
    # Get default image path
    base_dir = Path(__file__).resolve().parent.parent
    default_image = str(base_dir / "datasets" / "user_test" / "img_test_user1.jpg")
    
    parser = argparse.ArgumentParser(description="Test RBC2026 Detection on Image")
    parser.add_argument("image_path", type=str, nargs='?', default=default_image, help="Path to input image")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
     
    args = parser.parse_args()
    
    if not Path(args.image_path).exists():
        print(f"❌ Error: Image not found: {args.image_path}")
        sys.exit(1)
        
    try:
        tester = ImageTester(config_name=args.config)
        result = tester.process_image(args.image_path)
        
        if result is not None:
            print(f"✅ Processed: {args.image_path}")
            cv2.imshow("RBC2026_IMAGE_TEST", result)
            print("Press any key to exit...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("❌ Processing failed.")
    except Exception as e:
        print(f"💥 Fatal Error: {e}")


if __name__ == "__main__":
    main()
