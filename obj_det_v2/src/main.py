"""
Main application module for RBC2026 - SMOOTH & LINE RESTORED.
Features: 
- Fixed 30 FPS (Jitter-Free).
- Restored Visual Line (Đường nối mục tiêu).
- Watchdog 3-Frame Loss (Return to Center).
- Fixed Top-UI.
Author: Vu Duc Du + Gemini
"""

import cv2
import time
import json
import logging
import numpy as np
import ctypes
from pathlib import Path
from typing import Optional, Tuple, List
from openvino.runtime import Core

from vision import RobotVision
from connection import UARTManager
from camera import CameraStream
from label_smoother import LabelSmoother
from utils import preprocess_roi_for_cnn
from config_manager import ConfigManager

class RoboconSystem:
    def __init__(self, config_path: Optional[str] = None):
        try:
            self.winmm = ctypes.WinDLL('winmm')
            self.winmm.timeBeginPeriod(1)
        except: pass

        self._setup_logging()
        self.logger = logging.getLogger("RBC2026")
        
        # --- CẤU HÌNH ĐIỀU KHIỂN ---
        self.target_mode = "R1" # Chỉnh: "R1", "REAL", "FAKE"
        self.frame_idx = 0
        self.loss_counter = 0
        self.max_loss_frames = 7 
        
        self.latest_target_point = None
        self.latest_label = "NONE"
        self.status_text = "SEARCHING..."
        
        # FPS Sampling & Capping
        self.display_fps = 0.0
        self.last_fps_update_time = time.time()
        self.frame_count_since_update = 0
        self.target_fps = 30
        self.frame_duration = 1.0 / self.target_fps -0.012
        
        try:
            self.config = ConfigManager(config_path)
            self._init_models()
            self._init_hardware()
            self._init_tracking()
            self.logger.info("--- SYSTEM READY ---")
        except Exception as e:
            self.logger.error(f"Init Error: {e}")
            raise

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    def _init_models(self):
        self.ie = Core()
        cache_path = Path("model_cache")
        cache_path.mkdir(exist_ok=True)
        self.ie.set_property("GPU", {"CACHE_DIR": str(cache_path)})
        
        self.compiled_cnn = self.ie.compile_model(
            model=self.ie.read_model(self.config.get_path("cnn_xml")), device_name="GPU")
        self.cnn_output = self.compiled_cnn.output(0)
        self.labels_cnn = {int(v): k for k, v in json.load(open(self.config.get_path("labels_json"))).items()}
        
        self.vision = RobotVision(self.config.get_path("yolo_xml"), device="GPU")

    def _init_hardware(self):
        self.camera = CameraStream(src=0, buffer_size=1).start()
        self.uart = UARTManager(port="COM3", baudrate=115200)

    def _init_tracking(self):
        self.smoother = LabelSmoother(window_size=5)
        self.conf_yolo, self.conf_cnn = 0.45, 0.5
        self.grid_size = 40

    def _process_with_center_priority(self, frame, detections):
        h_f, w_f = frame.shape[:2]
        center = (w_f // 2, h_f // 2)
        candidates = []
        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            dist = np.sqrt((cx - center[0])**2 + (cy - center[1])**2)
            candidates.append({'box': (x1, y1, x2, y2), 'dist': dist, 'center': (cx, cy)})

        candidates = sorted(candidates, key=lambda x: x['dist'])[:2]
        for cand in candidates:
            x1, y1, x2, y2 = cand['box']
            roi = frame[max(0, y1):min(h_f, y2), max(0, x1):min(w_f, x2)]
            input_data = preprocess_roi_for_cnn(roi)
            if input_data is None: continue
            res = self.compiled_cnn([input_data])[self.cnn_output]
            idx = np.argmax(res[0])
            label = self.labels_cnn.get(idx, "UNK")
            if label and self.target_mode.upper() in label.upper() and res[0][idx] >= self.conf_cnn:
                smoothed, _ = self.smoother.smooth(f"{x1//self.grid_size}", label, res[0][idx])
                return cand['center'], smoothed
        return None, "NONE"

    def run(self):
        try:
            while not self.camera.stopped:
                start_time = time.time()
                frame = self.camera.read()
                if frame is None: continue
                
                h_f, w_f = frame.shape[:2]
                screen_center = (w_f // 2, h_f // 2)

                # --- AI LOGIC (SKIP 2) ---
                if self.frame_idx % 3 == 0:
                    detections = self.vision.predict(frame, conf_threshold=self.conf_yolo, imgsz=512)
                    found = False
                    if detections:
                        new_pt, label = self._process_with_center_priority(frame, detections)
                        if new_pt:
                            tx, ty = self.vision.update_kalman(new_pt[0], new_pt[1])
                            self.latest_target_point, self.latest_label = (tx, ty), label
                            self.loss_counter, self.status_text, found = 0, "LOCKED", True
                    
                    if not found:
                        self.loss_counter += 1
                        if self.loss_counter >= self.max_loss_frames:
                            self.latest_target_point, self.status_text, self.latest_label = screen_center, "NOT FOUND", "NONE"
                        else:
                            tx, ty = self.vision.update_kalman()
                            self.latest_target_point = (tx, ty)
                else:
                    if self.loss_counter < self.max_loss_frames:
                        tx, ty = self.vision.update_kalman()
                        self.latest_target_point = (tx, ty)

                # --- UI & UART ---
                # Vẽ thanh trạng thái cố định
                cv2.rectangle(frame, (0, 0), (w_f, 40), (40, 40, 40), -1)
                cv2.putText(frame, f"MODE: {self.target_mode} | {self.status_text} | {self.latest_label}", 
                            (10, 27), 0, 0.6, (255, 255, 255), 1)

                if self.latest_target_point:
                    # Gửi UART dựa trên sai số x
                    self.uart.send_error(int(self.latest_target_point[0] - screen_center[0]))
                    
                    # Màu sắc: Xanh khi Lock, Đỏ khi mất dấu
                    color = (0, 255, 0) if self.status_text == "LOCKED" else (0, 0, 255)
                    
                    # Vẽ Chấm tròn
                    cv2.circle(frame, self.latest_target_point, 10, color, -1)
                    
                    # PHỤC HỒI: Vẽ đường nối từ đáy màn hình lên mục tiêu
                    cv2.line(frame, (screen_center[0], h_f), self.latest_target_point, color, 2)
                    
                    # Vẽ tâm ngắm cố định (Crosshair) để ông căn cơ khí
                    cv2.drawMarker(frame, screen_center, (255, 255, 255), cv2.MARKER_CROSS, 20, 1)

                # FPS Sampling
                self.frame_count_since_update += 1
                if time.time() - self.last_fps_update_time >= 0.5:
                    self.display_fps = self.frame_count_since_update / (time.time() - self.last_fps_update_time)
                    self.frame_count_since_update, self.last_fps_update_time = 0, time.time()
                cv2.putText(frame, f"FPS: {self.display_fps:.1f}", (w_f - 100, 27), 0, 0.6, (0, 255, 0), 2)
                
                cv2.imshow("RBC2026_SMOOTH_FINAL", frame)
                self.frame_idx += 1
                if cv2.waitKey(1) & 0xFF == ord('q'): break

                # Hybrid Sleep (Khử giật nhịp)
                elapsed = time.time() - start_time
                if elapsed < self.frame_duration:
                    time.sleep(self.frame_duration - elapsed)
        finally:
            self.cleanup()

    def cleanup(self):
        self.camera.stop(); self.uart.stop(); cv2.destroyAllWindows()
        try: self.winmm.timeEndPeriod(1)
        except: pass

if __name__ == "__main__":
    RoboconSystem().run()