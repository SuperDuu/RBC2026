"""
Main application module for RBC2026 - ERROR X MONITOR VERSION.
Features:
- Real-time Error X display (Relative to center) in both UI and Terminal.
- Headless Terminal Logging (FPS + Target + ErrX).
- Instant Kalman Seeding (Fix sliding dot).
- Selective Leftmost Priority.
- Fixed 30 FPS Capping.
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
    def __init__(self, config_name: str = "config.yaml"):
        try:
            self.winmm = ctypes.WinDLL('winmm')
            self.winmm.timeBeginPeriod(1)
        except: pass

        self._setup_logging()
        self.logger = logging.getLogger("RBC2026")
        
        # --- TỰ ĐỘNG XÁC ĐỊNH ĐƯỜNG DẪN CONFIG ---
        possible_paths = [Path(config_name), Path(__file__).parent.parent / config_name]
        config_path = next((str(p) for p in possible_paths if p.exists()), None)
        if not config_path: raise FileNotFoundError(f"Config file {config_name} not found.")

        self.config = ConfigManager(config_path)
        
        # --- CẤU HÌNH CHIẾN THUẬT ---
        self.target_mode = self.config.get("classification.target_types", ["REAL"])[0].upper()
        self.frame_idx = 0
        self.loss_counter = 0
        self.max_loss_frames = self.config.get("detection.max_loss_frames", 7)
        
        self.latest_target_point = None
        self.latest_label = "NONE"
        self.latest_error_x = 0  # Biến lưu trữ sai số X
        self.status_text = "SEARCHING"
        self.last_target_x = None 
        self.target_switch_threshold = 80
        
        self.is_headless = self.config.get("display.headless", False)
        self.target_fps = 30
        self.frame_duration = 1.0 / self.target_fps - 0.012
        
        self.display_fps = 0.0
        self.last_fps_update_time = time.time()
        self.frame_count_since_update = 0

        try:
            self._init_models()
            self._init_hardware()
            self._init_tracking()
            self.logger.info(f"--- SYSTEM READY | MODE: {self.target_mode} | HEADLESS: {self.is_headless} ---")
        except Exception as e:
            self.logger.error(f"Init Failed: {e}")
            raise

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    def _init_models(self):
        self.ie = Core()
        yolo_device = self.config.get("models.yolo.device", "GPU")
        self.vision = RobotVision(self.config.get_path("yolo_xml"), device=yolo_device)
        
        cnn_device = self.config.get("models.cnn.device", "CPU")
        self.compiled_cnn = self.ie.compile_model(
            model=self.ie.read_model(self.config.get_path("cnn_xml")), device_name=cnn_device)
        self.cnn_output = self.compiled_cnn.output(0)
        self.labels_cnn = {int(v): k for k, v in json.load(open(self.config.get_path("labels_json"))).items()}

    def _init_hardware(self):
        cam_id = self.config.get("hardware.camera.device_id", 0)
        # Sửa src theo yêu cầu Camera của ông
        self.camera = CameraStream(src=cam_id + 1, buffer_size=1).start()
        self.uart = UARTManager(port=self.config.get("hardware.uart.port", "COM3"), 
                                baudrate=self.config.get("hardware.uart.baudrate", 115200))

    def _init_tracking(self):
        self.smoother = LabelSmoother(window_size=self.config.get("detection.label_smoothing.window_size", 7))
        self.conf_yolo = self.config.get("models.yolo.conf_threshold", 0.6)
        self.conf_cnn = self.config.get("models.cnn.conf_threshold", 0.5)

    def _reset_kalman_at_pos(self, x: float, y: float):
        """Mồi vị trí tức thì để tránh chấm tròn bay từ (0,0)."""
        self.vision._init_kalman_filter()
        self.vision.kalman.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.vision.kalman.statePre = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.smoother.history.clear()

    def _process_selective_leftmost(self, frame, detections):
        h_f, w_f = frame.shape[:2]
        # Ưu tiên vật thể bên trái nhất
        sorted_dets = sorted(detections, key=lambda d: (d.xyxy[0][0] + d.xyxy[0][2]) / 2)

        for det in sorted_dets:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            roi = frame[max(0, y1):min(h_f, y2), max(0, x1):min(w_f, x2)]
            input_data = preprocess_roi_for_cnn(roi)
            if input_data is None: continue
            
            res = self.compiled_cnn([input_data])[self.cnn_output]
            idx = np.argmax(res[0])
            label_raw = self.labels_cnn.get(idx, "UNK").upper()
            
            # Chỉ nhận diện đúng Mode (R1, REAL, FAKE)
            if label_raw.startswith(self.target_mode) and res[0][idx] >= self.conf_cnn:
                if self.last_target_x is None or abs(cx - self.last_target_x) > self.target_switch_threshold:
                    self._reset_kalman_at_pos(cx, cy)
                
                self.last_target_x = cx
                smoothed_label, _ = self.smoother.smooth("target", label_raw, res[0][idx])
                return (cx, cy), smoothed_label

        return None, "NONE"

    def run(self):
        try:
            while not self.camera.stopped:
                loop_start = time.time()
                frame = self.camera.read()
                if frame is None: continue
                
                h_f, w_f = frame.shape[:2]
                screen_center_x = w_f // 2

                # --- AI LOGIC (SKIP-2) ---
                if self.frame_idx % 3 == 0:
                    detections = self.vision.predict(frame, conf_threshold=self.conf_yolo, imgsz=512)
                    found_curr = False
                    if detections:
                        new_pt, label = self._process_selective_leftmost(frame, detections)
                        if new_pt:
                            tx, ty = self.vision.update_kalman(new_pt[0], new_pt[1])
                            self.latest_target_point, self.latest_label = (tx, ty), label
                            self.loss_counter, self.status_text, found_curr = 0, "LOCKED", True
                    
                    if not found_curr:
                        self.loss_counter += 1
                        if self.loss_counter >= self.max_loss_frames:
                            self.latest_target_point = (screen_center_x, h_f // 2)
                            self.status_text, self.latest_label, self.last_target_x = "LOST", "NONE", None
                        else:
                            tx, ty = self.vision.update_kalman()
                            self.latest_target_point = (tx, ty)
                else:
                    if self.loss_counter < self.max_loss_frames:
                        tx, ty = self.vision.update_kalman()
                        self.latest_target_point = (tx, ty)

                # --- CALCULATE ERROR X ---
                if self.latest_target_point:
                    self.latest_error_x = int(self.latest_target_point[0] - screen_center_x)
                    self.uart.send_error(self.latest_error_x)
                else:
                    self.latest_error_x = 0

                # --- FPS & TERMINAL LOGGING ---
                self.frame_count_since_update += 1
                curr_t = time.time()
                if curr_t - self.last_fps_update_time >= 0.5:
                    self.display_fps = self.frame_count_since_update / (curr_t - self.last_fps_update_time)
                    # LUÔN IN RA TERMINAL KHI CHẠY HEADLESS
                    if self.is_headless:
                        # In Error X ra Terminal
                        print(f"[RBC2026] FPS: {self.display_fps:.1f} | {self.status_text} | Target: {self.latest_label} | ErrX: {self.latest_error_x: >4}", end='\r')
                    
                    self.frame_count_since_update, self.last_fps_update_time = 0, curr_t

                # --- DISPLAY (NON-HEADLESS ONLY) ---
                if not self.is_headless:
                    cv2.rectangle(frame, (0, 0), (w_f, 40), (40, 40, 40), -1)
                    # Hiển thị thêm ErrX lên UI
                    cv2.putText(frame, f"T:{self.target_mode} | {self.status_text} | {self.latest_label} | ErrX:{self.latest_error_x}", (10, 27), 0, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, f"FPS: {self.display_fps:.1f}", (w_f - 110, 27), 0, 0.6, (0, 255, 0), 2)
                    
                    if self.latest_target_point:
                        color = (0, 255, 0) if self.status_text == "LOCKED" else (0, 0, 255)
                        cv2.circle(frame, self.latest_target_point, 10, color, -1)
                        cv2.line(frame, (screen_center_x, h_f), self.latest_target_point, color, 2)
                        cv2.drawMarker(frame, (screen_center_x, h_f // 2), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
                    
                    cv2.imshow("RBC2026_PROD", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break

                self.frame_idx += 1
                elapsed = time.time() - loop_start
                if elapsed < self.frame_duration:
                    time.sleep(self.frame_duration - elapsed)
        finally:
            print("\nShutting down...")
            self.cleanup()

    def cleanup(self):
        self.camera.stop(); self.uart.stop(); cv2.destroyAllWindows()
        try: self.winmm.timeEndPeriod(1)
        except: pass

if __name__ == "__main__":
    RoboconSystem().run()