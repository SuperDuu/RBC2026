import cv2
import numpy as np
import threading
import time
import logging
import json
import serial
import sys
import os
from pathlib import Path
from openvino.runtime import Core
import yaml

# Import from core
from core.config_manager import ConfigManager
from core.vision import RobotVision
from core.camera import CameraStream
from core.label_smoother import LabelSmoother
from core.utils import letterbox, preprocess_roi_for_cnn

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("SystemManager")

class SystemManager:
    def __init__(self, config_path="global_config.yaml"):
        print(f"--- INIT START: {config_path} ---", flush=True)
        # The core ConfigManager handles relative paths automatically
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        
        self.state = self.config['system']['initial_state']
        self.force_square = self.config['system']['force_square']
        self.headless = self.config['system']['headless']
        self.target_fps = self.config['system']['target_fps']
        self.frame_duration = 1.0 / self.target_fps
        
        # OpenVINO Shared Core
        self.ie = Core()
        
        # Async Inference State
        self.inference_lock = threading.Lock()
        self.latest_inference_data = ((None, "NONE"), [])
        self.inference_running = True
        
        try:
            print("  Initializing hardware...", flush=True)
            self._init_hardware()
            print("  Initializing models...", flush=True)
            self._init_models()
            
            # Start Background Inference Thread
            print("  Starting inference thread...", flush=True)
            self.inf_thread = threading.Thread(target=self._inference_loop, daemon=True)
            self.inf_thread.start()
            
            logger.info(f"SystemManager initialized. Initial State: {self.state}")
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def _init_hardware(self):
        self.use_test_image = self.config['system'].get('test_image', False)
        if self.use_test_image:
            self.test_img_path = self.config['system']['test_image_path']
            abs_img_path = self.config_manager.resolve_path(self.test_img_path) if hasattr(self.config_manager, 'resolve_path') else self.test_img_path
            self.test_frame = cv2.imread(abs_img_path)
            if self.test_frame is None:
                print(f"FAILED TO LOAD TEST IMAGE: {abs_img_path}", flush=True)
                self.use_test_image = False
            else:
                print(f"TEST IMAGE LOADED: {abs_img_path} {self.test_frame.shape}", flush=True)
        
        if not self.use_test_image:
            cam_id = self.config['hardware']['camera']['device_id']
            self.camera = CameraStream(src=cam_id).start()
        else:
            self.camera = None
        
        try:
            self.serial_port = serial.Serial(
                port=self.config['hardware']['serial']['port'],
                baudrate=self.config['hardware']['serial']['baudrate'],
                timeout=0.1
            )
        except Exception as e:
            self.serial_port = None

    def _init_models(self):
        print("\n" + "═"*50, flush=True)
        print(f"  [SYSTEM MANAGER] LOADING DUAL MODELS", flush=True)
        
        # Model V1 (SpearHead)
        v1_cfg = self.config['v1_model']
        v1_yolo = v1_cfg['yolo_xml']
        print(f"  Loading V1: {v1_yolo} on {v1_cfg['device']}...", flush=True)
        self.v1_vision = RobotVision(v1_yolo, device=v1_cfg['device'], ie_core=self.ie)
        print(f"  - Model V1 Load (SpearHead): SUCCESS", flush=True)
        
        # Model V2 (KFS)
        v2_cfg = self.config['v2_model']
        v2_yolo = v2_cfg['yolo_xml']
        print(f"  Loading V2 YOLO: {v2_yolo} on {v2_cfg['yolo_device']}...", flush=True)
        self.v2_vision = RobotVision(v2_yolo, device=v2_cfg['yolo_device'], ie_core=self.ie)
        
        v2_cnn_path = v2_cfg['cnn_xml']
        print(f"  Loading V2 CNN: {v2_cnn_path} on {v2_cfg['cnn_device']}...", flush=True)
        v2_cnn_model = self.ie.read_model(v2_cnn_path)
        self.v2_cnn = self.ie.compile_model(v2_cnn_model, device_name=v2_cfg['cnn_device'])
        self.v2_cnn_out = self.v2_cnn.output(0)
        
        v2_labels = v2_cfg['labels_json']
        with open(v2_labels, 'r') as f:
            self.v2_labels = {int(v): k for k, v in json.load(f).items()}
            
        print(f"  - Model V2 Load (KFS): SUCCESS", flush=True)
        print("" + "═"*50 + "\n", flush=True)
            
        self.v2_smoother = LabelSmoother(window_size=7)

    def _inference_loop(self):
        """Background thread for continuous AI processing."""
        while self.inference_running:
            if self.use_test_image:
                frame = self.test_frame.copy()
            elif self.camera and not self.camera.stopped:
                frame = self.camera.read()
            else:
                frame = None
            
            if frame is None:
                time.sleep(0.01); continue
            
            # --- Inference Logic ---
            res = (None, "NONE")
            all_dets = [] # For test mode visualization: List of (box, label, score)
            
            if self.state == 1: # V1 SpearHead
                dets = self.v1_vision.predict(frame, conf_threshold=self.config['v1_model']['conf_threshold'])
                if dets:
                    # Target selection: Leftmost
                    target_det = min(dets, key=lambda d: (d.xyxy[0][0] + d.xyxy[0][2]) / 2)
                    cx, cy = int((target_det.xyxy[0][0] + target_det.xyxy[0][2]) // 2), int((target_det.xyxy[0][1] + target_det.xyxy[0][3]) // 2)
                    res = (cx, cy), "TARGET"
                    
                    if self.use_test_image:
                        for d in dets:
                            x1, y1, x2, y2 = map(int, d.xyxy[0])
                            all_dets.append(([x1, y1, x2, y2], "YOLO", d.conf))
                
            elif self.state == 2: # V2 KFS
                dets = self.v2_vision.predict(frame, conf_threshold=self.config['v2_model']['conf_threshold_yolo'])
                if dets:
                    sorted_dets = sorted(dets, key=lambda d: (d.xyxy[0][0] + d.xyxy[0][2]) / 2) # Leftmost first
                    target_found = False
                    
                    for det in sorted_dets:
                        x1, y1, x2, y2 = map(int, det.xyxy[0])
                        roi = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                        input_data = preprocess_roi_for_cnn(roi, input_size=self.config['v2_model']['cnn_input_size'])
                        
                        if input_data is None:
                            if self.use_test_image: all_dets.append(([x1, y1, x2, y2], "INVALID_ROI", 0.0))
                            continue
                        
                        cnn_res = self.v2_cnn([input_data])[self.v2_cnn_out]
                        idx = np.argmax(cnn_res[0])
                        label_raw = self.v2_labels.get(idx, "UNK").upper()
                        score = float(cnn_res[0][idx])
                        
                        if self.use_test_image:
                            all_dets.append(([x1, y1, x2, y2], label_raw, score))
                        
                        is_target = any(label_raw.startswith(t.upper()) for t in self.config['v2_model']['target_types'])
                        if not target_found and is_target and score >= self.config['v2_model']['conf_threshold_cnn']:
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            label, _ = self.v2_smoother.smooth("target", label_raw, score)
                            res = (cx, cy), label
                            target_found = True
                            if not self.use_test_image: break # Optimized for live run

            with self.inference_lock:
                self.latest_inference_data = (res, all_dets)
            time.sleep(0.001)

    def run(self):
        print(f"--- RUN LOOP START | HEADLESS: {self.headless} ---", flush=True)
        self.last_fps_update_time = time.time()
        self.frame_count_since_update = 0
        self.display_fps = 0.0

        self.loss_counter = 0
        self.max_loss_frames = self.config['detection']['max_loss_frames']

        try:
            while self.inference_running:
                loop_start = time.time()
                if self.use_test_image:
                    frame = self.test_frame.copy()
                elif self.camera and not self.camera.stopped:
                    frame = self.camera.read()
                else:
                    break
                    
                if frame is None: continue
                
                h_orig, w_orig = frame.shape[:2]

                # Consume latest AI results
                new_pt, label = (None, "NONE")
                all_dets = []
                with self.inference_lock:
                    if self.latest_inference_data:
                        (new_pt, label), all_dets = self.latest_inference_data
                        self.latest_inference_data = None
                
                # Kalman Tracking & Smoothing
                current_vision = self.v1_vision if self.state == 1 else self.v2_vision
                if new_pt:
                    tx, ty = current_vision.update_kalman(new_pt[0], new_pt[1])
                    status = "LOCKED"
                    self.loss_counter = 0
                else:
                    self.loss_counter += 1
                    tx, ty = current_vision.update_kalman()
                    status = "SEARCHING" if self.loss_counter < self.max_loss_frames else "LOST"
                    if status == "LOST":
                        tx, ty = w_orig // 2, h_orig // 2
                
                # Display Mapping
                if self.force_square:
                    imgsz = 512
                    _, scale, (pw, ph) = letterbox(frame, (imgsz, imgsz))
                    dtx, dty = int(tx * scale + pw), int(ty * scale + ph)
                    sc_x = imgsz // 2
                    target_point = (dtx, dty)
                    curr_h, curr_w = imgsz, imgsz
                else:
                    sc_x = w_orig // 2
                    target_point = (tx, ty)
                    curr_h, curr_w = h_orig, w_orig
                    scale = 1.0; pw = ph = 0
                
                # Error Calculation & Serial
                err_x = int(target_point[0] - sc_x) if status == "LOCKED" else 999
                if self.serial_port:
                    self.serial_port.write(f"{err_x}\n".encode())
                    
                    # UART Read (Bi-directional control)
                    if self.serial_port.in_waiting > 0:
                        try:
                            cmd_data = self.serial_port.read(self.serial_port.in_waiting).decode('utf-8', errors='ignore')
                            # Find the last valid command '0', '1', or '2' in buffer
                            for char in reversed(cmd_data):
                                if char in ['0', '1', '2']:
                                    new_st = int(char)
                                    if new_st != self.state:
                                        self.state = new_st
                                        logger.info(f"UART Mode Sync: {self.state}")
                                    break
                        except Exception as e:
                            pass # Silently handle decode errors
                
                # FPS Calculation
                self.frame_count_since_update += 1
                curr_t = time.time()
                if curr_t - self.last_fps_update_time >= 0.5:
                    self.display_fps = self.frame_count_since_update / (curr_t - self.last_fps_update_time)
                    self.frame_count_since_update, self.last_fps_update_time = 0, curr_t

                # UI
                if not self.headless:
                    df = frame if not self.force_square else letterbox(frame, (512, 512))[0]
                    color = (0, 255, 0) if status == "LOCKED" else (255, 255, 0) if status == "SEARCHING" else (0, 0, 255)
                    
                    # Draw All Boxes in Test Mode
                    if self.use_test_image and all_dets:
                        for box, b_label, b_score in all_dets:
                            # Map box to display coords
                            bx1, by1, bx2, by2 = box
                            dbx1, dby1 = int(bx1 * scale + pw), int(by1 * scale + ph)
                            dbx2, dby2 = int(bx2 * scale + pw), int(by2 * scale + ph)
                            
                            cv2.rectangle(df, (dbx1, dby1), (dbx2, dby2), (0, 255, 0), 1)
                            cv2.putText(df, f"{b_label} {b_score:.2f}", (dbx1, dby1 - 5), 0, 0.4, (0, 255, 0), 1)

                    # Target Point and Line
                    cv2.line(df, (sc_x, curr_h), target_point, color, 2)
                    cv2.circle(df, target_point, 10, color, -1)
                    
                    # Status Header
                    cv2.rectangle(df, (0, 0), (curr_w, 35), (40, 40, 40), -1)
                    cv2.putText(df, f"MODE:{self.state} | {status} | {label} | EX:{err_x}", (10, 25), 0, 0.6, (255, 255, 255), 1)
                    cv2.putText(df, f"FPS: {self.display_fps:.1f}", (curr_w - 100, 25), 0, 0.6, (0, 255, 0), 2)
                    
                    cv2.imshow("SystemManager", df)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'): break
                    elif key in [ord('1'), ord('2'), ord('0')]:
                        self.state = int(chr(key))
                        logger.info(f"Switched state to: {self.state}")
                
                # FPS Capping
                wait = self.frame_duration - (time.time() - loop_start)
                if wait > 0: time.sleep(wait)
        finally:
            self.cleanup()

    def cleanup(self):
        self.inference_running = False
        if hasattr(self, 'inf_thread'): self.inf_thread.join(timeout=1.0)
        if self.camera: self.camera.stop()
        if self.serial_port: self.serial_port.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    SystemManager().run()
