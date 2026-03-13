import cv2
import numpy as np
import threading
import time
import logging
import json
import serial
import sys
import subprocess
from pathlib import Path
from openvino.runtime import Core
import yaml

class LabelSmoother:
    def __init__(self, window_size=7):
        self.window_size = window_size
        self.history = {}

    def smooth(self, obj_id, label, conf):
        if obj_id not in self.history:
            self.history[obj_id] = []
        self.history[obj_id].append((label, conf))
        if len(self.history[obj_id]) > self.window_size:
            self.history[obj_id].pop(0)
        
        labels = [h[0] for h in self.history[obj_id]]
        final_label = max(set(labels), key=labels.count)
        return final_label, conf

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("SuperManager")

class ConfigManager:
    """Manages application configuration from YAML file."""
    def __init__(self, config_path="super_config.yaml"):
        self.config_path = Path(config_path)
        self.config = {}
        self.load_config()

    def load_config(self):
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            sys.exit(1)

    def get(self, key_path, default=None):
        keys = key_path.split('.')
        value = self.config
        try:
            for key in keys:
                value = value[key]
            return value
        except:
            return default

# --- UTILS (Isolated Preprocessing) ---

def letterbox(img, new_shape=(512, 512), color=(128, 128, 128)):
    """Standard letterbox resize used by YOLO."""
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)

class CameraStream:
    """Threaded camera capture for low-latency video streaming."""
    def __init__(self, src=0, buffer_size=1):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        self.status, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            if self.cap.isOpened():
                (status, frame) = self.cap.read()
                if status:
                    with self.lock:
                        self.frame = frame
                else:
                    self.stopped = True

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.cap.release()

class UARTManager:
    """50Hz background sender for STM32 error packets (S+0000E)."""
    def __init__(self, port, baudrate):
        self.lock = threading.Lock()
        self.last_packet = "S+0000E\n"
        self.running = True
        try:
            self.ser = serial.Serial(port, baudrate, timeout=0, write_timeout=None)
            self.connected = True
        except:
            self.ser = None
            self.connected = False
            logger.warning(f"UART port {port} not available.")
        
        threading.Thread(target=self._send_loop, daemon=True).start()

    def _send_loop(self):
        interval = 1.0 / 50.0 # 50Hz
        next_t = time.perf_counter()
        while self.running:
            if self.ser and self.ser.is_open:
                with self.lock:
                    pkt = self.last_packet
                try:
                    self.ser.write(pkt.encode())
                    self.ser.flush()
                except:
                    pass
            
            next_t += interval
            sleep_time = next_t - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_t = time.perf_counter()

    def send_error(self, error_x):
        error_val = max(-9999, min(9999, int(error_x)))
        sign = "+" if error_val >= 0 else "-"
        pkt = f"S{sign}{abs(error_val):04d}E\n"
        with self.lock:
            self.last_packet = pkt

    def stop(self):
        self.running = False
        if self.ser: self.ser.close()

class SuperManager:
    """
    Unified manager for RBC2026 Vision Systems.
    Implements State Machine, Unified Inference Core, and Serial Integration.
    """
    def __init__(self, config_path="super_config.yaml"):
        self.config_manager = ConfigManager(config_path)
        self.state = self.config_manager.get("system.initial_state", 1) # 0: OFF, 1: Target, 2: KFS
        self.running = True
        self.ie = Core()
        
        # State & Counters
        self.frame_idx = 0
        self.loss_counter = 0
        self.status_text = "SEARCHING"
        self.latest_label = "NONE"
        self.max_loss_frames = self.config_manager.get("detection.max_loss_frames", 15)
        self.inference_skip = self.config_manager.get("system.inference_skip", 3)
        self.ema_alpha = self.config_manager.get("detection.ema_alpha", 0.7)
        self.last_boxes = {} # For EMA
        self.debug_boxes = [] # For test_image mode visualization
        
        # Test Mode
        self.test_image_mode = self.config_manager.get("system.test_image", False)
        self.test_image_path = self.config_manager.get("system.test_image_path", "")
        
        # OpenVINO Models
        self.models = {}
        self._init_openvino()
        
        # Tracking
        self._init_tracking()
        self.smoother = LabelSmoother(window_size=7)
        
        # UART (50Hz background)
        self.serial_port = self.config_manager.get("hardware.serial.port", "/dev/ttyUSB0")
        self.baudrate = self.config_manager.get("hardware.serial.baudrate", 115200)
        self.uart = UARTManager(self.serial_port, self.baudrate)
        
        # Camera Stream (Threaded)
        self.cam_id = self.config_manager.get("hardware.camera.device_id", 0)
        self.camera = CameraStream(self.cam_id)
        
        # Hardware Config
        self.device_path = f"/dev/video{self.cam_id}"
        self._lock_hardware_controls()
        
        # Display & Flow Control
        self.is_headless = self.config_manager.get("system.headless", False)
        self.target_fps = self.config_manager.get("system.target_fps", 30)
        self.frame_duration = 1.0 / self.target_fps if self.target_fps > 0 else 0
        self.last_fps_update_time = time.time()
        self.frame_count = 0
        self.display_fps = 0.0

        # Asynchronous Inference State
        self.inference_lock = threading.Lock()
        self.latest_inference_data = None
        self.inference_running = True

        # State Switch Listener
        self.serial_thread = threading.Thread(target=self._serial_reader, daemon=True)
        self.serial_thread.start()

        # Inference Thread (Background AI)
        self.inf_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inf_thread.start()

        logger.info(f"SuperManager ASYNC initialized. Target FPS: {self.target_fps}")

    def _lock_hardware_controls(self):
        """Lock camera settings to manual for consistency."""
        try:
            subprocess.run(['v4l2-ctl', '-d', self.device_path, '-c', 'white_balance_automatic=0'], check=False)
            subprocess.run(['v4l2-ctl', '-d', self.device_path, '-c', 'auto_exposure=1'], check=False)
            subprocess.run(['v4l2-ctl', '-d', self.device_path, '-c', 'exposure_time_absolute=600'], check=False)
            logger.info(f"Hardware controls locked for {self.device_path}.")
        except:
            logger.warning("V4L2 controls not available.")

    def _enhance(self, frame, clip_limit):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    def _init_tracking(self):
        """Initialize Kalman filter for trajectory smoothing."""
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        # Bumping up process noise for responsiveness on Go 2
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05

    def _reset_kalman(self, x, y):
        self._init_tracking()
        self.kalman.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.kalman.statePre = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.smoother.history.clear()

    def _init_openvino(self):
        """Load and compile all models to designated devices."""
        logger.info("Initializing OpenVINO Core...")
        try:
            # Model V1 YOLO
            m_v1_yolo_xml = self.config_manager.get("v1_model.yolo_xml")
            m_v1_yolo_dev = self.config_manager.get("v1_model.device", "GPU")
            m_v1_yolo = self.ie.read_model(m_v1_yolo_xml)
            self.models['v1_yolo'] = self.ie.compile_model(m_v1_yolo, m_v1_yolo_dev)
            self.output_v1_yolo = self.models['v1_yolo'].output(0)
            
            # Model V2 YOLO
            m_v2_yolo_xml = self.config_manager.get("v2_model.yolo_xml")
            m_v2_yolo_dev = self.config_manager.get("v2_model.yolo_device", "GPU")
            m_v2_yolo = self.ie.read_model(m_v2_yolo_xml)
            self.models['v2_yolo'] = self.ie.compile_model(m_v2_yolo, m_v2_yolo_dev)
            self.output_v2_yolo = self.models['v2_yolo'].output(0)
            
            # Model V2 CNN
            m_v2_cnn_xml = self.config_manager.get("v2_model.cnn_xml")
            m_v2_cnn_dev = self.config_manager.get("v2_model.cnn_device", "CPU")
            m_v2_cnn = self.ie.read_model(m_v2_cnn_xml)
            self.models['v2_cnn'] = self.ie.compile_model(m_v2_cnn, m_v2_cnn_dev)
            self.output_v2_cnn = self.models['v2_cnn'].output(0)
            
            # Load Labels
            labels_json = self.config_manager.get("v2_model.labels_json")
            with open(labels_json, 'r') as f:
                self.v2_labels = {int(v): k for k, v in json.load(f).items()}
                
            logger.info(f"Models loaded: V1(YOLO)->{m_v1_yolo_dev}, V2(YOLO)->{m_v2_yolo_dev}, V2(CNN)->{m_v2_cnn_dev}")
        except Exception as e:
            logger.error(f"Failed to load OpenVINO models: {e}")
            sys.exit(1)

    def _serial_reader(self):
        """Background thread to handle state switching via Serial."""
        logger.info(f"Background Serial Listener started...")
        try:
            while self.running:
                if self.uart.ser and self.uart.ser.is_open and self.uart.ser.in_waiting > 0:
                    cmd = self.uart.ser.read(1)
                    if cmd == b'\x00':
                        self.state = 0
                        logger.info("State switched to: OFF (0x00)")
                    elif cmd == b'\x01':
                        self.state = 1
                        logger.info("State switched to: TARGET (0x01)")
                    elif cmd == b'\x02':
                        self.state = 2
                        logger.info("State switched to: KFS (0x02)")
                time.sleep(0.01)
        except Exception as e:
            logger.error(f"Serial Input Error: {e}")

    def _inference_loop(self):
        """Background thread for continuous AI Inference."""
        logger.info("Inference Background Loop started.")
        while self.inference_running:
            if self.state == 0:
                time.sleep(0.1)
                continue
            
            frame = self.camera.read()
            if frame is None:
                time.sleep(0.01)
                continue
            
            # Run the appropriate detector based on current state
            try:
                if self.state == 1:
                    result = self.process_v1(frame)
                elif self.state == 2:
                    result = self.process_v2(frame)
                else:
                    result = None
                
                if result:
                    with self.inference_lock:
                        self.latest_inference_data = result
            except Exception as e:
                logger.error(f"Inference error: {e}")
            
            # Nano-sleep to prevent thread starvation
            time.sleep(0.001)

    def preprocess_v1(self, frame):
        """Isolated preprocessing for Weapon Detection."""
        use_clahe = self.config_manager.get("v1_model.use_clahe", False)
        clip_limit = self.config_manager.get("v1_model.clip_limit", 1.5)
        input_size = self.config_manager.get("v1_model.input_size", 512)

        if use_clahe:
            frame = self._enhance(frame, clip_limit)
        
        canvas, scale, pad = letterbox(frame, (input_size, input_size))
        blob = canvas.transpose((2, 0, 1)).reshape((1, 3, input_size, input_size)).astype(np.float32) / 255.0
        return blob, scale, pad

    def preprocess_v2_yolo(self, frame):
        """Isolated preprocessing for KFS YOLO."""
        use_clahe = self.config_manager.get("v2_model.use_clahe", False)
        clip_limit = self.config_manager.get("v2_model.clip_limit", 1.2)
        input_size = self.config_manager.get("v2_model.yolo_input_size", 512)
        mean = self.config_manager.get("v2_model.mean", [0,0,0])
        std = self.config_manager.get("v2_model.std", [255,255,255])

        if use_clahe:
            frame = self._enhance(frame, clip_limit)
        canvas, scale, pad = letterbox(frame, (input_size, input_size))
        blob = (canvas.astype(np.float32) - mean) / std
        blob = blob.transpose((2, 0, 1)).reshape((1, 3, input_size, input_size))
        return blob, scale, pad

    def preprocess_v2_cnn(self, roi):
        """Isolated preprocessing for KFS CNN (Grayscale, Resize, Pad)."""
        if roi is None or roi.size == 0: return None
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        size = self.config_manager.get("v2_model.cnn_input_size", 64)
        bg_val = self.config_manager.get("v2_model.background_value", 128)
        
        scale = size / max(h, w)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.full((size, size), bg_val, dtype=np.uint8)
        y_off, x_off = (size - nh) // 2, (size - nw) // 2
        canvas[y_off:y_off+nh, x_off:x_off+nw] = resized
        return canvas.reshape(1, size, size, 1).astype(np.float32) / 255.0

    def process_v1(self, frame):
        """Detection logic for Weapon mode."""
        blob, scale, (pad_x, pad_y) = self.preprocess_v1(frame)
        conf_thresh = self.config_manager.get("v1_model.conf_threshold", 0.6)
        nms_thresh = self.config_manager.get("v1_model.nms_threshold", 0.35)

        results = self.models['v1_yolo']([blob])[self.output_v1_yolo]
        predictions = np.squeeze(results)
        if predictions.shape[0] < predictions.shape[1]: predictions = predictions.T
        
        boxes, confs, classes = [], [], []
        if len(predictions) > 0:
            scores = predictions[:, 4:]
            max_scores = np.max(scores, axis=1)
            cls_ids = np.argmax(scores, axis=1)
            mask = (max_scores > conf_thresh)
            valid_preds = predictions[mask]
            
            if len(valid_preds) > 0:
                xc, yc, w, h = valid_preds[:, 0], valid_preds[:, 1], valid_preds[:, 2], valid_preds[:, 3]
                x1 = ((xc - w/2 - pad_x) / scale).astype(int)
                y1 = ((yc - h/2 - pad_y) / scale).astype(int)
                boxes = np.column_stack([x1, y1, (w/scale).astype(int), (h/scale).astype(int)]).tolist()
                confs = max_scores[mask].tolist()
                classes = cls_ids[mask].tolist()
        
        if not boxes: return None
        indices = cv2.dnn.NMSBoxes(boxes, confs, conf_thresh, nms_thresh)
        if len(indices) == 0: return None

        # Leftmost Priority
        best_idx = indices.flatten()[np.argmin([boxes[i][0] for i in indices.flatten()])]
        bx, by, bw, bh = boxes[best_idx]
        cx, cy = bx + bw // 2, by + bh // 2
        
        cls_id = classes[best_idx]
        label_text = "TARGET" if cls_id == 0 else f"CL:{cls_id}"
        label, _ = self.smoother.smooth("v1", label_text, confs[best_idx])
        return (cx, cy), label

    def process_v2(self, frame):
        """Detection logic for KFS mode."""
        h_orig, w_orig = frame.shape[:2]
        blob, scale, (pad_x, pad_y) = self.preprocess_v2_yolo(frame)
        yolo_conf = self.config_manager.get("v2_model.conf_threshold_yolo", 0.6)
        cnn_conf = self.config_manager.get("v2_model.conf_threshold_cnn", 0.5)
        target_types = self.config_manager.get("v2_model.target_types", ["REAL"])

        yolo_res = self.models['v2_yolo']([blob])[self.output_v2_yolo]
        predictions = np.squeeze(yolo_res)
        if predictions.shape[0] < predictions.shape[1]: predictions = predictions.T
        
        boxes, confs = [], []
        if len(predictions) > 0:
            mask = (np.max(predictions[:, 4:], axis=1) > yolo_conf) & (np.argmax(predictions[:, 4:], axis=1) == 0)
            valid_preds = predictions[mask]
            if len(valid_preds) > 0:
                xc, yc, w, h = valid_preds[:, 0], valid_preds[:, 1], valid_preds[:, 2], valid_preds[:, 3]
                x1, y1 = ((xc - w/2 - pad_x) / scale).astype(int), ((yc - h/2 - pad_y) / scale).astype(int)
                boxes = np.column_stack([x1, y1, (w/scale).astype(int), (h/scale).astype(int)]).tolist()
                confs = np.max(predictions[mask][:, 4:], axis=1).tolist()
        
        if not boxes: return None
        indices = cv2.dnn.NMSBoxes(boxes, confs, yolo_conf, 0.45)
        if len(indices) == 0: return None
        
        sorted_indices = indices.flatten()[np.argsort([boxes[i][0] for i in indices.flatten()])]
        for idx in sorted_indices:
            bx, by, bw, bh = boxes[idx]
            roi = frame[max(0, by):min(h_orig, by+bh), max(0, bx):min(w_orig, bx+bw)]
            cnn_blob = self.preprocess_v2_cnn(roi)
            if cnn_blob is None: continue
            
            cnn_res = self.models['v2_cnn']([cnn_blob])[self.output_v2_cnn]
            cnn_idx = np.argmax(cnn_res[0])
            label_raw = self.v2_labels.get(cnn_idx, "UNK").upper()
            
            if any(label_raw.startswith(t) for t in target_types) and cnn_res[0][cnn_idx] >= cnn_conf:
                cx, cy = bx + bw // 2, by + bh // 2
                label_smooth, _ = self.smoother.smooth("v2", label_raw, cnn_res[0][cnn_idx])
                return (cx, cy), label_smooth
        return None

    def run(self):
        """Main rendering loop at steady 30 FPS."""
        logger.info(f"Steady 30 FPS Loop started (State {self.state}).")
        v1_name = self.config_manager.get("v1_model.name", "SpearHead")
        v2_name = self.config_manager.get("v2_model.name", "KFS")
        force_sq = self.config_manager.get("system.force_square", True)
        imgsz = self.config_manager.get("v1_model.input_size", 512)

        try:
            while not self.camera.stopped:
                loop_start = time.time()
                
                if self.test_image_mode:
                    frame = cv2.imread(self.test_image_path)
                else:
                    frame = self.camera.read()
                
                if frame is None: continue
                h_f, w_f = frame.shape[:2]

                # ─── 1. ASYNC DATA PICKUP ──────────────────────────
                new_ai_res = None
                with self.inference_lock:
                    if self.latest_inference_data:
                        new_ai_res = self.latest_inference_data
                        self.latest_inference_data = None
                
                if self.state == 0:
                    self.status_text, self.latest_label = "IDLE", "OFF"
                elif new_ai_res:
                    (ax, ay), self.latest_label = new_ai_res
                    self.kalman.correct(np.array([[np.float32(ax)], [np.float32(ay)]], dtype=np.float32))
                    self.loss_counter, self.status_text = 0, "LOCKED"
                else:
                    if self.state != 0:
                        self.loss_counter += 1
                        if self.loss_counter >= self.max_loss_frames:
                            self.status_text, self.latest_label = "LOST", "NONE"

                # ─── 2. KALMAN PREDICT (EVERY FRAME) ───────────────
                pred = self.kalman.predict()
                tx, ty = int(pred[0, 0]), int(pred[1, 0])

                # ─── 3. UI COORDINATE MAPPING ──────────────────────
                if force_sq:
                    display_frame, display_scale, (pad_w, pad_h) = letterbox(frame, (imgsz, imgsz))
                    screen_center_x = imgsz // 2
                    dtx = int(tx * display_scale + pad_w)
                    dty = int(ty * display_scale + pad_h)
                    target_pt = (dtx, dty)
                    curr_h, curr_w = imgsz, imgsz
                else:
                    display_frame = frame.copy()
                    screen_center_x = w_f // 2
                    target_pt = (tx, ty)
                    curr_h, curr_w = h_f, w_f

                if self.status_text == "LOST":
                    target_pt = (screen_center_x, curr_h // 2)

                # ─── 4. ERROR & UART (30 FPS) ──────────────────────
                error_x = 999
                if self.status_text == "LOCKED":
                    error_x = int(target_pt[0] - screen_center_x)
                self.uart.send_error(error_x)

                # ─── 5. UI & LOGGING ──────────────────────────────
                self.frame_count += 1
                curr_time = time.time()
                if curr_time - self.last_fps_update_time >= 0.5:
                    self.display_fps = self.frame_count / (curr_time - self.last_fps_update_time)
                    self.frame_count, self.last_fps_update_time = 0, curr_time
                    if self.is_headless:
                        print(f"[RBC2026] FPS:{self.display_fps:.1f} | {self.status_text} | M:{self.state} | ErrX:{error_x}   ", end='\r')

                if not self.is_headless:
                    cv2.rectangle(display_frame, (0, 0), (curr_w, 40), (40, 40, 40), -1)
                    mode_name = v1_name if self.state == 1 else (v2_name if self.state == 2 else "OFF")
                    cv2.putText(display_frame, f"M:{mode_name} | {self.status_text} | {self.latest_label} | ErrX:{error_x}", 
                                (15, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(display_frame, f"FPS: {self.display_fps:.1f}", (curr_w - 110, 27), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    cv2.drawMarker(display_frame, (curr_w // 2, curr_h // 2), (200, 200, 200), cv2.MARKER_CROSS, 25, 1)
                    
                    color = (0, 255, 0) if self.status_text == "LOCKED" else (0, 0, 255)
                    if self.state == 0: color = (100, 100, 100)
                    cv2.circle(display_frame, target_pt, 8, color, -1)
                    cv2.line(display_frame, (curr_w // 2, curr_h), target_pt, color, 2)
                    cv2.imshow("SuperManager RBC2026", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break

                # ─── 6. FPS CAP ──────────────────────────────────
                elapsed = time.time() - loop_start
                if elapsed < self.frame_duration:
                    time.sleep(self.frame_duration - elapsed)
        finally:
            self.cleanup()

    def cleanup(self):
        """Graceful shutdown of all threads and devices."""
        self.inference_running = False
        self.running = False
        if hasattr(self, 'inf_thread'): self.inf_thread.join(timeout=1.0)
        if hasattr(self, 'camera'): self.camera.stop()
        if hasattr(self, 'uart'): self.uart.stop()
        cv2.destroyAllWindows()
        logger.info("System Shutdown.")

if __name__ == "__main__":
    manager = SuperManager()
    manager.run()
