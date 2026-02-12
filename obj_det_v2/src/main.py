import cv2
import time
import logging
import numpy as np
import json
import threading
from pathlib import Path
from collections import Counter
from openvino.runtime import Core
from vision import RobotVision
from connection import UARTManager
from preprocessing import RobotPreprocessing

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class LabelSmoother:
    def __init__(self, window_size=5):
        self.history = {}
        self.window_size = window_size

    def smooth(self, box_id, label, conf):
        if box_id not in self.history:
            self.history[box_id] = []
        self.history[box_id].append((label, conf))
        if len(self.history[box_id]) > self.window_size:
            self.history[box_id].pop(0)
        labels = [x[0] for x in self.history[box_id]]
        most_common_label = Counter(labels).most_common(1)[0][0]
        avg_conf = np.mean([x[1] for x in self.history[box_id] if x[0] == most_common_label])
        return most_common_label, avg_conf

class CameraStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.success, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            success, frame = self.cap.read()
            if success: self.frame = frame
            else: self.stopped = True

    def read(self): return self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

# --- CONFIG ---
BASE_DIR = Path(__file__).resolve().parent.parent
YOLO_XML = str(BASE_DIR / "models" / "best_openvino_model" / "best.xml")
CNN_XML = str(BASE_DIR / "models" / "openvino_cnn_v2" / "classifier_v2.xml")
LABEL_JSON = str(BASE_DIR / "models" / "labels_v2.json")

ie = Core()
try:
    with open(LABEL_JSON, 'r') as f:
        data = json.load(f)
        LABELS_CNN = {int(v): k for k, v in data.items()}
    logging.info(f"Nạp {len(LABELS_CNN)} nhãn thành công.")
except Exception as e:
    logging.error(f"Lỗi nhãn: {e}"); exit()

try:
    cnn_model = ie.read_model(model=CNN_XML)
    compiled_cnn = ie.compile_model(model=cnn_model, device_name="GPU")
    cnn_output = compiled_cnn.output(0)
    logging.info("CNN chạy trên iGPU.")
except:
    compiled_cnn = ie.compile_model(model=cnn_model, device_name="CPU")
    cnn_output = compiled_cnn.output(0)

vision = RobotVision(YOLO_XML)
uart = UARTManager(port='/dev/ttyUSB0', baud=115200)
preprocessor = RobotPreprocessing(device_path="/dev/video0")
smoother = LabelSmoother(window_size=6)

def classify_roi(roi_color):
    if roi_color is None or roi_color.size == 0: return "EMPTY", 0.0
    try:
        gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        max_side = max(h, w)
        canvas = np.zeros((max_side, max_side), dtype=np.uint8)
        canvas[(max_side-h)//2:(max_side-h)//2+h, (max_side-w)//2:(max_side-w)//2+w] = gray
        resized = cv2.resize(canvas, (64, 64), interpolation=cv2.INTER_AREA)
        input_data = resized.reshape(1, 64, 64, 1).astype(np.float32) / 255.0
        res = compiled_cnn([input_data])[cnn_output]
        idx = int(np.argmax(res))
        return LABELS_CNN.get(idx, f"ID_{idx}"), res[0][idx]
    except: return "ERR", 0.0

def run_main():
    cam = CameraStream(src=0).start()
    prev_time = 0
    frame_count = 0
    SKIP_FREQ = 2
    cached_boxes = []
    proc_frame = None

    try:
        while not cam.stopped:
            frame_raw = cam.read()
            if frame_raw is None: continue
            
            frame_count += 1
            h_f, w_f = frame_raw.shape[:2]
            display_frame = frame_raw.copy()
            screen_center_x = w_f // 2

            if frame_count % SKIP_FREQ == 0:
                proc_frame = preprocessor.run_pipeline(frame_raw)
                cached_boxes = vision.predict(proc_frame, conf_threshold=0.5)

            target_point = None
            best_conf = 0
            current_label = "NONE"

            if proc_frame is not None:
                for box in cached_boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    roi = proc_frame[max(0,y1):min(h_f,y2), max(0,x1):min(w_f,x2)]
                    
                    label_raw, conf_raw = classify_roi(roi)
                    box_id = f"{x1//20}_{y1//20}"
                    label, conf = smoother.smooth(box_id, label_raw, conf_raw)
                    
                    is_target = "real" in label.lower() or "r1" in label.upper()
                    
                    if is_target and conf > 0.5:
                        color = (0, 255, 0)
                    elif is_target and conf > 0.80:
                        color = (0, 255, 255)
                    else:
                        color = (0, 0, 255)

                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame, f"{label} {conf:.2f}", (x1, y1-10), 0, 0.5, color, 2)

                    if is_target and conf > 0.5: 
                        if conf > best_conf:
                            best_conf = conf
                            target_point = (int((x1+x2)/2), int((y1+y2)/2))
                            current_label = label

            if target_point:
                tx, ty = vision.update_kalman(target_point[0], target_point[1])
                error_x = tx - screen_center_x
                uart.send_error(error_x) 
                line_color = (0, 255, 0) if best_conf > 0.5 else (0, 255, 255)
                cv2.line(display_frame, (screen_center_x, h_f), (tx, ty), line_color, 2)
                cv2.circle(display_frame, (tx, ty), 8, (0, 255, 255), -1)
                cv2.putText(display_frame, f"E:{error_x}", (tx+10, ty), 0, 0.6, (0, 255, 255), 2)
            else:
                uart.send_error(0)
                cv2.line(display_frame, (screen_center_x, h_f), (screen_center_x, h_f - 50), (255, 255, 255), 1)

            now = time.time()
            fps = 1 / (now - prev_time) if prev_time else 0
            prev_time = now
            cv2.putText(display_frame, f"FPS: {fps:.1f} | {current_label}", (10, 30), 0, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("RBC2026_FINAL", display_frame)
            if cv2.waitKey(1) == ord('q'): break
    finally:
        cam.stop(); uart.stop(); cv2.destroyAllWindows()

if __name__ == "__main__":
    run_main()