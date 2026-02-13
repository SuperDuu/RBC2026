import cv2
import time
import numpy as np
import json
import threading
from pathlib import Path
from collections import Counter
from openvino.runtime import Core
from vision import RobotVision
from connection import UARTManager

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

BASE_DIR = Path(__file__).resolve().parent.parent
YOLO_XML = str(BASE_DIR / "models" / "best_openvino_model" / "best.xml")
CNN_XML = str(BASE_DIR / "models" / "openvino_cnn_v2_3" / "classifier_v2_3.xml")
LABEL_JSON = str(BASE_DIR / "models" / "labels_v2.3.json")

ie = Core()
with open(LABEL_JSON, 'r') as f:
    LABELS_CNN = {int(v): k for k, v in json.load(f).items()}

compiled_cnn = ie.compile_model(model=ie.read_model(model=CNN_XML), device_name="GPU")
cnn_output = compiled_cnn.output(0)
vision = RobotVision(YOLO_XML)
uart = UARTManager(port='/dev/ttyUSB0', baud=115200)
smoother = LabelSmoother(window_size=5)

def preprocess_realtime(roi):
    if roi is None or roi.size == 0: return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = 64 / max(h, w)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((64, 64), 128, dtype=np.uint8)
    canvas[(64-nh)//2:(64-nh)//2+nh, (64-nw)//2:(64-nw)//2+nw] = resized
    return canvas.reshape(1, 64, 64, 1).astype(np.float32) / 255.0

def run_main():
    cam = CameraStream(src=0).start()
    prev_time = time.time()
    
    try:
        while not cam.stopped:
            frame_raw = cam.read()
            if frame_raw is None: continue
            
            h_f, w_f = frame_raw.shape[:2]
            display_frame = frame_raw.copy()
            screen_center_x = w_f // 2

            cached_boxes = vision.predict(frame_raw, conf_threshold=0.4)
            
            target_point = None
            best_conf = 0
            current_label = "NONE"

            for box in cached_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = frame_raw[max(0,y1):min(h_f,y2), max(0,x1):min(w_f,x2)]
                
                input_data = preprocess_realtime(roi)
                if input_data is None: continue
                
                res = compiled_cnn([input_data])[cnn_output]
                idx = np.argmax(res[0])
                label_raw = LABELS_CNN.get(idx, "UNK")
                conf_raw = res[0][idx]
                
                if conf_raw < 0.5: continue

                label, conf = smoother.smooth(f"{x1//40}_{y1//40}", label_raw, conf_raw)
                
                name = label.lower()
                if "r1" in name: color, t_type = (255, 255, 0), "R1"
                elif "real" in name: color, t_type = (0, 255, 0), "REAL"
                else: color, t_type = (0, 0, 255), "FAKE"

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, f"{label} {conf:.2f}", (x1, y1-10), 0, 0.5, color, 2)

                if t_type in ["R1", "REAL"] and conf > best_conf:
                    best_conf = conf
                    target_point = ((x1+x2)//2, (y1+y2)//2)
                    current_label = label

            if target_point:
                tx, ty = vision.update_kalman(target_point[0], target_point[1])
                uart.send_error(tx - screen_center_x)
                cv2.circle(display_frame, (tx, ty), 8, (0, 255, 255), -1)
            else:
                uart.send_error(0)

            now = time.time()
            fps = 1 / (now - prev_time)
            prev_time = now
            cv2.putText(display_frame, f"FPS: {fps:.1f} | {current_label}", (10, 30), 0, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("RBC2026_V2.3_REAL", display_frame)
            if cv2.waitKey(1) == ord('q'): break
    finally:
        cam.stop(); uart.stop(); cv2.destroyAllWindows()

if __name__ == "__main__":
    run_main()