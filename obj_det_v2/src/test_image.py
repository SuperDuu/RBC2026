import cv2
import numpy as np
import json
import sys
import time
from pathlib import Path
from openvino.runtime import Core
from vision import RobotVision

BASE_DIR = Path(__file__).resolve().parent.parent
YOLO_XML = str(BASE_DIR / "models" / "best_openvino_model" / "best.xml")
CNN_XML = str(BASE_DIR / "models" / "openvino_cnn_v2_3" / "classifier_v2_3.xml")
LABEL_JSON = str(BASE_DIR / "models" / "labels_v2.3.json")

ie = Core()

try:
    with open(LABEL_JSON, 'r') as f:
        data = json.load(f)
        LABELS_CNN = {int(v): k for k, v in data.items()}
except Exception as e:
    sys.exit()

try:
    cnn_model = ie.read_model(model=CNN_XML)
    compiled_cnn = ie.compile_model(model=cnn_model, device_name="GPU")
    cnn_output = compiled_cnn.output(0)
    vision = RobotVision(YOLO_XML, class_id=0)
except Exception as e:
    sys.exit()

def preprocess_roi(roi_raw):
    if roi_raw is None or roi_raw.size == 0:
        return None
    gray = cv2.cvtColor(roi_raw, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = 64 / max(h, w)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((64, 64), 128, dtype=np.uint8)
    canvas[(64-nh)//2:(64-nh)//2+nh, (64-nw)//2:(64-nw)//2+nw] = resized
    return canvas.reshape(1, 64, 64, 1).astype(np.float32) / 255.0

def get_color_logic(label_name):
    name = label_name.lower()
    if "r1" in name: return (255, 255, 0), "R1"
    elif "real" in name: return (0, 255, 0), "REAL"
    else: return (0, 0, 255), "FAKE"

def main_process(frame):
    if frame is None: return None
    h_orig, w_orig = frame.shape[:2]
    display_frame = frame.copy()
    
    detected_objects = vision.predict(frame, conf_threshold=0.4)

    for obj in detected_objects:
        x1, y1, x2, y2 = map(int, obj.xyxy[0])
        roi_color = frame[max(0, y1):min(h_orig, y2), max(0, x1):min(w_orig, x2)]
        
        cnn_input = preprocess_roi(roi_color)
        
        if cnn_input is not None:
            res = compiled_cnn([cnn_input])[cnn_output]
            idx = np.argmax(res[0])
            label = LABELS_CNN.get(idx, "UNKNOWN")
            conf = res[0][idx]
            
            if conf > 0.4:
                color, t_type = get_color_logic(label)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, f"{label} {conf:.2f}", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                if t_type in ["R1", "REAL"]:
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    vision.update_kalman(cx, cy)

    return display_frame

if __name__ == "__main__":
    IMAGE_PATH = str(BASE_DIR / "datasets" / "user_test" / "img_test_user8.jpg")
    img = cv2.imread(IMAGE_PATH)
    if img is not None:
        result = main_process(img)
        cv2.imshow("RBC2026_V2.3_CLEAN", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()