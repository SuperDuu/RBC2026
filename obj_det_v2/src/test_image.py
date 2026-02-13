import cv2
import sys
import numpy as np
import json
from pathlib import Path
from openvino.runtime import Core
from vision import RobotVision
from connection import UARTManager
from preprocessing import RobotPreprocessing

BASE_DIR = Path(__file__).resolve().parent.parent
YOLO_XML = str(BASE_DIR / "models" / "best_openvino_model" / "best.xml")
CNN_XML = str(BASE_DIR / "models" / "openvino_cnn_v2" / "classifier_v2.xml")
LABEL_JSON = str(BASE_DIR / "models" / "labels_v2.json")
IMAGE_PATH = str(BASE_DIR / "datasets" / "user_test" / "img_test_user6.jpg")

ie = Core()
try:
    with open(LABEL_JSON, 'r') as f:
        data = json.load(f)
        LABELS_CNN = {int(v): k for k, v in data.items()}
except Exception as e:
    print(f"Lỗi nhãn: {e}"); sys.exit()

cnn_model = ie.read_model(model=CNN_XML)
compiled_cnn = ie.compile_model(model=cnn_model, device_name="CPU")
cnn_output = compiled_cnn.output(0)

vision = RobotVision(YOLO_XML, class_id=0)
uart = UARTManager(port='/dev/ttyUSB0', baud=115200)
preprocessor = RobotPreprocessing()

def classify_roi(roi_color):
    if roi_color is None or roi_color.size == 0: return "EMPTY", 0.0
    gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    max_side = max(h, w)
    canvas = np.zeros((max_side, max_side), dtype=np.uint8)
    canvas[(max_side-h)//2:(max_side-h)//2+h, (max_side-w)//2:(max_side-w)//2+w] = gray
    resized = cv2.resize(canvas, (64, 64), interpolation=cv2.INTER_AREA)
    input_data = resized.reshape(1, 64, 64, 1).astype(np.float32) / 255.0
    res = compiled_cnn([input_data])[cnn_output]
    idx = np.argmax(res)
    return LABELS_CNN.get(idx, f"ID_{idx}"), res[0][idx]

frame = cv2.imread(IMAGE_PATH)
if frame is None:
    print(f"Không tìm thấy ảnh tại: {IMAGE_PATH}"); sys.exit()

h_orig, w_orig, _ = frame.shape
screen_center_x = w_orig // 2
display_frame = frame.copy()
proc_frame = preprocessor.run_pipeline(frame)
detected_boxes = vision.predict(proc_frame, conf_threshold=0.5, imgsz=512)
target_point = None
max_area = 0

for box in detected_boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    roi = proc_frame[max(0,y1):min(h_orig,y2), max(0,x1):min(w_orig,x2)]
    label, conf = classify_roi(roi)
    is_target = "real" in label.lower() or "r1" in label.upper()
    color = (0, 255, 0) if (is_target and conf > 0.85) else (0, 0, 255)
    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(display_frame, f"{label} {conf:.2f}", (x1-10, y1-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    if is_target and conf > 0.85:
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area = area
            target_point = (int((x1 + x2) / 2), int((y1 + y2) / 2))

if target_point:
    px, py = target_point
    error_x = px - screen_center_x
    uart.send_error(error_x) 
    
    cv2.circle(display_frame, (px, py), 12, (0, 255, 255), -1) 
    cv2.line(display_frame, (screen_center_x, h_orig), (px, py), (255, 255, 0), 2)
    cv2.putText(display_frame, f"ERR: {error_x}", (px + 15, py), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
display_resized = cv2.resize(display_frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
print(f"Test kết thúc. Error gửi UART: {error_x if target_point else 0}")
cv2.imshow("Test Robocon 2026 - Image Mode", display_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()