from ultralytics import YOLO
import shutil
import os

model_path = '/home/du/Desktop/RBC2026/obj_det_v1/runs/detect/RBC2026/yolov8n_512px2/weights/best.pt'
model = YOLO(model_path)
export_path = model.export(format='openvino', half=True)

target_dir = 'models'
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

shutil.move(export_path, os.path.join(target_dir, os.path.basename(export_path)))

print(f"--- Đã hoàn thành! Model OpenVINO FP16 nằm tại: {target_dir} ---")