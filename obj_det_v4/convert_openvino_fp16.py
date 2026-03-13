from ultralytics import YOLO
import shutil
import os

model_path = '/home/du/Desktop/RBC2026/obj_det_v4/models/best.pt'
model = YOLO(model_path)

# The model class now handles fuse, yaml, stride, and names internally.
export_path = model.export(format='openvino', half=True)

target_dir = 'models'
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

target_path = os.path.join(target_dir, os.path.basename(export_path))
if os.path.exists(target_path):
    shutil.rmtree(target_path)
shutil.move(export_path, target_path)

print(f"--- Đã hoàn thành! Model OpenVINO FP16 nằm tại: {target_dir} ---")