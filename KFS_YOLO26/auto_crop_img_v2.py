import cv2
import os
import numpy as np
from pathlib import Path

dataset_path = Path('datasets/advimg') 
output_path = Path('cnn_dataset_64x64')
target_size = 64
class_names = [str(i) for i in range(32)] 

def square_pad_and_resize(img, size=64):
    h, w = img.shape[:2]
    if h == 0 or w == 0: return None
    
    max_side = max(h, w)
    canvas = np.zeros((max_side, max_side, 3), dtype=np.uint8)
    offset_h = (max_side - h) // 2
    offset_w = (max_side - w) // 2
    canvas[offset_h:offset_h+h, offset_w:offset_w+w] = img
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)

for name in class_names:
    (output_path / name).mkdir(parents=True, exist_ok=True)

img_dir = dataset_path / 'images'
lbl_dir = dataset_path / 'labels'
print("Bắt đầu xử lý dữ liệu hỗn hợp (Bbox & Polygon)...")
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

for lbl_file in lbl_dir.glob('*.txt'):
    img_file = None
    for ext in valid_extensions:
        temp_file = img_dir / f"{lbl_file.stem}{ext}"
        if temp_file.exists():
            img_file = temp_file
            break
            
    if img_file is None:
        continue
        
    image = cv2.imread(str(img_file))
    if image is None: continue
    h_img, w_img = image.shape[:2]
    
    with open(lbl_file, 'r') as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        data = list(map(float, line.strip().split()))
        if not data: continue
        
        class_id = int(data[0])
        coords = data[1:]
        
        if len(coords) == 4:
            x_c, y_c, w, h = coords
            x1 = int((x_c - w/2) * w_img)
            y1 = int((y_c - h/2) * h_img)
            x2 = int((x_c + w/2) * w_img)
            y2 = int((y_c + h/2) * h_img)
            
        elif len(coords) > 4:
            xs = coords[0::2] 
            ys = coords[1::2] 
            x1, x2 = int(min(xs) * w_img), int(max(xs) * w_img)
            y1, y2 = int(min(ys) * h_img), int(max(ys) * h_img)
        else:
            continue
        
        crop = image[max(0, y1):min(h_img, y2), max(0, x1):min(w_img, x2)]
        if crop.size == 0: continue
        final_crop = square_pad_and_resize(crop, target_size)
        
        if final_crop is not None:
            class_folder = class_names[class_id]
            save_name = f"{lbl_file.stem}_{i}.jpg"
            cv2.imwrite(str(output_path / class_folder / save_name), final_crop)

print(f"Xong! Toàn bộ ảnh 64x64 đã nằm tại: {output_path}")