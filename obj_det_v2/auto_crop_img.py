import cv2
import os
import numpy as np

IMG_DIR = 'datasets/train/images'   
LABEL_DIR = 'datasets/train/labels' 
OUTPUT_DIR = 'CNN_Dataset_Raw'      
TARGET_SIZE = 64                    

def resize_with_padding(img, target_size=64):
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    padded_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    padded_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
    return padded_img

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Đang bắt đầu cắt ảnh...")
count = 0

label_files = [f for f in os.listdir(LABEL_DIR) if f.endswith('.txt')]

for label_file in label_files:
    img_name = label_file.replace('.txt', '.jpg') 
    img_path = os.path.join(IMG_DIR, img_name)
    img = cv2.imread(img_path)
    
    if img is None:
        img_path = os.path.join(IMG_DIR, label_file.replace('.txt', '.png'))
        img = cv2.imread(img_path)
        if img is None: continue

    h_orig, w_orig, _ = img.shape

    with open(os.path.join(LABEL_DIR, label_file), 'r') as f:
        for line in f:
            data = line.split()
            if len(data) < 5: continue 
            
            try:
                x_c = float(data[1])
                y_c = float(data[2])
                w_b = float(data[3])
                h_b = float(data[4])
            except ValueError:
                continue

            x1 = int((x_c - w_b/2) * w_orig)
            y1 = int((y_c - h_b/2) * h_orig)
            x2 = int((x_c + w_b/2) * w_orig)
            y2 = int((y_c + h_b/2) * h_orig)
            
            crop = img[max(0,y1):min(h_orig,y2), max(0,x1):min(w_orig,x2)]
            
            if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5: 
                continue
            
            final_img = resize_with_padding(crop, TARGET_SIZE)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"crop_{count:06d}.jpg"), final_img)
            count += 1

print(f"Xong! Đã cắt được {count} ảnh vào thư mục {OUTPUT_DIR}")