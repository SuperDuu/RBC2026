from ultralytics import YOLO
import torch
import os

def train_best_model():
    # B1 - Thiết kế & Ràng buộc phần cứng (Giữ nguyên tinh chỉnh cá nhân của bạn)
    if torch.cuda.is_available():
        device = '0'
        torch.cuda.empty_cache() 
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    else:
        print("No GPU!")
        return

    # Khởi tạo mô hình Nano
    model = YOLO('/home/du/Desktop/RBC2026/yolov8n.pt') 

    # B2 - Thảo luận & Tích hợp tham số (Phản biện & Nâng cấp)
    model.train(
        # --- Cấu trúc hệ thống & Tài nguyên ---
        data='datasets/data.yaml', 
        epochs=150,               # Bạn để 105, tôi giữ 105 (nhưng khuyến cáo 150 nếu mAP chưa đạt)
        imgsz=512,               # Giữ nguyên 512px theo cấu hình của bạn
        batch=20,                # Giữ nguyên batch 20 theo phần cứng của bạn
        device=device,                    
        workers=4,                
        project='RBC2026',                
        name='yolov8n_512px_v3', 
        patience=0,              
        save=True,
        cache=False,              
        overlap_mask=True,
        
        # --- Tối ưu hóa Learning Rate ---
        lr0=0.01,                 
        cos_lr=True,              

        # --- CHIẾN LƯỢC ONLINE AUGMENTATION MẠNH (Thay thế việc x3 ảnh trên Roboflow) ---
        mosaic=1.0,           # Ghép ảnh để nhận diện đa quy mô
        copy_paste=0.2,       # Giữ nguyên mức 0.3 bạn đã chọn
        mixup=0.1,           # Tăng lên 0.15 giúp robot nhận diện trong môi trường chồng lấn
        degrees=20.0,         # Xoay nhẹ ảnh (bù đắp robot nghiêng)
        scale=0.2,            # Tăng biên độ zoom để nhận diện xa/gần tốt hơn
        translate=0.1,        # Dịch chuyển nhẹ để tránh vật thể luôn ở tâm
        shear=2.0,            # Cắt nghiêng giả lập góc nhìn camera robot
        perspective=0.001,    # Biến đổi phối cảnh (cần thiết cho Humanoid)
        fliplr=0.5,           # Lật trái/phải 50%
        
        # --- Bù đắp ánh sáng & Màu sắc (Rất quan trọng cho Robot thực tế) ---
        hsv_h=0.015,          # Hue
        hsv_s=0.7,            # Saturation (độ bão hòa)
        hsv_v=0.4,            # Value (độ sáng) - giúp nhận diện tốt khi tối/ngược sáng

        # --- Giai đoạn kết thúc (B3 - Kiểm chứng) ---
        close_mosaic=50       # Giữ nguyên 20 epoch cuối tắt Augmentation để hội tụ chính xác
    )

if __name__ == '__main__':
    train_best_model()