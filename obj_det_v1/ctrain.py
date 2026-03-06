from ultralytics import YOLO
import torch
import os

def continue_training_rbc():
    # B1 - Thiết kế & Ràng buộc phần cứng (Bảo tồn tinh chỉnh cá nhân)
    if torch.cuda.is_available():
        device = '0'
        torch.cuda.empty_cache() 
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    else:
        print("Không có GPU để chạy!")
        return

    # B2 - Thảo luận: Load model từ kết quả tốt nhất trước đó
    # Lưu ý: Dùng file best.pt để bắt đầu từ đỉnh cao nhất của lần train trước
    model_path = '/home/du/Desktop/RBC2026/obj_det_v1/runs/detect/RBC2026/yolov8n_512px/weights/best.pt'
    
    if not os.path.exists(model_path):
        print(f"Lỗi: Không tìm thấy file tại {model_path}")
        return

    model = YOLO(model_path)

    # B3 - Kiểm chứng & Tích hợp (Cấu hình chạy tiếp 50 Epoch)
    model.train(
        # --- Giữ nguyên cấu trúc dữ liệu ---
        data='datasets/data.yaml', 
        epochs=50,                # Train thêm đúng 50 epoch nữa
        imgsz=512,               
        batch=20,                
        device=device,                    
        workers=4,                
        project='RBC2026',                
        name='yolov8n_512px_v2',  # Lưu vào folder mới để so sánh
        
        # --- PHẢN BIỆN: THAY ĐỔI ĐỂ ÉP CHẠY TIẾP ---
        patience=0,               # Tắt EarlyStopping, ép chạy đủ 50 epoch
        lr0=0.001,                # Giảm 10 lần so với ban đầu để tinh chỉnh (Fine-tune)
        cos_lr=True,              # Vẫn dùng Cosine Learning Rate để hội tụ êm
        
        # --- Giữ nguyên các thông số Augmentation mạnh của bạn ---
        mosaic=1.0,           
        copy_paste=0.3,       
        mixup=0.15,           
        degrees=10.0,         
        scale=0.2,            
        translate=0.1,        
        shear=2.0,            
        perspective=0.001,    
        fliplr=0.5,           
        flipud=0.5,           
        hsv_h=0.015,          
        hsv_s=0.7,            
        hsv_v=0.4,            

        # --- Giai đoạn hội tụ cuối ---
        close_mosaic=30           # Tắt mosaic trong 15 epoch cuối để Box bám chặt hơn
    )

if __name__ == '__main__':
    continue_training_rbc()