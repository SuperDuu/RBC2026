import os
import sys
import numpy as np
import cv2
from pathlib import Path
import openvino as ov

# Kiểm tra NNCF sau khi bạn đã cài lại bản 2.14.0
try:
    import nncf
    from nncf import QuantizationPreset
except ImportError:
    print("❌ NNCF chưa được cài đặt đúng. Hãy chạy: pip install nncf==2.14.0")
    sys.exit(1)

def create_calibration_dataset(dataset_path: str, input_size: int, max_samples: int):
    dataset_dir = Path(dataset_path)
    # Hỗ trợ cả file lẻ và thư mục
    image_paths = []
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    for ext in extensions:
        image_paths.extend(list(dataset_dir.glob(f"**/{ext}")))
    
    image_paths = sorted(image_paths)[:max_samples]
    
    if not image_paths:
        raise ValueError(f"❌ Không tìm thấy ảnh tại: {dataset_path}")

    print(f"📂 Đang chuẩn bị {len(image_paths)} mẫu ảnh cho calibration...")
    
    calibration_data = []
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None: continue
        
        # Tiền xử lý Letterbox chuẩn YOLO
        h, w = img.shape[:2]
        r = input_size / max(h, w)
        nh, nw = int(h * r), int(w * r)
        img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        
        canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        top = (input_size - nh) // 2
        left = (input_size - nw) // 2
        canvas[top:top+nh, left:left+nw] = img_resized
        
        # BGR to RGB, HWC to NCHW, Normalize 0-1
        input_tensor = canvas[:, :, ::-1].transpose((2, 0, 1))[np.newaxis, ...].astype(np.float32) / 255.0
        calibration_data.append(input_tensor)
        
    return calibration_data

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Quantize YOLO INT8 for RBC2026")
    parser.add_argument("--model", required=True, help="Path to FP16 .xml model")
    parser.add_argument("--output", required=True, help="Path to save INT8 .xml model")
    parser.add_argument("--dataset", required=True, help="Path to calibration images")
    parser.add_argument("--input-size", type=int, default=512, help="Model input size")
    parser.add_argument("--max-samples", type=int, default=300, help="Max samples for calibration")
    args = parser.parse_args()

    # 1. Khởi tạo Core và đọc model
    core = ov.Core()
    print(f"📦 Đang đọc model: {args.model}")
    model = core.read_model(args.model)
    
    # 2. Tạo dữ liệu Calibration
    try:
        calib_list = create_calibration_dataset(args.dataset, args.input_size, args.max_samples)
    except Exception as e:
        print(e)
        return

    # 3. Tạo NNCF Dataset
    # Lưu ý: Nếu model có nhiều input, cần truyền dictionary. YOLO thường chỉ có 1 input.
    nncf_dataset = nncf.Dataset(calib_list)

    print(f"🚀 Bắt đầu quá trình nén INT8 (Subset size: {len(calib_list)})...")
    
    # 4. Thực hiện Quantization
    try:
        # Sử dụng PERFORMANCE để tối ưu FPS cho Robot Humanoid
        quantized_model = nncf.quantize(
            model, 
            nncf_dataset,
            preset=QuantizationPreset.PERFORMANCE,
            subset_size=len(calib_list)
        )

        # 5. Lưu kết quả (OpenVINO 2024+ dùng serialize hoặc save_model)
        out_xml = Path(args.output)
        out_xml.parent.mkdir(parents=True, exist_ok=True)
        ov.save_model(quantized_model, str(out_xml), compress_to_fp16=False)
        
        print(f"\n{'='*50}")
        print(f"✅ THÀNH CÔNG! Model INT8 đã được lưu tại:")
        print(f"👉 {out_xml}")
        print(f"{'='*50}")
        
    except Exception as e:
        print(f"❌ Lỗi trong khi quantize: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()