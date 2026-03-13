import time
import numpy as np
from openvino.runtime import Core

"""
Kịch bản Thực nghiệm đánh giá Kiến trúc Phân loại 1 giai đoạn YOLO so với Đường ống YOLO+CNN đề xuất.
Giải quyết trực tiếp yêu cầu số 2 của Phản biện A liên quan đến sự lựa chọn kiến trúc two-stage.
"""

def evaluate_single_stage(yolo_cls_model_path, test_images, device="CPU"):
    print(f"\n--- Đánh giá độ trễ Hệ thống phân loại 1 Giai đoạn (Single-stage) ({yolo_cls_model_path}) trên thiết bị {device} ---")
    ie = Core()
    try:
        model_ir = ie.read_model(model=yolo_cls_model_path)
        compiled_model = ie.compile_model(model=model_ir, device_name=device)
        output_layer = compiled_model.output(0)
    except Exception as e:
        print(f"Không thể khởi tạo mô hình một giai đoạn: {e}")
        return

    latencies = []
    
    for img in test_images:
        # Tiền xử lý theo tiêu chuẩn phân loại YOLO
        input_data = cv2.resize(img, (224, 224)) 
        input_data = input_data.transpose((2, 0, 1)).reshape((1, 3, 224, 224)).astype(np.float32) / 255.0
        
        # Truy xuất đặc trưng và xác suất
        start_time = time.time()
        _ = compiled_model([input_data])[output_layer]
        latencies.append((time.time() - start_time) * 1000)
        
    avg_latency = np.mean(latencies)
    print(f"Độ trễ trung bình hệ 1 Giai đoạn = {avg_latency:.2f} ms | Số khung hình/s (FPS) = {1000.0/avg_latency:.2f}")

def evaluate_two_stage(yolo_det_path, cnn_cls_path, test_images, yolo_device="GPU", cnn_device="CPU"):
    print(f"\n--- Đánh giá độ trễ Đường ống 2 Giai đoạn trên hệ thống kết hợp YOLO({yolo_device}) + CNN({cnn_device}) ---")
    ie = Core()
    try:
        yolo_ir = ie.read_model(model=yolo_det_path)
        cnn_ir = ie.read_model(model=cnn_cls_path)
        yolo_compiled = ie.compile_model(model=yolo_ir, device_name=yolo_device, config={"PERFORMANCE_HINT": "LATENCY"})
        cnn_compiled = ie.compile_model(model=cnn_ir, device_name=cnn_device)
        yolo_out = yolo_compiled.output(0)
        cnn_out = cnn_compiled.output(0)
    except Exception as e:
        print(f"Không thể khởi tạo đường ống hai giai đoạn: {e}")
        return

    latencies = []
    import cv2
    
    for img in test_images:
        start_time = time.time()
        
        # Giai đoạn 1: Phát hiện ROI và Định hướng đối tượng (Trên iGPU)
        input_detect = cv2.resize(img, (512, 512))
        input_detect = input_detect.transpose((2, 0, 1)).reshape((1, 3, 512, 512)).astype(np.float32) / 255.0
        _ = yolo_compiled([input_detect])[yolo_out]
        
        # Quá trình mô phỏng thao tác Crop + Giai đoạn 2: Phân loại ký tự (Trên CPU)
        # Giả định YOLO đã thực thi thành công và trả về ROI crop kích thước 64x64 Grayscale
        dummy_crop = np.random.rand(1, 1, 64, 64).astype(np.float32)
        _ = cnn_compiled([dummy_crop])[cnn_out]
        
        latencies.append((time.time() - start_time) * 1000)
        
    avg_latency = np.mean(latencies)
    print(f"Độ trễ trung bình hệ 2 Giai đoạn = {avg_latency:.2f} ms | Tổng FPS Hệ thống thực = {1000.0/avg_latency:.2f}")

if __name__ == "__main__":
    import cv2
    dummy_test_images = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(100)]
    
    print("Khởi động thực nghiệm xác định tính ưu việt của hệ thống hai giai đoạn...")
    
    # Giả định tồn tại file YOLOv8-cls xuất ra OpenVINO để đánh giá tổng hợp
    yolo_cls_path = "models/yolov8n_cls.xml" 
    
    # Đường dẫn ứng với Hệ thống hiện hữu trong thư mục tài nguyên dự án
    yolo_det_path = "models/best_openvino_model_int8/best_int8.xml"
    cnn_path = "models/openvino_cnn_v2_4/classifier_v2_4.xml"
    
    evaluate_single_stage(yolo_cls_path, dummy_test_images, device="CPU")
    evaluate_two_stage(yolo_det_path, cnn_path, dummy_test_images, yolo_device="GPU", cnn_device="CPU")
