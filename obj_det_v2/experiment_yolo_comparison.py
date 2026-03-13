import time
import cv2
import numpy as np
from openvino.runtime import Core

"""
Kịch bản Thực nghiệm so sánh giữa YOLOv5n, YOLOv7n, YOLOv8n và YOLOv8s.
Giải quyết trực tiếp yêu cầu số 1 của Phản biện A về tính hợp lý khi chọn YOLOv8n.
"""

def evaluate_yolo_model(model_path, test_images, device="GPU"):
    print(f"--- Đang phân tích mô hình {model_path} trên {device} ---")
    ie = Core()
    try:
        model_ir = ie.read_model(model=model_path)
        # Sử dụng PERFORMANCE_HINT = LATENCY để tối ưu thời gian phản hồi cho từng khung hình
        compiled_model = ie.compile_model(model=model_ir, device_name=device, config={"PERFORMANCE_HINT": "LATENCY"})
        output_layer = compiled_model.output(0)
    except Exception as e:
        print(f"Lỗi: Không thể tải mô hình {model_path}: {e}")
        return

    # Khởi động trước vùng nhớ mốc ban đầu (Warmup)
    dummy_input = np.zeros((1, 3, 512, 512), dtype=np.float32)
    for _ in range(5):
        compiled_model([dummy_input])

    latencies = []
    
    for img in test_images:
        # Tiền xử lý khung hình giả lập
        input_data = cv2.resize(img, (512, 512))
        input_data = input_data.transpose((2, 0, 1)).reshape((1, 3, 512, 512)).astype(np.float32) / 255.0
        
        # Bắt đầu suy luận (Inference)
        start_time = time.time()
        _ = compiled_model([input_data])[output_layer]
        latencies.append((time.time() - start_time) * 1000) # Tính bằng mili giây
        
    avg_latency = np.mean(latencies)
    fps = 1000.0 / avg_latency
    print(f"Kết quả phân tích {model_path}:\n - Độ trễ trung bình (Avg Latency) = {avg_latency:.2f} ms\n - Tốc độ phân tích (FPS) = {fps:.2f}\n")

if __name__ == "__main__":
    # Khởi tạo mô phỏng 100 khung hình dạng kích cỡ camera ngẫu nhiên.
    # (*Lưu ý cho tác giả: Trong thực tế, hãy nạp tập dữ liệu thực nghiệm để tính chính xác mAP tại đây)
    dummy_test_images = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(100)]
    
    # Đường dẫn đến các mô hình định dạng ONNX/OpenVINO IR cần đối chiếu tương quan
    models_to_test = [
        "models/yolov5n.xml", 
        "models/yolov7n.xml", 
        "models/best_openvino_model_int8/best_int8.xml", # Đại diện YOLOv8n
        "models/yolov8s.xml"
    ]
    
    print("Bắt đầu khởi động thực nghiệm Đánh giá các biến thể mô hình YOLO...")
    for model in models_to_test:
        # Trong bối cảnh thực tế hệ thống chạy Surface Go 2, kiến trúc OpenVINO ưu tiên iGPU 
        evaluate_yolo_model(model, dummy_test_images, device="GPU") 
