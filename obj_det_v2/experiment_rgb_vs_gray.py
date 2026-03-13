import tensorflow as tf
from tensorflow.keras import layers, models
import time
import numpy as np

"""
Kịch bản Thực nghiệm cường độ phân tích hiệu năng: Đầu vào Hình ảnh Đa kênh Khách quan (RGB) vs Kênh Đơn Tỉnh (Grayscale) trong CNN định tuyến Tùy chỉnh.
Giải quyết trực tiếp yêu cầu số 6 của Phản biện A về luận chứng cho sự chuyển nhượng Grayscale.
"""

def build_rgb_model(num_classes):
    model = models.Sequential([
        layers.Input(shape=(64, 64, 3)), # <--- Định dạng Đầu vào RGB
        layers.SeparableConv2D(32, (3, 3), padding='same', activation='elu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.SeparableConv2D(64, (3, 3), padding='same', activation='elu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.SeparableConv2D(128, (3, 3), padding='same', activation='elu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='elu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def build_gray_model(num_classes):
    model = models.Sequential([
        layers.Input(shape=(64, 64, 1)), # <--- Định dạng Đầu vào Ảnh Mức Xám
        layers.SeparableConv2D(32, (3, 3), padding='same', activation='elu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.SeparableConv2D(64, (3, 3), padding='same', activation='elu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.SeparableConv2D(128, (3, 3), padding='same', activation='elu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='elu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def measure_throughput(model, input_shape, num_samples=500):
    dummy_data = np.random.rand(num_samples, *input_shape).astype(np.float32)
    # Khởi động trước (Warmup) để quá trình đo thời gian loại bỏ được khoảng trễ lúc hệ thống tải ban đầu
    model.predict(dummy_data[:10], batch_size=1, verbose=0)
    
    start = time.time()
    for i in range(num_samples):
        _ = model.predict(np.expand_dims(dummy_data[i], axis=0), verbose=0)
    end = time.time()
    
    avg_ms = ((end - start) / num_samples) * 1000
    print(f"Bố cục Đầu vào (Input Shape): {input_shape} | Độ trễ Tính toán: {avg_ms:.2f} ms | Tổng khối tham số mạng CNN: {model.count_params()}")

if __name__ == "__main__":
    num_classes = 31 # Giả lập 31 lớp tự hình học theo thực tế chữ Giáp Cốt Robocon 2026
    print("--- Phân tích Hiệu suất Tải lượng Phân Nhánh CNN: Mô hình Hệ RGB so với Hệ Ảnh Mức Xám (Grayscale) ---")
    
    # Bước 1: Thử nghiệm mạng Grayscale (như kiến trúc quy chuẩn hiện hành)
    gray_model = build_gray_model(num_classes)
    measure_throughput(gray_model, (64, 64, 1))
    
    # Bước 2: Thử nghiệm không gian màu cơ bản RGB
    rgb_model = build_rgb_model(num_classes)
    measure_throughput(rgb_model, (64, 64, 3))
    
    print("\nLưu ý: Bài huấn luyện trên CSDL đầy đủ sẽ cho ra so sánh hoàn chỉnh nhất về Độ chính xác % Accuracy, ở đây ta chỉ minh hoạ khả năng tối giản tham số và tăng tốc độ xử lý.")
