import cv2
import numpy as np
from openvino.runtime import Core

class RobotVision:
    def __init__(self, model_path, class_id=0):
        self.ie = Core()
        xml_path = model_path if model_path.endswith('.xml') else f"{model_path}/best.xml"
        self.model_ir = self.ie.read_model(model=xml_path)
        # Chạy YOLO trên CPU để dành iGPU cho CNN (Chiến thuật phân tầng) [cite: 24, 62]
        self.compiled_model = self.ie.compile_model(model=self.model_ir, device_name="CPU")
        self.output_layer = self.compiled_model.output(0)
        
        self.class_id = class_id
        # Khởi tạo bộ lọc Kalman để mịn quỹ đạo [cite: 125, 158]
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

    def predict(self, frame, conf_threshold=0.5, imgsz=512):
        h_orig, w_orig = frame.shape[:2]
        
        # 1. Tối ưu: Letterbox thay vì Resize bóp méo ảnh [cite: 49, 52]
        scale = min(imgsz / h_orig, imgsz / w_orig)
        nh, nw = int(h_orig * scale), int(w_orig * scale)
        input_img = cv2.resize(frame, (nw, nh))
        canvas = np.full((imgsz, imgsz, 3), 128, dtype=np.uint8) # Nền xám trung tính [cite: 52]
        canvas[(imgsz-nh)//2 : (imgsz-nh)//2 + nh, (imgsz-nw)//2 : (imgsz-nw)//2 + nw] = input_img
        
        # 2. Tiền xử lý cho OpenVINO [cite: 30, 144]
        input_data = canvas.transpose((2, 0, 1)).reshape((1, 3, imgsz, imgsz)).astype(np.float32) / 255.0
        
        # 3. Inference
        results = self.compiled_model([input_data])[self.output_layer]
        predictions = np.squeeze(results)
        if predictions.shape[0] < predictions.shape[1]: predictions = predictions.T
        
        raw_boxes, confidences = [], []
        pad_x, pad_y = (imgsz - nw) // 2, (imgsz - nh) // 2

        for pred in predictions:
            scores = pred[4:]
            score = np.max(scores)
            cls_id = np.argmax(scores)
            
            if score > conf_threshold and cls_id == self.class_id:
                xc, yc, w, h = pred[:4]
                # Khôi phục tọa độ về kích thước gốc sau khi bù Letterbox
                x1 = int((xc - w/2 - pad_x) / scale)
                y1 = int((yc - h/2 - pad_y) / scale)
                width = int(w / scale)
                height = int(h / scale)
                raw_boxes.append([x1, y1, width, height])
                confidences.append(float(score))
        
        indices = cv2.dnn.NMSBoxes(raw_boxes, confidences, conf_threshold, 0.45)
        final_boxes = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = raw_boxes[i]
                class DetectedObject:
                    def __init__(self, x1, y1, x2, y2, conf):
                        self.xyxy = [np.array([x1, y1, x2, y2])]
                        self.conf = conf
                final_boxes.append(DetectedObject(x, y, x + w, y + h, confidences[i]))
        return final_boxes

    def update_kalman(self, x, y):
        measured = np.array([[np.float32(x)], [np.float32(y)]])
        self.kalman.correct(measured)
        pred = self.kalman.predict()
        # FIX LỖI: Flatten ma trận dự báo về mảng 1D trước khi ép kiểu int
        px, py = pred.flatten()[:2]
        return int(px), int(py)