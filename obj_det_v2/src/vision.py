import cv2
import numpy as np
from openvino.runtime import Core

class RobotVision:
    def __init__(self, model_path, class_id=0):
        self.ie = Core()
        xml_path = model_path if model_path.endswith('.xml') else f"{model_path}/best.xml"
        self.model_ir = self.ie.read_model(model=xml_path)
        self.compiled_model = self.ie.compile_model(model=self.model_ir, device_name="CPU")
        self.output_layer = self.compiled_model.output(0)
        
        self.class_id = class_id
        # Kalman Filter: Thông số tối ưu cho chuyển động Robot
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

    def predict(self, frame, conf_threshold=0.5, imgsz=512):
        h_orig, w_orig = frame.shape[:2]
        input_img = cv2.resize(frame, (imgsz, imgsz))
        input_img = input_img.transpose((2, 0, 1)).reshape((1, 3, imgsz, imgsz)).astype(np.float32) / 255.0
        
        results = self.compiled_model([input_img])[self.output_layer]
        predictions = np.squeeze(results)
        if predictions.shape[0] < predictions.shape[1]: predictions = predictions.T
        
        raw_boxes, confidences = [], []
        for pred in predictions:
            scores = pred[4:]
            score = np.max(scores)
            cls_id = np.argmax(scores)
            
            if score > conf_threshold and cls_id == self.class_id:
                xc, yc, w, h = pred[:4]
                x1 = int((xc - w/2) * w_orig / imgsz)
                y1 = int((yc - h/2) * h_orig / imgsz)
                width = int(w * w_orig / imgsz)
                height = int(h * h_orig / imgsz)
                raw_boxes.append([x1, y1, width, height])
                confidences.append(float(score))
        
        indices = cv2.dnn.NMSBoxes(raw_boxes, confidences, conf_threshold, 0.45)
        
        final_boxes = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = raw_boxes[i]
                # Cấu trúc phẳng, trực tiếp có thuộc tính .xyxy
                class DetectedObject:
                    def __init__(self, x1, y1, x2, y2, conf):
                        self.xyxy = [np.array([x1, y1, x2, y2])]
                        self.conf = conf
                final_boxes.append(DetectedObject(x, y, x + w, y + h, confidences[i]))
        
        return final_boxes # Trả về list các DetectedObject

    def update_kalman(self, x, y):
        measured = np.array([[np.float32(x)], [np.float32(y)]])
        self.kalman.correct(measured)
        pred = self.kalman.predict()
        return int(pred[0]), int(pred[1])