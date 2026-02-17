"""
Main application module for RBC2026 Robocon Vision System.

This module provides the main RoboconSystem class that orchestrates
detection, classification, tracking, and control.
"""

import cv2
import time
import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from openvino.runtime import Core

from vision import RobotVision, DetectedObject
from connection import UARTManager
from camera import CameraStream
from label_smoother import LabelSmoother
from utils import (
    preprocess_roi_for_cnn
)
from config_manager import ConfigManager


class RoboconSystem:
    """
    Main system class for Robocon vision and control.
    
    Orchestrates YOLO detection, CNN classification, Kalman tracking,
    and UART communication.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Robocon system.
        
        Args:
            config_path: Path to config YAML file. If None, uses default.
        
        Raises:
            RuntimeError: If initialization fails
        """
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(f"{__name__}.RoboconSystem")
        self.frame_counters = {}
        # Load configuration
        try:
            self.config = ConfigManager(config_path)
            self.logger.info("Configuration loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise RuntimeError(f"Configuration loading failed: {e}") from e
        
        # Initialize components
        self._init_models()
        self._init_hardware()
        self._init_tracking()
        
        self.logger.info("RoboconSystem initialized successfully")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_level = getattr(logging, "INFO", logging.INFO)
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.StreamHandler(),
            ]
        )
    
    def _init_models(self) -> None:
        """Initialize YOLO and CNN models."""
        try:
            # Load CNN labels
            labels_path = self.config.get_path("labels_json")
            with open(labels_path, 'r', encoding='utf-8') as f:
                labels_data = json.load(f)
                self.labels_cnn = {int(v): k for k, v in labels_data.items()}
            
            self.logger.info(f"Loaded {len(self.labels_cnn)} CNN labels")
            
            # Initialize OpenVINO Core
            self.ie = Core()
            cache_path = Path("model_cache")
            cache_path.mkdir(exist_ok=True)
            self.ie.set_property("GPU", {"CACHE_DIR": str(cache_path)})
            # Load and compile CNN model
            cnn_xml = self.config.get_path("cnn_xml")
            cnn_device = self.config.get("models.cnn.device", "GPU")
            
            self.logger.info(f"Loading CNN model from {cnn_xml} on {cnn_device}")
            cnn_model = self.ie.read_model(model=cnn_xml)

            self.compiled_cnn = self.ie.compile_model(
                model=cnn_model, 
                device_name=cnn_device,
                config={"PERFORMANCE_HINT": "LATENCY"}
            )


            # self.compiled_cnn = self.ie.compile_model(model=cnn_model, device_name=cnn_device)
            self.cnn_output = self.compiled_cnn.output(0)
            
            # Initialize YOLO vision system
            yolo_xml = self.config.get_path("yolo_xml")
            yolo_device = self.config.get("models.yolo.device", "CPU")
            yolo_class_id = self.config.get("models.yolo.class_id", 0)
            
            self.logger.info(f"Loading YOLO model from {yolo_xml} on {yolo_device}")
            self.vision = RobotVision(yolo_xml, class_id=yolo_class_id, device=yolo_device)
            
            self.logger.info("All models loaded successfully")
        
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {e}")
            raise RuntimeError(f"Model initialization failed: {e}") from e
    
    def _init_hardware(self) -> None:
        """Initialize camera and UART hardware."""
        try:
            # Initialize camera
            camera_id = self.config.get("hardware.camera.device_id", 0)
            buffer_size = self.config.get("hardware.camera.buffer_size", 1)
            
            self.camera = CameraStream(src=camera_id, buffer_size=buffer_size)
            
            # Initialize UART
            uart_port = self.config.get("hardware.uart.port", "/dev/ttyUSB0")
            uart_baud = self.config.get("hardware.uart.baudrate", 115200)
            
            self.uart = UARTManager(port=uart_port, baudrate=uart_baud)
            
            self.logger.info("Hardware initialized successfully")
        
        except Exception as e:
            self.logger.error(f"Failed to initialize hardware: {e}")
            raise RuntimeError(f"Hardware initialization failed: {e}") from e
    
    def _init_tracking(self) -> None:
        """Initialize tracking components."""
        window_size = self.config.get("detection.label_smoothing.window_size", 5)
        self.smoother = LabelSmoother(window_size=window_size)
        
        # Get configuration values
        self.conf_threshold_yolo = self.config.get("models.yolo.conf_threshold", 0.4)
        self.conf_threshold_cnn = self.config.get("models.cnn.conf_threshold", 0.5)
        self.target_types = self.config.get("classification.target_types", ["R1", "REAL"])
        self.color_map = {
            "R1": tuple(self.config.get("classification.colors.R1", [255, 255, 0])),
            "REAL": tuple(self.config.get("classification.colors.REAL", [0, 255, 0])),
            "FAKE": tuple(self.config.get("classification.colors.FAKE", [0, 0, 255]))
        }
        self.grid_size = self.config.get("detection.label_smoothing.grid_size", 40)
    
    def _classify_roi(self, roi: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Classify ROI using CNN model.
        
        Args:
            roi: Region of Interest image (BGR format)
        
        Returns:
            Tuple of (label, confidence) or (None, 0.0) if classification fails
        """
        try:
            # Preprocess ROI
            input_data = preprocess_roi_for_cnn(roi)
            if input_data is None:
                return None, 0.0
            
            # Run CNN inference
            result = self.compiled_cnn([input_data])[self.cnn_output]
            idx = np.argmax(result[0])
            label = self.labels_cnn.get(idx, "UNKNOWN")
            confidence = float(result[0][idx])
            
            return label, confidence
        
        except Exception as e:
            self.logger.error(f"Error classifying ROI: {e}")
            return None, 0.0
    


    def _process_detections(
        self, 
        frame: np.ndarray, 
        detections: List[DetectedObject]
    ) -> Tuple[Optional[Tuple[int, int]], str]:
        """
        Hàm xử lý detections tích hợp:
        1. Lọc box lồng nhau (IoU Filter).
        2. Chạy CNN mỗi 3 frame để ổn định FPS.
        3. Logic phân loại màu sắc và khóa mục tiêu.
        """
        h_frame, w_frame = frame.shape[:2]
        target_point = None
        best_conf = 0.0
        current_label = "NONE"
        
        # --- BƯỚC 1: LỌC BOX LỒNG NHAU (IOU FILTER) ---
        # Sắp xếp theo diện tích từ lớn đến nhỏ
        detections = sorted(detections, key=lambda x: (x.xyxy[0][2]-x.xyxy[0][0])*(x.xyxy[0][3]-x.xyxy[0][1]), reverse=True)
        filtered_detections = []

        for box_a in detections:
            is_nested = False
            a_x1, a_y1, a_x2, a_y2 = map(int, box_a.xyxy[0])
            area_a = (a_x2 - a_x1) * (a_y2 - a_y1)

            for box_b in filtered_detections:
                b_x1, b_y1, b_x2, b_y2 = map(int, box_b.xyxy[0])
                
                # Tính diện tích vùng giao nhau
                inter_x1 = max(a_x1, b_x1)
                inter_y1 = max(a_y1, b_y1)
                inter_x2 = min(a_x2, b_x2)
                inter_y2 = min(a_y2, b_y2)
                
                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    # Nếu diện tích giao thoa chiếm > 80% diện tích box nhỏ -> Coi là lồng nhau
                    if inter_area / min(area_a, (b_x2-b_x1)*(b_y2-b_y1)) > 0.8:
                        is_nested = True
                        break
            
            if not is_nested:
                filtered_detections.append(box_a)

        # Giới hạn xử lý tối đa 5 vật thể tốt nhất sau khi lọc để duy trì FPS
        filtered_detections = sorted(filtered_detections, key=lambda x: x.conf, reverse=True)[:5]

        # --- BƯỚC 2: PHÂN LOẠI CNN VÀ LOGIC MÀU SẮC ---
        for box in filtered_detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Grid ID để tracking nội bộ (dùng lưới 50px để ổn định ID)
            track_id = f"{cx//50}_{cy//50}"
            
            if track_id not in self.frame_counters:
                self.frame_counters[track_id] = {'count': 0, 'label': "UNKNOWN", 'conf': 0.0}
            
            # CHỈ CHẠY CNN MỖI 3 FRAME
            if self.frame_counters[track_id]['count'] % 3 == 0:
                roi = frame[max(0, y1):min(h_frame, y2), max(0, x1):min(w_frame, x2)]
                
                if roi.size > 0:
                    label_raw, conf_raw = self._classify_roi(roi)
                    
                    if label_raw is not None and conf_raw >= self.conf_threshold_cnn:
                        # Làm mượt nhãn bằng LabelSmoother có sẵn
                        box_id = f"{x1//self.grid_size}_{y1//self.grid_size}"
                        label, conf = self.smoother.smooth(box_id, label_raw, conf_raw)
                        
                        self.frame_counters[track_id]['label'] = label
                        self.frame_counters[track_id]['conf'] = conf
            
            # Lấy kết quả từ cache (cho frame skip) hoặc mới cập nhật
            label = self.frame_counters[track_id]['label']
            conf = self.frame_counters[track_id]['conf']
            self.frame_counters[track_id]['count'] += 1
            
            if label == "UNKNOWN" or conf < self.conf_threshold_cnn:
                continue
            
            # --- LOGIC MÀU SẮC & PHÂN LOẠI (Theo yêu cầu của bạn) ---
            name_lower = label.lower()
            if "r1" in name_lower:
                color, t_type = self.color_map["R1"], "R1"
            elif "real" in name_lower:
                color, t_type = self.color_map["REAL"], "REAL"
            else:
                color, t_type = self.color_map["FAKE"], "FAKE"
            
            # Vẽ visualization
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Khóa mục tiêu tốt nhất
            if t_type in self.target_types and conf > best_conf:
                best_conf = conf
                target_point = (cx, cy)
                current_label = label
        
        # Dọn dẹp cache ID cũ để tránh Memory Leak
        if len(self.frame_counters) > 50:
            self.frame_counters.clear()
            
        return target_point, current_label
    def _draw_tracking(
        self, 
        frame: np.ndarray, 
        target_point: Tuple[int, int], 
        error_x: int
    ) -> None:
        """
        Draw tracking visualization on frame.
        
        Args:
            frame: Frame to draw on
            target_point: Target point (x, y)
            error_x: Error in x direction
        """
        h_frame, w_frame = frame.shape[:2]
        screen_center_x = w_frame // 2
        
        # Update Kalman filter and get predicted position
        tx, ty = self.vision.update_kalman(target_point[0], target_point[1])
        
        # Draw line from center bottom to target
        cv2.line(
            frame, 
            (screen_center_x, h_frame), 
            (tx, ty), 
            (255, 255, 0), 
            2
        )
        
        # Draw target circle
        cv2.circle(frame, (tx, ty), 8, (0, 255, 255), -1)
        
        # Draw error text
        cv2.putText(
            frame, 
            f"ERR: {error_x}", 
            (tx + 15, ty), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (0, 255, 255), 
            2
        )
    
    def run(self) -> None:
        """Run main detection and tracking loop."""
        self.logger.info("Starting main loop...")
        
        self.camera.start()
        prev_time = time.time()
        prev_time = time.time()
        fps_accum = 0
        frame_count = 0
        avg_fps = 0.0
        last_fps_update = time.time()
        window_name = self.config.get("display.window_name", "RBC2026")
        show_fps = self.config.get("display.show_fps", True)
        font_scale = self.config.get("display.font_scale", 0.7)
        
        try:
            while not self.camera.stopped:
                frame = self.camera.read()
                if frame is None:
                    continue
                
                h_frame, w_frame = frame.shape[:2]
                screen_center_x = w_frame // 2
                
                # Run YOLO detection
                detections = self.vision.predict(
                    frame, 
                    conf_threshold=self.conf_threshold_yolo
                )
                
                # Process detections: classify and find best target
                target_point, current_label = self._process_detections(frame, detections)
                
                # Handle tracking
                if target_point:
                    error_x = target_point[0] - screen_center_x
                    self.uart.send_error(error_x)
                    self._draw_tracking(frame, target_point, error_x)
                else:
                    self.uart.send_error(0)
                
                # Calculate and display FPS (inline for performance)
                if show_fps:
                        # CẬP NHẬT LOGIC FPS
                    now = time.time()
                    frame_count += 1
                    fps_accum += 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0
                    prev_time = now
                    
                    # Cập nhật hiển thị mỗi 1 giây
                    if now - last_fps_update >= 0.25:
                        avg_fps = fps_accum / frame_count
                        fps_accum = 0
                        frame_count = 0
                        last_fps_update = now
                    
                    if self.config.get("display.show_fps", True):
                        cv2.putText(frame, f"AVG FPS: {avg_fps:.1f}", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow(window_name, frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.logger.info("Quit signal received")
                    break
        
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}", exc_info=True)
        
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        self.logger.info("Cleaning up resources...")
        
        try:
            self.camera.stop()
            self.uart.stop()
            cv2.destroyAllWindows()
            
            self.logger.info("Cleanup completed")
        
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


def main():
    """Main entry point."""
    import sys
    
    try:
        # Allow config path from command line
        config_path = sys.argv[1] if len(sys.argv) > 1 else None
        
        system = RoboconSystem(config_path=config_path)
        system.run()
    
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
