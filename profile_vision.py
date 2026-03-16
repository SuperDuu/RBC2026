"""
Model Performance Profiling Script for RBC2026

Measures inference time for YOLO and CNN models across all project configurations.
Based on user-provided template and tailored for the RBC2026 project structure.
"""

import cv2
import numpy as np
import time
import statistics
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Optional
from openvino.runtime import Core

# Add project root to path to allow importing from core
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from core.vision import RobotVision
from core.utils import preprocess_roi_for_cnn
from core.config_manager import ConfigManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class ModelProfiler:
    """Profiles YOLO and CNN model performance."""
    
    def __init__(self, folder_name: str):
        """Initialize profiler for a specific folder configuration."""
        self.folder_name = folder_name
        config_path = project_root / folder_name / "config.yaml"
        
        # Initialize ConfigManager with the specific config
        self.config = ConfigManager(str(config_path))
        self.ie = Core()
        
        self.has_cnn = "cnn_xml" in self.config.get("paths.models", {}) or "cnn" in self.config.get("models", {})
        self.has_yolo = "yolo_xml" in self.config.get("paths.models", {}) or "yolo" in self.config.get("models", {})
        
        # Override for SpearHead_YOLO26 special model path
        if folder_name == "SpearHead_YOLO26":
            self.yolo_xml = str(project_root / "SpearHead_YOLO26/models/yolo26n_openvino_int8/best_int8.xml")
        else:
            self.yolo_xml = self.config.get_path("paths.models.yolo_xml")
            
        self.cnn_xml = self.config.get_path("paths.models.cnn_xml")
        self.labels_path = self.config.get_path("paths.models.labels_json")
        
        self._init_models()
    
    def _init_models(self):
        """Initialize YOLO and CNN models if paths are valid."""
        # Setup CNN
        if self.cnn_xml and Path(self.cnn_xml).exists():
            cnn_device = self.config.get("models.cnn.device", "CPU")
            logger.info(f"[{self.folder_name}] Loading CNN model: {self.cnn_xml} on {cnn_device}")
            try:
                cnn_model = self.ie.read_model(model=self.cnn_xml)
                self.compiled_cnn = self.ie.compile_model(model=cnn_model, device_name=cnn_device)
                self.cnn_output = self.compiled_cnn.output(0)
            except Exception as e:
                logger.error(f"Failed to load CNN for {self.folder_name}: {e}")
                self.has_cnn = False
        else:
            self.has_cnn = False

        # Setup YOLO
        if self.yolo_xml and Path(self.yolo_xml).exists():
            yolo_device = self.config.get("models.yolo.device", "CPU")
            yolo_class_id = self.config.get("models.yolo.class_id", 0)
            logger.info(f"[{self.folder_name}] Loading YOLO model: {self.yolo_xml} on {yolo_device}")
            try:
                self.vision = RobotVision(self.yolo_xml, class_id=yolo_class_id, device=yolo_device)
            except Exception as e:
                logger.error(f"Failed to load YOLO for {self.folder_name}: {e}")
                self.has_yolo = False
        else:
            self.has_yolo = False

    def profile_cnn(self, num_iterations: int = 100, warmup: int = 10) -> Optional[dict]:
        if not self.has_cnn: return None
        
        dummy_roi = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        input_data = preprocess_roi_for_cnn(dummy_roi)
        
        # Warmup
        for _ in range(warmup):
            _ = self.compiled_cnn([input_data])[self.cnn_output]
        
        # Profile
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = self.compiled_cnn([input_data])[self.cnn_output]
            times.append((time.perf_counter() - start) * 1000)
            
        return {
            'mean': statistics.mean(times),
            'min': min(times),
            'max': max(times),
            'fps': 1000.0 / statistics.mean(times)
        }

    def profile_yolo(self, frame: np.ndarray, num_iterations: int = 100, warmup: int = 10) -> Optional[dict]:
        if not self.has_yolo: return None
        
        conf_threshold = self.config.get("models.yolo.conf_threshold", 0.4)
        input_size = self.config.get("models.yolo.input_size", 512)
        
        # Warmup
        for _ in range(warmup):
            _ = self.vision.predict(frame, conf_threshold=conf_threshold, imgsz=input_size)
            
        # Profile
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = self.vision.predict(frame, conf_threshold=conf_threshold, imgsz=input_size)
            times.append((time.perf_counter() - start) * 1000)
            
        return {
            'mean': statistics.mean(times),
            'min': min(times),
            'max': max(times),
            'fps': 1000.0 / statistics.mean(times)
        }

def main():
    folders = [
        "SpearHead_Standard",
        "SpearHead_HighPerformance",
        "SpearHead_YOLO26",
        "KFS_Standard",
        "KFS_HighPerformance"
    ]
    
    # Create dummy frame
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    results = []
    
    for folder in folders:
        try:
            logger.info(f"\nProcessing Folder: {folder}")
            profiler = ModelProfiler(folder)
            
            row = {"Folder": folder}
            
            if profiler.has_yolo:
                yolo_stats = profiler.profile_yolo(frame, num_iterations=50)
                row["YOLO_Latency"] = f"{yolo_stats['mean']:.2f}ms"
                row["YOLO_FPS"] = f"{yolo_stats['fps']:.1f}"
                row["YOLO_Device"] = profiler.config.get("models.yolo.device", "CPU")
            else:
                row["YOLO_Latency"] = "N/A"
                row["YOLO_FPS"] = "N/A"
                row["YOLO_Device"] = "N/A"
                
            if profiler.has_cnn:
                cnn_stats = profiler.profile_cnn(num_iterations=100)
                row["CNN_Latency"] = f"{cnn_stats['mean']:.2f}ms"
                row["CNN_FPS"] = f"{cnn_stats['fps']:.1f}"
                row["CNN_Device"] = profiler.config.get("models.cnn.device", "CPU")
            else:
                row["CNN_Latency"] = "N/A"
                row["CNN_FPS"] = "N/A"
                row["CNN_Device"] = "N/A"
                
            results.append(row)
            
        except Exception as e:
            logger.error(f"Error profiling {folder}: {e}")

    # Output Results Table
    print("\n" + "="*85)
    print(f"{'Folder':<25} | {'Model':<5} | {'Device':<6} | {'Avg Latency':<12} | {'FPS':<6}")
    print("-"*85)
    
    for res in results:
        # YOLO entry
        print(f"{res['Folder']:<25} | {'YOLO':<5} | {res['YOLO_Device']:<6} | {res['YOLO_Latency']:<12} | {res['YOLO_FPS']:<6}")
        # CNN entry
        if res['CNN_Latency'] != "N/A":
            print(f"{'':<25} | {'CNN':<5} | {res['CNN_Device']:<6} | {res['CNN_Latency']:<12} | {res['CNN_FPS']:<6}")
        print("-" * 85)

if __name__ == "__main__":
    main()
