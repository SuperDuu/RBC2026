"""
Model Performance Profiling Script - Detailed Version for KFS_HighPerformance
Đo thời gian inference của YOLO và CNN models để xác định bottleneck.
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
project_root = Path(__file__).resolve().parent.parent
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
    
    def __init__(self, config_path: str = None, force_device: str = None):
        """Initialize profiler with models."""
        self.config = ConfigManager(config_path)
        self.force_device = force_device
        self._init_models()
    
    def _init_models(self):
        """Initialize YOLO and CNN models."""
        # Initialize OpenVINO Core
        self.ie = Core()
        
        # Load CNN model
        cnn_xml = self.config.get_path("paths.models.cnn_xml")
        cnn_device = self.force_device if self.force_device else self.config.get("models.cnn.device", "CPU")
        logger.info(f"Loading CNN model: {cnn_xml} on {cnn_device} (This might take a while to compile on GPU)...")
        start_load_cnn = time.time()
        
        cnn_model = self.ie.read_model(model=cnn_xml)
        self.compiled_cnn = self.ie.compile_model(model=cnn_model, device_name=cnn_device)
        self.cnn_output = self.compiled_cnn.output(0)
        logger.info(f"CNN Model successfully loaded and compiled in {time.time() - start_load_cnn:.2f} seconds.")
        
        # Load YOLO model
        yolo_xml = self.config.get_path("paths.models.yolo_xml")
        yolo_device = self.force_device if self.force_device else self.config.get("models.yolo.device", "GPU")
        yolo_class_id = self.config.get("models.yolo.class_id", 0)
        logger.info(f"Loading YOLO model: {yolo_xml} on {yolo_device}")
        
        self.vision = RobotVision(yolo_xml, class_id=yolo_class_id, device=yolo_device)
        
        # Get model info
        logger.info(f"CNN Input shape: {self.compiled_cnn.input(0).get_partial_shape()}")
        logger.info(f"CNN Output shape: {self.compiled_cnn.output(0).get_partial_shape()}")
    
    def profile_cnn(self, num_iterations: int = 100, warmup: int = 20) -> dict:
        """Profile CNN inference time."""
        logger.info(f"\n{'='*60}")
        logger.info("Profiling CNN Model")
        logger.info(f"{'='*60}")
        
        dummy_roi = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        input_data = preprocess_roi_for_cnn(dummy_roi)
        
        if input_data is None: raise ValueError("Failed to preprocess ROI")
        
        # Warmup
        logger.info(f"Warming up ({warmup} iterations) to stabilize hardware clocks...")
        start_warmup = time.time()
        for _ in range(warmup):
            _ = self.compiled_cnn([input_data])[self.cnn_output]
        
        # Profile
        logger.info(f"Warmup completed in {time.time() - start_warmup:.2f} seconds.")
        logger.info(f"Profiling ({num_iterations} iterations)...")
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = self.compiled_cnn([input_data])[self.cnn_output]
            times.append((time.perf_counter() - start) * 1000)
            
        # Remove Top 5% slowest iterations as outliers (often OS jitter or first-frame JITs)
        sorted_times = sorted(times)
        trim_idx = max(1, int(len(sorted_times) * 0.95))
        trimmed_times = sorted_times[:trim_idx]
        
        stats = {
            'mean': statistics.mean(times),
            'trimmed_mean': statistics.mean(trimmed_times),
            'median': statistics.median(times),
            'min': min(times),
            'max': max(times),
            'std': statistics.stdev(times) if len(times) > 1 else 0,
            'p95': np.percentile(times, 95),
            'p99': np.percentile(times, 99),
            'fps': 1000.0 / statistics.mean(trimmed_times),
            'iterations': num_iterations
        }
        
        logger.info("\nCNN Performance Results:")
        logger.info(f"  Mean:          {stats['trimmed_mean']:.3f} ms")
        logger.info(f"  Median:        {stats['median']:.3f} ms")
        logger.info(f"  Min:           {stats['min']:.3f} ms")
        logger.info(f"  Max:           {stats['max']:.3f} ms")
        if 'std' in stats: logger.info(f"  Std Dev:       {stats['std']:.3f} ms")
        if 'p95' in stats: logger.info(f"  P95:           {stats['p95']:.3f} ms")
        if 'p99' in stats: logger.info(f"  P99:           {stats['p99']:.3f} ms")
        logger.info(f"  FPS:           {stats['fps']:.1f}")
        return stats
    
    def profile_yolo(self, frame: np.ndarray, num_iterations: int = 100, warmup: int = 20) -> dict:
        """Profile YOLO inference time."""
        logger.info(f"\n{'='*60}")
        logger.info("Profiling YOLO Model")
        logger.info(f"{'='*60}")
        
        conf_threshold = self.config.get("models.yolo.conf_threshold", 0.4)
        input_size = self.config.get("models.yolo.input_size", 512)
        
        # Warmup
        for _ in range(warmup):
            _ = self.vision.predict(frame, conf_threshold=conf_threshold, imgsz=input_size)
            
        times = []
        num_detections = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            detections = self.vision.predict(frame, conf_threshold=conf_threshold, imgsz=input_size)
            times.append((time.perf_counter() - start) * 1000)
            num_detections.append(len(detections))
            
        # Remove Top 5% slowest iterations as outliers (often OS jitter or first-frame JITs)
        sorted_times = sorted(times)
        trim_idx = max(1, int(len(sorted_times) * 0.95))
        trimmed_times = sorted_times[:trim_idx]
        
        stats = {
            'mean': statistics.mean(times),
            'trimmed_mean': statistics.mean(trimmed_times),
            'median': statistics.median(times),
            'min': min(times),
            'max': max(times),
            'fps': 1000.0 / statistics.mean(trimmed_times),
            'avg_detections': statistics.mean(num_detections),
            'iterations': num_iterations
        }
        
        logger.info("\nYOLO Performance Results:")
        logger.info(f"  Mean:          {stats['trimmed_mean']:.3f} ms")
        logger.info(f"  Median:        {stats['median']:.3f} ms")
        logger.info(f"  Min:           {stats['min']:.3f} ms")
        logger.info(f"  Max:           {stats['max']:.3f} ms")
        if 'std' in stats: logger.info(f"  Std Dev:       {stats['std']:.3f} ms")
        if 'p95' in stats: logger.info(f"  P95:           {stats['p95']:.3f} ms")
        if 'p99' in stats: logger.info(f"  P99:           {stats['p99']:.3f} ms")
        logger.info(f"  FPS:           {stats['fps']:.1f}")
        logger.info(f"  Avg Detections: {stats.get('avg_detections', 0.0):.1f}")
        return stats
    
    def profile_full_pipeline(self, frame: np.ndarray, num_iterations: int = 50, warmup: int = 5) -> dict:
        """Profile full pipeline: YOLO + CNN classification."""
        logger.info(f"\n{'='*60}")
        logger.info("Profiling Full Pipeline (YOLO + CNN)")
        logger.info(f"{'='*60}")
        
        h_frame, w_frame = frame.shape[:2]
        conf_yolo = self.config.get("models.yolo.conf_threshold", 0.4)
        imgsz = self.config.get("models.yolo.input_size", 512)
        
        # Warmup
        for _ in range(warmup):
            detections = self.vision.predict(frame, conf_threshold=conf_yolo, imgsz=imgsz)
            for box in detections[:3]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = frame[max(0, y1):min(h_frame, y2), max(0, x1):min(w_frame, x2)]
                if roi.size > 0:
                    input_data = preprocess_roi_for_cnn(roi)
                    if input_data is not None:
                        _ = self.compiled_cnn([input_data])[self.cnn_output]
        
        # Profile
        times_yolo, times_cnn_total, times_total, num_cnn_calls = [], [], [], []
        for _ in range(num_iterations):
            start_yolo = time.perf_counter()
            detections = self.vision.predict(frame, conf_threshold=conf_yolo, imgsz=imgsz)
            end_yolo = time.perf_counter()
            
            start_cnn = time.perf_counter()
            cnn_count = 0
            for box in detections:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = frame[max(0, y1):min(h_frame, y2), max(0, x1):min(w_frame, x2)]
                if roi.size > 0:
                    input_data = preprocess_roi_for_cnn(roi)
                    if input_data is not None:
                        _ = self.compiled_cnn([input_data])[self.cnn_output]
                        cnn_count += 1
            end_cnn = time.perf_counter()
            
            times_yolo.append((end_yolo - start_yolo) * 1000)
            times_cnn_total.append((end_cnn - start_cnn) * 1000)
            times_total.append(times_yolo[-1] + times_cnn_total[-1])
            num_cnn_calls.append(cnn_count)
            
        # Remove Top 5% slowest iterations as outliers (often OS jitter or first-frame JITs)
        sorted_times = sorted(times)
        trim_idx = max(1, int(len(sorted_times) * 0.95))
        trimmed_times = sorted_times[:trim_idx]
        
        stats = {
            'yolo_mean': statistics.mean(times_yolo),
            'cnn_total_mean': statistics.mean(times_cnn_total),
            'total_mean': statistics.mean(times_total),
            'total_fps': 1000.0 / statistics.mean(times_total),
            'avg_cnn_calls': statistics.mean(num_cnn_calls)
        }
        
        logger.info(f"\nFull Pipeline Results: Total Mean: {stats['total_mean']:.3f} ms, FPS: {stats['total_fps']:.1f}")
        return stats

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    
    try:
        config_path = args.config if args.config else str(Path(__file__).resolve().parent / "config.yaml")
        for dev in ['CPU', 'GPU']:
            logger.info(f"\n{'#'*70}\n### RUNNING ON DEVICE: {dev}\n{'#'*70}")
            profiler = ModelProfiler(config_path=config_path, force_device=dev)
            profiler.profile_cnn(num_iterations=args.iterations, warmup=args.warmup)
            frame = cv2.imread(args.image) if args.image else np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            profiler.profile_yolo(frame, num_iterations=args.iterations, warmup=args.warmup)
            if args.full: profiler.profile_full_pipeline(frame, num_iterations=args.iterations//2)
        logger.info(f"\n{'='*60}\nAll Profiling completed!\n{'='*60}")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True); return 1
    return 0

if __name__ == "__main__": exit(main())
