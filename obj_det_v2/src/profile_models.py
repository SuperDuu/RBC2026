"""
Model Performance Profiling Script

Đo thời gian inference của YOLO và CNN models để xác định bottleneck.
"""

import cv2
import numpy as np
import time
import statistics
from pathlib import Path
from typing import List, Tuple
import logging

from vision import RobotVision
from utils import preprocess_roi_for_cnn
from config_manager import ConfigManager
from openvino.runtime import Core
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelProfiler:
    """Profiles YOLO and CNN model performance."""
    
    def __init__(self, config_path: str = None):
        """Initialize profiler with models."""
        self.config = ConfigManager(config_path)
        self._init_models()
    
    def _init_models(self):
        """Initialize YOLO and CNN models."""
        # Load CNN labels
        labels_path = self.config.get_path("labels_json")
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels_data = json.load(f)
            self.labels_cnn = {int(v): k for k, v in labels_data.items()}
        
        # Initialize OpenVINO Core
        self.ie = Core()
        
        # Load CNN model
        cnn_xml = self.config.get_path("cnn_xml")
        cnn_device = self.config.get("models.cnn.device", "GPU")
        logger.info(f"Loading CNN model: {cnn_xml} on {cnn_device}")
        
        cnn_model = self.ie.read_model(model=cnn_xml)
        self.compiled_cnn = self.ie.compile_model(model=cnn_model, device_name=cnn_device)
        self.cnn_output = self.compiled_cnn.output(0)
        
        # Load YOLO model
        yolo_xml = self.config.get_path("yolo_xml")
        yolo_device = self.config.get("models.yolo.device", "CPU")
        yolo_class_id = self.config.get("models.yolo.class_id", 0)
        logger.info(f"Loading YOLO model: {yolo_xml} on {yolo_device}")
        
        self.vision = RobotVision(yolo_xml, class_id=yolo_class_id, device=yolo_device)
        
        # Get model info
        logger.info(f"CNN Input shape: {self.compiled_cnn.input(0).shape}")
        logger.info(f"CNN Output shape: {self.compiled_cnn.output(0).shape}")
    
    def profile_cnn(self, num_iterations: int = 100, warmup: int = 10) -> dict:
        """
        Profile CNN inference time.
        
        Args:
            num_iterations: Number of inference iterations
            warmup: Number of warmup iterations
        
        Returns:
            Dictionary with timing statistics
        """
        logger.info(f"\n{'='*60}")
        logger.info("Profiling CNN Model")
        logger.info(f"{'='*60}")
        
        # Create dummy ROI (64x64 grayscale)
        dummy_roi = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        input_data = preprocess_roi_for_cnn(dummy_roi)
        
        if input_data is None:
            raise ValueError("Failed to preprocess ROI")
        
        # Warmup
        logger.info(f"Warming up ({warmup} iterations)...")
        for _ in range(warmup):
            _ = self.compiled_cnn([input_data])[self.cnn_output]
        
        # Profile
        logger.info(f"Profiling ({num_iterations} iterations)...")
        times = []
        
        for i in range(num_iterations):
            start = time.perf_counter()
            _ = self.compiled_cnn([input_data])[self.cnn_output]
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
        
        # Calculate statistics
        stats = {
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'min': min(times),
            'max': max(times),
            'std': statistics.stdev(times) if len(times) > 1 else 0,
            'p95': np.percentile(times, 95),
            'p99': np.percentile(times, 99),
            'fps': 1000.0 / statistics.mean(times),
            'iterations': num_iterations
        }
        
        logger.info(f"\nCNN Performance Results:")
        logger.info(f"  Mean:     {stats['mean']:.3f} ms")
        logger.info(f"  Median:   {stats['median']:.3f} ms")
        logger.info(f"  Min:      {stats['min']:.3f} ms")
        logger.info(f"  Max:      {stats['max']:.3f} ms")
        logger.info(f"  Std Dev:  {stats['std']:.3f} ms")
        logger.info(f"  P95:      {stats['p95']:.3f} ms")
        logger.info(f"  P99:      {stats['p99']:.3f} ms")
        logger.info(f"  FPS:      {stats['fps']:.1f}")
        
        return stats
    
    def profile_yolo(self, frame: np.ndarray, num_iterations: int = 100, warmup: int = 10) -> dict:
        """
        Profile YOLO inference time.
        
        Args:
            frame: Input frame
            num_iterations: Number of inference iterations
            warmup: Number of warmup iterations
        
        Returns:
            Dictionary with timing statistics
        """
        logger.info(f"\n{'='*60}")
        logger.info("Profiling YOLO Model")
        logger.info(f"{'='*60}")
        logger.info(f"Frame shape: {frame.shape}")
        
        conf_threshold = self.config.get("models.yolo.conf_threshold", 0.4)
        input_size = self.config.get("models.yolo.input_size", 512)
        
        # Warmup
        logger.info(f"Warming up ({warmup} iterations)...")
        for _ in range(warmup):
            _ = self.vision.predict(frame, conf_threshold=conf_threshold, imgsz=input_size)
        
        # Profile
        logger.info(f"Profiling ({num_iterations} iterations) with input_size={input_size}...")
        times = []
        num_detections = []
        
        for i in range(num_iterations):
            start = time.perf_counter()
            detections = self.vision.predict(frame, conf_threshold=conf_threshold, imgsz=input_size)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
            num_detections.append(len(detections))
        
        # Calculate statistics
        stats = {
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'min': min(times),
            'max': max(times),
            'std': statistics.stdev(times) if len(times) > 1 else 0,
            'p95': np.percentile(times, 95),
            'p99': np.percentile(times, 99),
            'fps': 1000.0 / statistics.mean(times),
            'avg_detections': statistics.mean(num_detections),
            'iterations': num_iterations
        }
        
        logger.info(f"\nYOLO Performance Results:")
        logger.info(f"  Mean:          {stats['mean']:.3f} ms")
        logger.info(f"  Median:        {stats['median']:.3f} ms")
        logger.info(f"  Min:           {stats['min']:.3f} ms")
        logger.info(f"  Max:           {stats['max']:.3f} ms")
        logger.info(f"  Std Dev:       {stats['std']:.3f} ms")
        logger.info(f"  P95:           {stats['p95']:.3f} ms")
        logger.info(f"  P99:           {stats['p99']:.3f} ms")
        logger.info(f"  FPS:           {stats['fps']:.1f}")
        logger.info(f"  Avg Detections: {stats['avg_detections']:.1f}")
        
        return stats
    
    def profile_full_pipeline(
        self, 
        frame: np.ndarray, 
        num_iterations: int = 50, 
        warmup: int = 5
    ) -> dict:
        """
        Profile full pipeline: YOLO + CNN classification.
        
        Args:
            frame: Input frame
            num_iterations: Number of iterations
            warmup: Number of warmup iterations
        
        Returns:
            Dictionary with timing statistics
        """
        logger.info(f"\n{'='*60}")
        logger.info("Profiling Full Pipeline (YOLO + CNN)")
        logger.info(f"{'='*60}")
        
        conf_threshold_yolo = self.config.get("models.yolo.conf_threshold", 0.4)
        conf_threshold_cnn = self.config.get("models.cnn.conf_threshold", 0.5)
        input_size = self.config.get("models.yolo.input_size", 512)
        h_frame, w_frame = frame.shape[:2]
        
        # Warmup
        logger.info(f"Warming up ({warmup} iterations)...")
        for _ in range(warmup):
            detections = self.vision.predict(frame, conf_threshold=conf_threshold_yolo, imgsz=input_size)
            for box in detections[:3]:  # Process first 3 detections
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = frame[max(0, y1):min(h_frame, y2), max(0, x1):min(w_frame, x2)]
                if roi.size > 0:
                    input_data = preprocess_roi_for_cnn(roi)
                    if input_data is not None:
                        _ = self.compiled_cnn([input_data])[self.cnn_output]
        
        # Profile
        logger.info(f"Profiling ({num_iterations} iterations)...")
        times_yolo = []
        times_cnn_total = []
        times_total = []
        num_cnn_calls = []
        
        for i in range(num_iterations):
            # YOLO
            start_yolo = time.perf_counter()
            detections = self.vision.predict(frame, conf_threshold=conf_threshold_yolo, imgsz=input_size)
            end_yolo = time.perf_counter()
            times_yolo.append((end_yolo - start_yolo) * 1000)
            
            # CNN for each detection
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
            times_cnn_total.append((end_cnn - start_cnn) * 1000)
            num_cnn_calls.append(cnn_count)
            
            times_total.append(times_yolo[-1] + times_cnn_total[-1])
        
        # Calculate statistics
        stats = {
            'yolo_mean': statistics.mean(times_yolo),
            'cnn_total_mean': statistics.mean(times_cnn_total),
            'total_mean': statistics.mean(times_total),
            'yolo_fps': 1000.0 / statistics.mean(times_yolo),
            'total_fps': 1000.0 / statistics.mean(times_total),
            'avg_cnn_calls': statistics.mean(num_cnn_calls),
            'iterations': num_iterations
        }
        
        logger.info(f"\nFull Pipeline Performance Results:")
        logger.info(f"  YOLO Mean:        {stats['yolo_mean']:.3f} ms ({stats['yolo_fps']:.1f} FPS)")
        logger.info(f"  CNN Total Mean:   {stats['cnn_total_mean']:.3f} ms")
        logger.info(f"  Total Mean:       {stats['total_mean']:.3f} ms ({stats['total_fps']:.1f} FPS)")
        logger.info(f"  Avg CNN Calls:    {stats['avg_cnn_calls']:.1f} per frame")
        logger.info(f"\n  YOLO %:           {(stats['yolo_mean']/stats['total_mean']*100):.1f}%")
        logger.info(f"  CNN %:            {(stats['cnn_total_mean']/stats['total_mean']*100):.1f}%")
        
        return stats


def main():
    """Main profiling function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Profile YOLO and CNN model performance")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to test image (default: use dummy frame)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of profiling iterations (default: 100)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also profile full pipeline"
    )
    
    args = parser.parse_args()
    
    try:
        profiler = ModelProfiler(config_path=args.config)
        
        # Profile CNN
        cnn_stats = profiler.profile_cnn(
            num_iterations=args.iterations,
            warmup=args.warmup
        )
        
        # Profile YOLO
        if args.image:
            frame = cv2.imread(args.image)
            if frame is None:
                raise ValueError(f"Failed to load image: {args.image}")
        else:
            # Create dummy frame (640x480)
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            logger.info("Using dummy frame (640x480)")
        
        yolo_stats = profiler.profile_yolo(
            frame=frame,
            num_iterations=args.iterations,
            warmup=args.warmup
        )
        
        # Profile full pipeline if requested
        if args.full:
            pipeline_stats = profiler.profile_full_pipeline(
                frame=frame,
                num_iterations=args.iterations // 2,  # Fewer iterations for full pipeline
                warmup=args.warmup // 2
            )
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"CNN Inference:     {cnn_stats['mean']:.3f} ms ({cnn_stats['fps']:.1f} FPS)")
        logger.info(f"YOLO Inference:    {yolo_stats['mean']:.3f} ms ({yolo_stats['fps']:.1f} FPS)")
        
        if args.full:
            logger.info(f"Full Pipeline:     {pipeline_stats['total_mean']:.3f} ms ({pipeline_stats['total_fps']:.1f} FPS)")
            logger.info(f"\nBottleneck Analysis:")
            logger.info(f"  YOLO: {(pipeline_stats['yolo_mean']/pipeline_stats['total_mean']*100):.1f}% of total time")
            logger.info(f"  CNN:  {(pipeline_stats['cnn_total_mean']/pipeline_stats['total_mean']*100):.1f}% of total time")
        
        logger.info(f"\n{'='*60}")
        logger.info("Profiling completed!")
        
    except Exception as e:
        logger.error(f"Error during profiling: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
