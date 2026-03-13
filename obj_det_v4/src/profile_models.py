"""
Model Performance Profiling Script

Đo thời gian inference của YOLO model để xác định bottleneck.
"""

import cv2
import numpy as np
import time
import statistics
from pathlib import Path
from typing import List, Tuple
import logging

from vision import RobotVision
from vision import RobotVision
from config_manager import ConfigManager
from openvino.runtime import Core

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelProfiler:
    """Profiles YOLO model performance."""
    
    def __init__(self, config_path: str = None):
        """Initialize profiler with models."""
        self.config = ConfigManager(config_path)
        self._init_models()
    
    def _init_models(self):
        """Initialize YOLO model."""
        # Initialize OpenVINO Core
        self.ie = Core()
        
        # Load YOLO model
        yolo_xml = self.config.get_path("yolo_xml")
        yolo_device = self.config.get("models.yolo.device", "CPU")
        yolo_class_id = self.config.get("models.yolo.class_id", 0)
        logger.info(f"Loading YOLO model: {yolo_xml} on {yolo_device}")
        
        self.vision = RobotVision(yolo_xml, class_id=yolo_class_id, device=yolo_device)
    
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
            # Assume all YOLO detections are of interest 
            # or filtered by class_id in RobotVision.predict()
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
            'p95': float(np.percentile(times, 95)),
            'p99': float(np.percentile(times, 99)),
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


def main():
    """Main profiling function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Profile YOLO model performance")
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
    
    args = parser.parse_args()
    
    try:
        profiler = ModelProfiler(config_path=args.config)
        
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
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"YOLO Inference:    {yolo_stats['mean']:.3f} ms ({yolo_stats['fps']:.1f} FPS)")
        
        logger.info(f"\n{'='*60}")
        logger.info("Profiling completed!")
        
    except Exception as e:
        logger.error(f"Error during profiling: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
