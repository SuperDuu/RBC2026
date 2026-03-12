"""
Test image processing module for RBC2026 Robocon Vision System.

This module provides functionality to test detection and classification on single images.
"""

import cv2
import json
import logging
import numpy as np
import sys
from pathlib import Path
from typing import Optional, Tuple

from vision import RobotVision, DetectedObject
from utils import preprocess_roi_for_cnn, get_label_type_and_color
from config_manager import ConfigManager


class ImageTester:
    """
    Test detection and classification on single images.
    
    Uses the same models and processing pipeline as main system.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize image tester.
        
        Args:
            config_path: Path to config YAML file. If None, uses default.
        
        Raises:
            RuntimeError: If initialization fails
        """
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        try:
            self.config = ConfigManager(config_path)
            self.logger.info("Configuration loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise RuntimeError(f"Configuration loading failed: {e}") from e
        
        # Initialize models
        self._init_models()
    
    def _init_models(self) -> None:
        """Initialize YOLO and CNN models."""
        try:
            from openvino.runtime import Core
            
            # Load CNN labels
            labels_path = self.config.get_path("labels_json")
            with open(labels_path, 'r', encoding='utf-8') as f:
                labels_data = json.load(f)
                self.labels_cnn = {int(v): k for k, v in labels_data.items()}
            
            self.logger.info(f"Loaded {len(self.labels_cnn)} CNN labels")
            
            # Initialize OpenVINO Core
            self.ie = Core()
            
            # Load and compile CNN model
            cnn_xml = self.config.get_path("cnn_xml")
            cnn_device = self.config.get("models.cnn.device", "GPU")
            
            self.logger.info(f"Loading CNN model from {cnn_xml} on {cnn_device}")
            cnn_model = self.ie.read_model(model=cnn_xml)
            self.compiled_cnn = self.ie.compile_model(model=cnn_model, device_name=cnn_device)
            self.cnn_output = self.compiled_cnn.output(0)
            
            # Initialize YOLO vision system
            yolo_xml = self.config.get_path("yolo_xml")
            yolo_device = self.config.get("models.yolo.device", "CPU")
            yolo_class_id = self.config.get("models.yolo.class_id", 0)
            
            self.logger.info(f"Loading YOLO model from {yolo_xml} on {yolo_device}")
            self.vision = RobotVision(yolo_xml, class_id=yolo_class_id, device=yolo_device)
            
            # Get configuration values
            self.conf_threshold_yolo = self.config.get("models.yolo.conf_threshold", 0.4)
            self.conf_threshold_cnn = self.config.get("models.cnn.conf_threshold", 0.5)
            self.color_map = {
                "R1": tuple(self.config.get("classification.colors.R1", [255, 255, 0])),
                "REAL": tuple(self.config.get("classification.colors.REAL", [0, 255, 0])),
                "FAKE": tuple(self.config.get("classification.colors.FAKE", [0, 0, 255]))
            }
            
            self.logger.info("All models loaded successfully")
        
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {e}")
            raise RuntimeError(f"Model initialization failed: {e}") from e
    
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
    
    def process_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Process single image: detect and classify objects.
        
        Args:
            image_path: Path to input image
        
        Returns:
            Processed image with annotations, or None if processing fails
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                self.logger.error(f"Failed to load image: {image_path}")
                return None
            
            h_orig, w_orig = img.shape[:2]
            display_frame = img.copy()
            
            # Run YOLO detection
            detections = self.vision.predict(
                img, 
                conf_threshold=self.conf_threshold_yolo
            )
            
            self.logger.info(f"Found {len(detections)} detections")
            
            # Process each detection
            for obj in detections:
                try:
                    x1, y1, x2, y2 = map(int, obj.xyxy[0])
                    
                    # Extract ROI
                    roi = img[
                        max(0, y1):min(h_orig, y2),
                        max(0, x1):min(w_orig, x2)
                    ]
                    
                    if roi.size == 0:
                        continue
                    
                    # Classify ROI
                    label, conf = self._classify_roi(roi)
                    
                    if label is None or conf < self.conf_threshold_cnn:
                        continue
                    
                    # Get color and type
                    color, t_type = get_label_type_and_color(label, self.color_map)
                    
                    # Draw bounding box and label
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        display_frame, 
                        f"{label} {conf:.2f}", 
                        (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        color, 
                        2
                    )
                    
                    # Update Kalman filter for tracking visualization
                    if t_type in ["R1", "REAL"]:
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        self.vision.update_kalman(cx, cy)
                
                except Exception as e:
                    self.logger.error(f"Error processing detection: {e}")
                    continue
            
            return display_frame
        
        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            return None


def main():
    """Main entry point for image testing."""
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Get default image path
    base_dir = Path(__file__).resolve().parent.parent
    default_image = str(base_dir / "datasets" / "user_test" / "img_test_user2.jpg")
    
    parser = argparse.ArgumentParser(
        description="Test detection and classification on image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python test_image.py /path/to/image.jpg
  python test_image.py /path/to/image.jpg --config custom_config.yaml
  python test_image.py  # Uses default: {default_image}
        """
    )
    parser.add_argument(
        "image_path",
        type=str,
        nargs='?',  # Make optional
        default=default_image,
        help=f"Path to input image (default: {default_image})"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file (optional)"
    )
    
    args = parser.parse_args()
    
    # Check if image exists
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"❌ Error: Image file not found: {args.image_path}")
        print(f"\nPlease provide a valid image path:")
        print(f"  python test_image.py /path/to/your/image.jpg")
        print(f"\nOr check if default image exists:")
        print(f"  {default_image}")
        sys.exit(1)
    
    try:
        print(f"📷 Processing image: {args.image_path}")
        tester = ImageTester(config_path=args.config)
        result = tester.process_image(str(image_path))
        
        if result is not None:
            print("✅ Processing completed successfully")
            print("Press any key in the image window to close...")
            cv2.imshow("RBC2026 Test Result", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("❌ Processing failed")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
