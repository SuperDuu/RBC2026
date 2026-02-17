"""
Utility functions for RBC2026 Robocon Vision System.

This module contains shared utility functions used across the project.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)

# Constants
CNN_INPUT_SIZE = 64
BACKGROUND_VALUE = 128


def preprocess_roi_for_cnn(roi: np.ndarray, input_size: int = CNN_INPUT_SIZE) -> Optional[np.ndarray]:
    """
    Preprocess ROI (Region of Interest) for CNN classification.
    
    Converts ROI to grayscale, resizes with aspect ratio preservation,
    and pads to fixed size with gray background.
    
    Args:
        roi: Input ROI image (BGR format)
        input_size: Target input size (default: 64)
    
    Returns:
        Preprocessed array of shape (1, input_size, input_size, 1) normalized to [0, 1],
        or None if ROI is invalid
    """
    if roi is None or roi.size == 0:
        logger.debug("Invalid ROI: empty or None")
        return None
    
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Calculate scale to fit within input_size while preserving aspect ratio
        scale = input_size / max(h, w)
        nw, nh = int(w * scale), int(h * scale)
        
        # Resize with high-quality interpolation
        resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
        
        # Create canvas with gray background
        canvas = np.full((input_size, input_size), BACKGROUND_VALUE, dtype=np.uint8)
        
        # Center the resized image on canvas
        y_offset = (input_size - nh) // 2
        x_offset = (input_size - nw) // 2
        canvas[y_offset:y_offset+nh, x_offset:x_offset+nw] = resized
        
        # Reshape and normalize to [0, 1]
        return canvas.reshape(1, input_size, input_size, 1).astype(np.float32) / 255.0
    
    except Exception as e:
        logger.error(f"Error preprocessing ROI: {e}")
        return None


def get_label_type_and_color(label_name: str, color_map: Dict[str, Tuple[int, int, int]]) -> Tuple[Tuple[int, int, int], str]:
    """
    Determine label type and color based on label name.
    
    Args:
        label_name: Label name string
        color_map: Dictionary mapping types to RGB colors
    
    Returns:
        Tuple of (color, type) where:
        - color: RGB tuple (B, G, R) for OpenCV
        - type: String type ("R1", "REAL", or "FAKE")
    """
    name_lower = label_name.lower()
    
    if "r1" in name_lower:
        return tuple(color_map.get("R1", (255, 255, 0))), "R1"
    elif "real" in name_lower:
        return tuple(color_map.get("REAL", (0, 255, 0))), "REAL"
    else:
        return tuple(color_map.get("FAKE", (0, 0, 255))), "FAKE"


def validate_roi_bounds(roi: np.ndarray, frame_shape: Tuple[int, int, int]) -> bool:
    """
    Validate ROI bounds are within frame dimensions.
    
    Args:
        roi: ROI array to validate
        frame_shape: Frame shape tuple (height, width, channels)
    
    Returns:
        True if ROI is valid, False otherwise
    """
    if roi is None or roi.size == 0:
        return False
    
    h_frame, w_frame = frame_shape[:2]
    h_roi, w_roi = roi.shape[:2]
    
    return 0 < h_roi <= h_frame and 0 < w_roi <= w_frame


def calculate_fps(current_time: float, previous_time: float) -> float:
    """
    Calculate FPS from time difference.
    
    Args:
        current_time: Current timestamp
        previous_time: Previous timestamp
    
    Returns:
        FPS value (0 if time difference is invalid)
    """
    time_diff = current_time - previous_time
    if time_diff > 0:
        return 1.0 / time_diff
    return 0.0


def clamp_value(value: int, min_val: int, max_val: int) -> int:
    """
    Clamp value between min and max.
    
    Args:
        value: Value to clamp
        min_val: Minimum value
        max_val: Maximum value
    
    Returns:
        Clamped value
    """
    return max(min_val, min(value, max_val))
