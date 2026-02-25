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
BACKGROUND_VALUE = 128


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
