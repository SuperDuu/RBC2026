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


def letterbox(img: np.ndarray, new_shape: Tuple[int, int] = (512, 512), color: Tuple[int, int, int] = (BACKGROUND_VALUE, BACKGROUND_VALUE, BACKGROUND_VALUE)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize image to a 32-pixel-multiple rectangle while preserving aspect ratio using padding.
    
    Args:
        img: Input image
        new_shape: Target shape (height, width)
        color: Padding color (BGR)
        
    Returns:
        - Resized and padded image
        - Resize ratio
        - (padding_w, padding_h)
    """
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, r, (left, top)
