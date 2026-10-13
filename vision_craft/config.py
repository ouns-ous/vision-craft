"""
Configuration constants, color definitions, and path resolvers for VisionCraft.
"""

from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np

# Base paths
PACKAGE_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PACKAGE_ROOT / "data"
DEFAULT_YUNET_MODEL = DATA_DIR / "face_detection_yunet.onnx"

# Supported file formats
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

# BGR UI Palette for annotations and visual overlays
UI_COLORS: Dict[str, Tuple[int, int, int]] = {
    "cyan": (255, 255, 0),
    "magenta": (255, 0, 255),
    "yellow": (0, 255, 255),
    "green": (0, 255, 0),
    "red": (0, 0, 255),
    "blue": (255, 0, 0),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "orange": (0, 165, 255),
    "lime": (50, 205, 50),
    "purple": (128, 0, 128),
    "gray": (128, 128, 128),
}

# Predefined HSV ranges for robust color tracking: (Lower_HSV, Upper_HSV)
# Hue is in [0, 179], Saturation in [0, 255], Value in [0, 255]
HSV_COLOR_RANGES: Dict[str, List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]] = {
    # Red wraps around 0 and 180 degrees in OpenCV HSV
    "red": [
        ((0, 100, 70), (10, 255, 255)),
        ((170, 100, 70), (179, 255, 255)),
    ],
    "green": [
        ((35, 80, 50), (85, 255, 255)),
    ],
    "blue": [
        ((90, 80, 50), (130, 255, 255)),
    ],
    "yellow": [
        ((20, 100, 100), (35, 255, 255)),
    ],
    "orange": [
        ((10, 120, 70), (25, 255, 255)),
    ],
    "purple": [
        ((125, 50, 50), (155, 255, 255)),
    ],
    "cyan": [
        ((80, 100, 100), (100, 255, 255)),
    ],
    "white": [
        ((0, 0, 200), (180, 40, 255)),
    ],
    "black": [
        ((0, 0, 0), (180, 255, 50)),
    ],
}

# Default parameters
DEFAULT_MIN_CONTOUR_AREA = 300
DEFAULT_MAX_CONTOUR_AREA = 500000
DEFAULT_MOTION_MIN_AREA = 500
DEFAULT_TRAIL_LENGTH = 32
DEFAULT_YUNET_SCORE_THRESHOLD = 0.6
DEFAULT_YUNET_NMS_THRESHOLD = 0.3
