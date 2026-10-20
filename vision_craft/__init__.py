"""
VisionCraft: Advanced OpenCV Computer Vision Studio & Processing Engine.

A modular, extensible Computer Vision toolkit featuring image filters,
shape and contour classification, multi-color centroid tracking, motion detection,
face detection and privacy blurring, invisible cloaking, video frame extraction,
and an interactive live CV studio.
"""

__version__ = "1.1.0"
__author__ = "ouns-ous"

from vision_craft.filters import (
    ImageFilter,
    apply_filter,
    get_available_filters,
)
from vision_craft.shapes import (
    ShapeDetector,
    DetectedShape,
)
from vision_craft.colors import (
    ColorTracker,
    TrackedColorObject,
)
from vision_craft.motion import (
    MotionTracker,
    MotionResult,
)
from vision_craft.detection import (
    FaceDetector,
    FaceDetection,
)
from vision_craft.cloak import (
    InvisibleCloak,
)
from vision_craft.video_tools import (
    VideoFrameExtractor,
    VideoSplitter,
    get_video_metadata,
)
from vision_craft.synthetic import (
    SyntheticGenerator,
)
from vision_craft.document import (
    DocumentScanner,
    ScannedDocument,
)
from vision_craft.qrcode_scanner import (
    QRCodeScanner,
    QRScanResult,
)
from vision_craft.stitcher import (
    ImageStitcher,
)
from vision_craft.features import (
    FeatureMatcher,
    MatchResult,
)
from vision_craft.optical_flow import (
    OpticalFlowAnalyzer,
    FlowResult,
)

__all__ = [
    "ImageFilter",
    "apply_filter",
    "get_available_filters",
    "ShapeDetector",
    "DetectedShape",
    "ColorTracker",
    "TrackedColorObject",
    "MotionTracker",
    "MotionResult",
    "FaceDetector",
    "FaceDetection",
    "InvisibleCloak",
    "VideoFrameExtractor",
    "VideoSplitter",
    "get_video_metadata",
    "SyntheticGenerator",
    "DocumentScanner",
    "ScannedDocument",
    "QRCodeScanner",
    "QRScanResult",
    "ImageStitcher",
    "FeatureMatcher",
    "MatchResult",
    "OpticalFlowAnalyzer",
    "FlowResult",
]
