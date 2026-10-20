"""
Unit tests for ImageStitcher module.
"""

from pathlib import Path
import pytest
import numpy as np
import cv2

from vision_craft.stitcher import ImageStitcher


@pytest.fixture
def overlapping_pair() -> Tuple[np.ndarray, np.ndarray]:
    """Create two overlapping images with rich distinct features."""
    canvas = np.zeros((300, 600, 3), dtype=np.uint8)
    # Background texture
    for x in range(0, 600, 20):
        cv2.line(canvas, (x, 0), (x, 300), (40, 50, 60), 1)

    # Left image features
    cv2.circle(canvas, (100, 100), 40, (0, 0, 255), -1)
    cv2.rectangle(canvas, (50, 180), (150, 260), (0, 255, 0), -1)

    # Overlap region features (x=200 to x=400)
    cv2.putText(canvas, "STITCH-OVERLAP", (210, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.circle(canvas, (300, 220), 30, (255, 255, 0), -1)

    # Right image features
    cv2.rectangle(canvas, (450, 80), (550, 220), (255, 0, 255), -1)

    left_half = canvas[:, 0:380].copy()
    right_half = canvas[:, 220:600].copy()
    return left_half, right_half


def test_stitcher_instantiation():
    stitcher = ImageStitcher()
    assert stitcher is not None


def test_stitcher_crop_borders():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (90, 90), (120, 150, 200), -1)
    cropped = ImageStitcher.crop_black_borders(img)
    assert cropped.shape[0] <= 100
    assert cropped.shape[1] <= 100


def test_stitch_synthetic_pair(overlapping_pair, tmp_path: Path):
    left, right = overlapping_pair
    stitcher = ImageStitcher()

    success, pano, msg = stitcher.stitch([left, right], crop_borders=True)
    # Either OpenCV Stitcher or pairwise ORB fallback must succeed
    assert success, f"Stitching failed: {msg}"
    assert pano is not None
    assert pano.shape[1] > 300

    # Test file saving
    left_path = tmp_path / "left.png"
    right_path = tmp_path / "right.png"
    pano_path = tmp_path / "pano.png"
    cv2.imwrite(str(left_path), left)
    cv2.imwrite(str(right_path), right)

    ok, _, _ = stitcher.stitch_files([left_path, right_path], output_path=pano_path)
    assert ok
    assert pano_path.exists()
