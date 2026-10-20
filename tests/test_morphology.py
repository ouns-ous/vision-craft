"""
Unit tests for MorphologyEngine module.
"""

import pytest
import numpy as np
import cv2

from vision_craft.morphology import MorphologyEngine


@pytest.fixture
def binary_shapes() -> np.ndarray:
    """Create test binary image with rectangle and noisy dots."""
    img = np.zeros((120, 120), dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (90, 90), 255, -1)
    # Add noise specks
    img[10, 10] = 255
    img[15, 110] = 255
    return img


def test_morphology_kernels():
    k_rect = MorphologyEngine.get_kernel("rect", 5)
    assert k_rect.shape == (5, 5)

    k_cross = MorphologyEngine.get_kernel("cross", 7)
    assert k_cross.shape == (7, 7)

    k_ellipse = MorphologyEngine.get_kernel("ellipse", (3, 3))
    assert k_ellipse.shape == (3, 3)


def test_morphological_operations(binary_shapes):
    ops = ["erode", "dilate", "open", "close", "gradient", "tophat", "blackhat"]
    for op in ops:
        out = MorphologyEngine.apply_operation(binary_shapes, op_name=op, ksize=3)
        assert out.shape == binary_shapes.shape
        assert out.dtype == np.uint8

    with pytest.raises(ValueError):
        MorphologyEngine.apply_operation(binary_shapes, op_name="invalid_op")


def test_skeletonize(binary_shapes):
    skel = MorphologyEngine.skeletonize(binary_shapes)
    assert skel.shape == binary_shapes.shape
    assert skel.dtype == np.uint8
    # Skeleton should thin the solid rectangle
    assert np.count_nonzero(skel) < np.count_nonzero(binary_shapes)
    assert np.count_nonzero(skel) > 0
