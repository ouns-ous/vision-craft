"""
Unit tests for ShapeDetector and contour classification.
"""

import pytest
import numpy as np
import cv2

from vision_craft.shapes import ShapeDetector, DetectedShape
from vision_craft.synthetic import SyntheticGenerator


def test_detect_synthetic_shapes():
    img = SyntheticGenerator.generate_shapes_image(width=640, height=480)
    detector = ShapeDetector()
    shapes = detector.detect(img, min_area=300)

    # We drew 5 distinct shapes: circle, square, rectangle, triangle, pentagon
    assert len(shapes) >= 4

    shape_names = {s.name for s in shapes}
    assert "Circle" in shape_names
    assert ("Square" in shape_names or "Rectangle" in shape_names)
    assert "Triangle" in shape_names

    for s in shapes:
        assert s.area > 300
        assert s.perimeter > 0
        assert len(s.bounding_box) == 4
        assert len(s.centroid) == 2


def test_draw_shapes_overlay():
    img = SyntheticGenerator.generate_shapes_image(width=640, height=480)
    detector = ShapeDetector()
    shapes = detector.detect(img)
    annotated = detector.draw_shapes(img, shapes, show_details=True)

    assert annotated.shape == img.shape
    assert annotated.dtype == np.uint8
    # Annotated image should have differences where drawings occurred
    assert not np.array_equal(img, annotated)
