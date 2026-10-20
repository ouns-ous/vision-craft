"""
Unit tests for DocumentScanner module.
"""

import pytest
import numpy as np
import cv2

from vision_craft.document import DocumentScanner, ScannedDocument
from vision_craft.synthetic import SyntheticGenerator


def test_order_points():
    # Unordered points
    pts = np.array([
        [100, 200],  # BL
        [20, 20],    # TL
        [220, 190],  # BR
        [180, 30],   # TR
    ], dtype=np.float32)

    ordered = DocumentScanner.order_points(pts)
    assert np.array_equal(ordered[0], [20, 20])     # TL
    assert np.array_equal(ordered[1], [180, 30])    # TR
    assert np.array_equal(ordered[2], [220, 190])   # BR
    assert np.array_equal(ordered[3], [100, 200])   # BL


def test_four_point_transform():
    # Create simple image with red square
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(img, (40, 40), (160, 160), (0, 0, 255), -1)

    pts = np.array([
        [40, 40],
        [160, 40],
        [160, 160],
        [40, 160]
    ], dtype=np.float32)

    warped = DocumentScanner.four_point_transform(img, pts)
    assert warped is not None
    assert warped.shape[0] >= 115
    assert warped.shape[1] >= 115
    # Warped should be solid red
    assert warped[50, 50, 2] == 255


def test_document_enhancement_modes():
    img = np.full((100, 100, 3), 180, dtype=np.uint8)
    cv2.putText(img, "DOC", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    for mode in ["bw", "color_enhanced", "gray", "original"]:
        enhanced = DocumentScanner.enhance(img, mode=mode)
        assert enhanced.shape[:2] == img.shape[:2]
        assert enhanced.dtype == np.uint8

    with pytest.raises(ValueError):
        DocumentScanner.enhance(img, mode="unsupported_mode")


def test_document_scanner_synthetic():
    sample = SyntheticGenerator.generate_document_sample(width=640, height=480)
    scanner = DocumentScanner()

    doc = scanner.scan(sample, mode="color_enhanced")
    assert isinstance(doc, ScannedDocument)
    assert doc.warped is not None
    assert doc.enhanced is not None
    assert doc.corners.shape == (4, 2)
    assert doc.enhanced.shape[:2] == doc.warped.shape[:2]

    # Test bounds visualization
    annotated = scanner.draw_document_bounds(sample, doc.corners)
    assert annotated.shape == sample.shape
    assert not np.array_equal(annotated, sample)
