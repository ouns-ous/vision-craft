"""
Unit tests for FaceDetector and anonymization filters.
"""

import pytest
import numpy as np
import cv2

from vision_craft.detection import FaceDetector, FaceDetection
from vision_craft.synthetic import SyntheticGenerator


def test_face_detector_instantiation():
    detector = FaceDetector()
    assert detector is not None


def test_face_detection_synthetic():
    face_img = SyntheticGenerator.generate_face_like_image(width=320, height=240)
    detector = FaceDetector(use_fallback=True)
    faces = detector.detect(face_img)

    # Fallback or YuNet should locate the synthetic face
    assert len(faces) >= 1
    f = faces[0]
    assert len(f.bbox) == 4
    assert f.confidence > 0.0

    # Test visualization
    drawn = detector.draw_faces(face_img, faces, show_landmarks=True)
    assert drawn.shape == face_img.shape


def test_anonymize_faces():
    img = np.full((100, 100, 3), 150, dtype=np.uint8)
    dummy_faces = [FaceDetection(bbox=(20, 20, 50, 50), confidence=0.9)]

    # 1. Pixelate
    pixelated = FaceDetector.anonymize_faces(img, dummy_faces, method="pixelate", factor=0.1)
    assert pixelated.shape == img.shape

    # 2. Gaussian blur
    blurred = FaceDetector.anonymize_faces(img, dummy_faces, method="gaussian_blur")
    assert blurred.shape == img.shape

    # 3. Blackout
    blackout = FaceDetector.anonymize_faces(img, dummy_faces, method="blackout")
    assert blackout.shape == img.shape
    # Inside face region, pixels should be zero
    assert np.all(blackout[25:35, 25:35] == 0)
