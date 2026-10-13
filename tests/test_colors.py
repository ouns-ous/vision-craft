"""
Unit tests for ColorTracker and HSV segmentation.
"""

import pytest
import numpy as np
import cv2

from vision_craft.colors import ColorTracker, TrackedColorObject
from vision_craft.synthetic import SyntheticGenerator


def test_color_mask_extraction():
    img = SyntheticGenerator.generate_color_palette_image(width=640, height=480)
    tracker = ColorTracker()

    # Mask for red
    mask_red = tracker.get_mask(img, "red")
    assert mask_red.shape == (480, 640)
    assert np.count_nonzero(mask_red) > 500

    # Mask for green
    mask_green = tracker.get_mask(img, "green")
    assert np.count_nonzero(mask_green) > 500

    # Mask for blue
    mask_blue = tracker.get_mask(img, "blue")
    assert np.count_nonzero(mask_blue) > 500


def test_color_tracking_and_trail():
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    # Draw solid green circle
    cv2.circle(img, (150, 150), 30, (0, 255, 0), -1)

    tracker = ColorTracker(max_trail_length=10)
    mask, objs = tracker.track(img, "green", min_area=100)

    assert len(objs) == 1
    obj = objs[0]
    assert obj.color_label == "green"
    # Centroid should be very close to (150, 150)
    assert abs(obj.centroid[0] - 150) <= 2
    assert abs(obj.centroid[1] - 150) <= 2

    # Check trail is recorded
    assert len(tracker.trails["green"]) == 1

    annotated = tracker.draw_tracked(img, objs, draw_trail=True)
    assert annotated.shape == img.shape


def test_sample_color_at():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (80, 80), (0, 255, 255), -1)  # Yellow
    ranges = ColorTracker.sample_color_at(img, 50, 50, radius=5)

    assert len(ranges) == 1
    lower, upper = ranges[0]
    assert len(lower) == 3 and len(upper) == 3
    # Check that sample correctly tracks yellow patch
    tracker = ColorTracker()
    mask = tracker.get_mask(img, ranges)
    assert np.count_nonzero(mask) > 100
