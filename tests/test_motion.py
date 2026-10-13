"""
Unit tests for MotionTracker and Heatmap generation.
"""

import pytest
import numpy as np
import cv2

from vision_craft.motion import MotionTracker, MotionResult


def test_motion_frame_diff():
    tracker = MotionTracker(method="frame_diff", min_area=100)

    # Frame 1: empty static frame
    f1 = np.full((200, 200, 3), 50, dtype=np.uint8)
    res1 = tracker.process_frame(f1)
    # First frame has no prior reference, so no motion
    assert not res1.has_motion

    # Frame 2: white rectangle appears in frame
    f2 = f1.copy()
    cv2.rectangle(f2, (50, 50), (120, 120), (255, 255, 255), -1)
    res2 = tracker.process_frame(f2)
    assert res2.has_motion
    assert len(res2.motion_boxes) >= 1
    assert res2.total_motion_area >= 100

    # Draw motion overlay
    annotated = tracker.draw_motion(f2, res2, draw_boxes=True, blend_heatmap=True)
    assert annotated.shape == f2.shape

    # Check heatmap export
    heatmap = tracker.get_heatmap()
    assert heatmap.shape == (200, 200, 3)
    assert heatmap.dtype == np.uint8


def test_motion_mog2():
    tracker = MotionTracker(method="mog2", min_area=100)
    base = np.full((150, 150, 3), 80, dtype=np.uint8)

    # Prime background model with several identical frames
    for _ in range(5):
        tracker.process_frame(base)

    # Inject motion
    moving = base.copy()
    cv2.circle(moving, (75, 75), 25, (255, 255, 255), -1)
    res = tracker.process_frame(moving)

    assert res.foreground_mask.shape == (150, 150)
    assert np.count_nonzero(res.foreground_mask) > 50
