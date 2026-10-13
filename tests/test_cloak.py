"""
Unit tests for InvisibleCloak and chroma keying.
"""

import pytest
import numpy as np
import cv2

from vision_craft.cloak import InvisibleCloak


def test_invisible_cloak_pipeline():
    # Background: plain blue scene
    bg = np.zeros((200, 200, 3), dtype=np.uint8)
    bg[:, :] = (255, 0, 0)  # Blue

    cloak = InvisibleCloak(cloak_color="green", background=bg)

    # Frame: Red object wearing green cloak
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[:, :] = (0, 0, 255)  # Red background person
    # Green cloaked region in center
    cv2.rectangle(frame, (50, 50), (150, 150), (0, 255, 0), -1)

    composite = cloak.process_frame(frame)
    assert composite.shape == frame.shape

    # Outside cloak, pixels should be predominantly red (0, 0, 255)
    assert composite[10, 10, 2] > 200

    # Inside cloak, pixels should be replaced by background (blue: 255, 0, 0)
    assert composite[100, 100, 0] > 200
    assert composite[100, 100, 1] < 50


def test_calibrate_background():
    cloak = InvisibleCloak(cloak_color="green")
    frame = np.full((100, 100, 3), 120, dtype=np.uint8)

    calibrated = False
    for _ in range(10):
        if cloak.calibrate_background(frame, num_frames=10):
            calibrated = True

    assert calibrated
    assert cloak.background is not None
    assert cloak.background.shape == (100, 100, 3)
