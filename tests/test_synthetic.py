"""
Unit tests for SyntheticGenerator.
"""

from pathlib import Path
import pytest
import numpy as np

from vision_craft.synthetic import SyntheticGenerator


def test_generate_synthetic_assets(tmp_path: Path):
    shapes = SyntheticGenerator.generate_shapes_image(320, 240)
    assert shapes.shape == (240, 320, 3)
    assert shapes.dtype == np.uint8

    colors = SyntheticGenerator.generate_color_palette_image(320, 240)
    assert colors.shape == (240, 320, 3)

    face = SyntheticGenerator.generate_face_like_image(320, 240)
    assert face.shape == (240, 320, 3)

    v1 = tmp_path / "v1.mp4"
    SyntheticGenerator.generate_motion_video(v1, num_frames=10, fps=10)
    assert v1.exists()
    assert v1.stat().st_size > 0

    v2 = tmp_path / "v2.mp4"
    SyntheticGenerator.generate_tracking_video(v2, num_frames=10, fps=10)
    assert v2.exists()
    assert v2.stat().st_size > 0
