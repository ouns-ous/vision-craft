"""
Unit tests for OpticalFlowAnalyzer module.
"""

from pathlib import Path
import pytest
import numpy as np
import cv2

from vision_craft.optical_flow import OpticalFlowAnalyzer, FlowResult
from vision_craft.synthetic import SyntheticGenerator


@pytest.fixture
def moving_circle_frames() -> Tuple[np.ndarray, np.ndarray]:
    """Create two consecutive frames with a horizontally moving circle."""
    f1 = np.full((160, 160, 3), 30, dtype=np.uint8)
    cv2.circle(f1, (60, 80), 20, (255, 255, 255), -1)

    f2 = np.full((160, 160, 3), 30, dtype=np.uint8)
    cv2.circle(f2, (85, 80), 20, (255, 255, 255), -1)  # Shifted right by 25px
    return f1, f2


def test_dense_optical_flow(moving_circle_frames):
    f1, f2 = moving_circle_frames
    analyzer = OpticalFlowAnalyzer()
    res = analyzer.compute_dense_flow(f1, f2)

    assert isinstance(res, FlowResult)
    assert res.flow.shape == (160, 160, 2)
    assert res.magnitude.shape == (160, 160)
    assert res.angle.shape == (160, 160)
    assert res.vis.shape == f1.shape

    # Motion should be detected in the region of the circle
    assert np.max(res.magnitude) > 5.0


def test_draw_vector_field(moving_circle_frames):
    f1, f2 = moving_circle_frames
    analyzer = OpticalFlowAnalyzer()
    res = analyzer.compute_dense_flow(f1, f2)

    vis = analyzer.draw_vector_field(f2, res.flow, step=8, min_mag=2.0)
    assert vis.shape == f2.shape
    assert vis.dtype == np.uint8
    assert not np.array_equal(vis, f2)


def test_process_video_optical_flow(tmp_path: Path):
    vid_in = tmp_path / "in.mp4"
    vid_out = tmp_path / "out.mp4"
    SyntheticGenerator.generate_motion_video(vid_in, num_frames=15, width=160, height=120, fps=10)

    analyzer = OpticalFlowAnalyzer()
    count = analyzer.process_video(vid_in, vid_out, mode="vectors", max_frames=10)

    assert count >= 5
    assert vid_out.exists()
    assert vid_out.stat().st_size > 0
