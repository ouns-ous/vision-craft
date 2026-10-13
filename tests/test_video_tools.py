"""
Unit tests for VideoFrameExtractor and VideoSplitter.
"""

from pathlib import Path
import pytest
import numpy as np

from vision_craft.video_tools import (
    VideoFrameExtractor,
    VideoSplitter,
    get_video_metadata,
    parse_time_to_seconds,
)
from vision_craft.synthetic import SyntheticGenerator


@pytest.fixture
def test_video(tmp_path: Path) -> Path:
    vid_path = tmp_path / "sample_video.mp4"
    return SyntheticGenerator.generate_motion_video(vid_path, num_frames=60, fps=30)


def test_parse_time_to_seconds():
    assert parse_time_to_seconds(15) == 15.0
    assert parse_time_to_seconds("00:01:30") == 90.0
    assert parse_time_to_seconds("02:15") == 135.0
    assert parse_time_to_seconds("45.5") == 45.5


def test_video_metadata(test_video: Path):
    meta = get_video_metadata(test_video)
    assert meta["width"] == 320
    assert meta["height"] == 240
    assert meta["frame_count"] == 60
    assert meta["fps"] == 30.0
    assert meta["duration_sec"] == 2.0


def test_extract_frames_by_step(test_video: Path, tmp_path: Path):
    out_dir = tmp_path / "extracted_steps"
    extractor = VideoFrameExtractor(out_dir)
    extracted = extractor.extract_by_step(test_video, step_frames=20)

    # 60 frames / step 20 = 3 frames (0, 20, 40)
    assert len(extracted) == 3
    for p in extracted:
        assert p.exists()


def test_extract_frames_by_interval(test_video: Path, tmp_path: Path):
    out_dir = tmp_path / "extracted_interval"
    extractor = VideoFrameExtractor(out_dir)
    extracted = extractor.extract_by_interval(test_video, interval_seconds=1.0)

    assert len(extracted) >= 2
    assert (out_dir / "extraction_manifest.json").exists()


def test_video_splitter(test_video: Path, tmp_path: Path):
    out_path = tmp_path / "trimmed.mp4"
    split_res = VideoSplitter.split(test_video, out_path, start_time="00:00:00", end_time="00:00:01")

    assert split_res.exists()
    meta = get_video_metadata(split_res)
    # 1 second clip at 30 fps ~= 30-31 frames
    assert 28 <= meta["frame_count"] <= 32
