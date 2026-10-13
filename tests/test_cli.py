"""
Unit tests for VisionCraft CLI execution.
"""

from pathlib import Path
import pytest
import cv2
import numpy as np

from vision_craft.cli import main, build_parser
from vision_craft.synthetic import SyntheticGenerator


def test_cli_help():
    parser = build_parser()
    assert parser is not None
    # main with empty args prints help and returns 0
    assert main([]) == 0


def test_cli_filter_shapes_and_demo(tmp_path: Path):
    # 1. Generate demo
    demo_dir = tmp_path / "demos"
    code = main(["generate-demo", "-o", str(demo_dir)])
    assert code == 0
    assert (demo_dir / "shapes_sample.png").exists()

    # 2. Filter command
    filtered_out = tmp_path / "canny.png"
    code = main(["filter", "-i", str(demo_dir / "shapes_sample.png"), "-t", "canny", "-o", str(filtered_out)])
    assert code == 0
    assert filtered_out.exists()

    # 3. Shapes command
    shapes_out = tmp_path / "shapes_detected.png"
    code = main(["shapes", "-i", str(demo_dir / "shapes_sample.png"), "-o", str(shapes_out)])
    assert code == 0
    assert shapes_out.exists()

    # 4. Color tracking command
    color_out = tmp_path / "color_tracked.png"
    code = main(["color", "-i", str(demo_dir / "colors_sample.png"), "-c", "red", "-o", str(color_out)])
    assert code == 0
    assert color_out.exists()


def test_cli_motion_and_split(tmp_path: Path):
    vid = tmp_path / "motion.mp4"
    SyntheticGenerator.generate_motion_video(vid, num_frames=30, fps=15)

    motion_out = tmp_path / "motion_annotated.mp4"
    heatmap_out = tmp_path / "heatmap.png"
    code = main([
        "motion",
        "-i", str(vid),
        "-o", str(motion_out),
        "-m", "frame_diff",
        "--heatmap", str(heatmap_out)
    ])
    assert code == 0
    assert motion_out.exists()
    assert heatmap_out.exists()

    split_out = tmp_path / "split.mp4"
    code = main([
        "split",
        "-i", str(vid),
        "-o", str(split_out),
        "--start", "00:00:00",
        "--end", "00:00:01"
    ])
    assert code == 0
    assert split_out.exists()
