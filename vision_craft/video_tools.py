"""
Video frame extraction, keyframe detection, video splitting and trimming tools.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import json
import cv2
import numpy as np


def parse_time_to_seconds(time_str_or_num: Union[str, float, int]) -> float:
    """Parse time represented as float seconds or 'HH:MM:SS' / 'MM:SS' strings."""
    if isinstance(time_str_or_num, (int, float)):
        return float(time_str_or_num)

    parts = str(time_str_or_num).strip().split(":")
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 1:
        return float(parts[0])
    else:
        raise ValueError(f"Invalid timestamp format: '{time_str_or_num}'. Use 'HH:MM:SS' or seconds.")


def get_video_metadata(video_path: Union[str, Path]) -> Dict[str, Any]:
    """Retrieve technical metadata from video container."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
    duration_sec = frame_count / fps if fps > 0 else 0.0

    cap.release()

    return {
        "file_name": path.name,
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "duration_sec": round(duration_sec, 2),
        "fourcc": fourcc_str,
    }


class VideoFrameExtractor:
    """Extracts frames from video files based on intervals, frame steps, or scene change detection."""

    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_by_interval(
        self,
        video_path: Union[str, Path],
        interval_seconds: float = 1.0,
        prefix: str = "frame",
        image_format: str = "jpg",
    ) -> List[Path]:
        """Extract one frame every N seconds."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(round(fps * interval_seconds)))
        extracted_paths: List[Path] = []
        metadata_records: List[Dict[str, Any]] = []

        frame_idx = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / fps
                filename = f"{prefix}_{saved_count:05d}_t{timestamp:.2f}s.{image_format}"
                out_path = self.output_dir / filename
                cv2.imwrite(str(out_path), frame)
                extracted_paths.append(out_path)
                metadata_records.append({
                    "index": saved_count,
                    "frame_number": frame_idx,
                    "timestamp_sec": round(timestamp, 2),
                    "file_path": str(out_path.name),
                })
                saved_count += 1

            frame_idx += 1

        cap.release()

        # Save manifest
        report_file = self.output_dir / "extraction_manifest.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump({"total_extracted": saved_count, "frames": metadata_records}, f, indent=2)

        return extracted_paths

    def extract_by_step(
        self,
        video_path: Union[str, Path],
        step_frames: int = 30,
        prefix: str = "frame",
        image_format: str = "jpg",
    ) -> List[Path]:
        """Extract every Nth frame."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        extracted_paths: List[Path] = []
        frame_idx = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step_frames == 0:
                filename = f"{prefix}_{saved_count:05d}.{image_format}"
                out_path = self.output_dir / filename
                cv2.imwrite(str(out_path), frame)
                extracted_paths.append(out_path)
                saved_count += 1

            frame_idx += 1

        cap.release()
        return extracted_paths

    def extract_keyframes(
        self,
        video_path: Union[str, Path],
        diff_threshold: float = 35.0,
        min_interval_frames: int = 15,
        prefix: str = "keyframe",
        image_format: str = "jpg",
    ) -> List[Path]:
        """Extract keyframes when scene or visual changes exceed diff_threshold."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        extracted_paths: List[Path] = []
        prev_gray: Optional[np.ndarray] = None
        last_saved_frame = -min_interval_frames
        frame_idx = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Downsample for fast scene difference calculation
            small_gray = cv2.resize(gray, (160, 120))

            is_keyframe = False
            if prev_gray is None:
                is_keyframe = True
            elif (frame_idx - last_saved_frame) >= min_interval_frames:
                diff = cv2.absdiff(prev_gray, small_gray)
                mean_diff = float(np.mean(diff))
                if mean_diff >= diff_threshold:
                    is_keyframe = True

            if is_keyframe:
                timestamp = frame_idx / fps
                filename = f"{prefix}_{saved_count:04d}_t{timestamp:.2f}s.{image_format}"
                out_path = self.output_dir / filename
                cv2.imwrite(str(out_path), frame)
                extracted_paths.append(out_path)
                last_saved_frame = frame_idx
                saved_count += 1

            prev_gray = small_gray
            frame_idx += 1

        cap.release()
        return extracted_paths


class VideoSplitter:
    """Trims or splits video files into specified time intervals."""

    @staticmethod
    def split(
        video_path: Union[str, Path],
        output_path: Union[str, Path],
        start_time: Union[str, float, int],
        end_time: Union[str, float, int],
        fourcc_code: str = "mp4v",
    ) -> Path:
        """Trim video segment between start_time and end_time and save to output_path."""
        vpath = Path(video_path)
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        start_sec = parse_time_to_seconds(start_time)
        end_sec = parse_time_to_seconds(end_time)

        if end_sec <= start_sec:
            raise ValueError(f"End time ({end_sec}s) must be greater than start time ({start_sec}s).")

        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {vpath}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        start_frame = int(round(start_sec * fps))
        end_frame = int(round(end_sec * fps))

        fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame
        saved_frames = 0

        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            saved_frames += 1
            current_frame += 1

        writer.release()
        cap.release()

        if saved_frames == 0:
            raise RuntimeError(f"No frames were written to {out_path}.")

        return out_path
