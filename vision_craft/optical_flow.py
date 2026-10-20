"""
Dense and sparse optical flow motion estimation and vector field visualization module.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Union
import cv2
import numpy as np

from vision_craft.config import UI_COLORS


@dataclass
class FlowResult:
    """Encapsulates computed flow matrix and colorwheel / vector visualization."""
    flow: np.ndarray  # (H, W, 2) array of (dx, dy)
    magnitude: np.ndarray
    angle: np.ndarray
    vis: np.ndarray


class OpticalFlowAnalyzer:
    """Computes motion vector fields using Farneback dense and Lucas-Kanade sparse algorithms."""

    def __init__(self, pyr_scale: float = 0.5, levels: int = 3, winsize: int = 15, iterations: int = 3):
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations

    def compute_dense_flow(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
    ) -> FlowResult:
        """Calculate Gunnar Farneback dense optical flow and encode into HSV colorwheel representation."""
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if len(prev_frame.shape) == 3 else prev_frame
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY) if len(curr_frame.shape) == 3 else curr_frame

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            self.pyr_scale,
            self.levels,
            self.winsize,
            self.iterations,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # HSV colorwheel encoding: Hue = direction angle, Value = normalized velocity magnitude
        hsv = np.zeros_like(prev_frame)
        hsv[..., 0] = ang * 180 / np.pi / 2  # Hue [0, 179]
        hsv[..., 1] = 255                    # Saturation full
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value [0, 255]

        vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return FlowResult(
            flow=flow,
            magnitude=mag,
            angle=ang,
            vis=vis,
        )

    @staticmethod
    def draw_vector_field(
        frame: np.ndarray,
        flow: np.ndarray,
        step: int = 16,
        color: Optional[Tuple[int, int, int]] = None,
        min_mag: float = 1.0,
    ) -> np.ndarray:
        """Render optical flow motion vector arrows over a frame grid."""
        output = frame.copy()
        stroke_color = color or UI_COLORS["lime"]
        h, w = frame.shape[:2]

        y, x = np.mgrid[step // 2:h:step, step // 2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T

        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)

        for (x1, y1), (x2, y2) in lines:
            if np.hypot(x2 - x1, y2 - y1) >= min_mag:
                cv2.arrowedLine(output, (x1, y1), (x2, y2), stroke_color, 1, tipLength=0.3)
                cv2.circle(output, (x1, y1), 1, UI_COLORS["yellow"], -1)

        return output

    def process_video(
        self,
        video_path: Union[str, Path],
        output_path: Union[str, Path],
        mode: str = "dense",
        max_frames: Optional[int] = None,
    ) -> int:
        """Process video and render dense HSV colorwheel or vector field arrows."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            writer.release()
            return 0

        frames_written = 0
        while True:
            if max_frames and frames_written >= max_frames:
                break
            ret, curr_frame = cap.read()
            if not ret:
                break

            flow_res = self.compute_dense_flow(prev_frame, curr_frame)
            if mode == "vectors":
                vis = self.draw_vector_field(curr_frame, flow_res.flow, step=16)
            else:
                vis = flow_res.vis

            writer.write(vis)
            prev_frame = curr_frame
            frames_written += 1

        cap.release()
        writer.release()
        return frames_written
