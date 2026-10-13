"""
Motion detection, background subtraction, intrusion tracking, and heatmap generation module.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
import cv2
import numpy as np

from vision_craft.config import UI_COLORS, DEFAULT_MOTION_MIN_AREA


@dataclass
class MotionResult:
    """Encapsulates motion detection outputs for a single frame."""
    has_motion: bool
    motion_boxes: List[Tuple[int, int, int, int]]  # x, y, w, h
    motion_centroids: List[Tuple[int, int]]
    total_motion_area: float
    foreground_mask: np.ndarray


class MotionTracker:
    """Tracks visual motion via MOG2, KNN, or frame differencing, with accumulated heatmap support."""

    def __init__(
        self,
        method: str = "mog2",
        min_area: float = DEFAULT_MOTION_MIN_AREA,
        history: int = 500,
        var_threshold: float = 16.0,
        detect_shadows: bool = True,
        on_motion_callback: Optional[Callable[[MotionResult], None]] = None,
    ):
        self.method = method.lower()
        self.min_area = min_area
        self.on_motion_callback = on_motion_callback
        self.prev_gray: Optional[np.ndarray] = None
        self.heatmap_accumulator: Optional[np.ndarray] = None

        if self.method == "mog2":
            self.subtractor = cv2.createBackgroundSubtractorMOG2(
                history=history, varThreshold=var_threshold, detectShadows=detect_shadows
            )
        elif self.method == "knn":
            self.subtractor = cv2.createBackgroundSubtractorKNN(
                history=history, dist2Threshold=400.0, detectShadows=detect_shadows
            )
        elif self.method == "frame_diff":
            self.subtractor = None
        else:
            raise ValueError(f"Unknown motion method: '{method}'. Choose 'mog2', 'knn', or 'frame_diff'.")

    def reset(self) -> None:
        """Reset motion state and accumulated heatmaps."""
        self.prev_gray = None
        self.heatmap_accumulator = None
        if hasattr(self, "subtractor") and self.subtractor is not None:
            # Recreate subtractor
            if self.method == "mog2":
                self.subtractor = cv2.createBackgroundSubtractorMOG2()
            elif self.method == "knn":
                self.subtractor = cv2.createBackgroundSubtractorKNN()

    def process_frame(self, frame: np.ndarray) -> MotionResult:
        """Process incoming video frame and detect motion regions."""
        h, w = frame.shape[:2]
        if self.heatmap_accumulator is None:
            self.heatmap_accumulator = np.zeros((h, w), dtype=np.float32)

        # Compute foreground mask
        if self.method in ("mog2", "knn"):
            fg_mask = self.subtractor.apply(frame)
            # Remove shadows (shadows are marked as 127 in MOG2/KNN)
            _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        else:
            # Frame differencing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            if self.prev_gray is None:
                self.prev_gray = gray
                fg_mask = np.zeros((h, w), dtype=np.uint8)
            else:
                frame_diff = cv2.absdiff(self.prev_gray, gray)
                _, fg_mask = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
                self.prev_gray = gray

        # Morphological noise filtering
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned_mask = cv2.dilate(fg_mask, kernel, iterations=2)
        cleaned_mask = cv2.erode(cleaned_mask, kernel, iterations=1)

        # Accumulate motion intensity into heatmap
        self.heatmap_accumulator += (cleaned_mask > 0).astype(np.float32)

        # Extract contours
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_boxes: List[Tuple[int, int, int, int]] = []
        motion_centroids: List[Tuple[int, int]] = []
        total_area = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.min_area:
                x, y, bw, bh = cv2.boundingRect(cnt)
                motion_boxes.append((x, y, bw, bh))
                motion_centroids.append((x + bw // 2, y + bh // 2))
                total_area += area

        has_motion = len(motion_boxes) > 0
        result = MotionResult(
            has_motion=has_motion,
            motion_boxes=motion_boxes,
            motion_centroids=motion_centroids,
            total_motion_area=total_area,
            foreground_mask=cleaned_mask,
        )

        if has_motion and self.on_motion_callback is not None:
            self.on_motion_callback(result)

        return result

    def get_heatmap(self, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """Render normalized accumulated motion heatmap."""
        if self.heatmap_accumulator is None or np.max(self.heatmap_accumulator) == 0:
            h, w = (480, 640) if self.heatmap_accumulator is None else self.heatmap_accumulator.shape
            return np.zeros((h, w, 3), dtype=np.uint8)

        # Normalize to 0-255
        max_val = np.max(self.heatmap_accumulator)
        norm = (self.heatmap_accumulator / max_val * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(norm, colormap)
        return heatmap

    def draw_motion(
        self,
        frame: np.ndarray,
        result: MotionResult,
        draw_boxes: bool = True,
        blend_heatmap: bool = False,
        heatmap_alpha: float = 0.4,
    ) -> np.ndarray:
        """Render motion bounding boxes, alert badge, and optional heatmap blend."""
        output = frame.copy()

        if blend_heatmap and self.heatmap_accumulator is not None:
            heatmap = self.get_heatmap()
            output = cv2.addWeighted(output, 1.0 - heatmap_alpha, heatmap, heatmap_alpha, 0)

        if draw_boxes:
            for x, y, bw, bh in result.motion_boxes:
                cv2.rectangle(output, (x, y), (x + bw, y + bh), UI_COLORS["red"], 2)
                cv2.circle(output, (x + bw // 2, y + bh // 2), 4, UI_COLORS["yellow"], -1)

        # Render status banner
        if result.has_motion:
            msg = f"[MOTION DETECTED] Regions: {len(result.motion_boxes)} | Area: {int(result.total_motion_area)}px"
            bg_color = (0, 0, 180)  # Red alert
        else:
            msg = "[STATUS: SECURE] No Motion"
            bg_color = (0, 120, 0)  # Green safe

        cv2.rectangle(output, (10, 10), (450, 40), bg_color, -1)
        cv2.putText(
            output,
            msg,
            (18, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return output
