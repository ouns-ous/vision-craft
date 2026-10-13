"""
Real-time HSV color detection, segmentation, and trajectory tracking module.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Union
import cv2
import numpy as np

from vision_craft.config import (
    HSV_COLOR_RANGES,
    UI_COLORS,
    DEFAULT_MIN_CONTOUR_AREA,
    DEFAULT_TRAIL_LENGTH,
)


@dataclass
class TrackedColorObject:
    """Represents a detected color entity within a frame."""
    color_label: str
    centroid: Tuple[int, int]
    bounding_box: Tuple[int, int, int, int]  # x, y, w, h
    radius: int
    area: float
    contour: np.ndarray


class ColorTracker:
    """Tracks objects of specific colors and computes motion trajectories."""

    def __init__(self, max_trail_length: int = DEFAULT_TRAIL_LENGTH):
        self.max_trail_length = max_trail_length
        # Key: color_label -> deque of (cx, cy)
        self.trails: Dict[str, deque[Tuple[int, int]]] = {}

    def get_mask(
        self,
        frame: np.ndarray,
        color_target: Union[str, List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]],
    ) -> np.ndarray:
        """Create a clean binary mask for the given target color or custom HSV range."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Resolve ranges
        if isinstance(color_target, str):
            ranges = HSV_COLOR_RANGES.get(color_target.lower())
            if ranges is None:
                raise ValueError(
                    f"Unknown color target '{color_target}'. Available presets: {list(HSV_COLOR_RANGES.keys())}"
                )
        else:
            ranges = color_target

        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_np, upper_np)
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Morphological noise removal
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.erode(combined_mask, kernel, iterations=2)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)
        return combined_mask

    def track(
        self,
        frame: np.ndarray,
        color_target: Union[str, List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]],
        min_area: float = DEFAULT_MIN_CONTOUR_AREA,
        color_label: Optional[str] = None,
    ) -> Tuple[np.ndarray, List[TrackedColorObject]]:
        """Identify color regions, compute centroids, and update trajectory trail."""
        label = color_label or (color_target if isinstance(color_target, str) else "custom")
        mask = self.get_mask(frame, color_target)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tracked_objects: List[TrackedColorObject] = []

        if label not in self.trails:
            self.trails[label] = deque(maxlen=self.max_trail_length)

        # Sort by contour area descending to track primary target first
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        primary_centroid: Optional[Tuple[int, int]] = None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            m = cv2.moments(cnt)
            if m["m00"] != 0:
                cx = int(m["m10"] / m["m00"])
                cy = int(m["m01"] / m["m00"])
            else:
                cx, cy = int(x), int(y)

            bx, by, bw, bh = cv2.boundingRect(cnt)

            obj = TrackedColorObject(
                color_label=label,
                centroid=(cx, cy),
                bounding_box=(bx, by, bw, bh),
                radius=int(radius),
                area=area,
                contour=cnt,
            )
            tracked_objects.append(obj)

            if primary_centroid is None:
                primary_centroid = (cx, cy)

        # Update trajectory trail for the primary tracked entity
        if primary_centroid is not None:
            self.trails[label].appendleft(primary_centroid)
        else:
            # Gradually clear trail when object disappears
            if len(self.trails[label]) > 0:
                self.trails[label].pop()

        return mask, tracked_objects

    def draw_tracked(
        self,
        frame: np.ndarray,
        tracked_objects: List[TrackedColorObject],
        draw_trail: bool = True,
    ) -> np.ndarray:
        """Render bounding boxes, centroids, labels, and motion trails on frame."""
        output = frame.copy()

        for obj in tracked_objects:
            cx, cy = obj.centroid
            # Enclosing circle and centroid point
            cv2.circle(output, (cx, cy), obj.radius, UI_COLORS["cyan"], 2)
            cv2.circle(output, (cx, cy), 5, UI_COLORS["red"], -1)

            # Bounding box
            bx, by, bw, bh = obj.bounding_box
            cv2.rectangle(output, (bx, by), (bx + bw, by + bh), UI_COLORS["yellow"], 1)

            # Label badge
            label = f"{obj.color_label.upper()} ({cx},{cy})"
            cv2.putText(
                output,
                label,
                (bx, max(15, by - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Draw motion trajectory trail
        if draw_trail:
            for label, trail in self.trails.items():
                pts = list(trail)
                for i in range(1, len(pts)):
                    if pts[i - 1] is None or pts[i] is None:
                        continue
                    # Fade thickness along history
                    thickness = int(np.sqrt(self.max_trail_length / float(i + 1)) * 2)
                    thickness = max(1, thickness)
                    cv2.line(output, pts[i - 1], pts[i], UI_COLORS["lime"], thickness)

        return output

    def reset_trails(self) -> None:
        """Clear all active trajectory paths."""
        self.trails.clear()

    @staticmethod
    def sample_color_at(
        frame: np.ndarray, x: int, y: int, radius: int = 5
    ) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
        """Sample average HSV color around (x, y) and return a tight tracking HSV range."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]

        x1 = max(0, x - radius)
        x2 = min(w, x + radius)
        y1 = max(0, y - radius)
        y2 = min(h, y + radius)

        roi = hsv[y1:y2, x1:x2]
        avg_h = int(np.mean(roi[:, :, 0]))
        avg_s = int(np.mean(roi[:, :, 1]))
        avg_v = int(np.mean(roi[:, :, 2]))

        lower_hsv = (max(0, avg_h - 15), max(50, avg_s - 50), max(50, avg_v - 50))
        upper_hsv = (min(179, avg_h + 15), min(255, avg_s + 50), min(255, avg_v + 50))

        return [(lower_hsv, upper_hsv)]
