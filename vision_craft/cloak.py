"""
Invisible cloak and chroma key compositing module.
"""

from typing import Union, List, Tuple, Optional
import cv2
import numpy as np

from vision_craft.config import HSV_COLOR_RANGES


class InvisibleCloak:
    """Creates the 'Harry Potter' invisible cloaking or chroma keying effect."""

    def __init__(
        self,
        cloak_color: Union[str, List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]] = "green",
        background: Optional[np.ndarray] = None,
    ):
        self.cloak_color = cloak_color
        self.background = background
        self._bg_accumulator: Optional[np.ndarray] = None
        self._bg_count = 0

    def set_background(self, bg_image: np.ndarray) -> None:
        """Explicitly set static background frame."""
        self.background = bg_image.copy()

    def calibrate_background(self, frame: np.ndarray, num_frames: int = 30) -> bool:
        """Accumulate multiple static frames to compute a clean noise-free background model."""
        if self._bg_accumulator is None:
            self._bg_accumulator = np.zeros(frame.shape, dtype=np.float32)
            self._bg_count = 0

        self._bg_accumulator += frame.astype(np.float32)
        self._bg_count += 1

        if self._bg_count >= num_frames:
            self.background = (self._bg_accumulator / self._bg_count).astype(np.uint8)
            return True
        return False

    def get_mask(self, frame: np.ndarray) -> np.ndarray:
        """Generate smooth binary mask for cloaking color."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if isinstance(self.cloak_color, str):
            ranges = HSV_COLOR_RANGES.get(self.cloak_color.lower())
            if ranges is None:
                raise ValueError(f"Unknown cloak color '{self.cloak_color}'. Available presets: {list(HSV_COLOR_RANGES.keys())}")
        else:
            ranges = self.cloak_color

        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)
            m = cv2.inRange(hsv, lower_np, upper_np)
            mask = cv2.bitwise_or(mask, m)

        # Refine mask edges
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=2)
        mask = cv2.dilate(mask, kernel_dilate, iterations=1)
        return mask

    def process_frame(
        self,
        frame: np.ndarray,
        replacement_background: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Composite current frame with background through color mask."""
        bg = replacement_background if replacement_background is not None else self.background
        if bg is None:
            # Fallback: if no background calibrated yet, return original frame
            return frame.copy()

        # Ensure background matches frame dimensions
        if bg.shape[:2] != frame.shape[:2]:
            bg = cv2.resize(bg, (frame.shape[1], frame.shape[0]))

        mask = self.get_mask(frame)
        mask_inv = cv2.bitwise_not(mask)

        # Segment out the cloak area from the live frame
        fg_part = cv2.bitwise_and(frame, frame, mask=mask_inv)

        # Segment in the background through the cloak mask
        bg_part = cv2.bitwise_and(bg, bg, mask=mask)

        # Seamless composite
        output = cv2.add(fg_part, bg_part)
        return output
