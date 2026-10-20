"""
Mathematical morphology engine with structuring elements and morphological operators.
"""

from typing import Union, Tuple, Optional
import cv2
import numpy as np


class MorphologyEngine:
    """Performs mathematical morphology transformations on grayscale or binary images."""

    @staticmethod
    def get_kernel(shape: str = "rect", ksize: Union[int, Tuple[int, int]] = 5) -> np.ndarray:
        """Create structuring element of desired shape ('rect', 'cross', 'ellipse')."""
        if isinstance(ksize, int):
            sz = (ksize, ksize)
        else:
            sz = ksize

        s = shape.lower()
        if s == "cross":
            morph_shape = cv2.MORPH_CROSS
        elif s == "ellipse":
            morph_shape = cv2.MORPH_ELLIPSE
        else:
            morph_shape = cv2.MORPH_RECT

        return cv2.getStructuringElement(morph_shape, sz)

    @classmethod
    def apply_operation(
        cls,
        image: np.ndarray,
        op_name: str,
        ksize: int = 5,
        shape: str = "rect",
        iterations: int = 1,
    ) -> np.ndarray:
        """Execute selected morphological operation by name."""
        kernel = cls.get_kernel(shape, ksize)
        op = op_name.lower().replace("-", "_")

        if op in ("erode", "erosion"):
            return cv2.erode(image, kernel, iterations=iterations)
        elif op in ("dilate", "dilation"):
            return cv2.dilate(image, kernel, iterations=iterations)
        elif op in ("open", "opening"):
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
        elif op in ("close", "closing"):
            return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        elif op in ("gradient", "morph_gradient"):
            return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
        elif op in ("tophat", "top_hat", "whitehat"):
            return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        elif op in ("blackhat", "black_hat"):
            return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        elif op in ("skeleton", "skeletonize"):
            return cls.skeletonize(image)
        else:
            valid = "erode, dilate, open, close, gradient, tophat, blackhat, skeleton"
            raise ValueError(f"Unknown morphology operation: '{op_name}'. Valid: {valid}")

    @staticmethod
    def skeletonize(image: np.ndarray, max_iterations: int = 100) -> np.ndarray:
        """Iteratively thin binary contours down to a 1-pixel-wide topological skeleton."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        skel = np.zeros(binary.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        temp = binary.copy()

        for _ in range(max_iterations):
            eroded = cv2.erode(temp, element)
            opened = cv2.dilate(eroded, element)
            subset = cv2.subtract(temp, opened)
            skel = cv2.bitwise_or(skel, subset)
            temp = eroded.copy()
            if cv2.countNonZero(temp) == 0:
                break

        if len(image.shape) == 3:
            return cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)
        return skel
