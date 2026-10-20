"""
Smart Document Scanner and 4-Point Perspective Transform Module.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Union
import numpy as np
import cv2

from vision_craft.config import UI_COLORS


@dataclass
class ScannedDocument:
    """Encapsulates the detected corners, perspective warped image, and enhanced scan."""
    warped: np.ndarray
    enhanced: np.ndarray
    corners: np.ndarray  # (4, 2) ordered: TL, TR, BR, BL
    contour: Optional[np.ndarray] = None


class DocumentScanner:
    """Detects document edges, performs perspective correction, and applies scan enhancements."""

    @staticmethod
    def order_points(pts: np.ndarray) -> np.ndarray:
        """
        Order coordinates in clockwise order:
        [top-left, top-right, bottom-right, bottom-left].
        """
        pts = pts.reshape(4, 2).astype(np.float32)
        rect = np.zeros((4, 2), dtype=np.float32)

        # Top-left has smallest sum (x + y), bottom-right has largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # Top-right has smallest diff (y - x), bottom-left has largest diff
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    @classmethod
    def four_point_transform(cls, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Obtain a top-down flattened perspective transform of the quadrangle specified by pts."""
        rect = cls.order_points(pts)
        (tl, tr, br, bl) = rect

        # Compute max width
        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        max_width = max(int(width_a), int(width_b))

        # Compute max height
        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_height = max(int(height_a), int(height_b))

        # Enforce minimum size to avoid degenerate matrices
        max_width = max(max_width, 10)
        max_height = max(max_height, 10)

        # Destination rectangle points
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
        return warped

    def find_document_corners(
        self,
        image: np.ndarray,
        min_area_ratio: float = 0.05,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Locate document quadrangle corners in the image.
        Returns (corners, best_contour).
        """
        h, w = image.shape[:2]
        total_area = w * h

        # Downscale for edge detection if large
        ratio = 1.0
        proc_img = image
        if h > 800 or w > 800:
            ratio = 800.0 / max(h, w)
            proc_img = cv2.resize(image, (int(w * ratio), int(h * ratio)))

        gray = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY) if len(proc_img.shape) == 3 else proc_img
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 180)

        # Close gaps in edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            area = cv2.contourArea(cnt)

            if len(approx) == 4 and area >= (min_area_ratio * (proc_img.shape[0] * proc_img.shape[1])):
                # Scale back to original resolution
                scaled_pts = (approx.reshape(4, 2) / ratio).astype(np.float32)
                return self.order_points(scaled_pts), cnt

        # Fallback 1: Convex hull of the largest contour
        if contours:
            largest = contours[0]
            if cv2.contourArea(largest) >= (min_area_ratio * (proc_img.shape[0] * proc_img.shape[1])):
                hull = cv2.convexHull(largest)
                peri = cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, 0.03 * peri, True)
                if len(approx) == 4:
                    scaled_pts = (approx.reshape(4, 2) / ratio).astype(np.float32)
                    return self.order_points(scaled_pts), largest
                # Bounding box of largest
                bx, by, bw, bh = cv2.boundingRect(largest)
                scaled_box = np.array([
                    [bx, by],
                    [bx + bw, by],
                    [bx + bw, by + bh],
                    [bx, by + bh]
                ], dtype=np.float32) / ratio
                return self.order_points(scaled_box), largest

        # Fallback 2: Entire image frame
        default_corners = np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]
        ], dtype=np.float32)
        return default_corners, None

    @staticmethod
    def enhance(warped: np.ndarray, mode: str = "color_enhanced") -> np.ndarray:
        """
        Enhance scanned document image.
        Modes:
          - 'bw': Crisp black & white binary threshold.
          - 'color_enhanced': Shadow removal while retaining colors.
          - 'gray': High-contrast smoothed grayscale.
          - 'original': No post-processing filter.
        """
        m = mode.lower()

        if m == "original":
            return warped.copy()

        elif m == "bw":
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if len(warped.shape) == 3 else warped
            blurred = cv2.medianBlur(gray, 3)
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
            )
            return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        elif m == "gray":
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if len(warped.shape) == 3 else warped
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(gray)
            return cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

        elif m == "color_enhanced":
            # Shadow removal via background division
            channels = cv2.split(warped)
            out_channels = []
            for ch in channels:
                dilated = cv2.dilate(ch, np.ones((7, 7), np.uint8))
                bg = cv2.medianBlur(dilated, 21)
                diff = 255 - cv2.absdiff(ch, bg)
                norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                out_channels.append(norm)
            return cv2.merge(out_channels)

        else:
            raise ValueError(f"Unknown document enhance mode: '{mode}'. Use 'bw', 'color_enhanced', 'gray', or 'original'.")

    def scan(self, image: np.ndarray, mode: str = "color_enhanced") -> ScannedDocument:
        """Execute complete scanning workflow: corner detection, perspective rectification, and enhancement."""
        corners, contour = self.find_document_corners(image)
        warped = self.four_point_transform(image, corners)
        enhanced = self.enhance(warped, mode=mode)
        return ScannedDocument(
            warped=warped,
            enhanced=enhanced,
            corners=corners,
            contour=contour,
        )

    def draw_document_bounds(
        self,
        image: np.ndarray,
        corners: np.ndarray,
        color: Optional[Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """Render detected document quadrangle border and corner handles."""
        output = image.copy()
        stroke_color = color or UI_COLORS["lime"]
        pts = corners.astype(int).reshape((-1, 1, 2))
        cv2.polylines(output, [pts], isClosed=True, color=stroke_color, thickness=3)

        corner_labels = ["TL", "TR", "BR", "BL"]
        for i, pt in enumerate(corners.astype(int)):
            cv2.circle(output, tuple(pt), 6, UI_COLORS["cyan"], -1)
            cv2.putText(
                output,
                corner_labels[i],
                (pt[0] + 8, pt[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return output
