"""
Shape and contour analysis module with geometric classification.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import math
import cv2
import numpy as np

from vision_craft.config import UI_COLORS, DEFAULT_MIN_CONTOUR_AREA, DEFAULT_MAX_CONTOUR_AREA


@dataclass
class DetectedShape:
    """Represents a classified geometric contour."""
    name: str
    vertices_count: int
    area: float
    perimeter: float
    centroid: Tuple[int, int]
    bounding_box: Tuple[int, int, int, int]  # x, y, w, h
    contour: np.ndarray
    approx_contour: np.ndarray
    aspect_ratio: float
    circularity: float
    solidity: float


class ShapeDetector:
    """Detects and classifies 2D geometric shapes from binary/grayscale contours."""

    def __init__(self, epsilon_factor: float = 0.04):
        self.epsilon_factor = epsilon_factor

    def classify_contour(self, contour: np.ndarray) -> DetectedShape:
        """Analyze and classify a single contour into a geometric shape."""
        perimeter = cv2.arcLength(contour, True)
        epsilon = self.epsilon_factor * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        area = cv2.contourArea(contour)

        # Centroid from spatial moments
        m = cv2.moments(contour)
        if m["m00"] != 0:
            cx = int(m["m01"] / m["m00"]) # Wait: moments m10 is x, m01 is y!
            cy = int(m["m01"] / m["m00"])
            cx = int(m["m10"] / m["m00"])
        else:
            cx, cy = 0, 0

        # Bounding box & metrics
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0.0

        # Circularity metric: 4 * pi * Area / (Perimeter^2)
        circularity = (4.0 * math.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0.0

        # Solidity: Area / Hull Area
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0.0

        # Geometric shape classification
        vertices = len(approx)
        shape_name = "Unknown"

        if vertices == 3:
            shape_name = "Triangle"
        elif vertices == 4:
            # Check aspect ratio to distinguish Square vs Rectangle
            if 0.90 <= aspect_ratio <= 1.10:
                shape_name = "Square"
            else:
                shape_name = "Rectangle"
        elif vertices == 5:
            shape_name = "Pentagon"
        elif vertices == 6:
            shape_name = "Hexagon"
        else:
            if circularity >= 0.75:
                shape_name = "Circle"
            elif solidity < 0.70:
                shape_name = "Star"
            else:
                shape_name = f"Polygon ({vertices}-gon)"

        return DetectedShape(
            name=shape_name,
            vertices_count=vertices,
            area=area,
            perimeter=perimeter,
            centroid=(cx, cy),
            bounding_box=(x, y, w, h),
            contour=contour,
            approx_contour=approx,
            aspect_ratio=round(aspect_ratio, 3),
            circularity=round(circularity, 3),
            solidity=round(solidity, 3),
        )

    def detect(
        self,
        image: np.ndarray,
        min_area: float = DEFAULT_MIN_CONTOUR_AREA,
        max_area: float = DEFAULT_MAX_CONTOUR_AREA,
        preprocess: bool = True
    ) -> List[DetectedShape]:
        """Extract contours from an image and return detected shapes."""
        img_h, img_w = image.shape[:2]
        max_frame_area = 0.85 * (img_w * img_h)

        if preprocess:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            if np.mean(blurred) < 127:
                # Dark background, bright shapes
                _, thresh = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                # Light background, dark shapes
                _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            thresh = image

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_shapes: List[DetectedShape] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area <= area <= min(max_area, max_frame_area):
                shape = self.classify_contour(cnt)
                detected_shapes.append(shape)

        return detected_shapes

    def draw_shapes(
        self,
        image: np.ndarray,
        shapes: List[DetectedShape],
        show_details: bool = True,
        color: Optional[Tuple[int, int, int]] = None
    ) -> np.ndarray:
        """Render detected shapes, contours, centroids, and metadata on image."""
        output = image.copy()
        stroke_color = color if color is not None else UI_COLORS["lime"]

        for s in shapes:
            # Draw contour and approximated polygon
            cv2.drawContours(output, [s.contour], -1, stroke_color, 2)
            cv2.drawContours(output, [s.approx_contour], -1, UI_COLORS["cyan"], 1)

            # Draw centroid point
            cx, cy = s.centroid
            cv2.circle(output, (cx, cy), 4, UI_COLORS["red"], -1)

            # Draw bounding box
            x, y, w, h = s.bounding_box
            cv2.rectangle(output, (x, y), (x + w, y + h), UI_COLORS["yellow"], 1)

            # Draw text label
            label = s.name
            if show_details:
                label += f" | A:{int(s.area)}"

            # Label badge
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            tx = max(0, x)
            ty = max(text_size[1] + 4, y - 6)
            cv2.rectangle(
                output,
                (tx, ty - text_size[1] - 4),
                (tx + text_size[0] + 6, ty + 2),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                output,
                label,
                (tx + 3, ty - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return output
