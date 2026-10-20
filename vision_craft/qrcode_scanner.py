"""
QR Code encoding, detection, decoding, and visual overlay module.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Any
import json
import cv2
import numpy as np

from vision_craft.config import UI_COLORS


@dataclass
class QRScanResult:
    """Represents a localized and decoded QR code."""
    data: str
    polygon: np.ndarray  # (4, 2) or (4, 1, 2) corner points
    bounding_box: Tuple[int, int, int, int]  # x, y, w, h
    straight_qr: Optional[np.ndarray] = None


class QRCodeScanner:
    """Detects, decodes, and generates QR codes with visual annotation support."""

    def __init__(self):
        self.detector = cv2.QRCodeDetector()
        self.encoder = cv2.QRCodeEncoder_create() if hasattr(cv2, "QRCodeEncoder_create") else None

    def generate(self, data: str, scale: int = 8, border: int = 4) -> np.ndarray:
        """Encode text or URL into a standardized high-contrast QR code image."""
        if self.encoder is None:
            raise RuntimeError("QRCodeEncoder is not supported by current OpenCV build.")

        raw_qr = self.encoder.encode(data)
        # Add quiet zone (white border)
        padded = cv2.copyMakeBorder(raw_qr, border, border, border, border, cv2.BORDER_CONSTANT, value=255)
        # Scale up using nearest-neighbor interpolation to maintain sharp modules
        h, w = padded.shape[:2]
        scaled = cv2.resize(padded, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
        return cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)

    def scan(self, image: np.ndarray) -> List[QRScanResult]:
        """Detect and decode all QR codes within an image."""
        if len(image.shape) == 2:
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        results: List[QRScanResult] = []

        # Try multi-decode first
        ret, decoded_info, points, straight_qrs = self.detector.detectAndDecodeMulti(image)
        if ret and decoded_info:
            for i, text in enumerate(decoded_info):
                if not text:
                    continue
                pts = points[i].astype(int)
                x, y, w, h = cv2.boundingRect(pts)
                straight = straight_qrs[i] if straight_qrs is not None and i < len(straight_qrs) else None
                results.append(
                    QRScanResult(
                        data=text,
                        polygon=pts,
                        bounding_box=(x, y, w, h),
                        straight_qr=straight,
                    )
                )

        # Fallback to single detector if multi didn't find decoded text
        if not results:
            text, points, straight = self.detector.detectAndDecode(image)
            if text and points is not None:
                pts = points.reshape(-1, 2).astype(int)
                x, y, w, h = cv2.boundingRect(pts)
                results.append(
                    QRScanResult(
                        data=text,
                        polygon=pts,
                        bounding_box=(x, y, w, h),
                        straight_qr=straight,
                    )
                )

        return results

    def scan_video(
        self,
        video_path: Union[str, Path],
        max_frames: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Scan a video stream for QR codes and return unique detections with timestamps."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_idx = 0
        seen_data = set()
        manifest: List[Dict[str, Any]] = []

        while True:
            if max_frames and frame_idx >= max_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break

            results = self.scan(frame)
            for res in results:
                if res.data not in seen_data:
                    seen_data.add(res.data)
                    timestamp = frame_idx / fps
                    manifest.append({
                        "data": res.data,
                        "frame": frame_idx,
                        "timestamp_sec": round(timestamp, 2),
                        "bounding_box": list(res.bounding_box),
                    })

            frame_idx += 1

        cap.release()
        return manifest

    def draw_results(
        self,
        image: np.ndarray,
        results: List[QRScanResult],
        stroke_color: Optional[Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """Annotate detected QR codes with polygon boundary and text badge."""
        output = image.copy()
        color = stroke_color or UI_COLORS["lime"]

        for res in results:
            pts = res.polygon.reshape((-1, 1, 2))
            cv2.polylines(output, [pts], isClosed=True, color=color, thickness=3)

            x, y, w, h = res.bounding_box
            # Draw corner dots
            for pt in res.polygon.reshape(-1, 2):
                cv2.circle(output, tuple(pt), 4, UI_COLORS["cyan"], -1)

            # Draw text banner
            badge_text = res.data if len(res.data) <= 32 else res.data[:29] + "..."
            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            tx = max(0, x)
            ty = max(th + 6, y - 6)
            cv2.rectangle(output, (tx, ty - th - 4), (tx + tw + 8, ty + 2), (0, 0, 0), -1)
            cv2.rectangle(output, (tx, ty - th - 4), (tx + tw + 8, ty + 2), color, 1)
            cv2.putText(
                output,
                badge_text,
                (tx + 4, ty - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return output
