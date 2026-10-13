"""
Face detection, facial landmark extraction, and privacy anonymization module.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Union
import cv2
import numpy as np

from vision_craft.config import (
    DEFAULT_YUNET_MODEL,
    DEFAULT_YUNET_SCORE_THRESHOLD,
    DEFAULT_YUNET_NMS_THRESHOLD,
    UI_COLORS,
)


@dataclass
class FaceDetection:
    """Represents a localized face with confidence and optional landmarks."""
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    landmarks: Optional[List[Tuple[int, int]]] = None  # 5 facial points


class FaceDetector:
    """Detects faces via OpenCV YuNet deep learning model or skin-tone morphological fallback."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        score_threshold: float = DEFAULT_YUNET_SCORE_THRESHOLD,
        nms_threshold: float = DEFAULT_YUNET_NMS_THRESHOLD,
        use_fallback: bool = True,
    ):
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.use_fallback = use_fallback
        self.yunet_detector = None
        self._current_input_size: Optional[Tuple[int, int]] = None

        # Resolve model path
        path = Path(model_path) if model_path else DEFAULT_YUNET_MODEL
        if path.exists() and hasattr(cv2, "FaceDetectorYN_create"):
            try:
                self.yunet_detector = cv2.FaceDetectorYN_create(
                    str(path),
                    "",
                    (320, 240),
                    score_threshold=self.score_threshold,
                    nms_threshold=self.nms_threshold,
                )
            except Exception:
                self.yunet_detector = None

    def _detect_yunet(self, frame: np.ndarray) -> List[FaceDetection]:
        """Run deep learning YuNet face detector."""
        h, w = frame.shape[:2]
        if self._current_input_size != (w, h):
            self.yunet_detector.setInputSize((w, h))
            self._current_input_size = (w, h)

        _, raw_faces = self.yunet_detector.detect(frame)
        if raw_faces is None:
            return []

        results: List[FaceDetection] = []
        for face in raw_faces:
            x, y, fw, fh = map(int, face[0:4])
            # Ensure coordinates stay within frame bounds
            x = max(0, x)
            y = max(0, y)
            fw = min(w - x, fw)
            fh = min(h - y, fh)
            score = float(face[14])

            # Extract 5 landmarks: right eye, left eye, nose, right mouth, left mouth
            landmarks: List[Tuple[int, int]] = []
            for j in range(5):
                lx = int(face[4 + j * 2])
                ly = int(face[5 + j * 2])
                landmarks.append((lx, ly))

            results.append(
                FaceDetection(
                    bbox=(x, y, fw, fh),
                    confidence=round(score, 3),
                    landmarks=landmarks,
                )
            )
        return results

    def _detect_skin_tone_fallback(self, frame: np.ndarray) -> List[FaceDetection]:
        """Morphological skin-tone segmentation fallback."""
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Standard human skin tone HSV range
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([25, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=4)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results: List[FaceDetection] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < (w * h * 0.015):  # Ignore small patches
                continue

            x, y, fw, fh = cv2.boundingRect(cnt)
            aspect_ratio = float(fh) / fw if fw > 0 else 0
            # Human faces typically have height/width ratio between 1.0 and 1.8
            if 0.9 <= aspect_ratio <= 2.2:
                results.append(
                    FaceDetection(
                        bbox=(x, y, fw, fh),
                        confidence=0.75,
                        landmarks=None,
                    )
                )

        return results

    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        """Detect faces in the given frame."""
        if self.yunet_detector is not None:
            try:
                faces = self._detect_yunet(frame)
                if len(faces) > 0 or not self.use_fallback:
                    return faces
            except Exception:
                pass

        if self.use_fallback:
            return self._detect_skin_tone_fallback(frame)
        return []

    def draw_faces(
        self,
        frame: np.ndarray,
        faces: List[FaceDetection],
        show_landmarks: bool = True,
    ) -> np.ndarray:
        """Render face bounding boxes, confidence badges, and landmarks."""
        output = frame.copy()

        for face in faces:
            x, y, w, h = face.bbox
            cv2.rectangle(output, (x, y), (x + w, y + h), UI_COLORS["lime"], 2)

            # Draw confidence label
            label = f"Face: {int(face.confidence * 100)}%"
            cv2.putText(
                output,
                label,
                (x, max(15, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                UI_COLORS["lime"],
                1,
                cv2.LINE_AA,
            )

            # Draw facial landmarks if available
            if show_landmarks and face.landmarks:
                for lx, ly in face.landmarks:
                    cv2.circle(output, (lx, ly), 3, UI_COLORS["yellow"], -1)

        return output

    @staticmethod
    def anonymize_faces(
        frame: np.ndarray,
        faces: List[FaceDetection],
        method: str = "pixelate",
        factor: float = 0.08,
    ) -> np.ndarray:
        """Obfuscate face regions for privacy protection using pixelation, blur, or blackout."""
        output = frame.copy()
        img_h, img_w = frame.shape[:2]

        for face in faces:
            x, y, w, h = face.bbox
            if w <= 0 or h <= 0:
                continue

            # Clip bounds
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(img_w, x + w)
            y2 = min(img_h, y + h)

            roi = output[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            if method == "pixelate":
                # Downscale and upscale with nearest neighbor interpolation
                small_w = max(1, int((x2 - x1) * factor))
                small_h = max(1, int((y2 - y1) * factor))
                temp = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                pixelated = cv2.resize(temp, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
                output[y1:y2, x1:x2] = pixelated
            elif method == "gaussian_blur":
                # Heavy Gaussian blur
                k = int((x2 - x1) / 3)
                k = k if k % 2 != 0 else k + 1
                k = max(15, k)
                output[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
            elif method == "blackout":
                output[y1:y2, x1:x2] = 0
            else:
                raise ValueError(f"Unknown anonymize method '{method}'. Choose 'pixelate', 'gaussian_blur', or 'blackout'.")

        return output
