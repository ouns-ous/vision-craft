"""
Synthetic visual assets generator for automated testing and standalone demonstrations.
"""

from pathlib import Path
from typing import Union, Tuple
import math
import cv2
import numpy as np


class SyntheticGenerator:
    """Generates synthetic test images and video files for headless environments."""

    @staticmethod
    def generate_shapes_image(width: int = 640, height: int = 480) -> np.ndarray:
        """Create a synthetic test image containing clean, distinct geometric shapes."""
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # 1. Circle (Top-Left)
        cv2.circle(img, (120, 120), 50, (255, 255, 255), -1)

        # 2. Square (Top-Right)
        cv2.rectangle(img, (480, 70), (580, 170), (255, 255, 255), -1)

        # 3. Rectangle (Bottom-Left)
        cv2.rectangle(img, (80, 320), (220, 390), (255, 255, 255), -1)

        # 4. Triangle (Bottom-Right)
        tri_pts = np.array([[520, 300], [450, 420], [590, 420]], np.int32)
        cv2.fillPoly(img, [tri_pts], (255, 255, 255))

        # 5. Pentagon (Center)
        center_x, center_y = 320, 240
        radius = 55
        pent_pts = []
        for i in range(5):
            angle = math.radians(i * 72 - 90)
            px = int(center_x + radius * math.cos(angle))
            py = int(center_y + radius * math.sin(angle))
            pent_pts.append([px, py])
        cv2.fillPoly(img, [np.array(pent_pts, np.int32)], (255, 255, 255))

        return img

    @staticmethod
    def generate_color_palette_image(width: int = 640, height: int = 480) -> np.ndarray:
        """Create a synthetic test image with vibrant colored patches."""
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Red patch (BGR: 0, 0, 255)
        cv2.rectangle(img, (50, 50), (180, 180), (0, 0, 255), -1)

        # Green patch (BGR: 0, 255, 0)
        cv2.rectangle(img, (250, 50), (380, 180), (0, 255, 0), -1)

        # Blue patch (BGR: 255, 0, 0)
        cv2.rectangle(img, (450, 50), (580, 180), (255, 0, 0), -1)

        # Yellow patch (BGR: 0, 255, 255)
        cv2.rectangle(img, (150, 280), (280, 410), (0, 255, 255), -1)

        # Orange patch (BGR: 0, 165, 255)
        cv2.rectangle(img, (350, 280), (480, 410), (0, 165, 255), -1)

        return img

    @staticmethod
    def generate_face_like_image(width: int = 320, height: int = 240) -> np.ndarray:
        """Create an image with skin-tone ellipse and facial features for fallback detector tests."""
        img = np.full((height, width, 3), 40, dtype=np.uint8)

        center = (width // 2, height // 2)
        axes = (55, 75)  # Aspect ratio ~ 1.36 (typical face)
        # Skin tone in BGR (roughly HSV H~15, S~140, V~220)
        skin_bgr = (140, 175, 235)

        cv2.ellipse(img, center, axes, 0, 0, 360, skin_bgr, -1)

        # Eyes
        cv2.circle(img, (center[0] - 22, center[1] - 20), 7, (40, 30, 20), -1)
        cv2.circle(img, (center[0] + 22, center[1] - 20), 7, (40, 30, 20), -1)

        # Nose line
        cv2.line(img, (center[0], center[1] - 10), (center[0], center[1] + 10), (110, 140, 200), 2)

        # Mouth arc
        cv2.ellipse(img, (center[0], center[1] + 32), (20, 8), 0, 0, 180, (60, 60, 160), 3)

        return img

    @staticmethod
    def generate_motion_video(
        output_path: Union[str, Path],
        num_frames: int = 60,
        width: int = 320,
        height: int = 240,
        fps: int = 30,
    ) -> Path:
        """Create a synthetic video featuring a moving object across a static background."""
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

        # Static background
        bg = np.full((height, width, 3), 50, dtype=np.uint8)
        # Draw some background texture
        cv2.rectangle(bg, (20, 20), (100, 100), (80, 80, 80), -1)
        cv2.rectangle(bg, (220, 140), (300, 220), (70, 70, 70), -1)

        # Moving ball
        for i in range(num_frames):
            frame = bg.copy()
            # Ball moves from left to right diagonally
            bx = int(30 + (width - 60) * (i / float(num_frames)))
            by = int(120 + 40 * math.sin(i * 0.2))
            cv2.circle(frame, (bx, by), 18, (220, 220, 220), -1)
            writer.write(frame)

        writer.release()
        return out_path

    @staticmethod
    def generate_tracking_video(
        output_path: Union[str, Path],
        num_frames: int = 60,
        width: int = 320,
        height: int = 240,
        fps: int = 30,
        color_bgr: Tuple[int, int, int] = (0, 255, 0),  # Default bright green
    ) -> Path:
        """Create a video with a moving colored target for tracking verification."""
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

        for i in range(num_frames):
            frame = np.full((height, width, 3), 20, dtype=np.uint8)
            # Circular orbit trajectory
            cx = int(width / 2 + 70 * math.cos(i * 0.15))
            cy = int(height / 2 + 50 * math.sin(i * 0.15))
            cv2.circle(frame, (cx, cy), 20, color_bgr, -1)
            writer.write(frame)

        writer.release()
        return out_path

    @staticmethod
    def generate_document_sample(width: int = 640, height: int = 480) -> np.ndarray:
        """Create a synthetic tilted document on a textured background with text and colored stamps."""
        # Textured wooden/dark table background
        bg = np.zeros((height, width, 3), dtype=np.uint8)
        bg[:, :] = (35, 45, 60)
        # Add background grain/stripes
        for y in range(0, height, 15):
            cv2.line(bg, (0, y), (width, y), (40, 50, 70), 2)

        # Angled document polygon
        pts_src = np.array([
            [120, 60],
            [540, 110],
            [480, 420],
            [80, 360]
        ], np.int32)

        # Fill white page
        cv2.fillPoly(bg, [pts_src], (245, 248, 250))
        # Draw document border
        cv2.polylines(bg, [pts_src], isClosed=True, color=(180, 190, 200), thickness=2)

        # Draw a red official stamp inside the document
        cv2.circle(bg, (220, 220), 28, (30, 30, 200), 3)
        cv2.circle(bg, (220, 220), 20, (30, 30, 200), 2)

        # Draw simulated text lines
        lines = [
            ((150, 130), (380, 160)),
            ((145, 160), (450, 200)),
            ((140, 270), (400, 305)),
            ((135, 300), (350, 330)),
        ]
        for start, end in lines:
            cv2.line(bg, start, end, (50, 50, 60), 3)

        return bg

    @staticmethod
    def generate_qrcode_sample(data: str = "https://github.com/ouns-ous/vision-craft") -> np.ndarray:
        """Generate a clean synthetic QR code image with quiet zone."""
        from vision_craft.qrcode_scanner import QRCodeScanner
        scanner = QRCodeScanner()
        return scanner.generate(data, scale=6, border=4)
