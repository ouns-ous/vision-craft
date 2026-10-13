"""
Interactive Live Computer Vision Studio and Multi-Mode Viewer.
"""

from pathlib import Path
from typing import Optional, Union
import time
import cv2
import numpy as np

from vision_craft.config import UI_COLORS
from vision_craft.filters import ImageFilter
from vision_craft.shapes import ShapeDetector
from vision_craft.colors import ColorTracker
from vision_craft.motion import MotionTracker
from vision_craft.detection import FaceDetector
from vision_craft.cloak import InvisibleCloak
from vision_craft.synthetic import SyntheticGenerator


class VisionStudio:
    """Real-time processing dashboard for camera feed or video stream."""

    def __init__(
        self,
        source: Union[int, str, Path] = 0,
        window_title: str = "VisionCraft Interactive Studio",
        headless: bool = False,
    ):
        self.source = source
        self.window_title = window_title
        self.headless = headless

        # Engines
        self.shape_detector = ShapeDetector()
        self.color_tracker = ColorTracker()
        self.motion_tracker = MotionTracker()
        self.face_detector = FaceDetector()
        self.cloak = InvisibleCloak(cloak_color="green")

        # Interactive state flags
        self.active_filter = "original"
        self.mode_shapes = False
        self.mode_color = False
        self.tracked_color = "green"
        self.mode_motion = False
        self.mode_face = False
        self.mode_anonymize = False
        self.mode_cloak = False
        self.show_hud = True
        self.paused = False

    def toggle_filter(self) -> str:
        """Cycle through image filters."""
        filters = ["original", "gray", "canny", "sepia", "cartoon", "sketch", "binary"]
        idx = (filters.index(self.active_filter) + 1) % len(filters)
        self.active_filter = filters[idx]
        return self.active_filter

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply active processors and overlays to single frame."""
        processed = frame.copy()

        # 1. Base filter
        if self.active_filter == "gray":
            processed = ImageFilter.grayscale(processed)
        elif self.active_filter == "canny":
            processed = ImageFilter.canny_edges(processed)
        elif self.active_filter == "sepia":
            processed = ImageFilter.sepia(processed)
        elif self.active_filter == "cartoon":
            processed = ImageFilter.cartoon(processed)
        elif self.active_filter == "sketch":
            processed = ImageFilter.pencil_sketch(processed)
        elif self.active_filter == "binary":
            processed = ImageFilter.threshold_binary(processed)

        # 2. Shape detection overlay
        if self.mode_shapes:
            shapes = self.shape_detector.detect(frame)
            processed = self.shape_detector.draw_shapes(processed, shapes)

        # 3. Color tracking overlay
        if self.mode_color:
            _, tracked_objs = self.color_tracker.track(frame, self.tracked_color)
            processed = self.color_tracker.draw_tracked(processed, tracked_objs, draw_trail=True)

        # 4. Motion tracking overlay
        if self.mode_motion:
            res = self.motion_tracker.process_frame(frame)
            processed = self.motion_tracker.draw_motion(processed, res, draw_boxes=True, blend_heatmap=True)

        # 5. Face detection / Privacy blur
        if self.mode_face or self.mode_anonymize:
            faces = self.face_detector.detect(frame)
            if self.mode_anonymize:
                processed = self.face_detector.anonymize_faces(processed, faces, method="pixelate")
            elif self.mode_face:
                processed = self.face_detector.draw_faces(processed, faces, show_landmarks=True)

        # 6. Invisible cloak
        if self.mode_cloak:
            processed = self.cloak.process_frame(processed)

        # 7. Heads-Up Display (HUD)
        if self.show_hud:
            h, w = processed.shape[:2]
            hud_bg = (20, 20, 20)
            cv2.rectangle(processed, (0, h - 36), (w, h), hud_bg, -1)

            status_str = (
                f"Filter: {self.active_filter.upper()} | "
                f"Shapes [S]: {'ON' if self.mode_shapes else 'OFF'} | "
                f"Color [C]: {'ON' if self.mode_color else 'OFF'} | "
                f"Motion [M]: {'ON' if self.mode_motion else 'OFF'} | "
                f"Face [F]: {'ON' if self.mode_face else 'OFF'} | "
                f"Blur [B]: {'ON' if self.mode_anonymize else 'OFF'}"
            )
            cv2.putText(
                processed,
                status_str,
                (12, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (200, 240, 255),
                1,
                cv2.LINE_AA,
            )

        return processed

    def run(self, max_frames: Optional[int] = None) -> int:
        """Run studio loop. Returns number of frames processed."""
        if str(self.source).lower() == "synthetic":
            # Generate temporary synthetic motion video to stream
            temp_vid = Path("temp_synthetic_stream.mp4")
            SyntheticGenerator.generate_motion_video(temp_vid, num_frames=120)
            cap = cv2.VideoCapture(str(temp_vid))
        else:
            cap = cv2.VideoCapture(self.source)

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")

        frames_processed = 0
        last_frame: Optional[np.ndarray] = None

        try:
            while True:
                if max_frames and frames_processed >= max_frames:
                    break

                if not self.paused:
                    ret, frame = cap.read()
                    if not ret:
                        # If video ended, rewind or exit
                        if isinstance(self.source, (str, Path)):
                            break
                        continue
                    last_frame = frame
                else:
                    frame = last_frame

                if frame is None:
                    break

                output = self.process_frame(frame)
                frames_processed += 1

                if self.headless:
                    continue

                try:
                    cv2.imshow(self.window_title, output)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):  # 'q' or ESC
                        break
                    elif key == ord("f"):
                        self.toggle_filter()
                    elif key == ord("s"):
                        self.mode_shapes = not self.mode_shapes
                    elif key == ord("c"):
                        self.mode_color = not self.mode_color
                    elif key == ord("m"):
                        self.mode_motion = not self.mode_motion
                    elif key == ord("d"):
                        self.mode_face = not self.mode_face
                    elif key == ord("b"):
                        self.mode_anonymize = not self.mode_anonymize
                    elif key == ord("k"):
                        self.mode_cloak = not self.mode_cloak
                    elif key == ord("h"):
                        self.show_hud = not self.show_hud
                    elif key == ord(" "):
                        self.paused = not self.paused
                    elif key == ord("p"):
                        snapshot_path = f"snapshot_{int(time.time())}.jpg"
                        cv2.imwrite(snapshot_path, output)
                except cv2.error:
                    # In case of headless environment without display
                    self.headless = True
        finally:
            cap.release()
            if not self.headless:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass

        return frames_processed
