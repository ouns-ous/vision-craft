"""
Computer Vision performance benchmarking and latency profiler module.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
import time
import json
import cv2
import numpy as np

from vision_craft.filters import apply_filter
from vision_craft.shapes import ShapeDetector
from vision_craft.colors import ColorTracker
from vision_craft.motion import MotionTracker
from vision_craft.document import DocumentScanner


@dataclass
class BenchmarkScore:
    """Encapsulates latency, frame rate, and throughput metrics for an operation."""
    operation: str
    resolution: str
    iterations: int
    mean_ms: float
    fps: float
    megapixels_per_sec: float


class VisionBenchmark:
    """Profiles computer vision processing pipelines across resolution tiers."""

    RESOLUTIONS = {
        "480p": (640, 480),
        "720p": (1280, 720),
        "1080p": (1920, 1080),
    }

    @classmethod
    def run_benchmark(
        cls,
        resolution: str = "480p",
        iterations: int = 15,
        target_ops: Optional[List[str]] = None,
    ) -> List[BenchmarkScore]:
        """Benchmark core vision operations on synthetic frames of specified resolution."""
        w, h = cls.RESOLUTIONS.get(resolution.lower(), (640, 480))
        megapixels = (w * h) / 1_000_000.0

        # Create textured synthetic test frame
        test_frame = np.full((h, w, 3), 40, dtype=np.uint8)
        cv2.circle(test_frame, (w // 2, h // 2), min(w, h) // 4, (0, 255, 0), -1)
        cv2.rectangle(test_frame, (w // 4, h // 4), (w // 4 * 3, h // 4 * 3), (255, 0, 255), 3)

        shape_detector = ShapeDetector()
        color_tracker = ColorTracker()
        motion_tracker = MotionTracker(method="frame_diff")
        doc_scanner = DocumentScanner()

        pipeline_tasks = {
            "Filter: Grayscale": lambda: apply_filter(test_frame, "gray"),
            "Filter: Canny Edges": lambda: apply_filter(test_frame, "canny"),
            "Filter: Gaussian Blur": lambda: apply_filter(test_frame, "blur"),
            "Filter: Cartoon": lambda: apply_filter(test_frame, "cartoon"),
            "Shape Detection": lambda: shape_detector.detect(test_frame),
            "Color Tracking (HSV)": lambda: color_tracker.track(test_frame, "green"),
            "Motion Diff Tracking": lambda: motion_tracker.process_frame(test_frame),
            "Document Rectification": lambda: doc_scanner.scan(test_frame),
        }

        scores: List[BenchmarkScore] = []

        for name, task in pipeline_tasks.items():
            if target_ops and not any(t.lower() in name.lower() for t in target_ops):
                continue

            # Warmup
            task()

            durations = []
            for _ in range(iterations):
                start = time.perf_counter()
                task()
                durations.append((time.perf_counter() - start) * 1000.0)

            mean_ms = float(np.mean(durations))
            fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
            mp_per_sec = megapixels * fps

            scores.append(
                BenchmarkScore(
                    operation=name,
                    resolution=f"{w}x{h} ({resolution})",
                    iterations=iterations,
                    mean_ms=round(mean_ms, 2),
                    fps=round(fps, 1),
                    megapixels_per_sec=round(mp_per_sec, 2),
                )
            )

        return scores

    @staticmethod
    def format_markdown_table(scores: List[BenchmarkScore]) -> str:
        """Format benchmark scores into a standard markdown table."""
        lines = [
            "| Operation | Resolution | Mean Latency (ms) | Throughput (FPS) | Megapixels / Sec |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]
        for s in scores:
            lines.append(
                f"| {s.operation} | {s.resolution} | {s.mean_ms} ms | {s.fps} fps | {s.megapixels_per_sec} MP/s |"
            )
        return "\n".join(lines)
