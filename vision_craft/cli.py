"""
Command-line interface for VisionCraft.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List
import cv2
import numpy as np

from vision_craft import __version__
from vision_craft.filters import apply_filter, get_available_filters
from vision_craft.shapes import ShapeDetector
from vision_craft.colors import ColorTracker
from vision_craft.motion import MotionTracker
from vision_craft.detection import FaceDetector
from vision_craft.cloak import InvisibleCloak
from vision_craft.video_tools import VideoFrameExtractor, VideoSplitter, get_video_metadata
from vision_craft.synthetic import SyntheticGenerator
from vision_craft.studio import VisionStudio


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="visioncraft",
        description="VisionCraft: Advanced OpenCV Computer Vision Studio & Processing Engine",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 1. Filter
    p_filter = subparsers.add_parser("filter", help="Apply visual filters to an image or video")
    p_filter.add_argument("-i", "--input", required=True, help="Input image or video path")
    p_filter.add_argument("-t", "--type", required=True, choices=get_available_filters(), help="Filter algorithm")
    p_filter.add_argument("-o", "--output", required=True, help="Output destination path")

    # 2. Shapes
    p_shapes = subparsers.add_parser("shapes", help="Detect and classify geometric shapes")
    p_shapes.add_argument("-i", "--input", required=True, help="Input image path")
    p_shapes.add_argument("-o", "--output", required=True, help="Annotated output image path")
    p_shapes.add_argument("--min-area", type=float, default=300.0, help="Minimum contour area")

    # 3. Color
    p_color = subparsers.add_parser("color", help="Track color targets and trajectories")
    p_color.add_argument("-i", "--input", required=True, help="Input image or video path")
    p_color.add_argument("-c", "--color", required=True, help="Color preset name (e.g. red, green, blue, yellow)")
    p_color.add_argument("-o", "--output", required=True, help="Output destination path")

    # 4. Motion
    p_motion = subparsers.add_parser("motion", help="Detect visual motion and generate heatmaps")
    p_motion.add_argument("-i", "--input", required=True, help="Input video path")
    p_motion.add_argument("-o", "--output", required=True, help="Annotated video output path")
    p_motion.add_argument("-m", "--method", default="mog2", choices=["mog2", "knn", "frame_diff"], help="Algorithm")
    p_motion.add_argument("--heatmap", help="Optional path to export accumulated heatmap image")

    # 5. Face
    p_face = subparsers.add_parser("face", help="Detect faces and optionally blur for privacy")
    p_face.add_argument("-i", "--input", required=True, help="Input image or video path")
    p_face.add_argument("-o", "--output", required=True, help="Output destination path")
    p_face.add_argument("--blur", action="store_true", help="Apply privacy obfuscation")
    p_face.add_argument("--method", default="pixelate", choices=["pixelate", "gaussian_blur", "blackout"])

    # 6. Cloak
    p_cloak = subparsers.add_parser("cloak", help="Run invisible cloak or chroma key effect")
    p_cloak.add_argument("-i", "--input", required=True, help="Input video path")
    p_cloak.add_argument("-o", "--output", required=True, help="Output video path")
    p_cloak.add_argument("-c", "--color", default="green", help="Cloak color preset (default: green)")
    p_cloak.add_argument("--bg", help="Optional static background image path")

    # 7. Extract
    p_extract = subparsers.add_parser("extract", help="Extract video frames or keyframes")
    p_extract.add_argument("-i", "--input", required=True, help="Input video path")
    p_extract.add_argument("-o", "--output-dir", required=True, help="Output directory")
    p_extract.add_argument("--interval", type=float, help="Interval in seconds")
    p_extract.add_argument("--step", type=int, help="Frame step count")
    p_extract.add_argument("--keyframes", action="store_true", help="Scene-change keyframes only")

    # 8. Split
    p_split = subparsers.add_parser("split", help="Trim or split a video clip")
    p_split.add_argument("-i", "--input", required=True, help="Input video path")
    p_split.add_argument("-o", "--output", required=True, help="Output video path")
    p_split.add_argument("--start", required=True, help="Start time (seconds or HH:MM:SS)")
    p_split.add_argument("--end", required=True, help="End time (seconds or HH:MM:SS)")

    # 9. Generate Demo
    p_demo = subparsers.add_parser("generate-demo", help="Generate synthetic test assets")
    p_demo.add_argument("-o", "--output-dir", default="demo_assets", help="Directory for demo assets")

    # 10. Studio
    p_studio = subparsers.add_parser("studio", help="Launch interactive live CV studio")
    p_studio.add_argument("-s", "--source", default="0", help="Camera index, video file, or 'synthetic'")
    p_studio.add_argument("--headless", action="store_true", help="Run without UI window")
    p_studio.add_argument("--max-frames", type=int, help="Limit frames to process")

    return parser


def main(args: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint."""
    parser = build_parser()
    parsed = parser.parse_args(args)

    if not parsed.command:
        parser.print_help()
        return 0

    try:
        if parsed.command == "filter":
            img = cv2.imread(parsed.input)
            if img is not None:
                res = apply_filter(img, parsed.type)
                Path(parsed.output).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(parsed.output, res)
                print(f"[OK] Filter '{parsed.type}' applied: {parsed.output}")
            else:
                # Video processing
                cap = cv2.VideoCapture(parsed.input)
                if not cap.isOpened():
                    print(f"[Error] Failed to open: {parsed.input}", file=sys.stderr)
                    return 1
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out_path = Path(parsed.output)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    filtered = apply_filter(frame, parsed.type)
                    writer.write(filtered)
                cap.release()
                writer.release()
                print(f"[OK] Video filter '{parsed.type}' saved: {parsed.output}")

        elif parsed.command == "shapes":
            img = cv2.imread(parsed.input)
            if img is None:
                print(f"[Error] Failed to read image: {parsed.input}", file=sys.stderr)
                return 1
            detector = ShapeDetector()
            shapes = detector.detect(img, min_area=parsed.min_area)
            annotated = detector.draw_shapes(img, shapes)
            Path(parsed.output).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(parsed.output, annotated)
            print(f"[OK] Detected {len(shapes)} shapes: {', '.join(s.name for s in shapes)}")
            print(f"[OK] Output saved: {parsed.output}")

        elif parsed.command == "color":
            img = cv2.imread(parsed.input)
            tracker = ColorTracker()
            if img is not None:
                _, objs = tracker.track(img, parsed.color)
                annotated = tracker.draw_tracked(img, objs)
                Path(parsed.output).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(parsed.output, annotated)
                print(f"[OK] Tracked {len(objs)} objects of color '{parsed.color}' saved to: {parsed.output}")
            else:
                # Video
                cap = cv2.VideoCapture(parsed.input)
                if not cap.isOpened():
                    print(f"[Error] Failed to open: {parsed.input}", file=sys.stderr)
                    return 1
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out_path = Path(parsed.output)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    _, objs = tracker.track(frame, parsed.color)
                    writer.write(tracker.draw_tracked(frame, objs))
                cap.release()
                writer.release()
                print(f"[OK] Video color tracking saved to: {parsed.output}")

        elif parsed.command == "motion":
            cap = cv2.VideoCapture(parsed.input)
            if not cap.isOpened():
                print(f"[Error] Failed to open video: {parsed.input}", file=sys.stderr)
                return 1
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_path = Path(parsed.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            tracker = MotionTracker(method=parsed.method)

            motion_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                res = tracker.process_frame(frame)
                if res.has_motion:
                    motion_count += 1
                writer.write(tracker.draw_motion(frame, res, draw_boxes=True))

            cap.release()
            writer.release()
            print(f"[OK] Motion tracking finished. Frames with motion: {motion_count}")
            print(f"[OK] Video saved: {parsed.output}")

            if parsed.heatmap:
                heatmap = tracker.get_heatmap()
                Path(parsed.heatmap).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(parsed.heatmap, heatmap)
                print(f"[OK] Heatmap saved: {parsed.heatmap}")

        elif parsed.command == "face":
            detector = FaceDetector()
            img = cv2.imread(parsed.input)
            if img is not None:
                faces = detector.detect(img)
                if parsed.blur:
                    res = detector.anonymize_faces(img, faces, method=parsed.method)
                else:
                    res = detector.draw_faces(img, faces)
                Path(parsed.output).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(parsed.output, res)
                print(f"[OK] Found {len(faces)} faces. Saved to: {parsed.output}")
            else:
                cap = cv2.VideoCapture(parsed.input)
                if not cap.isOpened():
                    print(f"[Error] Failed to open: {parsed.input}", file=sys.stderr)
                    return 1
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out_path = Path(parsed.output)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    faces = detector.detect(frame)
                    if parsed.blur:
                        res = detector.anonymize_faces(frame, faces, method=parsed.method)
                    else:
                        res = detector.draw_faces(frame, faces)
                    writer.write(res)
                cap.release()
                writer.release()
                print(f"[OK] Face video processing saved to: {parsed.output}")

        elif parsed.command == "cloak":
            cloak = InvisibleCloak(cloak_color=parsed.color)
            if parsed.bg:
                bg = cv2.imread(parsed.bg)
                if bg is not None:
                    cloak.set_background(bg)

            cap = cv2.VideoCapture(parsed.input)
            if not cap.isOpened():
                print(f"[Error] Failed to open: {parsed.input}", file=sys.stderr)
                return 1
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_path = Path(parsed.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

            # Calibrate background on initial frames if not provided
            if cloak.background is None:
                calibrated = False
                for _ in range(30):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if cloak.calibrate_background(frame, 30):
                        calibrated = True
                        break
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                composite = cloak.process_frame(frame)
                writer.write(composite)

            cap.release()
            writer.release()
            print(f"[OK] Invisible cloak composite saved: {parsed.output}")

        elif parsed.command == "extract":
            extractor = VideoFrameExtractor(parsed.output_dir)
            if parsed.keyframes:
                paths = extractor.extract_keyframes(parsed.input)
            elif parsed.step:
                paths = extractor.extract_by_step(parsed.input, step_frames=parsed.step)
            else:
                interval = parsed.interval if parsed.interval else 1.0
                paths = extractor.extract_by_interval(parsed.input, interval_seconds=interval)
            print(f"[OK] Extracted {len(paths)} frames to: {parsed.output_dir}")

        elif parsed.command == "split":
            out = VideoSplitter.split(parsed.input, parsed.output, parsed.start, parsed.end)
            print(f"[OK] Trimmed video saved to: {out}")

        elif parsed.command == "generate-demo":
            out_dir = Path(parsed.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / "shapes_sample.png"), SyntheticGenerator.generate_shapes_image())
            cv2.imwrite(str(out_dir / "colors_sample.png"), SyntheticGenerator.generate_color_palette_image())
            cv2.imwrite(str(out_dir / "face_sample.png"), SyntheticGenerator.generate_face_like_image())
            SyntheticGenerator.generate_motion_video(out_dir / "motion_sample.mp4", num_frames=90)
            SyntheticGenerator.generate_tracking_video(out_dir / "tracking_sample.mp4", num_frames=90)
            print(f"[OK] Synthetic demo assets generated in: {out_dir}")

        elif parsed.command == "studio":
            src = int(parsed.source) if parsed.source.isdigit() else parsed.source
            studio = VisionStudio(source=src, headless=parsed.headless)
            processed = studio.run(max_frames=parsed.max_frames)
            print(f"[OK] Vision Studio session finished. Processed {processed} frames.")

    except Exception as e:
        print(f"[Error] Execution failed: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
