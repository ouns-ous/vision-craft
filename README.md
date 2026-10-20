# 👁️ VisionCraft

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/opencv-5.0%20%7C%204.8+-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-50%20passed-brightgreen.svg)]()
[![Version](https://img.shields.io/badge/version-1.2.0-blue.svg)]()

**VisionCraft** is a modular, high-performance Computer Vision Studio and processing engine built on top of OpenCV and NumPy. It unifies advanced digital image filtering, geometric contour classification, multi-color centroid tracking, background subtraction with accumulated motion heatmaps, deep-learning face detection with privacy obfuscation, chroma-key/invisible cloaking, smart document scanning, QR code encoding/decoding, panorama stitching, feature matching & homography localization, dense optical flow vector fields, mathematical morphology, and automated performance profiling.

---

## 🌟 Key Capabilities

### 1. 🎨 Image Filters & Artistic Processing
- **Color Spaces**: Grayscale, Sepia, Invert, HSV, CIELAB.
- **Edge Detection**: Canny edge detector, Sobel gradient ($X$, $Y$, magnitude), Laplacian 2nd derivative.
- **Adaptive Thresholding**: Standard Binary, Otsu's bimodal threshold, Adaptive Gaussian, and Adaptive Mean windowing.
- **Artistic Stylization**: Fine pencil sketch, Cartoon cel-shading (bilateral filter + adaptive edge mask), Emboss texture, Vignette radial shading, and K-Means color quantization.

### 2. 📐 Geometric Shape & Contour Classifier
- Automatic extraction and contour hierarchy analysis.
- Polygon approximation via the Ramer–Douglas–Peucker algorithm (`cv2.approxPolyDP`).
- Real-time classification of **Circles**, **Squares**, **Rectangles**, **Triangles**, **Pentagons**, **Hexagons**, **Stars**, and arbitrary Polygons.
- Computes spatial moments, centroid $(c_x, c_y)$, circularity ($4\pi A / P^2$), aspect ratio, and solidity ($A / A_{hull}$).

### 3. 🎯 Multi-Color Tracking & Trajectory Trails
- Calibrated HSV presets for `red`, `green`, `blue`, `yellow`, `orange`, `cyan`, `purple`, `white`, and `black` (with proper $0^\circ \leftrightarrow 180^\circ$ hue wrapping for reds).
- Dynamic ROI color sampling: click or sample any image coordinate to automatically generate optimal HSV bounds.
- Trajectory breadcrumb tracking: records object velocity and trajectory with smooth fading trails.

### 4. 🚨 Motion Detection & Accumulated Heatmaps
- Triple background subtraction algorithms:
  - **MOG2**: Gaussian Mixture-based background/foreground segmentation with shadow elimination.
  - **KNN**: K-Nearest Neighbors background modeling.
  - **Frame Differencing**: Ultra-fast consecutive frame difference for lightweight or embedded systems.
- Intrusion alert triggers with callback hooks.
- **Cumulative Thermal Heatmap**: records long-term pixel motion activity and exports colormapped heatmaps (`cv2.COLORMAP_JET`).

### 5. 👤 Deep Learning Face Detection & Privacy Blurring
- Powered by OpenCV's modern **YuNet** deep learning ONNX detector (`cv2.FaceDetectorYN`) with 5-point facial landmark localization.
- Morphological skin-tone segmentation fallback for environments without DNN acceleration.
- **Privacy Anonymization**:
  - `pixelate`: Mosaic block downscale/upscale.
  - `gaussian_blur`: Heavy localized Gaussian blur.
  - `blackout`: Solid privacy redaction.

### 6. 🧙‍♂️ Invisible Cloak & Chroma Key Compositor
- Real-time "Harry Potter" invisibility cloak effect via color segmentation and background replacement.
- Automatic static background calibration averaging initial frames.
- Virtual background replacement support for video conferencing and streaming.

### 7. 📄 Smart Document Scanner & Perspective Warper
- Automatic document boundary detection using edge filtering, contour hierarchy, and quadrangle approximation.
- 4-point perspective transform (`cv2.warpPerspective`) generating top-down flattened digital scans.
- Scan enhancement modes:
  - `bw`: Adaptive Gaussian binarization for clean printed document text.
  - `color_enhanced`: Background illumination division for shadow and glare elimination.
  - `gray`: CLAHE-enhanced smooth grayscale.
  - `original`: Perspective-rectified raw color scan.

### 8. 📱 QR Code Engine (Scanner, Decoder & Generator)
- Real-time QR code localization, polygon boundary drawing, and data payload extraction (`cv2.QRCodeDetector`).
- High-contrast module generator with customizable scale and quiet zone padding (`cv2.QRCodeEncoder`).
- Video stream QR scanning: tracks unique QR detections across frames with timestamps and JSON report export.

### 9. 🎬 Video Frame Extractor & Video Splitter
- **Interval Extraction**: Extract frames at fixed second intervals (e.g., every 1.0s) with JSON manifest metadata.
- **Step Extraction**: Extract every $N$-th frame.
- **Keyframe Extraction**: Scene change detection via inter-frame difference metric thresholding.
- **Video Trimmer / Splitter**: Frame-accurate video clipping using `HH:MM:SS` or decimal seconds.

### 10. 🎮 Interactive Live CV Studio
- Real-time webcam or video stream dashboard with HUD diagnostics (FPS, active algorithms, region counters).
- Keyboard toggles:
  - `F`: Cycle filters (`gray`, `canny`, `sepia`, `cartoon`, `sketch`, `binary`).
  - `S`: Toggle shape detection overlay.
  - `C`: Toggle color tracking.
  - `M`: Toggle motion detection.
  - `D`: Toggle face detection.
  - `B`: Toggle privacy anonymization.
  - `K`: Toggle invisible cloak.
  - `R`: Toggle QR code detection & payload decoding.
  - `P`: Capture instant high-res snapshot.
  - `Space`: Pause / Resume feed.
  - `Q` / `ESC`: Exit studio.

### 11. 🧪 Synthetic Test Asset Generator
- Headless-ready synthetic generator creating deterministic test cards, moving targets, and bouncing ball videos for 100% CI automated testing without physical cameras.

---

## 📁 Project Architecture

```
vision-craft/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── main.py                          # CLI wrapper entry point
├── vision_craft/
│   ├── __init__.py                  # Public API exports
│   ├── cli.py                       # Full-featured argument parser
│   ├── config.py                    # Constants, HSV ranges, palettes
│   ├── filters.py                   # Filters, edges, thresholding, art
│   ├── shapes.py                    # Shape detection & contour math
│   ├── colors.py                    # Multi-color HSV tracking & trails
│   ├── motion.py                    # MOG2/KNN/Diff motion & heatmaps
│   ├── detection.py                 # YuNet face detector & anonymization
│   ├── cloak.py                     # Invisible cloak & chroma keying
│   ├── document.py                  # Smart document scanner & perspective warper
│   ├── qrcode_scanner.py            # QR code generator, detector & decoder
│   ├── video_tools.py               # Frame extractor, keyframes, splitter
│   ├── synthetic.py                 # Synthetic images & video generator
│   ├── studio.py                    # Interactive live viewer & HUD
│   └── data/
│       └── face_detection_yunet.onnx # Bundled YuNet ONNX neural network
└── tests/
    ├── __init__.py
    ├── test_cli.py
    ├── test_cloak.py
    ├── test_colors.py
    ├── test_detection.py
    ├── test_document.py
    ├── test_filters.py
    ├── test_motion.py
    ├── test_qrcode.py
    ├── test_shapes.py
    ├── test_synthetic.py
    └── test_video_tools.py
```

---

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ouns-ous/vision-craft.git
   cd vision-craft
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Command-Line Usage

### Generate Synthetic Demo Assets
```bash
python main.py generate-demo --output-dir demo_assets
```

### Shape & Contour Detection
```bash
python main.py shapes --input demo_assets/shapes_sample.png --output shapes_detected.png
```

### Image Filtering & Stylization
```bash
# Cartoon effect
python main.py filter --input demo_assets/shapes_sample.png --type cartoon --output cartoon.png

# Canny edge detection
python main.py filter --input demo_assets/shapes_sample.png --type canny --output canny.png

# Vintage Sepia
python main.py filter --input photo.jpg --type sepia --output sepia.jpg
```

### Motion Tracking & Thermal Heatmap
```bash
python main.py motion --input demo_assets/motion_sample.mp4 --output motion_tracked.mp4 --method mog2 --heatmap heatmap.png
```

### Face Detection & Privacy Obfuscation
```bash
# Detect and draw bounding boxes + facial landmarks
python main.py face --input demo_assets/face_sample.png --output face_detected.png

# Anonymize face with pixelation (mosaic)
python main.py face --input demo_assets/face_sample.png --blur --method pixelate --output face_anonymized.png
```

### Color Tracking with Trajectory Trail
```bash
python main.py color --input demo_assets/colors_sample.png --color red --output color_tracked.png
```

### Smart Document Scanner & Perspective Rectification
```bash
# Scan with automatic shadow removal
python main.py scan --input receipt.jpg --output receipt_scanned.png --mode color_enhanced

# Scan with high-contrast black & white binarization
python main.py scan --input document.jpg --output document_bw.png --mode bw --bounds-output detected_bounds.png
```

### QR Code Generator & Scanner
```bash
# Generate high-resolution QR code
python main.py qrcode --generate "https://github.com/ouns-ous/vision-craft" --output my_qr.png

# Scan QR codes from image and export metadata
python main.py qrcode --input photo.jpg --output qr_annotated.png --json qr_results.json

# Scan QR codes throughout a video stream
python main.py qrcode --input video.mp4 --json video_qrs.json
```

### Video Frame Extraction
```bash
# Extract 1 frame every 0.5 seconds
python main.py extract --input video.mp4 --output-dir frames/ --interval 0.5

# Extract only scene-change keyframes
python main.py extract --input video.mp4 --output-dir keyframes/ --keyframes
```

### Video Trimmer / Splitter
```bash
python main.py split --input video.mp4 --output clip.mp4 --start 00:00:05 --end 00:00:15
```

### Launch Interactive Studio
```bash
# Live webcam
python main.py studio --source 0

# Test video or synthetic stream
python main.py studio --source synthetic
```

---

## 🐍 Python Library Example

```python
import cv2
from vision_craft import (
    ShapeDetector,
    ColorTracker,
    MotionTracker,
    FaceDetector,
    DocumentScanner,
    QRCodeScanner,
    apply_filter,
)

# 1. Apply Cartoon Filter
img = cv2.imread("input.jpg")
cartoon = apply_filter(img, "cartoon")

# 2. Detect Geometric Shapes
detector = ShapeDetector()
shapes = detector.detect(img)
for s in shapes:
    print(f"Detected: {s.name} (Vertices: {s.vertices_count}, Area: {s.area})")

# 3. Detect and Anonymize Faces
face_engine = FaceDetector()
faces = face_engine.detect(img)
anonymized = face_engine.anonymize_faces(img, faces, method="pixelate")

# 4. Color Tracking with Motion Trail
tracker = ColorTracker()
mask, objects = tracker.track(img, "blue")
result_frame = tracker.draw_tracked(img, objects)

# 5. Smart Document Scanning & Perspective Transform
scanner = DocumentScanner()
doc = scanner.scan(img, mode="color_enhanced")
cv2.imwrite("scanned_doc.png", doc.enhanced)

# 6. QR Code Detection & Decoding
qr_engine = QRCodeScanner()
qr_results = qr_engine.scan(img)
for qr in qr_results:
    print(f"Found QR Payload: {qr.data}")
```

---

## 🧪 Running Unit Tests

VisionCraft includes a comprehensive suite of 37 tests covering all modules:

```bash
pytest tests -v
```

All tests execute headlessly and generate synthetic assets dynamically.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) - Copyright (c) 2026 [ouns-ous](https://github.com/ouns-ous).
