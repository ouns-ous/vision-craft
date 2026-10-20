"""
Unit tests for QRCodeScanner module.
"""

from pathlib import Path
import pytest
import numpy as np
import cv2

from vision_craft.qrcode_scanner import QRCodeScanner, QRScanResult
from vision_craft.synthetic import SyntheticGenerator


def test_qrcode_generate_and_decode():
    scanner = QRCodeScanner()
    payload = "https://github.com/ouns-ous/vision-craft"

    # Generate QR Code image
    qr_img = scanner.generate(payload, scale=8, border=4)
    assert qr_img is not None
    assert len(qr_img.shape) == 3
    assert qr_img.dtype == np.uint8

    # Scan and decode
    results = scanner.scan(qr_img)
    assert len(results) >= 1

    matched = False
    for res in results:
        if res.data == payload:
            matched = True
            assert res.polygon.shape[0] == 4
            assert len(res.bounding_box) == 4
            break
    assert matched, f"Expected decoded payload '{payload}', got {[r.data for r in results]}"


def test_qrcode_draw_results():
    scanner = QRCodeScanner()
    payload = "VISION-CRAFT-TEST"
    qr_img = scanner.generate(payload, scale=6, border=4)
    results = scanner.scan(qr_img)

    annotated = scanner.draw_results(qr_img, results)
    assert annotated.shape == qr_img.shape
    assert annotated.dtype == np.uint8
    assert not np.array_equal(annotated, qr_img)


def test_qrcode_scan_video(tmp_path: Path):
    scanner = QRCodeScanner()
    payload = "VIDEO-QR-CODE"
    qr_img = scanner.generate(payload, scale=4, border=4)
    qr_h, qr_w = qr_img.shape[:2]

    # Create video where frame 5-15 has QR code
    vid_path = tmp_path / "qr_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(vid_path), fourcc, 10, (320, 240))

    for i in range(20):
        frame = np.full((240, 320, 3), 100, dtype=np.uint8)
        if 5 <= i <= 15:
            # Place QR code centered
            frame[20:20 + qr_h, 20:20 + qr_w] = qr_img
        writer.write(frame)

    writer.release()

    manifest = scanner.scan_video(vid_path)
    assert len(manifest) >= 1
    assert manifest[0]["data"] == payload
