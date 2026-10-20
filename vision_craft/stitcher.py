"""
Panorama and multi-image stitching engine with automatic exposure and seam compensation.
"""

from pathlib import Path
from typing import List, Tuple, Union, Optional
import cv2
import numpy as np


class ImageStitcher:
    """Stitches multiple overlapping images into a seamless panorama."""

    def __init__(self, mode: str = "panorama"):
        self.mode = mode.lower()
        if self.mode == "scans":
            stitch_mode = cv2.Stitcher_SCANS if hasattr(cv2, "Stitcher_SCANS") else 1
        else:
            stitch_mode = cv2.Stitcher_PANORAMA if hasattr(cv2, "Stitcher_PANORAMA") else 0

        self.stitcher = cv2.Stitcher_create(stitch_mode) if hasattr(cv2, "Stitcher_create") else None

    @staticmethod
    def crop_black_borders(image: np.ndarray) -> np.ndarray:
        """Crop irregular black borders created by perspective warping during stitching."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        non_zeros = cv2.findNonZero(thresh)
        if non_zeros is None:
            return image

        x, y, w, h = cv2.boundingRect(non_zeros)
        if w > 10 and h > 10:
            return image[y:y + h, x:x + w]
        return image

    def stitch_fallback_pairwise(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[bool, Optional[np.ndarray], str]:
        """Pairwise homography stitching fallback using ORB feature matching."""
        orb = cv2.ORB_create(nfeatures=1500)
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return False, None, "Insufficient keypoints for pairwise stitching"

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)
        matches = sorted(matches, key=lambda m: m.distance)

        if len(matches) < 4:
            return False, None, "Not enough valid feature matches found"

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
        if H is None:
            return False, None, "Homography estimation failed"

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        canvas_w = w1 + w2
        canvas_h = max(h1, h2)

        warped = cv2.warpPerspective(img2, H, (canvas_w, canvas_h))
        # Overlay img1 on the left
        warped[0:h1, 0:w1] = img1
        cropped = self.crop_black_borders(warped)
        return True, cropped, "Pairwise ORB stitch successful"

    def stitch(
        self,
        images: List[np.ndarray],
        crop_borders: bool = True,
    ) -> Tuple[bool, Optional[np.ndarray], str]:
        """Stitch a sequence of images into a unified composite."""
        if len(images) < 2:
            return False, None, "At least two images are required for stitching."

        # Attempt OpenCV built-in Stitcher
        if self.stitcher is not None:
            status, pano = self.stitcher.stitch(images)
            if status == (cv2.Stitcher_OK if hasattr(cv2, "Stitcher_OK") else 0) and pano is not None:
                target_h = max(img.shape[0] for img in images)
                if 0 < pano.shape[0] < target_h // 2:
                    scale = target_h / float(pano.shape[0])
                    pano = cv2.resize(pano, (int(pano.shape[1] * scale), target_h), interpolation=cv2.INTER_LINEAR)
                if crop_borders and pano is not None:
                    pano = self.crop_black_borders(pano)
                return True, pano, "OpenCV Stitcher succeeded"

        # Fallback to pairwise stitching if 2 images
        if len(images) == 2:
            return self.stitch_fallback_pairwise(images[0], images[1])

        return False, None, "Stitching failed: images could not be aligned."

    def stitch_files(
        self,
        image_paths: List[Union[str, Path]],
        output_path: Optional[Union[str, Path]] = None,
        crop_borders: bool = True,
    ) -> Tuple[bool, Optional[np.ndarray], str]:
        """Load image files, stitch panorama, and optionally save output."""
        images = []
        for p in image_paths:
            img = cv2.imread(str(p))
            if img is None:
                return False, None, f"Failed to load image file: {p}"
            images.append(img)

        success, pano, msg = self.stitch(images, crop_borders=crop_borders)
        if success and output_path and pano is not None:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), pano)

        return success, pano, msg
