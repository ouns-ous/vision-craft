"""
Keypoint detection, feature matching, and homography object localization module.
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional, Union
from pathlib import Path
import cv2
import numpy as np

from vision_craft.config import UI_COLORS


@dataclass
class MatchResult:
    """Encapsulates feature matching metrics and localized scene corners."""
    num_keypoints_query: int
    num_keypoints_scene: int
    num_good_matches: int
    homography: Optional[np.ndarray]
    scene_corners: Optional[np.ndarray]  # (4, 2) projected corners in scene coordinates
    matched_vis: np.ndarray


class FeatureMatcher:
    """Matches distinctive visual features between target query and scene."""

    def __init__(self, algorithm: str = "orb", n_features: int = 2000, ratio_threshold: float = 0.75):
        self.algorithm = algorithm.lower()
        self.ratio_threshold = ratio_threshold

        if self.algorithm == "sift" and hasattr(cv2, "SIFT_create"):
            self.detector = cv2.SIFT_create(nfeatures=n_features)
            self.norm = cv2.NORM_L2
        else:
            self.detector = cv2.ORB_create(nfeatures=n_features)
            self.norm = cv2.NORM_HAMMING

    def detect_and_compute(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """Detect keypoints and compute descriptors for an image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return self.detector.detectAndCompute(gray, None)

    def match(
        self,
        query_image: np.ndarray,
        scene_image: np.ndarray,
        min_match_count: int = 10,
    ) -> MatchResult:
        """Find query object inside scene image and localize with projected quadrangle."""
        kp_query, des_query = self.detect_and_compute(query_image)
        kp_scene, des_scene = self.detect_and_compute(scene_image)

        if des_query is None or des_scene is None or len(kp_query) < 4 or len(kp_scene) < 4:
            vis = np.hstack([query_image, scene_image]) if query_image.shape[0] == scene_image.shape[0] else scene_image
            return MatchResult(
                num_keypoints_query=len(kp_query),
                num_keypoints_scene=len(kp_scene),
                num_good_matches=0,
                homography=None,
                scene_corners=None,
                matched_vis=vis,
            )

        # KNN matching with k=2 for Lowe's ratio test
        bf = cv2.BFMatcher(self.norm, crossCheck=False)
        raw_matches = bf.knnMatch(des_query, des_scene, k=2)

        good_matches = []
        for pair in raw_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)

        scene_corners = None
        H = None

        if len(good_matches) >= min_match_count:
            src_pts = np.float32([kp_query[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is not None:
                qh, qw = query_image.shape[:2]
                query_box = np.float32([[0, 0], [qw - 1, 0], [qw - 1, qh - 1], [0, qh - 1]]).reshape(-1, 1, 2)
                try:
                    transformed = cv2.perspectiveTransform(query_box, H)
                    scene_corners = transformed.reshape(4, 2)
                except cv2.error:
                    scene_corners = None

        # Draw matches visualization
        matched_vis = cv2.drawMatches(
            query_image, kp_query,
            scene_image, kp_scene,
            good_matches[:50],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        # Draw projected polygon onto scene in visualization
        if scene_corners is not None:
            qw = query_image.shape[1]
            vis_corners = scene_corners.copy()
            vis_corners[:, 0] += qw  # Offset by query image width
            pts_int = vis_corners.astype(int).reshape((-1, 1, 2))
            cv2.polylines(matched_vis, [pts_int], isClosed=True, color=UI_COLORS["lime"], thickness=3)

        return MatchResult(
            num_keypoints_query=len(kp_query),
            num_keypoints_scene=len(kp_scene),
            num_good_matches=len(good_matches),
            homography=H,
            scene_corners=scene_corners,
            matched_vis=matched_vis,
        )

    def match_files(
        self,
        query_path: Union[str, Path],
        scene_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> MatchResult:
        """Execute matching from file paths and optionally save annotated visual."""
        q_img = cv2.imread(str(query_path))
        s_img = cv2.imread(str(scene_path))
        if q_img is None or s_img is None:
            raise FileNotFoundError("Could not read query or scene image file.")

        res = self.match(q_img, s_img)
        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), res.matched_vis)

        return res
