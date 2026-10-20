"""
Unit tests for FeatureMatcher and Homography alignment module.
"""

from pathlib import Path
import pytest
import numpy as np
import cv2

from vision_craft.features import FeatureMatcher, MatchResult


@pytest.fixture
def query_and_scene() -> Tuple[np.ndarray, np.ndarray]:
    """Create a distinctive query object and embed it inside a larger textured scene."""
    # Query object: 120x120 pattern
    query = np.full((120, 120, 3), 40, dtype=np.uint8)
    cv2.circle(query, (60, 60), 30, (0, 0, 255), 3)
    cv2.rectangle(query, (20, 20), (100, 100), (255, 255, 0), 2)
    cv2.putText(query, "TARGET", (25, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Scene: 400x400 with background texture
    scene = np.zeros((400, 400, 3), dtype=np.uint8)
    for i in range(0, 400, 25):
        cv2.line(scene, (0, i), (400, i), (25, 30, 35), 1)
        cv2.line(scene, (i, 0), (i, 400), (25, 30, 35), 1)

    # Embed query inside scene at (150, 120)
    scene[120:240, 150:270] = query
    # Extra distractor shapes
    cv2.circle(scene, (60, 300), 40, (100, 100, 100), -1)

    return query, scene


def test_feature_matcher_detect_and_compute(query_and_scene):
    query, scene = query_and_scene
    matcher = FeatureMatcher(algorithm="orb")
    kp, des = matcher.detect_and_compute(query)

    assert len(kp) > 0
    assert des is not None
    assert des.shape[0] == len(kp)


def test_feature_matcher_match_and_localize(query_and_scene, tmp_path: Path):
    query, scene = query_and_scene
    matcher = FeatureMatcher(algorithm="orb")
    res = matcher.match(query, scene, min_match_count=5)

    assert isinstance(res, MatchResult)
    assert res.num_keypoints_query > 0
    assert res.num_keypoints_scene > 0
    assert res.num_good_matches >= 5
    assert res.matched_vis is not None

    # Test file matching
    q_path = tmp_path / "query.png"
    s_path = tmp_path / "scene.png"
    out_path = tmp_path / "matched.png"
    cv2.imwrite(str(q_path), query)
    cv2.imwrite(str(s_path), scene)

    res_file = matcher.match_files(q_path, s_path, output_path=out_path)
    assert res_file.num_good_matches >= 5
    assert out_path.exists()
