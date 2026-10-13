"""
Unit tests for ImageFilter operations.
"""

import pytest
import numpy as np
import cv2

from vision_craft.filters import (
    ImageFilter,
    apply_filter,
    get_available_filters,
    FILTER_REGISTRY,
)


@pytest.fixture
def sample_image() -> np.ndarray:
    """Fixture providing a standard test color pattern."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(img, (50, 50), 30, (100, 150, 200), -1)
    cv2.rectangle(img, (10, 10), (40, 40), (255, 50, 50), -1)
    return img


def test_grayscale(sample_image: np.ndarray):
    res = ImageFilter.grayscale(sample_image)
    assert res.shape == sample_image.shape
    # Check that channels are identical (3-channel BGR representation of gray)
    assert np.array_equal(res[:, :, 0], res[:, :, 1])
    assert np.array_equal(res[:, :, 1], res[:, :, 2])


def test_sepia(sample_image: np.ndarray):
    res = ImageFilter.sepia(sample_image)
    assert res.shape == sample_image.shape
    assert res.dtype == np.uint8


def test_invert(sample_image: np.ndarray):
    res = ImageFilter.invert(sample_image)
    assert res.shape == sample_image.shape
    assert res[0, 0, 0] == 255 - sample_image[0, 0, 0]


def test_blur_filters(sample_image: np.ndarray):
    gblur = ImageFilter.gaussian_blur(sample_image, ksize=5)
    assert gblur.shape == sample_image.shape

    mblur = ImageFilter.median_blur(sample_image, ksize=5)
    assert mblur.shape == sample_image.shape

    bilat = ImageFilter.bilateral_filter(sample_image, d=5)
    assert bilat.shape == sample_image.shape


def test_edge_filters(sample_image: np.ndarray):
    canny = ImageFilter.canny_edges(sample_image)
    assert canny.shape == sample_image.shape

    sobel = ImageFilter.sobel(sample_image, direction="both")
    assert sobel.shape == sample_image.shape

    lap = ImageFilter.laplacian(sample_image)
    assert lap.shape == sample_image.shape


def test_thresholds(sample_image: np.ndarray):
    bin_thresh = ImageFilter.threshold_binary(sample_image, thresh=100)
    assert bin_thresh.shape == sample_image.shape

    otsu = ImageFilter.threshold_otsu(sample_image)
    assert otsu.shape == sample_image.shape

    adap = ImageFilter.threshold_adaptive_gaussian(sample_image)
    assert adap.shape == sample_image.shape


def test_artistic_filters(sample_image: np.ndarray):
    sketch = ImageFilter.pencil_sketch(sample_image)
    assert sketch.shape == sample_image.shape

    cartoon = ImageFilter.cartoon(sample_image, num_down=1, num_bilateral=2)
    assert cartoon.shape == sample_image.shape

    emboss = ImageFilter.emboss(sample_image)
    assert emboss.shape == sample_image.shape

    vignette = ImageFilter.vignette(sample_image)
    assert vignette.shape == sample_image.shape

    quant = ImageFilter.color_quantization(sample_image, k=4)
    assert quant.shape == sample_image.shape


def test_filter_registry(sample_image: np.ndarray):
    available = get_available_filters()
    assert len(available) > 10
    assert "canny" in available
    assert "cartoon" in available

    for f_name in ["gray", "sepia", "invert", "blur", "canny", "binary"]:
        out = apply_filter(sample_image, f_name)
        assert out.shape == sample_image.shape

    with pytest.raises(ValueError):
        apply_filter(sample_image, "non_existent_filter_name")
