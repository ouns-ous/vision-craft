"""
Image filtering, thresholding, edge detection, and artistic processing module.
"""

from typing import Dict, Callable, Any, Optional
import cv2
import numpy as np


class ImageFilter:
    """Provides high-performance image transformations and filtering operations."""

    @staticmethod
    def grayscale(image: np.ndarray) -> np.ndarray:
        """Convert BGR image to single-channel Grayscale (returned as 3-channel for pipeline compatibility)."""
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def sepia(image: np.ndarray) -> np.ndarray:
        """Apply warm vintage Sepia tone filter."""
        kernel = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ])
        transformed = cv2.transform(image, kernel)
        return np.clip(transformed, 0, 255).astype(np.uint8)

    @staticmethod
    def invert(image: np.ndarray) -> np.ndarray:
        """Invert color channels (negative effect)."""
        return cv2.bitwise_not(image)

    @staticmethod
    def hsv(image: np.ndarray) -> np.ndarray:
        """Convert to HSV color representation."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    @staticmethod
    def lab(image: np.ndarray) -> np.ndarray:
        """Convert to CIELAB color representation."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    @staticmethod
    def gaussian_blur(image: np.ndarray, ksize: int = 15, sigma: float = 0.0) -> np.ndarray:
        """Apply Gaussian smoothing filter."""
        k = ksize if ksize % 2 != 0 else ksize + 1
        return cv2.GaussianBlur(image, (k, k), sigma)

    @staticmethod
    def median_blur(image: np.ndarray, ksize: int = 9) -> np.ndarray:
        """Apply salt-and-pepper removing median filter."""
        k = ksize if ksize % 2 != 0 else ksize + 1
        return cv2.medianBlur(image, k)

    @staticmethod
    def bilateral_filter(image: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
        """Edge-preserving bilateral smoothing filter."""
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    @staticmethod
    def canny_edges(image: np.ndarray, threshold1: float = 50, threshold2: float = 150) -> np.ndarray:
        """Canny edge detection returned as 3-channel visual overlay."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        edges = cv2.Canny(blurred, threshold1, threshold2)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def sobel(image: np.ndarray, direction: str = "both", ksize: int = 3) -> np.ndarray:
        """Sobel gradient operator along x, y, or both axes."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        if direction == "x":
            grad = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        elif direction == "y":
            grad = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        else:
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
            grad = cv2.magnitude(grad_x, grad_y)
        grad_scaled = np.uint8(np.absolute(grad) / (np.max(grad) + 1e-6) * 255)
        return cv2.cvtColor(grad_scaled, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def laplacian(image: np.ndarray, ksize: int = 3) -> np.ndarray:
        """Laplacian 2nd derivative edge detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
        lap_abs = np.uint8(np.absolute(lap) / (np.max(lap) + 1e-6) * 255)
        return cv2.cvtColor(lap_abs, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def threshold_binary(image: np.ndarray, thresh: int = 127, maxval: int = 255) -> np.ndarray:
        """Standard binary threshold."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        _, binary = cv2.threshold(gray, thresh, maxval, cv2.THRESH_BINARY)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def threshold_otsu(image: np.ndarray) -> np.ndarray:
        """Automatic bimodal Otsu thresholding."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def threshold_adaptive_gaussian(image: np.ndarray, block_size: int = 11, c: int = 2) -> np.ndarray:
        """Adaptive Gaussian window thresholding."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        b = block_size if block_size % 2 != 0 else block_size + 1
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, b, c
        )
        return cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def pencil_sketch(image: np.ndarray) -> np.ndarray:
        """Generate high-contrast fine pencil sketch effect."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def cartoon(image: np.ndarray, num_down: int = 2, num_bilateral: int = 5) -> np.ndarray:
        """Generate cartoon cel-shading effect using bilateral filtering and edge masking."""
        img_color = image.copy()
        for _ in range(num_down):
            img_color = cv2.pyrDown(img_color)

        for _ in range(num_bilateral):
            img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)

        for _ in range(num_down):
            img_color = cv2.pyrUp(img_color)

        # Resize to original if slight pyr dimensions mismatch
        if img_color.shape[:2] != image.shape[:2]:
            img_color = cv2.resize(img_color, (image.shape[1], image.shape[0]))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        med = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(
            med, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2
        )
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return cv2.bitwise_and(img_color, edges_bgr)

    @staticmethod
    def emboss(image: np.ndarray) -> np.ndarray:
        """Emboss texture relief filter."""
        kernel = np.array([
            [-2, -1, 0],
            [-1, 1, 1],
            [0, 1, 2]
        ])
        emb = cv2.filter2D(image, -1, kernel) + 128
        return np.clip(emb, 0, 255).astype(np.uint8)

    @staticmethod
    def vignette(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """Apply radial dark vignette border."""
        rows, cols = image.shape[:2]
        kernel_x = cv2.getGaussianKernel(cols, cols / (2 + strength))
        kernel_y = cv2.getGaussianKernel(rows, rows / (2 + strength))
        mask = kernel_y * kernel_x.T
        mask = mask / np.max(mask)

        vignette_img = np.empty_like(image, dtype=np.float32)
        for i in range(3):
            vignette_img[:, :, i] = image[:, :, i] * mask
        return np.clip(vignette_img, 0, 255).astype(np.uint8)

    @staticmethod
    def color_quantization(image: np.ndarray, k: int = 8) -> np.ndarray:
        """Reduce image color depth to K dominant clusters."""
        data = np.float32(image).reshape((-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        quantized = centers[labels.flatten()]
        return quantized.reshape(image.shape)


FILTER_REGISTRY: Dict[str, Callable[..., np.ndarray]] = {
    "gray": ImageFilter.grayscale,
    "grayscale": ImageFilter.grayscale,
    "sepia": ImageFilter.sepia,
    "invert": ImageFilter.invert,
    "negative": ImageFilter.invert,
    "hsv": ImageFilter.hsv,
    "lab": ImageFilter.lab,
    "blur": ImageFilter.gaussian_blur,
    "gaussian_blur": ImageFilter.gaussian_blur,
    "median_blur": ImageFilter.median_blur,
    "bilateral": ImageFilter.bilateral_filter,
    "canny": ImageFilter.canny_edges,
    "edges": ImageFilter.canny_edges,
    "sobel": ImageFilter.sobel,
    "laplacian": ImageFilter.laplacian,
    "binary": ImageFilter.threshold_binary,
    "threshold": ImageFilter.threshold_binary,
    "otsu": ImageFilter.threshold_otsu,
    "adaptive": ImageFilter.threshold_adaptive_gaussian,
    "sketch": ImageFilter.pencil_sketch,
    "pencil": ImageFilter.pencil_sketch,
    "cartoon": ImageFilter.cartoon,
    "emboss": ImageFilter.emboss,
    "vignette": ImageFilter.vignette,
    "quantize": ImageFilter.color_quantization,
}


def get_available_filters() -> list[str]:
    """Return list of supported filter identifiers."""
    return sorted(list(FILTER_REGISTRY.keys()))


def apply_filter(image: np.ndarray, filter_name: str, **kwargs: Any) -> np.ndarray:
    """Apply a registered filter by name with optional parameter forwarding."""
    fn = FILTER_REGISTRY.get(filter_name.lower())
    if fn is None:
        valid = ", ".join(get_available_filters())
        raise ValueError(f"Filter '{filter_name}' not found. Available filters: {valid}")
    return fn(image, **kwargs)
