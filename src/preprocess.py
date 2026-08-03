"""
Image preprocessing: denoise, CLAHE, sharpen, deskew, adaptive threshold.
No hardcoded outputs — every step is derived from the input pixels.
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple


def load_and_normalize(image_path: str, max_dim: int = 2200) -> np.ndarray:
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not load image: {image_path}")
    h, w = bgr.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return bgr


def to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)


def clahe(gray: np.ndarray, clip: float = 2.5, tile: int = 8) -> np.ndarray:
    op = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return op.apply(gray)


def sharpen(gray: np.ndarray) -> np.ndarray:
    k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(gray, -1, k)


def deskew(gray: np.ndarray) -> Tuple[np.ndarray, float]:
    """Estimate skew from text-heavy regions and correct it."""
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thr > 0))
    if coords.size == 0:
        return gray, 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.4:
        return gray, 0.0
    (h, w) = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated, float(angle)


def adaptive_binary(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 9
    )


def full_pipeline(bgr: np.ndarray) -> dict:
    """Return every intermediate image the rest of the pipeline needs."""
    gray = to_gray(bgr)
    dn = denoise(gray)
    eq = clahe(dn)
    sharp = sharpen(eq)
    desk, angle = deskew(sharp)
    binary = adaptive_binary(desk)
    return {
        "gray": gray,
        "denoised": dn,
        "clahe": eq,
        "sharp": sharp,
        "deskewed": desk,
        "binary": binary,
        "skew_angle": angle,
    }
