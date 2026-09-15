"""
metrics.py
==========
Quantitative evaluation metrics for forensic license plate enhancement:
- OCR character accuracy: exact match, Levenshtein edit distance, normalized similarity.
- Image quality metrics: Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM).
- No-reference image sharpness: Laplacian variance (Tenengrad-style focus measure).
- Processing time tracking for pipeline latency measurement.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Maximum finite PSNR value to prevent JSON serialization failures with inf
_MAX_PSNR = 100.0


def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes the Levenshtein edit distance between two strings."""
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)
    
    dp = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, 1):
        new_dp = [i] + [0] * len(s2)
        for j, c2 in enumerate(s2, 1):
            cost = 0 if c1 == c2 else 1
            new_dp[j] = min(dp[j] + 1, new_dp[j - 1] + 1, dp[j - 1] + cost)
        dp = new_dp
    return dp[-1]


def char_accuracy(ground_truth: str, predicted: str) -> dict:
    """Computes exact match, edit distance, and normalized character similarity."""
    gt = ground_truth.strip().upper().replace(" ", "").replace("-", "")
    pred = predicted.strip().upper().replace(" ", "").replace("-", "")
    dist = levenshtein_distance(gt, pred)
    max_len = max(len(gt), 1)
    similarity = max(0.0, 1.0 - (dist / max_len))
    
    return {
        "exact_match": gt == pred,
        "edit_distance": dist,
        "char_similarity": round(similarity, 4),
        "ground_truth": gt,
        "predicted": pred,
    }


def compute_sharpness(img: np.ndarray) -> float:
    """Variance of the Laplacian: higher value indicates sharper edges."""
    if img is None or img.size == 0:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _compute_ssim_cv(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes SSIM using OpenCV operations (~5x faster than skimage).
    Uses the standard SSIM formula with Gaussian-weighted windows."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 * img1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator
    return float(ssim_map.mean())


def compute_psnr_ssim(reference: np.ndarray, test: np.ndarray) -> tuple[float, float]:
    """Computes PSNR and SSIM between a test image and reference image.
    Uses OpenCV PSNR (~5x faster than skimage) and a custom CV-based SSIM.
    Automatically handles resizing and grayscale conversion.
    PSNR is clamped to a finite maximum to prevent JSON serialization failures."""
    if reference is None or test is None or reference.size == 0 or test.size == 0:
        return 0.0, 0.0
        
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY) if reference.ndim == 3 else reference
    test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY) if test.ndim == 3 else test
    
    if test_gray.shape != ref_gray.shape:
        test_gray = cv2.resize(test_gray, (ref_gray.shape[1], ref_gray.shape[0]), interpolation=cv2.INTER_AREA)
    
    try:
        p = cv2.PSNR(ref_gray, test_gray)
        # Clamp inf or very high values to finite maximum
        # (B8: identical images produce inf with skimage, or very high finite with cv2.PSNR)
        if not np.isfinite(p) or p > _MAX_PSNR:
            p = _MAX_PSNR
            logger.debug("PSNR clamped to %.1f (identical or near-identical images)", _MAX_PSNR)
    except Exception:
        logger.warning("PSNR computation failed, defaulting to 0.0", exc_info=True)
        p = 0.0

    try:
        s = _compute_ssim_cv(ref_gray, test_gray)
    except Exception:
        logger.warning("SSIM computation failed, defaulting to 0.0", exc_info=True)
        s = 0.0

    return float(p), float(s)


@contextmanager
def timer(label: str = "operation"):
    """Context manager to measure elapsed time for a pipeline stage.

    Usage::

        with timer("enhancement") as t:
            result = enhance(image)
        print(t.elapsed)  # seconds as float
    """
    class _Timer:
        elapsed: float = 0.0

    t = _Timer()
    start = time.perf_counter()
    try:
        yield t
    finally:
        t.elapsed = time.perf_counter() - start
        logger.debug("%s completed in %.3fs", label, t.elapsed)
