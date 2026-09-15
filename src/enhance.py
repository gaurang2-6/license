"""
enhance.py
==========
Forensic image and video enhancement engine for license plates.
Techniques:
1. Perspective unwarping (Homography / 4-point perspective rectification).
2. Motion & Defocus blur correction (Wiener deconvolution with boundary padding).
3. Unsharp masking and high-frequency edge reinforcement.
4. CLAHE contrast adjustment & automatic gamma exposure calibration.
5. Fast Non-Local Means (NLM) & Bilateral denoising.
6. Edge-directed super-resolution & upsampling.
7. Multi-frame ECC video stabilization & temporal median averaging.
8. Adaptive binarization, morphological cleaning, and OCR border conditioning.
9. Auto-rotation correction using Hough line analysis.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Perspective & Geometric Correction
# ---------------------------------------------------------------------------

def rectify_perspective(frame: np.ndarray, corners: np.ndarray, out_size: tuple[int, int] = (400, 120)) -> np.ndarray:
    """Warps a 4-point quadrilateral (top-left, top-right, bottom-right, bottom-left)
    to a standard frontal horizontal license plate rectangle."""
    if frame is None or frame.size == 0:
        logger.warning("rectify_perspective received empty frame")
        return np.zeros((out_size[1], out_size[0], 3), dtype=np.uint8)
    w, h = out_size
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    src = np.float32(corners)
    H = cv2.getPerspectiveTransform(src, dst)
    rectified = cv2.warpPerspective(frame, H, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rectified


# ---------------------------------------------------------------------------
# 2. Blur Correction (Wiener Deconvolution & Unsharp Mask)
# ---------------------------------------------------------------------------

def create_motion_psf(length: int, angle_deg: float, shape: tuple[int, int]) -> np.ndarray:
    """Generates a normalized Point Spread Function (PSF) for linear motion blur."""
    h, w = shape
    psf = np.zeros((h, w), dtype=np.float32)
    center = (w // 2, h // 2)
    half_len = max(1, length // 2)
    
    cv2.line(psf, (center[0] - half_len, center[1]), (center[0] + half_len, center[1]), 1.0, thickness=1)
    
    if abs(angle_deg) > 0.1:
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        psf = cv2.warpAffine(psf, M, (w, h), flags=cv2.INTER_LINEAR)
        
    s = psf.sum()
    return psf / s if s > 0 else psf


def create_defocus_psf(radius: int, shape: tuple[int, int]) -> np.ndarray:
    """Generates a circular disk/pillbox PSF modeling out-of-focus defocus blur."""
    h, w = shape
    psf = np.zeros((h, w), dtype=np.float32)
    center = (w // 2, h // 2)
    cv2.circle(psf, center, max(1, radius), 1.0, -1)
    s = psf.sum()
    return psf / s if s > 0 else psf


def deblur_wiener(img: np.ndarray, length: int = 15, angle: float = 0.0, k: float = 0.015, is_defocus: bool = False) -> np.ndarray:
    """Frequency-domain Wiener deconvolution with boundary reflection padding
    to minimize Gibbs ringing artifacts."""
    if img is None or img.size == 0:
        return img
        
    is_color = (img.ndim == 3)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if is_color else img.copy()
    h, w = gray.shape
    
    # Pad image to reduce edge boundary discontinuities
    pad_h = min(h // 2, 32)
    pad_w = min(w // 2, 32)
    padded = cv2.copyMakeBorder(gray.astype(np.float32) / 255.0, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REFLECT)
    ph, pw = padded.shape

    # Construct PSF matching padded dimensions
    if is_defocus:
        psf = create_defocus_psf(radius=max(2, length // 2), shape=(ph, pw))
    else:
        psf = create_motion_psf(length=max(3, length), angle_deg=angle, shape=(ph, pw))

    # Shift PSF center to (0, 0)
    psf_shifted = np.roll(psf, -ph // 2, axis=0)
    psf_shifted = np.roll(psf_shifted, -pw // 2, axis=1)

    # FFT — use rfft2/irfft2 for real-valued images (~2x faster than full fft2)
    G = np.fft.rfft2(padded)
    H = np.fft.rfft2(psf_shifted, s=padded.shape)
    H_conj = np.conj(H)
    
    # Wiener filter transfer function
    denom = (H * H_conj) + k
    F_hat = (H_conj / denom) * G
    
    restored_padded = np.fft.irfft2(F_hat, s=padded.shape)
    restored = restored_padded[pad_h:pad_h + h, pad_w:pad_w + w]
    restored = np.clip(restored * 255.0, 0, 255).astype(np.uint8)

    if is_color:
        # Preserve original color chrominance while replacing luminance
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = restored
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return restored


def sharpen_unsharp(img: np.ndarray, sigma: float = 1.2, amount: float = 1.5) -> np.ndarray:
    """Enhance high-frequency details via unsharp masking."""
    if img is None or img.size == 0:
        return img
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    sharp = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 3. Denoising & Contrast Enhancement
# ---------------------------------------------------------------------------

def denoise(img: np.ndarray, strength: int = 7) -> np.ndarray:
    """Removes sensor noise and JPEG compression artifacts using Fast Non-Local Means."""
    if img is None or img.size == 0:
        return img
    if img.ndim == 3:
        return cv2.fastNlMeansDenoisingColored(img, None, strength, strength, 7, 21)
    return cv2.fastNlMeansDenoising(img, None, strength, 7, 21)


def denoise_bilateral(img: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """Fast bilateral denoising — preserves edges better than NLM for speed-sensitive use cases."""
    if img is None or img.size == 0:
        return img
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def enhance_contrast(img: np.ndarray, clip_limit: float = 3.0) -> np.ndarray:
    """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    in the LAB color space to boost local contrast without color distortion."""
    if img is None or img.size == 0:
        return img
    if img.ndim == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        return clahe.apply(img)


# Cached gamma LUT tables to avoid recomputing on every call
_GAMMA_LUT_CACHE: dict[float, np.ndarray] = {}


def _get_gamma_lut(gamma: float) -> np.ndarray:
    """Returns a cached gamma correction LUT for the given gamma value."""
    if gamma not in _GAMMA_LUT_CACHE:
        _GAMMA_LUT_CACHE[gamma] = np.array(
            [((i / 255.0) ** gamma) * 255 for i in range(256)]
        ).astype(np.uint8)
    return _GAMMA_LUT_CACHE[gamma]


def auto_exposure_correct(img: np.ndarray) -> np.ndarray:
    """Estimates mean brightness and applies calibrated gamma correction:
    brightens underexposed night footage and tones down overexposed glare."""
    if img is None or img.size == 0:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    mean_val = float(gray.mean())
    
    if mean_val < 85:
        # Underexposed: lighten shadows
        gamma = 0.55
    elif mean_val > 170:
        # Overexposed: darken highlights
        gamma = 1.75
    else:
        gamma = 1.0
        
    table = _get_gamma_lut(gamma)
    lut_img = cv2.LUT(img, table)
    # Boost contrast if image has sufficient variance; avoid CLAHE on flat/uniform images
    if float(gray.std()) > 5.0:
        return enhance_contrast(lut_img, clip_limit=2.5)
    return lut_img


# ---------------------------------------------------------------------------
# 4. Super-Resolution & Spatial Upscaling
# ---------------------------------------------------------------------------

def super_resolve(img: np.ndarray, scale: int = 3) -> np.ndarray:
    """Multi-scale edge-directed upscaling. Enlarges character glyphs
    using bicubic interpolation followed by unsharp edge reconstruction."""
    if img is None or img.size == 0:
        return img
    h, w = img.shape[:2]
    # Use INTER_LINEAR for intermediate upscaling (3x faster than INTER_CUBIC);
    # sharpening step below recovers edge detail
    upscaled = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)
    sharpened = sharpen_unsharp(upscaled, sigma=1.4, amount=1.2)
    return sharpened


def add_ocr_border(img: np.ndarray, padding: int = 24) -> np.ndarray:
    """Adds a clean neutral border around the crop to prevent character strokes
    from touching image edges, which dramatically improves OCR recognition."""
    if img is None or img.size == 0:
        return img
    return cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])


def binarize_for_ocr(img: np.ndarray) -> np.ndarray:
    """Adaptive Gaussian thresholding tailored for OCR text segmentation."""
    if img is None or img.size == 0:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9)
    return th


def morphological_clean(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Applies morphological open then close to remove small noise specks
    from binarized plate images while preserving character strokes."""
    if img is None or img.size == 0:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    cleaned = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cleaned


# ---------------------------------------------------------------------------
# 5. Auto-Rotation Detection
# ---------------------------------------------------------------------------

def auto_deskew(img: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """Detects slight rotation using Hough line analysis and corrects tilt.
    Only corrects angles within ±max_angle degrees to avoid false corrections."""
    if img is None or img.size == 0:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)

    if lines is None or len(lines) == 0:
        return img

    # Compute median angle from detected lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line.ravel()
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) > 5:  # Ignore near-vertical lines
            angle = np.degrees(np.arctan2(dy, dx))
            if abs(angle) <= max_angle:
                angles.append(angle)

    if not angles:
        return img

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:
        return img  # Not worth correcting sub-half-degree tilts

    logger.debug("Auto-deskew: correcting %.1f° rotation", median_angle)
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def ensure_bgr(img: np.ndarray) -> np.ndarray:
    """Ensures an image is 3-channel BGR. Converts grayscale to BGR if needed.
    This prevents channel mismatch errors when compositing images."""
    if img is None or img.size == 0:
        return img
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


# ---------------------------------------------------------------------------
# 6. Video Stabilization & Multi-Frame Fusion
# ---------------------------------------------------------------------------

def align_frames_ecc(ref_frame: np.ndarray, frames: list[np.ndarray], warp_mode: int = cv2.MOTION_EUCLIDEAN) -> list[np.ndarray]:
    """Aligns video frames to a reference frame using OpenCV's Enhanced Correlation
    Coefficient (ECC) algorithm to compensate for camera jitter and vibration."""
    aligned = []
    ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    h, w = ref_frame.shape[:2]

    for idx, f in enumerate(frames):
        f_gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        try:
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 1e-4)
            _, warp_matrix = cv2.findTransformECC(ref_gray, f_gray, warp_matrix, warp_mode, criteria)
            warped = cv2.warpAffine(f, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
            aligned.append(warped)
        except cv2.error:
            # If alignment fails for a shaky frame, preserve original frame
            logger.debug("ECC alignment failed for frame %d, using original", idx)
            aligned.append(f)
    return aligned


def multiframe_average(frames: list[np.ndarray]) -> np.ndarray:
    """Temporal median stacking across aligned frames to cancel out noise
    and recover persistent character structures."""
    if not frames:
        raise ValueError("Frames list cannot be empty")
    if len(frames) == 1:
        return frames[0]
    stack = np.stack(frames, axis=0).astype(np.float32)
    median_frame = np.median(stack, axis=0)
    return np.clip(median_frame, 0, 255).astype(np.uint8)
