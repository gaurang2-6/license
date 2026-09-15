"""
detect.py
=========
License plate detection module with cascading detector strategy:
1. YOLOv8 deep learning detector (highest accuracy, requires trained weights).
2. Full-scene OCR text box detection (RapidOCR, good for dark/night captures).
3. Classical CV multi-pass fallback (Sobel edge density + morphology + aspect ratio).
"""
from __future__ import annotations

import logging
import os

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Resolve project root relative to this file, not CWD
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_YOLO_MODEL = None


def load_yolo_model():
    """Loads the trained YOLOv8 license plate detector if weights are present.
    Model paths are resolved relative to the project root, not the CWD."""
    global _YOLO_MODEL
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL

    # B1: Use project-root-relative paths instead of CWD-relative
    # Prefer ONNX (2-4x faster CPU inference, no PyTorch overhead) over .pt
    model_paths = [
        os.path.join(_PROJECT_ROOT, "models", "license_plate_yolo.onnx"),
        os.path.join(_PROJECT_ROOT, "models", "license_plate_yolo.pt"),
        os.path.join(_PROJECT_ROOT, "models", "yolo_plate_run", "weights", "best.pt"),
    ]

    for p in model_paths:
        if os.path.exists(p):
            try:
                from ultralytics import YOLO
                _YOLO_MODEL = YOLO(p)
                logger.info("Deep Learning YOLO model loaded from: %s", p)
                return _YOLO_MODEL
            except Exception as e:
                logger.warning("Failed to load YOLO model from %s: %s", p, e)

    logger.debug("No YOLO model weights found, will use fallback detectors")
    return None


def detect_plate_yolo(frame: np.ndarray, conf_threshold: float = 0.20) -> dict:
    """Runs deep learning YOLO object detector on input frame."""
    model = load_yolo_model()
    if model is None:
        return {"found": False, "box": None, "axis_aligned_bbox": None, "score": 0.0}

    try:
        results = model.predict(source=frame, verbose=False, conf=conf_threshold)
        if results and len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            best_idx = int(np.argmax(boxes.conf.cpu().numpy()))
            conf = float(boxes.conf[best_idx].cpu().item())

            xyxy = boxes.xyxy[best_idx].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)

            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            axis_aligned_bbox = (x1, y1, w, h)

            pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            ordered_pts = order_corners(pts)

            return {
                "found": True,
                "box": ordered_pts,
                "axis_aligned_bbox": axis_aligned_bbox,
                "score": round(conf, 4),
                "detector_type": "yolo_deep_learning"
            }
    except Exception as e:
        # B3: Log the error instead of silently swallowing
        logger.warning("YOLO prediction error: %s", e, exc_info=True)

    return {"found": False, "box": None, "axis_aligned_bbox": None, "score": 0.0}


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 quadrilateral points as:
    [top-left, top-right, bottom-right, bottom-left]."""
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _score_candidate(bw: float, bh: float, edge_density: float, frame_area: float) -> float:
    """Computes a confidence score for a candidate contour based on aspect ratio,
    area, and vertical character edge density, cleanly rejecting small noise and tall vertical background objects."""
    if bh <= 0 or bw <= 0:
        return -1.0
        
    aspect = bw / bh
    # License plates are horizontal rectangular plates: aspect 1.1 - 6.5
    if not (1.1 <= aspect <= 6.5):
        return -1.0
        
    area = bw * bh
    min_area = max(700.0, 0.0035 * frame_area)
    if area < min_area or area > (0.45 * frame_area):
        return -1.0

    if bw < 32 or bh < 10:
        return -1.0

    # Reject circular headlights: aspect close to 1.0 AND small area (< 7500 px)
    if aspect < 1.5 and area < 7500:
        return -1.0
        
    area_norm = min(area / 25000.0, 1.0)
    aspect_score = 1.0 - min(abs(aspect - 3.2) / 3.2, 1.0)
        
    score = 0.40 * aspect_score + 0.40 * min(edge_density * 2.5, 1.0) + 0.20 * area_norm
    return float(score)


def _find_candidates_from_binary(thresh: np.ndarray, gray: np.ndarray, frame_area: float, kernel_w: int, kernel_h: int):
    """Extracts candidate contours and scores them from a binary edge/morph map."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, np.ones((3, 3), np.uint8), iterations=1)
    closed = cv2.dilate(closed, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for c in contours:
        rect = cv2.minAreaRect(c)
        x, y, bw, bh = cv2.boundingRect(c)
        
        roi_edges = thresh[y:y + bh, x:x + bw]
        edge_density = float((roi_edges > 0).mean()) if roi_edges.size > 0 else 0.0
        
        score = _score_candidate(float(bw), float(bh), edge_density, frame_area)
        if score > 0.18:
            candidates.append((score, rect, (x, y, bw, bh), closed))

    return candidates


def preprocess_frame_if_dark(frame: np.ndarray) -> np.ndarray:
    """Automatically enhances contrast on under-exposed or night surveillance frames."""
    if frame is None or frame.size == 0:
        return frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    if gray.mean() < 45.0 or gray.max() < 85:
        logger.debug("Dark frame detected (mean=%.1f, max=%d), applying CLAHE", gray.mean(), gray.max())
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(l)
        return cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)
    return frame


# Cached RapidOCR instance for detection pass (avoids ~200ms init per call)
_DETECT_RAPID_OCR = None
_DETECT_RAPID_OCR_INIT_ATTEMPTED = False


def _get_detect_rapid_ocr():
    """Lazily initializes and caches a RapidOCR instance for detection."""
    global _DETECT_RAPID_OCR, _DETECT_RAPID_OCR_INIT_ATTEMPTED
    if _DETECT_RAPID_OCR_INIT_ATTEMPTED:
        return _DETECT_RAPID_OCR
    _DETECT_RAPID_OCR_INIT_ATTEMPTED = True
    try:
        from rapidocr_onnxruntime import RapidOCR
        _DETECT_RAPID_OCR = RapidOCR()
    except ImportError:
        logger.debug("RapidOCR not available for full-scene detection pass")
    except Exception as e:
        logger.warning("Failed to init RapidOCR for detection: %s", e)
    return _DETECT_RAPID_OCR


def detect_plate_ocr_pass(proc_frame: np.ndarray) -> dict:
    """Runs a full-scene RapidOCR text box pass for ultra-degraded/dark night captures."""
    try:
        ocr = _get_detect_rapid_ocr()
        if ocr is None:
            return {"found": False, "box": None, "axis_aligned_bbox": None, "score": 0.0}
        results, _ = ocr(proc_frame)
        if results:
            best_cand = None
            best_score = 0.0
            for item in results:
                pts, text, conf = item
                if conf < 0.40 or len(text) < 3:
                    continue
                pts_arr = np.array(pts, dtype=np.float32)
                x_min, y_min = np.min(pts_arr, axis=0)
                x_max, y_max = np.max(pts_arr, axis=0)
                bw, bh = max(1.0, x_max - x_min), max(1.0, y_max - y_min)
                aspect = bw / bh
                if 1.5 <= aspect <= 7.0:
                    score = float(conf)
                    if score > best_score:
                        best_score = score
                        ordered_pts = order_corners(pts_arr)
                        axis_aligned = (int(x_min), int(y_min), int(bw), int(bh))
                        best_cand = {
                            "found": True,
                            "box": ordered_pts,
                            "axis_aligned_bbox": axis_aligned,
                            "score": round(score, 4),
                            "detector_type": "full_scene_ocr"
                        }
            if best_cand is not None:
                return best_cand
    except Exception as e:
        # B3: Log the error instead of silently swallowing
        logger.warning("Full-scene OCR detection pass failed: %s", e, exc_info=True)
    return {"found": False, "box": None, "axis_aligned_bbox": None, "score": 0.0}


def detect_plate(frame: np.ndarray, debug: bool = False) -> dict:
    """Detects and isolates the most probable license plate candidate in a frame.
    First attempts YOLO deep learning detector. If unavailable or low confidence,
    attempts full-scene OCR text box detection and multi-pass Sobel/BlackHat morphology.
    """
    if frame is None or frame.size == 0:
        return {"found": False, "box": None, "axis_aligned_bbox": None, "score": 0.0}

    # Pre-process dark/under-exposed frames
    proc_frame = preprocess_frame_if_dark(frame)

    # 1. Try Deep Learning YOLO Detector (raw frame first, pre-enhanced only if needed)
    yolo_res = detect_plate_yolo(frame)
    if not yolo_res["found"] or yolo_res["score"] < 0.20:
        # Only retry on preprocessed frame if the raw frame failed
        yolo_res = detect_plate_yolo(proc_frame)

    if yolo_res["found"] and yolo_res["score"] >= 0.20:
        logger.debug("Plate detected via YOLO (score=%.3f)", yolo_res["score"])
        return yolo_res

    # 2. Try Full-scene OCR text detection pass (for dark/night surveillance)
    ocr_res = detect_plate_ocr_pass(proc_frame)
    if ocr_res["found"] and ocr_res["score"] >= 0.50:
        logger.debug("Plate detected via full-scene OCR (score=%.3f)", ocr_res["score"])
        return ocr_res

    h, w = proc_frame.shape[:2]
    frame_area = float(h * w)

    # 3. Classical CV Multi-pass Fallback
    gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY) if proc_frame.ndim == 3 else proc_frame
    all_candidates = []

    # Pass 1: Standard Bilateral + Sobel Vertical Edges + Otsu
    smoothed = cv2.bilateralFilter(gray, d=7, sigmaColor=75, sigmaSpace=75)  # d=7 is ~40% faster than d=9
    sobelx = cv2.Sobel(smoothed, cv2.CV_16S, 1, 0, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    _, thresh1 = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    kw1 = max(int(w * 0.025), 15)
    kh1 = max(int(h * 0.010), 3)
    candidates1 = _find_candidates_from_binary(thresh1, gray, frame_area, kw1, kh1)
    all_candidates.extend(candidates1)

    # Pass 2: CLAHE + BlackHat morphology (for low-light/garage images)
    clahe_obj = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe = clahe_obj.apply(gray)
    blackhat = cv2.morphologyEx(clahe, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5)))
    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    minVal, maxVal = np.min(gradX), np.max(gradX)
    if maxVal > minVal:
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
        gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
        _, thresh2 = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        candidates2 = _find_candidates_from_binary(thresh2, gray, frame_area, 13, 5)
        all_candidates.extend(candidates2)

    result = {
        "found": False,
        "box": None,
        "axis_aligned_bbox": None,
        "score": 0.0,
        "edge_map": thresh1 if debug else None,
        "detector_type": "classical_cv_fallback"
    }

    if all_candidates:
        all_candidates.sort(key=lambda t: t[0], reverse=True)
        best_score, rect, bbox, best_closed = all_candidates[0]
        box_pts = cv2.boxPoints(rect)
        ordered_pts = order_corners(box_pts)
        result.update({
            "found": True,
            "box": ordered_pts,
            "axis_aligned_bbox": bbox,
            "score": round(float(best_score), 4),
            "edge_map": best_closed if debug else None
        })
        logger.debug("Plate detected via classical CV (score=%.3f)", best_score)
    else:
        logger.debug("No plate candidates found in frame")

    return result


def crop_with_margin(frame: np.ndarray, bbox: tuple[int, int, int, int], margin_ratio: float = 0.12) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop a bounding box with an optional margin, clamped to frame boundaries.
    Returns a guaranteed non-empty crop (minimum 1x1 pixel)."""
    x, y, w, h = bbox
    mx = int(w * margin_ratio)
    my = int(h * margin_ratio)
    H, W = frame.shape[:2]
    
    x0 = max(0, x - mx)
    y0 = max(0, y - my)
    x1 = min(W, x + w + mx)
    y1 = min(H, y + h + my)

    # B10: Guard against empty crops
    if x1 <= x0:
        x0 = max(0, x)
        x1 = min(W, x + max(1, w))
    if y1 <= y0:
        y0 = max(0, y)
        y1 = min(H, y + max(1, h))

    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        logger.warning("crop_with_margin produced empty crop for bbox=%s, returning 1x1 fallback", bbox)
        crop = frame[0:1, 0:1]

    return crop, (x0, y0, x1, y1)
