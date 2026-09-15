"""
ocr_engine.py
=============
OCR engine wrapper for forensic license plate recognition.
Supports:
1. RapidOCR (ONNX Runtime-based text detection & recognition - offline, robust).
2. Pytesseract (optional fallback if system Tesseract is installed).
3. Multi-variant candidate ranking: scores candidates across multiple enhanced
   image variants and selects the optimal prediction with full transparency.
4. Consensus voting: boosts confidence when multiple variants agree on the same text.
5. Indian plate format validation for confidence boosting.
"""
from __future__ import annotations

import logging
import re
from collections import Counter

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Precompiled regex for plate text cleaning (avoids recompilation on every call)
_PLATE_CLEAN_RE = re.compile(r"[^A-Za-z0-9]")

# Lazy-initialized singletons (Q4: deferred from module import time)
_RAPID_OCR = None
_RAPID_OCR_INIT_ATTEMPTED = False

_HAS_TESSERACT = None  # None = not checked yet

# Cached plate_validator functions (avoids per-variant import overhead)
_PLATE_VALIDATOR_LOADED = False
_validate_plate_format = None
_boost_confidence = None


def _load_plate_validator():
    """Lazily loads plate_validator functions once, caching the result."""
    global _PLATE_VALIDATOR_LOADED, _validate_plate_format, _boost_confidence
    if _PLATE_VALIDATOR_LOADED:
        return
    _PLATE_VALIDATOR_LOADED = True
    try:
        from plate_validator import boost_confidence, validate_plate_format
        _validate_plate_format = validate_plate_format
        _boost_confidence = boost_confidence
    except ImportError:
        pass


def _get_rapid_ocr():
    """Lazily initializes RapidOCR on first use instead of at import time."""
    global _RAPID_OCR, _RAPID_OCR_INIT_ATTEMPTED
    if _RAPID_OCR_INIT_ATTEMPTED:
        return _RAPID_OCR
    _RAPID_OCR_INIT_ATTEMPTED = True
    try:
        from rapidocr_onnxruntime import RapidOCR
        _RAPID_OCR = RapidOCR()
        logger.debug("RapidOCR initialized successfully")
    except Exception as e:
        _RAPID_OCR = None
        logger.info("RapidOCR not available: %s", e)
    return _RAPID_OCR


def _has_tesseract() -> bool:
    """Lazily checks for pytesseract availability."""
    global _HAS_TESSERACT
    if _HAS_TESSERACT is not None:
        return _HAS_TESSERACT
    try:
        import pytesseract  # noqa: F401
        _HAS_TESSERACT = True
    except ImportError:
        _HAS_TESSERACT = False
        logger.debug("pytesseract not available")
    return _HAS_TESSERACT


def clean_plate_text(text: str) -> str:
    """Sanitize plate text by removing non-alphanumeric characters,
    stripping whitespace, and converting to uppercase."""
    if not text:
        return ""
    # Remove symbols, dashes, spaces (uses precompiled regex)
    cleaned = _PLATE_CLEAN_RE.sub("", text).upper()
    return cleaned


def ocr_single_image(img: np.ndarray) -> tuple[str, float]:
    """Run OCR on a single preprocessed crop.
    Returns (cleaned_text, confidence)."""
    if img is None or img.size == 0:
        return "", 0.0
        
    # Ensure 3-channel BGR for RapidOCR
    if img.ndim == 2:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = img

    best_text = ""
    best_conf = 0.0

    # 1. Try RapidOCR
    rapid_ocr = _get_rapid_ocr()
    if rapid_ocr is not None:
        try:
            results, elapse = rapid_ocr(img_bgr)
            if not results:
                # Fallback: direct text recognition without DBNet detector for crops
                results, elapse = rapid_ocr(img_bgr, use_det=False)
            if results:
                texts = []
                confs = []
                for item in results:
                    if len(item) == 2:
                        text, score = item
                    else:
                        _, text, score = item
                    cleaned = clean_plate_text(text)
                    if cleaned:
                        texts.append(cleaned)
                        confs.append(float(score))
                if texts:
                    joined = "".join(texts)
                    mean_conf = sum(confs) / len(confs)
                    if len(joined) >= len(best_text) and mean_conf > best_conf:
                        best_text = joined
                        best_conf = mean_conf
        except Exception as e:
            # B2: Log the error instead of silently swallowing
            logger.warning("RapidOCR failed on image (shape=%s): %s", img_bgr.shape, e)

    # 2. Try Pytesseract as secondary / fallback if available
    # Skip Tesseract sweep when RapidOCR already returned high confidence
    if (not best_text or best_conf < 0.7) and _has_tesseract():
        import pytesseract
        from pytesseract import Output
        for psm in (7, 8, 6):
            try:
                config = f"--oem 3 --psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                data = pytesseract.image_to_data(img, config=config, output_type=Output.DICT)
                texts, confs = [], []
                for t, c in zip(data["text"], data["conf"]):
                    t_clean = clean_plate_text(t)
                    c_val = float(c)
                    if t_clean and c_val > 0:
                        texts.append(t_clean)
                        confs.append(c_val / 100.0)
                if texts:
                    joined = "".join(texts)
                    avg_c = sum(confs) / len(confs)
                    if avg_c > best_conf:
                        best_text = joined
                        best_conf = avg_c
            except Exception as e:
                logger.debug("Tesseract PSM=%d failed: %s", psm, e)
                continue

    return best_text, round(float(best_conf), 4)


def _compute_consensus_bonus(candidates: list[dict]) -> dict[str, float]:
    """Computes consensus bonus for candidates that multiple variants agree on.
    If ≥3 variants produce the same text, that text gets a confidence boost."""
    if len(candidates) < 3:
        return {}

    text_counts = Counter(c["text"] for c in candidates)
    bonuses = {}
    for text, count in text_counts.items():
        if count >= 3:
            # Significant consensus: strong boost
            bonuses[text] = 0.08
        elif count >= 2:
            # Mild consensus: small boost
            bonuses[text] = 0.03
    return bonuses


def read_plate_variants(variants: dict[str, np.ndarray]) -> dict:
    """Evaluates an ensemble of enhanced image variants (e.g. contrast,
    deblurred, super_resolved, binarized) and selects the best prediction.
    
    Includes consensus voting and Indian plate format validation for
    confidence boosting.
    
    Returns a dict with:
      - text: final predicted text
      - confidence: confidence score [0.0 - 1.0]
      - winning_variant: label of the variant that produced the best reading
      - all_candidates: list of all variant outputs sorted by confidence
      - consensus_count: how many variants agreed on the winning text
    """
    candidates = []
    
    for variant_name, img in variants.items():
        if img is None or img.size == 0:
            continue
            
        text, conf = ocr_single_image(img)
        if text:
            # Score balances confidence and reasonable plate length (typically 6-10 characters for Indian plates)
            len_bonus = min(len(text) / 10.0, 1.0) * 0.1
            composite_score = conf + len_bonus

            # Apply Indian plate format validation boost (uses cached import)
            _load_plate_validator()
            if _validate_plate_format is not None:
                validation = _validate_plate_format(text)
                if validation["is_valid"]:
                    composite_score += validation["format_score"] * 0.05
                    logger.debug("Variant '%s': plate '%s' matches format (state=%s)",
                                 variant_name, text, validation.get("state_name", "?"))

            candidates.append({
                "variant": variant_name,
                "text": text,
                "confidence": conf,
                "score": round(composite_score, 4)
            })

            # Early exit: if we have a high-confidence valid-format result, skip remaining variants
            if conf > 0.92 and _validate_plate_format is not None:
                validation = _validate_plate_format(text)
                if validation["is_valid"] and validation["format_score"] >= 0.7:
                    logger.debug("Early exit: high-confidence valid plate '%s' (conf=%.3f)", text, conf)
                    break

    if not candidates:
        return {
            "text": "",
            "confidence": 0.0,
            "winning_variant": "none",
            "all_candidates": [],
            "consensus_count": 0,
        }

    # Apply consensus voting bonus
    consensus_bonuses = _compute_consensus_bonus(candidates)
    for cand in candidates:
        bonus = consensus_bonuses.get(cand["text"], 0.0)
        if bonus > 0:
            cand["score"] = round(cand["score"] + bonus, 4)
            cand["consensus_bonus"] = bonus

    # Rank by composite score
    candidates.sort(key=lambda c: c["score"], reverse=True)
    best = candidates[0]

    # Count how many variants agreed on the winning text
    consensus_count = sum(1 for c in candidates if c["text"] == best["text"])

    logger.debug("OCR ensemble: winner='%s' (conf=%.3f, variant=%s, consensus=%d/%d)",
                 best["text"], best["confidence"], best["variant"], consensus_count, len(candidates))

    return {
        "text": best["text"],
        "confidence": best["confidence"],
        "winning_variant": best["variant"],
        "all_candidates": candidates,
        "consensus_count": consensus_count,
    }
