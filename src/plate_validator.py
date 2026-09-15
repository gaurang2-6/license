"""
plate_validator.py
==================
Indian license plate format validation and confidence boosting.
Validates OCR outputs against known Indian registration plate patterns,
providing format scores and state code verification.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# All 37 Indian State/UT RTO codes
INDIAN_STATE_CODES: dict[str, str] = {
    "AN": "Andaman & Nicobar",
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CG": "Chhattisgarh",
    "CH": "Chandigarh",
    "DD": "Dadra & Nagar Haveli and Daman & Diu",
    "DL": "Delhi",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HP": "Himachal Pradesh",
    "HR": "Haryana",
    "JH": "Jharkhand",
    "JK": "Jammu & Kashmir",
    "KA": "Karnataka",
    "KL": "Kerala",
    "LA": "Ladakh",
    "LD": "Lakshadweep",
    "MH": "Maharashtra",
    "ML": "Meghalaya",
    "MN": "Manipur",
    "MP": "Madhya Pradesh",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OD": "Odisha",
    "PB": "Punjab",
    "PY": "Puducherry",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TN": "Tamil Nadu",
    "TR": "Tripura",
    "TS": "Telangana",
    "UK": "Uttarakhand",
    "UP": "Uttar Pradesh",
    "WB": "West Bengal",
}

# Indian plate format patterns (most common formats)
# Format: SS DD XX DDDD (State-code, District, Series, Number)
# Examples: MH12AB1234, DL8CAF5566, KA05MZ7788
PLATE_PATTERNS: list[re.Pattern] = [
    # Standard: 2-letter state + 2-digit district + 1-2 letter series + 4-digit number
    re.compile(r"^([A-Z]{2})(\d{2})([A-Z]{1,3})(\d{4})$"),
    # Variant with single-digit district (Delhi style): DL8CAF5566
    re.compile(r"^([A-Z]{2})(\d{1})([A-Z]{1,3})(\d{4})$"),
    # BH-series (Bharat series, new national format)
    re.compile(r"^(\d{2})(BH)(\d{4})([A-Z]{2})$"),
    # Diplomatic/special
    re.compile(r"^(\d{2,3})([A-Z]{1,2})(\d{4})$"),
]


def validate_plate_format(text: str) -> dict:
    """Validates a plate text string against known Indian plate formats.

    Returns a dict with:
      - is_valid: whether it matches a known format
      - state_code: the detected state code (or None)
      - state_name: full state name (or None)
      - format_score: 0.0 - 1.0 confidence boost based on format validity
      - pattern_matched: which pattern matched (or None)
    """
    if not text or len(text) < 4:
        return {
            "is_valid": False,
            "state_code": None,
            "state_name": None,
            "format_score": 0.0,
            "pattern_matched": None,
        }

    cleaned = text.strip().upper().replace(" ", "").replace("-", "")

    # Check state code
    state_code = cleaned[:2] if len(cleaned) >= 2 else None
    state_name = INDIAN_STATE_CODES.get(state_code) if state_code else None

    # Try each pattern
    for idx, pattern in enumerate(PLATE_PATTERNS):
        match = pattern.match(cleaned)
        if match:
            # Compute format score
            score = 0.7  # Base score for pattern match
            if state_name is not None:
                score += 0.3  # Bonus for valid state code
            logger.debug("Plate '%s' matched pattern %d (state: %s)", cleaned, idx, state_name or "unknown")
            return {
                "is_valid": True,
                "state_code": state_code,
                "state_name": state_name,
                "format_score": round(score, 2),
                "pattern_matched": idx,
            }

    # No pattern matched — partial credit if state code is valid
    partial_score = 0.15 if state_name is not None else 0.0
    return {
        "is_valid": False,
        "state_code": state_code,
        "state_name": state_name,
        "format_score": round(partial_score, 2),
        "pattern_matched": None,
    }


def boost_confidence(text: str, base_confidence: float) -> float:
    """Applies a format-aware confidence boost to OCR predictions.
    Valid Indian plate formats receive a small confidence bump."""
    validation = validate_plate_format(text)
    format_score = validation["format_score"]

    # Blend: 85% original confidence + 15% format validation
    boosted = base_confidence * 0.85 + format_score * 0.15
    return round(min(1.0, boosted), 4)
