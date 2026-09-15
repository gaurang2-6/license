"""
synth_data.py
=============
Generates synthetic forensic test cases with known ground-truth text and geometry.
Provides an objective benchmark across 7 canonical degradation scenarios:
1. Horizontal motion blur (vehicle speeding past camera).
2. Defocus blur (incorrect lens focus).
3. Low resolution / pixelation (distant camera capture).
4. Low-light under-exposure with sensor noise.
5. Over-exposure / sunlight glare.
6. Perspective skew (vehicle captured at an oblique angle).
7. Compound degradation (low-res + motion blur + noise + JPEG compression).
"""
from __future__ import annotations

import json
import logging
import os

import cv2
import numpy as np

logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "synthetic")
PLATE_W, PLATE_H = 400, 120
SCENE_W, SCENE_H = 960, 540
FONT = cv2.FONT_HERSHEY_SIMPLEX


def render_clean_plate(text: str) -> np.ndarray:
    """Renders a crisp synthetic license plate with high contrast."""
    plate = np.full((PLATE_H, PLATE_W, 3), 240, dtype=np.uint8)
    # Plate border
    cv2.rectangle(plate, (5, 5), (PLATE_W - 6, PLATE_H - 6), (20, 20, 20), 5)
    # Character text
    (tw, th), _ = cv2.getTextSize(text, FONT, 2.0, 5)
    tx = (PLATE_W - tw) // 2
    ty = (PLATE_H + th) // 2
    cv2.putText(plate, text, (tx, ty), FONT, 2.0, (15, 15, 15), 5, cv2.LINE_AA)
    # Mounting screws
    for cx in (25, PLATE_W - 25):
        cv2.circle(plate, (cx, 22), 4, (80, 80, 80), -1)
        cv2.circle(plate, (cx, PLATE_H - 22), 4, (80, 80, 80), -1)
    return plate


def place_plate_in_scene(plate: np.ndarray, corners_dst=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Warps plate onto a vehicle scene background."""
    scene = np.zeros((SCENE_H, SCENE_W, 3), dtype=np.uint8)
    # Road gradient
    for row in range(SCENE_H):
        val = 60 + int(45 * row / SCENE_H)
        scene[row, :] = (val, val, val)
    # Vehicle body
    cv2.rectangle(scene, (180, 120), (780, 430), (35, 40, 48), -1)
    cv2.rectangle(scene, (180, 120), (780, 430), (15, 18, 22), 4)
    # Headlights
    cv2.circle(scene, (230, 290), 30, (205, 215, 225), -1)
    cv2.circle(scene, (730, 290), 30, (205, 215, 225), -1)

    h, w = plate.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    if corners_dst is None:
        dst = np.float32([[375, 300], [615, 300], [615, 400], [375, 400]])
    else:
        dst = np.float32(corners_dst)

    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(plate, H, (SCENE_W, SCENE_H))
    mask = cv2.warpPerspective(np.full((h, w), 255, dtype=np.uint8), H, (SCENE_W, SCENE_H))
    mask3 = cv2.merge([mask, mask, mask]) > 0
    scene = np.where(mask3, warped, scene)
    return scene.astype(np.uint8), H, dst


# ---------------------------------------------------------------------------
# Degradation implementations
# ---------------------------------------------------------------------------

def _degrade_motion(img):
    k = np.zeros((25, 25), dtype=np.float32)
    k[12, :] = 1.0
    M = cv2.getRotationMatrix2D((12.5, 12.5), 8.0, 1.0)
    k = cv2.warpAffine(k, M, (25, 25))
    k = k / k.sum()
    return cv2.filter2D(img, -1, k)


def _degrade_defocus(img):
    return cv2.GaussianBlur(img, (17, 17), 0)


def _degrade_low_res(img):
    h, w = img.shape[:2]
    small = cv2.resize(img, (int(w * 0.16), int(h * 0.16)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def _degrade_noise_dark(img):
    table = np.array([((i / 255.0) ** (1.0 / 2.2)) * 255 for i in range(256)]).astype(np.uint8)
    dark = cv2.LUT(img, table)
    noise = np.random.normal(0, 20.0, dark.shape).astype(np.float32)
    return np.clip(dark.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _degrade_overexposed(img):
    table = np.array([((i / 255.0) ** 2.4) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, table)


def _degrade_compound(img):
    small = _degrade_low_res(img)
    blurred = _degrade_motion(small)
    noisy = _degrade_noise_dark(blurred)
    ok, enc = cv2.imencode(".jpg", noisy, [cv2.IMWRITE_JPEG_QUALITY, 20])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


CASES = [
    {
        "name": "case1_motion_blur",
        "text": "MH12AB1234",
        "degrade": _degrade_motion,
        "desc": "Horizontal vehicle motion blur during high-speed passage",
    },
    {
        "name": "case2_defocus_blur",
        "text": "DL8CAF5566",
        "degrade": _degrade_defocus,
        "desc": "Severe camera lens defocus blur",
    },
    {
        "name": "case3_low_res_pixelation",
        "text": "KA05MZ7788",
        "degrade": _degrade_low_res,
        "desc": "Severe downsampling and pixelation from distant CCTV",
    },
    {
        "name": "case4_noise_lowlight",
        "text": "TN10CX9012",
        "degrade": _degrade_noise_dark,
        "desc": "Low-light under-exposure combined with high sensor noise",
    },
    {
        "name": "case5_overexposed",
        "text": "GJ01HN3344",
        "degrade": _degrade_overexposed,
        "desc": "Overexposed headlight and daytime solar glare",
    },
    {
        "name": "case6_perspective_skew",
        "text": "RJ14GB6677",
        "degrade": None,  # Only perspective distortion, no pixel-level degradation
        "desc": "Severe side angle perspective distortion (oblique capture)",
    },
    {
        "name": "case7_compound",
        "text": "UP32KD4455",
        "degrade": _degrade_compound,
        "desc": "Compound degradation: low-res + motion blur + noise + JPEG compression",
    },
]


def make_jitter_video(clean_scene: np.ndarray, degrade_fn, out_path: str, n_frames: int = 12):
    """Simulates a short shaky video clip for multi-frame stabilization evaluation."""
    h, w = clean_scene.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, 10.0, (w, h))
    rng = np.random.RandomState(42)

    for _ in range(n_frames):
        dx, dy = rng.uniform(-6.0, 6.0, size=2)
        ang = rng.uniform(-1.5, 1.5)
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, 1.0)
        M[0, 2] += dx
        M[1, 2] += dy
        jittered = cv2.warpAffine(clean_scene, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        frame = degrade_fn(jittered) if degrade_fn is not None else jittered
        vw.write(frame)

    vw.release()


def generate_synthetic_suite(out_dir: str = OUTPUT_DIR) -> list[dict]:
    """Generates all 7 synthetic benchmark cases, scenes, reference plates, and video clips."""
    os.makedirs(out_dir, exist_ok=True)
    manifest = []

    for case in CASES:
        case_dir = os.path.join(out_dir, case["name"])
        os.makedirs(case_dir, exist_ok=True)
        plate = render_clean_plate(case["text"])

        if case["name"] == "case6_perspective_skew":
            skewed_corners = [[380, 300], [520, 275], [520, 425], [380, 400]]
            scene, H, corners = place_plate_in_scene(plate, corners_dst=skewed_corners)
            degraded = scene.copy()
        else:
            scene, H, corners = place_plate_in_scene(plate)
            degraded = case["degrade"](scene)

        cv2.imwrite(os.path.join(case_dir, "clean_scene.png"), scene)
        cv2.imwrite(os.path.join(case_dir, "degraded_scene.png"), degraded)
        cv2.imwrite(os.path.join(case_dir, "clean_plate_flat.png"), plate)

        video_path = os.path.join(case_dir, "degraded_clip.mp4")
        # Use an explicit identity function for None degradation to avoid late-binding issues
        deg_fn = case["degrade"] if case["degrade"] is not None else _identity
        make_jitter_video(scene, deg_fn, video_path)

        meta = {
            "name": case["name"],
            "ground_truth_text": case["text"],
            "description": case["desc"],
            "plate_corners": corners.tolist(),
            "clean_scene": "clean_scene.png",
            "degraded_scene": "degraded_scene.png",
            "clean_plate_flat": "clean_plate_flat.png",
            "degraded_clip": "degraded_clip.mp4",
        }
        with open(os.path.join(case_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        manifest.append(meta)
        logger.info("Generated %s -> Ground Truth: '%s'", case["name"], case["text"])

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def _identity(img: np.ndarray) -> np.ndarray:
    """Identity function — returns the image unchanged.
    Used as explicit no-op degradation for perspective-only test cases."""
    return img


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    generate_synthetic_suite()
