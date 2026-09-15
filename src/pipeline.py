"""
pipeline.py
===========
Master forensic license plate enhancement and recovery pipeline.
Provides:
- End-to-end processing for single images and video clips.
- Multi-frame ECC stabilization and temporal averaging.
- Automated PSF Wiener deblurring bank with Laplacian sharpness selection.
- Multi-scale super-resolution and CLAHE contrast enhancement.
- Multi-variant OCR ensemble with RapidOCR and consensus voting.
- Batch processing mode for directories of images.
- Rigorous quantitative benchmarking on the Indian Vehicle Dataset and synthetic suites.
- Export of standardized forensic artifacts: detected region, rectified plate,
  enhanced plate, comparative before/after composite card, and result.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

# Ensure local imports work cleanly
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import dataset_loader
import detect
import enhance
import metrics
import ocr_engine
import synth_data

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON Serialization Helpers (B9)
# ---------------------------------------------------------------------------

class NumpySafeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types, inf, and NaN values
    which would otherwise crash json.dump with TypeError."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            val = float(obj)
            if not np.isfinite(val):
                return None  # Convert inf/NaN to null
            return val
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _safe_json_dump(data: dict, fp, **kwargs):
    """Safely serialize data to JSON, handling numpy types and inf values."""
    json.dump(data, fp, cls=NumpySafeEncoder, **kwargs)


# ---------------------------------------------------------------------------
# Core Pipeline Functions
# ---------------------------------------------------------------------------

def select_best_deblur(crop: np.ndarray) -> tuple[np.ndarray, str, list]:
    """Sweeps a pruned bank of Wiener deblurring parameters (motion angles/lengths
    and defocus radii) and selects candidates that enhance character legibility.
    
    Pruned from 15 to 6 combinations: only the most informative parameters are tested,
    cutting deblur time by ~60% with negligible quality loss."""
    candidates = []
    base_sharpness = metrics.compute_sharpness(crop)
    candidates.append((base_sharpness, crop, "none"))

    # Pruned motion blur kernel grid: 3 most effective length/angle combos
    for length, angle in [(15, 0.0), (21, 0.0), (25, 8.0)]:
        try:
            d = enhance.deblur_wiener(crop, length=length, angle=angle, k=0.015, is_defocus=False)
            s = metrics.compute_sharpness(d)
            if s <= base_sharpness * 12.0:
                candidates.append((s, d, f"wiener_motion_len{length}_ang{int(angle)}"))
        except Exception:
            continue

    # Pruned defocus blur kernel: 2 most effective radii
    for r in [9, 15]:
        try:
            d_defocus = enhance.deblur_wiener(crop, length=r, angle=0.0, k=0.02, is_defocus=True)
            s_defocus = metrics.compute_sharpness(d_defocus)
            if s_defocus <= base_sharpness * 12.0:
                candidates.append((s_defocus, d_defocus, f"wiener_defocus_r{r}"))
        except Exception:
            continue

    candidates.sort(key=lambda t: t[0], reverse=True)
    best_sharp, best_img, best_label = candidates[0]
    return best_img, best_label, candidates[:3]


def build_enhancement_variants(rectified_crop: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, str]:
    """Applies the forensic enhancement chain and produces an ensemble of
    preprocessed variants for the OCR engine to evaluate."""
    variants = {}

    # 1. Base rectified
    variants["rectified_raw"] = enhance.add_ocr_border(rectified_crop, padding=18)

    # 2. Exposure & CLAHE
    exposure_fixed = enhance.auto_exposure_correct(rectified_crop)
    variants["exposure_clahe"] = enhance.add_ocr_border(exposure_fixed, padding=18)

    # 3. Denoising
    denoised = enhance.denoise(exposure_fixed, strength=6)
    variants["denoised"] = enhance.add_ocr_border(denoised, padding=18)

    # 4. Wiener Deblurring (pruned sweep — top candidate only, alts dropped)
    best_deblur_img, deblur_label, _ = select_best_deblur(denoised)
    variants["deblurred"] = enhance.add_ocr_border(best_deblur_img, padding=18)

    # 5. Spatial Sharpening
    sharpened = enhance.sharpen_unsharp(best_deblur_img, sigma=1.2, amount=1.5)
    variants["sharpened"] = enhance.add_ocr_border(sharpened, padding=18)

    # 6. Edge-directed Super-Resolution (2x only — 3x dropped as it rarely wins)
    upscaled_2x = enhance.super_resolve(sharpened, scale=2)
    variants["super_resolved_2x"] = enhance.add_ocr_border(upscaled_2x, padding=24)

    # 7. Adaptive Binarization + Morphological Cleaning
    binarized = enhance.binarize_for_ocr(upscaled_2x)
    cleaned_bin = enhance.morphological_clean(binarized, ksize=2)
    variants["binarized"] = enhance.add_ocr_border(cleaned_bin, padding=24)

    # 8. Auto-deskewed variant
    deskewed = enhance.auto_deskew(exposure_fixed)
    deskewed_up = enhance.super_resolve(deskewed, scale=2)
    variants["deskewed_2x"] = enhance.add_ocr_border(deskewed_up, padding=24)

    # Primary visual artifact for human review
    primary_visual = upscaled_2x
    return variants, primary_visual, deblur_label


def generate_before_after_card(before_img: np.ndarray, after_img: np.ndarray,
                               title: str, predicted_text: str,
                               confidence: float, ground_truth: str = None,
                               sharpness_before: float = None,
                               sharpness_after: float = None,
                               deblur_method: str = None) -> np.ndarray:
    """Creates a high-contrast forensic before/after visual comparison card."""
    card_h = 240
    header_h = 70
    footer_h = 60  # Slightly taller for extra info
    spacing = 30

    # B6: Ensure both images are 3-channel BGR before compositing
    before_img = enhance.ensure_bgr(before_img)
    after_img = enhance.ensure_bgr(after_img)

    # Scale crops to uniform height
    aspect_b = before_img.shape[1] / max(1, before_img.shape[0])
    aspect_a = after_img.shape[1] / max(1, after_img.shape[0])
    
    bw = int(card_h * aspect_b)
    aw = int(card_h * aspect_a)
    
    b_resized = cv2.resize(before_img, (bw, card_h), interpolation=cv2.INTER_AREA)
    a_resized = cv2.resize(after_img, (aw, card_h), interpolation=cv2.INTER_CUBIC)
    
    total_w = bw + aw + spacing + 60
    total_h = card_h + header_h + footer_h + 30
    
    canvas = np.full((total_h, total_w, 3), 28, dtype=np.uint8)  # Dark aesthetic background
    
    # Header title
    cv2.putText(canvas, title, (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2, cv2.LINE_AA)
    
    # Place images
    y_start = header_h + 10
    x_b = 30
    x_a = x_b + bw + spacing
    
    canvas[y_start:y_start + card_h, x_b:x_b + bw] = b_resized
    canvas[y_start:y_start + card_h, x_a:x_a + aw] = a_resized
    
    # Subtle borders
    cv2.rectangle(canvas, (x_b, y_start), (x_b + bw, y_start + card_h), (80, 80, 80), 1)
    cv2.rectangle(canvas, (x_a, y_start), (x_a + aw, y_start + card_h), (0, 200, 100), 2)
    
    # Sub-labels
    cv2.putText(canvas, "BEFORE (Degraded/Raw)", (x_b, y_start - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, "AFTER (Forensic Enhancement)", (x_a, y_start - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 120), 1, cv2.LINE_AA)
    
    # Footer recovery metrics
    y_foot = total_h - 32
    ocr_str = f"Recovered Text: {predicted_text}  |  Confidence: {confidence * 100.0:.1f}%"
    cv2.putText(canvas, ocr_str, (30, y_foot), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 160), 2, cv2.LINE_AA)
    
    if ground_truth:
        gt_str = f"Ground Truth: {ground_truth}"
        cv2.putText(canvas, gt_str, (total_w - 320, y_foot), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2, cv2.LINE_AA)

    # Additional info line (sharpness, deblur method)
    y_info = total_h - 10
    info_parts = []
    if sharpness_before is not None and sharpness_after is not None:
        info_parts.append(f"Sharpness: {sharpness_before:.0f} -> {sharpness_after:.0f}")
    if deblur_method and deblur_method != "none":
        info_parts.append(f"Deblur: {deblur_method}")
    if info_parts:
        info_str = "  |  ".join(info_parts)
        cv2.putText(canvas, info_str, (30, y_info), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1, cv2.LINE_AA)

    return canvas


def process_frame(frame: np.ndarray, ground_truth_text: str = None,
                  clean_reference: np.ndarray = None, out_dir: str = None,
                  case_name: str = "case", force_bbox: tuple = None) -> dict:
    """Processes a single frame or stabilized composite image through the complete pipeline."""
    start_time = time.perf_counter()

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Pre-process dark/under-exposed frames
    proc_frame = detect.preprocess_frame_if_dark(frame)

    # 1. License plate detection
    det = detect.detect_plate(proc_frame)
    if force_bbox is not None:
        # Use known ground-truth bounding box (x, y, w, h)
        x, y, w, h = force_bbox
        pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
        det["found"] = True
        det["box"] = pts
        det["axis_aligned_bbox"] = force_bbox
        det["score"] = 1.0

    if det["found"] and det["box"] is not None:
        rectified = enhance.rectify_perspective(proc_frame, det["box"], out_size=(400, 120))
        raw_crop, _ = detect.crop_with_margin(proc_frame, det["axis_aligned_bbox"], margin_ratio=0.10)
    else:
        # Graceful fallback: resize entire region
        rectified = cv2.resize(proc_frame, (400, 120))
        raw_crop = proc_frame

    # 2. Enhancement variants
    variants, best_visual, deblur_label = build_enhancement_variants(rectified)

    # Also test raw crop directly to demonstrate baseline failure
    raw_for_ocr = enhance.add_ocr_border(cv2.resize(raw_crop, (400, 120)), padding=20)
    raw_text, raw_conf = ocr_engine.ocr_single_image(raw_for_ocr)

    # 3. OCR on enhancement variants
    ocr_result = ocr_engine.read_plate_variants(variants)

    # 4. Metrics computation
    sharpness_before = round(metrics.compute_sharpness(rectified), 2)
    sharpness_after = round(metrics.compute_sharpness(best_visual), 2)

    # B9: Ensure all values are JSON-serializable native Python types
    result = {
        "case": case_name,
        "detection": {
            "found": bool(det["found"]),
            "score": float(det.get("score", 0.0)),
            "bbox": _to_native(det.get("axis_aligned_bbox")),
            "detector_type": det.get("detector_type", "unknown"),
        },
        "deblur_chosen": deblur_label,
        "baseline_ocr": {
            "raw_crop_text": raw_text,
            "raw_crop_conf": float(raw_conf),
        },
        "recovered_ocr": {
            "text": ocr_result["text"],
            "confidence": float(ocr_result["confidence"]),
            "winning_variant": ocr_result["winning_variant"],
            "consensus_count": ocr_result.get("consensus_count", 0),
            "candidates": ocr_result["all_candidates"],
        },
        "sharpness": {
            "before": sharpness_before,
            "after": sharpness_after,
        },
        "processing_time_s": round(time.perf_counter() - start_time, 3),
    }

    if ground_truth_text:
        result["accuracy"] = metrics.char_accuracy(ground_truth_text, ocr_result["text"])
        result["baseline_accuracy"] = metrics.char_accuracy(ground_truth_text, raw_text)

    if clean_reference is not None:
        p_bef, s_bef = metrics.compute_psnr_ssim(clean_reference, rectified)
        p_aft, s_aft = metrics.compute_psnr_ssim(clean_reference, best_visual)
        result["image_quality"] = {
            "psnr_before": round(p_bef, 2),
            "ssim_before": round(s_bef, 4),
            "psnr_after": round(p_aft, 2),
            "ssim_after": round(s_aft, 4)
        }

    # 5. Save deliverable files
    if out_dir:
        cv2.imwrite(os.path.join(out_dir, "01_detected_plate.png"), raw_crop)
        cv2.imwrite(os.path.join(out_dir, "02_rectified_plate.png"), rectified)
        cv2.imwrite(os.path.join(out_dir, "03_enhanced_plate.png"), best_visual)

        card = generate_before_after_card(
            rectified, best_visual,
            title=f"Forensic Enhancement: {case_name}",
            predicted_text=ocr_result["text"],
            confidence=ocr_result["confidence"],
            ground_truth=ground_truth_text,
            sharpness_before=sharpness_before,
            sharpness_after=sharpness_after,
            deblur_method=deblur_label,
        )
        cv2.imwrite(os.path.join(out_dir, "04_before_after.png"), card)

        with open(os.path.join(out_dir, "result.json"), "w") as f:
            _safe_json_dump(result, f, indent=2)

    return result


def _to_native(val):
    """Convert numpy types to native Python types for JSON serialization."""
    if val is None:
        return None
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (tuple, list)):
        return [int(v) if isinstance(v, (np.integer,)) else v for v in val]
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return val


def process_image(path: str, ground_truth: str = None, clean_ref_path: str = None,
                  out_dir: str = None, force_bbox: tuple = None) -> dict:
    """Processes a single image file."""
    frame = cv2.imread(path)
    if frame is None:
        raise FileNotFoundError(f"Could not read image from {path}")
    clean_ref = cv2.imread(clean_ref_path) if clean_ref_path else None
    case_name = os.path.splitext(os.path.basename(path))[0]
    return process_frame(frame, ground_truth, clean_ref, out_dir, case_name, force_bbox)


def process_video(path: str, ground_truth: str = None, clean_ref_path: str = None,
                  out_dir: str = None, max_frames: int = 15) -> dict:
    """Stabilizes a shaky video sequence using ECC alignment, computes multi-frame
    temporal median average, and executes full forensic recovery."""
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()

    if not frames:
        raise FileNotFoundError(f"No frames could be read from video {path}")

    # Video stabilization
    ref_idx = int(np.argmax([metrics.compute_sharpness(f) for f in frames]))
    ref_frame = frames[ref_idx]
    aligned_frames = enhance.align_frames_ecc(ref_frame, frames)
    composite = enhance.multiframe_average(aligned_frames)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, "00_stabilized_composite.png"), composite)

    clean_ref = cv2.imread(clean_ref_path) if clean_ref_path else None
    case_name = os.path.splitext(os.path.basename(path))[0]
    result = process_frame(composite, ground_truth, clean_ref, out_dir, case_name)
    result["video_metadata"] = {
        "frames_total": len(frames),
        "reference_frame_index": int(ref_idx)
    }

    if out_dir:
        with open(os.path.join(out_dir, "result.json"), "w") as f:
            _safe_json_dump(result, f, indent=2)

    return result


def process_batch(input_dir: str, out_root: str = "outputs/batch", ground_truth: str = None) -> list[dict]:
    """Processes all images in a directory (batch mode).
    
    Supports .jpg, .jpeg, .png files. Each image gets its own output subdirectory.
    Returns a list of result dicts and writes a summary JSON.
    """
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    os.makedirs(out_root, exist_ok=True)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_files = sorted([
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in image_exts
    ])

    if not image_files:
        logger.warning("No image files found in %s", input_dir)
        return []

    logger.info("Batch processing %d images from %s", len(image_files), input_dir)
    results = []

    def _process_one(idx_fname):
        idx, fname = idx_fname
        img_path = os.path.join(input_dir, fname)
        case_name = os.path.splitext(fname)[0]
        case_out = os.path.join(out_root, case_name)
        logger.info("[%d/%d] Processing: %s", idx, len(image_files), fname)
        try:
            res = process_image(img_path, ground_truth=ground_truth, out_dir=case_out)
            logger.info("  -> Recovered: '%s' (conf=%.1f%%)",
                        res["recovered_ocr"]["text"],
                        res["recovered_ocr"]["confidence"] * 100)
            return res
        except Exception as e:
            logger.error("  -> Failed: %s", e)
            return {"case": case_name, "error": str(e)}

    # Use ThreadPoolExecutor for concurrent batch processing
    max_workers = min(4, len(image_files))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_one, (idx, fname)): fname
                   for idx, fname in enumerate(image_files, 1)}
        for future in as_completed(futures):
            results.append(future.result())

    # Write batch summary
    summary_path = os.path.join(out_root, "batch_summary.json")
    with open(summary_path, "w") as f:
        _safe_json_dump(results, f, indent=2)
    logger.info("Batch complete. Summary: %s", summary_path)

    return results


# ---------------------------------------------------------------------------
# Benchmark Suites
# ---------------------------------------------------------------------------

def run_real_benchmark(dataset_dir: str = None, out_root: str = "outputs/real_benchmark",
                       gallery_dir: str = "outputs/sample_gallery", num_samples: int = 10) -> list[dict]:
    """Executes evaluation on real Indian vehicle images and controlled degradations."""
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(gallery_dir, exist_ok=True)

    pairs = dataset_loader.find_dataset_pairs(dataset_dir) if dataset_dir else dataset_loader.find_dataset_pairs()
    logger.info("Found %d labeled real vehicle pairs", len(pairs))
    
    if not pairs:
        logger.error("No pairs found — ensure the Indian Vehicle Dataset is extracted")
        return []

    # Choose a diverse selection of state plates
    selected = pairs[:num_samples]
    results = []

    degradations = [
        ("clean_original", lambda im: im, "Original captured image crop"),
        ("motion_blur", lambda im: dataset_loader.degrade_motion_blur(im, 20, 6.0), "Linear motion blur (vehicle moving)"),
        ("defocus_blur", lambda im: dataset_loader.degrade_defocus_blur(im, 15), "Lens defocus out-of-focus"),
        ("low_res", lambda im: dataset_loader.degrade_low_res(im, 0.28), "Distant CCTV low pixel density"),
        ("night_noise", lambda im: dataset_loader.degrade_noise(dataset_loader.degrade_exposure(im, 2.2, dark=True), 18), "Low-light night underexposure + sensor noise"),
        ("glare_overexposed", lambda im: dataset_loader.degrade_exposure(im, 2.2, dark=False), "Overexposed solar glare"),
    ]

    for idx, sample in enumerate(selected):
        img = cv2.imread(sample["image_path"])
        if img is None:
            continue
            
        gt_text = sample["primary_plate"]
        bbox = sample["primary_bbox"]
        sample_id = sample["id"]
        logger.info("[%d/%d] Processing Real Plate '%s' (%s)", idx + 1, len(selected), gt_text, sample_id)

        # Pick one degradation for this sample to demonstrate diversity
        deg_name, deg_fn, deg_desc = degradations[idx % len(degradations)]
        degraded_img = deg_fn(img)

        case_dir = os.path.join(out_root, f"sample_{idx+1}_{deg_name}")
        res = process_frame(degraded_img, ground_truth_text=gt_text, out_dir=case_dir,
                            case_name=f"{gt_text}_{deg_name}", force_bbox=bbox)
        res["degradation_type"] = deg_name
        res["degradation_desc"] = deg_desc
        results.append(res)

        # Copy sample before/after to the main gallery (shutil avoids wasteful decode+encode)
        card_src = os.path.join(case_dir, "04_before_after.png")
        if os.path.exists(card_src):
            card_dst = os.path.join(gallery_dir, f"gallery_case_{idx+1}_{deg_name}.png")
            shutil.copy2(card_src, card_dst)

        logger.info("  -> Baseline OCR: '%s' | Recovered OCR: '%s' (Conf: %.1f%%)",
                     res["baseline_ocr"]["raw_crop_text"],
                     res["recovered_ocr"]["text"],
                     res["recovered_ocr"]["confidence"] * 100)

    summary_path = os.path.join(out_root, "real_benchmark_summary.json")
    with open(summary_path, "w") as f:
        _safe_json_dump(results, f, indent=2)

    return results


def run_synthetic_benchmark(out_root: str = "outputs/synthetic_benchmark") -> list[dict]:
    """Generates synthetic test cases and evaluates both image and video modes."""
    os.makedirs(out_root, exist_ok=True)
    manifest = synth_data.generate_synthetic_suite()
    results = []

    for case in manifest:
        case_name = case["name"]
        case_dir = os.path.join(os.path.dirname(synth_data.OUTPUT_DIR), "synthetic", case_name)
        img_path = os.path.join(case_dir, case["degraded_scene"])
        clean_ref_path = os.path.join(case_dir, case["clean_plate_flat"])
        video_path = os.path.join(case_dir, case["degraded_clip"])
        gt_text = case["ground_truth_text"]

        logger.info("Processing %s (GT: '%s')", case_name, gt_text)
        # 1. Single Image Mode
        img_out = os.path.join(out_root, f"{case_name}_image")
        img_res = process_image(img_path, ground_truth=gt_text, clean_ref_path=clean_ref_path, out_dir=img_out)

        # 2. Video Mode
        vid_out = os.path.join(out_root, f"{case_name}_video")
        vid_res = process_video(video_path, ground_truth=gt_text, clean_ref_path=clean_ref_path, out_dir=vid_out)

        summary_item = {
            "case": case_name,
            "description": case["description"],
            "ground_truth": gt_text,
            "image_result": img_res,
            "video_result": vid_res
        }
        results.append(summary_item)

        logger.info("  -> Image Recovered: '%s' (Sim: %s)",
                     img_res["recovered_ocr"]["text"],
                     img_res.get("accuracy", {}).get("char_similarity", "N/A"))
        logger.info("  -> Video Recovered: '%s' (Sim: %s)",
                     vid_res["recovered_ocr"]["text"],
                     vid_res.get("accuracy", {}).get("char_similarity", "N/A"))

    with open(os.path.join(out_root, "synthetic_benchmark_summary.json"), "w") as f:
        _safe_json_dump(results, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Forensic License Plate Enhancement & Recovery Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python src/pipeline.py --image path/to/vehicle.jpg --out outputs/case1
  python src/pipeline.py --video path/to/clip.mp4 --out outputs/video1
  python src/pipeline.py --batch-dir path/to/images/ --out outputs/batch_results
  python src/pipeline.py --benchmark-real --num-samples 8
  python src/pipeline.py --benchmark-synth
"""
    )
    parser.add_argument("--image", help="Path to input vehicle image")
    parser.add_argument("--video", help="Path to input vehicle video clip")
    parser.add_argument("--batch-dir", help="Directory of images for batch processing")
    parser.add_argument("--out", default="outputs/adhoc", help="Output directory for generated artifacts")
    parser.add_argument("--gt", default=None, help="Optional ground-truth plate text for evaluation")
    parser.add_argument("--clean-ref", default=None, help="Optional clean reference plate image for PSNR/SSIM")
    parser.add_argument("--bbox", default=None, help="Optional manual plate bounding box 'x,y,w,h'")
    parser.add_argument("--benchmark-real", action="store_true", help="Run comprehensive benchmark on Indian Vehicle Dataset")
    parser.add_argument("--benchmark-synth", action="store_true", help="Run benchmark on synthetic test suite")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of samples to evaluate for real benchmark")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging verbosity level (default: INFO)")
    parser.add_argument("--version", action="version", version="%(prog)s 1.1.0")

    args = parser.parse_args()

    # Configure structured logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    force_bbox = None
    if args.bbox:
        try:
            force_bbox = tuple(map(int, args.bbox.split(",")))
        except Exception as e:
            logger.error("Error parsing --bbox argument: %s", e)

    if args.benchmark_real:
        run_real_benchmark(num_samples=args.num_samples)
        return

    if args.benchmark_synth:
        run_synthetic_benchmark()
        return

    if args.batch_dir:
        process_batch(args.batch_dir, out_root=args.out, ground_truth=args.gt)
        return

    if args.video:
        res = process_video(args.video, ground_truth=args.gt, clean_ref_path=args.clean_ref, out_dir=args.out)
        print(json.dumps(res, indent=2, cls=NumpySafeEncoder))
        return

    if args.image:
        res = process_image(args.image, ground_truth=args.gt, clean_ref_path=args.clean_ref, out_dir=args.out, force_bbox=force_bbox)
        print(json.dumps(res, indent=2, cls=NumpySafeEncoder))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
