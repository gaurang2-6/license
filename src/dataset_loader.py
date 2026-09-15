"""
dataset_loader.py
=================
Loader and benchmark generator for the Indian Vehicle Dataset:
- Parses Pascal VOC XML files across State-wise_OLX, google_images, and video_images.
- Extracts ground-truth bounding boxes and license plate strings.
- Implements forensic degradation simulators to evaluate recovery on real Indian vehicles.
- Synthesizes shaky video sequences from real vehicle images for video stabilization testing.
"""
from __future__ import annotations

import glob
import logging
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DATASET_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "indian_vehicle_dataset")


def parse_voc_xml(xml_path: str) -> list[dict]:
    """Extracts ground-truth objects from a Pascal VOC XML annotation."""
    records = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            name_el = obj.find("name")
            name = name_el.text.strip() if name_el is not None and name_el.text else ""
            bndbox = obj.find("bndbox")
            if bndbox is not None:
                xmin = int(float(bndbox.find("xmin").text))
                ymin = int(float(bndbox.find("ymin").text))
                xmax = int(float(bndbox.find("xmax").text))
                ymax = int(float(bndbox.find("ymax").text))
                records.append({
                    "plate_text": name,
                    "bbox": (xmin, ymin, xmax - xmin, ymax - ymin),
                    "box_coords": (xmin, ymin, xmax, ymax)
                })
    except ET.ParseError as e:
        # B7: Log XML parse errors instead of silently swallowing
        logger.warning("Failed to parse VOC XML '%s': %s", xml_path, e)
    except Exception as e:
        logger.warning("Unexpected error reading XML '%s': %s", xml_path, e)
    return records


def find_dataset_pairs(dataset_dir: str = DATASET_ROOT) -> list[dict]:
    """Finds all valid image + XML annotation pairs in the dataset."""
    pairs = []
    valid_img_exts = {".jpg", ".jpeg", ".png"}

    if not os.path.isdir(dataset_dir):
        logger.warning("Dataset directory not found: %s", dataset_dir)
        return pairs

    for root, _, files in os.walk(dataset_dir):
        xml_map = {}
        img_map = {}
        for f in files:
            base, ext = os.path.splitext(f)
            ext_lower = ext.lower()
            if ext_lower == ".xml":
                # Handle extended names like xxx.jpg.xml
                clean_base = base.replace(".jpg", "").replace(".png", "")
                xml_map[clean_base] = os.path.join(root, f)
            elif ext_lower in valid_img_exts:
                img_map[base] = os.path.join(root, f)

        for base_name, img_path in img_map.items():
            if base_name in xml_map:
                xml_path = xml_map[base_name]
                annotations = parse_voc_xml(xml_path)
                if annotations:
                    pairs.append({
                        "id": base_name,
                        "image_path": img_path,
                        "xml_path": xml_path,
                        "annotations": annotations,
                        "primary_plate": annotations[0]["plate_text"],
                        "primary_bbox": annotations[0]["bbox"],
                    })

    logger.debug("Found %d labeled pairs in %s", len(pairs), dataset_dir)
    return pairs


# ---------------------------------------------------------------------------
# Forensic Degradation Simulators
# ---------------------------------------------------------------------------

def degrade_motion_blur(img: np.ndarray, length: int = 24, angle: float = 6.0) -> np.ndarray:
    """Simulates linear vehicle or camera motion blur."""
    k = np.zeros((length, length), dtype=np.float32)
    k[length // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((length / 2.0, length / 2.0), angle, 1.0)
    k = cv2.warpAffine(k, M, (length, length))
    s = k.sum()
    k = k / s if s > 0 else k
    return cv2.filter2D(img, -1, k)


def degrade_defocus_blur(img: np.ndarray, ksize: int = 15) -> np.ndarray:
    """Simulates camera lens defocus blur."""
    k = max(3, ksize | 1)
    return cv2.GaussianBlur(img, (k, k), 0)


def degrade_low_res(img: np.ndarray, scale: float = 0.25) -> np.ndarray:
    """Simulates long-distance CCTV capture with low pixel density."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(8, int(w * scale)), max(8, int(h * scale))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def degrade_noise(img: np.ndarray, sigma: float = 22.0) -> np.ndarray:
    """Simulates low-light sensor noise."""
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def degrade_exposure(img: np.ndarray, gamma: float = 2.4, dark: bool = True) -> np.ndarray:
    """Simulates severe over-exposure (glare) or under-exposure (night)."""
    inv = 1.0 / gamma if dark else gamma
    table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, table)


def degrade_jpeg_compression(img: np.ndarray, quality: int = 15) -> np.ndarray:
    """Simulates low-bitrate CCTV compression artifacts."""
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def create_jitter_video(clean_frame: np.ndarray, degrade_fn, out_path: str, n_frames: int = 12) -> str:
    """Generates a synthetic jittery video clip from an image to test stabilization."""
    h, w = clean_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    vw = cv2.VideoWriter(out_path, fourcc, 10.0, (w, h))
    rng = np.random.RandomState(42)

    for _ in range(n_frames):
        dx, dy = rng.uniform(-7.0, 7.0, size=2)
        ang = rng.uniform(-1.8, 1.8)
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, 1.0)
        M[0, 2] += dx
        M[1, 2] += dy
        jittered = cv2.warpAffine(clean_frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        degraded = degrade_fn(jittered) if degrade_fn is not None else jittered
        vw.write(degraded)

    vw.release()
    return out_path
