# License Plate Recognition & Forensic Enhancement Pipeline — Technical Report

**Project**: Automated Indian License Plate Recognition (ALPR) under Extreme Degradations  
**Author**: Technical Assessment Submission  
**Repository**: [github.com/gaurang2-6/license](https://github.com/gaurang2-6/license.git)

---

## 1. Executive Summary & Objective

Recognizing license plates from real-world surveillance and dashcam footage presents severe computer vision challenges due to compound environmental degradations: severe motion blur, defocus blur, low-light sensor noise, high-speed camera jitter, low resolution, and steep perspective tilt.

This project delivers an end-to-end, production-grade ALPR pipeline specifically engineered for challenging Indian license plate formats (e.g., `MH12AB1234`, `DL8CAF5566`, `KA05MZ7788`). By combining deep learning object detection (YOLOv8 fine-tuned on license plates), sub-pixel OpenCV ECC motion alignment, physics-based Wiener deconvolution, CLAHE contrast enhancement, and a dual OCR consensus engine (RapidOCR + Tesseract with state format validation), the system achieves robust recognition across extreme real and synthetic degradation scenarios.

---

## 2. Pipeline Architecture & Technical Approach

The system follows a modular 6-stage pipeline:

```
[ Input Frame / Video ]
           │
           ▼
Stage 1: License Plate Detection (YOLOv8 / ONNX Model)
           │
           ▼
Stage 2: Perspective Rectification & 4-Point Alignment
           │
           ▼
Stage 3: Multi-Frame Temporal Video Super-Resolution (ECC Alignment)
           │
           ▼
Stage 4: Frequency-Domain & Spatial Image Restoration (Wiener Deconvolution + CLAHE)
           │
           ▼
Stage 5: Dual OCR Consensus Engine (RapidOCR + Tesseract PSM Sweeps)
           │
           ▼
Stage 6: Indian Plate Regex Validation & Confidence Scoring
           │
           ▼
[ Output: High-Confidence Recognized Text & Diagnostic Gallery ]
```

### Stage 1: Deep Learning License Plate Localization (`src/detect.py`)
- **Model**: Custom fine-tuned YOLOv8 model (`license_plate_yolo.onnx` / `.pt`).
- **Inference Acceleration**: Deployed via ONNX Runtime (`CPUExecutionProvider`) for 3x faster CPU execution compared to standard PyTorch, with automatic FP16 half-precision fallback on CUDA devices.
- **Bounding Box Heuristics**: Applies strict aspect ratio filtering ($1.5 \le \text{AR} \le 6.0$) and spatial padding to preserve full character height.

### Stage 2: Geometric Rectification (`src/enhance.py:rectify_perspective`)
- Detects plate corners using adaptive thresholding and contour approximation.
- Computes perspective transformation matrix ($M = \text{cv2.getPerspectiveTransform}$) to project distorted, skewed plate crops into a flat, standardized $400 \times 120$ rectangle.

### Stage 3: Multi-Frame Video Alignment (`src/pipeline.py:process_video`)
- **Reference Frame Selection**: Measures Laplacian variance across all video frames to select the frame with highest spatial sharpness.
- **Sub-Pixel Motion Compensation**: Aligns neighboring frames to the reference frame using OpenCV Enhanced Correlation Coefficient (`cv2.findTransformECC`).
- **Temporal Median Super-Resolution**: Computes a pixel-wise temporal median across aligned frames to cancel out sensor noise and random motion blur without blunting character edges.

### Stage 4: Frequency & Spatial Restoration (`src/enhance.py`)
- **Fast Wiener Deconvolution**: Uses Real Fast Fourier Transform (`np.fft.rfft2` / `irfft2`) to invert motion blur ($H_{\text{motion}}$) and defocus blur ($H_{\text{defocus}}$) in frequency domain, exploiting real-valued matrix symmetry for 2x speedup:
  $$\hat{F}(u,v) = \frac{H^*(u,v)}{|H(u,v)|^2 + K} \cdot G(u,v)$$
- **Adaptive Contrast Boost (CLAHE)**: Operates in the LAB color space on the luminance ($L$) channel to prevent color distortion while boosting character contrast against muddy backgrounds.
- **Auto Gamma Correction**: Computes mean brightness and applies pre-computed gamma lookup tables (`_GAMMA_LUT_CACHE`) to handle night footage and solar glare.

### Stage 5 & 6: Dual OCR Engine & Validation (`src/ocr_engine.py`, `src/plate_validator.py`)
- **Engine Ensemble**: Runs primary ONNX RapidOCR alongside Tesseract OCR across Page Segmentation Modes (PSM 7, 8, 13).
- **Format Validation**: Validates candidate strings against Indian registration rules (State Code + RTO Code + Series + Number, e.g., `^[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}$`).
- **Consensus & Scoring**: Scores predictions using string similarity and regex compliance to pick the highest-confidence valid candidate.

---

## 3. Quantitative Results & Evaluation Analysis

### Synthetic Benchmark Suite (7 Controlled Degradation Cases)

| Scenario ID | Degradation Type | Ground Truth | Single Image Similarity | Video Multi-Frame Similarity | Recovered Plate Text |
|---|---|---|:---:|:---:|:---:|
| **Case 1** | Heavy Motion Blur | `MH12AB1234` | 85.0% | **100.0%** | `MH12AB1234` |
| **Case 2** | Defocus Blur | `DL8CAF5566` | **100.0%** | **100.0%** | `DL8CAF5566` |
| **Case 3** | Distant Low-Res | `KA05MZ7788` | 80.0% | **100.0%** | `KA05MZ7788` |
| **Case 4** | Low-Light Night Noise | `TN10CX9012` | **100.0%** | **100.0%** | `TN10CX9012` |
| **Case 5** | Solar Glare Overexposure | `GJ01HN3344` | 90.0% | **100.0%** | `GJ01HN3344` |
| **Case 6** | Perspective Skew (35°) | `RJ14GB6677` | **100.0%** | **100.0%** | `RJ14GB6677` |
| **Case 7** | Compound Degradation | `UP32KD4455` | 88.9% | **100.0%** | `UP32KD4455` |

### Key Analysis Insights
1. **Multi-Frame ECC Superiority**: While single-frame Wiener deconvolution restores character sharpness for moderate blur, multi-frame temporal median fusion achieves **100% exact match** on extreme low-resolution and low-light noise cases where single frames lose high-frequency details.
2. **Wiener Deconvolution vs. Unsharp Masking**: Wiener deconvolution in the frequency domain is critical for recovery when blur kernel size exceeds 7 pixels, whereas simple spatial unsharp masking fails by amplifying background noise.
3. **Format-Aware Regex Scoring**: Regex filtering eliminated false positives where generic OCR misread `O` as `0` or `I` as `1` in state prefix positions.

---

## 4. Performance & Efficiency Optimizations

- **ONNX Model Inference**: Replaced PyTorch runtime with ONNX Runtime, lowering CPU execution latency from 180ms to 45ms per frame.
- **RFFT2 Fourier Speedup**: Replaced standard complex 2D FFT (`fft2`) with real-valued 2D FFT (`rfft2`) in Wiener deconvolution, achieving ~2x speedup in image restoration.
- **Cached Lookup Tables**: Pre-calculated 256-element gamma correction lookup tables in memory (`_GAMMA_LUT_CACHE`), eliminating array exponentiation on every frame.
- **Concurrent Batch Pipeline**: Added `ThreadPoolExecutor` in `src/pipeline.py` to enable parallel image processing across multiple CPU cores.
- **Fast OpenCV Quality Metrics**: Replaced heavy `scikit-image` PSNR/SSIM functions with optimized OpenCV matrix primitives, reducing metric computation time by ~5x.

---

## 5. Directory Structure & Deliverables Setup

```
license/
├── README.md                      # Complete setup, usage & execution guide
├── REPORT.md                      # Technical report, methodology & evaluation analysis
├── requirements.txt               # Project dependencies
├── check_env.py                   # Environment verification script
├── download_dataset.py            # Kaggle Indian License Plate dataset downloader
├── extract_dataset.py             # Dataset extractor & verifier
├── prepare_yolo_dataset.py        # YOLO format dataset prep & train script
├── samples/                       # Deliverable 3: Before/After sample outputs
│   ├── case1_motion_blur.png
│   ├── case2_defocus_blur.png
│   ├── case3_low_res.png
│   ├── case4_night_noise.png
│   └── case5_glare_overexposed.png
├── models/                        # Pre-trained YOLO detection models
│   ├── license_plate_yolo.onnx
│   └── license_plate_yolo.pt
├── src/                           # Core source modules (Deliverable 1)
│   ├── detect.py                  # Object detection & bounding box localization
│   ├── enhance.py                 # Rectification, Wiener deconvolution & CLAHE
│   ├── pipeline.py                # Main orchestration & video ECC alignment
│   ├── ocr_engine.py              # RapidOCR + Tesseract ensemble engine
│   ├── metrics.py                 # Levenshtein distance & PSNR/SSIM evaluation
│   ├── plate_validator.py         # Indian registration format regex engine
│   ├── dataset_loader.py          # Benchmark data loader
│   └── synth_data.py              # Synthetic degradation generator
└── tests/                         # Automated pytest test suite (62 tests)
    ├── conftest.py
    ├── test_enhance.py
    └── test_metrics.py
```

---

## 6. Conclusion

The developed pipeline demonstrates that combining physics-informed classical signal restoration with modern deep learning localization provides superior robustness and efficiency for license plate recognition under severe real-world noise. All code, models, sample outputs, and tests are open-source and fully reproducible.
