# Forensic License Plate Enhancement & Recovery Pipeline

A modular, explainable computer-vision and forensic engineering pipeline designed to localize, rectify, enhance, and transcribe degraded vehicle license plates from real-world and synthetic footage.

The pipeline specifically addresses real-world capture defects including:
- **Motion blur** (vehicle moving past a stationary camera).
- **Defocus blur** (incorrect lens focal plane).
- **Low resolution & pixelation** (distant CCTV surveillance).
- **Sensor noise & night-time under-exposure** (low-light conditions).
- **Over-exposure & glare** (sunlight and headlight washout).
- **Perspective skew** (cameras capturing vehicles at sharp angles).
- **Camera shake & jitter** (compensating via multi-frame ECC video stabilization).

---

## System Architecture

```
d:\google\license\
├── data/
│   ├── indian_vehicle_dataset/   # Extracted real vehicle images + Pascal VOC XML annotations
│   └── synthetic/                # 7 canonical forensic test cases + jitter videos
├── src/
│   ├── detect.py                 # Plate localization (Sobel edge density + morphology + aspect ratio)
│   ├── enhance.py                # Forensic restoration (Wiener deblur, CLAHE, NLM, super-resolution, ECC)
│   ├── ocr_engine.py             # RapidOCR (ONNX Runtime) + Tesseract fallback + ensemble ranking
│   ├── metrics.py                # PSNR, SSIM, Levenshtein character accuracy, Laplacian sharpness
│   ├── dataset_loader.py         # Real dataset loader and degradation testbed
│   ├── synth_data.py             # Controlled benchmark generator with ground truth
│   └── pipeline.py               # Master CLI orchestrator
├── outputs/
│   ├── sample_gallery/           # 22 before/after comparative visual cards
│   ├── real_benchmark/           # Quantitative evaluation on Indian Vehicle Dataset
│   └── synthetic_benchmark/      # Quantitative evaluation on synthetic cases
├── REPORT.md                     # Comprehensive forensic evaluation & failure analysis report
├── README.md                     # Usage instructions & interview guide
└── requirements.txt              # Project dependencies
```

---

## Quickstart & Installation

### 1. Environment Setup

```bash
# Clone or navigate to the repository
cd d:\google\license

# Create a virtual environment using Python 3.10+ or uv
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running on New / Unseen Inputs (Interview Evaluation)

The pipeline is completely ready to accept new test images and video clips:

### 1. Process a Single Image

```bash
python src/pipeline.py --image path/to/vehicle.jpg --out outputs/interview_case_1
```

If you have an optional ground-truth plate string or clean reference for quantitative scoring:
```bash
python src/pipeline.py --image path/to/vehicle.jpg --gt "MH12AB1234" --out outputs/interview_case_1
```

### 2. Process a Shaky / Degraded Video Clip

The pipeline automatically identifies the sharpest reference frame, performs sub-pixel ECC motion alignment across all frames to cancel camera jitter, computes a temporal median super-resolution composite, and extracts the plate:
```bash
python src/pipeline.py --video path/to/dashcam.mp4 --out outputs/interview_video_case
```

### 3. Generated Artifacts per Input

For every processed input, the output directory contains 5 structured artifacts:
1. `01_detected_plate.png` — Raw localized plate region.
2. `02_rectified_plate.png` — Perspective-corrected frontal plate crop.
3. `03_enhanced_plate.png` — High-resolution deblurred and contrast-restored plate.
4. `04_before_after.png` — Side-by-side comparative inspection panel with OCR predictions and confidence score.
5. `result.json` — Full machine-readable metrics (sharpness before/after, OCR candidates, winning enhancement variant, detection bounding box).

---

## Running Replicable Benchmarks

### Benchmark on Real Indian Vehicles (`saisirishan/indian-vehicle-dataset`)

Evaluates detection and enhancement across real Indian vehicles spanning diverse states (TN, UP, RJ, KL, MH, etc.) and controlled real-world degradations:
```bash
python src/pipeline.py --benchmark-real --num-samples 8
```

### Benchmark on Synthetic Benchmark Suite

Generates 7 controlled degradation scenarios with known ground-truth geometry and text, evaluating both single-frame and multi-frame video stabilization:
```bash
python src/pipeline.py --benchmark-synth
```

All summary metrics are written to `outputs/real_benchmark/real_benchmark_summary.json` and `outputs/synthetic_benchmark/synthetic_benchmark_summary.json`.

---

## Key Results Summary

- **Real Indian Plates**:
  - `TN21AT0492`: Recovered with **99.5% confidence** (Exact match).
  - `KL63C8800`: Baseline missed state code (`63C8800`), pipeline recovered full string `'KL63C8800'` (**98.3% confidence**).
  - `RJ27TC0530`: Recovered with **97.7% confidence** (Exact match).
  - `MH46X9996`: Recovered with **92.2% confidence** (Exact match).
- **Defocus Blur (`DL8CAF5566`)**: **100% exact match** (Sim 1.0) in both image and video modes.
- **Distant Low-Res Pixelation (`KA05MZ7788`)**: Single image recovered 80% (`KA05427788`); video ECC multi-frame averaging recovered **100% exact match** (`KA05MZ7788`).
- **Low-Light Sensor Noise (`TN10CX9012`)**: **100% exact match** (Sim 1.0) in both image and video modes.
- **Overexposed Solar Glare (`GJ01HN3344`)**: Video multi-frame averaging achieved **100% exact match** (Sim 1.0).
