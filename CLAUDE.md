# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Human Action Recognition (HAR)** system built as a university thesis (Licenta). It implements an end-to-end pipeline: extracting skeleton pose data from the HAA500 video dataset using MediaPipe, then training a Spatio-Temporal Graph Convolutional Network (ST-GCN) to classify 500 human action classes.

Two model variants live in this repo:
- **v1 (`st_gcn_pipeline.py`)** — vanilla ST-GCN on 2D landmarks
- **v2 (`st_gcn_v2.py`)** — two-stream (joint + bone) ST-GCN with multi-scale TCN, temporal attention, and richer training (mixup, AdamW, warmup+cosine LR, AMP, 3D world landmarks)

## Setup

**Environment:** Python 3.11 venv at `venv/` with CUDA-enabled PyTorch (`torch 2.11.0+cu126`). All paths in source use `Path(__file__).resolve().parent.parent`, so the project is portable — no edits needed when relocating.

```powershell
# Activate venv (Windows PowerShell)
venv\Scripts\Activate.ps1

# Re-create from scratch on a fresh Windows machine:
py -3.11 -m venv venv
venv\Scripts\pip.exe install --upgrade pip
venv\Scripts\pip.exe install torch --index-url https://download.pytorch.org/whl/cu126
venv\Scripts\pip.exe install -r requirements_windows.txt
venv\Scripts\pip.exe install seaborn
```

`requirements_windows.txt` is the Linux-stripped freeze (no `nvidia-*` / `triton` / `cuda-*` pip wheels — those come bundled with the Windows torch CUDA build). `requirements_full.txt` is the original WSL freeze, kept for reference. `requirements.txt` is incomplete (no torch/mediapipe/opencv) and should not be used directly.

The MediaPipe pose landmarker task file (`pose_landmarker_full.task`) must be present in the project root — `pose_extraction.py` auto-downloads it if missing.

## Pipeline Execution (in order)

```powershell
# 1. Generate metadata CSV and EDA report
python src\eda.py

# 2. Create stratified train/val/test splits (70/15/15)
python src\dataset_split.py

# 3a. Extract MediaPipe 2D skeleton keypoints (image-space x/y/visibility)
python src\pose_extraction.py

# 3b. Extract MediaPipe 3D world landmarks (x/y/z/visibility) — used by v2
python src\pose_extraction_3d.py

# 4. (Optional) Visually verify skeleton overlay on a single video
python src\pose_visualizer.py

# 5a. Train v1 ST-GCN
python src\st_gcn_pipeline.py

# 5b. Train v2 two-stream ST-GCN (recommended)
python src\st_gcn_v2.py

# (Optional) Build a debug compilation video of skeleton overlays
python src\skeleton_video_compiler.py
```

The `run_*.sh` scripts at the project root are Linux/WSL convenience wrappers and are not used on Windows.

## Architecture

### Data Flow
- **Input**: Raw MP4 videos in `haa500_v1_1/video/<class>/`
- **2D extraction** (`pose_extraction.py`): MediaPipe `PoseLandmarker` → 33 joints/frame as `(T, 33, 3)` (x, y, visibility), saved under `extracted_skeletons/{train|val|test}/<class>/<video>.npy`
- **3D extraction** (`pose_extraction_3d.py`): same landmarker, but uses `pose_world_landmarks` → `(T, 33, 4)` (x, y, z, visibility) under `extracted_skeletons_world/`
- **Temporal normalization**: padded or sampled to exactly 60 frames at training time

### ST-GCN v1 (`src/st_gcn_pipeline.py`)
- **Input tensor**: `(N, C=3, T=60, V=33, M=1)` — batch, channels, time-steps, vertices, persons
- **Graph**: 34 edges encoding the MediaPipe skeleton topology
- **Blocks**: 10 ST-GCN blocks with spatial graph convolution + temporal convolution (kernel 9×1), residual connections, dropout 0.5
- **Channel progression**: 64 → 64 → 64 → 128 → 128 → 128 → 256 → 256 → 256 (stride-2 at blocks 4 and 7)
- **Output**: Global average pool → FC → 500-class softmax
- **Training**: Adam (lr=0.01), CrossEntropyLoss, 50 epochs, batch 32; checkpoint → `best_stgcn_model.pth`

### ST-GCN v2 (`src/st_gcn_v2.py`)
- **Two-stream** model: separate `SingleStream` networks for **joint** and **bone** features; logits averaged at fusion
- **Per-stream input**: `(N, C=7, T=60, V=33, M=1)` — joint stream packs `[x, y, z, vis, vx, vy, vz]`; bone stream packs bone vectors + visibility + bone velocities
- **Skeleton normalization**: prefers 3D world landmarks (`extracted_skeletons_world/`), falls back to 2D (`extracted_skeletons/`) per-file. Hip-centered (2D) or shoulder-scaled (3D)
- **Augmentation**: horizontal flip with FLIP_PAIRS swap, occasional time reverse, scale jitter, Gaussian noise; **mixup** (α=0.3) at batch level
- **Block stack** (10 total, with stochastic depth `drop_path` 0–0.3): 4 × 64ch → 3 × 128ch (stride-2) → 3 × 256ch (stride-2). Last 2 blocks add a `TemporalAttention` (8-head MHA over time)
- **Temporal conv**: `MultiScaleTCN` — 4 parallel branches (kernels 3/7/9 + max-pool) concatenated along channel
- **Training**: AdamW (lr=1e-3, wd=5e-4), `LinearLR` warmup 5 epochs → `CosineAnnealingLR`, label smoothing 0.15, grad clip 1.0, **AMP/autocast** on CUDA, 150 epochs, batch 32; checkpoint → `best_stgcn_v2.pth` / resume from `checkpoint_v2.pth`
- **Optional** `combined_train=True`: trains on `train+val`, monitors on `test` (used for final runs)

### Key Files
| File | Role |
|------|------|
| `src/eda.py` | Loads `haa500_v1_1/raw/*.txt` metadata, analyzes video properties, outputs `HAA500_consolidated_metadata.csv` and `haa500_eda_report.png` |
| `src/dataset_split.py` | Stratified split → `splits/{train,val,test}_split.csv` |
| `src/pose_extraction.py` | Video → 2D skeleton arrays (`extracted_skeletons/`) |
| `src/pose_extraction_3d.py` | Video → 3D world-landmark arrays (`extracted_skeletons_world/`) |
| `src/pose_visualizer.py` | Debug tool: overlays skeleton on video frames |
| `src/skeleton_video_compiler.py` | Builds `skeleton_compilation.mp4` — concatenated skeleton overlays for visual QA |
| `src/st_gcn_pipeline.py` | v1 model + dataset + training loop |
| `src/st_gcn_v2.py` | v2 two-stream model + dataset + training loop |

### Checkpoints at project root
| File | Produced by |
|------|-------------|
| `best_stgcn_model.pth` | v1 best-val checkpoint |
| `training_checkpoint.pth` | v1 resume state |
| `best_stgcn_v2.pth` | v2 best-val checkpoint |
| `checkpoint_v2.pth` | v2 resume state |
