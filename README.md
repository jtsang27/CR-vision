# CR-vision

Computer vision pipeline for **Clash Royale** gameplay: training and running a YOLOv8 model to detect units / structures on screen and exporting those detections into a structured **game state interface (GSI)** format for downstream analysis or reinforcement learning.

> ⚠️ This project is for **research and educational purposes only**. Clash Royale and all related assets are the property of Supercell.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Environment](#environment)
  - [Dataset](#dataset)
- [Training](#training)
- [Inference](#inference)
  - [Running YOLOv8](#running-yolov8)
  - [Exporting to GSI JSONL](#exporting-to-gsi-jsonl)
- [Notes & Limitations](#notes--limitations)
- [Acknowledgements](#acknowledgements)

---

## Overview

**CR-vision** is a small, focused repo that:

1. Uses **YOLOv8** to detect objects (e.g., units, towers, projectiles, etc.) from Clash Royale frames.
2. Stores the dataset configuration in a standard YOLO `data.yaml`.
3. Provides a utility script (`infer_to_gsi.py`) that converts raw detection outputs into a **JSONL “game state interface”** (`gsi.jsonl`) that can be consumed by other systems (e.g., simulators, RL agents, analytics pipelines).

The dataset was originally prepared and exported via **Roboflow**, and the raw dataset metadata is preserved in:

- `README.dataset.txt`
- `README.roboflow.txt`

---

## Repository Structure

```text
CR-vision/
├── cr_vision/            # Core scripts / helpers (model, utils, etc.)
├── train/                # Training images and labels
├── valid/                # Validation images and labels
├── test/                 # Test images and labels
├── runs/
│   └── detect/           # Sample YOLOv8 detection runs / outputs
├── data.yaml             # YOLO dataset configuration (paths, classes, etc.)
├── gsi.jsonl             # Example GSI output (JSONL: one game state per line)
├── infer_to_gsi.py       # Convert detections -> structured game state JSONL
├── yolov8n.pt            # Base YOLOv8 weights (from Ultralytics)
├── README.dataset.txt    # Dataset description (auto-generated)
├── README.roboflow.txt   # Roboflow export info
├── .gitignore
└── .DS_Store             # macOS metadata (safe to ignore)
```

> If you’re viewing this from GitHub and don’t see some of these paths, check the repo tree directly; this structure is based on the current layout.

---

## Getting Started

### Environment

You’ll need **Python 3.10+** (or similar) and typical CV / DL libraries.

Example setup:

```bash
# clone the repo
git clone https://github.com/jtsang27/CR-vision.git
cd CR-vision

# create & activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate     # on Windows: .venv\Scripts\activate

# install dependencies
pip install ultralytics opencv-python numpy tqdm
```

If you already have a global YOLOv8 / Ultralytics environment set up, you can just reuse that.

### Dataset

The dataset is configured via `data.yaml`, which follows the standard Ultralytics YOLO format. It typically includes:

- `train`: path to training images
- `val`: path to validation images
- (optionally) `test`: path to test images
- `nc`: number of classes
- `names`: list of class names

You can inspect and edit `data.yaml` to:

- Change the dataset root paths (e.g., if you move `train/`, `valid/`, `test/`).
- Add / rename classes.

There are also two extra docs:

- `README.dataset.txt` – explains the dataset structure.
- `README.roboflow.txt` – metadata from Roboflow export.

---

## Training

Training is handled using **Ultralytics YOLOv8**. A typical training command (from this repo root) would look like:

```bash
yolo detect train   data=data.yaml   model=yolov8n.pt   epochs=100   imgsz=640   project=runs   name=cr-vision
```

Key knobs you can adjust:

- `model=` – change to a different YOLOv8 variant (e.g., `yolov8s.pt`, `yolov8m.pt`) if you want a bigger/smaller model.
- `epochs=` – training length.
- `imgsz=` – input image size.
- `project=` / `name=` – where training runs and logs are saved.

Training outputs (weights, metrics, predictions) will appear under `runs/`.

---

## Inference

### Running YOLOv8

To run inference on images or video:

```bash
# images
yolo detect predict   model=path/to/weights.pt   source=path/to/images_or_folder   save=True   project=runs   name=cr-vision-pred

# video
yolo detect predict   model=path/to/weights.pt   source=path/to/video.mp4   save=True   project=runs   name=cr-vision-video
```

This will:

- Produce annotated frames with bounding boxes in `runs/detect/...`.
- Save raw prediction data (depending on your YOLOv8 configuration).

### Exporting to GSI JSONL

The script `infer_to_gsi.py` is intended to take YOLO detections and convert them into a **GSI JSONL** file, where each line represents one “game state” (e.g., a frame) with structured detections.

A typical usage pattern (update to match your actual flags) might look like:

```bash
python infer_to_gsi.py   --source path/to/video_or_frames   --weights path/to/weights.pt   --output gsi.jsonl
```

Common ideas of what `infer_to_gsi.py` may do:

- Run YOLOv8 inference over a sequence of frames.
- Group detections by frame / timestamp.
- Emit a JSON object per line with fields like:
  - `frame_id` or `timestamp`
  - `entities`: list of detected units/structures with:
    - class name / ID
    - bounding box
    - confidence score
    - (optionally) side/team, lane, or other derived attributes

If you are using this as input to a simulator / RL agent, you can adapt the script to match exactly the schema that your downstream code expects.

---

## Notes & Limitations

- This repo does **not** include any game assets; you’re expected to collect gameplay footage yourself under the terms allowed for personal / research use.
- Model quality depends heavily on:
  - Dataset size & labeling quality.
  - Coverage of different arenas, skins, and UI states.
- Paths in `data.yaml` and scripts may be hard-coded to a local file layout; if things break, double-check that the paths match your environment.

---

## Acknowledgements

- **Ultralytics YOLOv8** – core detection model used for training and inference.  
- **Roboflow** – dataset management / export tooling originally used to prepare the dataset.  
- **Supercell** – developers of Clash Royale. This project is not affiliated with or endorsed by Supercell in any way; it is purely a personal / academic experiment.
