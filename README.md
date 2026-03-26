# Ecuadorian ID Classifier

A computer vision system that classifies images of Ecuadorian identity documents using a fine-tuned deep learning model. It distinguishes between **cédulas** (new and old format), **pasaportes**, and **unknown/rejected** images, and exposes a production-ready REST API via Flask.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Data Structure](#data-structure)
- [Training](#training)
- [Running the API](#running-the-api)
- [Model Output Classes](#model-output-classes)
- [Configuration](#configuration)
- [License](#license)

---

## Overview

| Detail | Value |
|---|---|
| Model backbone | EfficientNet / ViT via `timm` (stored in `best_model.pth`) |
| Internal classes | `id_new`, `id_old`, `passport`, `unknown` (4 classes) |
| Public API classes | `cedula`, `pasaporte`, `desconocido` (merged) |
| Image size | 224 × 224 px (ImageNet normalisation) |
| Confidence threshold | 0.70 |
| Device | CUDA if available, otherwise CPU |

---

## Project Structure

```
ecuadorian-id-classifier/
├── app/                    ← Flask inference microservice
│   ├── app.py
│   ├── templates/
│   │   └── index.html      ← Drag-and-drop web UI
│   └── README.md           ← App-specific docs
├── data/
│   ├── raw/                ← Original, unprocessed images
│   ├── train/              ← Training split (per-class subfolders)
│   ├── val/                ← Validation split
│   ├── test/               ← Test split
│   └── backup/
├── global/
│   ├── config.py           ← Central configuration (paths, hyperparameters)
│   └── utils.py            ← Dataset class & transform helpers
├── notebooks/
│   ├── cleanning/          ← Data cleaning notebooks
│   ├── load/               ← Data loading & EDA notebooks
│   └── model/              ← Training & evaluation notebooks
├── results/                ← Saved metrics, plots, confusion matrices
├── best_model.pth          ← Trained model checkpoint
├── class_map.json          ← Label ↔ class-name mapping
├── requirements.txt        ← Python dependencies
└── LICENSE
```

---

## Setup

> **Prerequisites:** [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

### 1 — Create and activate the environment

```bash
conda create -n ecuadorian-id python=3.10 -y
conda activate ecuadorian-id
```

### 2 — Install PyTorch (choose one)

**GPU (CUDA 11.8):**
```bash
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

**CPU only:**
```bash
conda install pytorch torchvision cpuonly -c pytorch -y
```

### 3 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 4 — Verify the installation

```python
import torch, timm, albumentations, flask
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
```

---

## Data Structure

Place raw images under `data/raw/` in per-class subfolders. After running the data-preparation notebooks the split folders are created automatically:

```
data/
├── raw/
│   ├── id_new/
│   ├── id_old/
│   ├── passport/
│   └── unknown/
├── train/   (70 %)
├── val/     (15 %)
└── test/    (15 %)
```

Split ratios are configured in `global/config.py` (`VAL_RATIO`, `TEST_RATIO`).

---

## Training

Open and run the notebooks in order:

```
notebooks/cleanning/   → Data cleaning & deduplication
notebooks/load/        → Dataset analysis & visualisation
notebooks/model/       → Training, evaluation, checkpoint export
```

The best model checkpoint is saved to `best_model.pth` in the project root.

---

## Running the API

See [`app/README.md`](app/README.md) for full API documentation.

```bash
# Quick start (from repo root, env must be active)
conda activate ecuadorian-id
python app/app.py
```

Server starts at `http://localhost:5000`.

---

## Model Output Classes

| Internal label | Public API label | Description |
|---|---|---|
| `id_new` | `cedula` | Modern Ecuadorian cédula |
| `id_old` | `cedula` | Legacy Ecuadorian cédula |
| `passport` | `pasaporte` | Ecuadorian passport |
| `unknown` | `desconocido` | Rejected / unrecognised document |

When the top-class confidence is below **0.70** the result is always `desconocido`, regardless of the predicted class.

---

## Configuration

All shared constants live in [`global/config.py`](global/config.py):

| Constant | Default | Description |
|---|---|---|
| `IMAGE_SIZE` | `224` | Resize target (px) |
| `BATCH_SIZE` | `16` | Training batch size |
| `EPOCHS` | `10` | Total training epochs |
| `LEARNING_RATE` | `0.001` | Initial LR |
| `CONFIDENCE_THRESHOLD` | `0.70` | Min confidence for a valid prediction |
| `RANDOM_SEED` | `120699` | Global random seed |
| `DEVICE` | auto | `cuda` if available, else `cpu` |

---

## License

[MIT](LICENSE)
