"""
config.py — Central configuration for the whole project.

I keep every "magic number" and path in this one file so that
changing something like the image size or the learning rate
only ever requires editing one place, not hunting through notebooks.

Any script or notebook that needs these values just does:
    import config
    config.IMAGE_SIZE   # → 224
"""

from pathlib import Path
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# __file__ resolves to wherever config.py lives (global/).
# Going one level up gets us to the project root, which is where the data,
# results, and the trained checkpoint are stored.

CONFIG_DIR   = Path(__file__).resolve().parent   # …/ecuadorian-id-classifier/global
PROJECT_ROOT = CONFIG_DIR.parent                 # …/ecuadorian-id-classifier

DATA_DIR   = PROJECT_ROOT / "data"
RAW_DIR    = DATA_DIR / "raw"
TRAIN_DIR  = DATA_DIR / "train"
TEST_DIR   = DATA_DIR / "test"
VAL_DIR    = DATA_DIR / "val"

BEST_MODEL  = PROJECT_ROOT / "best_model.pth"
RESULTS_DIR = PROJECT_ROOT / "results"

# ---------------------------------------------------------------------------
# Image validation
# ---------------------------------------------------------------------------
# These are the only extensions we accept when scanning data folders.
# 50×50 is the absolute floor — anything smaller is almost certainly
# not a real document photo.

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMAGE_MIN_W = 50
IMAGE_MIN_H = 50

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------
# The model was trained on four distinct folder categories.
# "id_new" and "id_old" are later merged into a single "cedula" output
# at inference time, because from the API caller's perspective they are
# the same document type.

CLASSES = {
    "id_new":   0,
    "id_old":   1,
    "passport": 2,
    "unknown":  3,
}
CLASS_NAMES = list(CLASSES.keys())
NUM_CLASSES = len(CLASSES)

# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------
# 70 % train / 15 % val / 15 % test.
# These ratios are applied when the data-preparation notebook splits
# the raw images into the three split folders.

VAL_RATIO  = 0.15
TEST_RATIO = 0.15

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
# 224×224 is the ImageNet standard and matches what timm backbones expect.
# A batch size of 16 comfortably fits in most laptop GPUs with 4 GB VRAM.

IMAGE_SIZE    = 224
BATCH_SIZE    = 16
EPOCHS        = 10
LEARNING_RATE = 0.001

# ---------------------------------------------------------------------------
# ImageNet normalisation
# ---------------------------------------------------------------------------
# Because we use a pretrained backbone (EfficientNet / ViT), the pixel
# values must be normalised with the same mean and std the backbone was
# originally trained with. These numbers are the standard ImageNet values.

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
# PyTorch will automatically use the GPU if one is available.
# DEVICE_NAME is just for logging — it's nice to know which GPU is running.

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
# Fixing a seed means shuffles, weight initialisations, and dropout
# patterns are the same every run, making results reproducible.

RANDOM_SEED = 120699

# ---------------------------------------------------------------------------
# Inference threshold
# ---------------------------------------------------------------------------
# If the model's top-class probability is below this value, we return
# "desconocido" instead of committing to a potentially wrong answer.
# 0.70 felt like the right balance after inspecting the validation curves.

CONFIDENCE_THRESHOLD = 0.70