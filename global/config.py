from pathlib import Path
import torch
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- PATHS ---
CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONFIG_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
VAL_DIR = DATA_DIR / "val"
BEST_MODEL = PROJECT_ROOT / "best_model.pth"
RESULTS_DIR    = PROJECT_ROOT / "results"

# --- Images ---
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMAGE_MIN_W = 50
IMAGE_MIN_H = 50

# --- Class folders and Category ---
CLASSES = {
    "id_new": 0,
    "id_old": 1,
    "passport": 2,
    "unknown": 3,
}
CLASS_NAMES = ["id_new", "id_old", "passport", "unknown"]
NUM_CLASSES = len(CLASSES)

# --- Split Info ---
VAL_RATIO  = 0.15
TEST_RATIO = 0.15

# --- MODEL HYPERPARAMETERS ---
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001

# --- IMAGE NORMALIZATION (ImageNet Standards) ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# --- HARDWARE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'

# --- RANDOM ---
RANDOM_SEED = 120699

# --- Evaluate ---
CONFIDENCE_THRESHOLD = 0.70