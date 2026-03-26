import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import config
from pathlib import Path

class EcuadorianDocumentsDataset(Dataset):
    """
    Loads document images from a split folder.

    Expected folder structure:
        folder_name/
            id/        → label 0
            passport/  → label 1
            unknown/   → label 2
    """

    def __init__(self, folder_name: Path, classes: dict, transform=None):
        self.folder_name      = folder_name
        self.classes   = classes          # {"id": 0, "passport": 1, "unknown": 2}
        self.transform = transform
        self.samples   = []               # list of (path, label)
        self.images = [] # list pf paths

        for class_name, label in classes.items():
            class_dir = folder_name / class_name
            if not class_dir.exists():
                print(f"  ⚠️  Folder not found: {class_dir}")
                continue
            images = list_images(class_dir)
            self.images.extend(images)
            self.samples.extend([(img, label) for img in images])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image as numpy array (albumentations requires numpy, not PIL)
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label

    def class_counts(self) -> dict:
        """Returns number of samples per class."""
        label_to_name = {v: k for k, v in self.classes.items()}
        counts = {name: 0 for name in self.classes}
        for _, label in self.samples:
            counts[label_to_name[label]] += 1
        return counts

def get_transforms(is_train: bool = False):
    """
    Centralized transforms for the whole project.
    Modes: 'train', 'val', or 'test'
    """
    # Common normalization
    norm_step = A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    tensor_step = ToTensorV2()
    target_size = (config.IMAGE_SIZE, config.IMAGE_SIZE)

    if is_train:
        return A.Compose([
            # 1. Geometric & Perspective
            A.Perspective(scale=(0.05, 0.1), p=0.5),
            A.Rotate(limit=20, p=0.7),
            A.RandomResizedCrop(size=target_size, scale=(0.8, 1.0)),

            # 2. Lighting & Color
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0.3),

            # 3. Camera Artifacts & Obstructions
            A.GaussNoise(std_range=(0.1, 0.4), p=0.3),
            A.MotionBlur(blur_limit=3, p=0.2),
            A.ImageCompression(quality_range=(70, 100), p=0.3),
            A.CoarseDropout(num_holes_range=(1,3), hole_height_range=(0.05, 0.15), hole_width_range=(0.05, 0.15), fill='random', p=0.4),

            norm_step,
            tensor_step
        ])

    # Validation/Test (No random flips/crops, just resize)
    return A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        norm_step,
        tensor_step
    ])

def list_images(folder: Path) -> list[Path]:
    """Returns sorted list of valid image files in a folder."""
    return sorted([f for f in folder.iterdir() if f.suffix in config.VALID_EXTENSIONS])