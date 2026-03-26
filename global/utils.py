"""
utils.py — Shared dataset class and image transforms for the whole project.

Both the training notebooks and the Flask app rely on the same preprocessing
pipeline. Keeping it here means we can't accidentally end up with a slightly
different resize or normalisation between training and inference — a mismatch
like that would silently hurt model performance and be very difficult to debug.
"""

import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import config


class EcuadorianDocumentsDataset(Dataset):
    """
    A PyTorch Dataset that loads Ecuadorian document images from a split folder.

    Each split (train / val / test) is expected to follow this layout:

        split_folder/
            id_new/       ← class 0
            id_old/       ← class 1
            passport/     ← class 2
            unknown/      ← class 3

    The class → integer mapping comes from config.CLASSES so it stays
    consistent across the whole project.

    Usage:
        from utils import EcuadorianDocumentsDataset, get_transforms
        ds = EcuadorianDocumentsDataset(config.TRAIN_DIR, config.CLASSES,
                                        transform=get_transforms(is_train=True))
        image_tensor, label = ds[0]
    """

    def __init__(self, folder_name: Path, classes: dict, transform=None):
        self.folder_name = folder_name
        self.classes     = classes    # {"id_new": 0, "id_old": 1, ...}
        self.transform   = transform
        self.samples     = []         # list of (path, label) tuples
        self.images      = []         # flat list of paths, useful for EDA

        # Walk through each class subfolder and collect every valid image.
        # If a subfolder is missing we just warn and skip — this is handy
        # during early experiments when not all classes are set up yet.
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

        # Albumentations works with numpy arrays, not PIL images, so we
        # convert here. We also force RGB so that grayscale or RGBA images
        # don't cause shape mismatches downstream.
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label

    def class_counts(self) -> dict:
        """
        Returns how many samples belong to each class.
        Useful for spotting class imbalance before training.
        """
        label_to_name = {v: k for k, v in self.classes.items()}
        counts = {name: 0 for name in self.classes}
        for _, label in self.samples:
            counts[label_to_name[label]] += 1
        return counts


def get_transforms(is_train: bool = False) -> A.Compose:
    """
    Returns the albumentations pipeline for either training or evaluation.

    Why two separate pipelines?
    — During training we want the model to see the document from different
      angles, lighting conditions, and with partial obstructions so it
      generalises to real-world phone photos.
    — During validation and inference we want a deterministic, clean
      transform so the metrics and predictions are reproducible.

    Args:
        is_train: Pass True for the training split, False for val/test/inference.

    Returns:
        An albumentations Compose pipeline ready to be called as:
            output = pipeline(image=numpy_image)["image"]
    """
    norm       = A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    to_tensor  = ToTensorV2()
    target_size = (config.IMAGE_SIZE, config.IMAGE_SIZE)

    if is_train:
        return A.Compose([
            # --- Geometric distortions ---
            # Documents are often photographed at an angle or slightly rotated.
            # Perspective simulates the keystoning effect of a tilted camera,
            # and Rotate covers the simple case of a document that isn't
            # perfectly straight on a desk.
            A.Perspective(scale=(0.05, 0.1), p=0.5),
            A.Rotate(limit=20, p=0.7),
            A.RandomResizedCrop(size=target_size, scale=(0.8, 1.0)),

            # --- Lighting and colour ---
            # Phone photos vary wildly in brightness and white balance.
            # RandomBrightnessContrast covers dim or overexposed shots,
            # and HueSaturationValue covers yellow indoor lighting, etc.
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0.3
            ),

            # --- Sensor noise, blur, and compression artefacts ---
            # GaussNoise mimics a shaky or low-light shot.
            # MotionBlur covers movement during the capture.
            # ImageCompression simulates the JPEG artefacts that appear when
            # a document is sent over WhatsApp or stored at low quality.
            # CoarseDropout mimics fingers, stamps, or stickers partially
            # covering the document.
            A.GaussNoise(std_range=(0.1, 0.4), p=0.3),
            A.MotionBlur(blur_limit=3, p=0.2),
            A.ImageCompression(quality_range=(70, 100), p=0.3),
            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(0.05, 0.15),
                hole_width_range=(0.05, 0.15),
                fill="random",
                p=0.4,
            ),

            norm,
            to_tensor,
        ])

    # Validation / test / inference — just resize and normalise.
    # No randomness here so metrics are stable across epochs.
    return A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        norm,
        to_tensor,
    ])


def list_images(folder: Path) -> list[Path]:
    """
    Returns a sorted list of every valid image file inside a folder.
    Sorting makes the order deterministic across different operating systems,
    which matters for reproducibility when we fix a random seed.
    """
    return sorted([
        f for f in folder.iterdir()
        if f.suffix.lower() in config.VALID_EXTENSIONS
    ])