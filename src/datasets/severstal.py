import csv
import os
import random
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset


def _resolve_image_id(row: dict) -> str:
    if "ImageId" in row and row["ImageId"]:
        return row["ImageId"].strip()
    image_class_id = row.get("ImageId_ClassId", "").strip()
    if "_" in image_class_id:
        return image_class_id.split("_", 1)[0]
    return image_class_id


def _encoded_pixels_present(row: dict) -> bool:
    val = row.get("EncodedPixels", "")
    if val is None:
        return False
    sval = str(val).strip()
    return sval != "" and sval.lower() != "nan"


def collect_severstal_normal_images(
    data_root: str,
    train_csv: str = "train.csv",
    images_dir: str = "train_images",
) -> List[str]:
    csv_path = os.path.join(data_root, train_csv)
    img_root = os.path.join(data_root, images_dir)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing Severstal CSV: {csv_path}")
    if not os.path.isdir(img_root):
        raise FileNotFoundError(f"Missing Severstal image dir: {img_root}")

    has_defect = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = _resolve_image_id(row)
            if not image_id:
                continue
            has_defect.setdefault(image_id, False)
            if _encoded_pixels_present(row):
                has_defect[image_id] = True

    normal_images = []
    for image_id, defect_flag in has_defect.items():
        if defect_flag:
            continue
        path = os.path.join(img_root, image_id)
        if os.path.exists(path):
            normal_images.append(path)

    # Some Severstal exports contain only positive-mask rows in train.csv.
    # In that case, images present in train_images but absent from CSV are normal.
    if not normal_images:
        csv_image_ids = set(has_defect.keys())
        for fname in os.listdir(img_root):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue
            if fname in csv_image_ids:
                continue
            normal_images.append(os.path.join(img_root, fname))
    return normal_images


def build_non_overlapping_patch_index(
    image_paths: List[str],
    patch_size: int = 224,
    stride: int = 224,
) -> List[Tuple[str, int, int, int]]:
    index = []
    for path in image_paths:
        with Image.open(path) as img:
            width, height = img.size
        if width < patch_size or height < patch_size:
            continue
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                index.append((path, x, y, patch_size))
    return index


class SeverstalPatchDataset(Dataset):
    def __init__(self, patch_index, transform=None):
        self.patch_index = patch_index
        self.transform = transform

    def __len__(self):
        return len(self.patch_index)

    def __getitem__(self, idx):
        path, x, y, size = self.patch_index[idx]
        img = Image.open(path).convert("RGB")
        patch = img.crop((x, y, x + size, y + size))
        if self.transform:
            patch = self.transform(patch)
        return patch


def train_val_split(paths: List[str], val_ratio: float, seed: int):
    rng = random.Random(seed)
    paths = list(paths)
    rng.shuffle(paths)
    n_val = int(len(paths) * val_ratio)
    return paths[n_val:], paths[:n_val]
