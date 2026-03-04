import csv
import os
import random
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
from torch.utils.data import Dataset


def _is_mask_present(encoded_pixels: str) -> bool:
    if encoded_pixels is None:
        return False
    v = str(encoded_pixels).strip()
    return v != "" and v.lower() != "nan"


def load_severstal_image_labels(data_root: str, train_csv: str = "train.csv") -> Dict[str, set]:
    csv_path = Path(data_root) / train_csv
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing Severstal CSV: {csv_path}")

    image_labels = defaultdict(set)
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row["ImageId"].strip()
            cls = int(row["ClassId"])
            if _is_mask_present(row.get("EncodedPixels", "")):
                image_labels[image_id].add(cls)
    return image_labels


def collect_single_label_defect_samples(
    data_root: str,
    train_csv: str = "train.csv",
    images_dir: str = "train_images",
) -> List[Tuple[str, str]]:
    labels = load_severstal_image_labels(data_root, train_csv)
    img_root = Path(data_root) / images_dir
    samples = []
    for image_id, cls_set in labels.items():
        if len(cls_set) != 1:
            continue
        cls = next(iter(cls_set))
        path = img_root / image_id
        if path.exists():
            samples.append((str(path), f"Class_{cls}"))
    return samples


def collect_normal_image_paths(
    data_root: str,
    train_csv: str = "train.csv",
    images_dir: str = "train_images",
) -> List[str]:
    labels = load_severstal_image_labels(data_root, train_csv)
    img_root = Path(data_root) / images_dir
    all_imgs = [p for p in img_root.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    normals = [str(p) for p in all_imgs if p.name not in labels]
    return normals


def severstal_stats(data_root: str, train_csv: str = "train.csv", images_dir: str = "train_images") -> dict:
    labels = load_severstal_image_labels(data_root, train_csv)
    csv_path = Path(data_root) / train_csv
    class_row_counts = Counter()
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if _is_mask_present(row.get("EncodedPixels", "")):
                class_row_counts[int(row["ClassId"])] += 1

    cardinality = Counter(len(v) for v in labels.values())
    single_counts = Counter()
    for cls_set in labels.values():
        if len(cls_set) == 1:
            single_counts[next(iter(cls_set))] += 1

    img_root = Path(data_root) / images_dir
    all_imgs = [p for p in img_root.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    return {
        "defect_class_ids": sorted(class_row_counts.keys()),
        "mask_rows_per_class": dict(sorted(class_row_counts.items())),
        "images_with_defects": len(labels),
        "defect_label_cardinality": dict(sorted(cardinality.items())),
        "single_label_images_per_class": dict(sorted(single_counts.items())),
        "all_train_images": len(all_imgs),
        "normal_images": len([p for p in all_imgs if p.name not in labels]),
    }


class ImageLabelDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, str]], class_to_idx=None, transform=None):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cls = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.class_to_idx is None:
            return img, cls
        return img, self.class_to_idx[cls]


def stratified_split(samples: List[Tuple[str, str]], train_ratio=0.7, val_ratio=0.15, seed=42):
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for path, cls in samples:
        by_class[cls].append((path, cls))

    train, val, test = [], [], []
    for items in by_class.values():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train += items[:n_train]
        val += items[n_train:n_train + n_val]
        test += items[n_train + n_val:]
    return train, val, test
