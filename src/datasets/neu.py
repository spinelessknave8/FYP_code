import os
import random
from typing import List, Tuple
from PIL import Image
from torch.utils.data import Dataset


class NEUDataset(Dataset):
    def __init__(self, root: str, class_mapping: dict, classes: List[str], transform=None):
        self.root = root
        self.class_mapping = class_mapping
        self.classes = classes
        self.transform = transform
        self.samples = self._gather_samples()

    def _gather_samples(self):
        samples = []
        for cls in self.classes:
            folder = self.class_mapping[cls]
            cls_dir = os.path.join(self.root, folder)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    samples.append((os.path.join(cls_dir, fname), cls))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cls = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, cls


def collect_neu_samples(root: str, class_mapping: dict, splits: List[str], images_dir: str = "images"):
    samples = []
    for split in splits:
        split_root = os.path.join(root, split, images_dir)
        for cls, folder in class_mapping.items():
            cls_dir = os.path.join(split_root, folder)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    samples.append((os.path.join(cls_dir, fname), cls))
    return samples


class NEUListDataset(Dataset):
    def __init__(self, samples, class_to_idx=None, transform=None):
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


def split_known_unknown(samples: List[Tuple[str, str]], known_classes: List[str], seed: int):
    known = [s for s in samples if s[1] in known_classes]
    unknown = [s for s in samples if s[1] not in known_classes]
    random.Random(seed).shuffle(known)
    return known, unknown


def stratified_split(samples: List[Tuple[str, str]], train_ratio=0.7, val_ratio=0.15, seed=42):
    rng = random.Random(seed)
    by_class = {}
    for path, cls in samples:
        by_class.setdefault(cls, []).append((path, cls))
    train, val, test = [], [], []
    for cls, items in by_class.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train += items[:n_train]
        val += items[n_train:n_train + n_val]
        test += items[n_train + n_val:]
    return train, val, test
