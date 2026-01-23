import os
import random
from PIL import Image
from torch.utils.data import Dataset


class MVTecGoodDataset(Dataset):
    def __init__(self, root: str, categories: list, transform=None):
        self.root = root
        self.categories = categories
        self.transform = transform
        self.samples = self._gather_samples()

    def _gather_samples(self):
        samples = []
        for cat in self.categories:
            good_dir = os.path.join(self.root, cat, "train", "good")
            if not os.path.isdir(good_dir):
                continue
            for fname in os.listdir(good_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    samples.append(os.path.join(good_dir, fname))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def train_val_split(paths: list, val_ratio: float, seed: int):
    rng = random.Random(seed)
    paths = list(paths)
    rng.shuffle(paths)
    n_val = int(len(paths) * val_ratio)
    return paths[n_val:], paths[:n_val]
