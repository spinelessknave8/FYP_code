import os
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from ..utils.io import load_yaml, deep_update, save_json
from ..utils.device import get_device
from ..utils.seed import set_seed
from ..datasets.mvtec import MVTecGoodDataset, train_val_split
from ..patchcore.patchcore_simple import PatchCoreBackbone, build_memory_bank, coreset_subsample, infer_anomaly_scores
from ..patchcore.calibrate import calibrate_anomaly_threshold

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ListImageDataset(torch.utils.data.Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def load_config(config_path: str) -> dict:
    default_path = os.path.join("configs", "default.yaml")
    cfg = load_yaml(default_path)
    if os.path.basename(config_path) != "default.yaml":
        cfg = deep_update(cfg, load_yaml(config_path))
    return cfg


def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])

    device = get_device(cfg["device"])
    out_dir = os.path.join(cfg["output_dir"], "patchcore")
    os.makedirs(out_dir, exist_ok=True)

    mvtec_root = cfg["mvtec"]["data_root"]
    categories = [d for d in os.listdir(mvtec_root) if os.path.isdir(os.path.join(mvtec_root, d))]

    tf = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    base_ds = MVTecGoodDataset(mvtec_root, categories, transform=None)
    train_paths, val_paths = train_val_split(base_ds.samples, val_ratio=0.1, seed=cfg["seed"])

    train_ds = ListImageDataset(train_paths, transform=tf)
    val_ds = ListImageDataset(val_paths, transform=tf)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    backbone = PatchCoreBackbone(cfg["patchcore"]["backbone"]).to(device)

    memory = build_memory_bank(train_loader, device, backbone, cfg["patchcore"]["num_patches_per_sample"])
    memory = coreset_subsample(memory, cfg["patchcore"]["coreset_sampling_ratio"], seed=cfg["seed"])

    val_scores = infer_anomaly_scores(val_loader, device, backbone, memory)
    threshold = calibrate_anomaly_threshold(val_scores, cfg["patchcore"]["accept_rate"])

    np.save(os.path.join(out_dir, "memory.npy"), memory)
    save_json(os.path.join(out_dir, "threshold.json"), {"threshold": float(threshold)})
    save_json(os.path.join(out_dir, "meta.json"), {
        "backbone": cfg["patchcore"]["backbone"],
        "accept_rate": cfg["patchcore"]["accept_rate"],
        "num_patches_per_sample": cfg["patchcore"]["num_patches_per_sample"],
        "coreset_sampling_ratio": cfg["patchcore"]["coreset_sampling_ratio"],
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
