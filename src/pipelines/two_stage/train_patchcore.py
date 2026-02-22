import os
import argparse
import json
import numpy as np
import time
from torch.utils.data import DataLoader
from torchvision import transforms

from ...utils.io import load_yaml, deep_update, save_json
from ...utils.device import get_device
from ...utils.seed import set_seed
from ...datasets.severstal import (
    collect_severstal_normal_images,
    build_non_overlapping_patch_index,
    SeverstalPatchDataset,
    train_val_split,
)
from ...patchcore.patchcore_simple import PatchCoreBackbone, build_memory_bank, coreset_subsample, infer_anomaly_scores
from ...patchcore.calibrate import calibrate_anomaly_threshold

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_config(config_path: str) -> dict:
    default_path = os.path.join("configs", "default.yaml")
    cfg = load_yaml(default_path)
    if os.path.basename(config_path) != "default.yaml":
        cfg = deep_update(cfg, load_yaml(config_path))
    return cfg


def main(config_path: str):
    t_start = time.time()
    cfg = load_config(config_path)
    set_seed(cfg["seed"])

    device = get_device(cfg["device"])
    out_dir = os.path.join(cfg["output_dir"], "patchcore")
    os.makedirs(out_dir, exist_ok=True)
    memory_pre_path = os.path.join(out_dir, "memory_precoreset.npy")
    memory_path = os.path.join(out_dir, "memory.npy")
    val_scores_path = os.path.join(out_dir, "val_scores.npy")
    threshold_path = os.path.join(out_dir, "threshold.json")

    sev_cfg = cfg["severstal"]
    sev_root = sev_cfg["data_root"]
    normal_image_paths = collect_severstal_normal_images(
        sev_root,
        train_csv=sev_cfg.get("train_csv", "train.csv"),
        images_dir=sev_cfg.get("images_dir", "train_images"),
    )
    if not normal_image_paths:
        raise ValueError("No Severstal normal images found from CSV filter.")

    val_ratio = float(sev_cfg.get("val_ratio", 0.1))
    train_paths, val_paths = train_val_split(normal_image_paths, val_ratio=val_ratio, seed=cfg["seed"])

    patch_size = int(sev_cfg.get("patch_size", cfg["image_size"]))
    patch_stride = int(sev_cfg.get("patch_stride", patch_size))
    train_patch_index = build_non_overlapping_patch_index(train_paths, patch_size=patch_size, stride=patch_stride)
    val_patch_index = build_non_overlapping_patch_index(val_paths, patch_size=patch_size, stride=patch_stride)
    if not train_patch_index or not val_patch_index:
        raise ValueError("Severstal patch extraction produced an empty train/val set.")

    print("[two-stage] Stage 1 PatchCore setup complete")
    print(f"  normal images total: {len(normal_image_paths)}")
    print(f"  normal train images: {len(train_paths)}")
    print(f"  normal val images: {len(val_paths)}")
    print(f"  train patches: {len(train_patch_index)}")
    print(f"  val patches: {len(val_patch_index)}")
    print(f"  patch size/stride: {patch_size}/{patch_stride}")
    print(f"  elapsed setup: {time.time() - t_start:.1f}s")

    tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = SeverstalPatchDataset(train_patch_index, transform=tf)
    val_ds = SeverstalPatchDataset(val_patch_index, transform=tf)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    backbone = PatchCoreBackbone(cfg["patchcore"]["backbone"]).to(device)

    if os.path.exists(memory_path):
        print("[two-stage] Found existing coreset memory.npy, loading...")
        memory = np.load(memory_path)
    else:
        if os.path.exists(memory_pre_path):
            print("[two-stage] Found existing memory_precoreset.npy, loading...")
            memory_pre = np.load(memory_pre_path)
        else:
            t_memory = time.time()
            print("[two-stage] Building memory bank...")
            memory_pre = build_memory_bank(train_loader, device, backbone, cfg["patchcore"]["num_patches_per_sample"])
            np.save(memory_pre_path, memory_pre)
            print(f"[two-stage] Memory bank built: {len(memory_pre)} vectors in {time.time() - t_memory:.1f}s")
            print(f"[two-stage] Saved pre-coreset memory: {memory_pre_path}")

        t_coreset = time.time()
        print("[two-stage] Applying coreset subsampling...")
        memory = coreset_subsample(memory_pre, cfg["patchcore"]["coreset_sampling_ratio"], seed=cfg["seed"])
        np.save(memory_path, memory)
        print(f"[two-stage] Coreset done: {len(memory)} vectors in {time.time() - t_coreset:.1f}s")
        print(f"[two-stage] Saved coreset memory: {memory_path}")

    if os.path.exists(threshold_path):
        with open(threshold_path, "r") as f:
            threshold = float(json.load(f)["threshold"])
        print(f"[two-stage] Found existing threshold.json, skipping calibration (tau={threshold:.6f})")
    else:
        if os.path.exists(val_scores_path):
            print("[two-stage] Found existing val_scores.npy, loading...")
            val_scores = np.load(val_scores_path)
        else:
            t_val = time.time()
            print("[two-stage] Scoring validation patches for threshold calibration...")
            val_scores = infer_anomaly_scores(val_loader, device, backbone, memory)
            np.save(val_scores_path, val_scores)
            print(f"[two-stage] Validation scoring done in {time.time() - t_val:.1f}s")
            print(f"[two-stage] Saved validation scores: {val_scores_path}")

        threshold = calibrate_anomaly_threshold(val_scores, cfg["patchcore"]["accept_rate"])
        save_json(threshold_path, {"threshold": float(threshold)})
        print(f"[two-stage] Calibrated threshold (tau): {float(threshold):.6f}")
    save_json(os.path.join(out_dir, "meta.json"), {
        "source": "severstal",
        "severstal_root": sev_root,
        "backbone": cfg["patchcore"]["backbone"],
        "accept_rate": cfg["patchcore"]["accept_rate"],
        "num_patches_per_sample": cfg["patchcore"]["num_patches_per_sample"],
        "coreset_sampling_ratio": cfg["patchcore"]["coreset_sampling_ratio"],
        "normal_images_total": len(normal_image_paths),
        "normal_images_train": len(train_paths),
        "normal_images_val": len(val_paths),
        "patches_train": len(train_patch_index),
        "patches_val": len(val_patch_index),
        "patch_size": patch_size,
        "patch_stride": patch_stride,
        "val_ratio": val_ratio,
    })
    print(f"[two-stage] Saved outputs to: {out_dir}")
    print(f"[two-stage] Total elapsed: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
