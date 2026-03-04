import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms

from common import load_config, split_name_from_config
from data import ImageLabelDataset, collect_single_label_defect_samples, stratified_split

from src.models.resnet50 import build_resnet50
from src.patchcore.calibrate import calibrate_anomaly_threshold
from src.patchcore.patchcore_simple import PatchCoreBackbone, infer_anomaly_scores
from src.utils.device import get_device
from src.utils.io import save_json
from src.utils.seed import set_seed

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


def main(config_path: str):
    t0 = time.time()
    cfg = load_config(config_path)
    set_seed(cfg["seed"])

    split_name = split_name_from_config(config_path)
    out_dir = os.path.join(cfg["output_dir"], split_name)
    cascade_dir = os.path.join(out_dir, "cascade")
    os.makedirs(cascade_dir, exist_ok=True)

    known_classes = cfg["known_classes"]
    data_root = cfg["severstal"]["data_root"]
    train_csv = cfg["severstal"].get("train_csv", "train.csv")
    images_dir = cfg["severstal"].get("images_dir", "train_images")

    # Load stage1 artifacts
    patchcore_dir = os.path.join(cfg["output_dir"], "patchcore")
    memory = np.load(os.path.join(patchcore_dir, "memory.npy"))
    source_threshold = json.load(open(os.path.join(patchcore_dir, "threshold.json"), "r"))["threshold"]

    device = get_device(cfg["device"])
    backbone = PatchCoreBackbone(cfg["patchcore"]["backbone"]).to(device)

    # Load stage2 classifier
    clf_dir = os.path.join(out_dir, "classifier")
    ckpt = torch.load(os.path.join(clf_dir, "classifier.pt"), map_location="cpu")
    model = build_resnet50(num_classes=len(known_classes), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    kappa_path = os.path.join(out_dir, "osr", "thresholds.json")
    if os.path.exists(kappa_path):
        kappa = json.load(open(kappa_path, "r")).get("kappa")
    else:
        kappa = 0.0

    samples = collect_single_label_defect_samples(data_root, train_csv, images_dir)
    known_samples = [s for s in samples if s[1] in known_classes]
    unknown_samples = [s for s in samples if s[1] not in known_classes]

    _, known_val, known_test = stratified_split(
        known_samples,
        train_ratio=cfg["splits"]["train_ratio"],
        val_ratio=cfg["splits"]["val_ratio"],
        seed=cfg["seed"],
    )

    class_to_idx = {c: i for i, c in enumerate(known_classes)}
    tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    known_val_ds = ImageLabelDataset(known_val, class_to_idx=class_to_idx, transform=tf)
    known_ds = ImageLabelDataset(known_test, class_to_idx=class_to_idx, transform=tf)
    unk_ds = ImageLabelDataset(unknown_samples, class_to_idx=None, transform=tf)

    known_val_loader = DataLoader(known_val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    known_loader = DataLoader(known_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    unk_loader = DataLoader(unk_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    two_stage_cfg = cfg.get("two_stage", {})
    tau_source = two_stage_cfg.get("tau_source", "target_known_val")
    tau_accept_rate = float(two_stage_cfg.get("tau_accept_rate", 0.5))
    fusion_rule = str(two_stage_cfg.get("fusion_rule", "and")).lower()

    threshold = float(source_threshold)
    if tau_source == "target_known_val":
        known_val_scores = infer_anomaly_scores(known_val_loader, device, backbone, memory)
        np.save(os.path.join(cascade_dir, "stage1_known_val_scores.npy"), known_val_scores)
        threshold = float(calibrate_anomaly_threshold(known_val_scores, tau_accept_rate))
        save_json(os.path.join(cascade_dir, "stage1_tau.json"), {
            "tau_source": tau_source,
            "tau_accept_rate": tau_accept_rate,
            "threshold": threshold,
            "source_patchcore_threshold": float(source_threshold),
        })

    known_scores = infer_anomaly_scores(known_loader, device, backbone, memory)
    unk_scores = infer_anomaly_scores(unk_loader, device, backbone, memory)

    def get_confidences(loader):
        confs = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                logits = model(x)
                confs.append(np.max(softmax(logits.cpu().numpy()), axis=1))
        return np.concatenate(confs)

    conf_known = get_confidences(known_loader)
    conf_unknown = get_confidences(unk_loader)

    known_no_defect = known_scores <= threshold
    unk_no_defect = unk_scores <= threshold

    if fusion_rule == "and":
        known_reject = (~known_no_defect) & (conf_known < kappa)
        unk_reject = (~unk_no_defect) & (conf_unknown < kappa)
    elif fusion_rule == "or":
        known_reject = (~known_no_defect) | (conf_known < kappa)
        unk_reject = (~unk_no_defect) | (conf_unknown < kappa)
    else:
        raise ValueError(f"Unsupported fusion_rule: {fusion_rule}")

    known_mask = ~known_no_defect
    unk_mask = ~unk_no_defect

    if np.sum(known_mask) > 0 and np.sum(unk_mask) > 0:
        y_true = np.concatenate([np.zeros(np.sum(known_mask)), np.ones(np.sum(unk_mask))])
        scores = np.concatenate([1.0 - conf_known[known_mask], 1.0 - conf_unknown[unk_mask]])
        auroc_conf = roc_auc_score(y_true, scores)
    else:
        auroc_conf = float("nan")

    metrics = {
        "patchcore_threshold": float(threshold),
        "patchcore_threshold_source": tau_source,
        "patchcore_threshold_source_value": float(source_threshold),
        "patchcore_threshold_accept_rate": float(tau_accept_rate),
        "fusion_rule": fusion_rule,
        "kappa": float(kappa),
        "auroc_known_unknown_conf_conditional": float(auroc_conf),
        "tpr_unknown_conditional": float(np.mean(unk_reject[unk_mask])) if np.sum(unk_mask) > 0 else float("nan"),
        "fpr_known_conditional": float(np.mean(known_reject[known_mask])) if np.sum(known_mask) > 0 else float("nan"),
        "tpr_unknown_system": float(np.mean(unk_reject)),
        "fpr_known_system": float(np.mean(known_reject)),
        "stage1_pass_rate_known": float(np.mean(known_mask)),
        "stage1_pass_rate_unknown": float(np.mean(unk_mask)),
    }

    save_json(os.path.join(cascade_dir, "metrics.json"), metrics)
    np.save(os.path.join(cascade_dir, "known_scores.npy"), known_scores)
    np.save(os.path.join(cascade_dir, "unknown_scores.npy"), unk_scores)
    np.save(os.path.join(cascade_dir, "known_conf.npy"), conf_known)
    np.save(os.path.join(cascade_dir, "unknown_conf.npy"), conf_unknown)

    print(f"[severstal-cascade:{split_name}] done in {time.time() - t0:.1f}s")
    print(f"  stage1_pass_rate_known={metrics['stage1_pass_rate_known']:.4f}")
    print(f"  stage1_pass_rate_unknown={metrics['stage1_pass_rate_unknown']:.4f}")
    print(f"  tpr_unknown_system={metrics['tpr_unknown_system']:.4f} fpr_known_system={metrics['fpr_known_system']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
