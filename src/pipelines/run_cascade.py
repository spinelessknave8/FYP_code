import os
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score

from ..utils.io import load_yaml, deep_update, save_json
from ..utils.device import get_device
from ..utils.seed import set_seed
from ..datasets.neu import NEUListDataset, stratified_split, collect_neu_samples
from ..models.resnet50 import build_resnet50
from ..patchcore.patchcore_simple import PatchCoreBackbone, infer_anomaly_scores

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_config(config_path: str) -> dict:
    default_path = os.path.join("configs", "default.yaml")
    cfg = load_yaml(default_path)
    if os.path.basename(config_path) != "default.yaml":
        cfg = deep_update(cfg, load_yaml(config_path))
    return cfg


def split_name_from_config(config_path: str) -> str:
    name = os.path.splitext(os.path.basename(config_path))[0]
    if name.startswith("neu_"):
        return name.replace("neu_", "")
    return name


def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])

    device = get_device(cfg["device"])
    split_name = split_name_from_config(config_path)
    out_dir = os.path.join(cfg["output_dir"], split_name)
    cascade_dir = os.path.join(out_dir, "cascade")
    os.makedirs(cascade_dir, exist_ok=True)

    known_classes = cfg.get("known_classes")
    if not known_classes:
        raise ValueError("known_classes not set in split config")

    # Load patchcore
    patchcore_dir = os.path.join(cfg["output_dir"], "patchcore")
    memory = np.load(os.path.join(patchcore_dir, "memory.npy"))
    threshold = json.load(open(os.path.join(patchcore_dir, "threshold.json"), "r"))["threshold"]
    backbone = PatchCoreBackbone(cfg["patchcore"]["backbone"]).to(device)

    # Load classifier
    clf_dir = os.path.join(out_dir, "classifier")
    ckpt = torch.load(os.path.join(clf_dir, "classifier.pt"), map_location="cpu")
    model = build_resnet50(num_classes=len(known_classes), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    # Load kappa if exists
    kappa_path = os.path.join(out_dir, "osr", "thresholds.json")
    if os.path.exists(kappa_path):
        kappa = json.load(open(kappa_path, "r")).get("kappa")
    else:
        kappa = None

    neu_root = cfg["neu"]["data_root"]
    class_mapping = cfg["neu"]["class_mapping"]
    splits = cfg["neu"].get("splits", ["train", "validation"])
    images_dir = cfg["neu"].get("images_dir", "images")

    samples = collect_neu_samples(neu_root, class_mapping, splits, images_dir=images_dir)
    known_samples = [s for s in samples if s[1] in known_classes]
    unknown_samples = [s for s in samples if s[1] not in known_classes]

    _, _, known_test = stratified_split(known_samples, train_ratio=0.7, val_ratio=0.15, seed=cfg["seed"])

    class_to_idx = {c: i for i, c in enumerate(known_classes)}

    tf = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    known_ds = NEUListDataset(known_test, class_to_idx=class_to_idx, transform=tf)
    unk_ds = NEUListDataset(unknown_samples, class_to_idx=None, transform=tf)

    known_loader = DataLoader(known_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    unk_loader = DataLoader(unk_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    # PatchCore anomaly scores
    known_scores = infer_anomaly_scores(known_loader, device, backbone, memory)
    unk_scores = infer_anomaly_scores(unk_loader, device, backbone, memory)

    # Classifier confidences
    def get_confidences(loader):
        confs = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                logits = model(x)
                conf = np.max(softmax(logits.cpu().numpy()), axis=1)
                confs.append(conf)
        return np.concatenate(confs)

    conf_known = get_confidences(known_loader)
    conf_unknown = get_confidences(unk_loader)

    # Cascade decision
    if kappa is None:
        kappa = 0.0

    known_no_defect = known_scores <= threshold
    unk_no_defect = unk_scores <= threshold

    known_reject = (~known_no_defect) & (conf_known < kappa)
    unk_reject = (~unk_no_defect) & (conf_unknown < kappa)

    # For NEU (all defects), we ignore "no_defect" in evaluation
    known_mask = ~known_no_defect
    unk_mask = ~unk_no_defect

    if np.sum(known_mask) > 0 and np.sum(unk_mask) > 0:
        y_true = np.concatenate([np.zeros(np.sum(known_mask)), np.ones(np.sum(unk_mask))])
        scores = np.concatenate([1.0 - conf_known[known_mask], 1.0 - conf_unknown[unk_mask]])
        auroc_conf = roc_auc_score(y_true, scores)
    else:
        auroc_conf = float("nan")

    tpr_unknown = float(np.mean(unk_reject[unk_mask])) if np.sum(unk_mask) > 0 else float("nan")
    fpr_known = float(np.mean(known_reject[known_mask])) if np.sum(known_mask) > 0 else float("nan")

    metrics = {
        "patchcore_threshold": float(threshold),
        "kappa": float(kappa),
        "auroc_known_unknown_conf": float(auroc_conf),
        "tpr_unknown": tpr_unknown,
        "fpr_known": fpr_known,
        "ignored_no_defect_known": int(np.sum(known_no_defect)),
        "ignored_no_defect_unknown": int(np.sum(unk_no_defect)),
    }

    save_json(os.path.join(cascade_dir, "metrics.json"), metrics)

    global_metrics_path = os.path.join(out_dir, "metrics.json")
    if os.path.exists(global_metrics_path):
        with open(global_metrics_path, "r") as f:
            global_metrics = json.load(f)
    else:
        global_metrics = {}
    global_metrics["cascade"] = metrics
    save_json(global_metrics_path, global_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
