import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from ..utils.io import load_yaml, deep_update
from ..utils.device import get_device
from ..utils.seed import set_seed
from ..datasets.neu import NEUListDataset, stratified_split, collect_neu_samples
from ..models.resnet50 import build_resnet50
from ..models.embedding import EmbeddingExtractor

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


def extract_embeddings(model, extractor, loader, device, is_known=True):
    model.eval()
    extractor.eval()
    all_emb, all_logits, all_labels = [], [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="extract", leave=False):
            x = x.to(device)
            logits = model(x)
            emb = extractor(x)
            all_emb.append(emb.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
            if is_known:
                all_labels.append(y.numpy())
            else:
                all_labels.append(-1 * np.ones(len(x), dtype=np.int64))
    return np.concatenate(all_emb), np.concatenate(all_logits), np.concatenate(all_labels)


def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])

    device = get_device(cfg["device"])
    split_name = split_name_from_config(config_path)
    out_dir = os.path.join(cfg["output_dir"], split_name)
    emb_dir = os.path.join(out_dir, "embeddings")
    clf_dir = os.path.join(out_dir, "classifier")
    os.makedirs(emb_dir, exist_ok=True)

    known_classes = cfg.get("known_classes")
    if not known_classes:
        raise ValueError("known_classes not set in split config")

    neu_root = cfg["neu"]["data_root"]
    class_mapping = cfg["neu"]["class_mapping"]
    splits = cfg["neu"].get("splits", ["train", "validation"])
    images_dir = cfg["neu"].get("images_dir", "images")

    samples = collect_neu_samples(neu_root, class_mapping, splits, images_dir=images_dir)
    known_samples = [s for s in samples if s[1] in known_classes]
    unknown_samples = [s for s in samples if s[1] not in known_classes]

    train_samples, val_samples, test_samples = stratified_split(
        known_samples, train_ratio=0.7, val_ratio=0.15, seed=cfg["seed"]
    )

    class_to_idx = {c: i for i, c in enumerate(known_classes)}

    eval_tf = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = NEUListDataset(train_samples, class_to_idx=class_to_idx, transform=eval_tf)
    val_ds = NEUListDataset(val_samples, class_to_idx=class_to_idx, transform=eval_tf)
    test_ds = NEUListDataset(test_samples, class_to_idx=class_to_idx, transform=eval_tf)
    unk_ds = NEUListDataset(unknown_samples, class_to_idx=None, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    unk_loader = DataLoader(unk_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    ckpt = torch.load(os.path.join(clf_dir, "classifier.pt"), map_location="cpu")
    model = build_resnet50(num_classes=len(known_classes), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    extractor = EmbeddingExtractor(model).to(device)

    print("extracting known_train embeddings...")
    emb_train, log_train, y_train = extract_embeddings(model, extractor, train_loader, device, is_known=True)
    print("extracting known_val embeddings...")
    emb_val, log_val, y_val = extract_embeddings(model, extractor, val_loader, device, is_known=True)
    print("extracting known_test embeddings...")
    emb_test, log_test, y_test = extract_embeddings(model, extractor, test_loader, device, is_known=True)
    print("extracting unknown_test embeddings...")
    emb_unk, log_unk, y_unk = extract_embeddings(model, extractor, unk_loader, device, is_known=False)

    np.savez(os.path.join(emb_dir, "known_train.npz"), embeddings=emb_train, logits=log_train, labels=y_train)
    np.savez(os.path.join(emb_dir, "known_val.npz"), embeddings=emb_val, logits=log_val, labels=y_val)
    np.savez(os.path.join(emb_dir, "known_test.npz"), embeddings=emb_test, logits=log_test, labels=y_test)
    np.savez(os.path.join(emb_dir, "unknown_test.npz"), embeddings=emb_unk, logits=log_unk, labels=y_unk)

    extractor.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
