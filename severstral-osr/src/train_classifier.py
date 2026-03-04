import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from common import load_config, split_name_from_config
from data import ImageLabelDataset, collect_single_label_defect_samples, stratified_split

from src.models.resnet50 import build_resnet50
from src.utils.device import get_device
from src.utils.io import save_json
from src.utils.plots import plot_training_curves
from src.utils.seed import set_seed

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def train_one_epoch(model, loader, device, optimizer, criterion):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, total_correct / total


def eval_one_epoch(model, loader, device, criterion):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)
    return total_loss / total, total_correct / total


def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])

    split_name = split_name_from_config(config_path)
    out_dir = os.path.join(cfg["output_dir"], split_name)
    clf_dir = os.path.join(out_dir, "classifier")
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(clf_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    known_classes = cfg["known_classes"]
    data_root = cfg["severstal"]["data_root"]
    train_csv = cfg["severstal"].get("train_csv", "train.csv")
    images_dir = cfg["severstal"].get("images_dir", "train_images")

    samples = collect_single_label_defect_samples(data_root, train_csv, images_dir)
    known_samples = [s for s in samples if s[1] in known_classes]

    train_samples, val_samples, test_samples = stratified_split(
        known_samples,
        train_ratio=cfg["splits"]["train_ratio"],
        val_ratio=cfg["splits"]["val_ratio"],
        seed=cfg["seed"],
    )

    class_to_idx = {c: i for i, c in enumerate(known_classes)}

    tf = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = ImageLabelDataset(train_samples, class_to_idx=class_to_idx, transform=tf)
    val_ds = ImageLabelDataset(val_samples, class_to_idx=class_to_idx, transform=tf)
    test_ds = ImageLabelDataset(test_samples, class_to_idx=class_to_idx, transform=tf)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    device = get_device(cfg["device"])
    model = build_resnet50(num_classes=len(known_classes), pretrained=cfg["resnet"]["pretrained"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["resnet"]["lr"]),
        weight_decay=float(cfg["resnet"]["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()

    epochs = int(cfg["resnet"]["epochs"])
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val = -1.0
    best_state = None

    for epoch in range(epochs):
        tl, ta = train_one_epoch(model, train_loader, device, optimizer, criterion)
        vl, va = eval_one_epoch(model, val_loader, device, criterion)
        train_losses.append(tl)
        val_losses.append(vl)
        train_accs.append(ta)
        val_accs.append(va)
        print(f"epoch {epoch+1}/{epochs} train_acc={ta:.3f} val_acc={va:.3f}")
        if va > best_val:
            best_val = va
            best_state = {"model_state": model.state_dict(), "class_to_idx": class_to_idx}

    if best_state is None:
        best_state = {"model_state": model.state_dict(), "class_to_idx": class_to_idx}

    model.load_state_dict(best_state["model_state"])
    torch.save(best_state, os.path.join(clf_dir, "classifier.pt"))
    plot_training_curves(plot_dir, train_losses, val_losses, train_accs, val_accs)

    _, test_acc = eval_one_epoch(model, test_loader, device, criterion)
    save_json(os.path.join(clf_dir, "train_metrics.json"), {"best_val_acc": best_val, "test_acc": test_acc})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
