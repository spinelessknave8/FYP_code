import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from ...utils.io import load_yaml, deep_update, save_json
from ...utils.seed import set_seed
from ...utils.device import get_device
from ...utils.plots import plot_training_curves
from ...datasets.neu import NEUListDataset, stratified_split, collect_neu_samples
from ...models.resnet50 import build_resnet50


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


def train_one_epoch(model, loader, device, optimizer, criterion):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    print(f"  training batches: {len(loader)}")
    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total += x.size(0)
        if (i + 1) % 50 == 0 or (i + 1) == len(loader):
            print(f"  batch {i+1}/{len(loader)} - loss {loss.item():.4f}")
    return total_loss / total, total_correct / total


def eval_one_epoch(model, loader, device, criterion):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    with torch.no_grad():
        print(f"  validation batches: {len(loader)}")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total += x.size(0)
    return total_loss / total, total_correct / total


def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])

    device = get_device(cfg["device"])
    split_name = split_name_from_config(config_path)
    out_dir = os.path.join(cfg["output_dir"], split_name)
    clf_dir = os.path.join(out_dir, "classifier")
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(clf_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    known_classes = cfg.get("known_classes")
    if not known_classes:
        raise ValueError("known_classes not set in split config")

    neu_root = cfg["neu"]["data_root"]
    class_mapping = cfg["neu"]["class_mapping"]
    splits = cfg["neu"].get("splits", ["train", "validation"])
    images_dir = cfg["neu"].get("images_dir", "images")

    samples = collect_neu_samples(neu_root, class_mapping, splits, images_dir=images_dir)
    known_samples = [s for s in samples if s[1] in known_classes]

    train_samples, val_samples, test_samples = stratified_split(
        known_samples, train_ratio=0.7, val_ratio=0.15, seed=cfg["seed"]
    )

    class_to_idx = {c: i for i, c in enumerate(known_classes)}

    train_tf = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = NEUListDataset(train_samples, class_to_idx=class_to_idx, transform=train_tf)
    val_ds = NEUListDataset(val_samples, class_to_idx=class_to_idx, transform=eval_tf)
    test_ds = NEUListDataset(test_samples, class_to_idx=class_to_idx, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    model = build_resnet50(num_classes=len(known_classes), pretrained=cfg["resnet"]["pretrained"]).to(device)
    lr = float(cfg["resnet"]["lr"])
    weight_decay = float(cfg["resnet"]["weight_decay"])
    epochs = int(cfg["resnet"]["epochs"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val = -1.0
    best_state = None
    start_epoch = 0
    checkpoint_last_path = os.path.join(clf_dir, "checkpoint_last.pt")

    if os.path.exists(checkpoint_last_path):
        ckpt = torch.load(checkpoint_last_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        train_losses = ckpt.get("train_losses", [])
        val_losses = ckpt.get("val_losses", [])
        train_accs = ckpt.get("train_accs", [])
        val_accs = ckpt.get("val_accs", [])
        best_val = float(ckpt.get("best_val", -1.0))
        best_state = ckpt.get("best_state")
        start_epoch = int(ckpt.get("epoch", 0))
        print(f"[classifier:{split_name}] Resuming from epoch {start_epoch+1}/{epochs}")

    for epoch in range(start_epoch, epochs):
        print(f"epoch {epoch+1}/{epochs} - start")
        tl, ta = train_one_epoch(model, train_loader, device, optimizer, criterion)
        vl, va = eval_one_epoch(model, val_loader, device, criterion)
        train_losses.append(tl)
        val_losses.append(vl)
        train_accs.append(ta)
        val_accs.append(va)
        if va > best_val:
            best_val = va
            best_state = {"model_state": model.state_dict(), "class_to_idx": class_to_idx}
        print(f"epoch {epoch+1}/{epochs} - train_acc {ta:.3f} val_acc {va:.3f}")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val": best_val,
                "best_state": best_state,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "train_accs": train_accs,
                "val_accs": val_accs,
            },
            checkpoint_last_path,
        )

    if best_state is None:
        best_state = {"model_state": model.state_dict(), "class_to_idx": class_to_idx}

    torch.save(best_state, os.path.join(clf_dir, "classifier.pt"))

    plot_training_curves(plot_dir, train_losses, val_losses, train_accs, val_accs)

    test_loss, test_acc = eval_one_epoch(model, test_loader, device, criterion)
    save_json(os.path.join(clf_dir, "train_metrics.json"), {
        "best_val_acc": best_val,
        "test_acc": test_acc,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
