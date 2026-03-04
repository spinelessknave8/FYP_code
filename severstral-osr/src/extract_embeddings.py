import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from common import load_config, split_name_from_config
from data import ImageLabelDataset, collect_single_label_defect_samples, stratified_split

from src.models.embedding import EmbeddingExtractor
from src.models.resnet50 import build_resnet50
from src.utils.device import get_device
from src.utils.seed import set_seed

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def extract_embeddings(model, extractor, loader, device, is_known=True):
    model.eval()
    extractor.eval()
    all_emb, all_logits, all_labels = [], [], []
    with torch.no_grad():
        for x, y in loader:
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

    split_name = split_name_from_config(config_path)
    out_dir = os.path.join(cfg["output_dir"], split_name)
    emb_dir = os.path.join(out_dir, "embeddings")
    clf_dir = os.path.join(out_dir, "classifier")
    os.makedirs(emb_dir, exist_ok=True)

    known_classes = cfg["known_classes"]
    data_root = cfg["severstal"]["data_root"]
    train_csv = cfg["severstal"].get("train_csv", "train.csv")
    images_dir = cfg["severstal"].get("images_dir", "train_images")

    samples = collect_single_label_defect_samples(data_root, train_csv, images_dir)
    known_samples = [s for s in samples if s[1] in known_classes]
    unknown_samples = [s for s in samples if s[1] not in known_classes]

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
    unk_ds = ImageLabelDataset(unknown_samples, class_to_idx=None, transform=tf)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    unk_loader = DataLoader(unk_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    ckpt = torch.load(os.path.join(clf_dir, "classifier.pt"), map_location="cpu")
    model = build_resnet50(num_classes=len(known_classes), pretrained=False)
    model.load_state_dict(ckpt["model_state"])

    device = get_device(cfg["device"])
    model = model.to(device)
    extractor = EmbeddingExtractor(model).to(device)

    emb_train, log_train, y_train = extract_embeddings(model, extractor, train_loader, device, True)
    emb_val, log_val, y_val = extract_embeddings(model, extractor, val_loader, device, True)
    emb_test, log_test, y_test = extract_embeddings(model, extractor, test_loader, device, True)
    emb_unk, log_unk, y_unk = extract_embeddings(model, extractor, unk_loader, device, False)

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
