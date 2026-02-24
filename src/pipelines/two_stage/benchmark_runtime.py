import os
import argparse
import numpy as np
import torch
from ...utils.io import load_yaml, deep_update, save_json
from ...utils.device import get_device
from ...utils.timing import measure_time
from ...models.resnet50 import build_resnet50
from ...models.embedding import EmbeddingExtractor
from ...osr.gaussian_mahalanobis import fit_gaussians, min_mahalanobis
from ...patchcore.patchcore_simple import PatchCoreBackbone, extract_patches, anomaly_score


def load_config(config_path: str) -> dict:
    default_path = os.path.join("configs", "default.yaml")
    cfg = load_yaml(default_path)
    if os.path.basename(config_path) != "default.yaml":
        cfg = deep_update(cfg, load_yaml(config_path))
    return cfg


def split_name_from_config(config_path: str) -> str:
    name = os.path.splitext(os.path.basename(config_path))[0]
    if name.startswith("neu_"):
        name = name.replace("neu_", "")
    if name.endswith(".colab"):
        name = name[: -len(".colab")]
    return name


def main(config_path: str):
    cfg = load_config(config_path)
    device = get_device(cfg["device"])
    split_name = split_name_from_config(config_path)
    out_dir = os.path.join(cfg["output_dir"], split_name)

    known_classes = cfg.get("known_classes")
    if not known_classes:
        raise ValueError("known_classes not set in split config")

    # Classifier
    clf_dir = os.path.join(out_dir, "classifier")
    ckpt = torch.load(os.path.join(clf_dir, "classifier.pt"), map_location="cpu")
    model = build_resnet50(num_classes=len(known_classes), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    extractor = EmbeddingExtractor(model).to(device)

    # OSR params
    emb_dir = os.path.join(out_dir, "embeddings")
    known_train = np.load(os.path.join(emb_dir, "known_train.npz"))
    cov_reg_lambda = float(cfg["osr"]["cov_reg_lambda"])
    params = fit_gaussians(known_train["embeddings"], known_train["labels"], cov_reg_lambda)

    # PatchCore
    patchcore_dir = os.path.join(cfg["output_dir"], "patchcore")
    memory = np.load(os.path.join(patchcore_dir, "memory.npy"))
    backbone = PatchCoreBackbone(cfg["patchcore"]["backbone"]).to(device)
    backbone.eval()

    x = torch.randn(1, 3, cfg["image_size"], cfg["image_size"], device=device)

    def run_classifier():
        with torch.no_grad():
            _ = model(x)

    def run_osr():
        with torch.no_grad():
            emb = extractor(x)[0].cpu().numpy()
            _ = min_mahalanobis(emb, params)

    def run_patchcore():
        with torch.no_grad():
            feat = backbone(x)
            patches = extract_patches(feat).cpu().numpy()
            _ = anomaly_score(patches, memory)

    def run_cascade():
        with torch.no_grad():
            feat = backbone(x)
            patches = extract_patches(feat).cpu().numpy()
            score = anomaly_score(patches, memory)
            if score > 0.0:
                _ = model(x)

    rt = cfg.get("runtime", {"num_warmup": 5, "num_runs": 30})
    results = {
        "classifier_ms": measure_time(run_classifier, rt["num_warmup"], rt["num_runs"]),
        "osr_ms": measure_time(run_osr, rt["num_warmup"], rt["num_runs"]),
        "patchcore_ms": measure_time(run_patchcore, rt["num_warmup"], rt["num_runs"]),
        "cascade_ms": measure_time(run_cascade, rt["num_warmup"], rt["num_runs"]),
    }

    save_json(os.path.join(out_dir, "runtime.json"), results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
