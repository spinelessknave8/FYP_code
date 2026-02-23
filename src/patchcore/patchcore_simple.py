import random
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm


class PatchCoreBackbone(nn.Module):
    def __init__(self, backbone="resnet18"):
        super().__init__()
        if backbone == "resnet18":
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Use layers up to layer3 for mid-level features
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
        )

    def forward(self, x):
        return self.features(x)


def extract_patches(feat_map: torch.Tensor):
    # feat_map: [B, C, H, W] -> patches [B*H*W, C]
    b, c, h, w = feat_map.shape
    patches = feat_map.permute(0, 2, 3, 1).reshape(b * h * w, c)
    return patches


def build_memory_bank(dataloader, device, backbone, num_patches_per_sample: int):
    backbone.eval()
    memory = []
    with torch.no_grad():
        for x in tqdm(dataloader, desc="patchcore: build memory", leave=False):
            x = x.to(device)
            feat = backbone(x)
            patches = extract_patches(feat)
            patches = patches.cpu().numpy()
            if num_patches_per_sample < len(patches):
                idx = np.random.choice(len(patches), num_patches_per_sample, replace=False)
                patches = patches[idx]
            memory.append(patches)
    memory = np.concatenate(memory, axis=0)
    return memory


def coreset_subsample(memory: np.ndarray, ratio: float, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(memory)
    k = max(1, int(n * ratio))
    idx = rng.choice(n, k, replace=False)
    return memory[idx]


def anomaly_score(patches: np.ndarray, memory: np.ndarray):
    # nearest neighbor distance for each patch
    # simple L2 distance
    dists = []
    for p in patches:
        diff = memory - p
        dd = np.sqrt(np.sum(diff * diff, axis=1))
        dists.append(np.min(dd))
    dists = np.array(dists)
    # aggregate using max distance
    return float(np.max(dists))


def infer_anomaly_scores(dataloader, device, backbone, memory):
    backbone.eval()
    scores = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="patchcore: score val", leave=False):
            # Support image-only tensors, (image, label) batches, and list[tensor] batches.
            x = batch
            if isinstance(batch, (tuple, list)):
                x = batch[0]
            if isinstance(x, (tuple, list)):
                if len(x) == 0:
                    continue
                if isinstance(x[0], torch.Tensor):
                    x = torch.stack(list(x), dim=0)
                else:
                    raise TypeError(f"Unsupported batch image type: {type(x[0])}")
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"Unsupported batch type for images: {type(x)}")
            x = x.to(device)
            feat = backbone(x)
            b = feat.shape[0]
            for i in range(b):
                patches = extract_patches(feat[i : i + 1]).cpu().numpy()
                score = anomaly_score(patches, memory)
                scores.append(score)
    return np.array(scores)
