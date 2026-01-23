import torch
import torch.nn as nn


class EmbeddingExtractor(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.features = None
        self._hook = self.model.avgpool.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, inputs, outputs):
        self.features = outputs

    def forward(self, x):
        _ = self.model(x)
        feats = self.features
        feats = torch.flatten(feats, 1)
        return feats

    def close(self):
        if self._hook:
            self._hook.remove()
