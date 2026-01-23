import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


def build_resnet50(num_classes: int, pretrained: bool = True) -> nn.Module:
    if pretrained:
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
    else:
        model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
