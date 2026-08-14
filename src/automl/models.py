"""Backbone construction with pretrained weights, channel adaptation, partial
freezing and a dropout classification head.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _replace_head(model, in_features, num_classes, dropout):
    import torch.nn as nn
    return nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))


def build_model(
    backbone: str,
    num_classes: int,
    channels: int,
    dropout: float = 0.0,
    frozen_stage: int = 0,
    pretrained: bool = True,
):
    """Return a torchvision backbone adapted for the task.

    frozen_stage: 0 = train all params; 1..3 = freeze increasing fractions of the
    feature extractor (coarse but effective on small data).
    """
    import torch.nn as nn
    import torchvision.models as tvm

    weights_arg = "DEFAULT" if pretrained else None

    if backbone == "resnet18":
        model = tvm.resnet18(weights=weights_arg)
        if channels == 1:
            _adapt_first_conv(model.conv1)
        in_features = model.fc.in_features
        model.fc = _replace_head(model, in_features, num_classes, dropout)
        feature_blocks = [model.conv1, model.layer1, model.layer2, model.layer3, model.layer4]

    elif backbone == "efficientnet_b0":
        model = tvm.efficientnet_b0(weights=weights_arg)
        if channels == 1:
            _adapt_first_conv(model.features[0][0])
        in_features = model.classifier[1].in_features
        model.classifier = _replace_head(model, in_features, num_classes, dropout)
        feature_blocks = list(model.features)

    elif backbone == "mobilenetv3":
        model = tvm.mobilenet_v3_small(weights=weights_arg)
        if channels == 1:
            _adapt_first_conv(model.features[0][0])
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        # add dropout in front of the final linear
        model.classifier.insert(-1, nn.Dropout(p=dropout))
        feature_blocks = list(model.features)

    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    _apply_freezing(feature_blocks, frozen_stage)
    return model


def _adapt_first_conv(conv):
    """Average pretrained RGB filters to a single input channel."""
    import torch
    import torch.nn as nn

    if not isinstance(conv, nn.Conv2d):
        return
    if conv.in_channels == 1:
        return
    w = conv.weight.data
    new_w = w.mean(dim=1, keepdim=True)
    conv.in_channels = 1
    conv.weight = nn.Parameter(new_w)


def _apply_freezing(feature_blocks, frozen_stage: int) -> None:
    """Freeze the first `frac` of feature blocks based on frozen_stage in [0,3]."""
    if frozen_stage <= 0:
        return
    n = len(feature_blocks)
    frac = {1: 0.25, 2: 0.5, 3: 0.75}.get(frozen_stage, 0.0)
    n_freeze = int(round(n * frac))
    for block in feature_blocks[:n_freeze]:
        for p in block.parameters():
            p.requires_grad = False
