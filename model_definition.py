"""VGG model definition aligned with checkpoints saved from torchvision VGG."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import vgg16


class FederatedVGG(nn.Module):
    """Model keys match torchvision VGG (`features.*`, `classifier.*`)."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.vgg = vgg16(weights=None)
        in_features = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Linear(in_features, num_classes)

    def load_state_dict(self, state_dict, strict: bool = True):
        """Handle checkpoints saved with optional DataParallel `module.` prefix."""
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        return self.vgg.load_state_dict(state_dict, strict=strict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vgg(x)
