"""
model.py — helpers to load BDAPPV baseline models from HuggingFace Hub.

Usage:
    from huggingface_hub import hf_hub_download
    import importlib.util, sys

    path = hf_hub_download("gabrielkasmi/bdappv-models", "model.py")
    spec = importlib.util.spec_from_file_location("bdappv_model", path)
    mod  = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

    seg = mod.load_segmentation_model("google")   # or "ign"
    clf = mod.load_classification_model("google") # or "ign"
"""

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models import inception_v3

REPO = "gabrielkasmi/bdappv-models"


def load_segmentation_model(provider: str = "google", device: str = "cpu"):
    """
    Load the DeepLabV3-ResNet101 segmentation model.

    Args:
        provider : "google" or "ign"
        device   : "cpu", "cuda", or "mps"

    Returns:
        model in eval mode
    """
    assert provider in ("google", "ign"), "provider must be 'google' or 'ign'"

    path  = hf_hub_download(REPO, f"deeplab_{provider}_best.pth")
    model = deeplabv3_resnet101(weights=None, aux_loss=False)
    model.classifier[-1] = nn.Conv2d(256, 1, kernel_size=1)

    state      = torch.load(path, map_location=device, weights_only=False)
    model_dict = model.state_dict()
    compatible = {k: v for k, v in state.items()
                  if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(compatible)
    model.load_state_dict(model_dict)

    return model.eval().to(device)


def load_classification_model(provider: str = "google", device: str = "cpu"):
    """
    Load the InceptionV3 classification model (panel / no panel).

    Args:
        provider : "google" or "ign"
        device   : "cpu", "cuda", or "mps"

    Returns:
        model in eval mode
    """
    assert provider in ("google", "ign"), "provider must be 'google' or 'ign'"

    path  = hf_hub_download(REPO, f"inception_{provider}_best.pth")
    model = inception_v3(weights=None, aux_logits=True)
    model.fc           = nn.Linear(model.fc.in_features, 1)
    model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, 1)

    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.aux_logits = False  # désactive pour l'inférence

    return model.eval().to(device)
