from typing import Callable, Optional

import torch.nn as nn
from torchvision import models


def get_model(
    model_arch: str,
    num_classes: Optional[int] = None,
    logger: Optional[Callable] = None,
) -> nn.Module:
    """
    Tworzy model o podanej architekturze.

    Args:
        model_arch: Architektura modelu (np. 'efficientnet_b0')
        num_classes: Liczba klas (opcjonalnie)
        logger: Funkcja do logowania (opcjonalnie)

    Returns:
        nn.Module: Model PyTorch
    """
    if logger:
        logger(f"\nKonfiguracja modelu:")
        logger(f"- Architektura: {model_arch}")
        if num_classes:
            logger(f"- Liczba klas: {num_classes}")

    # Mapowanie nazw architektur na funkcje tworzące modele
    model_factories = {
        "efficientnet_b0": lambda: models.efficientnet_b0(pretrained=True),
        "efficientnet_b1": lambda: models.efficientnet_b1(pretrained=True),
        "efficientnet_b2": lambda: models.efficientnet_b2(pretrained=True),
        "efficientnet_b3": lambda: models.efficientnet_b3(pretrained=True),
        "efficientnet_b4": lambda: models.efficientnet_b4(pretrained=True),
        "efficientnet_b5": lambda: models.efficientnet_b5(pretrained=True),
        "efficientnet_b6": lambda: models.efficientnet_b6(pretrained=True),
        "efficientnet_b7": lambda: models.efficientnet_b7(pretrained=True),
        "resnet18": lambda: models.resnet18(pretrained=True),
        "resnet34": lambda: models.resnet34(pretrained=True),
        "resnet50": lambda: models.resnet50(pretrained=True),
        "resnet101": lambda: models.resnet101(pretrained=True),
        "resnet152": lambda: models.resnet152(pretrained=True),
        "mobilenet_v2": lambda: models.mobilenet_v2(pretrained=True),
        "mobilenet_v3_large": lambda: models.mobilenet_v3_large(pretrained=True),
        "mobilenet_v3_small": lambda: models.mobilenet_v3_small(pretrained=True),
        "convnext_tiny": lambda: models.convnext_tiny(pretrained=True),
        "convnext_small": lambda: models.convnext_small(pretrained=True),
        "convnext_base": lambda: models.convnext_base(pretrained=True),
        "convnext_large": lambda: models.convnext_large(pretrained=True),
        "vit_b_16": lambda: models.vit_b_16(pretrained=True),
        "vit_b_32": lambda: models.vit_b_32(pretrained=True),
        "vit_l_16": lambda: models.vit_l_16(pretrained=True),
        "vit_l_32": lambda: models.vit_l_32(pretrained=True),
    }

    if model_arch not in model_factories:
        raise ValueError(f"Nieznana architektura modelu: {model_arch}")

    # Utwórz model
    model = model_factories[model_arch]()

    # Dostosuj liczbę klas jeśli podano
    if num_classes is not None:
        if hasattr(model, "fc"):  # ResNet
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif hasattr(model, "classifier"):  # EfficientNet, MobileNet, ConvNeXt
            if isinstance(model.classifier, nn.Sequential):
                in_features = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(in_features, num_classes)
            else:
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, num_classes)
        elif hasattr(model, "heads"):  # ViT
            in_features = model.heads.head.in_features
            model.heads.head = nn.Linear(in_features, num_classes)

    return model
