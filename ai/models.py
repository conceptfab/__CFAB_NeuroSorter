from typing import Callable, Optional

import torch.nn as nn
from torchvision import models


def get_model(
    model_arch: str,
    num_classes: Optional[int] = None,
    logger: Optional[Callable] = None,
    drop_connect_rate: float = 0.2,
    dropout_rate: float = 0.3,
) -> nn.Module:
    """
    Tworzy model o podanej architekturze.

    Args:
        model_arch: Architektura modelu (np. 'efficientnet_b0')
        num_classes: Liczba klas (opcjonalnie)
        logger: Funkcja do logowania (opcjonalnie)
        drop_connect_rate: Wartość drop connect rate dla EfficientNet (opcjonalnie)
        dropout_rate: Wartość dropout rate dla warstw klasyfikacji (opcjonalnie)

    Returns:
        nn.Module: Model PyTorch
    """
    if logger:
        logger(f"\nKonfiguracja modelu:")
        logger(f"- Architektura: {model_arch}")
        if num_classes:
            logger(f"- Liczba klas: {num_classes}")
        if model_arch.startswith("efficientnet"):
            logger(f"- Drop connect rate: {drop_connect_rate}")
            logger(f"- Dropout rate: {dropout_rate}")

    # Mapowanie nazw architektur na funkcje tworzące modele
    model_factories = {
        "efficientnet_b0": lambda: models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1,
            drop_rate=drop_connect_rate,
        ),
        "efficientnet_b1": lambda: models.efficientnet_b1(
            weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1,
            drop_rate=drop_connect_rate,
        ),
        "efficientnet_b2": lambda: models.efficientnet_b2(
            weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1,
            drop_rate=drop_connect_rate,
        ),
        "efficientnet_b3": lambda: models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1,
            drop_rate=drop_connect_rate,
        ),
        "efficientnet_b4": lambda: models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1,
            drop_rate=drop_connect_rate,
        ),
        "efficientnet_b5": lambda: models.efficientnet_b5(
            weights=models.EfficientNet_B5_Weights.IMAGENET1K_V1,
            drop_rate=drop_connect_rate,
        ),
        "efficientnet_b6": lambda: models.efficientnet_b6(
            weights=models.EfficientNet_B6_Weights.IMAGENET1K_V1,
            drop_rate=drop_connect_rate,
        ),
        "efficientnet_b7": lambda: models.efficientnet_b7(
            weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1,
            drop_rate=drop_connect_rate,
        ),
        "resnet18": lambda: models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        ),
        "resnet34": lambda: models.resnet34(
            weights=models.ResNet34_Weights.IMAGENET1K_V1
        ),
        "resnet50": lambda: models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
        ),
        "resnet101": lambda: models.resnet101(
            weights=models.ResNet101_Weights.IMAGENET1K_V1
        ),
        "resnet152": lambda: models.resnet152(
            weights=models.ResNet152_Weights.IMAGENET1K_V1
        ),
        "mobilenet_v2": lambda: models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
        ),
        "mobilenet_v3_large": lambda: models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
        ),
        "mobilenet_v3_small": lambda: models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        ),
        "convnext_tiny": lambda: models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        ),
        "convnext_small": lambda: models.convnext_small(
            weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1
        ),
        "convnext_base": lambda: models.convnext_base(
            weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1
        ),
        "convnext_large": lambda: models.convnext_large(
            weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1
        ),
        "vit_b_16": lambda: models.vit_b_16(
            weights=models.ViT_B_16_Weights.IMAGENET1K_V1
        ),
        "vit_b_32": lambda: models.vit_b_32(
            weights=models.ViT_B_32_Weights.IMAGENET1K_V1
        ),
        "vit_l_16": lambda: models.vit_l_16(
            weights=models.ViT_L_16_Weights.IMAGENET1K_V1
        ),
        "vit_l_32": lambda: models.vit_l_32(
            weights=models.ViT_L_32_Weights.IMAGENET1K_V1
        ),
        "efficientnetv2_s": lambda: models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1,
            dropout=dropout_rate,
        ),
        "efficientnetv2_m": lambda: models.efficientnet_v2_m(
            weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1,
            dropout=dropout_rate,
        ),
        "efficientnetv2_l": lambda: models.efficientnet_v2_l(
            weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1,
            dropout=dropout_rate,
        ),
    }

    if model_arch not in model_factories:
        raise ValueError(f"Nieznana architektura modelu: {model_arch}")

    # Utwórz model
    model = model_factories[model_arch]()

    # Dostosuj liczbę klas jeśli podano
    if num_classes is not None:
        if hasattr(model, "fc"):  # ResNet
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(in_features, 1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Dropout(dropout_rate),
                nn.Linear(1024, num_classes),
            )
        elif hasattr(model, "classifier"):  # EfficientNet, MobileNet, ConvNeXt
            if isinstance(model.classifier, nn.Sequential):
                in_features = model.classifier[-1].in_features
                model.classifier[-1] = nn.Sequential(
                    nn.Dropout(dropout_rate), nn.Linear(in_features, num_classes)
                )
            else:
                in_features = model.classifier.in_features
                model.classifier = nn.Sequential(
                    nn.Dropout(dropout_rate),
                    nn.Linear(in_features, 1024),
                    nn.ReLU(),
                    nn.BatchNorm1d(1024),
                    nn.Dropout(dropout_rate),
                    nn.Linear(1024, num_classes),
                )
        elif hasattr(model, "heads"):  # ViT
            in_features = model.heads.head.in_features
            model.heads.head = nn.Sequential(
                nn.Dropout(dropout_rate), nn.Linear(in_features, num_classes)
            )

    return model
