Cześć! Przeanalizowałem kod pakietu AI do klasyfikacji obrazów. Na podstawie twoich wcześniejszych pytań i przesłanych plików, zidentyfikowałem kilka problemów, które mogą przyczyniać się do przeuczenia modelu. Przedstawię teraz propozycje zmian, które pomogą poprawić stosunek dokładności treningowej do walidacyjnej (87.42% vs 68.86%).
Zmiany w pliku ai/preprocessing.py:
python# Zmiana w funkcji get_augmentation_transforms()

def get_augmentation_transforms(image_size=(224, 224)):
"""
Zwraca przekształcenia z augmentacją danych dla treningu.

    Args:
        image_size: Krotka (szerokość, wysokość)

    Returns:
        Obiekt transforms.Compose z przekształceniami
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),  # Zmiana parametru scale
        transforms.RandomHorizontalFlip(p=0.7),  # Zwiększenie prawdopodobieństwa
        transforms.RandomVerticalFlip(p=0.3),  # Dodano odbicie pionowe
        transforms.RandomRotation(25),  # Dodano rotację
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),  # Zwiększone parametry
        transforms.RandomGrayscale(p=0.1),  # Dodano konwersję do skali szarości
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),  # Dodano perspektywę
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),  # Dodano usuwanie części obrazu
    ])

Zmiany w pliku ai/models.py:
python# Dodanie funkcji dla EfficientNetV2 i modyfikacja istniejącej funkcji get_model()

from typing import Callable, Optional
import torch.nn as nn
from torchvision import models

def get_model(
model_arch: str,
num_classes: Optional[int] = None,
logger: Optional[Callable] = None,
drop_connect_rate: float = 0.2,
dropout_rate: float = 0.3, # Dodany parametr dropout_rate
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
            logger(f"- Dropout rate: {dropout_rate}")  # Dodany log

    # Mapowanie nazw architektur na funkcje tworzące modele
    model_factories = {
        # Dotychczasowe modele...

        # Dodane modele EfficientNetV2
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

    # Reszta funkcji bez zmian...

    # Modyfikacja warstwy klasyfikacyjnej dla zwiększenia regularyzacji
    if num_classes is not None:
        if hasattr(model, "fc"):  # ResNet
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(in_features, 1024),  # Dodana warstwa pośrednia
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Dropout(dropout_rate),
                nn.Linear(1024, num_classes)
            )
        elif hasattr(model, "classifier"):  # EfficientNet, MobileNet, ConvNeXt
            if isinstance(model.classifier, nn.Sequential):
                in_features = model.classifier[-1].in_features
                model.classifier[-1] = nn.Sequential(
                    nn.Dropout(dropout_rate),
                    nn.Linear(in_features, num_classes)
                )
            else:
                in_features = model.classifier.in_features
                model.classifier = nn.Sequential(
                    nn.Dropout(dropout_rate),
                    nn.Linear(in_features, 1024),  # Dodana warstwa pośrednia
                    nn.ReLU(),
                    nn.BatchNorm1d(1024),
                    nn.Dropout(dropout_rate),
                    nn.Linear(1024, num_classes)
                )
        elif hasattr(model, "heads"):  # ViT
            in_features = model.heads.head.in_features
            model.heads.head = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(in_features, num_classes)
            )

    return model

Zmiany w pliku ai/optimized_training.py:
python# Dodanie implementacji cross-validation

def train_model_optimized(
model,
train_dir,
val_dir=None,
num_epochs=10,
batch_size=None,
learning_rate=0.001,
device=None,
progress_callback=None,
freeze_backbone=False,
lr_scheduler_type="plateau",
early_stopping=True,
mixup=True, # Włączenie mixup
label_smoothing=0.1,
weight_decay=0.03, # Zwiększony weight_decay
optimizer_type="adamw",
profiler=None,
augmentation_mode="extended", # Zmiana domyślnej augmentacji
augmentation_params=None,
should_stop_callback=None,
use_cross_validation=False, # Dodany parametr cross-validation
k_folds=5, # Liczba foldów dla cross-validation
freeze_layers_ratio=0.7, # Zamrożenie części modelu dla transfer learning
):
"""
Trenuje model na podanym zbiorze danych z wykorzystaniem optymalnych parametrów sprzętowych.
""" # Istniejący kod ...

    # Dodać po konfiguracji modelu:
    if freeze_backbone:
        # print("Zamrażanie głównej części modelu...")
        if hasattr(model, "fc"):
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
        elif hasattr(model, "classifier"):
            for param in model.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif hasattr(model, "heads"):
            for param in model.parameters():
                param.requires_grad = False
            for param in model.heads.parameters():
                param.requires_grad = True
    elif freeze_layers_ratio > 0:
        # Zamrożenie określonego procentu warstw, zaczynając od początku
        # print(f"Zamrażanie {freeze_layers_ratio * 100}% warstw modelu...")
        parameters = list(model.parameters())
        num_to_freeze = int(len(parameters) * freeze_layers_ratio)

        for i, param in enumerate(parameters):
            if i < num_to_freeze:
                param.requires_grad = False
            else:
                param.requires_grad = True

    # Implementacja mixup (dodaj do głównej pętli treningowej)
    if mixup and alpha > 0:
        # Wewnątrz pętli treningu po batchach, przed optimizer.zero_grad():
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=0.2)

        # Wewnątrz forward pass:
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

Zmiany w pliku ai/classifier.py:
python# Dodanie implementacji bardziej odpornego klasyfikatora

# W metodzie \_create_model:

def \_create_model(self):
"""Tworzenie modelu bazowego z pretrenowanymi wagami"""
if self.model_type == "resnet50": # Istniejący kod...
elif self.model_type == "efficientnet":
model = models.efficientnet_b0(
weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
)

        # Zamiast prostej warstwy klasyfikatora, dodajemy bardziej złożoną:
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes),
        )
        # Zamrożenie warstw bazowych dla lepszej generalizacji
        for param in list(model.parameters())[:-10]:
            param.requires_grad = False
    # Reszta istniejącego kodu...

Zmiany w pliku ai/training.py:
python# Dodanie funkcji implementującej walidację krzyżową

def cross_validation_train(
model_arch: str,
train_dir: str,
epochs: int = 50,
batch_size: int = 64,
learning_rate: float = 0.0005, # Zmniejszony learning rate
optimizer: str = "adamw", # Zmiana na AdamW
scheduler: Optional[str] = "cosine",
n_folds: int = 5,
\*\*kwargs
):
"""
Trenuje model używając walidacji krzyżowej.

    Args:
        model_arch: Architektura modelu (np. 'efficientnet_b0')
        train_dir: Ścieżka do katalogu z danymi treningowymi
        epochs: Liczba epok treningu
        batch_size: Rozmiar batcha
        learning_rate: Współczynnik uczenia
        optimizer: Nazwa optymalizatora
        scheduler: Nazwa schedulera (opcjonalnie)
        n_folds: Liczba foldów dla walidacji krzyżowej
        **kwargs: Dodatkowe parametry dla funkcji train_model

    Returns:
        Dict: Wyniki treningu z walidacją krzyżową
    """
    from sklearn.model_selection import KFold
    import numpy as np

    # Wczytanie wszystkich danych
    all_data = ImageDataset(train_dir, transform=get_default_transforms())
    indices = np.arange(len(all_data))

    # Implementacja KFold
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []
    best_val_loss = float('inf')
    best_model = None

    for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        print(f"\n--- Fold {fold+1}/{n_folds} ---")

        # Przygotowanie dataloaderów
        from torch.utils.data import Subset

        train_subset = Subset(all_data, train_idx)
        val_subset = Subset(all_data, val_idx)

        # Zastosowanie augmentacji tylko do zbioru treningowego
        train_subset.dataset.transform = get_augmentation_transforms()

        # Utwórz nowy model dla każdego foldu
        from .models import get_model
        model = get_model(
            model_arch,
            num_classes=len(all_data.classes),
            drop_connect_rate=0.3,  # Zwiększony drop connect
            dropout_rate=0.3,  # Wyższy dropout
        )

        # Trenuj model
        result = train_model(
            model=model,
            train_dataset=train_subset,
            val_dataset=val_subset,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            optimizer=optimizer,
            scheduler=scheduler,
            **kwargs
        )

        fold_results.append(result)

        # Zapisz najlepszy model
        if result['best_val_loss'] < best_val_loss:
            best_val_loss = result['best_val_loss']
            best_model = model

    # Oblicz średnie metryki
    avg_metrics = {
        'val_acc': np.mean([r['history']['val_acc'][-1] for r in fold_results]),
        'val_loss': np.mean([r['history']['val_loss'][-1] for r in fold_results]),
        'train_acc': np.mean([r['history']['train_acc'][-1] for r in fold_results]),
        'train_loss': np.mean([r['history']['train_loss'][-1] for r in fold_results]),
    }

    print("\n--- Podsumowanie walidacji krzyżowej ---")
    print(f"Średnia dokładność walidacyjna: {avg_metrics['val_acc']:.4f}")
    print(f"Średnia strata walidacyjna: {avg_metrics['val_loss']:.4f}")

    return {
        'best_model': best_model,
        'fold_results': fold_results,
        'avg_metrics': avg_metrics,
        'class_names': {str(i): cls for i, cls in enumerate(all_data.classes)}
    }

Podsumowanie rekomendowanych zmian:

Silniejsza augmentacja danych:

Dodanie szerszego zakresu transformacji (rotate, vertical flip, perspective)
Zwiększenie intensywności transformacji (większe zmiany jasności, kontrastu)
Dodanie random erasing dla symulacji okluzji

Ulepszony model:

Dodanie EfficientNetV2 jako nowszej i potencjalnie lepszej architektury
Bardziej złożony klasyfikator z warstwami pośrednimi
Dodanie batch normalization dla lepszej stabilności treningu

Techniki regularyzacji:

Zwiększony dropout (0.3-0.5) w warstwie klasyfikacyjnej
Wyższy weight decay (0.03) dla silniejszej regularyzacji L2
Implementacja mixup do syntezy nowych przykładów treningowych

Transfer learning:

Zamrożenie części warstw bazowych modelu (freeze_layers_ratio)
Zachowanie tylko wyższych warstw do dostrajania

Walidacja krzyżowa:

Implementacja k-fold cross-validation dla małych zbiorów danych
Lepsze wykorzystanie dostępnych danych treningowych i walidacyjnych

Optymalizacja hiperparametrów:

Zmniejszony learning rate (0.0005)
Zmiana optymalizatora na AdamW (lepszy z regularyzacją)
Zwiększony label smoothing (0.1)

Te zmiany powinny znacząco zmniejszyć różnicę między dokładnością treningową a walidacyjną, redukując problem przeuczenia modelu dla zaledwie 50 obrazów na klasę. Sugeruję przetestowanie tych zmian i monitorowanie postępu treningu.
