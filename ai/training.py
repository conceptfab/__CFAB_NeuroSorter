import datetime
import json
import logging
import os
import time
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms

from .preprocessing import (
    get_augmentation_transforms,
    get_default_transforms,
    get_extended_augmentation_transforms,
)

# Konfiguracja loggera
logger = logging.getLogger(__name__)


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Dataset do trenowania modelu.

        Args:
            root_dir: Katalog z podkatalogami klas
            transform: Opcjonalne przekształcenia
        """
        # Sprawdź czy katalog istnieje
        if not os.path.exists(root_dir):
            error_msg = f"Katalog {root_dir} nie istnieje."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Sprawdź czy katalog zawiera podkatalogi
        has_subdirs = any(
            os.path.isdir(os.path.join(root_dir, item)) for item in os.listdir(root_dir)
        )

        if not has_subdirs:
            error_msg = f"Katalog {root_dir} nie zawiera podkatalogów z klasami."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Sprawdź czy istnieją jakiekolwiek obrazy w podkatalogach
        found_images = False
        supported_extensions = (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".ppm",
            ".pgm",
            ".tif",
            ".tiff",
            ".webp",
        )

        for item in os.listdir(root_dir):
            item_path = os.path.join(root_dir, item)
            if os.path.isdir(item_path):
                files = os.listdir(item_path)
                if not files:
                    logger.warning(f"Katalog {item_path} jest pusty")
                    continue

                valid_files = [
                    f for f in files if f.lower().endswith(supported_extensions)
                ]
                if not valid_files:
                    logger.warning(
                        f"W katalogu {item_path} nie znaleziono plików o obsługiwanych rozszerzeniach. "
                        f"Obsługiwane rozszerzenia: {', '.join(supported_extensions)}"
                    )
                else:
                    found_images = True
                    logger.info(
                        f"Znaleziono {len(valid_files)} plików w katalogu {item_path}"
                    )

        if not found_images:
            error_msg = (
                f"Nie znaleziono obrazów w podkatalogach katalogu {root_dir}. "
                f"Obsługiwane rozszerzenia: {', '.join(supported_extensions)}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            self.dataset = datasets.ImageFolder(root_dir, transform=transform)
            logger.info(
                f"Utworzono dataset z {len(self.dataset)} obrazami w {len(self.dataset.classes)} klasach"
            )
        except Exception as e:
            error_msg = f"Błąd podczas tworzenia datasetu: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    @property
    def classes(self):
        return self.dataset.classes


def get_optimal_training_params():
    """Zwraca optymalne parametry treningu na podstawie dostępnego sprzętu."""
    import os

    params = {
        "batch_size": 32,  # Domyślna wartość
        "num_workers": 2,  # Zmniejszona z 8 na 2 dla uniknięcia problemów z pickle
        "pin_memory": True,
        "prefetch_factor": (
            2 if hasattr(torch.utils.data.DataLoader, "prefetch_factor") else None
        ),
    }

    # Sprawdź, czy istnieje profil wydajności
    try:
        from app.utils.profiler import HardwareProfiler

        profiler = HardwareProfiler()
        profile = profiler.load_profile()

        if profile:
            if "recommended_batch_size" in profile:
                params["batch_size"] = profile["recommended_batch_size"]
            if "recommended_workers" in profile:
                # Ograniczamy liczbę workerów niezależnie od profilu
                params["num_workers"] = min(2, profile["recommended_workers"])
    except Exception:
        # Jeśli nie można załadować profilu, użyj domyślnych wartości
        pass

    # Dostosuj parametry na podstawie dostępnych zasobów
    if torch.cuda.is_available():
        # Na GPU używamy większy batch size i więcej workerów
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (
            1024**3
        )  # w GB

        if gpu_memory > 16:
            params["batch_size"] = 64
            params["num_workers"] = 2  # Ograniczamy do 2
        elif gpu_memory > 8:
            params["batch_size"] = 48
            params["num_workers"] = 2  # Ograniczamy do 2
        elif gpu_memory > 4:
            params["batch_size"] = 32
            params["num_workers"] = 2  # Ograniczamy do 2
        else:
            params["batch_size"] = 16
            params["num_workers"] = 1  # Dla słabszych GPU tylko 1 worker
    else:
        # Na CPU dostosowujemy na podstawie liczby rdzeni
        cpu_cores = os.cpu_count() or 4

        if cpu_cores > 12:
            params["num_workers"] = 2  # Ograniczamy do 2
        elif cpu_cores > 8:
            params["num_workers"] = 2  # Ograniczamy do 2
        elif cpu_cores > 4:
            params["num_workers"] = 1  # Ograniczamy do 1
        else:
            params["num_workers"] = 0  # Dla słabych CPU wyłącz wielowątkowość
            params["batch_size"] = 8

    return params


def get_folder_classes(train_dir):
    """
    Pobiera nazwy klas bezpośrednio z nazw folderów w katalogu treningowym.

    Args:
        train_dir: Ścieżka do katalogu treningowego

    Returns:
        dict: Słownik mapujący indeksy klas na ich nazwy
    """
    class_names = {}

    # Sprawdź czy katalog istnieje
    if not os.path.isdir(train_dir):
        print(f"UWAGA: Katalog {train_dir} nie istnieje")
        return class_names

    # Pobierz listę podkatalogów (kategorii)
    subdirs = [
        d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))
    ]

    # Sortuj, aby zapewnić spójność indeksowania
    subdirs.sort()

    # Utwórz mapowanie indeks -> nazwa klasy
    for idx, class_name in enumerate(subdirs):
        class_names[str(idx)] = class_name

    return class_names


def train_model(
    model_arch: str,
    train_dir: str,
    val_dir: Optional[str] = None,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    optimizer: str = "RMSprop",
    scheduler: Optional[str] = "cosine",
    use_mixed_precision: bool = True,
    num_workers: int = 16,
    weight_decay: float = 0.0001,
    gradient_clip: float = 0.1,
    early_stopping: int = 5,
    augmentation: Optional[Dict[str, bool]] = None,
    logger: Optional[Callable] = None,
    label_smoothing: float = 0.1,
    drop_connect_rate: float = 0.2,
    momentum: float = 0.9,
    epsilon: float = 0.001,
    warmup_epochs: int = 5,
) -> Tuple[str, Dict[str, float]]:
    """
    Trenuje model na podstawie podanych parametrów.

    Args:
        model_arch: Architektura modelu (np. 'efficientnet_b0')
        train_dir: Ścieżka do katalogu z danymi treningowymi
        val_dir: Opcjonalna ścieżka do katalogu z danymi walidacyjnymi
        epochs: Liczba epok treningu
        batch_size: Rozmiar batcha
        learning_rate: Współczynnik uczenia
        optimizer: Nazwa optymalizatora
        scheduler: Nazwa schedulera (opcjonalnie)
        use_mixed_precision: Czy używać mixed precision
        num_workers: Liczba workerów do ładowania danych
        weight_decay: Wartość weight decay
        gradient_clip: Wartość gradient clipping
        early_stopping: Liczba epok bez poprawy przed zatrzymaniem
        augmentation: Konfiguracja augmentacji
        logger: Funkcja do logowania (opcjonalnie)
        label_smoothing: Wartość label smoothing
        drop_connect_rate: Wartość drop connect rate
        momentum: Wartość momentum
        epsilon: Wartość epsilon
        warmup_epochs: Liczba epok warmup

    Returns:
        Tuple[str, Dict[str, float]]: Ścieżka do zapisanego modelu i metryki
    """
    if logger:
        logger(f"\nKonfiguracja treningu:")
        logger(f"- Architektura modelu: {model_arch}")
        logger(f"- Liczba epok: {epochs}")
        logger(f"- Katalog treningowy: {train_dir}")
        if val_dir:
            logger(f"- Katalog walidacyjny: {val_dir}")
        logger(f"- Batch size: {batch_size}")
        logger(f"- Learning rate: {learning_rate}")
        logger(f"- Optimizer: {optimizer}")
        if scheduler:
            logger(f"- Scheduler: {scheduler}")
        logger(f"- Mixed precision: {use_mixed_precision}")
        logger(f"- Liczba workerów: {num_workers}")
        logger(f"- Weight decay: {weight_decay}")
        logger(f"- Gradient clip: {gradient_clip}")
        logger(f"- Early stopping: {early_stopping}")
        if augmentation:
            logger(f"- Augmentacja: {augmentation}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Trening na urządzeniu: {device}")

    model = models.get_model(model_arch, drop_connect_rate=drop_connect_rate)
    model = model.to(device)

    folder_class_names = get_folder_classes(train_dir)

    # Pobierz optymalne parametry treningu
    training_params = get_optimal_training_params()
    if batch_size is None:
        batch_size = training_params["batch_size"]

    print(f"Używane parametry treningu:")
    print(f"- Batch size: {batch_size}")
    print(f"- Liczba workerów: {training_params['num_workers']}")
    print(f"- Prefetch factor: {training_params['prefetch_factor']}")
    print(f"- Pin memory: {training_params['pin_memory']}")

    train_transform = get_augmentation_transforms()
    val_transform = get_default_transforms()

    train_dataset = ImageDataset(train_dir, transform=train_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=training_params["num_workers"],
        pin_memory=training_params["pin_memory"],
        prefetch_factor=training_params["prefetch_factor"],
    )

    if val_dir:
        val_dataset = ImageDataset(val_dir, transform=val_transform)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=training_params["num_workers"],
            pin_memory=training_params["pin_memory"],
            prefetch_factor=training_params["prefetch_factor"],
        )

    # Użyj odpowiedniego kryterium straty
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Optymalizator z regularyzacją L2
    if optimizer.lower() == "rmsprop":
        optimizer = optim.RMSprop(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
            eps=epsilon,
        )
    elif optimizer.lower() == "adamw":
        optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay,
            eps=epsilon,
        )
    elif optimizer.lower() == "sgd":
        optimizer = optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
        )
    else:  # Adam jako domyślny
        optimizer = optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay,
            eps=epsilon,
        )

    # Wybierz scheduler learning rate
    if scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )
    elif scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
    elif scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    else:
        scheduler = None

    # Dodaj warmup scheduler jeśli warmup_epochs > 0
    if warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, scheduler],
            milestones=[warmup_epochs],
        )

    # Implementacja Early Stopping
    patience = early_stopping if early_stopping != float("inf") else float("inf")
    best_val_loss = float("inf")
    counter = 0

    # Historia treningu
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_times": [],
    }

    # Inicjalizacja scaler'a dla mieszanej precyzji
    scaler = torch.amp.GradScaler(
        enabled=use_mixed_precision and torch.cuda.is_available()
    )

    # Trenowanie
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zeruj gradienty
            optimizer.zero_grad()

            # Forward pass z mieszaną precyzją
            with torch.amp.autocast(
                "cuda", enabled=use_mixed_precision and torch.cuda.is_available()
            ):
                outputs = model(inputs)

                # Oblicz stratę
                loss = criterion(outputs, labels)

            # Backward pass i optymalizacja z GradScaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Aktualizuj statystyki
            train_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Średnia strata i dokładność treningu
        train_loss = train_loss / len(train_loader.dataset)

        # Obliczenie dokładności
        train_acc = train_correct / train_total if train_total > 0 else 0

        # Walidacja
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        if val_dir:
            model.eval()
            with torch.no_grad():
                for inputs, labels in val_loader:
                    # Upewnij się, że dane są na odpowiednim urządzeniu
                    inputs, labels = inputs.to(device), labels.to(device)

                    # Wykonuj obliczenia na odpowiednim urządzeniu
                    with torch.amp.autocast(
                        "cuda",
                        enabled=use_mixed_precision and torch.cuda.is_available(),
                    ):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            # Średnia strata i dokładność walidacji
            val_loss = val_loss / len(val_loader.dataset)
            val_acc = val_correct / val_total

            # Aktualizuj scheduler jeśli istnieje
            if scheduler is not None:
                if scheduler == "plateau":
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            # Early stopping
            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        print(f"Early stopping na epoce {epoch+1}")
                        break

        # Zapisz czas trwania epoki
        epoch_time = time.time() - epoch_start_time
        history["epoch_times"].append(epoch_time)

        # Zapisz statystyki treningu
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        if val_dir:
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

        # Wypisz postęp
        if val_dir:
            print(
                f"Epoka {epoch+1}/{epochs} | "
                f"Strata: {train_loss:.4f}, Dokładność: {train_acc:.2%}, "
                f"Val Strata: {val_loss:.4f}, Val Dokładność: {val_acc:.2%}"
            )
        else:
            print(
                f"Epoka {epoch+1}/{epochs} | "
                f"Strata: {train_loss:.4f}, Dokładność: {train_acc:.2%}"
            )

        # Wywołaj callback postępu jeśli istnieje
        if logger is not None:
            logger(
                f"Epoka {epoch+1}/{epochs} | Strata: {train_loss:.4f}, Dokładność: {train_acc:.2%}"
            )

    # Zwróć historię treningu oraz mapowanie klas
    result = {
        "history": history,
        "class_names": folder_class_names,
        "best_epoch": epochs - 1,
        "best_val_loss": best_val_loss,
    }

    # Weryfikuj mapowanie kategorii po zakończeniu treningu
    if hasattr(model, "predict"):
        _verify_model_categories(model, train_dir)

    return result


def mixup_data(x, y, alpha=0.2):
    """Wykonuje mixup na danych wejściowych i etykietach"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    # Upewnij się, że indeks jest na tym samym urządzeniu co dane
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Oblicza stratę dla mixup"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def _verify_model_categories(model, train_dir):
    """Weryfikuje poprawność mapowania kategorii po trenowaniu."""
    print("Weryfikowanie mapowania kategorii...")

    # Wybierz kilka losowych obrazów z katalogu treningowego
    import random

    # Znajdź wszystkie obrazy w katalogu treningowym
    all_images = []
    for root, _, files in os.walk(train_dir):
        rel_path = os.path.relpath(root, train_dir)
        if rel_path == ".":
            continue

        expected_category = rel_path.replace("\\", "/")

        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                all_images.append((os.path.join(root, f), expected_category))

    # Wybierz losowo maksymalnie 10 obrazów
    sample_images = random.sample(all_images, min(10, len(all_images)))

    # Sprawdź klasyfikację
    for img_path, expected_category in sample_images:
        result = model.predict(img_path)
        predicted = result["class_name"]

        match = "✓" if expected_category.lower() == predicted.lower() else "✗"
        print(
            f"{match} {os.path.basename(img_path)} - Oczekiwano: {expected_category}, Przewidziano: {predicted}"
        )
