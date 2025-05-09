import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets

from .classifier import ImageClassifier
from .preprocessing import get_augmentation_transforms, get_default_transforms


def fine_tune_model(
    base_model_path,
    train_dir,
    val_dir=None,
    num_epochs=10,
    batch_size=16,
    learning_rate=0.0001,  # Niższy learning rate dla fine-tuningu
    freeze_ratio=0.8,  # Zamrażamy 80% warstw bazowych
    output_dir="./models",
    optimizer_type="adamw",
    scheduler_type="plateau",
    device=None,
    progress_callback=None,
    should_stop_callback=None,
    label_smoothing=0.1,
    weight_decay=0.01,
    warmup_epochs=1,
    use_mixup=False,
    use_mixed_precision=True,
):
    """
    Przeprowadza fine-tuning istniejącego modelu na nowym zbiorze danych.

    Args:
        base_model_path: Ścieżka do modelu bazowego (.pt)
        train_dir: Katalog z danymi treningowymi
        val_dir: Katalog z danymi walidacyjnymi (opcjonalnie)
        num_epochs: Liczba epok
        batch_size: Rozmiar batcha
        learning_rate: Współczynnik uczenia (niższy niż przy pełnym treningu)
        freeze_ratio: Jaka część warstw powinna być zamrożona (0-1)
        output_dir: Katalog docelowy dla wytrenowanego modelu
        optimizer_type: Typ optymalizatora ('adamw', 'adam', 'sgd')
        scheduler_type: Typ schedulera ('plateau', 'cosine', 'onecycle')
        device: Urządzenie (CPU/GPU)
        progress_callback: Funkcja callback do śledzenia postępu
        should_stop_callback: Funkcja callback do przerywania treningu
        label_smoothing: Parametr wygładzania etykiet (0-1)
        weight_decay: Współczynnik regularyzacji wag
        warmup_epochs: Liczba epok z powolnym wzrostem learning rate
        use_mixup: Czy używać techniki mixup dla augmentacji
        use_mixed_precision: Czy używać mieszanej precyzji (float16/float32)

    Returns:
        Tuple: (ścieżka do zapisanego modelu, historia treningu, szczegóły modelu)
    """
    print("\n=== INICJALIZACJA FINE-TUNINGU ===")
    print(f"Data rozpoczęcia: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model bazowy: {base_model_path}")
    print(f"Katalog treningowy: {train_dir}")
    if val_dir:
        print(f"Katalog walidacyjny: {val_dir}")
    print(f"Liczba epok: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Zamrożenie warstw: {freeze_ratio*100:.0f}%")

    # Sprawdź urządzenie
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Urządzenie: {device}")

    # 1. Załaduj bazowy model
    print("\nŁadowanie modelu bazowego...")
    base_classifier = ImageClassifier(weights_path=base_model_path)

    # Wyświetl informacje o modelu bazowym
    model_info = base_classifier.get_model_info()
    print(f"Typ modelu: {model_info['model_type']}")
    print(f"Liczba klas w modelu bazowym: {model_info['num_classes']}")
    print(f"Łączna liczba parametrów: {model_info['total_parameters']:,}")

    # 2. Znajdź liczbę klas w nowym zbiorze danych
    print("\nAnalizowanie zbioru treningowego...")
    train_folders = [
        f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))
    ]
    new_num_classes = len(train_folders)
    print(f"Znaleziono {new_num_classes} klas w zbiorze treningowym:")
    for idx, folder in enumerate(sorted(train_folders)):
        print(f"  - ID {idx}: {folder}")

    # 3. Utwórz mapowanie klas na podstawie folderów
    new_class_names = {
        str(i): class_name for i, class_name in enumerate(sorted(train_folders))
    }

    # 4. Przygotowanie modelu do fine-tuningu
    print("\nPrzygotowanie modelu do fine-tuningu...")
    model = base_classifier.model
    model_type = base_classifier.model_type

    # 5. Dostosuj ostatnią warstwę modelu, jeśli liczba klas się zmieniła
    original_num_classes = base_classifier.num_classes
    if new_num_classes != original_num_classes:
        print(f"Zmiana liczby klas: {original_num_classes} -> {new_num_classes}")
        print("Dostosowanie ostatniej warstwy modelu...")

        if hasattr(model, "fc"):  # dla ResNet
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, new_num_classes)
            print(
                f"Zmodyfikowano warstwę fc: in_features={in_features}, out_features={new_num_classes}"
            )
        elif hasattr(model, "classifier"):  # dla EfficientNet, MobileNet
            if isinstance(model.classifier, nn.Sequential):
                in_features = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(in_features, new_num_classes)
                print(
                    f"Zmodyfikowano ostatnią warstwę classifier: in_features={in_features}, out_features={new_num_classes}"
                )
            else:
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, new_num_classes)
                print(
                    f"Zmodyfikowano warstwę classifier: in_features={in_features}, out_features={new_num_classes}"
                )
        elif hasattr(model, "heads"):  # dla ViT
            in_features = model.heads.head.in_features
            model.heads.head = nn.Linear(in_features, new_num_classes)
            print(
                f"Zmodyfikowano warstwę heads.head: in_features={in_features}, out_features={new_num_classes}"
            )
    else:
        print("Liczba klas nie uległa zmianie, zachowuję oryginalną warstwę wyjściową")

    # 6. Zamroź określony procent warstw (od początku modelu)
    parameters = list(model.named_parameters())
    num_to_freeze = int(len(parameters) * freeze_ratio)

    trainable_params = 0
    frozen_params = 0

    print("\nZamrażanie warstw modelu...")
    for i, (name, param) in enumerate(parameters):
        if i < num_to_freeze:
            param.requires_grad = False  # Zamroź warstwę
            frozen_params += param.numel()
            print(f"  ❄️ Zamrożono: {name} ({param.shape})")
        else:
            param.requires_grad = True  # Pozostaw do treningu
            trainable_params += param.numel()
            print(f"  🔥 Trenowane: {name} ({param.shape})")

    print(
        f"\nParametry zamrożone: {frozen_params:,} ({frozen_params/(frozen_params+trainable_params)*100:.1f}%)"
    )
    print(
        f"Parametry trenowane: {trainable_params:,} ({trainable_params/(frozen_params+trainable_params)*100:.1f}%)"
    )

    # 7. Przygotuj transformacje danych
    train_transform = get_augmentation_transforms()
    val_transform = get_default_transforms()

    # 8. Załaduj dane
    print("\nŁadowanie danych...")
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"Załadowano {len(train_dataset)} obrazów treningowych")

    val_loader = None
    if val_dir:
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )
        print(f"Załadowano {len(val_dataset)} obrazów walidacyjnych")

    # 9. Skonfiguruj optymalizator (tylko dla trenowanych warstw)
    print("\nKonfiguracja optymalizatora...")
    if optimizer_type.lower() == "adamw":
        optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        print(
            f"Wybrano optymalizator: AdamW (lr={learning_rate}, weight_decay={weight_decay})"
        )
    elif optimizer_type.lower() == "sgd":
        optimizer = optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )
        print(
            f"Wybrano optymalizator: SGD (lr={learning_rate}, momentum=0.9, weight_decay={weight_decay})"
        )
    else:  # Domyślnie Adam
        optimizer = optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        print(
            f"Wybrano optymalizator: Adam (lr={learning_rate}, weight_decay={weight_decay})"
        )

    # 10. Skonfiguruj scheduler
    print("\nKonfiguracja schedulera...")
    scheduler = None
    if scheduler_type.lower() == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=3
        )
        print("Wybrano scheduler: ReduceLROnPlateau (factor=0.1, patience=3)")
    elif scheduler_type.lower() == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=learning_rate / 100
        )
        print(
            f"Wybrano scheduler: CosineAnnealingLR (T_max={num_epochs}, eta_min={learning_rate/100})"
        )
    elif scheduler_type.lower() == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate * 10,
            epochs=num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
        )
        print(
            f"Wybrano scheduler: OneCycleLR (max_lr={learning_rate*10}, pct_start=0.3)"
        )

    # 11. Skonfiguruj kryterium straty
    print("\nKonfiguracja kryterium straty...")
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    print(f"Wybrano CrossEntropyLoss (label_smoothing={label_smoothing})")

    # 12. Inicjalizacja dla mieszanej precyzji
    if use_mixed_precision and torch.cuda.is_available():
        scaler = torch.amp.GradScaler()
        print("Włączono mieszaną precyzję (mixed precision training)")
    else:
        scaler = None
        print("Mieszana precyzja wyłączona")

    # 13. Przejdź do trybu treningu
    model.train()
    model = model.to(device)

    # 14. Inicjalizacja historii treningu
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "val_top3": [],
        "val_top5": [],
        "learning_rates": [],
        "epoch_times": [],
        "best_val_loss": float("inf"),
        "best_epoch": 0,
    }

    # 15. Parametry early stopping
    patience = 5
    counter = 0
    best_val_loss = float("inf")

    print("\n=== ROZPOCZYNAM FINE-TUNING ===")

    # 16. Pętla treningowa
    for epoch in range(num_epochs):
        # Sprawdź czy proces ma zostać przerwany
        if should_stop_callback and should_stop_callback():
            print(
                f"\n!!! Fine-tuning przerwany na epoce {epoch+1} przez użytkownika !!!"
            )
            break

        epoch_start_time = time.time()
        print(f"\n--- Epoka {epoch+1}/{num_epochs} ---")

        # Tryb treningu
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        batch_count = 0

        # Trening na batchu
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Mixup (opcjonalnie)
            if use_mixup:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets)
                inputs, targets_a, targets_b = (
                    inputs.to(device),
                    targets_a.to(device),
                    targets_b.to(device),
                )

            # Zerowanie gradientów
            optimizer.zero_grad()

            # Forward pass z mixed precision (jeśli włączone)
            if use_mixed_precision and scaler is not None:
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(inputs)
                    if use_mixup:
                        loss = mixup_criterion(
                            criterion, outputs, targets_a, targets_b, lam
                        )
                    else:
                        loss = criterion(outputs, targets)

                # Backward pass ze scalerem
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standardowy forward pass
                outputs = model(inputs)
                if use_mixup:
                    loss = mixup_criterion(
                        criterion, outputs, targets_a, targets_b, lam
                    )
                else:
                    loss = criterion(outputs, targets)

                # Standardowy backward pass
                loss.backward()
                optimizer.step()

            # Oblicz dokładność
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            if use_mixup:
                # Przybliżona dokładność dla mixup
                train_correct += (
                    lam * predicted.eq(targets_a).sum().float()
                    + (1 - lam) * predicted.eq(targets_b).sum().float()
                ).item()
            else:
                train_correct += predicted.eq(targets).sum().item()

            train_loss += loss.item()
            batch_count += 1

            # Wyświetl postęp co 10 batchy
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                print(
                    f"  Batch {batch_idx+1}/{len(train_loader)} | "
                    f"Strata: {loss.item():.4f} | "
                    f"Dokładność: {predicted.eq(targets).sum().item() / targets.size(0):.2%}"
                )

        # Oblicz średnie wartości dla epoki
        epoch_loss = train_loss / batch_count
        epoch_acc = train_correct / train_total

        # Walidacja
        val_loss = None
        val_acc = None
        val_metrics = {}

        if val_loader:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            batch_count = 0
            all_targets = []
            all_preds = []
            all_probs = []

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)

                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    # Oblicz dokładność
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
                    val_loss += loss.item()
                    batch_count += 1

                    # Zbieranie danych do dodatkowych metryk
                    all_targets.extend(targets.cpu().numpy())
                    all_preds.extend(predicted.cpu().numpy())
                    all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

            # Oblicz średnie wartości dla walidacji
            val_loss = val_loss / batch_count
            val_acc = val_correct / val_total

            # Oblicz dodatkowe metryki
            y_true = np.array(all_targets)
            y_pred = np.array(all_preds)
            y_prob = np.array(all_probs)

            try:
                val_metrics["precision"] = precision_score(
                    y_true, y_pred, average="macro", zero_division=0
                )
                val_metrics["recall"] = recall_score(
                    y_true, y_pred, average="macro", zero_division=0
                )
                val_metrics["f1"] = f1_score(
                    y_true, y_pred, average="macro", zero_division=0
                )

                # Top-k accuracy (jeśli więcej niż 2 klasy)
                if new_num_classes > 2:
                    k_values = min(5, new_num_classes)  # Nie więcej niż liczba klas
                    val_metrics["top3"] = (
                        top_k_accuracy_score(y_true, y_prob, k=min(3, k_values))
                        if k_values >= 3
                        else None
                    )
                    val_metrics["top5"] = (
                        top_k_accuracy_score(y_true, y_prob, k=min(5, k_values))
                        if k_values >= 5
                        else None
                    )

                # AUC dla wieloklasowości
                try:
                    if new_num_classes == 2:
                        val_metrics["auc"] = roc_auc_score(y_true, y_prob[:, 1])
                    else:
                        val_metrics["auc"] = roc_auc_score(
                            y_true, y_prob, multi_class="ovr"
                        )
                except Exception:
                    val_metrics["auc"] = None

            except Exception as e:
                print(f"Błąd podczas obliczania metryk: {e}")
                val_metrics = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "top3": None,
                    "top5": None,
                    "auc": None,
                }

            # Zapisz metryki do historii
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_precision"].append(val_metrics["precision"])
            history["val_recall"].append(val_metrics["recall"])
            history["val_f1"].append(val_metrics["f1"])
            if val_metrics["top3"] is not None:
                history["val_top3"].append(val_metrics["top3"])
            if val_metrics["top5"] is not None:
                history["val_top5"].append(val_metrics["top5"])

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                history["best_val_loss"] = best_val_loss
                history["best_epoch"] = epoch
                counter = 0
                print("✓ Nowa najlepsza strata walidacyjna!")

                # Zapisz najlepszy model
                os.makedirs(output_dir, exist_ok=True)
                best_model_path = os.path.join(
                    output_dir, f"{model_type}_finetuned_best.pt"
                )

                # Utwórz nowy klasyfikator z dostosowanym modelem
                best_classifier = ImageClassifier(
                    model_type=model_type, num_classes=new_num_classes
                )
                best_classifier.model = model
                best_classifier.class_names = new_class_names

                # Zapisz model
                best_classifier.save(best_model_path)
                print(f"✓ Zapisano najlepszy model: {best_model_path}")
            else:
                counter += 1
                print(f"✗ Brak poprawy ({counter}/{patience})")
                if counter >= patience:
                    print(f"Early stopping na epoce {epoch+1}")
                    break

            # Aktualizuj scheduler
            if scheduler is not None:
                if scheduler_type.lower() == "plateau":
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

                # Zapisz aktualny learning rate
                current_lr = optimizer.param_groups[0]["lr"]
                history["learning_rates"].append(current_lr)
                print(f"Aktualny learning rate: {current_lr:.6f}")

        # Zapisz czas trwania epoki
        epoch_time = time.time() - epoch_start_time
        history["epoch_times"].append(epoch_time)
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)

        # Wyświetl podsumowanie epoki
        print(f"\nPodsumowanie epoki {epoch+1}:")
        print(f"  Czas: {epoch_time:.2f}s")
        print(f"  Train loss: {epoch_loss:.4f}")
        print(f"  Train acc:  {epoch_acc:.2%}")

        if val_loader:
            print(f"  Val loss:   {val_loss:.4f}")
            print(f"  Val acc:    {val_acc:.2%}")
            print(f"  Val F1:     {val_metrics['f1']:.4f}")
            if val_metrics["top3"] is not None:
                print(f"  Val top-3:  {val_metrics['top3']:.2%}")

        # Wywołaj callback z postępem jeśli istnieje
        if progress_callback:
            try:
                top3 = val_metrics.get("top3", 0) if val_loader else 0
                top5 = val_metrics.get("top5", 0) if val_loader else 0
                precision = val_metrics.get("precision", 0) if val_loader else 0
                recall = val_metrics.get("recall", 0) if val_loader else 0
                f1 = val_metrics.get("f1", 0) if val_loader else 0
                auc = val_metrics.get("auc", 0) if val_loader else 0

                progress_callback(
                    epoch + 1,
                    num_epochs,
                    epoch_loss,
                    epoch_acc,
                    val_loss if val_loader else 0,
                    val_acc if val_loader else 0,
                    top3,
                    top5,
                    precision,
                    recall,
                    f1,
                    auc,
                )
            except Exception as e:
                print(f"Błąd podczas wywołania progress_callback: {str(e)}")

        # Czyszczenie pamięci GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 17. Zapisz końcowy model
    print("\n=== ZAPISYWANIE KOŃCOWEGO MODELU ===")
    os.makedirs(output_dir, exist_ok=True)
    final_model_path = os.path.join(output_dir, f"{model_type}_finetuned_final.pt")

    # Utwórz nowy klasyfikator z dostosowanym modelem
    final_classifier = ImageClassifier(
        model_type=model_type, num_classes=new_num_classes
    )
    final_classifier.model = model
    final_classifier.class_names = new_class_names

    # Zapisz model
    final_classifier.save(final_model_path)
    print(f"Zapisano końcowy model: {final_model_path}")

    # 18. Podsumowanie fine-tuningu
    print("\n=== PODSUMOWANIE FINE-TUNINGU ===")
    print(f"Data zakończenia: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if val_loader:
        best_epoch = history["best_epoch"]
        print(f"Najlepsza epoka: {best_epoch + 1}")
        print(f"Najlepsza strata walidacyjna: {history['best_val_loss']:.4f}")
        print(f"Dokładność walidacji: {history['val_acc'][best_epoch]:.2%}")

    # Przygotuj wynik
    result = {
        "model_path": final_model_path,
        "best_model_path": best_model_path if val_loader else final_model_path,
        "history": history,
        "class_names": new_class_names,
        "model_type": model_type,
        "num_classes": new_num_classes,
        "base_model": base_model_path,
    }

    return result


def mixup_data(x, y, alpha=0.2):
    """
    Wykonuje mixup na danych wejściowych i etykietach.

    Args:
        x: Dane wejściowe (obrazy)
        y: Etykiety
        alpha: Parametr mixup (0-1)

    Returns:
        mixed_x, y_a, y_b, lam: Zmiksowane dane i etykiety oraz współczynnik lambda
    """
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
    """
    Oblicza stratę dla mixup.

    Args:
        criterion: Funkcja straty
        pred: Predykcje modelu
        y_a, y_b: Oryginalne i zmieszane etykiety
        lam: Współczynnik mixup

    Returns:
        float: Wartość straty
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_best_finetuning_params(model_type, dataset_size):
    """
    Zwraca rekomendowane parametry fine-tuningu dla danego typu modelu i wielkości zbioru danych.

    Args:
        model_type: Typ modelu (np. 'b0', 'resnet50')
        dataset_size: Liczba obrazów w zbiorze treningowym

    Returns:
        dict: Rekomendowane parametry fine-tuningu
    """
    # Parametry bazowe
    params = {
        "learning_rate": 0.0001,
        "batch_size": 16,
        "num_epochs": 20,
        "freeze_ratio": 0.8,
        "optimizer_type": "adamw",
        "scheduler_type": "cosine",
        "label_smoothing": 0.1,
        "weight_decay": 0.01,
        "use_mixup": True,
        "use_mixed_precision": True,
    }

    # Dostosowywanie parametrów do modelu
    if "b0" in model_type or "mobile" in model_type:
        # Mniejsze modele
        params["learning_rate"] = 0.0005
        params["freeze_ratio"] = 0.7  # Mniej warstw do zamrożenia
    elif "resnet50" in model_type or "b4" in model_type:
        # Średnie modele
        params["learning_rate"] = 0.0002
        params["freeze_ratio"] = 0.8
    elif "large" in model_type or "b7" in model_type:
        # Duże modele
        params["learning_rate"] = 0.00008
        params["freeze_ratio"] = 0.9  # Więcej warstw do zamrożenia
        params["batch_size"] = 8  # Mniejszy batch size dla większych modeli

    # Dostosowywanie parametrów do wielkości zbioru danych
    if dataset_size < 500:
        # Mały zbiór danych
        params["num_epochs"] = 30
        params["freeze_ratio"] += 0.05  # Zamroź więcej warstw dla małych zbiorów
        params["use_mixup"] = True  # Włącz mixup dla małych zbiorów
        params["weight_decay"] = 0.02  # Zwiększ regularyzację
    elif dataset_size < 2000:
        # Średni zbiór danych
        params["num_epochs"] = 20
    else:
        # Duży zbiór danych
        params["num_epochs"] = 15
        params["freeze_ratio"] -= 0.1  # Zamroź mniej warstw dla dużych zbiorów
        params["learning_rate"] *= 2  # Zwiększ learning rate

    return params


def verify_fine_tuned_model(model_path, test_dir, top_n=5):
    """
    Weryfikuje skuteczność fine-tuningu, testując model na wybranych obrazach.

    Args:
        model_path: Ścieżka do wytrenowanego modelu
        test_dir: Katalog z obrazami testowymi
        top_n: Liczba przykładów do wyświetlenia

    Returns:
        dict: Wyniki weryfikacji
    """
    import random

    print(f"\n=== WERYFIKACJA MODELU {model_path} ===")

    # Załaduj model
    classifier = ImageClassifier(weights_path=model_path)

    # Zbierz obrazy z podkatalogów
    test_images = []
    for root, _, files in os.walk(test_dir):
        rel_path = os.path.relpath(root, test_dir)
        if rel_path == ".":
            continue

        expected_category = rel_path.replace("\\", "/")

        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                test_images.append((os.path.join(root, f), expected_category))

    if not test_images:
        return {"error": "Nie znaleziono obrazów testowych"}

    # Wybierz losowo kilka obrazów do testów
    selected_images = random.sample(test_images, min(len(test_images), top_n * 3))

    results = {
        "correct": [],
        "incorrect": [],
        "accuracy": 0.0,
        "total": len(selected_images),
    }

    # Testuj model
    for img_path, expected in selected_images:
        try:
            result = classifier.predict(img_path)
            predicted = result["class_name"]
            confidence = result["confidence"]

            # Zapisz wynik
            if expected.lower() == predicted.lower():
                results["correct"].append(
                    {
                        "path": img_path,
                        "expected": expected,
                        "predicted": predicted,
                        "confidence": confidence,
                    }
                )
            else:
                results["incorrect"].append(
                    {
                        "path": img_path,
                        "expected": expected,
                        "predicted": predicted,
                        "confidence": confidence,
                    }
                )
        except Exception as e:
            print(f"Błąd przy przetwarzaniu {img_path}: {e}")

    # Oblicz dokładność
    if results["total"] > 0:
        results["accuracy"] = len(results["correct"]) / results["total"]

    # Wyświetl wyniki
    print(f"\nWyniki testu modelu:")
    print(f"Łącznie przetestowano: {results['total']} obrazów")
    print(f"Poprawnie sklasyfikowanych: {len(results['correct'])}")
    print(f"Błędnie sklasyfikowanych: {len(results['incorrect'])}")
    print(f"Dokładność: {results['accuracy']:.2%}")

    print("\nPrzykłady poprawnych klasyfikacji:")
    for i, item in enumerate(results["correct"][:top_n]):
        print(
            f"  ✓ {i+1}. {os.path.basename(item['path'])}: {item['expected']} (pewność: {item['confidence']:.2%})"
        )

    print("\nPrzykłady błędnych klasyfikacji:")
    for i, item in enumerate(results["incorrect"][:top_n]):
        print(
            f"  ✗ {i+1}. {os.path.basename(item['path'])}: oczekiwano {item['expected']}, "
            + f"otrzymano {item['predicted']} (pewność: {item['confidence']:.2%})"
        )

    return results


def compare_base_and_finetuned(base_model_path, finetuned_model_path, test_dir):
    """
    Porównuje wydajność modelu bazowego i modelu po fine-tuningu na tym samym zbiorze testowym.

    Args:
        base_model_path: Ścieżka do modelu bazowego
        finetuned_model_path: Ścieżka do modelu po fine-tuningu
        test_dir: Katalog z obrazami testowymi

    Returns:
        dict: Wyniki porównania
    """
    print(f"\n=== PORÓWNANIE MODELI ===")
    print(f"Model bazowy: {base_model_path}")
    print(f"Model po fine-tuningu: {finetuned_model_path}")

    # Załaduj modele
    base_classifier = ImageClassifier(weights_path=base_model_path)
    finetuned_classifier = ImageClassifier(weights_path=finetuned_model_path)

    # Zbierz obrazy z podkatalogów
    test_images = []
    for root, _, files in os.walk(test_dir):
        rel_path = os.path.relpath(root, test_dir)
        if rel_path == ".":
            continue

        expected_category = rel_path.replace("\\", "/")

        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                test_images.append((os.path.join(root, f), expected_category))

    if not test_images:
        return {"error": "Nie znaleziono obrazów testowych"}

    # Maksymalnie 100 obrazów dla porównania
    import random

    selected_images = random.sample(test_images, min(len(test_images), 100))

    # Przygotuj wyniki
    results = {
        "base_correct": 0,
        "finetuned_correct": 0,
        "both_correct": 0,
        "both_incorrect": 0,
        "base_only_correct": 0,
        "finetuned_only_correct": 0,
        "total": len(selected_images),
        "examples": [],
    }

    # Testuj oba modele
    for img_path, expected in selected_images:
        try:
            base_result = base_classifier.predict(img_path)
            base_predicted = base_result["class_name"]
            base_correct = expected.lower() == base_predicted.lower()

            finetuned_result = finetuned_classifier.predict(img_path)
            finetuned_predicted = finetuned_result["class_name"]
            finetuned_correct = expected.lower() == finetuned_predicted.lower()

            # Aktualizuj liczniki
            if base_correct:
                results["base_correct"] += 1
            if finetuned_correct:
                results["finetuned_correct"] += 1

            if base_correct and finetuned_correct:
                results["both_correct"] += 1
            elif not base_correct and not finetuned_correct:
                results["both_incorrect"] += 1
            elif base_correct and not finetuned_correct:
                results["base_only_correct"] += 1
            elif not base_correct and finetuned_correct:
                results["finetuned_only_correct"] += 1

            # Zapisz przykład
            results["examples"].append(
                {
                    "path": img_path,
                    "expected": expected,
                    "base_predicted": base_predicted,
                    "base_confidence": base_result["confidence"],
                    "base_correct": base_correct,
                    "finetuned_predicted": finetuned_predicted,
                    "finetuned_confidence": finetuned_result["confidence"],
                    "finetuned_correct": finetuned_correct,
                }
            )

        except Exception as e:
            print(f"Błąd przy przetwarzaniu {img_path}: {e}")

    # Oblicz dokładności
    if results["total"] > 0:
        results["base_accuracy"] = results["base_correct"] / results["total"]
        results["finetuned_accuracy"] = results["finetuned_correct"] / results["total"]
        results["improvement"] = (
            results["finetuned_accuracy"] - results["base_accuracy"]
        )

    # Wyświetl wyniki
    print(f"\nWyniki porównania:")
    print(f"Łącznie przetestowano: {results['total']} obrazów")
    print(f"Dokładność modelu bazowego: {results['base_accuracy']:.2%}")
    print(f"Dokładność modelu po fine-tuningu: {results['finetuned_accuracy']:.2%}")
    print(f"Poprawa: {results['improvement']:.2%}")
    print(
        f"Obrazy poprawnie klasyfikowane przez oba modele: {results['both_correct']} ({results['both_correct']/results['total']:.2%})"
    )
    print(
        f"Obrazy poprawnie klasyfikowane tylko przez model bazowy: {results['base_only_correct']} ({results['base_only_correct']/results['total']:.2%})"
    )
    print(
        f"Obrazy poprawnie klasyfikowane tylko przez model po fine-tuningu: {results['finetuned_only_correct']} ({results['finetuned_only_correct']/results['total']:.2%})"
    )
    print(
        f"Obrazy błędnie klasyfikowane przez oba modele: {results['both_incorrect']} ({results['both_incorrect']/results['total']:.2%})"
    )

    # Pokaż kilka przykładów
    if results["finetuned_only_correct"] > 0:
        print(
            "\nPrzykłady obrazów poprawnie klasyfikowanych tylko przez model po fine-tuningu:"
        )
        count = 0
        for item in results["examples"]:
            if not item["base_correct"] and item["finetuned_correct"]:
                print(
                    f"  ✓ {os.path.basename(item['path'])}: oczekiwano {item['expected']}, "
                    + f"model bazowy: {item['base_predicted']} ({item['base_confidence']:.2%}), "
                    + f"model po fine-tuningu: {item['finetuned_predicted']} ({item['finetuned_confidence']:.2%})"
                )
                count += 1
                if count >= 5:
                    break

    if results["base_only_correct"] > 0:
        print("\nPrzykłady obrazów poprawnie klasyfikowanych tylko przez model bazowy:")
        count = 0
        for item in results["examples"]:
            if item["base_correct"] and not item["finetuned_correct"]:
                print(
                    f"  ✗ {os.path.basename(item['path'])}: oczekiwano {item['expected']}, "
                    + f"model bazowy: {item['base_predicted']} ({item['base_confidence']:.2%}), "
                    + f"model po fine-tuningu: {item['finetuned_predicted']} ({item['finetuned_confidence']:.2%})"
                )
                count += 1
                if count >= 5:
                    break

    return results
