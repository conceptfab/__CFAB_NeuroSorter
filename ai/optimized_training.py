import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets

from .preprocessing import (
    get_augmentation_transforms,
    get_default_transforms,
    get_extended_augmentation_transforms,
)


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
    mixup=False,
    label_smoothing=0.0,
    weight_decay=0.01,
    optimizer_type="adamw",
    profiler=None,
    augmentation_mode="basic",
    augmentation_params=None,
    should_stop_callback=None,
):
    """
    Trenuje model na podanym zbiorze danych z wykorzystaniem optymalnych parametrów sprzętowych.
    """
    # print("\n=== INICJALIZACJA OPTYMALIZOWANEGO TRENINGU ===")
    # print(f"Data rozpoczęcia: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Inicjalizacja profilera jeśli nie podano
    if profiler is None:
        # print("Profiler nie został podany, tworzę nowy...")
        from app.utils.profiler import HardwareProfiler

        profiler = HardwareProfiler()

    # Załaduj profil sprzętowy
    hardware_profile = profiler.load_profile()
    if hardware_profile:
        # print("Załadowano profil sprzętowy z optymalnymi parametrami")
        pass
    else:
        # print("Nie znaleziono profilu sprzętowego, generuję nowy...")
        hardware_profile = profiler.generate_recommendations()
        # print("Utworzono nowy profil sprzętowy")

    # Ustaw urządzenie
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Urządzenie treningu: {device}")

    # Ustaw optymalny rozmiar batcha jeśli nie został podany
    if batch_size is None:
        batch_size = hardware_profile.get("recommended_batch_size", 32)
        # print(f"Użyto optymalnego rozmiaru batcha z profilera: {batch_size}")

    # Ustaw optymalną liczbę workerów
    # print("DEBUG: TYMCZASOWE USTAWIANIE num_workers NA 0 DLA TESTU!")
    recommended_workers = 0  # <--- USTAW NA 0 DLA TESTU

    # Ustaw mixed precision
    use_mixed_precision = hardware_profile.get("use_mixed_precision", False)
    if use_mixed_precision and torch.cuda.is_available():
        # print("Używam mixed precision (zalecane przez profiler)")
        pass
    else:
        use_mixed_precision = False
        # print("Mixed precision wyłączone (brak GPU lub niezalecane przez profiler)")

    # Inicjalizacja historii treningu
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_times": [],
        "learning_rates": [],
        "best_val_loss": float("inf"),
        "best_epoch": 0,
        "hardware_profile": hardware_profile,
    }

    # Parametry early stopping
    patience = 5
    counter = 0
    best_val_loss = float("inf")

    # Przygotuj model
    try:
        model = model.to(device)
        # print(f"Model przeniesiony na urządzenie: {device}")
    except Exception as e:
        print(f"Błąd podczas przenoszenia modelu na urządzenie: {e}")
        raise

    # Przygotuj dane treningowe
    train_transform = None
    if augmentation_mode == "basic":
        train_transform = get_augmentation_transforms()
        # print("Użyto podstawowej augmentacji danych")
    elif augmentation_mode == "extended":
        train_transform = get_extended_augmentation_transforms(
            params=augmentation_params
        )
        # print("Użyto rozszerzonej augmentacji danych z parametrami")
    else:
        train_transform = get_default_transforms()
        # print("Użyto standardowych transformacji bez augmentacji")

    val_transform = get_default_transforms()

    # Przygotuj zbiory danych
    try:
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)

        # Zapisz mapowanie klas
        class_names = {}
        for idx, class_name in enumerate(train_dataset.classes):
            class_names[str(idx)] = class_name
            # print(f"  - ID {idx}: {class_name}")

        # Przypisanie class_names do modelu
        if hasattr(model, "class_names"):
            model.class_names = class_names
        else:
            # Jeśli model nie ma atrybutu class_names, tworzymy go dynamicznie
            setattr(model, "class_names", class_names)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=recommended_workers,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=2 if hasattr(DataLoader, "prefetch_factor") else None,
            persistent_workers=False,  # Wyłącz persistent_workers aby zredukować problemy z pickle
        )

        # Walidacja
        val_loader = None
        if val_dir:
            val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=recommended_workers,
                pin_memory=torch.cuda.is_available(),
                prefetch_factor=2 if hasattr(DataLoader, "prefetch_factor") else None,
                persistent_workers=False,  # Wyłącz persistent_workers aby zredukować problemy z pickle
            )
    except Exception as e:
        print(f"Błąd podczas przygotowania danych: {e}")
        raise

    # Konfiguracja optymalizatora
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

    # Wybór optymalizatora
    if optimizer_type.lower() == "adamw":
        optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        # print(
        #     f"Użyto optymalizatora AdamW z learning_rate={learning_rate}, weight_decay={weight_decay}"
        # )
    elif optimizer_type.lower() == "sgd":
        optimizer = optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )
        # print(f"Użyto optymalizatora SGD z learning_rate={learning_rate}, momentum=0.9")
    else:  # domyślnie Adam
        optimizer = optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        # print(
        #     f"Użyto optymalizatora Adam z learning_rate={learning_rate}, weight_decay={weight_decay}"
        # )

    # Konfiguracja kryterium straty
    if label_smoothing > 0:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        # print(f"Użyto CrossEntropyLoss z label smoothing: {label_smoothing}")
    else:
        criterion = nn.CrossEntropyLoss()
        # print("Użyto standardowego CrossEntropyLoss")

    # Konfiguracja schedulera
    scheduler = None
    if lr_scheduler_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )
        # print("Użyto schedulera ReduceLROnPlateau")
    elif lr_scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        # print("Użyto schedulera CosineAnnealingLR")
    elif lr_scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        # print("Użyto schedulera StepLR")

    # Konfiguracja scaler dla mixed precision
    scaler = None
    if use_mixed_precision and torch.cuda.is_available():
        scaler = torch.amp.GradScaler()
        # print("Włączono mixed precision z GradScaler")

    # print("\n=== ROZPOCZYNAM TRENING ===")

    # Główna pętla treningu
    # print(f"DEBUG optimized_training: Rozpoczynam pętlę po epokach. Liczba epok: {num_epochs}")
    for epoch in range(num_epochs):
        # print(f"DEBUG optimized_training: Początek epoki {epoch + 1}/{num_epochs}")
        # Sprawdź, czy przerwano trening
        if should_stop_callback and should_stop_callback():
            print(f"\n!!! Trening przerwany na epoce {epoch+1} przez użytkownika !!!")
            break

        epoch_start_time = time.time()
        # print(f"\n=== EPOKA {epoch+1}/{num_epochs} ===")

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        batch_count = 0

        # Trening na batchu
        try:
            # print(f"DEBUG optimized_training: Rozpoczynam pętlę po batchach. Liczba batchy (len(train_loader)): {len(train_loader)}")
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                # print(f"DEBUG Batch {batch_idx + 1}: POCZĄTEK PĘTLI BATCH")

                # print(f"DEBUG Batch {batch_idx + 1}: Przenoszenie danych na urządzenie...")
                inputs, targets = inputs.to(device), targets.to(device)
                # print(f"DEBUG Batch {batch_idx + 1}: Dane przeniesione. Typ inputs: {type(inputs)}, Typ targets: {type(targets)}")

                # print(f"DEBUG Batch {batch_idx + 1}: Zerowanie gradientów...")
                optimizer.zero_grad()
                # print(f"DEBUG Batch {batch_idx + 1}: Gradienty wyzerowane.")

                if use_mixed_precision and scaler is not None:
                    # print(f"DEBUG Batch {batch_idx + 1}: Używam mixed precision. Forward pass z autocast...")
                    with torch.amp.autocast(
                        device_type=device.type if device else "cuda"
                    ):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                    # print(f"DEBUG Batch {batch_idx + 1}: Forward pass (mixed precision) zakończony. Strata: {loss.item()}")

                    # print(f"DEBUG Batch {batch_idx + 1}: Scaler.scale(loss).backward()...")
                    scaler.scale(loss).backward()
                    # print(f"DEBUG Batch {batch_idx + 1}: Scaler.scale(loss).backward() zakończone.")

                    # print(f"DEBUG Batch {batch_idx + 1}: Scaler.step(optimizer)...")
                    scaler.step(optimizer)
                    # print(f"DEBUG Batch {batch_idx + 1}: Scaler.step(optimizer) zakończone.")

                    # print(f"DEBUG Batch {batch_idx + 1}: Scaler.update()...")
                    scaler.update()
                    # print(f"DEBUG Batch {batch_idx + 1}: Scaler.update() zakończone.")
                else:
                    # print(f"DEBUG Batch {batch_idx + 1}: Standardowy forward pass...")
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    # print(f"DEBUG Batch {batch_idx + 1}: Standardowy forward pass zakończony. Strata: {loss.item()}")

                    # print(f"DEBUG Batch {batch_idx + 1}: loss.backward()...")
                    loss.backward()
                    # print(f"DEBUG Batch {batch_idx + 1}: loss.backward() zakończone.")

                    # print(f"DEBUG Batch {batch_idx + 1}: optimizer.step()...")
                    optimizer.step()
                    # print(f"DEBUG Batch {batch_idx + 1}: optimizer.step() zakończone.")

                # Oblicz dokładność
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
                train_loss += loss.item()
                batch_count += 1

                # Debug - wyświetl wartości dla każdego batcha
                batch_loss = loss.item()
                batch_acc = predicted.eq(targets).sum().item() / targets.size(0)
                # print(f"\nDEBUG Batch {batch_idx + 1}/{len(train_loader)}:")
                # print(f"Strata: {batch_loss:.4f}")
                # print(f"Dokładność: {batch_acc:.4f}")
                # print(f"DEBUG Batch {batch_idx + 1}: Zakończono przetwarzanie.")

            # Koniec pętli po batchach
            # print(f"DEBUG optimized_training: Pętla po batchach dla epoki {epoch + 1} zakończona normalnie.")

        except Exception as e_batch:
            print(
                f"!!!!!!!!!! DEBUG optimized_training: BŁĄD KRYTYCZNY W PĘTLI PO BATCHACH (epoka {epoch + 1}) !!!!!!!!!!"
            )
            print(f"Błąd: {str(e_batch)}")
            print(traceback.format_exc())
            # Można tutaj dodać break lub return, jeśli błąd jest tak poważny, że dalszy trening nie ma sensu

        # Oblicz średnie wartości dla epoki
        epoch_loss = (
            train_loss / batch_count if batch_count > 0 else 0
        )  # Zabezpieczenie przed dzieleniem przez zero
        epoch_acc = (
            train_correct / train_total if train_total > 0 else 0
        )  # Zabezpieczenie przed dzieleniem przez zero

        # print(f"\nDEBUG Epoka {epoch + 1}/{num_epochs}:")
        # print(f"Średnia strata: {epoch_loss:.4f}")
        # print(f"Średnia dokładność: {epoch_acc:.4f}")

        # Walidacja
        val_loss = None
        val_acc = None
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            batch_count = 0

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
                    val_loss += loss.item()
                    batch_count += 1

            val_loss = val_loss / batch_count
            val_acc = val_correct / val_total

            # print(f"Walidacja - Strata: {val_loss:.4f}")
            # print(f"Walidacja - Dokładność: {val_acc:.4f}")

            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    history["best_epoch"] = epoch
                    history["best_val_loss"] = best_val_loss
                    counter = 0
                    # print("✓ Nowa najlepsza strata walidacyjna!")
                else:
                    counter += 1
                    # print(f"✗ Brak poprawy ({counter}/{patience})")
                    if counter >= patience:
                        print(f"Early stopping na epoce {epoch+1}")
                        break

            if scheduler is not None:
                if lr_scheduler_type == "plateau":
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

                current_lr = optimizer.param_groups[0]["lr"]
                history["learning_rates"].append(current_lr)
                # print(f"Aktualny learning rate: {current_lr:.6f}")

        epoch_time = time.time() - epoch_start_time
        history["epoch_times"].append(epoch_time)
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)

        if val_loader:
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

        print("\n--- PODSUMOWANIE EPOKI ---")
        print(f"Czas trwania: {epoch_time:.2f}s")
        print(f"Średnia strata: {epoch_loss:.4f}")
        print(f"Średnia dokładność: {epoch_acc:.4f}")

        if val_loader:
            print(f"Walidacja - Strata: {val_loss:.4f}")
            print(f"Walidacja - Dokładność: {val_acc:.4f}")

        print(
            f"DEBUG optimized_training: Koniec epoki {epoch + 1}. Zaraz wywołam progress_callback (jeśli istnieje)."
        )
        # Wywołaj callback z postępem
        if progress_callback:
            try:
                print("\nDEBUG: Przekazuję dane do callbacka:")
                print(f"Epoka: {epoch + 1}")
                print(f"Liczba epok: {num_epochs}")
                print(f"Strata treningowa: {epoch_loss:.4f}")
                print(f"Dokładność treningowa: {epoch_acc:.4f}")
                # Poprawione printy dla val_loss i val_acc
                val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
                val_acc_str = f"{val_acc:.4f}" if val_acc is not None else "N/A"
                print(f"Strata walidacyjna: {val_loss_str}")
                print(f"Dokładność walidacyjna: {val_acc_str}")

                progress_callback(
                    epoch + 1, num_epochs, epoch_loss, epoch_acc, val_loss, val_acc
                )
            except Exception as e_outer_callback:
                print(
                    f"BŁĄD PODCZAS WYWOŁANIA progress_callback (złapany w optimized_training): Typ błędu: {type(e_outer_callback).__name__}, Treść: {str(e_outer_callback)}"
                )
                print(traceback.format_exc())
                print(
                    f"Wartości przy błędzie: epoch={epoch + 1}, loss={epoch_loss:.4f}, acc={epoch_acc:.4f}"
                )

    print(
        f"DEBUG optimized_training: Zakończono pętlę po epokach (naturalnie lub przez break). Ostatnia przetworzona epoka (0-indexed): {epoch if 'epoch' in locals() else 'nie zdefiniowano'}"
    )
    print("\n=== ZAKOŃCZENIE TRENINGU ===")
    print(f"Data zakończenia: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if val_loader:
        print(f"Najlepsza epoka: {history['best_epoch'] + 1}")
        print(f"Najlepsza strata walidacyjna: {history['best_val_loss']:.4f}")

    result = {
        "history": history,
        "class_names": class_names,
        "best_epoch": history.get("best_epoch", 0),
        "best_val_loss": history.get("best_val_loss", float("inf")),
        "hardware_profile": hardware_profile,
    }

    return result


def mixup_data(x, y, alpha=0.2, device=None):
    """
    Wykonuje mixup na danych wejściowych i etykietach.
    Bezpieczna implementacja unikająca problemów z urządzeniami.
    """
    print(
        f"DEBUG MIXUP: Wejście x typu: {type(x)}, kształt: {x.shape if hasattr(x, 'shape') else 'brak'}"
    )
    print(
        f"DEBUG MIXUP: Wejście y typu: {type(y)}, kształt: {y.shape if hasattr(y, 'shape') else 'brak'}"
    )
    print(f"DEBUG MIXUP: Alpha: {alpha}, Device: {device}")

    # Jeśli nie podano urządzenia, użyj urządzenia x
    if device is None:
        device = x.device
        print(f"DEBUG MIXUP: Użycie urządzenia z tensora x: {device}")

    # Bezpieczne sprawdzenie CUDA
    if device.type == "cuda" and not torch.cuda.is_available():
        print("Ostrzeżenie: Urządzenie ustawione na CUDA, ale CUDA nie jest dostępne.")
        device = torch.device("cpu")
        print(f"DEBUG MIXUP: Zmiana urządzenia na CPU")

    # Parametr mixup
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    print(f"DEBUG MIXUP: Lambda: {lam}")

    batch_size = x.size()[0]
    print(f"DEBUG MIXUP: Rozmiar batcha: {batch_size}")

    # Bezpieczne tworzenie permutacji
    try:
        print(f"DEBUG MIXUP: Próba tworzenia permutacji dla batch_size={batch_size}")
        # Najlepsze podejście - tworzenie na CPU a potem przeniesienie
        # Używamy jawnie określonych typów
        index = torch.randperm(batch_size, dtype=torch.long, device="cpu")
        print(f"DEBUG MIXUP: Permutacja utworzona na CPU, kształt: {index.shape}")
        index = index.to(device)
        print(f"DEBUG MIXUP: Permutacja przeniesiona na {device}")

        # Mixup
        print(f"DEBUG MIXUP: Wykonywanie mixup")
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        print(f"DEBUG MIXUP: Mixup zakończony, kształt mixed_x: {mixed_x.shape}")

        return mixed_x, y_a, y_b, lam
    except Exception as e:
        print(f"BŁĄD podczas mixup: {e}")
        print(f"SZCZEGÓŁY: {traceback.format_exc()}")
        # Awaryjne podejście - zwróć oryginalne dane
        return x, y, y, 1.0


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Oblicza stratę dla mixup.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def progress_callback(epoch, num_epochs, train_loss, train_acc, val_loss, val_acc):
    # Sprawdź, czy parametry są prawidłowe
    if epoch < 1 or epoch > num_epochs:
        print(f"BŁĄD: Nieprawidłowy numer epoki: {epoch}/{num_epochs}")
        return  # Nie wyświetlaj nieprawidłowych informacji

    # W tej wersji funkcji nie drukujemy nic, ponieważ informacje o epoce
    # są już drukowane w głównej funkcji train_model_optimized.
    # To pozwoli uniknąć duplikowania komunikatów.
    pass
