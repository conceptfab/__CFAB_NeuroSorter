import logging
import os
import sys
import time
import traceback
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

from .preprocessing import (
    get_augmentation_transforms,
    get_default_transforms,
    get_extended_augmentation_transforms,
)


def log_model_info(model, log_path, extra_info=None):
    """Zapisuje informacje o modelu do pliku log."""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            f"\n=== INFORMACJE O MODELU ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===\n"
        )
        f.write(f"Typ modelu: {type(model)}\n")
        if hasattr(model, "classifier"):
            f.write(f"Głowica classifier: {model.classifier}\n")
        if hasattr(model, "fc"):
            f.write(f"Głowica fc: {model.fc}\n")
        if hasattr(model, "class_names"):
            f.write(f"class_names: {getattr(model, 'class_names', None)}\n")
        if hasattr(model, "num_classes"):
            f.write(f"num_classes: {getattr(model, 'num_classes', None)}\n")
        f.write(f"Parametry modelu: {sum(p.numel() for p in model.parameters())}\n")
        if extra_info:
            f.write(f"Dodatkowe info: {extra_info}\n")
        f.write("=== KONIEC INFO ===\n\n")


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
    mixup=True,
    label_smoothing=0.1,
    weight_decay=0.03,
    optimizer_type="adamw",
    profiler=None,
    augmentation_mode="extended",
    augmentation_params=None,
    should_stop_callback=None,
    use_cross_validation=False,
    k_folds=5,
    freeze_layers_ratio=0.7,
    model_log_path=None,
    model_source_info=None,
    output_dir=None,
    model_save_path=None,
):
    """
    Trenuje model na podanym zbiorze danych z wykorzystaniem optymalnych parametrów sprzętowych.
    Wszystkie parametry powinny być pobierane z pliku konfiguracyjnego JSON.
    """
    # Ustal ścieżkę logu modelu: zawsze w folderze docelowym modelu
    if model_save_path is not None:
        base, _ = os.path.splitext(model_save_path)
        model_log_path = base + ".log"
    else:
        # Jeśli ścieżka zapisu nie jest podana, użyj domyślnej nazwy z timestampem
        models_dir = os.path.join("data", "models")
        os.makedirs(models_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"training_log_{timestamp}.log"
        model_log_path = os.path.join(models_dir, log_filename)

    print(
        f"[INFO] Rozpoczynam logowanie informacji o modelu do pliku: {model_log_path}"
    )
    # Logowanie informacji o modelu na początku treningu
    log_model_info(model, model_log_path, extra_info=model_source_info)

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
    recommended_workers = hardware_profile.get("recommended_workers", 0)

    # Ustaw mixed precision
    use_mixed_precision = hardware_profile.get("use_mixed_precision", False)
    if use_mixed_precision and torch.cuda.is_available():
        # print("Używam mixed precision (zalecane przez profiler)")
        scaler = torch.amp.GradScaler()
    else:
        use_mixed_precision = False
        scaler = None
        # print("Mixed precision wyłączone (brak GPU lub niezalecane przez profiler)")

    # Inicjalizacja historii treningu
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_top_3_accuracy": [],
        "val_top_5_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1_score": [],
        "val_auc": [],
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
    elif freeze_layers_ratio > 0:
        parameters = list(model.parameters())
        num_to_freeze = int(len(parameters) * freeze_layers_ratio)

        for i, param in enumerate(parameters):
            if i < num_to_freeze:
                param.requires_grad = False
            else:
                param.requires_grad = True

    # Wybór optymalizatora
    optimizer = configure_optimizer(model, optimizer_type, learning_rate, weight_decay)

    # Konfiguracja kryterium straty
    if label_smoothing > 0:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        # print(f"Użyto CrossEntropyLoss z label smoothing: {label_smoothing}")
    else:
        criterion = nn.CrossEntropyLoss()
        # print("Użyto standardowego CrossEntropyLoss")

    # Konfiguracja schedulera
    scheduler = configure_scheduler(optimizer, lr_scheduler_type, num_epochs, patience)

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
            # Przerywamy pętlę, aby uniknąć dalszych błędów
            break

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
            all_targets = []
            all_preds = []
            all_probs = []

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

                    all_targets.extend(targets.cpu().numpy())
                    all_preds.extend(predicted.cpu().numpy())
                    all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

            val_loss = val_loss / batch_count
            val_acc = val_correct / val_total

            # Dodatkowe metryki
            y_true = np.array(all_targets)
            y_pred = np.array(all_preds)
            y_prob = np.array(all_probs)
            try:
                val_precision = precision_score(
                    y_true, y_pred, average="macro", zero_division=0
                )
                val_recall = recall_score(
                    y_true, y_pred, average="macro", zero_division=0
                )
                val_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            except Exception:
                val_precision = val_recall = val_f1 = 0.0
            try:
                if y_prob.shape[1] > 1:
                    val_auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
                else:
                    val_auc = roc_auc_score(y_true, y_prob[:, 0])
            except Exception:
                val_auc = 0.0
            try:
                val_top_3 = top_k_accuracy_score(y_true, y_prob, k=3)
            except Exception:
                val_top_3 = 0.0
            try:
                val_top_5 = top_k_accuracy_score(y_true, y_prob, k=5)
            except Exception:
                val_top_5 = 0.0

            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_precision"].append(val_precision)
            history["val_recall"].append(val_recall)
            history["val_f1_score"].append(val_f1)
            history["val_auc"].append(val_auc)
            history["val_top_3_accuracy"].append(val_top_3)
            history["val_top_5_accuracy"].append(val_top_5)

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

        if val_loader:
            print(f"Walidacja - Strata: {val_loss:.4f}")
            print(f"Walidacja - Dokładność: {val_acc:.4f}")
        else:
            val_loss = 0.0
            val_acc = 0.0
            val_top_3 = 0.0
            val_top_5 = 0.0
            val_precision = 0.0
            val_recall = 0.0
            val_f1 = 0.0
            val_auc = 0.0

        print("\n--- PODSUMOWANIE EPOKI ---")
        print(f"Czas trwania: {epoch_time:.2f}s")
        print(f"Średnia strata: {epoch_loss:.4f}")
        print(f"Średnia dokładność: {epoch_acc:.4f}")

        print(
            f"DEBUG optimized_training: Koniec epoki {epoch + 1}. Zaraz wywołam progress_callback (jeśli istnieje)."
        )
        # Wywołaj callback z postępem jeśli istnieje
        if progress_callback:
            try:
                progress_callback(
                    epoch + 1,
                    num_epochs,
                    epoch_loss,
                    epoch_acc,
                    val_loss,
                    val_acc,
                    val_top_3,
                    val_top_5,
                    val_precision,
                    val_recall,
                    val_f1,
                    val_auc,
                )
            except Exception as e_cb:
                print(f"BŁĄD podczas wywołania progress_callback: {str(e_cb)}")
                print(traceback.format_exc())

        # Na końcu każdej epoki dodajemy czyszczenie pamięci GPU:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Opcjonalnie sprawdź i wyświetl aktualnie używaną pamięć
            if device.type == "cuda":
                memory_allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
                memory_reserved = torch.cuda.memory_reserved(device) / (1024 * 1024)
                print(
                    f"GPU memory: allocated={memory_allocated:.2f}MB, reserved={memory_reserved:.2f}MB"
                )

    print(
        f"DEBUG optimized_training: Zakończono pętlę po epokach (naturalnie lub przez break). Ostatnia przetworzona epoka (0-indexed): {epoch if 'epoch' in locals() else 'nie zdefiniowano'}"
    )
    print("\n=== ZAKOŃCZENIE TRENINGU ===")
    print(f"Data zakończenia: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if val_loader:
        print(f"Najlepsza epoka: {history['best_epoch'] + 1}")
        print(f"Najlepsza strata walidacyjna: {history['best_val_loss']:.4f}")

    # Po zakończeniu treningu zapisz jeszcze raz info o modelu (np. po fine-tuningu)
    log_model_info(model, model_log_path, extra_info="Stan modelu po treningu")

    result = {
        "history": history,
        "class_names": class_names,
        "best_epoch": history.get("best_epoch", 0),
        "best_val_loss": history.get("best_val_loss", float("inf")),
        "hardware_profile": hardware_profile,
    }

    return result


def configure_optimizer(model, optimizer_type, learning_rate, weight_decay):
    """Konfiguruje optymalizator z parametrami dostosowanymi do modelu."""
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Dopasuj learning rate do wielkości modelu
    adjusted_lr = learning_rate
    if param_count > 50_000_000:  # Bardzo duży model
        adjusted_lr = learning_rate * 0.5
    elif param_count < 5_000_000:  # Mały model
        adjusted_lr = learning_rate * 2.0

    # Dopasuj weight_decay
    adjusted_wd = weight_decay
    if param_count > 20_000_000:  # Większy model
        adjusted_wd = max(weight_decay, 0.03)

    # Wybierz i skonfiguruj optymalizator
    if optimizer_type.lower() == "adamw":
        return optim.AdamW(
            model.parameters(), lr=adjusted_lr, weight_decay=adjusted_wd, eps=1e-8
        )
    elif optimizer_type.lower() == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=adjusted_lr,
            momentum=0.9,
            weight_decay=adjusted_wd,
            nesterov=True,
        )
    else:  # Adam jako domyślny
        return optim.Adam(
            model.parameters(), lr=adjusted_lr, weight_decay=adjusted_wd, eps=1e-8
        )


def configure_scheduler(
    optimizer, scheduler_type, epochs, patience=3, steps_per_epoch=100
):
    """
    Konfiguruje scheduler learning rate odpowiedni do typu
    optymalizatora i długości treningu.
    """
    if scheduler_type == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=patience
        )
    elif scheduler_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
    elif scheduler_type == "onecycle":
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]["lr"] * 10,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=10000.0,
        )
    else:
        return None


def mixup_data(x, y, alpha=0.2, device=None):
    """
    Wykonuje mixup na danych wejściowych i etykietach.
    Bezpieczna implementacja unikająca problemów z urządzeniami.
    """
    if device is None:
        device = x.device

    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    batch_size = x.size()[0]

    try:
        index = torch.randperm(batch_size, device="cpu")
        index = index.to(device)
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    except Exception as e:
        print(f"Błąd podczas mixup: {str(e)}")
        return x, y, y, 1.0


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Oblicza stratę dla mixup.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def progress_callback(
    epoch,
    num_epochs,
    train_loss,
    train_acc,
    val_loss,
    val_acc,
    val_top3,
    val_top5,
    val_precision,
    val_recall,
    val_f1,
    val_auc,
):
    # Sprawdź, czy parametry są prawidłowe
    if epoch < 1 or epoch > num_epochs:
        print(f"BŁĄD: Nieprawidłowy numer epoki: {epoch}/{num_epochs}")
        return  # Nie wyświetlaj nieprawidłowych informacji

    # W tej wersji funkcji nie drukujemy nic, ponieważ informacje o epoce
    # są już drukowane w głównej funkcji train_model_optimized.
    # To pozwoli uniknąć duplikowania komunikatów.
    pass


def track_forgetting(original_model, fine_tuned_model, original_testloader):
    """Mierzy utratę wydajności na oryginalnym zbiorze danych."""
    original_metrics = evaluate_model(original_model, original_testloader)
    finetuned_metrics = evaluate_model(fine_tuned_model, original_testloader)
    forgetting = {
        k: original_metrics[k] - finetuned_metrics[k] for k in original_metrics
    }
    return forgetting


def adaptive_layer_freezing(model, gradient_threshold=0.001):
    """Zamraża warstwy z małymi gradientami, odmraża te z dużymi."""
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            param.requires_grad = grad_norm > gradient_threshold
    return model


def compare_activations(original_model, fine_tuned_model, sample_batch):
    """Porównuje aktywacje warstw modeli dla tych samych danych wejściowych."""
    original_activations = {}
    finetuned_activations = {}

    def hook_fn(name, activations_dict):
        def hook(module, input, output):
            activations_dict[name] = output.detach()

        return hook

    # Rejestracja hooków dla obu modeli
    hooks = []
    for name, module in original_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(
                module.register_forward_hook(hook_fn(name, original_activations))
            )

    for name, module in fine_tuned_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(
                module.register_forward_hook(hook_fn(name, finetuned_activations))
            )

    # Forward pass
    with torch.no_grad():
        original_model(sample_batch)
        fine_tuned_model(sample_batch)

    # Usunięcie hooków
    for hook in hooks:
        hook.remove()

    # Porównanie aktywacji
    activation_diffs = {}
    for name in original_activations:
        if name in finetuned_activations:
            diff = torch.norm(
                original_activations[name] - finetuned_activations[name]
            ).item()
            activation_diffs[name] = diff

    return activation_diffs


def ewc_loss(model, old_model, fisher_diag, importance=1000.0):
    """Oblicza składnik straty EWC do ochrony istotnych wag."""
    loss = 0
    for (name, param), (_, param_old), (_, fisher) in zip(
        model.named_parameters(), old_model.named_parameters(), fisher_diag.items()
    ):
        loss += (fisher * (param - param_old).pow(2)).sum() * importance
    return loss
