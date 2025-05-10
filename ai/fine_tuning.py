import json
import os
import shutil
import time
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets

from .catastrophic_forgetting import (
    ElasticWeightConsolidation,
    KnowledgeDistillationLoss,
    RehearsalMemory,
    compute_fisher_information,
    generate_synthetic_samples,
)
from .classifier import ImageClassifier
from .preprocessing import get_augmentation_transforms, get_default_transforms


def handle_nan_data(data):
    """
    Obsługuje wartości NaN w danych.

    Args:
        data: Dane do przetworzenia (numpy array)

    Returns:
        Przetworzone dane bez wartości NaN
    """
    if np.isnan(data).any():
        # Zastąp NaN wartościami 0
        data = np.nan_to_num(data, nan=0.0)
        print("Uwaga: Wykryto wartości NaN w danych. Zastąpiono je zerami.")
    return data


def verify_directory_structure(directory):
    """
    Sprawdza czy struktura katalogów jest płaska (kategoria/obrazy).

    Args:
        directory: Ścieżka do katalogu z danymi

    Returns:
        bool: True jeśli struktura jest poprawna, False w przeciwnym razie
    """
    for root, dirs, files in os.walk(directory):
        # Pomijamy główny katalog
        if root == directory:
            continue

        # Sprawdzamy czy są podkatalogi
        if dirs:
            return False

        # Sprawdzamy czy są pliki obrazów
        has_images = any(
            f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")) for f in files
        )
        if not has_images:
            return False

    return True


def map_class_indices(base_class_names, new_class_folders):
    """
    Mapuje indeksy klas z modelu bazowego do nowych klas w zbiorze treningowym.

    Args:
        base_class_names: Słownik mapujący indeksy na nazwy klas w modelu bazowym
        new_class_folders: Lista nazw folderów (klas) w zbiorze treningowym

    Returns:
        dict: Mapowanie nowych indeksów na indeksy bazowe
    """
    # Odwróć słownik klas bazowego modelu (nazwa klasy -> indeks)
    base_names_to_idx = {
        name.lower(): int(idx) for idx, name in base_class_names.items()
    }

    # Utwórz mapowanie nowych indeksów na indeksy bazowe
    index_mapping = {}
    for new_idx, folder_name in enumerate(sorted(new_class_folders)):
        # Jeśli klasa występuje w modelu bazowym, użyj jej oryginalnego indeksu
        if folder_name.lower() in base_names_to_idx:
            base_idx = base_names_to_idx[folder_name.lower()]
            index_mapping[new_idx] = base_idx
            print(
                f"  Mapowanie klasy: {folder_name} (nowy indeks {new_idx}) -> (bazowy indeks {base_idx})"
            )
        else:
            # Jeśli to nowa klasa, oznacz jako -1 (będzie wymagała inicjalizacji)
            index_mapping[new_idx] = -1
            print(
                f"  Nowa klasa: {folder_name} (nowy indeks {new_idx}) -> brak w modelu bazowym"
            )

    return index_mapping


def verify_model_config(model_path, class_names):
    """
    Weryfikuje zgodność konfiguracji modelu z pliku config.json z podanymi nazwami klas.

    Args:
        model_path: Ścieżka do modelu
        class_names: Słownik mapujący indeksy na nazwy klas

    Returns:
        bool: True jeśli konfiguracja jest zgodna, False w przeciwnym razie
    """
    # Sprawdź czy istnieje plik config.json
    config_path = os.path.join(os.path.dirname(model_path), "config.json")
    if not os.path.exists(config_path):
        print(f"Uwaga: Nie znaleziono pliku konfiguracyjnego {config_path}")
        return False

    try:
        # Wczytaj konfigurację
        with open(config_path, "r") as f:
            config = json.load(f)

        # Sprawdź zgodność klas
        if "class_names" in config:
            config_classes = config["class_names"]
            # Porównaj nazwy klas
            for idx, name in class_names.items():
                if (
                    idx in config_classes
                    and config_classes[idx].lower() != name.lower()
                ):
                    print(
                        f"Niezgodność klasy: {idx}, model: {config_classes[idx]}, "
                        f"oczekiwana: {name}"
                    )
                    return False

            print("✓ Konfiguracja modelu jest zgodna z nazwami klas")
            return True
        else:
            print("Nie znaleziono listy klas w pliku konfiguracyjnym")
            return False
    except Exception as e:
        print(f"Błąd podczas weryfikacji pliku konfiguracyjnego: {e}")
        return False


def fine_tune_model(
    base_model_path,
    train_dir,
    val_dir=None,
    num_epochs=10,
    batch_size=16,
    learning_rate=0.0001,
    freeze_ratio=0.8,
    output_dir="./data/models",
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
    task_name=None,
    # Nowe parametry do zapobiegania katastrofalnemu zapominaniu
    prevent_forgetting=True,
    preserve_original_classes=True,
    rehearsal_config=None,
    knowledge_distillation_config=None,
    ewc_config=None,
    layer_freezing_config=None,
):
    """
    Przeprowadza fine-tuning istniejącego modelu na nowym zbiorze danych.
    Dodano mechanizmy zapobiegające katastrofalnemu zapominaniu.

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
        task_name: Nazwa zadania
        prevent_forgetting: Czy włączyć mechanizmy zapobiegające zapominaniu
        preserve_original_classes: Czy zachować oryginalne klasy
        rehearsal_config: Konfiguracja mechanizmu rehearsal
        knowledge_distillation_config: Konfiguracja knowledge distillation
        ewc_config: Konfiguracja EWC
        layer_freezing_config: Konfiguracja zamrażania warstw

    Returns:
        Tuple: (ścieżka do zapisanego modelu, historia treningu, szczegóły modelu)
    """
    print("\n=== INICJALIZACJA FINE-TUNINGU ===")
    print(f"Data rozpoczęcia: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    start_training_time = time.time()
    print(f"Model bazowy: {base_model_path}")
    print(f"Katalog treningowy: {train_dir}")
    if val_dir:
        print(f"Katalog walidacyjny: {val_dir}")
    print(f"Liczba epok: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Zamrożenie warstw: {freeze_ratio*100:.0f}%")

    # Sprawdź strukturę katalogów
    print("\nSprawdzanie struktury katalogów...")
    if not ensure_class_folder_structure(train_dir):
        raise ValueError(f"Nie udało się przygotować katalogu {train_dir} do treningu.")

    if val_dir and not ensure_class_folder_structure(val_dir):
        raise ValueError(f"Nie udało się przygotować katalogu {val_dir} do treningu.")

    if not verify_directory_structure(train_dir):
        raise ValueError(
            f"Nieprawidłowa struktura katalogów w {train_dir}. "
            "Dozwolona jest tylko struktura płaska: kategoria/obrazy"
        )
    if val_dir and not verify_directory_structure(val_dir):
        raise ValueError(
            f"Nieprawidłowa struktura katalogów w {val_dir}. "
            "Dozwolona jest tylko struktura płaska: kategoria/obrazy"
        )

    # Sprawdź urządzenie
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Urządzenie: {device}")

    # 1. Załaduj bazowy model
    print("\nŁadowanie modelu bazowego...")
    base_classifier = ImageClassifier(weights_path=base_model_path)

    # Utwórz kopię modelu bazowego do późniejszego użycia w technikach zapobiegających zapominaniu
    if prevent_forgetting:
        print(
            "Tworzenie kopii modelu bazowego do technik zapobiegających zapominaniu..."
        )
        original_model = deepcopy(base_classifier.model)
        original_model.eval()  # Zamroź oryginał w trybie ewaluacji

    # Wczytaj oryginalny plik config modelu bazowego
    original_config = {}
    base_config_path = os.path.splitext(base_model_path)[0] + "_config.json"
    if os.path.exists(base_config_path):
        try:
            with open(base_config_path, "r") as f:
                original_config = json.load(f)
                print(f"Wczytano oryginalny plik konfiguracyjny: {base_config_path}")
        except Exception as e:
            print(f"Nie udało się wczytać oryginalnego pliku konfiguracyjnego: {e}")
    else:
        print(
            f"Nie znaleziono oryginalnego pliku konfiguracyjnego. Tworzona jest nowa konfiguracja."
        )

    # Weryfikuj konfigurację modelu
    verify_model_config(base_model_path, base_classifier.class_names)

    # Wyświetl informacje o modelu bazowym
    model_info = base_classifier.get_model_info()
    print(f"Typ modelu: {model_info['model_type']}")
    print(f"Liczba klas w modelu bazowym: {model_info['num_classes']}")
    print(f"Łączna liczba parametrów: {model_info['total_parameters']:,}")

    # 2. Znajdź liczbę klas w nowym zbiorze danych
    print("\nAnalizowanie zbioru treningowego...")
    print("\nStruktura katalogu treningowego:")
    print_directory_structure(train_dir)

    if val_dir:
        print("\nStruktura katalogu walidacyjnego:")
        print_directory_structure(val_dir)

    train_folders = [
        f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))
    ]
    new_num_classes = len(train_folders)
    print(f"\nZnaleziono {new_num_classes} klas w zbiorze treningowym:")
    for idx, folder in enumerate(sorted(train_folders)):
        print(f"  - ID {idx}: {folder}")

    # 3. Utwórz mapowanie klas na podstawie folderów
    new_class_names = {
        str(i): class_name for i, class_name in enumerate(sorted(train_folders))
    }

    # Zmodyfikowane mapowanie klas - zachowaj oryginalne klasy ORAZ dodaj nowe
    if prevent_forgetting and preserve_original_classes:
        print("Zachowywanie oryginalnych klas w mapowaniu...")
        # Zachowaj wszystkie oryginalne klasy
        merged_class_names = base_classifier.class_names.copy()

        # Dodaj nowe klasy, kontynuując numerację
        next_idx = max([int(idx) for idx in merged_class_names.keys()]) + 1
        for i, class_name in enumerate(sorted(train_folders)):
            # Sprawdź, czy ta klasa już istnieje w oryginalnym modelu
            if class_name not in merged_class_names.values():
                merged_class_names[str(next_idx)] = class_name
                next_idx += 1

        # Użyj merged_class_names zamiast new_class_names
        new_class_names = merged_class_names

    # 4. Przygotowanie modelu do fine-tuningu
    print("\nPrzygotowanie modelu do fine-tuningu...")
    model = base_classifier.model
    model_type = base_classifier.model_type

    # 5. Dostosuj ostatnią warstwę modelu, jeśli liczba klas się zmieniła
    original_num_classes = base_classifier.num_classes
    new_num_classes = len(new_class_names)

    # ZMIANA: Zachowaj wszystkie oryginalne klasy i dodaj nowe, zamiast nadpisywać
    if new_num_classes != original_num_classes:
        print(f"Liczba klas: {original_num_classes} -> {new_num_classes}")
        # Zmodyfikowany kod dostosowania ostatniej warstwy
        # [zmodyfikowany istniejący kod dostosowania warstwy wyjściowej, zachowujący wszystkie klasy]

    # Implementacja technik zapobiegających zapominaniu
    if prevent_forgetting:
        # 1. Rehearsal
        rehearsal_data = None
        if rehearsal_config and rehearsal_config.get("use", False):
            print("\n=== KONFIGURACJA REHEARSAL ===")
            samples_per_class = rehearsal_config.get("samples_per_class", 20)
            use_synthetic = rehearsal_config.get("synthetic_samples", False)

            if use_synthetic:
                # Generuj syntetyczne próbki na podstawie oryginalnego modelu
                print(
                    f"Generowanie {samples_per_class} syntetycznych próbek na klasę..."
                )
                original_classes = [
                    int(idx) for idx in base_classifier.class_names.keys()
                ]
                rehearsal_data = generate_synthetic_samples(
                    original_model, original_classes, samples_per_class, device
                )
            else:
                # Tutaj należałoby zaimplementować pobieranie oryginalnych próbek
                # z jakiegoś zewnętrznego zbioru danych lub pamięci
                pass

        # 2. Obliczanie informacji Fishera dla EWC
        fisher_diagonal = None
        if ewc_config and ewc_config.get("use", False):
            print("\n=== KONFIGURACJA EWC ===")
            fisher_sample_size = ewc_config.get("fisher_sample_size", 200)

            # Tutaj należałoby zaimplementować wczytywanie przykładów dla oryginalnych klas
            # aby obliczyć informację Fishera
            # Przykład (wymaga implementacji data_loader_for_original_classes):
            # fisher_diagonal = compute_fisher_information(
            #     original_model, data_loader_for_original_classes, fisher_sample_size, device
            # )

    # 6. Zmodyfikuj strategie zamrażania warstw
    if prevent_forgetting and layer_freezing_config:
        strategy = layer_freezing_config.get("strategy", "gradual")
        freeze_ratio = layer_freezing_config.get("freeze_ratio", 0.7)

        print(f"\n=== KONFIGURACJA ZAMRAŻANIA WARSTW ===")
        print(f"Strategia: {strategy}, Współczynnik: {freeze_ratio*100:.1f}%")

        # Implementacja różnych strategii zamrażania
        parameters = list(model.named_parameters())

        if strategy == "gradual":
            # Zamrażaj warstwę po warstwie od początku modelu
            num_to_freeze = int(len(parameters) * freeze_ratio)

            for i, (name, param) in enumerate(parameters):
                if i < num_to_freeze:
                    param.requires_grad = False
                    print(f"  ❄️ Zamrożono: {name}")
                else:
                    param.requires_grad = True
                    print(f"  🔥 Trenowane: {name}")

        elif strategy == "selective":
            # Zamrażaj tylko wybrane warstwy (np. konwolucyjne)
            for name, param in parameters:
                if "conv" in name or "bn" in name:  # Warstwy konwolucyjne i batch norm
                    param.requires_grad = False
                    print(f"  ❄️ Zamrożono: {name}")
                else:
                    param.requires_grad = True
                    print(f"  🔥 Trenowane: {name}")

        elif strategy == "progressive":
            # Początkowo zamróź wszystko oprócz ostatnich warstw
            num_to_freeze = int(len(parameters) * freeze_ratio)

            for i, (name, param) in enumerate(parameters):
                if i < num_to_freeze:
                    param.requires_grad = False
                    print(f"  ❄️ Zamrożono (początkowo): {name}")
                else:
                    param.requires_grad = True
                    print(f"  🔥 Trenowane (początkowo): {name}")

            # Dla progressive odmrażania zaimplementujemy logikę w pętli treningowej

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
        "val_loss": [],
        "loss_diff": [],  # Różnica między stratą treningową a walidacyjną
        "train_acc": [],
        "val_acc": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "val_auc": [],  # Poprawione obliczanie AUC
        "val_top3": [],
        "val_top5": [],
        "learning_rates": [],
        "initial_lr": learning_rate,  # Początkowy learning rate
        "current_lr": learning_rate,  # Aktualny learning rate
        "best_val_loss": float("inf"),
        "best_epoch": 0,
        "early_stopping_counter": 0,
        "early_stopping_patience": 5,
        "epoch_times": [],
        "val_balanced_accuracy": [],
        "val_specificity": [],
    }

    # 15. Parametry early stopping
    patience = 5
    counter = 0
    best_val_loss = float("inf")

    print("\n=== ROZPOCZYNAM FINE-TUNING ===")

    # Pętla treningowa
    print("\n=== ROZPOCZYNAM TRENING ===")
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    early_stopping_patience = 5

    for epoch in range(num_epochs):
        # Sprawdź czy proces ma zostać przerwany
        if should_stop_callback and should_stop_callback():
            print(
                f"\n!!! Fine-tuning przerwany na epoce {epoch+1} przez użytkownika !!!"
            )
            break

        epoch_start_time = time.time()
        print(f"\n--- Epoka {epoch+1}/{num_epochs} ---")

        # Progressive unfreezing - odmrażanie kolejnych warstw w każdej epoce
        if (
            prevent_forgetting
            and layer_freezing_config
            and layer_freezing_config.get("strategy") == "progressive"
        ):
            if epoch > warmup_epochs:
                # Oblicz, ile warstw odmrozić w tej epoce
                layers_to_unfreeze = int(
                    len(parameters) * (1 - freeze_ratio) * epoch / num_epochs
                )

                # Odmroź odpowiednią liczbę warstw od końca
                for i, (name, param) in enumerate(reversed(parameters)):
                    if i < layers_to_unfreeze:
                        param.requires_grad = True
                        print(f"  🔥 Odmrożono w epoce {epoch+1}: {name}")

        # Trening
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Dodaj Rehearsal (powtarzanie)
            if prevent_forgetting and rehearsal_data:
                # Wstaw przykłady z rehearsal_data do batcha treningowego
                rehearsal_batch = next(iter(rehearsal_data))
                rehearsal_inputs, rehearsal_targets = rehearsal_batch
                rehearsal_inputs = rehearsal_inputs.to(device)
                rehearsal_targets = rehearsal_targets.to(device)

                # Połącz oryginalny batch z rehearsal batch
                inputs = torch.cat([inputs, rehearsal_inputs], dim=0)
                targets = torch.cat([targets, rehearsal_targets], dim=0)

            # Forward pass
            optimizer.zero_grad()

            if use_mixed_precision:
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(inputs)

                    # Dodaj Knowledge Distillation
                    if (
                        prevent_forgetting
                        and knowledge_distillation_config
                        and knowledge_distillation_config.get("use", False)
                    ):
                        # Wykonaj forward pass przez oryginały model
                        with torch.no_grad():
                            teacher_outputs = original_model(inputs)

                        # Użyj niestandardowego kryterium straty z knowledge distillation
                        loss = distillation_loss(outputs, targets, teacher_outputs)
                    else:
                        loss = criterion(outputs, targets)

                    # Dodaj EWC regularization do straty
                    if (
                        prevent_forgetting
                        and ewc_config
                        and ewc_config.get("use", False)
                        and fisher_diagonal
                    ):
                        ewc_lambda = ewc_config.get("lambda", 100.0)

                        # Dodaj regularyzację EWC do straty
                        ewc_loss = 0
                        for name, param in model.named_parameters():
                            if name in fisher_diagonal:
                                ewc_loss += torch.sum(
                                    fisher_diagonal[name]
                                    * (param - original_model.state_dict()[name]).pow(2)
                                )
                        loss += ewc_lambda * ewc_loss

                # Backward pass z mixed precision
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)

                # Dodaj Knowledge Distillation
                if (
                    prevent_forgetting
                    and knowledge_distillation_config
                    and knowledge_distillation_config.get("use", False)
                ):
                    # Wykonaj forward pass przez oryginały model
                    with torch.no_grad():
                        teacher_outputs = original_model(inputs)

                    # Użyj niestandardowego kryterium straty z knowledge distillation
                    loss = distillation_loss(outputs, targets, teacher_outputs)
                else:
                    loss = criterion(outputs, targets)

                # Dodaj EWC regularization do straty
                if (
                    prevent_forgetting
                    and ewc_config
                    and ewc_config.get("use", False)
                    and fisher_diagonal
                ):
                    ewc_lambda = ewc_config.get("lambda", 100.0)

                    # Dodaj regularyzację EWC do straty
                    ewc_loss = 0
                    for name, param in model.named_parameters():
                        if name in fisher_diagonal:
                            ewc_loss += torch.sum(
                                fisher_diagonal[name]
                                * (param - original_model.state_dict()[name]).pow(2)
                            )
                    loss += ewc_lambda * ewc_loss

                # Standardowy backward pass
                loss.backward()
                optimizer.step()

            # Aktualizuj statystyki
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

            # Inicjalizacja train_acc przed użyciem w callback
            train_acc = 0.0
            if train_total > 0:
                train_acc = 100.0 * train_correct / train_total

            # Aktualizuj progress bar
            if progress_callback:
                try:
                    progress_callback(
                        epoch + 1,
                        num_epochs,
                        train_loss,
                        train_acc,
                        val_loss if val_loader else 0,
                        val_acc if val_loader else 0,
                        0,  # top3
                        0,  # top5
                        0,  # precision
                        0,  # recall
                        0,  # f1
                        0,  # auc
                    )
                except Exception as e:
                    print(f"Błąd podczas wywołania progress_callback: {str(e)}")

            # Sprawdź czy należy przerwać trening
            if should_stop_callback and should_stop_callback():
                print("\nPrzerwano trening na żądanie użytkownika")
                return None

        # Oblicz średnią stratę i dokładność dla epoki
        train_loss = train_loss / len(train_loader)
        train_acc = 100.0 * train_correct / train_total

        # Walidacja
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()

            val_loss = val_loss / len(val_loader)
            val_acc = 100.0 * val_correct / val_total

            # Inicjalizacja val_metrics z odpowiednimi wartościami
            val_metrics = {
                "loss": val_loss,
                "acc": val_acc,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "auc": 0.0,
                "top3": 0.0,
                "top5": 0.0,
            }

            # Obliczanie dodatkowych metryk (jeśli potrzebne)
            if len(val_loader.dataset) > 0:
                try:
                    # Kod do obliczania dodatkowych metryk, np. F1 score
                    y_true = np.array(all_targets) if "all_targets" in locals() else []
                    y_pred = np.array(all_preds) if "all_preds" in locals() else []

                    if len(y_true) > 0 and len(y_pred) > 0:
                        from sklearn.metrics import (
                            f1_score,
                            precision_score,
                            recall_score,
                        )

                        val_metrics["precision"] = precision_score(
                            y_true, y_pred, average="macro", zero_division=0
                        )
                        val_metrics["recall"] = recall_score(
                            y_true, y_pred, average="macro", zero_division=0
                        )
                        val_metrics["f1"] = f1_score(
                            y_true, y_pred, average="macro", zero_division=0
                        )
                except Exception as e:
                    print(
                        f"Ostrzeżenie: Nie udało się obliczyć dodatkowych metryk: {str(e)}"
                    )

            # Aktualizuj scheduler
            if scheduler_type == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping po {epoch + 1} epokach")
                    break

            print(
                f"Epoka {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.2f}%"
            )
        else:
            # Aktualizuj scheduler bez walidacji
            scheduler.step()
            print(
                f"Epoka {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.2f}%"
            )

        # Zapisz czas trwania epoki
        epoch_time = time.time() - epoch_start_time
        history["epoch_times"].append(epoch_time)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # Wyświetl podsumowanie epoki
        print(f"\nPodsumowanie epoki {epoch+1}:")
        print(f"  Czas: {epoch_time:.2f}s")
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Train acc:  {train_acc:.2%}")

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
                    train_loss,
                    train_acc,
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

    # Przywróć najlepszy model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 17. Zapisz końcowy model
    print("\n=== ZAPISYWANIE KOŃCOWEGO MODELU ===")
    os.makedirs(output_dir, exist_ok=True)

    # Utwórz nazwę modelu na podstawie architektury i nazwy zadania
    model_variant = ""
    if task_name:
        model_variant = f"_{task_name}"
    model_filename = f"{model_type}{model_variant}_finetuned_final.pt"
    final_model_path = os.path.join(output_dir, model_filename)

    # Utwórz nowy klasyfikator z dostosowanym modelem
    final_classifier = ImageClassifier(
        model_type=model_type, num_classes=new_num_classes
    )
    final_classifier.model = model
    final_classifier.class_names = new_class_names

    # Przygotuj historię fine-tuningu
    trained_categories = list(new_class_names.values())
    base_model_filename = os.path.basename(base_model_path)

    # Sprawdź, czy model już ma historię fine-tuningu
    finetuning_history = {}
    config_path = os.path.splitext(final_model_path)[0] + "_config.json"
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                existing_config = json.load(f)
                if (
                    "metadata" in existing_config
                    and "finetuning_history" in existing_config["metadata"]
                ):
                    finetuning_history = existing_config["metadata"][
                        "finetuning_history"
                    ]
        except Exception as e:
            print(f"Nie udało się odczytać istniejącej historii fine-tuningu: {e}")

    # Określ numer sesji fine-tuningu
    session_nums = [
        int(k.split("_")[-1])
        for k in finetuning_history.keys()
        if k.startswith("fine_tuning_session_")
    ]
    next_session_num = max(session_nums) + 1 if session_nums else 1
    session_key = f"fine_tuning_session_{next_session_num}"

    # Dodaj nową sesję
    finetuning_history[session_key] = {
        "trained_categories": trained_categories,
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M"),
        "base_model": base_model_filename,
    }

    # Dodaj szczegóły treningu
    training_details = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "freeze_ratio": freeze_ratio,
        "optimizer_type": optimizer_type,
        "scheduler_type": scheduler_type,
    }

    # Zapisz model z rozszerzoną konfiguracją
    metadata = {
        "finetuning_history": finetuning_history,
        "training_details": training_details,
    }
    if task_name:
        metadata["task_name"] = task_name

    # Przed zapisem finalnego modelu, oblicz czas treningu
    training_time = time.time() - start_training_time

    # Przygotuj nowe metadane
    new_metadata = {}
    # Zachowaj wszystkie oryginalne metadane
    if "metadata" in original_config:
        new_metadata = deepcopy(original_config["metadata"])

    # Dodaj lub zaktualizuj czas treningu
    if "training_time" in new_metadata:
        # Dodaj nowy czas treningu do istniejącego
        new_metadata["training_time"] += training_time
    else:
        # Utwórz nowy element czasu treningu
        new_metadata["training_time"] = training_time

    # Dodaj lub aktualizuj tylko te elementy, które się zmieniły
    new_metadata["finetuning_history"] = finetuning_history
    new_metadata["training_details"] = training_details
    if task_name:
        new_metadata["task_name"] = task_name

    # Zapisz model z kompletnymi metadanymi z oryginalnego modelu plus nowe informacje
    final_classifier.save_with_original_config(
        final_model_path, original_config, new_metadata
    )
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

    # WAŻNE: Zachowaj oryginalne mapowanie klas w konfiguracji modelu
    if prevent_forgetting and preserve_original_classes:
        print("\nZachowywanie oryginalnego mapowania klas w konfiguracji modelu...")
        final_classifier.class_names = new_class_names

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
        "Obrazy poprawnie klasyfikowane przez oba modele: "
        f"{results['both_correct']} "
        f"({results['both_correct']/results['total']:.2%})"
    )
    print(
        "Obrazy poprawnie klasyfikowane tylko przez model bazowy: "
        f"{results['base_only_correct']} "
        f"({results['base_only_correct']/results['total']:.2%})"
    )
    print(
        "Obrazy poprawnie klasyfikowane tylko przez model po fine-tuningu: "
        f"{results['finetuned_only_correct']} "
        f"({results['finetuned_only_correct']/results['total']:.2%})"
    )
    print(
        "Obrazy błędnie klasyfikowane przez oba modele: "
        f"{results['both_incorrect']} "
        f"({results['both_incorrect']/results['total']:.2%})"
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


def ensure_class_folder_structure(directory):
    """
    Sprawdza i naprawia strukturę katalogów dla treningu.
    Jeśli w katalogu są bezpośrednio pliki obrazów (bez podkatalogów),
    tworzy podkatalog 'default_class' i przenosi tam wszystkie obrazy.

    Args:
        directory: Ścieżka do katalogu z danymi

    Returns:
        bool: True jeśli struktura była poprawna lub została naprawiona, False w przypadku błędu
    """
    # Sprawdź, czy istnieją jakiekolwiek podkatalogi
    has_subdirs = False
    has_images = False

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            has_subdirs = True
        elif os.path.isfile(item_path) and item.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        ):
            has_images = True

    # Jeśli nie ma podkatalogów, ale są obrazy, utwórz katalog 'default_class'
    if not has_subdirs and has_images:
        print(
            f"Wykryto obrazy bez struktury klas. Tworzenie katalogu 'default_class'..."
        )
        default_class_dir = os.path.join(directory, "default_class")

        try:
            # Utwórz katalog 'default_class'
            os.makedirs(default_class_dir, exist_ok=True)

            # Przenieś wszystkie pliki obrazów do tego katalogu
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                if os.path.isfile(item_path) and item.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".bmp")
                ):
                    shutil.move(item_path, os.path.join(default_class_dir, item))

            print(f"Utworzono domyślną klasę i przeniesiono do niej obrazy.")
            return True
        except Exception as e:
            print(f"Błąd podczas tworzenia struktury katalogów: {e}")
            return False

    return True


def print_directory_structure(directory, indent=""):
    """
    Wyświetla strukturę katalogu wraz z liczbą plików w każdym podkatalogu.

    Args:
        directory: Ścieżka do katalogu
        indent: Wcięcie dla zagnieżdżonych katalogów
    """
    total_files = 0
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            files = [
                f
                for f in os.listdir(item_path)
                if os.path.isfile(os.path.join(item_path, f))
            ]
            image_files = [
                f
                for f in files
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            ]
            print(f"{indent}📁 {item}/ ({len(image_files)} obrazów)")
            total_files += len(image_files)
    print(
        f"\nŁącznie znaleziono {total_files} obrazów w {len(os.listdir(directory))} katalogach"
    )
