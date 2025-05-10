import functools
import json
import os
import shutil
import time
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

from .catastrophic_forgetting import compute_fisher_information
from .classifier import ImageClassifier
from .preprocessing import get_augmentation_transforms, get_default_transforms


# Funkcja pomocnicza dla target_transform przy obliczaniu macierzy Fishera
def _target_transform_for_fisher(target_idx, valid_base_class_indices_str_set):
    """
    Transformuje indeksy klas.
    Zwraca target_idx, jeśli jego reprezentacja jako string znajduje się
    w zbiorze valid_base_class_indices_str_set. W przeciwnym razie zwraca -1.
    """
    target_str = str(target_idx)
    if target_str in valid_base_class_indices_str_set:
        return int(target_idx)
    return -1


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


def create_class_mapping(model_config, train_directories):
    """
    Tworzy szczegółowe mapowanie klas: model bazowy -> katalog treningowy dla celów logowania.
    (Poprzednio błędnie nazwane map_class_indices)

    Args:
        model_config: Konfiguracja modelu bazowego (słownik)
        train_directories: Lista nazw folderów (klas) w zbiorze treningowym

    Returns:
        dict: Słownik mapujący nazwy klas na ich finalne indeksy w nowym modelu.
    """
    print("\n=== SZCZEGÓŁOWE MAPOWANIE KLAS (create_class_mapping) ===")
    print("\n1. KLASY BAZOWE (z modelu):")
    print("--------------------------------")

    base_class_to_idx = {}
    class_names_source = None

    # Użycie model_config zamiast base_class_names
    if "class_names" in model_config:
        class_names_source = model_config["class_names"]
    elif "metadata" in model_config and "class_names" in model_config["metadata"]:
        class_names_source = model_config["metadata"]["class_names"]

    if class_names_source:
        for idx, name in class_names_source.items():
            base_class_to_idx[name.lower()] = int(idx)
            print(f"  [{idx}] {name}")
    else:
        print("  UWAGA: Nie znaleziono mapowania klas w konfiguracji modelu!")

    print("\n2. KLASY DO DOSZKALANIA (z folderów):")
    print("--------------------------------")
    # Użycie train_directories zamiast new_class_folders
    for class_name in sorted(train_directories):
        print(f"  - {class_name}")

    print("\n3. MAPOWANIE KLAS:")
    print("--------------------------------")
    class_mapping = {}
    max_idx = -1
    if base_class_to_idx:
        idx_values = [int(idx) for idx in base_class_to_idx.values()]
        if idx_values:
            max_idx = max(idx_values)
        else:
            max_idx = -1
        print(f"Maksymalny indeks w modelu bazowym: {max_idx}")

    for class_name in sorted(train_directories):
        class_lower = class_name.lower()
        if class_lower in base_class_to_idx:
            base_idx = base_class_to_idx[class_lower]
            class_mapping[class_name] = base_idx
            print(f"  ✓ '{class_name}' -> ID {base_idx} (z bazy)")
        else:
            max_idx += 1
            class_mapping[class_name] = max_idx
            print(f"  + '{class_name}' -> ID {max_idx} (nowa)")

    print("\n4. PODSUMOWANIE MAPOWANIA:")
    print("--------------------------------")
    num_base_classes = len(base_class_to_idx)
    num_train_classes = len(train_directories)
    print(f"- Klasy w modelu bazowym: {num_base_classes}")
    print(f"- Klasy do doszkalania: {num_train_classes}")

    num_new_classes = len([c for c in class_mapping.values() if c > max_idx])
    print(f"- Nowe klasy: {num_new_classes}")

    num_preserved_classes = 0
    if base_class_to_idx:  # Sprawdźmy czy base_class_to_idx nie jest puste
        original_max_idx = -1
        idx_values = [int(idx) for idx in base_class_to_idx.values()]
        if idx_values:
            original_max_idx = max(idx_values)

        num_preserved_classes = len(
            [c for c in class_mapping.values() if c <= original_max_idx]
        )
    print(f"- Zachowane klasy z bazy: {num_preserved_classes}")

    print("\n5. PEŁNE MAPOWANIE:")
    print("--------------------------------")
    print("Format: [ID] Nazwa klasy -> Mapped ID (Status)")
    for class_name, mapped_idx in sorted(class_mapping.items(), key=lambda x: x[1]):
        # Ustalenie statusu na podstawie pierwotnego max_idx z base_class_to_idx
        # (max_idx jest inkrementowane dla nowych klas)
        # Potrzebujemy max_idx PRZED dodaniem nowych klas

        # Rekalkulacja pierwotnego max_idx dla poprawnego statusu
        initial_max_idx = -1
        if base_class_to_idx:
            idx_values_initial = [
                int(idx) for idx in base_class_to_idx.values()
            ]  # Zmieniono nazwę by uniknąć konfliktu
            if idx_values_initial:
                initial_max_idx = max(idx_values_initial)

        status = "Nowa klasa" if mapped_idx > initial_max_idx else "Istniejąca klasa"
        print(f"  [{mapped_idx}] {class_name} -> {mapped_idx} ({status})")

    return class_mapping


def verify_model_config(model_path, class_names):
    """
    Weryfikuje zgodność konfiguracji modelu z pliku config.json
    z podanymi nazwami klas.

    Args:
        model_path: Ścieżka do modelu
        class_names: Słownik mapujący indeksy na nazwy klas

    Returns:
        bool: True jeśli konfiguracja jest zgodna, False w przeciwnym razie
    """
    config_path = os.path.splitext(model_path)[0] + "_config.json"
    if not os.path.exists(config_path):
        print(f"Uwaga: Nie znaleziono pliku konfiguracyjnego {config_path}")
        return False

    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        if "class_names" in config:
            config_classes = config["class_names"]
            for idx, name in class_names.items():
                str_idx = str(idx)  # Klucze w JSON są stringami
                if (
                    str_idx in config_classes
                    and config_classes[str_idx].lower() != name.lower()
                ):
                    print(
                        f"Niezgodność klasy: ID {idx}, model: {config_classes[str_idx]}, oczekiwana: {name}"
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


def verify_training_directories(
    train_dir, val_dir=None, valid_extensions=(".jpg", ".jpeg", ".png", ".bmp")
):
    """
    Weryfikuje strukturę katalogów treningowych i walidacyjnych.

    Args:
        train_dir: Ścieżka do katalogu treningowego
        val_dir: Ścieżka do katalogu walidacyjnego (opcjonalnie)
        valid_extensions: Dozwolone rozszerzenia plików

    Returns:
        dict: Słownik zawierający informacje o strukturze katalogów
    """
    result = {
        "train": {"directories": {}, "total_images": 0, "valid": True, "errors": []},
        "validation": {
            "directories": {},
            "total_images": 0,
            "valid": True,
            "errors": [],
        },
    }

    # Sprawdź katalog treningowy
    try:
        train_dirs = [
            d
            for d in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, d))
        ]
        if not train_dirs:
            result["train"]["errors"].append("Brak podkatalogów z klasami")
            result["train"]["valid"] = False

        for class_dir in train_dirs:
            class_path = os.path.join(train_dir, class_dir)
            image_files = [
                f
                for f in os.listdir(class_path)
                if os.path.isfile(os.path.join(class_path, f))
                and f.lower().endswith(valid_extensions)
            ]

            result["train"]["directories"][class_dir] = len(image_files)
            result["train"]["total_images"] += len(image_files)

            if len(image_files) == 0:
                error_msg = f"Brak obrazów w katalogu {class_dir}"
                result["train"]["errors"].append(error_msg)
                result["train"]["valid"] = False
    except Exception as e:
        error_msg = f"Błąd podczas sprawdzania katalogu treningowego: {str(e)}"
        result["train"]["errors"].append(error_msg)
        result["train"]["valid"] = False

    # Sprawdź katalog walidacyjny jeśli podano
    if val_dir:
        try:
            val_dirs = [
                d
                for d in os.listdir(val_dir)
                if os.path.isdir(os.path.join(val_dir, d))
            ]
            if not val_dirs:
                error_msg = "Brak podkatalogów z klasami"
                result["validation"]["errors"].append(error_msg)
                result["validation"]["valid"] = False

            for class_dir in val_dirs:
                class_path = os.path.join(val_dir, class_dir)
                image_files = [
                    f
                    for f in os.listdir(class_path)
                    if os.path.isfile(os.path.join(class_path, f))
                    and f.lower().endswith(valid_extensions)
                ]

                result["validation"]["directories"][class_dir] = len(image_files)
                result["validation"]["total_images"] += len(image_files)

                if len(image_files) == 0:
                    error_msg = f"Brak obrazów w katalogu {class_dir}"
                    result["validation"]["errors"].append(error_msg)
                    result["validation"]["valid"] = False

            train_classes = set(result["train"]["directories"].keys())
            val_classes = set(result["validation"]["directories"].keys())

            if train_classes != val_classes:
                missing_in_val = train_classes - val_classes
                missing_in_train = val_classes - train_classes

                if missing_in_val:
                    error_msg = (
                        "Brakujące klasy w katalogu walidacyjnym: "
                        f"{", ".join(missing_in_val)}"
                    )
                    result["validation"]["errors"].append(error_msg)

                if missing_in_train:
                    error_msg = (
                        "Klasy w walidacji nieobecne w treningowym: "
                        f"{", ".join(missing_in_train)}"
                    )
                    result["validation"]["errors"].append(error_msg)
                result["validation"]["valid"] = False
        except Exception as e:
            error_msg = f"Błąd podczas sprawdzania katalogu walidacyjnego: {str(e)}"
            result["validation"]["errors"].append(error_msg)
            result["validation"]["valid"] = False
    return result


def display_directory_structure(verify_result):
    """
    Wyświetla strukturę katalogów treningowych i walidacyjnych
    na podstawie wyniku weryfikacji.

    Args:
        verify_result: Wynik z `verify_training_directories`
    """
    print("\nStruktura katalogu treningowego:")
    for class_dir, count in verify_result["train"]["directories"].items():
        print(f"📁 {class_dir}/ ({count} obrazów)")
    train_total_images = verify_result["train"]["total_images"]
    train_num_dirs = len(verify_result["train"]["directories"])
    print(
        f"\nŁącznie znaleziono {train_total_images} obrazów w "
        f"{train_num_dirs} katalogach"
    )

    if verify_result["validation"]["directories"]:
        print("\nStruktura katalogu walidacyjnego:")
        val_dirs = verify_result["validation"]["directories"]
        for class_dir, count in val_dirs.items():
            print(f"📁 {class_dir}/ ({count} obrazów)")
        val_total_images = verify_result["validation"]["total_images"]
        val_num_dirs = len(verify_result["validation"]["directories"])
        print(
            f"\nŁącznie znaleziono {val_total_images} obrazów w "
            f"{val_num_dirs} katalogach"
        )

    if verify_result["train"]["errors"]:
        print("\n⚠️ Problemy w katalogu treningowym:")
        for error in verify_result["train"]["errors"]:
            print(f"  - {error}")

    if verify_result["validation"]["errors"]:
        print("\n⚠️ Problemy w katalogu walidacyjnym:")
        for error in verify_result["validation"]["errors"]:
            print(f"  - {error}")


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
    prevent_forgetting=True,
    preserve_original_classes=True,
    rehearsal_config=None,
    knowledge_distillation_config=None,
    ewc_config=None,
    layer_freezing_config=None,
    augmentation_params=None,
    preprocessing_params=None,
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
        augmentation_params: Parametry augmentacji
        preprocessing_params: Parametry preprocessingu

    Returns:
        Tuple: (ścieżka do zapisanego modelu, historia treningu, szczegóły modelu)
    """
    # Jawne ustawienie urządzenia, jeśli nie zostało podane
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używane urządzenie: {device}")

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

    # Weryfikacja struktury katalogów
    print("\n=== WERYFIKACJA STRUKTURY KATALOGÓW ===")
    directory_verification = verify_training_directories(train_dir, val_dir)
    display_directory_structure(directory_verification)

    # Sprawdź czy struktura katalogów jest poprawna
    if not directory_verification["train"]["valid"]:
        raise ValueError("Problemy ze strukturą katalogu treningowego")
    if val_dir and not directory_verification["validation"]["valid"]:
        raise ValueError("Problemy ze strukturą katalogu walidacyjnego")

    # Wczytaj konfigurację modelu bazowego
    base_config_path = os.path.splitext(base_model_path)[0] + "_config.json"
    base_config = {}
    if os.path.exists(base_config_path):
        try:
            with open(base_config_path, "r") as f:
                base_config = json.load(f)
                print(f"\nWczytano konfigurację modelu bazowego: {base_config_path}")
                # Wyświetl tablicę class_names
                if "class_names" in base_config:
                    print("\nKlasy z modelu bazowego:")
                    for idx, name in base_config["class_names"].items():
                        print(f"  - ID {idx}: {name}")
                elif (
                    "metadata" in base_config
                    and "class_names" in base_config["metadata"]
                ):
                    print("\nKlasy z modelu bazowego (z metadanych):")
                    for idx, name in base_config["metadata"]["class_names"].items():
                        print(f"  - ID {idx}: {name}")
        except Exception as e:
            print(f"Ostrzeżenie: Nie udało się wczytać konfiguracji modelu: {str(e)}")

    # Utwórz mapowanie klas
    print("\n=== MAPOWANIE KLAS ===")
    train_directories = directory_verification["train"]["directories"].keys()

    # 1. Załaduj bazowy model
    print("\nŁadowanie modelu bazowego...")
    base_classifier = ImageClassifier(weights_path=base_model_path)

    # Wyświetl klasy z modelu bazowego
    print("\nKlasy w modelu bazowym:")
    for idx, name in base_classifier.class_names.items():
        print(f"  - ID {idx}: {name}")

    # Wyświetl klasy do doszkolenia
    print("\nKlasy do doszkolenia (znalezione w folderach):")
    for idx, folder in enumerate(sorted(train_directories)):
        print(f"  - ID {idx}: {folder}")

    # Utwórz i wyświetl mapowanie
    print("\nMapowanie klas:")
    class_mapping = {}
    for class_name in sorted(train_directories):
        class_lower = class_name.lower()
        if class_lower in {
            name.lower(): idx for idx, name in base_classifier.class_names.items()
        }:
            base_idx = next(
                idx
                for idx, name in base_classifier.class_names.items()
                if name.lower() == class_lower
            )
            class_mapping[class_name] = base_idx
            print(f"  ✓ {class_name} -> ID {base_idx} (istniejąca w modelu bazowym)")
        else:
            print(f"  + {class_name} -> Nowa klasa (będzie dodana)")

    # Zachowaj oryginalne mapowanie klas
    new_class_names = base_classifier.class_names.copy()

    # Utwórz kopię modelu bazowego do późniejszego użycia w technikach zapobiegających zapominaniu
    if prevent_forgetting:
        print(
            "Tworzenie kopii modelu bazowego do technik zapobiegających zapominaniu..."
        )
        original_model_for_forgetting = deepcopy(base_classifier.model)
        original_model_for_forgetting.eval()  # Zamroź oryginał w trybie ewaluacji
        original_model_for_forgetting.to(
            device
        )  # Upewnij się, że jest na właściwym urządzeniu

    # Weryfikuj konfigurację modelu
    verify_model_config(base_model_path, base_classifier.class_names)

    # Wyświetl informacje o modelu bazowym
    model_info = base_classifier.get_model_info()
    print(f"Typ modelu: {model_info['model_type']}")
    print(f"Liczba klas w modelu bazowym: {model_info['num_classes']}")
    print(f"Łączna liczba parametrów: {model_info['total_parameters']:,}")

    # --- Ustalanie finalnej listy klas (`new_class_names`) dla nowego modelu ---
    train_directories_list = sorted(list(train_directories))

    if prevent_forgetting and preserve_original_classes:
        print(
            "\nPrzygotowywanie listy klas dla nowego modelu (zachowując klasy bazowe i dodając/aktualizując z treningowych):"
        )
        new_class_names = {str(k): v for k, v in base_classifier.class_names.items()}
        current_max_id = -1
        if new_class_names:
            current_max_id = max(int(idx_str) for idx_str in new_class_names.keys())
        base_model_classes_by_name_lower = {
            name.lower(): str(idx) for idx, name in base_classifier.class_names.items()
        }

        for train_class_name in train_directories_list:
            train_class_lower = train_class_name.lower()
            if train_class_lower in base_model_classes_by_name_lower:
                base_id_str = base_model_classes_by_name_lower[train_class_lower]
                new_class_names[base_id_str] = train_class_name
            else:
                current_max_id += 1
                new_class_names[str(current_max_id)] = train_class_name
    else:
        print(
            "\nPrzygotowywanie listy klas dla nowego modelu (tylko klasy z folderów treningowych):"
        )
        new_class_names = {}
        for i, train_class_name in enumerate(train_directories_list):
            new_class_names[str(i)] = train_class_name

    print(
        f"\nFinalna lista klas przygotowana dla nowego modelu ({len(new_class_names)} klas):"
    )
    # Tworzymy set nazw klas treningowych dla szybkiego sprawdzania
    training_class_names_set = set(train_directories_list)
    for id_str, name_val in sorted(new_class_names.items(), key=lambda x: int(x[0])):
        marker = "(doszkalana)" if name_val in training_class_names_set else ""
        print(f"  ID {id_str}: {name_val} {marker}")

    # 4. Przygotowanie modelu do fine-tuningu
    print("\nPrzygotowanie modelu do fine-tuningu...")
    model = base_classifier.model
    model_type = base_classifier.model_type

    # 5. Dostosuj ostatnią warstwę modelu, jeśli liczba klas się zmieniła
    original_num_classes = base_classifier.num_classes
    new_num_classes = len(new_class_names)

    # ZMIANA: Zachowaj wszystkie oryginalne klasy i dodaj nowe, zamiast nadpisywać
    # --- POCZĄTEK MODYFIKACJI: Adaptacja warstwy klasyfikacyjnej ---
    if new_num_classes != original_num_classes or (
        prevent_forgetting and preserve_original_classes
    ):
        print(
            f"Dostosowywanie warstwy klasyfikacyjnej. Stare klasy: {original_num_classes}, Nowe klasy: {new_num_classes}"
        )

        last_layer_name = None
        last_layer = None

        # Próba identyfikacji ostatniej warstwy nn.Linear
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                last_layer_name = name
                last_layer = module

        if last_layer is None:
            print(
                "OSTRZEŻENIE: Nie udało się automatycznie zidentyfikować ostatniej warstwy nn.Linear. Adaptacja może nie zadziałać poprawnie."
            )
            # W tym miejscu można by dodać bardziej specyficzną logikę, jeśli znamy nazwy warstw (np. fc, classifier)
            # Na przykład:
            # if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
            #     last_layer_name = 'fc'
            #     last_layer = model.fc
            # elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
            #    # itd. dla różnych typowych nazw
            #    ...

        if last_layer is not None:
            old_in_features = last_layer.in_features
            old_out_features = (
                last_layer.out_features
            )  # Powinno być równe original_num_classes

            print(
                f"  Znaleziona ostatnia warstwa liniowa: '{last_layer_name}' o wymiarach ({old_in_features}, {old_out_features})"
            )

            # Utwórz nową warstwę liniową
            new_fc = nn.Linear(old_in_features, new_num_classes)
            new_fc.to(device)  # Przenieś nową warstwę na odpowiednie urządzenie

            if prevent_forgetting and preserve_original_classes:
                print(
                    "  Zachowywanie wag dla oryginalnych klas i inicjalizacja nowych..."
                )
                # Stwórz mapowanie: nazwa_klasy -> stary_indeks_w_tensorze
                old_class_name_to_idx = {
                    name: int(idx_str)
                    for idx_str, name in base_classifier.class_names.items()
                }

                # Stwórz mapowanie: nazwa_klasy -> nowy_indeks_w_tensorze
                # new_class_names to {nowy_id_str: nazwa_klasy}
                new_idx_to_name = {
                    int(idx_str): name for idx_str, name in new_class_names.items()
                }
                # new_class_name_to_idx = {name: int(idx_str) for idx_str, name in new_class_names.items()} # Nie jest bezpośrednio używane, ale może być przydatne

                with torch.no_grad():
                    for new_idx_int, new_name in new_idx_to_name.items():
                        if new_name in old_class_name_to_idx:
                            old_idx_int = old_class_name_to_idx[new_name]
                            # Sprawdzenie, czy stary indeks jest w zakresie wag oryginalnej warstwy
                            if old_idx_int < last_layer.weight.size(
                                0
                            ) and old_idx_int < last_layer.bias.size(0):
                                new_fc.weight.data[new_idx_int] = (
                                    last_layer.weight.data[old_idx_int]
                                )
                                new_fc.bias.data[new_idx_int] = last_layer.bias.data[
                                    old_idx_int
                                ]
                                print(
                                    f"    -> Skopiowano wagi dla klasy '{new_name}' (stary idx: {old_idx_int} -> nowy idx: {new_idx_int})"
                                )
                            else:
                                print(
                                    f"    OSTRZEŻENIE: Stary indeks {old_idx_int} dla klasy '{new_name}' poza zakresem oryginalnej warstwy. Inicjalizuję losowo."
                                )
                                # Wagi dla tej klasy pozostaną losowo zainicjalizowane przez nn.Linear
                        else:
                            print(
                                f"    -> Klasa '{new_name}' (nowy idx: {new_idx_int}) jest nowa. Wagi zainicjalizowane losowo."
                            )
            else:
                print(
                    "  Inicjalizacja nowej warstwy klasyfikacyjnej od zera (losowe wagi)."
                )
                # Wagi są już losowo inicjalizowane przez nn.Linear

            # Zastąp starą warstwę nową
            # To jest nieco skomplikowane, jeśli last_layer_name zawiera kropki (np. "features.18.classifier")
            components = last_layer_name.split(".")
            current_module = model
            for comp in components[:-1]:
                if hasattr(current_module, comp):
                    current_module = getattr(current_module, comp)
                else:  # Jeśli to np. element Sequential bez nazwy, tylko indeks
                    current_module = current_module[int(comp)]

            setattr(current_module, components[-1], new_fc)
            # Zaktualizuj liczbę klas w głównym obiekcie modelu, jeśli posiada taki atrybut
            if hasattr(model, "num_classes"):
                model.num_classes = new_num_classes
            base_classifier.num_classes = (
                new_num_classes  # Oraz w naszym wrapperze Classifier
            )
            print(
                f"  Zastąpiono warstwę '{last_layer_name}' nową warstwą o wymiarach ({old_in_features}, {new_num_classes})"
            )
        else:
            print(
                "OSTRZEŻENIE: Nie można było dostosować ostatniej warstwy. Model może nie działać poprawnie."
            )
    # --- KONIEC MODYFIKACJI ---

    # Implementacja technik zapobiegających zapominaniu
    # USUNIĘTO BŁĘDNE RESETOWANIE original_model_for_forgetting
    rehearsal_data = None  # INICJALIZACJA
    fisher_diagonal = None  # INICJALIZACJA
    original_params = {}  # INICJALIZACJA

    if prevent_forgetting:
        # Domyślna konfiguracja dla EWC, jeśli nie została przekazana
        if ewc_config is None:
            ewc_config = {
                "use": True,
                "lambda": 5000.0,  # Współczynnik regularyzacji EWC
                "fisher_sample_size": 200,  # Liczba przykładów do obliczenia macierzy Fishera
            }
            print(f"Używam domyślnej konfiguracji EWC: {ewc_config}")

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
                    original_model_for_forgetting,
                    original_classes,
                    samples_per_class,
                    device,
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

            # Przygotuj data loader dla oryginalnych klas
            print("Przygotowywanie data loadera dla oryginalnych klas...")
            # Pobieramy zbiór kluczy (ID jako stringi) klas z modelu bazowego
            base_class_keys_set = set(base_classifier.class_names.keys())

            original_dataset = datasets.ImageFolder(
                train_dir,  # UWAGA: Może wymagać przefiltrowania danych tylko do klas bazowych
                transform=get_default_transforms(),
                # Używamy functools.partial zamiast lambdy
                target_transform=functools.partial(
                    _target_transform_for_fisher,
                    valid_base_class_indices_str_set=base_class_keys_set,
                ),
            )
            data_loader_for_original_classes = DataLoader(
                original_dataset,
                batch_size=min(fisher_sample_size, len(original_dataset)),
                shuffle=True,
                num_workers=2,
                pin_memory=torch.cuda.is_available(),
            )

            # Oblicz macierz Fishera dla oryginalnego modelu
            print(f"Obliczanie macierzy Fishera dla {fisher_sample_size} próbek...")
            fisher_diagonal = compute_fisher_information(
                original_model_for_forgetting,
                data_loader_for_original_classes,
                device=device,
                num_samples=fisher_sample_size,  # Dodano num_samples
            )
            print("✓ Macierz Fishera obliczona pomyślnie")

        # 3. Zapisz oryginalne parametry modelu dla EWC
        original_params = {}
        if ewc_config and ewc_config.get("use", False):
            print("Zapisywanie oryginalnych parametrów modelu (dla EWC)...")
            for name, param in original_model_for_forgetting.named_parameters():
                if param.requires_grad:
                    original_params[name] = param.data.clone()
            print("✓ Parametry (dla EWC) zapisane")

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
    # ZMIANA: Przekazanie konfiguracji do funkcji transformujących
    train_transform = get_augmentation_transforms(config=augmentation_params)
    val_transform = get_default_transforms(config=preprocessing_params)

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
    scaler = None
    if use_mixed_precision and device.type == "cuda":
        if not torch.cuda.is_available():
            print(
                "OSTRZEŻENIE: use_mixed_precision=True i device=cuda, ale torch.cuda.is_available() jest False. Sprawdź konfigurację CUDA."
            )
        scaler = torch.amp.GradScaler()
        print("Włączono CUDA mixed precision (autocast + GradScaler).")
    elif use_mixed_precision and device.type == "cpu":
        print(
            "Włączono CPU mixed precision (autocast dla bfloat16, jeśli wspierane). GradScaler nie jest używany."
        )
    else:
        print("Mieszana precyzja wyłączona lub nieobsługiwana na tym urządzeniu.")

    # 13. Przejdź do trybu treningu
    model.train()  # Model już powinien być na urządzeniu po wcześniejszym model.to(device)

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
        # Resetowanie liczników dla każdej epoki
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
            optimizer.zero_grad(
                set_to_none=True
            )  # Użyj set_to_none=True dla potencjalnej optymalizacji pamięci

            # Określ, czy autocast powinien być aktywny
            enable_autocast = use_mixed_precision and device.type in ["cuda", "cpu"]

            with torch.amp.autocast(device_type=device.type, enabled=enable_autocast):
                outputs = model(inputs)  # Linia ~946, gdzie występuje błąd

                # Dodaj Knowledge Distillation
                if (
                    prevent_forgetting
                    and knowledge_distillation_config
                    and knowledge_distillation_config.get("use", False)
                    and original_model_for_forgetting
                    is not None  # ZMIANA: Użycie original_model_for_forgetting i sprawdzenie
                ):
                    with torch.no_grad():
                        teacher_outputs = original_model_for_forgetting(
                            inputs
                        )  # ZMIANA: Użycie original_model_for_forgetting

                    # ZMIANA: Pobranie parametrów KD i wywołanie nowej funkcji straty
                    temp = knowledge_distillation_config.get("temperature", 2.0)
                    alpha_kd = knowledge_distillation_config.get("alpha", 0.5)

                    loss = distillation_loss(
                        outputs, targets, teacher_outputs, temp, alpha_kd
                    )
                    if batch_idx % 10 == 0:
                        print(
                            f"  KD loss component active. Temp: {temp}, Alpha: {alpha_kd}"
                        )
                else:
                    loss = criterion(outputs, targets)

                if (
                    prevent_forgetting
                    and ewc_config
                    and ewc_config.get("use", False)
                    and fisher_diagonal
                    and original_params
                ):
                    ewc_lambda = ewc_config.get("lambda", 100.0)
                    ewc_loss_val = 0  # Zmieniono nazwę zmiennej, aby uniknąć konfliktu
                    for name, param in model.named_parameters():
                        if name in fisher_diagonal and name in original_params:
                            diff = (param - original_params[name]) ** 2
                            ewc_loss_val += torch.sum(fisher_diagonal[name] * diff)
                    loss += ewc_lambda * ewc_loss_val
                    if batch_idx % 10 == 0:
                        print(
                            f"  EWC loss component: {ewc_loss_val.item():.6f}, Lambda: {ewc_lambda}"
                        )

            # Backward pass i krok optymalizatora
            if scaler:  # scaler istnieje tylko dla CUDA mixed precision
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:  # Standardowy backward pass dla CPU mixed precision lub braku mixed precision
                loss.backward()
                optimizer.step()

            # Aktualizuj statystyki
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

            # Inicjalizacja train_acc przed użyciem w callback
            cumulative_train_acc_for_epoch = 0.0  # Zmieniono nazwę dla jasności
            if train_total > 0:
                cumulative_train_acc_for_epoch = 100.0 * train_correct / train_total

            avg_train_loss_for_epoch = train_loss / (batch_idx + 1)

            # DODATKOWY WYDRUK KONTROLNY:
            print(
                f"DEBUG INFO: Epoka {epoch + 1}, Batch {batch_idx + 1}/{len(train_loader)}"
            )
            print(f"  Raw: train_correct={train_correct}, train_total={train_total}")
            print(
                f"  Przekazywane do callback: avg_loss={avg_train_loss_for_epoch:.4f}, cumulative_acc={cumulative_train_acc_for_epoch:.2f}%"
            )

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
            all_targets = []
            all_preds = []
            all_probs = []

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()

                    # Dodaj dla obliczenia dodatkowych metryk
                    all_targets.extend(targets.cpu().numpy())
                    all_preds.extend(predicted.cpu().numpy())
                    all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

            val_loss = val_loss / len(val_loader)
            val_acc = 100.0 * val_correct / val_total

            # Inicjalizacja val_metrics z odpowiednimi wartościami
            val_metrics = {
                "loss": val_loss,
                "acc": val_acc,
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "auc": 0.0,
                "top3": 0.0,
                "top5": 0.0,
            }

            # Próba obliczenia dodatkowych metryk, jeśli możliwe
            try:
                if len(all_targets) > 0 and len(all_preds) > 0:
                    y_true = np.array(all_targets)
                    y_pred = np.array(all_preds)
                    val_metrics["f1"] = f1_score(
                        y_true, y_pred, average="macro", zero_division=0
                    )
                    val_metrics["precision"] = precision_score(
                        y_true, y_pred, average="macro", zero_division=0
                    )
                    val_metrics["recall"] = recall_score(
                        y_true, y_pred, average="macro", zero_division=0
                    )

                    # Jeśli mamy prawdopodobieństwa, możemy obliczyć AUC i top-k
                    if len(all_probs) > 0:
                        y_prob = np.array(all_probs)
                        if y_prob.shape[1] > 2:  # Wieloklasowy problem
                            val_metrics["auc"] = roc_auc_score(
                                y_true,
                                y_prob,
                                multi_class="ovr",
                                average="macro",
                                labels=np.arange(y_prob.shape[1]),
                            )
                        elif y_prob.shape[1] == 2:  # Problem binarny
                            val_metrics["auc"] = roc_auc_score(y_true, y_prob[:, 1])

                        # Top-k metryki
                        if y_prob.shape[1] >= 3:
                            val_metrics["top3"] = top_k_accuracy_score(
                                y_true, y_prob, k=3
                            )
                        if y_prob.shape[1] >= 5:
                            val_metrics["top5"] = top_k_accuracy_score(
                                y_true, y_prob, k=5
                            )
            except Exception as e:
                print(f"Ostrzeżenie: Nie udało się obliczyć niektórych metryk: {e}")
                # Zachowamy domyślne wartości 0.0 dla metryk, których nie udało się obliczyć
        else:
            # Utwórz puste val_metrics, gdy walidacja jest wyłączona
            val_metrics = {
                "loss": 0.0,
                "acc": 0.0,
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "auc": 0.0,
                "top3": 0.0,
                "top5": 0.0,
            }

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
            print(
                f"  Val F1:     {val_metrics.get('f1', 0):.4f}"
            )  # Użyj .get() dla bezpieczeństwa
            if val_metrics.get("top3") is not None:
                print(f"  Val top-3:  {val_metrics.get('top3', 0):.2%}")
            if val_metrics.get("top5") is not None:  # Dodano wyświetlanie top5
                print(f"  Val top-5:  {val_metrics.get('top5', 0):.2%}")

        # Wywołaj callback z postępem jeśli istnieje
        if progress_callback:
            try:
                # Pobierz wartości z val_metrics, używając .get() z domyślną wartością 0
                val_loss_cb = val_metrics.get("loss", 0) if val_loader else 0
                val_acc_cb = val_metrics.get("acc", 0) if val_loader else 0
                top3_cb = val_metrics.get("top3", 0) if val_loader else 0
                top5_cb = val_metrics.get("top5", 0) if val_loader else 0
                precision_cb = val_metrics.get("precision", 0) if val_loader else 0
                recall_cb = val_metrics.get("recall", 0) if val_loader else 0
                f1_cb = val_metrics.get("f1", 0) if val_loader else 0
                auc_cb = val_metrics.get("auc", 0) if val_loader else 0

                progress_callback(
                    epoch + 1,
                    num_epochs,
                    avg_train_loss_for_epoch,  # Przekaż bieżącą średnią stratę treningową
                    cumulative_train_acc_for_epoch,  # Przekaż bieżącą dokładność treningową
                    val_loss_cb,
                    val_acc_cb,
                    top3_cb,
                    top5_cb,
                    precision_cb,
                    recall_cb,
                    f1_cb,
                    auc_cb,
                )
            except Exception as e:
                print(
                    f"Błąd podczas wywołania progress_callback na końcu epoki: {str(e)}"
                )

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
    if "metadata" in base_config:
        new_metadata = deepcopy(base_config["metadata"])

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
        final_model_path, base_config, new_metadata
    )
    print(f"Zapisano końcowy model: {final_model_path}")

    # 18. Podsumowanie fine-tuningu
    print("\n=== PODSUMOWANIE FINE-TUNINGU ===")
    print(f"Data zakończenia: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if val_loader:
        best_epoch = history["best_epoch"]
        print(f"Najlepsza epoka: {best_epoch + 1}")
        print(f"Najlepsza strata walidacyjna: {history['best_val_loss']:.4f}")

        # Bezpieczny dostęp do listy val_acc
        if "val_acc" in history and best_epoch < len(history["val_acc"]):
            print(f"Dokładność walidacji: {history['val_acc'][best_epoch]:.2%}")
        else:
            print("Nie znaleziono danych o dokładności walidacji dla najlepszej epoki.")

    # Przygotuj wynik
    result = {
        "model_path": final_model_path,
        "best_model_path": final_model_path,
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


# DODANA FUNKCJA DISTILLATION LOSS
def distillation_loss(student_outputs, labels, teacher_outputs, temperature, alpha):
    """
    Oblicza stratę dla destylacji wiedzy.
    :param student_outputs: Logity z modelu ucznia.
    :param labels: Prawdziwe etykiety (twarde).
    :param teacher_outputs: Logity z modelu nauczyciela.
    :param temperature: Temperatura do zmiękczania prawdopodobieństw.
    :param alpha: Współczynnik wagi między stratą destylacji a stratą na twardych etykietach.
    :return: Całkowita strata.
    """
    # Standardowa strata na twardych etykietach
    hard_loss = F.cross_entropy(student_outputs, labels)

    # Strata na miękkich etykietach (KL Divergence)
    soft_loss = nn.KLDivLoss(reduction="batchmean")(
        F.log_softmax(student_outputs / temperature, dim=1),
        F.softmax(teacher_outputs / temperature, dim=1),
    ) * (
        temperature * temperature
    )  # Skalowanie gradientu

    return alpha * hard_loss + (1.0 - alpha) * soft_loss
