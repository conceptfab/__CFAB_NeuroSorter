import json
import os
import time
import warnings
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
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

# Ignoruj ostrzeżenia z pyqtgraph o All-NaN slice
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="All-NaN slice encountered"
)
# Ignoruj ostrzeżenia z sklearn o undefined metrics
warnings.filterwarnings(
    "ignore", category=UserWarning, message="Only one class is present in y_true"
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Precision is ill-defined",  # Częste przy małej liczbie próbek
)
warnings.filterwarnings("ignore", category=UserWarning, message="Recall is ill-defined")

# Załóżmy, że te importy działają w Twoim środowunku
from .catastrophic_forgetting import compute_fisher_information
from .classifier import ImageClassifier
from .preprocessing import get_augmentation_transforms, get_default_transforms


# Funkcja pomocnicza dla target_transform przy obliczaniu macierzy Fishera
def _target_transform_for_fisher_ewc(
    original_target_idx, class_name_in_dataset, valid_original_class_map
):
    """
    Transformuje etykiety dla danych do obliczenia macierzy Fishera.
    `original_target_idx` to indeks z ImageFolder (0...k-1 dla klas w folderze).
    `class_name_in_dataset` to nazwa folderu/klasy.
    `valid_original_class_map` to słownik {oryg_nazwa_klasy_z_configu: oryg_idx_liczbowy_z_configu}.

    Zwraca oryginalny_idx_liczbowy_z_configu, jeśli klasa jest jedną z bazowych.
    W przeciwnym razie -1 (do odfiltrowania).
    """
    class_name_lower = class_name_in_dataset.lower()
    for original_name, original_numeric_idx in valid_original_class_map.items():
        if original_name.lower() == class_name_lower:
            return original_numeric_idx
    return -1


class EWCFilterDataset(Dataset):
    """
    Wrapper dla datasetu, filtruje próbki i transformuje etykiety dla EWC.
    """

    def __init__(self, original_dataset, valid_original_class_map_for_ewc):
        self.original_dataset = original_dataset
        # `valid_original_class_map_for_ewc` to {name_lower: original_numeric_idx}
        self.valid_original_class_map_for_ewc = valid_original_class_map_for_ewc
        self.indices = []
        self.transformed_targets = []

        for i in range(len(self.original_dataset.samples)):
            _, original_folder_idx = self.original_dataset.samples[i]
            class_name_in_folder = self.original_dataset.classes[original_folder_idx]

            transformed_target = -1
            # Mapowanie nazwy folderu na oryginalny indeks numeryczny z modelu bazowego.
            # `valid_original_class_map_for_ewc` to:
            # {oryg_nazwa_klasy_lower: oryginalny_idx_numeryczny_z_configu_modelu_bazowego}
            if class_name_in_folder.lower() in self.valid_original_class_map_for_ewc:
                transformed_target = self.valid_original_class_map_for_ewc[
                    class_name_in_folder.lower()
                ]

            if transformed_target != -1:
                self.indices.append(i)
                self.transformed_targets.append(transformed_target)

        if not self.indices:
            print(
                "OSTRZEŻENIE EWCFilterDataset: Nie znaleziono pasujących próbek dla klas bazowych."
            )

    def __getitem__(self, index):
        original_data_index = self.indices[index]
        # Ignorujemy oryginalną etykietę z datasetu
        data, _ = self.original_dataset[original_data_index]
        target = self.transformed_targets[index]
        return data, target

    def __len__(self):
        return len(self.indices)


def handle_nan_data(data):
    if np.isnan(data).any():
        data = np.nan_to_num(data, nan=0.0)
        print("Uwaga: Wykryto wartości NaN w danych. Zastąpiono je zerami.")
    return data


def handle_nan_in_plots(data):
    if isinstance(data, np.ndarray) and np.isnan(data).any():
        median_val = np.nanmedian(data)
        if np.isnan(median_val):  # Jeśli wszystko NaN, median też będzie NaN
            median_val = 0.0
        return np.nan_to_num(data, nan=median_val)
    return data


# ... (verify_directory_structure, create_class_mapping, verify_model_config, verify_training_directories, display_directory_structure - bez zmian)
# Te funkcje są głównie diagnostyczne lub przygotowawcze i nie wydają się być bezpośrednią przyczyną problemów z parametrami treningu.
# Kluczowe zmiany będą w `fine_tune_model`.
# Poniżej te funkcje bez zmian, aby zachować kompletność pliku.
def verify_directory_structure(directory):
    for root, dirs, files in os.walk(directory):
        if root == directory:
            continue
        if dirs:  # Jeśli są podkatalogi w podkatalogach klas
            # Sprawdźmy, czy te podkatalogi nie zawierają obrazów bezpośrednio
            is_nested_class_structure = True
            for sub_dir in dirs:
                sub_dir_path = os.path.join(root, sub_dir)
                has_images_in_sub_dir = any(
                    f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
                    for f in os.listdir(sub_dir_path)
                    if os.path.isfile(os.path.join(sub_dir_path, f))
                )
                if (
                    not has_images_in_sub_dir
                ):  # Jeśli podkatalog nie ma obrazów, to ok (np. pusty)
                    pass
                else:  # Jeśli ma obrazy, to struktura nie jest płaska
                    # Ale ImageFolder może sobie z tym poradzić, jeśli to jest `train/classA/type1/img.jpg`
                    # Problem jest, jeśli `train/classA/img.jpg` ORAZ `train/classA/type1/img.jpg`
                    pass  # Na razie akceptujemy głębszą strukturę, ImageFolder powinien to ogarnąć.
            # if not is_nested_class_structure:
            # return False

        # Sprawdzamy czy są pliki obrazów w bieżącym `root` (który powinien być folderem klasy)
        has_images_in_class_folder = any(
            f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")) for f in files
        )
        # Jeśli `root` to folder klasy i nie ma w nim obrazów, ale ma podfoldery `dirs` z obrazami, to jest ok.
        # Problem, jeśli folder klasy nie ma ani obrazów, ani podfolderów z obrazami.
        if (
            not has_images_in_class_folder and not dirs
        ):  # Jeśli jest to liść i nie ma obrazów
            # Sprawdźmy, czy to nie jest po prostu pusty folder klasy - to jest problem dla ImageFolder
            if not os.listdir(root):  # Jeśli folder jest kompletnie pusty
                print(f"Ostrzeżenie: Pusty folder klasy napotkany: {root}")
            # else: ma jakieś pliki, ale nie obrazy - ImageFolder to zignoruje
    return True  # Uproszczona weryfikacja, ImageFolder jest dość elastyczny


def create_class_mapping(model_config, train_directories_names):
    print("\n=== SZCZEGÓŁOWE MAPOWANIE KLAS (create_class_mapping - diagnostyczne) ===")
    print("UWAGA: Ta funkcja służy głównie do celów diagnostycznych i logowania.")
    print(
        "Faktyczne mapowanie klas używane przez model jest tworzone dynamicznie w `fine_tune_model`."
    )

    base_class_to_idx_from_config = {}
    class_names_source = None

    if "class_names" in model_config:
        class_names_source = model_config["class_names"]
    elif "metadata" in model_config and "class_names" in model_config["metadata"]:
        class_names_source = model_config["metadata"]["class_names"]

    if class_names_source:
        print("\n1. KLASY BAZOWE (z konfiguracji modelu):")
        for idx_str, name in class_names_source.items():
            try:
                base_class_to_idx_from_config[name.lower()] = int(idx_str)
                print(f"  [{idx_str}] {name}")
            except ValueError:
                print(
                    f"  OSTRZEŻENIE: Nieprawidłowy format indeksu '{idx_str}' dla klasy '{name}' w konfiguracji."
                )
    else:
        print("\n1. KLASY BAZOWE: Nie znaleziono mapowania klas w konfiguracji modelu!")

    print("\n2. KLASY DO DOSZKALANIA (z folderów treningowych):")
    for class_name in sorted(train_directories_names):
        print(f"  - {class_name}")

    print("\n3. MAPOWANIE KLAS (symulacja na podstawie configu i folderów):")
    final_mapping_simulation = {}
    current_max_idx_sim = -1
    if base_class_to_idx_from_config:
        try:
            idx_values = [int(idx) for idx in base_class_to_idx_from_config.values()]
            if idx_values:
                current_max_idx_sim = max(idx_values)
        except ValueError:
            print("  OSTRZEŻENIE: Nie można ustalić max_idx z konfiguracji bazowej.")

    initial_max_idx_from_config_sim = (
        current_max_idx_sim  # Zapamiętaj max_idx PRZED dodaniem nowych
    )

    for class_name_train in sorted(train_directories_names):
        class_lower_train = class_name_train.lower()
        if class_lower_train in base_class_to_idx_from_config:
            original_idx = base_class_to_idx_from_config[class_lower_train]
            final_mapping_simulation[class_name_train] = original_idx
            print(f"  ✓ '{class_name_train}' -> mapowana na ID {original_idx} (z bazy)")
        else:
            current_max_idx_sim += 1
            final_mapping_simulation[class_name_train] = current_max_idx_sim
            print(
                f"  + '{class_name_train}' -> mapowana na nowe ID {current_max_idx_sim} (nowa)"
            )

    # Dodaj klasy z bazy, które nie są w `train_directories_names` (jeśli `preserve_original_classes` byłoby True)
    # To jest tylko symulacja, więc nie komplikujmy zbytnio.

    print("\n4. PODSUMOWANIE MAPOWANIA (symulacja):")
    # ... (reszta logiki podsumowania może być skomplikowana i myląca, bo to tylko symulacja)
    return final_mapping_simulation


def verify_model_config(model_path, class_names_from_loaded_model_object):
    config_path = os.path.splitext(model_path)[0] + "_config.json"
    if not os.path.exists(config_path):
        print(f"Uwaga: Nie znaleziono pliku konfiguracyjnego {config_path}")
        return (
            True  # Nie ma czego weryfikować, więc uznajemy za "zgodne" (brak konfliktu)
        )

    try:
        with open(config_path, "r") as f:
            config_from_file = json.load(f)

        # Sprawdź class_names
        class_names_in_file = config_from_file.get(
            "class_names",
            (
                config_from_file.get("metadata", {}).get("class_names")
                if "metadata" in config_from_file
                else None
            ),
        )

        if class_names_in_file and class_names_from_loaded_model_object:
            # Porównaj czy zestawy kluczy (indeksów) są takie same
            if set(str(k) for k in class_names_in_file.keys()) != set(
                str(k) for k in class_names_from_loaded_model_object.keys()
            ):
                print(
                    f"Niezgodność w zestawie indeksów klas: Plik: {sorted(class_names_in_file.keys())}, Obiekt: {sorted(class_names_from_loaded_model_object.keys())}"
                )
                # return False # Może być OK, jeśli obiekt ma już nowe klasy

            # Porównaj nazwy dla wspólnych indeksów
            for idx_obj, name_obj in class_names_from_loaded_model_object.items():
                str_idx_obj = str(idx_obj)
                if str_idx_obj in class_names_in_file:
                    name_file = class_names_in_file[str_idx_obj]
                    if name_file.lower() != name_obj.lower():
                        print(
                            f"Niezgodność nazwy klasy dla ID {str_idx_obj}: Plik='{name_file}', Obiekt='{name_obj}'"
                        )
                        return False
            print(
                "✓ Konfiguracja class_names z pliku jest zgodna z obiektem modelu (dla wspólnych ID)."
            )
        elif (
            class_names_in_file is None
            and class_names_from_loaded_model_object is not None
        ):
            print(
                "Ostrzeżenie: Brak class_names w pliku konfiguracyjnym, ale są w obiekcie modelu."
            )
        elif (
            class_names_in_file is not None
            and class_names_from_loaded_model_object is None
        ):
            print(
                "Ostrzeżenie: Są class_names w pliku konfiguracyjnym, ale brak w obiekcie modelu."
            )
        # Jeśli oba None, to OK.

        # Można dodać weryfikację innych pól jak 'model_type', 'num_classes'
        if "model_type" in config_from_file and hasattr(
            ImageClassifier, "model_type_attribute_in_object"
        ):  # Załóżmy, że obiekt ma atrybut
            if config_from_file["model_type"] != getattr(
                ImageClassifier, "model_type_attribute_in_object", None
            ):
                # ...
                pass

        return True
    except Exception as e:
        print(f"Błąd podczas weryfikacji pliku konfiguracyjnego: {e}")
        return False


def verify_training_directories(
    train_dir, val_dir=None, valid_extensions=(".jpg", ".jpeg", ".png", ".bmp")
):
    result = {
        "train": {"directories": {}, "total_images": 0, "valid": True, "errors": []},
        "validation": {
            "directories": {},
            "total_images": 0,
            "valid": True,
            "errors": [],
        },
    }

    def check_dir(path, type_key):
        try:
            if not os.path.exists(path) or not os.path.isdir(path):
                result[type_key]["errors"].append(
                    f"Katalog {path} nie istnieje lub nie jest katalogiem."
                )
                result[type_key]["valid"] = False
                return

            class_dirs = [
                d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))
            ]
            if not class_dirs:
                result[type_key]["errors"].append("Brak podkatalogów z klasami.")
                result[type_key]["valid"] = False

            for class_dir_name in class_dirs:
                class_path_full = os.path.join(path, class_dir_name)
                try:
                    image_files = [
                        f
                        for f in os.listdir(class_path_full)
                        if os.path.isfile(os.path.join(class_path_full, f))
                        and f.lower().endswith(valid_extensions)
                    ]
                    num_images = len(image_files)
                    result[type_key]["directories"][class_dir_name] = num_images
                    result[type_key]["total_images"] += num_images
                    if num_images == 0:
                        result[type_key]["errors"].append(
                            f"Brak obrazów w katalogu klasy: {class_path_full}"
                        )
                        # result[type_key]["valid"] = False # Pusty folder klasy niekoniecznie jest błędem krytycznym dla ImageFolder
                except Exception as e_inner:
                    result[type_key]["errors"].append(
                        f"Błąd odczytu katalogu klasy {class_path_full}: {e_inner}"
                    )
                    result[type_key]["valid"] = False

        except Exception as e_outer:
            result[type_key]["errors"].append(
                f"Błąd podczas sprawdzania katalogu {type_key}: {e_outer}"
            )
            result[type_key]["valid"] = False

    check_dir(train_dir, "train")
    if val_dir:
        check_dir(val_dir, "validation")
        if (
            result["train"]["valid"] and result["validation"]["valid"]
        ):  # Tylko jeśli oba są wstępnie OK
            train_classes_set = set(result["train"]["directories"].keys())
            val_classes_set = set(result["validation"]["directories"].keys())
            if train_classes_set != val_classes_set:
                # To jest ostrzeżenie, niekoniecznie błąd krytyczny, jeśli celowo tak jest.
                # Ale dla typowego fine-tuningu powinny być te same klasy.
                msg = "Zestawy klas w katalogu treningowym i walidacyjnym różnią się."
                if train_classes_set - val_classes_set:
                    msg += (
                        f" Brakuje w walidacji: {train_classes_set - val_classes_set}."
                    )
                if val_classes_set - train_classes_set:
                    msg += f" Dodatkowe w walidacji: {val_classes_set - train_classes_set}."
                result["validation"]["errors"].append(msg)
                # result["validation"]["valid"] = False # Można odkomentować, jeśli to ma być błąd
    return result


def display_directory_structure(verify_result):
    # Bez zmian
    print("\nStruktura katalogu treningowego:")
    for class_dir, count in verify_result["train"]["directories"].items():
        print(f"📁 {class_dir}/ ({count} obrazów)")
    train_total_images = verify_result["train"]["total_images"]
    train_num_dirs = len(verify_result["train"]["directories"])
    print(
        f"\nŁącznie znaleziono {train_total_images} obrazów w "
        f"{train_num_dirs} katalogach (trening)"
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
            f"{val_num_dirs} katalogach (walidacja)"
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
    freeze_ratio=0.8,  # Domyślny freeze_ratio, może być nadpisany przez layer_freezing_config
    output_dir="./data/models",
    optimizer_type="adamw",
    scheduler_type="plateau",
    device=None,
    progress_callback=None,
    should_stop_callback=None,
    label_smoothing=0.1,
    weight_decay=0.01,
    warmup_epochs=1,
    use_mixup=False,  # Czy stosować mixup na danych WEJŚCIOWYCH
    use_mixed_precision=True,
    task_name=None,
    prevent_forgetting=True,
    preserve_original_classes=True,
    rehearsal_config=None,  # np. {"use": True, "rehearsal_data_path": "path/to/old_data", "samples_per_class": 10}
    knowledge_distillation_config=None,  # np. {"use": True, "temperature": 2.0, "alpha": 0.5}
    ewc_config=None,  # np. {"use": True, "lambda": 1000.0, "fisher_sample_size": 200, "adaptive_lambda": True}
    layer_freezing_config=None,  # np. {"strategy": "gradual", "freeze_ratio": 0.7}
    augmentation_params=None,
    preprocessing_params=None,
    use_green_diffusion=False,
    early_stopping_patience=5,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używane urządzenie: {device}")

    print("\n=== INICJALIZACJA FINE-TUNINGU ===")
    # ... (logowanie parametrów wejściowych - bez zmian)
    start_training_time = time.time()

    print("\n=== WERYFIKACJA STRUKTURY KATALOGÓW ===")
    dir_ver_result = verify_training_directories(train_dir, val_dir)
    display_directory_structure(dir_ver_result)
    if not dir_ver_result["train"]["valid"]:
        raise ValueError(
            f"Problemy ze strukturą katalogu treningowego: {dir_ver_result['train']['errors']}"
        )
    if val_dir and not dir_ver_result["validation"]["valid"]:
        # To może być tylko ostrzeżenie, jeśli np. klasy się różnią celowo
        print(
            f"Ostrzeżenie dotyczące katalogu walidacyjnego: {dir_ver_result['validation']['errors']}"
        )

    # --- Ładowanie modelu bazowego i jego konfiguracji ---
    print(f"\nŁadowanie modelu bazowego: {base_model_path}")
    try:
        base_classifier = ImageClassifier(weights_path=base_model_path)
        base_classifier.model.to(
            device
        )  # Kluczowe: przenieś model na urządzenie OD RAZU
    except Exception as e:
        raise RuntimeError(
            f"Nie udało się załadować modelu bazowego z {base_model_path}: {e}"
        )

    base_config_path = os.path.splitext(base_model_path)[0] + "_config.json"
    base_config_from_file = {}
    if os.path.exists(base_config_path):
        with open(base_config_path, "r") as f:
            base_config_from_file = json.load(f)
        print(f"Wczytano konfigurację modelu bazowego z pliku: {base_config_path}")

    # `base_classifier.class_names` to {idx_str: name} z pliku _config.json powiązanego z modelem
    # `base_classifier.num_classes` to liczba klas z tego samego pliku lub z modelu
    original_class_names_from_config = (
        base_classifier.class_names if base_classifier.class_names else {}
    )
    original_num_classes_from_config = base_classifier.num_classes

    # Rzeczywista liczba wyjść ostatniej warstwy modelu bazowego
    # To jest ważne dla adaptacji głowicy
    actual_base_model_output_neurons = -1
    for _, module in reversed(
        list(base_classifier.model.named_modules())
    ):  # Szukaj od końca
        if isinstance(module, nn.Linear):
            actual_base_model_output_neurons = module.out_features
            break
    if actual_base_model_output_neurons == -1:
        raise RuntimeError(
            "Nie udało się ustalić liczby neuronów wyjściowych w modelu bazowym."
        )

    print(
        f"Model bazowy - klasy z config: {len(original_class_names_from_config)} (num_classes: {original_num_classes_from_config}), "
        f"rzeczywiste neurony wyjściowe: {actual_base_model_output_neurons}"
    )

    if original_num_classes_from_config != actual_base_model_output_neurons:
        print(
            f"OSTRZEŻENIE: Liczba klas w konfiguracji ({original_num_classes_from_config}) "
            f"różni się od rzeczywistej liczby neuronów wyjściowych modelu bazowego ({actual_base_model_output_neurons}). "
            f"Będziemy polegać na liczbie neuronów."
        )
        # Można by próbować naprawić `original_class_names_from_config` lub rzucić błąd.

    # Kopia modelu bazowego dla technik zapobiegających zapominaniu
    original_model_for_forgetting_techniques = None
    if prevent_forgetting:
        print(
            "Tworzenie głębokiej kopii modelu bazowego dla technik zapobiegających zapominaniu..."
        )
        original_model_for_forgetting_techniques = deepcopy(base_classifier.model)
        original_model_for_forgetting_techniques.eval()
        original_model_for_forgetting_techniques.to(
            device
        )  # Już powinno być, ale dla pewności

    # --- Przygotowanie danych treningowych i walidacyjnych ---
    train_transform = get_augmentation_transforms(config=augmentation_params)
    val_transform = get_default_transforms(config=preprocessing_params)

    print("\nŁadowanie danych treningowych (ImageFolder)...")
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    # `train_dataset.class_to_idx` to {nazwa_folderu: idx_tensora_0_N-1}
    # To jest podstawowe mapowanie dla bieżącego zadania treningowego.
    current_task_class_to_idx = train_dataset.class_to_idx
    current_task_idx_to_class_name = {
        v: k for k, v in current_task_class_to_idx.items()
    }
    num_classes_in_current_task = len(current_task_class_to_idx)

    print(
        f"Znaleziono {len(train_dataset)} obrazów treningowych w {num_classes_in_current_task} klasach (folderach):"
    )
    for name, idx in sorted(
        current_task_class_to_idx.items(), key=lambda item: item[1]
    ):
        print(f"  Folder '{name}' -> ID tensora {idx}")

    # --- Definiowanie klas dla nowego modelu (`final_class_names_for_new_model`) ---
    # To będzie słownik {nowy_idx_str_globalny: nazwa_klasy_globalna}
    # Ten "globalny" indeks będzie używany w ostatecznej głowicy modelu i zapisany w configu.
    final_class_names_for_new_model = {}

    if prevent_forgetting and preserve_original_classes:
        print(
            "Zachowywanie klas z modelu bazowego i dodawanie/aktualizowanie z bieżącego zadania..."
        )
        # Kopiujemy wszystkie klasy z oryginalnego configu modelu bazowego
        # `original_class_names_from_config` to {oryg_idx_str: oryg_nazwa}
        for orig_idx_str, orig_name in original_class_names_from_config.items():
            final_class_names_for_new_model[str(orig_idx_str)] = (
                orig_name  # Zachowaj oryginalne indeksy i nazwy
            )

        max_existing_idx = -1
        if final_class_names_for_new_model:
            try:
                max_existing_idx = max(
                    int(k) for k in final_class_names_for_new_model.keys()
                )
            except ValueError:  # Jeśli klucze nie są liczbami
                print(
                    "OSTRZEŻENIE: Klucze w original_class_names_from_config nie są prawidłowymi liczbami. Resetuję max_existing_idx."
                )
                final_class_names_for_new_model = (
                    {}
                )  # Zacznij od nowa, bo indeksy są złe
                max_existing_idx = -1

        # Sprawdź, czy klasy z bieżącego zadania już istnieją (po nazwie)
        # Jeśli tak, użyj ich oryginalnego indeksu. Jeśli nie, dodaj jako nowe.
        existing_names_lower_to_idx_str = {
            name.lower(): idx_str
            for idx_str, name in final_class_names_for_new_model.items()
        }

        for (
            task_class_name,
            _,
        ) in current_task_class_to_idx.items():  # Iteruj po nazwach folderów
            task_class_name_lower = task_class_name.lower()
            if task_class_name_lower in existing_names_lower_to_idx_str:
                # Klasa z zadania już istnieje w `final_class_names_for_new_model`
                # Możemy zaktualizować nazwę, jeśli np. wielkość liter się zmieniła, ale zachowujemy jej stary ID
                existing_idx_str = existing_names_lower_to_idx_str[
                    task_class_name_lower
                ]
                final_class_names_for_new_model[existing_idx_str] = (
                    task_class_name  # Użyj nazwy z folderu
                )
            else:
                # To jest nowa klasa, której nie było w modelu bazowym
                max_existing_idx += 1
                new_idx_str = str(max_existing_idx)
                final_class_names_for_new_model[new_idx_str] = task_class_name
                existing_names_lower_to_idx_str[task_class_name_lower] = (
                    new_idx_str  # Dodaj do mapy na przyszłość
                )
    else:
        # Tryb "nadpisania" lub brak zachowania klas. Nowy model będzie miał tylko klasy z bieżącego zadania.
        # Indeksy będą 0... (num_classes_in_current_task - 1)
        print(
            "Tworzenie mapowania klas tylko na podstawie bieżącego zadania treningowego."
        )
        for task_idx_0_N, task_class_name in current_task_idx_to_class_name.items():
            final_class_names_for_new_model[str(task_idx_0_N)] = task_class_name

    num_total_classes_for_new_head = len(final_class_names_for_new_model)
    print(
        f"\nFinalna liczba klas dla głowicy nowego modelu: {num_total_classes_for_new_head}"
    )
    print("Mapowanie (Globalny ID -> Nazwa Klasy) dla nowego modelu:")
    for final_idx_str, final_name in sorted(
        final_class_names_for_new_model.items(), key=lambda x: int(x[0])
    ):
        marker = (
            "(z zadania)"
            if final_name in current_task_class_to_idx
            else "(z bazy, nie w zadaniu)"
        )
        print(f"  ID {final_idx_str}: {final_name} {marker}")

    # --- Modyfikacja głowicy modelu ---
    model_to_train = base_classifier.model  # To jest model, który będziemy modyfikować

    if num_total_classes_for_new_head != actual_base_model_output_neurons:
        print(
            f"Adaptacja głowicy: Stare neurony: {actual_base_model_output_neurons}, Nowe neurony: {num_total_classes_for_new_head}"
        )

        # Znajdź ostatnią warstwę liniową i jej rodzica
        last_layer_name_path = None
        last_linear_module = None
        parent_module = model_to_train
        last_attr_name = None

        for name, module in model_to_train.named_modules():
            if isinstance(module, nn.Linear):
                last_layer_name_path = name
                last_linear_module = module

        if last_linear_module is None:
            raise RuntimeError(
                "Nie znaleziono warstwy nn.Linear do zastąpienia w modelu."
            )

        path_components = last_layer_name_path.split(".")
        current_module_trace = model_to_train
        for component_name in path_components[:-1]:
            current_module_trace = getattr(current_module_trace, component_name)
        parent_module = current_module_trace
        last_attr_name = path_components[-1]

        old_in_features = last_linear_module.in_features
        new_head_layer = nn.Linear(old_in_features, num_total_classes_for_new_head)
        new_head_layer.to(device)

        if prevent_forgetting and preserve_original_classes:
            print("Kopiowanie wag ze starej głowicy do nowej dla zachowanych klas...")
            # `original_class_names_from_config` to {oryg_idx_str: oryg_nazwa}
            # `final_class_names_for_new_model` to {nowy_idx_str_globalny: nazwa_klasy_globalna}

            # Mapowanie: nazwa_oryginalna_lower -> oryginalny_idx_liczbowy
            orig_name_lower_to_orig_numeric_idx = {
                name.lower(): int(idx_str)
                for idx_str, name in original_class_names_from_config.items()
            }

            with torch.no_grad():
                for (
                    final_idx_str,
                    final_name,
                ) in final_class_names_for_new_model.items():
                    final_numeric_idx = int(final_idx_str)
                    final_name_lower = final_name.lower()

                    if final_name_lower in orig_name_lower_to_orig_numeric_idx:
                        # Ta klasa (wg nazwy) istniała w modelu bazowym.
                        orig_numeric_idx_for_this_name = (
                            orig_name_lower_to_orig_numeric_idx[final_name_lower]
                        )

                        # Sprawdź, czy ten oryginalny indeks jest w granicach starej warstwy
                        if (
                            orig_numeric_idx_for_this_name
                            < last_linear_module.out_features
                        ):
                            new_head_layer.weight.data[final_numeric_idx] = (
                                last_linear_module.weight.data[
                                    orig_numeric_idx_for_this_name
                                ]
                            )
                            new_head_layer.bias.data[final_numeric_idx] = (
                                last_linear_module.bias.data[
                                    orig_numeric_idx_for_this_name
                                ]
                            )
                            print(
                                f"  Skopiowano wagi dla '{final_name}' (orig ID {orig_numeric_idx_for_this_name} -> new ID {final_numeric_idx})"
                            )
                        else:
                            print(
                                f"  OSTRZEŻENIE: Oryginalny ID {orig_numeric_idx_for_this_name} dla '{final_name}' poza zakresem starej głowicy ({last_linear_module.out_features}). Inicjalizacja losowa dla nowego ID {final_numeric_idx}."
                            )
                    # else: Klasa jest nowa w `final_class_names_for_new_model`, wagi pozostają losowo zainicjalizowane.

        setattr(parent_module, last_attr_name, new_head_layer)
        print(
            f"Zastąpiono warstwę '{last_layer_name_path}' nową głowicą ({old_in_features} -> {num_total_classes_for_new_head})."
        )
    else:
        print("Głowica modelu nie wymaga adaptacji (liczba klas bez zmian).")

    # --- Konfiguracja technik zapobiegających zapominaniu ---
    # (EWC, Rehearsal, KD) - z `original_model_for_forgetting_techniques`
    # `model_to_train` to model z potencjalnie nową głowicą.

    rehearsal_loader = None
    if prevent_forgetting and rehearsal_config and rehearsal_config.get("use", False):
        print("\nKonfigurowanie Rehearsal...")
        rehearsal_data_path = rehearsal_config.get("rehearsal_data_path")
        if rehearsal_data_path and os.path.exists(rehearsal_data_path):
            # Dane do rehearsal powinny mieć taką samą strukturę folderów jak train_dir
            # i używać tych samych transformacji.
            # Etykiety z rehearsal_loader będą mapowane przez `current_task_class_to_idx`
            # jeśli klasy rehearsal są częścią bieżącego zadania.
            # Jeśli klasy rehearsal są "stare" i nie ma ich w `train_dir`, to jest problem.
            # Rehearsal powinien dostarczać pary (obraz, globalny_idx_z_final_class_names).
            # To jest skomplikowane.
            # Uproszczenie: zakładamy, że `rehearsal_data_path` zawiera podfoldery klas,
            # które są częścią `final_class_names_for_new_model`.
            # Najlepiej, jeśli `rehearsal_data_path` zawiera tylko stare klasy.
            # Wtedy potrzebujemy specjalnego `target_transform` dla rehearsal_loader.

            print(
                f"OSTRZEŻENIE: Implementacja Rehearsal zakłada, że etykiety z rehearsal_loader będą poprawnie zmapowane. "
                "To może wymagać niestandardowego Dataset/DataLoader dla rehearsal."
            )
            try:
                rehearsal_dataset = datasets.ImageFolder(
                    rehearsal_data_path, transform=train_transform
                )  # Użyj tych samych transformacji
                # Potrzebujemy zmapować etykiety z rehearsal_dataset (0..M-1 wg folderów rehearsal)
                # na globalne indeksy z `final_class_names_for_new_model`.

                class RehearsalDatasetWrapper(Dataset):
                    def __init__(
                        self,
                        img_folder_dataset,
                        final_model_classes_map_name_to_idx_str,
                    ):
                        self.img_folder_dataset = img_folder_dataset
                        # final_model_classes_map_name_to_idx_str to {nazwa_klasy_lower: globalny_idx_str}
                        self.target_map = {}
                        valid_samples = []
                        for i in range(len(self.img_folder_dataset.samples)):
                            _, folder_idx = self.img_folder_dataset.samples[i]
                            class_name_from_folder = self.img_folder_dataset.classes[
                                folder_idx
                            ]
                            global_idx_str = (
                                final_model_classes_map_name_to_idx_str.get(
                                    class_name_from_folder.lower()
                                )
                            )
                            if global_idx_str is not None:
                                valid_samples.append((i, int(global_idx_str)))

                        self.valid_samples = valid_samples  # lista (original_idx_in_img_folder, global_target_idx)
                        if not self.valid_samples:
                            print(
                                "OSTRZEŻENIE RehearsalWrapper: Brak pasujących klas z rehearsal_data_path w final_class_names_for_new_model."
                            )

                    def __getitem__(self, index):
                        original_idx, global_target = self.valid_samples[index]
                        data, _ = self.img_folder_dataset[original_idx]
                        return data, global_target

                    def __len__(self):
                        return len(self.valid_samples)

                final_map_name_lower_to_idx_str = {
                    name.lower(): idx_str
                    for idx_str, name in final_class_names_for_new_model.items()
                }

                rehearsal_wrapped_dataset = RehearsalDatasetWrapper(
                    rehearsal_dataset, final_map_name_lower_to_idx_str
                )

                if len(rehearsal_wrapped_dataset) > 0:
                    rehearsal_batch_size = rehearsal_config.get(
                        "batch_size", batch_size // 2 or 1
                    )
                    rehearsal_loader = DataLoader(
                        rehearsal_wrapped_dataset,
                        batch_size=rehearsal_batch_size,
                        shuffle=True,
                        num_workers=2,
                        pin_memory=torch.cuda.is_available(),
                    )
                    print(
                        f"Załadowano {len(rehearsal_wrapped_dataset)} próbek rehearsal z {len(rehearsal_dataset.classes)} folderów (po mapowaniu)."
                    )
                else:
                    print(
                        "Rehearsal: Brak danych po mapowaniu klas. Rehearsal nie będzie użyty."
                    )
            except Exception as e_rehearsal:
                print(
                    f"Błąd podczas ładowania danych rehearsal: {e_rehearsal}. Rehearsal nieaktywny."
                )
        else:
            if rehearsal_config and rehearsal_config.get("use", False):
                print(
                    "Rehearsal włączony, ale brak `rehearsal_data_path` lub ścieżka nie istnieje. Rehearsal nie będzie użyty."
                )

    fisher_diagonal_params = None
    original_ewc_model_params = None
    if (
        prevent_forgetting
        and ewc_config
        and ewc_config.get("use", False)
        and original_model_for_forgetting_techniques
    ):
        print("\nKonfigurowanie EWC...")
        # `original_model_for_forgetting_techniques` ma starą głowicę (`actual_base_model_output_neurons`)
        # i oczekuje etykiet z zakresu `original_class_names_from_config`.

        # Potrzebujemy danych z `train_dir`, które należą do klas z `original_class_names_from_config`.
        # Etykiety tych danych muszą być zmapowane na oryginalne indeksy numeryczne.

        # Mapa: nazwa_klasy_z_oryginalnego_configu_LOWER -> oryginalny_idx_NUMERYCZNY
        valid_original_class_map_for_ewc = {
            name.lower(): int(idx_str)
            for idx_str, name in original_class_names_from_config.items()
            if idx_str.isdigit()  # Upewnij się, że idx_str jest liczbą
        }

        if not valid_original_class_map_for_ewc:
            print(
                "OSTRZEŻENIE EWC: Brak prawidłowych klas w konfiguracji modelu bazowego. EWC nie może być obliczone."
            )
        else:
            # Użyjemy pełnego `train_dataset` (który jest już załadowany) i owiniemy go.
            ewc_filtered_dataset = EWCFilterDataset(
                train_dataset, valid_original_class_map_for_ewc
            )

            if len(ewc_filtered_dataset) == 0:
                print(
                    "OSTRZEŻENIE EWC: Nie znaleziono żadnych próbek z klas bazowych w zbiorze treningowym. EWC nie może być obliczone."
                )
            else:
                fisher_sample_size = ewc_config.get(
                    "fisher_sample_size", min(200, len(ewc_filtered_dataset))
                )
                if fisher_sample_size > len(ewc_filtered_dataset):
                    print(
                        f"  EWC: Dostępnych jest tylko {len(ewc_filtered_dataset)} próbek dla Fishera, używam wszystkich."
                    )
                    fisher_sample_size = len(ewc_filtered_dataset)

                ewc_dataloader = DataLoader(
                    ewc_filtered_dataset,
                    batch_size=batch_size,  # Może być inny batch size dla Fishera
                    shuffle=True,
                    num_workers=2,
                    pin_memory=torch.cuda.is_available(),
                )
                print(
                    f"Obliczanie macierzy Fishera dla {fisher_sample_size} próbek z {len(ewc_filtered_dataset)} dostępnych..."
                )

                try:
                    fisher_diagonal_params = compute_fisher_information(
                        original_model_for_forgetting_techniques,  # Model z oryginalną głowicą
                        ewc_dataloader,
                        device=device,
                        num_samples=fisher_sample_size,
                    )
                    print("✓ Macierz Fishera obliczona.")

                    original_ewc_model_params = {
                        name: param.data.clone()
                        for name, param in original_model_for_forgetting_techniques.named_parameters()
                        if param.requires_grad  # Tylko parametry, które mogły się uczyć
                    }
                    print("✓ Oryginalne parametry dla EWC zapisane.")
                except Exception as e_fisher:
                    print(
                        f"BŁĄD podczas obliczania macierzy Fishera: {e_fisher}. EWC nie będzie użyte."
                    )
                    fisher_diagonal_params = None
                    original_ewc_model_params = None

    # --- Zamrażanie warstw (w `model_to_train`) ---
    # `model_to_train` to model z potencjalnie nową głowicą.
    # Parametr `freeze_ratio` jest domyślny. `layer_freezing_config` może go nadpisać.
    effective_freeze_ratio = freeze_ratio
    if layer_freezing_config and "freeze_ratio" in layer_freezing_config:
        effective_freeze_ratio = layer_freezing_config["freeze_ratio"]

    layer_freezing_strategy = "gradual"  # Domyślnie
    if layer_freezing_config and "strategy" in layer_freezing_config:
        layer_freezing_strategy = layer_freezing_config["strategy"]

    print(
        f"\nKonfiguracja zamrażania warstw: Strategia '{layer_freezing_strategy}', Ratio: {effective_freeze_ratio*100:.1f}%"
    )

    all_model_parameters = list(model_to_train.named_parameters())
    num_total_param_groups = len(all_model_parameters)
    num_to_freeze = int(num_total_param_groups * effective_freeze_ratio)

    if layer_freezing_strategy == "gradual":  # Zamroź `num_to_freeze` pierwszych warstw
        for i, (name, param) in enumerate(all_model_parameters):
            param.requires_grad = i >= num_to_freeze
    elif layer_freezing_strategy == "selective":
        # Zamrażaj wszystkie warstwy oprócz tych w nowej głowicy (jeśli była zmieniona)
        # i ewentualnie kilku ostatnich bloków. To wymaga bardziej szczegółowej logiki.
        # Prostsza wersja "selective": zamroź wszystko oprócz parametrów nowej głowicy.
        print(
            "  Strategia 'selective': Domyślnie trenowana będzie tylko nowa głowica (jeśli zmieniona) i warstwy nie zamrożone przez 'gradual' z ratio."
        )
        # Ta logika jest niejasna. Użyjmy `gradual` jako podstawy, a `selective` może być interpretowane
        # jako zamrożenie np. wszystkich `conv` a trenowanie `fc`.
        # Na razie, jeśli `selective`, to trenujemy tylko ostatnie `1 - effective_freeze_ratio` warstw.
        # To jest to samo co `gradual`. Potrzebna by była bardziej zaawansowana logika dla `selective`.
        # Załóżmy, że jeśli 'selective', to użytkownik chce trenować TYLKO głowicę.
        if "new_head_layer" in locals() and new_head_layer is not None:
            print("  'selective' interpretowane jako trenowanie tylko nowej głowicy.")
            for param in model_to_train.parameters():  # Najpierw zamroź wszystko
                param.requires_grad = False
            for param in new_head_layer.parameters():  # Potem odmroź głowicę
                param.requires_grad = True
        else:  # Jeśli głowica nie była zmieniana, 'selective' zadziała jak 'gradual' z odwrotnym ratio
            print(
                "  'selective' bez zmiany głowicy: trenowanie ostatnich (1-ratio) warstw."
            )
            for i, (name, param) in enumerate(all_model_parameters):
                param.requires_grad = i >= num_to_freeze

    elif layer_freezing_strategy == "progressive":
        # Początkowe zamrożenie. Odmrażanie będzie w pętli epok.
        for i, (name, param) in enumerate(all_model_parameters):
            param.requires_grad = i >= num_to_freeze

    trainable_params_count = sum(
        p.numel() for p in model_to_train.parameters() if p.requires_grad
    )
    total_params_count = sum(p.numel() for p in model_to_train.parameters())
    print(
        f"Liczba parametrów trenowalnych: {trainable_params_count:,} z {total_params_count:,} ({trainable_params_count/total_params_count*100:.2f}%)"
    )

    if trainable_params_count == 0:
        raise RuntimeError(
            "Brak trenowalnych parametrów w modelu! Sprawdź logikę zamrażania."
        )

    # --- Optymalizator, Scheduler, Kryterium Straty ---
    # (Tworzone na `model_to_train.parameters()` które mają `requires_grad=True`)
    optimizer_params = [p for p in model_to_train.parameters() if p.requires_grad]
    if optimizer_type.lower() == "adamw":
        optimizer = optim.AdamW(
            optimizer_params, lr=learning_rate, weight_decay=weight_decay
        )
    # ... (reszta optymalizatorów, schedulerów, criterion - jak wcześniej, ale na `optimizer_params`)
    elif optimizer_type.lower() == "sgd":
        optimizer = optim.SGD(
            optimizer_params, lr=learning_rate, momentum=0.9, weight_decay=weight_decay
        )
    else:  # Domyślnie Adam
        optimizer = optim.Adam(
            optimizer_params, lr=learning_rate, weight_decay=weight_decay
        )
    print(f"Wybrano optymalizator: {optimizer_type.lower()}")

    scheduler = None
    if scheduler_type.lower() == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=max(1, early_stopping_patience // 2),
        )
    elif scheduler_type.lower() == "cosine":
        # T_max powinno być liczbą kroków, jeśli aktualizujemy co krok, lub epok, jeśli co epokę.
        # Dla CosineAnnealingLR, T_max to liczba epok (po warm-upie).
        t_max_cosine = num_epochs - warmup_epochs
        if t_max_cosine <= 0:
            print(
                f"OSTRZEŻENIE: num_epochs ({num_epochs}) <= warmup_epochs ({warmup_epochs}). Cosine scheduler może nie działać poprawnie."
            )
            t_max_cosine = 1  # Uniknięcie błędu
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max_cosine, eta_min=learning_rate / 100
        )
    elif scheduler_type.lower() == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate * 10,  # max_lr może być parametrem
            epochs=num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy="cos",
        )
    print(f"Wybrano scheduler: {scheduler_type.lower() if scheduler else 'brak'}")

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    print(f"Kryterium straty: CrossEntropyLoss (label_smoothing={label_smoothing})")

    # Mixed Precision Scaler
    scaler = torch.amp.GradScaler(
        enabled=(use_mixed_precision and device.type == "cuda")
    )

    # --- Przygotowanie DataLoaderów ---
    # `train_dataset` już załadowany.
    # Etykiety z `train_loader` będą od 0 do `num_classes_in_current_task - 1`.
    # Te etykiety muszą być zmapowane na globalne indeksy z `final_class_names_for_new_model`
    # PRZED podaniem do modelu, jeśli `preserve_original_classes=True` i globalne indeksy są inne.
    # To jest BARDZO WAŻNE. Model oczekuje globalnych indeksów.

    # Stwórz mapę: task_idx (0..N-1 z ImageFolder) -> global_idx (z final_class_names_for_new_model)
    task_idx_to_global_idx_map = {}
    name_lower_to_global_idx_str = {
        name.lower(): idx_str
        for idx_str, name in final_class_names_for_new_model.items()
    }

    for task_idx_0_N, task_class_name in current_task_idx_to_class_name.items():
        global_idx_str = name_lower_to_global_idx_str.get(task_class_name.lower())
        if global_idx_str is not None:
            task_idx_to_global_idx_map[task_idx_0_N] = int(global_idx_str)
        else:
            # To nie powinno się zdarzyć, jeśli `final_class_names_for_new_model` było poprawnie zbudowane.
            raise RuntimeError(
                f"Nie można zmapować klasy zadania '{task_class_name}' (ID {task_idx_0_N}) na globalny ID."
            )

    print("Mapowanie etykiet z DataLoader na globalne ID modelu:")
    for task_idx, global_idx in sorted(task_idx_to_global_idx_map.items()):
        print(
            f"  Etykieta z loadera {task_idx} -> Globalny ID {global_idx} ({final_class_names_for_new_model.get(str(global_idx))})"
        )

    # Konwersja mapy na tensor do szybkiego mapowania na GPU
    # Długość tensora musi być `num_classes_in_current_task`. Indeksy to task_idx. Wartości to global_idx.
    if not task_idx_to_global_idx_map:  # Jeśli train_dir był pusty
        # To powinno być obsłużone wcześniej, ale jako zabezpieczenie
        raise ValueError("Brak klas w zbiorze treningowym po mapowaniu.")

    max_task_idx = (
        max(task_idx_to_global_idx_map.keys()) if task_idx_to_global_idx_map else -1
    )
    mapping_tensor = torch.full(
        (max_task_idx + 1,), -1, dtype=torch.long, device=device
    )
    for task_idx, global_idx in task_idx_to_global_idx_map.items():
        mapping_tensor[task_idx] = global_idx

    # Sprawdzenie, czy wszystkie task_idx zostały zmapowane
    if (mapping_tensor == -1).any():
        unmapped_task_indices = (mapping_tensor == -1).nonzero(as_tuple=True)[0]
        print(
            f"OSTRZEŻENIE KRYTYCZNE: Nie wszystkie etykiety z DataLoader zostały zmapowane na globalne ID: {unmapped_task_indices.tolist()}"
        )
        # To może się zdarzyć, jeśli `current_task_idx_to_class_name` ma więcej klas niż `task_idx_to_global_idx_map`
        # co jest mało prawdopodobne, jeśli logika była spójna.
        # Lub jeśli `max_task_idx` jest większe niż największy klucz w mapie (dziury w task_idx).
        # ImageFolder daje ciągłe indeksy 0..N-1, więc to nie powinno być problemem.

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,  # drop_last może pomóc uniknąć problemów z batch norm przy małym ostatnim batchu
    )

    val_loader = None
    # Walidacja również potrzebuje mapowania etykiet, jeśli używa tych samych klas co trening.
    # `val_dataset.class_to_idx` może być inne niż `train_dataset.class_to_idx` jeśli foldery są inne.
    # Należy zapewnić spójność lub ostrożnie interpretować metryki.
    # Na razie zakładamy, że jeśli `val_dir` jest, to ma te same klasy co `train_dir`.
    if val_dir:
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
        # Sprawdź, czy klasy walidacyjne są podzbiorem klas treningowych (lub takie same)
        val_task_class_to_idx = val_dataset.class_to_idx
        val_task_idx_to_global_idx_map = {}
        for val_task_idx, val_class_name in enumerate(
            val_dataset.classes
        ):  # Użyj `val_dataset.classes` aby uzyskać listę nazw w kolejności indeksów
            global_idx_str = name_lower_to_global_idx_str.get(val_class_name.lower())
            if global_idx_str is not None:
                val_task_idx_to_global_idx_map[val_task_idx] = int(global_idx_str)
            # else: klasa z walidacji nie jest w `final_class_names_for_new_model` - będzie problem z metrykami
            # Można by ją pominąć lub dodać do `final_class_names_for_new_model` (jeśli nie `preserve_original_classes`)

        max_val_task_idx = (
            max(val_task_idx_to_global_idx_map.keys())
            if val_task_idx_to_global_idx_map
            else -1
        )
        val_mapping_tensor = torch.full(
            (max_val_task_idx + 1,), -1, dtype=torch.long, device=device
        )
        if val_task_idx_to_global_idx_map:  # tylko jeśli są jakieś klasy w walidacji
            for val_task_idx, global_idx in val_task_idx_to_global_idx_map.items():
                if val_task_idx < len(val_mapping_tensor):  # Zabezpieczenie
                    val_mapping_tensor[val_task_idx] = global_idx
            if (val_mapping_tensor == -1).any():
                unmapped_val_indices = (val_mapping_tensor == -1).nonzero(
                    as_tuple=True
                )[0]
                print(
                    f"OSTRZEŻENIE: Nie wszystkie etykiety z Val DataLoader zostały zmapowane na globalne ID: {unmapped_val_indices.tolist()}"
                )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )
        print(f"Załadowano {len(val_dataset)} obrazów walidacyjnych.")

    # --- Pętla treningowa ---
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "val_auc": [],
        "val_top3": [],
        "val_top5": [],
        "learning_rates": [],
        "best_val_loss": float("inf"),
        "best_epoch": 0,
        "epoch_times": [],
        "current_lr": learning_rate,  # Będzie aktualizowane
    }
    best_model_state_dict = deepcopy(
        model_to_train.state_dict()
    )  # Na wypadek przerwania lub braku poprawy
    patience_count = 0

    print(f"\n=== ROZPOCZYNAM TRENING ({num_epochs} epok) ===")
    for epoch in range(num_epochs):
        if should_stop_callback and should_stop_callback():
            print(f"Przerwano trening na epoce {epoch+1} przez użytkownika.")
            break

        epoch_start_t = time.time()
        current_lr_for_epoch = optimizer.param_groups[0]["lr"]  # Pobierz aktualny LR
        history["learning_rates"].append(current_lr_for_epoch)
        history["current_lr"] = current_lr_for_epoch
        print(
            f"\n--- Epoka {epoch+1}/{num_epochs} (LR: {current_lr_for_epoch:.2e}) ---"
        )

        # Progressive unfreezing (jeśli skonfigurowane)
        if layer_freezing_strategy == "progressive" and epoch >= warmup_epochs:
            # Prosta logika: co kilka epok odmrażaj kolejny blok warstw
            # Tutaj bardziej zaawansowana: stopniowo odmrażaj do `1 - effective_freeze_ratio`
            progress_in_unfreezing = (epoch - warmup_epochs) / max(
                1, num_epochs - warmup_epochs - 1
            )  # 0 do 1

            # Ile warstw powinno być docelowo trenowalnych (nie zamrożonych)
            target_trainable_param_groups = num_total_param_groups - int(
                num_total_param_groups
                * effective_freeze_ratio
                * (1 - progress_in_unfreezing)
            )
            target_trainable_param_groups = min(
                num_total_param_groups, max(0, target_trainable_param_groups)
            )

            newly_unfrozen_in_epoch = False
            for i, (name, param) in enumerate(all_model_parameters):
                # Odmrażamy od początku listy (najwcześniejsze warstwy)
                # Chcemy, aby `target_trainable_param_groups` było trenowalne, zaczynając od końca.
                # Więc pierwsze `num_total_param_groups - target_trainable_param_groups` są zamrożone.
                should_be_frozen = i < (
                    num_total_param_groups - target_trainable_param_groups
                )
                if param.requires_grad == should_be_frozen:  # Jeśli trzeba zmienić stan
                    param.requires_grad = not should_be_frozen
                    if not should_be_frozen:  # Czyli została odmrożona
                        # print(f"  🔥 Odmrożono (progressive): {name}")
                        newly_unfrozen_in_epoch = True

            if newly_unfrozen_in_epoch:
                print(
                    f"  Odmrożono warstwy (progressive). Docelowo trenowalnych grup: {target_trainable_param_groups}"
                )
                optimizer.param_groups[0]["params"] = [
                    p for p in model_to_train.parameters() if p.requires_grad
                ]

        # --- Trening epoki ---
        model_to_train.train()
        running_train_loss = 0.0
        running_train_corrects = 0
        running_train_samples = 0

        # Iterator dla rehearsal, resetowany co epokę
        rehearsal_iter = iter(rehearsal_loader) if rehearsal_loader else None

        for batch_idx, (inputs, task_targets) in enumerate(train_loader):
            inputs, task_targets = inputs.to(device), task_targets.to(device)

            # Mapowanie etykiet z DataLoader na globalne ID modelu
            global_targets = mapping_tensor[task_targets]

            # Opcjonalny Green Diffusion
            if use_green_diffusion:
                inputs = green_diffusion(
                    inputs, noise_level=0.05, apply_prob=0.3
                )  # Przykładowe wartości

            # Opcjonalny Mixup na danych wejściowych
            # Jeśli mixup, to `global_targets` stają się `y_a`, a `y_b` trzeba też zmapować.
            y_a_mixup, y_b_mixup, lam_mixup = None, None, 1.0
            if use_mixup and epoch >= warmup_epochs:  # Mixup po warmupie
                # `inputs` i `global_targets` są już na `device`
                mixed_inputs, y_a_mixup, y_b_mixup, lam_mixup = mixup_data(
                    inputs, global_targets, alpha=0.2, device=device
                )
                inputs = mixed_inputs  # Nadpisz inputs
                # `global_targets` nie jest już pojedynczą etykietą, ale parą (y_a, y_b) z lambda.

            # Rehearsal - dodawanie próbek
            if rehearsal_iter:
                try:
                    re_inputs, re_global_targets = next(rehearsal_iter)
                    re_inputs, re_global_targets = re_inputs.to(
                        device
                    ), re_global_targets.to(device)

                    if inputs.shape[1:] == re_inputs.shape[1:]:
                        inputs = torch.cat([inputs, re_inputs], dim=0)
                        if (
                            use_mixup and y_a_mixup is not None
                        ):  # Jeśli był mixup, trzeba ostrożnie łączyć etykiety
                            # To komplikuje sprawę. Prościej: nie używaj mixup i rehearsal jednocześnie
                            # lub dodaj próbki rehearsal PRZED mixupem.
                            # Na razie: jeśli mixup, nie dodajemy etykiet rehearsal w ten sposób.
                            print(
                                "OSTRZEŻENIE: Mixup i Rehearsal jednocześnie - obsługa etykiet uproszczona."
                            )
                            # Potrzebowalibyśmy y_a_re, y_b_re, lam_re.
                            # Bezpieczniej: jeśli rehearsal, to global_targets jest konkatenowane.
                            # Jeśli potem mixup, to na połączonym batchu.
                            # Jeśli mixup był przed rehearsal, to `inputs` są zmieszane, a `re_inputs` nie.
                            # Zmieńmy kolejność: Rehearsal -> Mixup
                            pass  # Załóżmy, że etykiety `global_targets` zostaną obsłużone poprawnie
                            # przez mixup_criterion, jeśli `lam_mixup` jest 1.0 dla części rehearsal.
                            # LUB: jeśli mixup, to y_a_mixup i y_b_mixup muszą być rozszerzone.
                            # Najprościej: jeśli mixup, `global_targets` nie jest już używane bezpośrednio.
                        else:  # Rehearsal był przed mixupem
                            global_targets = torch.cat(
                                [global_targets, re_global_targets], dim=0
                            )

                    else:  # Kształty rehearsal i batcha się nie zgadzają
                        print(
                            f"OSTRZEŻENIE Rehearsal: Niezgodne kształty! Batch: {inputs.shape}, Rehearsal: {re_inputs.shape}. Pomijam."
                        )
                except StopIteration:
                    rehearsal_iter = iter(rehearsal_loader)  # Zresetuj, jeśli krótszy

            optimizer.zero_grad()
            autocast_enabled = (
                use_mixed_precision and device.type == "cuda"
            )  # Uproszczone, CPU bfloat16 rzadziej używane

            with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                outputs = model_to_train(
                    inputs
                )  # `inputs` mogą być zmieszane lub połączone z rehearsal

                # Obliczanie straty
                if use_mixup and y_a_mixup is not None and epoch >= warmup_epochs:
                    # `outputs` są z `mixed_inputs`. `y_a_mixup`, `y_b_mixup` to globalne ID.
                    # `criterion` to nn.CrossEntropyLoss
                    current_batch_loss = mixup_criterion(
                        criterion, outputs, y_a_mixup, y_b_mixup, lam_mixup
                    )
                else:
                    # `outputs` są z oryginalnych (lub rehearsal) `inputs`. `global_targets` to globalne ID.
                    current_batch_loss = criterion(outputs, global_targets)

                # Knowledge Distillation
                if (
                    prevent_forgetting
                    and knowledge_distillation_config
                    and knowledge_distillation_config.get("use", False)
                    and original_model_for_forgetting_techniques is not None
                    and epoch >= warmup_epochs
                ):  # KD po warmupie

                    # Nauczyciel przetwarza te same `inputs` co student (mogą być zmieszane/rehearsal)
                    with torch.no_grad():
                        teacher_outputs = original_model_for_forgetting_techniques(
                            inputs
                        )

                    # Dopasowanie wyjść studenta do nauczyciela, jeśli trzeba
                    student_outputs_for_kd = outputs
                    if (
                        outputs.shape[1] > teacher_outputs.shape[1]
                    ):  # Student ma więcej klas
                        # Załóżmy, że pierwsze N klas studenta odpowiada nauczycielowi
                        # To jest OK, jeśli nowe klasy dodano na końcu globalnych ID
                        # i `final_class_names_for_new_model` jest posortowane wg ID.
                        # Ryzykowne założenie. Lepsze mapowanie jest potrzebne.
                        # Na razie proste obcięcie:
                        student_outputs_for_kd = outputs[:, : teacher_outputs.shape[1]]
                    elif outputs.shape[1] < teacher_outputs.shape[1]:
                        print(
                            "OSTRZEŻENIE KD: Student ma mniej klas niż nauczyciel. KD może nie działać poprawnie."
                        )
                        # W tym przypadku nie da się dopasować. KD nie powinno być stosowane.
                        # Ale poniższa funkcja straty sobie z tym poradzi, zwracając tylko hard_loss.

                    # Etykiety dla hard loss w KD:
                    # Jeśli był mixup, to `y_a_mixup`, `y_b_mixup`, `lam_mixup`.
                    # Jeśli nie, to `global_targets`.
                    if use_mixup and y_a_mixup is not None and epoch >= warmup_epochs:
                        kd_labels_arg = (y_a_mixup, y_b_mixup, lam_mixup)
                    else:
                        kd_labels_arg = global_targets

                    current_batch_loss = distillation_loss(
                        student_outputs=outputs,  # Pełne wyjścia studenta
                        labels_or_mixup_tuple=kd_labels_arg,  # Etykiety lub krotka mixup
                        teacher_outputs=teacher_outputs,
                        student_outputs_for_soft_loss=student_outputs_for_kd,
                        temperature=knowledge_distillation_config.get(
                            "temperature", 2.0
                        ),
                        alpha=knowledge_distillation_config.get("alpha", 0.5),
                        base_criterion=criterion,  # Przekaż bazowe kryterium dla hard loss
                    )

                # EWC
                if (
                    prevent_forgetting
                    and ewc_config
                    and ewc_config.get("use", False)
                    and fisher_diagonal_params
                    and original_ewc_model_params
                    and epoch >= warmup_epochs
                ):  # EWC po warmupie

                    ewc_lambda_val = ewc_config.get("lambda", 1000.0)
                    if ewc_config.get("adaptive_lambda", True):
                        progress = epoch / max(1, num_epochs - 1)  # 0 do 1
                        ewc_lambda_val *= progress

                    ewc_penalty = 0.0
                    for name, param in model_to_train.named_parameters():
                        if (
                            name in fisher_diagonal_params
                            and name in original_ewc_model_params
                        ):
                            if (
                                param.shape == original_ewc_model_params[name].shape
                            ):  # Upewnij się, że kształty pasują
                                diff_sq = (
                                    param
                                    - original_ewc_model_params[name].to(param.device)
                                ) ** 2
                                ewc_penalty += torch.sum(
                                    fisher_diagonal_params[name].to(param.device)
                                    * diff_sq
                                )
                            # else: Ostrzeżenie o niezgodności kształtów (powinno być rzadkie dla EWC na backbone)
                    current_batch_loss += ewc_lambda_val * ewc_penalty

            # Backward pass
            if torch.isnan(current_batch_loss):
                print(
                    f"OSTRZEŻENIE: NaN w stracie! Epoka {epoch+1}, Batch {batch_idx+1}. Pomijam ten batch."
                )
                continue

            scaler.scale(current_batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += current_batch_loss.item() * inputs.size(
                0
            )  # Ważone przez rozmiar batcha (inputs.size(0) może być większe przez rehearsal)
            running_train_samples += inputs.size(0)

            # Dokładność - trudna przy mixupie. Jeśli mixup, licz tylko na części bez mixupu lub uprość.
            # Na razie liczymy na podstawie `outputs.max(1)` i `global_targets` (jeśli nie mixup)
            # lub `y_a_mixup` (jeśli mixup, to tylko vs główna klasa).
            if not (
                use_mixup and y_a_mixup is not None and epoch >= warmup_epochs
            ):  # Jeśli nie było mixupu danych
                _, preds = torch.max(outputs, 1)
                running_train_corrects += torch.sum(preds == global_targets.data)
            # else: dla mixup, dokładność jest bardziej złożona do policzenia i często pomijana w pętli.

            # Aktualizacja LR dla schedulerów per-krok (np. OneCycleLR)
            # OneCycleLR jest aktualizowany PO kroku optymalizatora
            if (
                scheduler and scheduler_type.lower() == "onecycle"
            ):  # OneCycleLR działa od początku
                scheduler.step()

        # Koniec pętli po batchach - obliczanie średnich dla epoki
        epoch_train_loss = (
            running_train_loss / running_train_samples
            if running_train_samples > 0
            else 0
        )
        epoch_train_acc = (
            (running_train_corrects.double() / running_train_samples * 100).item()
            if running_train_samples > 0 and not (use_mixup and epoch >= warmup_epochs)
            else 0.0
        )  # Uproszczone dla mixup
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)

        print(
            f"  Koniec treningu epoki: Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%"
            if epoch_train_acc > 0
            else f"  Koniec treningu epoki: Train Loss: {epoch_train_loss:.4f} (Acc nie liczona dla mixup)"
        )

        # --- Walidacja epoki ---
        epoch_val_loss, epoch_val_acc, epoch_val_metrics = 0.0, 0.0, {}
        if val_loader:
            model_to_train.eval()
            running_val_loss = 0.0
            running_val_corrects = 0
            running_val_samples = 0
            all_val_global_targets_np = []
            all_val_preds_np = []
            all_val_probs_np = []

            with torch.no_grad():
                for inputs_val, task_targets_val in val_loader:
                    inputs_val, task_targets_val = inputs_val.to(
                        device
                    ), task_targets_val.to(device)

                    # Mapowanie etykiet walidacyjnych na globalne ID
                    # Użyj `val_mapping_tensor` jeśli zdefiniowany i ma pasujące wymiary
                    global_targets_val = task_targets_val  # Domyślnie
                    if (
                        "val_mapping_tensor" in locals()
                        and val_mapping_tensor is not None
                        and task_targets_val.max() < len(val_mapping_tensor)
                    ):
                        global_targets_val = val_mapping_tensor[task_targets_val]
                        # Odfiltruj, jeśli mapowanie dało -1 (klasa z walidacji nieznana globalnie)
                        valid_mask_val = global_targets_val != -1
                        if not valid_mask_val.all():  # Jeśli są jakieś nieznane
                            inputs_val = inputs_val[valid_mask_val]
                            global_targets_val = global_targets_val[valid_mask_val]
                            if inputs_val.size(0) == 0:
                                continue  # Pusty batch po filtracji

                    with torch.autocast(
                        device_type=device.type, enabled=autocast_enabled
                    ):
                        outputs_val = model_to_train(inputs_val)
                        loss_val = criterion(
                            outputs_val, global_targets_val
                        )  # Użyj globalnych etykiet

                    running_val_loss += loss_val.item() * inputs_val.size(0)
                    _, preds_val = torch.max(outputs_val, 1)
                    running_val_corrects += torch.sum(
                        preds_val == global_targets_val.data
                    )
                    running_val_samples += inputs_val.size(0)

                    all_val_global_targets_np.extend(global_targets_val.cpu().numpy())
                    all_val_preds_np.extend(preds_val.cpu().numpy())
                    all_val_probs_np.extend(
                        torch.softmax(outputs_val, dim=1).cpu().numpy()
                    )

            epoch_val_loss = (
                running_val_loss / running_val_samples if running_val_samples > 0 else 0
            )
            epoch_val_acc = (
                (running_val_corrects.double() / running_val_samples * 100).item()
                if running_val_samples > 0
                else 0
            )
            history["val_loss"].append(epoch_val_loss)
            history["val_acc"].append(epoch_val_acc)

            # Obliczanie dodatkowych metryk walidacyjnych
            y_true_val_np = np.array(all_val_global_targets_np)
            y_pred_val_np = np.array(all_val_preds_np)
            y_prob_val_np = np.array(all_val_probs_np)

            # Etykiety dla sklearn powinny być unikalnymi wartościami z y_true_val_np
            # lub pełnym zakresem `num_total_classes_for_new_head`
            sklearn_labels = (
                np.unique(y_true_val_np)
                if len(np.unique(y_true_val_np)) > 1
                else list(range(num_total_classes_for_new_head))
            )

            if running_val_samples > 0 and len(sklearn_labels) > 0:
                epoch_val_metrics["f1"] = f1_score(
                    y_true_val_np,
                    y_pred_val_np,
                    average="macro",
                    zero_division=0,
                    labels=sklearn_labels,
                )
                epoch_val_metrics["precision"] = precision_score(
                    y_true_val_np,
                    y_pred_val_np,
                    average="macro",
                    zero_division=0,
                    labels=sklearn_labels,
                )
                epoch_val_metrics["recall"] = recall_score(
                    y_true_val_np,
                    y_pred_val_np,
                    average="macro",
                    zero_division=0,
                    labels=sklearn_labels,
                )

                if (
                    len(sklearn_labels) > 1
                    and y_prob_val_np.shape[0] > 0
                    and y_prob_val_np.shape[1] == num_total_classes_for_new_head
                ):
                    try:
                        epoch_val_metrics["auc"] = roc_auc_score(
                            y_true_val_np,
                            y_prob_val_np,
                            multi_class="ovr",
                            average="macro",
                            labels=sklearn_labels,
                        )
                    except ValueError as e_auc:  # Np. tylko jedna klasa w y_true
                        # print(f"  Nie można obliczyć AUC: {e_auc}")
                        epoch_val_metrics["auc"] = 0.0
                else:
                    epoch_val_metrics["auc"] = 0.0

                # Top-k (k musi być <= liczby klas w y_prob_val_np)
                # `labels` dla top_k_accuracy_score to lista wszystkich możliwych klas (indeksów)
                top_k_labels = list(range(num_total_classes_for_new_head))
                if y_prob_val_np.shape[0] > 0:
                    k3 = min(3, num_total_classes_for_new_head)
                    if k3 > 0:
                        epoch_val_metrics["top3"] = top_k_accuracy_score(
                            y_true_val_np, y_prob_val_np, k=k3, labels=top_k_labels
                        )
                    else:
                        epoch_val_metrics["top3"] = 0.0

                    k5 = min(5, num_total_classes_for_new_head)
                    if k5 > 0:
                        epoch_val_metrics["top5"] = top_k_accuracy_score(
                            y_true_val_np, y_prob_val_np, k=k5, labels=top_k_labels
                        )
                    else:
                        epoch_val_metrics["top5"] = 0.0
            else:  # Brak próbek walidacyjnych lub etykiet
                epoch_val_metrics = {
                    k: 0.0 for k in ["f1", "precision", "recall", "auc", "top3", "top5"]
                }

            history["val_f1"].append(epoch_val_metrics.get("f1", 0))
            history["val_precision"].append(epoch_val_metrics.get("precision", 0))
            history["val_recall"].append(epoch_val_metrics.get("recall", 0))
            history["val_auc"].append(epoch_val_metrics.get("auc", 0))
            history["val_top3"].append(epoch_val_metrics.get("top3", 0))
            history["val_top5"].append(epoch_val_metrics.get("top5", 0))

            print(
                f"  Walidacja: Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%"
            )
            print(
                f"             Val F1: {epoch_val_metrics.get('f1',0):.4f}, AUC: {epoch_val_metrics.get('auc',0):.4f}, Top3: {epoch_val_metrics.get('top3',0):.2%}"
            )

        # Aktualizacja schedulera (poza OneCycleLR, który jest per-krok)
        if scheduler and scheduler_type.lower() != "onecycle":
            if epoch >= warmup_epochs:  # Aktywuj scheduler po warmupie
                if scheduler_type.lower() == "plateau":
                    scheduler.step(epoch_val_loss if val_loader else epoch_train_loss)
                else:
                    scheduler.step()

        # Early stopping
        loss_for_es = epoch_val_loss if val_loader else epoch_train_loss
        if loss_for_es < history["best_val_loss"]:
            history["best_val_loss"] = loss_for_es
            history["best_epoch"] = epoch
            best_model_state_dict = deepcopy(model_to_train.state_dict())
            patience_count = 0
            print(f"  ✨ Nowy najlepszy model (loss: {loss_for_es:.4f})")
        else:
            patience_count += 1
            if patience_count >= early_stopping_patience:
                print(
                    f"⚠️ Early stopping po {early_stopping_patience} epokach bez poprawy. Najlepsza epoka: {history['best_epoch']+1}"
                )
                break

        history["epoch_times"].append(time.time() - epoch_start_t)
        if progress_callback:
            # Upewnij się, że epoch_val_metrics ma wartości domyślne, jeśli nie było walidacji
            cb_val_metrics = (
                epoch_val_metrics
                if val_loader
                else {
                    k: 0.0 for k in ["f1", "precision", "recall", "auc", "top3", "top5"]
                }
            )
            progress_callback(
                epoch + 1,
                num_epochs,
                epoch_train_loss,
                epoch_train_acc,
                epoch_val_loss,
                epoch_val_acc,  # Te są 0.0 jeśli brak val_loader
                cb_val_metrics.get("top3", 0),
                cb_val_metrics.get("top5", 0),
                cb_val_metrics.get("precision", 0),
                cb_val_metrics.get("recall", 0),
                cb_val_metrics.get("f1", 0),
                cb_val_metrics.get("auc", 0),
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # Koniec pętli po epokach

    # Przywróć najlepszy model
    model_to_train.load_state_dict(best_model_state_dict)
    print(
        f"\nPrzywrócono model z epoki {history['best_epoch']+1} (najlepsza strata: {history['best_val_loss']:.4f})"
    )

    # --- Zapisywanie modelu i konfiguracji ---
    print("\n=== ZAPISYWANIE KOŃCOWEGO MODELU ===")
    os.makedirs(output_dir, exist_ok=True)

    # Nazwa pliku modelu
    model_name_suffix = f"_{task_name}" if task_name else ""
    final_model_filename = f"{base_classifier.model_type}{model_name_suffix}_ft_e{history['best_epoch']+1}.pt"
    final_model_path = os.path.join(output_dir, final_model_filename)

    # Przygotuj ostateczny obiekt Classifier do zapisu
    # Powinien on mieć model z załadowanym `best_model_state_dict`
    # oraz `class_names` odpowiadające `final_class_names_for_new_model`
    # i `num_classes` równe `num_total_classes_for_new_head`.

    final_classifier_to_save = ImageClassifier(
        model_type=base_classifier.model_type,
        num_classes=num_total_classes_for_new_head,
    )
    final_classifier_to_save.model = (
        model_to_train  # Już ma załadowany best_model_state_dict
    )
    final_classifier_to_save.class_names = (
        final_class_names_for_new_model  # {global_idx_str: global_name}
    )

    # Metadane do zapisu
    # Zacznij od metadanych z oryginalnego pliku konfiguracyjnego modelu bazowego
    final_metadata = deepcopy(base_config_from_file.get("metadata", {}))

    # Dodaj/zaktualizuj historię fine-tuningu
    if "finetuning_history" not in final_metadata:
        final_metadata["finetuning_history"] = {}

    ft_session_count = (
        len(
            [
                k
                for k in final_metadata["finetuning_history"].keys()
                if k.startswith("session_")
            ]
        )
        + 1
    )
    session_key = f"session_{ft_session_count}"

    # Przygotuj słownik class_FT na podstawie current_task_idx_to_class_name
    # current_task_idx_to_class_name to {id_klasy_w_zadaniu (int): nazwa_klasy (str)}
    # Chcemy {"id_klasy_w_zadaniu_jako_string": nazwa_klasy (str)}
    class_ft_dict = {
        str(idx): name for idx, name in current_task_idx_to_class_name.items()
    }

    session_data_for_history = {
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M"),
        "base_model_for_this_ft_session": os.path.basename(base_model_path),
        "task_name": task_name,
        "num_epochs_run": epoch + 1,  # Ile faktycznie przebiegło
        "best_epoch_in_this_ft": history["best_epoch"] + 1,
        "best_val_loss_in_this_ft": history["best_val_loss"],
        "training_params": {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "optimizer": optimizer_type,
            "scheduler": scheduler_type,
            "label_smoothing": label_smoothing,
            "weight_decay": weight_decay,
            "warmup_epochs": warmup_epochs,
            "early_stopping_patience": early_stopping_patience,
            "freeze_ratio_config": freeze_ratio,
            "layer_freezing_actual_strategy": layer_freezing_strategy,
            "effective_freeze_ratio": effective_freeze_ratio,
        },
        "forgetting_prevention": {
            "ewc": ewc_config if ewc_config and ewc_config.get("use") else None,
            "kd": (
                knowledge_distillation_config
                if knowledge_distillation_config
                and knowledge_distillation_config.get("use")
                else None
            ),
            "rehearsal": (
                rehearsal_config
                if rehearsal_config and rehearsal_config.get("use")
                else None
            ),
        },
        # Nowa zagnieżdżona struktura
        "session": {
            "session_nr": f"{ft_session_count:02d}",  # Numer sesji, np. "01"
            "class_FT": class_ft_dict,
        },
        # Usunięto "trained_classes_in_this_ft_task" z tej lokalizacji
        # (jeśli istniało, nie było go w kodzie bezpośrednio tutaj)
    }
    final_metadata["finetuning_history"][session_key] = session_data_for_history

    # Usuwamy stary klucz, jeśli istniał w session_data_for_history
    # (mało prawdopodobne, ale dla pewności)
    if (
        "trained_classes_in_this_ft_task"
        in final_metadata["finetuning_history"][session_key]
    ):
        del final_metadata["finetuning_history"][session_key][
            "trained_classes_in_this_ft_task"
        ]

    # Zaktualizuj czas treningu
    training_time_this_session = time.time() - start_training_time
    final_metadata["last_ft_duration_sec"] = training_time_this_session

    # Odczytaj poprzedni skumulowany czas treningu, preferując nowy klucz "training_time"
    previous_total_training_time = 0
    if "training_time" in final_metadata:
        try:
            previous_total_training_time = float(final_metadata["training_time"])
        except (ValueError, TypeError):
            # Jeśli "training_time" istnieje, ale nie jest liczbą, spróbuj starszego klucza
            if "total_ft_duration_sec" in final_metadata:
                try:
                    previous_total_training_time = float(
                        final_metadata["total_ft_duration_sec"]
                    )
                except (ValueError, TypeError):
                    previous_total_training_time = (
                        0  # Resetuj, jeśli oba są niepoprawne
                    )
            else:
                previous_total_training_time = (
                    0  # Resetuj, jeśli tylko "training_time" jest niepoprawne
                )
    elif (
        "total_ft_duration_sec" in final_metadata
    ):  # Jeśli "training_time" nie ma, użyj starszego
        try:
            previous_total_training_time = float(
                final_metadata["total_ft_duration_sec"]
            )
        except (ValueError, TypeError):
            previous_total_training_time = (
                0  # Resetuj, jeśli "total_ft_duration_sec" jest niepoprawne
            )

    # Oblicz nowy całkowity czas treningu
    new_total_training_time = previous_total_training_time + training_time_this_session

    # Zapisz nowy całkowity czas treningu pod kluczem "training_time"
    final_metadata["training_time"] = new_total_training_time

    # Usuń stary klucz "total_ft_duration_sec", jeśli istniał, aby uniknąć redundancji
    if "total_ft_duration_sec" in final_metadata:
        del final_metadata["total_ft_duration_sec"]

    # Zapisz model używając metody z ImageClassifier, która powinna też zapisać config
    try:
        final_classifier_to_save.save(final_model_path, metadata_to_save=final_metadata)
        print(f"Zapisano końcowy model: {final_model_path}")
        print(
            f"  Konfiguracja zapisana w: {os.path.splitext(final_model_path)[0] + '_config.json'}"
        )
    except Exception as e_save:
        print(f"BŁĄD podczas zapisu modelu z ImageClassifier.save(): {e_save}")
        print("Próba ręcznego zapisu state_dict i konfiguracji.")
        torch.save(model_to_train.state_dict(), final_model_path)
        manual_config_path = os.path.splitext(final_model_path)[0] + "_config.json"
        manual_config_data = {
            "model_type": final_classifier_to_save.model_type,
            "num_classes": final_classifier_to_save.num_classes,
            "class_names": final_classifier_to_save.class_names,
            "metadata": final_metadata,
        }
        with open(manual_config_path, "w") as f_cfg:
            json.dump(manual_config_data, f_cfg, indent=4)
        print(
            f"Ręcznie zapisano model: {final_model_path} i config: {manual_config_path}"
        )

    print("\n=== PODSUMOWANIE FINE-TUNINGU ===")
    # ... (podsumowanie jak wcześniej)

    return {
        "model_path": final_model_path,
        "history": history,
        "class_names_in_model": final_class_names_for_new_model,
        "model_type": base_classifier.model_type,
        "num_classes_in_model": num_total_classes_for_new_head,
        "base_model_path": base_model_path,
    }


# --- distillation_loss z modyfikacją ---
def distillation_loss(
    student_outputs,
    labels_or_mixup_tuple,
    teacher_outputs,
    student_outputs_for_soft_loss,
    temperature,
    alpha,
    base_criterion,
):
    """
    Oblicza stratę dla destylacji wiedzy.
    Args:
        student_outputs: Pełne logity z modelu ucznia
        labels_or_mixup_tuple: Prawdziwe etykiety (tensor) lub krotka (y_a, y_b, lam) dla mixup
        teacher_outputs: Logity z modelu nauczyciela
        student_outputs_for_soft_loss: Logity studenta dopasowane do nauczyciela (dla soft loss)
        temperature: Temperatura do zmiękczania prawdopodobieństw
        alpha: Współczynnik wagi: alpha * hard_loss + (1-alpha) * soft_loss
        base_criterion: np. nn.CrossEntropyLoss, używane do obliczenia hard_loss (obsługuje mixup)
    """
    # Hard loss na etykietach studenta
    if isinstance(labels_or_mixup_tuple, tuple):  # Mixup
        y_a, y_b, lam = labels_or_mixup_tuple
        hard_loss = mixup_criterion(base_criterion, student_outputs, y_a, y_b, lam)
    else:  # Standardowe etykiety
        hard_loss = base_criterion(student_outputs, labels_or_mixup_tuple)

    # Soft loss (KL Divergence)
    # Sprawdź, czy wymiary dla soft loss są zgodne
    if student_outputs_for_soft_loss.shape[-1] != teacher_outputs.shape[-1]:
        print(
            f"OSTRZEŻENIE KD: Niezgodne wymiary dla soft loss. Student: {student_outputs_for_soft_loss.shape}, Nauczyciel: {teacher_outputs.shape}. Zwracam tylko hard_loss."
        )
        return hard_loss

    student_log_softmax = F.log_softmax(
        student_outputs_for_soft_loss / temperature, dim=1
    )
    # Nauczyciel nie powinien propagować gradientów, więc detach() jest kluczowe
    teacher_softmax = F.softmax(teacher_outputs / temperature, dim=1).detach()

    soft_loss = F.kl_div(
        student_log_softmax, teacher_softmax, reduction="batchmean"
    ) * (temperature * temperature)

    if torch.isnan(
        soft_loss
    ).any():  # .any() jeśli soft_loss to tensor wieloelementowy (rzadko przy batchmean)
        print(
            "OSTRZEŻENIE: Wykryto NaN w soft_loss (destylacja). Używam tylko hard_loss."
        )
        if torch.isnan(hard_loss).any():
            return torch.tensor(
                0.0, device=student_outputs.device, requires_grad=True
            )  # Ostateczność
        return hard_loss
    if torch.isnan(hard_loss).any():
        print(
            "OSTRZEŻENIE: Wykryto NaN w hard_loss (destylacja). Używam tylko soft_loss (jeśli nie NaN)."
        )
        if torch.isnan(soft_loss).any():
            return torch.tensor(0.0, device=student_outputs.device, requires_grad=True)
        return (1.0 - alpha) * soft_loss

    return alpha * hard_loss + (1.0 - alpha) * soft_loss


# --- mixup_data i mixup_criterion (bez zmian, ale upewnij się, że są poprawne) ---
def mixup_data(x, y, alpha=0.2, device="cpu"):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0  # Zmienione z int na float dla spójności

    batch_size = x.size(0)  # Poprawione z x.size()[0]
    index = torch.randperm(batch_size, device=device)  # Użyj device

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# --- generate_synthetic_samples (jak w poprzedniej sugestii - placeholder) ---
def generate_synthetic_samples(
    model, classes_indices, samples_per_class, device, image_shape=(3, 224, 224)
):
    print(
        "OSTRZEŻENIE: `generate_synthetic_samples` tworzy LOSOWY SZUM, a nie realistyczne obrazy. "
        "Rehearsal z tymi danymi nie będzie efektywny i prawdopodobnie pogorszy wyniki."
    )

    if not classes_indices:  # Jeśli lista klas jest pusta
        return DataLoader(
            torch.utils.data.TensorDataset(
                torch.empty(0, *image_shape), torch.empty(0, dtype=torch.long)
            ),
            batch_size=1,
        )

    all_synthetic_images = []
    all_synthetic_labels = []

    for class_idx in classes_indices:  # Oczekujemy listy indeksów liczbowych
        for _ in range(samples_per_class):
            noise_image = torch.rand(
                1, *image_shape, device=device
            )  # Szum w zakresie [0,1]
            all_synthetic_images.append(noise_image)
            all_synthetic_labels.append(
                torch.tensor([class_idx], dtype=torch.long, device=device)
            )  # Etykieta jako tensor

    if not all_synthetic_images:
        return DataLoader(
            torch.utils.data.TensorDataset(
                torch.empty(0, *image_shape), torch.empty(0, dtype=torch.long)
            ),
            batch_size=1,
        )

    synthetic_images_tensor = torch.cat(all_synthetic_images, dim=0)
    synthetic_labels_tensor = torch.cat(all_synthetic_labels, dim=0)

    synthetic_dataset = torch.utils.data.TensorDataset(
        synthetic_images_tensor, synthetic_labels_tensor
    )
    # Batch size dla synthetic_loader można by ustawić inaczej
    loader_batch_size = min(
        samples_per_class * len(classes_indices) if classes_indices else 1, 32
    )  # Nie większy niż 32
    if loader_batch_size == 0 and len(synthetic_dataset) > 0:
        loader_batch_size = 1  # Unikaj batch_size=0

    if len(synthetic_dataset) == 0:  # Jeśli dataset jest pusty
        return DataLoader(
            synthetic_dataset, batch_size=1
        )  # DataLoader dla pustego datasetu

    return DataLoader(synthetic_dataset, batch_size=loader_batch_size, shuffle=True)


# --- green_diffusion (jak w poprzedniej sugestii) ---
def green_diffusion(inputs, noise_level=0.05, apply_prob=0.3):
    if torch.rand(1).item() > apply_prob:
        return inputs

    if inputs.ndim == 4 and inputs.shape[1] >= 3:
        noise = torch.randn_like(inputs[:, 1:2, :, :]) * noise_level

        # Modyfikuj inputs w miejscu lub kopię, zależnie od preferencji. Kopia jest bezpieczniejsza.
        # inputs_modified = inputs.clone()
        # inputs_modified[:, 1:2, :, :] += noise
        # inputs_modified = torch.clamp(inputs_modified, 0, 1) # Zakładając zakres [0,1]
        # return inputs_modified

        # Modyfikacja w miejscu (jeśli tensor na to pozwala i nie jest liściem z requires_grad=True)
        # Dla danych wejściowych to zwykle OK.
        with torch.no_grad():  # Modyfikacja danych wejściowych nie powinna być śledzona przez autograd
            inputs[:, 1:2, :, :].add_(noise)  # Modyfikacja w miejscu
            inputs.clamp_(0, 1)  # Modyfikacja w miejscu, zakładając zakres [0,1]
    return inputs


# --- Pozostałe funkcje pomocnicze (get_best_finetuning_params, verify_fine_tuned_model,
# compare_base_and_finetuned, ensure_class_folder_structure, print_directory_structure)
# Zakładam, że są one OK lub ich ewentualne błędy nie są przyczyną problemów z parametrami treningu.
# (Kod tych funkcji pominięty dla zwięzłości, ale powinien być w pliku)


def get_best_finetuning_params(model_type, dataset_size):
    """
    Zwraca rekomendowane parametry fine-tuningu dla danego typu modelu i wielkości zbioru danych.
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
        "warmup_epochs": 1,
        "early_stopping_patience": 5,
    }

    if (
        "b0" in model_type.lower()
        or "mobile" in model_type.lower()
        or "efficientnet" in model_type.lower()
        and "b0" in model_type.lower()
    ):
        params["learning_rate"] = 0.0005
        params["freeze_ratio"] = 0.7
    elif "resnet50" in model_type.lower() or (
        "efficientnet" in model_type.lower() and "b4" in model_type.lower()
    ):
        params["learning_rate"] = 0.0002
        params["freeze_ratio"] = 0.8
    elif "large" in model_type.lower() or (
        "efficientnet" in model_type.lower() and "b7" in model_type.lower()
    ):
        params["learning_rate"] = 0.00008
        params["freeze_ratio"] = 0.9
        params["batch_size"] = 8

    if dataset_size < 500:
        params["num_epochs"] = 30
        params["freeze_ratio"] = min(0.95, params["freeze_ratio"] + 0.1)
        params["use_mixup"] = True
        params["weight_decay"] = 0.02
        params["early_stopping_patience"] = 7
    elif dataset_size < 2000:
        params["num_epochs"] = 20
    else:  # Duży zbiór danych
        params["num_epochs"] = 15
        params["freeze_ratio"] = max(0.5, params["freeze_ratio"] - 0.1)
        params["learning_rate"] *= 1.5  # Ostrożniejsze zwiększenie LR

    # Sanity check dla batch_size
    if params["batch_size"] < 1:
        params["batch_size"] = 1
    return params


def verify_fine_tuned_model(model_path, test_dir, top_n=5):
    import random

    print(f"\n=== WERYFIKACJA MODELU {model_path} ===")
    try:
        classifier = ImageClassifier(weights_path=model_path)
    except Exception as e:
        return {
            "error": f"Nie udało się załadować modelu: {e}",
            "correct": [],
            "incorrect": [],
            "accuracy": 0.0,
            "total": 0,
        }

    test_images = []
    if not os.path.isdir(test_dir):
        return {
            "error": f"Katalog testowy {test_dir} nie istnieje.",
            "correct": [],
            "incorrect": [],
            "accuracy": 0.0,
            "total": 0,
        }

    for class_name in os.listdir(test_dir):
        class_path = os.path.join(test_dir, class_name)
        if os.path.isdir(class_path):
            for f_name in os.listdir(class_path):
                if f_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    test_images.append((os.path.join(class_path, f_name), class_name))

    if not test_images:
        return {
            "error": "Nie znaleziono obrazów testowych w podkatalogach klas.",
            "correct": [],
            "incorrect": [],
            "accuracy": 0.0,
            "total": 0,
        }

    selected_images = random.sample(
        test_images, min(len(test_images), top_n * 3 if top_n > 0 else len(test_images))
    )
    results = {
        "correct": [],
        "incorrect": [],
        "accuracy": 0.0,
        "total": len(selected_images),
        "model_classes": classifier.class_names,
    }

    for img_path, expected_class in selected_images:
        try:
            prediction_result = classifier.predict(img_path)
            predicted_class_name = prediction_result["class_name"]
            confidence = prediction_result["confidence"]
            item = {
                "path": img_path,
                "expected": expected_class,
                "predicted": predicted_class_name,
                "confidence": confidence,
            }
            if expected_class.lower() == predicted_class_name.lower():
                results["correct"].append(item)
            else:
                results["incorrect"].append(item)
        except Exception as e_pred:
            print(f"Błąd przy predykcji dla {img_path}: {e_pred}")
            results["incorrect"].append(
                {
                    "path": img_path,
                    "expected": expected_class,
                    "predicted": f"BŁĄD: {e_pred}",
                    "confidence": 0.0,
                }
            )

    if results["total"] > 0:
        results["accuracy"] = len(results["correct"]) / results["total"]

    print(f"Dokładność na wybranych próbkach: {results['accuracy']:.2%}")
    # ... (reszta logowania wyników)
    return results


def compare_base_and_finetuned(base_model_path, finetuned_model_path, test_dir):
    # (Implementacja bez zmian, zakładając, że jest poprawna)
    print(f"\n=== PORÓWNANIE MODELI ===")
    # ...
    return {}  # Placeholder


def ensure_class_folder_structure(directory):
    # (Implementacja bez zmian, zakładając, że jest poprawna)
    return True  # Placeholder


def print_directory_structure(directory, indent=""):
    # (Implementacja bez zmian, zakładając, że jest poprawna)
    pass  # Placeholder
