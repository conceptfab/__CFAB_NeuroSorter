import json
import os
import traceback

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


class ImageClassifier:
    def __init__(self, model_type="resnet50", num_classes=10, weights_path=None):
        self.model_type = model_type
        self.num_classes = num_classes
        self.class_names = {}

        # Bezpieczne sprawdzenie CUDA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            print(
                "UWAGA: Urządzenie ustawione na CUDA, ale CUDA nie jest dostępne. Przechodzę na CPU."
            )
            self.device = torch.device("cpu")

        # Dodanie opcji precision
        self.precision = "half" if self.device.type == "cuda" else "full"

        # Dodanie informacji diagnostycznych o CUDA
        if torch.cuda.is_available():
            print(
                f"CUDA jest dostępne. Wykryto urządzenie: {torch.cuda.get_device_name(0)}"
            )
            print(f"Wersja CUDA: {torch.version.cuda}")
        else:
            print("CUDA nie jest dostępne. Używanie CPU.")

        # Inicjalizacja modelu
        self.model = self._create_model()

        # Załaduj wagi jeśli podano ścieżkę
        if weights_path and os.path.exists(weights_path):
            self._load_weights(weights_path)

        # Standardowe przekształcenia dla obrazów
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.model.eval()

    def _create_model(self):
        """Tworzenie modelu bazowego z pretrenowanymi wagami"""
        if self.model_type == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(
                    model.fc.in_features,
                    self.num_classes,
                ),
            )
        elif self.model_type == "efficientnet":
            model = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
            )
            model.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(
                    model.classifier[1].in_features,
                    self.num_classes,
                ),
            )
        elif self.model_type == "mobilenet":
            model = models.mobilenet_v3_large(
                weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
            )
            model.classifier[3] = nn.Linear(
                model.classifier[3].in_features,
                self.num_classes,
            )
        elif self.model_type == "vit":
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
            model.heads.head = nn.Linear(
                model.heads.head.in_features,
                self.num_classes,
            )
        elif self.model_type == "convnext":
            model = models.convnext_tiny(
                weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            )
            model.classifier[2] = nn.Linear(
                model.classifier[2].in_features,
                self.num_classes,
            )
        else:
            raise ValueError(f"Nieobsługiwany typ modelu: {self.model_type}")

        # Przenieś model na odpowiednie urządzenie (bez zmiany typu danych)
        model = model.to(device=self.device)

        return model

    def _load_weights(self, weights_path):
        """Ładowanie wag modelu"""
        try:
            # Załaduj checkpoint bez parametru weights_only
            checkpoint = torch.load(weights_path, map_location=self.device)

            # Sprawdź, czy checkpoint to słownik czy bezpośrednio stan modelu
            if isinstance(checkpoint, dict):
                # Aktualizacja typu modelu z pliku checkpoint
                if (
                    "model_type" in checkpoint
                    and checkpoint["model_type"] != self.model_type
                ):
                    self.model_type = checkpoint["model_type"]

                # Aktualizacja liczby klas, jeśli jest dostępna
                if "num_classes" in checkpoint:
                    # Jeśli liczba klas jest inna, utwórz nowy model
                    # z właściwą liczbą klas
                    if checkpoint["num_classes"] != self.num_classes:
                        self.num_classes = checkpoint["num_classes"]
                        self.model = self._create_model()

                if "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                elif "state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["state_dict"])
                else:
                    # Próba bezpośredniego ładowania
                    self.model.load_state_dict(checkpoint)

                # Załaduj nazwy klas, jeśli istnieją
                if "class_names" in checkpoint:
                    self.class_names = checkpoint["class_names"]
                    print(f"Załadowano mapowanie klas: {self.class_names}")
                # Sprawdź również w metadanych
                elif (
                    "metadata" in checkpoint and "class_names" in checkpoint["metadata"]
                ):
                    self.class_names = checkpoint["metadata"]["class_names"]
                    print(
                        "Załadowano mapowanie klas z metadanych: " f"{self.class_names}"
                    )

                # Sprawdź, czy class_names zostały poprawnie załadowane
                # i czy klucze są stringami
                if self.class_names and not all(
                    isinstance(k, str) for k in self.class_names.keys()
                ):
                    print(
                        "OSTRZEŻENIE: Wykryto klucze niebędące stringami w "
                        "class_names załadowanych z pliku .pt. Konwertowanie..."
                    )
                    try:
                        self.class_names = {
                            str(k): v for k, v in self.class_names.items()
                        }
                        print(
                            "Mapowanie klas po konwersji kluczy: " f"{self.class_names}"
                        )
                    except Exception as e:
                        print(f"BŁĄD podczas konwersji kluczy class_names: {e}")
                        self.class_names = {}  # Resetuj w razie błędu konwersji

                # Jeśli class_names nadal są puste, spróbuj załadować
                # z pliku _config.json
                if not self.class_names:
                    print(
                        "Nie znaleziono mapowania klas w pliku .pt. "
                        "Próbuję załadować z pliku _config.json..."
                    )
                    config_path = os.path.splitext(weights_path)[0] + "_config.json"
                    if os.path.exists(config_path):
                        try:
                            with open(config_path, "r", encoding="utf-8") as f:
                                config_data = json.load(f)
                            if (
                                "class_names" in config_data
                                and isinstance(config_data["class_names"], dict)
                                and config_data["class_names"]
                            ):
                                potential_class_names = config_data["class_names"]
                                # Upewnij się, że klucze są stringami
                                if not all(
                                    isinstance(k, str)
                                    for k in potential_class_names.keys()
                                ):
                                    print(
                                        "OSTRZEŻENIE: Wykryto klucze niebędące "
                                        "stringami w class_names załadowanych z "
                                        "pliku _config.json. Konwertowanie..."
                                    )
                                    self.class_names = {
                                        str(k): v
                                        for k, v in potential_class_names.items()
                                    }
                                else:
                                    self.class_names = potential_class_names
                                print(
                                    "Załadowano mapowanie klas z pliku "
                                    f"{config_path}: {self.class_names}"
                                )
                            else:
                                print(
                                    f"Plik {config_path} nie zawiera "
                                    "poprawnego mapowania 'class_names'."
                                )
                        except json.JSONDecodeError:
                            print(
                                "BŁĄD: Nie można zdekodować pliku JSON: "
                                f"{config_path}"
                            )
                        except Exception as e:
                            print("BŁĄD podczas ładowania pliku " f"{config_path}: {e}")
                    else:
                        print(f"Plik konfiguracyjny {config_path} nie istnieje.")

            else:
                # Bezpośrednie ładowanie stanu modelu
                self.model.load_state_dict(checkpoint)
                # W tym przypadku również spróbuj załadować z JSON
                print(
                    "Checkpoint nie jest słownikiem. Ładowanie tylko wag modelu. "
                    "Próbuję załadować class_names z _config.json..."
                )
                config_path = os.path.splitext(weights_path)[0] + "_config.json"
                if os.path.exists(config_path):
                    try:
                        with open(config_path, "r", encoding="utf-8") as f:
                            config_data = json.load(f)
                        if (
                            "class_names" in config_data
                            and isinstance(config_data["class_names"], dict)
                            and config_data["class_names"]
                        ):
                            potential_class_names = config_data["class_names"]
                            # Upewnij się, że klucze są stringami
                            if not all(
                                isinstance(k, str) for k in potential_class_names.keys()
                            ):
                                print(
                                    "OSTRZEŻENIE: Wykryto klucze niebędące "
                                    "stringami w class_names załadowanych z "
                                    "pliku _config.json. Konwertowanie..."
                                )
                                self.class_names = {
                                    str(k): v for k, v in potential_class_names.items()
                                }
                            else:
                                self.class_names = potential_class_names
                            print(
                                "Załadowano mapowanie klas z pliku "
                                f"{config_path}: {self.class_names}"
                            )
                        else:
                            print(
                                f"Plik {config_path} nie zawiera "
                                "poprawnego mapowania 'class_names'."
                            )
                    except json.JSONDecodeError:
                        print(
                            "BŁĄD: Nie można zdekodować pliku JSON: " f"{config_path}"
                        )
                    except Exception as e:
                        print("BŁĄD podczas ładowania pliku " f"{config_path}: {e}")
                else:
                    print(f"Plik konfiguracyjny {config_path} nie istnieje.")

        except Exception as e:
            raise ValueError(f"Nie udało się załadować modelu: {str(e)}")

        # Dodano logowanie końcowego stanu class_names
        if self.class_names:
            print(
                "Końcowy stan self.class_names po załadowaniu: " f"{self.class_names}"
            )
        else:
            print(
                "OSTRZEŻENIE: Nie udało się załadować mapowania klas ani z "
                "pliku .pt, ani z _config.json."
            )

        # Ustaw model w tryb ewaluacji
        self.model.eval()

    def predict(self, image_path):
        """Przewidywanie kategorii dla jednego obrazu"""
        try:
            print(f"DEBUG: Rozpoczynam klasyfikację obrazu: {image_path}")

            # Dodana weryfikacja czy słownik class_names istnieje
            if not hasattr(self, "class_names") or not self.class_names:
                print("DEBUG: Brak słownika class_names, inicjalizuję pusty słownik")
                self.class_names = {}

            print(f"DEBUG: Próba otwarcia obrazu: {image_path}")
            image = Image.open(image_path).convert("RGB")
            print(
                f"DEBUG: Obraz otwarty pomyślnie, rozmiar: {image.size}, tryb: {image.mode}"
            )

            # Poprawka: Oddzielamy transformację od przeniesienia na urządzenie
            print("DEBUG: Rozpoczynam transformację obrazu")
            image_tensor = self.transform(image).unsqueeze(0)
            print(
                f"DEBUG: Transformacja zakończona, kształt tensora: {image_tensor.shape}"
            )

            # Dodajemy obsługę half precision z jawnie określonym typem
            if self.precision == "half" and torch.cuda.is_available():
                print("DEBUG: Używam half precision na CUDA")
                image_tensor = image_tensor.to(self.device, dtype=torch.float16)
            else:
                print("DEBUG: Używam full precision")
                image_tensor = image_tensor.to(self.device, dtype=torch.float32)

            print("DEBUG: Rozpoczynam predykcję modelu")
            with torch.no_grad():
                # Dodajemy autocast dla spójności z resztą kodu
                if self.precision == "half" and torch.cuda.is_available():
                    with torch.amp.autocast(device_type="cuda", enabled=True):
                        outputs = self.model(image_tensor)
                else:
                    outputs = self.model(image_tensor)

                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted_idx = torch.max(outputs, 1)
                confidence = probabilities[0][predicted_idx].item()

            predicted_class = predicted_idx.item()
            print(
                f"DEBUG: Predykcja zakończona - klasa: {predicted_class}, pewność: {confidence:.4f}"
            )

            # ---> START DODANEGO KODU <---
            key_to_find = str(predicted_class)  # Klucz, którego szukamy
            print(f"DEBUG: Szukam klucza: '{key_to_find}' (typ: {type(key_to_find)})")
            print(f"DEBUG: Dostępny słownik class_names: {self.class_names}")
            if self.class_names:
                print(
                    f"DEBUG: Typy kluczy w class_names: {[type(k) for k in self.class_names.keys()]}"
                )
            # ---> KONIEC DODANEGO KODU <---

            class_name = self.class_names.get(key_to_find)
            if class_name is None:
                print(
                    f"UWAGA: Nie znaleziono nazwy dla klasy {predicted_class} w słowniku class_names"
                )
                print(
                    f"DEBUG: Słownik class_names w momencie błędu: {self.class_names}"
                )
                class_name = f"Kategoria_{predicted_class}"

            print(
                f"DEBUG: Końcowy wynik - klasa: {class_name}, pewność: {confidence:.4f}"
            )

            return {
                "class_id": predicted_class,
                "class_name": class_name,
                "confidence": confidence,
            }
        except Exception as e:
            error_msg = f"Błąd w ai.classifier.predict: {str(e)}"
            print(f"DEBUG: Wystąpił błąd: {error_msg}")
            print(f"DEBUG: Traceback: {traceback.format_exc()}")
            raise ValueError(error_msg)

    def batch_predict(self, image_paths, batch_size=16):
        """Przewidywanie kategorii dla wielu obrazów w trybie wsadowym."""
        # Dynamiczne określanie optymalnego rozmiaru batcha, jeśli nie określono
        if batch_size is None:
            batch_size = self.auto_select_batch_size()

        results = []

        # Przetwarzanie wsadowe
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            batch_images = []

            # Wczytaj i przetwórz obrazy
            for path in batch_paths:
                image = Image.open(path).convert("RGB")
                image_tensor = self.transform(image).unsqueeze(0)
                batch_images.append(image_tensor)

            # Połącz w jeden tensor i przenieś na urządzenie
            batch_tensor = torch.cat(batch_images, dim=0)

            # Ustaw odpowiedni typ danych i urządzenie
            if self.precision == "half" and torch.cuda.is_available():
                batch_tensor = batch_tensor.to(device=self.device, dtype=torch.float16)
            else:
                batch_tensor = batch_tensor.to(device=self.device, dtype=torch.float32)

            # Wykonaj predykcję
            with torch.no_grad():
                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=(self.precision == "half" and torch.cuda.is_available()),
                ):
                    outputs = self.model(batch_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    _, predicted_idx = torch.max(outputs, 1)

            # Przetwórz wyniki
            for j, path in enumerate(batch_paths):
                predicted_class = predicted_idx[j].item()
                confidence = probabilities[j][predicted_class].item()

                # Używamy oryginalnej nazwy klasy, jeśli istnieje w słowniku class_names
                class_name = self.class_names.get(str(predicted_class))
                if class_name is None:
                    print(
                        f"UWAGA: Nie znaleziono nazwy dla klasy {predicted_class} w słowniku class_names"
                    )
                    class_name = f"Kategoria_{predicted_class}"

                # Dodaj logi diagnostyczne dla pierwszych kilku wyników
                if j < 3:  # Pokaż tylko pierwsze 3 wyniki dla przejrzystości
                    print(
                        f"DEBUG: batch_predict() - j={j}, predicted_class={predicted_class}, class_name={class_name}"
                    )

                results.append(
                    {
                        "class_id": predicted_class,
                        "class_name": class_name,
                        "confidence": confidence,
                    }
                )

            # Wyczyść pamięć GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results

    def batch_predict_threaded(self, image_paths, num_threads=4):
        """
        Przewidywanie kategorii dla wielu obrazów z wykorzystaniem wielu wątków.

        Args:
            image_paths: Lista ścieżek do obrazów
            num_threads: Liczba wątków do wykorzystania

        Returns:
            Lista wyników klasyfikacji
        """
        from concurrent.futures import ThreadPoolExecutor

        results = []

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(self.predict, path) for path in image_paths]
            results = [future.result() for future in futures]

        return results

    def save(self, save_path, metadata=None):
        """Zapisywanie modelu z metadanymi"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Zapisz model
        checkpoint = {
            "model_type": self.model_type,
            "num_classes": self.num_classes,
            "model_state_dict": self.model.state_dict(),
            "class_names": self.class_names,
        }

        if metadata:
            checkpoint["metadata"] = metadata

        torch.save(checkpoint, save_path)

        # Zapisz też konfigurację w formacie JSON dla łatwiejszego odczytu
        config_path = os.path.splitext(save_path)[0] + "_config.json"
        config = {
            "model_type": self.model_type,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "metadata": metadata or {},
        }

        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

        return save_path, config_path

    def get_model_info(self):
        """Zwraca informacje o modelu"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        return {
            "model_type": self.model_type,
            "num_classes": self.num_classes,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device),
            "input_size": (224, 224),
            "class_names": self.class_names,
        }

    def __del__(self):
        """Destruktor klasy, zwalniający zasoby."""
        # Wyczyść pamięć GPU
        if (
            hasattr(self, "device")
            and torch.cuda.is_available()
            and self.device.type == "cuda"
        ):
            torch.cuda.empty_cache()

    def enable_quantization(self):
        """Włącza kwantyzację modelu dla szybszej inferencji na CPU"""
        if self.device.type == "cpu":
            self.model = torch.quantization.quantize_dynamic(
                self.model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
            )
            print("Włączono kwantyzację modelu dla CPU")
        else:
            print("Kwantyzacja jest obsługiwana tylko dla CPU")

    def batch_predict_with_cache(self, image_paths, batch_size=16, use_cache=True):
        """
        Przewidywanie kategorii dla wielu obrazów w trybie wsadowym z cache'owaniem.

        Args:
            image_paths: Lista ścieżek do obrazów
            batch_size: Rozmiar wsadu do przetwarzania
            use_cache: Czy używać cache'a dla wcześniej przetworzonych obrazów

        Returns:
            Lista wyników klasyfikacji
        """
        results = []

        # Stwórz cache dla ścieżek plików, jeśli nie istnieje
        if not hasattr(self, "_prediction_cache"):
            self._prediction_cache = {}

        # Podziel obrazy na te w cache'u i te do przetworzenia
        to_process = []
        cached_results = {}

        if use_cache:
            for path in image_paths:
                file_hash = self._get_file_hash(path)
                if file_hash in self._prediction_cache:
                    cached_results[path] = self._prediction_cache[file_hash]
                else:
                    to_process.append(path)
        else:
            to_process = image_paths

        # Jeśli nie ma nic do przetworzenia, zwróć tylko cache'owane wyniki
        if not to_process:
            return [
                cached_results[path] for path in image_paths if path in cached_results
            ]

        # Przetwórz obrazy, które nie są w cache'u
        batch_results = self.batch_predict(to_process, batch_size)

        # Zaktualizuj cache
        for i, path in enumerate(to_process):
            if i < len(batch_results):  # Sprawdza czy indeks istnieje
                file_hash = self._get_file_hash(path)
                self._prediction_cache[file_hash] = batch_results[i]

        # Połącz wyniki
        for path in image_paths:
            if path in cached_results:
                results.append(cached_results[path])
            else:
                # Znajdź indeks w to_process i dodaj odpowiadający wynik
                if path in to_process:
                    idx = to_process.index(path)
                    if idx < len(batch_results):  # Dodaj sprawdzenie
                        results.append(batch_results[idx])

        return results

    def _get_file_hash(self, file_path):
        """Generuje hash dla pliku na podstawie ścieżki i czasu modyfikacji"""
        import hashlib

        mod_time = os.path.getmtime(file_path)
        file_size = os.path.getsize(file_path)
        hash_str = f"{file_path}_{mod_time}_{file_size}"
        return hashlib.md5(hash_str.encode()).hexdigest()

    def validate_model(self, validation_dir=None, validation_data=None, batch_size=32):
        """
        Sprawdza poprawność modelu na zbiorze walidacyjnym.

        Args:
            validation_dir: Katalog ze zbiorem walidacyjnym
            validation_data: Opcjonalnie predefiniowany zbiór danych
            batch_size: Rozmiar wsadu

        Returns:
            dict: Wyniki walidacji (dokładność, macierz pomyłek itp.)
        """
        import numpy as np
        from torch.utils.data import DataLoader
        from torchvision import datasets

        self.model.eval()

        # Przygotuj dane walidacyjne
        if validation_data is not None:
            val_loader = DataLoader(
                validation_data, batch_size=batch_size, shuffle=False
            )
        elif validation_dir is not None:
            val_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            val_dataset = datasets.ImageFolder(validation_dir, transform=val_transform)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            raise ValueError("Musisz podać validation_dir lub validation_data")

        # Inicjalizacja metryk
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        # Walidacja
        with torch.no_grad():
            for inputs, labels in val_loader:
                # Poprawka: Dodajemy jawne określenie typu dla inputów
                inputs = inputs.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device)

                # Używamy autocast także tutaj dla spójności
                if self.precision == "half" and torch.cuda.is_available():
                    with torch.amp.autocast(device_type="cuda", enabled=True):
                        outputs = self.model(inputs)
                else:
                    outputs = self.model(inputs)

                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Oblicz macierz pomyłek
        from sklearn.metrics import confusion_matrix

        conf_matrix = confusion_matrix(all_labels, all_preds)

        # Oblicz dokładność dla każdej klasy
        class_accuracy = {}
        for i in range(len(conf_matrix)):
            class_accuracy[i] = (
                conf_matrix[i, i] / conf_matrix[i].sum()
                if conf_matrix[i].sum() > 0
                else 0
            )

        # Zwróć wyniki
        return {
            "accuracy": correct / total,
            "confusion_matrix": conf_matrix,
            "class_accuracy": class_accuracy,
            "num_samples": total,
        }

    def set_precision(self, precision="half"):
        """Ustawia precyzję obliczeń (half/full) dla modelu."""
        if precision == "half":
            # Sprawdź czy CUDA jest dostępne zanim przełączasz na half precision
            if torch.cuda.is_available():
                self.precision = "half"
                self.model = self.model.half()  # Konwertuje model do FP16
            else:
                print("Half precision wymaga CUDA. Używanie standardowej precyzji.")
                self.precision = "full"
                self.model = self.model.float()
        else:
            self.precision = "full"
            self.model = self.model.float()  # Przywraca FP32

    def compress_model(self):
        """Kompresuje model za pomocą pruningu i kwantyzacji."""
        try:
            import torch.nn.utils.prune as prune

            # Zastosuj pruning do warstw konwolucyjnych (usuwanie mniej ważnych wag)
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    prune.l1_unstructured(
                        module, name="weight", amount=0.3
                    )  # Usuń 30% najmniej ważnych wag

            # Zastosuj kwantyzację
            self.enable_quantization()

            return True
        except Exception as e:
            print(f"Błąd podczas kompresji modelu: {e}")
            return False

    def auto_select_batch_size(self):
        """
        Automatycznie wybiera optymalny rozmiar batcha na podstawie dostępnego sprzętu.

        Returns:
            int: Optymalny rozmiar batcha
        """
        if not torch.cuda.is_available():
            # Na CPU używamy mniejszych batchów
            return 8

        # Na GPU dobieramy rozmiar batcha na podstawie ilości pamięci
        try:
            memory_info = torch.cuda.get_device_properties(0).total_memory
            memory_gb = memory_info / (1024**3)  # Konwersja na GB

            # Proste mapowanie pamięci GPU na rozmiar batcha
            if memory_gb > 10:
                return 64
            elif memory_gb > 8:
                return 32
            elif memory_gb > 4:
                return 16
            else:
                return 8
        except Exception:
            # W przypadku błędu, użyj bezpiecznej wartości
            return 16

    def get_class_mapping(self):
        """
        Zwraca mapowanie indeksów klas na nazwy klas.

        Returns:
            dict: Słownik mapujący indeksy klas na ich nazwy
        """
        # Jeśli nie mamy mapowania klas, zwróć pusty słownik
        if not self.class_names:
            print("UWAGA: Brak mapowania klas w modelu.")
            return {}

        # Zwróć kopię słownika class_names
        return self.class_names.copy()

    @staticmethod
    def auto_select_architecture(
        dataset_size=None,
        num_classes=None,
        cuda_available=None,
        gpu_memory=None,
        cpu_cores=None,
    ):
        """
        Automatycznie wybiera optymalną architekturę modelu na podstawie:
        - wielkości zbioru danych
        - liczby klas
        - dostępności CUDA
        - ilości pamięci GPU
        - liczby rdzeni CPU
        """
        import os

        import psutil

        # Automatyczne wykrywanie parametrów, jeśli nie zostały podane
        if cuda_available is None:
            cuda_available = torch.cuda.is_available()

        if gpu_memory is None and cuda_available:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (
                1024**3
            )  # w GB

        if cpu_cores is None:
            cpu_cores = os.cpu_count() or 4

        # Domyślne wartości
        if dataset_size is None:
            dataset_size = 1000  # Założenie średniej wielkości datasetu

        if num_classes is None:
            num_classes = 10  # Założenie standardowej liczby klas

        # Logika wyboru architektury
        # 1. Dla małych urządzeń lub bez CUDA - lekkie modele
        if not cuda_available or (gpu_memory is not None and gpu_memory < 4):
            return "mobilenet"  # Najlżejszy model

        # 2. Dla średnich zasobów
        if gpu_memory is not None and 4 <= gpu_memory < 8:
            if dataset_size < 5000:
                return "efficientnet"  # Dobry kompromis wydajność/jakość
            else:
                return "mobilenet"  # Szybszy trening dla większych zbiorów

        # 3. Dla dużych zasobów obliczeniowych
        if dataset_size < 3000:
            # Dla mniejszych zbiorów danych
            if num_classes > 50:
                return "convnext"  # Lepszy dla wielu klas
            else:
                return "vit"  # Dobry dla mniejszych zbiorów z umiarkowaną liczbą klas
        elif dataset_size < 10000:
            return "resnet50"  # Sprawdzony model dla średnich zbiorów
        else:
            # Dla bardzo dużych zbiorów danych
            if gpu_memory is not None and gpu_memory > 12:
                return "convnext"  # Zaawansowany model dla dużych zbiorów
            else:
                return "resnet50"  # Standardowy wybór dla dużych zbiorów

        # Domyślnie używamy ResNet50 jako sprawdzonego modelu
        return "resnet50"

    def debug_category_mapping(self, directory, top_n=5):
        """
        Funkcja debugująca, która klasyfikuje obrazy i porównuje z faktycznymi folderami.

        Args:
            directory: Katalog z obrazami w odpowiedniej strukturze katalogów
            top_n: Liczba przykładów do wyświetlenia

        Returns:
            dict: Raport z porównaniem
        """
        report = {"matches": [], "mismatches": []}

        for root, _, files in os.walk(directory):
            # Pobierz względną ścieżkę do katalogu
            rel_path = os.path.relpath(root, directory)
            if rel_path == ".":
                continue

            # Pobierz nazwę kategorii z ścieżki
            expected_category = rel_path.replace("\\", "/")

            # Klasyfikuj kilka obrazów z tego katalogu
            image_files = [
                os.path.join(root, f)
                for f in files
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            ]
            if not image_files:
                continue

            # Wybierz do 5 losowych obrazów
            import random

            sample_images = random.sample(image_files, min(top_n, len(image_files)))

            for img_path in sample_images:
                result = self.predict(img_path)
                predicted = result["class_name"]

                # Sprawdź zgodność
                if expected_category.lower() == predicted.lower():
                    report["matches"].append(
                        {
                            "image": os.path.basename(img_path),
                            "expected": expected_category,
                            "predicted": predicted,
                            "confidence": result["confidence"],
                        }
                    )
                else:
                    report["mismatches"].append(
                        {
                            "image": os.path.basename(img_path),
                            "expected": expected_category,
                            "predicted": predicted,
                            "confidence": result["confidence"],
                        }
                    )

        # Wyświetl raport
        print(f"Poprawnie sklasyfikowano: {len(report['matches'])} obrazów")
        print(f"Błędnie sklasyfikowano: {len(report['mismatches'])} obrazów")

        print("\nPrzykłady błędnych klasyfikacji:")
        for i, mismatch in enumerate(report["mismatches"][:top_n]):
            print(
                f"{i+1}. {mismatch['image']} - Oczekiwano: {mismatch['expected']}, "
                f"Otrzymano: {mismatch['predicted']} ({mismatch['confidence']:.2f})"
            )

        return report

    def verify_class_mapping(self):
        """Weryfikuje czy mapowanie klas zostało poprawnie przypisane."""
        if not hasattr(self, "class_names") or not self.class_names:
            print("UWAGA: Brak mapowania klas (class_names) w modelu.")
            return False

        # Sprawdź czy mapowanie zawiera sensowne wartości
        if len(self.class_names) == 0:
            print("UWAGA: Mapowanie klas jest puste.")
            return False

        print(f"Znaleziono {len(self.class_names)} klas w mapowaniu:")
        for idx, name in self.class_names.items():
            print(f"  - ID {idx}: {name}")

        return True
