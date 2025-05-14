import json
import os
import traceback

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

from ai.models import get_model


class ImageClassifier:
    def __init__(
        self, model_type="b0", num_classes=10, weights_path=None, input_size=None
    ):
        # Jeśli istnieje plik konfiguracyjny, pobierz z niego model_type, num_classes i input_size
        config_file_name = "_config.json"
        config_path = None
        if weights_path:
            config_path = os.path.splitext(weights_path)[0] + config_file_name
        config_data = None
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                # Nadpisz model_type i num_classes z configu
                if "model_type" in config_data:
                    model_type = config_data["model_type"]
                if "num_classes" in config_data:
                    num_classes = config_data["num_classes"]
                # Pobierz input_size z configu
                if "input_size" in config_data:
                    self.input_size = (
                        tuple(config_data["input_size"])
                        if isinstance(config_data["input_size"], list)
                        else (config_data["input_size"], config_data["input_size"])
                    )
            except Exception as e:
                print(
                    f"BŁĄD: Nie udało się wczytać pliku konfiguracyjnego {config_path}: {e}"
                )
        # Ustaw domyślny input_size, jeśli nie został określony
        if input_size:
            self.input_size = (
                input_size
                if isinstance(input_size, tuple)
                else (input_size, input_size)
            )
        elif not hasattr(self, "input_size"):
            self.input_size = (224, 224)  # Domyślny rozmiar

        # Normalizacja nazwy modelu
        self.model_type = model_type.lower()
        self.model_type = self.model_type.replace("efficientnet-", "").replace(
            "efficientnet_", ""
        )
        self.model_type = self.model_type.replace("resnet", "")
        self.model_type = self.model_type.replace("mobilenet", "mobile")
        self.model_type = self.model_type.replace("convnext", "")
        self.model_type = self.model_type.replace("vit", "")
        self.model_type = self.model_type.strip(
            "-_"
        )  # Usuń ewentualne myślniki i podkreślniki na końcach

        self.num_classes = num_classes
        self.class_names = {}
        self.weights_path = weights_path  # <-- DODANO: Zapisz ścieżkę do wag

        # Bezpieczne sprawdzenie CUDA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            print(
                "UWAGA: Urządzenie ustawione na CUDA, ale CUDA nie jest dostępne. "
                "Przechodzę na CPU."
            )
            self.device = torch.device("cpu")

        # Dodanie opcji precision
        self.precision = "half" if self.device.type == "cuda" else "full"

        # Dodanie informacji diagnostycznych o CUDA
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"CUDA jest dostępne. Wykryto urządzenie: {device_name}")
            print(f"Wersja CUDA: {torch.version.cuda}")
        else:
            print("CUDA nie jest dostępne. Używanie CPU.")

        # Inicjalizacja modelu
        self.model = self._create_model()

        # Załaduj wagi jeśli podano ścieżkę
        if weights_path and os.path.exists(weights_path):
            self._load_weights(weights_path)

        # Standardowe przekształcenia dla obrazów z dynamicznym rozmiarem
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.model.eval()

    def _create_model(self):
        """Tworzenie modelu bazowego z pretrenowanymi wagami"""
        try:
            # Użyj funkcji get_model z models.py
            model = get_model(
                model_arch=self.model_type, num_classes=self.num_classes, logger=print
            )

            # Przenieś model na odpowiednie urządzenie
            model = model.to(device=self.device)

            return model

        except Exception:
            msg = f"Nieobsługiwany typ modelu: {self.model_type}"
            raise ValueError(msg)

    def _load_weights(self, weights_path):
        """Ładowanie wag modelu"""
        try:
            # Załaduj checkpoint
            checkpoint = torch.load(
                weights_path, map_location=self.device, weights_only=False
            )

            class_names_checkpoint = None
            class_names_config = None

            # --- Krok 1: Odczytaj class_names z checkpointu (jeśli istnieją) ---
            if isinstance(checkpoint, dict):
                if "class_names" in checkpoint:
                    class_names_checkpoint = checkpoint["class_names"]
                elif (
                    "metadata" in checkpoint and "class_names" in checkpoint["metadata"]
                ):
                    class_names_checkpoint = checkpoint["metadata"]["class_names"]

                # Wstępna normalizacja kluczy w class_names_checkpoint
                if class_names_checkpoint:
                    try:
                        class_names_checkpoint = {
                            str(k): v for k, v in class_names_checkpoint.items()
                        }
                    except Exception as e_conv_ckpt_pre:
                        print(
                            "UWAGA: Problem z wstępną konwersją kluczy "
                            f"class_names z checkpointu: {e_conv_ckpt_pre}"
                        )
                        class_names_checkpoint = {}  # Reset w razie problemu

            # --- Krok 2: Odczytaj class_names z _config.json (jeśli istnieje) ---
            config_file_name = "_config.json"
            config_path = os.path.splitext(weights_path)[0] + config_file_name
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        config_data = json.load(f)
                    if "class_names" in config_data:
                        class_names_config = config_data["class_names"]
                        # Normalizacja kluczy w class_names_config
                        if class_names_config:
                            try:
                                class_names_config = {
                                    str(k): v for k, v in class_names_config.items()
                                }
                            except Exception as e_conv_json_pre:
                                print(
                                    "UWAGA: Problem z konwersją kluczy "
                                    f"class_names z JSON: {e_conv_json_pre}"
                                )
                                class_names_config = {}  # Reset w razie problemu
                except Exception as e_json:
                    print(
                        "BŁĄD: Nie udało się załadować lub przetworzyć pliku "
                        f"{config_path}: {e_json}"
                    )
            else:
                print(f"INFO: Plik konfiguracyjny {config_path} nie istnieje.")

            # --- Krok 3: Logika decyzyjna i ustawienie self.class_names ---
            print("\n" + "=" * 60)
            print("=== WERYFIKACJA MAPOWANIA KLAS (class_names) ===")
            print("=" * 60)

            if class_names_config is not None:
                self.class_names = class_names_config
                print(
                    f"[INFO] Użyto mapowania klas z pliku "
                    f"konfiguracyjnego: {config_path}"
                )
                if class_names_checkpoint is not None:
                    if class_names_config == class_names_checkpoint:
                        print(
                            "[INFO] Potwierdzono zgodność mapowania klas "
                            "między plikiem konfiguracyjnym a checkpointem."
                        )
                    else:
                        print(
                            "[OSTRZEŻENIE] Niezgodność mapowania klas! Plik "
                            "konfiguracyjny i checkpoint "
                            "zawierają różne definicje class_names."
                        )
                        print(f"  > Użyto danych z: {config_path}")
                        print(
                            f"  > Dane z checkpoint ({weights_path}): "
                            f"{class_names_checkpoint}"
                        )
                        print(
                            f"  > Dane z config ({config_path}): "
                            f"{class_names_config}"
                        )
                        print("  > ZALECANA WERYFIKACJA!")
                else:
                    print(
                        "[INFO] Mapowanie klas z checkpointu nie zostało "
                        "znalezione. Użyto tylko danych z pliku "
                        "konfiguracyjnego."
                    )
            elif class_names_checkpoint is not None:
                self.class_names = class_names_checkpoint
                print(
                    "[OSTRZEŻENIE] Użyto mapowania klas z pliku "
                    f"checkpointu: {weights_path}"
                )
                print(
                    f"  > Plik konfiguracyjny {config_path} nie istnieje "
                    "lub nie zawiera 'class_names'."
                )
                print(
                    f"  > Plik '{config_file_name}' jest preferowanym "
                    "źródłem mapowania klas. Rozważ jego utworzenie."
                )
            else:
                print(
                    "[BŁĄD] Nie znaleziono mapowania klas ani w pliku "
                    "konfiguracyjnym, ani w checkpointu!"
                )
                print("  > Tworzenie domyślnego mapowania klas...")
                self.class_names = {
                    str(i): f"Kategoria_{i}" for i in range(self.num_classes)
                }
                print(f"  > Utworzono {len(self.class_names)} domyślnych kategorii.")

            # --- Krok 4: Weryfikacja mapowania klas ---
            if not self.class_names:
                print("[BŁĄD] Mapowanie klas jest puste!")
                raise ValueError("Nie udało się załadować mapowania klas.")

            print("\nWczytane mapowanie klas:")
            for idx, name in sorted(self.class_names.items(), key=lambda x: int(x[0])):
                print(f"  {idx}: {name}")

            # Załaduj wagi modelu
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            elif not isinstance(checkpoint, dict):
                self.model.load_state_dict(checkpoint)
            else:
                print(
                    f"[OSTRZEŻENIE] Próba załadowania wag z pliku o "
                    f"nieoczekiwanej strukturze (brak klucza 'model_state_dict'). "
                    f"Plik: {weights_path}"
                )
                try:
                    self.model.load_state_dict(checkpoint)
                except RuntimeError as e_load:
                    print(
                        f"[BŁĄD] Nie udało się załadować wag nawet po "
                        f"ostrzeżeniu. Błąd: {e_load}"
                    )
                    # Rzucamy oryginalny wyjątek, aby zachować ślad błędu
                    raise  # Rzucenie oryginalnego wyjątku przechwyconego wcześniej

            print(f"\n[INFO] Pomyślnie załadowano wagi modelu z: {weights_path}")

        except Exception as e:
            print(f"BŁĄD podczas ładowania wag modelu: {str(e)}")
            print(traceback.format_exc())
            raise

    def predict(self, image_path, return_ranking=False):
        """
        Optymalizacja: Usunięcie zbędnych logów debugowania i uproszczenie logiki.
        """
        try:
            # Jeśli słownik class_names nie istnieje, inicjalizuj pusty słownik
            if not hasattr(self, "class_names") or not self.class_names:
                self.class_names = {}

            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0)

            # Obsługa half precision
            if self.precision == "half" and torch.cuda.is_available():
                image_tensor = image_tensor.to(self.device, dtype=torch.float16)
            else:
                image_tensor = image_tensor.to(self.device, dtype=torch.float32)

            with torch.no_grad():
                # Użyj autocast dla spójności
                if self.precision == "half" and torch.cuda.is_available():
                    with torch.amp.autocast(device_type="cuda", enabled=True):
                        outputs = self.model(image_tensor)
                else:
                    outputs = self.model(image_tensor)

                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted_idx = torch.max(outputs, 1)
                confidence = probabilities[0][predicted_idx].item()

            predicted_class = predicted_idx.item()

            # Znajdź nazwę klasy
            key_to_find = str(predicted_class)
            if self.class_names and key_to_find not in self.class_names:
                try:
                    int_key = int(key_to_find)
                    if str(int_key) in self.class_names:
                        key_to_find = str(int_key)
                    elif int_key in self.class_names:
                        key_to_find = int_key
                except ValueError:
                    pass

            class_name = self.class_names.get(
                key_to_find, f"Kategoria_{predicted_class}"
            )

            # Przygotuj wynik
            result = {
                "class_id": predicted_class,
                "class_name": class_name,
                "confidence": confidence,
            }

            # Dodaj ranking jeśli wymagany
            if return_ranking:
                all_probabilities = probabilities[0].cpu().numpy()
                class_ranking = []

                for idx, prob in enumerate(all_probabilities):
                    class_key = str(idx)
                    if class_key not in self.class_names:
                        try:
                            if int(class_key) in self.class_names:
                                class_key = int(class_key)
                        except ValueError:
                            pass

                    class_name_for_idx = self.class_names.get(
                        class_key, f"Kategoria_{idx}"
                    )
                    class_ranking.append(
                        {
                            "class_id": idx,
                            "class_name": class_name_for_idx,
                            "confidence": float(prob),
                        }
                    )

                # Sortowanie malejąco według pewności
                class_ranking.sort(key=lambda x: x["confidence"], reverse=True)
                result["class_ranking"] = class_ranking

            return result

        except Exception as e:
            error_msg = f"Błąd w ai.classifier.predict: {str(e)}"
            raise ValueError(error_msg)

    def batch_predict(self, image_paths, batch_size=16, return_ranking=False):
        """Przewidywanie kategorii dla wielu obrazów w trybie wsadowym.

        Args:
            image_paths: Lista ścieżek do obrazów
            batch_size: Rozmiar batcha do przetwarzania
            return_ranking: Czy zwrócić pełny ranking wszystkich klas (domyślnie False)

        Returns:
            Lista słowników z wynikami klasyfikacji
        """
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

                # Pobierz nazwę klasy z mapowania
                class_key = str(predicted_class)
                class_name = self.class_names.get(class_key)

                if class_name is None:
                    print(
                        f"UWAGA: Nie znaleziono nazwy dla klasy {predicted_class} "
                        f"w słowniku class_names. Używam domyślnej nazwy."
                    )
                    class_name = f"Kategoria_{predicted_class}"

                # Dodaj logi diagnostyczne dla pierwszych kilku wyników
                if j < 3:  # Pokaż tylko pierwsze 3 wyniki dla przejrzystości
                    print(
                        f"DEBUG: batch_predict() - j={j}, "
                        f"predicted_class={predicted_class}, "
                        f"class_name={class_name}"
                    )

                # Przygotuj podstawowy wynik
                result = {
                    "class_id": predicted_class,
                    "class_name": class_name,
                    "confidence": confidence,
                }

                # Jeśli potrzebny jest ranking, dodaj go
                if return_ranking:
                    all_probabilities = probabilities[j].cpu().numpy()
                    class_ranking = []

                    for idx, prob in enumerate(all_probabilities):
                        class_key = str(idx)
                        class_name_for_idx = self.class_names.get(class_key)

                        if class_name_for_idx is None:
                            print(
                                f"UWAGA: Nie znaleziono nazwy dla klasy {idx} "
                                f"w słowniku class_names. Używam domyślnej nazwy."
                            )
                            class_name_for_idx = f"Kategoria_{idx}"

                        class_ranking.append(
                            {
                                "class_id": idx,
                                "class_name": class_name_for_idx,
                                "confidence": float(prob),
                            }
                        )

                    # Sortowanie malejąco według pewności
                    class_ranking.sort(key=lambda x: x["confidence"], reverse=True)

                    # Dodaj ranking do wyniku
                    result["class_ranking"] = class_ranking

                results.append(result)

            # Wyczyść pamięć GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results

    def batch_predict_threaded(self, image_paths, num_threads=4, return_ranking=False):
        """
        Przewidywanie kategorii dla wielu obrazów z wykorzystaniem wielu wątków.

        Args:
            image_paths: Lista ścieżek do obrazów
            num_threads: Liczba wątków do wykorzystania
            return_ranking: Czy zwrócić pełny ranking wszystkich klas (domyślnie False)

        Returns:
            Lista wyników klasyfikacji
        """
        from concurrent.futures import ThreadPoolExecutor

        results = []

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(self.predict, path, return_ranking)
                for path in image_paths
            ]
            results = [future.result() for future in futures]

        return results

    def save(self, save_path, metadata=None):
        """Zapisywanie modelu z metadanymi"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Przygotuj podstawowy słownik checkpoint
        checkpoint = {
            "model_type": self.model_type,
            "num_classes": self.num_classes,
            "model_state_dict": self.model.state_dict(),
            "class_names": self.class_names,
        }

        # Obsługa metadanych
        if metadata:
            # Kopiuj metadane, aby uniknąć modyfikacji oryginału
            checkpoint["metadata"] = metadata.copy()

        # Zapisz model
        torch.save(checkpoint, save_path)

        # Zapisz też konfigurację w formacie JSON dla łatwiejszego odczytu
        config_path = os.path.splitext(save_path)[0] + "_config.json"
        config = {
            "model_type": self.model_type,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
        }

        # Dodaj metadane do konfiguracji
        if metadata:
            config["metadata"] = metadata.copy()

        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

        return save_path, config_path

    def save_with_original_config(self, save_path, original_config, metadata=None):
        """
        Zapisywanie modelu z zachowaniem KOMPLETNEJ oryginalnej struktury pliku config
        i dodaniem nowych metadanych.

        Args:
            save_path: Ścieżka zapisu modelu
            original_config: Oryginalny plik konfiguracyjny do rozszerzenia
            metadata: Nowe metadane do dodania

        Returns:
            Tuple: (ścieżka_modelu, ścieżka_konfiguracji)
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Przygotuj podstawowy słownik checkpoint
        checkpoint = {
            "model_type": self.model_type,
            "num_classes": self.num_classes,
            "model_state_dict": self.model.state_dict(),
            "class_names": self.class_names,
        }

        # Dodajemy metadane do checkpointu
        if metadata:
            checkpoint["metadata"] = metadata

        # Zapisz model
        torch.save(checkpoint, save_path)

        # Przygotuj plik konfiguracyjny bazujący na oryginalnym
        config = original_config.copy()

        # NIE MODYFIKUJ głównych kluczy konfiguracji, tylko metadane wewnątrz
        if "metadata" not in config:
            config["metadata"] = {}

        if metadata:
            for key, value in metadata.items():
                config["metadata"][key] = value

        # Zapisz kompletny plik konfiguracyjny
        config_path = os.path.splitext(save_path)[0] + "_config.json"
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
            "input_size": self.input_size,
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

    def batch_predict_with_cache(
        self, image_paths, batch_size=16, use_cache=True, return_ranking=False
    ):
        """
        Przewidywanie kategorii dla wielu obrazów w trybie wsadowym z cache'owaniem.

        Args:
            image_paths: Lista ścieżek do obrazów
            batch_size: Rozmiar wsadu do przetwarzania
            use_cache: Czy używać cache'a dla wcześniej przetworzonych obrazów
            return_ranking: Czy zwrócić pełny ranking wszystkich klas (domyślnie False)

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
        batch_results = self.batch_predict(to_process, batch_size, return_ranking)

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
            free_memory = memory_info - torch.cuda.memory_allocated(0)
            memory_gb = free_memory / (1024**3)  # Konwersja na GB

            # Bardziej precyzyjne mapowanie pamięci GPU na rozmiar batcha
            if memory_gb > 16:
                return 128
            elif memory_gb > 12:
                return 96
            elif memory_gb > 8:
                return 64
            elif memory_gb > 6:
                return 48
            elif memory_gb > 4:
                return 32
            elif memory_gb > 2:
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
        Automatycznie wybiera optymalną architekturę modelu.
        Zgodnie z dokumentacją, zawsze zwraca efficientnet_b0 jako rekomendowaną architekturę.
        """
        return "efficientnet_b0"  # Zawsze zwracamy efficientnet_b0 jako optymalną architekturę

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

    def get_weights_path(self):
        """Zwraca zapamiętaną ścieżkę do pliku wag modelu."""
        return self.weights_path


# Funkcja główna (odczytująca JSON zadania):
def train_from_json_config(json_config_path):
    """
    Rozpoczyna trening na podstawie pliku JSON z konfiguracją zadania.
    Args:
        json_config_path: Ścieżka do pliku JSON z konfiguracją zadania
    """
    import json

    from ai.models import get_model
    from ai.optimized_training import train_model_optimized

    with open(json_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    # Wyodrębnij parametry z JSON
    training_config = config.get("config", {})
    model_config = training_config.get("model", {})
    # Pobierz input_size
    input_size = model_config.get("input_size", 224)
    if isinstance(input_size, int):
        input_size = (input_size, input_size)
    # Inne parametry...
    architecture = model_config.get("architecture", "EfficientNet")
    variant = model_config.get("variant", "EfficientNet-B0")
    num_classes = model_config.get("num_classes", 10)
    # ... pobieranie innych parametrów z JSONa ...
    # Tworzenie modelu z odpowiednim input_size
    model = get_model(
        model_arch=variant.replace("EfficientNet-", "").lower(),
        num_classes=num_classes,
        input_size=input_size,
    )
    # Przekaż input_size do funkcji trenującej
    result = train_model_optimized(
        model=model,
        train_dir=training_config.get("train_dir"),
        val_dir=training_config.get("val_dir"),
        input_size=input_size,
        # ... inne parametry ...
    )
    return result
