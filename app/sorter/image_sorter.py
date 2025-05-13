import json
import logging
import os
import shutil
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

from ai.classifier import ImageClassifier
from app.core.logger import Logger

# Utworzenie katalogu na logi jeśli nie istnieje

# Użyj głównego loggera
logger = Logger()


class ImageSorter:
    """Klasa do sortowania obrazów na podstawie klasyfikacji AI."""

    def __init__(
        self,
        model_path,
        output_directory=None,
        preserve_original_classes=True,
        logger=None,
    ):
        """
        Inicjalizacja sortera obrazów.

        Args:
            model_path: Ścieżka do pliku modelu
            output_directory: Katalog wyjściowy dla posortowanych obrazów
            preserve_original_classes: Czy zachować oryginalne klasy podczas sortowania
            logger: Opcjonalny logger do rejestrowania działań
        """
        self.logger = logger or self._setup_logger()
        self.logger.info(f"Inicjalizacja sortera z modelem: {model_path}")

        # Załaduj model
        self.model = ImageClassifier(weights_path=model_path)
        # Dodaj atrybut model_path do instancji ImageClassifier,
        # aby był dostępny np. dla batch_processor.py
        self.model.model_path = model_path

        # Zapisz oryginalne mapowanie klas przy inicjalizacji
        self.original_class_mapping = self.model.class_names.copy()
        log_msg = f"Załadowano model z {len(self.original_class_mapping)} klasami"
        self.logger.info(log_msg)

        # Zachowaj flagi konfiguracyjne
        self.preserve_original_classes = preserve_original_classes
        self.output_directory = output_directory
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)

        # Sprawdź, czy model ma włączoną ochronę przed zapominaniem
        self.has_forgetting_prevention = self._check_forgetting_prevention(model_path)
        if not self.has_forgetting_prevention and preserve_original_classes:
            log_warning = (
                "Model nie ma włączonych mechanizmów zapobiegających zapominaniu, "
                "ale flaga preserve_original_classes jest włączona. "
                "Może to prowadzić do nieprawidłowych klasyfikacji dla "
                "oryginalnych klas."
            )
            self.logger.warning(log_warning)

    def _check_forgetting_prevention(self, model_path):
        """
        Sprawdza, czy model ma włączone mechanizmy zapobiegające zapominaniu.

        Args:
            model_path: Ścieżka do pliku modelu

        Returns:
            bool: True jeśli model ma włączone mechanizmy zapobiegające zapominaniu
        """
        # Próba wczytania pliku konfiguracyjnego modelu
        config_path = os.path.splitext(model_path)[0] + "_config.json"
        if not os.path.exists(config_path):
            log_msg = f"Nie znaleziono pliku konfiguracyjnego: {config_path}"
            self.logger.warning(log_msg)
            return False

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            # Sprawdź, czy w konfiguracji jest sekcja zapobiegająca zapominaniu
            forgetting_prevention = (
                config.get("advanced", {})
                .get("catastrophic_forgetting_prevention", {})
                .get("enable", False)
            )

            if forgetting_prevention:
                self.logger.info(
                    "Model ma włączone mechanizmy zapobiegające zapominaniu"
                )
                return True
            else:
                log_msg = (
                    "Model nie ma włączonych mechanizmów zapobiegających " "zapominaniu"
                )
                self.logger.warning(log_msg)
                return False
        except Exception as e:
            log_msg = f"Błąd podczas sprawdzania konfiguracji modelu: {str(e)}"
            self.logger.error(log_msg)
            return False

    def evaluate_on_original_classes(self, test_dir, batch_size=16):
        """
        Wykonuje ewaluację modelu na katalogach z oryginalnymi klasami.

        Args:
            test_dir: Katalog zawierający podkatalogi dla każdej oryginalnej klasy
            batch_size: Rozmiar wsadu do przetwarzania

        Returns:
            dict: Wyniki ewaluacji dla każdej klasy
        """
        self.logger.info(
            f"Ewaluacja modelu na oryginalnych klasach w katalogu: {test_dir}"
        )

        # Przygotuj strukturę wyników
        results = {
            "overall": {"correct": 0, "total": 0, "accuracy": 0.0},
            "classes": {},
        }

        # Dla każdego katalogu (zakładamy, że nazwa katalogu to nazwa klasy)
        for class_name in os.listdir(test_dir):
            class_dir = os.path.join(test_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            # Znajdź wszystkie obrazy w katalogu klasy
            image_paths = []
            for root, _, files in os.walk(class_dir):
                for file in files:
                    if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        image_paths.append(os.path.join(root, file))

            if not image_paths:
                self.logger.warning(f"Brak obrazów dla klasy {class_name}")
                continue

            # Inicjalizuj wyniki dla tej klasy
            results["classes"][class_name] = {
                "correct": 0,
                "total": len(image_paths),
                "accuracy": 0.0,
            }

            # Klasyfikuj obrazy wsadowo
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i : i + batch_size]
                batch_results = self.model.batch_predict(batch_paths)

                # Sprawdź wyniki
                for result in batch_results:
                    predicted_class = result["class_name"]
                    results["overall"]["total"] += 1
                    results["classes"][class_name]["total"] += 1

                    if predicted_class.lower() == class_name.lower():
                        results["overall"]["correct"] += 1
                        results["classes"][class_name]["correct"] += 1

        # Oblicz dokładności
        for class_name, data in results["classes"].items():
            data["accuracy"] = (
                data["correct"] / data["total"] if data["total"] > 0 else 0
            )

        results["overall"]["accuracy"] = (
            results["overall"]["correct"] / results["overall"]["total"]
            if results["overall"]["total"] > 0
            else 0
        )

        # Wyświetl podsumowanie
        self.logger.info(f"Ogólna dokładność: {results['overall']['accuracy']:.2%}")
        for class_name, data in results["classes"].items():
            self.logger.info(
                f"  - {class_name}: {data['accuracy']:.2%} "
                f"({data['correct']}/{data['total']})"
            )

        return results

    def _setup_logger(self):
        """
        Tworzy i konfiguruje logger.

        Returns:
            logging.Logger: Skonfigurowany logger
        """
        logger = logging.getLogger("ImageSorter")
        logger.setLevel(logging.INFO)

        # Dodaj handler do konsoli, jeśli nie istnieje
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Przetwarza pojedynczy obraz.

        Args:
            image_path: Ścieżka do obrazu

        Returns:
            Dict[str, Any]: Słownik z wynikami przetwarzania
        """
        result = {
            "status": "error",
            "message": "",
            "image_path": image_path,
            "category": None,
            "confidence": None,
        }

        try:
            # Sprawdź czy plik istnieje i ma uprawnienia do odczytu
            if not os.path.exists(image_path):
                result["message"] = f"Plik nie istnieje: {image_path}"
                return result

            if not os.access(image_path, os.R_OK):
                result["message"] = f"Brak uprawnień do odczytu pliku: {image_path}"
                return result

            # Klasyfikuj obraz
            logger.info(f"Klasyfikacja obrazu: {image_path}")
            classification_result = self.model.predict(image_path)

            # Sprawdź czy klasyfikacja się powiodła
            if classification_result is None:
                result["message"] = "Nie można sklasyfikować obrazu."
                logger.info("Nie można sklasyfikować obrazu.")
                return result

            # Sprawdź czy wynik zawiera wymagane pola
            if not all(
                key in classification_result for key in ["class_name", "confidence"]
            ):
                result["message"] = "Nieprawidłowy format wyniku klasyfikacji"
                logger.warning(f"Nieprawidłowy format wyniku: {classification_result}")
                return result

            # Pobierz mapowanie klas
            class_mapping = {}
            if hasattr(self.model, "get_class_mapping"):
                class_mapping = self.model.get_class_mapping() or {}

            # Aktualizuj wynik - bez sprawdzania czy kategoria jest w mapowaniu
            category = classification_result["class_name"]
            result.update(
                {
                    "status": "success",
                    "category": category,
                    "confidence": classification_result["confidence"],
                }
            )

            logger.info(
                f"Obraz sklasyfikowany jako {category} "
                f"(pewność: {classification_result['confidence']:.2f})"
            )

        except Exception as e:
            result["message"] = f"Błąd klasyfikacji: {str(e)}"
            logger.error(f"Błąd podczas przetwarzania obrazu {image_path}: {e}")
            logger.error(traceback.format_exc())

        return result

    def _ensure_category_dir(self, category, output_dir, created_dirs):
        """
        Upewnia się, że katalog dla kategorii istnieje i zwraca jego ścieżkę.

        Args:
            category: Nazwa kategorii
            output_dir: Katalog wyjściowy
            created_dirs: Słownik z utworzonymi już katalogami

        Returns:
            str: Ścieżka do katalogu dla tej kategorii
        """
        # Dodaj zabezpieczenie przed None
        if category is None:
            category = "nieskategoryzowane"
            logger.warning(f"Wykryto kategorię None, używam domyślnej: {category}")

        # Jeśli katalog był już tworzony, zwróć go bezpośrednio z pamięci podręcznej
        if category in created_dirs:
            logger.debug(f"Użyto istniejącego katalogu dla kategorii: {category}")
            return created_dirs[category]

        # Zabezpiecz nazwę kategorii, aby była poprawną nazwą katalogu
        safe_category = self._sanitize_filename(category)
        logger.debug(f"Zabezpieczona nazwa kategorii: {safe_category}")

        # Utwórz katalog kategorii
        dest_dir = os.path.join(output_dir, safe_category)
        os.makedirs(dest_dir, exist_ok=True)
        logger.info(f"Utworzono katalog dla kategorii: {dest_dir}")

        # Zapisz w pamięci podręcznej
        created_dirs[category] = dest_dir
        return dest_dir

    def _get_unique_dest_path(self, dest_dir, image_path):
        """
        Tworzy unikalną ścieżkę docelową dla pliku, unikając nadpisywania istniejących plików.

        Args:
            dest_dir: Katalog docelowy
            image_path: Oryginalna ścieżka do obrazu

        Returns:
            str: Unikalna ścieżka docelowa
        """
        # Przygotuj ścieżkę docelową
        dest_path = os.path.join(dest_dir, os.path.basename(image_path))
        logger.debug(f"Początkowa ścieżka docelowa: {dest_path}")

        # Obsługa duplikatów nazw plików
        if os.path.exists(dest_path):
            base_name, extension = os.path.splitext(os.path.basename(image_path))
            counter = 1
            while os.path.exists(dest_path):
                dest_path = os.path.join(dest_dir, f"{base_name}_{counter}{extension}")
                counter += 1
            logger.info(f"Znaleziono duplikat - utworzono nową ścieżkę: {dest_path}")

        return dest_path

    def _sanitize_filename(self, filename):
        """
        Usuwa niedozwolone znaki z nazwy pliku/katalogu.

        Args:
            filename: Nazwa pliku/katalogu do oczyszczenia

        Returns:
            str: Oczyszczona nazwa pliku/katalogu
        """
        # Znaki niedozwolone w nazwach plików w większości systemów
        illegal_chars = r'<>:"/\|?*'
        original = filename

        # Usuń wszystkie niedozwolone znaki
        sanitized = filename
        for char in illegal_chars:
            sanitized = sanitized.replace(char, "_")

        # Usuń wiodące i końcowe spacje/kropki
        sanitized = sanitized.strip(". ")

        # Jeśli nazwa jest pusta po sanityzacji, użyj wartości domyślnej
        if not sanitized or all(c in ". " for c in sanitized):
            sanitized = "nieskategoryzowane"
            logger.warning(
                f"Nieprawidłowa nazwa kategorii '{original}' - użyto domyślnej: {sanitized}"
            )

        if sanitized != original:
            logger.debug(f"Zabezpieczono nazwę: '{original}' -> '{sanitized}'")

        return sanitized

    def stop(self):
        """Żąda przerwania procesu sortowania."""
        self._stop_requested = True
        logger.info("Otrzymano żądanie przerwania sortowania")

    def sort_directory(
        self,
        input_dir: str,
        output_dir: str,
        min_confidence_threshold: float = 0.5,
        max_confidence_threshold: float = 1.0,
        callback: Optional[callable] = None,
        selected_classes: Optional[List[str]] = None,
        category_callback: Optional[callable] = None,
    ) -> Dict:
        """Sortuje pliki w katalogu na podstawie klasyfikacji.

        Args:
            input_dir: Katalog źródłowy
            output_dir: Katalog docelowy
            min_confidence_threshold: Minimalny próg pewności klasyfikacji
            max_confidence_threshold: Maksymalny próg pewności klasyfikacji
            callback: Funkcja wywoływana po każdym przetworzonym pliku
            selected_classes: Lista wybranych klas do sortowania (None = wszystkie klasy)
            category_callback: Funkcja wywoływana po przetworzeniu pliku dla kategorii
        """
        try:
            self._stop_requested = False  # Reset flagi przerwania
            start_time = datetime.now()
            logger.info(f"Rozpoczęcie sortowania katalogu: {input_dir}")
            logger.info(
                f"Parametry: output_dir={output_dir}, "
                f"min_confidence_threshold={min_confidence_threshold}, "
                f"max_confidence_threshold={max_confidence_threshold}"
            )
            if selected_classes:
                logger.info(f"Wybrane klasy: {', '.join(selected_classes)}")

            # Jeśli nie podano katalogu wyjściowego, użyj wejściowego
            if output_dir is None:
                output_dir = input_dir
                logger.info("Użyto katalogu wejściowego jako wyjściowego")

            # Upewniamy się, że mamy różne katalogi wejściowy i wyjściowy jeśli przenosimy pliki
            if not self.preserve_original_classes and input_dir == output_dir:
                error_msg = "Katalog wejściowy i docelowy muszą być różne przy przenoszeniu plików"
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Utwórz katalog wyjściowy jeśli nie istnieje
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Utworzono katalog wyjściowy: {output_dir}")

            # Znajdź wszystkie pliki obrazów w katalogu wejściowym
            image_files = []
            valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]

            logger.info("Skanowanie katalogu w poszukiwaniu obrazów...")
            for root, _, files in os.walk(input_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in valid_extensions):
                        image_files.append(os.path.join(root, file))

            total_files = len(image_files)
            logger.info(f"Znaleziono {total_files} plików do przetworzenia")

            # Statystyki
            stats = {
                "processed": 0,
                "moved": 0,
                "skipped": 0,
                "skipped_confidence": 0,  # Nowy licznik dla pominiętych z powodu progu pewności
                "categories": {},
            }

            # Sortuj każdy obraz
            for i, image_path in enumerate(image_files):
                if self._stop_requested:  # Sprawdź czy nie ma żądania przerwania
                    logger.info("Przerwano sortowanie podczas przetwarzania plików")
                    return stats

                try:
                    stats["processed"] += 1
                    logger.info(
                        f"Przetwarzanie pliku {i+1}/{total_files}: {image_path}"
                    )

                    # Klasyfikacja obrazu
                    result = self._process_image(image_path)

                    if result["status"] == "success":
                        category = result["category"]
                        confidence = result["confidence"]

                        logger.info(
                            f"DEBUG sort_directory: Plik: {image_path}, "
                            f"Kategoria z _process_image: '{category}' "
                            f"(typ: {type(category)}), Pewność: {confidence}"
                        )
                        logger.info(
                            f"DEBUG sort_directory: Wybrane klasy (selected_classes): "
                            f"{selected_classes} (typ pierwszego elementu, jeśli są: "
                            f"{type(selected_classes[0]) if selected_classes and selected_classes[0] is not None else 'N/A'})"
                        )

                        # Sprawdź próg pewności
                        if confidence is None or not (
                            min_confidence_threshold
                            <= confidence
                            <= max_confidence_threshold
                        ):
                            logger.info(
                                f"Pominięto plik - pewność {confidence} "
                                f"poza zakresem [{min_confidence_threshold} - {max_confidence_threshold}]"
                            )
                            stats["skipped"] += 1
                            stats[
                                "skipped_confidence"
                            ] += 1  # Zwiększ licznik pominiętych z powodu progu
                            continue

                        # Sprawdź czy kategoria jest na liście wybranych klas
                        if selected_classes:
                            logger.info(
                                f"DEBUG sort_directory: Sprawdzanie czy '{category}' "
                                f"jest w {selected_classes}"
                            )
                            if category not in selected_classes:
                                logger.info(
                                    f"Pominięto plik - kategoria '{category}' "
                                    f"nie jest wybrana w {selected_classes}"
                                )
                                stats["skipped"] += 1
                                continue
                            else:
                                logger.info(
                                    f"DEBUG sort_directory: Kategoria '{category}' "
                                    f"ZNALEZIONA w selected_classes."
                                )
                        else:  # Jeśli selected_classes jest None lub puste, nie filtrujemy
                            logger.info(
                                "DEBUG sort_directory: Brak wybranych klas "
                                "(selected_classes is None/empty), "
                                "nie filtruję po klasie."
                            )

                        target_dir = os.path.join(output_dir, category)
                        os.makedirs(target_dir, exist_ok=True)

                        # Kopiuj lub przenieś plik
                        target_path = self._get_unique_dest_path(target_dir, image_path)
                        if self.preserve_original_classes:
                            shutil.copy2(image_path, target_path)
                        else:
                            shutil.move(image_path, target_path)

                        stats["categories"][category] = (
                            stats["categories"].get(category, 0) + 1
                        )
                        stats["moved"] += 1
                        logger.info(f"Sukces - plik zaklasyfikowany jako: {category}")

                        # Wywołaj category_callback jeśli podano
                        if category_callback:
                            try:
                                category_callback(
                                    category,
                                    stats["categories"][category],
                                    "Przetworzono",
                                )
                            except Exception as e_cb:
                                logger.error(
                                    f"Błąd w category_callback dla {category}: {e_cb}"
                                )
                    else:
                        logger.warning(
                            f"Nie udało się sklasyfikować pliku: {image_path}"
                        )
                        stats["skipped"] += 1

                    # Wywołaj callback jeśli podano
                    if callback:
                        callback(i + 1, total_files)

                except Exception as e:
                    logger.error(
                        f"Błąd podczas przetwarzania pliku {image_path}: {str(e)}"
                    )
                    stats["skipped"] += 1

            # Podsumowanie
            end_time = datetime.now()
            duration = end_time - start_time
            logger.info(
                f"Sortowanie zakończone w {duration.total_seconds():.2f} sekund"
            )
            logger.info(
                f"Przetworzono: {stats['processed']}, "
                f"Przeniesiono: {stats['moved']}, "
                f"Pominięto: {stats['skipped']}, "
                f"Pominięto (pewność): {stats['skipped_confidence']}"
            )
            for category, count in stats["categories"].items():
                logger.info(f"Kategoria {category}: {count} plików")

            return stats

        except Exception as e:
            logger.error(f"Błąd podczas sortowania: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def sort_images(
        self,
        input_directory,
        batch_size=16,
        min_confidence_threshold=0.0,
        max_confidence_threshold=1.0,
        use_uncategorized_folder: bool = True,
        callback: Optional[callable] = None,
        category_callback: Optional[callable] = None,
    ):
        """
        Sortuje obrazy z katalogu wejściowego do podkatalogów na podstawie klasyfikacji.

        Args:
            input_directory: Katalog z obrazami do sortowania.
            batch_size: Rozmiar wsadu do przetwarzania obrazów.
            min_confidence_threshold: Minimalna pewność klasyfikacji, aby przenieść obraz.
            max_confidence_threshold: Maksymalna pewność klasyfikacji, aby przenieść obraz.
            use_uncategorized_folder: Czy użyć katalogu nieskategoryzowanych obrazów.
            callback: Opcjonalna funkcja zwrotna do raportowania postępu (current, total).
            category_callback: Opcjonalna funkcja zwrotna do raportowania postępu kategorii.

        Returns:
            dict: Statystyki sortowania.
        """
        self.logger.info(
            f"Rozpoczęcie sortowania obrazów w katalogu: {input_directory}"
        )
        if not self.output_directory:
            self.logger.error(
                "Katalog wyjściowy (output_directory) nie jest ustawiony."
            )
            raise ValueError("Katalog wyjściowy (output_directory) nie jest ustawiony.")

        self._stop_requested = False  # Zainicjalizuj flagę stopu dla tej operacji
        stats = {
            "total_processed": 0,
            "total_moved_or_copied": 0,
            "total_skipped_confidence": 0,
            "total_skipped_errors": 0,
            "categories": {},
        }
        created_dirs = {}

        image_paths = []
        for root, _, files in os.walk(input_directory):
            if self._stop_requested:  # Sprawdzaj _stop_requested
                self.logger.info("Sortowanie przerwane podczas zbierania plików.")
                return stats
            for file in files:
                if self._stop_requested:  # Sprawdzaj _stop_requested
                    self.logger.info("Sortowanie przerwane podczas zbierania plików.")
                    return stats
                if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    image_paths.append(os.path.join(root, file))

        total_images = len(image_paths)
        self.logger.info(f"Znaleziono {total_images} obrazów do przetworzenia.")
        if total_images == 0:
            self.logger.info("Brak obrazów do przetworzenia.")
            return stats

        for i in range(0, total_images, batch_size):
            if self._stop_requested:  # Sprawdzaj _stop_requested
                self.logger.info(
                    "Sortowanie przerwane przed przetworzeniem kolejnego wsadu."
                )
                return stats

            batch_paths = image_paths[i : i + batch_size]
            current_batch_size = len(batch_paths)
            self.logger.debug(
                f"Przetwarzanie wsadu {i//batch_size + 1}/{(total_images + batch_size - 1)//batch_size} z {current_batch_size} obrazami."
            )

            try:
                batch_results = self.model.batch_predict(batch_paths)
            except Exception as e:
                self.logger.error(f"Błąd podczas batch_predict dla wsadu: {e}")
                self.logger.error(traceback.format_exc())
                stats["total_skipped_errors"] += current_batch_size
                stats["total_processed"] += current_batch_size
                if callback:
                    try:
                        callback(stats["total_processed"], total_images)
                    except Exception as cb_exc:
                        self.logger.error(
                            f"Błąd w callbacku po błędzie wsadu: {cb_exc}"
                        )
                continue

            for idx_in_batch, (image_path, result) in enumerate(
                zip(batch_paths, batch_results)
            ):
                if self._stop_requested:  # Sprawdzaj _stop_requested
                    self.logger.info(
                        "Sortowanie przerwane podczas przetwarzania wyników wsadu."
                    )

                    stats["total_processed"] += idx_in_batch
                    if callback and stats["total_processed"] <= total_images:
                        try:
                            callback(stats["total_processed"], total_images)
                        except Exception as cb_exc:
                            self.logger.error(
                                f"Błąd w callbacku przy przerwaniu: {cb_exc}"
                            )
                    return stats

                stats["total_processed"] += 1

                category = result.get("class_name")
                confidence = result.get("confidence")

                if category is None or confidence is None:
                    self.logger.warning(
                        f"Pominięto {image_path} - brak kategorii lub pewności w wyniku: {result}"
                    )
                    stats["total_skipped_errors"] += 1
                    if callback:
                        try:
                            callback(stats["total_processed"], total_images)
                        except Exception as cb_exc:
                            self.logger.error(
                                f"Błąd w callbacku po pominięciu (brak danych): {cb_exc}"
                            )
                    continue

                if not (
                    min_confidence_threshold <= confidence <= max_confidence_threshold
                ):
                    self.logger.info(
                        f"Pominięto {image_path} (pewność: {confidence:.2f} poza zakresem "
                        f"[{min_confidence_threshold:.2f} - {max_confidence_threshold:.2f}])"
                    )
                    stats["total_skipped_confidence"] += 1
                else:
                    dest_category_dir = self._ensure_category_dir(
                        category, self.output_directory, created_dirs
                    )
                    unique_dest_path = self._get_unique_dest_path(
                        dest_category_dir, image_path
                    )

                    try:
                        if self.preserve_original_classes:
                            shutil.copy2(image_path, unique_dest_path)
                            self.logger.info(
                                f"Skopiowano: {image_path} -> {unique_dest_path}"
                            )
                        else:
                            shutil.move(image_path, unique_dest_path)
                            self.logger.info(
                                f"Przeniesiono: {image_path} -> {unique_dest_path}"
                            )

                        stats["total_moved_or_copied"] += 1
                        if category not in stats["categories"]:
                            stats["categories"][category] = 0
                        stats["categories"][category] += 1

                        # Wywołaj category_callback jeśli podano
                        if category_callback:
                            try:
                                category_callback(
                                    category,
                                    stats["categories"][category],
                                    "Przetworzono",
                                )
                            except Exception as e_cb:
                                logger.error(
                                    f"Błąd w category_callback dla {category}: {e_cb}"
                                )

                    except Exception as e:
                        self.logger.error(
                            f"Błąd podczas przenoszenia/kopiowania {image_path} do {unique_dest_path}: {e}"
                        )
                        self.logger.error(traceback.format_exc())
                        stats["total_skipped_errors"] += 1

                if callback:
                    try:
                        callback(stats["total_processed"], total_images)
                    except Exception as cb_exc:
                        self.logger.error(
                            f"Błąd w callbacku po przetworzeniu obrazu: {cb_exc}"
                        )

        self.logger.info(
            f"Sortowanie zakończone. Przetworzono: {stats['total_processed']}, "
            f"Przeniesiono/skopiowano: {stats['total_moved_or_copied']}, "
            f"Pominięto (pewność): {stats['total_skipped_confidence']}, "
            f"Pominięto (błędy): {stats['total_skipped_errors']}"
        )
        # self._is_running = False # Usunięto - niepotrzebne
        return stats

    def get_available_classes(self):
        """
        Zwraca dostępne klasy z modelu.

        Returns:
            dict: Mapowanie id -> nazwa klasy
        """
        return self.model.class_names.copy()

    def _get_category_names(self):
        """Zwraca listę wszystkich dostępnych nazw kategorii z modelu."""
        class_mapping = self.model.get_class_mapping()
        if not class_mapping:
            return []

        # Zwróć unikalne nazwy kategorii
        return list(set(class_mapping.values()))
