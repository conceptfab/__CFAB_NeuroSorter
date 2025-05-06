import logging
import os
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ai.classifier import ImageClassifier
from app.core.logger import Logger
from app.metadata.metadata_manager import MetadataManager

# Utworzenie katalogu na logi jeśli nie istnieje
os.makedirs("logs", exist_ok=True)

# Użyj głównego loggera
logger = Logger()


class ImageSorter:
    """Klasa do sortowania obrazów na podstawie klasyfikacji AI."""

    def __init__(self, classifier, copy_files=True):
        """
        Inicjalizacja sortera obrazów.

        Args:
            classifier: Instancja ImageClassifier
            copy_files: Czy kopiować pliki zamiast przenosić
        """
        self.classifier = classifier
        self.metadata_manager = MetadataManager()
        self.copy_files = copy_files
        self.uncategorized_dir = "__pliki_bez_kategorii"
        logger.info(
            f"Inicjalizacja ImageSorter - tryb: {'kopiowanie' if copy_files else 'przenoszenie'}"
        )

    def _process_image(
        self, image_path, output_dir, created_dirs, confidence_threshold=0.5
    ):
        """
        Przetwarza pojedynczy obraz - klasyfikuje i przenosi/kopiuje do odpowiedniego folderu.

        Args:
            image_path: Ścieżka do obrazu
            output_dir: Katalog wyjściowy
            created_dirs: Słownik z utworzonymi już katalogami
            confidence_threshold: Minimalny próg pewności klasyfikacji

        Returns:
            dict: Wynik przetwarzania z informacją o kategorii i statusie
        """
        result = {"status": "skipped", "category": None, "confidence": 0, "error": None}
        start_time = datetime.now()

        try:
            logger.info(f"Rozpoczęcie przetwarzania obrazu: {image_path}")
            logger.debug(f"Sprawdzanie istnienia pliku: {image_path}")

            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Plik nie istnieje: {image_path}")

            logger.debug(f"Sprawdzanie uprawnień do pliku: {image_path}")
            if not os.access(image_path, os.R_OK):
                raise PermissionError(f"Brak uprawnień do odczytu pliku: {image_path}")

            logger.debug(f"Próba klasyfikacji obrazu: {image_path}")
            # Klasyfikuj obraz
            classification = self.classifier.predict(image_path)
            logger.debug(f"Wynik klasyfikacji: {classification}")

            category = classification["class_name"]
            confidence = classification["confidence"]

            logger.info(f"Klasyfikacja: {category} (pewność: {confidence:.2f})")

            # Aktualizuj wynik
            result["category"] = category
            result["confidence"] = confidence

            # Sprawdź czy klasyfikacja jest wystarczająco pewna
            if confidence < confidence_threshold:
                logger.warning(
                    f"Pominięto obraz - zbyt niska pewność: {confidence:.2f} < {confidence_threshold}"
                )
                return result

            # Przygotuj ścieżkę docelową
            dest_dir = self._ensure_category_dir(category, output_dir, created_dirs)
            dest_path = self._get_unique_dest_path(dest_dir, image_path)

            logger.debug(f"Ścieżka docelowa: {dest_path}")

            # Dodaj metadane o kategorii do obrazu
            metadata = {
                "category": category,
                "confidence": confidence,
                "sorted_at": datetime.now().isoformat(),
            }
            self.metadata_manager.add_category_to_image(image_path, metadata)
            logger.debug("Dodano metadane do obrazu")

            # Kopiuj lub przenieś plik w zależności od ustawienia
            if self.copy_files:
                shutil.copy2(image_path, dest_path)
                logger.info(f"Skopiowano plik do: {dest_path}")
            else:
                shutil.move(image_path, dest_path)
                logger.info(f"Przeniesiono plik do: {dest_path}")

            result["status"] = "processed"

            processing_time = (datetime.now() - start_time).total_seconds()
            logger.debug(f"Czas przetwarzania: {processing_time:.2f}s")

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            logger.error(
                f"Błąd podczas przetwarzania obrazu {image_path}: {str(e)}",
                exc_info=True,
            )

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

    def sort_directory(
        self,
        input_dir: str,
        output_dir: str,
        confidence_threshold: float = 0.5,
        callback: Optional[callable] = None,
    ) -> Dict:
        """Sortuje pliki w katalogu na podstawie klasyfikacji."""
        try:
            start_time = datetime.now()
            logger.info(f"Rozpoczęcie sortowania katalogu: {input_dir}")
            logger.info(
                f"Parametry: output_dir={output_dir}, confidence_threshold={confidence_threshold}"
            )

            # Jeśli nie podano katalogu wyjściowego, użyj wejściowego
            if output_dir is None:
                output_dir = input_dir
                logger.info("Użyto katalogu wejściowego jako wyjściowego")

            # Upewniamy się, że mamy różne katalogi wejściowy i wyjściowy jeśli przenosimy pliki
            if not self.copy_files and input_dir == output_dir:
                error_msg = "Katalog wejściowy i wyjściowy muszą być różne przy przenoszeniu plików"
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Utwórz katalog wyjściowy jeśli nie istnieje
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Utworzono katalog wyjściowy: {output_dir}")

            # Utwórz katalog dla plików bez kategorii
            uncategorized_path = os.path.join(output_dir, self.uncategorized_dir)
            os.makedirs(uncategorized_path, exist_ok=True)
            logger.info(
                f"Utworzono katalog dla plików bez kategorii: {uncategorized_path}"
            )

            # Znajdź wszystkie pliki obrazów w katalogu wejściowym
            image_files = []
            valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]

            logger.info("Skanowanie katalogu w poszukiwaniu obrazów...")
            for root, _, files in os.walk(input_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in valid_extensions):
                        image_files.append(os.path.join(root, file))

            total_files = len(image_files)
            logger.info(f"Znaleziono {total_files} plików obrazów")

            # Inicjalizacja statystyk
            stats = {
                "processed": 0,
                "moved": 0,
                "skipped": 0,
                "uncategorized": 0,
                "categories": {},
            }
            created_dirs = {}

            # Sortuj każdy obraz
            for i, image_path in enumerate(image_files):
                try:
                    stats["processed"] += 1
                    logger.info(
                        f"Przetwarzanie pliku {i+1}/{total_files}: {image_path}"
                    )

                    # Klasyfikacja obrazu
                    result = self.classifier.predict(image_path)

                    if result and result["confidence"] >= confidence_threshold:
                        category = result["class_name"]
                        target_dir = os.path.join(output_dir, category)
                        os.makedirs(target_dir, exist_ok=True)

                        # Kopiuj lub przenieś plik
                        target_path = os.path.join(
                            target_dir, os.path.basename(image_path)
                        )
                        if self.copy_files:
                            shutil.copy2(image_path, target_path)
                        else:
                            shutil.move(image_path, target_path)

                        stats["categories"][category] = (
                            stats["categories"].get(category, 0) + 1
                        )
                        stats["moved"] += 1
                        logger.info(f"Sukces - plik zaklasyfikowany jako: {category}")
                    else:
                        # Przenieś do katalogu plików bez kategorii
                        target_path = os.path.join(
                            uncategorized_path, os.path.basename(image_path)
                        )
                        if self.copy_files:
                            shutil.copy2(image_path, target_path)
                        else:
                            shutil.move(image_path, target_path)

                        stats["uncategorized"] += 1
                        logger.warning(f"Plik przeniesiony do kategorii bez kategorii")

                    # Wywołaj callback z postępem
                    if callback:
                        callback(i + 1, total_files)

                except Exception as e:
                    logger.error(
                        f"Nieoczekiwany błąd podczas sortowania obrazu {image_path}: {str(e)}",
                        exc_info=True,
                    )
                    stats["skipped"] += 1

            # Podsumowanie
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info("=== PODSUMOWANIE SORTOWANIA ===")
            logger.info(f"Czas wykonania: {processing_time:.2f}s")
            logger.info(f"Przetworzono plików: {stats['processed']}")
            logger.info(f"Przeniesiono/skopiowano: {stats['moved']}")
            logger.info(f"Pliki bez kategorii: {stats['uncategorized']}")
            logger.info(f"Pominięto: {stats['skipped']}")
            logger.info("Statystyki kategorii:")
            for category, count in stats["categories"].items():
                logger.info(f"  - {category}: {count} plików")

            return stats
        except Exception as e:
            error_msg = f"Błąd podczas sortowania: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def sort_images(
        self, image_paths, output_dir, confidence_threshold=0.5, callback=None
    ):
        """
        Sortuje wybrane obrazy do podfolderów według klas.

        Args:
            image_paths: Lista ścieżek do obrazów
            output_dir: Katalog wyjściowy
            confidence_threshold: Minimalny próg pewności klasyfikacji
            callback: Opcjonalna funkcja callback do aktualizacji postępu (current, total)

        Returns:
            dict: Statystyki sortowania
        """
        start_time = datetime.now()
        logger.info(f"Rozpoczęcie sortowania {len(image_paths)} wybranych obrazów")
        logger.info(
            f"Parametry: output_dir={output_dir}, confidence_threshold={confidence_threshold}"
        )

        # Utwórz katalog wyjściowy jeśli nie istnieje
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Utworzono katalog wyjściowy: {output_dir}")

        # Inicjalizacja statystyk
        stats = {"processed": 0, "moved": 0, "skipped": 0, "categories": {}}
        created_dirs = {}

        # Sortuj każdy obraz
        for i, image_path in enumerate(image_paths):
            try:
                stats["processed"] += 1
                logger.info(
                    f"Przetwarzanie pliku {i+1}/{len(image_paths)}: {image_path}"
                )

                # Przetwarzanie obrazu
                result = self._process_image(
                    image_path, output_dir, created_dirs, confidence_threshold
                )

                # Aktualizuj statystyki
                if result["status"] == "processed":
                    stats["categories"][result["category"]] = (
                        stats["categories"].get(result["category"], 0) + 1
                    )
                    stats["moved"] += 1
                    logger.info(
                        f"Sukces - plik zaklasyfikowany jako: {result['category']}"
                    )
                elif result["status"] == "skipped":
                    stats["skipped"] += 1
                    logger.warning(f"Pominięto plik - zbyt niska pewność klasyfikacji")
                elif result["status"] == "error":
                    logger.error(f"Błąd podczas przetwarzania pliku: {result['error']}")

                # Wywołaj callback z postępem
                if callback:
                    callback(i + 1, len(image_paths))

            except Exception as e:
                logger.error(
                    f"Nieoczekiwany błąd podczas sortowania obrazu {image_path}: {str(e)}",
                    exc_info=True,
                )
                stats["skipped"] += 1

        # Podsumowanie
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info("=== PODSUMOWANIE SORTOWANIA ===")
        logger.info(f"Czas wykonania: {processing_time:.2f}s")
        logger.info(f"Przetworzono plików: {stats['processed']}")
        logger.info(f"Przeniesiono/skopiowano: {stats['moved']}")
        logger.info(f"Pominięto: {stats['skipped']}")
        logger.info("Statystyki kategorii:")
        for category, count in stats["categories"].items():
            logger.info(f"  - {category}: {count} plików")

        return stats
