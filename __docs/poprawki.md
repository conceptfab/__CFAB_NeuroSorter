Analiza kodu
Przeanalizowałem dostarczony kod i zidentyfikowałem kilka miejsc, które można zoptymalizować, poprawić lub usunąć nadmiarowy kod. Przedstawiam propozycje zmian w formie poszczególnych plików.
Zmiany w app/core/logger.py
python# Zmiana w metodzie error
def error(
    self,
    message: str,
    extra: dict = None,
    func_name: str = None,
    file_name: str = None,
):
    """Loguje błąd wraz z informacją o funkcji i pliku."""
    location_info = ""
    if func_name and file_name:
        location_info = f"[{file_name}::{func_name}] "
    elif func_name:
        location_info = f"[{func_name}] "
    elif file_name:
        location_info = f"[{file_name}] "

    log_message = f"{location_info}{message}"
    if extra:
        log_message = f"{log_message} - {extra}"
    self.logger.error(log_message)
Ta metoda jest dobrze zaimplementowana. Nie wymaga zmian.
Zmiany w app/core/notifications.py
python# Optymalizacja metody get_notification_count
def get_notification_count(self, type: Optional[NotificationType] = None) -> int:
    """Zwraca liczbę powiadomień.

    Args:
        type: Filtr po typie powiadomienia

    Returns:
        Liczba powiadomień
    """
    if type:
        return sum(1 for n in self._notifications if n.type == type)
    return len(self._notifications)
Zamiast tworzyć listę za pomocą list comprehension i potem liczyć jej długość, używam sum z generatorem, co jest bardziej wydajne, szczególnie dla dużej liczby powiadomień.
Zmiany w app/core/optimizations.py
python# Optymalizacja metody get_oldest w klasie CacheManager
def get_oldest(self) -> Optional[Tuple[str, Any]]:
    """Zwraca najstarszy element z cache.

    Returns:
        Krotka (klucz, wartość) lub None
    """
    if not self._cache:
        return None

    oldest_key = min(self._access_times, key=self._access_times.get)
    return oldest_key, self._cache[oldest_key]
Zastąpiłem min(self._access_times.items(), key=lambda x: x[1])[0] na min(self._access_times, key=self._access_times.get), co jest bardziej wydajne i czytelne.
python# Podobna optymalizacja dla metody get_newest
def get_newest(self) -> Optional[Tuple[str, Any]]:
    """Zwraca najnowszy element z cache.

    Returns:
        Krotka (klucz, wartość) lub None
    """
    if not self._cache:
        return None

    newest_key = max(self._access_times, key=self._access_times.get)
    return newest_key, self._cache[newest_key]
Zmiany w app/core/state_manager.py
python# Optymalizacja metody save_state
def save_state(self):
    """Zapisuje stan do pliku."""
    try:
        # Utwórz kopię zapasową
        if os.path.exists(self.state_file):
            backup_file = f"{self.state_file}.bak"
            os.replace(self.state_file, backup_file)

        # Zapisz nowy stan
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self._state, f, indent=2, ensure_ascii=False)

        # Usuń kopię zapasową tylko jeśli zapis się powiódł
        if os.path.exists(f"{self.state_file}.bak"):
            os.remove(f"{self.state_file}.bak")

        self._last_save = datetime.now()

    except Exception as e:
        # Przywróć kopię zapasową w przypadku błędu
        if os.path.exists(f"{self.state_file}.bak"):
            os.replace(f"{self.state_file}.bak", self.state_file)
        raise e
Nie ma tu dużo do optymalizacji, ale zmiana kolejności usuwania kopii zapasowej jest bezpieczniejsza - usuwamy ją tylko jeśli zapis nowego stanu się powiódł.
Zmiany w app/core/workers/batch_classification_thread.py
python# Dodanie typowania do metod klasy BatchClassificationThread
def stop(self) -> None:
    """Zatrzymuje wątek."""
    self.is_stopped = True

def pause(self) -> None:
    """Wstrzymuje wątek."""
    self.is_paused = True

def resume(self) -> None:
    """Wznawia wątek."""
    self.is_paused = False
Dodałem typowanie, aby kod był bardziej przejrzysty i zgodny z resztą aplikacji.
Zmiany w app/core/workers/single_training_thread.py
python# Optymalizacja w _save_training_time
def _save_training_time(
    self, task_path: str, training_time: float, result: dict = None
):
    """Zapisuje czas treningu, dokładność walidacyjną i informacje o epokach w pliku JSON z zadaniem."""
    try:
        self.logger.info(f"Zapisywanie wyników treningu do pliku: {task_path}")

        if not os.path.exists(task_path):
            self.logger.error(f"Plik zadania nie istnieje: {task_path}")
            return

        with open(task_path, "r", encoding="utf-8") as f:
            task_data = json.load(f)
            self.logger.info(
                f"Wczytano dane zadania: {json.dumps(task_data, indent=2)}"
            )

        # Dodaj czas treningu do danych zadania
        task_data["training_time"] = round(training_time, 2)
        import datetime as _dt

        td = _dt.timedelta(seconds=int(training_time))
        task_data["training_time_str"] = str(td)

        # Ustaw status zadania na Zakończony
        task_data["status"] = "Zakończony"
        self.logger.info(f"Zmieniono status zadania na: {task_data['status']}")

        # Dodaj dokładność walidacyjną jeśli jest dostępna
        if result and "history" in result:
            history = result["history"]
            for key in ["train_acc", "train_loss", "val_acc", "val_loss"]:
                if key in history and history[key]:
                    task_data[key.replace("_", "_")] = history[key][-1]

        # Dodaj informacje o epokach
        if result and "epoch" in result and "total_epochs" in result:
            task_data["completed_epochs"] = result["epoch"]
            task_data["total_epochs"] = result["total_epochs"]

        # Zapisz zaktualizowane dane z powrotem do pliku
        self.logger.info(
            f"Zapisuję zaktualizowane dane zadania: {json.dumps(task_data, indent=2)}"
        )
        with open(task_path, "w", encoding="utf-8") as f:
            json.dump(task_data, f, indent=4, ensure_ascii=False)

        self.logger.info("Pomyślnie zapisano wyniki treningu do pliku")

    except Exception as e:
        self.logger.error(f"Błąd podczas zapisywania wyników treningu: {str(e)}")
        self.logger.error(f"TRACEBACK: {traceback.format_exc()}")
        raise
Użyłem pętli do iteracji po kluczach historii zamiast powtarzania kodu.
Zmiany w app/database/db_manager.py
python# Optymalizacja importu datetime
import os
import sqlite3
from datetime import datetime
Ten import jest w porządku.
Zmiany w app/metadata/metadata_manager.py
python# Uproszczenie kodu w add_category_to_image
def add_category_to_image(self, image_path, category_data):
    """
    Dodaje informację o kategorii do metadanych obrazu.

    Args:
        image_path: Ścieżka do pliku obrazu
        category_data: Nazwa kategorii jako string lub słownik z metadanymi kategorii

    Returns:
        bool: True jeśli operacja się powiodła, False w przeciwnym wypadku
    """
    try:
        # Konwersja category_data na format JSON, jeśli to słownik
        category_value = (
            json.dumps(category_data) if isinstance(category_data, dict) else str(category_data)
        )

        # Sprawdź format pliku
        _, ext = os.path.splitext(image_path.lower())

        if ext in [".jpg", ".jpeg"]:
            # Dla plików JPEG używamy biblioteki piexif
            try:
                # Wczytaj istniejące dane EXIF
                exif_dict = piexif.load(image_path)
            except Exception as e:
                print(f"Nie można wczytać danych EXIF z {image_path}: {str(e)}")
                # Jeśli nie ma danych EXIF, utwórz pusty słownik
                exif_dict = {
                    "0th": {},
                    "Exif": {},
                    "GPS": {},
                    "1st": {},
                    "thumbnail": None,
                }

            # Dodaj kategorię do pola UserComment w sekcji Exif
            category_bytes = category_value.encode("utf-8")
            exif_dict["Exif"][piexif.ExifIFD.UserComment] = category_bytes

            # Zapisz zmodyfikowane dane EXIF
            exif_bytes = piexif.dump(exif_dict)
            piexif.insert(exif_bytes, image_path)
            return True

        elif ext in [".png", ".tiff", ".tif", ".webp"]:
            # Dla innych obsługiwanych formatów używamy PIL
            try:
                img = Image.open(image_path)

                # Sprawdź czy obraz ma już metadane
                metadata = img.info or {}

                # Dodaj kategorię do metadanych
                metadata["Category"] = category_value

                # Zapisz obraz z nowymi metadanymi
                img.save(image_path, **metadata)
                return True
            except Exception as e:
                print(f"Nie można zapisać metadanych do {image_path}: {str(e)}")
                return False

        else:
            print(f"Nieobsługiwany format pliku: {ext}")
            return False

    except Exception as e:
        print(f"Błąd podczas dodawania metadanych do {image_path}: {str(e)}")
        return False
Uproszczono warunek sprawdzający typ category_data.
Zmiany w app/sorter/image_sorter.py
python# Optymalizacja metody sort_directory
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
            "categories": {},
            "skipped_confidence": 0,  # Dodano licznik dla pominiętych z powodu progu pewności
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
                        stats["skipped_confidence"] += 1
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
            f"Pominięto z powodu pewności: {stats['skipped_confidence']}"
        )
        for category, count in stats["categories"].items():
            logger.info(f"Kategoria {category}: {count} plików")

        return stats

    except Exception as e:
        logger.error(f"Błąd podczas sortowania: {str(e)}")
        logger.error(traceback.format_exc())
        raise
Dodano licznik dla pominiętych plików z powodu progu pewności, co jest bardziej informatywne w logach i statystykach.
Zmiany w app/utils/config.py
python# Optymalizacja funkcji load_default_settings
def load_default_settings():
    """Ładuje domyślne ustawienia aplikacji."""
    return {
        # Ustawienia ogólne
        "data_dir": "data",
        "models_dir": "data/models",
        "reports_dir": "data/reports",
        "log_level": "INFO",
        "log_file": "app.log",
        # Ustawienia modelu
        "confidence_threshold": 0.5,
        "use_gpu": True,
        "batch_size": DEFAULT_TRAINING_PARAMS["batch_size"],
        "num_workers": DEFAULT_TRAINING_PARAMS["num_workers"],
        # Ustawienia treningu
        "epochs": DEFAULT_TRAINING_PARAMS["max_epochs"],
        "train_batch_size": DEFAULT_TRAINING_PARAMS["batch_size"],
        "learning_rate": DEFAULT_TRAINING_PARAMS["learning_rate"],
        "optimizer": DEFAULT_TRAINING_PARAMS["optimizer"],
        "scheduler": DEFAULT_TRAINING_PARAMS["scheduler"],
        "weight_decay": DEFAULT_TRAINING_PARAMS["weight_decay"],
        "gradient_clip_val": DEFAULT_TRAINING_PARAMS["gradient_clip_val"],
        "early_stopping_patience": DEFAULT_TRAINING_PARAMS["early_stopping_patience"],
        "validation_split": DEFAULT_TRAINING_PARAMS["validation_split"],
        "use_mixed_precision": DEFAULT_TRAINING_PARAMS["use_mixed_precision"],
        # Ustawienia augmentacji
        "augmentation_params": DEFAULT_AUGMENTATION_PARAMS,
        # Ustawienia interfejsu
        "theme": "Systemowy",
        "language": "Polski",
        "font_size": 11,
        "autosave": True,
        "confirm_exit": True,
        "notifications": True,
        # Ustawienia systemowe
        "memory_limit": 4096,
        "threads": 4,
        "backup_enabled": False,
        "backup_dir": "data/backup",
        "backup_interval": 24,
        # Ustawienia kolorów wykresu
        "chart_train_loss_color": "b",
        "chart_val_loss_color": "r",
        "chart_train_acc_color": "g",
        "chart_val_acc_color": "m",
        "chart_plot_area_background_color": "w",
    }
Nie ma tu dużo do optymalizacji, kod jest już dość dobrze napisany.
Zmiany w app/utils/file_utils.py
python# Optymalizacja funkcji fix_task_file_extensions
def fix_task_file_extensions():
    """Naprawia rozszerzenia plików zadań w katalogu data/tasks."""
    tasks_dir = Path("data/tasks")
    if not tasks_dir.exists():
        return

    for file_path in tasks_dir.iterdir():
        if not file_path.is_file():
            continue

        # Usuń podwójne rozszerzenia .json
        if file_path.name.endswith(".json.json"):
            new_name = file_path.name[:-5]  # usuń ostatnie .json
            new_path = file_path.parent / new_name
            file_path.rename(new_path)
            file_path = new_path

        # Dodaj rozszerzenie .json jeśli go brakuje
        if not file_path.name.endswith(".json"):
            new_path = file_path.with_suffix(".json")
            file_path.rename(new_path)
Kod jest OK.
Zmiany w app/utils/profiler.py
python# Optymalizacja metody _calculate_workers w klasie HardwareProfiler
def _calculate_workers(self, cpu_info):
    """Oblicza zalecana liczbę workerów do ładowania danych."""
    logical_cores = cpu_info.get("cores_logical", os.cpu_count() or 4)

    # Ograniczamy maksymalną ilość workerów, uwzględniając liczbę rdzeni
    return max(1, min(logical_cores - 1, logical_cores // 2, 16))
Uproszczona logika określania optymalnej liczby workerów.
Podsumowanie zmian

Optymalizacja wyrażeń i warunków - Uprościłem kilka skomplikowanych wyrażeń i warunków, co poprawia czytelność kodu.
Lepsze wykorzystanie funkcji wbudowanych - Zastosowano funkcje takie jak sum z generatorami zamiast tworzenia list tylko po to, aby policzyć ich długość.
Dodanie typowania - Dodano brakujące adnotacje typów w niektórych miejscach.
Uproszczenie powtarzalnych fragmentów kodu - Zastąpiono powtarzające się fragmenty kodu pętlami lub bardziej zwięzłymi wyrażeniami.
Poprawa zbierania statystyk - Dodano licznik pominiętych plików ze względu na próg pewności w metodzie sort_directory.
Zwiększenie bezpieczeństwa operacji - Zmieniono kolejność usuwania kopii zapasowych w StateManager, aby usuwać je tylko po pomyślnym zapisie nowych danych.

Wszystkie te zmiany zachowują obecną logikę i strukturę kodu, jednocześnie poprawiając jego jakość, czytelność i wydajność.