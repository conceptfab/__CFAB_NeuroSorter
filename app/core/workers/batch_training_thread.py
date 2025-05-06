import datetime
import json
import os
import time
import traceback
from typing import Dict, List

import torch
from PyQt6.QtCore import QThread, pyqtSignal
from torchvision import datasets

from ai.classifier import ImageClassifier
from ai.optimized_training import train_model_optimized
from ai.preprocessing import get_default_transforms
from app.core.logger import Logger
from app.utils.file_utils import (
    validate_model_path,
    validate_training_directory,
    validate_validation_directory,
)


class BatchTrainingThread(QThread):
    """Wątek do wykonywania wsadowego treningu modeli."""

    task_started = pyqtSignal(str, str)
    task_progress = pyqtSignal(str, int, dict)
    task_completed = pyqtSignal(str, dict)
    all_tasks_completed = pyqtSignal()
    error = pyqtSignal(str, str)

    def __init__(self, task_paths_or_data):
        """
        Inicjalizuje wątek wsadowego treningu.

        Args:
            task_paths_or_data (list): Lista ścieżek do plików JSON lub słowników opisujących zadania.
        """
        super().__init__()
        self.task_paths_or_data = task_paths_or_data
        self._stopped = False
        self.logger = Logger()

    def stop(self):
        """Zatrzymuje wątek treningu."""
        self.logger.info("BatchTrainingThread: Otrzymano żądanie zatrzymania.")
        self._stopped = True

    def run(self):
        """Wykonuje wsadowe zadania treningowe."""
        self.logger.info("BatchTrainingThread.run: Rozpoczęto działanie wątku.")

        try:
            self.logger.info("Rozpoczynam przetwarzanie zadań treningowych...")

            # Pobierz listę zadań
            tasks = self._get_training_tasks()
            if not tasks:
                self.logger.info("Brak zadań treningowych do wykonania.")
                return

            self.logger.info(f"Znaleziono {len(tasks)} zadań treningowych.")

            for task in tasks:
                try:
                    # Loguj konfigurację zadania
                    self.logger.info("\nKonfiguracja zadania:")
                    self.logger.info(f"- Nazwa: {task['name']}")
                    self.logger.info(f"- Typ: {task.get('type', 'Nieznany')}")

                    # Wczytaj dane zadania z pliku JSON
                    try:
                        with open(task["path"], "r", encoding="utf-8") as f:
                            task_data = json.load(f)
                        self.logger.info(f"Wczytano dane zadania z: {task['path']}")
                        self.logger.info(
                            f"Struktura danych zadania: {json.dumps(task_data, indent=2, ensure_ascii=False)}"
                        )
                    except Exception as e:
                        self.logger.error(
                            f"BatchTrainingThread.run: BŁĄD wczytywania pliku {task['path']}: {str(e)}"
                        )
                        self.error.emit(
                            "Główny wątek", f"Błąd wczytywania pliku: {str(e)}"
                        )
                        continue

                    task_name = task_data.get("name", "Bez nazwy")
                    task_type = task_data.get("type", "Trening")

                    # Ujednolicenie formatu typu zadania
                    if task_type.lower() in ["doszkalanie", "finetuning"]:
                        task_type = "Doszkalanie"
                    elif task_type.lower() in ["trening", "training"]:
                        task_type = "Trening"

                    self.logger.info(f"Rozpoznany typ zadania: {task_type}")

                    # Powiadom o rozpoczęciu zadania
                    self.task_started.emit(task_name, task_type)
                    self.logger.info(
                        f"BatchTrainingThread.run: Rozpoczęto zadanie '{task_name}' typu '{task_type}'."
                    )

                    # Wykonaj zadanie w zależności od typu
                    result = None
                    if task_type == "Trening":
                        self.logger.info(
                            f"  - Wykonywanie zadania treningu: {task_name}"
                        )
                        result = self._run_training_task(
                            task_data, task_name, task["path"]
                        )
                    elif task_type == "Doszkalanie":
                        self.logger.info(
                            f"  - Wykonywanie zadania doszkalania: {task_name}"
                        )
                        result = self._run_finetuning_task(
                            task_data, task_name, task["path"]
                        )
                    else:
                        self.logger.error(
                            f"  - BŁĄD: Nieznany typ zadania: {task_type} dla zadania {task_name}"
                        )
                        self.error.emit(task_name, f"Nieznany typ zadania: {task_type}")
                        continue

                    # Sprawdź czy wątek został zatrzymany podczas wykonywania zadania
                    if self._stopped:
                        self.logger.info(
                            f"BatchTrainingThread.run: Zadanie '{task_name}' przerwane z powodu zatrzymania wątku."
                        )
                        break

                    # Powiadom o zakończeniu zadania (tylko jeśli nie zatrzymano)
                    self.task_completed.emit(task_name, result)
                    self.logger.info(
                        f"BatchTrainingThread.run: Zakończono zadanie '{task_name}'."
                    )

                except Exception as e:
                    self.logger.error(
                        f"BatchTrainingThread.run: BŁĄD: {str(e)}\n{traceback.format_exc()}"
                    )
                    self.error.emit("Główny wątek", str(e))
                finally:
                    # Wyczyść zasoby po zakończeniu zadania (nawet w przypadku błędu)
                    self.logger.info("Rozpoczynam czyszczenie zasobów po zadaniu...")
                    self._cleanup_resources()
                    self.logger.info("Zakończono czyszczenie zasobów.")

            # Powiadom o zakończeniu wszystkich zadań
            if not self._stopped:
                self.all_tasks_completed.emit()
                self.logger.info(
                    "BatchTrainingThread.run: Wszystkie zadania zakończone."
                )

        except Exception as e:
            self.logger.error(
                f"BatchTrainingThread.run: BŁĄD: {str(e)}\n{traceback.format_exc()}"
            )
            self.error.emit("Główny wątek", str(e))
        finally:
            # Wyczyść zasoby po zakończeniu wszystkich zadań
            self.logger.info("Rozpoczynam końcowe czyszczenie zasobów...")
            self._cleanup_resources()
            self.logger.info("Zakończono końcowe czyszczenie zasobów.")

    def _get_training_tasks(self) -> List[Dict]:
        """
        Pobiera listę zadań treningowych z katalogu tasks.

        Returns:
            List[Dict]: Lista zadań treningowych
        """
        tasks = []
        tasks_dir = os.path.join("data", "tasks")

        if not os.path.exists(tasks_dir):
            self.logger.info(f"Katalog zadań nie istnieje: {tasks_dir}")
            return tasks

        for task_file in os.listdir(tasks_dir):
            if not task_file.endswith(".json"):
                continue

            task_path = os.path.join(tasks_dir, task_file)
            try:
                with open(task_path, "r", encoding="utf-8") as f:
                    task = json.load(f)

                # Sprawdzamy czy zadanie jest typu Trening lub Doszkalanie i ma status Nowy
                if task.get("status") == "Nowy" and task.get("type") in [
                    "Trening",
                    "Doszkalanie",
                ]:
                    task["path"] = task_path
                    tasks.append(task)
                    self.logger.info(
                        f"Znaleziono zadanie: {task.get('name')} typu {task.get('type')}"
                    )

            except Exception as e:
                self.logger.error(f"Błąd wczytywania zadania {task_file}: {str(e)}")

        return tasks

    def _run_training_task(self, task_data, task_name, task_path):
        """Wykonuje zadanie treningu modelu."""
        try:
            # Pobierz sekcję config z danych zadania
            config = task_data.get("config", {})

            # Pobierz parametry zadania z sekcji config
            model_type = config.get("model_arch", "resnet18")
            training_dir = config.get("data_dir", "")
            validation_dir = config.get("val_dir", None)
            epochs = config.get("epochs", 10)
            batch_size = config.get("batch_size", 32)
            learning_rate = config.get("learning_rate", 0.001)

            # Sprawdź czy ścieżka treningowa nie jest pusta lub None
            if not training_dir or training_dir.strip() == "":
                error_msg = "Ścieżka do katalogu treningowego jest pusta"
                self.logger.error(f"BŁĄD w _run_training_task: {error_msg}")
                raise ValueError(error_msg)

            # Walidacja ścieżek
            is_valid, error_msg = validate_training_directory(training_dir)
            if not is_valid:
                raise ValueError(error_msg)

            # Walidacja katalogu walidacyjnego (opcjonalny)
            if validation_dir:
                is_valid, error_msg = validate_validation_directory(validation_dir)
                if not is_valid:
                    raise ValueError(error_msg)

            # Sprawdź czy katalog zawiera podkatalogi klas
            has_subdirs = any(
                os.path.isdir(os.path.join(training_dir, d))
                for d in os.listdir(training_dir)
            )
            if not has_subdirs:
                raise ValueError(
                    f"Katalog treningowy nie zawiera podkatalogów klas: {training_dir}"
                )

            # Załaduj dane treningowe
            self.logger.info(
                "\nŁadowanie danych treningowych w celu określenia liczby klas..."
            )
            try:
                temp_transform = get_default_transforms()
                train_dataset = datasets.ImageFolder(
                    training_dir, transform=temp_transform
                )
                num_classes = len(train_dataset.classes)
                self.logger.info(f"Wykryto {num_classes} klas w katalogu treningowym:")
                for class_idx, class_name in enumerate(train_dataset.classes):
                    self.logger.info(f"  - Klasa {class_idx}: {class_name}")
                if num_classes == 0:
                    raise ValueError(
                        "Nie znaleziono żadnych podkatalogów klas w katalogu treningowym."
                    )
            except Exception as e:
                self.logger.error(f"BŁĄD podczas ładowania danych treningowych: {e}")
                self.logger.error(f"TRACEBACK: {traceback.format_exc()}")
                raise

            # Utwórz model
            self.logger.info(
                f"\nTworzenie modelu {model_type} dla {num_classes} klas..."
            )
            try:
                model = ImageClassifier(model_type=model_type, num_classes=num_classes)
                self.logger.info(f"Model utworzony pomyślnie: {model}")
                model_info = (
                    model.get_model_info() if hasattr(model, "get_model_info") else {}
                )
                self.logger.info(
                    f"DEBUG: Informacje o modelu: {json.dumps(model_info, indent=2)}"
                )
            except Exception as e:
                self.logger.error(f"BŁĄD podczas tworzenia modelu: {e}")
                self.logger.error(f"TRACEBACK: {traceback.format_exc()}")
                raise

            # Ustal urządzenie
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.model = model.model.to(device)

            # Wykonaj trening
            self.logger.info("DEBUG: Rozpoczynam trening z parametrami:")
            self.logger.info(f"DEBUG: model={type(model.model)}")
            self.logger.info(f"DEBUG: train_dir={training_dir}")
            self.logger.info(f"DEBUG: val_dir={validation_dir}")
            self.logger.info(f"DEBUG: num_epochs={epochs}")
            self.logger.info(f"DEBUG: batch_size={batch_size}")
            self.logger.info(f"DEBUG: learning_rate={learning_rate}")

            start_time = time.time()

            def progress_callback(
                epoch, num_epochs, train_loss, train_acc, val_loss, val_acc
            ):
                """Callback do śledzenia postępu treningu."""
                # Debug - sprawdź wartości przed emisją sygnału
                print(f"\nDEBUG Progress Callback:")
                print(f"Epoka: {epoch}/{num_epochs}")
                print(f"Strata treningowa: {train_loss}")
                print(f"Dokładność treningowa: {train_acc}")
                print(f"Strata walidacyjna: {val_loss}")
                print(f"Dokładność walidacyjna: {val_acc}")

                # Upewnij się, że wartości nie są None i są liczbami
                train_loss = float(train_loss) if train_loss is not None else 0.0
                train_acc = float(train_acc) if train_acc is not None else 0.0
                val_loss = float(val_loss) if val_loss is not None else None
                val_acc = float(val_acc) if val_acc is not None else None

                # Emituj sygnał z postępem
                self.task_progress.emit(
                    task_name,
                    int((epoch / num_epochs) * 100) if num_epochs > 0 else 0,
                    {
                        "epoch": epoch,
                        "total_epochs": num_epochs,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    },
                )

            result = train_model_optimized(
                model=model.model,
                train_dir=training_dir,
                val_dir=validation_dir,
                num_epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                progress_callback=progress_callback,
            )

            training_time = time.time() - start_time
            self.logger.info(f"Czas treningu: {training_time:.2f}s")
            self.logger.info(
                f"Parametry do zapisania: {json.dumps(result, indent=2, ensure_ascii=False)}"
            )
            self.logger.info(f"Ścieżka pliku zadania: {task_path}")
            if not task_path or not os.path.exists(task_path):
                self.logger.error(
                    f"BŁĄD: Ścieżka pliku zadania nie istnieje lub jest pusta: {task_path}"
                )
                raise ValueError(
                    f"Brak pliku zadania lub ścieżka nie istnieje: {task_path}"
                )
            self._save_training_time(task_path, training_time, result)

            # Zapisz model
            try:
                # Zapisz model do katalogu wyjściowego
                output_dir = os.path.join("data", "models")
                os.makedirs(output_dir, exist_ok=True)

                # Upewniamy się, że model ma przypisane class_names
                if "class_names" in result and result["class_names"]:
                    model.class_names = result["class_names"]
                    self.logger.info(
                        f"Przypisano mapowanie klas: {result['class_names']}"
                    )
                elif hasattr(model, "class_names") and model.class_names:
                    self.logger.info(
                        f"Użyto istniejącego mapowania klas modelu: {model.class_names}"
                    )
                else:
                    self.logger.warning(
                        "Brak mapowania klas w wynikach treningu i modelu!"
                    )
                    model.class_names = None

                # Generuj nazwę pliku modelu
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
                model_filename = f"{model_type}_{epochs}epok_{timestamp}.pt"
                model_path = os.path.join(output_dir, model_filename)

                self.logger.info(f"Generowana nazwa modelu: {model_filename}")

                # Zapisz stan modelu
                model.save(
                    model_path,
                    metadata={
                        "accuracy": result.get("val_acc", 0),
                        "training_time": training_time,
                        "training_params": task_data,
                        "timestamp": timestamp,
                        "class_names": model.class_names,
                    },
                )

                self.logger.info(f"Model zapisany w: {model_path}")

            except Exception as e:
                self.logger.error(f"BŁĄD podczas zapisywania modelu: {e}")
                self.logger.error(f"TRACEBACK: {traceback.format_exc()}")
                raise

            return result

        except Exception as e:
            self.logger.error(f"BŁĄD w _run_training_task: {str(e)}")
            self.logger.error(f"TRACEBACK: {traceback.format_exc()}")
            raise

    def _run_finetuning_task(self, task_data, task_name, task_path):
        """Wykonuje zadanie doszkalania modelu."""
        try:
            self.logger.info(f"[FINETUNE] ===== ROZPOCZYNAM DOSZKALANIE MODELU =====")
            self.logger.info(f"[FINETUNE] Zadanie: {task_name}")
            self.logger.info(f"[FINETUNE] Ścieżka zadania: {task_path}")
            self.logger.info(
                f"[FINETUNE] Pełne dane zadania: {json.dumps(task_data, indent=2, ensure_ascii=False)}"
            )

            # Obsługa różnych struktur danych wejściowych
            config = task_data.get("config", {})

            # Jeśli dane są w polu config, używamy ich, w przeciwnym razie używamy bezpośrednio task_data
            base_model_path = config.get("base_model", task_data.get("model_path", ""))
            training_dir = config.get(
                "train_dir",
                config.get("training_dir", task_data.get("training_dir", "")),
            )
            validation_dir = config.get("val_dir", task_data.get("val_dir", None))
            epochs = config.get("epochs", task_data.get("epochs", 5))
            batch_size = config.get("batch_size", task_data.get("batch_size", 32))
            learning_rate = config.get(
                "learning_rate", task_data.get("learning_rate", 0.0001)
            )
            freeze_backbone = config.get(
                "freeze_backbone", task_data.get("freeze_backbone", True)
            )

            self.logger.info(f"[FINETUNE] ===== PARAMETRY DOSZKALANIA =====")
            self.logger.info(f"[FINETUNE] Ścieżka modelu bazowego: {base_model_path}")
            self.logger.info(f"[FINETUNE] Katalog treningowy: {training_dir}")
            self.logger.info(f"[FINETUNE] Katalog walidacyjny: {validation_dir}")
            self.logger.info(f"[FINETUNE] Liczba epok: {epochs}")
            self.logger.info(f"[FINETUNE] Rozmiar batcha: {batch_size}")
            self.logger.info(f"[FINETUNE] Współczynnik uczenia: {learning_rate}")
            self.logger.info(f"[FINETUNE] Zamrożenie backbone: {freeze_backbone}")

            # Sprawdź poprawność rozszerzenia pliku modelu
            self.logger.info(f"[FINETUNE] ===== WALIDACJA PLIKU MODELU =====")
            if not base_model_path.endswith((".pt", ".pth")):
                error_msg = f"Niewłaściwy format pliku modelu: {base_model_path}. Wymagane rozszerzenie .pt lub .pth"
                self.logger.error(f"[FINETUNE] BŁĄD: {error_msg}")
                raise ValueError(error_msg)
            self.logger.info(f"[FINETUNE] ✓ Format pliku modelu poprawny")

            # Sprawdź czy plik modelu istnieje
            if not os.path.exists(base_model_path):
                error_msg = f"Plik modelu nie istnieje: {base_model_path}"
                self.logger.error(f"[FINETUNE] BŁĄD: {error_msg}")
                raise ValueError(error_msg)
            self.logger.info(f"[FINETUNE] ✓ Plik modelu istnieje")

            # Sprawdź czy istnieje plik konfiguracyjny
            config_path = os.path.splitext(base_model_path)[0] + "_config.json"
            if not os.path.exists(config_path):
                self.logger.warning(
                    f"[FINETUNE] ⚠️ Nie znaleziono pliku konfiguracyjnego: {config_path}"
                )
            else:
                self.logger.info(
                    f"[FINETUNE] ✓ Znaleziono plik konfiguracyjny: {config_path}"
                )

            # Walidacja katalogu treningowego
            self.logger.info(f"[FINETUNE] ===== WALIDACJA KATALOGÓW =====")
            if not training_dir or training_dir.strip() == "":
                error_msg = "Ścieżka do katalogu treningowego jest pusta"
                self.logger.error(f"[FINETUNE] BŁĄD: {error_msg}")
                raise ValueError(error_msg)
            self.logger.info(
                f"[FINETUNE] ✓ Ścieżka katalogu treningowego nie jest pusta"
            )

            is_valid, error_msg = validate_training_directory(training_dir)
            if not is_valid:
                self.logger.error(
                    f"[FINETUNE] BŁĄD walidacji katalogu treningowego: {error_msg}"
                )
                raise ValueError(error_msg)
            self.logger.info(f"[FINETUNE] ✓ Katalog treningowy poprawny")

            # Walidacja katalogu walidacyjnego (opcjonalny)
            if validation_dir:
                is_valid, error_msg = validate_validation_directory(validation_dir)
                if not is_valid:
                    self.logger.error(
                        f"[FINETUNE] BŁĄD walidacji katalogu walidacyjnego: {error_msg}"
                    )
                    raise ValueError(error_msg)
                self.logger.info(f"[FINETUNE] ✓ Katalog walidacyjny poprawny")
            else:
                self.logger.info(
                    f"[FINETUNE] ℹ️ Katalog walidacyjny nie został określony - pomijam walidację"
                )

            # Sprawdź czy katalog treningowy zawiera podkatalogi klas
            has_subdirs = any(
                os.path.isdir(os.path.join(training_dir, d))
                for d in os.listdir(training_dir)
            )
            if not has_subdirs:
                error_msg = (
                    f"Katalog treningowy nie zawiera podkatalogów klas: {training_dir}"
                )
                self.logger.error(f"[FINETUNE] BŁĄD: {error_msg}")
                raise ValueError(error_msg)
            self.logger.info(
                f"[FINETUNE] ✓ Katalog treningowy zawiera podkatalogi klas"
            )

            # Załaduj model bazowy
            self.logger.info(f"[FINETUNE] ===== ŁADOWANIE MODELU =====")
            self.logger.info(
                f"[FINETUNE] Próba załadowania modelu z: {base_model_path}"
            )
            model = ImageClassifier(weights_path=base_model_path)
            self.logger.info(f"[FINETUNE] ✓ Model załadowany pomyślnie")

            # Ustal urządzenie
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.model = model.model.to(device)
            self.logger.info(f"[FINETUNE] ✓ Model przeniesiony na urządzenie: {device}")
            if torch.cuda.is_available():
                self.logger.info(f"[FINETUNE] Informacje o GPU:")
                self.logger.info(f"[FINETUNE] - Nazwa: {torch.cuda.get_device_name(0)}")
                self.logger.info(
                    f"[FINETUNE] - Pamięć całkowita: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
                )
                self.logger.info(
                    f"[FINETUNE] - Pamięć dostępna: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB"
                )

            # Wykonaj doszkalanie
            self.logger.info(f"[FINETUNE] ===== ROZPOCZYNAM DOSZKALANIE =====")
            start_time = time.time()

            def log_progress(
                epoch, num_epochs, train_loss, train_acc, val_loss, val_acc
            ):
                self.logger.info(f"[FINETUNE][EPOKA {epoch}/{num_epochs}]")
                self.logger.info(f"[FINETUNE] - Strata treningowa: {train_loss:.4f}")
                self.logger.info(f"[FINETUNE] - Dokładność treningowa: {train_acc:.4f}")
                if val_loss is not None:
                    self.logger.info(f"[FINETUNE] - Strata walidacyjna: {val_loss:.4f}")
                if val_acc is not None:
                    self.logger.info(
                        f"[FINETUNE] - Dokładność walidacyjna: {val_acc:.4f}"
                    )
                self.logger.info(
                    f"[FINETUNE] - Postęp: {int((epoch / num_epochs) * 100)}%"
                )

            result = train_model_optimized(
                model=model.model,
                train_dir=training_dir,
                val_dir=validation_dir,
                num_epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                freeze_backbone=freeze_backbone,
                progress_callback=lambda epoch, num_epochs, train_loss, train_acc, val_loss, val_acc: (
                    log_progress(
                        epoch, num_epochs, train_loss, train_acc, val_loss, val_acc
                    ),
                    self.task_progress.emit(
                        task_name,
                        int((epoch / num_epochs) * 100) if num_epochs > 0 else 0,
                        {
                            "epoch": epoch,
                            "total_epochs": num_epochs,
                            "train_loss": train_loss if train_loss > 0 else 0.0001,
                            "train_acc": train_acc if train_acc > 0 else 0.0001,
                            "val_loss": val_loss,
                            "val_acc": val_acc,
                        },
                    ),
                ),
            )

            training_time = time.time() - start_time
            self.logger.info(f"[FINETUNE] ===== PODSUMOWANIE DOSZKALANIA =====")
            self.logger.info(f"[FINETUNE] Czas wykonania: {training_time:.2f}s")
            self.logger.info(f"[FINETUNE] Wyniki:")
            self.logger.info(
                f"[FINETUNE] - Końcowa strata treningowa: {result.get('train_loss', 0):.4f}"
            )
            self.logger.info(
                f"[FINETUNE] - Końcowa dokładność treningowa: {result.get('train_acc', 0):.4f}"
            )
            self.logger.info(
                f"[FINETUNE] - Końcowa strata walidacyjna: {result.get('val_loss', 0):.4f}"
            )
            self.logger.info(
                f"[FINETUNE] - Końcowa dokładność walidacyjna: {result.get('val_acc', 0):.4f}"
            )

            # Zapisz wyniki
            self._save_training_time(task_path, training_time, result)
            self.logger.info(f"[FINETUNE] ✓ Zapisano wyniki do pliku zadania")

            self.logger.info(f"[FINETUNE] ===== ZAKOŃCZONO DOSZKALANIE =====")
            return result

        except Exception as e:
            self.logger.error(f"[FINETUNE] ===== BŁĄD W DOSZKALANIU =====")
            self.logger.error(f"[FINETUNE] Typ błędu: {type(e).__name__}")
            self.logger.error(f"[FINETUNE] Treść błędu: {str(e)}")
            self.logger.error(f"[FINETUNE] Stack trace:")
            self.logger.error(traceback.format_exc())
            raise

    def _save_training_time(
        self, task_path: str, training_time: float, result: dict = None
    ):
        """Zapisuje czas treningu, dokładność walidacyjną i informacje o epokach w pliku JSON z zadaniem."""
        try:
            with open(task_path, "r", encoding="utf-8") as f:
                task_data = json.load(f)

            # Dodaj czas treningu do danych zadania
            task_data["training_time"] = round(training_time, 2)
            import datetime as _dt

            td = _dt.timedelta(seconds=int(training_time))
            task_data["training_time_str"] = str(td)

            # Ustaw status zadania na Zakończony
            task_data["status"] = "Zakończony"

            # Dodaj dokładność walidacyjną jeśli jest dostępna
            if result and "history" in result:
                history = result["history"]
                if history.get("train_acc"):
                    task_data["train_accuracy"] = history["train_acc"][-1]
                if history.get("train_loss"):
                    task_data["train_loss"] = history["train_loss"][-1]
                if history.get("val_acc"):
                    task_data["validation_accuracy"] = history["val_acc"][-1]
                if history.get("val_loss"):
                    task_data["validation_loss"] = history["val_loss"][-1]

            # Dodaj informacje o epokach
            if result and "epoch" in result and "total_epochs" in result:
                task_data["completed_epochs"] = result["epoch"]
                task_data["total_epochs"] = result["total_epochs"]

            # Zapisz zaktualizowane dane z powrotem do pliku
            with open(task_path, "w", encoding="utf-8") as f:
                json.dump(task_data, f, indent=4, ensure_ascii=False)

            self.logger.info(
                f"Zapisano informacje o treningu: czas={training_time:.2f}s, "
                f"dokładność={result.get('val_acc', 0):.4f} "
                f"(epoki: {result.get('epoch', 0)}/{result.get('total_epochs', 0)})"
            )
        except Exception as e:
            self.logger.error(
                f"Błąd podczas zapisywania informacji o treningu: {str(e)}"
            )

    def _cleanup_resources(self):
        """Czyści zasoby systemowe po zakończeniu treningu."""
        try:
            self.logger.info("=== Rozpoczynam czyszczenie zasobów systemowych ===")

            # Czyszczenie pamięci GPU (VRAM)
            if torch.cuda.is_available():
                self.logger.info("Czyszczenie pamięci VRAM GPU...")
                torch.cuda.empty_cache()
                self.logger.info("✓ Wyczyszczono pamięć VRAM GPU")
            else:
                self.logger.info("GPU nie jest dostępne - pomijam czyszczenie VRAM")

            # Czyszczenie pamięci RAM
            self.logger.info("Czyszczenie pamięci RAM...")
            import gc

            gc.collect()
            self.logger.info("✓ Wyczyszczono pamięć RAM")

            # Czyszczenie cache PyTorch
            if hasattr(torch, "cuda"):
                self.logger.info("Czyszczenie cache PyTorch...")
                torch.cuda.empty_cache()
                self.logger.info("✓ Wyczyszczono cache PyTorch")
            else:
                self.logger.info(
                    "PyTorch CUDA nie jest dostępne - pomijam czyszczenie cache"
                )

            self.logger.info("=== Zakończono czyszczenie zasobów systemowych ===")

        except Exception as e:
            self.logger.error(f"Błąd podczas czyszczenia zasobów: {str(e)}")
            self.logger.error(f"TRACEBACK: {traceback.format_exc()}")
