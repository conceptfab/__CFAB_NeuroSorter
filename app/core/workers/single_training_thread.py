import datetime
import json
import os
import time
import traceback
from typing import Dict

import torch
from PyQt6.QtCore import QThread, pyqtSignal
from torchvision import datasets

from ai.classifier import ImageClassifier
from ai.fine_tuning import fine_tune_model
from ai.optimized_training import train_model_optimized
from ai.preprocessing import get_default_transforms
from app.core.logger import Logger
from app.utils.file_utils import (
    validate_training_directory,
    validate_validation_directory,
)


class SingleTrainingThread(QThread):
    """Wątek do wykonywania pojedynczego zadania treningu modelu."""

    task_started = pyqtSignal(str, str)
    task_progress = pyqtSignal(str, int, dict)
    task_completed = pyqtSignal(str, dict)
    error = pyqtSignal(str, str)
    log_message_signal = pyqtSignal(str)

    def __init__(self, task_path_or_data):
        """
        Inicjalizuje wątek pojedynczego treningu.

        Args:
            task_path_or_data (str lub dict): Ścieżka do pliku JSON lub słownik opisujący zadanie.
        """
        super().__init__()
        self.task_path_or_data = task_path_or_data
        self._stopped = False
        self.logger = Logger()

    def stop(self):
        """Zatrzymuje wątek treningu."""
        self.logger.info("SingleTrainingThread: Otrzymano żądanie zatrzymania.")
        self._stopped = True

    def run(self):
        """Wykonuje pojedyncze zadanie treningowe."""
        self.logger.info("=== ROZPOCZYNAM WYKONYWANIE ZADANIA ===")
        self.logger.info("SingleTrainingThread.run: Rozpoczęto działanie wątku.")

        try:
            # Wczytaj dane zadania
            if isinstance(self.task_path_or_data, str):
                with open(self.task_path_or_data, "r", encoding="utf-8") as f:
                    task_data = json.load(f)
                task_path = self.task_path_or_data
            else:
                task_data = self.task_path_or_data
                # Zapisz dane zadania do pliku
                task_name = task_data.get("name", "Bez nazwy")
                task_path = os.path.join("data", "tasks", f"{task_name}.json")
                os.makedirs(os.path.dirname(task_path), exist_ok=True)
                with open(task_path, "w", encoding="utf-8") as f:
                    json.dump(task_data, f, indent=4, ensure_ascii=False)
                self.logger.info(f"Zapisano dane zadania do pliku: {task_path}")

            task_name = task_data.get("name", "Bez nazwy")
            task_type = task_data.get("type", "training")

            # Emituj sygnał o rozpoczęciu zadania
            self.task_started.emit(task_name, task_type)
            self.logger.info(
                f"EMITOWANO task_started dla: {task_name}, typ: {task_type}"
            )

            # Dodajemy szczegółowe logi
            self.logger.info(f"=== INFORMACJE O ZADANIU ===")
            self.logger.info(f"Nazwa zadania: {task_name}")
            self.logger.info(f"Typ zadania: {task_type}")
            self.logger.info(
                f"Pełne dane zadania: {json.dumps(task_data, indent=2, ensure_ascii=False)}"
            )

            # Ujednolicenie formatu typu zadania
            task_type = task_type.lower()
            if task_type in ["doszkalanie", "fine-tuning", "fine_tuning"]:
                task_type = "fine_tuning"
                self.logger.info("=== ROZPOZNANO ZADANIE FINE-TUNINGU ===")
                self.logger.info(
                    "Zadanie będzie wykonywane przez skrypt: fine_tuning.py"
                )
                result = self._run_finetuning_task(task_data, task_name, task_path)
            elif task_type in ["trening", "training"]:
                task_type = "training"
                self.logger.info(
                    "============ ROZPOZNANO ZADANIE TRENINGU ============"
                )
                self.logger.info(
                    "Zadanie będzie wykonywane przez skrypt: optimized_training.py"
                )
                result = self._run_training_task(task_data, task_name, task_path)
            else:
                error_msg = f"Nieznany typ zadania: {task_type}. Oczekiwano 'training' lub 'fine_tuning'."
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            # Sprawdź czy wątek został zatrzymany podczas wykonywania zadania
            if self._stopped:
                self.logger.info(
                    f"SingleTrainingThread.run: Zadanie '{task_name}' przerwane z powodu zatrzymania wątku."
                )
                return

            # Powiadom o zakończeniu zadania (tylko jeśli nie zatrzymano)
            self.task_completed.emit(task_name, result)
            self.logger.info(
                f"SingleTrainingThread.run: Zakończono zadanie '{task_name}'."
            )

        except Exception as e:
            self.logger.error(
                f"SingleTrainingThread.run: BŁĄD: {str(e)}\n{traceback.format_exc()}"
            )
            self.error.emit("Główny wątek", str(e))
        finally:
            # Wyczyść zasoby po zakończeniu zadania (nawet w przypadku błędu)
            self.logger.info("Rozpoczynam czyszczenie zasobów po zadaniu...")
            self._cleanup_resources()
            self.logger.info("Zakończono czyszczenie zasobów.")

    def _run_training_task(self, task_data, task_name, task_path):
        """Wykonuje zadanie treningu modelu."""
        try:
            self.logger.info(
                "============ ROZPOCZYNAM TRENING NOWEGO MODELU ============"
            )
            self.logger.info("Używam skryptu: optimized_training.py")
            self.logger.info(f"Zadanie: {task_name}")
            self.logger.info(f"Ścieżka zadania: {task_path}")

            # Pobierz sekcję config z danych zadania
            config = task_data.get("config", {})

            # Pobierz parametry modelu z konfiguracji
            model_config = config.get("model", {})
            model_type = model_config.get("architecture", "resnet18")
            model_variant = model_config.get("variant", "")
            if model_variant:
                model_type = f"{model_type}-{model_variant}"

            # Pobierz parametry treningu
            training_config = config.get("training", {})
            epochs = training_config.get("epochs", 10)
            batch_size = training_config.get("batch_size", 32)
            learning_rate = training_config.get("learning_rate", 0.001)
            optimizer_type = training_config.get("optimizer", "AdamW").lower()
            scheduler_type = training_config.get(
                "scheduler", "CosineAnnealingWarmRestarts"
            ).lower()
            num_workers = training_config.get("num_workers", 0)
            freeze_base_model = training_config.get("freeze_base_model", False)
            mixed_precision = training_config.get("mixed_precision", False)

            # Pobierz parametry regularizacji
            reg_config = config.get("regularization", {})
            weight_decay = reg_config.get("weight_decay", 0.00015)
            label_smoothing = reg_config.get("label_smoothing", 0.1)

            # Pobierz parametry augmentacji
            aug_config = config.get("augmentation", {})
            mixup_config = aug_config.get("mixup", {})
            mixup = mixup_config.get("use", False)

            # Pobierz parametry monitorowania
            monitoring_config = config.get("monitoring", {})
            early_stopping_config = monitoring_config.get("early_stopping", {})
            early_stopping = early_stopping_config.get("use", True)
            patience = early_stopping_config.get("patience", 5)

            # Pobierz parametry zaawansowane
            advanced_config = config.get("advanced", {})
            cross_validation_config = advanced_config.get("cross_validation", {})
            use_cross_validation = cross_validation_config.get("use", False)
            k_folds = cross_validation_config.get("folds", 5)

            # Pobierz katalogi danych
            training_dir = config.get("train_dir", "")
            if not training_dir:
                training_dir = config.get("data_dir", "")
            validation_dir = config.get("val_dir", None)

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

            # Zbuduj ścieżkę zapisu modelu na podstawie nazwy zadania
            output_dir = os.path.join("data", "models")
            os.makedirs(output_dir, exist_ok=True)
            # Upewnij się, że nazwa zadania jest bezpieczna dla systemu plików
            safe_task_name = (
                task_name.replace(":", "_")
                .replace(" ", "_")
                .replace("/", "_")
                .replace("\\", "_")
            )
            model_save_path = os.path.join(
                output_dir, f"{safe_task_name}.pt"
            )  # Użyj rozszerzenia .pt (lub .pth)
            self.logger.info(f"Wygenerowano ścieżkę zapisu modelu: {model_save_path}")

            def progress_callback(
                epoch,
                num_epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                val_top3=None,
                val_top5=None,
                val_precision=None,
                val_recall=None,
                val_f1=None,
                val_auc=None,
            ):
                """Callback do śledzenia postępu treningu."""
                details = {
                    "epoch": epoch,
                    "total_epochs": num_epochs,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_top3": val_top3,
                    "val_top5": val_top5,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "val_f1": val_f1,
                    "val_auc": val_auc,
                }

                self.task_progress.emit(task_name, epoch, details)
                self.logger.info(
                    f"EMITOWANO task_progress dla: {task_name}, epoka: {epoch}"
                )

            # Wywołanie funkcji trenującej z wszystkimi parametrami z konfiguracji
            result = train_model_optimized(
                model=model.model,
                train_dir=training_dir,
                val_dir=validation_dir,
                num_epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                progress_callback=progress_callback,
                should_stop_callback=lambda: self._stopped,
                freeze_backbone=freeze_base_model,
                lr_scheduler_type=scheduler_type,
                early_stopping=early_stopping,
                mixup=mixup,
                label_smoothing=label_smoothing,
                weight_decay=weight_decay,
                optimizer_type=optimizer_type,
                augmentation_mode="extended",  # To można również sparametryzować
                use_cross_validation=use_cross_validation,
                k_folds=k_folds,
                model_save_path=model_save_path,
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
                accuracy = result.get("val_acc", 0)
                num_classes = len(model.class_names) if model.class_names else 0

                # Normalizacja nazwy modelu
                model_type = model_type.lower()
                model_type = model_type.replace("efficientnet-", "").replace(
                    "efficientnet_", ""
                )
                model_type = model_type.replace("resnet", "")
                model_type = model_type.replace("mobilenet", "mobile")
                model_type = model_type.replace("convnext", "")
                model_type = model_type.replace("vit", "")
                model_type = model_type.strip("-_")

                model_version = (
                    task_data.get("config", {}).get("model", {}).get("variant", "")
                )
                if model_version:
                    model_version = model_version.replace(
                        "efficientnet", "EfficientNet"
                    )

                model_filename = f"{model_type}_{model_version}_{num_classes}klas_{accuracy:.2f}acc_{epochs}epok_{timestamp}.pt"
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
            self.logger.info("=== ROZPOCZYNAM DOSZKALANIE MODELU ===")
            self.logger.info("Używam skryptu: fine_tuning.py")
            self.logger.info(f"Zadanie: {task_name}")
            self.logger.info(f"Ścieżka zadania: {task_path}")
            self.logger.info(
                f"[FINETUNE] Pełne dane zadania: {json.dumps(task_data, indent=2, ensure_ascii=False)}"
            )

            # Obsługa różnych struktur danych wejściowych
            config = task_data.get("config", {})

            # Podstawowe parametry (już istniejące)
            base_model_path = config.get("base_model", task_data.get("model_path", ""))
            training_dir = config.get(
                "train_dir",
                config.get("training_dir", task_data.get("training_dir", "")),
            )
            validation_dir = config.get("val_dir", task_data.get("val_dir", None))

            # Parametry treningu
            training_config = config.get("training", {})
            epochs = training_config.get("epochs", task_data.get("epochs", 5))
            batch_size = training_config.get(
                "batch_size", task_data.get("batch_size", 32)
            )
            learning_rate = training_config.get(
                "learning_rate", task_data.get("learning_rate", 0.0001)
            )
            optimizer_type = training_config.get("optimizer", "adamw").lower()
            scheduler_type = training_config.get("scheduler", "plateau").lower()
            warmup_epochs = training_config.get("warmup_epochs", 1)
            use_mixed_precision = training_config.get("mixed_precision", True)

            # Parametry regularizacji
            reg_config = config.get("regularization", {})
            weight_decay = reg_config.get("weight_decay", 0.01)
            label_smoothing = reg_config.get("label_smoothing", 0.1)

            # Parametry zapobiegania zapominaniu katastroftalnemu
            advanced_config = config.get("advanced", {})
            cf_prevention = advanced_config.get(
                "catastrophic_forgetting_prevention", {}
            )
            prevent_forgetting = cf_prevention.get("enable", True)
            preserve_original_classes = cf_prevention.get(
                "preserve_original_classes", True
            )

            # Konfiguracja EWC
            ewc_config = cf_prevention.get("ewc_regularization")

            # Konfiguracja destylacji wiedzy
            knowledge_distillation_config = cf_prevention.get("knowledge_distillation")

            # Konfiguracja rehearsal
            rehearsal_config = cf_prevention.get("rehearsal")

            # Konfiguracja zamrażania warstw
            layer_freezing_config = cf_prevention.get("layer_freezing", {})
            freeze_ratio = layer_freezing_config.get(
                "freeze_ratio", task_data.get("freeze_ratio", 0.8)
            )

            # Parametry augmentacji
            augmentation_params = config.get("augmentation", {})

            # Parametry preprocessingu
            preprocessing_params = config.get("preprocessing", {})

            # Parametry monitorowania
            monitoring_config = config.get("monitoring", {})
            early_stopping_config = monitoring_config.get("early_stopping", {})
            early_stopping_patience = early_stopping_config.get("patience", 5)

            # Parametr użycia green diffusion
            use_green_diffusion = False  # Brak bezpośredniego odpowiednika w JSON

            self.logger.info("[FINETUNE] ===== PARAMETRY DOSZKALANIA =====")
            self.logger.info(f"[FINETUNE] Ścieżka modelu bazowego: {base_model_path}")
            self.logger.info(f"[FINETUNE] Katalog treningowy: {training_dir}")
            self.logger.info(f"[FINETUNE] Katalog walidacyjny: {validation_dir}")
            self.logger.info(f"[FINETUNE] Liczba epok: {epochs}")
            self.logger.info(f"[FINETUNE] Rozmiar batcha: {batch_size}")
            self.logger.info(f"[FINETUNE] Learning rate: {learning_rate}")
            self.logger.info(f"[FINETUNE] Współczynnik zamrożenia: {freeze_ratio}")

            # Sprawdź czy wątek został zatrzymany
            if self._stopped:
                self.logger.info(
                    "[FINETUNE] Wątek został zatrzymany przed rozpoczęciem treningu"
                )
                return

            # Definicja callbacka do śledzenia postępu
            def progress_callback(
                epoch,
                num_epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                val_top3=None,
                val_top5=None,
                val_precision=None,
                val_recall=None,
                val_f1=None,
                val_auc=None,
            ):
                """Callback do śledzenia postępu fine-tuningu."""
                details = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_top3": val_top3,
                    "val_top5": val_top5,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "val_f1": val_f1,
                    "val_auc": val_auc,
                }

                self.task_progress.emit(task_name, epoch, details)
                self.logger.info(
                    f"EMITOWANO task_progress dla: {task_name}, epoka: {epoch}"
                )

            # Uruchom fine-tuning
            start_time = time.time()
            result = fine_tune_model(
                base_model_path=base_model_path,
                train_dir=training_dir,
                val_dir=validation_dir,
                num_epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                freeze_ratio=freeze_ratio,
                optimizer_type=optimizer_type,
                scheduler_type=scheduler_type,
                label_smoothing=label_smoothing,
                weight_decay=weight_decay,
                warmup_epochs=warmup_epochs,
                use_mixed_precision=use_mixed_precision,
                prevent_forgetting=prevent_forgetting,
                preserve_original_classes=preserve_original_classes,
                rehearsal_config=rehearsal_config,
                knowledge_distillation_config=knowledge_distillation_config,
                ewc_config=ewc_config,
                layer_freezing_config=layer_freezing_config,
                augmentation_params=augmentation_params,
                preprocessing_params=preprocessing_params,
                early_stopping_patience=early_stopping_patience,
                use_green_diffusion=use_green_diffusion,
                task_name=task_name,
                progress_callback=progress_callback,
                should_stop_callback=lambda: self._stopped,
            )

            training_time = time.time() - start_time
            self.logger.info(f"Czas treningu: {training_time:.2f}s")
            self.logger.info(
                f"Parametry do zapisania: {json.dumps(result, indent=2, ensure_ascii=False)}"
            )

            # Zapisz czas treningu i wyniki
            self._save_training_time(task_path, training_time, result)

            return result

        except Exception as e:
            self.logger.error(f"Błąd podczas fine-tuningu: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

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
