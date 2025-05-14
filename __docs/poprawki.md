Zmiany w kodzie dla poprawy eksportu wizualizacji treningu z queue_manager
Przeanalizowałem dostarczony kod i zidentyfikowałem, że queue_manager nie zapisuje wizualizacji treningu po zakończeniu zadania treningowego. Oto zmiany, które należy wprowadzić:
Zmiana w pliku app/gui/tabs/training_manager.py
W klasie TrainingManager należy dodać funkcję przekazywania wizualizacji treningu do queue_managera oraz zmodyfikować funkcję _training_task_completed:
pythondef __init__(self, parent=None, settings=None):
    self.logger = logging.getLogger("TrainingManager")
    self.logger.setLevel(logging.INFO)
    super().__init__(parent)
    self.parent = parent
    self.settings = settings
    self.training_thread = None
    self.queue_manager = QueueManager(self)
    self.setup_ui()
    self.connect_signals()
    
    # Przekazanie referencji do wizualizacji treningu
    if hasattr(self, "training_visualization"):
        self.queue_manager.set_visualization_widget(self.training_visualization)
    
    # Automatyczne odświeżenie listy zadań przy starcie
    self.refresh()
W metodzie _training_task_completed należy dodać funkcję eksportu wizualizacji:
pythondef _training_task_completed(self, task_name, result):
    """Obsługuje zakończenie zadania treningowego."""
    try:
        self.parent.logger.info(
            f"DEBUG: _training_task_completed wywołane dla zadania: {task_name}"
        )
        self.parent.logger.info(
            f"DEBUG: _training_task_completed - Stan przycisku stop PRZED "
            f"ustawieniem na False: {self.parent.stop_task_btn.isEnabled()}"
        )

        # Odśwież zakładkę modeli
        self.parent.model_manager_tab.refresh()

        model_filename = result.get("model_filename", "")
        accuracy = result.get("accuracy", 0.0)
        epochs_trained = result.get("epochs_trained", 0)
        training_time = result.get("training_time", 0)

        self.parent.logger.info(
            f"Zakończono zadanie {task_name}. Model: {model_filename}, "
            f"Dokładność: {accuracy:.2%}, Epoki: {epochs_trained}, "
            f"Czas: {training_time:.1f}s"
        )

        # Aktualizacja UI w głównym oknie
        self.parent.current_task_info.setText("Brak aktywnego zadania")
        self.parent.task_progress_bar.setValue(0)
        self.parent.task_progress_details.setText("")
        self.parent.stop_task_btn.setEnabled(False)
        self.parent.logger.info(
            f"DEBUG: _training_task_completed - Stan przycisku stop PO "
            f"ustawieniu na False: {self.parent.stop_task_btn.isEnabled()}"
        )

        # Zapisz wyniki do pliku zadania
        tasks_dir = os.path.join("data", "tasks")
        # Usuń rozszerzenie .json jeśli już istnieje w nazwie
        task_name = task_name.replace(".json", "")
        task_file = os.path.join(tasks_dir, f"{task_name}.json")

        if os.path.exists(task_file):
            try:
                # Wczytaj aktualne dane zadania
                with open(task_file, "r", encoding="utf-8") as f:
                    task_data = json.load(f)

                # Dodaj wyniki trainingu
                task_data["status"] = "Zakończony"
                task_data["model_filename"] = model_filename
                task_data["accuracy"] = accuracy
                task_data["epochs_trained"] = epochs_trained
                task_data["training_time"] = training_time
                task_data["training_time_str"] = str(
                    datetime.timedelta(seconds=int(training_time))
                )

                # Dodaj dodatkowe metryki jeśli są dostępne
                if "history" in result:
                    history = result["history"]
                    if "train_acc" in history:
                        task_data["train_accuracy"] = history["train_acc"][-1]
                    if "train_loss" in history:
                        task_data["train_loss"] = history["train_loss"][-1]
                    if "val_acc" in history:
                        task_data["validation_accuracy"] = history["val_acc"][-1]
                    if "val_loss" in history:
                        task_data["validation_loss"] = history["val_loss"][-1]

                # Zapisz zaktualizowane dane
                with open(task_file, "w", encoding="utf-8") as f:
                    json.dump(task_data, f, indent=4, ensure_ascii=False)

                self.parent.logger.info(
                    f"Zapisano wyniki trainingu do pliku: {task_file}"
                )
            except Exception as e:
                self.parent.logger.error(
                    f"Błąd podczas zapisywania wyników: {str(e)}"
                )
        else:
            self.parent.logger.error(f"Nie znaleziono pliku zadania: {task_file}")

        # Zapisz wykres treningu
        if hasattr(self, "training_visualization") and self.training_visualization:
            try:
                # Utwórz katalog na wykresy jeśli nie istnieje
                plots_dir = os.path.join("data", "plots")
                os.makedirs(plots_dir, exist_ok=True)

                # Generuj nazwę pliku wykresu
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
                plot_filename = f"{task_name}_{timestamp}.png"
                plot_path = os.path.join(plots_dir, plot_filename)

                # Zapisz wykres
                if self.training_visualization.save_plot(plot_path):
                    self.parent.logger.info(
                        f"Wykres treningu zapisany w: {plot_path}"
                    )
                    
                    # Dodaj informację o ścieżce do wykresu w pliku zadania
                    if os.path.exists(task_file):
                        try:
                            with open(task_file, "r", encoding="utf-8") as f:
                                task_data = json.load(f)
                            task_data["plot_path"] = plot_path
                            with open(task_file, "w", encoding="utf-8") as f:
                                json.dump(task_data, f, indent=4, ensure_ascii=False)
                        except Exception as e:
                            self.parent.logger.error(
                                f"Błąd podczas aktualizacji ścieżki wykresu: {str(e)}"
                            )
                    
                    # Reset wizualizacji po zapisaniu
                    self.training_visualization.clear_data()
                    self.training_visualization.reset_plot()
                else:
                    self.parent.logger.error(
                        "Nie udało się zapisać wykresu treningu"
                    )
            except Exception as plot_error:
                self.parent.logger.error(
                    f"Błąd podczas zapisywania wykresu: {plot_error}"
                )

        # Odśwież listę zadań
        self.refresh()

    except Exception as e:
        self.parent.logger.error(
            f"Błąd podczas obsługi zakończenia zadania: {str(e)}"
        )
        self.parent.logger.error(f"TRACEBACK: {traceback.format_exc()}")
        QtWidgets.QMessageBox.critical(
            self,
            "Błąd",
            f"Nie udało się zakończyć zadania: {str(e)}",
        )
Zmiana w pliku app/gui/dialogs/queue_manager.py
Musimy dodać funkcję ustawiającą i używającą wizualizacji treningu w klasie QueueManager:
pythonclass QueueManager(QDialog):
    # ... istniejący kod ...
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("Zarządzanie kolejką zadań")
        self.setMinimumSize(800, 600)
        self.new_tasks = []
        self.training_threads = []
        self.training_visualization = None  # Dodajemy referencję do wizualizacji
        self.setup_ui()
        self.load_new_tasks()
        
    def set_visualization_widget(self, visualization_widget):
        """Ustawia widget wizualizacji treningu."""
        self.training_visualization = visualization_widget
        
    def _handle_task_progress(self, task_name, progress, details):
        """Obsługuje aktualizacje postępu treningu."""
        try:
            # Aktualizacja paska postępu i szczegółów
            # ... istniejący kod ...
            
            # Aktualizacja wizualizacji treningu
            if self.training_visualization:
                # Pobierz i weryfikuj wartości metryk
                epoch = int(details.get("epoch", 0))
                train_loss = details.get("train_loss")
                train_acc = details.get("train_acc")
                val_loss = details.get("val_loss")
                val_acc = details.get("val_acc")
                val_top3 = details.get("val_top3")
                val_top5 = details.get("val_top5")
                val_precision = details.get("val_precision")
                val_recall = details.get("val_recall")
                val_f1 = details.get("val_f1")
                val_auc = details.get("val_auc")
                
                # Aktualizacja wizualizacji
                if epoch > 0:
                    self.training_visualization.update_data(
                        epoch=epoch,
                        train_loss=train_loss,
                        train_acc=train_acc,
                        val_loss=val_loss,
                        val_acc=val_acc,
                        val_top3=val_top3,
                        val_top5=val_top5,
                        val_precision=val_precision,
                        val_recall=val_recall,
                        val_f1=val_f1,
                        val_auc=val_auc
                    )
                    
        except Exception as e:
            print(f"Błąd w _handle_task_progress: {e}")
            import traceback
            print(traceback.format_exc())
            
    def _handle_task_completed(self, task_name, result):
        """Obsługuje zakończenie zadania treningowego."""
        try:
            # ... istniejący kod ...
            
            # Zapisz wykres treningu jeśli wizualizacja istnieje
            if self.training_visualization:
                try:
                    # Utwórz katalog na wykresy jeśli nie istnieje
                    plots_dir = os.path.join("data", "plots")
                    os.makedirs(plots_dir, exist_ok=True)

                    # Generuj nazwę pliku wykresu
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
                    plot_filename = f"{task_name}_{timestamp}.png"
                    plot_path = os.path.join(plots_dir, plot_filename)

                    # Zapisz wykres
                    if self.training_visualization.save_plot(plot_path):
                        print(f"Wykres treningu zapisany w: {plot_path}")
                        
                        # Dodaj informację o ścieżce do wykresu w pliku zadania
                        task_file = os.path.join("data", "tasks", f"{task_name}.json")
                        if os.path.exists(task_file):
                            try:
                                with open(task_file, "r", encoding="utf-8") as f:
                                    task_data = json.load(f)
                                task_data["plot_path"] = plot_path
                                with open(task_file, "w", encoding="utf-8") as f:
                                    json.dump(task_data, f, indent=4, ensure_ascii=False)
                            except Exception as e:
                                print(f"Błąd podczas aktualizacji ścieżki wykresu: {str(e)}")
                        
                        # Reset wizualizacji po zapisaniu
                        self.training_visualization.clear_data()
                        self.training_visualization.reset_plot()
                    else:
                        print("Nie udało się zapisać wykresu treningu")
                except Exception as plot_error:
                    print(f"Błąd podczas zapisywania wykresu: {plot_error}")
            
            # ... kontynuacja istniejącego kodu ...
        
        except Exception as e:
            print(f"Błąd podczas obsługi zakończenia zadania: {str(e)}")
            QMessageBox.critical(
                self,
                "Błąd",
                f"Nie udało się zakończyć zadania: {str(e)}",
            )
            
    def _setup_next_task(self):
        """Konfiguruje i uruchamia następne zadanie z kolejki."""
        try:
            # ... istniejący kod ...

            # Wyczyść dane wizualizacji przed rozpoczęciem nowego zadania
            if self.training_visualization:
                self.training_visualization.clear_data()
                self.training_visualization.reset_plot()
                
            # ... kontynuacja istniejącego kodu ...
        
        except Exception as e:
            print(f"Błąd podczas konfiguracji następnego zadania: {str(e)}")
            QMessageBox.critical(
                self,
                "Błąd",
                f"Nie udało się skonfigurować następnego zadania: {str(e)}",
            )
Zmiana w pliku app/core/workers/single_training_thread.py
Upewnijmy się, że SingleTrainingThread przekazuje wszystkie niezbędne dane wizualizacji:
pythondef _update_progress(self, epoch, total_epochs, metrics):
    """Aktualizuje postęp treningu."""
    try:
        if self._stop_requested:
            return

        progress = int((epoch / total_epochs) * 100) if total_epochs > 0 else 0
        
        # Zbieramy wszystkie dostępne metryki
        details = {
            "epoch": epoch,
            "total_epochs": total_epochs,
            "train_loss": metrics.get("train_loss"),
            "train_acc": metrics.get("train_acc"),
            "val_loss": metrics.get("val_loss"),
            "val_acc": metrics.get("val_acc"),
            "val_top3": metrics.get("val_top3"),
            "val_top5": metrics.get("val_top5"),
            "val_precision": metrics.get("val_precision"),
            "val_recall": metrics.get("val_recall"),
            "val_f1": metrics.get("val_f1"),
            "val_auc": metrics.get("val_auc"),
            "learning_rate": metrics.get("learning_rate")
        }
        
        self.task_progress.emit(self.task_name, progress, details)
        
    except Exception as e:
        self.error.emit(self.task_name, f"Błąd aktualizacji postępu: {str(e)}")
Podsumowanie zmian

Dodajemy przekazanie referencji do widgetu wizualizacji treningu z TrainingManager do QueueManager.
Dodajemy aktualizację wizualizacji podczas przetwarzania zadań w QueueManager.
Dodajemy zapisywanie wykresu treningu po zakończeniu zadania w QueueManager.
Upewniamy się, że SingleTrainingThread przekazuje wszystkie dostępne metryki do wizualizacji.
Dodajemy zapisywanie ścieżki do wygenerowanego wykresu w pliku konfiguracyjnym zadania.

Te zmiany powinny zapewnić, że queue_manager będzie prawidłowo zapisywał wizualizację treningu w taki sam sposób, jak dzieje się to w zakładce TrainingManager.