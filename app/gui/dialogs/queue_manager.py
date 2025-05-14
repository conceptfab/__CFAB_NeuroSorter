import datetime
import json
import os
from pathlib import Path

from PyQt6 import QtWidgets

from app.core.workers.single_training_thread import SingleTrainingThread
from app.gui.widgets.training_visualization import TrainingVisualization

# Usunięto nieużywany import Qt
# from PyQt6.QtCore import Qt


class QueueManager(QtWidgets.QDialog):
    def __init__(self, parent=None, settings=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("Trening wsadowy")
        self.setMinimumSize(1200, 800)
        self.tasks_dir = Path("data/tasks")
        self.new_tasks = []
        self.current_task_index = 0
        self.training_thread = None
        self.training_visualization = None  # Dodajemy referencję do wizualizacji

        # Style z MainWindow
        primary_color = "#007ACC"
        success_color = "#10B981"
        warning_color = "#DC2626"
        background = "#1E1E1E"
        surface = "#252526"
        border_color = "#3F3F46"
        text_color = "#CCCCCC"

        self.setStyleSheet(
            f"""
            QDialog {{
                background-color: {background};
                color: {text_color};
            }}
            QGroupBox {{
                background-color: {surface};
                color: {text_color};
                border: 1px solid {border_color};
                border-radius: 2px;
                margin-top: 14px; /* Aby tytuł nie nachodził na ramkę */
                padding-top: 14px; /* Dodatkowy padding dla zawartości grupy */
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 5px;
                color: #FFFFFF; /* Jaśniejszy kolor dla tytułu grupy */
            }}
            QPushButton {{
                background-color: {surface};
                color: {text_color};
                border: 1px solid {border_color};
                border-radius: 2px;
                padding: 4px 12px;
                min-height: 24px;
                max-height: 24px;
            }}
            QPushButton:hover {{
                background-color: #2A2D2E; /* Ciemniejszy hover */
            }}
            QPushButton:pressed {{
                background-color: #3E3E40; /* Jeszcze ciemniejszy przy wciśnięciu */
            }}
            QPushButton[action="success"] {{
                background-color: {success_color};
                color: white;
                border: none;
            }}
            QPushButton[action="success"]:hover {{
                background-color: #34D399; /* Jaśniejszy zielony hover */
            }}
            QPushButton[action="success"]:pressed {{
                background-color: #059669; /* Ciemniejszy zielony pressed */
            }}
            QPushButton[action="warning"] {{
                background-color: {warning_color};
                color: white;
                border: none;
            }}
            QPushButton[action="warning"]:hover {{
                background-color: #EF4444; /* Jaśniejszy czerwony hover */
            }}
            QPushButton[action="warning"]:pressed {{
                background-color: #B91C1C; /* Ciemniejszy czerwony pressed */
            }}
            QTableWidget {{
                background-color: #1C1C1C; /* Ciemniejsze tło tabeli */
                color: {text_color};
                border: 1px solid {border_color};
                gridline-color: {border_color}; /* Kolor linii siatki */
            }}
            QTableWidget::item:selected {{
                background-color: #264F78; /* Kolor zaznaczenia VS Code */
                color: #FFFFFF;
            }}
            QHeaderView::section {{
                background-color: {surface};
                color: {text_color};
                padding: 2px;
                border: 1px solid {border_color};
            }}
            QProgressBar {{
                border: 1px solid {border_color};
                background-color: {surface};
                text-align: center;
                color: {text_color};
            }}
            QProgressBar::chunk {{
                background-color: {primary_color};
            }}
            QLabel {{
                color: {text_color};
            }}
        """
        )

        # Layout główny
        main_layout = QtWidgets.QVBoxLayout(self)

        # Górna grupa: Lista zadań
        self.top_group = QtWidgets.QGroupBox("Lista zadań")  # Dodano tytuł
        top_layout = QtWidgets.QVBoxLayout(self.top_group)

        # Przyciski kontrolne
        control_layout = QtWidgets.QHBoxLayout()
        self.refresh_button = QtWidgets.QPushButton("Odśwież")
        self.refresh_button.clicked.connect(self.load_new_tasks)
        control_layout.addWidget(self.refresh_button)
        top_layout.addLayout(control_layout)

        # Tabela zadań
        self.tasks_table = QtWidgets.QTableWidget()
        self.tasks_table.setColumnCount(8)
        self.tasks_table.setHorizontalHeaderLabels(
            [
                "Nazwa",
                "type",
                "variant",
                "epochs",
                "batch_size",
                "num_workers",
                "train_dir",
                "val_dir",
            ]
        )
        self.tasks_table.horizontalHeader().setStretchLastSection(True)
        top_layout.addWidget(self.tasks_table)

        # Przyciski akcji
        action_layout = QtWidgets.QHBoxLayout()
        self.start_queue_button = QtWidgets.QPushButton("Uruchom kolejkę")
        self.stop_training_button = QtWidgets.QPushButton("Zatrzymaj trening")

        # Ustawienie właściwości "action" dla stylizacji
        self.start_queue_button.setProperty("action", "success")
        self.stop_training_button.setProperty("action", "warning")
        # Usunięto bezpośrednie ustawianiesetStyleSheet, bo jest teraz globalnie
        # self.start_queue_button.setStyleSheet("background-color: green")
        # self.stop_training_button.setStyleSheet("background-color: red")

        self.start_queue_button.clicked.connect(self.start_queue)
        self.stop_training_button.clicked.connect(self.stop_training)
        action_layout.addWidget(self.start_queue_button)
        action_layout.addWidget(self.stop_training_button)
        top_layout.addLayout(action_layout)

        main_layout.addWidget(self.top_group, stretch=1)

        # Progress bar
        progress_layout = QtWidgets.QHBoxLayout()
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setMinimum(0)
        # Będzie ustawione na podstawie liczby zadań
        self.progress_bar.setMaximum(0)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Zadanie %v z %m (%p%)")
        progress_layout.addWidget(self.progress_bar)
        main_layout.addLayout(progress_layout)

        # Dolna grupa: Wizualizacja treningu
        self.bottom_group = QtWidgets.QGroupBox("Wizualizacja treningu")
        bottom_layout = QtWidgets.QVBoxLayout(self.bottom_group)
        self.training_visualization = TrainingVisualization(
            parent=self, settings=settings
        )
        bottom_layout.addWidget(self.training_visualization)
        main_layout.addWidget(self.bottom_group, stretch=1)

        # Wczytaj zadania przy starcie
        self.load_new_tasks()

    def load_new_tasks(self):
        """Wczytuje zadania ze statusem 'Nowy' z katalogu tasks."""
        self.new_tasks = []
        self.tasks_table.setRowCount(0)
        self.current_task_index = 0

        if not self.tasks_dir.exists():
            return

        for task_file in self.tasks_dir.glob("*.json"):
            try:
                with open(task_file, "r", encoding="utf-8") as f:
                    task_data = json.load(f)
                    if task_data.get("status") == "Nowy":
                        self.new_tasks.append(task_data)
                        row = self.tasks_table.rowCount()
                        self.tasks_table.insertRow(row)
                        # Pobieranie zagnieżdżonych wartości
                        config = task_data.get("config", {})
                        model = config.get("model", {})
                        training = config.get("training", {})
                        item_name = QtWidgets.QTableWidgetItem(
                            task_data.get("name", "")
                        )
                        self.tasks_table.setItem(row, 0, item_name)
                        item_type = QtWidgets.QTableWidgetItem(
                            str(task_data.get("type", ""))
                        )
                        self.tasks_table.setItem(row, 1, item_type)
                        item_variant = QtWidgets.QTableWidgetItem(
                            str(model.get("variant", ""))
                        )
                        self.tasks_table.setItem(row, 2, item_variant)
                        item_epochs = QtWidgets.QTableWidgetItem(
                            str(training.get("epochs", ""))
                        )
                        self.tasks_table.setItem(row, 3, item_epochs)
                        item_batch_size = QtWidgets.QTableWidgetItem(
                            str(training.get("batch_size", ""))
                        )
                        self.tasks_table.setItem(row, 4, item_batch_size)
                        item_num_workers = QtWidgets.QTableWidgetItem(
                            str(training.get("num_workers", ""))
                        )
                        self.tasks_table.setItem(row, 5, item_num_workers)
                        item_train_dir = QtWidgets.QTableWidgetItem(
                            str(config.get("train_dir", ""))
                        )
                        self.tasks_table.setItem(row, 6, item_train_dir)
                        item_val_dir = QtWidgets.QTableWidgetItem(
                            str(config.get("val_dir", ""))
                        )
                        self.tasks_table.setItem(row, 7, item_val_dir)
            except Exception as e:
                print(f"Błąd wczytywania pliku {task_file}: {e}")

        self.tasks_table.resizeColumnsToContents()
        # Zwiększ szerokość wybranych kolumn o 20%
        for col in [0, 1, 2, 6]:
            current_width = self.tasks_table.columnWidth(col)
            self.tasks_table.setColumnWidth(col, int(current_width * 1.2))
        self.update_progress_bar()

    def update_progress_bar(self):
        """Aktualizuje ustawienia progress bara na podstawie liczby zadań."""
        total_tasks = len(self.new_tasks)
        self.progress_bar.setMaximum(total_tasks)
        self.progress_bar.setValue(self.current_task_index)

    def start_queue(self):
        """Uruchamia kolejkę zadań."""
        if not self.new_tasks:
            print("Brak zadań do wykonania")
            return

        self.current_task_index = 0
        self.update_progress_bar()
        self.training_visualization.clear_data()
        print("Uruchamianie kolejki zadań...")

        # Utwórz i skonfiguruj wątek dla pierwszego zadania
        self.training_thread = SingleTrainingThread(self.new_tasks[0])

        # Podłącz sygnały
        self.training_thread.task_started.connect(self._on_task_started)
        self.training_thread.task_progress.connect(self._on_task_progress)
        self.training_thread.task_completed.connect(self._on_task_completed)
        self.training_thread.error.connect(self._on_task_error)

        # Uruchom wątek
        self.training_thread.start()

    def _on_task_started(self, task_name, task_type):
        """Obsługa rozpoczęcia zadania."""
        print(f"Rozpoczęto zadanie: {task_name} ({task_type})")
        self.start_queue_button.setEnabled(False)
        self.stop_training_button.setEnabled(True)

    def _on_task_progress(self, task_name, progress, details):
        """Obsługa postępu zadania."""
        # Aktualizuj wizualizację
        if self.training_visualization:
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
                    val_auc=val_auc,
                )
            # --- Early stopping ---
            patience_counter = details.get("patience_counter")
            patience_max = details.get("patience_max")
            if patience_counter is not None and patience_max is not None:
                self.training_visualization.update_early_stopping_status(
                    patience_counter=patience_counter,
                    patience_max=patience_max
                )

    def _on_task_completed(self, task_name, result):
        """Obsługa zakończenia zadania."""
        self.parent.logger.info(f"Zakończono zadanie: {task_name}")
        # Aktualizuj status zadania w pliku
        base_task_name = task_name.replace(".json", "")
        task_file = os.path.join("data", "tasks", f"{base_task_name}.json")
        if os.path.exists(task_file):
            try:
                with open(task_file, "r", encoding="utf-8") as f:
                    task_data = json.load(f)
                task_data["status"] = "Zakończony"
                with open(task_file, "w", encoding="utf-8") as f:
                    json.dump(task_data, f, indent=4)
            except Exception as e:
                self.parent.logger.error(f"Błąd aktualizacji statusu zadania: {e}")
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
                    self.parent.logger.info(f"Wykres treningu zapisany w: {plot_path}")
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
                    self.parent.logger.error("Nie udało się zapisać wykresu treningu")
            except Exception as plot_error:
                self.parent.logger.error(
                    f"Błąd podczas zapisywania wykresu: {plot_error}"
                )
        # Przejdź do następnego zadania
        self.current_task_index += 1
        self.update_progress_bar()

        # Bardzo czytelny komunikat do konsoli z nazwami pozostałych zadań
        remaining = len(self.new_tasks) - self.current_task_index
        if remaining > 0:
            pozostale_nazwy = [
                task.get("name", "BRAK_NAZWY")
                for task in self.new_tasks[self.current_task_index :]
            ]
            print("=" * 60)
            print(f"===   POZOSTAŁO {remaining} ZADAŃ W KOLEJCE   ===")
            print(f"===   NAZWY POZOSTAŁYCH ZADAŃ:")
            for nazwa in pozostale_nazwy:
                print(f"===     - {nazwa}")
            print("=" * 60)
        else:
            print("=" * 60)
            print("===   KOLEJKA ZADAŃ ZOSTAŁA WYKONANA   ===")
            print("=" * 60)

        # Zabezpieczenie: sprawdź, czy są kolejne zadania i uruchom następne
        if self.current_task_index < len(self.new_tasks):
            try:
                next_task = self.new_tasks[self.current_task_index]
                self.training_thread = SingleTrainingThread(next_task)
                self.training_thread.task_started.connect(self._on_task_started)
                self.training_thread.task_progress.connect(self._on_task_progress)
                self.training_thread.task_completed.connect(self._on_task_completed)
                self.training_thread.error.connect(self._on_task_error)
                self.training_thread.start()
                self.parent.logger.info(
                    f"Uruchomiono kolejne zadanie: "
                    f"{next_task.get('name', 'brak nazwy')}"
                )
            except Exception as e:
                self.parent.logger.error(
                    f"Błąd przy uruchamianiu kolejnego zadania: {e}"
                )
        else:
            # Wszystkie zadania zakończone
            self.parent.logger.info("Wszystkie zadania zostały zakończone")
            self.start_queue_button.setEnabled(True)
            self.stop_training_button.setEnabled(False)
            self.training_visualization.clear_data()

    def _on_task_error(self, task_name, error_message):
        """Obsługa błędu zadania."""
        print(f"Błąd w zadaniu {task_name}: {error_message}")
        self.start_queue_button.setEnabled(True)
        self.stop_training_button.setEnabled(False)
        # Zatrzymaj kolejkę w przypadku błędu
        self.current_task_index = len(self.new_tasks)
        self.update_progress_bar()

    def stop_training(self):
        """Zatrzymuje aktualnie wykonywany trening i całą kolejkę."""
        if self.training_thread and self.training_thread.isRunning():
            print("Zatrzymywanie treningu...")
            self.training_thread.stop()
            self.training_thread.wait()

            if self.current_task_index < len(self.new_tasks):
                task_name_to_update = self.new_tasks[self.current_task_index].get(
                    "name"
                )
                # Poprawiono długość linii
                base_task_name_to_update = task_name_to_update.replace(".json", "")
                task_file_to_update = os.path.join(
                    "data", "tasks", f"{base_task_name_to_update}.json"
                )
                if os.path.exists(task_file_to_update):
                    try:
                        with open(task_file_to_update, "r", encoding="utf-8") as f:
                            task_data = json.load(f)
                        task_data["status"] = "Przerwany"
                        with open(task_file_to_update, "w", encoding="utf-8") as f:
                            json.dump(task_data, f, indent=4)
                    except Exception as e:
                        print(f"Błąd aktualizacji statusu zadania: {e}")

            self.start_queue_button.setEnabled(True)
            self.stop_training_button.setEnabled(False)
            self.training_visualization.clear_data()
            self.current_task_index = len(self.new_tasks)
            self.update_progress_bar()
            print("Trening został zatrzymany")

    def next_task(self):
        """Przechodzi do następnego zadania w kolejce."""
        if self.current_task_index < len(self.new_tasks):
            self.current_task_index += 1
            self.update_progress_bar()
            return self.new_tasks[self.current_task_index - 1]
        return None
