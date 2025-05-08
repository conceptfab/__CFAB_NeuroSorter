import json
import os
from pathlib import Path

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt

from app.core.workers.single_training_thread import SingleTrainingThread
from app.gui.widgets.training_visualization import TrainingVisualization


class QueueManager(QtWidgets.QDialog):
    def __init__(self, parent=None, settings=None):
        super().__init__(parent)
        self.setWindowTitle("Trening wsadowy")
        self.setMinimumSize(1200, 800)
        self.tasks_dir = Path("data/tasks")
        self.new_tasks = []
        self.current_task_index = 0
        self.training_thread = None  # Dodajemy atrybut dla wątku treningu

        # Layout główny
        main_layout = QtWidgets.QVBoxLayout(self)

        # Górna grupa: Lista zadań
        self.top_group = QtWidgets.QGroupBox()
        top_layout = QtWidgets.QVBoxLayout(self.top_group)

        # Przyciski kontrolne
        control_layout = QtWidgets.QHBoxLayout()
        self.refresh_button = QtWidgets.QPushButton("Odśwież")
        self.refresh_button.clicked.connect(self.load_new_tasks)
        control_layout.addWidget(self.refresh_button)
        top_layout.addLayout(control_layout)

        # Tabela zadań
        self.tasks_table = QtWidgets.QTableWidget()
        self.tasks_table.setColumnCount(4)
        self.tasks_table.setHorizontalHeaderLabels(
            ["Nazwa", "Typ", "Status", "Data utworzenia"]
        )
        self.tasks_table.horizontalHeader().setStretchLastSection(True)
        top_layout.addWidget(self.tasks_table)

        # Przyciski akcji
        action_layout = QtWidgets.QHBoxLayout()
        self.start_queue_button = QtWidgets.QPushButton("Uruchom kolejkę")
        self.stop_training_button = QtWidgets.QPushButton("Zatrzymaj trening")
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
        self.progress_bar.setMaximum(0)  # Będzie ustawione na podstawie liczby zadań
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
                        self.tasks_table.setItem(
                            row,
                            0,
                            QtWidgets.QTableWidgetItem(task_data.get("name", "")),
                        )
                        self.tasks_table.setItem(
                            row, 1, QtWidgets.QTableWidgetItem(task_data.get("typ", ""))
                        )
                        self.tasks_table.setItem(
                            row,
                            2,
                            QtWidgets.QTableWidgetItem(task_data.get("status", "")),
                        )
                        self.tasks_table.setItem(
                            row,
                            3,
                            QtWidgets.QTableWidgetItem(task_data.get("created_at", "")),
                        )
            except Exception as e:
                print(f"Błąd wczytywania pliku {task_file}: {e}")

        self.tasks_table.resizeColumnsToContents()
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
        self.update_training_progress(
            epoch=details.get("epoch", 0),
            train_loss=details.get("train_loss"),
            train_acc=details.get("train_acc"),
            val_loss=details.get("val_loss"),
            val_acc=details.get("val_acc"),
            val_top3=details.get("val_top3"),
            val_top5=details.get("val_top5"),
            val_precision=details.get("val_precision"),
            val_recall=details.get("val_recall"),
            val_f1=details.get("val_f1"),
            val_auc=details.get("val_auc"),
        )

    def _on_task_completed(self, task_name, result):
        """Obsługa zakończenia zadania."""
        print(f"Zakończono zadanie: {task_name}")

        # Aktualizuj status zadania w pliku
        task_file = os.path.join("data", "tasks", f"{task_name}.json")
        if os.path.exists(task_file):
            try:
                with open(task_file, "r", encoding="utf-8") as f:
                    task_data = json.load(f)
                task_data["status"] = "Zakończony"
                with open(task_file, "w", encoding="utf-8") as f:
                    json.dump(task_data, f, indent=4)
            except Exception as e:
                print(f"Błąd aktualizacji statusu zadania: {e}")

        # Przejdź do następnego zadania
        self.current_task_index += 1
        self.update_progress_bar()

        if self.current_task_index < len(self.new_tasks):
            # Uruchom następne zadanie
            next_task = self.new_tasks[self.current_task_index]
            self.training_thread = SingleTrainingThread(next_task)
            self.training_thread.task_started.connect(self._on_task_started)
            self.training_thread.task_progress.connect(self._on_task_progress)
            self.training_thread.task_completed.connect(self._on_task_completed)
            self.training_thread.error.connect(self._on_task_error)
            self.training_thread.start()
        else:
            # Wszystkie zadania zakończone
            print("Wszystkie zadania zostały zakończone")
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
            self.training_thread.stop()  # Zatrzymaj aktualny wątek
            self.training_thread.wait()  # Poczekaj na zakończenie wątku

            # Aktualizuj status przerwanego zadania
            if self.current_task_index < len(self.new_tasks):
                task_name = self.new_tasks[self.current_task_index].get("name")
                task_file = os.path.join("data", "tasks", f"{task_name}.json")
                if os.path.exists(task_file):
                    try:
                        with open(task_file, "r", encoding="utf-8") as f:
                            task_data = json.load(f)
                        task_data["status"] = "Przerwany"
                        with open(task_file, "w", encoding="utf-8") as f:
                            json.dump(task_data, f, indent=4)
                    except Exception as e:
                        print(f"Błąd aktualizacji statusu zadania: {e}")

            # Resetuj interfejs
            self.start_queue_button.setEnabled(True)
            self.stop_training_button.setEnabled(False)
            self.training_visualization.clear_data()
            self.current_task_index = len(self.new_tasks)  # Zakończ kolejkę
            self.update_progress_bar()
            print("Trening został zatrzymany")

    def next_task(self):
        """Przechodzi do następnego zadania w kolejce."""
        if self.current_task_index < len(self.new_tasks):
            self.current_task_index += 1
            self.update_progress_bar()
            return self.new_tasks[self.current_task_index - 1]
        return None

    def update_training_progress(
        self,
        epoch,
        train_loss,
        train_acc,
        val_loss=None,
        val_acc=None,
        val_top3=None,
        val_top5=None,
        val_precision=None,
        val_recall=None,
        val_f1=None,
        val_auc=None,
    ):
        """Aktualizuje wizualizację treningu."""
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
