import json
import logging
import os
from pathlib import Path

from PyQt6 import QtCore, QtWidgets


class QueueManager(QtWidgets.QDialog):
    """Klasa zarządzająca kolejką zadań treningowych."""

    def __init__(self, parent=None, settings=None):
        super().__init__(parent)
        self.parent = parent
        self.settings = settings
        self.logger = logging.getLogger("QueueManager")
        self.logger.setLevel(logging.INFO)

        self.setup_ui()
        self.connect_signals()
        self.refresh()

    def setup_ui(self):
        """Tworzy i konfiguruje elementy interfejsu."""
        self.setWindowTitle("Zarządzanie kolejką zadań")
        self.setMinimumSize(800, 600)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Nagłówek
        header = QtWidgets.QLabel("KOLEJKA ZADAŃ TRENINGOWYCH")
        header.setStyleSheet(
            "font-weight: bold; color: #CCCCCC; "
            "font-size: 11px; padding-bottom: 4px;"
        )
        layout.addWidget(header)

        # Tabela zadań
        self.tasks_table = QtWidgets.QTableWidget()
        self.tasks_table.setColumnCount(11)
        self.tasks_table.setHorizontalHeaderLabels(
            [
                "Nazwa",
                "Typ",
                "Status",
                "Priorytet",
                "Utworzono",
                "Czas treningu",
                "Dokładność treningu",
                "Strata treningu",
                "Dokładność walidacji",
                "Strata walidacji",
                "Akcje",
            ]
        )
        self.tasks_table.horizontalHeader().setStretchLastSection(True)
        self.tasks_table.setSelectionBehavior(
            QtWidgets.QTableWidget.SelectionBehavior.SelectRows
        )
        layout.addWidget(self.tasks_table, 1)

        # Przyciski zarządzania
        buttons_layout = QtWidgets.QHBoxLayout()

        self.refresh_btn = QtWidgets.QPushButton("Odśwież")
        self.refresh_btn.setFixedHeight(24)
        buttons_layout.addWidget(self.refresh_btn)

        self.start_btn = QtWidgets.QPushButton("Uruchom kolejkę")
        self.start_btn.setFixedHeight(24)
        buttons_layout.addWidget(self.start_btn)

        self.clear_btn = QtWidgets.QPushButton("Wyczyść kolejkę")
        self.clear_btn.setFixedHeight(24)
        buttons_layout.addWidget(self.clear_btn)

        buttons_layout.addStretch()
        layout.addLayout(buttons_layout)

    def connect_signals(self):
        """Podłącza sygnały do slotów."""
        self.refresh_btn.clicked.connect(self.refresh)
        self.start_btn.clicked.connect(self.start_queue)
        self.clear_btn.clicked.connect(self.clear_queue)

    def refresh(self):
        """Odświeża listę zadań w kolejce."""
        try:
            self.tasks_table.setRowCount(0)
            tasks_dir = Path("data/tasks")
            tasks_dir.mkdir(parents=True, exist_ok=True)

            for task_file in sorted(tasks_dir.glob("*.json")):
                try:
                    with open(task_file, "r", encoding="utf-8") as f:
                        task_data = json.load(f)

                    row = self.tasks_table.rowCount()
                    self.tasks_table.insertRow(row)
                    self.tasks_table.setRowHeight(row, 38)

                    # Nazwa zadania
                    task_name = task_data.get("name", task_file.stem)
                    self.tasks_table.setItem(
                        row, 0, QtWidgets.QTableWidgetItem(task_name)
                    )

                    # Typ zadania
                    task_type = task_data.get("typ", "trening")
                    self.tasks_table.setItem(
                        row, 1, QtWidgets.QTableWidgetItem(task_type)
                    )

                    # Status
                    status = task_data.get("status", "Nowy")
                    self.tasks_table.setItem(
                        row, 2, QtWidgets.QTableWidgetItem(status)
                    )

                    # Priorytet
                    priority = task_data.get("priority", 0)
                    self.tasks_table.setItem(
                        row, 3, QtWidgets.QTableWidgetItem(str(priority))
                    )

                    # Data utworzenia
                    created_at = task_data.get("created_at", "")
                    self.tasks_table.setItem(
                        row, 4, QtWidgets.QTableWidgetItem(created_at)
                    )

                    # Przyciski akcji
                    action_widget = QtWidgets.QWidget()
                    action_layout = QtWidgets.QHBoxLayout(action_widget)
                    action_layout.setContentsMargins(0, 0, 0, 0)
                    action_layout.setAlignment(
                        QtCore.Qt.AlignmentFlag.AlignVCenter
                    )

                    run_btn = QtWidgets.QPushButton("Uruchom")
                    run_btn.setFixedWidth(100)
                    run_btn.setFixedHeight(20)
                    run_btn.clicked.connect(
                        lambda checked, file=str(task_file): self.run_task(file)
                    )

                    edit_btn = QtWidgets.QPushButton("Edytuj")
                    edit_btn.setFixedWidth(80)
                    edit_btn.setFixedHeight(20)
                    edit_btn.clicked.connect(
                        lambda checked, file=str(task_file): self.edit_task(file)
                    )

                    delete_btn = QtWidgets.QPushButton("Usuń")
                    delete_btn.setFixedWidth(80)
                    delete_btn.setFixedHeight(20)
                    delete_btn.clicked.connect(
                        lambda checked, file=str(task_file): self.delete_task(file)
                    )

                    action_layout.addWidget(run_btn)
                    action_layout.addWidget(edit_btn)
                    action_layout.addWidget(delete_btn)
                    action_layout.addStretch()

                    self.tasks_table.setCellWidget(row, 10, action_widget)

                    # Dodaj wyniki treningu jeśli zadanie jest zakończone
                    if status == "Zakończony":
                        # Czas treningu
                        training_time = task_data.get("training_time_str", "")
                        self.tasks_table.setItem(
                            row, 5, QtWidgets.QTableWidgetItem(training_time)
                        )

                        # Dokładność treningu
                        train_acc = task_data.get("train_accuracy", 0)
                        self.tasks_table.setItem(
                            row, 6, QtWidgets.QTableWidgetItem(f"{train_acc:.2f}")
                        )

                        # Strata treningu
                        train_loss = task_data.get("train_loss", 0)
                        self.tasks_table.setItem(
                            row, 7, QtWidgets.QTableWidgetItem(f"{train_loss:.2f}")
                        )

                        # Dokładność walidacji
                        val_acc = task_data.get("validation_accuracy", 0)
                        self.tasks_table.setItem(
                            row, 8, QtWidgets.QTableWidgetItem(f"{val_acc:.2f}")
                        )

                        # Strata walidacji
                        val_loss = task_data.get("validation_loss", 0)
                        self.tasks_table.setItem(
                            row, 9, QtWidgets.QTableWidgetItem(f"{val_loss:.2f}")
                        )

                except Exception as e:
                    self.logger.error(
                        f"Błąd podczas wczytywania zadania {task_file}: {str(e)}"
                    )

            # Dostosuj szerokość kolumn
            self.tasks_table.resizeColumnsToContents()
            header = self.tasks_table.horizontalHeader()
            for col in range(self.tasks_table.columnCount()):
                width = self.tasks_table.columnWidth(col)
                if col == 0:  # Kolumna "Nazwa"
                    new_width = int(width * 1.15)
                elif col == 5:  # Kolumna "Czas treningu"
                    new_width = int(width * 1.4)
                elif col in [6, 7, 8, 9]:  # Kolumny z wynikami treningu
                    new_width = int(width * 1.0)
                else:
                    new_width = int(width * 1.6)
                self.tasks_table.setColumnWidth(col, new_width)

            # Zmniejsz kolumnę 'Akcje' (ostatnia) do odpowiedniej szerokości
            last_col = self.tasks_table.columnCount() - 1
            self.tasks_table.setColumnWidth(last_col, 250)
            header.setStretchLastSection(False)

        except Exception as e:
            self.logger.error(f"Błąd podczas odświeżania kolejki: {str(e)}")
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Nie udało się odświeżyć kolejki: {str(e)}"
            )

    def start_queue(self):
        """Uruchamia kolejkę zadań."""
        try:
            # Pobierz listę zadań do wykonania
            task_files = []
            for row in range(self.tasks_table.rowCount()):
                task_name = self.tasks_table.item(row, 0).text()
                status = self.tasks_table.item(row, 2).text()

                if status == "Nowy":
                    task_file = os.path.join("data", "tasks", f"{task_name}.json")
                    if os.path.exists(task_file):
                        task_files.append(task_file)

            if not task_files:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Brak zadań",
                    "Nie znaleziono żadnych nowych zadań do wykonania.",
                )
                return None

            # Zamknij okno i zwróć listę zadań
            self.accept()
            return task_files

        except Exception as e:
            self.logger.error(f"Błąd podczas uruchamiania kolejki: {str(e)}")
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Nie udało się uruchomić kolejki: {str(e)}"
            )
            return None

    def clear_queue(self):
        """Czyści kolejkę zadań."""
        reply = QtWidgets.QMessageBox.question(
            self,
            "Potwierdzenie",
            "Czy na pewno chcesz usunąć wszystkie zadania z kolejki?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )

        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        try:
            tasks_dir = Path("data/tasks")
            if not tasks_dir.exists():
                return

            for task_file in tasks_dir.glob("*.json"):
                try:
                    task_file.unlink()
                except Exception as e:
                    self.logger.error(
                        f"Błąd podczas usuwania zadania {task_file}: {str(e)}"
                    )

            self.refresh()
            QtWidgets.QMessageBox.information(
                self,
                "Kolejka wyczyszczona",
                "Wszystkie zadania zostały usunięte z kolejki.",
            )

        except Exception as e:
            self.logger.error(f"Błąd podczas czyszczenia kolejki: {str(e)}")
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Wystąpił błąd podczas czyszczenia kolejki: {str(e)}"
            )

    def run_task(self, task_file):
        """Uruchamia pojedyncze zadanie."""
        try:
            # Emituj sygnał z pojedynczym zadaniem
            self.accept()
            return [task_file]

        except Exception as e:
            self.logger.error(f"Błąd podczas uruchamiania zadania: {str(e)}")
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Nie udało się uruchomić zadania: {str(e)}"
            )
            return None

    def edit_task(self, task_file):
        """Otwiera plik zadania w domyślnym edytorze."""
        try:
            import platform
            import subprocess

            if platform.system() == "Windows":
                os.startfile(task_file)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", task_file])
            else:  # Linux
                subprocess.run(["xdg-open", task_file])

        except Exception as e:
            self.logger.error(f"Błąd podczas otwierania pliku: {str(e)}")
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Nie udało się otworzyć pliku w edytorze: {str(e)}"
            )

    def delete_task(self, task_file):
        """Usuwa pojedyncze zadanie z kolejki."""
        try:
            # Pobierz nazwę zadania z pliku
            with open(task_file, "r", encoding="utf-8") as f:
                task_data = json.load(f)
                task_name = task_data.get("name", os.path.basename(task_file))

            # Potwierdzenie usunięcia
            reply = QtWidgets.QMessageBox.question(
                self,
                "Potwierdzenie",
                f"Czy na pewno chcesz usunąć zadanie '{task_name}'?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )

            if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                return

            # Usuń plik zadania
            os.remove(task_file)

            # Odśwież listę zadań
            self.refresh()

            # Wyświetl potwierdzenie
            QtWidgets.QMessageBox.information(
                self,
                "Zadanie usunięte",
                f"Zadanie '{task_name}' zostało usunięte z kolejki.",
            )

        except Exception as e:
            self.logger.error(f"Błąd podczas usuwania zadania: {str(e)}")
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Wystąpił błąd podczas usuwania zadania: {str(e)}"
            )
