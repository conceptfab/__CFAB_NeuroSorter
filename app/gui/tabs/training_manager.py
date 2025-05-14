import datetime
import glob
import json
import logging
import os
import traceback

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt

from app.core.workers.single_training_thread import SingleTrainingThread
from app.gui.dialogs.fine_tuning_task_config_dialog import FineTuningTaskConfigDialog
from app.gui.dialogs.queue_manager import QueueManager
from app.gui.dialogs.training_task_config_dialog import TrainingTaskConfigDialog
from app.gui.tab_interface import TabInterface
from app.gui.widgets.training_visualization import TrainingVisualization

# from pathlib import Path # Usunięto nieużywany import


# Usunięto nieużywane importy z app.utils
# from app.utils.config import DEFAULT_TRAINING_PARAMS
# from app.utils.file_utils import (
#     validate_task_config,
#     validate_task_file,
#     validate_training_directory,
#     validate_validation_directory,
# )


class TrainingManager(QtWidgets.QWidget, TabInterface):
    """Klasa zarządzająca zakładką trainingu."""

    def __init__(self, parent=None, settings=None):
        self.logger = logging.getLogger("TrainingManager")
        self.logger.setLevel(logging.INFO)
        super().__init__(parent)
        self.parent = parent
        self.settings = settings
        self.training_thread = None
        self.queue_manager = QueueManager(self)
        self.setup_ui()
        self.connect_signals()

        # Automatyczne odświeżenie listy zadań przy starcie
        self.refresh()

    def setup_ui(self):
        """Tworzy i konfiguruje elementy interfejsu zakładki."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Dodaj etykietę informacji o profilu sprzętowym
        profile_layout = QtWidgets.QHBoxLayout()

        # Lewa kolumna - panel dodawania zadania
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Nagłówek sekcji dodawania zadania
        add_task_header = QtWidgets.QLabel("DODAJ NOWE ZADANIE TRENINGOWE")
        add_task_header.setStyleSheet(
            "font-weight: bold; color: #CCCCCC; "
            "font-size: 11px; padding-bottom: 4px;"
        )
        add_task_header.setFixedHeight(20)  # Ustaw stałą wysokość
        left_layout.addWidget(add_task_header)

        # Kontener na elementy w jednej linii
        controls_container = QtWidgets.QWidget()
        controls_layout = QtWidgets.QHBoxLayout(controls_container)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)

        # Wybór typu zadania
        task_type_label = QtWidgets.QLabel("Typ zadania:")
        task_type_label.setFixedWidth(120)
        task_type_label.setStyleSheet("font-size: 13px;")  # Zwiększamy rozmiar czcionki
        self.task_type_combo = QtWidgets.QComboBox()
        self.task_type_combo.addItems(
            ["Trening nowego modelu", "Doszkalanie istniejącego modelu"]
        )

        # Przycisk dodania zadania
        self.add_task_btn = QtWidgets.QPushButton("Dodaj zadanie do kolejki")
        self.add_task_btn.setFixedHeight(24)
        self.add_task_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """
        )

        # Dodaj wszystkie elementy do layoutu poziomego
        controls_layout.addWidget(task_type_label)
        controls_layout.addWidget(self.task_type_combo)
        controls_layout.addWidget(self.add_task_btn)
        controls_layout.addStretch(1)  # Dodaj rozciągnięcie na końcu

        # Dodaj kontener do głównego layoutu
        left_layout.addWidget(controls_container)

        # Prawa kolumna - informacje o profilu i optymalizacji
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Nagłówek sekcji
        optimization_header = QtWidgets.QLabel("OPTYMALIZACJA TRENINGU")
        optimization_header.setStyleSheet(
            "font-weight: bold; color: #CCCCCC; "
            "font-size: 11px; padding-bottom: 4px;"
        )
        optimization_header.setFixedHeight(20)  # Ustaw stałą wysokość
        right_layout.addWidget(optimization_header)

        self.profile_info_label = QtWidgets.QLabel("Status profilu: Nieznany")
        self.profile_info_label.setStyleSheet("color: #CCCCCC; padding: 4px;")
        right_layout.addWidget(self.profile_info_label)

        # Checkbox optymalizacji
        self.use_optimization_checkbox = QtWidgets.QCheckBox(
            "Używaj optymalizacji sprzętowej"
        )
        self.use_optimization_checkbox.setChecked(True)
        self.use_optimization_checkbox.setToolTip(
            "Automatycznie dobiera parametry trainingu "
            "na podstawie profilu sprzętowego"
        )
        right_layout.addWidget(self.use_optimization_checkbox)

        # Przycisk profilowania
        self.run_profiler_btn = QtWidgets.QPushButton("Uruchom profilowanie sprzętu")
        self.run_profiler_btn.setFixedHeight(24)
        self.run_profiler_btn.clicked.connect(self._run_profiler)
        right_layout.addWidget(self.run_profiler_btn)

        # Dodaj obie kolumny do głównego layoutu
        profile_layout.addWidget(left_panel)
        profile_layout.addWidget(right_panel)
        layout.addLayout(profile_layout)

        # === Początek zmian: Dodanie QTabWidget ===
        self.tabs = QtWidgets.QTabWidget()

        # Zakładka 1: Kolejka zadań trainingowych
        queue_panel_widget = self._create_queue_panel_widget()
        self.tabs.addTab(queue_panel_widget, "Kolejka zadań trainingowych")

        # Zakładka 2: Wizualizacja trainingu
        self.training_visualization = TrainingVisualization(
            parent=self, settings=self.settings
        )
        self.tabs.addTab(self.training_visualization, "Wizualizacja trainingu")

        layout.addWidget(
            self.tabs, 1
        )  # Dodajemy QTabWidget do głównego layoutu, rozciągalny
        # === Koniec zmian: Dodanie QTabWidget ===

    def connect_signals(self):
        """Łączy sygnały z odpowiednimi slotami."""
        self.add_task_btn.clicked.connect(self._add_training_task)
        self.task_type_combo.currentTextChanged.connect(self._on_task_type_changed)
        self.refresh_queue_btn.clicked.connect(self.refresh)
        self.clear_queue_btn.clicked.connect(self._clear_task_queue)

    def _on_task_type_changed(self, task_type):
        """Obsługuje zmianę typu zadania."""
        try:
            # Aktualizuj interfejs w zależności od typu zadania
            pass
        except Exception as e:
            self.parent.logger.error(
                f"Błąd podczas aktualizacji interfejsu po zmianie typu zadania: {str(e)}"
            )

    def refresh(self):
        """Odświeża listę zadań w kolejce."""
        self._refresh_task_queue()

    def _refresh_task_queue(self):
        """Odświeża listę zadań trainingowych w kolejce."""
        try:
            # Wyczyść tabelę
            self.tasks_table.setRowCount(0)

            # Katalog z zadaniami
            tasks_dir = os.path.join("data", "tasks")
            os.makedirs(tasks_dir, exist_ok=True)

            # Pobierz pliki zadań
            task_files = sorted(glob.glob(os.path.join(tasks_dir, "*.json")))

            if not task_files:
                return

            # Dodaj zadania do tabeli
            for task_file in task_files:
                try:
                    with open(task_file, "r", encoding="utf-8") as f:
                        task_data = json.load(f)

                    # Dodaj wiersz do tabeli
                    row = self.tasks_table.rowCount()
                    self.tasks_table.insertRow(row)
                    self.tasks_table.setRowHeight(row, 38)

                    # Nazwa zadania
                    task_name = task_data.get("name", os.path.basename(task_file))
                    self.tasks_table.setItem(
                        row, 0, QtWidgets.QTableWidgetItem(task_name)
                    )

                    # Typ zadania
                    if "type" not in task_data:
                        self.parent.logger.error(
                            f"Brak klucza 'type' w zadaniu {task_file}"
                        )
                        task_type = "Nieznany"
                    else:
                        task_type = task_data["type"]
                    self.tasks_table.setItem(
                        row, 1, QtWidgets.QTableWidgetItem(task_type)
                    )

                    # Status
                    status = task_data.get("status", "Nowy")
                    self.tasks_table.setItem(row, 2, QtWidgets.QTableWidgetItem(status))

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
                    action_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter)

                    # Przycisk uruchomienia
                    run_btn = QtWidgets.QPushButton("Uruchom")
                    run_btn.setFixedWidth(100)
                    run_btn.setFixedHeight(20)
                    run_btn.setStyleSheet(
                        """
                        QPushButton {
                            background-color: #4CAF50;
                            color: white;
                            border: none;
                            padding: 5px;
                            border-radius: 3px;
                        }
                        QPushButton:hover {
                            background-color: #388E3C;
                        }
                        QPushButton:pressed {
                            background-color: #1B5E20;
                        }
                    """
                    )
                    run_btn.clicked.connect(
                        lambda checked, file=task_file: (
                            self.parent.logger.info(
                                f"DEBUG: run_btn.clicked - Lambda wywołana dla pliku: {file}"
                            ),
                            self._run_task_from_queue(file),
                        )
                    )

                    # Przycisk edycji
                    edit_btn = QtWidgets.QPushButton("Edytuj")
                    edit_btn.setFixedWidth(80)
                    edit_btn.setFixedHeight(20)
                    edit_btn.clicked.connect(
                        lambda checked, file=task_file: self._edit_task_from_queue(file)
                    )

                    # Przycisk usunięcia
                    delete_btn = QtWidgets.QPushButton("Usuń")
                    delete_btn.setFixedWidth(80)
                    delete_btn.setFixedHeight(20)
                    delete_btn.setStyleSheet(
                        """
                        QPushButton {
                            background-color: #F44336;
                            color: white;
                            border: none;
                            padding: 5px;
                            border-radius: 3px;
                        }
                        QPushButton:hover {
                            background-color: #D32F2F;
                        }
                        QPushButton:pressed {
                            background-color: #B71C1C;
                        }
                    """
                    )
                    delete_btn.clicked.connect(
                        lambda checked, file=task_file: self._delete_task_from_queue(
                            file
                        )
                    )

                    action_layout.addWidget(run_btn)
                    action_layout.addWidget(edit_btn)
                    action_layout.addWidget(delete_btn)
                    action_layout.addStretch()

                    self.tasks_table.setCellWidget(row, 10, action_widget)

                    # Dodaj wyniki trainingu jeśli zadanie jest zakończone
                    if status == "Zakończony":
                        # Czas trainingu
                        training_time = task_data.get("training_time_str", "")
                        self.tasks_table.setItem(
                            row, 5, QtWidgets.QTableWidgetItem(training_time)
                        )

                        # Dokładność trainingu
                        train_acc = task_data.get("train_accuracy", 0)
                        self.tasks_table.setItem(
                            row, 6, QtWidgets.QTableWidgetItem(f"{train_acc:.2f}")
                        )

                        # Strata trainingu
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
                    self.parent.logger.error(
                        f"Błąd podczas wczytywania zadania {task_file}: {str(e)}"
                    )

            # Dostosuj szerokość kolumn
            self.tasks_table.resizeColumnsToContents()
            header = self.tasks_table.horizontalHeader()
            for col in range(self.tasks_table.columnCount()):
                width = self.tasks_table.columnWidth(col)
                if col == 0:  # Kolumna "Nazwa"
                    new_width = int(width * 1.15)
                elif col == 5:  # Kolumna "Czas trainingu"
                    new_width = int(width * 1.4)
                elif col in [6, 7, 8, 9]:  # Kolumny z wynikami trainingu
                    new_width = int(width * 1.0)
                else:
                    new_width = int(width * 1.6)
                self.tasks_table.setColumnWidth(col, new_width)

            # Ustaw kolumnę 'Akcje' (ostatnia) na rozciąganie
            last_col = self.tasks_table.columnCount() - 1
            header.setSectionResizeMode(
                last_col, QtWidgets.QHeaderView.ResizeMode.Stretch
            )
            header.setStretchLastSection(True)

        except Exception as e:
            self.parent.logger.error(
                f"Błąd podczas odświeżania kolejki zadań: {str(e)}"
            )

    def _create_queue_panel_widget(self):
        """Tworzy i zwraca panel kolejki zadań trainingowych jako widget."""
        queue_panel = QtWidgets.QWidget()
        queue_layout = QtWidgets.QVBoxLayout(queue_panel)
        queue_layout.setContentsMargins(0, 0, 0, 0)
        queue_layout.setSpacing(8)

        # Nagłówek sekcji
        queue_header = QtWidgets.QLabel("KOLEJKA ZADAŃ TRAININGOWYCH")
        queue_header.setStyleSheet(
            "font-weight: bold; color: #CCCCCC; "
            "font-size: 11px; padding-bottom: 4px;"
        )
        queue_layout.addWidget(queue_header)

        # Tabela z zadaniami
        self.tasks_table = QtWidgets.QTableWidget()
        self.tasks_table.setColumnCount(11)
        self.tasks_table.setHorizontalHeaderLabels(
            [
                "Nazwa",
                "Typ",
                "Status",
                "Priorytet",
                "Utworzono",
                "Czas trainingu",
                "Dokładność trainingu",
                "Strata trainingu",
                "Dokładność walidacji",
                "Strata walidacji",
                "Akcje",
            ]
        )
        self.tasks_table.horizontalHeader().setStretchLastSection(True)
        self.tasks_table.setSelectionBehavior(
            QtWidgets.QTableWidget.SelectionBehavior.SelectRows
        )
        self.tasks_table.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        queue_layout.addWidget(self.tasks_table, 1)

        # Przyciski zarządzania kolejką
        buttons_container = QtWidgets.QWidget()
        buttons_layout = QtWidgets.QHBoxLayout(buttons_container)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(8)

        self.refresh_queue_btn = QtWidgets.QPushButton("Odśwież kolejkę")
        self.refresh_queue_btn.clicked.connect(self.refresh)
        self.refresh_queue_btn.setFixedHeight(24)
        buttons_layout.addWidget(self.refresh_queue_btn)

        self.null_btn = QtWidgets.QPushButton("Uruchom wsadowy training")
        self.null_btn.setFixedHeight(24)
        self.null_btn.clicked.connect(self._show_queue_manager)
        buttons_layout.addWidget(self.null_btn)

        self.clear_queue_btn = QtWidgets.QPushButton("Wyczyść kolejkę")
        self.clear_queue_btn.clicked.connect(self._clear_task_queue)
        self.clear_queue_btn.setFixedHeight(24)
        buttons_layout.addWidget(self.clear_queue_btn)

        buttons_layout.addStretch(1)
        queue_layout.addWidget(buttons_container)

        return queue_panel

    def _add_training_task(self):
        try:
            # Pobierz typ zadania
            task_type = self.task_type_combo.currentText()

            # Wybierz odpowiedni dialog w zależności od typu zadania
            if task_type == "Trening nowego modelu":
                dialog = TrainingTaskConfigDialog(
                    parent=self,
                    settings=self.settings,
                    hardware_profile=getattr(self.parent, "hardware_profile", None),
                )
            elif task_type == "Doszkalanie istniejącego modelu":
                dialog = FineTuningTaskConfigDialog(
                    parent=self,
                    settings=self.settings,
                    hardware_profile=getattr(self.parent, "hardware_profile", None),
                )
            else:
                raise ValueError(f"Nieznany typ zadania: {task_type}")

            result = dialog.exec()
            if result == QtWidgets.QDialog.DialogCode.Accepted:
                task_config = dialog.get_task_config()
                if task_config:
                    # Usuń rozszerzenie .json jeśli już istnieje w nazwie
                    task_name = task_config["name"].replace(".json", "")
                    task_file = os.path.join("data", "tasks", f"{task_name}.json")
                    os.makedirs(os.path.dirname(task_file), exist_ok=True)
                    with open(task_file, "w", encoding="utf-8") as f:
                        json.dump(task_config, f, indent=4)
                    self.refresh()

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Nie udało się dodać zadania treningowego: {str(e)}"
            )

    def _clear_task_queue(self):
        """Czyści kolejkę zadań trainingowych."""
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
            # Katalog z zadaniami
            tasks_dir = os.path.join("data", "tasks")

            if not os.path.exists(tasks_dir):
                return

            # Usuń wszystkie pliki zadań
            for task_file in glob.glob(os.path.join(tasks_dir, "*.json")):
                try:
                    os.remove(task_file)
                except Exception as e:
                    self.parent.logger.error(
                        f"Błąd podczas usuwania zadania {task_file}: {str(e)}"
                    )

            # Odśwież listę zadań
            self.refresh()

            # Wyświetl potwierdzenie
            QtWidgets.QMessageBox.information(
                self,
                "Kolejka wyczyszczona",
                "Wszystkie zadania zostały usunięte z kolejki.",
            )

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Wystąpił błąd podczas czyszczenia kolejki: {str(e)}"
            )

    def _training_task_started(self, task_name, task_type):
        """Obsługa rozpoczęcia zadania treningowego."""
        # Logowanie
        self.parent.logger.info(
            f"DEBUG: _training_task_started wywołane dla zadania: "
            f"{task_name}, typ: {task_type}"
        )
        self.parent.logger.info(
            f"DEBUG: _training_task_started - Stan przycisku stop PRZED "
            f"ustawieniem: {self.parent.stop_task_btn.isEnabled()}"
        )
        self.parent.current_task_info.setText(
            f"Aktywne zadanie: {task_name} ({task_type})"
        )
        self.parent.task_progress_bar.setValue(0)  # Zresetuj pasek postępu
        self.parent.stop_task_btn.setEnabled(True)  # Aktywuj przycisk zatrzymania
        self.parent.logger.info(
            f"DEBUG: _training_task_started - Stan przycisku stop PO "
            f"ustawieniu na True: {self.parent.stop_task_btn.isEnabled()}"
        )

        # Wyczyść dane wizualizacji
        if hasattr(self, "training_visualization") and self.training_visualization:
            self.training_visualization.clear_data()

    def _training_task_progress(self, task_name, progress, details):
        """Obsługuje aktualizacje postępu trainingu."""
        try:
            # Pobierz dane z details i upewnij się, że mają prawidłowe wartości
            epoch = int(details.get("epoch", 0))
            total_epochs = int(details.get("total_epochs", 1))

            # Zabezpieczenie przed dzieleniem przez zero
            if total_epochs <= 0:
                total_epochs = 1

            # Pobierz i weryfikuj wartości metryk
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

            # Aktualizacja paska postępu
            percentage = min(100, max(0, int((epoch / total_epochs) * 100)))
            self.parent.task_progress_bar.setValue(percentage)

            # Aktualizacja opisu postępu
            if epoch > 0:
                loss_text = f"{train_loss:.4f}" if train_loss is not None else "N/A"
                acc_text = f"{train_acc:.2%}" if train_acc is not None else "N/A"
                val_loss_text = f"{val_loss:.4f}" if val_loss is not None else "N/A"
                val_acc_text = f"{val_acc:.2%}" if val_acc is not None else "N/A"
                val_top3_text = f"{val_top3:.2%}" if val_top3 is not None else "N/A"
                val_top5_text = f"{val_top5:.2%}" if val_top5 is not None else "N/A"
                val_precision_text = (
                    f"{val_precision:.2%}" if val_precision is not None else "N/A"
                )
                val_recall_text = (
                    f"{val_recall:.2%}" if val_recall is not None else "N/A"
                )
                val_f1_text = f"{val_f1:.2%}" if val_f1 is not None else "N/A"
                val_auc_text = f"{val_auc:.2%}" if val_auc is not None else "N/A"

                details_text = (
                    f"Epoka {epoch}/{total_epochs} | "
                    f"Strata: {loss_text}, Dokładność: {acc_text} | "
                    f"Val Strata: {val_loss_text}, Val Acc: {val_acc_text} | "
                    f"Top-3: {val_top3_text}, Top-5: {val_top5_text} | "
                    f"Precision: {val_precision_text}, Recall: {val_recall_text} | "
                    f"F1: {val_f1_text}, AUC: {val_auc_text}"
                )
                self.parent.task_progress_details.setText(details_text)
                self.parent.logger.info(details_text)

            # Aktualizacja wizualizacji jeśli istnieje
            if hasattr(self, "training_visualization") and self.training_visualization:
                # Upewnij się, że epoka jest większa od zera
                if epoch > 0:
                    try:
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
                    except Exception as vis_error:
                        self.parent.logger.error(
                            f"Błąd aktualizacji wizualizacji: {vis_error}"
                        )

        except Exception as e:
            import traceback

            self.parent.logger.error(f"Błąd w _training_task_progress: {e}")
            self.parent.logger.error(traceback.format_exc())

    def _training_task_completed(self, task_name, result):
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
                                    json.dump(
                                        task_data, f, indent=4, ensure_ascii=False
                                    )
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

    def _all_training_tasks_completed(self):
        """Obsługa zakończenia wszystkich zadań treningowych."""
        self.parent.logger.info("Wszystkie zadania treningowe zostały zakończone")

        # Informacja dla użytkownika
        QtWidgets.QMessageBox.information(
            self,
            "Trening zakończony",
            "Wszystkie zadania treningowe zostały zakończone pomyślnie.",
        )

        # Aktualizacja UI w głównym oknie
        self.parent.current_task_info.setText("Brak aktywnego zadania")
        self.parent.task_progress_bar.setValue(0)
        self.parent.task_progress_details.setText("Oczekiwanie na zadania...")

        # Reset wizualizacji po zakończeniu wszystkich zadań
        if hasattr(self, "training_visualization") and self.training_visualization:
            self.training_visualization.clear_data()
            self.training_visualization.reset_plot()

        # Odśwież kolejkę, aby zaktualizować statusy zadań
        self._refresh_task_queue()

    def _training_task_error(self, task_name, error_message):
        """Obsługa błędu zadania treningowego."""
        # Logowanie do konsoli
        self.parent.logger.error(
            f"DEBUG: _training_task_error wywołane dla zadania: {task_name}, błąd: {error_message}"
        )
        self.parent.logger.error(
            f"DEBUG: _training_task_error - Stan przycisku stop PRZED "
            f"ustawieniem na False: {self.parent.stop_task_btn.isEnabled()}"
        )
        print(f"\n[KRYTYCZNY BŁĄD] Zadanie: {task_name}")
        print(f"Treść błędu: {error_message}")
        print("-" * 80)

        self.parent.logger.error(f"Błąd zadania {task_name}: {error_message}")

        # Pokaż komunikat o błędzie
        QtWidgets.QMessageBox.critical(
            self,
            "Błąd treningu",
            f"Wystąpił błąd w zadaniu {task_name}: {error_message}",
        )

        # Odświeżamy listę zadań
        self.refresh()
        # Dodatkowo upewnijmy się, że przycisk stop jest wyłączony
        self.parent.stop_task_btn.setEnabled(False)
        self.parent.logger.error(
            f"DEBUG: _training_task_error - Stan przycisku stop PO "
            f"ustawieniu na False: {self.parent.stop_task_btn.isEnabled()}"
        )

    def update_settings(self, settings):
        """Aktualizuje ustawienia zakładki."""
        self.settings = settings

    def save_state(self):
        """Zapisuje stan zakładki."""
        return {}

    def restore_state(self, state):
        """Przywraca zapisany stan zakładki."""
        pass

    def update_hardware_profile(self, profile):
        """Aktualizuje profil sprzętowy używany do optymalizacji treningu."""
        self.hardware_profile = profile
        self.parent.logger.info("Zaktualizowano profil sprzętowy w zakładce treningu")

        # Opcjonalnie: aktualizacja UI na podstawie nowego profilu
        if hasattr(self, "use_optimization_checkbox"):
            self.use_optimization_checkbox.setChecked(True)

        if profile and hasattr(self, "profile_info_label"):
            cpu_info = profile.get("cpu_info", {})
            if isinstance(cpu_info, str):
                try:
                    cpu_info = json.loads(cpu_info)
                except json.JSONDecodeError:
                    cpu_info = {}

            gpu_info = profile.get("gpu_info", {})
            if isinstance(gpu_info, str):
                try:
                    gpu_info = json.loads(gpu_info)
                except json.JSONDecodeError:
                    gpu_info = {}

            info_text = "Status profilu: Aktywny\n"
            info_text += f"CPU: {cpu_info.get('name', 'Nieznany')}\n"
            if gpu_info:
                info_text += f"GPU: {gpu_info.get('name', 'Nieznany')}\n"
            info_text += f"RAM: {profile.get('ram_total', 0):.1f} GB"

            self.profile_info_label.setText(info_text)

    def _run_profiler(self):
        """Uruchamia proces profilowania sprzętu."""
        try:
            # Wyświetl dialog potwierdzenia
            reply = QtWidgets.QMessageBox.question(
                self,
                "Profilowanie sprzętu",
                "Proces profilowania może potrwać kilka minut i okresowo "
                "obciążyć system.\n\nCzy chcesz kontynuować?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.Yes,
            )

            if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                return

            # Utwórz i skonfiguruj dialog postępu
            progress_dialog = QtWidgets.QProgressDialog(
                "Trwa profilowanie sprzętu...", "Anuluj", 0, 100, self
            )
            progress_dialog.setWindowTitle("Profilowanie")
            progress_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
            progress_dialog.setMinimumDuration(0)
            progress_dialog.show()

            # Aktualizacja interfejsu
            progress_dialog.setValue(10)
            QtCore.QCoreApplication.processEvents()

            # Uruchom profiler w tle
            from app.utils.profiler import HardwareProfiler

            profiler = HardwareProfiler()

            # Aktualizacja interfejsu
            progress_dialog.setValue(20)
            progress_dialog.setLabelText("Rozpoczynam profilowanie...")
            QtCore.QCoreApplication.processEvents()

            # Wykonaj pełne profilowanie
            profile = profiler.run_profile()

            # Aktualizacja interfejsu
            progress_dialog.setValue(100)
            progress_dialog.setLabelText("Profilowanie zakończone!")
            QtCore.QCoreApplication.processEvents()

            # Załaduj wygenerowany profil
            self.parent.hardware_profile = profile

            # Wyświetl podsumowanie profilowania
            QtWidgets.QMessageBox.information(
                self,
                "Profilowanie zakończone",
                "Profilowanie sprzętu zostało zakończone pomyślnie.\n\n"
                "Parametry aplikacji zostały zoptymalizowane dla Twojego systemu.",
            )

            # Aktualizuj informacje o profilu
            self._update_profile_info()

        except Exception as e:
            self.parent.logger.error(f"Błąd profilowania: {str(e)}")
            QtWidgets.QMessageBox.critical(
                self,
                "Błąd profilowania",
                f"Wystąpił nieoczekiwany błąd: {str(e)}",
            )

    def _update_profile_info(self):
        """Aktualizuje informacje o profilu sprzętowym."""
        try:
            if (
                hasattr(self.parent, "hardware_profile")
                and self.parent.hardware_profile
            ):
                profile = self.parent.hardware_profile
                cpu_info = profile.get("cpu_info", {})
                gpu_info = profile.get("gpu_info", {})
                ram_info = profile.get("ram_info", {})

                info_text = "Status profilu: Aktywny\n"
                info_text += f"CPU: {cpu_info.get('name', 'Nieznany')}\n"
                if gpu_info:
                    info_text += f"GPU: {gpu_info.get('name', 'Nieznany')}\n"
                info_text += f"RAM: {ram_info.get('total_gb', 0):.1f} GB"

                self.profile_info_label.setText(info_text)
            else:
                self.profile_info_label.setText("Status profilu: Brak")
        except Exception as e:
            self.parent._log_message(
                f"Błąd aktualizacji informacji o profilu: {str(e)}"
            )
            self.profile_info_label.setText("Status profilu: Błąd")

    def _run_task_from_queue(self, task_file):
        """Uruchamia pojedyncze zadanie z kolejki."""
        try:
            self.parent.logger.info(
                f"DEBUG: _run_task_from_queue wywołane dla pliku: {task_file}"
            )
            # Sprawdź, czy wątek treningowy już działa
            if (
                hasattr(self, "training_thread")
                and self.training_thread is not None
                and self.training_thread.isRunning()
            ):
                QtWidgets.QMessageBox.warning(
                    self,
                    "Trening w toku",
                    "Inny proces treningu jest już aktywny. "
                    "Poczekaj na jego zakończenie lub zatrzymaj go.",
                )
                return

            # Wyczyść dane wizualizacji przed rozpoczęciem nowego zadania
            if hasattr(self, "training_visualization") and self.training_visualization:
                self.training_visualization.clear_data()

            # Usuń rozszerzenie .json jeśli już istnieje w nazwie
            task_name = os.path.basename(task_file).replace(".json", "")
            task_file = os.path.join("data", "tasks", f"{task_name}.json")

            # Utwórz nowy wątek z pojedynczym zadaniem
            self.training_thread = SingleTrainingThread(task_file)

            # Podłącz sygnały
            self.parent.logger.info(
                f"DEBUG: _run_task_from_queue - PRZED podłączeniem sygnałów "
                f"dla {task_file}"
            )
            self.training_thread.task_started.connect(self._training_task_started)
            self.training_thread.task_progress.connect(self._training_task_progress)
            self.training_thread.task_completed.connect(self._training_task_completed)
            self.training_thread.error.connect(self._training_task_error)
            self.parent.logger.info(
                f"DEBUG: _run_task_from_queue - PO podłączeniu sygnałów "
                f"dla {task_file}"
            )

            # Uruchom wątek
            self.parent.logger.info(
                f"DEBUG: _run_task_from_queue - PRZED "
                f"self.training_thread.start() dla {task_file}"
            )
            self.training_thread.start()
            self.parent.logger.info(
                f"DEBUG: _run_task_from_queue - PO "
                f"self.training_thread.start() dla {task_file}"
            )

            # Zaktualizuj UI
            self.parent.current_task_info.setText("Rozpoczynanie zadania...")
            self.parent.logger.info("Uruchomiono pojedyncze zadanie.")

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Nie udało się uruchomić zadania: {str(e)}"
            )

    def _delete_task_from_queue(self, task_file):
        """Usuwa zadanie z kolejki."""
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
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Wystąpił błąd podczas usuwania zadania: {str(e)}"
            )
            self.parent.logger.error(f"Błąd podczas usuwania zadania: {str(e)}")

    def _set_task_status(self, task_name, new_status):
        """Zmienia status zadania na nowy i zapisuje do pliku."""
        tasks_dir = os.path.join("data", "tasks")
        task_file = os.path.join(tasks_dir, f"{task_name}.json")

        self.parent.logger.info(
            f"Próba zmiany statusu zadania {task_name} na {new_status}"
        )
        self.parent.logger.info(f"Ścieżka do pliku zadania: {task_file}")

        if os.path.exists(task_file):
            try:
                self.parent.logger.info(
                    f"Wczytywanie danych zadania z pliku: {task_file}"
                )
                with open(task_file, "r", encoding="utf-8") as f:
                    task_data = json.load(f)
                self.parent.logger.info(
                    f"Aktualny status zadania: {task_data.get('status')}"
                )

                task_data["status"] = new_status
                self.parent.logger.info(f"Nowy status zadania: {new_status}")

                self.parent.logger.info(
                    f"Zapisywanie zaktualizowanych danych do pliku: {task_file}"
                )
                with open(task_file, "w", encoding="utf-8") as f:
                    json.dump(task_data, f, indent=4, ensure_ascii=False)
                self.parent.logger.info("Pomyślnie zaktualizowano status zadania")

            except Exception as e:
                self.parent.logger.error(f"Błąd przy zmianie statusu zadania: {str(e)}")
                self.parent.logger.error(f"TRACEBACK: {traceback.format_exc()}")
        else:
            self.parent.logger.error(f"Plik zadania nie istnieje: {task_file}")

    def _stop_current_task(self):
        """Bezwzględnie zatrzymuje aktualnie wykonywane zadanie treningu."""
        try:
            self.parent.logger.info(
                f"DEBUG: _stop_current_task - Rozpoczęto zatrzymywanie zadania. "
                f"Stan przycisku stop PRZED: {self.parent.stop_task_btn.isEnabled()}"
            )
            if hasattr(self, "training_thread") and self.training_thread is not None:
                # Zatrzymaj wątek treningu
                if hasattr(self.training_thread, "stop"):
                    self.training_thread.stop()

                # Poczekaj na zakończenie wątku (maksymalnie 5 sekund)
                if self.training_thread.isRunning():
                    self.training_thread.wait(5000)

                    # Tylko jeśli wątek nadal działa po czekaniu, rozważ użycie terminate()
                    if self.training_thread.isRunning():
                        # Zapisz informację o wymuszonym zatrzymaniu
                        self.parent.logger.warning(
                            "Wymuszenie zatrzymania wątku treningowego"
                        )
                        self.training_thread.terminate()
                        self.training_thread.wait()

                # Wyczyść referencję do wątku
                self.training_thread = None

                # Zaktualizuj UI
                self.parent.current_task_info.setText("Zadanie zostało zatrzymane")
                self.parent.task_progress_bar.setValue(0)
                self.parent.task_progress_details.setText("")
                self.parent.stop_task_btn.setEnabled(False)
                self.parent.logger.info(
                    f"DEBUG: _stop_current_task - Zadanie zatrzymane. "
                    f"Stan przycisku stop PO: {self.parent.stop_task_btn.isEnabled()}"
                )

                # Odśwież listę zadań
                self.refresh()

                self.parent.logger.info("Zadanie treningu zostało zatrzymane")

        except Exception as e:
            self.parent.logger.error(f"Błąd podczas zatrzymywania zadania: {str(e)}")
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Wystąpił błąd podczas zatrzymywania zadania:\n{str(e)}"
            )

    def _edit_task_from_queue(self, task_file):
        """Otwiera plik zadania w domyślnym edytorze tekstowym/kodu."""
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
            self.parent.logger.error(f"Błąd podczas otwierania pliku: {str(e)}")
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Nie udało się otworzyć pliku w edytorze: {str(e)}"
            )

    def _run_batch_training(self):
        """Uruchamia wsadowe zadania treningowe."""
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
                return

            # Uruchom pierwsze zadanie
            self.training_thread = SingleTrainingThread(task_files[0])
            self.training_thread.task_started.connect(self._training_task_started)
            self.training_thread.task_progress.connect(self._training_task_progress)
            self.training_thread.task_completed.connect(self._training_task_completed)
            self.training_thread.error.connect(self._training_task_error)
            self.training_thread.start()

        except Exception as e:
            self.logger.error(f"Błąd podczas uruchamiania wsadowego treningu: {str(e)}")
            QtWidgets.QMessageBox.critical(
                self,
                "Błąd",
                f"Wystąpił błąd podczas uruchamiania wsadowego treningu:\n{str(e)}",
            )

    def _select_directory(self, line_edit):
        """Otwiera dialog wyboru katalogu i aktualizuje pole tekstowe."""
        try:
            directory = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                "Wybierz katalog",
                os.path.join("data"),
                QtWidgets.QFileDialog.Option.ShowDirsOnly
                | QtWidgets.QFileDialog.Option.DontResolveSymlinks,
            )
            if directory:
                line_edit.setText(directory)
                self.parent.logger.info(f"Wybrano katalog: {directory}")
        except Exception as e:
            self.parent.logger.error(f"Błąd podczas wyboru katalogu: {str(e)}")
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Wystąpił błąd podczas wyboru katalogu: {str(e)}"
            )

    def _show_queue_manager(self):
        """Uruchamia widget queue_manager i odświeża jego listę zadań."""
        try:
            if self.queue_manager:  # Upewnij się, że obiekt istnieje
                self.queue_manager.load_new_tasks()  # ODŚWIEŻ LISTĘ ZADAŃ
                # Bardzo czytelny komunikat do konsoli z nazwami zadań
                nazwy_zadan = [
                    task.get("name", "BRAK_NAZWY")
                    for task in self.queue_manager.new_tasks
                ]
                print("*" * 60)
                print(f"***   DODANO DO KOLEJKI {len(nazwy_zadan)} ZADAŃ   ***")
                print(f"***   NAZWY ZADAŃ:")
                for nazwa in nazwy_zadan:
                    print(f"***     - {nazwa}")
                print("*" * 60)
            self.queue_manager.show()
            self.queue_manager.raise_()
            self.queue_manager.activateWindow()
        except Exception as e:
            self.parent.logger.error(
                f"Błąd podczas uruchamiania Queue Manager: {str(e)}"
            )
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Nie udało się uruchomić Queue Manager: {str(e)}"
            )
