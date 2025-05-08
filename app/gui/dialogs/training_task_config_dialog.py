import datetime
import json
import logging
import os
import sys

import torch
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.utils.config import DEFAULT_TRAINING_PARAMS
from app.utils.file_utils import (
    validate_training_directory,
    validate_validation_directory,
)


class TrainingTaskConfigDialog(QDialog):
    """Dialog konfiguracji zadania treningowego z zaawansowanymi opcjami."""

    def __init__(self, parent=None, settings=None, hardware_profile=None):
        super().__init__(parent)
        self.settings = settings
        self.hardware_profile = hardware_profile
        self.task_config = None

        # Konfiguracja loggera
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # Handler do pliku
        log_file = os.path.join("logs", "training_config.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Handler do konsoli
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(levelname)s: %(message)s")
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        self.logger.info("Inicjalizacja okna konfiguracji treningu")

        self.setWindowTitle("Konfiguracja zadania treningu")
        self.setMinimumWidth(800)
        self.setMinimumHeight(600)
        self._init_ui()

    def _init_ui(self):
        """Inicjalizacja interfejsu użytkownika z zakładkami."""
        try:
            self.logger.debug("Rozpoczęcie inicjalizacji UI")
            layout = QVBoxLayout(self)

            # Utworzenie zakładek
            self.tabs = QTabWidget()

            # Zakładka 1: Dane i Model
            self.data_model_tab = self._create_data_model_tab()
            self.tabs.addTab(self.data_model_tab, "Dane i Model")

            # Dodanie zakładek do layoutu
            layout.addWidget(self.tabs)

            # Przyciski OK i Anuluj
            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok
                | QDialogButtonBox.StandardButton.Cancel
            )
            buttons.accepted.connect(self._on_accept)
            buttons.rejected.connect(self.reject)
            layout.addWidget(buttons)

            self.logger.debug("Zakończono inicjalizację podstawowego UI")

        except Exception as e:
            self.logger.error(f"Błąd podczas inicjalizacji UI: {str(e)}", exc_info=True)
            raise

    def _create_data_model_tab(self):
        """Tworzenie zakładki Dane i Model."""
        try:
            self.logger.debug("Tworzenie zakładki Dane i Model")
            tab = QWidget()
            layout = QVBoxLayout(tab)
            form = QFormLayout()

            # Katalog danych treningowych
            train_dir_layout = QHBoxLayout()
            self.train_dir_edit = QLineEdit()
            train_dir_btn = QPushButton("Przeglądaj...")
            train_dir_btn.clicked.connect(
                lambda: self._select_directory(self.train_dir_edit, "treningowych")
            )
            train_dir_layout.addWidget(self.train_dir_edit)
            train_dir_layout.addWidget(train_dir_btn)
            form.addRow("Katalog danych treningowych:", train_dir_layout)

            # Katalog danych walidacyjnych
            val_dir_layout = QHBoxLayout()
            self.val_dir_edit = QLineEdit()
            val_dir_btn = QPushButton("Przeglądaj...")
            val_dir_btn.clicked.connect(
                lambda: self._select_directory(self.val_dir_edit, "walidacyjnych")
            )
            val_dir_layout.addWidget(self.val_dir_edit)
            val_dir_layout.addWidget(val_dir_btn)
            form.addRow("Katalog danych walidacyjnych:", val_dir_layout)

            # Architektura modelu
            self.model_arch_combo = QComboBox()
            self.model_arch_combo.addItems(
                ["efficientnet", "resnet", "mobilenet", "vit", "convnext"]
            )
            form.addRow("Architektura modelu:", self.model_arch_combo)

            # Wariant modelu
            self.model_variant_combo = QComboBox()
            self.model_arch_combo.currentTextChanged.connect(
                self._update_model_variants
            )
            form.addRow("Wariant modelu:", self.model_variant_combo)

            # Rozmiar obrazu wejściowego
            self.input_size_spin = QSpinBox()
            self.input_size_spin.setRange(32, 1024)
            self.input_size_spin.setSingleStep(32)
            self.input_size_spin.setValue(224)
            form.addRow("Rozmiar obrazu wejściowego:", self.input_size_spin)

            # Liczba klas
            self.num_classes_spin = QSpinBox()
            self.num_classes_spin.setRange(2, 1000)
            self.num_classes_spin.setValue(2)
            form.addRow("Liczba klas:", self.num_classes_spin)

            layout.addLayout(form)
            self._update_model_variants(self.model_arch_combo.currentText())

            self.logger.debug("Zakończono tworzenie zakładki Dane i Model")
            return tab

        except Exception as e:
            self.logger.error(
                f"Błąd podczas tworzenia zakładki Dane i Model: {str(e)}", exc_info=True
            )
            raise

    def _update_model_variants(self, architecture):
        """Aktualizacja listy wariantów w zależności od wybranej architektury."""
        try:
            self.logger.debug(
                f"Aktualizacja wariantów dla architektury: {architecture}"
            )
            self.model_variant_combo.clear()

            variants = {
                "efficientnet": ["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7"],
                "resnet": ["18", "34", "50", "101", "152"],
                "mobilenet": ["v2", "v3_small", "v3_large"],
                "vit": ["tiny", "small", "base", "large"],
                "convnext": ["tiny", "small", "base", "large"],
            }

            if architecture in variants:
                self.model_variant_combo.addItems(variants[architecture])
                self.logger.debug(f"Dodano warianty: {variants[architecture]}")

        except Exception as e:
            self.logger.error(
                f"Błąd podczas aktualizacji wariantów modelu: {str(e)}", exc_info=True
            )

    def _select_directory(self, line_edit, data_type):
        """Wybór katalogu danych."""
        try:
            self.logger.debug(f"Otwieranie okna wyboru katalogu dla danych {data_type}")
            directory = QFileDialog.getExistingDirectory(
                self,
                f"Wybierz katalog danych {data_type}",
                os.path.join("data"),
                QFileDialog.Option.ShowDirsOnly,
            )
            if directory:
                line_edit.setText(directory)
                self.logger.info(f"Wybrano katalog danych {data_type}: {directory}")

        except Exception as e:
            self.logger.error(f"Błąd podczas wyboru katalogu: {str(e)}", exc_info=True)

    def _on_accept(self):
        """Obsługa zatwierdzenia konfiguracji."""
        try:
            self.logger.info("Rozpoczęcie walidacji i zapisu konfiguracji")

            # Walidacja katalogu treningowego
            train_dir = self.train_dir_edit.text()
            if not train_dir.strip():
                self.logger.warning("Nie wybrano katalogu treningowego")
                QMessageBox.critical(
                    self, "Błąd", "Musisz wybrać katalog danych treningowych!"
                )
                return

            if not validate_training_directory(train_dir):
                self.logger.error(f"Nieprawidłowy katalog treningowy: {train_dir}")
                return

            # Walidacja katalogu walidacyjnego
            val_dir = self.val_dir_edit.text()
            if val_dir and not validate_validation_directory(val_dir):
                self.logger.error(f"Nieprawidłowy katalog walidacyjny: {val_dir}")
                return

            # Przygotowanie konfiguracji
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{self.model_arch_combo.currentText()}_{self.model_variant_combo.currentText()}"
            task_name = f"{model_name}_{timestamp}.json"

            self.task_config = {
                "name": task_name,
                "type": "Trening",
                "status": "Nowy",
                "priority": 0,
                "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "config": {
                    "train_dir": train_dir,
                    "data_dir": train_dir,
                    "val_dir": val_dir,
                    "model": {
                        "architecture": self.model_arch_combo.currentText(),
                        "variant": self.model_variant_combo.currentText(),
                        "input_size": self.input_size_spin.value(),
                        "num_classes": self.num_classes_spin.value(),
                    },
                },
            }

            self.logger.info(f"Utworzono konfigurację zadania: {task_name}")
            self.accept()

        except Exception as e:
            self.logger.error("Błąd podczas zapisywania konfiguracji", exc_info=True)
            QMessageBox.critical(
                self,
                "Błąd",
                f"Wystąpił błąd podczas zapisywania konfiguracji: {str(e)}",
            )

    def get_task_config(self):
        """Zwraca konfigurację zadania."""
        return self.task_config
