import datetime
import json
import os
import shutil
import traceback

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ai.classifier import ImageClassifier
from app.gui.tab_interface import TabInterface


class ModelManager(QWidget, TabInterface):
    """Klasa zarządzająca zakładką modeli."""

    def __init__(self, parent=None, settings=None):
        super().__init__(parent)
        self.parent = parent
        self.settings = settings or {}
        # Dodaj domyślną wartość dla models_dir jeśli nie istnieje
        if "models_dir" not in self.settings:
            self.settings["models_dir"] = "data/models"
            self.parent.logger.info(
                "Dodano domyślną wartość dla models_dir: data/models"
            )
        self.models_table = None
        self.hardware_profile = None  # Dodano inicjalizację
        self.setup_ui()
        self.connect_signals()

    def setup_ui(self):
        """Tworzy i konfiguruje elementy interfejsu zakładki."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Panel modeli - bez ramki GroupBox
        models_panel = QWidget()
        models_layout = QVBoxLayout(models_panel)
        models_layout.setContentsMargins(0, 0, 0, 0)
        models_layout.setSpacing(8)

        # Nagłówek sekcji
        models_header = QLabel("MODELE")
        models_header.setStyleSheet(
            "font-weight: bold; color: #CCCCCC; "
            "font-size: 11px; padding-bottom: 4px;"
        )
        models_layout.addWidget(models_header)

        # Tabela modeli
        self.models_table = QTableWidget()
        self.models_table.setColumnCount(11)
        self.models_table.setHorizontalHeaderLabels(
            [
                "Wybierz",
                "Nazwa",
                "Status",
                "Sesje F-T",
                "Data utworzenia",
                "Rozmiar (MB)",
                "Czas treningu",
                "Architektura modelu",
                "Wariant modelu",
                "Kategorie",
                "Optymizer",
            ]
        )
        self.models_table.horizontalHeader().setStretchLastSection(True)
        self.models_table.setColumnWidth(0, 50)  # Wybierz
        self.models_table.setColumnWidth(1, 290)  # Nazwa
        self.models_table.setColumnWidth(2, 80)  # Status
        self.models_table.setColumnWidth(3, 80)  # Sesje F-T (dawniej Dokładność)
        self.models_table.setColumnWidth(4, 120)  # Data utworzenia
        self.models_table.setColumnWidth(5, 80)  # Rozmiar (MB)
        self.models_table.setColumnWidth(6, 80)  # Czas treningu
        self.models_table.setColumnWidth(7, 80)  # Architektura modelu
        self.models_table.setColumnWidth(8, 80)  # Wariant modelu
        self.models_table.setColumnWidth(9, 80)  # Kategorie
        self.models_table.setColumnWidth(10, 80)  # Optymizer
        self.models_table.verticalHeader().setVisible(False)
        self.models_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.models_table.setAlternatingRowColors(True)
        self.models_table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        models_layout.addWidget(self.models_table, 1)

        # Poszerz wszystkie kolumny o 30%
        for col in range(self.models_table.columnCount()):
            width = self.models_table.columnWidth(col)
            new_width = int(width * 1.3)
            self.models_table.setColumnWidth(col, new_width)

        # Przyciski
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(8)  # Odstęp między przyciskami

        # Ustawienia wspólne dla wszystkich przycisków
        button_size = QSize(120, 24)  # Stały rozmiar

        self.load_btn = QPushButton("Załaduj model")
        self.load_btn.setFixedSize(button_size)
        self.load_btn.setProperty("action", "success")
        buttons_layout.addWidget(self.load_btn)

        self.refresh_btn = QPushButton("Odśwież")
        self.refresh_btn.setFixedSize(button_size)
        self.refresh_btn.setProperty("action", "primary")
        buttons_layout.addWidget(self.refresh_btn)

        self.class_mapping_btn = QPushButton("Mapowanie klas")
        self.class_mapping_btn.setFixedSize(button_size)
        buttons_layout.addWidget(self.class_mapping_btn)

        self.stats_btn = QPushButton("Statystyki")
        self.stats_btn.setFixedSize(button_size)
        buttons_layout.addWidget(self.stats_btn)

        self.compare_btn = QPushButton("Porównaj modele")
        self.compare_btn.setFixedSize(button_size)
        buttons_layout.addWidget(self.compare_btn)

        buttons_layout.addStretch()

        self.rename_btn = QPushButton("Zmień nazwę")
        self.rename_btn.setFixedSize(button_size)
        buttons_layout.addWidget(self.rename_btn)

        self.clone_btn = QPushButton("Klonuj model")
        self.clone_btn.setFixedSize(button_size)
        buttons_layout.addWidget(self.clone_btn)

        self.export_btn = QPushButton("Eksportuj model")
        self.export_btn.setFixedSize(button_size)
        buttons_layout.addWidget(self.export_btn)

        self.export_config_btn = QPushButton("Eksportuj konfig")
        self.export_config_btn.setFixedSize(button_size)
        buttons_layout.addWidget(self.export_config_btn)

        self.import_config_btn = QPushButton("Importuj konfig")
        self.import_config_btn.setFixedSize(button_size)
        buttons_layout.addWidget(self.import_config_btn)

        buttons_layout.addStretch()  # Separator przed przyciskiem Usuń model

        # Przycisk Usuń model - umieszczony jako ostatni w tej grupie
        self.delete_btn = QPushButton("Usuń model")
        self.delete_btn.setFixedSize(button_size)
        self.delete_btn.setProperty(
            "action", "warning"
        )  # Zmieniono na warning zamiast bezpośredniego stylu
        buttons_layout.addWidget(self.delete_btn)

        buttons_layout.addStretch()  # Ten addStretch kończy linię przycisków

        models_layout.addLayout(buttons_layout)
        layout.addWidget(models_panel, 1)

    def connect_signals(self):
        """Podłącza sygnały do slotów."""
        self.load_btn.clicked.connect(self._handle_load_button_click)
        self.export_btn.clicked.connect(self._export_selected_model)
        self.export_config_btn.clicked.connect(self._export_model_config)
        self.import_config_btn.clicked.connect(self._import_model_config)
        self.rename_btn.clicked.connect(self._rename_selected_model)
        self.clone_btn.clicked.connect(self._clone_selected_model)
        self.class_mapping_btn.clicked.connect(self._show_class_mapping)
        self.stats_btn.clicked.connect(self._show_model_stats)
        self.compare_btn.clicked.connect(self._compare_models)
        self.refresh_btn.clicked.connect(self.refresh)
        self.delete_btn.clicked.connect(self._delete_selected_model)

    def update_settings(self, settings):
        """Aktualizuje ustawienia zakładki."""
        self.settings = settings
        # Można tu dodać logikę specyficzną dla ModelManager,
        # np. odświeżenie, jeśli zmiana ustawień tego wymaga
        # (np. zmiana models_dir).
        if self.parent and hasattr(self.parent, "logger"):
            self.parent.logger.info("ModelManager zaktualizował ustawienia.")
        # Odśwież listę modeli, bo np. models_dir mógł się zmienić
        self._refresh_models_list()

    def refresh(self):
        """Odświeża listę modeli."""
        self._refresh_models_list()

    def _refresh_models_list(self):
        """Odświeża listę modeli."""
        try:
            # Wyczyść tabelę
            self.models_table.setRowCount(0)

            # Pobierz listę plików modeli
            models_dir = self.settings.get("models_dir", "data/models")
            if not os.path.exists(models_dir):
                os.makedirs(models_dir, exist_ok=True)

            models = []
            for file in os.listdir(models_dir):
                if file.endswith((".pt", ".pth")):
                    models.append(file)

            # Dodaj modele do tabeli
            for model_filename in models:  # Zmieniono nazwę zmiennej
                row = self.models_table.rowCount()
                self.models_table.insertRow(row)

                # Checkbox wyboru
                checkbox = QTableWidgetItem()
                checkbox.setCheckState(Qt.CheckState.Unchecked)
                self.models_table.setItem(row, 0, checkbox)

                # Nazwa modelu
                self.models_table.setItem(row, 1, QTableWidgetItem(model_filename))

                # Ścieżka do modelu
                model_path = os.path.join(models_dir, model_filename)

                # Status
                status = (
                    "Aktywny"
                    if hasattr(self.parent, "current_model")
                    and self.parent.current_model == model_filename
                    and hasattr(self.parent, "model_loaded")
                    and self.parent.model_loaded
                    else ""
                )
                self.models_table.setItem(row, 2, QTableWidgetItem(status))

                # Dane z konfiguracji
                config_path = os.path.splitext(model_path)[0] + "_config.json"
                sesje_ft = "Brak"
                training_time = "Brak"
                categories = "Brak"
                architecture = "Brak"
                variant = "Brak"
                optimizer = "Brak"

                if os.path.exists(config_path):
                    try:
                        with open(
                            config_path, "r", encoding="utf-8"
                        ) as f:  # Dodano encoding
                            config_data = json.load(f)

                        metadata = config_data.get("metadata", {})

                        # Sesje F-T
                        session_nrs = []
                        if isinstance(metadata, dict) and "session_nr" in metadata:
                            try:
                                session_nrs.append(int(metadata["session_nr"]))
                            except ValueError:  # Lepsze łapanie błędu
                                pass

                        # Rekurencyjne szukanie session_nr
                        def find_session_nrs_recursive(data_struct):
                            if isinstance(data_struct, dict):
                                for k, v_item in data_struct.items():
                                    if k == "session_nr":
                                        try:
                                            session_nrs.append(int(v_item))
                                        except ValueError:
                                            pass
                                    else:
                                        find_session_nrs_recursive(v_item)
                            elif isinstance(data_struct, list):
                                for item_in_list in data_struct:
                                    find_session_nrs_recursive(item_in_list)

                        find_session_nrs_recursive(config_data)
                        if session_nrs:
                            sesje_ft = str(max(session_nrs))

                        # Czas treningu
                        train_time_val = metadata.get("training_time", 0)
                        if isinstance(train_time_val, (int, float)):
                            hours = int(train_time_val // 3600)
                            minutes = int((train_time_val % 3600) // 60)
                            seconds = int(train_time_val % 60)
                            training_time = f"{hours}:{minutes:02d}:{seconds:02d}"
                        elif isinstance(train_time_val, str):
                            training_time = train_time_val  # Jeśli już jest stringiem

                        # Kategorie
                        class_names = metadata.get("class_names", {})
                        if isinstance(class_names, (dict, list)):
                            categories = str(len(class_names))

                        # Architektura, Wariant, Optymizer
                        training_params = metadata.get("training_params", {})
                        if isinstance(training_params, dict):  # Sprawdzenie typu
                            config_from_training = training_params.get("config", {})
                            if isinstance(
                                config_from_training, dict
                            ):  # Sprawdzenie typu
                                model_config = config_from_training.get("model", {})
                                if isinstance(model_config, dict):  # Sprawdzenie typu
                                    architecture = model_config.get(
                                        "architecture", "Brak"
                                    )
                                    variant = model_config.get("variant", "Brak")
                                training_config = config_from_training.get(
                                    "training", {}
                                )
                                if isinstance(
                                    training_config, dict
                                ):  # Sprawdzenie typu
                                    optimizer = training_config.get("optimizer", "Brak")
                    except json.JSONDecodeError:
                        self.parent.logger.error(
                            f"Błąd dekodowania JSON: {config_path}"
                        )
                    except Exception as e_config:  # Ogólny błąd odczytu configu
                        self.parent.logger.error(
                            f"Błąd odczytu config {config_path}: {e_config}"
                        )

                self.models_table.setItem(row, 3, QTableWidgetItem(sesje_ft))

                # Data utworzenia
                try:
                    date_val = datetime.datetime.fromtimestamp(
                        os.path.getmtime(model_path)
                    ).strftime("%Y-%m-%d %H:%M:%S")
                except OSError:
                    date_val = "Brak danych"
                self.models_table.setItem(row, 4, QTableWidgetItem(date_val))

                # Rozmiar pliku
                try:
                    size_bytes = os.path.getsize(model_path)
                    size_mb = size_bytes / (1024 * 1024)
                    size_str = f"{size_mb:.2f} MB"
                except OSError:
                    size_str = "Brak danych"
                self.models_table.setItem(row, 5, QTableWidgetItem(size_str))

                # Pozostałe dane
                self.models_table.setItem(row, 6, QTableWidgetItem(training_time))
                self.models_table.setItem(row, 7, QTableWidgetItem(str(architecture)))
                self.models_table.setItem(row, 8, QTableWidgetItem(str(variant)))
                self.models_table.setItem(row, 9, QTableWidgetItem(str(categories)))
                self.models_table.setItem(row, 10, QTableWidgetItem(str(optimizer)))

        except Exception as e:
            self.parent.logger.error(f"Błąd podczas odświeżania listy modeli: {e}")

    def _handle_load_button_click(self):
        """Obsługuje kliknięcie przycisku 'Załaduj model'.
        Pobiera zaznaczony model i wywołuje self.load_model()."""
        try:
            self.parent.logger.info("Kliknięto przycisk Załaduj model")
            selected_rows = []
            for i in range(self.models_table.rowCount()):
                item = self.models_table.item(i, 0)  # Kolumna z checkboxem
                if item and item.checkState() == Qt.CheckState.Checked:
                    selected_rows.append(i)

            model_to_load_name = None
            if selected_rows:  # Jeśli są jakieś zaznaczone checkboxy
                row = selected_rows[0]  # Bierzemy pierwszy zaznaczony
                # Kolumna z nazwą modelu
                model_item = self.models_table.item(row, 1)
                if model_item:
                    model_to_load_name = model_item.text()
            # Jeśli brak zaznaczonych checkboxów, spróbuj z aktualnie
            # podświetlonym wierszem
            else:
                current_row = self.models_table.currentRow()
                if current_row >= 0:
                    # Kolumna z nazwą modelu
                    model_item = self.models_table.item(current_row, 1)
                    if model_item:
                        model_to_load_name = model_item.text()

            if model_to_load_name:
                self.parent.logger.info(
                    f"Model wybrany przez przycisk: {model_to_load_name}"
                )
                self.load_model(model_to_load_name)
            else:
                QMessageBox.warning(
                    self, "Ostrzeżenie", "Wybierz model do załadowania z listy."
                )
                self.parent.logger.warning(
                    "Nie wybrano modelu do załadowania przyciskiem."
                )

        except Exception as e:
            err_msg = f"Błąd w _handle_load_button_click: {e} {traceback.format_exc()}"
            self.parent.logger.error(err_msg)
            QMessageBox.critical(self, "Błąd", f"Wystąpił nieoczekiwany błąd: {e}")

    def load_model(self, model_name_to_load: str):
        """Ładuje model na podstawie przekazanej nazwy."""
        try:
            self.parent.logger.info(
                f"Rozpoczynam ładowanie modelu: {model_name_to_load}"
            )

            if not model_name_to_load:
                QMessageBox.warning(
                    self, "Ostrzeżenie", "Nie podano nazwy modelu do załadowania."
                )
                self.parent.logger.warning("Nie podano nazwy modelu w load_model.")
                return

            # Bezpieczne pobieranie wartości z ustawień
            models_dir = self.settings.get("models_dir", "data/models")
            model_path = os.path.join(models_dir, model_name_to_load)

            if not os.path.exists(model_path):
                QMessageBox.warning(
                    self,
                    "Ostrzeżenie",
                    f"Plik modelu {model_name_to_load} nie istnieje.",
                )
                self.parent.logger.error(f"Plik modelu nie istnieje: {model_path}")
                return

            self.parent.logger.info(f"Ładowanie modelu z: {model_path}")

            # Dodane: Zwolnienie zasobów poprzedniego modelu przed załadowaniem nowego
            if (
                hasattr(self.parent, "classifier")
                and self.parent.classifier is not None
            ):
                try:
                    self.parent.logger.info("Zwalnianie zasobów poprzedniego modelu...")
                    if (
                        hasattr(self.parent.classifier, "model")
                        and self.parent.classifier.model is not None
                    ):
                        # Usunięcie modelu z GPU jeśli był tam przeniesiony
                        self.parent.classifier.model.cpu()
                        # Usuń referencje do modelu
                        self.parent.classifier.model = None

                    # Wywołanie garbage collectora
                    import gc

                    gc.collect()

                    # Jeśli używamy PyTorch z CUDA, wyczyść również pamięć CUDA
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    self.parent.logger.info(
                        "Zasoby poprzedniego modelu zostały zwolnione"
                    )
                except Exception as e_cleanup:
                    self.parent.logger.warning(
                        f"Błąd podczas zwalniania zasobów poprzedniego modelu: {e_cleanup}"
                    )

            # Tworzenie nowego klasyfikatora
            try:
                self.parent.logger.info(
                    f"Tworzenie nowego klasyfikatora dla modelu: {model_name_to_load}"
                )
                self.parent.classifier = ImageClassifier(weights_path=model_path)
            except Exception as e_load:
                self.parent.logger.error(f"Błąd ładowania modelu: {e_load}")
                QMessageBox.critical(
                    self,
                    "Błąd",
                    f"Nie udało się załadować modelu {model_name_to_load}: {e_load}",
                )
                return

            self.parent.model_loaded = True
            self.parent.model_path = model_path
            self.parent.current_model = model_name_to_load

            if hasattr(self.parent, "_update_active_model_info"):
                self.parent._update_active_model_info()

            if hasattr(self.parent, "batch_processor_tab") and hasattr(
                self.parent.batch_processor_tab, "refresh"
            ):
                self.parent.batch_processor_tab.refresh()

            QMessageBox.information(
                self,
                "Sukces",
                f"Model {model_name_to_load} został załadowany pomyślnie.",
            )
            self.parent.logger.info(
                f"Model {model_name_to_load} został pomyślnie załadowany"
            )

        except Exception as e:
            err_msg = (
                f"Błąd ładowania modelu ({model_name_to_load}): "
                f"{e} {traceback.format_exc()}"
            )
            self.parent.logger.error(err_msg)
            QMessageBox.critical(
                self,
                "Błąd",
                f"Nie udało się załadować modelu {model_name_to_load}: {e}",
            )

    def _export_selected_model(self):
        """Eksportuje model wybrany z tabeli modeli."""
        try:
            self.parent.logger.info("Rozpoczynam eksport wybranego modelu...")
            selected_rows = []
            for i in range(self.models_table.rowCount()):
                item = self.models_table.item(i, 0)
                if item and item.checkState() == Qt.CheckState.Checked:
                    selected_rows.append(i)

            if not selected_rows:
                current_row = self.models_table.currentRow()
                if current_row >= 0:
                    selected_rows = [current_row]
                    self.parent.logger.info(
                        f"Wybrano aktualnie zaznaczony wiersz: {current_row}"
                    )

            if not selected_rows:
                self.parent.logger.warning("Nie wybrano modelu do eksportu")
                QMessageBox.warning(self, "Ostrzeżenie", "Wybierz model do eksportu.")
                return

            row = selected_rows[0]
            model_name = self.models_table.item(row, 1).text()
            self.parent.logger.info(f"Wybrano model do eksportu: {model_name}")
            source_path = os.path.join(self.settings["models_dir"], model_name)

            if not os.path.exists(source_path):
                self.parent.logger.error(f"Plik modelu nie istnieje: {source_path}")
                QMessageBox.warning(
                    self, "Ostrzeżenie", f"Plik modelu {model_name} nie istnieje."
                )
                return

            export_path, _ = QFileDialog.getSaveFileName(
                self,
                "Eksportuj model",
                model_name,
                "Pliki modeli (*.pt *.pth);;Wszystkie pliki (*.*)",
            )

            if not export_path:
                self.parent.logger.info("Anulowano eksport modelu")
                return

            self.parent.logger.info(f"Kopiowanie modelu do: {export_path}")
            shutil.copy2(source_path, export_path)

            config_path = os.path.splitext(source_path)[0] + "_config.json"
            if os.path.exists(config_path):
                export_config_path = os.path.splitext(export_path)[0] + "_config.json"
                self.parent.logger.info(
                    "Kopiowanie pliku konfiguracyjnego do: " f"{export_config_path}"
                )
                shutil.copy2(config_path, export_config_path)

            QMessageBox.information(
                self,
                "Sukces",
                f"Model {model_name} został wyeksportowany do {export_path}",
            )
            self.parent.logger.info(
                f"Model {model_name} pomyślnie wyeksportowany do {export_path}"
            )

        except Exception as e:
            self.parent.logger.error(f"Błąd podczas eksportu modelu: {e}")
            QMessageBox.critical(
                self, "Błąd", f"Nie udało się wyeksportować modelu: {e}"
            )

    def _export_model_config(self):
        """Eksportuje konfigurację wybranego modelu."""
        try:
            self.parent.logger.info("Rozpoczynam eksport konfiguracji modelu...")
            selected_models = []
            for row in range(self.models_table.rowCount()):
                item = self.models_table.item(row, 0)
                if item and item.checkState() == Qt.CheckState.Checked:
                    model_name = self.models_table.item(row, 1).text()
                    selected_models.append(model_name)

            if len(selected_models) != 1:
                err_msg = (
                    "Wybrano nieprawidłową liczbę modeli: " f"{len(selected_models)}"
                )
                self.parent.logger.warning(err_msg)
                QMessageBox.warning(
                    self,
                    "Błąd",
                    "Wybierz dokładnie jeden model do eksportu konfiguracji.",
                )
                return

            model_name = selected_models[0]
            self.parent.logger.info(
                f"Wybrano model do eksportu konfiguracji: {model_name}"
            )
            models_dir = self.settings.get("models_dir", "data/models")
            config_path = os.path.join(
                models_dir, os.path.splitext(model_name)[0] + "_config.json"
            )

            if not os.path.exists(config_path):
                self.parent.logger.error(
                    f"Plik konfiguracyjny nie istnieje: {config_path}"
                )
                QMessageBox.warning(
                    self, "Błąd", "Brak pliku konfiguracyjnego dla wybranego modelu."
                )
                return

            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Zapisz konfigurację",
                os.path.splitext(model_name)[0] + "_config.json",
                "Pliki JSON (*.json)",
            )

            if file_path:
                self.parent.logger.info(f"Kopiowanie konfiguracji do: {file_path}")
                shutil.copy2(config_path, file_path)
                QMessageBox.information(
                    self, "Sukces", "Konfiguracja została wyeksportowana pomyślnie."
                )
                self.parent.logger.info("Konfiguracja została pomyślnie wyeksportowana")
            else:
                self.parent.logger.info("Anulowano eksport konfiguracji")

        except Exception as e:
            err_msg = f"Błąd podczas eksportu konfiguracji: {e}"
            self.parent.logger.error(err_msg)
            QMessageBox.critical(
                self, "Błąd", f"Nie udało się wyeksportować konfiguracji: {e}"
            )

    def _import_model_config(self):
        """Importuje konfigurację dla wybranego modelu."""
        try:
            self.parent.logger.info("Rozpoczynam import konfiguracji modelu...")
            selected_models = []
            for row in range(self.models_table.rowCount()):
                item = self.models_table.item(row, 0)
                if item and item.checkState() == Qt.CheckState.Checked:
                    model_name = self.models_table.item(row, 1).text()
                    selected_models.append(model_name)

            if len(selected_models) != 1:
                err_msg = (
                    "Wybrano nieprawidłową liczbę modeli: " f"{len(selected_models)}"
                )
                self.parent.logger.warning(err_msg)
                QMessageBox.warning(
                    self,
                    "Błąd",
                    "Wybierz dokładnie jeden model do importu konfiguracji.",
                )
                return

            model_name = selected_models[0]
            self.parent.logger.info(
                f"Wybrano model do importu konfiguracji: {model_name}"
            )
            models_dir = self.settings.get("models_dir", "data/models")
            target_config_path = os.path.join(
                models_dir, os.path.splitext(model_name)[0] + "_config.json"
            )

            file_path, _ = QFileDialog.getOpenFileName(
                self, "Wybierz plik konfiguracyjny", "", "Pliki JSON (*.json)"
            )

            if file_path:
                self.parent.logger.info(f"Importowanie konfiguracji z: {file_path}")
                try:
                    with open(file_path, "r", encoding="utf-8") as f:  # Dodano encoding
                        config = json.load(f)
                    if not isinstance(config, dict):
                        raise ValueError("Nieprawidłowy format konfiguracji")

                    shutil.copy2(file_path, target_config_path)
                    QMessageBox.information(
                        self, "Sukces", "Konfiguracja została zaimportowana pomyślnie."
                    )
                    self.parent.logger.info(
                        "Konfiguracja została pomyślnie zaimportowana"
                    )
                    self.refresh()
                except json.JSONDecodeError:
                    self.parent.logger.error(
                        "Wybrany plik nie jest poprawnym plikiem JSON"
                    )
                    QMessageBox.critical(
                        self, "Błąd", "Wybrany plik nie jest poprawnym plikiem JSON."
                    )
                except ValueError as e_val:
                    self.parent.logger.error(f"Błąd w formacie konfiguracji: {e_val}")
                    QMessageBox.critical(self, "Błąd", str(e_val))
            else:
                self.parent.logger.info("Anulowano import konfiguracji")

        except Exception as e:
            err_msg = f"Błąd podczas importu konfiguracji: {e}"
            self.parent.logger.error(err_msg)
            QMessageBox.critical(
                self, "Błąd", f"Nie udało się zaimportować konfiguracji: {e}"
            )

    def _rename_selected_model(self):
        """Zmienia nazwę wybranego modelu."""
        try:
            self.parent.logger.info("Rozpoczynam zmianę nazwy modelu...")
            selected_rows = []
            for i in range(self.models_table.rowCount()):
                item = self.models_table.item(i, 0)
                if item and item.checkState() == Qt.CheckState.Checked:
                    selected_rows.append(i)

            if not selected_rows:
                current_row = self.models_table.currentRow()
                if current_row >= 0:
                    selected_rows = [current_row]
                    self.parent.logger.info(
                        "Wybrano aktualnie zaznaczony wiersz: " f"{current_row}"
                    )

            if not selected_rows:
                self.parent.logger.warning("Nie wybrano modelu do zmiany nazwy")
                QMessageBox.warning(
                    self, "Ostrzeżenie", "Wybierz model do zmiany nazwy."
                )
                return
            if len(selected_rows) > 1:
                self.parent.logger.warning(
                    "Wybrano więcej niż jeden model do zmiany nazwy"
                )
                QMessageBox.warning(
                    self, "Ostrzeżenie", "Wybierz tylko jeden model do zmiany nazwy."
                )
                return

            row = selected_rows[0]
            old_name = self.models_table.item(row, 1).text()
            old_path = os.path.join(self.settings["models_dir"], old_name)
            self.parent.logger.info(f"Zmiana nazwy modelu: {old_name}")

            new_name_base, ok = QInputDialog.getText(
                self,
                "Zmiana nazwy modelu",
                "Nowa nazwa modelu:",
                text=os.path.splitext(old_name)[0],  # Sugeruj bez rozszerzenia
            )

            if not ok or not new_name_base:
                self.parent.logger.info("Anulowano zmianę nazwy modelu")
                return

            # Dodaj oryginalne rozszerzenie
            new_name = new_name_base + os.path.splitext(old_name)[1]
            self.parent.logger.info(f"Pełna nowa nazwa klona: {new_name}")

            new_path = os.path.join(self.settings["models_dir"], new_name)
            if os.path.exists(new_path):
                self.parent.logger.warning(
                    f"Plik o nowej nazwie już istnieje: {new_path}"
                )
                QMessageBox.warning(
                    self, "Ostrzeżenie", f"Plik {new_name} już istnieje."
                )
                return

            os.rename(old_path, new_path)
            self.parent.logger.info(
                f"Zmieniono nazwę pliku modelu z {old_path} na {new_path}"
            )

            old_config_path = os.path.splitext(old_path)[0] + "_config.json"
            if os.path.exists(old_config_path):
                new_config_path = os.path.splitext(new_path)[0] + "_config.json"
                os.rename(old_config_path, new_config_path)
                self.parent.logger.info(
                    f"Zmieniono nazwę pliku konfiguracyjnego z "
                    f"{old_config_path} na {new_config_path}"
                )

            self.models_table.item(row, 1).setText(new_name)

            if (
                hasattr(self.parent, "model_loaded")
                and self.parent.model_loaded
                and hasattr(self.parent, "current_model")
                and self.parent.current_model == old_name
            ):
                self.parent.model_path = new_path
                self.parent.current_model = new_name  # Aktualizuj też current_model
                self.parent.logger.info(
                    f"Zaktualizowano ścieżkę aktywnego modelu na: {new_path}"
                )

            QMessageBox.information(
                self, "Sukces", f"Nazwa modelu została zmieniona na {new_name}."
            )
            self.parent.logger.info(
                f"Pomyślnie zmieniono nazwę modelu z {old_name} na {new_name}"
            )
            self.refresh()  # Odśwież listę

        except Exception as e:
            err_msg = f"Błąd podczas zmiany nazwy modelu: {e}"
            self.parent.logger.error(err_msg)
            QMessageBox.critical(
                self, "Błąd", f"Nie udało się zmienić nazwy modelu: {e}"
            )

    def _clone_selected_model(self):
        """Klonuje wybrany model."""
        try:
            self.parent.logger.info("Rozpoczynam klonowanie modelu...")
            selected_rows = []
            for i in range(self.models_table.rowCount()):
                item = self.models_table.item(i, 0)
                if item and item.checkState() == Qt.CheckState.Checked:
                    selected_rows.append(i)

            if not selected_rows:
                current_row = self.models_table.currentRow()
                if current_row >= 0:
                    selected_rows = [current_row]
                    self.parent.logger.info(
                        "Wybrano aktualnie zaznaczony wiersz: " f"{current_row}"
                    )

            if not selected_rows:
                self.parent.logger.warning("Nie wybrano modelu do klonowania")
                QMessageBox.warning(self, "Ostrzeżenie", "Wybierz model do klonowania.")
                return
            if len(selected_rows) > 1:
                self.parent.logger.warning(
                    "Wybrano więcej niż jeden model do klonowania"
                )
                QMessageBox.warning(
                    self, "Ostrzeżenie", "Wybierz tylko jeden model do klonowania."
                )
                return

            row = selected_rows[0]
            old_name = self.models_table.item(row, 1).text()
            old_path = os.path.join(self.settings["models_dir"], old_name)
            self.parent.logger.info(f"Klonowanie modelu: {old_name}")

            base_name, ext = os.path.splitext(old_name)
            new_name_base, ok = QInputDialog.getText(
                self,
                "Klonowanie modelu",
                "Nazwa kopii modelu:",
                text=f"{base_name}_copy",
            )

            if not ok or not new_name_base:
                self.parent.logger.info("Anulowano klonowanie modelu")
                return

            new_name = new_name_base + ext  # Dodaj oryginalne rozszerzenie
            self.parent.logger.info(f"Pełna nowa nazwa klona: {new_name}")

            new_path = os.path.join(self.settings["models_dir"], new_name)
            if os.path.exists(new_path):
                self.parent.logger.warning(
                    f"Plik o nowej nazwie już istnieje: {new_path}"
                )
                QMessageBox.warning(
                    self, "Ostrzeżenie", f"Plik {new_name} już istnieje."
                )
                return

            shutil.copy2(old_path, new_path)
            self.parent.logger.info(
                f"Skopiowano plik modelu z {old_path} do {new_path}"
            )

            old_config_path = os.path.splitext(old_path)[0] + "_config.json"
            if os.path.exists(old_config_path):
                new_config_path = os.path.splitext(new_path)[0] + "_config.json"
                shutil.copy2(old_config_path, new_config_path)
                self.parent.logger.info(
                    f"Skopiowano plik konfiguracyjny z "
                    f"{old_config_path} do {new_config_path}"
                )

            self.refresh()

            QMessageBox.information(
                self, "Sukces", f"Model został sklonowany jako {new_name}."
            )
            self.parent.logger.info(
                f"Pomyślnie sklonowano model {old_name} jako {new_name}"
            )

        except Exception as e:
            err_msg = f"Błąd podczas klonowania modelu: {e}"
            self.parent.logger.error(err_msg)
            QMessageBox.critical(self, "Błąd", f"Nie udało się sklonować modelu: {e}")

    def _show_class_mapping(self):
        """Wyświetla mapowanie indeksów klas na nazwy klas."""
        try:
            self.parent.logger.info("Rozpoczynam wyświetlanie mapowania klas...")
            if not hasattr(self.parent, "classifier") or self.parent.classifier is None:
                self.parent.logger.warning("Nie załadowano modelu")
                QMessageBox.warning(
                    self,
                    "Brak modelu",
                    "Nie załadowano modelu. Proszę najpierw załadować model.",
                )
                return

            self.parent.logger.info("Pobieranie mapowania klas z klasyfikatora")
            class_mapping = self.parent.classifier.get_class_mapping()

            if not class_mapping:  # Sprawdza czy słownik nie jest pusty
                self.parent.logger.warning("Brak dostępnego mapowania klas")
                QMessageBox.information(
                    self,
                    "Brak mapowania klas",
                    "Brak dostępnego mapowania klas dla aktualnego modelu.",
                )
                return

            dialog = QDialog(self)
            dialog.setWindowTitle("Mapowanie klas")
            dialog.setMinimumSize(400, 300)
            layout = QVBoxLayout(dialog)
            label = QLabel("Mapowanie indeksów klas na nazwy:")
            layout.addWidget(label)

            table = QTableWidget()
            table.setColumnCount(2)
            table.setHorizontalHeaderLabels(["Indeks", "Nazwa klasy"])
            table.setRowCount(len(class_mapping))

            for i, (class_idx, class_name) in enumerate(
                sorted(
                    class_mapping.items(),
                    key=lambda item: (
                        int(item[0]) if item[0].isdigit() else float("inf")
                    ),
                )
            ):  # Sortowanie po kluczach numerycznych
                idx_item = QTableWidgetItem(str(class_idx))
                name_item = QTableWidgetItem(
                    str(class_name)
                )  # Upewnij się, że nazwa jest stringiem
                table.setItem(i, 0, idx_item)
                table.setItem(i, 1, name_item)

            table.horizontalHeader().setSectionResizeMode(
                0, QHeaderView.ResizeMode.ResizeToContents
            )
            table.horizontalHeader().setSectionResizeMode(
                1, QHeaderView.ResizeMode.Stretch
            )
            layout.addWidget(table)

            button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
            button_box.rejected.connect(dialog.reject)
            layout.addWidget(button_box)

            self.parent.logger.info("Wyświetlanie okna z mapowaniem klas")
            dialog.exec()

        except Exception as e:
            err_msg = f"Błąd podczas wyświetlania mapowania klas: {e}"
            self.parent.logger.error(err_msg)
            QMessageBox.critical(
                self, "Błąd", f"Wystąpił błąd podczas pobierania mapowania klas: {e}"
            )

    def _show_model_stats(self):
        """Wyświetla statystyki modelu."""
        try:
            self.parent.logger.info("Rozpoczynam wyświetlanie statystyk modelu...")
            selected_rows = []
            for i in range(self.models_table.rowCount()):
                item = self.models_table.item(i, 0)
                if item and item.checkState() == Qt.CheckState.Checked:
                    selected_rows.append(i)

            if not selected_rows:
                current_row = self.models_table.currentRow()
                if current_row >= 0:
                    selected_rows = [current_row]
                    self.parent.logger.info(
                        "Wybrano aktualnie zaznaczony wiersz: " f"{current_row}"
                    )

            if not selected_rows:
                self.parent.logger.warning(
                    "Nie wybrano modelu do wyświetlenia statystyk"
                )
                QMessageBox.warning(
                    self, "Ostrzeżenie", "Wybierz model, aby wyświetlić statystyki."
                )
                return

            row = selected_rows[0]
            model_name = self.models_table.item(row, 1).text()
            self.parent.logger.info(
                f"Wybrano model do wyświetlenia statystyk: {model_name}"
            )
            model_path = os.path.join(self.settings["models_dir"], model_name)
            config_path = os.path.splitext(model_path)[0] + "_config.json"

            stats = {}
            stats["Nazwa modelu"] = model_name
            stats["Ścieżka"] = model_path
            try:
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                stats["Rozmiar"] = f"{size_mb:.2f} MB"
                date_mod = datetime.datetime.fromtimestamp(
                    os.path.getmtime(model_path)
                ).strftime("%Y-%m-%d %H:%M:%S")
                stats["Data modyfikacji"] = date_mod
            except OSError as e_os:
                stats["Rozmiar"] = f"Błąd: {e_os.strerror}"
                stats["Data modyfikacji"] = f"Błąd: {e_os.strerror}"

            # Próba załadowania modelu tymczasowo tylko dla informacji
            temp_classifier = None
            try:
                if os.path.exists(model_path):
                    self.parent.logger.info(
                        "Ładowanie modelu w celu pobrania dodatkowych informacji"
                    )
                    # Użyj parametrów z aktualnego klasyfikatora jeśli to możliwe
                    # lub domyślnych, jeśli klasyfikator nie istnieje
                    current_model_type = "b0"  # Domyślny
                    current_num_classes = 10  # Domyślny
                    if hasattr(self.parent, "classifier") and self.parent.classifier:
                        current_model_type = self.parent.classifier.model_type
                        current_num_classes = self.parent.classifier.num_classes

                    temp_classifier = ImageClassifier(
                        model_type=current_model_type,
                        num_classes=current_num_classes,  # Użyj num_classes z pliku .pt
                        weights_path=model_path,
                    )
                    model_info = temp_classifier.get_model_info()
                    for k, v_item in model_info.items():
                        # Prefiksujemy klucze z get_model_info aby uniknąć kolizji
                        stats[f"model_info.{k}"] = v_item
                else:
                    stats["model_info"] = (
                        "Plik modelu nie istnieje, nie można załadować."
                    )
            except Exception as e_load:
                self.parent.logger.error(
                    f"Błąd podczas tymczasowego ładowania modelu dla statystyk: {e_load}"
                )
                stats["model_info.Error"] = str(e_load)
            finally:
                if temp_classifier:
                    del temp_classifier  # Zwolnij zasoby

            def flatten_dict(d, parent_key="", sep="."):
                items = []
                for k, v_item in d.items():
                    new_key = f"{parent_key}{sep}{k}" if parent_key else k
                    if isinstance(v_item, dict):
                        # Ogranicz głębokość rekursji lub spłaszczaj ostrożnie
                        # Tutaj prostsze spłaszczenie dla przykładu
                        items.extend(flatten_dict(v_item, new_key, sep=sep).items())
                    elif isinstance(v_item, list):
                        # Konwertuj listy na stringi lub obsługuj inaczej
                        items.append((new_key, str(v_item)))
                    else:
                        items.append((new_key, v_item))
                return dict(items)

            if os.path.exists(config_path):
                try:
                    self.parent.logger.info(
                        "Odczyt wszystkich danych z pliku konfiguracyjnego"
                    )
                    with open(
                        config_path, "r", encoding="utf-8"
                    ) as f:  # Dodano encoding
                        config = json.load(f)
                    flat_config = flatten_dict(config)
                    for k, v_item in flat_config.items():
                        stats[f"config.{k}"] = v_item
                except Exception as e_cfg_comp:
                    self.parent.logger.error(
                        f"Błąd odczytu config dla {model_name}: {e_cfg_comp}"
                    )
                    stats[f"config.Error_{model_name}"] = str(e_cfg_comp)
            else:
                stats["config_file"] = "Brak"

            dialog = QDialog(self)
            dialog.setWindowTitle(f"Statystyki modelu: {model_name}")
            dialog.setMinimumSize(600, 500)  # Zwiększono rozmiar
            layout = QVBoxLayout(dialog)

            stats_table = QTableWidget()
            stats_table.setColumnCount(2)
            stats_table.setHorizontalHeaderLabels(["Parametr", "Wartość"])
            stats_table.horizontalHeader().setStretchLastSection(True)
            stats_table.setRowCount(len(stats))

            # Sortowanie kluczy dla lepszej czytelności
            sorted_keys = sorted(stats.keys())

            for i, key in enumerate(sorted_keys):
                value = stats[key]
                stats_table.setItem(i, 0, QTableWidgetItem(str(key)))
                # Konwertuj listy i słowniki na stringi dla wyświetlenia
                if isinstance(value, (list, dict)):
                    try:
                        value_str = json.dumps(value, indent=2, ensure_ascii=False)
                    except TypeError:
                        value_str = str(value)  # Fallback
                else:
                    value_str = str(value)
                stats_table.setItem(i, 1, QTableWidgetItem(value_str))

            stats_table.resizeColumnToContents(0)
            # stats_table.resizeRowsToContents() # Może być potrzebne dla długich wartości
            layout.addWidget(stats_table)

            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
            buttons.rejected.connect(dialog.reject)
            layout.addWidget(buttons)

            self.parent.logger.info("Wyświetlanie okna ze statystykami modelu")
            dialog.exec()

        except Exception as e:
            err_msg = f"Błąd podczas wyświetlania statystyk modelu: {e}"
            self.parent.logger.error(err_msg)
            QMessageBox.critical(
                self, "Błąd", f"Nie udało się wyświetlić statystyk modelu: {e}"
            )

    def _compare_models(self):
        """Porównuje wybrane modele."""
        try:
            self.parent.logger.info("Rozpoczynam porównywanie modeli...")
            selected_model_info = []  # Zmieniono na listę krotek (row, name)
            for row in range(self.models_table.rowCount()):
                item = self.models_table.item(row, 0)
                if item and item.checkState() == Qt.CheckState.Checked:
                    model_name = self.models_table.item(row, 1).text()
                    selected_model_info.append(model_name)  # Tylko nazwy

            if len(selected_model_info) < 2:
                self.parent.logger.warning("Wybrano za mało modeli do porównania")
                QMessageBox.warning(
                    self, "Ostrzeżenie", "Wybierz co najmniej dwa modele do porównania."
                )
                return

            self.parent.logger.info(
                f"Wybrano {len(selected_model_info)} modeli do porównania"
            )
            dialog = QDialog(self)
            dialog.setWindowTitle("Porównanie modeli")
            dialog.setMinimumWidth(800)
            dialog.setMinimumHeight(600)
            layout = QVBoxLayout(dialog)
            info_label = QLabel(f"Porównanie {len(selected_model_info)} modeli:")
            layout.addWidget(info_label)

            comp_table = QTableWidget()
            comp_table.setColumnCount(len(selected_model_info) + 1)
            headers = ["Parametr"] + selected_model_info  # Użyj nazw modeli
            comp_table.setHorizontalHeaderLabels(headers)

            # Zbieranie danych dla każdego modelu
            model_data_list = []
            all_param_keys = set()  # Zestaw wszystkich unikalnych kluczy parametrów

            def robust_flatten_dict(d, parent_key="", sep="."):
                items = []
                for k, v_item in d.items():
                    new_key = f"{parent_key}{sep}{k}" if parent_key else k
                    if isinstance(v_item, dict):
                        # Ogranicz głębokość rekursji lub spłaszczaj ostrożnie
                        # Tutaj prostsze spłaszczenie dla przykładu
                        items.extend(
                            robust_flatten_dict(v_item, new_key, sep=sep).items()
                        )
                    elif isinstance(v_item, list):
                        # Konwertuj listy na stringi lub obsługuj inaczej
                        items.append((new_key, str(v_item)))
                    else:
                        items.append((new_key, v_item))
                return dict(items)

            models_dir = self.settings.get("models_dir", "data/models")
            for model_name in selected_model_info:
                model_path = os.path.join(models_dir, model_name)
                config_path = os.path.splitext(model_path)[0] + "_config.json"
                data_for_model = {"Nazwa modelu": model_name}

                try:
                    data_for_model["Rozmiar (MB)"] = (
                        f"{os.path.getsize(model_path) / (1024 * 1024):.2f}"
                    )
                    data_for_model["Data modyfikacji"] = (
                        datetime.datetime.fromtimestamp(
                            os.path.getmtime(model_path)
                        ).strftime("%Y-%m-%d %H:%M:%S")
                    )
                except OSError:
                    data_for_model["Rozmiar (MB)"] = "Błąd"
                    data_for_model["Data modyfikacji"] = "Błąd"

                if os.path.exists(config_path):
                    try:
                        with open(
                            config_path, "r", encoding="utf-8"
                        ) as f:  # Dodano encoding
                            config = json.load(f)
                        # Spłaszcz konfigurację, prefiksując klucze, aby uniknąć kolizji
                        flat_config = robust_flatten_dict(config)
                        data_for_model.update(flat_config)
                    except Exception as e_cfg_comp:
                        self.parent.logger.error(
                            f"Błąd odczytu config dla {model_name}: {e_cfg_comp}"
                        )
                        data_for_model[f"config.Error_{model_name}"] = str(e_cfg_comp)
                else:
                    data_for_model[f"config_file_{model_name}"] = "Brak"

                # Dodaj wszystkie klucze z tego modelu do zbioru all_param_keys
                all_param_keys.update(data_for_model.keys())
                model_data_list.append(data_for_model)

            # Ustal kolejność wyświetlania parametrów (np. alfabetycznie)
            # Usuń "Nazwa modelu", bo jest w nagłówkach
            sorted_param_keys = sorted(
                [k for k in all_param_keys if k != "Nazwa modelu"]
            )

            comp_table.setRowCount(len(sorted_param_keys))
            for i, param_key in enumerate(sorted_param_keys):
                comp_table.setItem(i, 0, QTableWidgetItem(param_key))
                for col_idx, model_data in enumerate(model_data_list, start=1):
                    value = model_data.get(param_key, "N/A")
                    comp_table.setItem(i, col_idx, QTableWidgetItem(str(value)))

            comp_table.horizontalHeader().setSectionResizeMode(
                0, QHeaderView.ResizeMode.ResizeToContents
            )
            for i in range(1, comp_table.columnCount()):
                comp_table.horizontalHeader().setSectionResizeMode(
                    i, QHeaderView.ResizeMode.Stretch
                )
            layout.addWidget(comp_table)

            button_layout = QHBoxLayout()
            export_csv_btn = QPushButton("Eksportuj do CSV")
            export_csv_btn.clicked.connect(
                lambda: self._export_comparison_to_csv(comp_table)
            )
            button_layout.addWidget(export_csv_btn)
            close_btn = QPushButton("Zamknij")
            close_btn.clicked.connect(dialog.accept)
            button_layout.addWidget(close_btn)
            button_layout.addStretch(1)
            layout.addLayout(button_layout)

            self.parent.logger.info("Wyświetlanie okna porównania modeli")
            dialog.exec()

        except Exception as e:
            err_msg = f"Błąd podczas porównywania modeli: {e} {traceback.format_exc()}"
            self.parent.logger.error(err_msg)
            QMessageBox.critical(
                self, "Błąd", f"Wystąpił błąd podczas porównywania modeli: {e}"
            )

    def _export_comparison_to_csv(self, table):
        """Eksportuje porównanie modeli do pliku CSV."""
        try:
            self.parent.logger.info("Rozpoczynam eksport porównania do CSV...")
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Zapisz porównanie jako CSV", "", "Pliki CSV (*.csv)"
            )

            if not file_path:
                self.parent.logger.info("Anulowano eksport do CSV")
                return

            if not file_path.endswith(".csv"):
                file_path += ".csv"

            self.parent.logger.info(f"Zapisywanie porównania do pliku: {file_path}")
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                headers = [
                    table.horizontalHeaderItem(col).text()
                    for col in range(table.columnCount())
                ]
                f.write(",".join([f'"{h}"' for h in headers]) + "\n")

                for row in range(table.rowCount()):
                    row_data = []
                    for col in range(table.columnCount()):
                        item = table.item(row, col)
                        cell_text = item.text() if item else ""
                        # Ucieczka cudzysłowów w danych
                        cell_text = cell_text.replace('"', '""')
                        row_data.append(f'"{cell_text}"')
                    f.write(",".join(row_data) + "\n")

            QMessageBox.information(
                self,
                "Sukces",
                f"Porównanie zostało wyeksportowane do pliku:\n{file_path}",
            )
            self.parent.logger.info("Pomyślnie wyeksportowano porównanie do CSV")

        except Exception as e:
            err_msg = f"Błąd podczas eksportu porównania do CSV: {e}"
            self.parent.logger.error(err_msg)
            QMessageBox.critical(
                self, "Błąd", f"Wystąpił błąd podczas eksportu porównania: {e}"
            )

    def _delete_selected_model(self):
        """Usuwa wybrany model."""
        try:
            self.parent.logger.info("Rozpoczynam usuwanie modelu...")
            selected_rows = []
            for i in range(self.models_table.rowCount()):
                item = self.models_table.item(i, 0)
                if item and item.checkState() == Qt.CheckState.Checked:
                    selected_rows.append(i)

            if not selected_rows:
                current_row = self.models_table.currentRow()
                if current_row >= 0:
                    selected_rows = [current_row]
                    self.parent.logger.info(
                        "Wybrano aktualnie zaznaczony wiersz do usunięcia: "
                        f"{current_row}"
                    )

            if not selected_rows:
                self.parent.logger.warning("Nie wybrano modelu do usunięcia")
                QMessageBox.warning(self, "Ostrzeżenie", "Wybierz model do usunięcia.")
                return
            if len(selected_rows) > 1:
                self.parent.logger.warning(
                    "Wybrano więcej niż jeden model do usunięcia"
                )
                QMessageBox.warning(
                    self, "Ostrzeżenie", "Wybierz tylko jeden model do usunięcia."
                )
                return

            row_to_delete = selected_rows[0]
            model_name_to_delete = self.models_table.item(row_to_delete, 1).text()
            log_msg = f"Model do usunięcia: {model_name_to_delete}"
            self.parent.logger.info(log_msg)

            if (
                hasattr(self.parent, "model_loaded")
                and self.parent.model_loaded
                and hasattr(self.parent, "current_model")
                and self.parent.current_model == model_name_to_delete
            ):
                err_msg = "Próba usunięcia aktywnego modelu: " f"{model_name_to_delete}"
                self.parent.logger.warning(err_msg)
                QMessageBox.critical(
                    self,
                    "Błąd",
                    f"Nie można usunąć aktywnego modelu '{model_name_to_delete}'.\n"
                    "Załaduj inny model lub odznacz go jako aktywny.",
                )
                return

            reply = QMessageBox.question(
                self,
                "Potwierdź usunięcie",
                f"Czy na pewno chcesz usunąć model '{model_name_to_delete}' "
                f"oraz jego plik konfiguracyjny?\nTej operacji nie można cofnąć.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.No:
                self.parent.logger.info("Anulowano usuwanie modelu.")
                return

            models_dir = self.settings.get("models_dir", "data/models")
            model_path_to_delete = os.path.join(models_dir, model_name_to_delete)
            config_filename = os.path.splitext(model_name_to_delete)[0] + "_config.json"
            config_path_to_delete = os.path.join(models_dir, config_filename)

            deleted_model_file = False
            if os.path.exists(model_path_to_delete):
                try:
                    os.remove(model_path_to_delete)
                    deleted_model_file = True
                    self.parent.logger.info(
                        f"Usunięto plik modelu: {model_path_to_delete}"
                    )
                except OSError as e_os_model:
                    self.parent.logger.error(
                        "Błąd OS przy usuwaniu pliku modelu "
                        f"{model_path_to_delete}: {e_os_model}"
                    )
                    QMessageBox.critical(
                        self, "Błąd", f"Nie udało się usunąć pliku modelu: {e_os_model}"
                    )
                    return  # Nie kontynuuj, jeśli główny plik nie został usunięty

            deleted_config_file = False
            if os.path.exists(config_path_to_delete):
                try:
                    os.remove(config_path_to_delete)
                    deleted_config_file = True
                    self.parent.logger.info(
                        f"Usunięto plik konfiguracyjny: {config_path_to_delete}"
                    )
                except OSError as e_os_config:
                    # Loguj błąd, ale nie przerywaj, jeśli plik modelu został usunięty
                    self.parent.logger.error(
                        "Błąd OS przy usuwaniu pliku konfiguracyjnego "
                        f"{config_path_to_delete}: {e_os_config}"
                    )
                    QMessageBox.warning(
                        self,
                        "Ostrzeżenie",
                        "Nie udało się usunąć pliku konfiguracyjnego: "
                        f"{e_os_config}.\nPlik modelu mógł zostać usunięty.",
                    )

            if deleted_model_file or deleted_config_file:  # Jeśli cokolwiek usunięto
                self.refresh()
                QMessageBox.information(
                    self, "Sukces", f"Model '{model_name_to_delete}' został usunięty."
                )
                self.parent.logger.info(
                    f"Pomyślnie przetworzono usuwanie dla: {model_name_to_delete}"
                )
            else:
                QMessageBox.warning(
                    self,
                    "Informacja",
                    f"Nie znaleziono plików do usunięcia dla modelu '{model_name_to_delete}'.",
                )

        except Exception as e:
            detailed_error = traceback.format_exc()
            err_msg = f"Błąd podczas usuwania modelu: {e}\n{detailed_error}"
            self.parent.logger.error(err_msg)
            QMessageBox.critical(self, "Błąd", f"Nie udało się usunąć modelu: {e}")

    def update_hardware_profile(self, profile):
        """Aktualizuje profil sprzętowy używany do optymalizacji."""
        self.hardware_profile = profile
        self.parent.logger.info("Zaktualizowano profil sprzętowy w zakładce modeli")

        # Aktualizacja UI na podstawie nowego profilu
        # Przykład: Można by tu wyświetlić informacje o profilu
        # lub dostosować dostępne opcje.
        # if hasattr(self, "use_optimization_checkbox"):
        #     self.use_optimization_checkbox.setChecked(True)

        # if profile and hasattr(self, "profile_info_label"):
        #     cpu_info = profile.get("cpu_info", {})
        #     if isinstance(cpu_info, str):
        #         try:
        #             cpu_info = json.loads(cpu_info)
        #         except json.JSONDecodeError:
        #             cpu_info = {}

        #     gpu_info = profile.get("gpu_info", {})
        #     if isinstance(gpu_info, str):
        #         try:
        #             gpu_info = json.loads(gpu_info)
        #         except json.JSONDecodeError:
        #             gpu_info = {}

        #     info_text = "Status profilu: Aktywny\n"
        #     info_text += f"CPU: {cpu_info.get('name', 'Nieznany')}\n"
        #     if gpu_info:
        #         info_text += f"GPU: {gpu_info.get('name', 'Nieznany')}\n"
        #     info_text += f"RAM: {profile.get('ram_total', 0):.1f} GB"
        #     # self.profile_info_label.setText(info_text)
