import datetime
import json
import os
import shutil

from PyQt6.QtCore import Qt
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
                "Data utworzenia",
                "Dokładność",
                "Rozmiar (MB)",
                "Status",
                "Czas treningu",
                "Epoki",
                "Batch size",
                "Learning rate",
                "Optimizer",
            ]
        )
        self.models_table.horizontalHeader().setStretchLastSection(True)
        self.models_table.setColumnWidth(0, 50)
        self.models_table.setColumnWidth(1, 260)
        self.models_table.setColumnWidth(2, 150)
        self.models_table.setColumnWidth(3, 80)
        self.models_table.setColumnWidth(4, 80)
        self.models_table.setColumnWidth(5, 80)
        self.models_table.setColumnWidth(6, 100)
        self.models_table.setColumnWidth(7, 60)
        self.models_table.setColumnWidth(8, 80)
        self.models_table.setColumnWidth(9, 80)
        self.models_table.setColumnWidth(10, 100)
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

        # Przyciski zarządzania modelami
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(4)

        self.load_btn = QPushButton("Załaduj model")
        self.load_btn.setFixedHeight(24)
        buttons_layout.addWidget(self.load_btn)

        self.export_btn = QPushButton("Eksportuj model")
        self.export_btn.setFixedHeight(24)
        buttons_layout.addWidget(self.export_btn)

        self.export_config_btn = QPushButton("Eksportuj konfigurację")
        self.export_config_btn.setFixedHeight(24)
        buttons_layout.addWidget(self.export_config_btn)

        self.import_config_btn = QPushButton("Importuj konfigurację")
        self.import_config_btn.setFixedHeight(24)
        buttons_layout.addWidget(self.import_config_btn)

        self.delete_btn = QPushButton("Usuń model")
        self.delete_btn.setFixedHeight(24)
        buttons_layout.addWidget(self.delete_btn)

        self.rename_btn = QPushButton("Zmień nazwę")
        self.rename_btn.setFixedHeight(24)
        buttons_layout.addWidget(self.rename_btn)

        self.show_mapping_btn = QPushButton("Pokaż mapowanie")
        self.show_mapping_btn.setFixedHeight(24)
        buttons_layout.addWidget(self.show_mapping_btn)

        self.stats_btn = QPushButton("Statystyki")
        self.stats_btn.setFixedHeight(24)
        buttons_layout.addWidget(self.stats_btn)

        self.compare_btn = QPushButton("Porównaj modele")
        self.compare_btn.setFixedHeight(24)
        buttons_layout.addWidget(self.compare_btn)

        self.refresh_btn = QPushButton("Odśwież")
        self.refresh_btn.setFixedHeight(24)
        buttons_layout.addWidget(self.refresh_btn)

        models_layout.addLayout(buttons_layout)
        layout.addWidget(models_panel, 1)

    def connect_signals(self):
        """Podłącza sygnały do slotów."""
        self.load_btn.clicked.connect(self._load_selected_model)
        self.export_btn.clicked.connect(self._export_selected_model)
        self.export_config_btn.clicked.connect(self._export_model_config)
        self.import_config_btn.clicked.connect(self._import_model_config)
        self.delete_btn.clicked.connect(self._delete_selected_model)
        self.rename_btn.clicked.connect(self._rename_selected_model)
        self.show_mapping_btn.clicked.connect(self._show_class_mapping)
        self.stats_btn.clicked.connect(self._show_model_stats)
        self.compare_btn.clicked.connect(self._compare_models)
        self.refresh_btn.clicked.connect(self.refresh)

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
            for model in models:
                row = self.models_table.rowCount()
                self.models_table.insertRow(row)

                # Checkbox wyboru
                checkbox = QTableWidgetItem()
                checkbox.setCheckState(Qt.CheckState.Unchecked)
                self.models_table.setItem(row, 0, checkbox)

                # Nazwa modelu
                self.models_table.setItem(row, 1, QTableWidgetItem(model))

                # Data utworzenia
                model_path = os.path.join(models_dir, model)
                date = datetime.datetime.fromtimestamp(
                    os.path.getmtime(model_path)
                ).strftime("%Y-%m-%d %H:%M:%S")
                self.models_table.setItem(row, 2, QTableWidgetItem(date))

                # Dokładność - pobierz z konfiguracji jeśli istnieje
                config_path = os.path.splitext(model_path)[0] + "_config.json"
                accuracy = "Nieznana"
                training_time = "Nieznany"
                epochs = "Nieznana"
                batch_size = "Nieznany"
                learning_rate = "Nieznany"
                optimizer = "Nieznany"

                if os.path.exists(config_path):
                    try:
                        with open(config_path, "r") as f:
                            config = json.load(f)
                            metadata = config.get("metadata", {})
                            if metadata:
                                accuracy = f"{metadata.get('accuracy', 0):.2%}"
                                training_time = (
                                    f"{metadata.get('training_time', 0):.1f}s"
                                )

                                training_params = metadata.get(
                                    "training_params", {}
                                ).get("config", {})
                                if training_params:
                                    epochs = str(
                                        training_params.get("epochs", "Nieznana")
                                    )
                                    batch_size = str(
                                        training_params.get("batch_size", "Nieznany")
                                    )
                                    learning_rate = str(
                                        training_params.get("learning_rate", "Nieznany")
                                    )
                                    optimizer = training_params.get(
                                        "optimizer", "Nieznany"
                                    )
                    except Exception:
                        pass
                self.models_table.setItem(row, 3, QTableWidgetItem(accuracy))
                self.models_table.setItem(row, 6, QTableWidgetItem(training_time))
                self.models_table.setItem(row, 7, QTableWidgetItem(epochs))
                self.models_table.setItem(row, 8, QTableWidgetItem(batch_size))
                self.models_table.setItem(row, 9, QTableWidgetItem(learning_rate))
                self.models_table.setItem(row, 10, QTableWidgetItem(optimizer))

                # Rozmiar pliku
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                self.models_table.setItem(row, 4, QTableWidgetItem(f"{size_mb:.2f} MB"))

                # Status
                status = ""
                if hasattr(self.parent, "model_loaded") and hasattr(
                    self.parent, "model_path"
                ):
                    if (
                        self.parent.model_loaded
                        and self.parent.model_path
                        and os.path.basename(self.parent.model_path) == model
                    ):
                        status = "Aktywny"
                self.models_table.setItem(row, 5, QTableWidgetItem(status))

            # Dostosuj szerokość kolumn
            self.models_table.resizeColumnsToContents()

        except Exception as e:
            self.parent.logger.error(f"Błąd podczas odświeżania listy modeli: {str(e)}")

    def _load_selected_model(self):
        """Ładuje model wybrany z listy w tabeli modeli."""
        try:
            self.parent.logger.info("Rozpoczynam ładowanie wybranego modelu...")
            # Sprawdź, czy jest wybrany wiersz
            selected_rows = []
            for i in range(self.models_table.rowCount()):
                item = self.models_table.item(i, 0)
                if item and item.checkState() == Qt.CheckState.Checked:
                    selected_rows.append(i)

            # Jeśli nie wybrano żadnego modelu, wybierz aktualnie zaznaczony wiersz
            if not selected_rows:
                current_row = self.models_table.currentRow()
                if current_row >= 0:
                    selected_rows = [current_row]

            # Sprawdź czy wybrano model
            if not selected_rows:
                QMessageBox.warning(
                    self, "Ostrzeżenie", "Wybierz model do załadowania."
                )
                return

            # Używamy tylko pierwszego wybranego modelu
            row = selected_rows[0]
            model_name = self.models_table.item(row, 1).text()

            # Bezpieczne pobieranie wartości z ustawień z wartością domyślną
            models_dir = self.settings.get("models_dir", "data/models")
            model_path = os.path.join(models_dir, model_name)

            # Sprawdź czy plik istnieje
            if not os.path.exists(model_path):
                QMessageBox.warning(
                    self, "Ostrzeżenie", f"Plik modelu {model_name} nie istnieje."
                )
                return

            # Załaduj model w głównym oknie
            self.parent.logger.info(f"Ładowanie modelu z: {model_path}")
            self.parent.classifier = ImageClassifier(weights_path=model_path)
            self.parent.model_loaded = True
            self.parent.model_path = model_path
            self.parent.current_model = model_name

            # Aktualizuj informacje o aktywnym modelu
            self.parent._update_active_model_info()

            # Odśwież zakładkę Przetwarzanie wsadowe
            if hasattr(self.parent, "batch_processor_tab"):
                self.parent.batch_processor_tab.refresh()

            # Zapisz ostatnio używany model
            self.parent.settings["last_model"] = model_name
            self.parent._save_settings()

            # Aktualizuj status w tabeli
            for i in range(self.models_table.rowCount()):
                status_item = self.models_table.item(i, 5)
                status_item.setText("Aktywny" if i == row else "")

            # Wyświetl komunikat
            QMessageBox.information(
                self, "Sukces", f"Model {model_name} został załadowany pomyślnie."
            )

            self.parent.logger.info(f"Model {model_name} został pomyślnie załadowany")

        except Exception as e:
            QMessageBox.critical(
                self, "Błąd", f"Nie udało się załadować modelu: {str(e)}"
            )
            self.parent.logger.error(f"Błąd ładowania modelu: {str(e)}")

    def _export_selected_model(self):
        """Eksportuje model wybrany z tabeli modeli."""
        try:
            self.parent.logger.info("Rozpoczynam eksport wybranego modelu...")
            # Sprawdź, czy jest wybrany wiersz
            selected_rows = []
            for i in range(self.models_table.rowCount()):
                item = self.models_table.item(i, 0)
                if item and item.checkState() == Qt.CheckState.Checked:
                    selected_rows.append(i)

            # Jeśli nie wybrano żadnego modelu, wybierz aktualnie zaznaczony wiersz
            if not selected_rows:
                current_row = self.models_table.currentRow()
                if current_row >= 0:
                    selected_rows = [current_row]
                    self.parent.logger.info(
                        f"Wybrano aktualnie zaznaczony wiersz: {current_row}"
                    )

            # Sprawdź czy wybrano model
            if not selected_rows:
                self.parent.logger.warning("Nie wybrano modelu do eksportu")
                QMessageBox.warning(self, "Ostrzeżenie", "Wybierz model do eksportu.")
                return

            # Używamy tylko pierwszego wybranego modelu
            row = selected_rows[0]
            model_name = self.models_table.item(row, 1).text()
            self.parent.logger.info(f"Wybrano model do eksportu: {model_name}")
            source_path = os.path.join(self.settings["models_dir"], model_name)

            # Sprawdź czy plik istnieje
            if not os.path.exists(source_path):
                self.parent.logger.error(f"Plik modelu nie istnieje: {source_path}")
                QMessageBox.warning(
                    self, "Ostrzeżenie", f"Plik modelu {model_name} nie istnieje."
                )
                return

            # Wybierz miejsce eksportu
            export_path, _ = QFileDialog.getSaveFileName(
                self,
                "Eksportuj model",
                model_name,
                "Pliki modeli (*.pt *.pth);;Wszystkie pliki (*.*)",
            )

            if not export_path:
                self.parent.logger.info("Anulowano eksport modelu")
                return  # Użytkownik anulował

            # Skopiuj plik modelu
            self.parent.logger.info(f"Kopiowanie modelu do: {export_path}")
            shutil.copy2(source_path, export_path)

            # Sprawdź czy istnieje plik konfiguracyjny i skopiuj go także
            config_path = os.path.splitext(source_path)[0] + "_config.json"
            if os.path.exists(config_path):
                export_config_path = os.path.splitext(export_path)[0] + "_config.json"
                self.parent.logger.info(
                    f"Kopiowanie pliku konfiguracyjnego do: {export_config_path}"
                )
                shutil.copy2(config_path, export_config_path)

            # Wyświetl komunikat
            QMessageBox.information(
                self,
                "Sukces",
                f"Model {model_name} został wyeksportowany do {export_path}",
            )

            self.parent.logger.info(
                f"Model {model_name} został pomyślnie wyeksportowany do {export_path}"
            )

        except Exception as e:
            self.parent.logger.error(f"Błąd podczas eksportu modelu: {str(e)}")
            QMessageBox.critical(
                self, "Błąd", f"Nie udało się wyeksportować modelu: {str(e)}"
            )

    def _export_model_config(self):
        """Eksportuje konfigurację wybranego modelu."""
        try:
            self.parent.logger.info("Rozpoczynam eksport konfiguracji modelu...")
            selected_models = []
            for row in range(self.models_table.rowCount()):
                if self.models_table.item(row, 0).checkState() == Qt.CheckState.Checked:
                    model_name = self.models_table.item(row, 1).text()
                    selected_models.append(model_name)

            if len(selected_models) != 1:
                self.parent.logger.warning(
                    f"Wybrano nieprawidłową liczbę modeli: {len(selected_models)}"
                )
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
                    self,
                    "Błąd",
                    "Brak pliku konfiguracyjnego dla wybranego modelu.",
                )
                return

            # Wybierz miejsce zapisu
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
                    self,
                    "Sukces",
                    "Konfiguracja została wyeksportowana pomyślnie.",
                )
                self.parent.logger.info("Konfiguracja została pomyślnie wyeksportowana")
            else:
                self.parent.logger.info("Anulowano eksport konfiguracji")

        except Exception as e:
            self.parent.logger.error(f"Błąd podczas eksportu konfiguracji: {str(e)}")
            QMessageBox.critical(
                self,
                "Błąd",
                f"Nie udało się wyeksportować konfiguracji: {str(e)}",
            )

    def _import_model_config(self):
        """Importuje konfigurację dla wybranego modelu."""
        try:
            self.parent.logger.info("Rozpoczynam import konfiguracji modelu...")
            selected_models = []
            for row in range(self.models_table.rowCount()):
                if self.models_table.item(row, 0).checkState() == Qt.CheckState.Checked:
                    model_name = self.models_table.item(row, 1).text()
                    selected_models.append(model_name)

            if len(selected_models) != 1:
                self.parent.logger.warning(
                    f"Wybrano nieprawidłową liczbę modeli: {len(selected_models)}"
                )
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

            # Wybierz plik do importu
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Wybierz plik konfiguracyjny",
                "",
                "Pliki JSON (*.json)",
            )

            if file_path:
                self.parent.logger.info(f"Importowanie konfiguracji z: {file_path}")
                try:
                    # Sprawdź czy plik jest poprawnym JSON
                    with open(file_path, "r") as f:
                        config = json.load(f)
                        if not isinstance(config, dict):
                            raise ValueError("Nieprawidłowy format konfiguracji")

                    # Skopiuj plik
                    shutil.copy2(file_path, target_config_path)
                    QMessageBox.information(
                        self,
                        "Sukces",
                        "Konfiguracja została zaimportowana pomyślnie.",
                    )
                    self.parent.logger.info(
                        "Konfiguracja została pomyślnie zaimportowana"
                    )
                    self.refresh()  # Odśwież listę modeli
                except json.JSONDecodeError:
                    self.parent.logger.error(
                        "Wybrany plik nie jest poprawnym plikiem JSON"
                    )
                    QMessageBox.critical(
                        self,
                        "Błąd",
                        "Wybrany plik nie jest poprawnym plikiem JSON.",
                    )
                except ValueError as e:
                    self.parent.logger.error(f"Błąd w formacie konfiguracji: {str(e)}")
                    QMessageBox.critical(
                        self,
                        "Błąd",
                        str(e),
                    )
            else:
                self.parent.logger.info("Anulowano import konfiguracji")

        except Exception as e:
            self.parent.logger.error(f"Błąd podczas importu konfiguracji: {str(e)}")
            QMessageBox.critical(
                self,
                "Błąd",
                f"Nie udało się zaimportować konfiguracji: {str(e)}",
            )

    def _delete_selected_model(self):
        """Usuwa model wybrany z tabeli modeli."""
        try:
            self.parent.logger.info("Rozpoczynam usuwanie wybranego modelu...")
            # Sprawdź, czy jest wybrany wiersz
            selected_rows = []
            for i in range(self.models_table.rowCount()):
                item = self.models_table.item(i, 0)
                if item and item.checkState() == Qt.CheckState.Checked:
                    selected_rows.append(i)

            # Jeśli nie wybrano żadnego modelu, wybierz aktualnie zaznaczony wiersz
            if not selected_rows:
                current_row = self.models_table.currentRow()
                if current_row >= 0:
                    selected_rows = [current_row]
                    self.parent.logger.info(
                        f"Wybrano aktualnie zaznaczony wiersz: {current_row}"
                    )

            # Sprawdź czy wybrano model
            if not selected_rows:
                self.parent.logger.warning("Nie wybrano modelu do usunięcia")
                QMessageBox.warning(self, "Ostrzeżenie", "Wybierz model do usunięcia.")
                return

            # Potwierdzenie przed usunięciem
            if len(selected_rows) == 1:
                model_name = self.models_table.item(selected_rows[0], 1).text()
                confirm_msg = f"Czy na pewno chcesz usunąć model {model_name}?"
            else:
                confirm_msg = f"Czy na pewno chcesz usunąć {len(selected_rows)} modele?"

            reply = QMessageBox.question(
                self,
                "Potwierdzenie",
                confirm_msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if reply != QMessageBox.StandardButton.Yes:
                self.parent.logger.info("Anulowano usuwanie modelu")
                return

            # Usuń wybrane modele
            for row in sorted(selected_rows, reverse=True):
                model_name = self.models_table.item(row, 1).text()
                model_path = os.path.join(self.settings["models_dir"], model_name)
                self.parent.logger.info(f"Usuwanie modelu: {model_name}")

                # Sprawdź czy to nie jest aktualnie załadowany model
                if (
                    hasattr(self.parent, "model_loaded")
                    and hasattr(self.parent, "model_path")
                    and self.parent.model_loaded
                    and self.parent.model_path
                    and os.path.basename(self.parent.model_path) == model_name
                ):
                    self.parent.logger.warning(
                        f"Model {model_name} jest aktualnie załadowany"
                    )
                    QMessageBox.warning(
                        self,
                        "Ostrzeżenie",
                        f"Model {model_name} jest aktualnie załadowany i nie może być usunięty.",
                    )
                    continue

                # Usuń plik modelu
                if os.path.exists(model_path):
                    os.remove(model_path)
                    self.parent.logger.info(f"Usunięto plik modelu: {model_path}")

                    # Usuń również plik konfiguracyjny, jeśli istnieje
                    config_path = os.path.splitext(model_path)[0] + "_config.json"
                    if os.path.exists(config_path):
                        os.remove(config_path)
                        self.parent.logger.info(
                            f"Usunięto plik konfiguracyjny: {config_path}"
                        )

                else:
                    self.parent.logger.warning(
                        f"Plik modelu nie istnieje: {model_path}"
                    )

                # Usuń wiersz z tabeli
                self.models_table.removeRow(row)

            # Wyświetl komunikat o sukcesie
            QMessageBox.information(self, "Sukces", "Wybrane modele zostały usunięte.")
            self.parent.logger.info("Wybrane modele zostały pomyślnie usunięte")

        except Exception as e:
            self.parent.logger.error(f"Błąd podczas usuwania modelu: {str(e)}")
            QMessageBox.critical(self, "Błąd", f"Nie udało się usunąć modelu: {str(e)}")

    def _rename_selected_model(self):
        """Zmienia nazwę wybranego modelu."""
        try:
            self.parent.logger.info("Rozpoczynam zmianę nazwy modelu...")
            # Sprawdź, czy jest wybrany wiersz
            selected_rows = []
            for i in range(self.models_table.rowCount()):
                item = self.models_table.item(i, 0)
                if item and item.checkState() == Qt.CheckState.Checked:
                    selected_rows.append(i)

            # Jeśli nie wybrano żadnego modelu, wybierz aktualnie zaznaczony wiersz
            if not selected_rows:
                current_row = self.models_table.currentRow()
                if current_row >= 0:
                    selected_rows = [current_row]
                    self.parent.logger.info(
                        f"Wybrano aktualnie zaznaczony wiersz: {current_row}"
                    )

            # Sprawdź czy wybrano dokładnie jeden model
            if not selected_rows:
                self.parent.logger.warning("Nie wybrano modelu do zmiany nazwy")
                QMessageBox.warning(
                    self, "Ostrzeżenie", "Wybierz model do zmiany nazwy."
                )
                return
            elif len(selected_rows) > 1:
                self.parent.logger.warning(
                    "Wybrano więcej niż jeden model do zmiany nazwy"
                )
                QMessageBox.warning(
                    self, "Ostrzeżenie", "Wybierz tylko jeden model do zmiany nazwy."
                )
                return

            # Pobierz informacje o wybranym modelu
            row = selected_rows[0]
            old_name = self.models_table.item(row, 1).text()
            old_path = os.path.join(self.settings["models_dir"], old_name)
            self.parent.logger.info(f"Zmiana nazwy modelu: {old_name}")

            # Pobierz nową nazwę
            new_name, ok = QInputDialog.getText(
                self, "Zmiana nazwy modelu", "Nowa nazwa modelu:", text=old_name
            )

            if not ok or not new_name or new_name == old_name:
                self.parent.logger.info("Anulowano zmianę nazwy modelu")
                return

            # Sprawdź czy nowa nazwa kończy się odpowiednim rozszerzeniem
            if not new_name.endswith((".pt", ".pth")):
                new_name += os.path.splitext(old_name)[
                    1
                ]  # Dodaj oryginalne rozszerzenie
                self.parent.logger.info(
                    f"Dodano rozszerzenie do nowej nazwy: {new_name}"
                )

            # Sprawdź czy plik o nowej nazwie już istnieje
            new_path = os.path.join(self.settings["models_dir"], new_name)
            if os.path.exists(new_path):
                self.parent.logger.warning(
                    f"Plik o nowej nazwie już istnieje: {new_path}"
                )
                QMessageBox.warning(
                    self, "Ostrzeżenie", f"Plik {new_name} już istnieje."
                )
                return

            # Zmień nazwę pliku modelu
            os.rename(old_path, new_path)
            self.parent.logger.info(
                f"Zmieniono nazwę pliku modelu z {old_path} na {new_path}"
            )

            # Zmień nazwę pliku konfiguracyjnego, jeśli istnieje
            old_config_path = os.path.splitext(old_path)[0] + "_config.json"
            if os.path.exists(old_config_path):
                new_config_path = os.path.splitext(new_path)[0] + "_config.json"
                os.rename(old_config_path, new_config_path)
                self.parent.logger.info(
                    f"Zmieniono nazwę pliku konfiguracyjnego z {old_config_path} na {new_config_path}"
                )

            # Zaktualizuj nazwę w tabeli
            self.models_table.item(row, 1).setText(new_name)

            # Jeśli to był aktualnie załadowany model, zaktualizuj ścieżkę
            if (
                hasattr(self.parent, "model_loaded")
                and hasattr(self.parent, "model_path")
                and self.parent.model_loaded
                and self.parent.model_path
                and os.path.basename(self.parent.model_path) == old_name
            ):
                self.parent.model_path = new_path
                self.parent.settings["last_model"] = new_name
                self.parent._save_settings()
                self.parent.logger.info(
                    f"Zaktualizowano ścieżkę aktywnego modelu na: {new_path}"
                )

            # Wyświetl komunikat o sukcesie
            QMessageBox.information(
                self, "Sukces", f"Nazwa modelu została zmieniona na {new_name}."
            )
            self.parent.logger.info(
                f"Pomyślnie zmieniono nazwę modelu z {old_name} na {new_name}"
            )

        except Exception as e:
            self.parent.logger.error(f"Błąd podczas zmiany nazwy modelu: {str(e)}")
            QMessageBox.critical(
                self, "Błąd", f"Nie udało się zmienić nazwy modelu: {str(e)}"
            )

    def _show_class_mapping(self):
        """Wyświetla mapowanie indeksów klas na nazwy klas dla aktualnego modelu."""
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

            # Pobierz mapowanie klas z klasyfikatora
            self.parent.logger.info("Pobieranie mapowania klas z klasyfikatora")
            class_mapping = self.parent.classifier.get_class_mapping()

            if not class_mapping:
                self.parent.logger.warning("Brak dostępnego mapowania klas")
                QMessageBox.information(
                    self,
                    "Brak mapowania klas",
                    "Brak dostępnego mapowania klas dla aktualnego modelu.",
                )
                return

            # Utwórz okno dialogowe do wyświetlenia mapowania
            dialog = QDialog(self)
            dialog.setWindowTitle("Mapowanie klas")
            dialog.setMinimumSize(400, 300)

            layout = QVBoxLayout(dialog)

            label = QLabel("Mapowanie indeksów klas na nazwy:")
            layout.addWidget(label)

            # Tabela z mapowaniem
            table = QTableWidget()
            table.setColumnCount(2)
            table.setHorizontalHeaderLabels(["Indeks", "Nazwa klasy"])

            # Wypełnij tabelę danymi
            table.setRowCount(len(class_mapping))
            for i, (class_idx, class_name) in enumerate(sorted(class_mapping.items())):
                if isinstance(class_idx, str) and class_idx.isdigit():
                    class_idx = int(class_idx)

                idx_item = QTableWidgetItem(str(class_idx))
                name_item = QTableWidgetItem(class_name)

                table.setItem(i, 0, idx_item)
                table.setItem(i, 1, name_item)

            # Dopasuj szerokość kolumn
            table.horizontalHeader().setSectionResizeMode(
                0, QHeaderView.ResizeMode.ResizeToContents
            )
            table.horizontalHeader().setSectionResizeMode(
                1, QHeaderView.ResizeMode.Stretch
            )

            layout.addWidget(table)

            # Przycisk zamknięcia
            button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
            button_box.rejected.connect(dialog.reject)
            layout.addWidget(button_box)

            self.parent.logger.info("Wyświetlanie okna z mapowaniem klas")
            dialog.exec()

        except Exception as e:
            self.parent.logger.error(
                f"Błąd podczas wyświetlania mapowania klas: {str(e)}"
            )
            QMessageBox.critical(
                self,
                "Błąd",
                f"Wystąpił błąd podczas pobierania mapowania klas: {str(e)}",
            )

    def _show_model_stats(self):
        """Wyświetla statystyki modelu."""
        try:
            self.parent.logger.info("Rozpoczynam wyświetlanie statystyk modelu...")
            # Sprawdź, czy jest wybrany wiersz
            selected_rows = []
            for i in range(self.models_table.rowCount()):
                item = self.models_table.item(i, 0)
                if item and item.checkState() == Qt.CheckState.Checked:
                    selected_rows.append(i)

            # Jeśli nie wybrano żadnego modelu, wybierz aktualnie zaznaczony wiersz
            if not selected_rows:
                current_row = self.models_table.currentRow()
                if current_row >= 0:
                    selected_rows = [current_row]
                    self.parent.logger.info(
                        f"Wybrano aktualnie zaznaczony wiersz: {current_row}"
                    )

            # Sprawdź czy wybrano model
            if not selected_rows:
                self.parent.logger.warning(
                    "Nie wybrano modelu do wyświetlenia statystyk"
                )
                QMessageBox.warning(
                    self, "Ostrzeżenie", "Wybierz model, aby wyświetlić statystyki."
                )
                return

            # Używamy tylko pierwszego wybranego modelu
            row = selected_rows[0]
            model_name = self.models_table.item(row, 1).text()
            self.parent.logger.info(
                f"Wybrano model do wyświetlenia statystyk: {model_name}"
            )
            model_path = os.path.join(self.settings["models_dir"], model_name)
            config_path = os.path.splitext(model_path)[0] + "_config.json"

            # Statystyki do wyświetlenia
            stats = {}

            # Dodaj podstawowe informacje o modelu
            stats["Nazwa modelu"] = model_name
            stats["Ścieżka"] = model_path

            # Pobierz rozmiar modelu
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            stats["Rozmiar"] = f"{size_mb:.2f} MB"

            # Odczytaj datę utworzenia
            date = datetime.datetime.fromtimestamp(
                os.path.getmtime(model_path)
            ).strftime("%Y-%m-%d %H:%M:%S")
            stats["Data utworzenia"] = date

            # Spróbuj załadować model aby uzyskać więcej informacji
            if (
                hasattr(self.parent, "classifier")
                and self.parent.classifier is not None
            ):
                try:
                    self.parent.logger.info(
                        "Ładowanie modelu w celu pobrania dodatkowych informacji"
                    )
                    classifier = ImageClassifier(weights_path=model_path)
                    model_info = classifier.get_model_info()
                    for k, v in model_info.items():
                        stats[f"model_info.{k}"] = v
                except Exception as e:
                    self.parent.logger.error(f"Błąd podczas analizy modelu: {str(e)}")

            # Spróbuj odczytać wszystkie dane z pliku konfiguracyjnego
            def flatten_dict(d, parent_key="", sep="."):
                items = []
                for k, v in d.items():
                    new_key = f"{parent_key}{sep}{k}" if parent_key else k
                    if isinstance(v, dict):
                        items.extend(flatten_dict(v, new_key, sep=sep).items())
                    else:
                        items.append((new_key, v))
                return dict(items)

            if os.path.exists(config_path):
                try:
                    self.parent.logger.info(
                        "Odczyt wszystkich danych z pliku konfiguracyjnego"
                    )
                    with open(config_path, "r") as f:
                        config = json.load(f)
                    flat_config = flatten_dict(config)
                    for k, v in flat_config.items():
                        stats[f"config.{k}"] = v
                except Exception as e:
                    self.parent.logger.error(
                        f"Błąd podczas odczytu konfiguracji: {str(e)}"
                    )

            # Utwórz okno dialogowe
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Statystyki modelu: {model_name}")
            dialog.setMinimumSize(500, 400)

            layout = QVBoxLayout(dialog)

            # Utwórz tabelę statystyk
            stats_table = QTableWidget()
            stats_table.setColumnCount(2)
            stats_table.setHorizontalHeaderLabels(["Parametr", "Wartość"])
            stats_table.horizontalHeader().setStretchLastSection(True)
            stats_table.setRowCount(len(stats))

            for i, (key, value) in enumerate(stats.items()):
                stats_table.setItem(i, 0, QTableWidgetItem(str(key)))
                stats_table.setItem(i, 1, QTableWidgetItem(str(value)))

            stats_table.resizeColumnToContents(0)
            layout.addWidget(stats_table)

            # Przyciski zamknięcia
            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
            buttons.rejected.connect(dialog.reject)
            layout.addWidget(buttons)

            self.parent.logger.info("Wyświetlanie okna ze statystykami modelu")
            dialog.exec()

        except Exception as e:
            self.parent.logger.error(
                f"Błąd podczas wyświetlania statystyk modelu: {str(e)}"
            )
            QMessageBox.critical(
                self, "Błąd", f"Nie udało się wyświetlić statystyk modelu: {str(e)}"
            )

    def _compare_models(self):
        """Porównuje wybrane modele."""
        try:
            self.parent.logger.info("Rozpoczynam porównywanie modeli...")
            # Sprawdź, czy są wybrane modele
            selected_models = []
            for row in range(self.models_table.rowCount()):
                if self.models_table.item(row, 0).checkState() == Qt.CheckState.Checked:
                    model_name = self.models_table.item(row, 1).text()
                    selected_models.append((row, model_name))

            # Potrzebujemy co najmniej dwóch modeli do porównania
            if len(selected_models) < 2:
                self.parent.logger.warning("Wybrano za mało modeli do porównania")
                QMessageBox.warning(
                    self,
                    "Ostrzeżenie",
                    "Wybierz co najmniej dwa modele do porównania.",
                )
                return

            self.parent.logger.info(
                f"Wybrano {len(selected_models)} modeli do porównania"
            )
            # Utwórz okno dialogowe
            dialog = QDialog(self)
            dialog.setWindowTitle("Porównanie modeli")
            dialog.setMinimumWidth(800)
            dialog.setMinimumHeight(600)

            layout = QVBoxLayout(dialog)

            # Etykieta z informacją o porównaniu
            info_label = QLabel(f"Porównanie {len(selected_models)} modeli:")
            layout.addWidget(info_label)

            # Tabela porównawcza
            comp_table = QTableWidget()
            comp_table.setColumnCount(
                len(selected_models) + 1
            )  # +1 dla nazw parametrów

            # Nagłówki kolumn (nazwy modeli)
            headers = ["Parametr"]
            for _, model_name in selected_models:
                headers.append(model_name)
            comp_table.setHorizontalHeaderLabels(headers)

            # Parametry do porównania - podstawowe na górze
            basic_parameters = [
                ("Typ modelu", "model_type"),
                ("Liczba klas", "num_classes"),
                ("Rozmiar (MB)", "size_mb"),
                ("Data utworzenia", "created_at"),
                ("Dokładność", "accuracy"),
                ("Liczba parametrów", "parameters"),
                ("Czas treningu", "training_time"),
                ("Liczba epok", "epochs"),
                ("Batch size", "batch_size"),
                ("Learning rate", "learning_rate"),
                ("Optimizer", "optimizer"),
            ]

            # Zbierz wszystkie parametry z JSON-ów
            def flatten_dict(d, parent_key="", sep="."):
                items = []
                for k, v in d.items():
                    new_key = f"{parent_key}{sep}{k}" if parent_key else k
                    if isinstance(v, dict):
                        items.extend(flatten_dict(v, new_key, sep=sep).items())
                    else:
                        items.append((new_key, v))
                return dict(items)

            all_keys = set()
            model_data_list = []
            for _, model_name in selected_models:
                model_path = os.path.join(self.settings["models_dir"], model_name)
                config_path = os.path.splitext(model_path)[0] + "_config.json"
                data = {}
                # Podstawowe dane
                data["size_mb"] = os.path.getsize(model_path) / (1024 * 1024)
                data["created_at"] = datetime.datetime.fromtimestamp(
                    os.path.getmtime(model_path)
                ).strftime("%Y-%m-%d %H:%M:%S")
                # Domyślne wartości
                data["model_type"] = "Nieznany"
                data["num_classes"] = "Nieznana"
                data["accuracy"] = "Nieznana"
                data["parameters"] = "Nieznana"
                data["training_time"] = "Nieznany"
                data["epochs"] = "Nieznana"
                data["batch_size"] = "Nieznany"
                data["learning_rate"] = "Nieznany"
                data["optimizer"] = "Nieznany"
                # Dane z JSON
                flat_config = {}
                if os.path.exists(config_path):
                    try:
                        with open(config_path, "r") as f:
                            config = json.load(f)
                        flat_config = flatten_dict(config)
                        data.update(flat_config)
                    except Exception as e:
                        self.parent.logger.error(
                            f"Błąd podczas odczytu konfiguracji modelu {model_name}: {str(e)}"
                        )
                model_data_list.append(data)
                all_keys.update(flat_config.keys())
            # Usuń podstawowe parametry z all_keys, by nie dublować
            for _, key in basic_parameters:
                all_keys.discard(key)
            # Posortuj dodatkowe parametry
            extra_keys = sorted(all_keys)
            # Przygotuj listę wszystkich parametrów do wyświetlenia
            all_params = basic_parameters + [(k, k) for k in extra_keys]
            comp_table.setRowCount(len(all_params))
            for i, (label, key) in enumerate(all_params):
                comp_table.setItem(i, 0, QTableWidgetItem(label))
                for col, data in enumerate(model_data_list, start=1):
                    val = data.get(key, "Nieznany")
                    # Formatowanie dla niektórych pól
                    if key == "size_mb" and isinstance(val, (int, float)):
                        val = f"{val:.2f} MB"
                    if key == "accuracy" and isinstance(val, (int, float)):
                        val = f"{val:.2%}"
                    if key == "training_time" and isinstance(val, (int, float)):
                        val = f"{val:.2f}s"
                    comp_table.setItem(i, col, QTableWidgetItem(str(val)))

            # Dostosuj szerokość kolumn
            comp_table.horizontalHeader().setSectionResizeMode(
                0, QHeaderView.ResizeMode.ResizeToContents
            )
            for i in range(1, comp_table.columnCount()):
                comp_table.horizontalHeader().setSectionResizeMode(
                    i, QHeaderView.ResizeMode.Stretch
                )

            layout.addWidget(comp_table)

            # Dodaj wiersz przycisków
            button_layout = QHBoxLayout()

            # Przycisk eksportu do CSV
            export_csv_btn = QPushButton("Eksportuj do CSV")
            export_csv_btn.clicked.connect(
                lambda: self._export_comparison_to_csv(comp_table)
            )
            button_layout.addWidget(export_csv_btn)

            # Przycisk zamknięcia
            close_btn = QPushButton("Zamknij")
            close_btn.clicked.connect(dialog.accept)
            button_layout.addWidget(close_btn)

            button_layout.addStretch(1)
            layout.addLayout(button_layout)

            self.parent.logger.info("Wyświetlanie okna porównania modeli")
            dialog.exec()

        except Exception as e:
            self.parent.logger.error(f"Błąd podczas porównywania modeli: {str(e)}")
            QMessageBox.critical(
                self,
                "Błąd",
                f"Wystąpił błąd podczas porównywania modeli: {str(e)}",
            )

    def _export_comparison_to_csv(self, table):
        """Eksportuje porównanie modeli do pliku CSV."""
        try:
            self.parent.logger.info("Rozpoczynam eksport porównania do CSV...")
            # Wybierz ścieżkę do zapisu pliku
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Zapisz porównanie jako CSV",
                "",
                "Pliki CSV (*.csv)",
            )

            if not file_path:
                self.parent.logger.info("Anulowano eksport do CSV")
                return

            # Dodaj rozszerzenie .csv jeśli nie zostało podane
            if not file_path.endswith(".csv"):
                file_path += ".csv"

            # Otwórz plik do zapisu
            self.parent.logger.info(f"Zapisywanie porównania do pliku: {file_path}")
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                # Pobierz nagłówki
                headers = []
                for col in range(table.columnCount()):
                    headers.append(table.horizontalHeaderItem(col).text())

                # Zapisz nagłówki
                f.write(",".join([f'"{h}"' for h in headers]) + "\n")

                # Zapisz zawartość tabeli
                for row in range(table.rowCount()):
                    row_data = []
                    for col in range(table.columnCount()):
                        item = table.item(row, col)
                        cell_text = item.text() if item else ""
                        row_data.append(f'"{cell_text}"')
                    f.write(",".join(row_data) + "\n")

            QMessageBox.information(
                self,
                "Sukces",
                f"Porównanie zostało wyeksportowane do pliku:\n{file_path}",
            )
            self.parent.logger.info("Pomyślnie wyeksportowano porównanie do CSV")

        except Exception as e:
            self.parent.logger.error(
                f"Błąd podczas eksportu porównania do CSV: {str(e)}"
            )
            QMessageBox.critical(
                self,
                "Błąd",
                f"Wystąpił błąd podczas eksportu porównania: {str(e)}",
            )

    def update_hardware_profile(self, profile):
        """Aktualizuje profil sprzętowy używany do optymalizacji."""
        self.hardware_profile = profile
        self.parent.logger.info("Zaktualizowano profil sprzętowy w zakładce modeli")

        # Aktualizacja UI na podstawie nowego profilu
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
