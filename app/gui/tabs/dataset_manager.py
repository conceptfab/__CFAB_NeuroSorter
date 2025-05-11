import datetime
import json
import os
import shutil

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.gui.tab_interface import TabInterface
from app.utils.dataset_utils import split_dataset, validate_dataset


class DatasetManager(QWidget, TabInterface):
    """Klasa zarządzająca zbiorami danych."""

    def __init__(self, parent=None, settings=None):
        super().__init__(parent)
        self.parent = parent
        self.settings = settings
        self.setup_ui()
        self.connect_signals()

    def setup_ui(self):
        """Tworzy i konfiguruje elementy interfejsu zakładki."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Panel importu danych
        self._create_import_panel(layout)

        # Panel zarządzania zbiorami
        self._create_datasets_panel(layout)

        # Panel podziału danych
        self._create_split_panel(layout)

        # Dodaj elastyczną przestrzeń na dole
        layout.addStretch(1)

    def connect_signals(self):
        """Podłącza sygnały do slotów."""
        pass

    def refresh(self):
        """Odświeża zawartość zakładki."""
        self._refresh_datasets()

    def update_settings(self, settings):
        """Aktualizuje ustawienia zakładki."""
        self.settings = settings

    def save_state(self):
        """Zapisuje stan zakładki."""
        return {}

    def restore_state(self, state):
        """Przywraca zapisany stan zakładki."""
        pass

    def _create_import_panel(self, parent_layout):
        """Tworzy panel importu danych."""
        import_panel = QWidget()
        import_layout = QVBoxLayout(import_panel)
        import_layout.setContentsMargins(0, 0, 0, 0)

        # Nagłówek sekcji
        import_header = QLabel("IMPORT DANYCH")
        import_header.setStyleSheet(
            "font-weight: bold; color: #CCCCCC; font-size: 11px; padding-bottom: 4px;"
        )
        import_layout.addWidget(import_header)

        # Grupa importu
        import_group = QGroupBox("Import nowego zbioru danych")
        import_group_layout = QFormLayout(import_group)

        # Nazwa zbioru
        self.dataset_name_edit = QLineEdit()
        import_group_layout.addRow("Nazwa zbioru:", self.dataset_name_edit)

        # Katalog źródłowy
        source_layout = QHBoxLayout()
        self.source_dir_edit = QLineEdit()
        self.source_dir_edit.setReadOnly(True)
        source_dir_btn = QPushButton("Przeglądaj")
        source_dir_btn.clicked.connect(self._select_source_directory)
        source_layout.addWidget(self.source_dir_edit)
        source_layout.addWidget(source_dir_btn)
        import_group_layout.addRow("Katalog źródłowy:", source_layout)

        # Opcje importu
        self.recursive_checkbox = QCheckBox("Importuj podkatalogi")
        self.recursive_checkbox.setChecked(True)
        import_group_layout.addRow("", self.recursive_checkbox)

        self.copy_files_checkbox = QCheckBox("Kopiuj pliki (zamiast przenosić)")
        self.copy_files_checkbox.setChecked(True)
        import_group_layout.addRow("", self.copy_files_checkbox)

        # Przycisk importu
        self.import_btn = QPushButton("Importuj dane")
        self.import_btn.clicked.connect(self._import_dataset)
        import_group_layout.addRow("", self.import_btn)

        import_layout.addWidget(import_group)
        parent_layout.addWidget(import_panel)

    def _create_datasets_panel(self, parent_layout):
        """Tworzy panel zarządzania zbiorami danych."""
        datasets_panel = QWidget()
        datasets_layout = QVBoxLayout(datasets_panel)
        datasets_layout.setContentsMargins(0, 0, 0, 0)

        # Nagłówek sekcji
        datasets_header = QLabel("ZBIORY DANYCH")
        datasets_header.setStyleSheet(
            "font-weight: bold; color: #CCCCCC; font-size: 11px; padding-bottom: 4px;"
        )
        datasets_layout.addWidget(datasets_header)

        # Tabela zbiorów danych
        self.datasets_table = QTableWidget()
        self.datasets_table.setColumnCount(6)
        self.datasets_table.setHorizontalHeaderLabels(
            ["Nazwa", "Liczba klas", "Liczba obrazów", "Rozmiar", "Status", "Akcje"]
        )
        self.datasets_table.horizontalHeader().setStretchLastSection(True)
        self.datasets_table.verticalHeader().setVisible(False)
        self.datasets_table.setAlternatingRowColors(True)
        datasets_layout.addWidget(self.datasets_table)

        # Przyciski zarządzania
        buttons_layout = QHBoxLayout()

        self.refresh_btn = QPushButton("Odśwież")
        self.refresh_btn.clicked.connect(self.refresh)
        buttons_layout.addWidget(self.refresh_btn)

        self.validate_btn = QPushButton("Waliduj")
        self.validate_btn.clicked.connect(self._validate_selected_dataset)
        buttons_layout.addWidget(self.validate_btn)

        self.export_btn = QPushButton("Eksportuj")
        self.export_btn.clicked.connect(self._export_selected_dataset)
        buttons_layout.addWidget(self.export_btn)

        self.delete_btn = QPushButton("Usuń")
        self.delete_btn.clicked.connect(self._delete_selected_dataset)
        buttons_layout.addWidget(self.delete_btn)

        buttons_layout.addStretch(1)
        datasets_layout.addLayout(buttons_layout)

        parent_layout.addWidget(datasets_panel)

    def _create_split_panel(self, parent_layout):
        """Tworzy panel podziału danych."""
        split_panel = QWidget()
        split_layout = QVBoxLayout(split_panel)
        split_layout.setContentsMargins(0, 0, 0, 0)

        # Nagłówek sekcji
        split_header = QLabel("PODZIAŁ DANYCH")
        split_header.setStyleSheet(
            "font-weight: bold; color: #CCCCCC; font-size: 11px; padding-bottom: 4px;"
        )
        split_layout.addWidget(split_header)

        # Grupa podziału
        split_group = QGroupBox("Podział zbioru danych")
        split_group_layout = QFormLayout(split_group)

        # Proporcje podziału
        split_ratios_layout = QHBoxLayout()

        # Zbiór treningowy
        train_layout = QHBoxLayout()
        train_label = QLabel("Treningowy:")
        self.train_ratio_spin = QSpinBox()
        self.train_ratio_spin.setRange(0, 100)
        self.train_ratio_spin.setValue(70)
        self.train_ratio_spin.setSuffix("%")
        train_layout.addWidget(train_label)
        train_layout.addWidget(self.train_ratio_spin)
        split_ratios_layout.addLayout(train_layout)

        # Zbiór walidacyjny
        val_layout = QHBoxLayout()
        val_label = QLabel("Walidacyjny:")
        self.val_ratio_spin = QSpinBox()
        self.val_ratio_spin.setRange(0, 100)
        self.val_ratio_spin.setValue(15)
        self.val_ratio_spin.setSuffix("%")
        val_layout.addWidget(val_label)
        val_layout.addWidget(self.val_ratio_spin)
        split_ratios_layout.addLayout(val_layout)

        # Zbiór testowy
        test_layout = QHBoxLayout()
        test_label = QLabel("Testowy:")
        self.test_ratio_spin = QSpinBox()
        self.test_ratio_spin.setRange(0, 100)
        self.test_ratio_spin.setValue(15)
        self.test_ratio_spin.setSuffix("%")
        test_layout.addWidget(test_label)
        test_layout.addWidget(self.test_ratio_spin)
        split_ratios_layout.addLayout(test_layout)

        split_group_layout.addRow("Proporcje podziału:", split_ratios_layout)

        # Opcje podziału
        self.stratified_checkbox = QCheckBox("Zachowaj proporcje klas")
        self.stratified_checkbox.setChecked(True)
        split_group_layout.addRow("", self.stratified_checkbox)

        self.shuffle_checkbox = QCheckBox("Losowa kolejność")
        self.shuffle_checkbox.setChecked(True)
        split_group_layout.addRow("", self.shuffle_checkbox)

        # Przycisk podziału
        self.split_btn = QPushButton("Podziel dane")
        self.split_btn.clicked.connect(self._split_dataset)
        split_group_layout.addRow("", self.split_btn)

        split_layout.addWidget(split_group)
        parent_layout.addWidget(split_panel)

    def _select_source_directory(self):
        """Wyświetla dialog wyboru katalogu źródłowego."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Wybierz katalog ze zbiorem danych",
            "",
            QFileDialog.Option.ShowDirsOnly,
        )
        if directory:
            self.source_dir_edit.setText(directory)

    def _import_dataset(self):
        """Importuje nowy zbiór danych."""
        try:
            # Sprawdź czy podano nazwę zbioru
            dataset_name = self.dataset_name_edit.text().strip()
            if not dataset_name:
                QMessageBox.warning(self, "Ostrzeżenie", "Podaj nazwę zbioru danych.")
                return

            # Sprawdź czy podano katalog źródłowy
            source_dir = self.source_dir_edit.text()
            if not source_dir or not os.path.isdir(source_dir):
                QMessageBox.warning(
                    self, "Ostrzeżenie", "Wybierz prawidłowy katalog źródłowy."
                )
                return

            # Utwórz katalog docelowy
            target_dir = os.path.join(
                self.settings["data_dir"], "datasets", dataset_name
            )
            if os.path.exists(target_dir):
                reply = QMessageBox.question(
                    self,
                    "Potwierdzenie",
                    f"Zbiór danych {dataset_name} już istnieje. Czy chcesz go nadpisać?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return
                shutil.rmtree(target_dir)

            # Utwórz dialog postępu
            progress_dialog = QProgressDialog(
                "Importowanie danych...", "Anuluj", 0, 100, self
            )
            progress_dialog.setWindowTitle("Import danych")
            progress_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
            progress_dialog.setMinimumDuration(0)
            progress_dialog.setAutoClose(False)
            progress_dialog.setAutoReset(False)
            progress_dialog.show()
            QApplication.processEvents()

            # Importuj dane
            os.makedirs(target_dir, exist_ok=True)
            total_files = 0
            imported_files = 0

            # Przygotuj listę plików do importu
            files_to_import = []
            for root, _, files in os.walk(source_dir):
                if not self.recursive_checkbox.isChecked() and root != source_dir:
                    continue
                for file in files:
                    if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        source_path = os.path.join(root, file)
                        rel_path = os.path.relpath(source_path, source_dir)
                        target_path = os.path.join(target_dir, rel_path)
                        files_to_import.append((source_path, target_path))
                        total_files += 1

            # Importuj pliki
            for source_path, target_path in files_to_import:
                try:
                    # Utwórz katalog docelowy
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)

                    # Skopiuj lub przenieś plik
                    if self.copy_files_checkbox.isChecked():
                        shutil.copy2(source_path, target_path)
                    else:
                        shutil.move(source_path, target_path)

                    imported_files += 1
                    progress = int(imported_files / total_files * 100)
                    progress_dialog.setValue(progress)
                    QApplication.processEvents()

                    if progress_dialog.wasCanceled():
                        # Usuń częściowo zaimportowane dane
                        shutil.rmtree(target_dir)
                        QMessageBox.warning(
                            self,
                            "Import przerwany",
                            "Import danych został przerwany przez użytkownika.",
                        )
                        return

                except Exception as e:
                    self.parent._log_message(
                        f"Błąd podczas importu pliku {source_path}: {str(e)}"
                    )

            # Zamknij dialog postępu
            progress_dialog.close()

            # Zapisz metadane zbioru
            metadata = {
                "name": dataset_name,
                "source_dir": source_dir,
                "imported_files": imported_files,
                "import_date": datetime.datetime.now().isoformat(),
                "recursive": self.recursive_checkbox.isChecked(),
                "copied": self.copy_files_checkbox.isChecked(),
            }
            metadata_file = os.path.join(target_dir, "metadata.json")
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)

            # Odśwież listę zbiorów
            self.refresh()

            # Wyświetl komunikat o sukcesie
            QMessageBox.information(
                self,
                "Sukces",
                f"Zaimportowano {imported_files} plików do zbioru {dataset_name}.",
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Błąd", f"Nie udało się zaimportować danych: {str(e)}"
            )

    def _refresh_datasets(self):
        """Odświeża listę zbiorów danych."""
        try:
            # Wyczyść tabelę
            self.datasets_table.setRowCount(0)

            # Katalog ze zbiorami danych
            datasets_dir = os.path.join(self.settings["data_dir"], "datasets")
            if not os.path.exists(datasets_dir):
                return

            # Pobierz listę zbiorów
            for dataset_name in os.listdir(datasets_dir):
                dataset_path = os.path.join(datasets_dir, dataset_name)
                if not os.path.isdir(dataset_path):
                    continue

                try:
                    # Wczytaj metadane
                    metadata_file = os.path.join(dataset_path, "metadata.json")
                    if os.path.exists(metadata_file):
                        with open(metadata_file, "r", encoding="utf-8") as f:
                            metadata = json.load(f)
                    else:
                        metadata = {}

                    # Dodaj wiersz do tabeli
                    row = self.datasets_table.rowCount()
                    self.datasets_table.insertRow(row)

                    # Nazwa zbioru
                    self.datasets_table.setItem(row, 0, QTableWidgetItem(dataset_name))

                    # Liczba klas
                    num_classes = len(
                        [
                            d
                            for d in os.listdir(dataset_path)
                            if os.path.isdir(os.path.join(dataset_path, d))
                            and not d.startswith(".")
                        ]
                    )
                    self.datasets_table.setItem(
                        row, 1, QTableWidgetItem(str(num_classes))
                    )

                    # Liczba obrazów
                    num_images = metadata.get("imported_files", 0)
                    self.datasets_table.setItem(
                        row, 2, QTableWidgetItem(str(num_images))
                    )

                    # Rozmiar zbioru
                    total_size = 0
                    for root, _, files in os.walk(dataset_path):
                        total_size += sum(
                            os.path.getsize(os.path.join(root, file)) for file in files
                        )
                    size_mb = total_size / (1024 * 1024)
                    self.datasets_table.setItem(
                        row, 3, QTableWidgetItem(f"{size_mb:.2f} MB")
                    )

                    # Status
                    status = "Gotowy"
                    if os.path.exists(os.path.join(dataset_path, "train")):
                        status = "Podzielony"
                    self.datasets_table.setItem(row, 4, QTableWidgetItem(status))

                    # Przyciski akcji
                    action_widget = QWidget()
                    action_layout = QHBoxLayout(action_widget)
                    action_layout.setContentsMargins(2, 2, 2, 2)

                    view_btn = QPushButton("Podgląd")
                    view_btn.clicked.connect(
                        lambda checked, name=dataset_name: self._view_dataset(name)
                    )
                    action_layout.addWidget(view_btn)

                    edit_btn = QPushButton("Edytuj")
                    edit_btn.clicked.connect(
                        lambda checked, name=dataset_name: self._edit_dataset(name)
                    )
                    action_layout.addWidget(edit_btn)

                    action_layout.addStretch()
                    self.datasets_table.setCellWidget(row, 5, action_widget)

                except Exception as e:
                    self.parent._log_message(
                        f"Błąd podczas wczytywania zbioru {dataset_name}: {str(e)}"
                    )

            # Dostosuj szerokość kolumn
            self.datasets_table.resizeColumnsToContents()
            self.datasets_table.horizontalHeader().setStretchLastSection(True)

        except Exception as e:
            self.parent._log_message(
                f"Błąd podczas odświeżania listy zbiorów: {str(e)}"
            )

    def _validate_selected_dataset(self):
        """Waliduje wybrany zbiór danych."""
        try:
            # Sprawdź czy jest wybrany zbiór
            current_row = self.datasets_table.currentRow()
            if current_row < 0:
                QMessageBox.warning(
                    self, "Ostrzeżenie", "Wybierz zbiór danych do walidacji."
                )
                return

            # Pobierz nazwę zbioru
            dataset_name = self.datasets_table.item(current_row, 0).text()
            dataset_path = os.path.join(
                self.settings["data_dir"], "datasets", dataset_name
            )

            # Utwórz dialog postępu
            progress_dialog = QProgressDialog(
                "Walidacja zbioru danych...", "Anuluj", 0, 100, self
            )
            progress_dialog.setWindowTitle("Walidacja danych")
            progress_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
            progress_dialog.setMinimumDuration(0)
            progress_dialog.show()
            QApplication.processEvents()

            # Waliduj zbiór
            validation_result = validate_dataset(dataset_path, progress_dialog)

            # Wyświetl wyniki walidacji
            if validation_result["valid"]:
                message = "Zbiór danych jest poprawny.\n\n"
                message += f"Liczba klas: {validation_result['num_classes']}\n"
                message += f"Liczba obrazów: {validation_result['num_images']}\n"
                message += (
                    f"Minimalna rozdzielczość: {validation_result['min_resolution']}\n"
                )
                message += (
                    f"Maksymalna rozdzielczość: {validation_result['max_resolution']}\n"
                )
                QMessageBox.information(self, "Wyniki walidacji", message)
            else:
                message = "Znaleziono problemy w zbiorze danych:\n\n"
                for error in validation_result["errors"]:
                    message += f"• {error}\n"
                QMessageBox.warning(self, "Wyniki walidacji", message)

        except Exception as e:
            QMessageBox.critical(
                self, "Błąd", f"Nie udało się zwalidować zbioru danych: {str(e)}"
            )

    def _export_selected_dataset(self):
        """Eksportuje wybrany zbiór danych."""
        try:
            # Sprawdź czy jest wybrany zbiór
            current_row = self.datasets_table.currentRow()
            if current_row < 0:
                QMessageBox.warning(
                    self, "Ostrzeżenie", "Wybierz zbiór danych do eksportu."
                )
                return

            # Pobierz nazwę zbioru
            dataset_name = self.datasets_table.item(current_row, 0).text()
            dataset_path = os.path.join(
                self.settings["data_dir"], "datasets", dataset_name
            )

            # Wybierz katalog docelowy
            target_dir = QFileDialog.getExistingDirectory(
                self,
                "Wybierz katalog do eksportu",
                "",
                QFileDialog.Option.ShowDirsOnly,
            )

            if not target_dir:
                return

            # Utwórz dialog postępu
            progress_dialog = QProgressDialog(
                "Eksportowanie danych...", "Anuluj", 0, 100, self
            )
            progress_dialog.setWindowTitle("Eksport danych")
            progress_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
            progress_dialog.setMinimumDuration(0)
            progress_dialog.show()
            QApplication.processEvents()

            # Eksportuj dane
            target_path = os.path.join(target_dir, dataset_name)
            shutil.copytree(dataset_path, target_path)

            # Wyświetl komunikat o sukcesie
            QMessageBox.information(
                self,
                "Sukces",
                f"Zbiór danych został wyeksportowany do:\n{target_path}",
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Błąd", f"Nie udało się wyeksportować zbioru danych: {str(e)}"
            )

    def _delete_selected_dataset(self):
        """Usuwa wybrany zbiór danych."""
        try:
            # Sprawdź czy jest wybrany zbiór
            current_row = self.datasets_table.currentRow()
            if current_row < 0:
                QMessageBox.warning(
                    self, "Ostrzeżenie", "Wybierz zbiór danych do usunięcia."
                )
                return

            # Pobierz nazwę zbioru
            dataset_name = self.datasets_table.item(current_row, 0).text()

            # Potwierdzenie przed usunięciem
            reply = QMessageBox.question(
                self,
                "Potwierdzenie",
                f"Czy na pewno chcesz usunąć zbiór danych {dataset_name}?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if reply != QMessageBox.StandardButton.Yes:
                return

            # Usuń zbiór
            dataset_path = os.path.join(
                self.settings["data_dir"], "datasets", dataset_name
            )
            shutil.rmtree(dataset_path)

            # Odśwież listę zbiorów
            self.refresh()

            # Wyświetl komunikat o sukcesie
            QMessageBox.information(
                self, "Sukces", f"Zbiór danych {dataset_name} został usunięty."
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Błąd", f"Nie udało się usunąć zbioru danych: {str(e)}"
            )

    def _split_dataset(self):
        """Dzieli wybrany zbiór danych na podzbiory."""
        try:
            # Sprawdź czy jest wybrany zbiór
            current_row = self.datasets_table.currentRow()
            if current_row < 0:
                QMessageBox.warning(
                    self, "Ostrzeżenie", "Wybierz zbiór danych do podziału."
                )
                return

            # Pobierz nazwę zbioru
            dataset_name = self.datasets_table.item(current_row, 0).text()
            dataset_path = os.path.join(
                self.settings["data_dir"], "datasets", dataset_name
            )

            # Sprawdź czy suma proporcji wynosi 100%
            total_ratio = (
                self.train_ratio_spin.value()
                + self.val_ratio_spin.value()
                + self.test_ratio_spin.value()
            )
            if total_ratio != 100:
                QMessageBox.warning(
                    self,
                    "Ostrzeżenie",
                    "Suma proporcji podziału musi wynosić 100%.",
                )
                return

            # Utwórz dialog postępu
            progress_dialog = QProgressDialog(
                "Dzielenie zbioru danych...", "Anuluj", 0, 100, self
            )
            progress_dialog.setWindowTitle("Podział danych")
            progress_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
            progress_dialog.setMinimumDuration(0)
            progress_dialog.show()
            QApplication.processEvents()

            # Podziel zbiór
            split_result = split_dataset(
                dataset_path,
                train_ratio=self.train_ratio_spin.value() / 100,
                val_ratio=self.val_ratio_spin.value() / 100,
                test_ratio=self.test_ratio_spin.value() / 100,
                stratified=self.stratified_checkbox.isChecked(),
                shuffle=self.shuffle_checkbox.isChecked(),
                progress_callback=lambda p: progress_dialog.setValue(p),
            )

            # Wyświetl wyniki podziału
            message = "Zbiór danych został podzielony:\n\n"
            message += f"Zbiór treningowy: {split_result['train_size']} obrazów\n"
            message += f"Zbiór walidacyjny: {split_result['val_size']} obrazów\n"
            message += f"Zbiór testowy: {split_result['test_size']} obrazów"
            QMessageBox.information(self, "Sukces", message)

            # Odśwież listę zbiorów
            self.refresh()

        except Exception as e:
            QMessageBox.critical(
                self, "Błąd", f"Nie udało się podzielić zbioru danych: {str(e)}"
            )

    def _show_context_menu(self, pos):
        pass
