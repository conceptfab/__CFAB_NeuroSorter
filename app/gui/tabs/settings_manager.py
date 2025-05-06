import json

# from PyQt6.QtCore import Qt # Usunięto nieużywany import Qt
from PyQt6.QtWidgets import (  # QDialogButtonBox, # Usunięto nieużywany import; QFrame, # Usunięto nieużywany import; QLabel, # Usunięto nieużywany import; QScrollArea, # Usunięto nieużywany import
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.gui.tab_interface import TabInterface

# Poniższy import jest problematyczny - wymaga sprawdzenia struktury projektu
from app.utils.settings_utils import validate_settings

# import os # Usunięto nieużywany import


class SettingsManager(QDialog):
    """Klasa zarządzająca ustawieniami aplikacji."""

    def __init__(self, parent=None, settings=None):
        super().__init__(parent)
        self.parent = parent
        self.settings = settings
        self.setup_ui()
        self.connect_signals()

    def setup_ui(self):
        """Tworzy i konfiguruje elementy interfejsu okna."""
        self.setWindowTitle("Ustawienia globalne")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Zakładki ustawień
        self.settings_tabs = QTabWidget()
        self.settings_tabs.setDocumentMode(True)

        # Dodaj zakładki
        self.settings_tabs.addTab(self._create_general_tab(), "Ogólne")
        self.settings_tabs.addTab(self._create_model_tab(), "Model")
        self.settings_tabs.addTab(self._create_training_tab(), "Trening")
        self.settings_tabs.addTab(self._create_interface_tab(), "Interfejs")
        self.settings_tabs.addTab(self._create_system_tab(), "System")

        layout.addWidget(self.settings_tabs)

        # Przyciski akcji
        buttons_layout = QHBoxLayout()

        self.save_btn = QPushButton("Zapisz ustawienia")
        self.save_btn.clicked.connect(self._save_settings)
        self.save_btn.setFixedHeight(24)
        buttons_layout.addWidget(self.save_btn)

        self.reset_btn = QPushButton("Przywróć domyślne")
        self.reset_btn.clicked.connect(self._reset_settings)
        self.reset_btn.setFixedHeight(24)
        buttons_layout.addWidget(self.reset_btn)

        self.export_btn = QPushButton("Eksportuj ustawienia")
        self.export_btn.clicked.connect(self._export_settings)
        self.export_btn.setFixedHeight(24)
        buttons_layout.addWidget(self.export_btn)

        self.import_btn = QPushButton("Importuj ustawienia")
        self.import_btn.clicked.connect(self._import_settings)
        self.import_btn.setFixedHeight(24)
        buttons_layout.addWidget(self.import_btn)

        buttons_layout.addStretch(1)
        layout.addLayout(buttons_layout)

        # Przyciski OK i Anuluj
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        # Załaduj ustawienia
        self._load_settings()

    def connect_signals(self):
        """Podłącza sygnały do slotów."""
        pass

    def refresh(self):
        """Odświeża zawartość zakładki."""
        self._load_settings()

    def update_settings(self, settings):
        """Aktualizuje ustawienia zakładki."""
        self.settings = settings
        self._load_settings()

    def save_state(self):
        """Zapisuje stan zakładki."""
        return {}

    def restore_state(self, state):
        """Przywraca zapisany stan zakładki."""
        pass

    def _create_general_tab(self):
        """Tworzy zakładkę ustawień ogólnych."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Grupa katalogów
        directories_group = QGroupBox("Katalogi")
        directories_layout = QFormLayout(directories_group)

        # Katalog danych
        self.data_dir_edit = QLineEdit()
        self.data_dir_edit.setReadOnly(True)
        data_dir_btn = QPushButton("Przeglądaj")
        data_dir_btn.clicked.connect(
            lambda: self._select_directory(self.data_dir_edit, "Wybierz katalog danych")
        )
        data_dir_layout = QHBoxLayout()
        data_dir_layout.addWidget(self.data_dir_edit)
        data_dir_layout.addWidget(data_dir_btn)
        directories_layout.addRow("Katalog danych:", data_dir_layout)

        # Katalog modeli
        self.models_dir_edit = QLineEdit()
        self.models_dir_edit.setReadOnly(True)
        models_dir_btn = QPushButton("Przeglądaj")
        models_dir_btn.clicked.connect(
            lambda: self._select_directory(
                self.models_dir_edit, "Wybierz katalog modeli"
            )
        )
        models_dir_layout = QHBoxLayout()
        models_dir_layout.addWidget(self.models_dir_edit)
        models_dir_layout.addWidget(models_dir_btn)
        directories_layout.addRow("Katalog modeli:", models_dir_layout)

        # Katalog raportów
        self.reports_dir_edit = QLineEdit()
        self.reports_dir_edit.setReadOnly(True)
        reports_dir_btn = QPushButton("Przeglądaj")
        reports_dir_btn.clicked.connect(
            lambda: self._select_directory(
                self.reports_dir_edit, "Wybierz katalog raportów"
            )
        )
        reports_dir_layout = QHBoxLayout()
        reports_dir_layout.addWidget(self.reports_dir_edit)
        reports_dir_layout.addWidget(reports_dir_btn)
        directories_layout.addRow("Katalog raportów:", reports_dir_layout)

        layout.addWidget(directories_group)

        # Grupa logowania
        logging_group = QGroupBox("Logowanie")
        logging_layout = QFormLayout(logging_group)

        # Poziom logowania
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        logging_layout.addRow("Poziom logowania:", self.log_level_combo)

        # Plik logów
        self.log_file_edit = QLineEdit()
        self.log_file_edit.setReadOnly(True)
        log_file_btn = QPushButton("Przeglądaj")
        log_file_btn.clicked.connect(
            lambda: self._select_file(
                self.log_file_edit,
                "Wybierz plik logów",
                "Pliki logów (*.log);;Wszystkie pliki (*.*)",
            )
        )
        log_file_layout = QHBoxLayout()
        log_file_layout.addWidget(self.log_file_edit)
        log_file_layout.addWidget(log_file_btn)
        logging_layout.addRow("Plik logów:", log_file_layout)

        layout.addWidget(logging_group)

        # Dodaj elastyczną przestrzeń na dole
        layout.addStretch(1)

        return tab

    def _create_model_tab(self):
        """Tworzy zakładkę ustawień modelu."""
        model_tab = QWidget()
        model_layout = QFormLayout(model_tab)

        # Ustawienia modelu
        model_group = QGroupBox("Ustawienia modelu")
        model_group_layout = QFormLayout(model_group)

        # Próg pewności
        self.confidence_threshold = QSpinBox()
        self.confidence_threshold.setRange(0, 100)
        self.confidence_threshold.setValue(50)
        self.confidence_threshold.setSuffix("%")
        model_group_layout.addRow("Próg pewności:", self.confidence_threshold)

        # Użyj GPU jeśli dostępne
        self.use_gpu_checkbox = QCheckBox("Użyj GPU jeśli dostępne")
        model_group_layout.addRow("", self.use_gpu_checkbox)

        # Automatyczne ładowanie ostatniego modelu
        self.auto_load_last_model_checkbox = QCheckBox("Automatycznie ładuj ostatnio używany model przy starcie")
        model_group_layout.addRow("", self.auto_load_last_model_checkbox)

        model_layout.addRow(model_group)

        # Ustawienia klasyfikacji
        classification_group = QGroupBox("Ustawienia klasyfikacji")
        classification_layout = QFormLayout(classification_group)

        # Rozmiar wsadu
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 128)
        self.batch_size.setValue(32)
        classification_layout.addRow("Rozmiar wsadu:", self.batch_size)

        # Liczba wątków
        self.num_workers = QSpinBox()
        self.num_workers.setRange(0, 16)
        self.num_workers.setValue(4)
        classification_layout.addRow("Liczba wątków:", self.num_workers)

        model_layout.addRow(classification_group)

        return model_tab

    def _create_training_tab(self):
        """Tworzy zakładkę ustawień treningu."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Grupa ustawień treningu
        training_group = QGroupBox("Ustawienia treningu")
        training_layout = QFormLayout(training_group)

        # Liczba epok
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(100)
        training_layout.addRow("Liczba epok:", self.epochs_spin)

        # Rozmiar wsadu
        self.train_batch_size_spin = QSpinBox()
        self.train_batch_size_spin.setRange(1, 1000)
        self.train_batch_size_spin.setValue(32)
        training_layout.addRow("Rozmiar wsadu:", self.train_batch_size_spin)

        # Współczynnik uczenia
        self.learning_rate_combo = QComboBox()
        self.learning_rate_combo.addItems(["0.1", "0.01", "0.001", "0.0001"])
        training_layout.addRow("Współczynnik uczenia:", self.learning_rate_combo)

        # Optymalizator
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["Adam", "SGD", "RMSprop", "AdamW"])
        training_layout.addRow("Optymalizator:", self.optimizer_combo)

        layout.addWidget(training_group)

        # Grupa augmentacji danych
        augmentation_group = QGroupBox("Augmentacja danych")
        augmentation_layout = QFormLayout(augmentation_group)

        # Włącz augmentację
        self.use_augmentation_checkbox = QCheckBox("Używaj augmentacji danych")
        augmentation_layout.addRow("", self.use_augmentation_checkbox)

        # Rotacja
        self.rotation_spin = QSpinBox()
        self.rotation_spin.setRange(0, 360)
        self.rotation_spin.setValue(15)
        self.rotation_spin.setSuffix("°")
        augmentation_layout.addRow("Maksymalny kąt rotacji:", self.rotation_spin)

        # Jasność
        self.brightness_spin = QSpinBox()
        self.brightness_spin.setRange(0, 100)
        self.brightness_spin.setValue(20)
        self.brightness_spin.setSuffix("%")
        augmentation_layout.addRow("Zmiana jasności:", self.brightness_spin)

        layout.addWidget(augmentation_group)

        # Dodaj elastyczną przestrzeń na dole
        layout.addStretch(1)

        return tab

    def _create_interface_tab(self):
        """Tworzy zakładkę ustawień interfejsu."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Grupa wyglądu
        appearance_group = QGroupBox("Wygląd")
        appearance_layout = QFormLayout(appearance_group)

        # Motyw
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Jasny", "Ciemny", "Systemowy"])
        appearance_layout.addRow("Motyw:", self.theme_combo)

        # Język
        self.language_combo = QComboBox()
        self.language_combo.addItems(["Polski", "English"])
        appearance_layout.addRow("Język:", self.language_combo)

        # Rozmiar czcionki
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8, 24)
        self.font_size_spin.setValue(11)
        appearance_layout.addRow("Rozmiar czcionki:", self.font_size_spin)

        layout.addWidget(appearance_group)

        # Grupa zachowania
        behavior_group = QGroupBox("Zachowanie")
        behavior_layout = QFormLayout(behavior_group)

        # Automatyczne zapisywanie
        self.autosave_checkbox = QCheckBox("Automatyczne zapisywanie")
        behavior_layout.addRow("", self.autosave_checkbox)

        # Potwierdzenia
        self.confirm_exit_checkbox = QCheckBox("Potwierdzaj zamknięcie")
        behavior_layout.addRow("", self.confirm_exit_checkbox)

        # Powiadomienia
        self.notifications_checkbox = QCheckBox("Pokazuj powiadomienia")
        behavior_layout.addRow("", self.notifications_checkbox)

        layout.addWidget(behavior_group)

        # Dodaj elastyczną przestrzeń na dole
        layout.addStretch(1)

        return tab

    def _create_system_tab(self):
        """Tworzy zakładkę ustawień systemowych."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Grupa zasobów
        resources_group = QGroupBox("Zasoby systemowe")
        resources_layout = QFormLayout(resources_group)

        # Limit pamięci
        self.memory_limit_spin = QSpinBox()
        self.memory_limit_spin.setRange(1024, 32768)
        self.memory_limit_spin.setValue(4096)
        self.memory_limit_spin.setSuffix(" MB")
        resources_layout.addRow("Limit pamięci:", self.memory_limit_spin)

        # Liczba wątków
        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(1, 16)
        self.threads_spin.setValue(4)
        resources_layout.addRow("Liczba wątków:", self.threads_spin)

        layout.addWidget(resources_group)

        # Grupa kopii zapasowej
        backup_group = QGroupBox("Kopia zapasowa")
        backup_layout = QFormLayout(backup_group)

        # Włącz kopie zapasowe
        self.backup_enabled_checkbox = QCheckBox("Twórz kopie zapasowe")
        backup_layout.addRow("", self.backup_enabled_checkbox)

        # Katalog kopii
        self.backup_dir_edit = QLineEdit()
        self.backup_dir_edit.setReadOnly(True)
        backup_dir_btn = QPushButton("Przeglądaj")
        backup_dir_btn.clicked.connect(
            lambda: self._select_directory(
                self.backup_dir_edit, "Wybierz katalog kopii zapasowych"
            )
        )
        backup_dir_layout = QHBoxLayout()
        backup_dir_layout.addWidget(self.backup_dir_edit)
        backup_dir_layout.addWidget(backup_dir_btn)
        backup_layout.addRow("Katalog kopii:", backup_dir_layout)

        # Częstotliwość
        self.backup_interval_spin = QSpinBox()
        self.backup_interval_spin.setRange(1, 168)
        self.backup_interval_spin.setValue(24)
        self.backup_interval_spin.setSuffix(" h")
        backup_layout.addRow("Częstotliwość:", self.backup_interval_spin)

        layout.addWidget(backup_group)

        # Dodaj elastyczną przestrzeń na dole
        layout.addStretch(1)

        return tab

    def _select_directory(self, line_edit, title):
        """Wyświetla dialog wyboru katalogu."""
        directory = QFileDialog.getExistingDirectory(
            self, title, "", QFileDialog.Option.ShowDirsOnly
        )
        if directory:
            line_edit.setText(directory)

    def _select_file(self, line_edit, title, filter_str):
        """Wyświetla dialog wyboru pliku."""
        file_path, _ = QFileDialog.getOpenFileName(self, title, "", filter_str)
        if file_path:
            line_edit.setText(file_path)

    def _load_settings(self):
        """Ładuje ustawienia z pliku."""
        try:
            # Ustawienia interfejsu
            self.theme_combo.setCurrentText(self.settings.get("theme", "Ciemny"))
            self.language_combo.setCurrentText(self.settings.get("language", "Polski"))
            self.log_level_combo.setCurrentText(
                self.settings.get("log_level", "INFO")
            )
            self.log_file_edit.setText(self.settings.get("log_file", "logs/app.log"))

            # Ustawienia modelu
            self.confidence_threshold.setValue(
                int(self.settings.get("confidence_threshold", 50))
            )
            self.use_gpu_checkbox.setChecked(self.settings.get("use_gpu", True))
            self.batch_size.setValue(self.settings.get("batch_size", 32))
            self.num_workers.setValue(self.settings.get("num_workers", 4))
            self.auto_load_last_model_checkbox.setChecked(
                self.settings.get("auto_load_last_model", True)
            )

            # Ustawienia treningu
            self.learning_rate_combo.setCurrentText(
                str(self.settings.get("learning_rate", 0.001))
            )
            self.epochs_spin.setValue(int(self.settings.get("epochs", 10)))
            self.train_batch_size_spin.setValue(self.settings.get("batch_size", 32))
            self.optimizer_combo.setCurrentText(
                self.settings.get("optimizer", "Adam")
            )
            self.use_augmentation_checkbox.setChecked(
                self.settings.get("use_augmentation", True)
            )
            self.rotation_spin.setValue(self.settings.get("rotation_angle", 15))
            self.brightness_spin.setValue(self.settings.get("brightness_change", 20))

            # Ustawienia systemowe
            self.memory_limit_spin.setValue(self.settings.get("memory_limit", 4096))
            self.threads_spin.setValue(self.settings.get("threads", 4))
            self.backup_enabled_checkbox.setChecked(
                self.settings.get("backup_enabled", False)
            )
            self.backup_dir_edit.setText(self.settings.get("backup_dir", "data/backup"))
            self.backup_interval_spin.setValue(self.settings.get("backup_interval", 24))

        except Exception as e:
            self.parent.logger.error(f"Błąd ładowania ustawień: {str(e)}")
            QMessageBox.critical(
                self, "Błąd", f"Nie udało się załadować ustawień:\n{str(e)}"
            )

    def _save_settings(self):
        """Zapisuje ustawienia do pliku."""
        try:
            settings = {
                # Ustawienia interfejsu
                "theme": self.theme_combo.currentText(),
                "language": self.language_combo.currentText(),
                "log_level": self.log_level_combo.currentText(),
                "log_file": self.log_file_edit.text(),
                # Ustawienia modelu
                "confidence_threshold": self.confidence_threshold.value(),
                "use_gpu": self.use_gpu_checkbox.isChecked(),
                "batch_size": self.batch_size.value(),
                "num_workers": self.num_workers.value(),
                "auto_load_last_model": self.auto_load_last_model_checkbox.isChecked(),
                # Ustawienia treningu
                "learning_rate": float(self.learning_rate_combo.currentText()),
                "epochs": self.epochs_spin.value(),
                "batch_size": self.train_batch_size_spin.value(),
                "optimizer": self.optimizer_combo.currentText(),
                "use_augmentation": self.use_augmentation_checkbox.isChecked(),
                "rotation_angle": self.rotation_spin.value(),
                "brightness_change": self.brightness_spin.value(),
                # Ustawienia systemowe
                "memory_limit": self.memory_limit_spin.value(),
                "threads": self.threads_spin.value(),
                "backup_enabled": self.backup_enabled_checkbox.isChecked(),
                "backup_dir": self.backup_dir_edit.text(),
                "backup_interval": self.backup_interval_spin.value(),
            }

            # Zapisz ustawienia
            with open(self.settings_file, "w") as f:
                json.dump(settings, f, indent=4)

            # Zaktualizuj ustawienia w głównym oknie
            self.parent.settings.update(settings)
            self.parent._save_settings()

            QMessageBox.information(
                self, "Sukces", "Ustawienia zostały zapisane pomyślnie."
            )

        except Exception as e:
            self.parent.logger.error(f"Błąd zapisywania ustawień: {str(e)}")
            QMessageBox.critical(
                self, "Błąd", f"Nie udało się zapisać ustawień:\n{str(e)}"
            )

    def _reset_settings(self):
        """Przywraca domyślne ustawienia."""
        reply = QMessageBox.question(
            self,
            "Potwierdzenie",
            "Czy na pewno chcesz przywrócić domyślne ustawienia?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            # Wczytaj domyślne ustawienia
            self.parent._load_default_settings()
            self.settings = self.parent.settings

            # Odśwież interfejs
            self._load_settings()

            # Wyświetl komunikat o sukcesie
            QMessageBox.information(
                self,
                "Sukces",
                "Przywrócono domyślne ustawienia. " "Zapisz zmiany, aby je zastosować.",
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Błąd",
                f"Nie udało się przywrócić domyślnych ustawień: {str(e)}",
            )

    def _export_settings(self):
        """Eksportuje ustawienia do pliku."""
        try:
            # Wybierz miejsce zapisu
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Eksportuj ustawienia",
                "",
                "Pliki JSON (*.json);;Wszystkie pliki (*.*)",
            )

            if not file_path:
                return

            # Zapisz ustawienia do pliku
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=4, ensure_ascii=False)

            # Wyświetl komunikat o sukcesie
            QMessageBox.information(
                self,
                "Sukces",
                f"Ustawienia zostały wyeksportowane do pliku:\n{file_path}",
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Błąd",
                f"Nie udało się wyeksportować ustawień: {str(e)}",
            )

    def _import_settings(self):
        """Importuje ustawienia z pliku."""
        try:
            # Wybierz plik do importu
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Importuj ustawienia",
                "",
                "Pliki JSON (*.json);;Wszystkie pliki (*.*)",
            )

            if not file_path:
                return

            # Wczytaj ustawienia z pliku
            with open(file_path, "r", encoding="utf-8") as f:
                settings = json.load(f)

            # Walidacja ustawień
            if not validate_settings(settings):
                QMessageBox.warning(
                    self,
                    "Ostrzeżenie",
                    "Plik zawiera nieprawidłowe ustawienia. "
                    "Import został przerwany.",
                )
                return

            # Zaktualizuj ustawienia
            self.settings = settings
            self._load_settings()

            # Wyświetl komunikat o sukcesie
            QMessageBox.information(
                self,
                "Sukces",
                "Ustawienia zostały zaimportowane. "
                "Zapisz zmiany, aby je zastosować.",
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Błąd", f"Nie udało się zaimportować ustawień: {str(e)}"
            )
