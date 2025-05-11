import json
import logging

from PyQt6.QtWidgets import (
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

from app.utils.settings_utils import validate_settings

# Usunięto nieużywany import TabInterface
# Poniższy import jest problematyczny - wymaga sprawdzenia struktury projektu
# import os # Usunięto nieużywany import


class SettingsManager(QDialog):
    """Klasa zarządzająca ustawieniami aplikacji."""

    def __init__(self, parent=None, settings=None):
        super().__init__(parent)
        self.parent = parent
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.setup_ui()
        self.connect_signals()
        self.logger.info("Zainicjalizowano SettingsManager")

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
        self.settings_tabs.addTab(self._create_interface_tab(), "Interfejs")

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
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        button_box.button(QDialogButtonBox.StandardButton.Cancel).setText("Zamknij")
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

    def _create_interface_tab(self):
        """Tworzy zakładkę ustawień interfejsu."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Grupa kolorów wykresu treningu
        chart_colors_group = QGroupBox("Kolory wykresu treningu")
        chart_colors_layout = QFormLayout(chart_colors_group)

        self.train_loss_color_edit = QLineEdit()
        chart_colors_layout.addRow("Strata treningowa:", self.train_loss_color_edit)

        self.val_loss_color_edit = QLineEdit()
        chart_colors_layout.addRow("Strata walidacyjna:", self.val_loss_color_edit)

        self.train_acc_color_edit = QLineEdit()
        chart_colors_layout.addRow("Dokładność treningowa:", self.train_acc_color_edit)

        self.val_acc_color_edit = QLineEdit()
        chart_colors_layout.addRow("Dokładność walidacyjna:", self.val_acc_color_edit)

        self.chart_plot_area_background_color_edit = QLineEdit()
        chart_colors_layout.addRow(
            "Kolor tła obszaru kreślenia:", self.chart_plot_area_background_color_edit
        )

        layout.addWidget(chart_colors_group)

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
            self.logger.info("Rozpoczynam wczytywanie ustawień")
            self.logger.debug("Rozpoczynam ładowanie ustawień")
            if not self.settings:
                self.logger.warning("Brak ustawień do załadowania")
                return

            self.logger.debug(f"Aktualne ustawienia: {self.settings}")

            # Ustawienia ogólne (zakładka "Ogólne")
            if hasattr(self, "data_dir_edit"):
                value = self.settings.get("data_dir", "data")
                self.data_dir_edit.setText(value)
                self.logger.debug(f"Ustawiono data_dir: {value}")

            if hasattr(self, "models_dir_edit"):
                value = self.settings.get("models_dir", "data/models")
                self.models_dir_edit.setText(value)
                self.logger.debug(f"Ustawiono models_dir: {value}")

            if hasattr(self, "reports_dir_edit"):
                value = self.settings.get("reports_dir", "data/reports")
                self.reports_dir_edit.setText(value)
                self.logger.debug(f"Ustawiono reports_dir: {value}")

            if hasattr(self, "log_level_combo"):
                value = self.settings.get("log_level", "INFO")
                self.log_level_combo.setCurrentText(value)
                self.logger.debug(f"Ustawiono log_level: {value}")

            if hasattr(self, "log_file_edit"):
                value = self.settings.get("log_file", "app.log")
                self.log_file_edit.setText(value)
                self.logger.debug(f"Ustawiono log_file: {value}")

            # Kolory wykresu
            if hasattr(self, "train_loss_color_edit"):
                value = self.settings.get("chart_train_loss_color", "b")
                self.train_loss_color_edit.setText(value)
                self.logger.debug(f"Ustawiono chart_train_loss_color: {value}")
            if hasattr(self, "val_loss_color_edit"):
                value = self.settings.get("chart_val_loss_color", "r")
                self.val_loss_color_edit.setText(value)
                self.logger.debug(f"Ustawiono chart_val_loss_color: {value}")
            if hasattr(self, "train_acc_color_edit"):
                value = self.settings.get("chart_train_acc_color", "g")
                self.train_acc_color_edit.setText(value)
                self.logger.debug(f"Ustawiono chart_train_acc_color: {value}")
            if hasattr(self, "val_acc_color_edit"):
                value = self.settings.get("chart_val_acc_color", "m")
                self.val_acc_color_edit.setText(value)
                self.logger.debug(f"Ustawiono chart_val_acc_color: {value}")
            if hasattr(self, "chart_plot_area_background_color_edit"):
                value = self.settings.get("chart_plot_area_background_color", "w")
                self.chart_plot_area_background_color_edit.setText(value)
                self.logger.debug(
                    "Ustawiono chart_plot_area_background_color: %s", value
                )

            # Ustawienia zachowania
            if hasattr(self, "autosave_checkbox"):
                value = self.settings.get("autosave", True)
                self.autosave_checkbox.setChecked(value)
                self.logger.debug(f"Ustawiono autosave: {value}")
            if hasattr(self, "confirm_exit_checkbox"):
                value = self.settings.get("confirm_exit", True)
                self.confirm_exit_checkbox.setChecked(value)
                self.logger.debug(f"Ustawiono confirm_exit: {value}")
            if hasattr(self, "notifications_checkbox"):
                value = self.settings.get("notifications", True)
                self.notifications_checkbox.setChecked(value)
                self.logger.debug(f"Ustawiono notifications: {value}")

            self.logger.info("Pomyślnie załadowano wszystkie ustawienia")
            self.logger.debug("Zakończono wczytywanie ustawień")

        except Exception as e:
            self.logger.error(
                f"Błąd podczas ładowania ustawień: {str(e)}", exc_info=True
            )
            QMessageBox.critical(
                self, "Błąd", f"Nie udało się załadować ustawień:\n{str(e)}"
            )

    def _save_settings(self):
        """Zbiera ustawienia z UI i zapisuje je do pliku konfiguracyjnego."""
        try:
            self.logger.info("Rozpoczynam zapisywanie ustawień")
            self.logger.debug("Rozpoczynam zbieranie ustawień z UI")

            # Tworzymy nowy słownik tylko z aktualnie dostępnymi ustawieniami
            current_settings = {
                # Ustawienia ogólne
                "data_dir": (
                    self.data_dir_edit.text()
                    if hasattr(self, "data_dir_edit")
                    else "data"
                ),
                "models_dir": (
                    self.models_dir_edit.text()
                    if hasattr(self, "models_dir_edit")
                    else "data/models"
                ),
                "reports_dir": (
                    self.reports_dir_edit.text()
                    if hasattr(self, "reports_dir_edit")
                    else "data/reports"
                ),
                "log_level": (
                    self.log_level_combo.currentText()
                    if hasattr(self, "log_level_combo")
                    else "INFO"
                ),
                "log_file": (
                    self.log_file_edit.text()
                    if hasattr(self, "log_file_edit")
                    else "app.log"
                ),
                # Kolory wykresu
                "chart_train_loss_color": (
                    self.train_loss_color_edit.text()
                    if hasattr(self, "train_loss_color_edit")
                    else "b"
                ),
                "chart_val_loss_color": (
                    self.val_loss_color_edit.text()
                    if hasattr(self, "val_loss_color_edit")
                    else "r"
                ),
                "chart_train_acc_color": (
                    self.train_acc_color_edit.text()
                    if hasattr(self, "train_acc_color_edit")
                    else "g"
                ),
                "chart_val_acc_color": (
                    self.val_acc_color_edit.text()
                    if hasattr(self, "val_acc_color_edit")
                    else "m"
                ),
                "chart_plot_area_background_color": (
                    self.chart_plot_area_background_color_edit.text()
                    if hasattr(self, "chart_plot_area_background_color_edit")
                    else "w"
                ),
                # Ustawienia zachowania
                "autosave": (
                    self.autosave_checkbox.isChecked()
                    if hasattr(self, "autosave_checkbox")
                    else True
                ),
                "confirm_exit": (
                    self.confirm_exit_checkbox.isChecked()
                    if hasattr(self, "confirm_exit_checkbox")
                    else True
                ),
                "notifications": (
                    self.notifications_checkbox.isChecked()
                    if hasattr(self, "notifications_checkbox")
                    else True
                ),
            }

            self.logger.debug(f"Zebrane ustawienia: {current_settings}")

            # Walidacja ustawień
            self.logger.debug("Walidacja zebranych ustawień")
            if not validate_settings(current_settings):
                self.logger.error("Walidacja ustawień nie powiodła się")
                raise ValueError("Nieprawidłowe ustawienia")

            # Aktualizacja ustawień
            self.logger.debug("Aktualizuję ustawienia")
            self.settings = current_settings  # Zastępujemy cały słownik ustawień nowym
            if self.parent and hasattr(self.parent, "settings"):
                self.logger.debug("Aktualizuję ustawienia rodzica")
                self.parent.settings = (
                    current_settings  # Zastępujemy ustawienia rodzica nowym słownikiem
                )

                # Faktyczny zapis ustawień do pliku
                self.logger.debug("Zapisuję ustawienia do pliku")
                if hasattr(self.parent, "save_settings") and callable(
                    self.parent.save_settings
                ):
                    try:
                        self.parent.save_settings()
                        self.logger.info("Pomyślnie zapisano ustawienia do pliku")
                        self.logger.debug("Zakończono zapisywanie ustawień")
                        QMessageBox.information(
                            self,
                            "Sukces",
                            "Ustawienia zostały zapisane pomyślnie.",
                        )
                    except Exception as save_error:
                        self.logger.error(
                            "Błąd podczas zapisywania ustawień do pliku: %s",
                            str(save_error),
                            exc_info=True,
                        )
                        raise
                else:
                    self.logger.error("Brak metody save_settings w obiekcie rodzica")
                    QMessageBox.warning(
                        self,
                        "Ostrzeżenie",
                        "Ustawienia zostały zaktualizowane, "
                        "ale nie mogły zostać zapisane do pliku.",
                    )
            else:
                self.logger.warning(
                    "Brak rodzica lub ustawień rodzica - " "zmiany nie zostaną zapisane"
                )
                QMessageBox.warning(
                    self,
                    "Ostrzeżenie",
                    "Ustawienia zostały zaktualizowane w pamięci, "
                    "ale nie zostaną zapisane trwale.",
                )

        except Exception as e:
            self.logger.error(
                f"Błąd podczas zapisywania ustawień: {str(e)}", exc_info=True
            )
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

    def accept(self):
        """Zapisuje ustawienia i zamyka dialog."""
        # Usunięto automatyczny zapis ustawień przy zamykaniu okna
        super().accept()
