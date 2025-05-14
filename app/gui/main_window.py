# Standardowe biblioteki
# import datetime # Usunięto nieużywany import
import json
import logging  # Dodano import logging
import os

# import time # nieużywane
import traceback  # Dodano import traceback

# Zewnętrzne biblioteki
import torch
from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import QFrame  # Potrzebne do separatora
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.core.logger import Logger

# Wewnętrzne importy aplikacji
from app.gui.tabs import SettingsManager  # Dodano import SettingsManager
from app.gui.tabs import (
    BatchProcessor,
    HelpTab,
    ImageClassifierTab,
    ModelManager,
    ReportGenerator,
    TrainingManager,
)
from app.utils.profiler import HardwareProfiler  # Dodano import

# import shutil # Usunięto nieużywany import shutil


# import sys # Usunięto nieużywany import


# Klasa pomocnicza do logowania do UI
class QtLogHandler(logging.Handler, QObject):
    log_signal = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__()
        QObject.__init__(self, parent)  # Inicjalizuj QObject

    def emit(self, record):
        msg = self.format(record)
        self.log_signal.emit(msg)


class MainWindow(QMainWindow):
    def __init__(self, settings=None):
        """Inicjalizacja głównego okna aplikacji."""
        super().__init__()
        try:
            # Inicjalizacja loggera
            self.logger = Logger()
            self.logger.info("MainWindow __init__: Start")

            # Inicjalizuj self.settings jako pusty słownik na samym początku
            self.settings = settings if settings is not None else {}
            self.logger.info(
                f"MainWindow __init__: self.settings zainicjalizowane jako: {self.settings}"
            )

            self._setup_console_logging()  # Konfiguracja handlera dla konsoli UI

            # Inicjalizacja urządzenia do obliczeń
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Cache w pamięci zamiast pliku settings.json
            self._settings_cache = {}
            self.hardware_profile = None
            self.current_model = None  # Dodano atrybut dla aktualnego modelu
            self.current_image_path = None  # Dodano atrybut dla ścieżki obrazu

            # Załaduj ustawienia tylko jeśli nie zostały przekazane
            if settings is None:
                self.logger.info("MainWindow __init__: Przed _load_settings()")
                self._load_settings()
                self.logger.info(
                    f"MainWindow __init__: Po _load_settings(), self.settings = {self.settings}"
                )

            # Zastosuj motyw Material Design
            self._apply_material_theme()

            # Inicjalizacja pamięci podręcznej modeli i profilu wydajności
            self._init_model_cache()

            # Inicjalizacja profilu sprzętowego
            self._load_hardware_profile()  # Próba załadowania profilu

            # Utwórz interfejs
            self._create_menu()
            self.logger.info(
                f"MainWindow __init__: Przed _create_central_widget(), self.settings = {self.settings}"
            )  # Log przed tworzeniem zakładek
            self._create_central_widget()
            self._create_status_bar()

            # Ustaw parametry okna
            self.setWindowTitle("CFAB NeuroSorter")

            # Ustaw ikonę aplikacji
            icon_path = os.path.join(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                ),
                "resources",
                "img",
                "icon.png",
            )
            if os.path.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))
            else:
                self.logger.warning(f"Nie znaleziono pliku ikony: {icon_path}")

            self.setMinimumSize(1700, 900)

            # Jeśli nie mamy profilu, zaproponuj profilowanie
            if self.hardware_profile is None:
                self.logger.warning(
                    "Nie znaleziono profilu sprzętowego - proponuję profilowanie"
                )
                QTimer.singleShot(1000, self._propose_profiling)
            else:
                self.logger.info("Pomyślnie załadowano profil sprzętowy")

        except Exception as e:
            self.logger.error(f"Błąd podczas inicjalizacji aplikacji: {str(e)}")
            QMessageBox.critical(
                self,
                "Błąd inicjalizacji",
                f"Wystąpił błąd podczas uruchamiania aplikacji: {str(e)}",
            )

    def _apply_material_theme(self):
        """Aplikuje styl Material Design do aplikacji."""
        try:
            # Kolory zgodne z Material Design i VS Code
            primary_color = "#007ACC"  # Niebieski VS Code
            success_color = "#10B981"  # Zielony
            warning_color = "#DC2626"  # Czerwony
            background = "#1E1E1E"  # Ciemne tło
            surface = "#252526"  # Lekko jaśniejsze tło dla paneli
            border_color = "#3F3F46"  # Kolor obramowania
            text_color = "#CCCCCC"  # Kolor tekstu

            # Dla ciemnego motywu (motyw VS Code)
            self.setStyleSheet(
                f"""
                QMainWindow, QDialog {{
                    background-color: {background};
                    color: {text_color};
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
                    background-color: #2A2D2E;
                }}
                QPushButton:pressed {{
                    background-color: #3E3E40;
                }}
                QPushButton[action="primary"] {{
                    background-color: {primary_color};
                    color: white;
                    border: none;
                }}
                QPushButton[action="primary"]:hover {{
                    background-color: #1C97EA;
                }}
                QPushButton[action="primary"]:pressed {{
                    background-color: #005A9E;
                }}
                QPushButton[action="warning"] {{
                    background-color: {warning_color};
                    color: white;
                    border: none;
                }}
                QPushButton[action="warning"]:hover {{
                    background-color: #EF4444;
                }}
                QPushButton[action="warning"]:pressed {{
                    background-color: #B91C1C;
                }}
                QPushButton[action="success"] {{
                    background-color: {success_color};
                    color: white;
                    border: none;
                }}
                QPushButton[action="success"]:hover {{
                    background-color: #059669;
                }}
                QPushButton[action="success"]:pressed {{
                    background-color: #047857;
                }}
                QTabWidget::pane {{
                    border: 1px solid {border_color};
                    background-color: {surface};
                    color: {text_color};
                }}
                QTabBar::tab {{
                    background-color: {background};
                    color: {text_color};
                    padding: 5px 10px;
                    margin-right: 2px;
                    border-top-left-radius: 2px;
                    border-top-right-radius: 2px;
                    border: 1px solid {border_color};
                    min-width: 80px;
                    max-height: 25px;
                }}
                QTabBar::tab:selected {{
                    background-color: {surface};
                    border-bottom-color: {surface};
                }}
                QGroupBox {{
                    background-color: {surface};
                    color: {text_color};
                    border: 1px solid {border_color};
                    border-radius: 2px;
                    margin-top: 14px;
                    padding-top: 14px;
                    font-weight: normal;
                }}
                QGroupBox::title {{
                    subcontrol-origin: margin;
                    left: 8px;
                    padding: 0 5px;
                    color: #FFFFFF;
                }}
                QLineEdit, QTextEdit, QTableWidget {{
                    background-color: #1C1C1C;
                    color: {text_color};
                    border: 1px solid {border_color};
                    border-radius: 2px;
                    padding: 2px;
                    selection-background-color: #264F78;
                }}
                QTableWidget::item:selected {{
                    background-color: #264F78;
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
                }}
                QProgressBar::chunk {{
                    background-color: {primary_color};
                }}
                QLabel {{
                    color: {text_color};
                }}
                QMenu {{
                    background-color: {surface};
                    color: {text_color};
                    border: 1px solid {border_color};
                }}
                QMenu::item:selected {{
                    background-color: #264F78;
                }}
                """
            )
        except Exception as e:
            self.logger.error(
                f"Błąd aplikacji motywu: {str(e)}\n{traceback.format_exc()}"
            )

    def _create_menu(self):
        """Tworzy menu główne aplikacji."""
        menubar = self.menuBar()

        # Menu Plik
        file_menu = menubar.addMenu("Plik")

        exit_action = QAction("Zakończ", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Menu Ustawienia
        settings_menu = menubar.addMenu("Ustawienia")

        global_settings_action = QAction("Ustawienia globalne", self)
        global_settings_action.triggered.connect(self._show_global_settings)
        settings_menu.addAction(global_settings_action)

        # Menu Narzędzia
        tools_menu = menubar.addMenu("Narzędzia")

        # Dodaj akcje do menu Narzędzia
        profile_action = QAction("Profilowanie sprzętu", self)
        profile_action.triggered.connect(self._propose_profiling)
        tools_menu.addAction(profile_action)

        clear_cache_action = QAction("Wyczyść pamięć podręczną", self)
        clear_cache_action.triggered.connect(self._clear_model_cache)
        tools_menu.addAction(clear_cache_action)

        tools_menu.addSeparator()

        optimize_action = QAction("Optymalizuj wydajność", self)
        optimize_action.triggered.connect(self._optimize_performance)
        tools_menu.addAction(optimize_action)

        tools_menu.addSeparator()

        data_splitter_action = QAction("Przygotowanie danych AI", self)
        data_splitter_action.triggered.connect(self._run_data_splitter)
        tools_menu.addAction(data_splitter_action)

        model_viewer_action = QAction("Przeglądarka modeli", self)
        model_viewer_action.triggered.connect(self._run_model_viewer)
        tools_menu.addAction(model_viewer_action)

        folder_compare_action = QAction("Porównywarka folderów", self)
        folder_compare_action.triggered.connect(self._run_folder_compare)
        tools_menu.addAction(folder_compare_action)

        # Menu Pomoc
        help_menu = menubar.addMenu("Pomoc")

        about_action = QAction("O programie", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _create_central_widget(self):
        """Tworzy główny widget centralny."""
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)

        # Utwórz zakładki
        self.tab_widget = QTabWidget()

        # Dodanie zakładek z nowych klas w odpowiedniej kolejności
        self.image_classifier_tab = ImageClassifierTab(self, self.settings)
        self.model_manager_tab = ModelManager(self, self.settings)
        self.batch_processor_tab = BatchProcessor(self, self.settings)
        self.training_manager_tab = TrainingManager(self, self.settings)
        self.report_generator_tab = ReportGenerator(self, self.settings)
        self.help_tab = HelpTab(self, self.settings)

        # Nowa kolejność i nazwy zakładek
        self.tab_widget.addTab(self.model_manager_tab, "Modele")
        self.tab_widget.addTab(self.training_manager_tab, "Trening")
        self.tab_widget.addTab(self.image_classifier_tab, "Klasyfikacja")
        self.tab_widget.addTab(self.batch_processor_tab, "Klasyfikacja wsadowa")
        raporty_index = self.tab_widget.addTab(self.report_generator_tab, "Raporty")
        self.tab_widget.addTab(self.help_tab, "Pomoc")
        # Ustaw zakładkę "Raporty" jako nieaktywną (jeśli to możliwe)
        try:
            self.tab_widget.setTabEnabled(raporty_index, False)
        except Exception:
            pass

        layout.addWidget(self.tab_widget)

        # Dodaj panel konsoli poniżej zakładek
        self._create_console_panel(layout)  # Przekaż główny layout

        # Aktualizuj profil sprzętowy w zakładce treningu, jeśli jest dostępny
        if hasattr(self, "hardware_profile") and self.hardware_profile:
            self.training_manager_tab.update_hardware_profile(self.hardware_profile)

        # Odśwież listę modeli przy starcie
        self.model_manager_tab.refresh()

        self.setCentralWidget(central_widget)

    def _create_status_bar(self):
        """Tworzy pasek statusu."""
        # Dodaj informację o aktywnym modelu po lewej stronie
        self.active_model_label = QLabel("Aktywny model: brak")
        self.statusBar().addWidget(self.active_model_label)

        # Dodaj informacje o systemie
        self.system_info_label = QLabel()
        self.statusBar().addPermanentWidget(self.system_info_label)
        self._update_system_info()

        # Timer do aktualizacji informacji systemowych
        self.system_info_timer = QTimer()
        self.system_info_timer.timeout.connect(self._update_system_info)
        self.system_info_timer.start(5000)  # Aktualizuj co 5 sekund

    def _get_default_settings_dict(self):
        from app.utils.config import load_default_settings as util_load_defaults

        return util_load_defaults()

    def _load_default_settings(self):
        """Ładuje domyślne ustawienia aplikacji DO self.settings."""
        try:
            # from app.utils.config import load_default_settings # Już zaimportowane w _get_default_settings_dict
            self.settings = self._get_default_settings_dict()
            # Usuń 'last_model' jeśli istnieje po załadowaniu domyślnych
            if "last_model" in self.settings:
                del self.settings["last_model"]
                self.logger.info(
                    "MainWindow _load_default_settings: Usunięto klucz 'last_model' z domyślnych ustawień."
                )
            self.logger.info(
                f"MainWindow _load_default_settings: self.settings ustawione na domyślne: {self.settings}"
            )
            # Usunięto _save_settings() stąd, będzie wywołane w _load_settings() w razie potrzeby
        except Exception as e:
            self.logger.error(
                f"Błąd podczas _load_default_settings: {str(e)}. self.settings może być niekompletne."
            )
            # W ostateczności, można tu ustawić absolutne minimum, jeśli wszystko inne zawiedzie
            self.settings = {}  # Powrót do pustego słownika w razie krytycznego błędu

        # Ostateczne usunięcie 'last_model' z self.settings po wszystkich operacjach ładowania
        if "last_model" in self.settings:
            del self.settings["last_model"]
            self.logger.info(
                "MainWindow _load_settings: Ostatecznie usunięto klucz 'last_model' z self.settings."
            )

    def _load_settings(self):
        """Ładuje ustawienia aplikacji, łącząc z domyślnymi i tworząc plik w razie potrzeby."""
        try:
            settings_filepath = "settings.json"
            default_settings = self._get_default_settings_dict()

            if os.path.exists(settings_filepath):
                try:
                    with open(settings_filepath, "r", encoding="utf-8") as f:
                        loaded_settings_from_file = json.load(f)
                    self.logger.info(
                        f"MainWindow _load_settings: Wczytano z {settings_filepath}: {loaded_settings_from_file}"
                    )

                    # Scalanie: zacznij od domyślnych, nadpisz wczytanymi
                    merged_settings = default_settings.copy()
                    merged_settings.update(loaded_settings_from_file)
                    self.settings.update(
                        merged_settings
                    )  # Aktualizuj główny self.settings
                    self.logger.info(
                        f"MainWindow _load_settings: self.settings po scaleniu: {self.settings}"
                    )

                    # Sprawdź, czy plik wymaga aktualizacji o nowe domyślne klucze
                    if len(merged_settings) > len(loaded_settings_from_file):
                        self.logger.info(
                            f"Plik {settings_filepath} jest aktualizowany o nowe/brakujące klucze z ustawień domyślnych."
                        )
                        self._save_settings()  # Zapisz scalone ustawienia, aby zaktualizować plik

                except json.JSONDecodeError as je:
                    self.logger.error(
                        f"Błąd dekodowania JSON z {settings_filepath}: {str(je)}. Plik może być uszkodzony. Ładowanie domyślnych i próba nadpisania."
                    )
                    self.settings.update(default_settings)  # Użyj domyślnych
                    self._save_settings()  # Spróbuj nadpisać uszkodzony plik domyślnymi
                except Exception as e_file_read:
                    self.logger.error(
                        f"Nieoczekiwany błąd podczas czytania {settings_filepath}: {str(e_file_read)}. Ładowanie domyślnych."
                    )
                    self.settings.update(default_settings)
                    # Nie zapisujemy tutaj, aby uniknąć pętli w razie problemów z zapisem
            else:
                self.logger.warning(
                    f"Plik {settings_filepath} nie istnieje. Ładowanie i zapisywanie ustawień domyślnych."
                )
                self.settings.update(default_settings)  # Użyj domyślnych
                self._save_settings()  # Zapisz domyślne, aby utworzyć plik

        except Exception as e_outer:
            self.logger.error(
                f"Krytyczny błąd w _load_settings: {str(e_outer)}. Ładowanie awaryjnych ustawień domyślnych."
            )
            # Użyj self._load_default_settings(), która ma własny try-except
            # Ta metoda już ustawia self.settings
            # Na wszelki wypadek, jeśli _load_default_settings samo rzuci wyjątek i self.settings nie zostanie ustawione:
            if (
                not self.settings
            ):  # Jeśli self.settings jest nadal puste po błędzie w _load_default_settings
                current_defaults = (
                    self._get_default_settings_dict()
                )  # Pobierz świeże domyślne
                self.settings.update(current_defaults)
            self.logger.info(
                f"MainWindow _load_settings: self.settings po obsłudze błędu zewnętrznego: {self.settings}"
            )
            # Rozważ, czy zapisywać tutaj. Jeśli zapis jest problemem, może to prowadzić do pętli.

    def _save_settings(self):
        """Zapisuje ustawienia aplikacji, usuwając klucz 'last_model'."""
        try:
            settings_to_save = self.settings.copy()
            if "last_model" in settings_to_save:
                del settings_to_save["last_model"]
                self.logger.info(
                    "MainWindow _save_settings: Usunięto klucz 'last_model' przed zapisem do pliku."
                )

            print(f"DEBUG MW._save_settings: settings_to_save = {settings_to_save}")
            self.logger.info(
                f"DEBUG MW._save_settings: settings_to_save = {settings_to_save}"
            )
            with open("settings.json", "w") as f:
                json.dump(settings_to_save, f, indent=4)
        except Exception as e:
            self.logger.error(f"Błąd zapisywania ustawień: {str(e)}")

    def _load_hardware_profile(self):
        """Ładuje profil sprzętowy z bazy danych."""
        try:
            self.logger.info("Rozpoczynam ładowanie profilu sprzętowego...")

            # Utwórz instancję profilera
            profiler = HardwareProfiler()

            # Próba załadowania profilu
            self.hardware_profile = profiler.load_profile()

            if self.hardware_profile:
                self.logger.info(
                    f"Pomyślnie załadowano profil sprzętowy dla machine_id: {self.hardware_profile.get('machine_id')}"
                )
                # Aktualizuj cache
                self._settings_cache["hardware_profile"] = self.hardware_profile

                # Przekaż profil do wszystkich zakładek, które mogą go potrzebować
                if hasattr(self, "training_manager_tab"):
                    self.training_manager_tab.update_hardware_profile(
                        self.hardware_profile
                    )
                if hasattr(self, "model_manager_tab"):
                    self.model_manager_tab.update_hardware_profile(
                        self.hardware_profile
                    )
            else:
                self.logger.warning(
                    "Nie znaleziono profilu sprzętowego w bazie danych."
                )
                self.hardware_profile = None
                self._settings_cache["hardware_profile"] = None

        except Exception as e:
            self.logger.error(f"Błąd ładowania profilu sprzętowego: {str(e)}")
            self.hardware_profile = None
            self._settings_cache["hardware_profile"] = None

    def _open_image(self):
        """Otwiera dialog wyboru obrazu do klasyfikacji."""
        try:
            # Pobierz ostatnio używany katalog lub domyślny
            last_dir = self.settings.get(
                "last_image_dir", os.path.expanduser("~")
            )  # Domyślnie katalog domowy

            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Wybierz obraz",
                last_dir,  # Używa self.settings
                "Obrazy (*.png *.jpg *.jpeg *.bmp *.gif);;" "Wszystkie pliki (*.*)",
            )

            if file_path:
                self.current_image_path = file_path
                self.settings["last_image_dir"] = os.path.dirname(file_path)
                self._save_settings()

                # Przełącz na zakładkę Klasyfikacja
                for i in range(self.tab_widget.count()):
                    if self.tab_widget.tabText(i) == "Klasyfikacja":
                        self.tab_widget.setCurrentIndex(i)
                        break

                # Przekaż obraz do zakładki klasyfikacji
                if hasattr(self, "image_classifier_tab"):
                    self.image_classifier_tab._select_image(file_path)
                else:
                    self.logger.error("Nie znaleziono zakładki klasyfikacji")
                    QMessageBox.critical(
                        self, "Błąd", "Nie znaleziono zakładki klasyfikacji"
                    )
        except Exception as e:
            QMessageBox.critical(
                self, "Błąd", f"Nie udało się otworzyć obrazu: {str(e)}"
            )

    def _show_image_preview(self, image_path):
        """Wyświetla podgląd obrazu (tymczasowo pusta)."""
        self.logger.info(
            f"Wywołano _show_image_preview dla: {image_path} " f"(implementacja pusta)"
        )
        pass  # Pusta implementacja

    def _classify_image(self, image_path=None):
        """Klasyfikuje obraz (tymczasowo pusta)."""
        img_path = image_path if image_path else self.current_image_path
        self.logger.info(
            f"Wywołano _classify_image dla: {img_path} " f"(implementacja pusta)"
        )
        pass  # Pusta implementacja

    # Dodaj nowy slot do obsługi logów
    def _append_log_to_console(self, message):
        if hasattr(self, "console_text"):
            # Zapewnij przewijanie do dołu
            cursor = self.console_text.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            self.console_text.setTextCursor(cursor)
            self.console_text.insertPlainText(message + "\n")  # Dodaj nową linię
            # Można dodać opcjonalne przycinanie, jeśli logów jest za dużo
            # max_lines = 1000
            # if self.console_text.document().blockCount() > max_lines:
            #     cursor.select(cursor.BlockUnderCursor)
            #     cursor.removeSelectedText()
            #     cursor.deletePreviousChar()

    # Dodaj nową metodę do konfiguracji handlera
    def _setup_console_logging(self):
        """Konfiguruje handler logowania do konsoli UI."""
        self.qt_log_handler = QtLogHandler(self)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"
        )
        self.qt_log_handler.setFormatter(formatter)
        self.qt_log_handler.log_signal.connect(self._append_log_to_console)

        # ZAWSZE dodaj handler do loggera
        root_logger = logging.getLogger("CFAB_NeuroSorter")
        root_logger.addHandler(self.qt_log_handler)

    # Dodaj nową metodę tworzącą panel konsoli (wzorowana na starej wersji)
    def _create_console_panel(self, parent_layout):
        """Tworzy panel konsoli z logami i postępem zadania."""
        console_widget = QWidget()
        console_widget.setMinimumHeight(280)
        console_widget.setMaximumHeight(280)
        console_layout = QHBoxLayout(console_widget)
        console_layout.setContentsMargins(0, 8, 0, 0)  # Margines górny
        console_layout.setSpacing(8)  # Odstęp między lewą a prawą częścią

        # Lewa strona - konsola z logami
        console_left = QWidget()
        console_left.setMinimumHeight(260)
        console_left.setMaximumHeight(260)
        console_left_main_layout = QVBoxLayout(console_left)
        console_left_main_layout.setContentsMargins(0, 0, 0, 0)
        console_left_main_layout.setSpacing(4)

        console_header = QLabel("KONSOLA")
        console_header.setStyleSheet(
            "font-weight: bold; color: #CCCCCC; font-size: 11px; padding-bottom: 4px;"
        )
        console_left_main_layout.addWidget(console_header)

        self.console_text = QTextEdit()
        self.console_text.setReadOnly(True)
        self.console_text.setMinimumHeight(200)
        self.console_text.setMaximumHeight(200)
        self.console_text.setStyleSheet(
            "background-color: #1C1C1C; color: #CCCCCC; font-family: 'Consolas', 'Courier New', monospace; font-size: 10px;"
        )
        console_left_main_layout.addWidget(self.console_text)

        button_row = QWidget()
        button_row_layout = QHBoxLayout(button_row)
        button_row_layout.setContentsMargins(0, 0, 0, 0)
        button_row_layout.setSpacing(8)
        clear_btn = QPushButton("Wyczyść")
        clear_btn.clicked.connect(self.console_text.clear)
        clear_btn.setFixedHeight(24)
        button_row_layout.addWidget(clear_btn)
        save_btn = QPushButton("Zapisz logi")
        save_btn.clicked.connect(self._save_console_logs)
        save_btn.setFixedHeight(24)
        button_row_layout.addWidget(save_btn)
        button_row_layout.addStretch(1)
        console_left_main_layout.addWidget(button_row)

        # Prawa strona - panel postępu zadania (na razie podstawowy)
        console_right = QWidget()
        console_right.setMinimumHeight(150)
        console_right.setMaximumHeight(150)
        console_right_layout = QVBoxLayout(console_right)
        console_right_layout.setContentsMargins(24, 0, 24, 0)
        console_right_layout.setSpacing(4)

        progress_header = QLabel("POSTĘP AKTUALNEGO ZADANIA")
        progress_header.setStyleSheet(
            "font-weight: bold; color: #CCCCCC; font-size: 11px; padding-bottom: 4px;"
        )
        console_right_layout.addWidget(progress_header)

        self.current_task_info = QLabel("Brak aktywnego zadania")
        self.current_task_info.setStyleSheet("color: #CCCCCC;")
        console_right_layout.addWidget(self.current_task_info)

        self.task_progress_bar = QProgressBar()
        self.task_progress_bar.setRange(0, 100)
        self.task_progress_bar.setValue(0)
        self.task_progress_bar.setFixedHeight(12)
        self.task_progress_bar.setTextVisible(True)  # Pokaż procenty
        console_right_layout.addWidget(self.task_progress_bar)

        self.task_progress_details = QLabel("")  # Na szczegóły jak epoka/strata
        self.task_progress_details.setStyleSheet("color: #CCCCCC; font-style: italic;")
        console_right_layout.addWidget(self.task_progress_details)

        console_right_layout.addStretch(1)  # Wypchnij elementy do góry
        # Przycisk zatrzymania zadania
        self.stop_task_btn = QPushButton("Zatrzymaj bieżące zadanie")
        self.stop_task_btn.clicked.connect(self._stop_current_task)
        self.stop_task_btn.setFixedHeight(24)
        self.stop_task_btn.setEnabled(False)  # Domyślnie wyłączony
        self.stop_task_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #DC2626;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QPushButton:hover {
                background-color: #EF4444;
            }
            QPushButton:disabled {
                background-color: #4B5563;
                color: #9CA3AF;
            }
        """
        )
        console_right_layout.addWidget(self.stop_task_btn)

        # Dodaj obie strony do głównego układu konsoli
        console_layout.addWidget(console_left, 2)  # Konsola zajmuje 2/3 szerokości
        console_layout.addWidget(console_right, 1)  # Postęp zajmuje 1/3 szerokości

        # Dodaj separator i widget konsoli do głównego layoutu okna
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet("background-color: #3F3F46;")
        parent_layout.addWidget(separator)
        parent_layout.addWidget(console_widget)

    # Dodaj metodę do zapisywania logów
    def _save_console_logs(self):
        """Zapisuje logi konsoli do pliku."""
        if not hasattr(self, "console_text"):
            return
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Zapisz logi konsoli",
                "",
                "Pliki tekstowe (*.txt);;Wszystkie pliki (*.*)",
            )

            if file_path:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(self.console_text.toPlainText())
                QMessageBox.information(
                    self, "Sukces", f"Logi zostały zapisane do pliku:\n{file_path}"
                )
        except Exception as e:
            QMessageBox.critical(
                self, "Błąd", f"Wystąpił błąd podczas zapisywania logów:\n{str(e)}"
            )

    # Dodaj metodę closeEvent
    def closeEvent(self, event):
        """Obsługuje zdarzenie zamknięcia okna."""
        try:
            # Użyj loggera z self, jeśli jest dostępny
            if hasattr(self, "logger"):
                self.logger.info("Zamykanie aplikacji...")
            else:
                print("Zamykanie aplikacji...")  # Fallback

            # Najpierw odłącz sygnał handlera, jeśli istnieje i jest podłączony
            if hasattr(self, "qt_log_handler") and self.qt_log_handler:
                try:
                    # Sprawdź, czy sygnał jest nadal podłączony przed próbą odłączenia
                    # (To wymagałoby bardziej złożonej introspekcji sygnałów,
                    # prostszym podejściem jest po prostu próba usunięcia handlera)
                    pass  # Na razie pomijamy próbę disconnect

                    # Usuń handler z loggera
                    root_logger = logging.getLogger()
                    root_logger.removeHandler(self.qt_log_handler)

                    # Opcjonalnie: Zamknij handler (chociaż standardowy Handler nie ma close)
                    # if hasattr(self.qt_log_handler, 'close'):
                    #     self.qt_log_handler.close()

                    if hasattr(self, "logger"):
                        self.logger.info(
                            "Usunięto i potencjalnie zamknięto handler logowania do konsoli UI."
                        )
                    else:
                        print(
                            "Usunięto i potencjalnie zamknięto handler logowania do konsoli UI."
                        )

                    # Ustaw handler na None, aby uniknąć dalszych prób użycia
                    self.qt_log_handler = None

                except Exception as e:
                    # Użyj print, bo logger może już nie działać poprawnie
                    print(
                        f"Błąd podczas usuwania/zamykania handlera logów: {e}\n{traceback.format_exc()}"
                    )

            # Zaakceptuj zdarzenie zamknięcia (pozwól zamknąć okno)
            # Zrób to na końcu, po wyczyszczeniu zasobów
            event.accept()

        except Exception as e:
            # Logowanie błędu, jeśli coś pójdzie nie tak w samym closeEvent
            # Używamy print, bo stan loggera jest niepewny
            print(
                f"KRYTYCZNY BŁĄD podczas closeEvent: {str(e)}\n{traceback.format_exc()}"
            )
            # Mimo błędu, próbujemy zamknąć aplikację
            event.accept()

    def _stop_current_task(self):
        """Zatrzymuje aktualnie wykonywane zadanie."""
        # Znajdź aktywną zakładkę i zatrzymaj zadanie
        current_tab = self.tab_widget.currentWidget()
        if hasattr(current_tab, "_stop_current_task"):
            current_tab._stop_current_task()
        else:
            self.logger.warning("Aktywna zakładka nie obsługuje zatrzymywania zadań")

    def _update_active_model_info(self):
        """Aktualizuje informacje o aktywnym modelu we wszystkich zakładkach."""
        if hasattr(self, "model_manager_tab"):
            self.model_manager_tab.refresh()

        # Aktualizuj informację o aktywnym modelu w pasku statusu
        if hasattr(self, "active_model_label"):
            if hasattr(self, "current_model") and self.current_model:
                # Jeżeli klasyfikator jest załadowany, pobierz informacje o klasach
                if hasattr(self, "classifier") and self.classifier:
                    class_info = ""
                    if (
                        hasattr(self.classifier, "class_names")
                        and self.classifier.class_names
                    ):
                        class_count = len(self.classifier.class_names)
                        class_info = f" ({class_count} klas)"
                    self.active_model_label.setText(
                        f"Aktywny model: {self.current_model}{class_info}"
                    )
                else:
                    self.active_model_label.setText(
                        f"Aktywny model: {self.current_model}"
                    )
            else:
                self.active_model_label.setText("Aktywny model: brak")

    def _clear_model_cache(self):
        """Czyści pamięć podręczną modeli."""
        try:
            if self.model_cache:
                self.model_cache.clear()
                self.logger.info("Pamięć podręczna modeli została wyczyszczona")
                QMessageBox.information(
                    self,
                    "Sukces",
                    "Pamięć podręczna modeli została pomyślnie wyczyszczona.",
                )
            else:
                self.logger.info("Pamięć podręczna modeli jest już pusta")
                QMessageBox.information(
                    self, "Informacja", "Pamięć podręczna modeli jest już pusta."
                )
        except Exception as e:
            self.logger.error(f"Błąd podczas czyszczenia pamięci podręcznej: {str(e)}")
            QMessageBox.critical(
                self,
                "Błąd",
                f"Wystąpił błąd podczas czyszczenia pamięci podręcznej:\n{str(e)}",
            )

    def _optimize_performance(self):
        """Optymalizuje wydajność aplikacji."""
        try:
            self.logger.info("Rozpoczynam optymalizację wydajności...")

            # Pokaż dialog postępu
            progress_dialog = QProgressDialog(
                "Trwa optymalizacja wydajności...", "Anuluj", 0, 100, self
            )
            progress_dialog.setWindowTitle("Optymalizacja")
            progress_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
            progress_dialog.setMinimumDuration(0)
            progress_dialog.show()
            QApplication.processEvents()

            # Krok 1: Optymalizacja pamięci podręcznej
            progress_dialog.setValue(20)
            progress_dialog.setLabelText("Optymalizacja pamięci podręcznej...")
            QApplication.processEvents()
            self._optimize_model_cache_size()

            # Krok 2: Aktualizacja profilu sprzętowego
            progress_dialog.setValue(50)
            progress_dialog.setLabelText("Aktualizacja profilu sprzętowego...")
            QApplication.processEvents()
            self._load_hardware_profile()

            # Krok 3: Aktualizacja interfejsu
            progress_dialog.setValue(80)
            progress_dialog.setLabelText("Aktualizacja interfejsu...")
            QApplication.processEvents()
            self._update_active_model_info()

            # Zakończenie
            progress_dialog.setValue(100)
            progress_dialog.close()

            self.logger.info("Optymalizacja wydajności zakończona pomyślnie")
            QMessageBox.information(
                self, "Sukces", "Optymalizacja wydajności została zakończona pomyślnie."
            )

        except Exception as e:
            if "progress_dialog" in locals():
                progress_dialog.close()
            self.logger.error(f"Błąd podczas optymalizacji wydajności: {str(e)}")
            QMessageBox.critical(
                self,
                "Błąd",
                f"Wystąpił błąd podczas optymalizacji wydajności:\n{str(e)}",
            )

    def _run_data_splitter(self):
        """Uruchamia narzędzie do przygotowania danych AI."""
        try:
            from tools.data_splitter_gui import CombinedApp

            self.data_splitter = CombinedApp(self.settings)
            self.data_splitter.show()
        except Exception as e:
            self.logger.error(f"Błąd podczas uruchamiania DataSplitter: {str(e)}")
            QMessageBox.critical(
                self, "Błąd", f"Nie udało się uruchomić narzędzia: {str(e)}"
            )

    def _run_model_viewer(self):
        """Uruchamia przeglądarkę modeli."""
        try:
            from tools.model_viewer import ModelViewer

            self.model_viewer = ModelViewer()
            self.model_viewer.show()
        except Exception as e:
            self.logger.error(f"Błąd podczas uruchamiania ModelViewer: {str(e)}")
            QMessageBox.critical(
                self, "Błąd", f"Nie udało się uruchomić przeglądarki modeli: {str(e)}"
            )

    def _run_folder_compare(self):
        """Uruchamia narzędzie do porównywania folderów."""
        try:
            from tools.compare_folders import FolderViewer

            self.folder_viewer = FolderViewer()
            self.folder_viewer.show()
        except Exception as e:
            self.logger.error(f"Błąd podczas uruchamiania FolderViewer: {str(e)}")
            QMessageBox.critical(
                self, "Błąd", f"Nie udało się uruchomić porównywarki folderów: {str(e)}"
            )

    def _apply_settings(self):
        """Stosuje ustawienia do aplikacji."""
        try:
            # Zastosuj motyw
            self._apply_material_theme()

            # Zaktualizuj ustawienia w zakładkach
            if hasattr(self, "image_classifier_tab"):
                self.image_classifier_tab.update_settings(self.settings)
            if hasattr(self, "model_manager_tab"):
                self.model_manager_tab.update_settings(self.settings)
            if hasattr(self, "batch_processor_tab"):
                self.batch_processor_tab.update_settings(self.settings)
            if hasattr(self, "training_manager_tab"):
                self.training_manager_tab.update_settings(self.settings)
            if hasattr(self, "report_generator_tab"):
                self.report_generator_tab.update_settings(self.settings)

            # Zaktualizuj informacje systemowe
            self._update_system_info()

            self.logger.info("Zastosowano nowe ustawienia")

        except Exception as e:
            self.logger.error(f"Błąd stosowania ustawień: {str(e)}")
            QMessageBox.critical(
                self, "Błąd", f"Nie udało się zastosować ustawień:\n{str(e)}"
            )

    def save_settings(self):
        """Publiczna metoda do zapisu ustawień, wywoływana przez SettingsManager."""
        self._save_settings()

    def _init_model_cache(self):
        """Inicjalizuje pamięć podręczną modeli."""
        self.model_cache = {}
        self._optimize_model_cache_size()

    def _optimize_model_cache_size(self):
        """Optymalizuje rozmiar pamięci podręcznej modeli."""
        # Domyślna maksymalna wielkość cache w GB
        max_cache_size_gb = 2
        if torch.cuda.is_available():
            try:
                # Użyj 30% pamięci GPU jako limit cache, jeśli dostępna
                gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (
                    1024**3
                )
                max_cache_size_gb = gpu_mem_gb * 0.3
                self.logger.info(
                    f"Ustawiono limit cache modeli na "
                    f"{max_cache_size_gb:.2f} GB (30% GPU RAM)"
                )
            except Exception as e:
                self.logger.warning(
                    f"Nie udało się pobrać pamięci GPU dla optymalizacji " f"cache: {e}"
                )
                self.logger.info(
                    f"Używam domyślnego limitu cache modeli: " f"{max_cache_size_gb} GB"
                )
        else:
            self.logger.info(
                f"Używam domyślnego limitu cache modeli: "
                f"{max_cache_size_gb} GB (brak GPU)"
            )

        # Oblicz aktualny rozmiar cache
        current_size_gb = 0
        # Przykład obliczenia, jeśli modele mają metodę .size()
        # current_size_gb = sum(model.size() for model in self.model_cache.values()) / (1024**3)

        # Usuń najstarsze modele, jeśli przekroczono limit
        if current_size_gb > max_cache_size_gb:
            self.logger.info(
                f"Aktualny rozmiar cache ({current_size_gb:.2f} GB) "
                f"przekracza limit ({max_cache_size_gb:.2f} GB). "
                f"Usuwanie najstarszych modeli..."
            )
            # Sortuj klucze cache według czasu dodania (zakładając, że zachowują kolejność w Python 3.7+)
            keys_to_remove = list(self.model_cache.keys())

            while current_size_gb > max_cache_size_gb and keys_to_remove:
                oldest_key = keys_to_remove.pop(0)
                try:
                    # Oblicz rozmiar usuwanego modelu (zakładając metodę .element_size() i .nelement())
                    # removed_size_gb = (self.model_cache[oldest_key].element_size() *
                    #                    self.model_cache[oldest_key].nelement()) / (1024**3)
                    # Zastąp powyższe rzeczywistym sposobem obliczania rozmiaru modelu
                    removed_size_gb = 0  # Placeholder

                    del self.model_cache[oldest_key]
                    current_size_gb -= removed_size_gb
                    self.logger.info(f"Usunięto model '{oldest_key}' z cache.")
                except KeyError:
                    self.logger.warning(
                        f"Próbowano usunąć nieistniejący klucz "
                        f"'{oldest_key}' z cache."
                    )
                except AttributeError:
                    self.logger.warning(
                        f"Model dla klucza '{oldest_key}' nie ma metody "
                        f"do obliczenia rozmiaru."
                    )

            self.logger.info(
                f"Zakończono optymalizację cache. "
                f"Aktualny rozmiar: {current_size_gb:.2f} GB"
            )

    def _update_system_info(self):
        """Aktualizuje informacje systemowe w pasku statusu."""
        try:
            import psutil

            # Pobierz informacje o systemie
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Pobierz informacje o GPU jeśli dostępne
            gpu_info = ""
            if torch.cuda.is_available():
                try:
                    gpu_memory = torch.cuda.memory_allocated() / 1024**2
                    gpu_info = f" | GPU: {gpu_memory:.1f}MB"
                except Exception as e:
                    self.logger.warning(f"Błąd przy pobieraniu informacji o GPU: {e}")
                    gpu_info = " | GPU: Dane niedostępne"

            # Aktualizuj etykietę
            self.system_info_label.setText(
                f"CPU: {cpu_percent}% | RAM: {memory_percent}%{gpu_info}"
            )
        except ImportError:
            self.logger.warning(
                "Moduł psutil nie jest zainstalowany. "
                "Informacje systemowe nie będą dostępne."
            )
            self.system_info_label.setText(
                "Informacje systemowe niedostępne (brak psutil)"
            )
            # Zatrzymaj timer, aby nie próbować ponownie
            if hasattr(self, "system_info_timer") and self.system_info_timer.isActive():
                self.system_info_timer.stop()
        except Exception as e:
            self.logger.error(
                f"Błąd aktualizacji informacji systemowych: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            # Można dodać informację w UI
            self.system_info_label.setText("Błąd aktualizacji info sys.")

    def _show_about(self):
        """Wyświetla informacje o programie."""
        try:
            QMessageBox.about(
                self,
                "O programie",
                "CFAB NeuroSorter\n"
                "Wersja 0.4 alpha\n\n"
                "CFAB NeuroSorter to aplikacja do szkolenia i doszkalania modeli, automatyzująca przetwarzanie zadań wsadowo.",
            )
        except Exception as e:
            self.logger.error(
                f"Błąd wyświetlania informacji o programie: {str(e)}\n{traceback.format_exc()}"
            )

    def _show_global_settings(self):
        """Wyświetla okno ustawień globalnych."""
        try:
            # Utwórz instancję SettingsManager
            settings_dialog = SettingsManager(self, self.settings)

            # Wyświetl okno i czekaj na odpowiedź
            if settings_dialog.exec() == QDialog.DialogCode.Accepted:
                # Jeśli użytkownik kliknął OK, zastosuj ustawienia
                print(
                    f"DEBUG MW po SettingsDialog.Accepted: "
                    f"self.settings = {self.settings}"
                )
                self._apply_settings()

        except Exception as e:
            self.logger.error(f"Błąd wyświetlania ustawień globalnych: {str(e)}")
            QMessageBox.critical(
                self, "Błąd", f"Nie udało się wyświetlić ustawień globalnych:\n{str(e)}"
            )

    def _propose_profiling(self):
        """Proponuje uruchomienie profilowania sprzętu."""
        reply = QMessageBox.question(
            self,
            "Profilowanie sprzętu",
            "Czy chcesz uruchomić profilowanie sprzętu, aby "
            "zoptymalizować wydajność aplikacji?\n\n"
            "Profilowanie jest jednorazowym procesem, który analizuje wydajność "
            "Twojego sprzętu i dobiera optymalne parametry dla aplikacji.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.logger.info(
                "Użytkownik zaakceptował propozycję uruchomienia profilowania."
            )

            # Pokaż dialog postępu
            progress_dialog = QProgressDialog(
                "Trwa profilowanie sprzętu...", "Anuluj", 0, 100, self
            )
            progress_dialog.setWindowTitle("Profilowanie")
            progress_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
            progress_dialog.setMinimumDuration(0)
            progress_dialog.show()
            QApplication.processEvents()

            # Aktualizacja interfejsu
            progress_dialog.setValue(10)
            progress_dialog.setLabelText("Inicjalizacja profilowania...")
            QApplication.processEvents()

            try:
                # Wywołaj metodę profilowania w trybie głównym
                profiler = HardwareProfiler()

                # Aktualizacja interfejsu
                progress_dialog.setValue(20)
                progress_dialog.setLabelText(
                    "Profiler zainicjalizowany, rozpoczynam profilowanie..."
                )
                QApplication.processEvents()

                # Wykonaj profilowanie
                profile = profiler.run_profile()

                # Aktualizacja interfejsu
                progress_dialog.setValue(100)
                progress_dialog.setLabelText("Profilowanie zakończone!")
                QApplication.processEvents()

                # Zamknij dialog
                progress_dialog.close()

                # Zapisz profil do ustawień
                if profile:
                    self.hardware_profile = profile
                    self._settings_cache["hardware_profile"] = profile
                    self._save_settings()

                    # Zaktualizuj zakładki
                    if hasattr(self, "training_manager_tab"):
                        self.training_manager_tab.update_hardware_profile(profile)
                    if hasattr(self, "model_manager_tab"):
                        self.model_manager_tab.update_hardware_profile(profile)

                    # Wyświetl wyniki
                    QMessageBox.information(
                        self,
                        "Profilowanie zakończone",
                        f"Profilowanie sprzętu zostało zakończone pomyślnie.\n\n"
                        f"Zalecany rozmiar wsadu: {profile.get('recommended_batch_size', 32)}\n"
                        f"Zalecana liczba wątków: {profile.get('recommended_workers', 4)}\n\n"
                        f"Parametry aplikacji zostały zoptymalizowane dla Twojego systemu.",
                    )
                else:
                    QMessageBox.warning(
                        self,
                        "Profilowanie nieudane",
                        "Nie udało się wygenerować profilu sprzętowego. "
                        "Będą używane wartości domyślne.",
                    )

            except Exception as e:
                progress_dialog.close()
                self.logger.error(f"Błąd podczas profilowania: {str(e)}")
                QMessageBox.critical(
                    self,
                    "Błąd profilowania",
                    f"Wystąpił błąd podczas profilowania:\n{str(e)}",
                )

            self.profiling_proposed = True
        else:
            self.logger.info(
                "Użytkownik odrzucił propozycję uruchomienia profilowania."
            )
