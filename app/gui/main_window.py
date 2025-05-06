# Standardowe biblioteki
# import datetime # Usunięto nieużywany import
import json
import logging  # Dodano import logging
import os
import shutil

# import time # nieużywane
import traceback  # Dodano import traceback

# Zewnętrzne biblioteki
import torch
from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import QFrame  # Potrzebne do separatora
from PyQt6.QtWidgets import (  # Potrzebne; QApplication, # nieużywane; QListWidget, # nieużywane; QStatusBar, # nieużywane; QMenu, # nieużywane; QMenuBar, # nieużywane; QSizePolicy, # nieużywane; QDialogButtonBox, # Usunięto nieużywany import; QScrollArea, # Usunięto nieużywany import
    QApplication,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
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
    def __init__(self):
        """Inicjalizacja głównego okna aplikacji."""
        super().__init__()
        try:
            # Inicjalizacja loggera i podłączenie do konsoli UI
            self.logger = Logger()
            self.logger.info("Inicjalizacja głównego okna aplikacji...")
            self._setup_console_logging()  # Konfiguracja handlera dla konsoli UI

            # Inicjalizacja urządzenia do obliczeń
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Cache w pamięci zamiast pliku settings.json
            self._settings_cache = {}
            self.hardware_profile = None
            self.current_model = None  # Dodano atrybut dla aktualnego modelu

            # Załaduj ustawienia
            self._load_settings()
            self.settings = {}  # Dodano tymczasowy słownik ustawień
            self.current_image_path = None  # Dodano atrybut dla ścieżki obrazu

            # Zastosuj motyw Material Design
            self._apply_material_theme()

            # Inicjalizacja pamięci podręcznej modeli i profilu wydajności
            self._init_model_cache()

            # Inicjalizacja profilu sprzętowego
            self._load_hardware_profile()  # Próba załadowania profilu

            # Utwórz interfejs
            self._create_menu()
            self._create_central_widget()
            self._create_status_bar()

            # Ustaw parametry okna
            self.setWindowTitle("CFAB NeuroSorter")

            # Ustaw ikonę aplikacji
            icon_path = os.path.join(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                ),
                "app",
                "img",
                "icon.png",
            )
            if os.path.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))
            else:
                self.logger.warning(f"Nie znaleziono pliku ikony: {icon_path}")

            self.setMinimumSize(1200, 800)

            # Załaduj ostatnio używany model
            self._load_last_model()

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
            # primary_variant = "#005A9E"  # Ciemniejszy niebieski - usunięto nieużywaną zmienną
            # secondary_color = "#1E1E1E"  # Ciemny szary - usunięto nieużywaną zmienną
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

        tools_menu.addSeparator()

        verify_files_action = QAction("Weryfikacja plików", self)
        verify_files_action.triggered.connect(self._verify_files)
        tools_menu.addAction(verify_files_action)

        # Dodaj podmenu dla narzędzi plikowych
        file_tools_menu = tools_menu.addMenu("Narzędzia plikowe")

        jpeg_to_jpg_action = QAction("Konwerter JPEG -> JPG", self)
        jpeg_to_jpg_action.triggered.connect(self._run_jpeg_to_jpg)
        file_tools_menu.addAction(jpeg_to_jpg_action)

        copy_images_action = QAction("Kopiowanie plików graficznych", self)
        copy_images_action.triggered.connect(self._run_copy_images)
        file_tools_menu.addAction(copy_images_action)

        move_jpeg_action = QAction("Przenoszenie plików JPEG", self)
        move_jpeg_action.triggered.connect(self._run_move_jpeg)
        file_tools_menu.addAction(move_jpeg_action)

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

    def _load_settings(self):
        """Ładuje ustawienia aplikacji."""
        try:
            if os.path.exists("settings.json"):
                with open("settings.json", "r") as f:
                    self.settings = json.load(f)
            else:
                self.settings = {
                    "models_dir": "data/models",
                    "training_dir": "data/training",
                    "reports_dir": "reports",
                    "last_model": None,
                    "hardware_profile": None,
                }
                self._save_settings()
        except Exception as e:
            self.logger.error(f"Błąd ładowania ustawień: {str(e)}")
            self.settings = {
                "models_dir": "data/models",
                "training_dir": "data/training",
                "reports_dir": "reports",
                "last_model": None,
                "hardware_profile": None,
            }

    def _save_settings(self):
        """Zapisuje ustawienia aplikacji."""
        try:
            with open("settings.json", "w") as f:
                json.dump(self.settings, f, indent=4)
        except Exception as e:
            self.logger.error(f"Błąd zapisywania ustawień: {str(e)}")

    def _load_last_model(self):
        """Ładuje ostatnio używany model."""
        if self.settings.get("auto_load_last_model", True) and self.settings.get(
            "last_model"
        ):
            try:
                model_path = os.path.join(
                    self.settings["models_dir"], self.settings["last_model"]
                )
                if os.path.exists(model_path):
                    self.logger.info(
                        f"Ładowanie ostatnio używanego modelu: {self.settings['last_model']}"
                    )
                    self.model_manager_tab.load_model(self.settings["last_model"])
                    self.logger.info("Model został pomyślnie załadowany")
                else:
                    self.logger.warning(
                        f"Ostatnio używany model {self.settings['last_model']} nie istnieje"
                    )
            except Exception as e:
                self.logger.error(f"Błąd ładowania ostatniego modelu: {str(e)}")
                QMessageBox.critical(
                    self,
                    "Błąd",
                    f"Nie udało się załadować ostatniego modelu:\n{str(e)}",
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
                "Wersja 0.3 alpha\n\n"
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
                    f"Pomyślnie załadowano profil sprzętowy dla machine_id: "
                    f"{self.hardware_profile.get('machine_id')}"
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
            import subprocess
            import sys

            # Ścieżka do skryptu data_splitter_gui.py
            script_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data_splitter_gui.py",
            )

            if not os.path.exists(script_path):
                self.logger.error(f"Nie znaleziono skryptu: {script_path}")
                QMessageBox.critical(
                    self,
                    "Błąd",
                    f"Nie znaleziono skryptu przygotowania danych:\n{script_path}",
                )
                return

            # Uruchom skrypt w nowym procesie bez widocznej konsoli
            if sys.platform == "win32":
                subprocess.Popen(
                    [sys.executable, script_path],
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
            else:
                subprocess.Popen(
                    [sys.executable, script_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            self.logger.info("Uruchomiono narzędzie przygotowania danych AI")

        except Exception as e:
            self.logger.error(f"Błąd podczas uruchamiania narzędzia: {str(e)}")
            QMessageBox.critical(
                self, "Błąd", f"Wystąpił błąd podczas uruchamiania narzędzia:\n{str(e)}"
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

    def _load_default_settings(self):
        """Ładuje domyślne ustawienia aplikacji."""
        try:
            from app.utils.config import load_default_settings

            # Załaduj domyślne ustawienia
            self.settings = load_default_settings()

            # Zapisz ustawienia
            self._save_settings()

            self.logger.info("Załadowano domyślne ustawienia")

        except Exception as e:
            self.logger.error(f"Błąd ładowania domyślnych ustawień: {str(e)}")
            QMessageBox.critical(
                self, "Błąd", f"Nie udało się załadować domyślnych ustawień:\n{str(e)}"
            )

    def _verify_files(self):
        """Weryfikuje pliki w wybranym katalogu pod kątem błędów."""
        try:
            # Pobierz ostatnio używany katalog lub domyślny
            last_dir = self.settings.get("last_verify_dir", os.path.expanduser("~"))

            # Wybierz katalog do weryfikacji
            dir_path = QFileDialog.getExistingDirectory(
                self, "Wybierz katalog do weryfikacji", last_dir
            )

            if not dir_path:
                return

            # Zapisz wybrany katalog
            self.settings["last_verify_dir"] = dir_path
            self._save_settings()

            # Utwórz okno dialogowe postępu
            progress_dialog = QProgressDialog(
                "Weryfikacja plików...", "Anuluj", 0, 100, self
            )
            progress_dialog.setWindowTitle("Weryfikacja plików")
            progress_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
            progress_dialog.setMinimumDuration(0)
            progress_dialog.show()
            QApplication.processEvents()

            # Lista do przechowywania wyników weryfikacji
            results = {"total_files": 0, "valid_files": 0, "invalid_files": []}

            # Znajdź wszystkie pliki w katalogu
            all_files = []
            for root, _, files in os.walk(dir_path):
                for file in files:
                    all_files.append(os.path.join(root, file))

            total_files = len(all_files)

            # Sprawdź wszystkie pliki
            for i, file_path in enumerate(all_files):
                if progress_dialog.wasCanceled():
                    break

                results["total_files"] += 1

                try:
                    # Sprawdź czy plik można otworzyć
                    with open(file_path, "rb") as f:
                        # Próba odczytu pierwszych kilku bajtów
                        f.read(1024)

                    # Sprawdź czy to obraz
                    if file_path.lower().endswith(
                        (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")
                    ):
                        try:
                            from PIL import Image

                            img = Image.open(file_path)
                            img.verify()  # Weryfikacja integralności obrazu
                            img.close()

                            # Sprawdź rozmiar pliku
                            file_size = os.path.getsize(file_path)
                            if file_size > 10 * 1024 * 1024:  # Większe niż 10MB
                                results["invalid_files"].append(
                                    {
                                        "path": file_path,
                                        "error": f"Duży rozmiar pliku: {file_size / (1024*1024):.1f} MB",
                                    }
                                )
                            else:
                                results["valid_files"] += 1

                        except Exception as e:
                            results["invalid_files"].append(
                                {
                                    "path": file_path,
                                    "error": f"Błąd weryfikacji obrazu: {str(e)}",
                                }
                            )
                    else:
                        results["valid_files"] += 1

                except Exception as e:
                    results["invalid_files"].append(
                        {"path": file_path, "error": f"Błąd odczytu pliku: {str(e)}"}
                    )

                # Aktualizuj postęp
                progress = int((i + 1) / total_files * 100)
                progress_dialog.setValue(progress)
                QApplication.processEvents()

            progress_dialog.close()

            # Wyświetl wyniki w nowym oknie dialogowym
            dialog = FileVerificationDialog(results, self)
            dialog.exec()

        except Exception as e:
            self.logger.error(f"Błąd podczas weryfikacji plików: {str(e)}")
            QMessageBox.critical(
                self, "Błąd", f"Wystąpił błąd podczas weryfikacji plików:\n{str(e)}"
            )

    def _run_jpeg_to_jpg(self):
        """Uruchamia narzędzie do konwersji plików JPEG na JPG."""
        try:
            from app.utils.file_tools import JpegToJpgConverter

            converter = JpegToJpgConverter(self)
            converter.show()
        except Exception as e:
            self.logger.error(f"Błąd podczas uruchamiania konwertera JPEG: {str(e)}")
            QMessageBox.critical(
                self, "Błąd", f"Wystąpił błąd podczas uruchamiania narzędzia:\n{str(e)}"
            )

    def _run_copy_images(self):
        """Uruchamia narzędzie do kopiowania plików graficznych."""
        try:
            from app.utils.file_tools import ImageCopierApp

            copier = ImageCopierApp(self)
            copier.show()
        except Exception as e:
            self.logger.error(f"Błąd podczas uruchamiania kopiarki plików: {str(e)}")
            QMessageBox.critical(
                self, "Błąd", f"Wystąpił błąd podczas uruchamiania narzędzia:\n{str(e)}"
            )

    def _run_move_jpeg(self):
        """Uruchamia narzędzie do przenoszenia plików JPEG."""
        try:
            from app.utils.file_tools import JpegMoverApp

            mover = JpegMoverApp(self)
            mover.show()
        except Exception as e:
            self.logger.error(
                f"Błąd podczas uruchamiania narzędzia przenoszenia: {str(e)}"
            )
            QMessageBox.critical(
                self, "Błąd", f"Wystąpił błąd podczas uruchamiania narzędzia:\n{str(e)}"
            )


class FileVerificationDialog(QDialog):
    """Dialog wyświetlający wyniki weryfikacji plików w formie tabeli."""

    def __init__(self, results, parent=None):
        super().__init__(parent)
        self.results = results
        self.selected_files = []
        self.setup_ui()

    def setup_ui(self):
        """Konfiguruje interfejs użytkownika dialogu."""
        self.setWindowTitle("Wyniki weryfikacji plików")
        self.setMinimumWidth(800)
        self.setMinimumHeight(600)

        layout = QVBoxLayout(self)

        # Statystyki
        stats_group = QGroupBox("Statystyki")
        stats_layout = QHBoxLayout()
        stats_layout.addWidget(
            QLabel(f"Łączna liczba plików: {self.results['total_files']}")
        )
        stats_layout.addWidget(QLabel(f"Poprawne pliki: {self.results['valid_files']}"))
        stats_layout.addWidget(
            QLabel(f"Błędne pliki: {len(self.results['invalid_files'])}")
        )
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        # Tabela plików
        self.files_table = QTableWidget()
        self.files_table.setColumnCount(5)
        self.files_table.setHorizontalHeaderLabels(
            ["", "Nazwa pliku", "Ścieżka", "Błąd", "Rozmiar"]
        )
        self.files_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        self.files_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.Stretch
        )

        # Wypełnij tabelę danymi
        self.files_table.setRowCount(len(self.results["invalid_files"]))
        for i, file_info in enumerate(self.results["invalid_files"]):
            # Checkbox
            checkbox = QTableWidgetItem()
            checkbox.setFlags(
                Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled
            )
            checkbox.setCheckState(Qt.CheckState.Unchecked)
            self.files_table.setItem(i, 0, checkbox)

            # Nazwa pliku
            self.files_table.setItem(
                i, 1, QTableWidgetItem(os.path.basename(file_info["path"]))
            )

            # Ścieżka
            self.files_table.setItem(i, 2, QTableWidgetItem(file_info["path"]))

            # Błąd
            self.files_table.setItem(i, 3, QTableWidgetItem(file_info["error"]))

            # Rozmiar
            try:
                size = os.path.getsize(file_info["path"])
                size_str = self._format_size(size)
            except:
                size_str = "N/A"
            self.files_table.setItem(i, 4, QTableWidgetItem(size_str))

        layout.addWidget(self.files_table)

        # Przyciski akcji
        buttons_layout = QHBoxLayout()

        select_all_btn = QPushButton("Zaznacz wszystkie")
        select_all_btn.clicked.connect(self._select_all)
        buttons_layout.addWidget(select_all_btn)

        deselect_all_btn = QPushButton("Odznacz wszystkie")
        deselect_all_btn.clicked.connect(self._deselect_all)
        buttons_layout.addWidget(deselect_all_btn)

        buttons_layout.addStretch()

        move_selected_btn = QPushButton("Przenieś zaznaczone")
        move_selected_btn.clicked.connect(self._move_selected)
        buttons_layout.addWidget(move_selected_btn)

        close_btn = QPushButton("Zamknij")
        close_btn.clicked.connect(self.accept)
        buttons_layout.addWidget(close_btn)

        layout.addLayout(buttons_layout)

    def _format_size(self, size):
        """Formatuje rozmiar pliku do czytelnej postaci."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

    def _select_all(self):
        """Zaznacza wszystkie pliki w tabeli."""
        for row in range(self.files_table.rowCount()):
            self.files_table.item(row, 0).setCheckState(Qt.CheckState.Checked)

    def _deselect_all(self):
        """Odznacza wszystkie pliki w tabeli."""
        for row in range(self.files_table.rowCount()):
            self.files_table.item(row, 0).setCheckState(Qt.CheckState.Unchecked)

    def _move_selected(self):
        """Przenosi zaznaczone pliki do wybranego katalogu."""
        # Zbierz zaznaczone pliki
        selected_files = []
        for row in range(self.files_table.rowCount()):
            if self.files_table.item(row, 0).checkState() == Qt.CheckState.Checked:
                file_path = self.files_table.item(row, 2).text()
                selected_files.append(file_path)

        if not selected_files:
            QMessageBox.warning(self, "Ostrzeżenie", "Nie zaznaczono żadnych plików.")
            return

        # Wybierz katalog docelowy
        target_dir = QFileDialog.getExistingDirectory(
            self,
            "Wybierz katalog docelowy",
            "",
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks,
        )

        if not target_dir:
            return

        # Przenieś pliki
        moved_files = 0
        errors = []

        for file_path in selected_files:
            try:
                file_name = os.path.basename(file_path)
                target_path = os.path.join(target_dir, file_name)

                # Jeśli plik o takiej nazwie już istnieje, dodaj numer
                counter = 1
                while os.path.exists(target_path):
                    base_name, ext = os.path.splitext(file_name)
                    target_path = os.path.join(
                        target_dir, f"{base_name}_{counter}{ext}"
                    )
                    counter += 1

                shutil.move(file_path, target_path)
                moved_files += 1

            except Exception as e:
                errors.append(f"{file_name}: {str(e)}")

        # Wyświetl podsumowanie
        if errors:
            error_msg = "\n".join(errors)
            QMessageBox.warning(
                self,
                "Błędy podczas przenoszenia",
                f"Przeniesiono {moved_files} z {len(selected_files)} plików.\n\nBłędy:\n{error_msg}",
            )
        else:
            QMessageBox.information(
                self, "Sukces", f"Pomyślnie przeniesiono {moved_files} plików."
            )

        # Odśwież tabelę
        self.accept()
