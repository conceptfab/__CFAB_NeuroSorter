import sys
from pathlib import Path

# Dodaj ścieżkę główną projektu do sys.path, aby umożliwić importy względne
# przy bezpośrednim uruchamianiu skryptu.
# Zakładamy, że ten plik znajduje się w app/utils/file_tools/
# więc musimy cofnąć się o 3 poziomy, aby dotrzeć do korzenia projektu.
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
import logging
import random
import shutil
from datetime import datetime

from PyQt6.QtCore import (
    QObject,
    QRunnable,
    Qt,
    QThread,
    QThreadPool,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtGui import QColor, QFont, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.utils.file_tools.config import config

# Konfiguracja logowania
logger = logging.getLogger("DataSplitter")


def setup_logger():
    try:
        log_file = (
            Path("logs")
            / f"data_splitter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        log_file.parent.mkdir(exist_ok=True)
        logger.info(f"Tworzenie pliku logów: {log_file}")

        logger.setLevel(logging.DEBUG)

        # Handler pliku
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Handler konsoli
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Format
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Dodaj handlery
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        logger.info("Logger został poprawnie skonfigurowany")
        return logger
    except Exception as e:
        print(f"Błąd podczas konfiguracji loggera: {e}")
        raise


setup_logger()
logger.info("Inicjalizacja aplikacji DataSplitter")

# --- Konfiguracja ---
try:
    TRAIN_FOLDER_NAME = config.get("folders", "train_folder_name")
    VALID_FOLDER_NAME = config.get("folders", "valid_folder_name")
    DEFAULT_TRAIN_SPLIT_PERCENT = config.get("defaults", "train_split_percent")
    DEFAULT_FILES_PER_CATEGORY = config.get("defaults", "files_per_category")
    ALLOWED_IMAGE_EXTENSIONS = tuple(
        config.get("extensions", "allowed_image_extensions")
    )
    logger.info("Konfiguracja została wczytana pomyślnie")
except Exception as e:
    logger.error(f"Błąd podczas wczytywania konfiguracji: {e}")
    raise

# --- Style ---
try:
    PRIMARY_COLOR = config.get("ui", "colors")["primary_color"]
    BACKGROUND = config.get("ui", "colors")["background"]
    SURFACE = config.get("ui", "colors")["surface"]
    BORDER_COLOR = config.get("ui", "colors")["border_color"]
    TEXT_COLOR = config.get("ui", "colors")["text_color"]
    HIGHLIGHT_COLOR = config.get("ui", "colors")["highlight_color"]
    logger.info("Style UI zostały wczytane pomyślnie")
except Exception as e:
    logger.error(f"Błąd podczas wczytywania stylów UI: {e}")
    raise


class DataSplitterError(Exception):
    """Bazowa klasa dla wyjątków w aplikacji Data Splitter"""

    pass


class ConfigurationError(DataSplitterError):
    """Błędy związane z konfiguracją (ścieżki, wartości, etc.)"""

    pass


class ProcessingError(DataSplitterError):
    """Błędy występujące podczas przetwarzania plików"""

    pass


class FileSplitter:
    """Klasa odpowiedzialna za logikę podziału plików."""

    def __init__(
        self,
        input_dir,
        output_dir,
        split_mode,
        split_value,
        use_validation=True,
        selected_categories=None,
    ):
        logger.info("Inicjalizacja FileSplitter")
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.split_mode = split_mode
        self.split_value = split_value
        self.use_validation = use_validation
        self.selected_categories = selected_categories if selected_categories else []
        self.stats = {"train": {}, "valid": {}}
        self.json_report = {}
        self.min_files_in_selection_for_report = 0
        self.folders_with_min_for_report = []
        logger.info(
            f"FileSplitter zainicjalizowany: tryb={split_mode}, wartość={split_value}, walidacja={use_validation}"
        )

    def get_min_files_in_selected_categories_for_processing(self):
        if not self.input_dir or not self.selected_categories:
            return 0, []

        min_files_val = float("inf")
        folders_with_min = []
        root_path = Path(self.input_dir)

        for category_name in self.selected_categories:
            category_dir = root_path / category_name
            if category_dir.is_dir():
                count = sum(
                    1
                    for f in category_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
                )
                if 0 < count < min_files_val:
                    min_files_val = count
                    folders_with_min = [category_name]
                elif count == min_files_val and min_files_val != float("inf"):
                    folders_with_min.append(category_name)

        if min_files_val == float("inf"):
            return 0, []
        return min_files_val, folders_with_min

    def process_files(self, progress_callback=None, cancel_check=None):
        """
        Główna metoda procesująca pliki z obsługą postępu

        :param progress_callback: funkcja wywoływana do aktualizacji postępu (value, message)
        :param cancel_check: funkcja sprawdzająca czy operacja została anulowana
        :return: (status_string, error_message)
        """
        try:
            logger.info("Rozpoczynam przetwarzanie plików")
            if progress_callback:
                progress_callback(0, "Rozpoczynanie przetwarzania...")

            if not self.input_dir.is_dir():
                logger.error(f"Folder wejściowy nie istnieje: {self.input_dir}")
                raise ValueError(f"Folder wejściowy nie istnieje: {self.input_dir}")
            if not self.output_dir.exists():
                logger.info(f"Tworzenie folderu wyjściowego: {self.output_dir}")
                self.output_dir.mkdir(parents=True, exist_ok=True)
            elif not self.output_dir.is_dir():
                logger.error(f"Ścieżka wyjściowa nie jest folderem: {self.output_dir}")
                raise ValueError(
                    f"Ścieżka wyjściowa istnieje, ale nie jest folderem: {self.output_dir}"
                )

            train_base_path = self.output_dir / TRAIN_FOLDER_NAME
            valid_base_path = (
                self.output_dir / VALID_FOLDER_NAME if self.use_validation else None
            )

            if train_base_path.exists():
                logger.info(
                    f"Usuwanie istniejącego folderu treningowego: {train_base_path}"
                )
                shutil.rmtree(train_base_path)
            if valid_base_path and valid_base_path.exists():
                logger.info(
                    f"Usuwanie istniejącego folderu walidacyjnego: {valid_base_path}"
                )
                shutil.rmtree(valid_base_path)

            train_base_path.mkdir(parents=True, exist_ok=True)
            if valid_base_path:
                valid_base_path.mkdir(parents=True, exist_ok=True)

            subfolders_to_process = []
            total_files_to_process = 0
            if progress_callback:
                progress_callback(5, "Skanowanie wybranych folderów wejściowych...")

            if not self.selected_categories:
                logger.error("Nie wybrano żadnych kategorii do przetworzenia")
                raise ValueError("Nie wybrano żadnych kategorii do przetworzenia.")

            for category_name in self.selected_categories:
                category_dir = self.input_dir / category_name
                if category_dir.is_dir():
                    relative_path = category_dir.relative_to(self.input_dir)
                    files_in_subdir = [
                        f
                        for f in category_dir.iterdir()
                        if f.is_file() and f.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
                    ]
                    if files_in_subdir:
                        subfolders_to_process.append((relative_path, files_in_subdir))
                        total_files_to_process += len(files_in_subdir)
                        self.stats["train"][str(relative_path)] = 0
                        if self.use_validation:
                            self.stats["valid"][str(relative_path)] = 0
                        self.json_report[str(relative_path)] = {
                            "train": [],
                            "valid": [],
                        }
                        logger.info(
                            f"Znaleziono {len(files_in_subdir)} plików w kategorii {category_name}"
                        )
                    else:
                        logger.warning(
                            f"Kategoria '{category_name}' jest pusta lub nie zawiera obrazów"
                        )
                        if progress_callback:
                            progress_callback(
                                0,
                                f"Info: Wybrana kategoria '{category_name}' jest pusta lub nie zawiera obrazów.",
                            )

            if not subfolders_to_process:
                logger.error("Nie znaleziono plików w wybranych kategoriach")
                raise ValueError(
                    "Nie znaleziono żadnych plików obrazów w wybranych i niepustych podfolderach."
                )

            logger.info(
                f"Znaleziono {total_files_to_process} plików w {len(subfolders_to_process)} kategoriach"
            )
            if progress_callback:
                progress_callback(
                    10,
                    f"Znaleziono {total_files_to_process} plików w {len(subfolders_to_process)} wybranych podkategoriach.",
                )

            processed_files_count = 0

            if self.split_mode == "files":
                (
                    self.min_files_in_selection_for_report,
                    self.folders_with_min_for_report,
                ) = self.get_min_files_in_selected_categories_for_processing()
                if (
                    self.min_files_in_selection_for_report == 0
                    and self.use_validation
                    and self.split_value > 0
                ):
                    logger.warning(
                        "W trybie 'Limit plików' nie znaleziono plików w wybranych kategoriach"
                    )
                    if progress_callback:
                        progress_callback(
                            0,
                            "Ostrzeżenie: W trybie 'Limit plików' nie znaleziono plików w wybranych kategoriach, co może wpłynąć na podział walidacyjny.",
                        )

            for relative_path, files in subfolders_to_process:
                if cancel_check and cancel_check():
                    logger.info("Przetwarzanie zostało anulowane")
                    return "Anulowano.", None

                random.shuffle(files)
                num_train, num_valid = 0, 0

                if self.split_mode == "percent":
                    num_train_float = len(files) * self.split_value / 100
                    num_train = int(num_train_float)

                    if self.use_validation:
                        num_valid = len(files) - num_train
                        if (
                            num_valid == 0
                            and self.split_value < 100
                            and len(files) > 0
                            and num_train == len(files)
                        ):
                            if num_train > 0:
                                num_train -= 1
                                num_valid = 1
                    else:
                        num_train = len(files)
                else:  # "files" mode
                    num_train = min(self.split_value, len(files))
                    if self.use_validation:
                        available_for_valid_in_current_cat = len(files) - num_train
                        desired_valid_based_on_min_cat = 0
                        if self.min_files_in_selection_for_report > self.split_value:
                            desired_valid_based_on_min_cat = (
                                self.min_files_in_selection_for_report
                                - self.split_value
                            )

                        num_valid = min(
                            available_for_valid_in_current_cat,
                            desired_valid_based_on_min_cat,
                        )
                        if num_valid < 0:
                            num_valid = 0

                logger.info(
                    f"Przetwarzanie kategorii {relative_path}: {num_train} tren., {num_valid} walid."
                )

                current_train_path = train_base_path / relative_path
                current_valid_path = (
                    valid_base_path / relative_path
                    if valid_base_path and num_valid > 0
                    else None
                )

                current_train_path.mkdir(parents=True, exist_ok=True)
                if current_valid_path:
                    current_valid_path.mkdir(parents=True, exist_ok=True)

                train_files_to_copy = files[:num_train]
                valid_files_to_copy = (
                    files[num_train : num_train + num_valid] if num_valid > 0 else []
                )

                for file_path in train_files_to_copy:
                    if cancel_check and cancel_check():
                        break
                    try:
                        shutil.copy2(file_path, current_train_path / file_path.name)
                        self.stats["train"][str(relative_path)] += 1
                        self.json_report[str(relative_path)]["train"].append(
                            file_path.name
                        )
                        processed_files_count += 1
                    except Exception as e:
                        logger.error(f"Błąd kopiowania {file_path} (trening): {e}")
                        return None, f"Błąd kopiowania {file_path} (trening): {e}"
                if cancel_check and cancel_check():
                    break

                for file_path in valid_files_to_copy:
                    if cancel_check and cancel_check():
                        break
                    try:
                        if current_valid_path:
                            shutil.copy2(file_path, current_valid_path / file_path.name)
                            if self.use_validation:
                                self.stats["valid"][str(relative_path)] += 1
                                self.json_report[str(relative_path)]["valid"].append(
                                    file_path.name
                                )
                            processed_files_count += 1
                    except Exception as e:
                        logger.error(f"Błąd kopiowania {file_path} (walidacja): {e}")
                        return None, f"Błąd kopiowania {file_path} (walidacja): {e}"
                if cancel_check and cancel_check():
                    break

            logger.info("Przetwarzanie zakończone sukcesem")
            return self._generate_report(), None

        except Exception as e:
            logger.error(f"Wystąpił błąd podczas przetwarzania: {e}")
            return None, str(e)

    def _generate_report(self):
        """Generuje raport z przetwarzania"""
        report = ["=== RAPORT KOPIOWANIA ===", ""]
        if self.split_mode == "percent":
            report.append(f"Algorytm: Podział procentowy (dla wybranych kategorii)")
            train_percent = self.split_value
            valid_percent = 100 - train_percent if self.use_validation else 0
            report.append(
                f"Stosunek plików: {train_percent}% trening"
                + (f" / {valid_percent}% walidacja" if self.use_validation else "")
            )
        else:  # "files" mode
            min_files_val = self.min_files_in_selection_for_report
            folders_with_min_names = self.folders_with_min_for_report
            report.append(
                f"Algorytm: Podział z limitem plików (dla wybranych kategorii)"
            )
            if folders_with_min_names:
                report.append(
                    f"Folder(y) z najmniejszą liczbą plików (wśród wybranych): {', '.join(folders_with_min_names)} ({min_files_val} plików)"
                )
            elif self.selected_categories:  # Były wybrane, ale puste lub bez plików
                report.append(
                    "W wybranych kategoriach nie znaleziono plików do ustalenia globalnego limitu walidacji."
                )
            report.append(f"Z każdej wybranej kategorii próbowano wziąć:")
            report.append(
                f"  - {self.split_value} plików do treningu (lub mniej, jeśli kategoria miała mniej)"
            )
            if self.use_validation:
                num_valid_expected = 0
                if min_files_val > self.split_value:  # min_files_val to min z wybranych
                    num_valid_expected = min_files_val - self.split_value
                report.append(
                    f"  - do {num_valid_expected} plików do walidacji (lub mniej, jeśli było mniej dostępnych)"
                )
            else:
                report.append(f"  - 0 plików do walidacji (opcja wyłączona)")
        report.append("")

        for category in sorted(self.stats["train"].keys()):
            train_count = self.stats["train"][category]
            report.append(f"{category}")
            report.append(f"├── {TRAIN_FOLDER_NAME}: {train_count} plików")
            if self.use_validation and category in self.stats["valid"]:
                valid_count = self.stats["valid"][category]
                report.append(f"└── {VALID_FOLDER_NAME}: {valid_count} plików")
            else:
                report.append(f"└── {VALID_FOLDER_NAME}: 0 plików")
            report.append("")

        total_train = sum(self.stats["train"].values())
        total_valid = (
            sum(self.stats["valid"].values())
            if self.use_validation and self.stats.get("valid")
            else 0
        )
        report.append("=== PODSUMOWANIE OGÓLNE ===")
        report.append(f"Łącznie skopiowano: {total_train + total_valid} plików")
        report.append(f"  - Trening: {total_train} plików")
        report.append(f"  - Walidacja: {total_valid} plików")
        return "\n".join(report)


class Worker(QThread):
    """Wątek do przetwarzania danych w tle"""

    progress_updated = pyqtSignal(int, str)
    finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        input_dir,
        output_dir,
        split_mode,
        split_value,
        use_validation=True,
        selected_categories=None,
    ):
        super().__init__()
        logger.info("Inicjalizacja wątku Worker")
        self.splitter = FileSplitter(
            input_dir,
            output_dir,
            split_mode,
            split_value,
            use_validation,
            selected_categories,
        )
        self.is_cancelled = False

    def run(self):
        try:
            logger.info("Rozpoczynam przetwarzanie w wątku Worker")
            self.progress_updated.emit(0, "Rozpoczynanie przetwarzania...")

            result, error = self.splitter.process_files(
                progress_callback=self.progress_updated.emit,
                cancel_check=lambda: self.is_cancelled,
            )

            if self.is_cancelled:
                logger.info("Przetwarzanie zostało anulowane")
                self.finished.emit("Anulowano.")
            elif error:
                logger.error(f"Wystąpił błąd podczas przetwarzania: {error}")
                self.error_occurred.emit(f"Błąd: {error}")
                self.finished.emit(f"Błąd: {error}")
            else:
                logger.info("Przetwarzanie zakończone sukcesem")
                self.finished.emit(result)

        except ConfigurationError as ce:
            logger.error(f"Błąd konfiguracji: {ce}")
            self.error_occurred.emit(f"Błąd konfiguracji: {ce}")
            self.finished.emit(f"Błąd: {ce}")
        except ProcessingError as pe:
            logger.error(f"Błąd przetwarzania: {pe}")
            self.error_occurred.emit(f"Błąd przetwarzania: {pe}")
            self.finished.emit(f"Błąd: {pe}")
        except Exception as e:
            logger.error(f"Niespodziewany błąd: {e}")
            self.error_occurred.emit(f"Niespodziewany błąd: {e}")
            self.finished.emit(f"Niespodziewany błąd: {e}")

    def cancel(self):
        logger.info("Anulowanie przetwarzania")
        self.is_cancelled = True
        self.progress_updated.emit(0, "Anulowanie...")


# Sygnały dla skanera folderów
class ScannerSignals(QObject):
    """Sygnały emitowane przez skaner katalogów"""

    finished = pyqtSignal(dict)  # Przekazuje słownik z danymi skanowania
    progress = pyqtSignal(int, str)  # Postęp (procent, wiadomość)
    error = pyqtSignal(str)  # Komunikat błędu


class FolderScanner(QRunnable):
    """Zadanie skanowania katalogów w tle używające QRunnable"""

    def __init__(self, root_path, allowed_extensions):
        super().__init__()
        logger.info(f"Inicjalizacja FolderScanner dla ścieżki: {root_path}")
        self.root_path = Path(root_path)
        self.allowed_extensions = allowed_extensions
        self.signals = ScannerSignals()

    @pyqtSlot()
    def run(self):
        """Wykonuje skanowanie katalogów"""
        try:
            logger.info("Rozpoczynam skanowanie katalogów")
            result = {
                "folder_counts": {},
                "min_files_count": float("inf"),
                "min_files_folder": None,
                "total_files": 0,
            }

            # Określ całkowitą liczbę folderów
            folders = [f for f in self.root_path.iterdir() if f.is_dir()]
            total_folders = len(folders)
            logger.info(f"Znaleziono {total_folders} folderów do przeskanowania")

            for i, category_dir in enumerate(folders):
                # Aktualizuj postęp
                progress_percent = (
                    int((i / total_folders) * 100) if total_folders > 0 else 0
                )
                self.signals.progress.emit(
                    progress_percent,
                    f"Skanowanie: {category_dir.name} ({i+1}/{total_folders})",
                )

                # Policz pliki w kategorii
                file_count = sum(
                    1
                    for f in category_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in self.allowed_extensions
                )

                result["folder_counts"][category_dir.name] = file_count
                result["total_files"] += file_count

                logger.info(
                    f"Kategoria {category_dir.name}: znaleziono {file_count} plików"
                )

                # Aktualizuj minimum, jeśli kategoria ma pliki
                if 0 < file_count < result["min_files_count"]:
                    result["min_files_count"] = file_count
                    result["min_files_folder"] = category_dir.name
                    logger.info(
                        f"Nowa minimalna liczba plików: {file_count} w kategorii {category_dir.name}"
                    )

            # Zakończ, jeśli nie znaleziono minimum
            if result["min_files_count"] == float("inf"):
                result["min_files_count"] = 0
                result["min_files_folder"] = None
                logger.warning("Nie znaleziono żadnych plików w kategoriach")

            logger.info(
                f"Skanowanie zakończone: {result['total_files']} plików w {len(result['folder_counts'])} kategoriach"
            )
            self.signals.finished.emit(result)

        except Exception as e:
            logger.error(f"Błąd podczas skanowania katalogów: {e}")
            self.signals.error.emit(str(e))


# --- Główna klasa aplikacji GUI ---
class DataSplitterApp(QWidget):
    def __init__(self):
        super().__init__()
        logger.info("Inicjalizacja DataSplitterApp")
        self.input_dir = ""
        self.output_dir = ""
        self.processing_thread = None
        self.files_list = []

        # Inicjalizacja puli wątków
        self.threadpool = QThreadPool()
        logger.info(
            f"Używam puli wątków z maksymalnie {self.threadpool.maxThreadCount()} wątkami"
        )

        # Inicjalizacja interfejsu
        icon_path = Path("resources/img/icon.png")
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
            logger.info(f"Wczytano ikonę aplikacji: {icon_path}")
        else:
            logger.warning(f"Nie znaleziono pliku ikony: {icon_path}")

        self.initUI()
        self._apply_material_theme()
        self.update_files_limit_and_validation_based_on_selection()
        logger.info("Inicjalizacja DataSplitterApp zakończona")

    def _apply_material_theme(self):
        self.setStyleSheet(
            f"""
            QWidget {{ background-color: {BACKGROUND}; color: {TEXT_COLOR}; }}
            QPushButton {{ background-color: {SURFACE}; border: 1px solid {BORDER_COLOR}; border-radius: 2px; padding: 4px 12px; min-height: 24px; max-height: 24px; }}
            QPushButton:hover {{ background-color: #2A2D2E; }} QPushButton:pressed {{ background-color: #3E3E40; }}
            QLineEdit {{ background-color: #1C1C1C; border: 1px solid {BORDER_COLOR}; border-radius: 2px; padding: 2px; }}
            QSlider::groove:horizontal {{ border: 1px solid {BORDER_COLOR}; height: 8px; background: {SURFACE}; margin: 2px 0; border-radius: 4px; }}
            QSlider::handle:horizontal {{ background: {PRIMARY_COLOR}; border: 1px solid {PRIMARY_COLOR}; width: 18px; margin: -2px 0; border-radius: 9px; }}
            QSlider::sub-page:horizontal {{ background-color: {PRIMARY_COLOR}; border-radius: 4px; }}
            QProgressBar {{ border: 1px solid {BORDER_COLOR}; background-color: {SURFACE}; text-align: center; color: {TEXT_COLOR}; }}
            QProgressBar::chunk {{ background-color: {PRIMARY_COLOR}; }}
            QTextEdit {{ background-color: #1C1C1C; border: 1px solid {BORDER_COLOR}; border-radius: 2px; padding: 2px; font-family: 'Consolas', 'Courier New', monospace; font-size: 10px; }}
            QLabel {{ color: {TEXT_COLOR}; }}
            QTreeWidget {{ border: 1px solid {BORDER_COLOR}; background-color: #1C1C1C; }}
            QTreeWidget::item {{ padding: 3px; }}
            QTreeWidget::item:selected {{ background-color: {PRIMARY_COLOR}; color: white; }}
            QListWidget {{ border: 1px solid {BORDER_COLOR}; background-color: #1C1C1C; }}
            QComboBox {{ border: 1px solid {BORDER_COLOR}; border-radius: 2px; padding: 2px; background-color: {SURFACE}; }}
            QSpinBox {{ border: 1px solid {BORDER_COLOR}; border-radius: 2px; padding: 2px; background-color: {SURFACE}; }}
        """
        )

    def initUI(self):
        self.setWindowTitle("Przygotowanie Danych AI")
        self.setGeometry(200, 200, 850, 650)  # Trochę szersze
        layout = QVBoxLayout()

        folder_layout = QVBoxLayout()
        in_folder_layout = QHBoxLayout()
        in_label = QLabel("Folder z danymi źródłowymi:")
        self.in_path_edit = QLineEdit()
        self.in_path_edit.setReadOnly(True)
        in_button = QPushButton("Wybierz...")
        in_button.clicked.connect(self.select_input_folder)
        in_folder_layout.addWidget(in_label)
        in_folder_layout.addWidget(self.in_path_edit)
        in_folder_layout.addWidget(in_button)
        folder_layout.addLayout(in_folder_layout)

        out_folder_layout = QHBoxLayout()
        out_label = QLabel("Folder docelowy dla podziału:")
        self.out_path_edit = QLineEdit()
        self.out_path_edit.setReadOnly(True)
        out_button = QPushButton("Wybierz...")
        out_button.clicked.connect(self.select_output_folder)
        out_folder_layout.addWidget(out_label)
        out_folder_layout.addWidget(self.out_path_edit)
        out_folder_layout.addWidget(out_button)
        folder_layout.addLayout(out_folder_layout)
        layout.addLayout(folder_layout)

        self.tabs = QTabWidget()
        self.folder_tree = QTreeWidget()
        self.folder_tree.setHeaderLabels(
            ["Struktura folderów (zaznacz kategorie do przetworzenia)"]
        )
        self.folder_tree.setColumnCount(1)

        # Dodaję przyciski do zaznaczania/odznaczania
        folder_buttons_layout = QHBoxLayout()
        self.select_all_button = QPushButton("Zaznacz wszystkie")
        self.deselect_all_button = QPushButton("Odznacz wszystkie")
        self.select_all_button.clicked.connect(self.select_all_folders)
        self.deselect_all_button.clicked.connect(self.deselect_all_folders)
        folder_buttons_layout.addWidget(self.select_all_button)
        folder_buttons_layout.addWidget(self.deselect_all_button)

        folder_layout = QVBoxLayout()
        folder_layout.addLayout(folder_buttons_layout)
        folder_layout.addWidget(self.folder_tree)

        folder_widget = QWidget()
        folder_widget.setLayout(folder_layout)
        self.tabs.addTab(folder_widget, "Wybór kategorii")
        self.files_list_widget = QListWidget()
        self.tabs.addTab(self.files_list_widget, "Lista wszystkich plików")
        layout.addWidget(self.tabs)

        split_layout = QVBoxLayout()
        mode_layout = QHBoxLayout()
        mode_label = QLabel("Tryb podziału:")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Podział procentowy", "Limit plików na kategorię"])
        self.mode_combo.currentIndexChanged.connect(self.update_split_mode)
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo)
        split_layout.addLayout(mode_layout)

        self.percent_layout = QHBoxLayout()
        percent_label = QLabel("Podział Treningowe / Walidacyjne:")
        self.split_slider = QSlider(Qt.Orientation.Horizontal)
        self.split_slider.setMinimum(1)
        self.split_slider.setMaximum(99)
        self.split_slider.setValue(DEFAULT_TRAIN_SPLIT_PERCENT)
        self.split_slider.setTickInterval(10)
        self.split_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.split_value_label = QLabel(
            f"{DEFAULT_TRAIN_SPLIT_PERCENT}% / {100 - DEFAULT_TRAIN_SPLIT_PERCENT}%"
        )
        self.split_value_label.setMinimumWidth(80)
        self.split_slider.valueChanged.connect(self.update_split_label)
        self.percent_layout.addWidget(percent_label)
        self.percent_layout.addWidget(self.split_slider)
        self.percent_layout.addWidget(self.split_value_label)
        split_layout.addLayout(self.percent_layout)

        self.files_layout = QHBoxLayout()
        files_label = QLabel(
            "Liczba plików treningowych na kategorię:"
        )  # Zmieniona etykieta
        self.files_spin = QSpinBox()
        self.files_spin.setMinimum(1)
        self.files_spin.setMaximum(10000)
        self.files_spin.setValue(DEFAULT_FILES_PER_CATEGORY)
        self.files_spin.valueChanged.connect(
            self.update_files_limit_and_validation_based_on_selection
        )  # ZMIANA
        self.files_layout.addWidget(files_label)
        self.files_layout.addWidget(self.files_spin)
        split_layout.addLayout(self.files_layout)

        validation_layout = QHBoxLayout()
        self.validation_check = QCheckBox("Utwórz folder walidacyjny")
        self.validation_check.setChecked(True)
        self.validation_check.stateChanged.connect(
            self.update_files_limit_and_validation_based_on_selection
        )  # ZMIANA
        self.validation_label = QLabel("")  # Etykieta informacyjna obok checkboxa
        validation_layout.addWidget(self.validation_check)
        validation_layout.addWidget(self.validation_label)
        split_layout.addLayout(validation_layout)
        layout.addLayout(split_layout)

        control_layout = QHBoxLayout()
        self.start_button = QPushButton("Rozpocznij przetwarzanie")
        self.start_button.clicked.connect(self.start_processing)
        self.cancel_button = QPushButton("Anuluj")
        self.cancel_button.clicked.connect(self.cancel_processing)
        self.cancel_button.setEnabled(False)
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.cancel_button)
        layout.addLayout(control_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        log_label = QLabel("Log:")
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumHeight(100)
        layout.addWidget(log_label)
        layout.addWidget(self.log_edit)

        self.setLayout(layout)
        self.show()
        self.update_split_mode(0)  # Inicjalizacja widoczności na podstawie trybu

    def update_split_mode(self, index):
        is_percent_mode = index == 0
        for i in range(self.percent_layout.count()):
            widget = self.percent_layout.itemAt(i).widget()
            if widget:
                widget.setVisible(is_percent_mode)
        for i in range(self.files_layout.count()):
            widget = self.files_layout.itemAt(i).widget()
            if widget:
                widget.setVisible(not is_percent_mode)
        self.update_files_limit_and_validation_based_on_selection()

    def update_split_label(self):
        train_percent = self.split_slider.value()
        valid_percent = 100 - train_percent
        self.split_value_label.setText(f"{train_percent}% / {valid_percent}%")
        if (
            self.mode_combo.currentIndex() == 0
        ):  # Tylko jeśli tryb procentowy jest aktywny
            self.update_files_limit_and_validation_based_on_selection()

    def update_folder_tree(self):
        """Aktualizuje drzewo folderów z wykorzystaniem QThreadPool"""
        self.folder_tree.clear()
        if not self.input_dir:
            self.update_files_limit_and_validation_based_on_selection()
            return

        root_path = Path(self.input_dir)
        tree_root_item = QTreeWidgetItem(self.folder_tree, [root_path.name])
        self.folder_tree.addTopLevelItem(tree_root_item)
        tree_root_item.setExpanded(True)

        # Odłącz sygnał zmiany elementu
        try:
            self.folder_tree.itemChanged.disconnect(self.on_folder_tree_item_changed)
        except TypeError:
            pass

        # Uruchom skaner folderów w tle
        scanner = FolderScanner(root_path, ALLOWED_IMAGE_EXTENSIONS)

        # Funkcje obsługi sygnałów
        def on_scan_progress(percent, message):
            self.log_message(message)
            self.progress_bar.setValue(percent)
            QApplication.processEvents()

        def on_scan_finished(result):
            folder_counts = result["folder_counts"]
            min_files_folder = result["min_files_folder"]

            # Dodaj foldery do drzewa
            for category_dir in root_path.iterdir():
                if category_dir.is_dir():
                    file_count = folder_counts.get(category_dir.name, 0)
                    display_text = f"{category_dir.name} ({file_count} plików)"
                    item = QTreeWidgetItem(tree_root_item, [display_text])
                    item.setData(0, Qt.ItemDataRole.UserRole, category_dir.name)
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(0, Qt.CheckState.Checked)

                    # Wyróżnij folder z najmniejszą liczbą plików
                    if category_dir.name == min_files_folder and file_count > 0:
                        item.setForeground(0, QColor(HIGHLIGHT_COLOR))

                    QApplication.processEvents()  # Zapobiega blokowaniu UI

            # Podłącz ponownie sygnał i aktualizuj kontrolki
            self.folder_tree.itemChanged.connect(self.on_folder_tree_item_changed)
            self.update_files_limit_and_validation_based_on_selection()
            self.progress_bar.setValue(0)  # Zresetuj pasek postępu
            self.log_message(
                f"Znaleziono {result['total_files']} plików w {len(folder_counts)} kategoriach."
            )

        def on_scan_error(error_message):
            self.log_message(f"Błąd podczas skanowania: {error_message}", logging.ERROR)
            self.progress_bar.setValue(0)
            # Podłącz ponownie sygnał
            self.folder_tree.itemChanged.connect(self.on_folder_tree_item_changed)

        # Podłącz sygnały i rozpocznij skanowanie
        scanner.signals.progress.connect(on_scan_progress)
        scanner.signals.finished.connect(on_scan_finished)
        scanner.signals.error.connect(on_scan_error)

        self.threadpool.start(scanner)

    def on_folder_tree_item_changed(self, item, column):
        if (
            column == 0
            and item.parent()
            and item.parent() == self.folder_tree.topLevelItem(0)
        ):
            self.update_files_limit_and_validation_based_on_selection()

    def get_selected_categories_names(self):
        selected = []
        if not self.input_dir or not self.folder_tree.topLevelItem(0):
            return selected
        root_item = self.folder_tree.topLevelItem(0)
        for i in range(root_item.childCount()):
            child_item = root_item.child(i)
            if child_item.checkState(0) == Qt.CheckState.Checked:
                category_name = child_item.data(0, Qt.ItemDataRole.UserRole)
                if category_name:
                    selected.append(category_name)
        return selected

    def get_min_files_in_selected_categories(self, selected_category_names):
        if not self.input_dir or not selected_category_names:
            return (
                DEFAULT_FILES_PER_CATEGORY,
                [],
            )  # Zwraca wartość domyślną, jeśli nic nie wybrano

        min_val = float("inf")
        folders_with_min = []
        root = Path(self.input_dir)
        for name in selected_category_names:
            cat_dir = root / name
            if cat_dir.is_dir():
                count = sum(
                    1
                    for f in cat_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
                )
                if 0 < count < min_val:
                    min_val = count
                    folders_with_min = [name]
                elif count == min_val and min_val != float("inf"):
                    folders_with_min.append(name)

        return (min_val, folders_with_min) if min_val != float("inf") else (0, [])

    def update_files_limit_and_validation_based_on_selection(self):
        """Aktualizuje stan kontrolek na podstawie wybranych kategorii"""
        selected_cats = self.get_selected_categories_names()
        min_files_in_selection, folders_with_min = (
            self.get_min_files_in_selected_categories(selected_cats)
        )

        # Aktualizuj stan SpinBoxa dla plików
        self._update_files_spin_state(selected_cats, min_files_in_selection)

        # Aktualizuj stan checkboxa walidacji
        self._update_validation_checkbox_state(selected_cats, min_files_in_selection)

    def _update_files_spin_state(self, selected_cats, min_files_in_selection):
        """Aktualizuje stan kontrolki files_spin"""
        if not selected_cats:
            self.files_spin.setEnabled(False)
            self.files_spin.setValue(1)
            self.log_message("Wybierz kategorie, aby dostosować opcje podziału.")
        else:
            self.files_spin.setEnabled(self.mode_combo.currentIndex() == 1)
            if min_files_in_selection > 0:
                self.files_spin.setMaximum(min_files_in_selection)
                if self.files_spin.value() > min_files_in_selection:
                    self.files_spin.setValue(min_files_in_selection)
            else:
                self.files_spin.setMaximum(1)
                self.files_spin.setValue(1)

    def _update_validation_checkbox_state(self, selected_cats, min_files_in_selection):
        """Aktualizuje stan checkboxa walidacji i jego etykiety"""
        current_files_limit = self.files_spin.value()

        if self.mode_combo.currentIndex() == 0:  # Tryb procentowy
            self._update_validation_for_percent_mode(selected_cats)
        else:  # Tryb limitu plików
            self._update_validation_for_files_mode(
                selected_cats, min_files_in_selection, current_files_limit
            )

    def _update_validation_for_percent_mode(self, selected_cats):
        """Aktualizuje stan walidacji dla trybu procentowego"""
        can_create_validation = self.split_slider.value() < 100
        self.validation_check.setEnabled(can_create_validation and bool(selected_cats))

        if not can_create_validation:
            self.validation_check.setChecked(False)
            self.validation_label.setText("(100% tren.)")
        else:
            self.validation_label.setText(
                f"({100 - self.split_slider.value()}% walid.)"
            )

    def _update_validation_for_files_mode(
        self, selected_cats, min_files_in_selection, current_files_limit
    ):
        """Aktualizuje stan walidacji dla trybu limitu plików"""
        can_create_validation = (
            current_files_limit < min_files_in_selection
            and min_files_in_selection > 0
            and bool(selected_cats)
        )
        self.validation_check.setEnabled(can_create_validation)

        if not can_create_validation:
            self.validation_check.setChecked(False)
            self._set_validation_label_for_files_mode(
                selected_cats, min_files_in_selection
            )
        else:
            max_valid = min_files_in_selection - current_files_limit
            self.validation_label.setText(f"(do {max_valid} walid.)")

    def _set_validation_label_for_files_mode(
        self, selected_cats, min_files_in_selection
    ):
        """Ustawia tekst etykiety walidacji dla trybu plików gdy walidacja jest niedostępna"""
        if not selected_cats:
            self.validation_label.setText("(wybierz kat.)")
        elif min_files_in_selection == 0:
            self.validation_label.setText("(brak plików)")
        else:
            self.validation_label.setText("(limit >= min.)")

    def update_files_list(self):
        """Aktualizuje listę wszystkich plików w wybranych katalogach"""
        self.files_list_widget.clear()
        self.files_list = []

        if not self.input_dir:
            return

        # Dodaj element informujący o skanowaniu
        self.files_list_widget.addItem("Trwa skanowanie plików...")
        QApplication.processEvents()

        # Skaner plików w osobnym wątku
        class FileScanner(QThread):
            finished = pyqtSignal(list)
            progress = pyqtSignal(int)

            def __init__(self, root_path, allowed_extensions):
                super().__init__()
                self.root_path = root_path
                self.allowed_extensions = allowed_extensions
                self.files = []

            def run(self):
                for item in self.root_path.rglob("*"):
                    if (
                        item.is_file()
                        and item.suffix.lower() in self.allowed_extensions
                    ):
                        relative_path = item.relative_to(self.root_path)
                        self.files.append((item, str(relative_path)))
                        if len(self.files) % 100 == 0:
                            self.progress.emit(len(self.files))
                self.finished.emit(self.files)

        # Funkcje zwrotne dla skanera
        def on_scan_progress(count):
            self.files_list_widget.item(0).setText(
                f"Skanowanie w toku... Znaleziono {count} plików"
            )
            QApplication.processEvents()

        def on_scan_complete(files):
            self.files_list_widget.clear()
            self.files_list = [f[0] for f in files]  # Zapisz pełne ścieżki

            # Pokaż tylko pierwsze 1000 plików w widoku
            display_limit = 1000
            for i, (_, file_path_str) in enumerate(files[:display_limit]):
                self.files_list_widget.addItem(file_path_str)

            # Dodaj informację o limicie
            if len(files) > display_limit:
                self.files_list_widget.addItem(
                    f"... oraz {len(files) - display_limit} więcej plików"
                )

            self.log_message(f"Znaleziono łącznie {len(files)} plików.", logging.INFO)

        # Uruchom skaner
        scanner = FileScanner(Path(self.input_dir), ALLOWED_IMAGE_EXTENSIONS)
        scanner.progress.connect(on_scan_progress)
        scanner.finished.connect(on_scan_complete)
        scanner.start()

    def log_message(self, message, level=logging.INFO):
        """Loguje wiadomość w interfejsie i w systemie logowania"""
        self.log_edit.append(message)
        self.log_edit.verticalScrollBar().setValue(
            self.log_edit.verticalScrollBar().maximum()
        )
        QApplication.processEvents()

        # Logowanie do pliku
        if level == logging.DEBUG:
            logger.debug(message)
        elif level == logging.INFO:
            logger.info(message)
        elif level == logging.WARNING:
            logger.warning(message)
        elif level == logging.ERROR:
            logger.error(message)
        elif level == logging.CRITICAL:
            logger.critical(message)

    def select_input_folder(self):
        logger.info("Rozpoczynam wybór folderu źródłowego")
        folder = QFileDialog.getExistingDirectory(self, "Wybierz folder źródłowy")
        if folder:
            self.input_dir = folder
            self.in_path_edit.setText(folder)
            logger.info(f"Wybrano folder źródłowy: {folder}")
            self.update_folder_tree()
            self.update_files_list()
        else:
            logger.info("Anulowano wybór folderu źródłowego")

    def select_output_folder(self):
        logger.info("Rozpoczynam wybór folderu docelowego")
        folder = QFileDialog.getExistingDirectory(self, "Wybierz folder docelowy")
        if folder:
            self.output_dir = folder
            self.out_path_edit.setText(folder)
            logger.info(f"Wybrano folder docelowy: {folder}")
        else:
            logger.info("Anulowano wybór folderu docelowego")

    def update_progress(self, value, message):
        self.progress_bar.setValue(value)
        if message:
            self.log_message(message)

    def processing_finished(self, final_message):
        logger.info(f"Zakończono przetwarzanie: {final_message}")
        self.log_message(f"Status końcowy: {final_message}")
        QApplication.processEvents()

        if "Błąd" in final_message:
            logger.error(f"Wystąpił błąd podczas przetwarzania: {final_message}")
            QMessageBox.warning(
                self, "Błąd", f"Wystąpił błąd:\n{final_message.splitlines()[-1]}"
            )
        elif "Anulowano" not in final_message:
            logger.info("Przetwarzanie zakończone sukcesem")
            dialog = ReportDialog(final_message, self)
            dialog.exec()
        else:
            logger.info("Przetwarzanie zostało anulowane przez użytkownika")
            QMessageBox.information(
                self, "Anulowano", "Przetwarzanie zostało anulowane."
            )

        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.mode_combo.setEnabled(True)
        self.split_slider.setEnabled(self.mode_combo.currentIndex() == 0)
        self.files_spin.setEnabled(self.mode_combo.currentIndex() == 1)
        self.update_files_limit_and_validation_based_on_selection()
        self.processing_thread = None

    def processing_error(self, error_message):
        logger.error(f"Błąd w wątku przetwarzania: {error_message}")
        self.log_message(f"BŁĄD WĄTKU: {error_message}")

    def start_processing(self):
        logger.info("Rozpoczynam przetwarzanie")
        if not self.input_dir or not Path(self.input_dir).is_dir():
            logger.error("Nie wybrano prawidłowego folderu źródłowego")
            QMessageBox.warning(
                self, "Brak folderu", "Wybierz prawidłowy folder źródłowy."
            )
            return
        if not self.output_dir:
            logger.error("Nie wybrano folderu docelowego")
            QMessageBox.warning(self, "Brak folderu", "Wybierz folder docelowy.")
            return

        selected_categories = self.get_selected_categories_names()
        if not selected_categories:
            logger.error("Nie wybrano żadnych kategorii")
            QMessageBox.warning(
                self,
                "Brak kategorii",
                "Wybierz przynajmniej jedną kategorię do przetworzenia.",
            )
            return

        logger.info(f"Wybrane kategorie: {selected_categories}")

        # Sprawdzenie ścieżek
        input_path = Path(self.input_dir)
        output_path = Path(self.output_dir)
        try:
            if input_path == output_path or output_path.is_relative_to(input_path):
                logger.warning(
                    f"Folder docelowy jest taki sam jak źródłowy lub znajduje się wewnątrz niego: {output_path}"
                )
                reply = QMessageBox.question(
                    self,
                    "Potwierdzenie ścieżki",
                    f"Folder docelowy ('{output_path}') jest taki sam jak źródłowy lub znajduje się wewnątrz niego. "
                    f"Spowoduje to utworzenie '{TRAIN_FOLDER_NAME}' i '{VALID_FOLDER_NAME}' w '{output_path}'.\n"
                    "Kontynuować?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if reply == QMessageBox.StandardButton.No:
                    logger.info("Użytkownik anulował operację")
                    return
        except ValueError as e:
            logger.warning(f"Błąd podczas sprawdzania ścieżek: {e}")
            pass

        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.mode_combo.setEnabled(False)
        self.split_slider.setEnabled(False)
        self.files_spin.setEnabled(False)
        self.validation_check.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_edit.clear()

        split_mode = "percent" if self.mode_combo.currentIndex() == 0 else "files"
        split_value = (
            self.split_slider.value()
            if split_mode == "percent"
            else self.files_spin.value()
        )
        use_validation = self.validation_check.isChecked()

        logger.info(
            f"Parametry przetwarzania: tryb={split_mode}, wartość={split_value}, walidacja={use_validation}"
        )

        self.processing_thread = Worker(
            self.input_dir,
            self.output_dir,
            split_mode,
            split_value,
            use_validation,
            selected_categories,
        )
        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.finished.connect(self.processing_finished)
        self.processing_thread.error_occurred.connect(self.processing_error)
        logger.info("Uruchamiam wątek przetwarzania")
        self.processing_thread.start()

    def cancel_processing(self):
        if self.processing_thread and self.processing_thread.isRunning():
            logger.info("Wysyłanie sygnału anulowania...")
            self.processing_thread.cancel()
            self.cancel_button.setEnabled(False)

    def closeEvent(self, event):
        if self.processing_thread and self.processing_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Zamykanie",
                "Przetwarzanie w toku. Anulować i zamknąć?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.cancel_processing()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    def select_all_folders(self):
        """Zaznacza wszystkie foldery w drzewie."""
        if not self.folder_tree.topLevelItem(0):
            return
        root_item = self.folder_tree.topLevelItem(0)
        for i in range(root_item.childCount()):
            child_item = root_item.child(i)
            child_item.setCheckState(0, Qt.CheckState.Checked)
        self.update_files_limit_and_validation_based_on_selection()

    def deselect_all_folders(self):
        """Odznacza wszystkie foldery w drzewie."""
        if not self.folder_tree.topLevelItem(0):
            return
        root_item = self.folder_tree.topLevelItem(0)
        for i in range(root_item.childCount()):
            child_item = root_item.child(i)
            child_item.setCheckState(0, Qt.CheckState.Unchecked)
        self.update_files_limit_and_validation_based_on_selection()


# --- Dialog Raportu (bez zmian, ale dodaję dla kompletności) ---
class ReportDialog(QDialog):
    def __init__(self, report_text, parent=None):
        super().__init__(parent)
        logger.info("Inicjalizacja okna raportu")
        self.setWindowTitle("Raport kopiowania")
        self.setMinimumSize(800, 600)
        layout = QVBoxLayout()
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(QFont("Consolas", 10))
        self.text_edit.setStyleSheet(
            f"""
            QTextEdit {{ background-color: {BACKGROUND}; color: {TEXT_COLOR}; border: 1px solid {BORDER_COLOR}; padding: 10px; }}
        """
        )
        self.text_edit.setHtml(self._format_report_to_html(report_text))
        logger.info("Raport został sformatowany i wyświetlony")
        close_button = QPushButton("Zamknij")
        close_button.setStyleSheet(
            f"""
            QPushButton {{ background-color: {PRIMARY_COLOR}; color: white; border: none; padding: 8px 16px; border-radius: 4px; min-width: 100px; }}
            QPushButton:hover {{ background-color: #1C97EA; }}
        """
        )
        close_button.clicked.connect(self.accept)
        layout.addWidget(self.text_edit)
        layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignCenter)
        self.setLayout(layout)
        logger.info("Okno raportu zostało skonfigurowane")

    def _format_report_to_html(self, report_text):
        """Formatuje raport do HTML"""
        logger.debug("Formatowanie raportu do HTML")
        html = report_text.replace("\n", "<br>")
        html = html.replace("===", "<h3>").replace(" ===", "</h3>")
        return f"<body style='color:{TEXT_COLOR}; background-color:{BACKGROUND};'>{html}</body>"


if __name__ == "__main__":
    try:
        logger.info("Uruchamianie aplikacji")
        app = QApplication(sys.argv)
        logger.info("QApplication utworzona")
        ex = DataSplitterApp()
        logger.info("DataSplitterApp utworzona")
        sys.exit(app.exec())
    except Exception as e:
        logger.critical(f"Krytyczny błąd podczas uruchamiania aplikacji: {e}")
        raise
