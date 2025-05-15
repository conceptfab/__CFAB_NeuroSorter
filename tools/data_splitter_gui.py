import json
import logging
import os
import random
import shutil
import sys
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from PIL import Image, UnidentifiedImageError

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from button_styles import BUTTON_STYLES
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

# For ResolutionScanner's plot
from matplotlib.figure import Figure
from PyQt6.QtCore import (
    QObject,
    QRunnable,
    Qt,
    QThread,
    QThreadPool,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtGui import QAction, QColor, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

# --- Global Logger Setup ---
# Main logger for the combined application
global_logger = logging.getLogger("DataSplitterApp")
global_logger.setLevel(logging.DEBUG)

# Basic console handler for debugging during development
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
global_logger.addHandler(stream_handler)


# --- QtLogHandler (from main_window.py) ---
class QtLogHandler(logging.Handler, QObject):
    log_signal = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__()  # Wywołanie konstruktora klasy bazowej
        QObject.__init__(self, parent)  # Wywołanie konstruktora QObject

    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_signal.emit(msg)
        except Exception:
            self.handleError(record)


# --- BEGIN: data_splitter_gui.py components ---
# Simplified config for data_splitter_gui.py to avoid external file dependency for this example
data_splitter_config_values = {
    "folders": {
        "input": "",
        "output": "",
        "train_folder_name": "train",
        "valid_folder_name": "valid",
    },
    "defaults": {"train_split_percent": 80, "files_per_category": 10},
    "extensions": {
        "allowed_image_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"]
    },
    "ui": {
        "colors": {
            "primary_color": "#2196F3",
            "secondary_color": "#757575",
            "success_color": "#4CAF50",
            "warning_color": "#FFC107",
            "error_color": "#F44336",
            "background": "#FFFFFF",
            "surface": "#F5F5F5",
            "border_color": "#E0E0E0",
            "text_color": "#212121",
            "highlight_color": "#2196F3",
        }
    },
}


class DS_Config:  # Renamed to DS_Config to avoid potential conflicts
    def __init__(self, values):
        self.values = values

    def get(self, section, key=None):
        if key is None:  # implies section is the key, and we're looking in top-level
            return self.values.get(section)
        if section in self.values and key in self.values[section]:
            return self.values[section][key]
        return None


config = DS_Config(data_splitter_config_values)

# DataSplitter specific logger
ds_logger = logging.getLogger("DataSplitterApp")  # Use a specific name
ds_logger.setLevel(logging.DEBUG)
# Note: FileHandler for DataSplitter logs will be set up inside DataSplitterApp if needed
# For UI, it should emit signals or use the global_logger with QtLogHandler

TRAIN_FOLDER_NAME = config.get("folders", "train_folder_name")
VALID_FOLDER_NAME = config.get("folders", "valid_folder_name")
DEFAULT_TRAIN_SPLIT_PERCENT = config.get("defaults", "train_split_percent")
DEFAULT_FILES_PER_CATEGORY = config.get("defaults", "files_per_category")
ALLOWED_IMAGE_EXTENSIONS_DS = tuple(
    config.get("extensions", "allowed_image_extensions")
)  # Renamed

# Style colors (can be overridden by main app's theme)
DS_PRIMARY_COLOR = config.get("ui", "colors")["primary_color"]
DS_BACKGROUND = config.get("ui", "colors")["background"]
DS_SURFACE = config.get("ui", "colors")["surface"]
DS_BORDER_COLOR = config.get("ui", "colors")["border_color"]
DS_TEXT_COLOR = config.get("ui", "colors")["text_color"]
DS_HIGHLIGHT_COLOR = config.get("ui", "colors")["highlight_color"]


class DataSplitterError(Exception):
    pass


class ConfigurationError(DataSplitterError):
    pass


class ProcessingError(DataSplitterError):
    pass


class FileSplitter:
    def __init__(
        self,
        input_dir,
        output_dir,
        split_mode,
        split_value,
        use_validation=True,
        selected_categories=None,
        move_files=False,  # Nowy parametr
    ):
        ds_logger.info("Inicjalizacja FileSplitter")
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
        self.move_files = move_files  # Zapisanie parametru
        ds_logger.info(
            f"FileSplitter zainicjalizowany: tryb={split_mode}, wartość={split_value}, walidacja={use_validation}, przenoszenie={move_files}"
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
                    if f.is_file() and f.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS_DS
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
        try:
            ds_logger.info("Rozpoczynam przetwarzanie plików")
            if progress_callback:
                progress_callback(0, "Rozpoczynanie przetwarzania...")

            if not self.input_dir.is_dir():
                ds_logger.error(f"Folder wejściowy nie istnieje: {self.input_dir}")
                raise ValueError(f"Folder wejściowy nie istnieje: {self.input_dir}")
            if not self.output_dir.exists():
                ds_logger.info(f"Tworzenie folderu wyjściowego: {self.output_dir}")
                self.output_dir.mkdir(parents=True, exist_ok=True)
            elif not self.output_dir.is_dir():
                ds_logger.error(
                    f"Ścieżka wyjściowa nie jest folderem: {self.output_dir}"
                )
                raise ValueError(
                    f"Ścieżka wyjściowa istnieje, ale nie jest folderem: {self.output_dir}"
                )

            train_base_path = self.output_dir / TRAIN_FOLDER_NAME
            valid_base_path = (
                self.output_dir / VALID_FOLDER_NAME if self.use_validation else None
            )

            if train_base_path.exists():
                ds_logger.info(
                    f"Usuwanie istniejącego folderu treningowego: {train_base_path}"
                )
                shutil.rmtree(train_base_path)
            if valid_base_path and valid_base_path.exists():
                ds_logger.info(
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
                ds_logger.error("Nie wybrano żadnych kategorii do przetworzenia")
                raise ValueError("Nie wybrano żadnych kategorii do przetworzenia.")

            for category_name in self.selected_categories:
                category_dir = self.input_dir / category_name
                if category_dir.is_dir():
                    relative_path = category_dir.relative_to(self.input_dir)
                    files_in_subdir = [
                        f
                        for f in category_dir.iterdir()
                        if f.is_file()
                        and f.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS_DS
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
                        ds_logger.info(
                            f"Znaleziono {len(files_in_subdir)} plików w kategorii {category_name}"
                        )
                    else:
                        ds_logger.warning(
                            f"Kategoria '{category_name}' jest pusta lub nie zawiera obrazów"
                        )
                        if progress_callback:
                            progress_callback(
                                0,  # Progress value not updated, but message sent
                                f"Info: Wybrana kategoria '{category_name}' jest pusta lub nie zawiera obrazów.",
                            )

            if not subfolders_to_process:
                ds_logger.error("Nie znaleziono plików w wybranych kategoriach")
                raise ValueError(
                    "Nie znaleziono żadnych plików obrazów w wybranych i niepustych podfolderach."
                )

            ds_logger.info(
                f"Znaleziono {total_files_to_process} plików w {len(subfolders_to_process)} kategoriach"
            )
            if progress_callback:
                progress_callback(
                    10,
                    f"Znaleziono {total_files_to_process} plików w {len(subfolders_to_process)} wybranych podkategoriach.",
                )

            processed_files_count = 0
            current_progress_base = 10  # Start progress after initial scan

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
                    ds_logger.warning(
                        "W trybie 'Limit plików' nie znaleziono plików w wybranych kategoriach"
                    )
                    if progress_callback:
                        progress_callback(
                            current_progress_base,  # Keep current progress
                            "Ostrzeżenie: W trybie 'Limit plików' nie znaleziono plików w wybranych kategoriach, co może wpłynąć na podział walidacyjny.",
                        )

            for idx, (relative_path, files) in enumerate(subfolders_to_process):
                if cancel_check and cancel_check():
                    ds_logger.info("Przetwarzanie zostało anulowane")
                    return "Anulowano.", None

                if progress_callback:
                    # Calculate progress: 10% for setup, 80% for processing, 10% for report
                    # This is a rough estimate for progress during file copying.
                    # It assumes each subfolder contributes somewhat equally to the remaining 80%.
                    # A more accurate way would be to base it on total_files_to_process.
                    # For now, simple per-subfolder progress update.
                    progress_val = current_progress_base + int(
                        ((idx + 1) / len(subfolders_to_process)) * 80
                    )
                    progress_callback(
                        progress_val,
                        f"Przetwarzanie kategorii: {relative_path} ({idx+1}/{len(subfolders_to_process)})",
                    )

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
                            and num_train
                            == len(files)  # If rounding caused all files to go to train
                        ):
                            if (
                                num_train > 0
                            ):  # Ensure there's at least one file to move from train to valid
                                num_train -= 1
                                num_valid = 1
                    else:  # No validation
                        num_train = len(files)  # All files go to train
                else:  # "files" mode
                    num_train = min(self.split_value, len(files))
                    if self.use_validation:
                        available_for_valid_in_current_cat = len(files) - num_train
                        desired_valid_based_on_min_cat = 0
                        # self.min_files_in_selection_for_report is min files across ALL selected categories
                        if self.min_files_in_selection_for_report > self.split_value:
                            desired_valid_based_on_min_cat = (
                                self.min_files_in_selection_for_report
                                - self.split_value
                            )

                        num_valid = min(
                            available_for_valid_in_current_cat,
                            desired_valid_based_on_min_cat,  # Cap validation files by what the smallest category can provide
                        )
                        if num_valid < 0:
                            num_valid = 0  # Safety check

                ds_logger.info(
                    f"Przetwarzanie kategorii {relative_path}: {num_train} tren., {num_valid} walid."
                )

                current_train_path = train_base_path / relative_path
                current_valid_path = (
                    valid_base_path / relative_path
                    if valid_base_path
                    and num_valid > 0  # Only create valid path if needed
                    else None
                )

                current_train_path.mkdir(parents=True, exist_ok=True)
                if current_valid_path:
                    current_valid_path.mkdir(parents=True, exist_ok=True)

                train_files_to_copy = files[:num_train]
                valid_files_to_copy = (
                    files[num_train : num_train + num_valid] if num_valid > 0 else []
                )

                # Copy train files
                for file_path in train_files_to_copy:
                    if cancel_check and cancel_check():
                        break
                    try:
                        # Wybierz funkcję w zależności od tego, czy przenosimy czy kopiujemy
                        file_operation = (
                            shutil.move if self.move_files else shutil.copy2
                        )
                        file_operation(file_path, current_train_path / file_path.name)
                        self.stats["train"][str(relative_path)] += 1
                        self.json_report[str(relative_path)]["train"].append(
                            file_path.name
                        )
                        processed_files_count += 1
                    except Exception as e:
                        ds_logger.error(
                            f"Błąd {'przenoszenia' if self.move_files else 'kopiowania'} {file_path} (trening): {e}"
                        )
                        return (
                            None,
                            f"Błąd {'przenoszenia' if self.move_files else 'kopiowania'} {file_path} (trening): {e}",
                        )
                if cancel_check and cancel_check():
                    break

                # Copy validation files
                for file_path in valid_files_to_copy:
                    if cancel_check and cancel_check():
                        break
                    try:
                        if current_valid_path:  # Ensure path exists
                            # Wybierz funkcję w zależności od tego, czy przenosimy czy kopiujemy
                            file_operation = (
                                shutil.move if self.move_files else shutil.copy2
                            )
                            file_operation(
                                file_path, current_valid_path / file_path.name
                            )
                            if (
                                self.use_validation
                            ):  # Double check, though num_valid > 0 implies this
                                self.stats["valid"][str(relative_path)] += 1
                                self.json_report[str(relative_path)]["valid"].append(
                                    file_path.name
                                )
                            processed_files_count += 1
                    except Exception as e:
                        ds_logger.error(
                            f"Błąd {'przenoszenia' if self.move_files else 'kopiowania'} {file_path} (walidacja): {e}"
                        )
                        return (
                            None,
                            f"Błąd {'przenoszenia' if self.move_files else 'kopiowania'} {file_path} (walidacja): {e}",
                        )
                if cancel_check and cancel_check():
                    break

            if cancel_check and cancel_check():
                ds_logger.info("Przetwarzanie anulowane podczas kopiowania plików.")
                return "Anulowano.", None

            if progress_callback:
                progress_callback(95, "Generowanie raportu...")
            report_content = self._generate_report()
            if progress_callback:
                progress_callback(100, "Zakończono.")
            ds_logger.info("Przetwarzanie zakończone sukcesem")
            return report_content, None

        except Exception as e:
            ds_logger.error(f"Wystąpił błąd podczas przetwarzania: {e}", exc_info=True)
            if progress_callback:
                progress_callback(0, f"Błąd krytyczny: {e}")
            return None, str(e)

    def _generate_report(self):
        operation_type = "przenoszenia" if self.move_files else "kopiowania"
        report = [f"=== RAPORT {operation_type.upper()} ===", ""]
        if self.split_mode == "percent":
            report.append(f"Algorytm: Podział procentowy (dla wybranych kategorii)")
            train_percent = self.split_value
            valid_percent = 100 - train_percent if self.use_validation else 0
            report.append(
                f"Stosunek plików: {train_percent}% {TRAIN_FOLDER_NAME}"  # MODIFIED HERE
                + (
                    f" / {valid_percent}% {VALID_FOLDER_NAME}"
                    if self.use_validation
                    else ""
                )  # MODIFIED HERE
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
            elif self.selected_categories:
                report.append(
                    "W wybranych kategoriach nie znaleziono plików do ustalenia globalnego limitu walidacji."
                )
            report.append(f"Z każdej wybranej kategorii próbowano wziąć:")
            report.append(
                f"  - {self.split_value} plików do {TRAIN_FOLDER_NAME} (lub mniej, jeśli kategoria miała mniej)"  # MODIFIED HERE
            )
            if self.use_validation:
                num_valid_expected = 0
                if min_files_val > self.split_value:
                    num_valid_expected = min_files_val - self.split_value
                report.append(
                    f"  - do {num_valid_expected} plików do {VALID_FOLDER_NAME} (lub mniej, jeśli było mniej dostępnych)"  # MODIFIED HERE
                )
            else:
                report.append(
                    f"  - 0 plików do {VALID_FOLDER_NAME} (opcja wyłączona)"
                )  # MODIFIED HERE
        report.append("")

        for category in sorted(self.stats["train"].keys()):
            train_count = self.stats["train"][category]
            report.append(f"{category}")
            report.append(f"├── {TRAIN_FOLDER_NAME}: {train_count} plików")
            if self.use_validation and category in self.stats["valid"]:
                valid_count = self.stats["valid"][category]
                report.append(f"└── {VALID_FOLDER_NAME}: {valid_count} plików")
            else:
                report.append(
                    f"└── {VALID_FOLDER_NAME}: 0 plików"
                )  # Even if validation off, show 0
            report.append("")

        total_train = sum(self.stats["train"].values())
        total_valid = (
            sum(self.stats["valid"].values())
            if self.use_validation and self.stats.get("valid")
            else 0
        )
        report.append("=== PODSUMOWANIE OGÓLNE ===")
        report.append(
            f"Łącznie {'przeniesiono' if self.move_files else 'skopiowano'}: {total_train + total_valid} plików"
        )
        report.append(f"  - {TRAIN_FOLDER_NAME}: {total_train} plików")
        report.append(f"  - {VALID_FOLDER_NAME}: {total_valid} plików")
        return "\n".join(report)


class DS_Worker(QThread):  # Renamed to DS_Worker
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
        move_files=False,  # Nowy parametr
    ):
        super().__init__()
        ds_logger.info("Inicjalizacja wątku DS_Worker")
        self.splitter = FileSplitter(
            input_dir,
            output_dir,
            split_mode,
            split_value,
            use_validation,
            selected_categories,
            move_files,  # Przekazanie parametru
        )
        self.is_cancelled = False

    def run(self):
        try:
            ds_logger.info("Rozpoczynam przetwarzanie w wątku DS_Worker")
            self.progress_updated.emit(0, "Rozpoczynanie przetwarzania...")

            result, error = self.splitter.process_files(
                progress_callback=self.progress_updated.emit,
                cancel_check=lambda: self.is_cancelled,
            )

            if self.is_cancelled:
                ds_logger.info("Przetwarzanie zostało anulowane")
                self.finished.emit("Anulowano.")
            elif error:
                ds_logger.error(f"Wystąpił błąd podczas przetwarzania: {error}")
                self.error_occurred.emit(f"Błąd: {error}")
                self.finished.emit(f"Błąd: {error}")
            else:
                ds_logger.info("Przetwarzanie zakończone sukcesem")
                self.finished.emit(result)
        except ConfigurationError as ce:
            ds_logger.error(f"Błąd konfiguracji: {ce}")
            self.error_occurred.emit(f"Błąd konfiguracji: {ce}")
            self.finished.emit(f"Błąd: {ce}")
        except ProcessingError as pe:
            ds_logger.error(f"Błąd przetwarzania: {pe}")
            self.error_occurred.emit(f"Błąd przetwarzania: {pe}")
            self.finished.emit(f"Błąd: {pe}")
        except Exception as e:
            ds_logger.error(f"Niespodziewany błąd: {e}", exc_info=True)
            self.error_occurred.emit(f"Niespodziewany błąd: {e}")
            self.finished.emit(f"Niespodziewany błąd: {e}")

    def cancel(self):
        ds_logger.info("Anulowanie przetwarzania")
        self.is_cancelled = True
        self.progress_updated.emit(0, "Anulowanie...")


class DS_ScannerSignals(QObject):  # Renamed
    finished = pyqtSignal(dict)
    progress = pyqtSignal(int, str)
    error = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)


class DS_FolderScanner(QRunnable):  # Renamed
    def __init__(self, root_path, allowed_extensions):
        super().__init__()
        ds_logger.info(f"Inicjalizacja DS_FolderScanner dla ścieżki: {root_path}")
        self.root_path = Path(root_path)
        self.allowed_extensions = allowed_extensions
        self.signals = DS_ScannerSignals()

    @pyqtSlot()
    def run(self):
        try:
            ds_logger.info("Rozpoczynam skanowanie katalogów")
            result = {
                "folder_counts": {},
                "min_files_count": float("inf"),
                "min_files_folder": None,
                "total_files": 0,
            }
            folders = [f for f in self.root_path.iterdir() if f.is_dir()]
            total_folders = len(folders)
            ds_logger.info(f"Znaleziono {total_folders} folderów do przeskanowania")

            for i, category_dir in enumerate(folders):
                progress_percent = (
                    int((i / total_folders) * 100) if total_folders > 0 else 0
                )
                self.signals.progress.emit(
                    progress_percent,
                    f"Skanowanie: {category_dir.name} ({i+1}/{total_folders})",
                )
                file_count = sum(
                    1
                    for f in category_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in self.allowed_extensions
                )
                result["folder_counts"][category_dir.name] = file_count
                result["total_files"] += file_count
                ds_logger.info(
                    f"Kategoria {category_dir.name}: znaleziono {file_count} plików"
                )
                if 0 < file_count < result["min_files_count"]:
                    result["min_files_count"] = file_count
                    result["min_files_folder"] = category_dir.name
                    ds_logger.info(
                        f"Nowa minimalna liczba plików: {file_count} w kategorii {category_dir.name}"
                    )

            if result["min_files_count"] == float("inf"):
                result["min_files_count"] = 0
                result["min_files_folder"] = None
                ds_logger.warning("Nie znaleziono żadnych plików w kategoriach")

            ds_logger.info(
                f"Skanowanie zakończone: {result['total_files']} plików w {len(result['folder_counts'])} kategoriach"
            )
            self.signals.finished.emit(result)
        except Exception as e:
            ds_logger.error(f"Błąd podczas skanowania katalogów: {e}", exc_info=True)
            self.signals.error.emit(str(e))


class DataSplitterApp(QWidget):
    # Signal to emit log messages to the main application console
    log_to_main_console = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        # ds_logger.info("Inicjalizacja DataSplitterApp") # This logger is fine for file logging
        self.log_to_main_console.emit("Inicjalizacja DataSplitterApp")

        self.input_dir = ""
        self.output_dir = ""
        self.processing_thread = None
        self.files_list = []  # Stores Path objects
        self.files_scanner_thread = None  # For QThread based file scanner
        self.move_files = False  # Domyślnie kopiowanie plików

        self.threadpool = QThreadPool()
        self.log_to_main_console.emit(
            f"DataSplitterApp: Używam puli wątków z max {self.threadpool.maxThreadCount()} wątkami"
        )

        # Try to load icon, specific to DataSplitter if needed, or rely on main app's icon
        # icon_path = Path("resources/img/icon.png") # Relative to where data_splitter_gui.py was
        # For combined app, paths need to be more robust or icons embedded
        # if icon_path.exists(): self.setWindowIcon(QIcon(str(icon_path)))

        self.initUI()
        # Styling will be handled by the main application's theme
        # self._apply_material_theme() # Remove, main app handles this
        self.update_files_limit_and_validation_based_on_selection()
        self.log_to_main_console.emit("Inicjalizacja DataSplitterApp zakończona")

    def _apply_ds_theme(
        self,
    ):  # Keep this if DS needs specific styling not covered by main
        # This is an example, main app's theme will likely override most of this.
        # It's better to use the main app's theme for consistency.
        pass

    def initUI(self):
        # self.setWindowTitle("Przygotowanie Danych AI") # Title set by Tab name
        # self.setGeometry(200, 200, 850, 650)
        layout = QVBoxLayout(self)  # Add self

        # --- MODIFIED FOLDER SELECTION LAYOUT ---
        folder_selection_outer_group = QGroupBox("Ścieżki folderów")
        folder_selection_outer_group.setFixedHeight(120)  # Ustawienie stałej wysokości
        folder_selection_main_layout = QHBoxLayout(
            folder_selection_outer_group
        )  # Main layout for the group box

        # Input Path Section
        input_path_v_layout = QVBoxLayout()  # Vertical layout for this section

        in_label = QLabel("Folder z danymi źródłowymi:")
        input_path_v_layout.addWidget(in_label)

        input_controls_h_layout = (
            QHBoxLayout()
        )  # Horizontal for QLineEdit and QPushButton
        self.in_path_edit = QLineEdit()
        self.in_path_edit.setReadOnly(True)
        in_button = QPushButton("Wybierz...")
        in_button.clicked.connect(self.select_input_folder)
        input_controls_h_layout.addWidget(
            self.in_path_edit, 1
        )  # QLineEdit takes more space
        input_controls_h_layout.addWidget(in_button)
        input_path_v_layout.addLayout(input_controls_h_layout)
        input_path_v_layout.addStretch(
            1
        )  # Pushes controls to top if vertical space is available

        folder_selection_main_layout.addLayout(
            input_path_v_layout
        )  # Add input section to main horizontal layout

        # Output Path Section
        output_path_v_layout = QVBoxLayout()  # Vertical layout for this section

        out_label = QLabel("Folder docelowy dla podziału:")
        output_path_v_layout.addWidget(out_label)

        output_controls_h_layout = (
            QHBoxLayout()
        )  # Horizontal for QLineEdit and QPushButton
        self.out_path_edit = QLineEdit()
        self.out_path_edit.setReadOnly(True)
        out_button = QPushButton("Wybierz...")
        out_button.clicked.connect(self.select_output_folder)
        output_controls_h_layout.addWidget(
            self.out_path_edit, 1
        )  # QLineEdit takes more space
        output_controls_h_layout.addWidget(out_button)
        output_path_v_layout.addLayout(output_controls_h_layout)
        output_path_v_layout.addStretch(1)  # Pushes controls to top

        folder_selection_main_layout.addLayout(
            output_path_v_layout
        )  # Add output section to main horizontal layout

        layout.addWidget(folder_selection_outer_group)
        # --- END MODIFIED FOLDER SELECTION LAYOUT ---

        self.tabs = QTabWidget()

        # Folder tree tab
        folder_tree_widget = QWidget()
        folder_tree_main_layout = QVBoxLayout(folder_tree_widget)

        self.folder_tree = QTreeWidget()
        self.folder_tree.setHeaderLabels(
            ["Struktura folderów (zaznacz kategorie do przetworzenia)"]
        )
        self.folder_tree.setColumnCount(1)
        self.folder_tree.itemChanged.connect(self.on_folder_tree_item_changed)
        # QTreeWidget typically expands by default, so no specific size policy needed here
        # Its parent layouts (QVBoxLayout, QTabWidget) will manage its size.

        folder_buttons_layout = QHBoxLayout()
        self.select_all_button = QPushButton("Zaznacz wszystkie")
        self.deselect_all_button = QPushButton("Odznacz wszystkie")
        self.select_all_button.clicked.connect(self.select_all_folders)
        self.deselect_all_button.clicked.connect(self.deselect_all_folders)
        folder_buttons_layout.addWidget(self.select_all_button)
        folder_buttons_layout.addWidget(self.deselect_all_button)
        folder_tree_main_layout.addLayout(folder_buttons_layout)
        folder_tree_main_layout.addWidget(
            self.folder_tree, 1
        )  # Add stretch factor for tree
        self.tabs.addTab(folder_tree_widget, "Wybór kategorii")

        # Files list tab (optional, can be demanding for very large datasets)
        self.files_list_widget = QListWidget()
        self.tabs.addTab(self.files_list_widget, "Lista wszystkich plików (podgląd)")
        layout.addWidget(self.tabs)

        # Split options group
        split_options_group = QGroupBox("Opcje podziału")
        split_main_layout = QVBoxLayout(split_options_group)

        mode_layout = QHBoxLayout()
        mode_label = QLabel("Tryb podziału:")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Podział procentowy", "Limit plików na kategorię"])
        self.mode_combo.currentIndexChanged.connect(self.update_split_mode_visibility)
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo)
        split_main_layout.addLayout(mode_layout)

        self.percent_layout_widget = (
            QWidget()
        )  # Use a QWidget to toggle visibility of the whole layout
        self.percent_layout = QHBoxLayout(self.percent_layout_widget)
        percent_label = QLabel(
            f"Podział {TRAIN_FOLDER_NAME} / {VALID_FOLDER_NAME} (%):"
        )  # MODIFIED HERE
        self.split_slider = QSlider(Qt.Orientation.Horizontal)
        self.split_slider.setMinimum(1)
        self.split_slider.setMaximum(99)
        self.split_slider.setValue(DEFAULT_TRAIN_SPLIT_PERCENT)
        self.split_slider.setTickInterval(10)
        self.split_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.split_value_label = QLabel(
            f"{DEFAULT_TRAIN_SPLIT_PERCENT}% / {100 - DEFAULT_TRAIN_SPLIT_PERCENT}%"
        )
        self.split_value_label.setMinimumWidth(80)  # Ensure space for label
        self.split_slider.valueChanged.connect(self.update_split_label_text)
        self.percent_layout.addWidget(percent_label)
        self.percent_layout.addWidget(self.split_slider)
        self.percent_layout.addWidget(self.split_value_label)
        split_main_layout.addWidget(self.percent_layout_widget)

        self.files_layout_widget = QWidget()  # Use QWidget for visibility toggle
        self.files_layout = QHBoxLayout(self.files_layout_widget)
        files_label = QLabel(
            f"Liczba plików {TRAIN_FOLDER_NAME} na kategorię:"
        )  # MODIFIED HERE
        self.files_spin = QSpinBox()
        self.files_spin.setMinimum(1)
        self.files_spin.setMaximum(10000)  # Default max, will be adjusted
        self.files_spin.setValue(DEFAULT_FILES_PER_CATEGORY)
        self.files_spin.valueChanged.connect(
            self.update_files_limit_and_validation_based_on_selection
        )
        self.files_layout.addWidget(files_label)
        self.files_layout.addWidget(self.files_spin)
        split_main_layout.addWidget(self.files_layout_widget)

        validation_layout = QHBoxLayout()
        self.validation_check = QCheckBox(
            f"Utwórz folder {VALID_FOLDER_NAME}"
        )  # MODIFIED HERE
        self.validation_check.setChecked(True)
        self.validation_check.stateChanged.connect(
            self.update_files_limit_and_validation_based_on_selection
        )
        self.validation_label = QLabel("")  # Informational label
        validation_layout.addWidget(self.validation_check)
        validation_layout.addWidget(self.validation_label)
        validation_layout.addStretch()
        split_main_layout.addLayout(validation_layout)
        layout.addWidget(split_options_group)

        # Control buttons group
        # control_group = QGroupBox("Akcje") # Not really needed if progress bar is outside
        # control_main_layout = QVBoxLayout(control_group)

        control_buttons_layout = QHBoxLayout()
        self.start_button = QPushButton("Rozpocznij kopiowanie")
        self.start_button.setProperty("action", "success")
        self.start_button.clicked.connect(self.start_processing)
        self.empty_button = QPushButton("Przenieś pliki")  # Zmiana nazwy przycisku
        self.empty_button.clicked.connect(
            self.start_moving
        )  # Powiązanie z nową funkcją
        self.cancel_button = QPushButton("Anuluj")
        self.cancel_button.clicked.connect(self.cancel_processing)
        self.cancel_button.setEnabled(False)
        control_buttons_layout.addWidget(self.start_button)
        control_buttons_layout.addWidget(self.empty_button)
        control_buttons_layout.addWidget(self.cancel_button)
        layout.addLayout(control_buttons_layout)  # Add QHBoxLayout directly

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        # Log edit is handled by main application
        # log_label = QLabel("Log (Data Splitter):")
        # self.log_edit = QTextEdit() # This would be a local log for DS
        # self.log_edit.setReadOnly(True)
        # self.log_edit.setMaximumHeight(100)
        # layout.addWidget(log_label)
        # layout.addWidget(self.log_edit)

        self.setLayout(layout)
        # self.show() # Main app shows this widget
        self.update_split_mode_visibility(0)  # Initial setup based on combobox

        # Style dla przycisków w DataSplitterApp
        self.start_button.setStyleSheet(BUTTON_STYLES["success"])
        self.empty_button.setStyleSheet(BUTTON_STYLES["warning"])
        self.cancel_button.setStyleSheet(BUTTON_STYLES["stop"])
        in_button.setStyleSheet(BUTTON_STYLES["default"])
        out_button.setStyleSheet(BUTTON_STYLES["default"])
        self.select_all_button.setStyleSheet(BUTTON_STYLES["default"])
        self.deselect_all_button.setStyleSheet(BUTTON_STYLES["default"])

    def update_split_mode_visibility(self, index):
        is_percent_mode = index == 0
        self.percent_layout_widget.setVisible(is_percent_mode)
        self.files_layout_widget.setVisible(not is_percent_mode)
        self.update_files_limit_and_validation_based_on_selection()

    def update_split_label_text(self):
        train_percent = self.split_slider.value()
        valid_percent = 100 - train_percent
        self.split_value_label.setText(f"{train_percent}% / {valid_percent}%")
        if self.mode_combo.currentIndex() == 0:  # Percent mode
            self.update_files_limit_and_validation_based_on_selection()

    def update_folder_tree(self):
        self.folder_tree.clear()
        if not self.input_dir:
            self.update_files_limit_and_validation_based_on_selection()  # Ensure controls are updated
            return

        root_path = Path(self.input_dir)
        # Add a root item representing the input directory itself
        tree_root_item = QTreeWidgetItem(
            self.folder_tree, [f"{root_path.name} (katalog główny)"]
        )
        self.folder_tree.addTopLevelItem(tree_root_item)
        tree_root_item.setFlags(
            tree_root_item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable
        )  # Not checkable
        tree_root_item.setExpanded(True)

        try:  # Disconnect to prevent multiple calls during population
            self.folder_tree.itemChanged.disconnect(self.on_folder_tree_item_changed)
        except TypeError:
            pass  # Was not connected

        # Scanner for folder stats
        scanner = DS_FolderScanner(root_path, ALLOWED_IMAGE_EXTENSIONS_DS)

        # Use local progress bar and log_message for scanner updates
        def on_scan_progress(percent, message):
            self.log_message(
                f"Skaner folderów: {message}", level=logging.DEBUG
            )  # To main console
            self.progress_bar.setValue(percent)  # Update DS progress bar
            QApplication.processEvents()

        def on_scan_finished(result):
            folder_counts = result.get("folder_counts", {})
            min_files_folder_name = result.get("min_files_folder")  # Name of folder

            # Add categories (subfolders) to the tree_root_item
            sorted_category_names = sorted(
                [d.name for d in root_path.iterdir() if d.is_dir()]
            )

            for category_name in sorted_category_names:
                category_dir = root_path / category_name  # Reconstruct Path object
                file_count = folder_counts.get(category_name, 0)
                display_text = f"{category_name} ({file_count} plików)"
                item = QTreeWidgetItem(tree_root_item, [display_text])
                item.setData(
                    0, Qt.ItemDataRole.UserRole, category_name
                )  # Store category name
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(0, Qt.CheckState.Checked)  # Default to checked

                if category_name == min_files_folder_name and file_count > 0:
                    item.setForeground(
                        0,
                        QColor(DS_HIGHLIGHT_COLOR if DS_HIGHLIGHT_COLOR else "#00AACC"),
                    )
                QApplication.processEvents()

            self.folder_tree.itemChanged.connect(
                self.on_folder_tree_item_changed
            )  # Reconnect
            self.update_files_limit_and_validation_based_on_selection()
            self.progress_bar.setValue(0)  # Reset progress bar
            self.log_message(
                f"Skaner folderów: Zakończono. Znaleziono {result.get('total_files',0)} plików w {len(folder_counts)} kategoriach."
            )

        def on_scan_error(error_message):
            self.log_message(
                f"Skaner folderów BŁĄD: {error_message}", level=logging.ERROR
            )
            self.progress_bar.setValue(0)
            try:  # Reconnect even on error
                self.folder_tree.itemChanged.connect(self.on_folder_tree_item_changed)
            except TypeError:
                pass

        scanner.signals.progress.connect(on_scan_progress)
        scanner.signals.finished.connect(on_scan_finished)
        scanner.signals.error.connect(on_scan_error)
        self.threadpool.start(scanner)

    def on_folder_tree_item_changed(self, item, column):
        # Only react to changes in the first column for children of the root item
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
                category_name = child_item.data(
                    0, Qt.ItemDataRole.UserRole
                )  # Retrieve stored name
                if category_name:
                    selected.append(category_name)
        return selected

    def get_min_files_in_selected_categories(self, selected_category_names):
        if not self.input_dir or not selected_category_names:
            return (
                DEFAULT_FILES_PER_CATEGORY,
                [],
            )  # Default if no selection or input_dir

        min_val = float("inf")
        folders_with_min = []  # List of names of folders having the min_val
        root = Path(self.input_dir)

        for name in selected_category_names:
            cat_dir = root / name
            if cat_dir.is_dir():
                count = sum(
                    1
                    for f in cat_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS_DS
                )
                if 0 < count < min_val:
                    min_val = count
                    folders_with_min = [name]
                elif count == min_val and min_val != float(
                    "inf"
                ):  # Handle multiple folders with the same min count
                    folders_with_min.append(name)

        return (min_val, folders_with_min) if min_val != float("inf") else (0, [])

    def update_files_limit_and_validation_based_on_selection(self):
        selected_cats = self.get_selected_categories_names()
        min_files_in_selection, _ = self.get_min_files_in_selected_categories(
            selected_cats
        )

        # Update files_spin state (for "Limit plików" mode)
        if not selected_cats:
            self.files_spin.setEnabled(False)
            self.files_spin.setValue(1)  # Reset to a safe minimum
            self.log_message(
                "Wybierz kategorie, aby dostosować opcje podziału.", level=logging.DEBUG
            )
        else:
            self.files_spin.setEnabled(
                self.mode_combo.currentIndex() == 1
            )  # Only enable in "files" mode
            if min_files_in_selection > 0:
                self.files_spin.setMaximum(min_files_in_selection)
                # Adjust current value if it exceeds new maximum
                if self.files_spin.value() > min_files_in_selection:
                    self.files_spin.setValue(min_files_in_selection)
            else:  # No files in selected categories, or categories are empty
                self.files_spin.setMaximum(
                    1
                )  # Can't take more than 0 or 1 if categories are problematic
                self.files_spin.setValue(1)

        # Update validation_check state and label
        current_files_limit = self.files_spin.value()
        is_percent_mode = self.mode_combo.currentIndex() == 0

        if is_percent_mode:
            can_create_validation_percent = self.split_slider.value() < 100
            self.validation_check.setEnabled(
                can_create_validation_percent and bool(selected_cats)
            )
            if not can_create_validation_percent:
                self.validation_check.setChecked(
                    False
                )  # Can't have validation if 100% train
                self.validation_label.setText(
                    f"(100% {TRAIN_FOLDER_NAME})"
                )  # MODIFIED HERE
            else:
                self.validation_label.setText(
                    f"({100 - self.split_slider.value()}% {VALID_FOLDER_NAME})"  # MODIFIED HERE
                )
        else:  # Files mode
            # Validation possible if files_limit < min_files_in_selection AND there are selected categories with files
            can_create_validation_files = (
                current_files_limit < min_files_in_selection
                and min_files_in_selection > 0
                and bool(selected_cats)
            )
            self.validation_check.setEnabled(can_create_validation_files)
            if not can_create_validation_files:
                self.validation_check.setChecked(False)
                if not selected_cats:
                    self.validation_label.setText("(wybierz kat.)")
                elif min_files_in_selection == 0:
                    self.validation_label.setText("(brak plików)")
                else:
                    self.validation_label.setText(
                        "(limit >= min.)"
                    )  # Limit too high for validation
            else:
                max_valid_files = min_files_in_selection - current_files_limit
                self.validation_label.setText(
                    f"(do {max_valid_files} {VALID_FOLDER_NAME})"
                )  # MODIFIED HERE

    def update_files_list(self):
        self.files_list_widget.clear()
        self.files_list = []  # Reset stored files (Path objects)

        if not self.input_dir:
            return

        self.files_list_widget.addItem("Trwa skanowanie plików (podgląd)...")
        QApplication.processEvents()

        class DS_FileScannerThread(QThread):  # Renamed
            finished = pyqtSignal(list)  # list of (Path, str_rel_path) tuples
            progress = pyqtSignal(int)  # count of files found

            def __init__(self, root_path, allowed_extensions):
                super().__init__()
                self.root_path = Path(root_path)  # Ensure it's a Path object
                self.allowed_extensions = allowed_extensions
                self.files_data = []  # Store (Path_obj, str_rel_path)
                self._stop = False

            def run(self):
                try:
                    for item in self.root_path.rglob("*"):
                        if self._stop:
                            break
                        if (
                            item.is_file()
                            and item.suffix.lower() in self.allowed_extensions
                        ):
                            try:
                                relative_path_str = str(
                                    item.relative_to(self.root_path)
                                )
                                self.files_data.append((item, relative_path_str))
                                if (
                                    len(self.files_data) % 100 == 0
                                ):  # Update progress periodically
                                    self.progress.emit(len(self.files_data))
                            except (
                                ValueError
                            ):  # Not relative, should not happen with rglob from root
                                ds_logger.warning(
                                    f"DS_FileScannerThread: {item} not relative to {self.root_path}"
                                )
                    self.finished.emit(self.files_data)
                except Exception as e:
                    ds_logger.error(f"DS_FileScannerThread error: {e}", exc_info=True)
                    self.finished.emit([])  # Emit empty list on error

            def stop(self):
                self._stop = True

        def on_scan_progress(count):
            if self.files_list_widget.count() > 0:
                self.files_list_widget.item(0).setText(
                    f"Skanowanie w toku... Znaleziono {count} plików (podgląd)"
                )
            QApplication.processEvents()

        def on_scan_complete(files_data_list):  # Receives list of (Path, str_rel_path)
            self.files_list_widget.clear()
            self.files_list = [
                path_obj for path_obj, _ in files_data_list
            ]  # Store full Path objects

            display_limit = 500  # Limit display for performance
            for i, (_, rel_path_str) in enumerate(files_data_list[:display_limit]):
                self.files_list_widget.addItem(rel_path_str)

            if len(files_data_list) > display_limit:
                self.files_list_widget.addItem(
                    f"... oraz {len(files_data_list) - display_limit} więcej plików (nie wyświetlono wszystkich)."
                )

            self.log_message(
                f"Skaner plików: Znaleziono łącznie {len(files_data_list)} plików.",
                level=logging.INFO,
            )
            if self.files_scanner_thread:  # Clean up
                self.files_scanner_thread.finished.disconnect()
                self.files_scanner_thread.progress.disconnect()
            self.files_scanner_thread = None

        if self.files_scanner_thread and self.files_scanner_thread.isRunning():
            self.files_scanner_thread.stop()
            self.files_scanner_thread.wait()  # Wait for it to finish

        self.files_scanner_thread = DS_FileScannerThread(
            Path(self.input_dir), ALLOWED_IMAGE_EXTENSIONS_DS
        )
        self.files_scanner_thread.progress.connect(on_scan_progress)
        self.files_scanner_thread.finished.connect(on_scan_complete)
        self.files_scanner_thread.start()

    def log_message(self, message, level=logging.INFO):
        # self.log_edit.append(message) # Local log edit
        # self.log_edit.verticalScrollBar().setValue(self.log_edit.verticalScrollBar().maximum())
        self.log_to_main_console.emit(
            f"[DataSplitter] {message}"
        )  # Emit to main console

        # Log to DataSplitter's own logger (for file logging if configured)
        if level == logging.DEBUG:
            ds_logger.debug(message)
        elif level == logging.INFO:
            ds_logger.info(message)
        elif level == logging.WARNING:
            ds_logger.warning(message)
        elif level == logging.ERROR:
            ds_logger.error(message)
        elif level == logging.CRITICAL:
            ds_logger.critical(message)
        QApplication.processEvents()

    def select_input_folder(self):
        self.log_message("Rozpoczynam wybór folderu źródłowego", level=logging.DEBUG)
        folder = QFileDialog.getExistingDirectory(
            self,
            "Wybierz folder źródłowy",
            self.input_dir if self.input_dir else str(Path.home()),
        )
        if folder:
            self.input_dir = folder
            self.in_path_edit.setText(folder)
            self.log_message(f"Wybrano folder źródłowy: {folder}")
            self.update_folder_tree()
            self.update_files_list()
        else:
            self.log_message("Anulowano wybór folderu źródłowego", level=logging.DEBUG)

    def select_output_folder(self):
        self.log_message("Rozpoczynam wybór folderu docelowego", level=logging.DEBUG)
        folder = QFileDialog.getExistingDirectory(
            self,
            "Wybierz folder docelowy",
            self.output_dir if self.output_dir else str(Path.home()),
        )
        if folder:
            self.output_dir = folder
            self.out_path_edit.setText(folder)
            self.log_message(f"Wybrano folder docelowy: {folder}")
        else:
            self.log_message("Anulowano wybór folderu docelowego", level=logging.DEBUG)

    def update_progress_display(self, value, message):  # Renamed to avoid conflict
        self.progress_bar.setValue(value)
        if message:  # Only log if message is not empty
            self.log_message(
                message, level=logging.DEBUG
            )  # Progress messages often verbose

    def processing_finished_display(self, final_message):  # Renamed
        self.log_message(f"Status końcowy: {final_message}")
        QApplication.processEvents()

        if "Błąd" in final_message:
            QMessageBox.warning(
                self,
                "Błąd",
                f"Wystąpił błąd:\n{final_message.splitlines()[-1] if final_message.splitlines() else final_message}",
            )
        elif "Anulowano" not in final_message:
            dialog = DS_ReportDialog(final_message, self)  # Use DS_ReportDialog
            dialog.exec()
        else:  # Anulowano
            QMessageBox.information(
                self, "Anulowano", "Przetwarzanie zostało anulowane."
            )

        self._set_controls_enabled(True)
        self.cancel_button.setEnabled(False)
        self.processing_thread = None

    def processing_error_display(self, error_message):  # Renamed
        self.log_message(f"BŁĄD WĄTKU: {error_message}", level=logging.ERROR)
        # self._set_controls_enabled(True) # Already handled by finished signal

    def start_processing(self):
        self.log_message("Rozpoczynam przetwarzanie")
        if not self.input_dir or not Path(self.input_dir).is_dir():
            QMessageBox.warning(
                self, "Brak folderu", "Wybierz prawidłowy folder źródłowy."
            )
            return
        if not self.output_dir:
            QMessageBox.warning(self, "Brak folderu", "Wybierz folder docelowy.")
            return

        selected_categories = self.get_selected_categories_names()
        if not selected_categories:
            QMessageBox.warning(
                self,
                "Brak kategorii",
                "Wybierz przynajmniej jedną kategorię do przetworzenia.",
            )
            return

        self.log_message(f"Wybrane kategorie do kopiowania: {selected_categories}")

        input_path = Path(self.input_dir)
        output_path = Path(self.output_dir)
        if input_path == output_path or output_path.is_relative_to(input_path):
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
                self.log_message(
                    "Użytkownik anulował operację ze względu na ścieżki.",
                    level=logging.INFO,
                )
                return

        self._set_controls_enabled(False)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setValue(0)

        split_mode_str = "percent" if self.mode_combo.currentIndex() == 0 else "files"
        split_val = (
            self.split_slider.value()
            if split_mode_str == "percent"
            else self.files_spin.value()
        )
        use_val_check = self.validation_check.isChecked()

        self.log_message(
            f"Parametry: tryb={split_mode_str}, wartość={split_val}, walidacja={use_val_check}"
        )

        self.move_files = False  # Ustawiamy flagę przenoszenia na False
        self.processing_thread = DS_Worker(  # Use DS_Worker
            self.input_dir,
            self.output_dir,
            split_mode_str,
            split_val,
            use_val_check,
            selected_categories,
            move_files=False,  # Parametr move_files=False
        )
        self.processing_thread.progress_updated.connect(self.update_progress_display)
        self.processing_thread.finished.connect(self.processing_finished_display)
        self.processing_thread.error_occurred.connect(self.processing_error_display)
        self.log_message("Uruchamiam wątek przetwarzania DataSplitter")
        self.processing_thread.start()

    def cancel_processing(self):
        if self.processing_thread and self.processing_thread.isRunning():
            self.log_message("Wysyłanie sygnału anulowania do DataSplitter...")
            self.processing_thread.cancel()  # Call cancel on DS_Worker
            self.cancel_button.setEnabled(False)  # Disable to prevent multiple clicks

    def _set_controls_enabled(self, enabled: bool):
        self.start_button.setEnabled(enabled)
        self.in_path_edit.setEnabled(
            enabled
        )  # Though ReadOnly, better to disable interactions
        self.out_path_edit.setEnabled(enabled)
        # Assuming in_button and out_button are children of path edits or main layout
        # If they are direct members of self, disable them too:
        # self.in_button.setEnabled(enabled)
        # self.out_button.setEnabled(enabled)
        self.mode_combo.setEnabled(enabled)
        self.split_slider.setEnabled(enabled and self.mode_combo.currentIndex() == 0)
        self.files_spin.setEnabled(enabled and self.mode_combo.currentIndex() == 1)
        # self.validation_check.setEnabled(enabled) # This is handled by update_files_limit...
        self.update_files_limit_and_validation_based_on_selection()  # Re-evaluate validation check enable state

    def cleanup(self):  # Called by main window on close
        self.log_message("DataSplitterApp cleanup initiated.", level=logging.DEBUG)
        if self.files_scanner_thread and self.files_scanner_thread.isRunning():
            self.files_scanner_thread.stop()
            self.files_scanner_thread.wait()
            self.log_message("DS_FileScannerThread stopped.", level=logging.DEBUG)

        if self.processing_thread and self.processing_thread.isRunning():
            self.log_message(
                "DS_Worker still running during cleanup, attempting to cancel.",
                level=logging.WARNING,
            )
            self.processing_thread.cancel()
            # self.processing_thread.wait() # Wait can be long, consider if essential for main window close
            self.log_message("DS_Worker cancel signal sent.", level=logging.DEBUG)

        self.threadpool.waitForDone(-1)  # Wait indefinitely for all QRunnables
        self.log_message(
            "DataSplitterApp QThreadPool tasks finished.", level=logging.DEBUG
        )

    def select_all_folders(self):
        if not self.folder_tree.topLevelItem(0):
            return
        root_item = self.folder_tree.topLevelItem(0)
        for i in range(root_item.childCount()):
            child_item = root_item.child(i)
            child_item.setCheckState(0, Qt.CheckState.Checked)
        # No need to call update_files_limit... here, itemChanged will trigger it for each item

    def deselect_all_folders(self):
        if not self.folder_tree.topLevelItem(0):
            return
        root_item = self.folder_tree.topLevelItem(0)
        for i in range(root_item.childCount()):
            child_item = root_item.child(i)
            child_item.setCheckState(0, Qt.CheckState.Unchecked)
        # No need to call update_files_limit... here

    def start_moving(self):
        """Funkcja analogiczna do start_processing, ale przenosi pliki zamiast kopiować"""
        self.log_message("Rozpoczynam przenoszenie plików")
        if not self.input_dir or not Path(self.input_dir).is_dir():
            QMessageBox.warning(
                self, "Brak folderu", "Wybierz prawidłowy folder źródłowy."
            )
            return
        if not self.output_dir:
            QMessageBox.warning(self, "Brak folderu", "Wybierz folder docelowy.")
            return

        selected_categories = self.get_selected_categories_names()
        if not selected_categories:
            QMessageBox.warning(
                self,
                "Brak kategorii",
                "Wybierz przynajmniej jedną kategorię do przetworzenia.",
            )
            return

        self.log_message(f"Wybrane kategorie do przeniesienia: {selected_categories}")

        input_path = Path(self.input_dir)
        output_path = Path(self.output_dir)
        if input_path == output_path or output_path.is_relative_to(input_path):
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
                self.log_message(
                    "Użytkownik anulował operację ze względu na ścieżki.",
                    level=logging.INFO,
                )
                return

        # Dodatkowe ostrzeżenie, ponieważ przenoszenie jest operacją nieodwracalną
        reply = QMessageBox.question(
            self,
            "Potwierdzenie przenoszenia",
            "UWAGA: Przenoszenie plików jest operacją nieodwracalną i spowoduje usunięcie oryginałów."
            "\nCzy na pewno chcesz kontynuować?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.No:
            self.log_message(
                "Użytkownik anulował operację przenoszenia.", level=logging.INFO
            )
            return

        self._set_controls_enabled(False)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setValue(0)

        split_mode_str = "percent" if self.mode_combo.currentIndex() == 0 else "files"
        split_val = (
            self.split_slider.value()
            if split_mode_str == "percent"
            else self.files_spin.value()
        )
        use_val_check = self.validation_check.isChecked()

        self.log_message(
            f"Parametry przenoszenia: tryb={split_mode_str}, wartość={split_val}, walidacja={use_val_check}"
        )

        self.move_files = True  # Ustawiamy flagę przenoszenia na True
        self.processing_thread = DS_Worker(  # Use DS_Worker
            self.input_dir,
            self.output_dir,
            split_mode_str,
            split_val,
            use_val_check,
            selected_categories,
            move_files=True,  # Parametr move_files=True
        )
        self.processing_thread.progress_updated.connect(self.update_progress_display)
        self.processing_thread.finished.connect(self.processing_finished_display)
        self.processing_thread.error_occurred.connect(self.processing_error_display)
        self.log_message("Uruchamiam wątek przenoszenia DataSplitter")
        self.processing_thread.start()


class DS_ReportDialog(QDialog):  # Renamed
    def __init__(self, report_text, parent=None):
        super().__init__(parent)
        self.log_to_main_console = (
            parent.log_to_main_console
            if hasattr(parent, "log_to_main_console")
            else None
        )
        if self.log_to_main_console:
            self.log_to_main_console.emit("Inicjalizacja okna raportu DataSplitter")

        # Ustaw tytuł okna na podstawie zawartości raportu
        if "RAPORT PRZENOSZENIA" in report_text:
            self.setWindowTitle("Raport przenoszenia (Data Splitter)")
        else:
            self.setWindowTitle("Raport kopiowania (Data Splitter)")

        self.setMinimumSize(700, 500)  # Adjusted size
        layout = QVBoxLayout(self)
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(QFont("Consolas", 10))
        # Use main app's theme for consistency, but can have fallbacks
        self.text_edit.setStyleSheet(
            f"background-color: {DS_BACKGROUND}; color: {DS_TEXT_COLOR}; border: 1px solid {DS_BORDER_COLOR}; padding: 10px;"
        )
        self.text_edit.setHtml(self._format_report_to_html(report_text))

        close_button = QPushButton("Zamknij")
        close_button.setStyleSheet(
            f"background-color: {DS_PRIMARY_COLOR}; color: white; border: none; padding: 8px 16px; border-radius: 4px; min-width: 100px;"
        )
        close_button.clicked.connect(self.accept)

        layout.addWidget(self.text_edit)
        layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignCenter)
        self.setLayout(layout)
        if self.log_to_main_console:
            self.log_to_main_console.emit("Okno raportu DataSplitter skonfigurowane")

    def _format_report_to_html(self, report_text):
        html = report_text.replace("\n", "<br>")
        # Basic formatting for headers
        html = html.replace("=== RAPORT KOPIOWANIA ===", "<h2>RAPORT KOPIOWANIA</h2>")
        html = html.replace(
            "=== RAPORT PRZENOSZENIA ===", "<h2>RAPORT PRZENOSZENIA</h2>"
        )
        html = html.replace(
            "=== PODSUMOWANIE OGÓLNE ===", "<h2>PODSUMOWANIE OGÓLNE</h2>"
        )
        # Highlight folder names or key stats if needed, e.g. by wrapping with <strong>
        return f"<body style='color:{DS_TEXT_COLOR}; background-color:{DS_BACKGROUND};'>{html}</body>"


# --- END: data_splitter_gui.py components ---


# --- BEGIN: scaller.py components ---
# SCALLER_TARGET_DIMENSION = 300 # MODIFIED: This will become a configurable value
DEFAULT_SCALLER_TARGET_DIMENSION = 300  # MODIFIED: Added default
SCALLER_SUPPORTED_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".tiff",
    ".webp",
)


class ScallerWorker(QObject):  # Renamed
    finished = pyqtSignal()
    progress = pyqtSignal(int, int)  # current, total
    status_update = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(
        self, folder_path, target_dimension
    ):  # MODIFIED: Added target_dimension
        super().__init__()
        self.folder_path = folder_path
        self.target_dimension = target_dimension  # MODIFIED: Store target_dimension
        self.is_running = True

    def stop(self):
        self.is_running = False
        self.status_update.emit("Proces skalowania zatrzymywany...")

    def run(self):
        image_files = []
        self.status_update.emit(
            f"Wyszukiwanie plików graficznych w: {self.folder_path}..."
        )
        QApplication.processEvents()  # Allow UI to update
        for root, _, files in os.walk(self.folder_path):
            if not self.is_running:
                self.status_update.emit("Przerwano wyszukiwanie plików (Scaller).")
                self.finished.emit()
                return
            for file in files:
                if file.lower().endswith(SCALLER_SUPPORTED_EXTENSIONS):
                    image_files.append(os.path.join(root, file))

        if not self.is_running:  # Check again after loop
            self.status_update.emit("Przerwano wyszukiwanie plików (Scaller).")
            self.finished.emit()
            return

        if not image_files:
            self.status_update.emit("Nie znaleziono plików graficznych do skalowania.")
            self.finished.emit()
            return

        total_files = len(image_files)
        self.status_update.emit(
            f"Znaleziono {total_files} plików. Rozpoczynanie skalowania (cel: {self.target_dimension}px)..."  # MODIFIED: Added target_dimension to log
        )
        self.progress.emit(0, total_files)
        QApplication.processEvents()

        processed_count = 0
        skipped_count = 0
        resized_count = 0

        for i, file_path in enumerate(image_files):
            if not self.is_running:
                self.status_update.emit(f"Przerwano skalowanie po {i} plikach.")
                break

            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    if (
                        width
                        <= self.target_dimension  # MODIFIED: Use self.target_dimension
                        or height
                        <= self.target_dimension  # MODIFIED: Use self.target_dimension
                    ):
                        self.status_update.emit(
                            f"Pominięto (wymiar <= {self.target_dimension}px): {Path(file_path).name} ({width}x{height})"  # MODIFIED
                        )
                        skipped_count += 1
                    else:
                        original_format = img.format
                        if width < height:  # MODIFIED: Changed from width > height
                            new_width = self.target_dimension  # MODIFIED
                            new_height = int(
                                height * (self.target_dimension / width)  # MODIFIED
                            )
                        else:
                            new_height = self.target_dimension  # MODIFIED
                            new_width = int(
                                width * (self.target_dimension / height)
                            )  # MODIFIED

                        new_width = max(1, new_width)
                        new_height = max(1, new_height)

                        resized_img = img.resize(
                            (new_width, new_height), Image.Resampling.LANCZOS
                        )
                        save_kwargs = {}
                        if original_format and original_format.upper() == "JPEG":
                            save_kwargs["quality"] = 90
                            save_kwargs["optimize"] = True
                        elif original_format and original_format.upper() == "PNG":
                            save_kwargs["optimize"] = True

                        resized_img.save(
                            file_path, format=original_format, **save_kwargs
                        )
                        self.status_update.emit(
                            f"Przeskalowano: {Path(file_path).name} do {new_width}x{new_height}"
                        )
                        resized_count += 1
            except FileNotFoundError:
                self.status_update.emit(
                    f"BŁĄD (Scaller): Nie znaleziono pliku: {file_path}"
                )
                self.error_signal.emit(f"Nie znaleziono pliku: {file_path}")
            except UnidentifiedImageError:
                self.status_update.emit(
                    f"BŁĄD (Scaller): Nie można otworzyć: {Path(file_path).name}"
                )
                self.error_signal.emit(
                    f"Nie można otworzyć lub zidentyfikować jako obraz: {file_path}"
                )
            except Exception as e:
                self.status_update.emit(
                    f"BŁĄD (Scaller) {Path(file_path).name}: {str(e)}"
                )
                self.error_signal.emit(f"Błąd dla pliku {file_path}: {str(e)}")

            processed_count += 1
            self.progress.emit(processed_count, total_files)
            if i % 10 == 0:
                QApplication.processEvents()  # Allow UI to refresh periodically

        if self.is_running:
            self.status_update.emit(
                f"Skalowanie zakończone. Przeskalowano: {resized_count}, Pominięto: {skipped_count}, Razem przetworzono: {processed_count} z {total_files}."
            )
        else:
            self.status_update.emit(
                f"Skalowanie zatrzymane. Przetworzono {processed_count}/{total_files} plików."
            )
        self.finished.emit()


class ScallerApp(QWidget):  # Renamed
    log_to_main_console = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker_scaller = None  # Renamed attribute
        self.thread_scaller = None  # Renamed attribute
        self._init_ui()
        self.log_to_main_console.emit("[ScallerApp] Zainicjalizowano.")

    def _init_ui(self):
        layout = QVBoxLayout(self)

        folder_group = QGroupBox("Ustawienia skalowania")  # MODIFIED: Title
        folder_layout_internal = QVBoxLayout(folder_group)  # layout for the groupbox

        folder_select_layout = QHBoxLayout()
        self.folder_label = QLabel("Folder:")
        folder_select_layout.addWidget(self.folder_label)
        self.folder_path_edit = QLineEdit()
        self.folder_path_edit.setPlaceholderText(
            "Wybierz folder z obrazami do skalowania"
        )
        self.folder_path_edit.setReadOnly(True)
        folder_select_layout.addWidget(self.folder_path_edit)
        self.browse_button = QPushButton("Przeglądaj...")
        self.browse_button.clicked.connect(self.browse_folder)
        folder_select_layout.addWidget(self.browse_button)
        folder_layout_internal.addLayout(folder_select_layout)

        # --- MODIFIED: Target Dimension Control ---
        dim_control_layout = QHBoxLayout()
        dim_control_layout.addWidget(
            QLabel("Docelowy wymiar krótszego boku (px):")
        )  # MODIFIED: Changed text
        self.target_dimension_spinbox = QSpinBox()
        self.target_dimension_spinbox.setRange(50, 8192)  # Example range
        self.target_dimension_spinbox.setValue(DEFAULT_SCALLER_TARGET_DIMENSION)
        self.target_dimension_spinbox.setToolTip(
            "Obrazy, których krótszy bok jest mniejszy lub równy tej wartości, zostaną pominięte."  # MODIFIED: Changed text
        )
        dim_control_layout.addWidget(self.target_dimension_spinbox)
        dim_control_layout.addStretch()
        folder_layout_internal.addLayout(dim_control_layout)

        target_dim_info_label = QLabel(
            "Krótszy bok obrazu zostanie przeskalowany do wartości docelowej. Pliki, których oba wymiary są mniejsze lub równe wartości docelowej, zostaną pominięte."  # MODIFIED: Changed text
        )
        target_dim_info_label.setWordWrap(True)
        folder_layout_internal.addWidget(target_dim_info_label)
        # --- END MODIFIED ---

        layout.addWidget(folder_group)

        action_layout = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_processing)
        # self.start_button.setStyleSheet("background-color: lightgreen;") # Theme will handle
        action_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)
        # self.stop_button.setStyleSheet("background-color: lightcoral;") # Theme will handle
        action_layout.addWidget(self.stop_button)
        layout.addLayout(action_layout)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Status messages will go to main console
        # self.status_label = QLabel("Status (Scaller):")
        # layout.addWidget(self.status_label)
        # self.status_text_edit = QTextEdit()
        # self.status_text_edit.setReadOnly(True)
        # layout.addWidget(self.status_text_edit)

        self.setLayout(layout)

        # Style dla przycisków w ScallerApp
        self.start_button.setStyleSheet(BUTTON_STYLES["success"])
        self.stop_button.setStyleSheet(BUTTON_STYLES["stop"])
        self.browse_button.setStyleSheet(BUTTON_STYLES["default"])

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Wybierz folder", self.folder_path_edit.text() or str(Path.home())
        )
        if folder:
            self.folder_path_edit.setText(folder)
            # self.status_text_edit.clear() # Main console won't be cleared by this
            self.progress_bar.setValue(0)
            self.log_to_main_console.emit(f"[ScallerApp] Wybrano folder: {folder}")

    def start_processing(self):
        folder_path = self.folder_path_edit.text()
        if not folder_path or not os.path.isdir(folder_path):
            QMessageBox.warning(
                self, "Błąd (Scaller)", "Proszę wybrać prawidłowy folder."
            )
            return

        target_dim = (
            self.target_dimension_spinbox.value()
        )  # MODIFIED: Get value from spinbox

        self.start_button.setEnabled(False)
        self.browse_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.target_dimension_spinbox.setEnabled(
            False
        )  # MODIFIED: Disable spinbox during processing
        # self.status_text_edit.clear()
        self.progress_bar.setValue(0)
        self.log_to_main_console.emit(
            f"[ScallerApp] Rozpoczynanie skalowania w folderze: {folder_path} do wymiaru: {target_dim}px"  # MODIFIED
        )

        self.thread_scaller = QThread()
        self.worker_scaller = ScallerWorker(
            folder_path, target_dim
        )  # MODIFIED: Pass target_dim
        self.worker_scaller.moveToThread(self.thread_scaller)

        self.worker_scaller.finished.connect(self.thread_scaller.quit)
        self.worker_scaller.finished.connect(self.worker_scaller.deleteLater)
        self.thread_scaller.finished.connect(self.thread_scaller.deleteLater)
        self.thread_scaller.finished.connect(self.on_processing_finished)

        self.worker_scaller.progress.connect(self.update_progress_bar)  # Renamed
        self.worker_scaller.status_update.connect(self.update_status_log)  # Renamed
        self.worker_scaller.error_signal.connect(self.log_error_scaller)  # Renamed

        self.thread_scaller.started.connect(self.worker_scaller.run)
        self.thread_scaller.start()

    def stop_processing(self):
        if self.worker_scaller:
            self.worker_scaller.stop()
        self.stop_button.setEnabled(False)
        self.log_to_main_console.emit("[ScallerApp] Wysłano żądanie zatrzymania.")

    def on_processing_finished(self):
        self.start_button.setEnabled(True)
        self.browse_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.target_dimension_spinbox.setEnabled(True)  # MODIFIED: Re-enable spinbox
        # Message already sent by worker
        # if self.worker_scaller and not self.worker_scaller.is_running:
        #     self.update_status_log("Skalowanie zatrzymane przez użytkownika.")
        # else:
        #     self.update_status_log("Skalowanie zakończone.")

        self.worker_scaller = None
        self.thread_scaller = None
        self.log_to_main_console.emit("[ScallerApp] Wątek skalowania zakończony.")

    def update_progress_bar(self, current, total):  # Renamed
        if total > 0:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(current)
        else:
            self.progress_bar.setMaximum(1)
            self.progress_bar.setValue(0)

    def update_status_log(self, message):  # Renamed
        self.log_to_main_console.emit(f"[Scaller] {message}")

    def log_error_scaller(self, error_message):  # Renamed
        self.log_to_main_console.emit(f"[Scaller] BŁĄD: {error_message}")

    def cleanup(self):
        self.log_to_main_console.emit("[ScallerApp] Cleanup initiated.")
        if self.thread_scaller and self.thread_scaller.isRunning():
            if self.worker_scaller:
                self.worker_scaller.stop()
            self.thread_scaller.quit()
            if not self.thread_scaller.wait(3000):  # Wait 3 seconds
                self.log_to_main_console.emit(
                    "[ScallerApp] Wątek skalowania nie zakończył się w wyznaczonym czasie."
                )
            else:
                self.log_to_main_console.emit(
                    "[ScallerApp] Wątek skalowania zatrzymany."
                )


# --- END: scaller.py components ---


# --- BEGIN: fix_png.py components ---
class PngFixerImageProcessor(QThread):  # Renamed
    progress_updated = pyqtSignal(int)
    log_message = pyqtSignal(str)
    finished_processing = pyqtSignal(int, int)  # processed_count, problems_fixed

    def __init__(
        self,
        folder_path,
        recursive=True,
        replace_transparency=True,
        bg_color=(255, 255, 255),
    ):
        super().__init__()
        self.folder_path = folder_path
        self.recursive = recursive
        self.replace_transparency = replace_transparency
        self.bg_color = bg_color
        self.image_files = []
        self.processed_count = 0
        self.problems_fixed = 0
        self._is_running = True

    def stop(self):
        self._is_running = False
        self.log_message.emit("Zatrzymywanie procesu naprawy PNG...")

    def run(self):
        self.gather_image_files()
        total_files = len(self.image_files)

        if not self._is_running:
            self.log_message.emit("Proces naprawy PNG zatrzymany przed rozpoczęciem.")
            self.finished_processing.emit(0, 0)
            return

        if total_files == 0:
            self.log_message.emit(
                "Nie znaleziono plików obrazów PNG/GIF/WEBP/TGA w wybranym folderze."
            )
            self.finished_processing.emit(0, 0)
            return

        self.log_message.emit(
            f"Znaleziono {total_files} plików obrazów do potencjalnej naprawy."
        )
        QApplication.processEvents()

        for i, file_path in enumerate(self.image_files):
            if not self._is_running:
                self.log_message.emit(
                    f"Proces naprawy PNG zatrzymany po przetworzeniu {i} plików."
                )
                break
            try:
                fixed = self.process_image(file_path)
                if fixed:
                    self.problems_fixed += 1
                self.processed_count += 1
                progress = int((i + 1) / total_files * 100)
                self.progress_updated.emit(progress)
            except Exception as e:
                self.log_message.emit(
                    f"Błąd przy przetwarzaniu {Path(file_path).name}: {str(e)}"
                )

            if i % 5 == 0:
                QApplication.processEvents()  # UI refresh

        final_message = (
            f"Zakończono przetwarzanie PNG. Naprawiono {self.problems_fixed} z {self.processed_count} obrazów."
            if self._is_running
            else f"Przetwarzanie PNG zatrzymane. Dotychczas naprawiono {self.problems_fixed} z {self.processed_count} obrazów."
        )
        self.log_message.emit(final_message)
        self.finished_processing.emit(self.processed_count, self.problems_fixed)

    def gather_image_files(self):
        self.image_files = []
        # Relevant extensions for palette transparency issues
        image_extensions = {
            ".png",
            ".gif",
            ".webp",
            ".tga",
        }  # WEBP and TGA can also have palette issues

        if self.recursive:
            for root, _, files in os.walk(self.folder_path):
                if not self._is_running:
                    break
                for file in files:
                    if not self._is_running:
                        break
                    if Path(file).suffix.lower() in image_extensions:
                        self.image_files.append(os.path.join(root, file))
        else:
            for file in os.listdir(self.folder_path):
                if not self._is_running:
                    break
                file_p = Path(file)
                if file_p.is_file() and file_p.suffix.lower() in image_extensions:
                    self.image_files.append(os.path.join(self.folder_path, file))
        if not self._is_running:
            self.log_message.emit("Zbieranie plików przerwane (PNG Fixer).")

    def process_image(self, file_path):
        needs_fixing = False
        if not PIL_AVAILABLE:
            self.log_message.emit(
                "Biblioteka Pillow (PIL) nie jest dostępna. Nie można przetwarzać obrazów."
            )
            return False

        # Suppress specific PIL warnings related to palette transparency
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
            # For "Possibly corrupt EXIF data." or other non-critical warnings.
            warnings.filterwarnings("ignore", category=UserWarning)

            try:
                img = Image.open(file_path)
                original_format = img.format
                original_mode = img.mode

                # Check for paletted image with transparency key
                if img.mode == "P" and "transparency" in img.info:
                    self.log_message.emit(
                        f"Naprawianie obrazu paletowego: {Path(file_path).name}"
                    )
                    needs_fixing = True

                    img = img.convert(
                        "RGBA"
                    )  # Convert to RGBA to handle transparency properly

                    if self.replace_transparency:
                        background = Image.new("RGBA", img.size, self.bg_color)
                        # Paste using alpha channel of img as mask
                        background.alpha_composite(
                            img
                        )  # This is better for RGBA on RGBA
                        img = background.convert("RGB")  # Convert to RGB (no alpha)

                    # Save back in original format if possible, or a suitable one
                    save_format = (
                        original_format
                        if original_format in ["PNG", "GIF", "WEBP"]
                        else "PNG"
                    )
                    img.save(file_path, save_format)

                # Consider other potential fixes if added later
                # For example, re-saving to clean metadata or structure
                # elif original_format == "PNG" and some_other_condition:
                #    img.save(file_path, "PNG", optimize=True) # Example
                #    needs_fixing = True

            except FileNotFoundError:
                self.log_message.emit(
                    f"BŁĄD (PNG Fixer): Plik nie znaleziony - {file_path}"
                )
            except UnidentifiedImageError:
                self.log_message.emit(
                    f"BŁĄD (PNG Fixer): Nie można zidentyfikować formatu obrazu - {Path(file_path).name}"
                )
            except Exception as e:
                self.log_message.emit(
                    f"BŁĄD (PNG Fixer) przy {Path(file_path).name}: {str(e)}"
                )
                # Consider re-raising or specific error handling if needed
            finally:
                if "img" in locals() and hasattr(img, "close"):
                    img.close()

        return needs_fixing


class FixPngApp(QWidget):  # Renamed
    log_to_main_console = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.processor_fix_png = None  # Renamed
        self.selected_folder_fix_png = None  # Renamed
        self.initUI()
        self.log_to_main_console.emit("[FixPngApp] Zainicjalizowano.")

    def initUI(self):
        # self.setWindowTitle("Naprawa Obrazów z Przezroczystością")
        # self.setGeometry(100, 100, 700, 500) # Size managed by tab
        main_layout = QVBoxLayout(self)

        folder_group = QGroupBox("Wybór folderu")
        folder_layout = QHBoxLayout()
        self.folder_path_label = QLabel("Wybierz folder z obrazami:")
        folder_layout.addWidget(self.folder_path_label)
        self.select_folder_btn = QPushButton("Wybierz folder")
        self.select_folder_btn.clicked.connect(self.select_folder)
        folder_layout.addWidget(self.select_folder_btn)
        folder_group.setLayout(folder_layout)
        main_layout.addWidget(folder_group)

        options_group = QGroupBox("Opcje przetwarzania")
        options_layout = QVBoxLayout(options_group)  # Set layout for groupbox
        self.recursive_checkbox = QCheckBox("Przetwarzaj podfoldery (rekursywnie)")
        self.recursive_checkbox.setChecked(True)
        options_layout.addWidget(self.recursive_checkbox)

        self.transparency_checkbox = QCheckBox(
            "Zastąp przezroczystość kolorem tła (dla obrazów paletowych)"
        )
        self.transparency_checkbox.setChecked(True)
        options_layout.addWidget(self.transparency_checkbox)

        bg_color_layout = QHBoxLayout()
        bg_color_layout.addWidget(QLabel("Kolor tła:"))
        self.bg_color_group = QButtonGroup(self)  # Parent self
        self.white_bg_radio = QRadioButton("Biały")
        self.white_bg_radio.setChecked(True)
        self.bg_color_group.addButton(self.white_bg_radio)
        bg_color_layout.addWidget(self.white_bg_radio)
        self.black_bg_radio = QRadioButton("Czarny")
        self.bg_color_group.addButton(self.black_bg_radio)
        bg_color_layout.addWidget(self.black_bg_radio)
        bg_color_layout.addStretch()
        options_layout.addLayout(bg_color_layout)
        options_group.setLayout(
            options_layout
        )  # Ensure options_layout is set to options_group
        main_layout.addWidget(options_group)

        # Action buttons and Progress
        action_progress_layout = QHBoxLayout()
        self.process_btn = QPushButton("Rozpocznij kopiowanie")
        self.process_btn.setProperty("action", "success")
        self.process_btn.clicked.connect(self.start_processing)
        self.process_btn.setEnabled(False)
        action_progress_layout.addWidget(self.process_btn)

        self.stop_btn_fix_png = QPushButton("Zatrzymaj")
        self.stop_btn_fix_png.clicked.connect(self.stop_processing_fix_png)
        self.stop_btn_fix_png.setEnabled(False)
        action_progress_layout.addWidget(self.stop_btn_fix_png)
        main_layout.addLayout(action_progress_layout)

        progress_group = QGroupBox("Postęp")
        progress_layout_internal = QVBoxLayout(progress_group)
        self.progress_bar = QProgressBar()
        progress_layout_internal.addWidget(self.progress_bar)
        self.status_label = QLabel("Gotowy do przetwarzania.")  # Local status
        progress_layout_internal.addWidget(self.status_label)
        main_layout.addWidget(progress_group)

        # Log window for this tool's specific messages (if needed, or rely on main console)
        # log_group = QGroupBox("Dziennik działań (PNG Fixer)")
        # log_layout_internal = QVBoxLayout(log_group)
        # self.log_text_fix_png = QTextEdit() # Renamed
        # self.log_text_fix_png.setReadOnly(True)
        # log_layout_internal.addWidget(self.log_text_fix_png)
        # main_layout.addWidget(log_group)
        main_layout.addStretch()

        # Style dla przycisków w FixPngApp
        self.process_btn.setStyleSheet(BUTTON_STYLES["success"])
        self.stop_btn_fix_png.setStyleSheet(BUTTON_STYLES["stop"])
        self.select_folder_btn.setStyleSheet(BUTTON_STYLES["default"])

    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(
            self, "Wybierz folder", self.selected_folder_fix_png or str(Path.home())
        )
        if folder_path:
            self.folder_path_label.setText(
                f"Wybrany folder: {Path(folder_path).name}"
            )  # Show only name
            self.selected_folder_fix_png = folder_path
            self.process_btn.setEnabled(True)
            self.add_log_fix_png(f"Wybrano folder: {folder_path}")

    def start_processing(self):
        if self.selected_folder_fix_png:
            bg_color_tuple = (
                (255, 255, 255) if self.white_bg_radio.isChecked() else (0, 0, 0)
            )

            self.process_btn.setEnabled(False)
            self.select_folder_btn.setEnabled(False)
            self.stop_btn_fix_png.setEnabled(True)
            self.recursive_checkbox.setEnabled(False)
            self.transparency_checkbox.setEnabled(False)
            self.white_bg_radio.setEnabled(False)
            self.black_bg_radio.setEnabled(False)

            self.progress_bar.setValue(0)
            self.status_label.setText("Przetwarzanie (PNG Fixer)...")
            self.add_log_fix_png("Rozpoczynanie naprawy PNG...")

            self.processor_fix_png = PngFixerImageProcessor(
                self.selected_folder_fix_png,
                recursive=self.recursive_checkbox.isChecked(),
                replace_transparency=self.transparency_checkbox.isChecked(),
                bg_color=bg_color_tuple,
            )
            self.processor_fix_png.progress_updated.connect(
                self.update_progress_fix_png
            )  # Renamed
            self.processor_fix_png.log_message.connect(self.add_log_fix_png)
            self.processor_fix_png.finished_processing.connect(
                self.processing_finished_fix_png
            )  # Renamed
            self.processor_fix_png.start()
        else:
            self.add_log_fix_png("Najpierw wybierz folder!")

    def stop_processing_fix_png(self):
        if self.processor_fix_png and self.processor_fix_png.isRunning():
            self.processor_fix_png.stop()
            self.stop_btn_fix_png.setEnabled(False)  # Prevent multiple clicks
            self.add_log_fix_png("Wysłano żądanie zatrzymania naprawy PNG.")

    def update_progress_fix_png(self, value):  # Renamed
        self.progress_bar.setValue(value)

    def add_log_fix_png(self, message):  # Renamed
        self.log_to_main_console.emit(f"[PNG Fixer] {message}")
        # self.log_text_fix_png.append(message)
        # self.log_text_fix_png.verticalScrollBar().setValue(self.log_text_fix_png.verticalScrollBar().maximum())
        self.status_label.setText(message.splitlines()[-1])  # Show last line as status

    def processing_finished_fix_png(self, processed, fixed):  # Renamed
        self.status_label.setText(
            f"Zakończono! Przetworzono {processed} plików, naprawiono {fixed}."
        )
        self.process_btn.setEnabled(True)
        self.select_folder_btn.setEnabled(True)
        self.stop_btn_fix_png.setEnabled(False)
        self.recursive_checkbox.setEnabled(True)
        self.transparency_checkbox.setEnabled(True)
        self.white_bg_radio.setEnabled(True)
        self.black_bg_radio.setEnabled(True)
        self.add_log_fix_png(
            f"Naprawa PNG zakończona. Przetworzono: {processed}, Naprawiono: {fixed}."
        )
        self.processor_fix_png = None

    def cleanup(self):
        self.log_to_main_console.emit("[FixPngApp] Cleanup initiated.")
        if self.processor_fix_png and self.processor_fix_png.isRunning():
            self.processor_fix_png.stop()
            if not self.processor_fix_png.wait(3000):
                self.log_to_main_console.emit(
                    "[FixPngApp] Wątek naprawy PNG nie zakończył się."
                )
            else:
                self.log_to_main_console.emit(
                    "[FixPngApp] Wątek naprawy PNG zatrzymany."
                )


# --- END: fix_png.py components ---


# --- BEGIN: resolution_scanner.py components ---
class ResolutionScannerThread(QThread):
    progress_updated = pyqtSignal(int, int)
    file_problem_found = pyqtSignal(str, str)
    scan_completed = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    resolution_points_updated = pyqtSignal(list)

    def __init__(self, directory: str, min_size: int, max_size: int):
        super().__init__()
        self.directory = directory
        self.min_size = min_size
        self.max_size = max_size
        self._stopped = False
        self.resolution_points: List[Tuple[int, int]] = []

    def stop(self):
        self._stopped = True

    def run(self):
        try:
            image_files = []
            supported_formats = (
                ".png",
                ".jpg",
                ".jpeg",
                ".gif",
                ".bmp",
                ".tiff",
                ".tif",
                ".webp",
            )

            self.file_problem_found.emit(
                "INFO", f"Rozpoczynanie skanowania katalogu: {self.directory}"
            )
            QApplication.processEvents()

            for root, _, files in os.walk(self.directory):
                if self._stopped:
                    break
                for file in files:
                    if self._stopped:
                        break
                    if file.lower().endswith(supported_formats):
                        image_files.append(os.path.join(root, file))

            if self._stopped:
                self.error_occurred.emit(
                    "Skanowanie zatrzymane podczas wyszukiwania plików."
                )
                return

            total_files = len(image_files)
            if total_files == 0:
                self.error_occurred.emit(
                    "Nie znaleziono plików obrazów w wybranym katalogu."
                )
                return

            results = {
                "too_small": [],
                "too_large": [],
                "unreadable": [],
                "total_scanned": 0,
                "total_problems": 0,
                "total_unreadable": 0,
                "resolution_stats": {
                    "min_width": float("inf"),
                    "min_height": float("inf"),
                    "max_width": 0,
                    "max_height": 0,
                    "avg_width": 0,
                    "avg_height": 0,
                    "total_width": 0,
                    "total_height": 0,
                },
            }
            self.resolution_points.clear()

            for i, file_path in enumerate(image_files):
                if self._stopped:
                    break
                try:
                    with Image.open(file_path) as img:
                        width, height = img.size
                        self.resolution_points.append((width, height))
                        stats = results["resolution_stats"]
                        stats["min_width"] = min(stats["min_width"], width)
                        stats["min_height"] = min(stats["min_height"], height)
                        stats["max_width"] = max(stats["max_width"], width)
                        stats["max_height"] = max(stats["max_height"], height)
                        stats["total_width"] += width
                        stats["total_height"] += height

                        if width < self.min_size or height < self.min_size:
                            results["too_small"].append((file_path, (width, height)))
                            results["total_problems"] += 1
                            self.file_problem_found.emit(
                                file_path, f"Zbyt mała: {width}x{height}"
                            )
                        elif width > self.max_size or height > self.max_size:
                            results["too_large"].append((file_path, (width, height)))
                            results["total_problems"] += 1
                            self.file_problem_found.emit(
                                file_path, f"Zbyt duża: {width}x{height}"
                            )
                except UnidentifiedImageError:
                    results["unreadable"].append(file_path)
                    results["total_unreadable"] += 1
                    self.file_problem_found.emit(
                        file_path, "Błąd: Nie można zidentyfikować formatu."
                    )
                except Exception as e:
                    results["unreadable"].append(file_path)
                    results["total_unreadable"] += 1
                    self.file_problem_found.emit(file_path, f"Błąd analizy: {str(e)}")

                results["total_scanned"] += 1
                self.progress_updated.emit(i + 1, total_files)
                if i % 20 == 0:
                    QApplication.processEvents()  # UI refresh

            if self._stopped:
                self.error_occurred.emit("Skanowanie zatrzymane przez użytkownika.")
                return

            self.resolution_points_updated.emit(self.resolution_points)
            scanned_valid_images = (
                results["total_scanned"] - results["total_unreadable"]
            )
            if scanned_valid_images > 0:
                results["resolution_stats"]["avg_width"] = (
                    results["resolution_stats"]["total_width"] / scanned_valid_images
                )
                results["resolution_stats"]["avg_height"] = (
                    results["resolution_stats"]["total_height"] / scanned_valid_images
                )
            else:
                results["resolution_stats"]["min_width"] = 0
                results["resolution_stats"]["min_height"] = 0
            self.scan_completed.emit(results)
        except Exception as e:
            self.error_occurred.emit(
                f"Krytyczny błąd skanowania: {str(e)}\n{traceback.format_exc()}"
            )


class ResizeOptionsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Opcje przeskalowania")
        layout = QVBoxLayout(self)
        self.overwrite_rb = QRadioButton("Nadpisz oryginalne pliki (ryzykowne!)")
        self.save_in_subdir_rb = QRadioButton("Zapisz w podkatalogu '_resized'")
        self.save_in_subdir_rb.setChecked(True)
        layout.addWidget(QLabel("Wybierz metodę zapisu przeskalowanych plików:"))
        layout.addWidget(self.overwrite_rb)
        layout.addWidget(self.save_in_subdir_rb)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def get_selected_option(self):
        return "overwrite" if self.overwrite_rb.isChecked() else "subdir"


class ResolutionScannerWidget(QWidget):
    log_to_main_console = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scanner_thread_res = None  # Renamed
        self.all_resolution_points: List[Tuple[int, int]] = []
        self.scan_results_res: Dict = {}  # Renamed
        self.current_scan_dir = ""
        self.init_ui()
        self.log_to_main_console.emit("[ResScanner] Zainicjalizowano.")

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        controls_group = QGroupBox("Ustawienia skanowania")
        controls_layout_internal = QVBoxLayout(controls_group)

        dir_layout = QHBoxLayout()
        self.select_dir_btn = QPushButton("Wybierz katalog")
        self.select_dir_btn.clicked.connect(self.select_directory)
        dir_layout.addWidget(self.select_dir_btn)
        self.path_label = QLabel("Nie wybrano katalogu")
        self.path_label.setWordWrap(True)
        dir_layout.addWidget(self.path_label, 1)  # Give label more space
        controls_layout_internal.addLayout(dir_layout)

        resolution_limits_layout = QHBoxLayout()
        min_layout = QVBoxLayout()
        min_layout.addWidget(QLabel("Min. wymiar (px):"))
        self.min_size_spinbox = QSpinBox()
        self.min_size_spinbox.setRange(1, 20000)
        self.min_size_spinbox.setValue(500)
        min_layout.addWidget(self.min_size_spinbox)
        resolution_limits_layout.addLayout(min_layout)

        max_layout = QVBoxLayout()
        max_layout.addWidget(QLabel("Max. wymiar (px):"))
        self.max_size_spinbox = QSpinBox()
        self.max_size_spinbox.setRange(1, 20000)
        self.max_size_spinbox.setValue(4096)
        max_layout.addWidget(self.max_size_spinbox)
        resolution_limits_layout.addLayout(max_layout)
        controls_layout_internal.addLayout(resolution_limits_layout)
        main_layout.addWidget(controls_group)

        action_buttons_layout = QHBoxLayout()
        self.start_scan_btn = QPushButton("Rozpocznij skanowanie")
        self.start_scan_btn.clicked.connect(self.initiate_scan_from_current_path)
        self.start_scan_btn.setEnabled(False)
        action_buttons_layout.addWidget(self.start_scan_btn)

        self.stop_btn_res = QPushButton("Zatrzymaj skanowanie")  # Renamed
        self.stop_btn_res.clicked.connect(self.stop_scanning_res)  # Renamed
        self.stop_btn_res.setEnabled(False)
        action_buttons_layout.addWidget(self.stop_btn_res)

        self.resize_btn = QPushButton("Przeskaluj duże pliki")
        self.resize_btn.clicked.connect(self.resize_large_files)
        self.resize_btn.setEnabled(False)
        action_buttons_layout.addWidget(self.resize_btn)
        main_layout.addLayout(action_buttons_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)  # Initially hidden
        main_layout.addWidget(self.progress_bar)

        results_group = QGroupBox("Wyniki skanowania")
        results_layout_internal = QHBoxLayout(
            results_group
        )  # QHBoxLayout for group's content
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumWidth(300)
        results_layout_internal.addWidget(
            self.results_text, 1
        )  # Text edit takes 1 part

        self.figure = Figure(figsize=(5, 3), dpi=100)  # Slightly smaller default
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumWidth(400)
        results_layout_internal.addWidget(self.canvas, 2)  # Canvas takes 2 parts
        main_layout.addWidget(results_group)

        self.setLayout(main_layout)

        # Style dla przycisków w ResolutionScannerWidget
        self.start_scan_btn.setStyleSheet(BUTTON_STYLES["success"])
        self.stop_btn_res.setStyleSheet(BUTTON_STYLES["stop"])
        self.resize_btn.setStyleSheet(BUTTON_STYLES["warning"])
        self.select_dir_btn.setStyleSheet(BUTTON_STYLES["default"])

    def select_directory(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Wybierz katalog do skanowania",
            self.current_scan_dir or str(Path.home()),
            QFileDialog.Option.ShowDirsOnly,
        )
        if directory:
            self.current_scan_dir = directory
            self.path_label.setText(directory)
            self.start_scan_btn.setEnabled(True)
            self.log_to_main_console.emit(f"[ResScanner] Wybrano katalog: {directory}")

    def initiate_scan_from_current_path(self):
        if self.current_scan_dir and self.current_scan_dir != "Nie wybrano katalogu":
            self.start_scanning_res(self.current_scan_dir)  # Renamed
        else:
            QMessageBox.warning(
                self,
                "Brak katalogu (ResScanner)",
                "Najpierw wybierz katalog do skanowania.",
            )

    def start_scanning_res(self, directory: str):  # Renamed
        self.results_text.clear()
        self.all_resolution_points.clear()
        self.scan_results_res.clear()  # Use renamed var
        self.figure.clear()
        self.canvas.draw()

        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self._set_controls_enabled_res(False)  # Renamed

        self.log_to_main_console.emit(
            f"[ResScanner] Rozpoczynanie skanowania w: {directory}"
        )
        self.scanner_thread_res = ResolutionScannerThread(
            directory, self.min_size_spinbox.value(), self.max_size_spinbox.value()
        )
        self.scanner_thread_res.progress_updated.connect(self.update_progress_res)
        self.scanner_thread_res.file_problem_found.connect(
            self.add_file_problem_result_res
        )
        self.scanner_thread_res.scan_completed.connect(self.show_results_res)
        self.scanner_thread_res.error_occurred.connect(self.show_error_res)
        self.scanner_thread_res.resolution_points_updated.connect(
            self.update_resolution_points_res
        )
        self.scanner_thread_res.finished.connect(self.scan_thread_finished_res)
        self.scanner_thread_res.start()

    def _set_controls_enabled_res(self, enabled: bool):
        self.select_dir_btn.setEnabled(enabled)
        self.start_scan_btn.setEnabled(
            enabled
            and bool(
                self.current_scan_dir
                and self.current_scan_dir != "Nie wybrano katalogu"
            )
        )
        self.min_size_spinbox.setEnabled(enabled)
        self.max_size_spinbox.setEnabled(enabled)
        self.stop_btn_res.setEnabled(not enabled)  # Stop is enabled when scanning
        # resize_btn is handled separately based on results

    def scan_thread_finished_res(self):  # Renamed
        self._set_controls_enabled_res(True)
        self.progress_bar.setVisible(False)
        # resize_btn state handled in show_results_res
        self.log_to_main_console.emit("[ResScanner] Wątek skanowania zakończony.")

    def stop_scanning_res(self):  # Renamed
        if self.scanner_thread_res and self.scanner_thread_res.isRunning():
            self.scanner_thread_res.stop()
            self.log_to_main_console.emit(
                "[ResScanner] Wysłano żądanie zatrzymania skanowania."
            )
            self.results_text.append("\nZatrzymywanie skanowania...")
            self.stop_btn_res.setEnabled(False)  # Disable while stopping

    def update_progress_res(self, current: int, total: int):  # Renamed
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def add_file_problem_result_res(self, file_path: str, problem: str):  # Renamed
        log_entry = f"Problem: {Path(file_path).name if file_path != 'INFO' else ''} - {problem}"
        self.results_text.append(log_entry)
        self.log_to_main_console.emit(f"[ResScanner] {log_entry}")

    def update_resolution_points_res(self, points: list):  # Renamed
        self.all_resolution_points = points

    def resize_large_files(self):
        if not self.scan_results_res or not self.scan_results_res.get("too_large"):
            QMessageBox.information(
                self,
                "Informacja (ResScanner)",
                "Nie znaleziono plików do przeskalowania.",
            )
            return

        options_dialog = ResizeOptionsDialog(self)
        if not options_dialog.exec():
            self.log_to_main_console.emit(
                "[ResScanner] Anulowano opcje przeskalowania."
            )
            return

        resize_option = options_dialog.get_selected_option()
        target_dir_resize = ""  # Renamed
        if resize_option == "subdir":
            base_dir = self.current_scan_dir
            target_dir_resize = os.path.join(base_dir, "_resized")
            try:
                os.makedirs(target_dir_resize, exist_ok=True)
            except OSError as e:
                QMessageBox.critical(
                    self,
                    "Błąd tworzenia katalogu",
                    f"Nie można utworzyć {target_dir_resize}: {e}",
                )
                self.log_to_main_console.emit(
                    f"[ResScanner] BŁĄD: Nie można utworzyć {target_dir_resize}: {e}"
                )
                return

        files_to_resize = self.scan_results_res["too_large"]
        num_files = len(files_to_resize)

        msg_action = (
            "nadpisane"
            if resize_option == "overwrite"
            else f"zapisane w '{Path(target_dir_resize).name}'"
        )
        reply = QMessageBox.question(
            self,
            "Potwierdzenie (ResScanner)",
            f"Znaleziono {num_files} plików do przeskalowania. Zostaną one {msg_action}. Kontynuować?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            max_dim = self.max_size_spinbox.value()
            resized_count = 0
            errors = []

            progress_dialog_resize = QProgressDialog(
                "Przeskalowywanie plików...", "Anuluj", 0, num_files, self
            )
            progress_dialog_resize.setWindowModality(
                Qt.WindowModality.ApplicationModal
            )  # Make it modal
            progress_dialog_resize.setMinimumDuration(0)  # Show immediately
            progress_dialog_resize.show()

            self.log_to_main_console.emit(
                f"[ResScanner] Rozpoczynanie przeskalowywania {num_files} plików. Cel: max {max_dim}px. Opcja: {resize_option}."
            )

            for i, (file_path, (width, height)) in enumerate(files_to_resize):
                if progress_dialog_resize.wasCanceled():
                    errors.append("Operacja anulowana przez użytkownika.")
                    self.log_to_main_console.emit(
                        "[ResScanner] Przeskalowywanie anulowane przez użytkownika."
                    )
                    break
                progress_dialog_resize.setValue(i)
                progress_dialog_resize.setLabelText(
                    f"Przeskalowywanie: {Path(file_path).name}"
                )
                QApplication.processEvents()

                try:
                    with Image.open(file_path) as img:
                        original_format = img.format
                        if width > height:
                            new_width = max_dim
                            new_height = int(height * (max_dim / width))
                        else:
                            new_height = max_dim
                            new_width = int(width * (max_dim / height))

                        new_width = max(1, new_width)
                        new_height = max(1, new_height)
                        resized_img = img.resize(
                            (new_width, new_height), Image.Resampling.LANCZOS
                        )

                        save_path = file_path
                        if resize_option == "subdir":
                            relative_path = os.path.relpath(
                                file_path, self.current_scan_dir
                            )
                            save_path = os.path.join(target_dir_resize, relative_path)
                            os.makedirs(os.path.dirname(save_path), exist_ok=True)

                        save_params = {}
                        if original_format in ["JPEG", "WEBP"]:
                            save_params["quality"] = 95
                        elif original_format == "PNG":
                            save_params["optimize"] = True
                        resized_img.save(
                            save_path, format=original_format, **save_params
                        )
                        resized_count += 1
                except Exception as e:
                    error_msg = f"Błąd podczas przeskalowywania {Path(file_path).name}: {str(e)}"
                    errors.append(error_msg)
                    self.log_to_main_console.emit(f"[ResScanner] BŁĄD: {error_msg}")

            progress_dialog_resize.setValue(num_files)  # Ensure it reaches 100%
            progress_dialog_resize.close()  # Close it

            summary = f"Przeskalowano pomyślnie {resized_count} z {num_files} plików."
            if errors:
                summary += (
                    "\n\nWystąpiły błędy lub operacja została anulowana:\n"
                    + "\n".join(errors[:3])
                )  # Show first 3 errors
                if len(errors) > 3:
                    summary += f"\n...i {len(errors)-3} więcej."

            QMessageBox.information(
                self, "Podsumowanie przeskalowania (ResScanner)", summary
            )
            self.log_to_main_console.emit(
                f"[ResScanner] Podsumowanie: {summary.replace('Patron Saint (Default)n', ' ')}"  # Typo from original test?
            )

            if resize_option == "overwrite" and resized_count > 0:
                QMessageBox.information(
                    self,
                    "Informacja (ResScanner)",
                    "Pliki zostały nadpisane. Może być konieczne ponowne skanowanie.",
                )
                self.resize_btn.setEnabled(False)

    def show_results_res(self, results: Dict):  # Renamed
        self.scan_results_res = results
        self.resize_btn.setEnabled(len(results.get("too_large", [])) > 0)
        self.log_to_main_console.emit(
            f"[ResScanner] Skanowanie zakończone. Znaleziono problemów: {results.get('total_problems',0)}, nieczytelnych: {results.get('total_unreadable',0)}."
        )

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        if not self.all_resolution_points:
            ax.text(
                0.5,
                0.5,
                "Brak danych do wyświetlenia\n(żadne obrazy nie zostały poprawnie odczytane)",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )
        else:
            resolution_counts = {}
            for w, h in self.all_resolution_points:
                resolution_counts[(w, h)] = resolution_counts.get((w, h), 0) + 1

            plot_x = [w for (w, _), count in resolution_counts.items()]
            plot_y = [h for (_, h), count in resolution_counts.items()]
            # Scale size for visibility, ensure min size for single points
            plot_sizes = [max(10, count * 10) for count in resolution_counts.values()]

            if not plot_x or not plot_y:
                ax.text(
                    0.5,
                    0.5,
                    "Brak danych do wyświetlenia na wykresie",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )
            else:
                ax.scatter(
                    plot_x,
                    plot_y,
                    s=plot_sizes,
                    alpha=0.6,
                    c="skyblue",
                    edgecolors="dodgerblue",
                    linewidth=0.5,
                )
                ax.set_xlabel("Szerokość (px)")
                ax.set_ylabel("Wysokość (px)")
                ax.set_title("Rozkład rozdzielczości obrazów")
                min_val_plot = self.min_size_spinbox.value()  # Renamed
                max_val_plot = self.max_size_spinbox.value()  # Renamed
                ax.axhline(
                    y=min_val_plot,
                    color="orange",
                    linestyle="--",
                    alpha=0.7,
                    label=f"Min: {min_val_plot}px",
                )
                ax.axvline(x=min_val_plot, color="orange", linestyle="--", alpha=0.7)
                ax.axhline(
                    y=max_val_plot,
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    label=f"Max: {max_val_plot}px",
                )
                ax.axvline(x=max_val_plot, color="red", linestyle="--", alpha=0.7)
                if plot_x:  # Only add legend if there's data
                    ax.legend()
                ax.grid(True, linestyle=":", alpha=0.5)
                all_plot_x = (
                    plot_x + [min_val_plot, max_val_plot]
                    if plot_x
                    else [min_val_plot, max_val_plot]
                )
                all_plot_y = (
                    plot_y + [min_val_plot, max_val_plot]
                    if plot_y
                    else [min_val_plot, max_val_plot]
                )
                ax.set_xlim(
                    min(all_plot_x) * 0.9 if min(all_plot_x) > 0 else 0,
                    max(all_plot_x) * 1.1 + 1,
                )
                ax.set_ylim(
                    min(all_plot_y) * 0.9 if min(all_plot_y) > 0 else 0,
                    max(all_plot_y) * 1.1 + 1,
                )
        self.canvas.draw()

        stats_res = results.get("resolution_stats", {})  # Renamed
        summary = f"""Podsumowanie skanowania (ResScanner):
------------------------
Przeskanowano plików: {results.get('total_scanned', 0)}
Problemy z rozdzielczością: {results.get('total_problems', 0)}
Pliki nieczytelne: {results.get('total_unreadable', 0)}

Statystyki (dla odczytanych obrazów):
- Min szer.: {stats_res.get('min_width', 'N/A')}px, Max szer.: {stats_res.get('max_width', 'N/A')}px, Śr. szer.: {stats_res.get('avg_width', 0):.1f}px
- Min wys.: {stats_res.get('min_height', 'N/A')}px, Max wys.: {stats_res.get('max_height', 'N/A')}px, Śr. wys.: {stats_res.get('avg_height', 0):.1f}px
Unikalne rozdzielczości: {len(resolution_counts) if self.all_resolution_points and 'resolution_counts' in locals() else 0}
"""
        # Preserve existing problem logs if any, prepend summary
        current_problem_logs = self.results_text.toPlainText()
        self.results_text.setText(summary)
        if (
            current_problem_logs and "Problem:" in current_problem_logs
        ):  # If there were actual problem logs
            self.results_text.append(
                "\nSzczegółowe logi problemów z plikami:\n" + current_problem_logs
            )

        if results.get("too_small"):
            self.results_text.append(
                "\nPliki zbyt małe (<{}px):\n".format(self.min_size_spinbox.value())
                + "\n".join(
                    [
                        f"- {Path(fp).name} ({w}x{h})"
                        for fp, (w, h) in results["too_small"][:10]
                    ]
                )
                + (
                    f"\n...i {len(results['too_small'])-10} więcej."
                    if len(results["too_small"]) > 10
                    else ""
                )
            )
        if results.get("too_large"):
            self.results_text.append(
                "\nPliki zbyt duże (>{}px):\n".format(self.max_size_spinbox.value())
                + "\n".join(
                    [
                        f"- {Path(fp).name} ({w}x{h})"
                        for fp, (w, h) in results["too_large"][:10]
                    ]
                )
                + (
                    f"\n...i {len(results['too_large'])-10} więcej."
                    if len(results["too_large"]) > 10
                    else ""
                )
            )
        if results.get("unreadable"):
            self.results_text.append(
                "\nPliki nieczytelne:\n"
                + "\n".join([f"- {Path(fp).name}" for fp in results["unreadable"][:10]])
                + (
                    f"\n...i {len(results['unreadable'])-10} więcej."
                    if len(results["unreadable"]) > 10
                    else ""
                )
            )

    def show_error_res(self, error_msg: str):  # Renamed
        QMessageBox.critical(self, "Błąd krytyczny (ResScanner)", error_msg)
        self._set_controls_enabled_res(True)  # Ensure UI is usable
        self.progress_bar.setVisible(False)
        self.results_text.append(f"\nBŁĄD KRYTYCZNY: {error_msg}")
        self.log_to_main_console.emit(f"[ResScanner] BŁĄD KRYTYCZNY: {error_msg}")

    def cleanup(self):
        self.log_to_main_console.emit("[ResScanner] Cleanup initiated.")
        if self.scanner_thread_res and self.scanner_thread_res.isRunning():
            self.scanner_thread_res.stop()
            if not self.scanner_thread_res.wait(3000):
                self.log_to_main_console.emit(
                    "[ResScanner] Wątek skanowania rozdzielczości nie zakończył się."
                )
            else:
                self.log_to_main_console.emit(
                    "[ResScanner] Wątek skanowania rozdzielczości zatrzymany."
                )


# --- END: resolution_scanner.py components ---


# --- Main Application Window (Combined) ---
class CombinedApp(QMainWindow):
    def __init__(self, settings=None):
        super().__init__()
        self.settings = settings or {}
        self._setup_main_logger_and_qt_handler()
        self._apply_material_theme()
        self._create_main_menu()
        self._create_main_central_widget()
        self._create_main_status_bar()
        self._load_app_settings()
        # --- DODANE: ustawienie tytułu i ikony okna głównego ---
        self.setWindowTitle("Data Splitter")
        import os

        from PyQt6.QtGui import QIcon

        icon_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "resources",
            "img",
            "icon.png",
        )
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

    def _load_app_settings(self):
        """Wczytuje ustawienia aplikacji."""
        # Ustawienia domyślne
        self.settings = {
            "theme": "light",
            "language": "pl",
            "autosave": False,
            "last_directory": "",
            "window": {"width": 1200, "height": 800, "x": 100, "y": 100},
        }

    def _save_app_settings(self):
        """Wyłączone zapisywanie ustawień."""
        pass

    def closeEvent(self, event):
        """Obsługa zamknięcia aplikacji."""
        # Wyłączone zapisywanie ustawień
        event.accept()

    def _setup_main_logger_and_qt_handler(self):
        self.global_logger = logging.getLogger("CombinedApp")
        self.global_logger.handlers.clear()
        self.global_logger.setLevel(logging.DEBUG)
        self.global_logger.propagate = False

        # QtLogHandler for UI console
        self.qt_log_handler = QtLogHandler(self)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"
        )
        self.qt_log_handler.setFormatter(formatter)
        self.qt_log_handler.log_signal.connect(self._append_log_to_console)
        self.global_logger.addHandler(self.qt_log_handler)

    def _apply_material_theme(self):
        primary_color = "#007ACC"
        success_color = "#10B981"
        warning_color = "#DC2626"
        background = "#1E1E1E"
        surface = "#252526"
        border_color = "#3F3F46"
        text_color = "#CCCCCC"
        self.setStyleSheet(
            f"""
            QMainWindow, QDialog {{ background-color: {background}; color: {text_color}; }}
            QPushButton {{ background-color: {surface}; color: {text_color}; border: 1px solid {border_color}; border-radius: 2px; padding: 4px 12px; min-height: 24px; max-height: 24px; }}
            QPushButton:hover {{ background-color: #2A2D2E; }} QPushButton:pressed {{ background-color: #3E3E40; }}
            QPushButton[action="primary"] {{ background-color: {primary_color}; color: white; border: none; }}
            QPushButton[action="primary"]:hover {{ background-color: #1C97EA; }} QPushButton[action="primary"]:pressed {{ background-color: #005A9E; }}
            QTabWidget::pane {{ border: 1px solid {border_color}; background-color: {surface}; color: {text_color}; }}
            QTabBar::tab {{ background-color: {background}; color: {text_color}; padding: 5px 10px; margin-right: 2px; border-top-left-radius: 2px; border-top-right-radius: 2px; border: 1px solid {border_color}; min-width: 80px; max-height: 25px; }}
            QTabBar::tab:selected {{ background-color: {surface}; border-bottom-color: {surface}; }}
            QGroupBox {{ background-color: {surface}; color: {text_color}; border: 1px solid {border_color}; border-radius: 2px; margin-top: 14px; padding-top: 14px; font-weight: normal; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 5px; color: #FFFFFF; }}
            QLineEdit, QTextEdit, QTableWidget, QListWidget, QTreeWidget {{ background-color: #1C1C1C; color: {text_color}; border: 1px solid {border_color}; border-radius: 2px; padding: 2px; selection-background-color: #264F78; }}
            QTableWidget::item:selected {{ background-color: #264F78; color: #FFFFFF; }}
            QHeaderView::section {{ background-color: {surface}; color: {text_color}; padding: 2px; border: 1px solid {border_color}; }}
            QProgressBar {{ border: 1px solid {border_color}; background-color: {surface}; text-align: center; color: {text_color}; }}
            QProgressBar::chunk {{ background-color: {primary_color}; }}
            QLabel {{ color: {text_color}; }}
            QMenu {{ background-color: {surface}; color: {text_color}; border: 1px solid {border_color}; }}
            QMenu::item:selected {{ background-color: #264F78; }}
            QSpinBox, QComboBox {{ background-color: {surface}; color: {text_color}; border: 1px solid {border_color}; padding: 1px; }}
            QSlider::groove:horizontal {{ border: 1px solid {border_color}; height: 8px; background: {surface}; margin: 2px 0; border-radius: 4px; }}
            QSlider::handle:horizontal {{ background: {primary_color}; border: 1px solid {primary_color}; width: 14px; margin: -4px 0; border-radius: 7px; }}
            QSlider::sub-page:horizontal {{ background-color: {primary_color}; border-radius: 4px; }}
            QCheckBox {{ color: {text_color}; }}
            QRadioButton {{ color: {text_color}; }}
        """
        )

    def _create_main_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("Plik")
        exit_action = QAction("Zakończ", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        # Add other menus if needed (e.g., Pomoc -> O Programie)
        help_menu = menubar.addMenu("Pomoc")
        about_action = QAction("O programie", self)
        about_action.triggered.connect(self._show_about_combined)
        help_menu.addAction(about_action)

    def _create_main_central_widget(self):
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        self.tab_widget = QTabWidget()

        # Instantiate and add tabs
        self.data_splitter_tab = DataSplitterApp()
        self.data_splitter_tab.log_to_main_console.connect(self._append_log_to_console)
        self.tab_widget.addTab(self.data_splitter_tab, "Podział Danych (Splitter)")

        self.scaller_tab = ScallerApp()
        self.scaller_tab.log_to_main_console.connect(self._append_log_to_console)
        self.tab_widget.addTab(self.scaller_tab, "Skalowanie Obrazów (Scaller)")

        self.fix_png_tab = FixPngApp()
        self.fix_png_tab.log_to_main_console.connect(self._append_log_to_console)
        self.tab_widget.addTab(self.fix_png_tab, "Naprawa PNG (Fixer)")

        self.resolution_scanner_tab = ResolutionScannerWidget()
        self.resolution_scanner_tab.log_to_main_console.connect(
            self._append_log_to_console
        )
        self.tab_widget.addTab(self.resolution_scanner_tab, "Skaner Rozdzielczości")

        layout.addWidget(self.tab_widget)
        self._create_console_panel(layout)
        self.setCentralWidget(central_widget)

    def _create_main_status_bar(self):
        self.statusBar().showMessage("Gotowy")
        # Could add more info here if needed, e.g., from main_window.py's system info

    def _create_console_panel(self, parent_layout):
        # Simplified from main_window.py for brevity, can be expanded
        console_group = QGroupBox("Konsola Aplikacji")
        console_group.setFixedHeight(
            200
        )  # MODIFIED: Ustawienie stałej wysokości dla całej grupy
        console_layout_internal = QVBoxLayout(console_group)

        self.console_text = QTextEdit()
        self.console_text.setReadOnly(True)
        # Usunięto: self.console_text.setMinimumHeight(75)
        # Usunięto: self.console_text.setMaximumHeight(100)
        self.console_text.setStyleSheet(
            "font-family: 'Consolas', 'Courier New', monospace; font-size: 10px;"
        )
        console_layout_internal.addWidget(self.console_text)

        button_row_layout = QHBoxLayout()
        clear_btn = QPushButton("Wyczyść konsolę")
        clear_btn.clicked.connect(self.console_text.clear)
        button_row_layout.addWidget(clear_btn)
        button_row_layout.addStretch(1)
        console_layout_internal.addLayout(button_row_layout)

        parent_layout.addWidget(console_group)

        # Style dla przycisku czyszczenia konsoli
        clear_btn.setStyleSheet(BUTTON_STYLES["default"])

    def _append_log_to_console(self, message):
        if hasattr(self, "console_text"):
            self.console_text.append(message)
            # self.console_text.verticalScrollBar().setValue(self.console_text.verticalScrollBar().maximum())

    def _show_about_combined(self):
        QMessageBox.about(
            self,
            "O programie",
            "Zestaw Narzędzi Graficznych v1.0\n\n"
            "Aplikacja integrująca narzędzia:\n"
            "- Podział Danych (Data Splitter)\n"
            "- Skalowanie Obrazów (Image Scaller)\n"
            "- Naprawa PNG (PNG Fixer)\n"
            "- Skaner Rozdzielczości",
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Ensure PIL is available for some tools
    if not PIL_AVAILABLE:
        QMessageBox.critical(
            None,
            "Brak biblioteki Pillow",
            "Biblioteka Pillow (PIL) do przetwarzania obrazów nie jest zainstalowana. "
            "Niektóre funkcje mogą nie działać poprawnie. Zainstaluj ją używając: pip install Pillow",
        )
        # sys.exit(1) # Or allow to run with limited functionality

    main_win = CombinedApp()
    main_win.show()
    sys.exit(app.exec())
