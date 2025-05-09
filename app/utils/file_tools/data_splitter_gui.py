import json
import os
import random
import shutil
import sys
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
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

# --- Konfiguracja ---
TRAIN_FOLDER_NAME = "__dane_treningowe"
VALID_FOLDER_NAME = "__dane_walidacyjne"
DEFAULT_TRAIN_SPLIT_PERCENT = 80
DEFAULT_FILES_PER_CATEGORY = 100  # Używane gdy nie można określić inaczej
ALLOWED_IMAGE_EXTENSIONS = (
    ".png",
    ".webp",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".gif",
    ".tiff",
    ".tif",
)

# --- Style ---
PRIMARY_COLOR = "#007ACC"
BACKGROUND = "#1E1E1E"
SURFACE = "#252526"
BORDER_COLOR = "#3F3F46"
TEXT_COLOR = "#CCCCCC"
HIGHLIGHT_COLOR = "#FF0000"


# --- Wątek roboczy do przetwarzania danych ---
class Worker(QThread):
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
        selected_categories=None,  # NOWY ARGUMENT
    ):
        super().__init__()
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.split_mode = split_mode
        self.split_value = split_value
        self.use_validation = use_validation
        self.selected_categories = selected_categories if selected_categories else []
        self.is_cancelled = False
        self.stats = {"train": {}, "valid": {}}
        self.json_report = {}
        # Do raportu w trybie "files"
        self.min_files_in_selection_for_report = 0
        self.folders_with_min_for_report = []

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

        if min_files_val == float("inf"):  # Żadne wybrane kategorie nie miały plików
            return 0, []
        return min_files_val, folders_with_min

    def run(self):
        try:
            self.progress_updated.emit(0, "Rozpoczynanie przetwarzania...")
            if not self.input_dir.is_dir():
                raise ValueError(f"Folder wejściowy nie istnieje: {self.input_dir}")
            if not self.output_dir.exists():
                self.output_dir.mkdir(parents=True, exist_ok=True)
            elif not self.output_dir.is_dir():
                raise ValueError(
                    f"Ścieżka wyjściowa istnieje, ale nie jest folderem: {self.output_dir}"
                )

            train_base_path = self.output_dir / TRAIN_FOLDER_NAME
            valid_base_path = (
                self.output_dir / VALID_FOLDER_NAME if self.use_validation else None
            )

            if train_base_path.exists():
                shutil.rmtree(train_base_path)  # Czyszczenie
            if valid_base_path and valid_base_path.exists():
                shutil.rmtree(valid_base_path)  # Czyszczenie

            train_base_path.mkdir(parents=True, exist_ok=True)
            if valid_base_path:
                valid_base_path.mkdir(parents=True, exist_ok=True)

            subfolders_to_process = []
            total_files_to_process = 0
            self.progress_updated.emit(
                5, "Skanowanie wybranych folderów wejściowych..."
            )

            if not self.selected_categories:
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
                    else:
                        self.progress_updated.emit(
                            0,
                            f"Info: Wybrana kategoria '{category_name}' jest pusta lub nie zawiera obrazów.",
                        )

            if not subfolders_to_process:
                raise ValueError(
                    "Nie znaleziono żadnych plików obrazów w wybranych i niepustych podfolderach."
                )

            self.progress_updated.emit(
                10,
                f"Znaleziono {total_files_to_process} plików w {len(subfolders_to_process)} wybranych podkategoriach.",
            )

            processed_files_count = 0

            if self.split_mode == "files":  # Oblicz min tylko raz dla trybu "files"
                (
                    self.min_files_in_selection_for_report,
                    self.folders_with_min_for_report,
                ) = self.get_min_files_in_selected_categories_for_processing()
                if (
                    self.min_files_in_selection_for_report == 0
                    and self.use_validation
                    and self.split_value > 0
                ):
                    self.progress_updated.emit(
                        0,
                        "Ostrzeżenie: W trybie 'Limit plików' nie znaleziono plików w wybranych kategoriach, co może wpłynąć na podział walidacyjny.",
                    )

            for relative_path, files in subfolders_to_process:
                if self.is_cancelled:
                    self.finished.emit("Anulowano.")
                    return

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
                        # Używamy self.min_files_in_selection_for_report, które jest min z wybranych
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

                self.progress_updated.emit(
                    int(
                        10
                        + 80
                        * (
                            processed_files_count / total_files_to_process
                            if total_files_to_process > 0
                            else 0
                        )
                    ),
                    f"Przetwarzanie: {relative_path} ({num_train} tren., {num_valid} walid.)",
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
                    if self.is_cancelled:
                        break
                    try:
                        shutil.copy2(file_path, current_train_path / file_path.name)
                        self.stats["train"][str(relative_path)] += 1
                        self.json_report[str(relative_path)]["train"].append(
                            file_path.name
                        )
                        processed_files_count += 1
                        # Aktualizacja progressu co jakiś czas
                    except Exception as e:
                        self.error_occurred.emit(
                            f"Błąd kopiowania {file_path} (trening): {e}"
                        )
                if self.is_cancelled:
                    break

                for file_path in valid_files_to_copy:
                    if self.is_cancelled:
                        break
                    try:
                        if current_valid_path:  # Powinno być, bo num_valid > 0
                            shutil.copy2(file_path, current_valid_path / file_path.name)
                            if self.use_validation:  # Dodatkowe zabezpieczenie
                                self.stats["valid"][str(relative_path)] += 1
                                self.json_report[str(relative_path)]["valid"].append(
                                    file_path.name
                                )
                            processed_files_count += 1
                            # Aktualizacja progressu
                    except Exception as e:
                        self.error_occurred.emit(
                            f"Błąd kopiowania {file_path} (walidacja): {e}"
                        )
                if self.is_cancelled:
                    break

            if self.is_cancelled:
                self.finished.emit("Anulowano.")
            else:
                self.progress_updated.emit(100, "Zakończono kopiowanie plików.")
                report = self._generate_report()
                json_path = self.output_dir / "raport_kopiowania.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(self.json_report, f, ensure_ascii=False, indent=2)
                self.finished.emit(
                    f"Przetwarzanie zakończone pomyślnie!\n\n{report}\n\nZapisano raport JSON: {json_path}"
                )

        except ValueError as ve:
            self.error_occurred.emit(f"Błąd konfiguracji: {ve}")
            self.finished.emit(f"Błąd: {ve}")
        except Exception as e:
            self.error_occurred.emit(f"Niespodziewany błąd: {e}")
            self.finished.emit(f"Niespodziewany błąd: {e}")

    def _generate_report(self):
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

    def cancel(self):
        self.is_cancelled = True
        self.progress_updated.emit(0, "Anulowanie...")


# --- Główna klasa aplikacji GUI ---
class DataSplitterApp(QWidget):
    def __init__(self):
        super().__init__()
        self.input_dir = ""
        self.output_dir = ""
        self.processing_thread = None
        self.files_list = []

        icon_path = Path("app/img/icon.png")  # Użyj Path dla spójności
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        else:
            print(f"Nie znaleziono pliku ikony: {icon_path}")

        self.initUI()
        self._apply_material_theme()
        self.update_files_limit_and_validation_based_on_selection()  # Inicjalizacja kontrolek

    def _apply_material_theme(self):
        self.setStyleSheet(
            f"""
            QWidget {{ background-color: {BACKGROUND}; color: {TEXT_COLOR}; }}
            QPushButton {{ background-color: {SURFACE}; border: 1px solid {BORDER_COLOR}; border-radius: 2px; padding: 4px 12px; min-height: 24px; max-height: 24px; }}
            QPushButton:hover {{ background-color: #2A2D2E; }} QPushButton:pressed {{ background-color: #3E3E40; }}
            QLineEdit {{ background-color: #1C1C1C; border: 1px solid {BORDER_COLOR}; border-radius: 2px; padding: 2px; }}
            QSlider::groove:horizontal {{ border: 1px solid {BORDER_COLOR}; height: 8px; background: {SURFACE}; margin: 2px 0; border-radius: 4px; }}
            QSlider::handle:horizontal {{ background: {PRIMARY_COLOR}; border: 1px solid {PRIMARY_COLOR}; width: 18px; margin: -2px 0; border-radius: 9px; }}
            QSlider::sub-page:horizontal {{ background: {PRIMARY_COLOR}; border-radius: 4px; }}
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
        self.folder_tree.clear()
        if not self.input_dir:
            self.update_files_limit_and_validation_based_on_selection()  # Aktualizuj kontrolki nawet jeśli brak folderu
            return

        root_path = Path(self.input_dir)
        tree_root_item = QTreeWidgetItem(self.folder_tree, [root_path.name])
        self.folder_tree.addTopLevelItem(tree_root_item)
        tree_root_item.setExpanded(True)

        try:  # Odłącz, aby uniknąć wielokrotnego podłączania lub błędów
            self.folder_tree.itemChanged.disconnect(self.on_folder_tree_item_changed)
        except TypeError:
            pass  # Nie było podłączone

        # Znajdź folder z najmniejszą liczbą plików
        min_files_count = float("inf")
        min_files_folder = None
        folder_counts = {}

        for category_dir in root_path.iterdir():
            if category_dir.is_dir():
                direct_file_count = sum(
                    1
                    for f in category_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
                )
                folder_counts[category_dir.name] = direct_file_count
                if direct_file_count < min_files_count:
                    min_files_count = direct_file_count
                    min_files_folder = category_dir.name

        # Dodaj foldery do drzewa
        for category_dir in root_path.iterdir():
            if category_dir.is_dir():
                direct_file_count = folder_counts[category_dir.name]
                display_text = f"{category_dir.name} ({direct_file_count} plików)"
                item = QTreeWidgetItem(tree_root_item, [display_text])
                item.setData(0, Qt.ItemDataRole.UserRole, category_dir.name)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(0, Qt.CheckState.Checked)  # Domyślnie zaznaczony

                # Wyróżnij folder z najmniejszą liczbą plików
                if category_dir.name == min_files_folder and min_files_count > 0:
                    item.setForeground(0, QColor(HIGHLIGHT_COLOR))

        self.folder_tree.itemChanged.connect(self.on_folder_tree_item_changed)
        self.update_files_limit_and_validation_based_on_selection()  # Zawsze aktualizuj po przebudowie drzewa

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
        selected_cats = self.get_selected_categories_names()

        min_files_in_selection, _ = self.get_min_files_in_selected_categories(
            selected_cats
        )

        if not selected_cats:
            self.files_spin.setEnabled(False)
            self.files_spin.setValue(1)  # Reset do minimum, gdy nic nie wybrano
            self.log_message("Wybierz kategorie, aby dostosować opcje podziału.")
        else:
            self.files_spin.setEnabled(
                self.mode_combo.currentIndex() == 1
            )  # Aktywny tylko w trybie limitu
            if min_files_in_selection > 0:
                self.files_spin.setMaximum(min_files_in_selection)
                # Nie zmieniaj wartości spinboxa, jeśli obecna jest nadal ważna
                if self.files_spin.value() > min_files_in_selection:
                    self.files_spin.setValue(min_files_in_selection)
            else:  # Brak plików w wybranych kategoriach
                self.files_spin.setMaximum(1)  # Ustaw max na 1 (min)
                self.files_spin.setValue(1)
            # self.log_message(f"Min. plików w wybranych: {min_files_in_selection}.")

        # Logika dla checkboxa walidacji
        current_files_limit_from_spin = self.files_spin.value()

        if self.mode_combo.currentIndex() == 0:  # Tryb procentowy
            can_create_validation = self.split_slider.value() < 100
            self.validation_check.setEnabled(
                can_create_validation and bool(selected_cats)
            )  # Muszą być wybrane kategorie
            if not can_create_validation:
                self.validation_check.setChecked(False)
                self.validation_label.setText("(100% tren.)")
            else:
                self.validation_label.setText(
                    f"({100 - self.split_slider.value()}% walid.)"
                )
        else:  # Tryb limitu plików
            # min_files_in_selection to już minimum z *aktualnie wybranych*
            can_create_validation = (
                current_files_limit_from_spin < min_files_in_selection
                and min_files_in_selection > 0
                and bool(selected_cats)
            )
            self.validation_check.setEnabled(can_create_validation)
            if not can_create_validation:
                self.validation_check.setChecked(False)
                if not selected_cats:
                    self.validation_label.setText("(wybierz kat.)")
                elif min_files_in_selection == 0:
                    self.validation_label.setText("(brak plików)")
                else:
                    self.validation_label.setText("(limit >= min.)")
            else:
                max_valid = min_files_in_selection - current_files_limit_from_spin
                self.validation_label.setText(f"(do {max_valid} walid.)")

        # Jeśli validation_check jest odznaczone przez użytkownika, nie zaznaczaj go automatycznie
        # Powyższa logika tylko ustawia `setEnabled` i tekst etykiety.

    def update_files_list(self):
        self.files_list_widget.clear()
        self.files_list = []
        if not self.input_dir:
            return
        root_path = Path(self.input_dir)
        for item in root_path.rglob("*"):  # rglob przechodzi przez podfoldery
            if item.is_file() and item.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS:
                relative_path = item.relative_to(root_path)
                self.files_list.append(item)
                self.files_list_widget.addItem(str(relative_path))

    def log_message(self, message):
        self.log_edit.append(message)
        self.log_edit.verticalScrollBar().setValue(
            self.log_edit.verticalScrollBar().maximum()
        )
        QApplication.processEvents()

    def select_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Wybierz folder źródłowy")
        if folder:
            self.input_dir = folder
            self.in_path_edit.setText(folder)
            self.log_message(f"Wybrano folder źródłowy: {folder}")
            self.update_folder_tree()  # To wywoła aktualizację kontrolek
            self.update_files_list()
            # update_files_limit_and_validation_based_on_selection() jest wywoływane w update_folder_tree

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Wybierz folder docelowy")
        if folder:
            self.output_dir = folder
            self.out_path_edit.setText(folder)
            self.log_message(f"Wybrano folder docelowy: {folder}")

    def update_progress(self, value, message):
        self.progress_bar.setValue(value)
        if message:
            self.log_message(message)

    def processing_finished(self, final_message):
        self.log_message(f"Status końcowy: {final_message}")
        QApplication.processEvents()  # Upewnij się, że log jest widoczny przed dialogiem

        if "Błąd" in final_message:
            QMessageBox.warning(
                self, "Błąd", f"Wystąpił błąd:\n{final_message.splitlines()[-1]}"
            )
        elif "Anulowano" not in final_message:
            dialog = ReportDialog(final_message, self)
            dialog.exec()
            # QMessageBox.information(self, "Zakończono", "Przetwarzanie danych zakończone.") # Dialog raportu wystarczy
        else:
            QMessageBox.information(
                self, "Anulowano", "Przetwarzanie zostało anulowane."
            )

        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.mode_combo.setEnabled(True)
        self.split_slider.setEnabled(self.mode_combo.currentIndex() == 0)
        self.files_spin.setEnabled(self.mode_combo.currentIndex() == 1)
        # Stan validation_check jest zarządzany przez update_files_limit...
        self.update_files_limit_and_validation_based_on_selection()  # Przywróć stan kontrolek
        self.processing_thread = None

    def processing_error(self, error_message):
        self.log_message(f"BŁĄD WĄTKU: {error_message}")

    def start_processing(self):
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

        # Sprawdzenie ścieżek
        input_path = Path(self.input_dir)
        output_path = Path(self.output_dir)
        try:  # Aby uniknąć błędu, jeśli output_dir nie jest jeszcze absolutny lub poprawny
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
                    return
        except (
            ValueError
        ):  # Np. gdy jedna ścieżka jest względna a druga absolutna w inny sposób
            pass  # Pozwól Workerowi na dalszą walidację

        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.mode_combo.setEnabled(False)
        self.split_slider.setEnabled(False)
        self.files_spin.setEnabled(False)
        self.validation_check.setEnabled(False)  # Całkowicie blokujemy kontrolki
        self.progress_bar.setValue(0)
        self.log_edit.clear()

        split_mode = "percent" if self.mode_combo.currentIndex() == 0 else "files"
        split_value = (
            self.split_slider.value()
            if split_mode == "percent"
            else self.files_spin.value()
        )
        use_validation = self.validation_check.isChecked()

        self.log_message("=" * 30)
        self.log_message(
            f"Rozpoczynanie (tryb: {split_mode}, wartość: {split_value}, walidacja: {use_validation})"
        )
        self.log_message(f"Wybrane kategorie: {', '.join(selected_categories)}")
        # ... (reszta logów) ...

        self.processing_thread = Worker(
            self.input_dir,
            self.output_dir,
            split_mode,
            split_value,
            use_validation,
            selected_categories,  # Przekazanie wybranych kategorii
        )
        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.finished.connect(self.processing_finished)
        self.processing_thread.error_occurred.connect(self.processing_error)
        self.processing_thread.start()

    def cancel_processing(self):
        if self.processing_thread and self.processing_thread.isRunning():
            self.log_message("Wysyłanie sygnału anulowania...")
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
        self.setWindowTitle("Raport kopiowania")
        self.setMinimumSize(800, 600)
        layout = QVBoxLayout()
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(QFont("Consolas", 10))  # Zmniejszona czcionka
        self.text_edit.setStyleSheet(
            f"""
            QTextEdit {{ background-color: {BACKGROUND}; color: {TEXT_COLOR}; border: 1px solid {BORDER_COLOR}; padding: 10px; }}
        """
        )
        self.text_edit.setHtml(self._format_report_to_html(report_text))  # Użyj HTML
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

    def _format_report_to_html(self, report_text):
        # Proste formatowanie - można rozbudować
        html = report_text.replace("\n", "<br>")
        html = html.replace("===", "<h3>").replace(" ===", "</h3>")  # Nagłówki
        # Można dodać więcej reguł, np. dla list
        return f"<body style='color:{TEXT_COLOR}; background-color:{BACKGROUND};'>{html}</body>"


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = DataSplitterApp()
    sys.exit(app.exec())
