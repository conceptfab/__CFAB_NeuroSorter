import json
import os
import random
import shutil
import sys
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QIcon
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
DEFAULT_FILES_PER_CATEGORY = 100
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
PRIMARY_COLOR = "#007ACC"  # Niebieski VS Code
BACKGROUND = "#1E1E1E"  # Ciemne tło
SURFACE = "#252526"  # Lekko jaśniejsze tło dla paneli
BORDER_COLOR = "#3F3F46"  # Kolor obramowania
TEXT_COLOR = "#CCCCCC"  # Kolor tekstu


# --- Wątek roboczy do przetwarzania danych ---
class Worker(QThread):
    """Wątek do wykonywania operacji plikowych w tle"""

    progress_updated = pyqtSignal(int, str)  # percentage, message
    finished = pyqtSignal(str)  # final message (success or error)
    error_occurred = pyqtSignal(str)

    def __init__(
        self, input_dir, output_dir, split_mode, split_value, use_validation=True
    ):
        super().__init__()
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.split_mode = split_mode  # "percent" lub "files"
        self.split_value = split_value  # procent lub liczba plików
        self.use_validation = use_validation
        self.is_cancelled = False
        # Statystyki kopiowania
        self.stats = {
            "train": {},  # kategoria -> liczba plików
            "valid": {},  # kategoria -> liczba plików
        }
        # --- DODANE: Słownik do raportu JSON ---
        self.json_report = {}

    def run(self):
        """Główna logika przetwarzania danych"""
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

            # Utwórz główne foldery wyjściowe
            if train_base_path.exists():
                self.progress_updated.emit(
                    0, f"Czyszczenie istniejącego folderu: {train_base_path}"
                )
            if valid_base_path and valid_base_path.exists():
                self.progress_updated.emit(
                    0, f"Czyszczenie istniejącego folderu: {valid_base_path}"
                )

            train_base_path.mkdir(parents=True, exist_ok=True)
            if valid_base_path:
                valid_base_path.mkdir(parents=True, exist_ok=True)

            # 1. Przeskanuj folder wejściowy i zbierz wszystkie pliki pogrupowane według podkategorii
            subfolders_to_process = []
            total_files_to_process = 0
            self.progress_updated.emit(5, "Skanowanie folderu wejściowego...")

            for category_dir in self.input_dir.iterdir():
                if category_dir.is_dir():
                    relative_path = category_dir.relative_to(self.input_dir)
                    files_in_subdir = [
                        f
                        for f in category_dir.glob("*")
                        if f.is_file() and f.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
                    ]
                    if files_in_subdir:
                        subfolders_to_process.append((relative_path, files_in_subdir))
                        total_files_to_process += len(files_in_subdir)
                        # Inicjalizuj statystyki dla kategorii
                        self.stats["train"][str(relative_path)] = 0
                        self.stats["valid"][str(relative_path)] = 0
                        # --- DODANE: Inicjalizuj raport JSON ---
                        self.json_report[str(relative_path)] = {
                            "train": [],
                            "valid": [],
                        }

            if not subfolders_to_process:
                raise ValueError(
                    "Nie znaleziono żadnych plików obrazów w podfolderach folderu wejściowego."
                )

            self.progress_updated.emit(
                10,
                f"Znaleziono {total_files_to_process} plików w {len(subfolders_to_process)} podkategoriach.",
            )

            # 2. Przetwarzaj każdą podkategorię
            processed_files_count = 0
            for relative_path, files in subfolders_to_process:
                if self.is_cancelled:
                    self.progress_updated.emit(
                        0, "Przetwarzanie anulowane przez użytkownika."
                    )
                    self.finished.emit("Anulowano.")
                    return

                # Wymieszaj pliki losowo
                random.shuffle(files)

                if self.split_mode == "percent":
                    # Tryb procentowy - losowy podział zgodnie z procentem
                    random.shuffle(files)
                    num_train = int(len(files) * self.split_value / 100)
                    num_valid = len(files) - num_train
                else:
                    # Tryb z limitem plików
                    random.shuffle(files)
                    if len(files) > self.split_value:
                        num_train = self.split_value
                        num_valid = 1
                    else:
                        num_train = len(files)
                        num_valid = 0

                self.progress_updated.emit(
                    int(10 + 80 * (processed_files_count / total_files_to_process)),
                    f"Przetwarzanie: {relative_path} ({num_train} tren., {num_valid} walid.)",
                )

                # Stwórz odpowiednie foldery wyjściowe
                current_train_path = train_base_path / relative_path
                current_valid_path = (
                    valid_base_path / relative_path
                    if valid_base_path and num_valid > 0
                    else None
                )
                current_train_path.mkdir(parents=True, exist_ok=True)
                if current_valid_path:
                    current_valid_path.mkdir(parents=True, exist_ok=True)

                # Podziel pliki
                train_files = files[:num_train]
                valid_files = (
                    files[num_train : num_train + num_valid] if num_valid > 0 else []
                )

                # Kopiuj pliki treningowe
                for file_path in train_files:
                    if self.is_cancelled:
                        break
                    try:
                        shutil.copy2(file_path, current_train_path / file_path.name)
                        processed_files_count += 1
                        self.stats["train"][str(relative_path)] += 1
                        # --- DODANE: Zapisz do raportu JSON ---
                        self.json_report[str(relative_path)]["train"].append(
                            file_path.name
                        )
                        if (
                            processed_files_count % 10 == 0
                            or processed_files_count == total_files_to_process
                        ):
                            self.progress_updated.emit(
                                int(
                                    10
                                    + 80
                                    * (processed_files_count / total_files_to_process)
                                ),
                                f"Kopiowanie: {file_path.name} do {TRAIN_FOLDER_NAME}",
                            )
                    except Exception as e:
                        self.error_occurred.emit(f"Błąd kopiowania {file_path}: {e}")

                if self.is_cancelled:
                    break

                # Kopiuj pliki walidacyjne (zawsze próbuj, nawet jeśli 0)
                for file_path in valid_files:
                    if self.is_cancelled:
                        break
                    try:
                        if current_valid_path:
                            shutil.copy2(file_path, current_valid_path / file_path.name)
                        processed_files_count += 1
                        self.stats["valid"][str(relative_path)] += 1
                        # --- DODANE: Zapisz do raportu JSON ---
                        self.json_report[str(relative_path)]["valid"].append(
                            file_path.name
                        )
                        if (
                            processed_files_count % 10 == 0
                            or processed_files_count == total_files_to_process
                        ):
                            self.progress_updated.emit(
                                int(
                                    10
                                    + 80
                                    * (processed_files_count / total_files_to_process)
                                ),
                                f"Kopiowanie: {file_path.name} do {VALID_FOLDER_NAME}",
                            )
                    except Exception as e:
                        self.error_occurred.emit(f"Błąd kopiowania {file_path}: {e}")

                if self.is_cancelled:
                    break

            if self.is_cancelled:
                self.progress_updated.emit(0, "Przetwarzanie anulowane.")
                self.finished.emit("Anulowano.")
            else:
                self.progress_updated.emit(100, "Zakończono kopiowanie plików.")
                # Przygotuj raport końcowy
                report = self._generate_report()
                # --- DODANE: Zapisz raport JSON ---
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
        """Generuje raport końcowy z kopiowania w formie drzewa folderów."""
        report = []
        report.append("=== RAPORT KOPIOWANIA ===")
        report.append("")

        # Podsumowanie dla każdej kategorii w formie drzewa
        for category in sorted(self.stats["train"].keys()):
            train_count = self.stats["train"][category]
            valid_count = self.stats["valid"].get(category, 0)

            # Dodaj główną kategorię
            report.append(f"{category}")
            report.append(f"├── {TRAIN_FOLDER_NAME}: {train_count} plików")
            report.append(f"└── {VALID_FOLDER_NAME}: {valid_count} plików")
            report.append("")

        # Podsumowanie ogólne
        total_train = sum(self.stats["train"].values())
        total_valid = sum(self.stats["valid"].values())
        report.append("=== PODSUMOWANIE ===")
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
        self.files_list = []  # Lista wszystkich plików

        # Ustaw ikonę aplikacji
        icon_path = os.path.join("app", "img", "icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        else:
            print(f"Nie znaleziono pliku ikony: {icon_path}")

        self.initUI()
        self._apply_material_theme()

    def _apply_material_theme(self):
        """Aplikuje styl Material Design do aplikacji."""
        self.setStyleSheet(
            f"""
            QWidget {{
                background-color: {BACKGROUND};
                color: {TEXT_COLOR};
            }}
            QPushButton {{
                background-color: {SURFACE};
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
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
            QLineEdit {{
                background-color: #1C1C1C;
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 2px;
                padding: 2px;
            }}
            QSlider::groove:horizontal {{
                border: 1px solid {BORDER_COLOR};
                height: 8px;
                background: {SURFACE};
                margin: 2px 0;
                border-radius: 4px;
            }}
            QSlider::handle:horizontal {{
                background: {PRIMARY_COLOR};
                border: 1px solid {PRIMARY_COLOR};
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }}
            QSlider::sub-page:horizontal {{
                background: {PRIMARY_COLOR};
                border-radius: 4px;
            }}
            QProgressBar {{
                border: 1px solid {BORDER_COLOR};
                background-color: {SURFACE};
                text-align: center;
                color: {TEXT_COLOR};
            }}
            QProgressBar::chunk {{
                background-color: {PRIMARY_COLOR};
            }}
            QTextEdit {{
                background-color: #1C1C1C;
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 2px;
                padding: 2px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10px;
            }}
            QLabel {{
                color: {TEXT_COLOR};
            }}
        """
        )

    def initUI(self):
        self.setWindowTitle("Przygotowanie Danych AI")
        self.setGeometry(200, 200, 800, 600)

        layout = QVBoxLayout()

        # --- Sekcja wyboru folderów ---
        folder_layout = QVBoxLayout()

        # Folder wejściowy
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

        # Folder wyjściowy
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

        # --- Zakładki ---
        self.tabs = QTabWidget()

        # Zakładka ze strukturą folderów
        self.folder_tree = QTreeWidget()
        self.folder_tree.setHeaderLabels(["Struktura folderów"])
        self.folder_tree.setColumnCount(1)
        self.tabs.addTab(self.folder_tree, "Struktura folderów")

        # Zakładka z listą plików
        self.files_list_widget = QListWidget()
        self.tabs.addTab(self.files_list_widget, "Lista plików")

        layout.addWidget(self.tabs)

        # --- Sekcja podziału danych ---
        split_layout = QVBoxLayout()

        # Tryb podziału
        mode_layout = QHBoxLayout()
        mode_label = QLabel("Tryb podziału:")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Podział procentowy", "Limit plików na kategorię"])
        self.mode_combo.currentIndexChanged.connect(self.update_split_mode)
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo)
        split_layout.addLayout(mode_layout)

        # Suwak procentowy
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

        # Pole liczby plików
        self.files_layout = QHBoxLayout()
        files_label = QLabel("Liczba plików na kategorię:")
        self.files_spin = QSpinBox()
        self.files_spin.setMinimum(1)
        self.files_spin.setMaximum(10000)
        self.files_spin.setValue(DEFAULT_FILES_PER_CATEGORY)
        self.files_spin.valueChanged.connect(
            lambda value: self.update_validation_checkbox(value)
        )
        self.files_layout.addWidget(files_label)
        self.files_layout.addWidget(self.files_spin)
        split_layout.addLayout(self.files_layout)

        # Checkbox dla folderu walidacyjnego
        validation_layout = QHBoxLayout()
        self.validation_check = QCheckBox("Utwórz folder walidacyjny")
        self.validation_check.setChecked(True)
        self.validation_check.stateChanged.connect(
            lambda state: self.update_validation_checkbox(self.files_spin.value())
        )
        self.validation_label = QLabel("")
        validation_layout.addWidget(self.validation_check)
        validation_layout.addWidget(self.validation_label)
        split_layout.addLayout(validation_layout)

        layout.addLayout(split_layout)

        # --- Sekcja kontrolna i status ---
        control_layout = QHBoxLayout()
        self.start_button = QPushButton("Rozpocznij przetwarzanie")
        self.start_button.clicked.connect(self.start_processing)
        self.cancel_button = QPushButton("Anuluj")
        self.cancel_button.clicked.connect(self.cancel_processing)
        self.cancel_button.setEnabled(False)

        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.cancel_button)
        layout.addLayout(control_layout)

        # Pasek postępu
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        # Logi / Status
        log_label = QLabel("Log:")
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumHeight(100)
        layout.addWidget(log_label)
        layout.addWidget(self.log_edit)

        self.setLayout(layout)
        self.show()

        # Inicjalizacja widoczności elementów
        self.update_split_mode(0)

    def update_split_mode(self, index):
        """Aktualizuje widoczność elementów interfejsu w zależności od wybranego trybu"""
        is_percent_mode = index == 0

        # Ukryj/pokaż elementy trybu procentowego
        for i in range(self.percent_layout.count()):
            widget = self.percent_layout.itemAt(i).widget()
            if widget:
                widget.setVisible(is_percent_mode)

        # Ukryj/pokaż elementy trybu z limitem plików
        for i in range(self.files_layout.count()):
            widget = self.files_layout.itemAt(i).widget()
            if widget:
                widget.setVisible(not is_percent_mode)

        # Jeśli przełączamy na tryb z limitem plików, zaktualizuj limit i checkbox
        if not is_percent_mode:
            self.update_files_limit()
        else:
            # W trybie procentowym zawsze włącz checkbox walidacji
            self.validation_check.setEnabled(True)

    def update_folder_tree(self):
        """Aktualizuje drzewo folderów na podstawie wybranego katalogu wejściowego."""
        self.folder_tree.clear()
        if not self.input_dir:
            return

        root_path = Path(self.input_dir)
        root_item = QTreeWidgetItem(self.folder_tree, [root_path.name])
        root_item.setExpanded(True)

        def count_files_in_folder(folder_path):
            """Liczy pliki w folderze i jego podfolderach."""
            count = 0
            for item in folder_path.iterdir():
                if item.is_file() and item.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS:
                    count += 1
                elif item.is_dir():
                    count += count_files_in_folder(item)
            return count

        def add_folder_to_tree(folder_path, parent_item):
            for item in folder_path.iterdir():
                if item.is_dir():
                    file_count = count_files_in_folder(item)
                    folder_name = f"{item.name} ({file_count} plików)"
                    folder_item = QTreeWidgetItem(parent_item, [folder_name])
                    folder_item.setExpanded(True)
                    add_folder_to_tree(item, folder_item)

        add_folder_to_tree(root_path, root_item)

    def update_files_list(self):
        """Aktualizuje listę plików na podstawie wybranego katalogu wejściowego."""
        self.files_list_widget.clear()
        self.files_list = []

        if not self.input_dir:
            return

        root_path = Path(self.input_dir)

        def add_files_to_list(folder_path):
            for item in folder_path.iterdir():
                if item.is_file() and item.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS:
                    relative_path = item.relative_to(root_path)
                    self.files_list.append(item)
                    self.files_list_widget.addItem(str(relative_path))
                elif item.is_dir():
                    add_files_to_list(item)

        add_files_to_list(root_path)

    def log_message(self, message):
        """Dodaje wiadomość do pola logów"""
        self.log_edit.append(message)
        # Przewiń na dół, aby widzieć najnowsze wpisy
        self.log_edit.verticalScrollBar().setValue(
            self.log_edit.verticalScrollBar().maximum()
        )
        QApplication.processEvents()  # Odśwież UI

    def select_input_folder(self):
        """Otwiera dialog wyboru folderu wejściowego"""
        folder = QFileDialog.getExistingDirectory(self, "Wybierz folder źródłowy")
        if folder:
            self.input_dir = folder
            self.in_path_edit.setText(folder)
            self.log_message(f"Wybrano folder źródłowy: {folder}")
            self.update_folder_tree()
            self.update_files_list()
            self.update_files_limit()  # Aktualizuj limit plików po wybraniu folderu

    def select_output_folder(self):
        """Otwiera dialog wyboru folderu wyjściowego"""
        folder = QFileDialog.getExistingDirectory(self, "Wybierz folder docelowy")
        if folder:
            self.output_dir = folder
            self.out_path_edit.setText(folder)
            self.log_message(f"Wybrano folder docelowy: {folder}")

    def update_split_label(self):
        """Aktualizuje etykietę procentową podziału"""
        train_percent = self.split_slider.value()
        valid_percent = 100 - train_percent
        self.split_value_label.setText(f"{train_percent}% / {valid_percent}%")

    def update_progress(self, value, message):
        """Aktualizuje pasek postępu i log"""
        self.progress_bar.setValue(value)
        if message:
            self.log_message(message)

    def processing_finished(self, final_message):
        """Wywoływana po zakończeniu pracy wątku"""
        self.log_message(f"Status końcowy: {final_message}")
        if "Błąd" in final_message:
            QMessageBox.warning(
                self, "Błąd", f"Wystąpił błąd podczas przetwarzania:\n{final_message}"
            )
        elif "Anulowano" not in final_message:
            # Wyświetl raport w osobnym oknie
            dialog = ReportDialog(final_message, self)
            dialog.exec()
            QMessageBox.information(
                self, "Zakończono", "Przetwarzanie danych zakończone pomyślnie."
            )
        else:
            QMessageBox.warning(self, "Anulowano", "Przetwarzanie zostało anulowane.")

        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.split_slider.setEnabled(True)
        self.processing_thread = None

    def processing_error(self, error_message):
        """Loguje błędy z wątku roboczego"""
        self.log_message(f"BŁĄD: {error_message}")
        # Można dodać dodatkowe powiadomienie dla użytkownika, jeśli błąd jest krytyczny

    def start_processing(self):
        """Rozpoczyna proces kopiowania i dzielenia plików"""
        if not self.input_dir or not Path(self.input_dir).is_dir():
            QMessageBox.warning(
                self, "Brak folderu", "Proszę wybrać prawidłowy folder źródłowy."
            )
            return
        if not self.output_dir:
            QMessageBox.warning(self, "Brak folderu", "Proszę wybrać folder docelowy.")
            return
        if self.input_dir == self.output_dir or Path(self.output_dir).is_relative_to(
            Path(self.input_dir)
        ):
            reply = QMessageBox.question(
                self,
                "Potwierdzenie ścieżki",
                f"Folder docelowy ('{self.output_dir}') jest taki sam jak źródłowy lub znajduje się wewnątrz niego.\n"
                f"Spowoduje to utworzenie folderów '{TRAIN_FOLDER_NAME}' i '{VALID_FOLDER_NAME}' wewnątrz '{self.output_dir}'.\n"
                "Czy na pewno chcesz kontynuować?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                return

        # Wyłącz przyciski i suwak na czas przetwarzania
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.split_slider.setEnabled(False)
        self.files_spin.setEnabled(False)
        self.mode_combo.setEnabled(False)
        self.validation_check.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_edit.clear()

        # Pobierz parametry
        split_mode = "percent" if self.mode_combo.currentIndex() == 0 else "files"
        split_value = (
            self.split_slider.value()
            if split_mode == "percent"
            else self.files_spin.value()
        )
        use_validation = self.validation_check.isChecked()

        self.log_message("=" * 30)
        if split_mode == "percent":
            self.log_message(
                f"Rozpoczynanie przetwarzania z podziałem {split_value}% / {100-split_value}%"
            )
        else:
            self.log_message(
                f"Rozpoczynanie przetwarzania z limitem {split_value} plików na kategorię"
            )
        self.log_message(f"Źródło: {self.input_dir}")
        self.log_message(f"Cel: {self.output_dir}")
        self.log_message(f"Folder treningowy: {TRAIN_FOLDER_NAME}")
        if use_validation:
            self.log_message(f"Folder walidacyjny: {VALID_FOLDER_NAME}")
        self.log_message("=" * 30)

        # Uruchom przetwarzanie w osobnym wątku
        self.processing_thread = Worker(
            self.input_dir, self.output_dir, split_mode, split_value, use_validation
        )
        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.finished.connect(self.processing_finished)
        self.processing_thread.error_occurred.connect(self.processing_error)
        self.processing_thread.start()

    def cancel_processing(self):
        """Anuluje bieżące przetwarzanie"""
        if self.processing_thread and self.processing_thread.isRunning():
            self.log_message("Wysyłanie sygnału anulowania...")
            self.processing_thread.cancel()
            self.cancel_button.setEnabled(
                False
            )  # Wyłącz od razu, żeby nie klikać wielokrotnie
            # Wątek sam się zakończy i wywoła `processing_finished`

    def closeEvent(self, event):
        """Obsługa zamknięcia okna - próba anulowania wątku"""
        if self.processing_thread and self.processing_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Zamykanie aplikacji",
                "Trwa przetwarzanie danych. Czy na pewno chcesz zakończyć i anulować?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.cancel_processing()
                # Poczekaj chwilę na zakończenie wątku (opcjonalne, może blokować zamknięcie)
                # self.processing_thread.wait(1000)
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    def get_min_files_in_category(self):
        """Zwraca minimalną liczbę plików w kategorii."""
        if not self.input_dir:
            return DEFAULT_FILES_PER_CATEGORY

        min_files = float("inf")
        root_path = Path(self.input_dir)

        for category_dir in root_path.iterdir():
            if category_dir.is_dir():
                files_count = len(
                    [
                        f
                        for f in category_dir.glob("*")
                        if f.is_file() and f.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
                    ]
                )
                if files_count > 0:
                    min_files = min(min_files, files_count)

        return min_files if min_files != float("inf") else DEFAULT_FILES_PER_CATEGORY

    def update_files_limit(self):
        """Aktualizuje limit plików na podstawie minimalnej liczby plików w kategorii."""
        min_files = self.get_min_files_in_category()
        self.files_spin.setValue(min_files)
        self.log_message(f"Wykryto minimalną liczbę plików w kategorii: {min_files}")

        # Jeśli tryb z limitem plików jest aktywny, sprawdź czy można utworzyć folder walidacyjny
        if self.mode_combo.currentIndex() == 1:  # Tryb z limitem plików
            self.update_validation_checkbox(min_files)

    def update_validation_checkbox(self, files_limit):
        """Aktualizuje stan checkboxa folderu walidacyjnego."""
        min_files = self.get_min_files_in_category()
        can_have_validation = files_limit < min_files

        self.validation_check.setEnabled(can_have_validation)
        if not can_have_validation:
            self.validation_check.setChecked(False)
            self.validation_label.setText("")
            self.log_message(
                "Folder walidacyjny wyłączony - liczba plików musi być mniejsza niż minimalna"
            )
        else:
            extra_files = min_files - files_limit
            self.validation_label.setText(f"(do {extra_files} plików walidacyjnych)")
            self.log_message(
                f"Folder walidacyjny dostępny - nadmiarowe pliki ({extra_files}) będą przeniesione do walidacji"
            )


class ReportDialog(QDialog):
    """Okno dialogowe do wyświetlania raportu kopiowania."""

    def __init__(self, report_text, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Raport kopiowania")
        self.setMinimumSize(800, 600)

        layout = QVBoxLayout()

        # Pole tekstowe z raportem
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(QFont("Consolas", 12))
        self.text_edit.setStyleSheet(
            """
            QTextEdit {
                background-color: #1E1E1E;
                color: #CCCCCC;
                border: 1px solid #3F3F46;
                border-radius: 4px;
                padding: 10px;
            }
        """
        )

        # Formatuj raport
        formatted_report = self._format_report(report_text)
        self.text_edit.setText(formatted_report)

        # Przycisk zamknięcia
        close_button = QPushButton("Zamknij")
        close_button.setStyleSheet(
            """
            QPushButton {
                background-color: #007ACC;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #1C97EA;
            }
        """
        )
        close_button.clicked.connect(self.accept)

        layout.addWidget(self.text_edit)
        layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignCenter)

        self.setLayout(layout)

    def _format_report(self, report_text):
        """Formatuje raport dla lepszej czytelności."""
        lines = report_text.split("\n")
        formatted_lines = []

        for line in lines:
            if line.startswith("==="):
                # Nagłówki
                formatted_lines.append(f'<h2 style="color: #007ACC;">{line}</h2>')
            elif line.strip() == "":
                # Puste linie
                formatted_lines.append("<br>")
            elif ":" in line and "plików" in line:
                # Linie z liczbą plików
                parts = line.split(":")
                formatted_lines.append(
                    f'<span style="color: #CCCCCC;">{parts[0]}:</span>'
                    f'<span style="color: #4EC9B0;">{parts[1]}</span>'
                )
            else:
                # Zwykłe linie
                formatted_lines.append(f'<span style="color: #CCCCCC;">{line}</span>')

        return "<br>".join(formatted_lines)


# --- Uruchomienie aplikacji ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = DataSplitterApp()
    sys.exit(app.exec())
