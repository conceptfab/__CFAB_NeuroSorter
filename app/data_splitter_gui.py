import os
import random
import shutil
import sys
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# --- Konfiguracja ---
TRAIN_FOLDER_NAME = "__dane_treningowe"
VALID_FOLDER_NAME = "__dane_walidacyjne"
DEFAULT_TRAIN_SPLIT_PERCENT = 80
ALLOWED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff")

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

    def __init__(self, input_dir, output_dir, train_split_percent):
        super().__init__()
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.train_split_percent = train_split_percent
        self.is_cancelled = False

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
            valid_base_path = self.output_dir / VALID_FOLDER_NAME

            # Utwórz główne foldery wyjściowe, jeśli istnieją - wyczyść je? Na razie nadpisujemy
            if train_base_path.exists():
                self.progress_updated.emit(
                    0, f"Czyszczenie istniejącego folderu: {train_base_path}"
                )
                # shutil.rmtree(train_base_path) # Opcjonalnie: czyszczenie
            if valid_base_path.exists():
                self.progress_updated.emit(
                    0, f"Czyszczenie istniejącego folderu: {valid_base_path}"
                )
                # shutil.rmtree(valid_base_path) # Opcjonalnie: czyszczenie

            train_base_path.mkdir(parents=True, exist_ok=True)
            valid_base_path.mkdir(parents=True, exist_ok=True)

            # 1. Przeskanuj folder wejściowy i zbierz wszystkie pliki pogrupowane według podkategorii
            subfolders_to_process = []
            total_files_to_process = 0
            self.progress_updated.emit(5, "Skanowanie folderu wejściowego...")

            for category_dir in self.input_dir.iterdir():
                if category_dir.is_dir():
                    # Sprawdź tylko bezpośrednie podkatalogi pierwszego poziomu
                    relative_path = category_dir.relative_to(self.input_dir)
                    # Znajdź wszystkie pliki obrazów w tym podkatalogu
                    files_in_subdir = [
                        f
                        for f in category_dir.glob("*")
                        if f.is_file() and f.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
                    ]
                    if files_in_subdir:  # Przetwarzaj tylko jeśli są pliki obrazów
                        subfolders_to_process.append((relative_path, files_in_subdir))
                        total_files_to_process += len(files_in_subdir)

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

                num_files = len(files)
                num_train = int(num_files * self.train_split_percent / 100)
                num_valid = num_files - num_train

                self.progress_updated.emit(
                    int(10 + 80 * (processed_files_count / total_files_to_process)),
                    f"Przetwarzanie: {relative_path} ({num_train} tren., {num_valid} walid.)",
                )

                # Stwórz odpowiednie foldery wyjściowe
                current_train_path = train_base_path / relative_path
                current_valid_path = valid_base_path / relative_path
                current_train_path.mkdir(parents=True, exist_ok=True)
                current_valid_path.mkdir(parents=True, exist_ok=True)

                # Wymieszaj pliki losowo
                random.shuffle(files)

                # Podziel pliki
                train_files = files[:num_train]
                valid_files = files[num_train:]

                # Kopiuj pliki treningowe
                for file_path in train_files:
                    if self.is_cancelled:
                        break
                    try:
                        shutil.copy2(file_path, current_train_path / file_path.name)
                        processed_files_count += 1
                        # Aktualizuj progress rzadziej, żeby nie spowalniać UI
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
                        # Można zdecydować czy kontynuować czy przerwać
                        # continue

                if self.is_cancelled:
                    break

                # Kopiuj pliki walidacyjne
                for file_path in valid_files:
                    if self.is_cancelled:
                        break
                    try:
                        shutil.copy2(file_path, current_valid_path / file_path.name)
                        processed_files_count += 1
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
                        # continue

                if self.is_cancelled:
                    break  # Sprawdź ponownie po pętli walidacyjnej

            if self.is_cancelled:
                self.progress_updated.emit(0, "Przetwarzanie anulowane.")
                self.finished.emit("Anulowano.")
            else:
                self.progress_updated.emit(100, "Zakończono kopiowanie plików.")
                self.finished.emit("Przetwarzanie zakończone pomyślnie!")

        except ValueError as ve:
            self.error_occurred.emit(f"Błąd konfiguracji: {ve}")
            self.finished.emit(f"Błąd: {ve}")
        except Exception as e:
            self.error_occurred.emit(f"Niespodziewany błąd: {e}")
            self.finished.emit(f"Niespodziewany błąd: {e}")

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

        # Ustaw ikonę aplikacji
        icon_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "app",
            "img",
            "icon.png",
        )
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
        self.setGeometry(200, 200, 600, 450)  # x, y, width, height

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

        # --- Sekcja podziału danych ---
        split_layout = QHBoxLayout()
        split_label = QLabel("Podział Treningowe / Walidacyjne:")
        self.split_slider = QSlider(Qt.Orientation.Horizontal)
        self.split_slider.setMinimum(1)
        self.split_slider.setMaximum(99)
        self.split_slider.setValue(DEFAULT_TRAIN_SPLIT_PERCENT)
        self.split_slider.setTickInterval(10)
        self.split_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.split_value_label = QLabel(
            f"{DEFAULT_TRAIN_SPLIT_PERCENT}% / {100 - DEFAULT_TRAIN_SPLIT_PERCENT}%"
        )
        self.split_value_label.setMinimumWidth(80)  # Zapewnia stałą szerokość etykiety
        self.split_slider.valueChanged.connect(self.update_split_label)

        split_layout.addWidget(split_label)
        split_layout.addWidget(self.split_slider)
        split_layout.addWidget(self.split_value_label)
        layout.addLayout(split_layout)

        # --- Sekcja kontrolna i status ---
        control_layout = QHBoxLayout()
        self.start_button = QPushButton("Rozpocznij przetwarzanie")
        self.start_button.clicked.connect(self.start_processing)
        self.cancel_button = QPushButton("Anuluj")
        self.cancel_button.clicked.connect(self.cancel_processing)
        self.cancel_button.setEnabled(False)  # Początkowo nieaktywny

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
        layout.addWidget(log_label)
        layout.addWidget(self.log_edit)

        self.setLayout(layout)
        self.show()

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
            QMessageBox.information(
                self, "Zakończono", "Przetwarzanie danych zakończone pomyślnie."
            )
        else:
            QMessageBox.warning(self, "Anulowano", "Przetwarzanie zostało anulowane.")

        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.split_slider.setEnabled(True)  # Włącz suwak z powrotem
        self.processing_thread = None  # Wyczyść referencję do wątku

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
        self.progress_bar.setValue(0)
        self.log_edit.clear()  # Wyczyść logi przed nowym uruchomieniem

        split_percent = self.split_slider.value()
        self.log_message("=" * 30)
        self.log_message(
            f"Rozpoczynanie przetwarzania z podziałem {split_percent}% / {100-split_percent}%"
        )
        self.log_message(f"Źródło: {self.input_dir}")
        self.log_message(f"Cel: {self.output_dir}")
        self.log_message(f"Folder treningowy: {TRAIN_FOLDER_NAME}")
        self.log_message(f"Folder walidacyjny: {VALID_FOLDER_NAME}")
        self.log_message("=" * 30)

        # Uruchom przetwarzanie w osobnym wątku
        self.processing_thread = Worker(self.input_dir, self.output_dir, split_percent)
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


# --- Uruchomienie aplikacji ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = DataSplitterApp()
    sys.exit(app.exec())
