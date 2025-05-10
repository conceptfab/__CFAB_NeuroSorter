import os
import sys
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QLabel,
    QProgressBar,
    QTextEdit,
    QHBoxLayout,
    QCheckBox,
    QGroupBox,
    QRadioButton,
    QButtonGroup,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PIL import Image
import warnings


class ImageProcessor(QThread):
    progress_updated = pyqtSignal(int)
    log_message = pyqtSignal(str)
    finished_processing = pyqtSignal(int, int)

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

    def run(self):
        # Znajdź wszystkie pliki obrazów
        self.gather_image_files()
        total_files = len(self.image_files)

        if total_files == 0:
            self.log_message.emit("Nie znaleziono plików obrazów w wybranym folderze.")
            self.finished_processing.emit(0, 0)
            return

        self.log_message.emit(
            f"Znaleziono {total_files} plików obrazów do przetworzenia."
        )

        # Przetwarzaj każdy obraz
        for i, file_path in enumerate(self.image_files):
            try:
                fixed = self.process_image(file_path)
                if fixed:
                    self.problems_fixed += 1
                self.processed_count += 1

                # Aktualizuj pasek postępu
                progress = int((i + 1) / total_files * 100)
                self.progress_updated.emit(progress)

            except Exception as e:
                self.log_message.emit(f"Błąd przy przetwarzaniu {file_path}: {str(e)}")

        self.log_message.emit(
            f"Zakończono przetwarzanie. Naprawiono {self.problems_fixed} z {self.processed_count} obrazów."
        )
        self.finished_processing.emit(self.processed_count, self.problems_fixed)

    def gather_image_files(self):
        """Zbiera wszystkie pliki obrazów z określonego folderu i podfolderów."""
        self.image_files = []
        image_extensions = {".png", ".gif", ".webp", ".tga"}

        if self.recursive:
            for root, _, files in os.walk(self.folder_path):
                for file in files:
                    if Path(file).suffix.lower() in image_extensions:
                        self.image_files.append(os.path.join(root, file))
        else:
            for file in os.listdir(self.folder_path):
                if Path(file).suffix.lower() in image_extensions:
                    self.image_files.append(os.path.join(self.folder_path, file))

    def process_image(self, file_path):
        """Przetwarza pojedynczy obraz i naprawia problemy z przezroczystością."""
        # Tłumienie ostrzeżeń PIL, ponieważ właśnie naprawiamy te błędy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Otwieramy obraz
            img = Image.open(file_path)
            original_format = img.format

            # Sprawdź, czy to obraz paletowy z przezroczystością
            needs_fixing = False
            if img.mode == "P" and "transparency" in img.info:
                needs_fixing = True
                self.log_message.emit(
                    f"Naprawianie obrazu paletowego z przezroczystością: {file_path}"
                )

                # Konwersja do RGBA
                img = img.convert("RGBA")

                # Jeśli chcemy zastąpić przezroczystość białym tłem
                if self.replace_transparency:
                    # Tworzymy nowy obraz z białym tłem
                    background = Image.new("RGBA", img.size, self.bg_color)
                    background.paste(img, (0, 0), img)
                    img = background.convert(
                        "RGB"
                    )  # Konwersja do RGB bez przezroczystości

                # Zapisz obraz z powrotem
                if original_format == "PNG":
                    img.save(file_path, "PNG")
                elif original_format == "GIF":
                    img.save(file_path, "GIF")
                else:
                    img.save(file_path)  # Domyślny format

            return needs_fixing


class ImageProcessorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.processor = None

    def initUI(self):
        self.setWindowTitle("Naprawa Obrazów z Przezroczystością")
        self.setGeometry(100, 100, 700, 500)

        # Główny widget i layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Folder selection
        folder_group = QGroupBox("Wybór folderu")
        folder_layout = QHBoxLayout()

        self.folder_path_label = QLabel("Wybierz folder z obrazami:")
        folder_layout.addWidget(self.folder_path_label)

        self.select_folder_btn = QPushButton("Wybierz folder")
        self.select_folder_btn.clicked.connect(self.select_folder)
        folder_layout.addWidget(self.select_folder_btn)

        folder_group.setLayout(folder_layout)
        main_layout.addWidget(folder_group)

        # Opcje
        options_group = QGroupBox("Opcje przetwarzania")
        options_layout = QVBoxLayout()

        self.recursive_checkbox = QCheckBox("Przetwarzaj podfoldery")
        self.recursive_checkbox.setChecked(True)
        options_layout.addWidget(self.recursive_checkbox)

        self.transparency_checkbox = QCheckBox("Zastąp przezroczystość kolorem tła")
        self.transparency_checkbox.setChecked(True)
        options_layout.addWidget(self.transparency_checkbox)

        # Opcje koloru tła
        bg_color_layout = QHBoxLayout()
        bg_color_layout.addWidget(QLabel("Kolor tła:"))

        self.bg_color_group = QButtonGroup()

        self.white_bg_radio = QRadioButton("Biały")
        self.white_bg_radio.setChecked(True)
        self.bg_color_group.addButton(self.white_bg_radio)
        bg_color_layout.addWidget(self.white_bg_radio)

        self.black_bg_radio = QRadioButton("Czarny")
        self.bg_color_group.addButton(self.black_bg_radio)
        bg_color_layout.addWidget(self.black_bg_radio)

        options_layout.addLayout(bg_color_layout)
        options_group.setLayout(options_layout)
        main_layout.addWidget(options_group)

        # Progress bar
        progress_group = QGroupBox("Postęp")
        progress_layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Gotowy do przetwarzania.")
        progress_layout.addWidget(self.status_label)

        progress_group.setLayout(progress_layout)
        main_layout.addWidget(progress_group)

        # Log window
        log_group = QGroupBox("Dziennik działań")
        log_layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)

        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)

        # Process button
        self.process_btn = QPushButton("Rozpocznij przetwarzanie")
        self.process_btn.clicked.connect(self.start_processing)
        self.process_btn.setEnabled(False)
        main_layout.addWidget(self.process_btn)

    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(
            self, "Wybierz folder z obrazami"
        )
        if folder_path:
            self.folder_path_label.setText(f"Wybrany folder: {folder_path}")
            self.process_btn.setEnabled(True)
            self.log_text.append(f"Wybrano folder: {folder_path}")
            self.selected_folder = folder_path

    def start_processing(self):
        if hasattr(self, "selected_folder"):
            # Ustal kolor tła
            if self.white_bg_radio.isChecked():
                bg_color = (255, 255, 255)
            else:
                bg_color = (0, 0, 0)

            # Wyłącz przyciski podczas przetwarzania
            self.process_btn.setEnabled(False)
            self.select_folder_btn.setEnabled(False)

            # Resetuj pasek postępu
            self.progress_bar.setValue(0)
            self.status_label.setText("Przetwarzanie...")

            # Utwórz i uruchom wątek przetwarzania
            self.processor = ImageProcessor(
                self.selected_folder,
                recursive=self.recursive_checkbox.isChecked(),
                replace_transparency=self.transparency_checkbox.isChecked(),
                bg_color=bg_color,
            )

            self.processor.progress_updated.connect(self.update_progress)
            self.processor.log_message.connect(self.add_log)
            self.processor.finished_processing.connect(self.processing_finished)

            self.processor.start()
        else:
            self.log_text.append("Najpierw wybierz folder!")

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def add_log(self, message):
        self.log_text.append(message)
        # Przewiń log na dół
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def processing_finished(self, processed, fixed):
        self.status_label.setText(
            f"Zakończono! Przetworzono {processed} plików, naprawiono {fixed} problemów."
        )
        self.process_btn.setEnabled(True)
        self.select_folder_btn.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessorApp()
    window.show()
    sys.exit(app.exec())
