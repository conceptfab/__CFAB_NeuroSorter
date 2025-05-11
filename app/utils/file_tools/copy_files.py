import os
import shutil
import sys

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# Lista popularnych rozszerzeń plików graficznych
IMAGE_EXTENSIONS = [
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".tiff",
    ".webp",
    ".heic",
    ".heif",
]


class Worker(QThread):
    """
    Wątek roboczy do obsługi operacji plikowych, aby nie blokować GUI.
    """

    progress_update = pyqtSignal(int, str)  # (procent, wiadomość)
    finished_signal = pyqtSignal(int, int)  # (liczba znalezionych, liczba skopiowanych)
    error_signal = pyqtSignal(str)

    def __init__(self, source_dir, dest_dir, image_extensions):
        super().__init__()
        self.source_dir = source_dir
        self.dest_dir = dest_dir
        self.image_extensions = image_extensions
        self.running = True

    def run(self):
        found_count = 0
        copied_count = 0
        total_files_to_scan = 0

        try:
            # Pierwsze przejście, aby policzyć pliki do przeskanowania (dla paska postępu)
            for _, _, files in os.walk(self.source_dir):
                if not self.running:
                    self.progress_update.emit(100, "Przerwano przez użytkownika.")
                    self.finished_signal.emit(found_count, copied_count)
                    return
                total_files_to_scan += len(files)

            processed_files = 0

            for root, _, files in os.walk(self.source_dir):
                if not self.running:
                    self.progress_update.emit(100, "Przerwano przez użytkownika.")
                    self.finished_signal.emit(found_count, copied_count)
                    return

                for filename in files:
                    if not self.running:
                        self.progress_update.emit(100, "Przerwano przez użytkownika.")
                        self.finished_signal.emit(found_count, copied_count)
                        return

                    processed_files += 1
                    if total_files_to_scan > 0:
                        progress_percent = int(
                            (processed_files / total_files_to_scan) * 100
                        )
                    else:
                        progress_percent = 0

                    self.progress_update.emit(
                        progress_percent, f"Skanowanie: {filename}"
                    )

                    file_path = os.path.join(root, filename)
                    file_ext = os.path.splitext(filename)[1].lower()

                    if file_ext in self.image_extensions:
                        found_count += 1
                        dest_file_path_base = os.path.join(self.dest_dir, filename)
                        dest_file_path = dest_file_path_base

                        # Obsługa konfliktów nazw
                        copy_num = 1
                        base, ext = os.path.splitext(filename)
                        while os.path.exists(dest_file_path):
                            new_filename = f"{base}_copy_{copy_num}{ext}"
                            dest_file_path = os.path.join(self.dest_dir, new_filename)
                            copy_num += 1

                        try:
                            shutil.copy2(file_path, dest_file_path)
                            copied_count += 1
                            self.progress_update.emit(
                                progress_percent,
                                f"Skopiowano: {filename} -> {os.path.basename(dest_file_path)}",
                            )
                        except Exception as e:
                            self.progress_update.emit(
                                progress_percent, f"BŁĄD kopiowania {filename}: {e}"
                            )

            if self.running:  # Jeśli nie przerwano
                self.progress_update.emit(
                    100, "Zakończono przeszukiwanie i kopiowanie."
                )
            self.finished_signal.emit(found_count, copied_count)

        except Exception as e:
            self.error_signal.emit(f"Wystąpił krytyczny błąd: {e}")
            self.finished_signal.emit(
                found_count, copied_count
            )  # Zakończ nawet przy błędzie

    def stop(self):
        self.running = False


class ImageCopierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.source_dir = ""
        self.dest_dir = ""
        self.worker_thread = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Kopiowanie Plików Graficznych")
        self.setGeometry(300, 300, 600, 450)  # x, y, width, height

        layout = QVBoxLayout()

        # Folder źródłowy
        source_layout = QHBoxLayout()
        self.source_label = QLabel("Folder źródłowy:")
        self.source_entry = QLineEdit()
        self.source_entry.setReadOnly(True)
        self.source_button = QPushButton("Wybierz...")
        self.source_button.clicked.connect(self.select_source_dir)
        source_layout.addWidget(self.source_label)
        source_layout.addWidget(self.source_entry)
        source_layout.addWidget(self.source_button)
        layout.addLayout(source_layout)

        # Folder docelowy
        dest_layout = QHBoxLayout()
        self.dest_label = QLabel("Folder docelowy: ")
        self.dest_entry = QLineEdit()
        self.dest_entry.setReadOnly(True)
        self.dest_button = QPushButton("Wybierz...")
        self.dest_button.clicked.connect(self.select_dest_dir)
        dest_layout.addWidget(self.dest_label)
        dest_layout.addWidget(self.dest_entry)
        dest_layout.addWidget(self.dest_button)
        layout.addLayout(dest_layout)

        # Przycisk Start i Anuluj
        action_layout = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_copying)
        self.cancel_button = QPushButton("Anuluj")
        self.cancel_button.clicked.connect(self.cancel_copying)
        self.cancel_button.setEnabled(False)
        action_layout.addWidget(self.start_button)
        action_layout.addWidget(self.cancel_button)
        layout.addLayout(action_layout)

        # Pasek postępu
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        # Log
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        layout.addWidget(self.log_area)

        self.setLayout(layout)

    def select_source_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Wybierz folder źródłowy")
        if directory:
            self.source_dir = directory
            self.source_entry.setText(directory)
            self.log_message(f"Wybrano folder źródłowy: {directory}")

    def select_dest_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Wybierz folder docelowy")
        if directory:
            self.dest_dir = directory
            self.dest_entry.setText(directory)
            self.log_message(f"Wybrano folder docelowy: {directory}")

    def log_message(self, message):
        self.log_area.append(message)
        QApplication.processEvents()  # Aby GUI się odświeżało

    def start_copying(self):
        if not self.source_dir:
            QMessageBox.warning(self, "Brak folderu", "Proszę wybrać folder źródłowy.")
            return
        if not self.dest_dir:
            QMessageBox.warning(self, "Brak folderu", "Proszę wybrać folder docelowy.")
            return
        if self.source_dir == self.dest_dir:
            QMessageBox.warning(
                self,
                "Konflikt folderów",
                "Folder źródłowy i docelowy nie mogą być takie same.",
            )
            return

        if not os.path.exists(self.source_dir):
            QMessageBox.critical(
                self, "Błąd", f"Folder źródłowy nie istnieje: {self.source_dir}"
            )
            return

        # Utwórz folder docelowy, jeśli nie istnieje
        try:
            os.makedirs(self.dest_dir, exist_ok=True)
        except OSError as e:
            QMessageBox.critical(
                self,
                "Błąd tworzenia folderu",
                f"Nie można utworzyć folderu docelowego {self.dest_dir}: {e}",
            )
            return

        self.log_area.clear()
        self.log_message(
            f"Rozpoczynanie kopiowania z {self.source_dir} do {self.dest_dir}..."
        )
        self.log_message(f"Szukane rozszerzenia: {', '.join(IMAGE_EXTENSIONS)}")

        self.progress_bar.setValue(0)
        self.start_button.setEnabled(False)
        self.source_button.setEnabled(False)
        self.dest_button.setEnabled(False)
        self.cancel_button.setEnabled(True)

        self.worker_thread = Worker(self.source_dir, self.dest_dir, IMAGE_EXTENSIONS)
        self.worker_thread.progress_update.connect(self.update_progress)
        self.worker_thread.finished_signal.connect(self.on_finished)
        self.worker_thread.error_signal.connect(self.on_error)
        self.worker_thread.start()

    def cancel_copying(self):
        if self.worker_thread and self.worker_thread.isRunning():
            self.log_message("Próba anulowania operacji...")
            self.worker_thread.stop()
            self.cancel_button.setEnabled(False)  # Zapobiegaj wielokrotnemu klikaniu

    def update_progress(self, percent, message):
        self.progress_bar.setValue(percent)
        self.log_message(message)

    def on_finished(self, found_count, copied_count):
        final_message = f"Zakończono. Znaleziono plików graficznych: {found_count}. Skopiowano: {copied_count}."
        self.log_message(final_message)
        if (
            self.worker_thread and not self.worker_thread.running
        ):  # Jeśli było przerwane
            self.log_message("Operacja została przerwana przez użytkownika.")
        QMessageBox.information(self, "Zakończono", final_message)

        self.progress_bar.setValue(
            100 if found_count > 0 or copied_count > 0 else 0
        )  # Ustaw na 100% lub 0%
        self.start_button.setEnabled(True)
        self.source_button.setEnabled(True)
        self.dest_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.worker_thread = None  # Usuń referencję do wątku

    def on_error(self, error_message):
        self.log_message(f"BŁĄD KRYTYCZNY: {error_message}")
        QMessageBox.critical(self, "Błąd krytyczny", error_message)
        self.start_button.setEnabled(True)
        self.source_button.setEnabled(True)
        self.dest_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.worker_thread = None

    def closeEvent(self, event):
        """Obsługa zamknięcia okna."""
        if self.worker_thread and self.worker_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Zamykanie",
                "Operacja kopiowania jest w toku. Czy na pewno chcesz zamknąć?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.worker_thread.stop()
                self.worker_thread.wait()  # Poczekaj na zakończenie wątku
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def run_copier():
    app = QApplication(sys.argv)
    ex = ImageCopierApp()
    ex.show()
    sys.exit(app.exec())
