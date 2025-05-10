import sys
import os
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLineEdit,
    QFileDialog,
    QProgressBar,
    QTextEdit,
    QLabel,
    QMessageBox,
)
from PyQt6.QtCore import QThread, pyqtSignal, QObject
from PIL import Image, UnidentifiedImageError

TARGET_DIMENSION = 300
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp")


class Worker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int, int)  # current, total
    status_update = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path
        self.is_running = True

    def stop(self):
        self.is_running = False
        self.status_update.emit("Proces zatrzymywany...")

    def run(self):
        image_files = []
        self.status_update.emit(
            f"Wyszukiwanie plików graficznych w: {self.folder_path}..."
        )
        for root, _, files in os.walk(self.folder_path):
            if not self.is_running:
                self.status_update.emit("Przerwano wyszukiwanie plików.")
                self.finished.emit()
                return
            for file in files:
                if file.lower().endswith(SUPPORTED_EXTENSIONS):
                    image_files.append(os.path.join(root, file))

        if not image_files:
            self.status_update.emit("Nie znaleziono plików graficznych.")
            self.finished.emit()
            return

        total_files = len(image_files)
        self.status_update.emit(
            f"Znaleziono {total_files} plików graficznych. Rozpoczynanie przetwarzania..."
        )
        self.progress.emit(0, total_files)

        processed_count = 0
        skipped_count = 0
        resized_count = 0

        for i, file_path in enumerate(image_files):
            if not self.is_running:
                self.status_update.emit(f"Przerwano przetwarzanie po {i} plikach.")
                break

            self.status_update.emit(f"Przetwarzanie: {os.path.basename(file_path)}")
            try:
                with Image.open(file_path) as img:
                    width, height = img.size

                    # 1: Sprawdza rozdzielczość pliku, jeśli jedna z wartości ma 300 lub mniej px, pomija ten plik
                    if width <= TARGET_DIMENSION or height <= TARGET_DIMENSION:
                        self.status_update.emit(
                            f"Pominięto (wymiar <= {TARGET_DIMENSION}px): {file_path} ({width}x{height})"
                        )
                        skipped_count += 1
                        processed_count += 1
                        self.progress.emit(processed_count, total_files)
                        continue

                    # Jeśli doszliśmy tutaj, to obie wartości (width i height) są > TARGET_DIMENSION
                    # W tym przypadku, punkty 2 i 3 z zadania sprowadzają się do tego samego:
                    # przeskaluj tak, aby większy wymiar był równy TARGET_DIMENSION, zachowując proporcje.

                    self.status_update.emit(
                        f"Przeskalowywanie: {file_path} ({width}x{height})"
                    )

                    original_format = img.format  # Zapamiętujemy format

                    if width > height:
                        new_width = TARGET_DIMENSION
                        new_height = int(height * (TARGET_DIMENSION / width))
                    else:
                        new_height = TARGET_DIMENSION
                        new_width = int(width * (TARGET_DIMENSION / height))

                    # Upewnijmy się, że żaden wymiar nie jest zerowy po zaokrągleniu
                    new_width = max(1, new_width)
                    new_height = max(1, new_height)

                    resized_img = img.resize(
                        (new_width, new_height), Image.Resampling.LANCZOS
                    )

                    # Przygotowanie opcji zapisu
                    save_kwargs = {}
                    if original_format and original_format.upper() == "JPEG":
                        save_kwargs["quality"] = 90  # Dobra jakość dla JPEG
                        save_kwargs["optimize"] = True
                    elif original_format and original_format.upper() == "PNG":
                        save_kwargs["optimize"] = True

                    # Nadpisywanie pliku
                    resized_img.save(file_path, format=original_format, **save_kwargs)
                    self.status_update.emit(
                        f"Przeskalowano i zapisano: {file_path} do {new_width}x{new_height}"
                    )
                    resized_count += 1

            except FileNotFoundError:
                self.status_update.emit(f"BŁĄD: Nie znaleziono pliku: {file_path}")
                self.error_signal.emit(f"Nie znaleziono pliku: {file_path}")
            except UnidentifiedImageError:
                self.status_update.emit(
                    f"BŁĄD: Nie można otworzyć lub zidentyfikować pliku jako obraz: {file_path}"
                )
                self.error_signal.emit(
                    f"Nie można otworzyć lub zidentyfikować pliku jako obraz: {file_path}"
                )
            except Exception as e:
                self.status_update.emit(
                    f"BŁĄD podczas przetwarzania {file_path}: {str(e)}"
                )
                self.error_signal.emit(f"Błąd dla pliku {file_path}: {str(e)}")

            processed_count += 1
            self.progress.emit(processed_count, total_files)

        if self.is_running:
            self.status_update.emit(
                f"Zakończono. Przeskalowano: {resized_count}, Pominięto: {skipped_count}, Razem: {total_files}."
            )
        else:
            self.status_update.emit(
                f"Proces zatrzymany przez użytkownika. Przetworzono {processed_count-1}/{total_files} plików."
            )
        self.finished.emit()


class ImageProcessorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Skaler Obrazów do {TARGET_DIMENSION}px")
        self.setGeometry(100, 100, 700, 500)
        self.worker = None
        self.thread = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()

        # Folder selection
        folder_layout = QHBoxLayout()
        self.folder_label = QLabel("Folder:")
        folder_layout.addWidget(self.folder_label)
        self.folder_path_edit = QLineEdit()
        self.folder_path_edit.setPlaceholderText("Wybierz folder z obrazami")
        self.folder_path_edit.setReadOnly(True)
        folder_layout.addWidget(self.folder_path_edit)
        self.browse_button = QPushButton("Przeglądaj...")
        self.browse_button.clicked.connect(self.browse_folder)
        folder_layout.addWidget(self.browse_button)
        layout.addLayout(folder_layout)

        # Start/Stop buttons
        action_layout = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_processing)
        self.start_button.setStyleSheet("background-color: lightgreen;")
        action_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("background-color: lightcoral;")
        action_layout.addWidget(self.stop_button)
        layout.addLayout(action_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Status messages
        self.status_label = QLabel("Status:")
        layout.addWidget(self.status_label)
        self.status_text_edit = QTextEdit()
        self.status_text_edit.setReadOnly(True)
        layout.addWidget(self.status_text_edit)

        self.setLayout(layout)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Wybierz folder")
        if folder:
            self.folder_path_edit.setText(folder)
            self.status_text_edit.clear()
            self.progress_bar.setValue(0)

    def start_processing(self):
        folder_path = self.folder_path_edit.text()
        if not folder_path or not os.path.isdir(folder_path):
            QMessageBox.warning(self, "Błąd", "Proszę wybrać prawidłowy folder.")
            return

        self.start_button.setEnabled(False)
        self.browse_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_text_edit.clear()
        self.progress_bar.setValue(0)

        self.thread = QThread()
        self.worker = Worker(folder_path)
        self.worker.moveToThread(self.thread)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.on_processing_finished)

        self.worker.progress.connect(self.update_progress)
        self.worker.status_update.connect(self.update_status)
        self.worker.error_signal.connect(self.log_error)

        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def stop_processing(self):
        if self.worker:
            self.worker.stop()
        self.stop_button.setEnabled(False)  # Zapobiega wielokrotnemu klikaniu

    def on_processing_finished(self):
        self.start_button.setEnabled(True)
        self.browse_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if self.worker and not self.worker.is_running:
            self.status_text_edit.append("Przetwarzanie zatrzymane przez użytkownika.")
        else:
            self.status_text_edit.append("Przetwarzanie zakończone.")

        self.worker = None
        self.thread = None

    def update_progress(self, current, total):
        if total > 0:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(current)
        else:
            self.progress_bar.setMaximum(1)  # uniknięcie błędu dzielenia przez zero
            self.progress_bar.setValue(0)

    def update_status(self, message):
        self.status_text_edit.append(message)

    def log_error(self, error_message):
        # Można dodać logowanie do pliku lub bardziej rozbudowane powiadomienia
        self.status_text_edit.append(f"<font color='red'>BŁĄD: {error_message}</font>")

    def closeEvent(self, event):
        if self.thread and self.thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Zamykanie",
                "Proces przetwarzania jest w toku. Czy na pewno chcesz zamknąć?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                if self.worker:
                    self.worker.stop()
                self.thread.quit()
                self.thread.wait()  # Czekaj na zakończenie wątku
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessorApp()
    window.show()
    sys.exit(app.exec())
