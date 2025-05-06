import sys
import os
import shutil
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLineEdit,
    QLabel,
    QFileDialog,
    QTextEdit,
    QMessageBox,
)
from PyQt6.QtCore import Qt


class JpegMoverApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Przenoszenie plików JPEG")
        self.setGeometry(200, 200, 600, 400)  # x, y, width, height

        self.source_dir_path = ""
        self.dest_dir_path = ""

        # Layout główny
        main_layout = QVBoxLayout()

        # Sekcja folderu źródłowego
        source_layout = QHBoxLayout()
        self.source_label = QLabel("Folder źródłowy:")
        self.source_edit = QLineEdit()
        self.source_edit.setReadOnly(True)
        self.source_button = QPushButton("Przeglądaj...")
        self.source_button.clicked.connect(self.browse_source_folder)
        source_layout.addWidget(self.source_label)
        source_layout.addWidget(self.source_edit)
        source_layout.addWidget(self.source_button)
        main_layout.addLayout(source_layout)

        # Sekcja folderu docelowego
        dest_layout = QHBoxLayout()
        self.dest_label = QLabel("Folder docelowy:")
        self.dest_edit = QLineEdit()
        self.dest_edit.setReadOnly(True)
        self.dest_button = QPushButton("Przeglądaj...")
        self.dest_button.clicked.connect(self.browse_dest_folder)
        dest_layout.addWidget(self.dest_label)
        dest_layout.addWidget(self.dest_edit)
        dest_layout.addWidget(self.dest_button)
        main_layout.addLayout(dest_layout)

        # Przycisk akcji
        self.move_button = QPushButton("Przenieś pliki JPEG")
        self.move_button.clicked.connect(self.process_files)
        main_layout.addWidget(self.move_button)

        # Log
        self.log_label = QLabel("Log operacji:")
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        main_layout.addWidget(self.log_label)
        main_layout.addWidget(self.log_area)

        self.setLayout(main_layout)

    def browse_source_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Wybierz folder źródłowy")
        if folder_path:
            self.source_dir_path = folder_path
            self.source_edit.setText(folder_path)
            self.log_area.append(f"Wybrano folder źródłowy: {folder_path}")

    def browse_dest_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Wybierz folder docelowy")
        if folder_path:
            self.dest_dir_path = folder_path
            self.dest_edit.setText(folder_path)
            self.log_area.append(f"Wybrano folder docelowy: {folder_path}")

    def log_message(self, message):
        self.log_area.append(message)
        QApplication.processEvents()  # Aby odświeżyć UI

    def get_unique_dest_path(self, dest_folder, filename):
        """Generuje unikalną ścieżkę docelową, jeśli plik już istnieje."""
        base, ext = os.path.splitext(filename)
        counter = 1
        unique_filename = filename
        dest_path = os.path.join(dest_folder, unique_filename)
        while os.path.exists(dest_path):
            unique_filename = f"{base}_copy_{counter}{ext}"
            dest_path = os.path.join(dest_folder, unique_filename)
            counter += 1
        return dest_path

    def process_files(self):
        if not self.source_dir_path:
            QMessageBox.warning(self, "Błąd", "Wybierz folder źródłowy.")
            self.log_message("BŁĄD: Nie wybrano folderu źródłowego.")
            return

        if not self.dest_dir_path:
            QMessageBox.warning(self, "Błąd", "Wybierz folder docelowy.")
            self.log_message("BŁĄD: Nie wybrano folderu docelowego.")
            return

        if self.source_dir_path == self.dest_dir_path:
            QMessageBox.warning(
                self, "Błąd", "Folder źródłowy i docelowy nie mogą być takie same."
            )
            self.log_message("BŁĄD: Folder źródłowy i docelowy są takie same.")
            return

        self.log_message("\nRozpoczynam przetwarzanie plików...")
        self.move_button.setEnabled(False)  # Wyłącz przycisk podczas przetwarzania

        files_moved_count = 0
        files_found_count = 0

        try:
            # Utwórz folder docelowy, jeśli nie istnieje
            if not os.path.exists(self.dest_dir_path):
                os.makedirs(self.dest_dir_path)
                self.log_message(f"Utworzono folder docelowy: {self.dest_dir_path}")

            for root, _, files in os.walk(self.source_dir_path):
                for filename in files:
                    if filename.lower().endswith(
                        ".jpeg"
                    ):  # Możesz dodać .jpg jeśli chcesz: ('.jpeg', '.jpg')
                        files_found_count += 1
                        source_file_path = os.path.join(root, filename)

                        # Generuj unikalną nazwę w folderze docelowym
                        dest_file_path = self.get_unique_dest_path(
                            self.dest_dir_path, filename
                        )

                        try:
                            shutil.move(source_file_path, dest_file_path)
                            self.log_message(
                                f"Przeniesiono: {source_file_path} -> {dest_file_path}"
                            )
                            files_moved_count += 1
                        except Exception as e:
                            self.log_message(
                                f"BŁĄD podczas przenoszenia {source_file_path}: {e}"
                            )

            self.log_message(f"\nZakończono przetwarzanie.")
            self.log_message(f"Znaleziono plików JPEG: {files_found_count}")
            self.log_message(f"Przeniesiono plików: {files_moved_count}")
            QMessageBox.information(
                self,
                "Zakończono",
                f"Przeniesiono {files_moved_count} z {files_found_count} znalezionych plików JPEG.",
            )

        except Exception as e:
            self.log_message(f"Krytyczny błąd podczas przetwarzania: {e}")
            QMessageBox.critical(self, "Krytyczny błąd", f"Wystąpił błąd: {e}")
        finally:
            self.move_button.setEnabled(True)  # Włącz przycisk z powrotem


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = JpegMoverApp()
    window.show()
    sys.exit(app.exec())
