import os
import sys

from PyQt6.QtCore import Qt, QTimer  # QTimer for delayed status clear
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class JpegToJpgConverter(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Konwerter JPEG na JPG")
        self.setGeometry(300, 300, 600, 400)

        # Layout główny
        main_layout = QVBoxLayout()

        # Sekcja wyboru folderu
        folder_layout = QHBoxLayout()
        self.folder_label = QLabel("Folder:")
        self.folder_path_edit = QLineEdit()
        self.folder_path_edit.setReadOnly(True)
        self.browse_button = QPushButton("Wybierz folder...")
        self.browse_button.clicked.connect(self.browse_folder)

        folder_layout.addWidget(self.folder_label)
        folder_layout.addWidget(self.folder_path_edit)
        folder_layout.addWidget(self.browse_button)
        main_layout.addLayout(folder_layout)

        # Przycisk konwersji
        self.convert_button = QPushButton("Zamień .jpeg na .jpg")
        self.convert_button.clicked.connect(self.start_conversion)
        main_layout.addWidget(self.convert_button)

        # Logi
        self.log_label = QLabel("Log:")
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        main_layout.addWidget(self.log_label)
        main_layout.addWidget(self.log_area)

        # Status bar
        self.status_bar = QStatusBar()
        main_layout.addWidget(self.status_bar)

        self.setLayout(main_layout)

    def browse_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Wybierz folder")
        if folder_path:
            self.folder_path_edit.setText(folder_path)
            self.log_area.clear()
            self.status_bar.showMessage(f"Wybrano folder: {folder_path}", 5000)

    def log_message(self, message, is_error=False):
        if is_error:
            self.log_area.append(f"<font color='red'>{message}</font>")
        else:
            self.log_area.append(message)
        QApplication.processEvents()  # Aby UI się odświeżało

    def start_conversion(self):
        folder_path = self.folder_path_edit.text()
        if not folder_path:
            QMessageBox.warning(self, "Błąd", "Najpierw wybierz folder.")
            return

        if not os.path.isdir(folder_path):
            QMessageBox.critical(
                self,
                "Błąd krytyczny",
                f"Wybrana ścieżka nie jest folderem: {folder_path}",
            )
            self.folder_path_edit.clear()
            return

        self.log_area.clear()
        self.log_message(f"Rozpoczynanie konwersji w folderze: {folder_path}...")
        self.status_bar.showMessage("Przetwarzanie...", 0)  # 0 = bez limitu czasu
        self.convert_button.setEnabled(False)  # Zapobiega wielokrotnemu kliknięciu

        files_renamed_count = 0
        files_found_count = 0

        try:
            for root, _, files in os.walk(folder_path):
                for filename in files:
                    if filename.lower().endswith(".jpeg"):
                        files_found_count += 1
                        old_filepath = os.path.join(root, filename)
                        # Tworzenie nowej nazwy pliku
                        base, ext = os.path.splitext(filename)
                        new_filename = base + ".jpg"
                        new_filepath = os.path.join(root, new_filename)

                        # Sprawdzenie, czy plik .jpg już nie istnieje
                        if os.path.exists(new_filepath):
                            self.log_message(
                                f"UWAGA: Plik {new_filepath} już istnieje. Pomijam {old_filepath}",
                                is_error=True,
                            )
                            continue

                        try:
                            os.rename(old_filepath, new_filepath)
                            self.log_message(
                                f"Zmieniono: {old_filepath} -> {new_filepath}"
                            )
                            files_renamed_count += 1
                        except OSError as e:
                            self.log_message(
                                f"BŁĄD przy zmianie nazwy {old_filepath}: {e}",
                                is_error=True,
                            )
        except Exception as e:
            self.log_message(
                f"Wystąpił nieoczekiwany błąd podczas przetwarzania: {e}", is_error=True
            )
        finally:
            self.convert_button.setEnabled(True)
            self.log_message(
                f"\nZakończono. Znaleziono plików .jpeg: {files_found_count}. Zmieniono nazwę: {files_renamed_count} plików."
            )
            self.status_bar.showMessage(
                f"Zakończono. Zmieniono {files_renamed_count} plików.", 5000
            )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    converter = JpegToJpgConverter()
    converter.show()
    sys.exit(app.exec())
