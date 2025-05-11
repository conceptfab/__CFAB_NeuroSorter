import os
import sys

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


# --- Logika walidacji (może być w osobnym pliku) ---
class ValidationWorker(QThread):
    """
    Wątek roboczy do przeprowadzania walidacji w tle,
    aby nie blokować UI.
    """

    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(list)  # Lista komunikatów wynikowych
    error_signal = pyqtSignal(str)

    def __init__(self, train_path, val_path):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.results = []

    def log(self, message, level="INFO"):
        log_message = f"[{level}] {message}"
        self.results.append(log_message)
        self.progress_signal.emit(log_message)

    def run(self):
        try:
            self.log("Rozpoczęcie walidacji...")

            if not os.path.isdir(self.train_path):
                self.error_signal.emit(
                    f"Ścieżka do danych treningowych nie istnieje: {self.train_path}"
                )
                return
            if not os.path.isdir(self.val_path):
                self.error_signal.emit(
                    f"Ścieżka do danych walidacyjnych nie istnieje: {self.val_path}"
                )
                return

            self.log(f"Folder danych treningowych: {self.train_path}")
            self.log(f"Folder danych walidacyjnych: {self.val_path}")

            # 1. Pobierz listę folderów kategorii
            train_categories = sorted(
                [
                    d
                    for d in os.listdir(self.train_path)
                    if os.path.isdir(os.path.join(self.train_path, d))
                ]
            )
            val_categories = sorted(
                [
                    d
                    for d in os.listdir(self.val_path)
                    if os.path.isdir(os.path.join(self.val_path, d))
                ]
            )

            self.log(
                f"Znalezione kategorie w danych treningowych ({len(train_categories)}): {', '.join(train_categories) if train_categories else 'Brak'}"
            )
            self.log(
                f"Znalezione kategorie w danych walidacyjnych ({len(val_categories)}): {', '.join(val_categories) if val_categories else 'Brak'}"
            )

            # 2. Porównaj ilość kategorii
            if len(train_categories) == len(val_categories):
                self.log("Liczba kategorii w obu folderach jest zgodna.")
            else:
                self.log(
                    f"UWAGA: Różna liczba kategorii! Trening: {len(train_categories)}, Walidacja: {len(val_categories)}",
                    "WARNING",
                )

            # 3. Porównaj nazwy kategorii
            set_train_cat = set(train_categories)
            set_val_cat = set(val_categories)

            if set_train_cat == set_val_cat:
                self.log("Nazwy kategorii w obu folderach są zgodne.")
            else:
                self.log("UWAGA: Niezgodność nazw kategorii!", "WARNING")
                missing_in_val = set_train_cat - set_val_cat
                missing_in_train = set_val_cat - set_train_cat
                if missing_in_val:
                    self.log(
                        f"  Kategorie obecne w treningu, brakujące w walidacji: {', '.join(missing_in_val)}",
                        "WARNING",
                    )
                if missing_in_train:
                    self.log(
                        f"  Kategorie obecne w walidacji, brakujące w treningu: {', '.join(missing_in_train)}",
                        "WARNING",
                    )

            # 4. Sprawdź zawartość każdego folderu kategorii
            common_categories = set_train_cat.intersection(set_val_cat)
            all_categories_to_check = set_train_cat.union(set_val_cat)

            self.log("\nSprawdzanie zawartości folderów kategorii:")
            for category_set, base_path, dataset_name in [
                (train_categories, self.train_path, "treningowych"),
                (val_categories, self.val_path, "walidacyjnych"),
            ]:
                self.log(f"  Sprawdzanie danych {dataset_name}:")
                if not category_set:
                    self.log(
                        f"    Brak kategorii do sprawdzenia w danych {dataset_name}."
                    )
                    continue
                for cat_name in category_set:
                    cat_path = os.path.join(base_path, cat_name)
                    if not os.listdir(cat_path):  # Sprawdza czy folder jest pusty
                        self.log(
                            f"    Folder kategorii '{cat_name}' jest PUSTY.", "WARNING"
                        )
                    else:
                        # Można dodać bardziej szczegółowe sprawdzanie, np. typów plików
                        num_files = len(
                            [
                                f
                                for f in os.listdir(cat_path)
                                if os.path.isfile(os.path.join(cat_path, f))
                            ]
                        )
                        num_dirs = len(
                            [
                                d
                                for d in os.listdir(cat_path)
                                if os.path.isdir(os.path.join(cat_path, d))
                            ]
                        )
                        self.log(
                            f"    Folder kategorii '{cat_name}' zawiera {num_files} plików i {num_dirs} podfolderów.",
                            "INFO",
                        )

            self.log("\nWalidacja zakończona.", "INFO")
            self.finished_signal.emit(self.results)

        except Exception as e:
            self.error_signal.emit(f"Wystąpił błąd podczas walidacji: {str(e)}")


# --- UI ---
class ValidatorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Walidator Danych AI")
        self.setGeometry(200, 200, 700, 500)  # X, Y, Width, Height
        self._init_ui()
        self.worker = None  # Placeholder for the validation thread

    def _init_ui(self):
        main_layout = QVBoxLayout(self)

        # Sekcja wyboru folderów
        folder_section_layout = QVBoxLayout()

        # Folder danych treningowych
        train_layout = QHBoxLayout()
        train_label = QLabel("Folder danych treningowych:")
        self.train_path_edit = QLineEdit()
        self.train_path_edit.setPlaceholderText("Wybierz folder...")
        self.train_path_edit.setReadOnly(True)
        train_browse_btn = QPushButton("Przeglądaj...")
        train_browse_btn.clicked.connect(self.browse_train_folder)
        train_layout.addWidget(train_label)
        train_layout.addWidget(self.train_path_edit)
        train_layout.addWidget(train_browse_btn)
        folder_section_layout.addLayout(train_layout)

        # Folder danych walidacyjnych
        val_layout = QHBoxLayout()
        val_label = QLabel("Folder danych walidacyjnych:")
        self.val_path_edit = QLineEdit()
        self.val_path_edit.setPlaceholderText("Wybierz folder...")
        self.val_path_edit.setReadOnly(True)
        val_browse_btn = QPushButton("Przeglądaj...")
        val_browse_btn.clicked.connect(self.browse_val_folder)
        val_layout.addWidget(val_label)
        val_layout.addWidget(self.val_path_edit)
        val_layout.addWidget(val_browse_btn)
        folder_section_layout.addLayout(val_layout)

        main_layout.addLayout(folder_section_layout)

        # Przycisk walidacji
        self.validate_button = QPushButton("Rozpocznij Walidację")
        self.validate_button.clicked.connect(self.start_validation)
        main_layout.addWidget(self.validate_button)

        # Etykieta statusu
        self.status_label = QLabel("Status: Oczekuje na wybór folderów...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.status_label)

        # Pole wyników
        results_label = QLabel("Wyniki Walidacji:")
        self.results_text_edit = QTextEdit()
        self.results_text_edit.setReadOnly(True)
        # Ustawienie ciemniejszego motywu dla pola tekstowego (opcjonalne)
        # palette = self.results_text_edit.palette()
        # palette.setColor(QPalette.ColorRole.Base, QColor(40, 40, 40))
        # palette.setColor(QPalette.ColorRole.Text, QColor(220, 220, 220))
        # self.results_text_edit.setPalette(palette)

        main_layout.addWidget(results_label)
        main_layout.addWidget(self.results_text_edit)

        self.setLayout(main_layout)

    def browse_folder(self, line_edit_widget):
        folder_path = QFileDialog.getExistingDirectory(self, "Wybierz folder")
        if folder_path:
            line_edit_widget.setText(folder_path)
            self.update_status_ready()

    def browse_train_folder(self):
        self.browse_folder(self.train_path_edit)

    def browse_val_folder(self):
        self.browse_folder(self.val_path_edit)

    def update_status_ready(self):
        if self.train_path_edit.text() and self.val_path_edit.text():
            self.status_label.setText("Status: Gotowy do walidacji.")
        else:
            self.status_label.setText("Status: Oczekuje na wybór folderów...")

    def start_validation(self):
        train_path = self.train_path_edit.text()
        val_path = self.val_path_edit.text()

        if not train_path or not val_path:
            QMessageBox.warning(
                self,
                "Brakujące ścieżki",
                "Proszę wybrać oba foldery (treningowy i walidacyjny).",
            )
            return

        self.results_text_edit.clear()
        self.validate_button.setEnabled(False)
        self.status_label.setText("Status: Walidacja w toku...")

        # Uruchom walidację w osobnym wątku
        self.worker = ValidationWorker(train_path, val_path)
        self.worker.progress_signal.connect(self.append_log_message)
        self.worker.finished_signal.connect(self.on_validation_finished)
        self.worker.error_signal.connect(self.on_validation_error)
        self.worker.start()

    def append_log_message(self, message):
        if "WARNING" in message:
            self.results_text_edit.append(f"<font color='orange'>{message}</font>")
        elif "ERROR" in message or "BŁĄD" in message:  # Dodane dla error_signal
            self.results_text_edit.append(f"<font color='red'>{message}</font>")
        else:
            self.results_text_edit.append(message)

    def on_validation_finished(self, results):
        # Ostatnie logi mogły zostać wysłane przez progress_signal,
        # ale można tu dodać podsumowujący komunikat
        # self.results_text_edit.append("\n--- Walidacja zakończona pomyślnie ---")
        self.validate_button.setEnabled(True)
        self.status_label.setText("Status: Walidacja zakończona.")
        QMessageBox.information(
            self, "Zakończono", "Walidacja danych została zakończona."
        )

    def on_validation_error(self, error_message):
        self.append_log_message(
            f"[BŁĄD KRYTYCZNY] {error_message}"
        )  # Używamy append_log_message dla spójności formatowania
        self.validate_button.setEnabled(True)
        self.status_label.setText("Status: Błąd podczas walidacji.")
        QMessageBox.critical(self, "Błąd Walidacji", error_message)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Możesz ustawić styl, np. 'Fusion' dla bardziej nowoczesnego wyglądu
    # app.setStyle('Fusion')

    # # Opcjonalny ciemny motyw (prosty przykład)
    # dark_palette = QPalette()
    # dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    # dark_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    # dark_palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    # dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    # dark_palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    # dark_palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    # dark_palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    # dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    # dark_palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    # dark_palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    # dark_palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    # dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    # dark_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    # app.setPalette(dark_palette)
    # app.setStyleSheet("QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }")

    window = ValidatorApp()
    window.show()
    sys.exit(app.exec())
