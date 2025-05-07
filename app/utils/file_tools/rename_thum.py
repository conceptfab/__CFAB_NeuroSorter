import sys
import os
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLineEdit,
    QTextEdit,
    QFileDialog,
    QLabel,
    QMessageBox,
)
from PyQt6.QtCore import QThread, QObject, pyqtSignal, Qt


class Worker(QObject):
    """
    Obiekt roboczy do wykonywania zadań w osobnym wątku,
    aby nie blokować interfejsu użytkownika.
    """

    finished = pyqtSignal()
    progress = pyqtSignal(str)
    error = pyqtSignal(str)
    summary = pyqtSignal(
        int, int
    )  # Przesłane: (liczba_zmienionych_plikow, liczba_bledow)

    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path
        self.is_running = True

    def run(self):
        """Główna logika przeszukiwania i zmiany nazw plików."""
        changed_files_count = 0
        error_count = 0
        if not self.folder_path or not os.path.isdir(self.folder_path):
            self.error.emit(
                f"Wybrana ścieżka '{self.folder_path}' nie jest prawidłowym folderem."
            )
            self.finished.emit()
            return

        self.progress.emit(f"Rozpoczynam przeszukiwanie folderu: {self.folder_path}")

        for dirpath, _, filenames in os.walk(self.folder_path):
            if not self.is_running:
                self.progress.emit("Przerwano operację.")
                break

            folder_name = os.path.basename(dirpath)
            if not folder_name:  # Może się zdarzyć dla roota dysku
                self.progress.emit(
                    f"Pominięto folder bez nazwy (prawdopodobnie root): {dirpath}"
                )
                continue

            file_counter = {}  # Licznik dla unikalnych nazw w danym folderze

            for filename in filenames:
                if not self.is_running:
                    break

                file_ext_lower = os.path.splitext(filename)[1].lower()
                if file_ext_lower in [".jpg", ".jpeg"]:
                    original_full_path = os.path.join(dirpath, filename)

                    # Ustalanie nowej nazwy bazowej
                    new_name_base = folder_name

                    # Obsługa konfliktów nazw (jeśli w folderze jest więcej niż jeden jpg/jpeg)
                    current_count = file_counter.get(new_name_base, 0)
                    if current_count == 0:
                        new_filename_candidate = f"{new_name_base}{file_ext_lower}"
                    else:
                        new_filename_candidate = (
                            f"{new_name_base}_{current_count}{file_ext_lower}"
                        )

                    # Sprawdzenie, czy nazwa już istnieje (na wszelki wypadek, choć licznik powinien temu zapobiegać)
                    # Ta pętla jest bardziej na wypadek, gdyby plik o nazwie folderu_X.jpg już istniał z innego powodu
                    idx = current_count
                    while os.path.exists(os.path.join(dirpath, new_filename_candidate)):
                        idx += 1
                        new_filename_candidate = (
                            f"{new_name_base}_{idx}{file_ext_lower}"
                        )

                    new_full_path = os.path.join(dirpath, new_filename_candidate)

                    # Nie zmieniamy nazwy, jeśli jest taka sama (np. folder.jpg już istnieje)
                    if (
                        original_full_path.lower() == new_full_path.lower()
                        and original_full_path != new_full_path
                    ):
                        # Jeśli tylko różnica w wielkości liter, a system plików nie rozróżnia
                        self.progress.emit(
                            f"Pominięto (nazwa docelowa {new_filename_candidate} jest taka sama jak {filename}, ignorując wielkość liter): {original_full_path}"
                        )
                        continue
                    elif original_full_path == new_full_path:
                        self.progress.emit(
                            f"Pominięto (nazwa {filename} jest już poprawna): {original_full_path}"
                        )
                        file_counter[new_name_base] = idx + 1  # Aktualizujemy licznik
                        continue

                    try:
                        os.rename(original_full_path, new_full_path)
                        self.progress.emit(
                            f"Zmieniono: '{original_full_path}' -> '{new_full_path}'"
                        )
                        changed_files_count += 1
                        file_counter[new_name_base] = idx + 1  # Aktualizujemy licznik
                    except OSError as e:
                        self.error.emit(
                            f"Błąd zmiany nazwy '{original_full_path}': {e}"
                        )
                        error_count += 1
            if not self.is_running:
                break

        self.summary.emit(changed_files_count, error_count)
        self.finished.emit()

    def stop(self):
        self.is_running = False


class RenamerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Zmieniacz Nazw JPG/JPEG")
        self.setGeometry(100, 100, 700, 500)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Wybór folderu
        self.folder_layout = QHBoxLayout()
        self.folder_label = QLabel("Folder:")
        self.folder_path_edit = QLineEdit()
        self.folder_path_edit.setPlaceholderText("Wybierz folder do przeszukania...")
        self.browse_button = QPushButton("Przeglądaj...")
        self.browse_button.clicked.connect(self.browse_folder)
        self.folder_layout.addWidget(self.folder_label)
        self.folder_layout.addWidget(self.folder_path_edit)
        self.folder_layout.addWidget(self.browse_button)
        self.layout.addLayout(self.folder_layout)

        # Przycisk Start/Stop
        self.process_button = QPushButton("Rozpocznij Zmianę Nazw")
        self.process_button.clicked.connect(self.start_processing)
        self.layout.addWidget(self.process_button)

        # Logi
        self.log_label = QLabel("Log:")
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        self.layout.addWidget(self.log_label)
        self.layout.addWidget(self.log_text_edit)

        self.thread = None
        self.worker = None

    def browse_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Wybierz Folder")
        if folder_path:
            self.folder_path_edit.setText(folder_path)
            self.log_text_edit.clear()  # Czyścimy logi przy wyborze nowego folderu

    def start_processing(self):
        if self.worker and self.worker.is_running:
            self.log_message("Zatrzymywanie operacji...")
            self.worker.stop()
            self.process_button.setText(
                "Rozpocznij Zmianę Nazw"
            )  # Zmieniamy tekst przycisku
            # Nie czyścimy wątku od razu, pozwalamy mu dokończyć obecną iterację i zakończyć się grzecznie
            return

        folder = self.folder_path_edit.text()
        if not folder:
            QMessageBox.warning(self, "Błąd", "Najpierw wybierz folder.")
            return
        if not os.path.isdir(folder):
            QMessageBox.warning(
                self,
                "Błąd",
                f"Podana ścieżka '{folder}' nie jest prawidłowym folderem.",
            )
            return

        self.log_text_edit.clear()
        self.process_button.setText("Zatrzymaj Przetwarzanie")
        self.process_button.setEnabled(True)  # Upewnij się, że jest włączony

        # Utwórz wątek i pracownika
        self.thread = QThread()
        self.worker = Worker(folder)
        self.worker.moveToThread(self.thread)

        # Połączenie sygnałów
        self.worker.progress.connect(self.log_message)
        self.worker.error.connect(self.log_error)
        self.worker.summary.connect(self.log_summary)
        self.worker.finished.connect(self.on_worker_finished)

        self.thread.started.connect(self.worker.run)

        self.thread.start()

    def on_worker_finished(self):
        self.log_message("--- Zakończono przetwarzanie ---")
        self.process_button.setText("Rozpocznij Zmianę Nazw")
        self.process_button.setEnabled(True)
        if self.thread:
            self.thread.quit()
            self.thread.wait()  # Poczekaj na zakończenie wątku
            self.thread.deleteLater()
            self.thread = None
        if self.worker:
            self.worker.deleteLater()
            self.worker = None

    def log_message(self, message):
        self.log_text_edit.append(message)

    def log_error(self, message):
        self.log_text_edit.append(f"<font color='red'>BŁĄD: {message}</font>")

    def log_summary(self, changed_count, error_count):
        self.log_message(f"\n--- Podsumowanie ---")
        self.log_message(f"Zmieniono nazwy {changed_count} plików.")
        if error_count > 0:
            self.log_error(f"Wystąpiło {error_count} błędów.")
        else:
            self.log_message("Nie wystąpiły błędy.")

    def closeEvent(self, event):
        """Obsługa zamykania okna, aby zatrzymać wątek."""
        if self.worker and self.worker.is_running:
            self.log_message(
                "Próba zamknięcia aplikacji. Zatrzymywanie aktywnego przetwarzania..."
            )
            self.worker.stop()
            if self.thread:
                self.thread.quit()
                self.thread.wait(2000)  # Dajmy mu chwilę na zamknięcie
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = RenamerApp()
    main_window.show()
    sys.exit(app.exec())
