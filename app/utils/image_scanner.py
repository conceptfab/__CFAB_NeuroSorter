import os
import traceback
from typing import Dict, List, Optional

from PIL import Image
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class ImageFixerThread(QThread):
    """Wątek do naprawy obrazów."""

    progress_updated = pyqtSignal(int, int)  # aktualny, całkowity
    file_fixed = pyqtSignal(str, str)  # ścieżka, status
    fix_completed = pyqtSignal(dict)  # wyniki
    error_occurred = pyqtSignal(str)  # komunikat błędu

    def __init__(self, files_to_fix: List[str], delete_original: bool = False):
        super().__init__()
        self.files_to_fix = files_to_fix
        self.delete_original = delete_original
        self._stopped = False

    def stop(self):
        """Zatrzymuje naprawę."""
        self._stopped = True

    def run(self):
        """Wykonuje naprawę obrazów."""
        try:
            total_files = len(self.files_to_fix)
            if total_files == 0:
                self.error_occurred.emit("Brak plików do naprawy.")
                return

            results = {"fixed": [], "failed": [], "deleted": [], "total_processed": 0}

            for i, file_path in enumerate(self.files_to_fix):
                if self._stopped:
                    break

                try:
                    # Otwórz obraz
                    with Image.open(file_path) as img:
                        # Konwertuj do RGBA jeśli nie jest
                        if img.mode != "RGBA":
                            img = img.convert("RGBA")

                        # Utwórz nowy obraz w trybie RGB z białym tłem
                        new_img = Image.new("RGB", img.size, (255, 255, 255))

                        # Wklej oryginalny obraz na biały
                        new_img.paste(
                            img, mask=img.split()[3]
                        )  # użyj kanału alpha jako maski

                        # Zapisz jako JPG z maksymalną jakością
                        base, _ = os.path.splitext(file_path)
                        new_path = f"{base}.jpg"
                        new_img.save(new_path, "JPEG", quality=100, optimize=True)

                        # Usuń oryginalny plik jeśli zaznaczono
                        if self.delete_original:
                            os.remove(file_path)
                            results["deleted"].append(file_path)

                        results["fixed"].append((file_path, new_path))
                        self.file_fixed.emit(
                            file_path,
                            f"Konwertowano do JPG i zapisano jako: {new_path}",
                        )

                except Exception as e:
                    results["failed"].append((file_path, str(e)))
                    self.file_fixed.emit(file_path, f"Błąd konwersji: {str(e)}")

                results["total_processed"] += 1
                self.progress_updated.emit(i + 1, total_files)

            self.fix_completed.emit(results)

        except Exception as e:
            self.error_occurred.emit(
                f"Błąd podczas naprawy: {str(e)}\n{traceback.format_exc()}"
            )


class ImageScannerThread(QThread):
    """Wątek do skanowania obrazów."""

    progress_updated = pyqtSignal(int, int)  # aktualny, całkowity
    file_found = pyqtSignal(str, str)  # ścieżka, problem
    scan_completed = pyqtSignal(dict)  # wyniki
    error_occurred = pyqtSignal(str)  # komunikat błędu

    def __init__(self, directory: str):
        super().__init__()
        self.directory = directory
        self._stopped = False

    def stop(self):
        """Zatrzymuje skanowanie."""
        self._stopped = True

    def run(self):
        """Wykonuje skanowanie katalogu."""
        try:
            # Znajdź wszystkie pliki obrazów
            image_files = []
            for root, _, files in os.walk(self.directory):
                for file in files:
                    if file.lower().endswith((".png", ".gif", ".bmp", ".tiff", ".tif")):
                        image_files.append(os.path.join(root, file))

            total_files = len(image_files)
            if total_files == 0:
                self.error_occurred.emit(
                    "Nie znaleziono plików obrazów w wybranym katalogu."
                )
                return

            results = {
                "palette_with_transparency": [],
                "total_scanned": 0,
                "total_problems": 0,
            }

            # Skanuj każdy plik
            for i, file_path in enumerate(image_files):
                if self._stopped:
                    break

                try:
                    with Image.open(file_path) as img:
                        # Sprawdź czy obraz używa palety i ma przezroczystość
                        if img.mode == "P" and "transparency" in img.info:
                            results["palette_with_transparency"].append(file_path)
                            results["total_problems"] += 1
                            self.file_found.emit(
                                file_path, "Obraz z paletą i przezroczystością"
                            )

                except Exception as e:
                    self.file_found.emit(file_path, f"Błąd podczas analizy: {str(e)}")

                results["total_scanned"] += 1
                self.progress_updated.emit(i + 1, total_files)

            self.scan_completed.emit(results)

        except Exception as e:
            self.error_occurred.emit(
                f"Błąd podczas skanowania: {str(e)}\n{traceback.format_exc()}"
            )


class ImageScannerWidget(QWidget):
    """Widget do skanowania i naprawy obrazów z problematyczną paletą i przezroczystością."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scanner_thread = None
        self.fixer_thread = None
        self.problem_files = []
        self.init_ui()

    def init_ui(self):
        """Inicjalizuje interfejs użytkownika."""
        layout = QVBoxLayout()

        # Przycisk wyboru katalogu
        self.select_dir_btn = QPushButton("Wybierz katalog do skanowania")
        self.select_dir_btn.clicked.connect(self.select_directory)
        layout.addWidget(self.select_dir_btn)

        # Etykieta z wybraną ścieżką
        self.path_label = QLabel("Nie wybrano katalogu")
        layout.addWidget(self.path_label)

        # Pasek postępu
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Pole tekstowe z wynikami
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        layout.addWidget(self.results_text)

        # Przyciski akcji
        button_layout = QHBoxLayout()

        self.stop_btn = QPushButton("Zatrzymaj skanowanie")
        self.stop_btn.clicked.connect(self.stop_scanning)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)

        self.fix_btn = QPushButton("Konwertuj do JPG i usuń oryginały")
        self.fix_btn.clicked.connect(self.start_fixing)
        self.fix_btn.setEnabled(False)
        button_layout.addWidget(self.fix_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def select_directory(self):
        """Otwiera dialog wyboru katalogu."""
        directory = QFileDialog.getExistingDirectory(
            self, "Wybierz katalog do skanowania", "", QFileDialog.Option.ShowDirsOnly
        )
        if directory:
            self.path_label.setText(directory)
            self.start_scanning(directory)

    def start_scanning(self, directory: str):
        """Rozpoczyna proces skanowania."""
        self.results_text.clear()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.select_dir_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.fix_btn.setEnabled(False)

        self.scanner_thread = ImageScannerThread(directory)
        self.scanner_thread.progress_updated.connect(self.update_progress)
        self.scanner_thread.file_found.connect(self.add_file_result)
        self.scanner_thread.scan_completed.connect(self.show_results)
        self.scanner_thread.error_occurred.connect(self.show_error)
        self.scanner_thread.start()

    def start_fixing(self):
        """Rozpoczyna proces naprawy znalezionych plików."""
        if not self.problem_files:
            QMessageBox.warning(self, "Ostrzeżenie", "Brak plików do naprawy.")
            return

        reply = QMessageBox.question(
            self,
            "Potwierdzenie",
            f"Czy chcesz skonwertować {len(self.problem_files)} plików do JPG i usunąć oryginały?\n"
            "Ta operacja jest nieodwracalna!",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.results_text.clear()
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)
            self.select_dir_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.fix_btn.setEnabled(False)

            self.fixer_thread = ImageFixerThread(
                self.problem_files, delete_original=True
            )
            self.fixer_thread.progress_updated.connect(self.update_progress)
            self.fixer_thread.file_fixed.connect(self.add_fix_result)
            self.fixer_thread.fix_completed.connect(self.show_fix_results)
            self.fixer_thread.error_occurred.connect(self.show_error)
            self.fixer_thread.start()

    def stop_scanning(self):
        """Zatrzymuje proces skanowania lub naprawy."""
        if self.scanner_thread and self.scanner_thread.isRunning():
            self.scanner_thread.stop()
        if self.fixer_thread and self.fixer_thread.isRunning():
            self.fixer_thread.stop()

        self.stop_btn.setEnabled(False)
        self.results_text.append("\nOperacja zatrzymana przez użytkownika.")

    def update_progress(self, current: int, total: int):
        """Aktualizuje pasek postępu."""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def add_file_result(self, file_path: str, problem: str):
        """Dodaje wynik skanowania pojedynczego pliku."""
        self.results_text.append(
            f"Znaleziono problem: {file_path}\nProblem: {problem}\n"
        )
        self.problem_files.append(file_path)

    def add_fix_result(self, file_path: str, status: str):
        """Dodaje wynik naprawy pojedynczego pliku."""
        self.results_text.append(f"Plik: {file_path}\nStatus: {status}\n")

    def show_results(self, results: Dict):
        """Wyświetla podsumowanie wyników skanowania."""
        self.progress_bar.setVisible(False)
        self.select_dir_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.fix_btn.setEnabled(len(self.problem_files) > 0)

        summary = f"\n=== PODSUMOWANIE SKANOWANIA ===\n"
        summary += f"Przeskanowano plików: {results['total_scanned']}\n"
        summary += f"Znaleziono problemów: {results['total_problems']}\n"

        if results["palette_with_transparency"]:
            summary += "\nPliki z paletą i przezroczystością:\n"
            for file in results["palette_with_transparency"]:
                summary += f"- {file}\n"

        self.results_text.append(summary)

    def show_fix_results(self, results: Dict):
        """Wyświetla podsumowanie wyników naprawy."""
        self.progress_bar.setVisible(False)
        self.select_dir_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.fix_btn.setEnabled(False)

        summary = f"\n=== PODSUMOWANIE KONWERSJI ===\n"
        summary += f"Przetworzono plików: {results['total_processed']}\n"
        summary += f"Pomyślnie skonwertowano: {len(results['fixed'])}\n"
        summary += f"Usunięto oryginałów: {len(results['deleted'])}\n"
        summary += f"Nie udało się skonwertować: {len(results['failed'])}\n"

        if results["failed"]:
            summary += "\nPliki, których nie udało się skonwertować:\n"
            for file, error in results["failed"]:
                summary += f"- {file}\n  Błąd: {error}\n"

        self.results_text.append(summary)

    def show_error(self, error_msg: str):
        """Wyświetla komunikat o błędzie."""
        QMessageBox.critical(self, "Błąd", error_msg)
        self.progress_bar.setVisible(False)
        self.select_dir_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.fix_btn.setEnabled(False)
