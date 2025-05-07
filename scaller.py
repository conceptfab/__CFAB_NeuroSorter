import os
import sys
import traceback
from typing import Dict, List, Tuple  # Dodano List i Tuple dla lepszego typowania

import matplotlib

matplotlib.use("Qt5Agg")  # Używamy Qt5Agg zamiast Qt6Agg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image

# Dodano alias dla Image.Resampling.LANCZOS dla kompatybilności wstecznej z Pillow
try:
    from PIL.Image import Resampling

    LANCZOS_ALIAS = Resampling.LANCZOS
except ImportError:
    LANCZOS_ALIAS = Image.LANCZOS  # Dla starszych wersji Pillow

from PyQt6.QtCore import QObject, QThread, pyqtSignal  # Dodano QObject
from PyQt6.QtGui import QCloseEvent  # Dodano QCloseEvent dla typowania
from PyQt6.QtWidgets import QApplication  # Dodano
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class ResolutionScannerThread(QThread):
    """Wątek do skanowania rozdzielczości obrazów."""

    progress_updated = pyqtSignal(int, int)  # aktualny, całkowity
    file_found = pyqtSignal(str, str)  # ścieżka, problem
    scan_completed = pyqtSignal(dict)  # wyniki
    error_occurred = pyqtSignal(str)  # komunikat błędu
    resolution_data_updated = pyqtSignal(list, list)  # szerokości, wysokości

    def __init__(
        self, directory: str, min_size: int, max_size: int, parent: QObject = None
    ):
        super().__init__(parent)
        self.directory = directory
        self.min_size = min_size
        self.max_size = max_size
        self._stopped = False
        self.img_resolutions_width_height_pairs: List[Tuple[int, int]] = (
            []
        )  # Zmieniona nazwa dla jasności

    def stop(self):
        """Zatrzymuje skanowanie."""
        self._stopped = True

    def run(self):
        """Wykonuje skanowanie katalogu."""
        try:
            image_files = []
            supported_formats = (
                ".png",
                ".jpg",
                ".jpeg",
                ".gif",
                ".bmp",
                ".tiff",
                ".tif",
                ".webp",
            )

            for root, _, files in os.walk(self.directory):
                for file in files:
                    if self._stopped:
                        self.error_occurred.emit(
                            "Skanowanie przerwane przez użytkownika podczas wyszukiwania plików."
                        )
                        return
                    if file.lower().endswith(supported_formats):
                        image_files.append(os.path.join(root, file))

            total_files = len(image_files)
            if total_files == 0:
                self.error_occurred.emit(
                    "Nie znaleziono plików obrazów w wybranym katalogu."
                )
                return

            results: Dict = {
                "too_small": [],
                "too_large": [],
                "total_scanned": 0,
                "total_problems": 0,
                "resolution_stats": {
                    "min_width": float("inf"),
                    "min_height": float("inf"),
                    "max_width": 0,
                    "max_height": 0,
                    "avg_width": 0,
                    "avg_height": 0,
                    "total_width": 0,
                    "total_height": 0,
                },
                "format_stats": {},
            }

            # Wyczyszczenie danych z poprzedniego skanowania
            self.img_resolutions_width_height_pairs.clear()

            for i, file_path in enumerate(image_files):
                if self._stopped:
                    # Emitujemy wyniki częściowe, jeśli skanowanie zostało przerwane
                    if results["total_scanned"] > 0:
                        results["resolution_stats"]["avg_width"] = (
                            results["resolution_stats"]["total_width"]
                            / results["total_scanned"]
                        )
                        results["resolution_stats"]["avg_height"] = (
                            results["resolution_stats"]["total_height"]
                            / results["total_scanned"]
                        )
                    self.scan_completed.emit(
                        results
                    )  # Można by dodać flagę 'interrupted: True'
                    self.error_occurred.emit("Skanowanie zatrzymane przez użytkownika.")
                    return

                try:
                    with Image.open(file_path) as img:
                        width, height = img.size

                        self.img_resolutions_width_height_pairs.append((width, height))

                        stats = results["resolution_stats"]
                        stats["min_width"] = min(stats["min_width"], width)
                        stats["min_height"] = min(stats["min_height"], height)
                        stats["max_width"] = max(stats["max_width"], width)
                        stats["max_height"] = max(stats["max_height"], height)
                        stats["total_width"] += width
                        stats["total_height"] += height

                        if width < self.min_size or height < self.min_size:
                            results["too_small"].append((file_path, (width, height)))
                            results["total_problems"] += 1
                            self.file_found.emit(
                                file_path, f"Zbyt mała rozdzielczość: {width}x{height}"
                            )
                        elif width > self.max_size or height > self.max_size:
                            results["too_large"].append((file_path, (width, height)))
                            results["total_problems"] += 1
                            self.file_found.emit(
                                file_path, f"Zbyt duża rozdzielczość: {width}x{height}"
                            )
                except Exception as e:
                    self.file_found.emit(file_path, f"Błąd podczas analizy: {str(e)}")
                    results["total_problems"] += 1  # Liczymy to jako problem

                results["total_scanned"] += 1
                self.progress_updated.emit(i + 1, total_files)

                if (i + 1) % 10 == 0 or (
                    i + 1
                ) == total_files:  # Aktualizacja co 10 plików lub na końcu
                    # Przekazujemy kopie list, aby uniknąć problemów z wątkami
                    # W tym przypadku przekazujemy jedną listę par (szer,wys)
                    # Dla spójności z oryginalnym kodem, który oczekuje dwóch list,
                    # rozdzielmy to tutaj, chociaż efektywniej byłoby przekazać jedną.
                    current_widths_for_plot = [
                        w for w, h in self.img_resolutions_width_height_pairs
                    ]
                    current_heights_for_plot = [
                        h for w, h in self.img_resolutions_width_height_pairs
                    ]
                    self.resolution_data_updated.emit(
                        current_widths_for_plot, current_heights_for_plot
                    )

            if results["total_scanned"] > 0:
                results["resolution_stats"]["avg_width"] = (
                    results["resolution_stats"]["total_width"]
                    / results["total_scanned"]
                )
                results["resolution_stats"]["avg_height"] = (
                    results["resolution_stats"]["total_height"]
                    / results["total_scanned"]
                )

            # Ostatnia aktualizacja danych do wykresu (jeśli nie było w pętli)
            if total_files % 10 != 0:
                final_widths_for_plot = [
                    w for w, h in self.img_resolutions_width_height_pairs
                ]
                final_heights_for_plot = [
                    h for w, h in self.img_resolutions_width_height_pairs
                ]
                self.resolution_data_updated.emit(
                    final_widths_for_plot, final_heights_for_plot
                )

            self.scan_completed.emit(results)

        except Exception as e:
            detailed_error = (
                f"Krytyczny błąd podczas skanowania: {str(e)}\n{traceback.format_exc()}"
            )
            self.error_occurred.emit(detailed_error)


class ResolutionScannerWidget(QWidget):
    """Widget do skanowania rozdzielczości obrazów."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scanner_thread: ResolutionScannerThread | None = None
        # Zmieniono strukturę self.resolution_data dla uproszczenia
        self.plot_data_widths: List[int] = []
        self.plot_data_heights: List[int] = []
        self.scan_results: Dict | None = None
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        controls_layout = QVBoxLayout()

        self.select_dir_btn = QPushButton("Wybierz katalog do skanowania")
        self.select_dir_btn.clicked.connect(self.select_directory)
        controls_layout.addWidget(self.select_dir_btn)

        self.path_label = QLabel("Nie wybrano katalogu")
        controls_layout.addWidget(self.path_label)

        resolution_layout = QHBoxLayout()
        min_layout = QVBoxLayout()
        min_layout.addWidget(QLabel("Minimalna rozdzielczość (bok):"))
        self.min_size_spinbox = QSpinBox()
        self.min_size_spinbox.setRange(1, 20000)
        self.min_size_spinbox.setValue(500)
        min_layout.addWidget(self.min_size_spinbox)
        resolution_layout.addLayout(min_layout)

        max_layout = QVBoxLayout()
        max_layout.addWidget(QLabel("Maksymalna rozdzielczość (bok):"))
        self.max_size_spinbox = QSpinBox()
        self.max_size_spinbox.setRange(1, 20000)
        self.max_size_spinbox.setValue(4096)
        max_layout.addWidget(self.max_size_spinbox)
        resolution_layout.addLayout(max_layout)
        controls_layout.addLayout(resolution_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        controls_layout.addWidget(self.progress_bar)

        buttons_layout = QHBoxLayout()
        self.start_btn = QPushButton(
            "Rozpocznij skanowanie"
        )  # Dodano przycisk start, aby nie startować od razu po wyborze katalogu
        self.start_btn.clicked.connect(self.trigger_scan_from_ui)
        self.start_btn.setEnabled(False)  # Aktywny po wybraniu katalogu
        buttons_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Zatrzymaj skanowanie")
        self.stop_btn.clicked.connect(self.stop_scanning)
        self.stop_btn.setEnabled(False)
        buttons_layout.addWidget(self.stop_btn)

        self.resize_btn = QPushButton("Przeskaluj duże pliki")
        self.resize_btn.clicked.connect(self.resize_large_files)
        self.resize_btn.setEnabled(False)
        buttons_layout.addWidget(self.resize_btn)
        controls_layout.addLayout(buttons_layout)

        results_layout = QHBoxLayout()
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)

        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        results_layout.addWidget(self.canvas)

        main_layout.addLayout(controls_layout)
        main_layout.addLayout(results_layout)
        self.setLayout(main_layout)

    def select_directory(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Wybierz katalog do skanowania",
            (
                self.path_label.text()
                if self.path_label.text() != "Nie wybrano katalogu"
                else ""
            ),
            QFileDialog.Option.ShowDirsOnly,
        )
        if directory:
            self.path_label.setText(directory)
            self.start_btn.setEnabled(True)  # Uaktywnij przycisk Start
            # self.start_scanning(directory) # Usunięto automatyczne startowanie

    def trigger_scan_from_ui(self):
        directory = self.path_label.text()
        if directory and directory != "Nie wybrano katalogu":
            self.start_scanning(directory)
        else:
            QMessageBox.warning(
                self, "Brak katalogu", "Najpierw wybierz katalog do skanowania."
            )

    def start_scanning(self, directory: str):
        self.results_text.clear()
        self.plot_data_widths.clear()  # Wyczyść dane do wykresu
        self.plot_data_heights.clear()  # Wyczyść dane do wykresu
        self.figure.clear()  # Wyczyść poprzedni wykres
        self.canvas.draw()  # Odśwież pusty canvas

        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.select_dir_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.resize_btn.setEnabled(False)  # Dezaktywuj na czas skanowania

        min_val = self.min_size_spinbox.value()
        max_val = self.max_size_spinbox.value()

        self.scanner_thread = ResolutionScannerThread(
            directory, min_val, max_val, self
        )  # Przekazanie self jako parent
        self.scanner_thread.progress_updated.connect(self.update_progress)
        self.scanner_thread.file_found.connect(self.add_file_result)
        self.scanner_thread.scan_completed.connect(self.show_results)
        self.scanner_thread.error_occurred.connect(self.show_error)
        self.scanner_thread.resolution_data_updated.connect(
            self.update_resolution_plot_data
        )
        self.scanner_thread.finished.connect(
            self.on_scan_thread_finished
        )  # Połączenie z sygnałem finished
        self.scanner_thread.start()

    def stop_scanning(self):
        if self.scanner_thread and self.scanner_thread.isRunning():
            self.results_text.append("\nZatrzymywanie skanowania...")
            self.scanner_thread.stop()
            # Nie blokujemy UI przez wait() tutaj, pozwalamy wątkowi zakończyć się naturalnie
            # Stan przycisków zostanie zaktualizowany w on_scan_thread_finished lub show_error

    def on_scan_thread_finished(self):
        """Slot wywoływany po zakończeniu wątku (normalnie lub przez stop)."""
        self.select_dir_btn.setEnabled(True)
        self.start_btn.setEnabled(self.path_label.text() != "Nie wybrano katalogu")
        self.stop_btn.setEnabled(False)
        # Przycisk resize_btn jest zarządzany w show_results lub show_error

    def update_progress(self, current: int, total: int):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def add_file_result(self, file_path: str, problem: str):
        self.results_text.append(f"Problem: {file_path}\nOpis: {problem}\n")

    def update_resolution_plot_data(self, widths: List[int], heights: List[int]):
        # Zastąp dane nowymi danymi, a nie dołączaj (append)
        self.plot_data_widths = widths
        self.plot_data_heights = heights
        # Wykres będzie rysowany w show_results lub można dodać opcjonalne odświeżanie tutaj
        # self.plot_resolutions() # Można dodać metodę do rysowania wykresu na bieżąco

    def resize_large_files(self):
        if not self.scan_results or not self.scan_results["too_large"]:
            QMessageBox.information(
                self,
                "Informacja",
                "Nie znaleziono plików do przeskalowania lub brak wyników skanowania.",
            )
            return

        reply = QMessageBox.question(
            self,
            "Potwierdzenie",
            (
                f"Znaleziono {len(self.scan_results['too_large'])} plików o zbyt dużej "
                f"rozdzielczości (większej niż {self.max_size_spinbox.value()}px). "
                "Czy chcesz je przeskalować do tej wartości, nadpisując oryginały?"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            max_size_val = self.max_size_spinbox.value()
            resized_count = 0
            errors = []

            # Utwórz kopię listy, aby uniknąć modyfikacji podczas iteracji jeśli byłaby taka potrzeba
            files_to_resize = list(self.scan_results["too_large"])

            self.results_text.append("\nRozpoczynanie przeskalowywania...")
            QApplication.processEvents()  # Pozwól na odświeżenie UI

            for file_path, (original_width, original_height) in files_to_resize:
                try:
                    with Image.open(file_path) as img:
                        if original_width > original_height:
                            new_width = max_size_val
                            new_height = int(
                                original_height * (max_size_val / original_width)
                            )
                        else:
                            new_height = max_size_val
                            new_width = int(
                                original_width * (max_size_val / original_height)
                            )

                        new_width = max(1, new_width)  # Zapobiegaj zerowym wymiarom
                        new_height = max(1, new_height)

                        resized_img = img.resize((new_width, new_height), LANCZOS_ALIAS)

                        # Ustalenie parametrów zapisu w zależności od formatu
                        save_params = {}
                        img_format = img.format
                        if img_format in ["JPEG", "JPG"]:
                            save_params["quality"] = 95
                            save_params["optimize"] = True
                        elif img_format == "PNG":
                            save_params["compress_level"] = 6  # Dobry kompromis
                        # Dla innych formatów Pillow użyje domyślnych ustawień

                        resized_img.save(file_path, **save_params)
                        resized_count += 1
                        self.results_text.append(
                            f"Przeskalowano: {file_path} do {new_width}x{new_height}"
                        )
                except Exception as e:
                    error_msg = f"Błąd podczas przeskalowywania {file_path}: {str(e)}"
                    errors.append(error_msg)
                    self.results_text.append(error_msg)
                QApplication.processEvents()

            summary = (
                f"\nZakończono przeskalowywanie. Przeskalowano {resized_count} plików."
            )
            if errors:
                summary += (
                    "\n\nWystąpiły błędy podczas przeskalowywania:\n"
                    + "\n".join(errors)
                )
            self.results_text.append(summary)
            QMessageBox.information(self, "Podsumowanie przeskalowywania", summary)

            # Po przeskalowaniu, wyniki są nieaktualne, można by zasugerować ponowne skanowanie
            self.resize_btn.setEnabled(False)  # Dezaktywuj, bo pliki zostały zmienione

    def plot_resolutions(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if not self.plot_data_widths or not self.plot_data_heights:
            ax.text(
                0.5,
                0.5,
                "Brak danych do wyświetlenia na wykresie",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            self.canvas.draw()
            return

        # Zliczanie wystąpień każdej unikalnej rozdzielczości
        resolution_counts: Dict[Tuple[int, int], int] = {}
        for w, h in zip(self.plot_data_widths, self.plot_data_heights):
            key = (w, h)
            resolution_counts[key] = resolution_counts.get(key, 0) + 1

        # Przygotowanie danych do wykresu rozproszenia
        # x_coords to unikalne szerokości, y_coords to odpowiadające im unikalne wysokości
        unique_widths = [key[0] for key in resolution_counts.keys()]
        unique_heights = [key[1] for key in resolution_counts.keys()]
        # Rozmiar kropki proporcjonalny do liczby obrazów o danej rozdzielczości
        dot_sizes = [
            count * 30 for count in resolution_counts.values()
        ]  # Skalowanie rozmiaru kropek

        ax.scatter(
            unique_widths,
            unique_heights,
            s=dot_sizes,
            alpha=0.6,
            c="blue",
            edgecolors="w",
            linewidth=0.5,
        )
        ax.set_xlabel("Szerokość (px)")
        ax.set_ylabel("Wysokość (px)")
        ax.set_title("Rozkład unikalnych rozdzielczości obrazów")

        min_size_val = self.min_size_spinbox.value()
        max_size_val = self.max_size_spinbox.value()
        ax.axhline(
            y=min_size_val,
            color="orange",
            linestyle="--",
            alpha=0.5,
            label=f"Min. bok: {min_size_val}px",
        )
        ax.axvline(x=min_size_val, color="orange", linestyle="--", alpha=0.5)
        ax.axhline(
            y=max_size_val,
            color="red",
            linestyle="--",
            alpha=0.5,
            label=f"Max. bok: {max_size_val}px",
        )
        ax.axvline(x=max_size_val, color="red", linestyle="--", alpha=0.5)

        ax.legend(fontsize="small")
        ax.grid(True, linestyle=":", alpha=0.4)

        if unique_widths and unique_heights:
            ax.set_xlim(
                0, max(unique_widths) * 1.1 if unique_widths else max_size_val * 1.2
            )
            ax.set_ylim(
                0, max(unique_heights) * 1.1 if unique_heights else max_size_val * 1.2
            )

        self.canvas.draw()

    def show_results(self, results: Dict):
        self.progress_bar.setVisible(False)
        # Stan przycisków zarządzany przez on_scan_thread_finished

        self.scan_results = results
        self.resize_btn.setEnabled(len(results.get("too_large", [])) > 0)

        self.plot_resolutions()  # Wywołanie metody rysującej wykres

        stats = results.get("resolution_stats", {})
        min_w = stats.get("min_width", "N/A")
        max_w = stats.get("max_width", "N/A")
        avg_w = (
            f"{stats.get('avg_width', 0):.1f}"
            if isinstance(stats.get("avg_width"), (int, float))
            else "N/A"
        )
        min_h = stats.get("min_height", "N/A")
        max_h = stats.get("max_height", "N/A")
        avg_h = (
            f"{stats.get('avg_height', 0):.1f}"
            if isinstance(stats.get("avg_height"), (int, float))
            else "N/A"
        )

        # Liczba unikalnych rozdzielczości z danych użytych do wykresu
        num_unique_resolutions = len(
            set(zip(self.plot_data_widths, self.plot_data_heights))
        )

        summary = f"""Podsumowanie skanowania:
------------------------
Przeskanowano plików: {results.get('total_scanned', 0)}
Znaleziono problemów (z rozdzielczością lub odczytem): {results.get('total_problems', 0)}

Statystyki rozdzielczości:
- Minimalna szerokość: {min_w if min_w != float("inf") else "N/A"}px
- Maksymalna szerokość: {max_w if max_w != 0 else "N/A"}px
- Średnia szerokość: {avg_w}px
- Minimalna wysokość: {min_h if min_h != float("inf") else "N/A"}px
- Maksymalna wysokość: {max_h if max_h != 0 else "N/A"}px
- Średnia wysokość: {avg_h}px

Liczba unikalnych rozdzielczości na wykresie: {num_unique_resolutions}
"""
        current_text = self.results_text.toPlainText()
        separator = "\n" if current_text and not current_text.endswith("\n\n") else ""
        self.results_text.append(f"{separator}{summary}")

        if results.get("too_small"):
            self.results_text.append(
                f"\nPliki o zbyt małej rozdzielczości (<{self.min_size_spinbox.value()}px):"
            )
            for file_path, (width, height) in results["too_small"]:
                self.results_text.append(f"- {file_path} ({width}x{height})")

        if results.get("too_large"):
            self.results_text.append(
                f"\nPliki o zbyt dużej rozdzielczości (>{self.max_size_spinbox.value()}px):"
            )
            for file_path, (width, height) in results["too_large"]:
                self.results_text.append(f"- {file_path} ({width}x{height})")

        self.results_text.append("\nSkanowanie zakończone.")

    def show_error(self, error_msg: str):
        QMessageBox.critical(self, "Błąd krytyczny", error_msg)
        self.progress_bar.setVisible(False)
        # Stan przycisków zarządzany przez on_scan_thread_finished
        # Jeśli błąd wystąpił przed lub w trakcie startu wątku, on_scan_thread_finished może nie być wywołane
        # dlatego dodatkowo ustawiamy tutaj:
        self.select_dir_btn.setEnabled(True)
        self.start_btn.setEnabled(self.path_label.text() != "Nie wybrano katalogu")
        self.stop_btn.setEnabled(False)
        self.resize_btn.setEnabled(False)
        self.results_text.append(f"\nBŁĄD: {error_msg}")

    def closeEvent(self, event: QCloseEvent):
        """Obsługa zdarzenia zamknięcia okna."""
        self.results_text.append("\nZamykanie aplikacji...")
        self.stop_scanning()  # Poproś o zatrzymanie wątku
        if self.scanner_thread and self.scanner_thread.isRunning():
            self.results_text.append("Oczekiwanie na zakończenie pracy wątku...")
            QApplication.processEvents()  # Daj UI szansę na odświeżenie
            if not self.scanner_thread.wait(3000):  # Poczekaj max 3s
                self.results_text.append(
                    "Wątek nie zakończył się w wyznaczonym czasie. Wymuszanie zamknięcia."
                )
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ResolutionScannerWidget()
    window.setWindowTitle("Skaner Rozdzielczości Obrazów")
    window.setGeometry(100, 100, 1100, 750)  # Dostosowany rozmiar okna
    window.show()
    sys.exit(app.exec())
