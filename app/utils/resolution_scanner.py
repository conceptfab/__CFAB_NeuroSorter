import os
import traceback
from typing import Dict

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image
from PyQt6.QtCore import QThread, pyqtSignal
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

    def __init__(self, directory: str, min_size: int, max_size: int):
        super().__init__()
        self.directory = directory
        self.min_size = min_size
        self.max_size = max_size
        self._stopped = False
        self.widths = []
        self.heights = []

    def stop(self):
        """Zatrzymuje skanowanie."""
        self._stopped = True

    def run(self):
        """Wykonuje skanowanie katalogu."""
        try:
            # Znajdź wszystkie pliki obrazów
            image_files = []
            supported_formats = (
                ".png",  # PNG
                ".jpg",  # JPEG
                ".jpeg",  # JPEG
                ".gif",  # GIF
                ".bmp",  # Bitmap
                ".tiff",  # TIFF
                ".tif",  # TIFF
                ".webp",  # WebP
            )

            for root, _, files in os.walk(self.directory):
                for file in files:
                    if file.lower().endswith(supported_formats):
                        image_files.append(os.path.join(root, file))

            total_files = len(image_files)
            if total_files == 0:
                self.error_occurred.emit(
                    "Nie znaleziono plików obrazów w wybranym katalogu."
                )
                return

            results = {
                "too_small": [],  # obrazy mniejsze niż min_size
                "too_large": [],  # obrazy większe niż max_size
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
                "format_stats": {},  # statystyki dla każdego formatu
            }

            # Skanuj każdy plik
            for i, file_path in enumerate(image_files):
                if self._stopped:
                    break

                try:
                    with Image.open(file_path) as img:
                        width, height = img.size

                        # Dodaj dane do wykresu
                        self.widths.append((width, height))
                        self.heights.append((width, height))

                        # Aktualizuj statystyki
                        results["resolution_stats"]["min_width"] = min(
                            results["resolution_stats"]["min_width"], width
                        )
                        results["resolution_stats"]["min_height"] = min(
                            results["resolution_stats"]["min_height"], height
                        )
                        results["resolution_stats"]["max_width"] = max(
                            results["resolution_stats"]["max_width"], width
                        )
                        results["resolution_stats"]["max_height"] = max(
                            results["resolution_stats"]["max_height"], height
                        )
                        results["resolution_stats"]["total_width"] += width
                        results["resolution_stats"]["total_height"] += height

                        # Sprawdź rozdzielczość
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

                results["total_scanned"] += 1
                self.progress_updated.emit(i + 1, total_files)

                # Aktualizuj dane do wykresu co 10 plików
                if i % 10 == 0:
                    self.resolution_data_updated.emit(self.widths, self.heights)

            # Ostatnia aktualizacja danych
            self.resolution_data_updated.emit(self.widths, self.heights)

            # Oblicz średnie
            if results["total_scanned"] > 0:
                results["resolution_stats"]["avg_width"] = (
                    results["resolution_stats"]["total_width"]
                    / results["total_scanned"]
                )
                results["resolution_stats"]["avg_height"] = (
                    results["resolution_stats"]["total_height"]
                    / results["total_scanned"]
                )

            self.scan_completed.emit(results)

        except Exception as e:
            self.error_occurred.emit(
                f"Błąd podczas skanowania: {str(e)}\n{traceback.format_exc()}"
            )


class ResolutionScannerWidget(QWidget):
    """Widget do skanowania rozdzielczości obrazów."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scanner_thread = None
        self.resolution_data = {"widths": [], "heights": []}
        self.scan_results = None  # Dodajemy przechowywanie wyników skanowania
        self.init_ui()

    def init_ui(self):
        """Inicjalizuje interfejs użytkownika."""
        main_layout = QVBoxLayout()

        # Górna część - kontrole
        controls_layout = QVBoxLayout()

        # Przycisk wyboru katalogu
        self.select_dir_btn = QPushButton("Wybierz katalog do skanowania")
        self.select_dir_btn.clicked.connect(self.select_directory)
        controls_layout.addWidget(self.select_dir_btn)

        # Etykieta z wybraną ścieżką
        self.path_label = QLabel("Nie wybrano katalogu")
        controls_layout.addWidget(self.path_label)

        # Ustawienia rozdzielczości
        resolution_layout = QHBoxLayout()

        # Minimalna rozdzielczość
        min_layout = QVBoxLayout()
        min_layout.addWidget(QLabel("Minimalna rozdzielczość:"))
        self.min_size = QSpinBox()
        self.min_size.setRange(1, 10000)
        self.min_size.setValue(500)
        min_layout.addWidget(self.min_size)
        resolution_layout.addLayout(min_layout)

        # Maksymalna rozdzielczość
        max_layout = QVBoxLayout()
        max_layout.addWidget(QLabel("Maksymalna rozdzielczość:"))
        self.max_size = QSpinBox()
        self.max_size.setRange(1, 10000)
        self.max_size.setValue(4096)
        max_layout.addWidget(self.max_size)
        resolution_layout.addLayout(max_layout)

        controls_layout.addLayout(resolution_layout)

        # Pasek postępu
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        controls_layout.addWidget(self.progress_bar)

        # Przyciski
        buttons_layout = QHBoxLayout()

        # Przycisk zatrzymania
        self.stop_btn = QPushButton("Zatrzymaj skanowanie")
        self.stop_btn.clicked.connect(self.stop_scanning)
        self.stop_btn.setEnabled(False)
        buttons_layout.addWidget(self.stop_btn)

        # Przycisk przeskalowania
        self.resize_btn = QPushButton("Przeskaluj duże pliki")
        self.resize_btn.clicked.connect(self.resize_large_files)
        self.resize_btn.setEnabled(False)
        buttons_layout.addWidget(self.resize_btn)

        controls_layout.addLayout(buttons_layout)

        # Dolna część - wyniki i wykres
        results_layout = QHBoxLayout()

        # Pole tekstowe z wynikami
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)

        # Wykres
        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figure)
        results_layout.addWidget(self.canvas)

        # Dodaj wszystkie elementy do głównego layoutu
        main_layout.addLayout(controls_layout)
        main_layout.addLayout(results_layout)

        self.setLayout(main_layout)

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

        self.scanner_thread = ResolutionScannerThread(
            directory, self.min_size.value(), self.max_size.value()
        )
        self.scanner_thread.progress_updated.connect(self.update_progress)
        self.scanner_thread.file_found.connect(self.add_file_result)
        self.scanner_thread.scan_completed.connect(self.show_results)
        self.scanner_thread.error_occurred.connect(self.show_error)
        self.scanner_thread.resolution_data_updated.connect(self.update_resolution_data)
        self.scanner_thread.start()

    def stop_scanning(self):
        """Zatrzymuje proces skanowania."""
        if self.scanner_thread and self.scanner_thread.isRunning():
            self.scanner_thread.stop()
            self.stop_btn.setEnabled(False)
            self.results_text.append("\nSkanowanie zatrzymane przez użytkownika.")

    def update_progress(self, current: int, total: int):
        """Aktualizuje pasek postępu."""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def add_file_result(self, file_path: str, problem: str):
        """Dodaje wynik skanowania pojedynczego pliku."""
        self.results_text.append(
            f"Znaleziono problem: {file_path}\nProblem: {problem}\n"
        )

    def update_resolution_data(self, widths: list, heights: list):
        """Aktualizuje dane do wykresu."""
        self.resolution_data["widths"] = widths
        self.resolution_data["heights"] = heights

    def resize_large_files(self):
        """Przeskalowuje pliki większe niż maksymalna rozdzielczość."""
        if not self.scan_results or not self.scan_results["too_large"]:
            QMessageBox.information(
                self,
                "Informacja",
                "Nie znaleziono plików do przeskalowania.",
            )
            return

        reply = QMessageBox.question(
            self,
            "Potwierdzenie",
            f"Znaleziono {len(self.scan_results['too_large'])} plików do "
            "przeskalowania. Czy chcesz kontynuować?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            max_size = self.max_size.value()
            resized_count = 0
            errors = []

            for file_path, (width, height) in self.scan_results["too_large"]:
                try:
                    # Sprawdź rozmiar i rozdzielczość przed zmniejszeniem
                    original_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                    print(f"\nPróba zmniejszenia pliku: {file_path}")
                    print(f"Oryginalny rozmiar: {original_size:.2f} MB")
                    print(f"Oryginalna rozdzielczość: {width}x{height}")

                    with Image.open(file_path) as img:
                        # Oblicz nowe wymiary zachowując proporcje
                        if width > height:
                            new_width = max_size
                            new_height = int(height * (max_size / width))
                        else:
                            new_height = max_size
                            new_width = int(width * (max_size / height))

                        print(f"Nowe wymiary: {new_width}x{new_height}")

                        # Przeskaluj obraz
                        resized_img = img.resize(
                            (new_width, new_height), Image.Resampling.LANCZOS
                        )

                        # Zapisz przeskalowany obraz nadpisując oryginalny plik
                        print(f"Zapisywanie do: {file_path}")
                        resized_img.save(file_path, quality=95)

                        # Sprawdź rozmiar i rozdzielczość po zmniejszeniu
                        new_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                        with Image.open(file_path) as new_img:
                            new_width, new_height = new_img.size

                        print(f"Nowy rozmiar: {new_size:.2f} MB")
                        print(f"Nowa rozdzielczość: {new_width}x{new_height}")
                        print(
                            f"Zmniejszenie rozmiaru: {((original_size - new_size) / original_size * 100):.1f}%"
                        )

                        resized_count += 1
                        print(f"Pomyślnie zmniejszono plik: {file_path}")

                except Exception as e:
                    error_msg = f"Błąd podczas przeskalowywania {file_path}: {str(e)}"
                    print(error_msg)
                    errors.append(error_msg)

            # Wyświetl podsumowanie
            summary = f"Przeskalowano {resized_count} plików."
            if errors:
                summary += "\n\nWystąpiły błędy:\n" + "\n".join(errors)

            QMessageBox.information(self, "Podsumowanie", summary)

    def show_results(self, results: Dict):
        """Wyświetla podsumowanie wyników skanowania i wykres."""
        self.progress_bar.setVisible(False)
        self.select_dir_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.scan_results = results  # Zapisz wyniki skanowania
        self.resize_btn.setEnabled(
            len(results["too_large"]) > 0
        )  # Włącz przycisk jeśli są duże pliki

        # Wyczyść poprzedni wykres
        self.figure.clear()

        # Utwórz nowy wykres
        ax = self.figure.add_subplot(111)

        # Przygotuj dane do wykresu
        widths = [w for w, _ in self.resolution_data["widths"]]
        heights = [h for _, h in self.resolution_data["heights"]]

        if not widths or not heights:
            ax.text(
                0.5,
                0.5,
                "Brak danych do wyświetlenia",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            self.canvas.draw()
            return

        # Oblicz liczbę wystąpień każdej rozdzielczości
        resolution_counts = {}
        for w, h in zip(widths, heights):
            key = (w, h)
            resolution_counts[key] = resolution_counts.get(key, 0) + 1

        # Przygotuj dane do wykresu
        x = [w for (w, _), _ in resolution_counts.items()]
        y = [h for (_, h), _ in resolution_counts.items()]
        sizes = [
            count * 50 for count in resolution_counts.values()
        ]  # Skalujemy wielkość kropek

        # Narysuj wykres rozproszenia
        scatter = ax.scatter(x, y, s=sizes, alpha=0.6, c="blue")
        ax.set_xlabel("Szerokość (px)")
        ax.set_ylabel("Wysokość (px)")
        ax.set_title("Rozkład rozdzielczości obrazów")

        # Dodaj linię referencyjną dla minimalnej i maksymalnej rozdzielczości
        min_size = self.min_size.value()
        max_size = self.max_size.value()
        ax.axhline(y=min_size, color="r", linestyle="--", alpha=0.3)
        ax.axvline(x=min_size, color="r", linestyle="--", alpha=0.3)
        ax.axhline(y=max_size, color="r", linestyle="--", alpha=0.3)
        ax.axvline(x=max_size, color="r", linestyle="--", alpha=0.3)

        # Dodaj siatkę
        ax.grid(True, linestyle="--", alpha=0.3)

        # Dostosuj zakres osi
        ax.set_xlim(min(x) * 0.9 if min(x) > 0 else 0, max(x) * 1.1)
        ax.set_ylim(min(y) * 0.9 if min(y) > 0 else 0, max(y) * 1.1)

        # Odśwież canvas
        self.canvas.draw()

        # Wyświetl podsumowanie w polu tekstowym
        stats = results["resolution_stats"]
        summary = f"""
Podsumowanie skanowania:
------------------------
Przeskanowano plików: {results['total_scanned']}
Znaleziono problemów: {results['total_problems']}

Statystyki rozdzielczości:
- Minimalna szerokość: {stats['min_width']}px
- Maksymalna szerokość: {stats['max_width']}px
- Średnia szerokość: {stats['avg_width']:.1f}px
- Minimalna wysokość: {stats['min_height']}px
- Maksymalna wysokość: {stats['max_height']}px
- Średnia wysokość: {stats['avg_height']:.1f}px

Liczba unikalnych rozdzielczości: {len(resolution_counts)}

Pliki poza skalą:
-----------------
"""

        # Dodaj listę plików o zbyt małej rozdzielczości
        if results["too_small"]:
            summary += "\nPliki o zbyt małej rozdzielczości (<{}px):\n".format(min_size)
            for file_path, (width, height) in results["too_small"]:
                summary += f"- {file_path} ({width}x{height})\n"

        # Dodaj listę plików o zbyt dużej rozdzielczości
        if results["too_large"]:
            summary += "\nPliki o zbyt dużej rozdzielczości (>{}px):\n".format(max_size)
            for file_path, (width, height) in results["too_large"]:
                summary += f"- {file_path} ({width}x{height})\n"

        self.results_text.setText(summary)

    def show_error(self, error_msg: str):
        """Wyświetla komunikat o błędzie."""
        QMessageBox.critical(self, "Błąd", error_msg)
