import os
import time
from typing import List

from PyQt6.QtCore import QThread, pyqtSignal

from app.sorter.image_sorter import ImageSorter


class BatchClassificationThread(QThread):
    """Wątek do sortowania plików w tle."""

    progress_updated = pyqtSignal(int, int)  # current, total
    result_ready = pyqtSignal(dict)  # result dict
    error_occurred = pyqtSignal(str)  # error message

    def __init__(
        self,
        files: List[str],
        output_dir: str,
        copy_files: bool = True,
        confidence_threshold: float = 0.5,
    ):
        """
        Inicjalizacja wątku sortowania.

        Args:
            files: Lista plików do sortowania
            output_dir: Katalog docelowy
            copy_files: Czy kopiować pliki zamiast przenosić
            confidence_threshold: Minimalny próg pewności klasyfikacji
        """
        super().__init__()
        self.files = files
        self.output_dir = output_dir
        self.copy_files = copy_files
        self.confidence_threshold = confidence_threshold
        self.is_paused = False
        self.is_stopped = False

    def run(self):
        """Główna metoda wątku."""
        try:
            # Inicjalizacja sortera
            sorter = ImageSorter(
                classifier=self.parent().classifier, copy_files=self.copy_files
            )

            # Sortowanie plików
            stats = sorter.sort_directory(
                input_dir=os.path.dirname(self.files[0]),
                output_dir=self.output_dir,
                confidence_threshold=self.confidence_threshold,
                callback=self._progress_callback,
            )

            # Emituj wyniki
            for file_path in self.files:
                if self.is_stopped:
                    break

                while self.is_paused:
                    time.sleep(0.1)
                    if self.is_stopped:
                        break

                try:
                    # Klasyfikacja pliku
                    result = self.parent().classifier.predict(file_path)

                    if result and result["confidence"] >= self.confidence_threshold:
                        category = result["class_name"]
                        status = "Sukces"
                    else:
                        category = "__pliki_bez_kategorii"
                        status = "Brak kategorii"

                    self.result_ready.emit(
                        {
                            "file": file_path,
                            "category": category,
                            "confidence": result["confidence"] if result else 0.0,
                            "status": status,
                        }
                    )

                except Exception as e:
                    self.error_occurred.emit(
                        f"Błąd podczas przetwarzania {file_path}: {str(e)}"
                    )
                    self.result_ready.emit(
                        {
                            "file": file_path,
                            "category": "Błąd",
                            "confidence": 0.0,
                            "status": f"Błąd: {str(e)}",
                        }
                    )

        except Exception as e:
            self.error_occurred.emit(f"Błąd krytyczny: {str(e)}")

    def _progress_callback(self, current: int, total: int):
        """Callback do aktualizacji postępu."""
        self.progress_updated.emit(current, total)

    def pause(self):
        """Wstrzymuje przetwarzanie."""
        self.is_paused = True

    def resume(self):
        """Wznawia przetwarzanie."""
        self.is_paused = False

    def stop(self):
        """Zatrzymuje przetwarzanie."""
        self.is_stopped = True
