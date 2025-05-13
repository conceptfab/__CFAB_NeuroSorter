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
        model_path: str,
        preserve_original_classes: bool = True,
        confidence_threshold: float = 0.5,
    ):
        """
        Inicjalizacja wątku sortowania.

        Args:
            files: Lista plików do sortowania
            output_dir: Katalog docelowy
            model_path: Ścieżka do modelu
            preserve_original_classes: Czy zachować oryginalne klasy
            confidence_threshold: Minimalny próg pewności klasyfikacji
        """
        super().__init__()
        self.files = files
        self.output_dir = output_dir
        self.model_path = model_path
        self.preserve_original_classes = preserve_original_classes
        self.confidence_threshold = confidence_threshold
        self.is_paused = False
        self.is_stopped = False

    def run(self):
        """Główna metoda wątku."""
        try:
            # Inicjalizacja sortera
            sorter = ImageSorter(
                model_path=self.model_path,
                output_directory=self.output_dir,
                preserve_original_classes=self.preserve_original_classes,
            )

            # Sortowanie plików
            stats = sorter.sort_images(
                input_directory=os.path.dirname(self.files[0]),
                batch_size=16,
                confidence_threshold=self.confidence_threshold,
            )

            # Emituj wyniki
            self.result_ready.emit(stats)

        except Exception as e:
            self.error_occurred.emit(str(e))

    def stop(self) -> None:
        """Zatrzymuje wątek."""
        self.is_stopped = True

    def pause(self) -> None:
        """Wstrzymuje wątek."""
        self.is_paused = True

    def resume(self) -> None:
        """Wznawia wątek."""
        self.is_paused = False
