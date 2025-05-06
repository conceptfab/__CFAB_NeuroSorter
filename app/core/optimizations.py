import json
import os
import threading
import time
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


class CacheManager:
    """Klasa zarządzająca cache'owaniem wyników."""

    def __init__(self, cache_dir: str = "cache", max_size: int = 1000):
        """Inicjalizuje menedżer cache.

        Args:
            cache_dir: Katalog cache
            max_size: Maksymalna liczba elementów w cache
        """
        self.cache_dir = cache_dir
        self.max_size = max_size
        self._cache: Dict[str, Any] = {}
        self._access_times: Dict[str, float] = {}

        # Utwórz katalog cache jeśli nie istnieje
        os.makedirs(cache_dir, exist_ok=True)

    def get(self, key: str) -> Optional[Any]:
        """Pobiera wartość z cache.

        Args:
            key: Klucz

        Returns:
            Wartość z cache lub None
        """
        if key in self._cache:
            self._access_times[key] = time.time()
            return self._cache[key]
        return None

    def set(self, key: str, value: Any):
        """Ustawia wartość w cache.

        Args:
            key: Klucz
            value: Wartość
        """
        # Sprawdź czy cache nie jest pełny
        if len(self._cache) >= self.max_size:
            # Usuń najstarszy element
            oldest_key = min(self._access_times.items(), key=lambda x: x[1])[0]
            del self._cache[oldest_key]
            del self._access_times[oldest_key]

        self._cache[key] = value
        self._access_times[key] = time.time()

    def remove(self, key: str):
        """Usuwa wartość z cache.

        Args:
            key: Klucz
        """
        if key in self._cache:
            del self._cache[key]
            del self._access_times[key]

    def clear(self):
        """Czyści cały cache."""
        self._cache.clear()
        self._access_times.clear()

    def get_size(self) -> int:
        """Zwraca rozmiar cache.

        Returns:
            Liczba elementów w cache
        """
        return len(self._cache)

    def get_oldest(self) -> Optional[Tuple[str, Any]]:
        """Zwraca najstarszy element z cache.

        Returns:
            Krotka (klucz, wartość) lub None
        """
        if not self._cache:
            return None

        oldest_key = min(self._access_times.items(), key=lambda x: x[1])[0]
        return oldest_key, self._cache[oldest_key]

    def get_newest(self) -> Optional[Tuple[str, Any]]:
        """Zwraca najnowszy element z cache.

        Returns:
            Krotka (klucz, wartość) lub None
        """
        if not self._cache:
            return None

        newest_key = max(self._access_times.items(), key=lambda x: x[1])[0]
        return newest_key, self._cache[newest_key]


class MemoryOptimizer:
    """Klasa zarządzająca optymalizacją pamięci."""

    def __init__(self, max_memory_mb: int = 1024):
        """Inicjalizuje optymalizator pamięci.

        Args:
            max_memory_mb: Maksymalna ilość pamięci w MB
        """
        self.max_memory = max_memory_mb * 1024 * 1024  # Konwersja na bajty
        self._current_memory = 0
        self._lock = threading.Lock()

    def allocate(self, size: int) -> bool:
        """Próbuje zaalokować pamięć.

        Args:
            size: Rozmiar w bajtach

        Returns:
            True jeśli udało się zaalokować, False w przeciwnym razie
        """
        with self._lock:
            if self._current_memory + size <= self.max_memory:
                self._current_memory += size
                return True
            return False

    def free(self, size: int):
        """Zwalnia pamięć.

        Args:
            size: Rozmiar w bajtach
        """
        with self._lock:
            self._current_memory = max(0, self._current_memory - size)

    def get_available_memory(self) -> int:
        """Zwraca dostępną pamięć.

        Returns:
            Dostępna pamięć w bajtach
        """
        with self._lock:
            return max(0, self.max_memory - self._current_memory)

    def get_used_memory(self) -> int:
        """Zwraca używaną pamięć.

        Returns:
            Używana pamięć w bajtach
        """
        with self._lock:
            return self._current_memory

    def get_memory_usage_percent(self) -> float:
        """Zwraca procent użycia pamięci.

        Returns:
            Procent użycia pamięci (0-100)
        """
        with self._lock:
            return (self._current_memory / self.max_memory) * 100


class ImageOptimizer:
    """Klasa zarządzająca optymalizacją obrazów."""

    def __init__(self, max_size: Tuple[int, int] = (1024, 1024)):
        """Inicjalizuje optymalizator obrazów.

        Args:
            max_size: Maksymalny rozmiar obrazu (szerokość, wysokość)
        """
        self.max_size = max_size
        self._processed_count = 0
        self._error_count = 0
        self._start_time = None
        self._lock = threading.Lock()

    def _update_stats(self, success: bool = True):
        """Aktualizuje statystyki przetwarzania.

        Args:
            success: Czy operacja się powiodła
        """
        with self._lock:
            self._processed_count += 1
            if not success:
                self._error_count += 1

    def get_stats(self) -> Dict[str, Any]:
        """Zwraca statystyki przetwarzania.

        Returns:
            Słownik ze statystykami
        """
        with self._lock:
            elapsed_time = time.time() - self._start_time if self._start_time else 0
            success_rate = (
                (self._processed_count - self._error_count)
                / max(1, self._processed_count)
                * 100
            )
            return {
                "processed": self._processed_count,
                "errors": self._error_count,
                "success_rate": success_rate,
                "elapsed_time": elapsed_time,
                "avg_time_per_image": elapsed_time / max(1, self._processed_count),
            }

    def optimize_image(self, image_path: str, output_path: Optional[str] = None) -> str:
        """Optymalizuje obraz.

        Args:
            image_path: Ścieżka do obrazu
            output_path: Ścieżka wyjściowa (opcjonalnie)

        Returns:
            Ścieżka do zoptymalizowanego obrazu

        Raises:
            FileNotFoundError: Jeśli plik wejściowy nie istnieje
            ValueError: Jeśli plik nie jest obrazem
            Exception: W przypadku innych błędów
        """
        if not os.path.exists(image_path):
            self._update_stats(False)
            raise FileNotFoundError(f"Plik nie istnieje: {image_path}")

        if output_path is None:
            output_path = image_path

        try:
            # Otwórz obraz
            with Image.open(image_path) as img:
                # Sprawdź czy to jest obraz
                if not hasattr(img, "size"):
                    self._update_stats(False)
                    raise ValueError(f"Plik nie jest obrazem: {image_path}")

                # Konwertuj do RGB jeśli to PNG
                if img.mode in ("RGBA", "LA"):
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])
                    img = background
                elif img.mode != "RGB":
                    img = img.convert("RGB")

                # Zmniejsz rozmiar jeśli za duży
                if img.size[0] > self.max_size[0] or img.size[1] > self.max_size[1]:
                    img.thumbnail(self.max_size, Image.Resampling.LANCZOS)

                # Zapisz zoptymalizowany obraz
                img.save(output_path, "JPEG", quality=85, optimize=True)

            self._update_stats(True)
            return output_path

        except Exception as e:
            self._update_stats(False)
            raise Exception(f"Błąd optymalizacji obrazu {image_path}: {str(e)}")

    def optimize_batch(
        self, image_paths: List[str], output_dir: Optional[str] = None
    ) -> List[str]:
        """Optymalizuje partię obrazów.

        Args:
            image_paths: Lista ścieżek do obrazów
            output_dir: Katalog wyjściowy (opcjonalnie)

        Returns:
            Lista ścieżek do zoptymalizowanych obrazów
        """
        if not image_paths:
            return []

        if output_dir is None:
            output_dir = os.path.dirname(image_paths[0])

        os.makedirs(output_dir, exist_ok=True)

        # Resetuj statystyki
        self._processed_count = 0
        self._error_count = 0
        self._start_time = time.time()

        optimized_paths = []
        errors = []

        for image_path in image_paths:
            output_path = os.path.join(
                output_dir, f"opt_{os.path.basename(image_path)}"
            )
            try:
                optimized_path = self.optimize_image(image_path, output_path)
                optimized_paths.append(optimized_path)
            except Exception as e:
                errors.append((image_path, str(e)))
                print(f"Błąd optymalizacji {image_path}: {str(e)}")

        # Wyświetl podsumowanie
        stats = self.get_stats()
        print("\nPodsumowanie optymalizacji:")
        print(f"Przetworzono obrazów: {stats['processed']}")
        print(f"Błędy: {stats['errors']}")
        print(f"Wskaźnik sukcesu: {stats['success_rate']:.1f}%")
        print(f"Czas wykonania: {stats['elapsed_time']:.1f}s")
        print(f"Średni czas na obraz: {stats['avg_time_per_image']:.2f}s")

        if errors:
            print("\nLista błędów:")
            for path, error in errors:
                print(f"- {path}: {error}")

        return optimized_paths
