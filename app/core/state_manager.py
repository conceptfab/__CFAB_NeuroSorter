import json
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional


class StateManager:
    """Klasa zarządzająca automatycznym zapisem stanu aplikacji."""

    def __init__(
        self, state_file: str = "app_state.json", auto_save_interval: int = 300
    ):
        """Inicjalizuje menedżer stanu.

        Args:
            state_file: Ścieżka do pliku stanu
            auto_save_interval: Interwał automatycznego zapisu w sekundach
        """
        self.state_file = state_file
        self.auto_save_interval = auto_save_interval
        self._state: Dict[str, Any] = {}
        self._last_save = datetime.now()
        self._auto_save_thread: Optional[threading.Thread] = None
        self._stop_auto_save = threading.Event()

        # Wczytaj stan jeśli plik istnieje
        self.load_state()

    def start_auto_save(self):
        """Uruchamia automatyczny zapis stanu."""
        if self._auto_save_thread is None or not self._auto_save_thread.is_alive():
            self._stop_auto_save.clear()
            self._auto_save_thread = threading.Thread(
                target=self._auto_save_loop, daemon=True
            )
            self._auto_save_thread.start()

    def stop_auto_save(self):
        """Zatrzymuje automatyczny zapis stanu."""
        if self._auto_save_thread and self._auto_save_thread.is_alive():
            self._stop_auto_save.set()
            self._auto_save_thread.join()

    def _auto_save_loop(self):
        """Pętla automatycznego zapisu stanu."""
        while not self._stop_auto_save.is_set():
            time.sleep(self.auto_save_interval)
            if not self._stop_auto_save.is_set():
                self.save_state()

    def set_state(self, key: str, value: Any):
        """Ustawia wartość w stanie.

        Args:
            key: Klucz
            value: Wartość
        """
        self._state[key] = value
        self._last_save = datetime.now()

    def get_state(self, key: str, default: Any = None) -> Any:
        """Pobiera wartość ze stanu.

        Args:
            key: Klucz
            default: Wartość domyślna

        Returns:
            Wartość ze stanu lub wartość domyślna
        """
        return self._state.get(key, default)

    def remove_state(self, key: str):
        """Usuwa wartość ze stanu.

        Args:
            key: Klucz
        """
        if key in self._state:
            del self._state[key]
            self._last_save = datetime.now()

    def clear_state(self):
        """Czyści cały stan."""
        self._state.clear()
        self._last_save = datetime.now()

    def save_state(self):
        """Zapisuje stan do pliku."""
        try:
            # Utwórz kopię zapasową
            if os.path.exists(self.state_file):
                backup_file = f"{self.state_file}.bak"
                os.replace(self.state_file, backup_file)

            # Zapisz nowy stan
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(self._state, f, indent=2, ensure_ascii=False)

            # Usuń kopię zapasową
            if os.path.exists(f"{self.state_file}.bak"):
                os.remove(f"{self.state_file}.bak")

            self._last_save = datetime.now()

        except Exception as e:
            # Przywróć kopię zapasową w przypadku błędu
            if os.path.exists(f"{self.state_file}.bak"):
                os.replace(f"{self.state_file}.bak", self.state_file)
            raise e

    def load_state(self):
        """Wczytuje stan z pliku."""
        if not os.path.exists(self.state_file):
            return

        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                self._state = json.load(f)

        except Exception as e:
            # Spróbuj wczytać z kopii zapasowej
            backup_file = f"{self.state_file}.bak"
            if os.path.exists(backup_file):
                try:
                    with open(backup_file, "r", encoding="utf-8") as f:
                        self._state = json.load(f)
                except:
                    self._state = {}
            else:
                self._state = {}

    def get_last_save_time(self) -> datetime:
        """Zwraca czas ostatniego zapisu.

        Returns:
            Czas ostatniego zapisu
        """
        return self._last_save

    def get_state_size(self) -> int:
        """Zwraca rozmiar stanu w bajtach.

        Returns:
            Rozmiar stanu w bajtach
        """
        return len(json.dumps(self._state).encode("utf-8"))

    def get_state_keys(self) -> list:
        """Zwraca listę kluczy w stanie.

        Returns:
            Lista kluczy
        """
        return list(self._state.keys())
