import logging
from typing import Optional


class Logger:
    """Klasa zarządzająca systemem logowania."""

    def __init__(self, log_dir: str = None):
        """Inicjalizuje logger.

        Args:
            log_dir: Katalog do zapisu logów (opcjonalny)
        """
        self.log_dir = log_dir
        self._setup_logger()

    def _setup_logger(self):
        """Konfiguruje logger."""
        # Utwórz logger
        self.logger = logging.getLogger("CFAB_NeuroSorter")
        self.logger.setLevel(logging.INFO)

        # Sprawdź czy logger już ma handlery
        if not self.logger.handlers:
            # Utwórz formatter
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

            # Utwórz handler do konsoli
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)

            # Dodaj handler do loggera
            self.logger.addHandler(console_handler)

    def info(self, message: str, extra: dict = None):
        """Loguje informację.

        Args:
            message: Wiadomość do zalogowania
            extra: Dodatkowe dane do zalogowania
        """
        if extra:
            self.logger.info(f"{message} - {extra}")
        else:
            self.logger.info(message)

    def warning(self, message: str, extra: dict = None):
        """Loguje ostrzeżenie.

        Args:
            message: Wiadomość do zalogowania
            extra: Dodatkowe dane do zalogowania
        """
        if extra:
            self.logger.warning(f"{message} - {extra}")
        else:
            self.logger.warning(message)

    def error(
        self,
        message: str,
        extra: dict = None,
        func_name: str = None,
        file_name: str = None,
    ):
        """Loguje błąd wraz z informacją o funkcji i pliku."""
        location_info = ""
        if func_name and file_name:
            location_info = f"[{file_name}::{func_name}] "
        elif func_name:
            location_info = f"[{func_name}] "
        elif file_name:
            location_info = f"[{file_name}] "

        log_message = f"{location_info}{message}"
        if extra:
            log_message = f"{log_message} - {extra}"
        self.logger.error(log_message)

    def debug(self, message: str, extra: dict = None):
        """Loguje informację debugowania.

        Args:
            message: Wiadomość do zalogowania
            extra: Dodatkowe dane do zalogowania
        """
        if extra:
            self.logger.debug(f"{message} - {extra}")
        else:
            self.logger.debug(message)

    def critical(self, message: str, extra: dict = None):
        """Loguje błąd krytyczny.

        Args:
            message: Wiadomość do zalogowania
            extra: Dodatkowe dane do zalogowania
        """
        if extra:
            self.logger.critical(f"{message} - {extra}")
        else:
            self.logger.critical(message)

    def exception(self, message: str, extra: dict = None):
        """Loguje wyjątek.

        Args:
            message: Wiadomość do zalogowania
            extra: Dodatkowe dane do zalogowania
        """
        if extra:
            self.logger.exception(f"{message} - {extra}")
        else:
            self.logger.exception(message)

    def get_log_file(self) -> Optional[str]:
        """Zwraca ścieżkę do aktualnego pliku logu.

        Returns:
            Zawsze None, ponieważ logi nie są zapisywane do pliku.
        """
        return None  # Logi nie są zapisywane do pliku

    def clear_old_logs(self, days: int = 7):
        """Usuwa stare pliki logów.

        Args:
            days: Liczba dni po których pliki logów zostaną usunięte (nieużywane).
        """
        # Ta metoda teraz nic nie robi, ponieważ nie zapisujemy logów do plików
        pass
