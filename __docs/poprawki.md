Zmiana w pliku app/core/logger.py
Zmiana 1: Zmodyfikowanie wszystkich metod loggujących
pythondef info(
    self,
    message: str,
    extra: dict = None,
    func_name: str = None,
    file_name: str = None,
):
    """Loguje informację.

    Args:
        message: Wiadomość do zalogowania
        extra: Dodatkowe dane do zalogowania
        func_name: Nazwa funkcji, w której wystąpiło zdarzenie
        file_name: Nazwa pliku, w którym wystąpiło zdarzenie
    """
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
    self.logger.info(log_message)
Zmiana 2: Zmodyfikowanie pozostałych metod loggujących (warning, debug, critical, exception)
Podobne zmiany należy wprowadzić w metodach warning, debug, critical i exception. Poniżej przedstawiam pełną propozycję zmodyfikowanego kodu dla tych metod:
pythondef warning(
    self,
    message: str,
    extra: dict = None,
    func_name: str = None,
    file_name: str = None,
):
    """Loguje ostrzeżenie.

    Args:
        message: Wiadomość do zalogowania
        extra: Dodatkowe dane do zalogowania
        func_name: Nazwa funkcji, w której wystąpiło zdarzenie
        file_name: Nazwa pliku, w którym wystąpiło zdarzenie
    """
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
    self.logger.warning(log_message)

def debug(
    self,
    message: str,
    extra: dict = None,
    func_name: str = None,
    file_name: str = None,
):
    """Loguje informację debugowania.

    Args:
        message: Wiadomość do zalogowania
        extra: Dodatkowe dane do zalogowania
        func_name: Nazwa funkcji, w której wystąpiło zdarzenie
        file_name: Nazwa pliku, w którym wystąpiło zdarzenie
    """
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
    self.logger.debug(log_message)

def critical(
    self,
    message: str,
    extra: dict = None,
    func_name: str = None,
    file_name: str = None,
):
    """Loguje błąd krytyczny.

    Args:
        message: Wiadomość do zalogowania
        extra: Dodatkowe dane do zalogowania
        func_name: Nazwa funkcji, w której wystąpiło zdarzenie
        file_name: Nazwa pliku, w którym wystąpiło zdarzenie
    """
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
    self.logger.critical(log_message)

def exception(
    self,
    message: str,
    extra: dict = None,
    func_name: str = None,
    file_name: str = None,
):
    """Loguje wyjątek.

    Args:
        message: Wiadomość do zalogowania
        extra: Dodatkowe dane do zalogowania
        func_name: Nazwa funkcji, w której wystąpiło zdarzenie
        file_name: Nazwa pliku, w którym wystąpiło zdarzenie
    """
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
    self.logger.exception(log_message)
Te zmiany pozwolą na bardziej precyzyjne logowanie dla wszystkich poziomów logów, nie tylko dla błędów. Kod zachowuje istniejącą logikę i strukturę, a jedynie rozszerza funkcjonalność pozostałych metod o możliwość dodania informacji o pliku i funkcji, w której wystąpiło zdarzenie.
Teraz można będzie używać wszystkich metod loggera z dodatkowymi parametrami, np.:
pythonlogger.info("Rozpoczęto przetwarzanie", func_name="process_data", file_name="processor.py")
logger.warning("Wykryto nieprawidłowe dane", extra={"id": 123}, func_name="validate", file_name="validator.py")