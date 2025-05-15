import logging

from app.core.logger import Logger


def setup_logger(name):
    """Konfiguruje i zwraca logger dla podanego modułu.

    Args:
        name: Nazwa modułu (zazwyczaj __name__)

    Returns:
        logging.Logger: Skonfigurowany logger
    """
    # Użyj głównego loggera aplikacji
    logger = Logger()

    # Ustaw poziom logowania
    logger.logger.setLevel(logging.DEBUG)

    return logger.logger
