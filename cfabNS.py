import json
import logging
import os
import sys
import traceback

try:
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtGui import QPixmap
    from PyQt6.QtWidgets import QApplication, QSplashScreen
except ImportError as e:
    msg_part1 = "Krytyczny błąd: Nie można zaimportować podstawowych"
    msg_part2 = f" modułów PyQt6: {e}"
    print(msg_part1 + msg_part2)
    sys.exit(1)

from app.core.logger import Logger
from app.gui.main_window import MainWindow
from app.utils.file_utils import fix_task_file_extensions

# Dodaj katalog główny projektu do PYTHONPATH
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def sanitize_settings(settings_dict):
    """Maskuje wrażliwe dane przed logowaniem."""
    safe_settings = settings_dict.copy()
    sensitive_keys = ["password", "api_key", "token", "secret"]

    for key in sensitive_keys:
        if key in safe_settings:
            safe_settings[key] = "******"

    return safe_settings


def update_splash(splash, message):
    """Aktualizuje komunikat na ekranie powitalnym, jeśli istnieje."""
    if splash:
        splash.showMessage(
            message,
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
            Qt.GlobalColor.white,
        )
        QApplication.processEvents()


def handle_exception(e, logger=None):
    """Centralna funkcja obsługi wyjątków."""
    if logger is not None:
        logger.error(f"Błąd krytyczny: {str(e)}")
        logger.error(traceback.format_exc())
    else:
        print(f"Błąd krytyczny: {str(e)}")
        traceback.print_exc()

    # Specjalna obsługa dla znanych błędów
    if "empty() received an invalid combination of arguments" in str(e):
        error_info = {
            "type": type(e).__name__,
            "message": str(e),
            "location": {
                "file": traceback.extract_tb(sys.exc_info()[2])[-1].filename,
                "line": traceback.extract_tb(sys.exc_info()[2])[-1].lineno,
                "function": traceback.extract_tb(sys.exc_info()[2])[-1].name,
            },
        }

        if logger:
            logger.error("=== ZŁAPANO BŁĄD empty() ===", extra=error_info)
        else:
            print("=== ZŁAPANO BŁĄD empty() ===")
            print(f"Typ błędu: {error_info['type']}")
            print(f"Treść błędu: {error_info['message']}")
            print(f"\nBłąd wystąpił w pliku: {error_info['location']['file']}")
            print(f"W linii: {error_info['location']['line']}")
            print(f"W funkcji: {error_info['location']['function']}")

        # Informacje o wersji PyTorch
        try:
            import torch

            torch_info = {
                "version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": (
                    torch.version.cuda if torch.cuda.is_available() else "N/A"
                ),
            }
            if logger:
                logger.debug("Informacje o PyTorch", extra=torch_info)
            else:
                print("\nInformacje o wersji PyTorch:")
                print(f"PyTorch: {torch_info['version']}")
                print(f"CUDA dostępna: {torch_info['cuda_available']}")
                if torch_info["cuda_available"]:
                    print(f"CUDA wersja: {torch_info['cuda_version']}")
        except ImportError:
            if logger:
                logger.warning("PyTorch nie jest zainstalowany")
            else:
                print("PyTorch nie jest zainstalowany")


def main():
    # Napraw rozszerzenia plików zadań
    fix_task_file_extensions()

    # Wczytaj ustawienia z pliku settings.json
    settings = {}
    try:
        with open("settings.json", "r", encoding="utf-8") as f:
            settings = json.load(f)
            # Dodanie logowania ustawień tylko raz w aplikacji
            print("Wczytano plik ustawień: settings.json")
    except Exception as e:
        print(f"Błąd wczytywania settings.json: {e}")

    log_file = settings.get("log_file", "app.log")

    # Ustawienia skalowania DPI
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

    # Inicjalizacja aplikacji
    app = QApplication(sys.argv)

    # Inicjalizacja loggera
    logger = Logger(log_dir=os.path.dirname(log_file))

    # Ustawienie poziomu logowania w zależności od środowiska
    if os.environ.get("APP_ENV") == "development":
        logger.logger.setLevel(logging.DEBUG)
    else:
        logger.logger.setLevel(logging.INFO)

    # Zaloguj wczytane ustawienia tylko raz
    logger.info("Wczytano plik ustawień: settings.json")

    # Inicjalizacja ekranu powitalnego
    splash = None
    splash_path = os.path.join("resources", "img", "splash.jpg")
    splash_pix = QPixmap(splash_path)

    if not splash_pix.isNull():
        splash = QSplashScreen(splash_pix, Qt.WindowType.WindowStaysOnTopHint)
        splash.show()
        app.processEvents()

        # Ustaw timer na 2.5 sekundy (2500 ms)
        QTimer.singleShot(2500, splash.close)

    # Sprawdź wymagane katalogi
    update_splash(splash, "Weryfikacja struktury katalogów...")

    required_dirs = ["data/models", "data/tasks", "resources/img"]
    missing_dirs = [dir for dir in required_dirs if not os.path.exists(dir)]

    if missing_dirs:
        error_msg = "Brakujące wymagane katalogi:\n" + "\n".join(missing_dirs)
        logger.error(error_msg, extra={"missing_dirs": missing_dirs})
        sys.exit(1)

    try:
        # Przygotowanie do inicjalizacji głównego okna
        update_splash(splash, "Ładowanie interfejsu użytkownika...")

        # Inicjalizacja głównego okna - przekazanie wczytanych ustawień
        window = MainWindow(
            settings=settings
        )  # Zmiana: przekazanie ustawień do MainWindow

        # Pokazanie głównego okna
        window.show()

        # Jeśli mamy splash screen, użyj metody finish()
        if splash:
            splash.finish(window)

        # Bezpieczne zamknięcie loggera przed zakończeniem aplikacji
        logger.shutdown()

        sys.exit(app.exec())
    except Exception as e:
        handle_exception(e, logger)


if __name__ == "__main__":
    main()
