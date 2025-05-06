import os
import sys
import traceback

# Sprawdź dostępne moduły w PyQt6 i zaimportuj QApplication
try:
    # Standardowe importy dla PyQt6
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtGui import QPixmap
    from PyQt6.QtWidgets import QApplication, QSplashScreen
except ImportError as e:
    # Jeśli podstawowe moduły nie są dostępne, zakończ
    error_msg = (
        "Krytyczny błąd: Nie można zaimportować podstawowych " f"modułów PyQt6: {e}"
    )
    print(error_msg)
    sys.exit(1)

# Importuj główne okno aplikacji i loggera
from app.core.logger import Logger
from app.gui.main_window import MainWindow

# Dodaj katalog główny projektu do PYTHONPATH
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def main():
    # Inicjalizacja loggera (przeniesiona po inicjalizacji app)
    logger = None  # Zainicjuj logger jako None na początku

    # Ustawienia skalowania DPI
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

    # Inicjalizacja aplikacji musi nastąpić przed utworzeniem jakichkolwiek widgetów, w tym splash
    app = QApplication(sys.argv)

    # --- Ekran powitalny ---
    splash = None  # Zainicjuj jako None
    try:
        splash_path = os.path.join("app", "img", "splash.jpg")
        splash_pix = QPixmap(splash_path)
        if not splash_pix.isNull():  # Sprawdź, czy obraz został poprawnie załadowany
            splash = QSplashScreen(splash_pix, Qt.WindowType.WindowStaysOnTopHint)
            splash.showMessage(
                "Ładowanie...",
                Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
                Qt.GlobalColor.white,
            )
            splash.show()
            app.processEvents()  # Pozwól na odświeżenie UI

            # Ustaw timer na 3 sekundy (3000 ms)
            QTimer.singleShot(2500, splash.close)
        else:
            warning_msg = (
                "Ostrzeżenie: Nie można załadować obrazu "
                f"dla ekranu powitalnego: {splash_path}"
            )
            print(warning_msg)

    except (ImportError, AttributeError, Exception) as splash_error:
        warning_msg = (
            "Ostrzeżenie: Nie można utworzyć ekranu powitalnego: " f"{splash_error}"
        )
        print(warning_msg)
        # splash pozostaje None, aplikacja będzie kontynuować bez niego

    # Upewnij się, że katalogi istnieją
    if splash:
        splash.showMessage(
            "Tworzenie katalogów...",
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
            Qt.GlobalColor.white,
        )
        app.processEvents()

    os.makedirs("data/models", exist_ok=True)
    os.makedirs("data/tasks", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    try:
        # Inicjalizacja loggera teraz, przed MainWindow
        logger = Logger()

        # Pokaż komunikat na splash screen przed załadowaniem głównego okna
        if splash:
            splash.showMessage(
                "Ładowanie interfejsu użytkownika...",
                Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
                Qt.GlobalColor.white,
            )
            app.processEvents()

        # Inicjalizacja głównego okna
        window = MainWindow()  # Logger jest już dostępny dla MainWindow

        # Pokaż główne okno (splash zamknie się sam po 3 sekundach)
        window.show()

        sys.exit(app.exec())
    except Exception as e:
        # Użyj loggera jeśli jest zainicjalizowany
        if logger is not None:
            logger.error(f"Błąd krytyczny: {str(e)}")
            logger.error(traceback.format_exc())
        else:
            print(f"Błąd krytyczny przed inicjalizacją loggera: {str(e)}")
            traceback.print_exc()

        if "empty() received an invalid combination of arguments" in str(e):
            print("=== ZŁAPANO BŁĄD empty() ===")
            print(f"Typ błędu: {type(e).__name__}")
            print(f"Treść błędu: {str(e)}")
            print("\nPełny stack trace:")
            traceback.print_exc()

            # Dodatkowe informacje o miejscu błędu
            exc_type, exc_obj, exc_tb = sys.exc_info()
            frame = traceback.extract_tb(exc_tb)[-1]
            print(f"\nBłąd wystąpił w pliku: {frame.filename}")
            print(f"W linii: {frame.lineno}")
            print(f"W funkcji: {frame.name}")
            print(f"Kod linii: {frame.line}")

            # Informacje o wersji PyTorch
            print("\nInformacje o wersji PyTorch:")
            import torch

            print(f"PyTorch: {torch.__version__}")
            print(f"CUDA dostępna: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA wersja: {torch.version.cuda}")
        else:
            # Standardowa obsługa innych błędów
            print(f"Wystąpił błąd: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
