import os
import random  # <--- DODANO: Import modułu random
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
from app.utils.file_utils import fix_task_file_extensions

# Dodaj katalog główny projektu do PYTHONPATH
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def main():
    # Napraw rozszerzenia plików zadań
    fix_task_file_extensions()

    # Inicjalizacja loggera (przeniesiona po inicjalizacji app)
    logger = None  # Zainicjuj logger jako None na początku

    # Ustawienia skalowania DPI
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

    # Inicjalizacja aplikacji musi nastąpić przed utworzeniem
    # jakichkolwiek widgetów, w tym splash
    app = QApplication(sys.argv)

    # --- Ekran powitalny ---
    splash = None  # Zainicjuj jako None
    try:
        # --- POCZĄTEK ZMIAN DLA LOSOWEGO SPLASH SCREEN ---
        splash_dir = os.path.join("resources", "img")
        splash_file_prefix = "splash_"
        # Można dodać więcej obsługiwanych rozszerzeń, np. ".png"
        allowed_extensions = (".jpg", ".jpeg", ".png")

        potential_splash_files = []
        if os.path.isdir(splash_dir):
            for f_name in os.listdir(splash_dir):
                # Sprawdź prefix i rozszerzenie (ignorując wielkość liter dla rozszerzenia)
                if f_name.startswith(splash_file_prefix) and f_name.lower().endswith(
                    allowed_extensions
                ):
                    potential_splash_files.append(f_name)

        if potential_splash_files:
            selected_splash_file = random.choice(potential_splash_files)
            splash_path = os.path.join(splash_dir, selected_splash_file)

            # Komunikat informacyjny o wybranym pliku (opcjonalny, do debugowania)
            # print(f"INFO: Wybrano plik splash: {splash_path}")

            splash_pix = QPixmap(splash_path)
            # Sprawdź, czy obraz został poprawnie załadowany
            if not splash_pix.isNull():
                splash = QSplashScreen(splash_pix, Qt.WindowType.WindowStaysOnTopHint)
                splash.showMessage(
                    "Ładowanie...",
                    Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
                    Qt.GlobalColor.white,
                )
                splash.show()
                app.processEvents()  # Pozwól na odświeżenie UI

                # Ustaw timer na zamknięcie splash screen (np. 2500 ms)
                QTimer.singleShot(2500, splash.close)
            else:
                warning_msg = (
                    "Ostrzeżenie: Nie można załadować obrazu "
                    f"dla ekranu powitalnego: {splash_path}"
                )
                print(warning_msg)
        else:
            # Komunikat, jeśli nie znaleziono pasujących plików lub katalog nie istnieje
            warning_msg = (
                f"Ostrzeżenie: Nie znaleziono odpowiednich plików ekranu powitalnego w '{splash_dir}' "
                f"z prefiksem '{splash_file_prefix}' i rozszerzeniami {allowed_extensions}."
            )
            print(warning_msg)
            # splash pozostaje None, aplikacja będzie kontynuować bez niego
        # --- KONIEC ZMIAN DLA LOSOWEGO SPLASH SCREEN ---

    except (ImportError, AttributeError, Exception) as splash_error:
        warning_msg = (
            "Ostrzeżenie: Nie można utworzyć ekranu powitalnego: " f"{splash_error}"
        )
        print(warning_msg)
        # splash pozostaje None, aplikacja będzie kontynuować bez niego

    # Upewnij się, że katalogi istnieją
    if splash:  # Komunikaty na splashu tylko jeśli splash istnieje
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
        if splash:  # Komunikaty na splashu tylko jeśli splash istnieje
            splash.showMessage(
                "Ładowanie interfejsu użytkownika...",
                Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
                Qt.GlobalColor.white,
            )
            app.processEvents()

        # Inicjalizacja głównego okna
        window = MainWindow()  # Logger jest już dostępny dla MainWindow

        # Pokaż główne okno (splash zamknie się sam po upływie timera)
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
            import torch  # Importuj torch tylko w razie tego konkretnego błędu

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
