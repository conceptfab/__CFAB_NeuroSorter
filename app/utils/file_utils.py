import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

# Konfiguracja loggera
logger = logging.getLogger(__name__)


def validate_training_directory(training_dir: str) -> Tuple[bool, Optional[str]]:
    """
    Waliduje katalog treningowy.

    Args:
        training_dir: Ścieżka do katalogu treningowego

    Returns:
        Tuple[bool, Optional[str]]: (czy_walidny, komunikat_błędu)
    """
    if not training_dir:
        return False, "Ścieżka do katalogu treningowego jest pusta"

    if not os.path.exists(training_dir):
        return False, f"Katalog treningowy nie istnieje: {training_dir}"

    if not os.path.isdir(training_dir):
        return False, f"Ścieżka nie jest katalogiem: {training_dir}"

    # Sprawdź czy katalog zawiera podkatalogi klas
    has_subdirs = any(
        os.path.isdir(os.path.join(training_dir, d)) for d in os.listdir(training_dir)
    )
    if not has_subdirs:
        return (
            False,
            f"Katalog treningowy nie zawiera podkatalogów klas: {training_dir}",
        )

    # Sprawdź, czy nie ma zagnieżdżonych podkatalogów kategorii
    for category_dir in os.listdir(training_dir):
        category_path = os.path.join(training_dir, category_dir)
        if os.path.isdir(category_path):
            for item in os.listdir(category_path):
                item_path = os.path.join(category_path, item)
                if os.path.isdir(item_path):
                    # Znaleziono zagnieżdżony podkatalog - upewnij się, że nie zawiera obrazów
                    for file in os.listdir(item_path):
                        file_path = os.path.join(item_path, file)
                        if os.path.isfile(file_path) and file.lower().endswith(
                            (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff")
                        ):
                            return (
                                False,
                                f"Znaleziono zagnieżdżony podkatalog z obrazami: {item_path}. Dozwolona jest tylko struktura płaska: kategoria/obrazy",
                            )

    return True, None


def validate_model_path(model_path: str) -> Tuple[bool, Optional[str]]:
    """
    Waliduje ścieżkę do modelu.

    Args:
        model_path: Ścieżka do pliku modelu

    Returns:
        Tuple[bool, Optional[str]]: (czy_walidny, komunikat_błędu)
    """
    if not model_path:
        return False, "Ścieżka do modelu jest pusta"

    if not os.path.exists(model_path):
        return False, f"Model nie istnieje: {model_path}"

    if not os.path.isfile(model_path):
        return False, f"Ścieżka nie jest plikiem: {model_path}"

    return True, None


def validate_validation_directory(val_dir: Optional[str]) -> Tuple[bool, Optional[str]]:
    """
    Waliduje katalog walidacyjny (opcjonalny).

    Args:
        val_dir: Ścieżka do katalogu walidacyjnego

    Returns:
        Tuple[bool, Optional[str]]: (czy_walidny, komunikat_błędu)
    """
    if not val_dir:
        return True, None  # Katalog walidacyjny jest opcjonalny

    if not os.path.exists(val_dir):
        return False, f"Katalog walidacyjny nie istnieje: {val_dir}"

    if not os.path.isdir(val_dir):
        return False, f"Ścieżka nie jest katalogiem: {val_dir}"

    return True, None


def validate_task_file(task_file: str) -> Tuple[bool, Optional[str]]:
    """
    Waliduje plik zadania treningowego.

    Args:
        task_file: Ścieżka do pliku zadania

    Returns:
        Tuple[bool, Optional[str]]: (czy_walidny, komunikat_błędu)
    """
    logger.info(f"Rozpoczęto walidację pliku zadania: {task_file}")

    if not task_file:
        logger.error("Ścieżka do pliku zadania jest pusta")
        return False, "Ścieżka do pliku zadania jest pusta"

    if not os.path.exists(task_file):
        logger.error(f"Plik zadania nie istnieje: {task_file}")
        return False, f"Plik zadania nie istnieje: {task_file}"

    if not task_file.endswith(".json"):
        logger.error(f"Nieprawidłowe rozszerzenie pliku: {task_file}")
        return False, f"Plik zadania musi mieć rozszerzenie .json: {task_file}"

    logger.info("Sprawdzanie poprawności formatu JSON...")
    try:
        with open(task_file, "r", encoding="utf-8") as f:
            task_data = json.load(f)
        logger.info("Plik JSON poprawnie odczytany")
    except json.JSONDecodeError:
        logger.error(f"Błąd dekodowania JSON w pliku: {task_file}")
        return False, f"Plik zadania nie jest poprawnym plikiem JSON: {task_file}"
    except Exception as e:
        logger.error(f"Nieoczekiwany błąd podczas odczytu pliku: {str(e)}")
        return False, f"Błąd podczas odczytu pliku zadania: {str(e)}"

    # Sprawdź wymagane pola na najwyższym poziomie
    logger.info("Sprawdzanie wymaganych pól na najwyższym poziomie...")
    required_top_fields = ["type", "name", "status", "created_at", "config"]
    for field in required_top_fields:
        if field not in task_data:
            logger.error(f"Brak wymaganego pola na najwyższym poziomie: {field}")
            return False, f"Brak wymaganego pola '{field}' w pliku zadania"
    logger.info("Wszystkie wymagane pola na najwyższym poziomie obecne")

    # Sprawdź wymagane pola w sekcji config
    logger.info("Sprawdzanie wymaganych pól w sekcji config...")
    config = task_data.get("config", {})
    required_config_fields = [
        "model_arch",
        "data_dir",
        "epochs",
        "batch_size",
        "learning_rate",
        "optimizer",
    ]
    for field in required_config_fields:
        if field not in config:
            logger.error(f"Brak wymaganego pola w konfiguracji: {field}")
            return False, f"Brak wymaganego pola '{field}' w sekcji config"
    logger.info("Wszystkie wymagane pola w konfiguracji obecne")

    # Sprawdź wymagane pola w sekcji model
    logger.info("Sprawdzanie wymaganych pól w sekcji model...")
    model_config = config.get("model", {})
    if task_data.get("type") == "doszkalanie":
        if "model_path" not in model_config:
            logger.error("Brak ścieżki do modelu w konfiguracji doszkalania")
            return (
                False,
                "Brak wymaganego pola 'model_path' w sekcji model dla zadania doszkalania",
            )
        if not os.path.exists(model_config["model_path"]):
            logger.error(
                f"Model do doszkalania nie istnieje: {model_config['model_path']}"
            )
            return (
                False,
                f"Model do doszkalania nie istnieje: {model_config['model_path']}",
            )
    logger.info("Wszystkie wymagane pola w sekcji model obecne")

    # Sprawdź typy wartości w konfiguracji
    logger.info("Sprawdzanie typów wartości w konfiguracji...")
    if not isinstance(config["epochs"], (int, float)) or config["epochs"] <= 0:
        logger.error(f"Nieprawidłowa wartość epok: {config['epochs']}")
        return False, "Liczba epok musi być dodatnią liczbą"

    if not isinstance(config["batch_size"], int) or config["batch_size"] <= 0:
        logger.error(f"Nieprawidłowa wartość batch_size: {config['batch_size']}")
        return False, "Rozmiar wsadu musi być dodatnią liczbą całkowitą"

    if (
        not isinstance(config["learning_rate"], (int, float))
        or config["learning_rate"] <= 0
    ):
        logger.error(f"Nieprawidłowa wartość learning_rate: {config['learning_rate']}")
        return False, "Współczynnik uczenia musi być dodatnią liczbą"
    logger.info("Wszystkie wartości liczbowe poprawne")

    # Sprawdź poprawność ścieżek
    logger.info("Sprawdzanie ścieżek do katalogów...")
    if not os.path.exists(config["data_dir"]):
        logger.error(f"Katalog treningowy nie istnieje: {config['data_dir']}")
        return (
            False,
            f"Katalog danych treningowych nie istnieje: {config['data_dir']}",
        )

    if config.get("val_dir") and not os.path.exists(config["val_dir"]):
        logger.error(f"Katalog walidacyjny nie istnieje: {config['val_dir']}")
        return (
            False,
            f"Katalog danych walidacyjnych nie istnieje: {config['val_dir']}",
        )
    logger.info("Ścieżki do katalogów poprawne")

    logger.info("Walidacja pliku zadania zakończona pomyślnie")
    return True, None


def validate_task_config(config: dict) -> tuple[bool, str]:
    """
    Waliduje konfigurację zadania przed zapisem.

    Args:
        config (dict): Konfiguracja zadania do walidacji

    Returns:
        tuple[bool, str]: (czy_walidacja_przeszła, komunikat_błędu)
    """
    try:
        # Sprawdź czy konfiguracja nie jest pusta
        if not config:
            return False, "Konfiguracja zadania jest pusta"

        # Sprawdź wymagane pola
        required_fields = ["name", "type", "status", "created_at", "config"]
        for field in required_fields:
            if field not in config:
                return False, f"Brak wymaganego pola: {field}"

        # Sprawdź pola w konfiguracji
        config_fields = config.get("config", {})
        required_config_fields = [
            "model_arch",
            "data_dir",
            "epochs",
            "batch_size",
            "learning_rate",
            "optimizer",
        ]
        for field in required_config_fields:
            if field not in config_fields:
                return False, f"Brak wymaganego pola w konfiguracji: {field}"

        # Sprawdź typy wartości
        if not isinstance(config["name"], str):
            return False, "Nazwa zadania musi być tekstem"
        if not isinstance(config["type"], str):
            return False, "Typ zadania musi być tekstem"
        if not isinstance(config["status"], str):
            return False, "Status zadania musi być tekstem"
        if not isinstance(config["created_at"], str):
            return False, "Data utworzenia musi być tekstem"

        # Sprawdź wartości w konfiguracji
        if not isinstance(config_fields["epochs"], int):
            return False, "Liczba epok musi być liczbą całkowitą"
        if not isinstance(config_fields["batch_size"], int):
            return False, "Rozmiar batcha musi być liczbą całkowitą"
        if not isinstance(config_fields["learning_rate"], (int, float)):
            return False, "Współczynnik uczenia musi być liczbą"

        return True, ""

    except Exception as e:
        return False, f"Błąd podczas walidacji konfiguracji: {str(e)}"


def fix_task_file_extensions():
    """Naprawia rozszerzenia plików zadań w katalogu data/tasks."""
    tasks_dir = Path("data/tasks")
    if not tasks_dir.exists():
        return

    for file_path in tasks_dir.iterdir():
        if not file_path.is_file():
            continue

        # Usuń podwójne rozszerzenia .json
        if file_path.name.endswith(".json.json"):
            new_name = file_path.name[:-5]  # usuń ostatnie .json
            new_path = file_path.parent / new_name
            file_path.rename(new_path)
            file_path = new_path

        # Dodaj rozszerzenie .json jeśli go brakuje
        if not file_path.name.endswith(".json"):
            new_path = file_path.with_suffix(".json")
            file_path.rename(new_path)
