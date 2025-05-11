"""
Moduł zawierający funkcje pomocnicze do zarządzania klasyfikatorem w aplikacji.
"""

import json
import logging
import os
from datetime import datetime


def find_latest_model_version(models_base_dir):
    """
    Znajduje najnowszą wersję modelu w katalogu wersji.

    Args:
        models_base_dir: Katalog bazowy zawierający wersje modeli

    Returns:
        tuple: (ścieżka_katalogu, ścieżka_modelu, ścieżka_mapowania) lub (None, None, None) jeśli nie znaleziono
    """
    if not os.path.exists(models_base_dir) or not os.path.isdir(models_base_dir):
        logging.warning(f"Katalog modeli {models_base_dir} nie istnieje")
        return None, None, None

    # Znajdź wszystkie katalogi run_*
    version_dirs = []
    for item in os.listdir(models_base_dir):
        full_path = os.path.join(models_base_dir, item)
        if os.path.isdir(full_path) and item.startswith("run_"):
            version_dirs.append(full_path)

    if not version_dirs:
        logging.warning(f"Nie znaleziono katalogów wersji w {models_base_dir}")
        return None, None, None

    # Sortuj według daty w nazwie (format run_YYYYMMDD_HHMMSS)
    version_dirs.sort(reverse=True)  # Sortuj malejąco, aby najnowsze były na początku

    # Wybierz najnowszy katalog
    latest_dir = version_dirs[0]

    # Znajdź model i mapowanie klas w katalogu
    model_path = None
    class_names_path = None

    for item in os.listdir(latest_dir):
        full_path = os.path.join(latest_dir, item)
        if os.path.isfile(full_path):
            if item.endswith(".keras"):
                model_path = full_path
            elif item == "class_names.json":
                class_names_path = full_path

    if not model_path or not class_names_path:
        logging.warning(
            f"Nie znaleziono pliku modelu lub mapowania klas w {latest_dir}"
        )
        return None, None, None

    logging.info(f"Znaleziono najnowszą wersję modelu: {latest_dir}")
    return latest_dir, model_path, class_names_path


def load_class_names(class_names_path):
    """
    Wczytuje mapowanie klas z pliku JSON.

    Args:
        class_names_path: Ścieżka do pliku JSON z mapowaniem klas

    Returns:
        dict: Mapowanie klas lub pusty słownik w przypadku błędu
    """
    try:
        with open(class_names_path, "r", encoding="utf-8") as f:
            class_names = json.load(f)
        return class_names
    except Exception as e:
        logging.error(f"Błąd podczas wczytywania mapowania klas: {str(e)}")
        return {}


def load_latest_classifier(app_config, classifier_class):
    """
    Wczytuje najnowszą wersję klasyfikatora.

    Args:
        app_config: Konfiguracja aplikacji zawierająca MODELS_DIR
        classifier_class: Klasa klasyfikatora do załadowania

    Returns:
        tuple: (klasyfikator, ścieżka_modelu, ścieżka_mapowania) lub (None, None, None) w przypadku błędu
    """
    models_dir = app_config.get("MODELS_DIR")
    if not models_dir:
        logging.error("Nie zdefiniowano katalogu modeli w konfiguracji")
        return None, None, None

    # Znajdź najnowszą wersję
    version_dir, model_path, class_names_path = find_latest_model_version(models_dir)

    if not model_path or not class_names_path:
        logging.error("Nie znaleziono plików modelu lub mapowania klas")
        return None, None, None

    try:
        # Załaduj klasyfikator z odpowiednimi ścieżkami
        classifier = classifier_class(
            weights_path=model_path, class_names_path=class_names_path
        )
        logging.info(f"Załadowano klasyfikator z {model_path} i {class_names_path}")
        return classifier, model_path, class_names_path
    except Exception as e:
        logging.error(f"Błąd podczas ładowania klasyfikatora: {str(e)}")
        return None, None, None
