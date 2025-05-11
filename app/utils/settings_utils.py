"""
Moduł zawierający funkcje pomocnicze do walidacji ustawień aplikacji.
"""

import os
from typing import Dict, Any


def validate_settings(settings: Dict[str, Any]) -> bool:
    """
    Waliduje ustawienia aplikacji.
    
    Args:
        settings (Dict[str, Any]): Słownik zawierający ustawienia do walidacji
        
    Returns:
        bool: True jeśli ustawienia są poprawne, False w przeciwnym razie
    """
    try:
        # Sprawdź wymagane katalogi
        required_dirs = ["data_dir", "models_dir", "reports_dir"]
        for dir_key in required_dirs:
            if dir_key not in settings:
                return False
            if not isinstance(settings[dir_key], str):
                return False
            # Utwórz katalog jeśli nie istnieje
            os.makedirs(settings[dir_key], exist_ok=True)

        # Sprawdź ustawienia modelu
        if "confidence_threshold" in settings:
            if not isinstance(settings["confidence_threshold"], (int, float)):
                return False
            if not 0 <= settings["confidence_threshold"] <= 1:
                return False

        if "use_gpu" in settings:
            if not isinstance(settings["use_gpu"], bool):
                return False

        if "batch_size" in settings:
            if not isinstance(settings["batch_size"], int):
                return False
            if settings["batch_size"] < 1:
                return False

        if "num_workers" in settings:
            if not isinstance(settings["num_workers"], int):
                return False
            if settings["num_workers"] < 0:
                return False

        if "auto_load_last_model" in settings:
            if not isinstance(settings["auto_load_last_model"], bool):
                return False

        # Sprawdź ustawienia treningu
        if "epochs" in settings:
            if not isinstance(settings["epochs"], int):
                return False
            if settings["epochs"] < 1:
                return False

        if "train_batch_size" in settings:
            if not isinstance(settings["train_batch_size"], int):
                return False
            if settings["train_batch_size"] < 1:
                return False

        if "learning_rate" in settings:
            if not isinstance(settings["learning_rate"], (int, float)):
                return False
            if settings["learning_rate"] <= 0:
                return False

        # Sprawdź ustawienia interfejsu
        if "theme" in settings:
            if not isinstance(settings["theme"], str):
                return False
            if settings["theme"] not in ["Jasny", "Ciemny", "Systemowy"]:
                return False

        if "language" in settings:
            if not isinstance(settings["language"], str):
                return False
            if settings["language"] not in ["Polski", "English"]:
                return False

        if "font_size" in settings:
            if not isinstance(settings["font_size"], int):
                return False
            if not 8 <= settings["font_size"] <= 24:
                return False

        # Sprawdź ustawienia systemowe
        if "memory_limit" in settings:
            if not isinstance(settings["memory_limit"], int):
                return False
            if settings["memory_limit"] < 1024:
                return False

        if "threads" in settings:
            if not isinstance(settings["threads"], int):
                return False
            if settings["threads"] < 1:
                return False

        if "backup_enabled" in settings:
            if not isinstance(settings["backup_enabled"], bool):
                return False

        if "backup_interval" in settings:
            if not isinstance(settings["backup_interval"], int):
                return False
            if not 1 <= settings["backup_interval"] <= 168:
                return False

        return True

    except Exception:
        return False 