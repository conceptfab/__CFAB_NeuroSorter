import json
import os

# Ścieżki do katalogów
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(DATA_DIR, "models")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
DATABASE_PATH = os.path.join(DATA_DIR, "database.sqlite")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
CONFIG_DIR = os.path.join(BASE_DIR, "config")

# Konfiguracja modelu
DEFAULT_MODEL_TYPE = "efficientnet"
DEFAULT_NUM_CLASSES = 10
IMAGE_SIZE = (224, 224)

# Domyślne parametry treningu
DEFAULT_TRAINING_PARAMS = {
    "model": "efficientnet",
    "batch_size": 32,
    "num_workers": 16,
    "max_epochs": 50,
    "learning_rate": 0.001,
    "optimizer": "RMSprop",
    "scheduler": "cosine",
    "weight_decay": 1e-4,
    "gradient_clip_val": 0.1,
    "early_stopping_patience": 5,
    "validation_split": 0.2,
    "use_mixed_precision": True,
    "label_smoothing": 0.1,
    "drop_connect_rate": 0.2,
    "momentum": 0.9,
    "epsilon": 0.001,
    "warmup_epochs": 5,
}

# Domyślne parametry augmentacji
DEFAULT_AUGMENTATION_PARAMS = {
    "brightness": 0.2,
    "contrast": 0.2,
    "saturation": 0.2,
    "hue": 0.1,
    "rotation": 15,
    "vertical_flip": False,
    "grayscale": False,
    "perspective": False,
}

# Konfiguracja interfejsu
WINDOW_SIZE = (1200, 800)
MAX_IMAGE_SIZE = 500

# Style ciemnego motywu
DARK_THEME = {
    "bg": "#1e1e1e",  # Ciemniejsze tło
    "fg": "#e0e0e0",  # Jaśniejszy tekst
    "select_bg": "#3a3d41",  # Kolor zaznaczenia
    "select_fg": "#ffffff",  # Biały tekst dla zaznaczenia
    "inactive_bg": "#2d2d2d",  # Tło nieaktywnych elementów
    "inactive_fg": "#cccccc",  # Tekst nieaktywnych elementów
    "highlight_bg": "#264f78",  # Kolor podświetlenia (niebieski)
    "highlight_fg": "#ffffff",  # Biały tekst dla podświetlenia
    "tree_bg": "#1e1e1e",  # Tło dla tabelki
    "tree_fg": "#e0e0e0",  # Tekst dla tabelki
    "menu_bg": "#1e1e1e",  # Tło menu
    "menu_fg": "#e0e0e0",  # Tekst menu
    "border": "#3a3d41",  # Kolor obramowań
}

# Dozwolone rozszerzenia plików
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Tworzenie katalogów jeśli nie istnieją
directories = [DATA_DIR, MODELS_DIR, IMAGES_DIR, LOGS_DIR, REPORTS_DIR, CONFIG_DIR]
for directory in directories:
    os.makedirs(directory, exist_ok=True)

# Cache w pamięci dla ustawień
_settings_cache = {}


def get_settings():
    """Pobiera ustawienia z cache."""
    return _settings_cache.copy()


def update_settings(settings):
    """Aktualizuje cache ustawień."""
    global _settings_cache
    _settings_cache.update(settings)


def clear_settings():
    """Czyści cache ustawień."""
    global _settings_cache
    _settings_cache.clear()


def get_hardware_profile():
    """Pobiera profil sprzętowy z cache."""
    return _settings_cache.get("hardware_profile")


def update_hardware_profile(profile):
    """Aktualizuje profil sprzętowy w cache."""
    _settings_cache["hardware_profile"] = profile


def load_default_settings():
    """Ładuje domyślne ustawienia aplikacji."""
    try:
        settings = {
            # Ustawienia ogólne
            "data_dir": "data",
            "models_dir": "data/models",
            "reports_dir": "data/reports",
            "log_level": "INFO",
            "log_file": "app.log",
            # Ustawienia modelu
            "confidence_threshold": 0.5,
            "use_gpu": True,
            "batch_size": DEFAULT_TRAINING_PARAMS["batch_size"],
            "num_workers": DEFAULT_TRAINING_PARAMS["num_workers"],
            # Ustawienia treningu
            "epochs": DEFAULT_TRAINING_PARAMS["max_epochs"],
            "train_batch_size": DEFAULT_TRAINING_PARAMS["batch_size"],
            "learning_rate": DEFAULT_TRAINING_PARAMS["learning_rate"],
            "optimizer": DEFAULT_TRAINING_PARAMS["optimizer"],
            "scheduler": DEFAULT_TRAINING_PARAMS["scheduler"],
            "weight_decay": DEFAULT_TRAINING_PARAMS["weight_decay"],
            "gradient_clip_val": DEFAULT_TRAINING_PARAMS["gradient_clip_val"],
            "early_stopping_patience": DEFAULT_TRAINING_PARAMS[
                "early_stopping_patience"
            ],
            "validation_split": DEFAULT_TRAINING_PARAMS["validation_split"],
            "use_mixed_precision": DEFAULT_TRAINING_PARAMS["use_mixed_precision"],
            # Ustawienia augmentacji
            "augmentation_params": DEFAULT_AUGMENTATION_PARAMS,
            # Ustawienia interfejsu
            "theme": "Systemowy",
            "language": "Polski",
            "font_size": 11,
            "autosave": True,
            "confirm_exit": True,
            "notifications": True,
            # Ustawienia systemowe
            "memory_limit": 4096,
            "threads": 4,
            "backup_enabled": False,
            "backup_dir": "data/backup",
            "backup_interval": 24,
            # Ustawienia kolorów wykresu
            "chart_train_loss_color": "b",
            "chart_val_loss_color": "r",
            "chart_train_acc_color": "g",
            "chart_val_acc_color": "m",
            "chart_plot_area_background_color": "w",
        }

        # Walidacja wartości
        if not 0 <= settings["confidence_threshold"] <= 1:
            raise ValueError("confidence_threshold musi być między 0 a 1")

        if settings["batch_size"] < 1:
            raise ValueError("batch_size musi być większe od 0")

        if settings["num_workers"] < 0:
            raise ValueError("num_workers nie może być ujemne")

        if settings["epochs"] < 1:
            raise ValueError("epochs musi być większe od 0")

        if settings["train_batch_size"] < 1:
            raise ValueError("train_batch_size musi być większe od 0")

        if settings["learning_rate"] <= 0:
            raise ValueError("learning_rate musi być większe od 0")

        if settings["weight_decay"] < 0:
            raise ValueError("weight_decay nie może być ujemne")

        if settings["gradient_clip_val"] < 0:
            raise ValueError("gradient_clip_val nie może być ujemne")

        if settings["early_stopping_patience"] < 1:
            raise ValueError("early_stopping_patience musi być większe od 0")

        if not 0 < settings["validation_split"] < 1:
            raise ValueError("validation_split musi być między 0 a 1")

        if settings["font_size"] < 8 or settings["font_size"] > 24:
            raise ValueError("font_size musi być między 8 a 24")

        if settings["memory_limit"] < 1024:
            raise ValueError("memory_limit musi być większe lub równe 1024")

        if settings["threads"] < 1:
            raise ValueError("threads musi być większe od 0")

        if not 1 <= settings["backup_interval"] <= 168:
            raise ValueError("backup_interval musi być między 1 a 168")

        return settings

    except Exception as e:
        print(f"Błąd podczas ładowania domyślnych ustawień: {str(e)}")
        # Zwróć podstawowe ustawienia w przypadku błędu
        return {
            "data_dir": "data",
            "models_dir": "data/models",
            "reports_dir": "data/reports",
            "log_level": "INFO",
            "log_file": "app.log",
            "confidence_threshold": 0.5,
            "use_gpu": True,
            "batch_size": 32,
            "num_workers": 4,
            "epochs": 50,
            "train_batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "RMSprop",
            "scheduler": "cosine",
            "weight_decay": 1e-4,
            "gradient_clip_val": 0.1,
            "early_stopping_patience": 5,
            "validation_split": 0.2,
            "use_mixed_precision": True,
            "augmentation_params": DEFAULT_AUGMENTATION_PARAMS,
            "theme": "Systemowy",
            "language": "Polski",
            "font_size": 11,
            "autosave": True,
            "confirm_exit": True,
            "notifications": True,
            "memory_limit": 4096,
            "threads": 4,
            "backup_enabled": False,
            "backup_dir": "data/backup",
            "backup_interval": 24,
            "chart_train_loss_color": "b",
            "chart_val_loss_color": "r",
            "chart_train_acc_color": "g",
            "chart_val_acc_color": "m",
            "chart_plot_area_background_color": "w",
        }


def save_default_settings():
    """Zapisuje domyślne ustawienia do pliku."""
    settings = load_default_settings()
    settings_path = os.path.join(CONFIG_DIR, "settings.json")

    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=4)
