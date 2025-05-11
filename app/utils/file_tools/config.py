import json
from pathlib import Path

DEFAULT_CONFIG = {
    "folders": {
        "train_folder_name": "__dane_treningowe",
        "valid_folder_name": "__dane_walidacyjne",
    },
    "defaults": {
        "train_split_percent": 80,
        "files_per_category": 100,
    },
    "extensions": {
        "allowed_image_extensions": [
            ".png",
            ".webp",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".gif",
            ".tiff",
            ".tif",
        ]
    },
    "ui": {
        "colors": {
            "primary_color": "#007ACC",
            "background": "#1E1E1E",
            "surface": "#252526",
            "border_color": "#3F3F46",
            "text_color": "#CCCCCC",
            "highlight_color": "#FF0000",
        }
    },
}


class Config:
    """Klasa przechowująca konfigurację aplikacji"""

    def __init__(self):
        self.config_path = Path("config") / "data_splitter_config.json"
        self.config = DEFAULT_CONFIG
        self.load()

    def load(self):
        """Ładuje konfigurację z pliku"""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    loaded_config = json.load(f)
                    # Aktualizuj tylko istniejące klucze
                    self._update_config_recursive(self.config, loaded_config)
            except Exception as e:
                print(f"Błąd ładowania konfiguracji: {e}")

    def save(self):
        """Zapisuje konfigurację do pliku"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Błąd zapisywania konfiguracji: {e}")

    def _update_config_recursive(self, target, source):
        """Aktualizuje konfigurację rekurencyjnie"""
        for key, value in source.items():
            if key in target:
                if isinstance(value, dict) and isinstance(target[key], dict):
                    self._update_config_recursive(target[key], value)
                else:
                    target[key] = value

    def get(self, section, key, default=None):
        """Pobiera wartość z konfiguracji"""
        try:
            return self.config[section][key]
        except KeyError:
            return default

    def set(self, section, key, value):
        """Ustawia wartość w konfiguracji"""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        self.save()


# Singleton konfiguracji
config = Config()
