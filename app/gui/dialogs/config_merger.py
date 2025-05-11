import json
from pathlib import Path
from typing import Any, Dict


def generate_output_filename(config: Dict[str, Any]) -> str:
    """
    Generuje nazwę pliku wyjściowego na podstawie variant i num_classes.

    Args:
        config (Dict[str, Any]): Słownik konfiguracji

    Returns:
        str: Wygenerowana nazwa pliku
    """
    variant = config["model"]["variant"].lower().replace("-", "_")
    num_classes = config["model"]["num_classes"]
    return f"{variant}_classes_{num_classes}.json"


def merge_configs(
    default_profile_path: str, extracted_config_path: str, output_path: str
) -> None:
    """
    Łączy strukturę z default_profile.json z wartościami z extracted_config.json

    Args:
        default_profile_path (str): Ścieżka do pliku default_profile.json
        extracted_config_path (str): Ścieżka do pliku extracted_config.json
        output_path (str): Ścieżka do pliku wyjściowego
    """
    # Wczytanie plików
    with open(default_profile_path, "r", encoding="utf-8") as f:
        default_profile = json.load(f)

    with open(extracted_config_path, "r", encoding="utf-8") as f:
        extracted_config = json.load(f)

    # Funkcja pomocnicza do rekurencyjnego aktualizowania wartości
    def update_values(
        default_dict: Dict[str, Any], extracted_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        result = default_dict.copy()

        for key, value in extracted_dict.items():
            if key in result:
                if isinstance(value, dict) and isinstance(result[key], dict):
                    result[key] = update_values(result[key], value)
                else:
                    result[key] = value

        return result

    # Aktualizacja konfiguracji
    merged_config = default_profile.copy()
    merged_config["config"] = update_values(default_profile["config"], extracted_config)

    # Poniższe linie zostaną usunięte lub zakomentowane, aby użyć output_path przekazanego do funkcji
    # output_filename = generate_output_filename(merged_config["config"])
    # output_path = str(Path(output_path).parent / output_filename)

    # Zapisanie wyniku
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged_config, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # Przykład użycia
    default_profile_path = "data/models/default_profile.json"
    extracted_config_path = "data/models/extracted_config.json"
    output_path = "data/models/merged_config.json"

    merge_configs(default_profile_path, extracted_config_path, output_path)
