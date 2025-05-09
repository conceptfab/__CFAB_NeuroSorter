import json
from typing import Any, Dict, List


def read_config_file(
    input_path: str,
    output_path: str,
) -> None:
    """
    Wczytuje plik konfiguracyjny JSON i wyodrębnia klucze modelu.

    Args:
        input_path (str): Ścieżka do wejściowego pliku JSON
        output_path (str): Ścieżka do wyjściowego pliku JSON
    """
    try:
        # Wczytanie pliku JSON
        with open(input_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        # Tworzenie nowego słownika z wyodrębnionymi kluczami
        extracted_data = {
            "model": {
                "architecture": config_data.get("metadata", {})
                .get("training_params", {})
                .get("config", {})
                .get("model", {})
                .get("architecture"),
                "variant": config_data.get("metadata", {})
                .get("training_params", {})
                .get("config", {})
                .get("model", {})
                .get("variant"),
                "input_size": config_data.get("metadata", {})
                .get("training_params", {})
                .get("config", {})
                .get("model", {})
                .get("input_size"),
                "num_classes": config_data.get("metadata", {})
                .get("training_params", {})
                .get("config", {})
                .get("model", {})
                .get("num_classes"),
            },
            "training": {
                "batch_size": config_data.get("metadata", {})
                .get("training_params", {})
                .get("config", {})
                .get("training", {})
                .get("batch_size"),
                "learning_rate": config_data.get("metadata", {})
                .get("training_params", {})
                .get("config", {})
                .get("training", {})
                .get("learning_rate"),
                "optimizer": config_data.get("metadata", {})
                .get("training_params", {})
                .get("config", {})
                .get("training", {})
                .get("optimizer"),
                "scheduler": config_data.get("metadata", {})
                .get("training_params", {})
                .get("config", {})
                .get("training", {})
                .get("scheduler"),
                "warmup_epochs": config_data.get("metadata", {})
                .get("training_params", {})
                .get("config", {})
                .get("training", {})
                .get("warmup_epochs"),
                "mixed_precision": config_data.get("metadata", {})
                .get("training_params", {})
                .get("config", {})
                .get("training", {})
                .get("mixed_precision"),
                "unfreeze_layers": config_data.get("metadata", {})
                .get("training_params", {})
                .get("config", {})
                .get("training", {})
                .get("unfreeze_layers"),
                "unfreeze_strategy": config_data.get("metadata", {})
                .get("training_params", {})
                .get("config", {})
                .get("training", {})
                .get("unfreeze_strategy"),
            },
            "regularization": {
                "weight_decay": config_data.get("metadata", {})
                .get("training_params", {})
                .get("config", {})
                .get("regularization", {})
                .get("weight_decay"),
                "drop_connect_rate": config_data.get("metadata", {})
                .get("training_params", {})
                .get("config", {})
                .get("regularization", {})
                .get("drop_connect_rate"),
                "dropout_rate": config_data.get("metadata", {})
                .get("training_params", {})
                .get("config", {})
                .get("regularization", {})
                .get("dropout_rate"),
                "label_smoothing": config_data.get("metadata", {})
                .get("training_params", {})
                .get("config", {})
                .get("regularization", {})
                .get("label_smoothing"),
                "swa": config_data.get("metadata", {})
                .get("training_params", {})
                .get("config", {})
                .get("regularization", {})
                .get("swa"),
            },
            "augmentation": {
                "basic": config_data.get("metadata", {})
                .get("training_params", {})
                .get("config", {})
                .get("augmentation", {})
                .get("basic"),
                "mixup": config_data.get("metadata", {})
                .get("training_params", {})
                .get("config", {})
                .get("augmentation", {})
                .get("mixup"),
                "cutmix": config_data.get("metadata", {})
                .get("training_params", {})
                .get("config", {})
                .get("augmentation", {})
                .get("cutmix"),
            },
            "preprocessing": {
                "normalization": config_data.get("metadata", {})
                .get("training_params", {})
                .get("config", {})
                .get("preprocessing", {})
                .get("normalization")
            },
            "monitoring": {
                "early_stopping": config_data.get("metadata", {})
                .get("training_params", {})
                .get("config", {})
                .get("monitoring", {})
                .get("early_stopping"),
                "checkpointing": config_data.get("metadata", {})
                .get("training_params", {})
                .get("config", {})
                .get("monitoring", {})
                .get("checkpointing"),
            },
        }

        # Zapisanie wyodrębnionych danych do nowego pliku
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(extracted_data, f, indent=4, ensure_ascii=False)

        print(f"Pomyślnie zapisano wyodrębnione dane do: {output_path}")

    except FileNotFoundError:
        print(f"Nie znaleziono pliku: {input_path}")
    except json.JSONDecodeError:
        print(f"Błąd dekodowania JSON w pliku: {input_path}")
    except Exception as e:
        print(f"Wystąpił nieoczekiwany błąd: {str(e)}")


if __name__ == "__main__":
    # Ścieżki do plików
    input_file = "data/models/glob_config.json"
    output_file = "data/models/extracted_config.json"

    # Wywołanie funkcji
    read_config_file(input_file, output_file)
