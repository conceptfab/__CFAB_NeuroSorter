Rozumiem - ważne jest, aby nie zmieniać podstawowej struktury pliku konfiguracyjnego w głównych elementach takich jak model_type, num_classes i struktura class_names w głównym poziomie pliku, ponieważ kod sortowania używa tego mapowania w swojej pracy.
Zmodyfikuję odpowiednio rozwiązanie. Poniżej znajduje się poprawiona wersja funkcji save_with_original_config w klasie ImageClassifier, która zachowa odpowiednią strukturę:
pythondef save_with_original_config(self, save_path, original_config, metadata=None):
    """
    Zapisywanie modelu z zachowaniem KOMPLETNEJ oryginalnej struktury pliku config
    i dodaniem nowych metadanych.

    Args:
        save_path: Ścieżka zapisu modelu
        original_config: Oryginalny plik konfiguracyjny do rozszerzenia
        metadata: Nowe metadane do dodania

    Returns:
        Tuple: (ścieżka_modelu, ścieżka_konfiguracji)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Przygotuj podstawowy słownik checkpoint
    checkpoint = {
        "model_type": self.model_type,
        "num_classes": self.num_classes,
        "model_state_dict": self.model.state_dict(),
        "class_names": self.class_names,
    }

    # Dodajemy metadane do checkpointu
    if metadata:
        checkpoint["metadata"] = metadata

    # Zapisz model
    torch.save(checkpoint, save_path)

    # Przygotuj plik konfiguracyjny bazujący na oryginalnym
    config = original_config.copy()

    # NIE MODYFIKUJ głównych kluczy konfiguracji, tylko metadane wewnątrz
    if "metadata" not in config:
        config["metadata"] = {}

    if metadata:
        for key, value in metadata.items():
            config["metadata"][key] = value

    # Zapisz kompletny plik konfiguracyjny
    config_path = os.path.splitext(save_path)[0] + "_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    return save_path, config_path
Następnie, zaktualizuję fragment kodu w funkcji fine_tune_model, gdzie dodajemy training_time do metadanych:
python# Przed zapisem modelu, oblicz czas treningu
training_time = time.time() - start_training_time

# Przygotuj nowe metadane
new_metadata = {}
# Zachowaj wszystkie oryginalne metadane
if "metadata" in original_config:
    new_metadata = deepcopy(original_config["metadata"])

# Dodaj lub zaktualizuj czas treningu
if "training_time" in new_metadata:
    # Dodaj nowy czas treningu do istniejącego
    new_metadata["training_time"] += training_time
else:
    # Utwórz nowy element czasu treningu
    new_metadata["training_time"] = training_time

# Dodaj lub aktualizuj tylko te elementy, które się zmieniły
new_metadata["finetuning_history"] = finetuning_history
new_metadata["training_details"] = training_details
if task_name:
    new_metadata["task_name"] = task_name

# Zapisz model z zachowaniem oryginalnej konfiguracji
final_classifier.save_with_original_config(final_model_path, original_config, new_metadata)
W ten sposób:

Podstawowa struktura pliku konfiguracyjnego pozostanie niezmieniona
W metadanych zostanie zaktualizowany czas treningu, dodając go do istniejącej wartości
Informacje o historii fine-tuningu i szczegółach treningu zostaną dodane/zaktualizowane w metadanych
Kod sortowania będzie nadal działał poprawnie, ponieważ mapowanie klas w głównym poziomie pliku pozostanie bez zmian

Ta poprawka powinna zapewnić zachowanie wymaganej struktury pliku konfiguracyjnego, jednocześnie przenosząc wszystkie dodatkowe informacje do metadanych.