Rozumiem Twoją frustrację i masz całkowitą rację. W moim poprzednim rozwiązaniu nadal brakuje kluczowych danych z oryginalnego pliku konfiguracyjnego. Pliki konfiguracyjne powinny zawierać wszystkie oryginalne dane, a nie tylko część metadanych.
Oto poprawione rozwiązanie, które faktycznie zachowa całą strukturę oryginalnego pliku:
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
    
    # Kopiujemy wszystkie oryginalne metadane
    if "metadata" in original_config:
        if metadata is None:
            metadata = {}
        
        # Kopiujemy wszystkie oryginalne metadane oprócz tych, które będziemy aktualizować
        for key, value in original_config["metadata"].items():
            if key not in metadata and key != "finetuning_history":
                metadata[key] = value
    
    # Dodajemy metadane do checkpointu
    if metadata:
        checkpoint["metadata"] = metadata
    
    # Zapisz model
    torch.save(checkpoint, save_path)
    
    # Przygotuj plik konfiguracyjny bazujący na oryginalnym
    # WAŻNE: Kopiujemy WSZYSTKO z oryginalnego pliku
    config = original_config.copy()
    
    # Aktualizuj tylko te pola, które faktycznie się zmieniły
    config["model_type"] = self.model_type
    config["num_classes"] = self.num_classes
    config["class_names"] = self.class_names
    
    # Dodaj lub aktualizuj metadane, zachowując resztę oryginalnych danych
    if not "metadata" in config:
        config["metadata"] = {}
    
    if metadata:
        for key, value in metadata.items():
            config["metadata"][key] = value
    
    # Zapisz kompletny plik konfiguracyjny
    config_path = os.path.splitext(save_path)[0] + "_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    
    return save_path, config_path
W funkcji fine_tune_model trzeba również zmodyfikować kod, aby faktycznie zachować wszystkie oryginalne dane:
python# W części funkcji fine_tune_model odpowiedzialnej za zapisywanie modelu:

# Wczytaj oryginalny plik config modelu bazowego, aby go rozszerzyć
original_config = {}
base_config_path = os.path.splitext(base_model_path)[0] + "_config.json"
if os.path.exists(base_config_path):
    try:
        with open(base_config_path, 'r') as f:
            original_config = json.load(f)
            print(f"Wczytano oryginalny plik konfiguracyjny: {base_config_path}")
    except Exception as e:
        print(f"Nie udało się wczytać oryginalnego pliku konfiguracyjnego: {e}")
else:
    print(f"Nie znaleziono oryginalnego pliku konfiguracyjnego. Tworzona jest nowa konfiguracja.")

# Przygotuj historię fine-tuningu
trained_categories = list(new_class_names.values())
base_model_filename = os.path.basename(base_model_path)

# Sprawdź, czy model już ma historię fine-tuningu
finetuning_history = {}
if "metadata" in original_config and "finetuning_history" in original_config["metadata"]:
    finetuning_history = deepcopy(original_config["metadata"]["finetuning_history"])

# Określ numer sesji fine-tuningu
session_nums = [int(k.split('_')[-1]) for k in finetuning_history.keys() 
                if k.startswith("fine_tuning_session_")]
next_session_num = max(session_nums) + 1 if session_nums else 1
session_key = f"fine_tuning_session_{next_session_num}"

# Dodaj nową sesję fine-tuningu
finetuning_history[session_key] = {
    "trained_categories": trained_categories,
    "timestamp": datetime.now().strftime('%Y-%m-%d_%H-%M'),
    "base_model": base_model_filename
}

# Przygotuj szczegóły treningu
training_details = {
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "freeze_ratio": freeze_ratio,
    "optimizer_type": optimizer_type,
    "scheduler_type": scheduler_type
}

# Przygotuj nowe metadane, które zostaną połączone z oryginalnymi
new_metadata = {}
# Zachowaj wszystkie oryginalne metadane
if "metadata" in original_config:
    new_metadata = deepcopy(original_config["metadata"])

# Dodaj lub aktualizuj tylko te elementy, które się zmieniły
new_metadata["finetuning_history"] = finetuning_history
new_metadata["training_details"] = training_details
if task_name:
    new_metadata["task_name"] = task_name

# Zapisz model z kompletnymi metadanymi z oryginalnego modelu plus nowe informacje
final_classifier.save_with_original_config(model_path, original_config, new_metadata)
print(f"Zapisano model: {model_path}")
Kluczowe elementy, które poprawiłem:

Pełna kopia oryginalnego pliku config - teraz zachowujemy wszystkie dane z oryginalnego pliku, a nie tylko część metadanych.
Głębokie kopiowanie (deepcopy) - używamy deepcopy zamiast zwykłej kopii, aby uniknąć problemów z referencjami.
Zachowanie wszystkich oryginalnych metadanych - wszystkie metadane z oryginalnego pliku są zachowywane i łączone z nowymi metadanymi.
Selektywna aktualizacja - aktualizujemy tylko te pola, które faktycznie się zmieniły (model_type, num_classes, class_names), zachowując wszystkie inne oryginalne dane.

Ten kod powinien w pełni zachować wszystkie oryginalne dane i strukturę pliku konfiguracyjnego, dodając jednocześnie nowe informacje o sesji fine-tuningu. To powinno dać plik config, który zawiera WSZYSTKIE dane z oryginalnego pliku oraz nowe informacje o historii fine-tuningu, zachowując kompletną historię modelu.