Zmiany w pliku ai/classifier.py:
python# W klasie ImageClassifier

def __init__(self, model_type="b0", num_classes=10, weights_path=None, input_size=None):
    # Jeśli istnieje plik konfiguracyjny, pobierz z niego model_type, num_classes i input_size
    config_file_name = "_config.json"
    config_path = None
    if weights_path:
        config_path = os.path.splitext(weights_path)[0] + config_file_name
    config_data = None
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            # Nadpisz model_type i num_classes z configu
            if "model_type" in config_data:
                model_type = config_data["model_type"]
            if "num_classes" in config_data:
                num_classes = config_data["num_classes"]
            # Pobierz input_size z configu
            if "input_size" in config_data:
                self.input_size = tuple(config_data["input_size"]) if isinstance(config_data["input_size"], list) else (config_data["input_size"], config_data["input_size"])
        except Exception as e:
            print(f"BŁĄD: Nie udało się wczytać pliku konfiguracyjnego {config_path}: {e}")
    
    # Ustaw domyślny input_size, jeśli nie został określony
    if input_size:
        self.input_size = input_size if isinstance(input_size, tuple) else (input_size, input_size)
    elif not hasattr(self, 'input_size'):
        self.input_size = (224, 224)  # Domyślny rozmiar
        
    # ... reszta kodu init ...
    
    # Standardowe przekształcenia dla obrazów z dynamicznym rozmiarem
    self.transform = transforms.Compose(
        [
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
Zmiany w pliku ai/fine_tuning.py:
Gdy tworzymy nowy model podczas fine-tuningu, musimy przekazać input_size z pliku JSON zadania:
pythondef fine_tune_model(
    base_model_path,
    train_dir,
    val_dir=None,
    num_epochs=10,
    batch_size=16,
    learning_rate=0.0001,
    freeze_ratio=0.8,
    output_dir="./data/models",
    optimizer_type="adamw",
    scheduler_type="plateau",
    device=None,
    progress_callback=None,
    should_stop_callback=None,
    label_smoothing=0.1,
    weight_decay=0.01,
    warmup_epochs=1,
    use_mixup=False,
    use_mixed_precision=False,
    task_name=None,
    prevent_forgetting=True,
    preserve_original_classes=True,
    rehearsal_config=None,
    knowledge_distillation_config=None,
    ewc_config=None,
    layer_freezing_config=None,
    augmentation_params=None,
    preprocessing_params=None,
    use_green_diffusion=False,
    early_stopping_patience=5,
    input_size=None,  # Dodajemy nowy parametr
):
    # ...
    
    # Ładując model bazowy, przekazujemy input_size
    print(f"\nŁadowanie modelu bazowego: {base_model_path}")
    try:
        base_classifier = ImageClassifier(weights_path=base_model_path, input_size=input_size)
        base_classifier.model.to(device)
        # ...
    except Exception as e:
        raise RuntimeError(f"Nie udało się załadować modelu bazowego z {base_model_path}: {e}")
    
    # ...
    
    # Przy tworzeniu datasetu, przekazujemy odpowiedni input_size do przekształceń:
    val_transform = get_default_transforms(config={"image_size": base_classifier.input_size})
    train_transform = get_augmentation_transforms(config={"image_size": base_classifier.input_size})
    
    # ...
    
    # Kod dalszego fine-tuningu...
Dostosowanie funkcji w ai/optimized_training.py:
Dodajemy obsługę input_size w funkcji train_model_optimized:
pythondef train_model_optimized(
    model,
    train_dir,
    val_dir=None,
    num_epochs=10,
    batch_size=None,
    learning_rate=0.001,
    device=None,
    progress_callback=None,
    freeze_backbone=False,
    lr_scheduler_type="plateau",
    early_stopping=True,
    mixup=True,
    label_smoothing=0.1,
    weight_decay=0.03,
    optimizer_type="adamw",
    profiler=None,
    augmentation_mode="extended",
    augmentation_params=None,
    should_stop_callback=None,
    use_cross_validation=False,
    k_folds=5,
    freeze_layers_ratio=0.7,
    model_log_path=None,
    model_source_info=None,
    output_dir=None,
    model_save_path=None,
    input_size=None,  # Dodajemy nowy parametr
):
    # ...
    
    # Ustalamy input_size do przekazania do transformacji
    if input_size is None and hasattr(model, "input_size"):
        input_size = model.input_size
    elif input_size is None:
        input_size = (224, 224)  # Domyślny rozmiar
    
    # Przygotuj dane treningowe
    train_transform = None
    if augmentation_mode == "basic":
        train_transform = get_augmentation_transforms(config={"image_size": input_size})
    elif augmentation_mode == "extended":
        # Upewnij się, że augmentation_params ma image_size
        if augmentation_params is None:
            augmentation_params = {}
        augmentation_params["image_size"] = input_size
        train_transform = get_extended_augmentation_transforms(image_size=input_size, params=augmentation_params)
    else:
        train_transform = get_default_transforms(config={"image_size": input_size})
    
    val_transform = get_default_transforms(config={"image_size": input_size})
    
    # ... reszta kodu ...
Funkcja główna (odczytująca JSON zadania):
Teraz potrzebujemy funkcji głównej, która będzie odczytywać plik JSON zadania i przekazywać parametr input_size do odpowiednich funkcji:
pythondef train_from_json_config(json_config_path):
    """
    Rozpoczyna trening na podstawie pliku JSON z konfiguracją zadania.
    
    Args:
        json_config_path: Ścieżka do pliku JSON z konfiguracją zadania
    """
    with open(json_config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Wyodrębnij parametry z JSON
    training_config = config.get('config', {})
    model_config = training_config.get('model', {})
    
    # Pobierz input_size
    input_size = model_config.get('input_size', 224)
    if isinstance(input_size, int):
        input_size = (input_size, input_size)
    
    # Inne parametry...
    architecture = model_config.get('architecture', 'EfficientNet')
    variant = model_config.get('variant', 'EfficientNet-B0')
    num_classes = model_config.get('num_classes', 10)
    
    # ... pobieranie innych parametrów z JSONa ...
    
    # Tworzenie modelu z odpowiednim input_size
    model = get_model(
        model_arch=variant.replace('EfficientNet-', '').lower(),
        num_classes=num_classes,
        input_size=input_size
    )
    
    # Przekaż input_size do funkcji trenującej
    result = train_model_optimized(
        model=model,
        train_dir=training_config.get('train_dir'),
        val_dir=training_config.get('val_dir'),
        input_size=input_size,
        # ... inne parametry ...
    )
    
    return result
Modyfikacja ai/models.py:
Dodajemy obsługę input_size w funkcji get_model:
pythondef get_model(
    model_arch: str,
    num_classes: Optional[int] = None,
    logger: Optional[Callable] = None,
    drop_connect_rate: float = 0.2,
    dropout_rate: float = 0.3,
    input_size: Optional[Union[int, Tuple[int, int]]] = None
) -> nn.Module:
    """
    Tworzy model o podanej architekturze.

    Args:
        model_arch: Architektura modelu (np. 'b0' dla EfficientNet, '50' dla ResNet)
        num_classes: Liczba klas (opcjonalnie)
        logger: Funkcja do logowania (opcjonalnie)
        drop_connect_rate: Wartość drop connect rate dla EfficientNet (opcjonalnie)
        dropout_rate: Wartość dropout rate dla warstw klasyfikacji (opcjonalnie)
        input_size: Rozmiar wejściowy obrazu (opcjonalnie)

    Returns:
        nn.Module: Model PyTorch
    """
    # ... istniejący kod ...
    
    # Dodanie input_size jako atrybutu modelu
    if input_size is not None:
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        setattr(model, "input_size", input_size)
    else:
        setattr(model, "input_size", (224, 224))  # Domyślny rozmiar
        
    return model
Dzięki tym zmianom, input_size będzie dynamicznie odczytywany z pliku JSON zadania i używany w całym procesie treningu, zamiast być ustawionym na stałe jako 224x224.