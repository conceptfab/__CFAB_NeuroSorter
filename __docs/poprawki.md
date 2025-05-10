Poprawki dla pliku ImageSorter
Poniżej przedstawiam zestaw poprawek dla klasy ImageSorter, które uwzględniają mechanizmy zapobiegające katastrofalnemu zapominaniu, zgodne z wcześniej wprowadzonymi zmianami:
python# Modyfikacje klasy ImageSorter

# 1. Zmiana konstruktora, aby uwzględniał mechanizmy zapobiegające zapominaniu
def __init__(self, model_path, output_directory=None, preserve_original_classes=True, logger=None):
    """
    Inicjalizacja sortera obrazów.
    
    Args:
        model_path: Ścieżka do pliku modelu
        output_directory: Katalog wyjściowy dla posortowanych obrazów
        preserve_original_classes: Czy zachować oryginalne klasy podczas sortowania
        logger: Opcjonalny logger do rejestrowania działań
    """
    self.logger = logger or self._setup_logger()
    self.logger.info(f"Inicjalizacja sortera z modelem: {model_path}")
    
    # Załaduj model
    from ai.classifier import ImageClassifier
    self.model = ImageClassifier(weights_path=model_path)
    
    # Zapisz oryginalne mapowanie klas przy inicjalizacji
    self.original_class_mapping = self.model.class_names.copy()
    self.logger.info(f"Załadowano model z {len(self.original_class_mapping)} klasami")
    
    # Zachowaj flagi konfiguracyjne
    self.preserve_original_classes = preserve_original_classes
    self.output_directory = output_directory
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    
    # Sprawdź, czy model ma włączoną ochronę przed zapominaniem
    self.has_forgetting_prevention = self._check_forgetting_prevention(model_path)
    if not self.has_forgetting_prevention and preserve_original_classes:
        self.logger.warning(
            "Model nie ma włączonych mechanizmów zapobiegających zapominaniu, "
            "ale flaga preserve_original_classes jest włączona. "
            "Może to prowadzić do nieprawidłowych klasyfikacji dla oryginalnych klas."
        )

# 2. Dodanie metody sprawdzającej mechanizmy zapobiegające zapominaniu
def _check_forgetting_prevention(self, model_path):
    """
    Sprawdza, czy model ma włączone mechanizmy zapobiegające zapominaniu.
    
    Args:
        model_path: Ścieżka do pliku modelu
    
    Returns:
        bool: True jeśli model ma włączone mechanizmy zapobiegające zapominaniu
    """
    # Próba wczytania pliku konfiguracyjnego modelu
    config_path = os.path.splitext(model_path)[0] + "_config.json"
    if not os.path.exists(config_path):
        self.logger.warning(f"Nie znaleziono pliku konfiguracyjnego: {config_path}")
        return False
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # Sprawdź, czy w konfiguracji jest sekcja zapobiegająca zapominaniu
        forgetting_prevention = (
            config.get("advanced", {})
            .get("catastrophic_forgetting_prevention", {})
            .get("enable", False)
        )
        
        if forgetting_prevention:
            self.logger.info("Model ma włączone mechanizmy zapobiegające zapominaniu")
            return True
        else:
            self.logger.warning("Model nie ma włączonych mechanizmów zapobiegających zapominaniu")
            return False
    except Exception as e:
        self.logger.error(f"Błąd podczas sprawdzania konfiguracji modelu: {str(e)}")
        return False

# 3. Modyfikacja głównej metody sortowania
def sort_images(self, input_directory, batch_size=16, confidence_threshold=0.0):
    """
    Sortuje obrazy z input_directory do self.output_directory na podstawie klasyfikacji modelu.
    
    Args:
        input_directory: Katalog z obrazami do posortowania
        batch_size: Rozmiar wsadu przy przetwarzaniu
        confidence_threshold: Próg pewności poniżej którego obrazy są pomijane
    
    Returns:
        dict: Statystyki sortowania
    """
    # Sprawdź czy jest ustawiony katalog wyjściowy
    if not self.output_directory:
        raise ValueError("Nie ustawiono katalogu wyjściowego (output_directory)")
    
    # Zbierz wszystkie obrazy z katalogu wejściowego
    self.logger.info(f"Szukam obrazów w katalogu: {input_directory}")
    image_paths = []
    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_paths.append(os.path.join(root, file))
    
    if not image_paths:
        self.logger.warning("Nie znaleziono obrazów w katalogu wejściowym")
        return {"sorted": 0, "skipped": 0, "classes": {}}
    
    self.logger.info(f"Znaleziono {len(image_paths)} obrazów do posortowania")
    
    # Przygotuj statystyki sortowania
    stats = {"sorted": 0, "skipped": 0, "classes": {}}
    
    # Sortuj obrazy wsadowo z paskiem postępu
    from tqdm import tqdm
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        
        # Użyj batch_predict zamiast batch_predict_with_cache dla większej pewności
        results = self.model.batch_predict(batch_paths, return_ranking=True)
        
        # Przetwórz wyniki dla każdego obrazu w batchu
        for j, result in enumerate(results):
            # Pobierz wynik klasyfikacji
            class_id = result['class_id']
            class_name = result['class_name']
            confidence = result['confidence']
            
            # Jeśli pewność jest niższa niż próg, pomiń ten obraz
            if confidence < confidence_threshold:
                self.logger.debug(
                    f"Pomijam obraz {batch_paths[j]} - pewność {confidence:.2f} < {confidence_threshold}"
                )
                stats["skipped"] += 1
                continue
            
            # Utwórz katalog dla klasy, jeśli nie istnieje
            class_dir = os.path.join(self.output_directory, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Kopiuj obraz do odpowiedniego katalogu
            src_path = batch_paths[j]
            dst_path = os.path.join(class_dir, os.path.basename(src_path))
            shutil.copy2(src_path, dst_path)
            
            # Aktualizuj statystyki
            stats["sorted"] += 1
            if class_name not in stats["classes"]:
                stats["classes"][class_name] = 0
            stats["classes"][class_name] += 1
            
            self.logger.debug(
                f"Obraz {src_path} sklasyfikowany jako {class_name} (pewność: {confidence:.2f})"
            )
    
    # Podsumowanie sortowania
    self.logger.info(f"Sortowanie zakończone: {stats['sorted']} obrazów posortowanych, "
                    f"{stats['skipped']} pominiętych")
    for class_name, count in stats["classes"].items():
        self.logger.info(f"  - {class_name}: {count} obrazów")
    
    return stats

# 4. Dodanie metody do sprawdzania dostępnych klas
def get_available_classes(self):
    """
    Zwraca dostępne klasy z modelu.
    
    Returns:
        dict: Mapowanie id -> nazwa klasy
    """
    return self.model.class_names.copy()

# 5. Dodanie metody do testowania modelu na oryginalnych klasach
def evaluate_on_original_classes(self, test_dir, batch_size=16):
    """
    Wykonuje ewaluację modelu na katalogach z oryginalnymi klasami.
    
    Args:
        test_dir: Katalog zawierający podkatalogi dla każdej oryginalnej klasy
        batch_size: Rozmiar wsadu do przetwarzania
    
    Returns:
        dict: Wyniki ewaluacji dla każdej klasy
    """
    self.logger.info(f"Ewaluacja modelu na oryginalnych klasach w katalogu: {test_dir}")
    
    # Przygotuj strukturę wyników
    results = {
        "overall": {"correct": 0, "total": 0, "accuracy": 0.0},
        "classes": {}
    }
    
    # Dla każdego katalogu (zakładamy, że nazwa katalogu to nazwa klasy)
    for class_name in os.listdir(test_dir):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        # Znajdź wszystkie obrazy w katalogu klasy
        image_paths = []
        for root, _, files in os.walk(class_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_paths.append(os.path.join(root, file))
        
        if not image_paths:
            self.logger.warning(f"Brak obrazów dla klasy {class_name}")
            continue
        
        # Inicjalizuj wyniki dla tej klasy
        results["classes"][class_name] = {"correct": 0, "total": len(image_paths), "accuracy": 0.0}
        
        # Klasyfikuj obrazy wsadowo
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_results = self.model.batch_predict(batch_paths)
            
            # Sprawdź wyniki
            for result in batch_results:
                predicted_class = result["class_name"]
                results["overall"]["total"] += 1
                results["classes"][class_name]["total"] += 1
                
                if predicted_class.lower() == class_name.lower():
                    results["overall"]["correct"] += 1
                    results["classes"][class_name]["correct"] += 1
    
    # Oblicz dokładności
    for class_name, data in results["classes"].items():
        data["accuracy"] = data["correct"] / data["total"] if data["total"] > 0 else 0
    
    results["overall"]["accuracy"] = (
        results["overall"]["correct"] / results["overall"]["total"]
        if results["overall"]["total"] > 0 else 0
    )
    
    # Wyświetl podsumowanie
    self.logger.info(f"Ogólna dokładność: {results['overall']['accuracy']:.2%}")
    for class_name, data in results["classes"].items():
        self.logger.info(f"  - {class_name}: {data['accuracy']:.2%} ({data['correct']}/{data['total']})")
    
    return results

# 6. Dodanie metody do konfiguracji logowania
def _setup_logger(self):
    """
    Tworzy i konfiguruje logger.
    
    Returns:
        logging.Logger: Skonfigurowany logger
    """
    logger = logging.getLogger("ImageSorter")
    logger.setLevel(logging.INFO)
    
    # Dodaj handler do konsoli, jeśli nie istnieje
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger
Dodatkowe funkcje do wykorzystania w głównym programie:
pythondef update_sorter_to_use_forgetting_prevention(sorter_instance, config_path):
    """
    Aktualizuje konfigurację sortera, aby używał mechanizmów zapobiegających zapominaniu.
    
    Args:
        sorter_instance: Instancja ImageSorter do zaktualizowania
        config_path: Ścieżka do pliku konfiguracyjnego
    
    Returns:
        bool: True jeśli aktualizacja się powiodła
    """
    try:
        # Wczytaj konfigurację
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # Dodaj sekcję zapobiegającą zapominaniu, jeśli nie istnieje
        if "advanced" not in config:
            config["advanced"] = {}
        
        if "catastrophic_forgetting_prevention" not in config["advanced"]:
            config["advanced"]["catastrophic_forgetting_prevention"] = {
                "enable": True,
                "preserve_original_classes": True,
                "rehearsal": {
                    "use": True,
                    "samples_per_class": 20,
                    "synthetic_samples": True
                },
                "knowledge_distillation": {
                    "use": True,
                    "temperature": 2.0,
                    "alpha": 0.4
                },
                "ewc_regularization": {
                    "use": True,
                    "lambda": 100.0,
                    "fisher_sample_size": 200
                },
                "layer_freezing": {
                    "strategy": "gradual",
                    "freeze_ratio": 0.7
                }
            }
        else:
            # Upewnij się, że opcja enable jest włączona
            config["advanced"]["catastrophic_forgetting_prevention"]["enable"] = True
        
        # Zapisz zaktualizowaną konfigurację
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        
        # Zaktualizuj instancję sortera
        sorter_instance.has_forgetting_prevention = True
        
        return True
    except Exception as e:
        print(f"Błąd podczas aktualizacji konfiguracji: {str(e)}")
        return False

def create_sorter_with_forgetting_prevention(model_path, output_dir=None):
    """
    Tworzy instancję sortera z włączonymi mechanizmami zapobiegającymi zapominaniu.
    
    Args:
        model_path: Ścieżka do pliku modelu
        output_dir: Katalog wyjściowy dla posortowanych obrazów
    
    Returns:
        ImageSorter: Instancja sortera
    """
    # Utwórz instancję sortera
    sorter = ImageSorter(
        model_path=model_path,
        output_directory=output_dir,
        preserve_original_classes=True
    )
    
    # Sprawdź, czy model ma włączone mechanizmy zapobiegające zapominaniu
    if not sorter.has_forgetting_prevention:
        # Dodaj mechanizmy zapobiegające zapominaniu do konfiguracji modelu
        config_path = os.path.splitext(model_path)[0] + "_config.json"
        if os.path.exists(config_path):
            update_sorter_to_use_forgetting_prevention(sorter, config_path)
    
    return sorter
Te poprawki wprowadzają kompleksowe zmiany w klasie ImageSorter, dodając mechanizmy wykrywania i obsługi funkcji zapobiegających katastrofalnemu zapominaniu. Dodatkowo zawierają narzędzia do testowania i analizy modelu pod kątem zachowania oryginalnych klas, co pomoże w monitorowaniu i rozwiązywaniu ewentualnych problemów z zapominaniem.