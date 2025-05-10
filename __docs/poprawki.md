Rozumiem problem - podczas fine-tuningu występuje niepoprawne mapowanie klas, co powoduje, że model się gubi. Potrzebujemy funkcję, która będzie odpowiedzialna za weryfikację struktury katalogów treningowych i walidacyjnych oraz poprawne mapowanie klas na podstawie pliku konfiguracyjnego modelu bazowego. Przygotuję implementację takiej funkcji.
Poniżej przedstawiam zmiany, które należy wprowadzić:
markdown# Zmiany w pliku ai/fine_tuning.py

## 1. Nowa funkcja verify_training_directories

```python
def verify_training_directories(train_dir, val_dir=None, valid_extensions=(".jpg", ".jpeg", ".png", ".bmp")):
    """
    Weryfikuje strukturę katalogów treningowych i walidacyjnych.
    
    Args:
        train_dir: Ścieżka do katalogu treningowego
        val_dir: Ścieżka do katalogu walidacyjnego (opcjonalnie)
        valid_extensions: Dozwolone rozszerzenia plików
        
    Returns:
        dict: Słownik zawierający informacje o strukturze katalogów
    """
    result = {
        "train": {"directories": {}, "total_images": 0, "valid": True, "errors": []},
        "validation": {"directories": {}, "total_images": 0, "valid": True, "errors": []},
    }
    
    # Sprawdź katalog treningowy
    try:
        train_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        if not train_dirs:
            result["train"]["errors"].append("Brak podkatalogów z klasami")
            result["train"]["valid"] = False
        
        for class_dir in train_dirs:
            class_path = os.path.join(train_dir, class_dir)
            image_files = [f for f in os.listdir(class_path) 
                          if os.path.isfile(os.path.join(class_path, f)) and 
                          f.lower().endswith(valid_extensions)]
            
            result["train"]["directories"][class_dir] = len(image_files)
            result["train"]["total_images"] += len(image_files)
            
            if len(image_files) == 0:
                result["train"]["errors"].append(f"Brak obrazów w katalogu {class_dir}")
                result["train"]["valid"] = False
    except Exception as e:
        result["train"]["errors"].append(f"Błąd podczas sprawdzania katalogu treningowego: {str(e)}")
        result["train"]["valid"] = False
    
    # Sprawdź katalog walidacyjny jeśli podano
    if val_dir:
        try:
            val_dirs = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]
            if not val_dirs:
                result["validation"]["errors"].append("Brak podkatalogów z klasami")
                result["validation"]["valid"] = False
            
            for class_dir in val_dirs:
                class_path = os.path.join(val_dir, class_dir)
                image_files = [f for f in os.listdir(class_path) 
                              if os.path.isfile(os.path.join(class_path, f)) and 
                              f.lower().endswith(valid_extensions)]
                
                result["validation"]["directories"][class_dir] = len(image_files)
                result["validation"]["total_images"] += len(image_files)
                
                if len(image_files) == 0:
                    result["validation"]["errors"].append(f"Brak obrazów w katalogu {class_dir}")
                    result["validation"]["valid"] = False
            
            # Sprawdź zgodność klas między katalogami
            train_classes = set(result["train"]["directories"].keys())
            val_classes = set(result["validation"]["directories"].keys())
            
            if train_classes != val_classes:
                missing_in_val = train_classes - val_classes
                missing_in_train = val_classes - train_classes
                
                if missing_in_val:
                    result["validation"]["errors"].append(
                        f"Brakujące klasy w katalogu walidacyjnym: {', '.join(missing_in_val)}"
                    )
                
                if missing_in_train:
                    result["validation"]["errors"].append(
                        f"Klasy w katalogu walidacyjnym nieobecne w treningowym: {', '.join(missing_in_train)}"
                    )
                
                result["validation"]["valid"] = False
        
        except Exception as e:
            result["validation"]["errors"].append(f"Błąd podczas sprawdzania katalogu walidacyjnego: {str(e)}")
            result["validation"]["valid"] = False
    
    return result
2. Nowa funkcja display_directory_structure
pythondef display_directory_structure(verify_result):
    """
    Wyświetla strukturę katalogów treningowych i walidacyjnych na podstawie wyniku weryfikacji.
    
    Args:
        verify_result: Wynik weryfikacji z funkcji verify_training_directories
    """
    print("\nStruktura katalogu treningowego:")
    for class_dir, count in verify_result["train"]["directories"].items():
        print(f"📁 {class_dir}/ ({count} obrazów)")
    print(f"\nŁącznie znaleziono {verify_result['train']['total_images']} obrazów w {len(verify_result['train']['directories'])} katalogach")
    
    if verify_result["validation"]["directories"]:
        print("\nStruktura katalogu walidacyjnego:")
        for class_dir, count in verify_result["validation"]["directories"].items():
            print(f"📁 {class_dir}/ ({count} obrazów)")
        print(f"\nŁącznie znaleziono {verify_result['validation']['total_images']} obrazów w {len(verify_result['validation']['directories'])} katalogach")
    
    # Wyświetl ewentualne błędy
    if verify_result["train"]["errors"]:
        print("\n⚠️ Problemy w katalogu treningowym:")
        for error in verify_result["train"]["errors"]:
            print(f"  - {error}")
    
    if verify_result["validation"]["errors"]:
        print("\n⚠️ Problemy w katalogu walidacyjnym:")
        for error in verify_result["validation"]["errors"]:
            print(f"  - {error}")
3. Nowa funkcja create_class_mapping
pythondef create_class_mapping(model_config, train_directories):
    """
    Tworzy mapowanie między klasami z modelu bazowego a klasami w katalogu treningowym.
    
    Args:
        model_config: Konfiguracja modelu bazowego wczytana z pliku config.json
        train_directories: Lista katalogów klas w zbiorze treningowym
        
    Returns:
        dict: Słownik mapujący nazwy klas na ich indeksy
    """
    # Utwórz odwrotne mapowanie "nazwa klasy -> indeks" z modelu bazowego
    base_class_to_idx = {}
    if "class_names" in model_config:
        base_class_to_idx = {name.lower(): int(idx) for idx, name in model_config["class_names"].items()}
    elif "metadata" in model_config and "class_names" in model_config["metadata"]:
        base_class_to_idx = {name.lower(): int(idx) for idx, name in model_config["metadata"]["class_names"].items()}
    
    # Mapuj klasy treningowe na indeksy z modelu bazowego lub utwórz nowe indeksy
    class_mapping = {}
    max_idx = -1
    if base_class_to_idx:
        max_idx = max([int(idx) for idx in base_class_to_idx.values()], default=-1)
    
    for class_name in sorted(train_directories):
        class_lower = class_name.lower()
        if class_lower in base_class_to_idx:
            # Klasa istnieje w modelu bazowym, użyj jej oryginalnego indeksu
            class_mapping[class_name] = base_class_to_idx[class_lower]
            print(f"  ✓ Klasa '{class_name}' mapowana na istniejący indeks {base_class_to_idx[class_lower]}")
        else:
            # Nowa klasa, przypisz jej nowy indeks
            max_idx += 1
            class_mapping[class_name] = max_idx
            print(f"  + Nowa klasa '{class_name}' mapowana na nowy indeks {max_idx}")
    
    return class_mapping
4. Modyfikacja funkcji fine_tune_model
W funkcji fine_tune_model należy dodać wywołania nowych funkcji na początku:
pythondef fine_tune_model(
    base_model_path,
    train_dir,
    val_dir=None,
    # ... pozostałe parametry ...
):
    """
    Przeprowadza fine-tuning istniejącego modelu na nowym zbiorze danych.
    """
    print("\n=== INICJALIZACJA FINE-TUNINGU ===")
    print(f"Data rozpoczęcia: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    start_training_time = time.time()
    print(f"Model bazowy: {base_model_path}")
    print(f"Katalog treningowy: {train_dir}")
    if val_dir:
        print(f"Katalog walidacyjny: {val_dir}")
    
    # Weryfikacja struktury katalogów
    print("\n=== WERYFIKACJA STRUKTURY KATALOGÓW ===")
    directory_verification = verify_training_directories(train_dir, val_dir)
    display_directory_structure(directory_verification)
    
    # Sprawdź czy struktura katalogów jest poprawna
    if not directory_verification["train"]["valid"]:
        raise ValueError("Problemy ze strukturą katalogu treningowego")
    if val_dir and not directory_verification["validation"]["valid"]:
        raise ValueError("Problemy ze strukturą katalogu walidacyjnego")
    
    # Wczytaj konfigurację modelu bazowego
    base_config_path = os.path.splitext(base_model_path)[0] + "_config.json"
    base_config = {}
    if os.path.exists(base_config_path):
        try:
            with open(base_config_path, "r") as f:
                base_config = json.load(f)
                print(f"\nWczytano konfigurację modelu bazowego: {base_config_path}")
        except Exception as e:
            print(f"Ostrzeżenie: Nie udało się wczytać konfiguracji modelu: {str(e)}")
    
    # Utwórz mapowanie klas
    print("\n=== MAPOWANIE KLAS ===")
    train_directories = directory_verification["train"]["directories"].keys()
    class_mapping = create_class_mapping(base_config, train_directories)
    
    # ... pozostała część funkcji ...
5. Modyfikacja tworzenia słownika class_names w fine_tune_model
W istniejącej implementacji fine_tune_model zmodyfikuj kod odpowiedzialny za tworzenie słownika new_class_names:
python# Zamiast:
new_class_names = {
    str(i): class_name for i, class_name in enumerate(sorted(train_folders))
}

# Użyj:
new_class_names = {
    str(idx): class_name for class_name, idx in class_mapping.items()
}

Powyższe zmiany rozwiążą problem z mapowaniem klas podczas fine-tuningu. Nowe funkcje wykonają następujące zadania:

1. `verify_training_directories` - sprawdzi poprawność struktury katalogów treningowych i walidacyjnych, w tym czy zawierają pliki o odpowiednich rozszerzeniach.
2. `display_directory_structure` - wyświetli przejrzystą strukturę katalogów z informacją o liczbie obrazów w każdej klasie.
3. `create_class_mapping` - utworzy poprawne mapowanie między klasami z modelu bazowego a klasami w zbiorze treningowym, zachowując oryginalne indeksy dla istniejących klas.

Ten kod zapewni, że podczas fine-tuningu klasy będą poprawnie mapowane na podstawie pliku konfiguracyjnego modelu bazowego, co powinno rozwiązać problem z gubionymi klasami.