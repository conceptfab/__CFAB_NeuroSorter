Analiza problemu
Mamy następujące dane:

Katalog treningowy zawiera 3 klasy: indoor_plant, outdoor_plant i tree
Te same klasy istnieją już w modelu bazowym (widoczne w pliku base_A_config.json)
Podczas fine-tuningu kod próbuje zachować oryginalne klasy i dodać nowe, ale robi to nieefektywnie

Główny problem polega na tym, że kod wykonuje niepotrzebne operacje mapowania klas, które już istnieją w modelu bazowym. Zamiast po prostu używać istniejących indeksów klas, tworzy nowe mapowanie.
Proponowane zmiany
Oto proponowane zmiany w pliku ai/fine_tuning.py:
python# Zmiana w funkcji map_class_indices (linie około 110-147)
def map_class_indices(base_class_names, new_class_folders):
    """
    Mapuje indeksy klas z modelu bazowego do nowych klas w zbiorze treningowym.
    
    Args:
        base_class_names: Słownik mapujący indeksy na nazwy klas w modelu bazowym
        new_class_folders: Lista nazw folderów (klas) w zbiorze treningowym
        
    Returns:
        dict: Mapowanie nowych indeksów na indeksy bazowe
    """
    # Odwróć słownik klas bazowego modelu (nazwa klasy -> indeks)
    base_names_to_idx = {name.lower(): int(idx) for idx, name in base_class_names.items()}
    
    # Utwórz mapowanie nowych indeksów na indeksy bazowe
    index_mapping = {}
    for new_idx, folder_name in enumerate(sorted(new_class_folders)):
        # Sprawdzamy czy klasa istnieje w modelu bazowym
        if folder_name.lower() in base_names_to_idx:
            base_idx = base_names_to_idx[folder_name.lower()]
            index_mapping[new_idx] = base_idx
            print(f"  Mapowanie klasy: {folder_name} (nowy indeks {new_idx}) -> (bazowy indeks {base_idx})")
        else:
            # Jeśli to nowa klasa, oznacz jako -1 (będzie wymagała inicjalizacji)
            index_mapping[new_idx] = -1
            print(f"  Nowa klasa: {folder_name} (nowy indeks {new_idx}) -> brak w modelu bazowym")
    
    return index_mapping
python# Modyfikacja w funkcji fine_tune_model (linie około 200-300)
# Zmiana w bloku dotyczącym zachowywania oryginalnych klas

# Zamiast:
if prevent_forgetting and preserve_original_classes:
    print("Zachowywanie oryginalnych klas w mapowaniu...")
    # Zachowaj wszystkie oryginalne klasy
    merged_class_names = base_classifier.class_names.copy()
    
    # Dodaj nowe klasy, kontynuując numerację
    next_idx = max([int(idx) for idx in merged_class_names.keys()]) + 1
    for i, class_name in enumerate(sorted(train_folders)):
        # Sprawdź, czy ta klasa już istnieje w oryginalnym modelu
        if class_name not in merged_class_names.values():
            merged_class_names[str(next_idx)] = class_name
            next_idx += 1
    
    # Użyj merged_class_names zamiast new_class_names
    new_class_names = merged_class_names

# Zastosuj:
if prevent_forgetting and preserve_original_classes:
    print("Inteligentne mapowanie klas - zachowanie indeksów oryginalnych klas...")
    
    # Mapowanie klas treningowych do oryginalnych indeksów
    class_mapping = {}
    for class_name in sorted(train_folders):
        added = False
        # Sprawdź czy klasa istnieje w modelu bazowym
        for idx, base_name in base_classifier.class_names.items():
            if class_name.lower() == base_name.lower():
                class_mapping[class_name] = idx
                print(f"  Klasa {class_name} już istnieje w modelu bazowym z indeksem {idx}")
                added = True
                break
        
        if not added:
            # Dodajemy nową klasę z nowym indeksem
            next_idx = str(max([int(idx) for idx in base_classifier.class_names.keys()]) + 1)
            class_mapping[class_name] = next_idx
            print(f"  Dodajemy nową klasę {class_name} z indeksem {next_idx}")
    
    # Zachowaj wszystkie oryginalne klasy
    new_class_names = base_classifier.class_names.copy()
    
    # Dodaj/zaktualizuj klasy treningowe zgodnie z mapowaniem
    for class_name, idx in class_mapping.items():
        new_class_names[idx] = class_name
    
    print(f"  Finalne mapowanie klas: {len(new_class_names)} klas w modelu")
python# Implementacja ElasticWeightConsolidation w funkcji fine_tune_model
# Dodaj w sekcji zapobiegania katastrofalnemu zapominaniu (około linii 415-475)

def compute_fisher_information(model, data_loader, device):
    """
    Oblicza diagonalną macierz informacji Fishera dla EWC.
    
    Args:
        model: Model, dla którego obliczamy macierz Fishera
        data_loader: DataLoader z danymi do obliczenia macierzy
        device: Urządzenie (CPU/GPU)
        
    Returns:
        dict: Macierz Fishera (tylko wartości diagonalne)
    """
    fisher_diagonal = {}
    # Inicjalizuj macierz Fishera zerami dla każdego parametru
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_diagonal[name] = torch.zeros_like(param.data)
    
    # Ustaw model w trybie ewaluacji
    model.eval()
    
    # Funkcja straty
    criterion = nn.CrossEntropyLoss()
    
    # Przebieg przez dane
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Wyzeruj gradienty
        model.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Oblicz straty
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Aktualizuj diagonalną macierz Fishera dodając kwadraty gradientów
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher_diagonal[name] += param.grad.data.pow(2).clone() / len(data_loader)
    
    return fisher_diagonal
python# W pętli treningowej dodaj obliczanie straty EWC (około linii 500-550)
# Dodaj to w miejscu gdzie jest obliczana strata podczas treningowa

# Dodaj poniższy kod w bloku gdzie jest obliczana strata
if prevent_forgetting and ewc_config and ewc_config.get("use", False) and fisher_diagonal:
    ewc_lambda = ewc_config.get("lambda", 100.0)
    
    # Dodaj regularyzację EWC do straty
    ewc_loss = 0
    for name, param in model.named_parameters():
        if name in fisher_diagonal and name in original_params:
            # Oblicz kwadrat różnicy między aktualnymi parametrami a oryginalnymi
            diff = (param - original_params[name]) ** 2
            # Pomnóż przez ważoność parametru (macierz Fishera)
            ewc_loss += torch.sum(fisher_diagonal[name] * diff)
    
    # Dodaj ważoną stratę EWC do głównej straty
    loss += ewc_lambda * ewc_loss
    
    if batch_idx % 10 == 0:  # Wyświetlaj co 10 batchy
        print(f"  EWC loss: {ewc_loss.item():.6f}, Lambda: {ewc_lambda}")
Dodatkowo - konfiguracja EWC
Dodaj na początku funkcji fine_tune_model konfigurację dla EWC:
python# Domyślna konfiguracja dla EWC, jeśli nie została przekazana
if prevent_forgetting and ewc_config is None:
    ewc_config = {
        "use": True,
        "lambda": 5000.0,  # Współczynnik regularyzacji EWC
        "fisher_sample_size": 200  # Liczba przykładów do obliczenia macierzy Fishera
    }
    print(f"Używam domyślnej konfiguracji EWC: {ewc_config}")
Wnioski
Problem w oryginalnym kodzie polega na niewydajnym mechanizmie mapowania klas, który nie uwzględnia poprawnie istniejących klas modelu. Zaproponowane zmiany:

Usprawniają mapowanie klas, pozwalając na zachowanie oryginalnych indeksów dla klas, które już istnieją w modelu
Implementują technikę Elastic Weight Consolidation (EWC), która pomaga w zapobieganiu katastrofalnemu zapominaniu
Konfigurują EWC z domyślnymi, sensownymi wartościami

EWC działa poprzez określenie, które parametry są krytyczne dla poprzednich zadań i karanie modelu za ich zbyt duże zmiany podczas uczenia nowych klas, co powinno rozwiązać problem z utratą wiedzy o oryginalnych klasach podczas fine-tuningu.