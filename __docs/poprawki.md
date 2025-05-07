to 0.35
Analiza i propozycje optymalizacji kodu w projekcie AI
Na podstawie analizy przedstawionego kodu, przygotowałem zestaw poprawek skupiających się na optymalizacji procesu trenowania modeli, poprawie wydajności, usunięciu błędów oraz zwiększeniu stabilności aplikacji.

1. Błędy w module ai/classifier.py
1.1. Problem z konwersją kluczy w klasie ImageClassifier
W metodzie _load_weights występuje problem z konwersją kluczy w słowniku class_names. Metoda niepoprawnie obsługuje klucze numeryczne, a później jest problem podczas klasyfikacji obrazów.

python
# Zmiana w pliku ai/classifier.py, w metodzie predict
# Linie dotyczące debugowania i wyszukiwania klucza

# Przed poprawką:
key_to_find = str(predicted_class)  # Klucz, którego szukamy
print(f"DEBUG: Szukam klucza: '{key_to_find}' (typ: {type(key_to_find)})")

# Po poprawce:
key_to_find = str(predicted_class)  # Konwersja na string
if self.class_names and key_to_find not in self.class_names:
    # Spróbuj znaleźć klucz jako liczbę
    try:
        int_key = int(key_to_find)
        if str(int_key) in self.class_names:
            key_to_find = str(int_key)
        elif int_key in self.class_names:
            key_to_find = int_key
    except ValueError:
        pass
1.2. Dodanie obsługi błędów przy ładowaniu modelu
python
# Zmiana w pliku ai/classifier.py, w metodzie __init__

# Dodaj obsługę błędów przy ładowaniu modelu
if weights_path and os.path.exists(weights_path):
    try:
        self._load_weights(weights_path)
    except Exception as e:
        print(f"BŁĄD podczas ładowania wag modelu: {str(e)}")
        traceback.print_exc()
        raise ValueError(f"Nie udało się załadować modelu: {str(e)}")
2. Optymalizacja procesu trenowania
2.1. Optymalizacja zarządzania pamięcią w ai/optimized_training.py
Aktualne zarządzanie pamięcią przy treningu jest nieoptymalne - należy dodać regularne czyszczenie pamięci GPU podczas treningu.

python
# Zmiana w pliku ai/optimized_training.py, w metodzie train_model_optimized
# Dodaj po zakończeniu każdej epoki

# Po aktualizacji historii i wywołaniu progress_callback
if torch.cuda.is_available():
    # Wyczyść nieużywane tensory z pamięci CUDA
    torch.cuda.empty_cache()
2.2. Optymalizacja rozkładu batch_size
Aktualna metoda auto_select_batch_size może zostać ulepszona, aby lepiej dostosować rozmiar batch'a do dostępnej pamięci GPU.

python
# Zmiana w pliku ai/classifier.py, nowa implementacja metody auto_select_batch_size

def auto_select_batch_size(self):
    """
    Automatycznie wybiera optymalny rozmiar batcha na podstawie dostępnego sprzętu.
    
    Returns:
        int: Optymalny rozmiar batcha
    """
    if not torch.cuda.is_available():
        # Na CPU używamy mniejszych batchów
        return 8

    # Na GPU dobieramy rozmiar batcha na podstawie ilości pamięci
    try:
        memory_info = torch.cuda.get_device_properties(0).total_memory
        free_memory = memory_info - torch.cuda.memory_allocated(0)
        memory_gb = free_memory / (1024**3)  # Konwersja na GB
        
        # Bardziej precyzyjne mapowanie pamięci GPU na rozmiar batcha
        if memory_gb > 16:
            return 128
        elif memory_gb > 12:
            return 96
        elif memory_gb > 8:
            return 64
        elif memory_gb > 6:
            return 48
        elif memory_gb > 4:
            return 32
        elif memory_gb > 2:
            return 16
        else:
            return 8
    except Exception:
        # W przypadku błędu, użyj bezpiecznej wartości
        return 16
2.3. Poprawa funkcji mixup_data w ai/optimized_training.py
Funkcja mixup_data ma problemy ze stabilnością, szczególnie gdy używana jest na GPU.

python
# Zmiana w pliku ai/optimized_training.py, poprawiona funkcja mixup_data

def mixup_data(x, y, alpha=0.2, device=None):
    """
    Wykonuje mixup na danych wejściowych i etykietach.
    Bezpieczna implementacja unikająca problemów z urządzeniami.
    """
    # Jeśli nie podano urządzenia, użyj urządzenia x
    if device is None:
        device = x.device
    
    # Bezpieczne sprawdzenie CUDA
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    
    # Parametr mixup
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    
    # Bezpieczne tworzenie permutacji
    try:
        # Tworzenie na CPU a potem przeniesienie
        index = torch.randperm(batch_size, dtype=torch.long, device="cpu")
        index = index.to(device)
        
        # Mixup
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    except Exception as e:
        # Awaryjne podejście - zwróć oryginalne dane
        return x, y, y, 1.0
3. Poprawa obsługi błędów i stabilności
3.1. Poprawka w obsłudze wyjątków w app/core/workers/batch_training_thread.py
python
# Zmiana w pliku app/core/workers/batch_training_thread.py, metoda run

# Poprawa obsługi błędów przy zamykaniu wątku
try:
    # Bieżący kod funkcji
    pass
except Exception as e:
    self.logger.error(f"BatchTrainingThread.run: BŁĄD: {str(e)}\n{traceback.format_exc()}")
    self.error.emit("Główny wątek", str(e))
finally:
    # Zabezpieczenie przed nieoczekiwanymi błędami przy czyszczeniu zasobów
    try:
        # Wyczyść zasoby po zakończeniu wszystkich zadań
        self.logger.info("Rozpoczynam końcowe czyszczenie zasobów...")
        self._cleanup_resources()
        self.logger.info("Zakończono końcowe czyszczenie zasobów.")
    except Exception as cleanup_error:
        self.logger.error(f"Błąd podczas czyszczenia zasobów: {str(cleanup_error)}")
3.2. Zabezpieczenie przed wyciekami pamięci w app/gui/main_window.py
python
# Zmiana w pliku app/gui/main_window.py, metoda closeEvent

def closeEvent(self, event):
    """Obsługuje zdarzenie zamknięcia okna."""
    try:
        # Czyszczenie pamięci GPU przy zamykaniu aplikacji
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Użyj loggera z self, jeśli jest dostępny
        if hasattr(self, "logger"):
            self.logger.info("Zamykanie aplikacji...")
        else:
            print("Zamykanie aplikacji...")  # Fallback
            
        # Usunięcie referencji do klasyfikatora
        if hasattr(self, "classifier") and self.classifier is not None:
            del self.classifier
            
        # Zatrzymaj aktualnie działający wątek treningowy, jeśli istnieje
        if hasattr(self, "training_thread") and self.training_thread is not None:
            try:
                if self.training_thread.isRunning():
                    self.training_thread.stop()
                    self.training_thread.wait(1000)  # Poczekaj maksymalnie 1 sekundę
            except Exception as thread_error:
                print(f"Błąd podczas zatrzymywania wątku treningowego: {thread_error}")
                
        # Zaakceptuj zdarzenie zamknięcia (pozwól zamknąć okno)
        event.accept()
        
    except Exception as e:
        # Logowanie błędu, jeśli coś pójdzie nie tak w samym closeEvent
        print(f"KRYTYCZNY BŁĄD podczas closeEvent: {str(e)}\n{traceback.format_exc()}")
        # Mimo błędu, próbujemy zamknąć aplikację
        event.accept()
4. Optymalizacja wydajności
4.1. Optymalizacja wczytywania danych w ai/preprocessing.py
python
# Zmiana w pliku ai/preprocessing.py, dodanie nowej funkcji

def get_preprocessing_workers():
    """
    Zwraca optymalną liczbę workerów do przetwarzania danych.
    
    Returns:
        int: Optymalna liczba workerów
    """
    import os
    cpu_count = os.cpu_count() or 2
    
    # Zapewniamy minimum 2 wątki dla maszyn jednordzeniowych
    if cpu_count < 2:
        return 1
    
    # Używamy 75% dostępnych rdzeni dla przetwarzania danych
    optimal_workers = max(1, int(cpu_count * 0.75))
    
    # Ograniczamy do maksymalnie 8 workerów, aby uniknąć problemów z pamięcią
    return min(optimal_workers, 8)
4.2. Optymalizacja przetwarzania wsadowego w ai/classifier.py
python
# Zmiana w pliku ai/classifier.py, poprawiona metoda batch_predict

def batch_predict(self, image_paths, batch_size=16):
    """Przewidywanie kategorii dla wielu obrazów w trybie wsadowym."""
    # Dynamiczne określanie optymalnego rozmiaru batcha, jeśli nie określono
    if batch_size is None:
        batch_size = self.auto_select_batch_size()
        
    # Oblicz optymalną liczbę wątków
    optimal_workers = min(4, len(image_paths))  # Maksymalnie 4 wątki
    
    results = []
    total_batches = (len(image_paths) + batch_size - 1) // batch_size
    
    # Przetwarzanie wsadowe z użyciem progress_callback jeśli dostępny
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_images = []
        
        # Wczytaj i przetwórz obrazy równolegle
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
            futures = [
                executor.submit(self.transform, Image.open(path).convert("RGB"))
                for path in batch_paths
            ]
            batch_images = [future.result().unsqueeze(0) for future in futures]
            
        # Połącz w jeden tensor i przenieś na urządzenie
        try:
            batch_tensor = torch.cat(batch_images, dim=0)
        except RuntimeError as e:
            # Gdy obrazy mają różne rozmiary, przetwórz je pojedynczo
            results.extend([self.predict(path) for path in batch_paths])
            continue
            
        # Ustaw odpowiedni typ danych i urządzenie
        if self.precision == "half" and torch.cuda.is_available():
            batch_tensor = batch_tensor.to(device=self.device, dtype=torch.float16)
        else:
            batch_tensor = batch_tensor.to(device=self.device, dtype=torch.float32)
            
        # Wykonaj predykcję
        with torch.no_grad():
            with torch.amp.autocast(
                device_type="cuda",
                enabled=(self.precision == "half" and torch.cuda.is_available()),
            ):
                outputs = self.model(batch_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted_idx = torch.max(outputs, 1)
                
        # Przetwórz wyniki
        batch_results = []
        for j, path in enumerate(batch_paths):
            predicted_class = predicted_idx[j].item()
            confidence = probabilities[j][predicted_class].item()
            
            # Znajdź nazwę klasy
            key_to_find = str(predicted_class)
            if key_to_find not in self.class_names and predicted_class in self.class_names:
                key_to_find = predicted_class
                
            class_name = self.class_names.get(key_to_find)
            if class_name is None:
                class_name = f"Kategoria_{predicted_class}"
                
            batch_results.append({
                "class_id": predicted_class,
                "class_name": class_name,
                "confidence": confidence,
            })
            
        results.extend(batch_results)
        
        # Wyczyść pamięć GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    return results
5. Poprawki błędów w importach i definicjach
5.1. Brakujący import w app/gui/main_window.py
python
# Dodaj import w pliku app/gui/main_window.py

import glob  # Dodanie brakującego importu dla funkcji glob.glob
5.2. Poprawa importu w app/gui/tabs/report_generator.py
python
# Dodaj import w pliku app/gui/tabs/report_generator.py

import glob  # Dodanie brakującego importu dla funkcji glob.glob
6. Poprawa wydajności treningu
6.1. Dodanie automatycznego wyboru optymalizatora w ai/optimized_training.py
python
# Zmiana w pliku ai/optimized_training.py, metoda train_model_optimized

# Dodaj funkcję wyboru optymalizatora na podstawie rozmiaru modelu
def _select_optimizer(model_parameters, learning_rate, weight_decay, optimizer_type):
    """Wybiera odpowiedni optymalizator na podstawie wielkości modelu i dostępnego sprzętu."""
    param_count = sum(p.numel() for p in model_parameters if p.requires_grad)
    
    # Dla bardzo dużych modeli lepiej sprawdza się AdamW z większym weight_decay
    if param_count > 50_000_000:  # Ponad 50M parametrów
        if optimizer_type.lower() == "adamw":
            return optim.AdamW(
                model_parameters, 
                lr=learning_rate,
                weight_decay=max(weight_decay, 0.05)
            )
        elif optimizer_type.lower() == "sgd":
            return optim.SGD(
                model_parameters,
                lr=learning_rate,
                momentum=0.9,
                weight_decay=max(weight_decay, 0.01),
                nesterov=True
            )
            
    # Dla średnich modeli
    if param_count > 10_000_000:  # Ponad 10M parametrów
        if optimizer_type.lower() == "adamw":
            return optim.AdamW(
                model_parameters, 
                lr=learning_rate,
                weight_decay=weight_decay
            )
            
    # Dla mniejszych modeli
    if optimizer_type.lower() == "adamw":
        return optim.AdamW(
            model_parameters,
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == "sgd":
        return optim.SGD(
            model_parameters,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )
    else:  # domyślnie Adam
        return optim.Adam(
            model_parameters,
            lr=learning_rate,
            weight_decay=weight_decay
        )

# Zastąp istniejący kod wyboru optymalizatora tym wywołaniem:
optimizer = _select_optimizer(
    [p for p in model.parameters() if p.requires_grad],
    learning_rate,
    weight_decay,
    optimizer_type
)
7. Poprawa komunikatów debugowania
7.1. Poprawa logowania w ai/optimized_training.py
python
# Zmiana w pliku ai/optimized_training.py

# Przed debugowaniem batch'a:
# print(f"DEBUG Batch {batch_idx + 1}: Dane przeniesione. Typ inputs: {type(inputs)}, Typ targets: {type(targets)}")

# Po poprawce:
# print(f"DEBUG Batch {batch_idx + 1}: Dane przeniesione. Typ inputs: {type(inputs)}, kształt: {inputs.shape}, Typ targets: {type(targets)}, kształt: {targets.shape}")
8. Poprawa stabilności treningu
8.1. Dodanie obsługi błędów w przypadku niepoprawnych obrazów
python
# Zmiana w pliku ai/optimized_training.py, dodać obsługę błędów w pętli treningowej

# W miejscu pętli po batchach:
try:
    # Kod przetwarzania batcha
    pass
except RuntimeError as e:
    if "CUDA out of memory" in str(e):
        # Obsługa braku pamięci CUDA - zredukuj batch_size
        print(f"BŁĄD CUDA: Brak pamięci. Zredukuj rozmiar wsadu lub użyj CPU.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        # Inne błędy runtime
        print(f"Błąd podczas przetwarzania batcha: {str(e)}")
        
    # Kontynuuj z następnym batchem
    continue
except Exception as e_batch:
    # Loguj inne błędy
    print(f"!!!!!!!!!! Nieoczekiwany błąd w pętli po batchach !!!!!!!!!!")
    print(f"Błąd: {str(e_batch)}")
    print(traceback.format_exc())
    
    # Kontynuuj z następnym batchem
    continue
Podsumowanie
Powyższe poprawki skupiają się na:

Naprawie błędów w klasyfikatorze obrazów, szczególnie związanych z konwersją kluczy w class_names
Optymalizacji zarządzania pamięcią GPU podczas treningu
Polepszeniu doboru rozmiaru batch'a w zależności od dostępnego sprzętu
Zwiększeniu stabilności funkcji mixup_data używanej przy augmentacji danych
Poprawie obsługi wyjątków w wątkach treningowych
Zabezpieczeniu przed wyciekami pamięci przy zamykaniu aplikacji
Optymalizacji przetwarzania wsadowego obrazów
Dodaniu brakujących importów i poprawie istniejących
Optymalizacji wyboru optymalizatora w zależności od wielkości modelu
Poprawie komunikatów debugowania
Zwiększeniu stabilności podczas treningu
Te zmiany powinny znacznie poprawić wydajność i stabilność aplikacji, szczególnie w kontekście trenowania modeli AI i przetwarzania dużych zestawów obrazów.




czy sam proces treningu jest prawidłowy?

Edit
Po dokładnej analizie kodu, ocenię sam proces treningu zawarty w module ai/optimized_training.py i powiązanych plikach.

Analiza procesu treningu
Proces treningu zaimplementowany w funkcji train_model_optimized jest ogólnie poprawny, ale zawiera kilka problemów i obszarów do poprawy:

Poprawne elementy implementacji:
Architektura procesu treningu jest prawidłowa - obejmuje:
Inicjalizację modelu
Definicję dataloader'ów dla danych treningowych i walidacyjnych
Odpowiednie pętle treningowe
Walidację modelu
Zapisywanie historii treningu
Obsługę wczesnego zatrzymania (early stopping)
Właściwa implementacja optymalizacji sprzętowych:
Wykorzystanie GPU jeśli jest dostępne
Wsparcie dla mixed precision (autocast)
Dynamiczne dostosowanie batch size
Poprawna implementacja monitorowania treningu:
Obsługa funkcji callback do raportowania postępu
Zbieranie historii treningu
Problemy w procesie treningu:
Zarządzanie pamięcią:
Brak systematycznego czyszczenia pamięci GPU po każdej epoce
Nieprawidłowa obsługa wycieków pamięci podczas błędów
Stabilność podczas treningu:
Niedostateczna obsługa błędów CUDA (out-of-memory)
Brak mechanizmu automatycznego zmniejszania batch size przy problemach z pamięcią
Obsługa optymalizatora:
Wybór optymalizatora nie jest dostosowany do architektury modelu
Parametry optymalizatora nie są optymalizowane dla różnych wielkości modeli
Mechanizm augmentacji danych:
Niestabilna implementacja funkcji mixup_data
Brak obsługi błędów przy augmentacji
Implementacja schedulera:
Niepoprawne użycie schedulera typu plateau w połączeniu z wczesnym zatrzymaniem
Brak adaptacyjnej zmiany learning rate podczas treningu
Rekomendowane poprawki w procesie treningu
1. Poprawa obsługi pamięci GPU
python
# ai/optimized_training.py - w funkcji train_model_optimized
# Dodaj systematyczne czyszczenie pamięci po każdej epoce

# Po zakończeniu pętli po batchach, a przed walidacją:
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Także po zakończeniu walidacji:
if val_loader and torch.cuda.is_available():
    torch.cuda.empty_cache()
2. Poprawa stabilności treningu
python
# ai/optimized_training.py - w funkcji train_model_optimized
# W głównej pętli treningowej, dodaj obsługę błędów pamięci

current_batch_size = batch_size
reduced_batch_size = False

try:
    # Pętla po epokach
    for epoch in range(num_epochs):
        # Obsługa przerwania treningu
        if should_stop_callback and should_stop_callback():
            print(f"\n!!! Trening przerwany na epoce {epoch+1} przez użytkownika !!!")
            break
            
        # Reset batch size na początek każdej epoki jeśli była zmniejszona
        if reduced_batch_size and torch.cuda.is_available():
            # Sprawdź dostępną pamięć
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            if free_memory > 2 * 1024 * 1024 * 1024:  # 2GB wolnej pamięci
                current_batch_size = min(current_batch_size * 2, batch_size)
                print(f"Zwiększam batch size do {current_batch_size}")
        
        # Pętla po batchach
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            try:
                # Kod treningu batcha
                # ...
            except RuntimeError as e:
                if "CUDA out of memory" in str(e) and torch.cuda.is_available():
                    # Zmniejsz batch size i wyczyść pamięć
                    torch.cuda.empty_cache()
                    current_batch_size = max(1, current_batch_size // 2)
                    reduced_batch_size = True
                    print(f"CUDA out of memory! Zmniejszam batch size do {current_batch_size}")
                    
                    # Podziel bieżący batch na mniejsze części
                    half_size = inputs.size(0) // 2
                    if half_size >= 1:
                        # Przetwórz pierwszą połowę
                        inputs_half = inputs[:half_size]
                        targets_half = targets[:half_size]
                        # Kontynuuj przetwarzanie z mniejszym batchem
                        # ...
                    continue
                else:
                    # Inne błędy
                    print(f"Błąd podczas przetwarzania batcha: {str(e)}")
                    continue
3. Poprawa wyboru i konfiguracji optymalizatora
python
# ai/optimized_training.py - w funkcji train_model_optimized
# Lepsze dopasowanie optymalizatora do modelu

def configure_optimizer(model, optimizer_type, learning_rate, weight_decay):
    """Konfiguruje optymalizator z parametrami dostosowanymi do modelu."""
    # Policz liczbę parametrów modelu
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Dopasuj learning rate do wielkości modelu
    adjusted_lr = learning_rate
    if param_count > 50_000_000:  # Bardzo duży model (>50M parametrów)
        adjusted_lr = learning_rate * 0.5
    elif param_count < 5_000_000:  # Mały model (<5M parametrów)
        adjusted_lr = learning_rate * 2.0
    
    # Dopasuj weight_decay
    adjusted_wd = weight_decay
    if param_count > 20_000_000:  # Większy model potrzebuje silniejszej regularyzacji
        adjusted_wd = max(weight_decay, 0.03)
    
    # Wybierz i skonfiguruj optymalizator
    if optimizer_type.lower() == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=adjusted_lr,
            weight_decay=adjusted_wd,
            eps=1e-8
        )
    elif optimizer_type.lower() == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=adjusted_lr, 
            momentum=0.9,
            weight_decay=adjusted_wd,
            nesterov=True
        )
    else:  # Adam jako domyślny
        return optim.Adam(
            model.parameters(),
            lr=adjusted_lr,
            weight_decay=adjusted_wd,
            eps=1e-8
        )

# Użycie:
optimizer = configure_optimizer(model, optimizer_type, learning_rate, weight_decay)
4. Poprawa mechanizmu augmentacji
python
# ai/optimized_training.py - zastąp funkcję mixup_data

def mixup_data(x, y, alpha=0.2, device=None):
    """
    Wykonuje mixup na danych wejściowych i etykietach.
    Bezpieczna implementacja unikająca problemów z urządzeniami.
    """
    try:
        # Jeśli nie podano urządzenia, użyj urządzenia x
        if device is None:
            device = x.device
        
        # Bezpieczne sprawdzenie CUDA
        if device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")
        
        # Parametr mixup
        if alpha > 0:
            lam = float(np.random.beta(alpha, alpha))
        else:
            lam = 1.0
        
        batch_size = x.size()[0]
        
        # Bezpieczne tworzenie permutacji (zawsze na CPU)
        index = torch.randperm(batch_size, device='cpu')
        # Przenieś indeksy na odpowiednie urządzenie
        index = index.to(device)
        
        # Wykonaj mixup
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    except Exception as e:
        print(f"Błąd podczas mixup: {str(e)}")
        # Zwróć oryginalne dane w przypadku błędu
        return x, y, y, 1.0
5. Poprawa implementacji schedulera
python
# ai/optimized_training.py - poprawa konfiguracji schedulera

# Dodaj funkcję tworzenia optymalnego schedulera
def configure_scheduler(optimizer, scheduler_type, epochs, patience=3):
    """Konfiguruje scheduler learning rate odpowiedni do typu optymalizatora i długości treningu."""
    if scheduler_type == "plateau":
        # ReduceLROnPlateau z lepszymi parametrami
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode="min", 
            factor=0.5,     # Zmniejsz learning rate o połowę
            patience=patience,
            verbose=True,
            min_lr=1e-7
        )
    elif scheduler_type == "cosine":
        # CosineAnnealingLR z odpowiednim T_max
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=1e-7,
            verbose=True
        )
    elif scheduler_type == "onecycle":
        # OneCycleLR - często najlepszy wybór dla krótszych treningów
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]['lr'] * 10,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=10000.0,
            verbose=True
        )
    elif scheduler_type == "step":
        # StepLR z odpowiednią częstotliwością kroków
        step_size = max(5, epochs // 4)  # Co najmniej 5 epok lub 1/4 całkowitej liczby
        return optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=step_size, 
            gamma=0.5,
            verbose=True
        )
    else:
        # Brak schedulera
        return None

# Użycie:
scheduler = configure_scheduler(optimizer, lr_scheduler_type, num_epochs)
6. Lepsze zapisywanie historii treningu
python
# ai/optimized_training.py - ulepszone zapisywanie historii

# W funkcji train_model_optimized

# Inicjalizacja historii treningu z dodatkowymi metrykami
history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
    "epoch_times": [],
    "learning_rates": [],
    "batch_sizes": [],  # Śledzenie rozmiaru wsadu
    "memory_usage": [],  # Śledzenie użycia pamięci GPU
    "best_val_loss": float("inf"),
    "best_epoch": 0,
    "hardware_profile": hardware_profile,
}

# W pętli treningowej, po zakończeniu epoki:
current_lr = optimizer.param_groups[0]["lr"]
history["learning_rates"].append(current_lr)
history["batch_sizes"].append(current_batch_size)

if torch.cuda.is_available():
    memory_used = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)  # GB
    history["memory_usage"].append(float(memory_used))
    # Reset licznika maksymalnej pamięci
    torch.cuda.reset_peak_memory_stats()
else:
    history["memory_usage"].append(0.0)
Podsumowanie stanu procesu treningu
Ogólnie rzecz biorąc, proces treningu jest poprawny pod względem podstawowych koncepcji i struktury, ale zawiera wiele elementów, które można zoptymalizować dla lepszej wydajności, stabilności i rezultatów. Główne obszary wymagające poprawy to:

Lepsze zarządzanie pamięcią GPU
Zwiększenie stabilności przez adaptacyjny batch size
Optymalizacja doborów parametrów uczenia (optymalizator, scheduler)
Poprawki w implementacji augmentacji danych
Dokładniejsze monitorowanie i zapisywanie postępu treningu
Wprowadzenie tych poprawek powinno znacząco ulepszyć proces treningu, zwłaszcza na różnorodnym sprzęcie i dla różnych wielkości modeli.







