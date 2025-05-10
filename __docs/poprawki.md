Analiza obsługi parametrów konfiguracyjnych
Nieobsługiwane lub częściowo obsługiwane parametry
1. Parametry EWC w pliku fine_tuning.py
python# W pliku fine_tuning.py, w funkcji fine_tune_model

# Parametr Lambda z konfiguracji EWC jest pobierany, ale wartość domyślna jest zbyt niska
ewc_lambda = ewc_config.get("lambda", 100.0)  # Zbyt niska wartość domyślna

# Parametr fisher_sample_size jest zadeklarowany, ale nigdzie nie jest używany 
# w funkcji compute_fisher_information
fisher_sample_size = ewc_config.get("fisher_sample_size", 200)
# ...
fisher_diagonal = compute_fisher_information(
    original_model_for_forgetting,
    data_loader_for_original_classes,
    device=device,
    num_samples=fisher_sample_size,  # Ten parametr jest przekazywany, ale nie jest używany w funkcji
)
2. Brakujące parametry w oknie dialogowym
python# W pliku fine_tuning_task_config_dialog.py

# Brakuje kontrolki dla parametru "adaptive_lambda" w konfiguracji EWC
# która była wspomniana w proponowanych zmianach
3. Parametry z pliku b2_classes_fine_tuning_5.json nie są w pełni wykorzystywane
json// W pliku b2_classes_fine_tuning_5.json

// Te parametry są ustawione, ale wydają się być ignorowane:
"catastrophic_forgetting_prevention": {
  "enable": true,
  "preserve_original_classes": true,
  "rehearsal": {
    "use": true,
    "samples_per_class": 25,
    "synthetic_samples": true
  },
  "knowledge_distillation": {
    "use": true,
    "temperature": 2.0,
    "alpha": 0.4
  },
  "ewc_regularization": {
    "use": true,
    "lambda": 120.0,  // Ta wartość jest zbyt niska
    "fisher_sample_size": 200
  },
  "layer_freezing": {
    "strategy": "gradual",
    "freeze_ratio": 0.7
  }
}
Proponowane zmiany w obsłudze parametrów
1. Poprawki w pliku catastrophic_forgetting.py
python# Funkcja compute_fisher_information powinna uwzględniać parametr num_samples

def compute_fisher_information(model, data_loader, device=None, num_samples=200):
    """
    Oblicza diagonalę macierzy Fishera dla wszystkich parametrów modelu.
    
    Args:
        model: Model do obliczenia macierzy Fishera
        data_loader: DataLoader z danymi
        device: Urządzenie (CPU/GPU)
        num_samples: Liczba próbek do obliczenia macierzy Fishera
    
    Returns:
        dict: Diagonala macierzy Fishera dla każdego parametru
    """
    # Ustawienie urządzenia
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Przełączenie modelu w tryb ewaluacji
    model.eval()
    
    # Przygotowanie słownika na diagonalę macierzy Fishera
    fisher_diagonal = {}
    for name, param in model.named_parameters():
        fisher_diagonal[name] = torch.zeros_like(param)
    
    # Licznik próbek
    sample_count = 0
    
    # Iteracja po batchu danych
    for inputs, targets in data_loader:
        # Sprawdź czy osiągnięto limit próbek
        if sample_count >= num_samples:
            break
            
        # Liczba próbek w bieżącym batchu
        batch_size = inputs.size(0)
        
        # Nie przekraczamy limitu próbek
        samples_to_process = min(batch_size, num_samples - sample_count)
        if samples_to_process <= 0:
            break
            
        # Wycinamy odpowiednią liczbę próbek z batcha
        inputs = inputs[:samples_to_process]
        targets = targets[:samples_to_process]
        
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        log_probs = torch.log_softmax(model(inputs), dim=1)
        
        # Iteracja po próbkach
        for i in range(samples_to_process):
            # Filtruj nieprawidłowe etykiety (np. etykiety -1 używane do oznaczenia ignorowanych klas)
            if targets[i] < 0:
                continue
                
            # Obliczenie pochodnej logarytmu prawdopodobieństwa dla prawdziwej klasy
            log_prob = log_probs[i, targets[i]]
            
            # Zerowanie gradientów
            model.zero_grad()
            
            # Backward pass
            log_prob.backward(retain_graph=(i < samples_to_process-1))
            
            # Aktualizacja diagonali macierzy Fishera
            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher_diagonal[name] += param.grad.data ** 2 / samples_to_process
        
        # Aktualizacja licznika próbek
        sample_count += samples_to_process
    
    # Normalizacja przez liczbę próbek
    if sample_count > 0:
        for name in fisher_diagonal:
            fisher_diagonal[name] /= sample_count
    
    return fisher_diagonal
2. Zmiany w interfejsie użytkownika w fine_tuning_task_config_dialog.py
python# Dodanie brakujących kontrolek w klasie FineTuningTaskConfigDialog

def _init_controls(self):
    # Istniejący kod...
    
    # Zmiana zakresu i wartości domyślnej dla parametru Lambda w EWC
    self.ewc_lambda_spin = QtWidgets.QDoubleSpinBox()
    self.ewc_lambda_spin.setRange(100.0, 10000.0)
    self.ewc_lambda_spin.setValue(5000.0)
    self.ewc_lambda_spin.setDecimals(1)
    
    # Dodanie nowej kontrolki dla adaptacyjnej Lambda
    self.adaptive_ewc_lambda_check = QtWidgets.QCheckBox()
    self.adaptive_ewc_lambda_check.setChecked(True)
    
    # Istniejący kod...
    
def _create_advanced_tab(self) -> QtWidgets.QWidget:
    # Istniejący kod...
    
    # Dodanie nowej kontrolki do układu
    forgetting_layout.addRow("Adaptacyjna Lambda EWC:", self.adaptive_ewc_lambda_check)
    
    # Istniejący kod...
    
def _on_accept(self):
    # Istniejący kod...
    
    # Dodanie nowego parametru do konfiguracji
    config["advanced"]["catastrophic_forgetting_prevention"]["ewc_regularization"]["adaptive_lambda"] = self.adaptive_ewc_lambda_check.isChecked()
    
    # Istniejący kod...
3. Zmiany w fine_tuning.py do obsługi wszystkich parametrów
python# W pliku fine_tuning.py, w funkcji fine_tune_model

if prevent_forgetting and ewc_config and ewc_config.get("use", False) and fisher_diagonal and original_params:
    # Pobranie parametrów z konfiguracji
    ewc_lambda = ewc_config.get("lambda", 5000.0)  # Zmieniona wartość domyślna
    adaptive_lambda = ewc_config.get("adaptive_lambda", True)  # Nowy parametr
    
    # Dynamiczna lambda, jeśli włączona
    if adaptive_lambda:
        # Zwiększamy Lambda w miarę postępu treningu, aby wzmocnić ochronę wiedzy
        current_ewc_lambda = ewc_lambda * (1 + epoch / (num_epochs * 0.5))
    else:
        current_ewc_lambda = ewc_lambda
    
    # Obliczanie komponentu straty EWC
    ewc_loss_val = 0
    for name, param in model.named_parameters():
        if name in fisher_diagonal and name in original_params:
            diff = (param - original_params[name]) ** 2
            ewc_loss_val += torch.sum(fisher_diagonal[name] * diff)
    
    # Dodanie do całkowitej straty
    loss += current_ewc_lambda * ewc_loss_val
    
    # Logowanie
    if batch_idx % 10 == 0:
        print(f"EWC loss component: {ewc_loss_val.item():.6f}, Lambda: {current_ewc_lambda}")
4. Dodanie do pliku b2_classes_fine_tuning_5.json parametru adaptive_lambda
json"ewc_regularization": {
  "use": true,
  "lambda": 5000.0,
  "fisher_sample_size": 200,
  "adaptive_lambda": true
}
Podsumowanie brakujących parametrów i proponowanych zmian

Parametry EWC:

Wartość lambda jest zbyt niska (120.0) - powinna być zwiększona do około 5000.0
Brakuje parametru adaptive_lambda w konfiguracji
Parametr fisher_sample_size nie jest poprawnie obsługiwany w funkcji compute_fisher_information


Parametry UI:

Brakuje kontrolki dla adaptive_lambda w interfejsie użytkownika
Zakres dla parametru ewc_lambda_spin jest zbyt niski


Parametry wizualizacji:

Brakuje obsługi błędów NaN w danych wizualizacji
Brakuje zapisywania komponentów straty EWC do historii treningu



Powyższe zmiany zapewnią, że wszystkie parametry konfiguracyjne będą prawidłowo obsługiwane w procesie fine-tuningu, co pozwoli na lepszą kontrolę nad procesem zapobiegania katastrofalnemu zapominaniu.