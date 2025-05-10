Zmiany w pliku ai/fine_tuning.py
Problem 1: Inicjalizacja zmiennych przed używaniem ich w wywołaniu progress_callback
python# Zmiana 1: Inicjalizacja zmiennych val_loss i val_acc przed użyciem w callback
# Dodać na początku pętli po batchach, przed pierwszym wywołaniem progress_callback

# Inicjalizacja zmiennych, które będą używane w progress_callback
val_loss = 0.0
val_acc = 0.0
train_acc = 0.0  # Ta jest już poprawiona w drugim logu

# Aktualizuj progress bar
if progress_callback:
    try:
        progress_callback(
            epoch + 1,
            num_epochs,
            train_loss,
            train_acc,
            val_loss,  # Teraz te zmienne są zdefiniowane
            val_acc,
            0,  # top3
            0,  # top5
            0,  # precision
            0,  # recall
            0,  # f1
            0,  # auc
        )
    except Exception as e:
        print(f"Błąd podczas wywołania progress_callback: {str(e)}")
Problem 2: Błąd indeksu w historii walidacji podczas podsumowania fine-tuningu
python# Zmiana 2: Naprawienie błędu "list index out of range" w podsumowaniu fine-tuningu
# W funkcji fine_tune_model, w sekcji podsumowania fine-tuningu

# Obecny problematyczny kod:
print(f"Najlepsza epoka: {best_epoch + 1}")
print(f"Najlepsza strata walidacyjna: {history['best_val_loss']:.4f}")
print(f"Dokładność walidacji: {history['val_acc'][best_epoch]:.2%}")  # Ten wiersz powoduje błąd

# Zmodyfikowany kod z bezpiecznym dostępem do indeksu:
print(f"Najlepsza epoka: {best_epoch + 1}")
print(f"Najlepsza strata walidacyjna: {history['best_val_loss']:.4f}")

# Sprawdź czy indeks best_epoch istnieje w history['val_acc'] zanim go użyjesz
if 'val_acc' in history and best_epoch < len(history['val_acc']):
    print(f"Dokładność walidacji: {history['val_acc'][best_epoch]:.2%}")
else:
    print("Nie znaleziono danych o dokładności walidacji dla najlepszej epoki.")
Problem 3: Inicjalizacja val_metrics przed jego użyciem (główny błąd)
python# Zmiana 3: Inicjalizacja val_metrics przed jego użyciem
# Po pętli walidacji, przed użyciem val_metrics

# Walidacja
if val_loader:
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()
            
            # Dodaj dla obliczenia dodatkowych metryk
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    val_loss = val_loss / len(val_loader)
    val_acc = 100.0 * val_correct / val_total

    # Inicjalizacja val_metrics z odpowiednimi wartościami
    val_metrics = {
        "loss": val_loss,
        "acc": val_acc,
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "auc": 0.0,
        "top3": 0.0,
        "top5": 0.0
    }
    
    # Próba obliczenia dodatkowych metryk, jeśli możliwe
    try:
        if len(all_targets) > 0 and len(all_preds) > 0:
            y_true = np.array(all_targets)
            y_pred = np.array(all_preds)
            val_metrics["f1"] = f1_score(y_true, y_pred, average='macro', zero_division=0)
            val_metrics["precision"] = precision_score(y_true, y_pred, average='macro', zero_division=0)
            val_metrics["recall"] = recall_score(y_true, y_pred, average='macro', zero_division=0)
            
            # Jeśli mamy prawdopodobieństwa, możemy obliczyć AUC i top-k
            if len(all_probs) > 0:
                y_prob = np.array(all_probs)
                if y_prob.shape[1] > 2:  # Wieloklasowy problem
                    val_metrics["auc"] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
                elif y_prob.shape[1] == 2:  # Problem binarny
                    val_metrics["auc"] = roc_auc_score(y_true, y_prob[:, 1])
                
                # Top-k metryki
                if y_prob.shape[1] >= 3:
                    val_metrics["top3"] = top_k_accuracy_score(y_true, y_prob, k=3)
                if y_prob.shape[1] >= 5:
                    val_metrics["top5"] = top_k_accuracy_score(y_true, y_prob, k=5)
    except Exception as e:
        print(f"Ostrzeżenie: Nie udało się obliczyć niektórych metryk: {e}")
        # Zachowamy domyślne wartości 0.0 dla metryk, których nie udało się obliczyć
else:
    # Utwórz puste val_metrics, gdy walidacja jest wyłączona
    val_metrics = {
        "loss": 0.0,
        "acc": 0.0,
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "auc": 0.0,
        "top3": 0.0,
        "top5": 0.0
    }
Kompletna poprawka - fragment pliku fine_tuning.py
Poniżej przedstawiam pełny fragment kodu z wprowadzonymi poprawkami, który należy dodać w odpowiednich miejscach (zbliżony do linii 819):
python# Walidacja
if val_loader:
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()
            
            # Dodaj dla obliczenia dodatkowych metryk
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    val_loss = val_loss / len(val_loader)
    val_acc = 100.0 * val_correct / val_total

    # Inicjalizacja val_metrics z odpowiednimi wartościami
    val_metrics = {
        "loss": val_loss,
        "acc": val_acc,
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "auc": 0.0,
        "top3": 0.0,
        "top5": 0.0
    }
    
    # Próba obliczenia dodatkowych metryk, jeśli możliwe
    try:
        if len(all_targets) > 0 and len(all_preds) > 0:
            y_true = np.array(all_targets)
            y_pred = np.array(all_preds)
            val_metrics["f1"] = f1_score(y_true, y_pred, average='macro', zero_division=0)
            val_metrics["precision"] = precision_score(y_true, y_pred, average='macro', zero_division=0)
            val_metrics["recall"] = recall_score(y_true, y_pred, average='macro', zero_division=0)
            
            # Jeśli mamy prawdopodobieństwa, możemy obliczyć AUC i top-k
            if len(all_probs) > 0:
                y_prob = np.array(all_probs)
                try:
                    if y_prob.shape[1] > 2:  # Wieloklasowy problem
                        val_metrics["auc"] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
                    elif y_prob.shape[1] == 2:  # Problem binarny
                        val_metrics["auc"] = roc_auc_score(y_true, y_prob[:, 1])
                except Exception as e:
                    print(f"Ostrzeżenie: Nie udało się obliczyć AUC: {e}")
                
                # Top-k metryki
                try:
                    if y_prob.shape[1] >= 3:
                        val_metrics["top3"] = top_k_accuracy_score(y_true, y_prob, k=3)
                    if y_prob.shape[1] >= 5:
                        val_metrics["top5"] = top_k_accuracy_score(y_true, y_prob, k=5)
                except Exception as e:
                    print(f"Ostrzeżenie: Nie udało się obliczyć top-k metryk: {e}")
    except Exception as e:
        print(f"Ostrzeżenie: Nie udało się obliczyć niektórych metryk: {e}")
        # Zachowamy domyślne wartości 0.0 dla metryk, których nie udało się obliczyć
else:
    # Utwórz puste val_metrics, gdy walidacja jest wyłączona
    val_metrics = {
        "loss": 0.0,
        "acc": 0.0,
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "auc": 0.0,
        "top3": 0.0,
        "top5": 0.0
    }

# Również przy wyświetlaniu podsumowania na końcu funkcji
if val_loader:
    best_epoch = history["best_epoch"]
    print(f"Najlepsza epoka: {best_epoch + 1}")
    print(f"Najlepsza strata walidacyjna: {history['best_val_loss']:.4f}")
    
    # Bezpieczny dostęp do listy val_acc
    if 'val_acc' in history and best_epoch < len(history['val_acc']):
        print(f"Dokładność walidacji: {history['val_acc'][best_epoch]:.2%}")
    else:
        print("Nie znaleziono danych o dokładności walidacji dla najlepszej epoki.")
Powyższe zmiany powinny rozwiązać wszystkie zgłaszane błędy w funkcji fine_tune_model, bez ryzyka wprowadzenia nowych problemów.