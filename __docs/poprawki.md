Rozwiązanie błędu w funkcji fine_tune_model
Analiza problemu
W pliku paste.txt widoczne są następujące błędy:
Błąd podczas wywołania progress_callback: cannot access local variable 'train_acc' where it is not associated with a value
...
2025-05-10 18:06:10,991 [ERROR] Błąd podczas fine-tuningu: name 'val_metrics' is not defined
Błąd występuje w pliku ai/fine_tuning.py, w funkcji fine_tune_model, linia 819, gdzie program próbuje odwołać się do niezdefiniowanej zmiennej val_metrics.
Proponowane zmiany
Zmiana w pliku ai/fine_tuning.py
python# Przed linią 819, dodajemy inicjalizację zmiennej val_metrics, która powinna zawierać metryki walidacyjne

# Obliczamy niezbędne metryki walidacyjne po pętli walidacyjnej
if val_loader:
    # ... istniejący kod walidacji ...

    # Inicjalizacja val_metrics z odpowiednimi wartościami
    val_metrics = {
        "loss": val_loss,
        "acc": val_acc,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "auc": 0.0,
        "top3": 0.0,
        "top5": 0.0
    }
    
    # Obliczanie dodatkowych metryk (jeśli potrzebne)
    if len(val_loader.dataset) > 0:
        try:
            # Kod do obliczania dodatkowych metryk, np. F1 score
            # Przykład:
            y_true = np.array(all_targets) if 'all_targets' in locals() else []
            y_pred = np.array(all_preds) if 'all_preds' in locals() else []
            
            if len(y_true) > 0 and len(y_pred) > 0:
                from sklearn.metrics import f1_score, precision_score, recall_score
                val_metrics["precision"] = precision_score(y_true, y_pred, average='macro', zero_division=0)
                val_metrics["recall"] = recall_score(y_true, y_pred, average='macro', zero_division=0)
                val_metrics["f1"] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        except Exception as e:
            print(f"Ostrzeżenie: Nie udało się obliczyć dodatkowych metryk: {str(e)}")
else:
    # Domyślne wartości, gdy brak walidacji
    val_metrics = {
        "loss": 0.0,
        "acc": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "auc": 0.0,
        "top3": 0.0,
        "top5": 0.0
    }
Ten kod powinien zostać umieszczony przed miejsce, gdzie występuje odwołanie do val_metrics w funkcji fine_tune_model.
Drugi problem: zmienna train_acc
W logach widać również liczne błędy związane ze zmienną train_acc w funkcji progress_callback. Prawdopodobnie jest to spowodowane tym, że zmienna ta jest używana przed inicjalizacją. Należy dodać również inicjalizację tej zmiennej.
python# Na początku pętli epokami, przed wywołaniem progress_callback
train_acc = 0.0
if train_total > 0:
    train_acc = 100.0 * train_correct / train_total

# Później w kodzie przed wywołaniem progress_callback
if progress_callback:
    try:
        progress_callback(
            epoch + 1,
            num_epochs,
            train_loss,
            train_acc,  # Teraz ta zmienna jest zdefiniowana
            val_loss if val_loader else 0,
            val_acc if val_loader else 0,
            # ... pozostałe parametry ...
        )
    except Exception as e:
        print(f"Błąd podczas wywołania progress_callback: {str(e)}")
Podsumowanie zmian

Dodanie inicjalizacji słownika val_metrics z metrykami walidacyjnymi przed jego użyciem
Dodanie poprawnej inicjalizacji zmiennej train_acc przed wywołaniem funkcji callback

Te zmiany powinny rozwiązać główny problem zgłaszany w logach i pozwolić na poprawne działanie funkcji fine-tuning.
Zmiany powinny zostać wprowadzone w pliku ai/fine_tuning.py w funkcji fine_tune_model.