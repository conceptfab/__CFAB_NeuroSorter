Zmiany w pliku ai/fine_tuning.py
1. Naprawa błędnych komunikatów przy obliczaniu metryk top-k
python# Linia około 1059 (w funkcji fine_tune_model, w sekcji obliczania metryk walidacyjnych)

# Obecny problematyczny kod:
if y_prob.shape[1] >= 3:
    val_metrics["top3"] = top_k_accuracy_score(y_true, y_prob, k=3)
if y_prob.shape[1] >= 5:
    val_metrics["top5"] = top_k_accuracy_score(y_true, y_prob, k=5)

# Proponowana zmiana:
if y_prob.shape[1] >= 3:
    try:
        val_metrics["top3"] = top_k_accuracy_score(
            y_true, y_prob, k=min(3, y_prob.shape[1]), labels=np.unique(y_true)
        )
    except Exception as e:
        print(f"Błąd przy obliczaniu top-3: {e}")
        val_metrics["top3"] = 0.0
        
if y_prob.shape[1] >= 5:
    try:
        val_metrics["top5"] = top_k_accuracy_score(
            y_true, y_prob, k=min(5, y_prob.shape[1]), labels=np.unique(y_true)
        )
    except Exception as e:
        print(f"Błąd przy obliczaniu top-5: {e}")
        val_metrics["top5"] = 0.0
2. Naprawa błędu w wyświetlaniu dokładności treningowej w progress_callback
python# Linia około 1186 (w funkcji fine_tune_model, wewnątrz pętli epok)

# Obecny problematyczny kod:
cumulative_train_acc_for_epoch = 0.0
if train_total > 0:
    cumulative_train_acc_for_epoch = 100.0 * train_correct / train_total

# Proponowana zmiana:
cumulative_train_acc_for_epoch = 0.0
if train_total > 0:
    cumulative_train_acc_for_epoch = 100.0 * train_correct / train_total

# Usuń lub popraw debugowy wydruk:
# DODATKOWY WYDRUK KONTROLNY:
if batch_idx % 50 == 0:  # Wyświetlaj rzadziej, co 50 batchy
    print(f"DEBUG INFO: Epoka {epoch + 1}, Batch {batch_idx + 1}/{len(train_loader)}")
    print(f"  Raw: train_correct={train_correct}, train_total={train_total}")
    print(f"  Aktualna dokładność: {cumulative_train_acc_for_epoch:.2f}%")
3. Poprawa obsługi metryki AUC dla danych jednokklasowych
python# Linia około 1046 (w funkcji fine_tune_model, w sekcji obliczania metryk walidacyjnych)

# Obecny problematyczny kod:
if y_prob.shape[1] > 2:  # Wieloklasowy problem
    val_metrics["auc"] = roc_auc_score(
        y_true,
        y_prob,
        multi_class="ovr",
        average="macro",
        labels=np.arange(y_prob.shape[1]),
    )
elif y_prob.shape[1] == 2:  # Problem binarny
    val_metrics["auc"] = roc_auc_score(y_true, y_prob[:, 1])

# Proponowana zmiana:
try:
    n_classes = len(np.unique(y_true))
    if n_classes > 1:  # Sprawdzamy liczbę unikalnych klas w danych
        if y_prob.shape[1] > 2:  # Wieloklasowy problem
            val_metrics["auc"] = roc_auc_score(
                y_true,
                y_prob,
                multi_class="ovr",
                average="macro",
                labels=np.unique(y_true),  # Użyj tylko unikalnych klas występujących w batchu
            )
        elif y_prob.shape[1] == 2:  # Problem binarny
            val_metrics["auc"] = roc_auc_score(y_true, y_prob[:, 1])
    else:
        print("Tylko jedna klasa w danych walidacyjnych, AUC nie jest zdefiniowany.")
        val_metrics["auc"] = 0.0
except Exception as e:
    print(f"Błąd podczas obliczania AUC: {e}")
    val_metrics["auc"] = 0.0
4. Dodanie lepszej obsługi ostrzeżeń pyqtgraph i sklearn
python# Dodaj na początku pliku (po importach)

# Dodaj po istniejących importach:
import warnings
# Ignoruj ostrzeżenia z pyqtgraph o All-NaN slice
warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")
# Ignoruj ostrzeżenia z sklearn o undefined metrics
warnings.filterwarnings("ignore", category=UserWarning, message="Only one class is present in y_true")
5. Poprawa formatowania wartości procentowych w log progress
python# Linia około 1401 (w funkcji fine_tune_model, wydruk postępu na koniec epoki)

# Obecny problematyczny kod:
print(f"  Train acc:  {train_acc:.2%}")
# i
print(f"  Val acc:    {val_acc:.2%}")

# Proponowana zmiana:
print(f"  Train acc:  {train_acc/100:.2%}")  # Dzielenie przez 100, ponieważ wartość jest już w procentach
# i
print(f"  Val acc:    {val_acc/100:.2%}")  # Dzielenie przez 100, ponieważ wartość jest już w procentach
6. Poprawienie przekazywania wartości do progress_callback
python# Linia około 1412 (w funkcji fine_tune_model, wywołanie progress_callback)

# Obecny problematyczny kod w sekcji wywołania progress_callback:
progress_callback(
    epoch + 1,
    num_epochs,
    avg_train_loss_for_epoch,  # Przekaż bieżącą średnią stratę treningową
    cumulative_train_acc_for_epoch,  # Przekaż bieżącą dokładność treningową
    val_loss_cb,
    val_acc_cb,
    top3_cb,
    top5_cb,
    precision_cb,
    recall_cb,
    f1_cb,
    auc_cb,
)

# Proponowana zmiana - upewnienie się, że wartości są w odpowiednim formacie:
progress_callback(
    epoch + 1,
    num_epochs,
    float(avg_train_loss_for_epoch),  # Upewnienie się, że jest typu float
    float(cumulative_train_acc_for_epoch),  # Upewnienie się, że jest typu float
    float(val_loss_cb),
    float(val_acc_cb),
    float(top3_cb),
    float(top5_cb),
    float(precision_cb),
    float(recall_cb),
    float(f1_cb),
    float(auc_cb),
)
Te zmiany powinny rozwiązać problemy z błędami w logach oraz poprawić stabilność funkcji fine_tune_model, szczególnie w zakresie obliczania i wyświetlania metryk treningowych.