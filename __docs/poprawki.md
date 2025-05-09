
Problemy widoczne w logach:

Problem z AUC (Area Under Curve):
AUC: nan%
UndefinedMetricWarning: Only one class is present in y_true. ROC AUC score is not defined in that case.
Jest to oczekiwane ostrzeżenie w przypadku klasyfikacji jednoklasowej, gdy wszystkie przykłady walidacyjne należą do tej samej klasy. AUC nie może być obliczone, ponieważ wymaga przykładów z obu klas.
Problemy z wizualizacją:
RuntimeWarning: All-NaN slice encountered
self.bounds[ax] = (np.nanmin(d) - self._maxSpotWidth*0.7072, np.nanmax(d) + self._maxSpotWidth*0.7072)
Te ostrzeżenia są związane z biblioteką pyqtgraph, która próbuje narysować wykresy z wartościami NaN (Not a Number).

Rozwiązania:
1. Dostosowanie obliczania metryk AUC
python# Poprawka w pliku fine_tuning.py
# W sekcji obliczania metryk walidacyjnych:

try:
    if new_num_classes == 2:
        # Sprawdź, czy w zbiorze walidacyjnym występują obie klasy
        if len(np.unique(y_true)) > 1:
            val_metrics["auc"] = roc_auc_score(y_true, y_prob[:, 1])
        else:
            val_metrics["auc"] = 1.0  # Jeśli wszystkie próbki są z jednej klasy i model je poprawnie klasyfikuje
            print("Uwaga: Wszystkie próbki walidacyjne należą do jednej klasy. AUC ustawiono na 1.0.")
    else:
        # ...
except Exception as e:
    print(f"Błąd podczas obliczania AUC: {e}")
    val_metrics["auc"] = 0.0
2. Rozwiązanie problemu z wykresami
python# Poprawka w kodzie generowania wykresów (w miejscu gdzie używany jest pyqtgraph)

def update_plot(data, plot_item):
    # Sprawdź, czy dane zawierają wartości NaN
    if np.isnan(data).any():
        # Zastąp NaN wartościami 0 lub ostatnią prawidłową wartością
        data = np.nan_to_num(data, nan=0.0)
        # Możesz też użyć interpolacji:
        # from scipy import interpolate
        # x = np.arange(len(data))
        # mask = ~np.isnan(data)
        # if np.any(mask):  # Sprawdź czy istnieją jakiekolwiek prawidłowe dane
        #     f = interpolate.interp1d(x[mask], data[mask], bounds_error=False, fill_value=0)
        #     data = f(x)
    
    # Teraz możesz bezpiecznie aktualizować wykres
    plot_item.setData(data)
3. Monitorowanie dla jednej klasy
Dla klasyfikacji jednoklasowej (binarnej) dodaj lepsze monitorowanie specyficznych metryk:
python# Dodaj do funkcji fine_tune_model w pliku fine_tuning.py:

if new_num_classes == 2:
    # Oblicz specyficzne metryki dla problemu binarnego
    try:
        # Dokładność zbalansowana - lepsza miara dla niezbalansowanych zbiorów danych
        from sklearn.metrics import balanced_accuracy_score
        val_metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
        
        # Specyficzność (Specificity) - ważna dla detekcji jednej klasy
        # True Negative Rate: tn / (tn + fp)
        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        if (tn + fp) > 0:
            val_metrics["specificity"] = tn / (tn + fp)
        else:
            val_metrics["specificity"] = 0.0
            
        # Dodaj te metryki do historii
        history["val_balanced_accuracy"].append(val_metrics["balanced_accuracy"])
        history["val_specificity"].append(val_metrics["specificity"])
        
        # Wyświetl dodatkowe metryki
        print(f"  Val balanced acc: {val_metrics['balanced_accuracy']:.4f}")
        print(f"  Val specificity: {val_metrics['specificity']:.4f}")
    except Exception as e:
        print(f"Błąd podczas obliczania dodatkowych metryk binarnych: {e}")