Problem w pliku ai/fine_tuning.py
Ostrzeżenie pojawia się, gdy parametr k (w twoim przypadku 3) jest równy lub większy od liczby klas (n_classes, również 3) w modelu klasyfikacji. Jak wskazuje komunikat, taka sytuacja zawsze daje "idealny wynik" (1.0 lub 100%) i jest bezużyteczna w ocenie modelu, ponieważ wszystkie możliwe klasy są uwzględniane w przewidywaniach.
Proponowana zmiana
Zmiana w pliku ai/fine_tuning.py, w funkcji fine_tune_model, gdzie funkcja top_k_accuracy_score jest używana (około linii 577-587):
python# Top-k accuracy (jeśli więcej niż 2 klasy)
if new_num_classes > 2:
    k_values = min(5, new_num_classes)
    try:
        # Zmiana tutaj - dodanie warunku sprawdzającego, czy k jest mniejsze od liczby klas
        val_metrics["top3"] = (
            top_k_accuracy_score(
                y_true, y_prob, k=min(3, new_num_classes-1), normalize=True
            )
            if k_values >= 3 and new_num_classes > 3
            else 0.0
        )
        val_metrics["top5"] = (
            top_k_accuracy_score(
                y_true, y_prob, k=min(5, new_num_classes-1), normalize=True
            )
            if k_values >= 5 and new_num_classes > 5
            else 0.0
        )
    except Exception as e:
        print(f"Błąd podczas obliczania Top-k accuracy: {e}")
        val_metrics["top3"] = 0.0
        val_metrics["top5"] = 0.0
Teraz parametr k będzie zawsze o 1 mniejszy niż liczba klas, co zapewni, że metryka top-k będzie miała praktyczne znaczenie. Dodano również dodatkowe warunki sprawdzające, czy liczba klas jest wystarczająca do obliczenia konkretnej metryki top-k.
Alternatywnie, jeśli chcesz zachować wartości k=3 i k=5, ale po prostu wyciszyć ostrzeżenie, możesz również użyć kontekstu warnings.catch_warnings() wokół wywołania funkcji:
pythonimport warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UndefinedMetricWarning)
    # kod wywołujący top_k_accuracy_score
Jednak zmiana parametru k jest lepszym rozwiązaniem, ponieważ zapewnia, że metryka faktycznie dostarcza użytecznej informacji o jakości modelu.