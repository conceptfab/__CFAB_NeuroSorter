1. Zakładka: Dane i Model

Katalog danych treningowych
Katalog danych walidacyjnych
Architektura modelu
Wariant modelu
Rozmiar obrazu wejściowego (nowy parametr)
Liczba klas (nowy parametr)


2. Zakładka: Parametry Treningu

Liczba epok
Rozmiar wsadu (batch size)
Współczynnik uczenia (learning rate)
Optymalizator (Adam, AdamW, SGD itp.)
Harmonogram uczenia (scheduler)
Liczba wątków do ładowania danych
Liczba epok rozgrzewki (warmup)
Użyj mixed precision

3. Zakładka: Regularyzacja i Optymalizacja

Współczynnik regularyzacji L2 (weight decay)
Wartość przycinania gradientów (gradient clip)
Label Smoothing
Drop Connect Rate
Dropout Rate
Momentum
Epsilon
Stochastic Weight Averaging (SWA)
Epoka rozpoczęcia SWA

4. Zakładka: Augmentacja Danych

Używaj podstawowej augmentacji
Używaj zaawansowanej augmentacji
Parametry augmentacji:

Maksymalny kąt rotacji
Zmiana jasności
Współczynnik przesunięcia (shift)
Współczynnik przybliżenia (zoom)

Techniki mieszania danych:

Używaj Mixup (+ wartość alpha)
Używaj CutMix (+ wartość alpha)

5. Zakładka: Monitorowanie i Zapis

Metryki monitorowania:

Accuracy
Precision
Recall
F1 Score
Top-k Accuracy

Wczesne zatrzymanie:

Liczba epok bez poprawy
Monitorowana metryka

Checkpointowanie:

Zapisuj tylko najlepszy model
Częstotliwość zapisu (epoki)

Katalog zapisu modelu (nowy parametr)
Zapisuj logi treningu (nowy parametr)

6. Zakładka: Zaawansowane

Parametry harmonogramu uczenia:

Patience
Factor
Minimalna wartość LR

Inicjalizacja wag (nowy parametr)
Zamrożenie warstw konwolucyjnych (nowy parametr)
Dodatkowe parametry dla specyficznych architektur
