
Optymalne ustawienia treningu dla EfficientNet
Aktualne ustawienia są poprawne:

Architektura: efficientnet
Liczba epok: 50 (odpowiednia wartość)
Rozmiar wsadu (input): 128 (dobra wartość)
Współczynnik uczenia: 0.001 (dobra wartość początkowa)
Optymalizator: Adam (lepiej zmienić na RMSProp)
Liczba wątków do ładowania danych: 16 (odpowiednia)
Współczynnik regularyzacji L2: 1e-4 (dobra wartość)
Wartość przycinania gradientów: 0.10 (dobra wartość)
Early stopping: 5 epok (odpowiednia)
Mixed precision: włączone (bardzo dobrze)
Augmentacja: podstawowa i zaawansowana (bardzo dobrze)

Optymalizator: Zmień z Adam na RMSProp (zalecany dla EfficientNet)
Parametry: momentum = 0.9, epsilon = 0.001
Harmonogram uczenia: Ustaw na "cosine" zamiast "None"
Dodaj rozgrzewkę liniową przez pierwsze 5 epok
Rozmiar batcha: Brak tego ustawienia w interfejsie
Zalecana wartość: 32-64 dla EfficientNet-B0
Funkcja straty: Brak tego ustawienia w interfejsie
Zalecana: categorical_crossentropy (dla klasyfikacji wieloklasowej)
Lub binary_crossentropy (dla klasyfikacji binarnej)
Parametr drop_connect_rate: Brak tego ustawienia w interfejsie
Zalecana wartość: 0.2
Metryki monitorowania: Brak tego ustawienia w interfejsie => Dodaj: accuracy, precision, recall
Label smoothing: Brak tego ustawienia w interfejsie
Zalecana wartość: 0.1
Podsumowanie optymalizacji:

Zmień optymalizator na RMSProp
Ustaw harmonogram uczenia na cosinusowy
Jeśli to możliwe, dodaj pozostałe brakujące parametry przez dodatkową konfigurację

Twoje aktualne ustawienia są już całkiem dobre, ale wprowadzenie tych zmian powinno poprawić wyniki treningu modelu EfficientNet.


Główne metryki monitorowania dla EfficientNet:

val_loss - najważniejsza metryka do monitorowania dla early stopping, pozwala wykryć przeuczenie
val_accuracy - podstawowa metryka oceny jakości klasyfikacji
val_top_k_accuracy (gdzie k=3 lub k=5) - przydatne w przypadku klasyfikacji wieloklasowej z wieloma klasami

Dodatkowe metryki warte monitorowania:

val_precision - ważna w sytuacjach, gdy fałszywie pozytywne wyniki są kosztowne
val_recall - istotna, gdy fałszywie negatywne wyniki są niepożądane
val_f1_score - zbalansowana miara między precision i recall
val_auc - dobra miara ogólnej jakości modelu klasyfikacyjnego

Które metryki używać do early stopping?

Jeśli zależy Ci na ogólnej skuteczności modelu, użyj val_loss jako metryki do early stopping
Jeśli optymalizujesz pod kątem konkretnej miary (np. dokładności), możesz użyć val_accuracy

W przypadku imbalanced datasets (niezrównoważonych zbiorów danych), warto rozważyć monitorowanie val_f1_score lub val_auc zamiast prostej dokładności.