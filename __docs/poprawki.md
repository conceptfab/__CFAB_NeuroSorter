
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

Brakujące/rekomendowane zmiany:

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


Metryki monitorowania: Brak tego ustawienia w interfejsie

Dodaj: accuracy, precision, recall


Label smoothing: Brak tego ustawienia w interfejsie

Zalecana wartość: 0.1



Podsumowanie optymalizacji:

Zmień optymalizator na RMSProp
Ustaw harmonogram uczenia na cosinusowy
Jeśli to możliwe, dodaj pozostałe brakujące parametry przez dodatkową konfigurację

Twoje aktualne ustawienia są już całkiem dobre, ale wprowadzenie tych zmian powinno poprawić wyniki treningu modelu EfficientNet.