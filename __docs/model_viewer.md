Kod w pliku model_checker.py umożliwia odczytanie i analizę następujących informacji z modelu PyTorch (zarówno w formie torch.nn.Module, jak i state_dict):
Struktura modelu:
Hierarchia warstw i podmodułów (drzewo modułów).
Nazwy warstw, podmodułów oraz parametrów.
Typy warstw (np. Linear, Conv2d, itp.).
Liczba parametrów w każdej warstwie.
Parametry modelu:
Kształty (shape) i typy (dtype) tensorów parametrów.
Możliwość podglądu wartości parametrów (np. średnia wartość).
Możliwość edycji wartości pojedynczego parametru (wszystkie wartości w tensorze można ustawić na wybraną liczbę).
Analiza modelu:
Całkowita liczba parametrów w modelu.
Liczba warstw z parametrami.
Szczegółowy raport z liczbą parametrów w każdej warstwie.
Wizualizacja parametrów:
Histogram wartości wszystkich parametrów.
Wykres rozkładu wartości parametrów (sortowanych).
Porównanie modeli:
Porównanie struktury dwóch modeli (różnice w warstwach i parametrach).
Porównanie wartości parametrów (wykres rozrzutu wartości parametrów obu modeli).
Eksport i zapis:
Eksport struktury modelu do pliku JSON lub TXT.
Eksport modelu do formatu ONNX lub TorchScript.
Zapis modelu do pliku PyTorch.
Podstawowe operacje:
Wczytywanie modelu z pliku.
Zapis modelu do pliku.