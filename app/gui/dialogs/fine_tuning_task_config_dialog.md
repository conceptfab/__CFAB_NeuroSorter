# Raport implementacji parametrów fine-tuningu modelu

## Główne informacje

### type
- **Opis**: Typ konfiguracji
- **Wartości**: "fine_tuning"
- **Kontrolka UI**: Pole tekstowe (tylko do odczytu)
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Informacje ogólne

### info
- **Opis**: Nazwa profilu modelu
- **Wartości**: Dowolny tekst
- **Kontrolka UI**: Pole tekstowe
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Informacje ogólne

### description
- **Opis**: Opis profilu
- **Wartości**: Dowolny tekst
- **Kontrolka UI**: Obszar tekstowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Informacje ogólne

### data_required
- **Opis**: Informacja o wymaganych danych
- **Wartości**: Dowolny tekst
- **Kontrolka UI**: Obszar tekstowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Informacje ogólne

### hardware_required
- **Opis**: Wymagania sprzętowe
- **Wartości**: Dowolny tekst
- **Kontrolka UI**: Pole tekstowe
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Informacje ogólne

## Model

### architecture
- **Opis**: Architektura modelu
- **Wartości**: "EfficientNet"
- **Kontrolka UI**: Lista rozwijana
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Model

### variant
- **Opis**: Wariant architektury
- **Wartości**: "EfficientNet-B0", "EfficientNet-B1", "EfficientNet-B2", "EfficientNet-B3", itd.
- **Kontrolka UI**: Lista rozwijana
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Model

### input_size
- **Opis**: Rozmiar wejściowy obrazu
- **Wartości**: Liczba całkowita (np. 260)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Model

### num_classes
- **Opis**: Liczba klas do klasyfikacji
- **Wartości**: Liczba całkowita (np. 32)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Model

### pretrained
- **Opis**: Czy używać wstępnie wytrenowanych wag
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Model

### pretrained_weights
- **Opis**: Źródło wstępnie wytrenowanych wag
- **Wartości**: "imagenet", inne źródła
- **Kontrolka UI**: Lista rozwijana
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Model

### feature_extraction_only
- **Opis**: Czy używać tylko ekstrakcji cech
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Model

### activation
- **Opis**: Funkcja aktywacji
- **Wartości**: "swish", "relu", "leaky_relu", "sigmoid"
- **Kontrolka UI**: Lista rozwijana
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Model

### dropout_at_inference
- **Opis**: Czy używać dropout podczas wnioskowania
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Model

### global_pool
- **Opis**: Metoda globalnego poolingu
- **Wartości**: "avg", "max", "concat"
- **Kontrolka UI**: Lista rozwijana
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Model

### last_layer_activation
- **Opis**: Aktywacja ostatniej warstwy
- **Wartości**: "softmax", "sigmoid", "none"
- **Kontrolka UI**: Lista rozwijana
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Model

## Trening

### epochs
- **Opis**: Liczba epok treningu
- **Wartości**: Liczba całkowita (np. 100)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Trening

### batch_size
- **Opis**: Rozmiar partii danych
- **Wartości**: Liczba całkowita (np. 48)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Trening

### learning_rate
- **Opis**: Współczynnik uczenia
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.0002)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Trening

### optimizer
- **Opis**: Optymalizator
- **Wartości**: "AdamW", "Adam", "SGD", "RMSprop"
- **Kontrolka UI**: Lista rozwijana
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Trening

### scheduler.type
- **Opis**: Typ schedulera
- **Wartości**: "CosineAnnealingWarmRestarts", "StepLR", "ReduceLROnPlateau", "OneCycleLR"
- **Kontrolka UI**: Lista rozwijana
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Trening

### scheduler.T_0
- **Opis**: Parametr T_0 dla CosineAnnealingWarmRestarts
- **Wartości**: Liczba całkowita (np. 10)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Trening

### scheduler.T_mult
- **Opis**: Parametr T_mult dla CosineAnnealingWarmRestarts
- **Wartości**: Liczba całkowita (np. 2)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Trening

### scheduler.eta_min
- **Opis**: Minimalna wartość learning rate
- **Wartości**: Liczba zmiennoprzecinkowa (np. 1e-7)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Trening

### num_workers
- **Opis**: Liczba wątków do ładowania danych
- **Wartości**: Liczba całkowita (np. 4)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Trening

### warmup_epochs
- **Opis**: Liczba epok rozgrzewki
- **Wartości**: Liczba całkowita (np. 5)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Trening

### warmup_lr_init
- **Opis**: Początkowy learning rate dla rozgrzewki
- **Wartości**: Liczba zmiennoprzecinkowa (np. 5e-7)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Trening

### mixed_precision
- **Opis**: Czy używać mieszanej precyzji
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Trening

### gradient_accumulation_steps
- **Opis**: Liczba kroków do akumulacji gradientu
- **Wartości**: Liczba całkowita (np. 2)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Trening

### gradient_clip
- **Opis**: Wartość przycinania gradientu
- **Wartości**: Liczba zmiennoprzecinkowa (np. 1.0)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Trening

### evaluation_freq
- **Opis**: Częstotliwość ewaluacji
- **Wartości**: Liczba całkowita (np. 1)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Trening

### use_ema
- **Opis**: Czy używać Exponential Moving Average
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Trening

### ema_decay
- **Opis**: Współczynnik zaniku EMA
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.9999)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Trening

### unfreeze_strategy
- **Opis**: Strategia odmrażania warstw
- **Wartości**: "gradual_with_lr_scaling", "all_at_once", "none"
- **Kontrolka UI**: Lista rozwijana
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Trening

### unfreeze_after_epochs
- **Opis**: Po ilu epokach odmrozić warstwy
- **Wartości**: Liczba całkowita (np. 5)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Trening

### unfreeze_layers
- **Opis**: Liczba warstw do odmrożenia
- **Wartości**: Liczba całkowita (np. 20)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Trening

### frozen_lr
- **Opis**: Learning rate dla zamrożonych warstw
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.0001)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Trening

### unfrozen_lr
- **Opis**: Learning rate dla odmrożonych warstw
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.001)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Trening

### layer_specific_lr
- **Opis**: Czy używać specyficznych learning rate dla warstw
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Trening

### lr_scaling_factor
- **Opis**: Współczynnik skalowania learning rate
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.1)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Trening

### loss_function
- **Opis**: Funkcja straty
- **Wartości**: "focal_loss", "cross_entropy", "binary_cross_entropy"
- **Kontrolka UI**: Lista rozwijana
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Trening

### focal_loss_gamma
- **Opis**: Parametr gamma dla focal loss
- **Wartości**: Liczba zmiennoprzecinkowa (np. 2.0)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Trening

### focal_loss_alpha
- **Opis**: Parametr alpha dla focal loss
- **Wartości**: "auto", wartość zmiennoprzecinkowa lub lista wartości
- **Kontrolka UI**: Pole tekstowe/numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Trening

## Regularyzacja

### weight_decay
- **Opis**: Współczynnik regularyzacji wag
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.00015)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Regularyzacja

### label_smoothing
- **Opis**: Współczynnik wygładzania etykiet
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.1)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Regularyzacja

### dropout_rate
- **Opis**: Współczynnik dropout
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.3)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Regularyzacja

### drop_connect_rate
- **Opis**: Współczynnik drop connect
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.2)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Regularyzacja

### momentum
- **Opis**: Współczynnik momentum
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.9)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Regularyzacja

### epsilon
- **Opis**: Epsilon do stabilności numerycznej
- **Wartości**: Liczba zmiennoprzecinkowa (np. 1e-6)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Regularyzacja

### swa.use
- **Opis**: Czy używać Stochastic Weight Averaging
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Regularyzacja

### swa.start_epoch
- **Opis**: Epoka rozpoczęcia SWA
- **Wartości**: Liczba całkowita (np. 80)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Regularyzacja

### stochastic_depth.use
- **Opis**: Czy używać stochastic depth
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Regularyzacja

### stochastic_depth.drop_rate
- **Opis**: Współczynnik drop rate dla stochastic depth
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.2)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Regularyzacja

### stochastic_depth.survival_probability
- **Opis**: Prawdopodobieństwo przetrwania warstwy
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.8)
- **Kontrolka UI**: Pole numeryczne
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Wczytywanie z profilu: Wartość jest poprawnie odczytywana z pliku JSON, ale należy sprawdzić przypisanie do kontrolki UI
- **Zakładka**: Regularyzacja

### random_erase.use
- **Opis**: Czy używać random erase
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Regularyzacja

### random_erase.probability
- **Opis**: Prawdopodobieństwo random erase
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.25)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Regularyzacja

### random_erase.mode
- **Opis**: Tryb random erase
- **Wartości**: "pixel", "block"
- **Kontrolka UI**: Lista rozwijana
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Regularyzacja

## Augmentacja

### augmentation.image_size
- **Opis**: Rozmiar obrazu po augmentacji
- **Wartości**: Lista dwóch liczb całkowitych [szerokość, wysokość]
- **Kontrolka UI**: Dwa pola numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Augmentacja

### basic.use
- **Opis**: Czy używać podstawowej augmentacji
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Augmentacja

### basic.rotation
- **Opis**: Zakres rotacji (stopnie)
- **Wartości**: Liczba całkowita (np. 30)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Augmentacja

### basic.brightness
- **Opis**: Zakres zmiany jasności
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.3)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Augmentacja

### basic.shift
- **Opis**: Zakres przesunięcia
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.15)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Augmentacja

### basic.zoom
- **Opis**: Zakres zoomu
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.2)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Augmentacja

### basic.horizontal_flip
- **Opis**: Czy używać odbicia poziomego
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Augmentacja

### basic.vertical_flip
- **Opis**: Czy używać odbicia pionowego
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Augmentacja

### mixup.use
- **Opis**: Czy używać augmentacji mixup
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Augmentacja

### mixup.alpha
- **Opis**: Parametr alpha dla mixup
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.3)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Augmentacja

### cutmix.use
- **Opis**: Czy używać augmentacji cutmix
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Augmentacja

### cutmix.alpha
- **Opis**: Parametr alpha dla cutmix
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.4)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Augmentacja

### autoaugment.use
- **Opis**: Czy używać autoaugment
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Augmentacja

### randaugment.use
- **Opis**: Czy używać randaugment
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Augmentacja

### randaugment.n
- **Opis**: Liczba operacji do zastosowania
- **Wartości**: Liczba całkowita (np. 2)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Augmentacja

### randaugment.m
- **Opis**: Siła operacji
- **Wartości**: Liczba całkowita (np. 7)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Augmentacja

### advanced.contrast
- **Opis**: Zakres zmiany kontrastu
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.2)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Augmentacja

### advanced.saturation
- **Opis**: Zakres zmiany nasycenia
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.2)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Augmentacja

### advanced.hue
- **Opis**: Zakres zmiany odcienia
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.1)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Augmentacja

### advanced.shear
- **Opis**: Zakres shear
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.1)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Augmentacja

### advanced.channel_shift
- **Opis**: Zakres przesunięcia kanałów
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.0)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Augmentacja

## Preprocessing

### preprocessing.image_size
- **Opis**: Rozmiar obrazu do preprocessingu
- **Wartości**: Lista dwóch liczb całkowitych [szerokość, wysokość]
- **Kontrolka UI**: Dwa pola numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Preprocessing

### normalization.mean
- **Opis**: Średnie dla normalizacji kanałów RGB
- **Wartości**: Lista trzech liczb zmiennoprzecinkowych
- **Kontrolka UI**: Trzy pola numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Preprocessing

### normalization.std
- **Opis**: Odchylenia standardowe dla normalizacji kanałów RGB
- **Wartości**: Lista trzech liczb zmiennoprzecinkowych
- **Kontrolka UI**: Trzy pola numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Preprocessing

### resize_mode
- **Opis**: Metoda zmiany rozmiaru
- **Wartości**: "bilinear", "bicubic", "nearest"
- **Kontrolka UI**: Lista rozwijana
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Preprocessing

### cache_dataset
- **Opis**: Czy buforować zbiór danych
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Preprocessing

## Monitoring

### metrics.accuracy
- **Opis**: Czy mierzyć dokładność
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring

### metrics.precision
- **Opis**: Czy mierzyć precyzję
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring

### metrics.recall
- **Opis**: Czy mierzyć recall
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring

### metrics.f1
- **Opis**: Czy mierzyć F1 score
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring

### metrics.topk
- **Opis**: Czy mierzyć top-k accuracy
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring

### metrics.confusion_matrix
- **Opis**: Czy generować macierz pomyłek
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring

### metrics.auc
- **Opis**: Czy mierzyć AUC
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring

### metrics.balanced_accuracy
- **Opis**: Czy mierzyć zbalansowaną dokładność
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring

### metrics.specificity
- **Opis**: Czy mierzyć specyficzność
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring

### metrics.kappa
- **Opis**: Czy mierzyć współczynnik kappa
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring

### metrics.per_class_metrics
- **Opis**: Czy mierzyć metryki per klasa
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring

### metrics.class_distribution_monitoring
- **Opis**: Czy monitorować rozkład klas
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring

### logging.use_tensorboard
- **Opis**: Czy używać TensorBoard do logowania
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring

### logging.use_wandb
- **Opis**: Czy używać Weights & Biases do logowania
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring

### logging.save_to_csv
- **Opis**: Czy zapisywać metryki do CSV
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring

### logging.logging_freq
- **Opis**: Częstotliwość logowania
- **Wartości**: "epoch", "batch", "step"
- **Kontrolka UI**: Lista rozwijana
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring

### visualization.use_gradcam
- **Opis**: Czy używać GradCAM do wizualizacji
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring

### visualization.use_feature_maps
- **Opis**: Czy używać map cech do wizualizacji
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring

### visualization.use_pred_samples
- **Opis**: Czy wizualizować przykłady predykcji
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring

### visualization.num_samples
- **Opis**: Liczba przykładów do wizualizacji
- **Wartości**: Liczba całkowita (np. 10)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring

### early_stopping.patience
- **Opis**: Cierpliwość dla early stopping
- **Wartości**: Liczba całkowita (np. 15)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring

### early_stopping.min_delta
- **Opis**: Minimalna zmiana uznawana za poprawę
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.0005)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring

### early_stopping.monitor
- **Opis**: Metrika do monitorowania
- **Wartości**: "val_loss", "val_accuracy", "val_balanced_accuracy"
- **Kontrolka UI**: Lista rozwijana
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring

### checkpointing.best_only
- **Opis**: Czy zapisywać tylko najlepszy model
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring

### checkpointing.save_frequency
- **Opis**: Częstotliwość zapisywania modelu
- **Wartości**: Liczba całkowita (np. 1)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring

### checkpointing.metric
- **Opis**: Metrika do monitorowania
- **Wartości**: "val_loss", "val_accuracy", "val_balanced_accuracy"
- **Kontrolka UI**: Lista rozwijana
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring

## Zaawansowane

### seed
- **Opis**: Ziarno losowości
- **Wartości**: Liczba całkowita (np. 42)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### deterministic
- **Opis**: Czy używać deterministycznych operacji
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### class_weights
- **Opis**: Wagi klas
- **Wartości**: "auto", "balanced", lista wartości
- **Kontrolka UI**: Lista rozwijana/pole tekstowe
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### sampler
- **Opis**: Metoda próbkowania
- **Wartości**: "balanced_weighted_random", "random"
- **Kontrolka UI**: Lista rozwijana
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### image_channels
- **Opis**: Liczba kanałów obrazu
- **Wartości**: Liczba całkowita (np. 3)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### tta.use
- **Opis**: Czy używać Test Time Augmentation
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### tta.num_augmentations
- **Opis**: Liczba augmentacji dla TTA
- **Wartości**: Liczba całkowita (np. 3)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### export_onnx
- **Opis**: Czy eksportować model do ONNX
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### quantization.use
- **Opis**: Czy kwantyzować model
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### quantization.precision
- **Opis**: Precyzja kwantyzacji
- **Wartości**: "int8", "fp16"
- **Kontrolka UI**: Lista rozwijana
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### imbalanced_data_handling.enable
- **Opis**: Czy włączyć obsługę niezbalansowanych danych
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### imbalanced_data_handling.strategy
- **Opis**: Strategia obsługi niezbalansowanych danych
- **Wartości**: "oversampling", "undersampling", "hybrid"
- **Kontrolka UI**: Lista rozwijana
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### imbalanced_data_handling.oversampling_ratio
- **Opis**: Współczynnik oversamplingu
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.8)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### imbalanced_data_handling.undersampling_threshold
- **Opis**: Próg undersamplingu
- **Wartości**: Liczba całkowita (np. 500)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### imbalanced_data_handling.dynamic_class_weights
- **Opis**: Czy używać dynamicznych wag klas
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### imbalanced_data_handling.focal_loss.use
- **Opis**: Czy używać focal loss dla niezbalansowanych danych
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### imbalanced_data_handling.focal_loss.gamma
- **Opis**: Parametr gamma dla focal loss
- **Wartości**: Liczba zmiennoprzecinkowa (np. 2.0)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### imbalanced_data_handling.focal_loss.alpha
- **Opis**: Parametr alpha dla focal loss
- **Wartości**: "auto", wartości numeryczne
- **Kontrolka UI**: Lista rozwijana/pole tekstowe
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### catastrophic_forgetting_prevention.enable
- **Opis**: Czy włączyć zapobieganie katastroficznemu zapominaniu
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### catastrophic_forgetting_prevention.preserve_original_classes
- **Opis**: Czy zachować oryginalne klasy
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### catastrophic_forgetting_prevention.rehearsal.use
- **Opis**: Czy używać rehearsal
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### catastrophic_forgetting_prevention.rehearsal.samples_per_class
- **Opis**: Liczba próbek na klasę
- **Wartości**: Liczba całkowita (np. 25)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### catastrophic_forgetting_prevention.rehearsal.synthetic_samples
- **Opis**: Czy używać syntetycznych próbek
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### catastrophic_forgetting_prevention.knowledge_distillation.use
- **Opis**: Czy używać destylacji wiedzy
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### catastrophic_forgetting_prevention.knowledge_distillation.temperature
- **Opis**: Temperatura dla destylacji wiedzy
- **Wartości**: Liczba zmiennoprzecinkowa (np. 2.0)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### catastrophic_forgetting_prevention.knowledge_distillation.alpha
- **Opis**: Waga dla destylacji wiedzy
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.4)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### catastrophic_forgetting_prevention.ewc_regularization.use
- **Opis**: Czy używać regularyzacji EWC
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### catastrophic_forgetting_prevention.ewc_regularization.lambda
- **Opis**: Waga regularyzacji EWC
- **Wartości**: Liczba zmiennoprzecinkowa (np. 5000.0)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### catastrophic_forgetting_prevention.ewc_regularization.fisher_sample_size
- **Opis**: Liczba próbek do obliczenia macierzy Fishera
- **Wartości**: Liczba całkowita (np. 200)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### catastrophic_forgetting_prevention.ewc_regularization.adaptive_lambda
- **Opis**: Czy używać adaptacyjnego lambda
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### catastrophic_forgetting_prevention.layer_freezing.strategy
- **Opis**: Strategia zamrażania warstw
- **Wartości**: "gradual", "fixed", "none"
- **Kontrolka UI**: Lista rozwijana
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### catastrophic_forgetting_prevention.layer_freezing.freeze_ratio
- **Opis**: Współczynnik zamrożenia warstw
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.7)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### evaluation_on_original_classes.enable
- **Opis**: Czy włączyć ewaluację na oryginalnych klasach
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### evaluation_on_original_classes.frequency
- **Opis**: Częstotliwość ewaluacji
- **Wartości**: Liczba całkowita (np. 5)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### evaluation_on_original_classes.report_metrics
- **Opis**: Czy raportować metryki
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### model_merging.enable
- **Opis**: Czy włączyć łączenie modeli
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### model_merging.method
- **Opis**: Metoda łączenia modeli
- **Wartości**: "fisher_weighted", "average", "max"
- **Kontrolka UI**: Lista rozwijana
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### model_merging.interpolation_factor
- **Opis**: Współczynnik interpolacji
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.5)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane