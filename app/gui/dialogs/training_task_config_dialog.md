# Parametry konfiguracyjne treningu modelu

## Informacje ogólne

### type

- **Opis**: Typ konfiguracji
- **Wartości**: "training"
- **Kontrolka UI**: Pole tekstowe (tylko do odczytu)
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Dane i Model

### info

- **Opis**: Nazwa profilu modelu
- **Wartości**: Dowolny tekst
- **Kontrolka UI**: Pole tekstowe
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Dane i Model

### description

- **Opis**: Opis profilu
- **Wartości**: Dowolny tekst
- **Kontrolka UI**: Obszar tekstowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Dane i Model

### data_required

- **Opis**: Wymagania dotyczące danych treningowych
- **Wartości**: Dowolny tekst
- **Kontrolka UI**: Obszar tekstowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Dane i Model

### hardware_required

- **Opis**: Wymagania sprzętowe
- **Wartości**: Dowolny tekst
- **Kontrolka UI**: Obszar tekstowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Dane i Model

## Architektura modelu (model)

### architecture

- **Opis**: Typ architektury
- **Wartości**: "EfficientNet"
- **Kontrolka UI**: Dropdown
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Dane i Model

### variant

- **Opis**: Wariant modelu
- **Wartości**: "EfficientNet-B0" do "EfficientNet-B7"
- **Kontrolka UI**: Dropdown
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Dane i Model

### input_size

- **Opis**: Rozmiar wejściowy obrazu (piksele)
- **Wartości**: Liczba całkowita (np. 260)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Dane i Model

### num_classes

- **Opis**: Liczba klas do klasyfikacji
- **Wartości**: Liczba całkowita (np. 32, 40)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Dane i Model

### pretrained

- **Opis**: Czy używać wstępnie wytrenowanych wag
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik (toggle)
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Brak uwag
- **Zakładka**: Parametry Treningu

### pretrained_weights

- **Opis**: Źródło wag pretrenowanych
- **Wartości**: "imagenet" lub inne
- **Kontrolka UI**: Dropdown
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Brak uwag
- **Zakładka**: Parametry Treningu

### feature_extraction_only

- **Opis**: Czy używać modelu tylko do ekstrakcji cech
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Brak uwag
- **Zakładka**: Parametry Treningu

### activation

- **Opis**: Funkcja aktywacji w modelu
- **Wartości**: "swish", "relu", "sigmoid", itp.
- **Kontrolka UI**: Dropdown
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Brak uwag
- **Zakładka**: Parametry Treningu

### dropout_at_inference

- **Opis**: Czy używać dropoutu podczas inferencji
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Brak uwag
- **Zakładka**: Parametry Treningu

### global_pool

- **Opis**: Typ global pooling
- **Wartości**: "avg", "max"
- **Kontrolka UI**: Dropdown
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Brak uwag
- **Zakładka**: Parametry Treningu

### last_layer_activation

- **Opis**: Aktywacja ostatniej warstwy
- **Wartości**: "softmax", "sigmoid", "none"
- **Kontrolka UI**: Dropdown
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Kontrolka UI: Istnieje (self.last_layer_activation_combo), ale jest w zakładce "Dane i Model", podczas gdy plik .md sugeruje "Parametry Treningu".
- **Zakładka**: Parametry Treningu

## Parametry treningu (training)

### epochs

- **Opis**: Liczba epok treningu
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Parametry Treningu

### batch_size

- **Opis**: Rozmiar batcha
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Parametry Treningu

### learning_rate

- **Opis**: Współczynnik uczenia
- **Wartości**: Liczba zmiennoprzecinkowa (0 < x < 1)
- **Kontrolka UI**: Slider liczbowy lub pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Parametry Treningu

### optimizer

- **Opis**: Optymalizator
- **Wartości**: "AdamW", "Adam", "SGD", "RMSprop"
- **Kontrolka UI**: Dropdown
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Parametry Treningu

### scheduler.type

- **Opis**: Typ harmonogramu uczenia
- **Wartości**: "CosineAnnealingWarmRestarts", "StepLR", "OneCycleLR"
- **Kontrolka UI**: Dropdown
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Parametry Treningu

### scheduler.T_0

- **Opis**: Parametr T_0 dla CosineAnnealingWarmRestarts
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Brak uwag
- **Zakładka**: Parametry Treningu

### scheduler.T_mult

- **Opis**: Parametr T_mult dla CosineAnnealingWarmRestarts
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Brak uwag
- **Zakładka**: Parametry Treningu

### scheduler.eta_min

- **Opis**: Minimalna wartość learning rate
- **Wartości**: Liczba zmiennoprzecinkowa (bliska 0)
- **Kontrolka UI**: Pole numeryczne
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Brak uwag
- **Zakładka**: Parametry Treningu

### num_workers

- **Opis**: Liczba wątków do ładowania danych
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Parametry Treningu

### warmup_epochs

- **Opis**: Liczba epok rozgrzewki (warmup)
- **Wartości**: Liczba całkowita (≥0)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Parametry Treningu

### warmup_lr_init

- **Opis**: Początkowy learning rate dla rozgrzewki
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Pole numeryczne
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Brak uwag
- **Zakładka**: Parametry Treningu

### mixed_precision

- **Opis**: Czy używać mieszanej precyzji
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Parametry Treningu

### gradient_accumulation_steps

- **Opis**: Liczba kroków akumulacji gradientu
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Parametry Treningu

### gradient_clip

- **Opis**: Wartość przycinania gradientu
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Parametry Treningu

### evaluation_freq

- **Opis**: Częstotliwość ewaluacji (co ile epok)
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Brak uwag
- **Zakładka**: Parametry Treningu

### use_ema

- **Opis**: Czy używać Exponential Moving Average
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Brak uwag
- **Zakładka**: Parametry Treningu

### ema_decay

- **Opis**: Współczynnik EMA decay
- **Wartości**: Liczba zmiennoprzecinkowa (0 < x < 1)
- **Kontrolka UI**: Slider liczbowy
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Brak uwag
- **Zakładka**: Parametry Treningu

### freeze_base_model

- **Opis**: Czy zamrozić wagi bazowego modelu
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Parametry Treningu

### unfreeze_layers

- **Opis**: Które warstwy odmrozić
- **Wartości**: "all", "last_n", lista warstw
- **Kontrolka UI**: Dropdown lub wielowybór
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Parametry Treningu

### unfreeze_strategy

- **Opis**: Strategia odmrażania warstw
- **Wartości**: "gradual", "all_at_once"
- **Kontrolka UI**: Dropdown
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Parametry Treningu

### unfreeze_after_epochs

- **Opis**: Po ilu epokach odmrozić warstwy
- **Wartości**: Liczba całkowita (≥0)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Wczytywanie z profilu: Odczytuje wartość, ale używa bezpośredniego dostępu training_config["unfreeze_after_epochs"] co może prowadzić do błędu, jeśli klucz nie istnieje. Zalecane użycie .get(). Kontrolka UI: Istnieje (self.unfreeze_after_epochs_spin), ale znajduje się w zakładce "Zaawansowane", a nie "Parametry Treningu" jak w .md.
- **Zakładka**: Parametry Treningu

### frozen_lr

- **Opis**: Learning rate dla zamrożonych warstw
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Pole numeryczne
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Problem: Istnieją dwie kontrolki UI o tej samej nazwie self.frozen_lr_spin (jedna w zakładce "Parametry Treningu", druga w "Zaawansowane"). Metoda \_apply_profile próbuje ustawić wartość dla obu, a \_on_accept odczytuje wartość z kontrolki z zakładki "Zaawansowane". Powoduje to niespójność i potencjalne błędy. Należy ujednolicić do jednej kontrolki i poprawić logikę.
- **Zakładka**: Parametry Treningu

### unfrozen_lr

- **Opis**: Learning rate dla odmrożonych warstw
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Pole numeryczne
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Problem: Analogiczna sytuacja jak z frozen_lr. Dwie kontrolki self.unfrozen_lr_spin w różnych zakładkach.
- **Zakładka**: Parametry Treningu

### validation_split

- **Opis**: Część danych do walidacji
- **Wartości**: Liczba zmiennoprzecinkowa (0 < x < 1)
- **Kontrolka UI**: Slider liczbowy
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Problem: Brak implementacji wczytywania z profilu, brak kontrolki UI, brak zapisu do zadania.
- **Zakładka**: Parametry Treningu

## Parametry regularyzacji (regularization)

### weight_decay

- **Opis**: Współczynnik weight decay
- **Wartości**: Liczba zmiennoprzecinkowa (≥0)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Regularyzacja

### label_smoothing

- **Opis**: Współczynnik wygładzania etykiet
- **Wartości**: Liczba zmiennoprzecinkowa (0 ≤ x < 1)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Regularyzacja

### dropout_rate

- **Opis**: Współczynnik dropoutu
- **Wartości**: Liczba zmiennoprzecinkowa (0 ≤ x < 1)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Regularyzacja

### drop_connect_rate

- **Opis**: Współczynnik drop connect
- **Wartości**: Liczba zmiennoprzecinkowa (0 ≤ x < 1)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Regularyzacja

### momentum

- **Opis**: Współczynnik momentum (dla SGD)
- **Wartości**: Liczba zmiennoprzecinkowa (0 < x < 1)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Regularyzacja

### epsilon

- **Opis**: Epsilon dla optymalizatora
- **Wartości**: Liczba zmiennoprzecinkowa (>0, bliska 0)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Regularyzacja

### stochastic_depth.use

- **Opis**: Czy używać stochastic depth
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Wczytywanie z profilu: Wartość jest odczytywana z config["regularization"] ale nie jest przypisywana do kontrolki UI self.use_stoch_depth_check w metodzie \_apply_profile.
- **Zakładka**: Regularyzacja

### stochastic_depth.survival_probability

- **Opis**: Prawdopodobieństwo przetrwania dla stochastic depth
- **Wartości**: Liczba zmiennoprzecinkowa (0 < x ≤ 1)
- **Kontrolka UI**: Slider liczbowy
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Wczytywanie z profilu: Wartość jest odczytywana, ale nie przypisywana do kontrolki UI self.stoch_depth_survival_prob.
- **Zakładka**: Regularyzacja

### swa.use

- **Opis**: Czy używać Stochastic Weight Averaging
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Regularyzacja

### swa.start_epoch

- **Opis**: Od której epoki rozpocząć SWA
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Regularyzacja

### swa.lr_swa

- **Opis**: Learning rate dla SWA
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Pole numeryczne
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Wczytywanie z profilu: Wartość jest odczytywana, ale nie przypisywana do kontrolki UI self.swa_lr_spin.
- **Zakładka**: Regularyzacja

## Parametry augmentacji danych (augmentation)

### basic.use

- **Opis**: Czy używać podstawowych augmentacji
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Augmentacja

### basic.rotation

- **Opis**: Maksymalny kąt rotacji (stopnie)
- **Wartości**: Liczba zmiennoprzecinkowa (≥0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Augmentacja

### basic.brightness

- **Opis**: Zakres zmiany jasności
- **Wartości**: Liczba zmiennoprzecinkowa (≥0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Augmentacja

### basic.contrast

- **Opis**: Zakres zmiany kontrastu
- **Wartości**: Liczba zmiennoprzecinkowa (≥0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Brak uwag
- **Zakładka**: Augmentacja

### basic.saturation

- **Opis**: Zakres zmiany nasycenia
- **Wartości**: Liczba zmiennoprzecinkowa (≥0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Brak uwag
- **Zakładka**: Augmentacja

### basic.hue

- **Opis**: Zakres zmiany odcienia
- **Wartości**: Liczba zmiennoprzecinkowa (≥0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Brak uwag
- **Zakładka**: Augmentacja

### basic.shift

- **Opis**: Maksymalne przesunięcie (piksele lub %)
- **Wartości**: Liczba zmiennoprzecinkowa (≥0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Brak uwag
- **Zakładka**: Augmentacja

### basic.zoom

- **Opis**: Zakres powiększenia/zmniejszenia
- **Wartości**: Liczba zmiennoprzecinkowa (≥0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Brak uwag
- **Zakładka**: Augmentacja

### basic.horizontal_flip

- **Opis**: Czy stosować odbicia poziome
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Brak uwag
- **Zakładka**: Augmentacja

### basic.vertical_flip

- **Opis**: Czy stosować odbicia pionowe
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Brak uwag
- **Zakładka**: Augmentacja

### mixup.use

- **Opis**: Czy używać augmentacji Mixup
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Brak uwag
- **Zakładka**: Augmentacja

### mixup.alpha

- **Opis**: Parametr alpha dla Mixup
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Brak uwag
- **Zakładka**: Augmentacja

### cutmix.use

- **Opis**: Czy używać augmentacji CutMix
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Brak uwag
- **Zakładka**: Augmentacja

### cutmix.alpha

- **Opis**: Parametr alpha dla CutMix
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Brak uwag
- **Zakładka**: Augmentacja

### autoaugment.use

- **Opis**: Czy używać AutoAugment
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Augmentacja

### autoaugment.policy

- **Opis**: Polityka AutoAugment
- **Wartości**: "imagenet", "cifar", "svhn"
- **Kontrolka UI**: Dropdown
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Wczytywanie z profilu: Brak wczytywania wartości do self.autoaugment_policy_combo w \_apply_profile.
- **Zakładka**: Augmentacja

### randaugment.use

- **Opis**: Czy używać RandAugment
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Wczytywanie z profilu: Brak wczytywania wartości do self.randaugment_check.
- **Zakładka**: Augmentacja

### randaugment.n

- **Opis**: Liczba operacji RandAugment
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Wczytywanie z profilu: Brak wczytywania wartości do self.randaugment_n_spin.
- **Zakładka**: Augmentacja

### randaugment.m

- **Opis**: Intensywność operacji RandAugment
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Wczytywanie z profilu: Brak wczytywania wartości do self.randaugment_m_spin.
- **Zakładka**: Augmentacja

### trivialaugment.use

- **Opis**: Czy używać TrivialAugment
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Wczytywanie z profilu: Brak wczytywania wartości do self.trivialaugment_check.
- **Zakładka**: Augmentacja

### random_erase.use

- **Opis**: Czy używać Random Erase
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Wczytywanie z profilu: Brak wczytywania wartości do self.random_erase_check.
- **Zakładka**: Augmentacja

### random_erase.probability

- **Opis**: Prawdopodobieństwo Random Erase
- **Wartości**: Liczba zmiennoprzecinkowa (0 ≤ x ≤ 1)
- **Kontrolka UI**: Slider liczbowy
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Wczytywanie z profilu: Brak wczytywania wartości do self.random_erase_prob_spin.
- **Zakładka**: Augmentacja

### random_erase.scale

- **Opis**: Zakres skali dla Random Erase
- **Wartości**: [min, max] gdzie min, max to liczby zmiennoprzecinkowe
- **Kontrolka UI**: Podwójny slider (range slider)
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Wczytywanie z profilu: Brak wczytywania wartości do self.random_erase_scale_min_spin i self.random_erase_scale_max_spin.
- **Zakładka**: Augmentacja

### random_erase.ratio

- **Opis**: Zakres proporcji dla Random Erase
- **Wartości**: [min, max] gdzie min, max to liczby zmiennoprzecinkowe
- **Kontrolka UI**: Podwójny slider (range slider)
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Wczytywanie z profilu: Brak wczytywania wartości do self.random_erase_ratio_min_spin i self.random_erase_ratio_max_spin.
- **Zakładka**: Augmentacja

### grid_distortion.enabled

- **Opis**: Czy używać zniekształcenia siatki
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Wczytywanie z profilu: Brak wczytywania wartości do self.grid_distortion_check.
- **Zakładka**: Augmentacja

### grid_distortion.probability

- **Opis**: Prawdopodobieństwo zniekształcenia siatki
- **Wartości**: Liczba zmiennoprzecinkowa (0 ≤ x ≤ 1)
- **Kontrolka UI**: Slider liczbowy
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Wczytywanie z profilu: Brak wczytywania wartości do self.grid_distortion_prob_spin.
- **Zakładka**: Augmentacja

### grid_distortion.distort_limit

- **Opis**: Limit zniekształcenia siatki
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Wczytywanie z profilu: Brak wczytywania wartości do self.grid_distortion_limit_spin.
- **Zakładka**: Augmentacja

## Parametry przetwarzania wstępnego (preprocessing)

### resize.enabled

- **Opis**: Czy włączyć zmianę rozmiaru obrazów
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Wczytywanie z profilu: Brak wczytywania wartości z sekcji preprocessing profilu do self.resize_check. Kontrolka UI: Istnieje (self.resize_check), ale w zakładce "Augmentacja", a nie "Preprocessing". Zapis do zadania: OK (zapisuje jako config["augmentation"]["resize"]["enabled"]).
- **Zakładka**: Preprocessing

### resize.size

- **Opis**: Docelowy rozmiar obrazów
- **Wartości**: [width, height] gdzie width, height to liczby całkowite
- **Kontrolka UI**: Podwójny spinner liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Preprocessing

### resize.mode

- **Opis**: Tryb zmiany rozmiaru
- **Wartości**: "bilinear", "bicubic", "nearest", "lanczos"
- **Kontrolka UI**: Dropdown
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak uwag
- **Zakładka**: Preprocessing

### normalize.enabled

- **Opis**: Czy włączyć normalizację
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Preprocessing

### normalize.mean

- **Opis**: Średnie wartości dla normalizacji
- **Wartości**: [R, G, B] gdzie R, G, B to liczby zmiennoprzecinkowe
- **Kontrolka UI**: Trzy pola liczbowe
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Preprocessing

### normalize.std

- **Opis**: Odchylenia standardowe dla normalizacji
- **Wartości**: [R, G, B] gdzie R, G, B to liczby zmiennoprzecinkowe
- **Kontrolka UI**: Trzy pola liczbowe
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Preprocessing

### grayscale.enabled

- **Opis**: Czy konwertować do skali szarości
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak uwag
- **Zakładka**: Preprocessing

### grayscale.num_output_channels

- **Opis**: Liczba kanałów wyjściowych
- **Wartości**: 1 lub 3
- **Kontrolka UI**: Spinner liczbowy
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak uwag
- **Zakładka**: Preprocessing

### color_jitter.enabled

- **Opis**: Czy włączyć modyfikację kolorów
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak uwag
- **Zakładka**: Preprocessing

### color_jitter.brightness

- **Opis**: Zakres zmiany jasności
- **Wartości**: Liczba zmiennoprzecinkowa (≥0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak uwag
- **Zakładka**: Preprocessing

### color_jitter.contrast

- **Opis**: Zakres zmiany kontrastu
- **Wartości**: Liczba zmiennoprzecinkowa (≥0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Problem: Brak implementacji.
- **Zakładka**: Preprocessing

### color_jitter.saturation

- **Opis**: Zakres zmiany nasycenia
- **Wartości**: Liczba zmiennoprzecinkowa (≥0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak uwag
- **Zakładka**: Preprocessing

### color_jitter.hue

- **Opis**: Zakres zmiany odcienia
- **Wartości**: Liczba zmiennoprzecinkowa (≥0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Problem: Brak implementacji.
- **Zakładka**: Preprocessing

### gaussian_blur.enabled

- **Opis**: Czy włączyć rozmycie Gaussa
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak uwag
- **Zakładka**: Preprocessing

### gaussian_blur.kernel_size

- **Opis**: Rozmiar jądra rozmycia
- **Wartości**: Liczba nieparzysta (>0)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak uwag
- **Zakładka**: Preprocessing

### gaussian_blur.sigma

- **Opis**: Odchylenie standardowe rozmycia
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak uwag
- **Zakładka**: Preprocessing

### cache_dataset

- **Opis**: Czy cachować zestaw danych
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak uwag

### scaling.method

- **Opis**: Metoda skalowania obrazów
- **Wartości**: "Bicubic", "Bilinear", "Nearest"
- **Kontrolka UI**: Dropdown
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag

### scaling.maintain_aspect_ratio

- **Opis**: Czy zachować proporcje obrazu
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag

### scaling.pad_to_square

- **Opis**: Czy dopełniać obraz do kwadratu
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag

### scaling.pad_mode

- **Opis**: Typ dopełnienia
- **Wartości**: "reflection", "constant", "edge"
- **Kontrolka UI**: Dropdown
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag

### scaling.pad_value

- **Opis**: Wartość dopełnienia (dla "constant")
- **Wartości**: Liczba całkowita (0-255)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag

### random_resize_crop.enabled

- **Opis**: Czy używać losowego przycinania przy zmianie rozmiaru
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak uwag

### random_resize_crop.size

- **Opis**: Docelowy rozmiar po przycięciu
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak uwag

### random_resize_crop.scale

- **Opis**: Zakres skali dla losowego przycinania
- **Wartości**: [min, max] gdzie min, max to liczby zmiennoprzecinkowe
- **Kontrolka UI**: Podwójny slider (range slider)
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak uwag

### random_resize_crop.ratio

- **Opis**: Zakres proporcji dla losowego przycinania
- **Wartości**: [min, max] gdzie min, max to liczby zmiennoprzecinkowe
- **Kontrolka UI**: Podwójny slider (range slider)
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak uwag

## Parametry monitorowania i logowania (monitoring)

### metrics.accuracy

- **Opis**: Czy obliczać dokładność
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag

### metrics.precision

- **Opis**: Czy obliczać precyzję
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag

### metrics.recall

- **Opis**: Czy obliczać recall
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag

### metrics.f1

- **Opis**: Czy obliczać F1-score
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag

### metrics.topk

- **Opis**: Lista k dla top-k accuracy
- **Wartości**: Lista liczb całkowitych
- **Kontrolka UI**: Wielowybór lub pole tagów
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag

### metrics.confusion_matrix

- **Opis**: Czy generować macierz pomyłek
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag

### metrics.auc

- **Opis**: Czy obliczać AUC-ROC
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak uwag

### metrics.gpu_utilization

- **Opis**: Czy monitorować wykorzystanie GPU
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak uwag

### metrics.memory_usage

- **Opis**: Czy monitorować zużycie pamięci
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak uwag

### tensorboard.enabled

- **Opis**: Czy włączyć logowanie do TensorBoard
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring i Logging

### tensorboard.log_dir

- **Opis**: Katalog do zapisywania logów TensorBoard
- **Wartości**: Ścieżka do katalogu
- **Kontrolka UI**: Pole tekstowe + przycisk wyboru katalogu
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring i Logging

### tensorboard.update_freq

- **Opis**: Częstotliwość aktualizacji logów
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring i Logging

### wandb.enabled

- **Opis**: Czy włączyć logowanie do Weights & Biases
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring i Logging

### wandb.project

- **Opis**: Nazwa projektu w W&B
- **Wartości**: Nazwa projektu
- **Kontrolka UI**: Pole tekstowe
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring i Logging

### wandb.entity

- **Opis**: Nazwa użytkownika/zespołu w W&B
- **Wartości**: Nazwa użytkownika/zespołu
- **Kontrolka UI**: Pole tekstowe
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring i Logging

### wandb.tags

- **Opis**: Tagi dla eksperymentu w W&B
- **Wartości**: Lista tagów
- **Kontrolka UI**: Pole tekstowe z możliwością dodawania wielu tagów
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring i Logging

### checkpoint.enabled

- **Opis**: Czy włączyć zapisywanie checkpointów
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring i Logging

### checkpoint.dir

- **Opis**: Katalog do zapisywania checkpointów
- **Wartości**: Ścieżka do katalogu
- **Kontrolka UI**: Pole tekstowe + przycisk wyboru katalogu
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring i Logging

### checkpoint.save_best_only

- **Opis**: Czy zapisywać tylko najlepszy model
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring i Logging

### checkpoint.monitor

- **Opis**: Metryka do monitorowania
- **Wartości**: Nazwa metryki
- **Kontrolka UI**: Dropdown
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring i Logging

### checkpoint.mode

- **Opis**: Tryb monitorowania
- **Wartości**: "min" lub "max"
- **Kontrolka UI**: Dropdown
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring i Logging

### checkpoint.save_freq

- **Opis**: Częstotliwość zapisywania checkpointów
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring i Logging

### early_stopping.enabled

- **Opis**: Czy włączyć wczesne zatrzymywanie
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring i Logging

### early_stopping.monitor

- **Opis**: Metryka do monitorowania
- **Wartości**: Nazwa metryki
- **Kontrolka UI**: Dropdown
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring i Logging

### early_stopping.mode

- **Opis**: Tryb monitorowania
- **Wartości**: "min" lub "max"
- **Kontrolka UI**: Dropdown
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring i Logging

### early_stopping.patience

- **Opis**: Liczba epok bez poprawy przed zatrzymaniem
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring i Logging

### early_stopping.min_delta

- **Opis**: Minimalna zmiana uznawana za poprawę
- **Wartości**: Liczba zmiennoprzecinkowa (≥0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring i Logging

### reduce_lr.enabled

- **Opis**: Czy włączyć redukcję learning rate
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring i Logging

### reduce_lr.monitor

- **Opis**: Metryka do monitorowania
- **Wartości**: Nazwa metryki
- **Kontrolka UI**: Dropdown
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring i Logging

### reduce_lr.mode

- **Opis**: Tryb monitorowania
- **Wartości**: "min" lub "max"
- **Kontrolka UI**: Dropdown
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring i Logging

### reduce_lr.factor

- **Opis**: Współczynnik redukcji learning rate
- **Wartości**: Liczba zmiennoprzecinkowa (0 < x < 1)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring i Logging

### reduce_lr.patience

- **Opis**: Liczba epok bez poprawy przed redukcją
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring i Logging

### reduce_lr.min_delta

- **Opis**: Minimalna zmiana uznawana za poprawę
- **Wartości**: Liczba zmiennoprzecinkowa (≥0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring i Logging

### reduce_lr.min_lr

- **Opis**: Minimalna wartość learning rate
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Monitoring i Logging

## Parametry zaawansowane (advanced)

### seed

- **Opis**: Ziarno losowości
- **Wartości**: Liczba całkowita
- **Kontrolka UI**: Spinner liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### deterministic

- **Opis**: Czy używać deterministycznych operacji
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### benchmark

- **Opis**: Czy włączyć benchmark CUDA
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### num_workers

- **Opis**: Liczba workerów do ładowania danych
- **Wartości**: Liczba całkowita (≥0)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### pin_memory

- **Opis**: Czy używać pin memory
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### prefetch_factor

- **Opis**: Liczba próbek do prefetchowania
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### persistent_workers

- **Opis**: Czy używać persistent workers
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### gradient_clip_val

- **Opis**: Wartość przycinania gradientów
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### gradient_clip_algorithm

- **Opis**: Algorytm przycinania gradientów
- **Wartości**: "norm", "value"
- **Kontrolka UI**: Dropdown
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### accumulate_grad_batches

- **Opis**: Liczba batchy do akumulacji gradientów
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### sync_batchnorm

- **Opis**: Czy synchronizować BatchNorm
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### precision

- **Opis**: Precyzja obliczeń
- **Wartości**: 16, 32, 64, "bf16", "mixed"
- **Kontrolka UI**: Dropdown
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### amp_level

- **Opis**: Poziom automatycznej mieszanej precyzji
- **Wartości**: "O0", "O1", "O2", "O3"
- **Kontrolka UI**: Dropdown
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### gradient_clip_norm

- **Opis**: Maksymalna norma gradientów
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### gradient_clip_mode

- **Opis**: Tryb przycinania gradientów
- **Wartości**: "norm", "value", "agc"
- **Kontrolka UI**: Dropdown
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### gradient_clip_agc

- **Opis**: Czy używać Adaptive Gradient Clipping
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### gradient_clip_agc_clipping

- **Opis**: Wartość przycinania dla AGC
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### gradient_clip_agc_eps

- **Opis**: Epsilon dla AGC
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### gradient_clip_agc_eps_inside

- **Opis**: Epsilon wewnętrzny dla AGC
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### gradient_clip_agc_eps_outside

- **Opis**: Epsilon zewnętrzny dla AGC
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### gradient_clip_agc_eps_inside_scale

- **Opis**: Skala epsilon wewnętrznego dla AGC
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### gradient_clip_agc_eps_outside_scale

- **Opis**: Skala epsilon zewnętrznego dla AGC
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### gradient_clip_agc_eps_inside_scale_min

- **Opis**: Minimalna skala epsilon wewnętrznego dla AGC
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### gradient_clip_agc_eps_outside_scale_min

- **Opis**: Minimalna skala epsilon zewnętrznego dla AGC
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### gradient_clip_agc_eps_inside_scale_max

- **Opis**: Maksymalna skala epsilon wewnętrznego dla AGC
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane

### gradient_clip_agc_eps_outside_scale_max

- **Opis**: Maksymalna skala epsilon zewnętrznego dla AGC
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag
- **Zakładka**: Zaawansowane
