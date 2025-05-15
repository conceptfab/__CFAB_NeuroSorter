# Parametry konfiguracyjne treningu modelu

## Informacje ogólne
Schemat działania i weryfikacji poprawności implementacji danego parametru: 
wczytywanie z profilu plik_profilu*.json -> ręczne dostrojenie przez UI /sprawdzenie kontrolki UI -> walidacja i zapisanie do zadania plik_zadania*.json

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
- **Do zrobienia**: Brak uwag. Wyświetlane w `self.profile_info` (ReadOnly). Zapisywane przy tworzeniu nowego profilu.
- **Zakładka**: Dane i Model

### description

- **Opis**: Opis profilu
- **Wartości**: Dowolny tekst
- **Kontrolka UI**: Obszar tekstowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Wyświetlane w `self.profile_description` (ReadOnly). Zapisywane przy tworzeniu nowego profilu.
- **Zakładka**: Dane i Model

### data_required

- **Opis**: Wymagania dotyczące danych treningowych
- **Wartości**: Dowolny tekst
- **Kontrolka UI**: Obszar tekstowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Wyświetlane w `self.profile_data_required` (ReadOnly). Zapisywane przy tworzeniu nowego profilu.
- **Zakładka**: Dane i Model

### hardware_required

- **Opis**: Wymagania sprzętowe
- **Wartości**: Dowolny tekst
- **Kontrolka UI**: Obszar tekstowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Wyświetlane w `self.profile_hardware_required` (ReadOnly). Zapisywane przy tworzeniu nowego profilu.
- **Zakładka**: Dane i Model

## Architektura modelu (model)

### architecture

- **Opis**: Typ architektury
- **Wartości**: "EfficientNet", "ConvNeXt"
- **Kontrolka UI**: Dropdown
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.arch_combo`. Wczytywanie z profilu: TAK. Zapis do zadania: TAK. Zapis do profilu: TAK.
- **Zakładka**: Dane i Model

### variant

- **Opis**: Wariant modelu
- **Wartości**: "EfficientNet-B0" do "EfficientNet-B7", "ConvNeXt-Tiny" do "ConvNeXt-Large"
- **Kontrolka UI**: Dropdown
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.variant_combo`. Wczytywanie z profilu: TAK. Zapis do zadania: TAK. Zapis do profilu: TAK.
- **Zakładka**: Dane i Model

### input_size

- **Opis**: Rozmiar wejściowy obrazu (piksele)
- **Wartości**: Liczba całkowita (np. 260)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.input_size_spin`. Wczytywanie z profilu: TAK. Zapis do zadania: TAK. Zapis do profilu: TAK.
- **Zakładka**: Dane i Model

### num_classes

- **Opis**: Liczba klas do klasyfikacji
- **Wartości**: Liczba całkowita (np. 32, 40)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.num_classes_spin`. Wczytywanie z profilu: TAK. Zapis do zadania: TAK. Zapis do profilu: TAK.
- **Zakładka**: Dane i Model

### pretrained

- **Opis**: Czy używać wstępnie wytrenowanych wag
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik (toggle)
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.pretrained_check` (instancja z `_create_training_params_tab` jest używana, jeśli ta zakładka jest tworzona po "Dane i Model", w przeciwnym razie instancja z `_create_data_model_tab`). Wczytywanie z profilu (`config.model.pretrained`): TAK. Zapis do zadania (`config.model.pretrained`): TAK. Zapis do profilu (`config.model.pretrained`): TAK.
- **Zakładka**: Parametry Treningu

### pretrained_weights

- **Opis**: Źródło wag pretrenowanych
- **Wartości**: "imagenet" lub inne
- **Kontrolka UI**: Dropdown
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.pretrained_weights_combo` (podobnie jak `pretrained`). Wczytywanie z profilu (`config.model.pretrained_weights`): TAK. Zapis do zadania (`config.model.pretrained_weights`): TAK. Zapis do profilu (`config.model.pretrained_weights`): TAK.
- **Zakładka**: Parametry Treningu

### feature_extraction_only

- **Opis**: Czy używać modelu tylko do ekstrakcji cech
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.feature_extraction_check` (podobnie jak `pretrained`). Wczytywanie z profilu (`config.model.feature_extraction_only`): TAK. Zapis do zadania (`config.model.feature_extraction_only`): TAK. Zapis do profilu (`config.model.feature_extraction_only`): TAK.
- **Zakładka**: Parametry Treningu

### activation

- **Opis**: Funkcja aktywacji w modelu
- **Wartości**: "swish", "relu", "sigmoid", itp.
- **Kontrolka UI**: Dropdown
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.activation_combo` (podobnie jak `pretrained`). Wczytywanie z profilu (`config.model.activation`): TAK. Zapis do zadania (`config.model.activation`): TAK. Zapis do profilu (`config.model.activation`): TAK.
- **Zakładka**: Parametry Treningu

### dropout_at_inference

- **Opis**: Czy używać dropoutu podczas inferencji
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.dropout_at_inference_check` (podobnie jak `pretrained`). Wczytywanie z profilu (`config.model.dropout_at_inference`): TAK. Zapis do zadania (`config.model.dropout_at_inference`): TAK. Zapis do profilu (`config.model.dropout_at_inference`): TAK.
- **Zakładka**: Parametry Treningu

### global_pool

- **Opis**: Typ global pooling
- **Wartości**: "avg", "max"
- **Kontrolka UI**: Dropdown
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.global_pool_combo` (podobnie jak `pretrained`). Wczytywanie z profilu (`config.model.global_pool`): TAK. Zapis do zadania (`config.model.global_pool`): TAK. Zapis do profilu (`config.model.global_pool`): TAK.
- **Zakładka**: Parametry Treningu

### last_layer_activation

- **Opis**: Aktywacja ostatniej warstwy
- **Wartości**: "softmax", "sigmoid", "none"
- **Kontrolka UI**: Dropdown
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.last_layer_activation_combo` (podobnie jak `pretrained`). Wczytywanie z profilu (`config.model.last_layer_activation`): TAK. Zapis do zadania (`config.model.last_layer_activation`): TAK. Zapis do profilu (`config.model.last_layer_activation`): TAK.
- **Zakładka**: Parametry Treningu

## Parametry treningu (training)

### epochs

- **Opis**: Liczba epok treningu
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka `self.epochs_spin` (instancja z `_create_training_params_tab` lub `_create_data_model_tab`). Wczytywanie z profilu (`config.training.epochs`): TAK. Zapis do zadania (`config.training.epochs`): TAK. Zapis do profilu (`config.training.epochs`): TAK.
- **Zakładka**: Parametry Treningu

### batch_size

- **Opis**: Rozmiar batcha
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Kontrolka UI (`self.parameter_rows["batch_size"]["value_widget"]`) istnieje w zakładce "Optymalizacja treningu". Brak bezpośredniego wczytywania z profilu (`config.training.batch_size`) do kontrolki UI (wczytywane z `hardware_profile` lub wartość użytkownika). Zapis do zadania (`config.training.batch_size`): TAK. Zapis do profilu (`config.training.batch_size`): TAK. Przenieść/zduplikować kontrolkę do "Parametry Treningu" lub zaktualizować opis zakładki.
- **Zakładka**: Parametry Treningu

### learning_rate

- **Opis**: Współczynnik uczenia
- **Wartości**: Liczba zmiennoprzecinkowa (0 < x < 1)
- **Kontrolka UI**: Slider liczbowy lub pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka `self.lr_spin` (instancja z `_create_training_params_tab` lub `_create_data_model_tab`). Wczytywanie z profilu (`config.training.learning_rate`): TAK. Zapis do zadania (`config.training.learning_rate`): TAK. Zapis do profilu (`config.training.learning_rate`): TAK.
- **Zakładka**: Parametry Treningu

### optimizer

- **Opis**: Optymalizator
- **Wartości**: "AdamW", "Adam", "SGD", "RMSprop"
- **Kontrolka UI**: Dropdown
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.optimizer_combo`. Wczytywanie z profilu (`config.training.optimizer`): TAK. Zapis do zadania (`config.training.optimizer`): TAK. Zapis do profilu (`config.training.optimizer`): TAK.
- **Zakładka**: Parametry Treningu

### scheduler.type

- **Opis**: Typ harmonogramu uczenia
- **Wartości**: "CosineAnnealingWarmRestarts", "StepLR", "OneCycleLR", "ReduceLROnPlateau", "CosineAnnealingLR", "None"
- **Kontrolka UI**: Dropdown
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.scheduler_combo`. Wczytywanie z profilu (`config.training.scheduler.type`): TAK. Zapis do zadania (`config.training.scheduler.type`): TAK. Zapis do profilu (`config.training.scheduler.type`): TAK.
- **Zakładka**: Parametry Treningu

### scheduler.T_0

- **Opis**: Parametr T_0 dla CosineAnnealingWarmRestarts
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.scheduler_t0_spin`. Wczytywanie z profilu (`config.training.scheduler.T_0` lub `config.training.T_0` w `_apply_profile`): TAK. Zapis do zadania (`config.training.scheduler.T_0`): TAK. Zapis do profilu (`config.training.scheduler.T_0`): TAK.
- **Zakładka**: Parametry Treningu

### scheduler.T_mult

- **Opis**: Parametr T_mult dla CosineAnnealingWarmRestarts
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.scheduler_tmult_spin`. Wczytywanie z profilu (`config.training.scheduler.T_mult` lub `config.training.T_mult`): TAK. Zapis do zadania (`config.training.scheduler.T_mult`): TAK. Zapis do profilu (`config.training.scheduler.T_mult`): TAK.
- **Zakładka**: Parametry Treningu

### scheduler.eta_min

- **Opis**: Minimalna wartość learning rate
- **Wartości**: Liczba zmiennoprzecinkowa (bliska 0)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.scheduler_eta_min_spin`. Wczytywanie z profilu (`config.training.scheduler.eta_min` lub `config.training.eta_min`): TAK. Zapis do zadania (`config.training.scheduler.eta_min`): TAK. Zapis do profilu (`config.training.scheduler.eta_min`): TAK.
- **Zakładka**: Parametry Treningu

### num_workers

- **Opis**: Liczba wątków do ładowania danych
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Kontrolka UI (`self.parameter_rows["num_workers"]["value_widget"]`) istnieje w zakładce "Optymalizacja treningu". Brak bezpośredniego wczytywania z profilu (`config.training.num_workers`) do kontrolki UI. Zapis do zadania (`config.training.num_workers`): TAK. Zapis do profilu (`config.training.num_workers`): TAK. Przenieść/zduplikować kontrolkę do "Parametry Treningu" lub zaktualizować opis zakładki.
- **Zakładka**: Parametry Treningu

### warmup_epochs

- **Opis**: Liczba epok rozgrzewki (warmup)
- **Wartości**: Liczba całkowita (≥0)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.warmup_epochs_spin`. Wczytywanie z profilu (`config.training.warmup_epochs`): TAK. Zapis do zadania (`config.training.warmup_epochs`): TAK. Zapis do profilu (`config.training.warmup_epochs`): TAK.
- **Zakładka**: Parametry Treningu

### warmup_lr_init

- **Opis**: Początkowy learning rate dla rozgrzewki
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.warmup_lr_init_spin`. Wczytywanie z profilu (`config.training.warmup_lr_init`): TAK. Zapis do zadania (`config.training.warmup_lr_init`): TAK. Zapis do profilu (`config.training.warmup_lr_init`): TAK.
- **Zakładka**: Parametry Treningu

### mixed_precision

- **Opis**: Czy używać mieszanej precyzji
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Kontrolka UI (`self.parameter_rows["use_mixed_precision"]["value_widget"]`) istnieje w zakładce "Optymalizacja treningu". Brak bezpośredniego wczytywania z profilu (`config.training.mixed_precision`) do kontrolki UI. Zapis do zadania (`config.training.mixed_precision`): TAK. Zapis do profilu (`config.training.mixed_precision`): TAK. Przenieść/zduplikować kontrolkę do "Parametry Treningu" lub zaktualizować opis zakładki.
- **Zakładka**: Parametry Treningu

### gradient_accumulation_steps

- **Opis**: Liczba kroków akumulacji gradientu
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Kontrolka UI (`self.grad_accum_steps_spin`) istnieje w zakładce "Dane i Model". Przenieść do "Parametry Treningu". Brak wczytywania z profilu. Brak zapisu do profilu. Zapis do zadania (`config.training.gradient_accumulation_steps`): TAK.
- **Zakładka**: Parametry Treningu

### gradient_clip

- **Opis**: Wartość przycinania gradientu
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI dla ścieżki `training.gradient_clip`. Parametr `regularization.gradient_clip` jest zaimplementowany (`self.gradient_clip_spin` w zakładce "Regularyzacja").
- **Zakładka**: Parametry Treningu

### evaluation_freq

- **Opis**: Częstotliwość ewaluacji (co ile epok)
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.evaluation_freq_spin`. Wczytywanie z profilu (`config.training.evaluation_freq`): TAK. Zapis do zadania (`config.training.evaluation_freq`): TAK. Zapis do profilu (`config.training.evaluation_freq`): TAK.
- **Zakładka**: Parametry Treningu

### use_ema

- **Opis**: Czy używać Exponential Moving Average
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.use_ema_check`. Wczytywanie z profilu (`config.training.use_ema`): TAK. Zapis do zadania (`config.training.use_ema`): TAK. Zapis do profilu (`config.training.use_ema`): TAK.
- **Zakładka**: Parametry Treningu

### ema_decay

- **Opis**: Współczynnik EMA decay
- **Wartości**: Liczba zmiennoprzecinkowa (0 < x < 1)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.ema_decay_spin` (QDoubleSpinBox). Wczytywanie z profilu (`config.training.ema_decay`): TAK. Zapis do zadania (`config.training.ema_decay`): TAK. Zapis do profilu (`config.training.ema_decay`): TAK.
- **Zakładka**: Parametry Treningu

### freeze_base_model

- **Opis**: Czy zamrozić wagi bazowego modelu
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Kontrolka UI (`self.freeze_base_model`) istnieje w zakładce "Zaawansowane". Wczytywanie z profilu (`config.training.freeze_base_model`): TAK. Zapis do zadania (`config.training.freeze_base_model`): TAK. Zapis do profilu (`config.training.freeze_base_model`): NIE. Przenieść do "Parametry Treningu" lub zaktualizować opis.
- **Zakładka**: Parametry Treningu

### unfreeze_layers

- **Opis**: Które warstwy odmrozić
- **Wartości**: "all", "last_n", lista warstw
- **Kontrolka UI**: Dropdown lub wielowybór
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Kontrolka UI (`self.unfreeze_layers` - QLineEdit) istnieje w zakładce "Zaawansowane". Wczytywanie z profilu (`config.training.unfreeze_layers`): TAK. Zapis do zadania (`config.training.unfreeze_layers`): TAK. Zapis do profilu (`config.training.unfreeze_layers`): NIE. Przenieść do "Parametry Treningu" lub zaktualizować opis.
- **Zakładka**: Parametry Treningu

### unfreeze_strategy

- **Opis**: Strategia odmrażania warstw
- **Wartości**: "gradual", "all_at_once" (w kodzie: "unfreeze_all", "unfreeze_gradual_end", "unfreeze_gradual_start", "unfreeze_after_epoochs")
- **Kontrolka UI**: Dropdown
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Kontrolka UI (`self.unfreeze_strategy`) istnieje w zakładce "Zaawansowane". Wczytywanie z profilu (`config.training.unfreeze_strategy`): TAK (z mapowaniem). Zapis do zadania (`config.training.unfreeze_strategy`): TAK. Zapis do profilu (`config.training.unfreeze_strategy`): NIE. Przenieść do "Parametry Treningu" lub zaktualizować opis.
- **Zakładka**: Parametry Treningu

### unfreeze_after_epochs

- **Opis**: Po ilu epokach odmrozić warstwy
- **Wartości**: Liczba całkowita (≥0)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Kontrolka UI (`self.unfreeze_after_epochs_spin`) istnieje w zakładce "Zaawansowane". Wczytywanie z profilu (`config.training.unfreeze_after_epochs`): TAK. Zapis do zadania (`config.training.unfreeze_after_epochs`): TAK. Zapis do profilu (`config.training.unfreeze_after_epochs`): NIE. Przenieść do "Parametry Treningu" lub zaktualizować opis.
- **Zakładka**: Parametry Treningu

### frozen_lr

- **Opis**: Learning rate dla zamrożonych warstw
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Pole numeryczne
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Kontrolka UI (`self.frozen_lr_spin`) istnieje w zakładce "Parametry Treningu". Wczytywanie z profilu (`config.training.frozen_lr`): TAK. Zapis do zadania (`config.training.frozen_lr`): TAK. Zapis do profilu (`config.training.frozen_lr`): NIE (w `_save_profile` brak tej ścieżki).
- **Zakładka**: Parametry Treningu

### unfrozen_lr

- **Opis**: Learning rate dla odmrożonych warstw
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Pole numeryczne
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Kontrolka UI (`self.unfrozen_lr_spin`) istnieje w zakładce "Parametry Treningu". Wczytywanie z profilu (`config.training.unfrozen_lr`): TAK. Zapis do zadania (`config.training.unfrozen_lr`): TAK. Zapis do profilu (`config.training.unfrozen_lr`): NIE.
- **Zakładka**: Parametry Treningu

### validation_split

- **Opis**: Część danych do walidacji
- **Wartości**: Liczba zmiennoprzecinkowa (0 < x < 1)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania.
- **Zakładka**: Parametry Treningu

## Parametry regularyzacji (regularization)

### weight_decay

- **Opis**: Współczynnik weight decay
- **Wartości**: Liczba zmiennoprzecinkowa (≥0)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.weight_decay_spin`. Wczytywanie z profilu: TAK. Zapis do zadania: TAK. Zapis do profilu: TAK.
- **Zakładka**: Regularyzacja

### label_smoothing

- **Opis**: Współczynnik wygładzania etykiet
- **Wartości**: Liczba zmiennoprzecinkowa (0 ≤ x < 1)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.label_smoothing_spin` (QDoubleSpinBox). Wczytywanie z profilu: TAK. Zapis do zadania: TAK. Zapis do profilu: TAK.
- **Zakładka**: Regularyzacja

### dropout_rate

- **Opis**: Współczynnik dropoutu
- **Wartości**: Liczba zmiennoprzecinkowa (0 ≤ x < 1)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.dropout_spin` (QDoubleSpinBox). Wczytywanie z profilu: TAK. Zapis do zadania: TAK. Zapis do profilu: TAK.
- **Zakładka**: Regularyzacja

### drop_connect_rate

- **Opis**: Współczynnik drop connect
- **Wartości**: Liczba zmiennoprzecinkowa (0 ≤ x < 1)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.drop_connect_spin` (QDoubleSpinBox). Wczytywanie z profilu: TAK. Zapis do zadania: TAK. Zapis do profilu: TAK.
- **Zakładka**: Regularyzacja

### momentum

- **Opis**: Współczynnik momentum (dla SGD)
- **Wartości**: Liczba zmiennoprzecinkowa (0 < x < 1)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.momentum_spin` (QDoubleSpinBox). Wczytywanie z profilu: TAK. Zapis do zadania: TAK. Zapis do profilu: TAK.
- **Zakładka**: Regularyzacja

### epsilon

- **Opis**: Epsilon dla optymalizatora
- **Wartości**: Liczba zmiennoprzecinkowa (>0, bliska 0)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.epsilon_spin`. Wczytywanie z profilu: TAK. Zapis do zadania: TAK. Zapis do profilu: TAK.
- **Zakładka**: Regularyzacja

### stochastic_depth.use

- **Opis**: Czy używać stochastic depth
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.use_stoch_depth_check`. Wczytywanie z profilu (`reg_config.get("stochastic_depth", {}).get("use", False)` w `_save_profile` - powinno być w `_apply_profile`). Zapis do zadania: TAK. Zapis do profilu: TAK. W `_apply_profile` brak wczytywania.
- **Zakładka**: Regularyzacja

### stochastic_depth.survival_probability

- **Opis**: Prawdopodobieństwo przetrwania dla stochastic depth
- **Wartości**: Liczba zmiennoprzecinkowa (0 < x ≤ 1)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.stoch_depth_survival_prob` (QDoubleSpinBox). Wczytywanie z profilu (`reg_config.get("stochastic_depth", {}).get("survival_probability", 0.8)` w `_save_profile` - powinno być w `_apply_profile`). Zapis do zadania: TAK. Zapis do profilu: TAK. W `_apply_profile` brak wczytywania.
- **Zakładka**: Regularyzacja

### swa.use

- **Opis**: Czy używać Stochastic Weight Averaging
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.use_swa_check`. Wczytywanie z profilu: TAK. Zapis do zadania: TAK. Zapis do profilu: TAK.
- **Zakładka**: Regularyzacja

### swa.start_epoch

- **Opis**: Od której epoki rozpocząć SWA
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.swa_start_epoch_spin`. Wczytywanie z profilu: TAK. Zapis do zadania: TAK. Zapis do profilu: TAK.
- **Zakładka**: Regularyzacja

### swa.lr_swa

- **Opis**: Learning rate dla SWA
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Pole numeryczne
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.swa_lr_spin`. Wczytywanie z profilu (`reg_config.get("swa", {}).get("lr_swa", 5e-5)` w `_apply_profile` - brak). Zapis do zadania: TAK. Zapis do profilu: TAK. W `_apply_profile` brak wczytywania.
- **Zakładka**: Regularyzacja

## Parametry augmentacji danych (augmentation)

### basic.use

- **Opis**: Czy używać podstawowych augmentacji
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.basic_aug_check`. Wczytywanie: TAK. Zapis: TAK.
- **Zakładka**: Augmentacja

### basic.rotation

- **Opis**: Maksymalny kąt rotacji (stopnie)
- **Wartości**: Liczba zmiennoprzecinkowa (≥0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.rotation_spin` (QSpinBox). Wczytywanie: TAK. Zapis: TAK.
- **Zakładka**: Augmentacja

### basic.brightness

- **Opis**: Zakres zmiany jasności
- **Wartości**: Liczba zmiennoprzecinkowa (≥0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.brightness_spin` (QDoubleSpinBox). Wczytywanie: TAK. Zapis: TAK.
- **Zakładka**: Augmentacja

### basic.contrast

- **Opis**: Zakres zmiany kontrastu
- **Wartości**: Liczba zmiennoprzecinkowa (≥0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.contrast_spin` (QDoubleSpinBox). Wczytywanie: TAK. Zapis: TAK.
- **Zakładka**: Augmentacja

### basic.saturation

- **Opis**: Zakres zmiany nasycenia
- **Wartości**: Liczba zmiennoprzecinkowa (≥0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.saturation_spin` (QDoubleSpinBox). Wczytywanie: TAK. Zapis: TAK.
- **Zakładka**: Augmentacja

### basic.hue

- **Opis**: Zakres zmiany odcienia
- **Wartości**: Liczba zmiennoprzecinkowa (≥0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.hue_spin` (QDoubleSpinBox). Wczytywanie: TAK. Zapis: TAK.
- **Zakładka**: Augmentacja

### basic.shift

- **Opis**: Maksymalne przesunięcie (piksele lub %)
- **Wartości**: Liczba zmiennoprzecinkowa (≥0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.shift_spin` (QDoubleSpinBox). Wczytywanie: TAK. Zapis: TAK.
- **Zakładka**: Augmentacja

### basic.zoom

- **Opis**: Zakres powiększenia/zmniejszenia
- **Wartości**: Liczba zmiennoprzecinkowa (≥0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.zoom_spin` (QDoubleSpinBox). Wczytywanie: TAK. Zapis: TAK.
- **Zakładka**: Augmentacja

### basic.horizontal_flip

- **Opis**: Czy stosować odbicia poziome
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.horizontal_flip_check`. Wczytywanie: TAK. Zapis: TAK.
- **Zakładka**: Augmentacja

### basic.vertical_flip

- **Opis**: Czy stosować odbicia pionowe
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.vertical_flip_check`. Wczytywanie: TAK. Zapis: TAK.
- **Zakładka**: Augmentacja

### mixup.use

- **Opis**: Czy używać augmentacji Mixup
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.mixup_check`. Wczytywanie: TAK. Zapis: TAK.
- **Zakładka**: Augmentacja

### mixup.alpha

- **Opis**: Parametr alpha dla Mixup
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.mixup_alpha_spin` (QDoubleSpinBox). Wczytywanie: TAK. Zapis: TAK.
- **Zakładka**: Augmentacja

### cutmix.use

- **Opis**: Czy używać augmentacji CutMix
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.cutmix_check`. Wczytywanie: TAK. Zapis: TAK.
- **Zakładka**: Augmentacja

### cutmix.alpha

- **Opis**: Parametr alpha dla CutMix
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.cutmix_alpha_spin` (QDoubleSpinBox). Wczytywanie: TAK. Zapis: TAK.
- **Zakładka**: Augmentacja

### autoaugment.use

- **Opis**: Czy używać AutoAugment
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.autoaugment_check`. Wczytywanie z profilu: NIE. Zapis: TAK.
- **Zakładka**: Augmentacja

### autoaugment.policy

- **Opis**: Polityka AutoAugment
- **Wartości**: "imagenet", "cifar", "svhn"
- **Kontrolka UI**: Dropdown
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.autoaugment_policy_combo`. Wczytywanie z profilu: NIE. Zapis: TAK.
- **Zakładka**: Augmentacja

### randaugment.use

- **Opis**: Czy używać RandAugment
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.randaugment_check`. Wczytywanie z profilu: NIE. Zapis: TAK.
- **Zakładka**: Augmentacja

### randaugment.n

- **Opis**: Liczba operacji RandAugment
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.randaugment_n_spin`. Wczytywanie z profilu: NIE. Zapis: TAK.
- **Zakładka**: Augmentacja

### randaugment.m

- **Opis**: Intensywność operacji RandAugment
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.randaugment_m_spin`. Wczytywanie z profilu: NIE. Zapis: TAK.
- **Zakładka**: Augmentacja

### trivialaugment.use

- **Opis**: Czy używać TrivialAugment
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.trivialaugment_check`. Wczytywanie z profilu: NIE. Zapis: TAK.
- **Zakładka**: Augmentacja

### random_erase.use

- **Opis**: Czy używać Random Erase
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.random_erase_check`. Wczytywanie z profilu: NIE. Zapis: TAK.
- **Zakładka**: Augmentacja

### random_erase.probability

- **Opis**: Prawdopodobieństwo Random Erase
- **Wartości**: Liczba zmiennoprzecinkowa (0 ≤ x ≤ 1)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.random_erase_prob_spin` (QDoubleSpinBox). Wczytywanie z profilu: NIE. Zapis: TAK.
- **Zakładka**: Augmentacja

### random_erase.scale

- **Opis**: Zakres skali dla Random Erase
- **Wartości**: [min, max] gdzie min, max to liczby zmiennoprzecinkowe
- **Kontrolka UI**: Podwójny slider (range slider)
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolki: `self.random_erase_scale_min_spin`, `self.random_erase_scale_max_spin` (QDoubleSpinBox). Wczytywanie z profilu: NIE. Zapis: TAK.
- **Zakładka**: Augmentacja

### random_erase.ratio

- **Opis**: Zakres proporcji dla Random Erase
- **Wartości**: [min, max] gdzie min, max to liczby zmiennoprzecinkowe
- **Kontrolka UI**: Podwójny slider (range slider)
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolki: `self.random_erase_ratio_min_spin`, `self.random_erase_ratio_max_spin` (QDoubleSpinBox). Wczytywanie z profilu: NIE. Zapis: TAK.
- **Zakładka**: Augmentacja

### grid_distortion.enabled

- **Opis**: Czy używać zniekształcenia siatki
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.grid_distortion_check`. Wczytywanie z profilu: NIE. Zapis: TAK.
- **Zakładka**: Augmentacja

### grid_distortion.probability

- **Opis**: Prawdopodobieństwo zniekształcenia siatki
- **Wartości**: Liczba zmiennoprzecinkowa (0 ≤ x ≤ 1)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.grid_distortion_prob_spin` (QDoubleSpinBox). Wczytywanie z profilu: NIE. Zapis: TAK.
- **Zakładka**: Augmentacja

### grid_distortion.distort_limit

- **Opis**: Limit zniekształcenia siatki
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.grid_distortion_limit_spin` (QDoubleSpinBox). Wczytywanie z profilu: NIE. Zapis: TAK.
- **Zakładka**: Augmentacja

## Parametry przetwarzania wstępnego (preprocessing)

### resize.enabled

- **Opis**: Czy włączyć zmianę rozmiaru obrazów
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Kontrolka UI (`self.resize_check`) istnieje w zakładce "Augmentacja". Ścieżka w konfiguracji to `augmentation.resize.enabled`. Brak wczytywania z profilu dla tej ścieżki. Zapis do zadania i profilu dla `augmentation.resize.enabled`: TAK. Wymagana implementacja dla `preprocessing.resize.enabled` w zakładce "Preprocessing" lub zmiana opisu.
- **Zakładka**: Preprocessing

### resize.size

- **Opis**: Docelowy rozmiar obrazów
- **Wartości**: [width, height] gdzie width, height to liczby całkowite
- **Kontrolka UI**: Podwójny spinner liczbowy
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania.
- **Zakładka**: Preprocessing

### resize.mode

- **Opis**: Tryb zmiany rozmiaru
- **Wartości**: "bilinear", "bicubic", "nearest", "lanczos"
- **Kontrolka UI**: Dropdown
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI. W `_on_accept` domyślnie `config["preprocessing"]["resize_mode"] = "bilinear"`. Brak wczytywania/zapisu z UI.
- **Zakładka**: Preprocessing

### normalize.enabled

- **Opis**: Czy włączyć normalizację
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania. W `_on_accept` domyślnie `config["preprocessing"]["normalization"] = "RGB"`, co implikuje włączenie.
- **Zakładka**: Preprocessing

### normalize.mean

- **Opis**: Średnie wartości dla normalizacji
- **Wartości**: [R, G, B] gdzie R, G, B to liczby zmiennoprzecinkowe
- **Kontrolka UI**: Trzy pola liczbowe
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania.
- **Zakładka**: Preprocessing

### normalize.std

- **Opis**: Odchylenia standardowe dla normalizacji
- **Wartości**: [R, G, B] gdzie R, G, B to liczby zmiennoprzecinkowe
- **Kontrolka UI**: Trzy pola liczbowe
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania.
- **Zakładka**: Preprocessing

### grayscale.enabled

- **Opis**: Czy konwertować do skali szarości
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania.
- **Zakładka**: Preprocessing

### grayscale.num_output_channels

- **Opis**: Liczba kanałów wyjściowych
- **Wartości**: 1 lub 3
- **Kontrolka UI**: Spinner liczbowy
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania.
- **Zakładka**: Preprocessing

### color_jitter.enabled

- **Opis**: Czy włączyć modyfikację kolorów
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania. (Parametry Color Jitter są w Augmentacji).
- **Zakładka**: Preprocessing

### color_jitter.brightness

- **Opis**: Zakres zmiany jasności
- **Wartości**: Liczba zmiennoprzecinkowa (≥0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania. (Parametry Color Jitter są w Augmentacji).
- **Zakładka**: Preprocessing

### color_jitter.contrast

- **Opis**: Zakres zmiany kontrastu
- **Wartości**: Liczba zmiennoprzecinkowa (≥0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania. (Parametry Color Jitter są w Augmentacji).
- **Zakładka**: Preprocessing

### color_jitter.saturation

- **Opis**: Zakres zmiany nasycenia
- **Wartości**: Liczba zmiennoprzecinkowa (≥0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania. (Parametry Color Jitter są w Augmentacji).
- **Zakładka**: Preprocessing

### color_jitter.hue

- **Opis**: Zakres zmiany odcienia
- **Wartości**: Liczba zmiennoprzecinkowa (≥0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania. (Parametry Color Jitter są w Augmentacji).
- **Zakładka**: Preprocessing

### gaussian_blur.enabled

- **Opis**: Czy włączyć rozmycie Gaussa
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania.
- **Zakładka**: Preprocessing

### gaussian_blur.kernel_size

- **Opis**: Rozmiar jądra rozmycia
- **Wartości**: Liczba nieparzysta (>0)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania.
- **Zakładka**: Preprocessing

### gaussian_blur.sigma

- **Opis**: Odchylenie standardowe rozmycia
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania.
- **Zakładka**: Preprocessing

### cache_dataset

- **Opis**: Czy cachować zestaw danych
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI. W `_on_accept` domyślnie `config["preprocessing"]["cache_dataset"] = False`. Brak wczytywania/zapisu z UI.
- **Zakładka**: Preprocessing (nie ma jawnej zakładki w kodzie, ale logicznie tu pasuje)

### scaling.method

- **Opis**: Metoda skalowania obrazów
- **Wartości**: "Bicubic", "Bilinear", "Nearest" (w kodzie: "Bilinear", "Bicubic", "Lanczos", "Nearest", "Area")
- **Kontrolka UI**: Dropdown
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Kontrolka UI (`self.scaling_method`) istnieje. Brak wczytywania z profilu. Brak zapisu do zadania/profilu (nie jest używana do ustawienia `config.preprocessing.resize_mode`).
- **Zakładka**: Preprocessing

### scaling.maintain_aspect_ratio

- **Opis**: Czy zachować proporcje obrazu
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Kontrolka UI (`self.maintain_aspect_ratio`) istnieje. Brak wczytywania z profilu. Brak zapisu do zadania/profilu.
- **Zakładka**: Preprocessing

### scaling.pad_to_square

- **Opis**: Czy dopełniać obraz do kwadratu
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Kontrolka UI (`self.pad_to_square`) istnieje. Brak wczytywania z profilu. Brak zapisu do zadania/profilu.
- **Zakładka**: Preprocessing

### scaling.pad_mode

- **Opis**: Typ dopełnienia
- **Wartości**: "reflection", "constant", "edge" (w kodzie: "constant", "edge", "reflect", "symmetric")
- **Kontrolka UI**: Dropdown
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Kontrolka UI (`self.pad_mode`) istnieje. Brak wczytywania z profilu. Brak zapisu do zadania/profilu.
- **Zakładka**: Preprocessing

### scaling.pad_value

- **Opis**: Wartość dopełnienia (dla "constant")
- **Wartości**: Liczba całkowita (0-255)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Kontrolka UI (`self.pad_value`) istnieje. Brak wczytywania z profilu. Brak zapisu do zadania/profilu.
- **Zakładka**: Preprocessing

### random_resize_crop.enabled

- **Opis**: Czy używać losowego przycinania przy zmianie rozmiaru
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania.
- **Zakładka**: Preprocessing

### random_resize_crop.size

- **Opis**: Docelowy rozmiar po przycięciu
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania.
- **Zakładka**: Preprocessing

### random_resize_crop.scale

- **Opis**: Zakres skali dla losowego przycinania
- **Wartości**: [min, max] gdzie min, max to liczby zmiennoprzecinkowe
- **Kontrolka UI**: Podwójny slider (range slider)
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania.
- **Zakładka**: Preprocessing

### random_resize_crop.ratio

- **Opis**: Zakres proporcji dla losowego przycinania
- **Wartości**: [min, max] gdzie min, max to liczby zmiennoprzecinkowe
- **Kontrolka UI**: Podwójny slider (range slider)
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania.
- **Zakładka**: Preprocessing

## Parametry monitorowania i logowania (monitoring)

### metrics.accuracy

- **Opis**: Czy obliczać dokładność
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.accuracy_check`. Wczytywanie: TAK. Zapis: TAK.
- **Zakładka**: Monitoring i Logging (w kodzie "Monitorowanie")

### metrics.precision

- **Opis**: Czy obliczać precyzję
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.precision_check`. Wczytywanie: TAK. Zapis: TAK.
- **Zakładka**: Monitoring i Logging

### metrics.recall

- **Opis**: Czy obliczać recall
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.recall_check`. Wczytywanie: TAK. Zapis: TAK.
- **Zakładka**: Monitoring i Logging

### metrics.f1

- **Opis**: Czy obliczać F1-score
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.f1_check`. Wczytywanie: TAK. Zapis: TAK.
- **Zakładka**: Monitoring i Logging

### metrics.topk

- **Opis**: Lista k dla top-k accuracy
- **Wartości**: Lista liczb całkowitych
- **Kontrolka UI**: Wielowybór lub pole tagów
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.topk_check` (Przełącznik). Wczytywanie obsługuje listę/bool. Zapis: TAK (jako bool).
- **Zakładka**: Monitoring i Logging

### metrics.confusion_matrix

- **Opis**: Czy generować macierz pomyłek
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.confusion_matrix_check`. Wczytywanie: TAK. Zapis: TAK.
- **Zakładka**: Monitoring i Logging

### metrics.auc

- **Opis**: Czy obliczać AUC-ROC
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania.
- **Zakładka**: Monitoring i Logging

### metrics.gpu_utilization

- **Opis**: Czy monitorować wykorzystanie GPU
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania.
- **Zakładka**: Monitoring i Logging

### metrics.memory_usage

- **Opis**: Czy monitorować zużycie pamięci
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania.
- **Zakładka**: Monitoring i Logging

### tensorboard.enabled

- **Opis**: Czy włączyć logowanie do TensorBoard
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Kontrolka UI (`self.use_tensorboard_check`) istnieje. Brak wczytywania z profilu. Brak zapisu do zadania/profilu.
- **Zakładka**: Monitoring i Logging

### tensorboard.log_dir

- **Opis**: Katalog do zapisywania logów TensorBoard
- **Wartości**: Ścieżka do katalogu
- **Kontrolka UI**: Pole tekstowe + przycisk wyboru katalogu
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Kontrolka UI (`self.tensorboard_dir_edit`) istnieje. Brak wczytywania z profilu. Brak zapisu do zadania/profilu.
- **Zakładka**: Monitoring i Logging

### tensorboard.update_freq

- **Opis**: Częstotliwość aktualizacji logów
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania.
- **Zakładka**: Monitoring i Logging

### wandb.enabled

- **Opis**: Czy włączyć logowanie do Weights & Biases
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania.
- **Zakładka**: Monitoring i Logging

### wandb.project

- **Opis**: Nazwa projektu w W&B
- **Wartości**: Nazwa projektu
- **Kontrolka UI**: Pole tekstowe
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania.
- **Zakładka**: Monitoring i Logging

### wandb.entity

- **Opis**: Nazwa użytkownika/zespołu w W&B
- **Wartości**: Nazwa użytkownika/zespołu
- **Kontrolka UI**: Pole tekstowe
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania.
- **Zakładka**: Monitoring i Logging

### wandb.tags

- **Opis**: Tagi dla eksperymentu w W&B
- **Wartości**: Lista tagów
- **Kontrolka UI**: Pole tekstowe z możliwością dodawania wielu tagów
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania.
- **Zakładka**: Monitoring i Logging

### checkpoint.enabled

- **Opis**: Czy włączyć zapisywanie checkpointów
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak dedykowanej kontrolki "enabled". Zapisywanie jest implikowane przez inne ustawienia checkpointów.
- **Zakładka**: Monitoring i Logging

### checkpoint.dir

- **Opis**: Katalog do zapisywania checkpointów
- **Wartości**: Ścieżka do katalogu
- **Kontrolka UI**: Pole tekstowe + przycisk wyboru katalogu
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Kontrolka UI (`self.model_dir_edit`) istnieje. Brak wczytywania z profilu. Brak zapisu do zadania/profilu dla tej konkretnej ścieżki (`monitoring.checkpoint.dir`).
- **Zakładka**: Monitoring i Logging

### checkpoint.save_best_only

- **Opis**: Czy zapisywać tylko najlepszy model
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.best_only_check`. Wczytywanie: TAK. Zapis: TAK.
- **Zakładka**: Monitoring i Logging

### checkpoint.monitor

- **Opis**: Metryka do monitorowania
- **Wartości**: Nazwa metryki
- **Kontrolka UI**: Dropdown
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.checkpoint_metric_combo`. Wczytywanie: TAK. Zapis: TAK.
- **Zakładka**: Monitoring i Logging

### checkpoint.mode

- **Opis**: Tryb monitorowania
- **Wartości**: "min" lub "max"
- **Kontrolka UI**: Dropdown
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania. Tryb jest implikowany przez metrykę (np. val_loss to "min").
- **Zakładka**: Monitoring i Logging

### checkpoint.save_freq

- **Opis**: Częstotliwość zapisywania checkpointów
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.save_freq_spin`. Wczytywanie: TAK. Zapis: TAK.
- **Zakładka**: Monitoring i Logging

### early_stopping.enabled

- **Opis**: Czy włączyć wczesne zatrzymywanie
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.use_early_stopping_check`. Wczytywanie: TAK. Zapis: TAK.
- **Zakładka**: Monitoring i Logging

### early_stopping.monitor

- **Opis**: Metryka do monitorowania
- **Wartości**: Nazwa metryki
- **Kontrolka UI**: Dropdown
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.monitor_combo`. Wczytywanie: TAK. Zapis: TAK.
- **Zakładka**: Monitoring i Logging

### early_stopping.mode

- **Opis**: Tryb monitorowania
- **Wartości**: "min" lub "max"
- **Kontrolka UI**: Dropdown
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania. Tryb jest implikowany przez metrykę.
- **Zakładka**: Monitoring i Logging

### early_stopping.patience

- **Opis**: Liczba epok bez poprawy przed zatrzymaniem
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.patience_spin`. Wczytywanie: TAK. Zapis: TAK.
- **Zakładka**: Monitoring i Logging

### early_stopping.min_delta

- **Opis**: Minimalna zmiana uznawana za poprawę
- **Wartości**: Liczba zmiennoprzecinkowa (≥0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ✅ Zaimplementowane
- **Do zrobienia**: Brak uwag. Kontrolka: `self.min_delta_spin` (QDoubleSpinBox). Wczytywanie: TAK. Zapis: TAK.
- **Zakładka**: Monitoring i Logging

### reduce_lr.enabled

- **Opis**: Czy włączyć redukcję learning rate
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania. (Parametry podobne są w zakładce "Zaawansowane" dla schedulera, ale nie pod tą ścieżką).
- **Zakładka**: Monitoring i Logging

### reduce_lr.monitor

- **Opis**: Metryka do monitorowania
- **Wartości**: Nazwa metryki
- **Kontrolka UI**: Dropdown
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania.
- **Zakładka**: Monitoring i Logging

### reduce_lr.mode

- **Opis**: Tryb monitorowania
- **Wartości**: "min" lub "max"
- **Kontrolka UI**: Dropdown
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania.
- **Zakładka**: Monitoring i Logging

### reduce_lr.factor

- **Opis**: Współczynnik redukcji learning rate
- **Wartości**: Liczba zmiennoprzecinkowa (0 < x < 1)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI (`self.scheduler_factor` istnieje w "Zaawansowane", ale inna ścieżka), brak wczytywania z profilu, brak zapisu do zadania.
- **Zakładka**: Monitoring i Logging

### reduce_lr.patience

- **Opis**: Liczba epok bez poprawy przed redukcją
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI (`self.scheduler_patience` istnieje w "Zaawansowane", ale inna ścieżka), brak wczytywania z profilu, brak zapisu do zadania.
- **Zakładka**: Monitoring i Logging

### reduce_lr.min_delta

- **Opis**: Minimalna zmiana uznawana za poprawę
- **Wartości**: Liczba zmiennoprzecinkowa (≥0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania.
- **Zakładka**: Monitoring i Logging

### reduce_lr.min_lr

- **Opis**: Minimalna wartość learning rate
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI (`self.min_lr` istnieje w "Zaawansowane", ale inna ścieżka), brak wczytywania z profilu, brak zapisu do zadania.
- **Zakładka**: Monitoring i Logging

## Parametry zaawansowane (advanced)

### seed

- **Opis**: Ziarno losowości
- **Wartości**: Liczba całkowita
- **Kontrolka UI**: Spinner liczbowy
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania.
- **Zakładka**: Zaawansowane

### deterministic

- **Opis**: Czy używać deterministycznych operacji
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania.
- **Zakładka**: Zaawansowane

### benchmark

- **Opis**: Czy włączyć benchmark CUDA
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Kontrolka UI (`self.parameter_rows["cudnn_benchmark"]["value_widget"]`) istnieje w zakładce "Optymalizacja treningu". Wczytywanie z `hardware_profile`. Zapis do `config.optimization.cudnn_benchmark`. Brak wczytywania/zapisu dla `config.advanced.benchmark`.
- **Zakładka**: Zaawansowane

### num_workers

- **Opis**: Liczba workerów do ładowania danych
- **Wartości**: Liczba całkowita (≥0)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Parametr `training.num_workers` jest obsługiwany (patrz wyżej). Jeśli to odrębny parametr `advanced.num_workers`, to brak implementacji. Prawdopodobnie duplikat w opisie.
- **Zakładka**: Zaawansowane

### pin_memory

- **Opis**: Czy używać pin memory
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Kontrolka UI (`self.parameter_rows["pin_memory"]["value_widget"]`) istnieje w zakładce "Optymalizacja treningu". Wczytywanie z `hardware_profile`. Zapis do `config.optimization.pin_memory`. Brak wczytywania/zapisu dla `config.advanced.pin_memory`.
- **Zakładka**: Zaawansowane

### prefetch_factor

- **Opis**: Liczba próbek do prefetchowania
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Kontrolka UI (`self.parameter_rows["prefetch_factor"]["value_widget"]`) istnieje w zakładce "Optymalizacja treningu". Wczytywanie z `hardware_profile`. Zapis do `config.optimization.dataloader.prefetch_factor`. Brak wczytywania/zapisu dla `config.advanced.prefetch_factor`.
- **Zakładka**: Zaawansowane

### persistent_workers

- **Opis**: Czy używać persistent workers
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Kontrolka UI (`self.parameter_rows["persistent_workers"]["value_widget"]`) istnieje w zakładce "Optymalizacja treningu". Wczytywanie z `hardware_profile`. Zapis do `config.optimization.dataloader.persistent_workers`. Brak wczytywania/zapisu dla `config.advanced.persistent_workers`.
- **Zakładka**: Zaawansowane

### gradient_clip_val

- **Opis**: Wartość przycinania gradientów
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Kontrolka UI (`self.grad_clip` - QDoubleSpinBox) istnieje w zakładce "Zaawansowane", ale nie jest podłączona do wczytywania/zapisu dla ścieżki `advanced.gradient_clip_val`. Parametr `regularization.gradient_clip` jest zaimplementowany.
- **Zakładka**: Zaawansowane

### gradient_clip_algorithm

- **Opis**: Algorytm przycinania gradientów
- **Wartości**: "norm", "value"
- **Kontrolka UI**: Dropdown
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania.
- **Zakładka**: Zaawansowane

### accumulate_grad_batches

- **Opis**: Liczba batchy do akumulacji gradientów
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy
- **Status**: 🔄 Do sprawdzenia
- **Do zrobienia**: Parametr `training.gradient_accumulation_steps` jest częściowo obsługiwany. Jeśli to odrębny parametr `advanced.accumulate_grad_batches`, to brak implementacji. Prawdopodobnie duplikat w opisie.
- **Zakładka**: Zaawansowane

### sync_batchnorm

- **Opis**: Czy synchronizować BatchNorm
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania.
- **Zakładka**: Zaawansowane

### precision

- **Opis**: Precyzja obliczeń
- **Wartości**: 16, 32, 64, "bf16", "mixed"
- **Kontrolka UI**: Dropdown
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania. Parametr `training.mixed_precision` jest obsługiwany.
- **Zakładka**: Zaawansowane

### amp_level

- **Opis**: Poziom automatycznej mieszanej precyzji
- **Wartości**: "O0", "O1", "O2", "O3"
- **Kontrolka UI**: Dropdown
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania.
- **Zakładka**: Zaawansowane

### gradient_clip_norm

- **Opis**: Maksymalna norma gradientów
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania.
- **Zakładka**: Zaawansowane

### gradient_clip_mode

- **Opis**: Tryb przycinania gradientów
- **Wartości**: "norm", "value", "agc"
- **Kontrolka UI**: Dropdown
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania.
- **Zakładka**: Zaawansowane

### gradient_clip_agc

- **Opis**: Czy używać Adaptive Gradient Clipping
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania.
- **Zakładka**: Zaawansowane

### gradient_clip_agc_clipping

- **Opis**: Wartość przycinania dla AGC
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania.
- **Zakładka**: Zaawansowane

### gradient_clip_agc_eps

- **Opis**: Epsilon dla AGC
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania.
- **Zakładka**: Zaawansowane

### gradient_clip_agc_eps_inside

- **Opis**: Epsilon wewnętrzny dla AGC
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania.
- **Zakładka**: Zaawansowane

### gradient_clip_agc_eps_outside

- **Opis**: Epsilon zewnętrzny dla AGC
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania.
- **Zakładka**: Zaawansowane

### gradient_clip_agc_eps_inside_scale

- **Opis**: Skala epsilon wewnętrznego dla AGC
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania.
- **Zakładka**: Zaawansowane

### gradient_clip_agc_eps_outside_scale

- **Opis**: Skala epsilon zewnętrznego dla AGC
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania.
- **Zakładka**: Zaawansowane

### gradient_clip_agc_eps_inside_scale_min

- **Opis**: Minimalna skala epsilon wewnętrznego dla AGC
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania.
- **Zakładka**: Zaawansowane

### gradient_clip_agc_eps_outside_scale_min

- **Opis**: Minimalna skala epsilon zewnętrznego dla AGC
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania.
- **Zakładka**: Zaawansowane

### gradient_clip_agc_eps_inside_scale_max

- **Opis**: Maksymalna skala epsilon wewnętrznego dla AGC
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania.
- **Zakładka**: Zaawansowane

### gradient_clip_agc_eps_outside_scale_max

- **Opis**: Maksymalna skala epsilon zewnętrznego dla AGC
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Slider liczbowy
- **Status**: ❌ Brak implementacji
- **Do zrobienia**: Brak kontrolki UI, brak wczytywania z profilu, brak zapisu do zadania.
- **Zakładka**: Zaawansowane
