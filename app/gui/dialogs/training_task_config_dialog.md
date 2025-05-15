# Parametry konfiguracyjne treningu modelu

## Informacje ogólne

### type

- **Opis**: Typ konfiguracji
- **Wartości**: "training"
- **Kontrolka UI**: Pole tekstowe (tylko do odczytu)

### info

- **Opis**: Nazwa profilu modelu
- **Wartości**: Dowolny tekst
- **Kontrolka UI**: Pole tekstowe

### description

- **Opis**: Opis profilu
- **Wartości**: Dowolny tekst
- **Kontrolka UI**: Obszar tekstowy

### data_required

- **Opis**: Wymagania dotyczące danych treningowych
- **Wartości**: Dowolny tekst
- **Kontrolka UI**: Obszar tekstowy

### hardware_required

- **Opis**: Wymagania sprzętowe
- **Wartości**: Dowolny tekst
- **Kontrolka UI**: Obszar tekstowy

## Architektura modelu (model)

### architecture

- **Opis**: Typ architektury
- **Wartości**: "EfficientNet"
- **Kontrolka UI**: Dropdown

### variant

- **Opis**: Wariant modelu
- **Wartości**: "EfficientNet-B0" do "EfficientNet-B7"
- **Kontrolka UI**: Dropdown

### input_size

- **Opis**: Rozmiar wejściowy obrazu (piksele)
- **Wartości**: Liczba całkowita (np. 260)
- **Kontrolka UI**: Spinner liczbowy

### num_classes

- **Opis**: Liczba klas do klasyfikacji
- **Wartości**: Liczba całkowita (np. 32, 40)
- **Kontrolka UI**: Spinner liczbowy

### pretrained

- **Opis**: Czy używać wstępnie wytrenowanych wag
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik (toggle)

### pretrained_weights

- **Opis**: Źródło wag pretrenowanych
- **Wartości**: "imagenet" lub inne
- **Kontrolka UI**: Dropdown

### feature_extraction_only

- **Opis**: Czy używać modelu tylko do ekstrakcji cech
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### activation

- **Opis**: Funkcja aktywacji w modelu
- **Wartości**: "swish", "relu", "sigmoid", itp.
- **Kontrolka UI**: Dropdown

### dropout_at_inference

- **Opis**: Czy używać dropoutu podczas inferencji
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### global_pool

- **Opis**: Typ global pooling
- **Wartości**: "avg", "max"
- **Kontrolka UI**: Dropdown

### last_layer_activation

- **Opis**: Aktywacja ostatniej warstwy
- **Wartości**: "softmax", "sigmoid", "none"
- **Kontrolka UI**: Dropdown

## Parametry treningu (training)

### epochs

- **Opis**: Liczba epok treningu
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy

### batch_size

- **Opis**: Rozmiar batcha
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy

### learning_rate

- **Opis**: Współczynnik uczenia
- **Wartości**: Liczba zmiennoprzecinkowa (0 < x < 1)
- **Kontrolka UI**: Slider liczbowy lub pole numeryczne

### optimizer

- **Opis**: Optymalizator
- **Wartości**: "AdamW", "Adam", "SGD", "RMSprop"
- **Kontrolka UI**: Dropdown

### scheduler.type

- **Opis**: Typ harmonogramu uczenia
- **Wartości**: "CosineAnnealingWarmRestarts", "StepLR", "OneCycleLR"
- **Kontrolka UI**: Dropdown

### scheduler.T_0

- **Opis**: Parametr T_0 dla CosineAnnealingWarmRestarts
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy

### scheduler.T_mult

- **Opis**: Parametr T_mult dla CosineAnnealingWarmRestarts
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy

### scheduler.eta_min

- **Opis**: Minimalna wartość learning rate
- **Wartości**: Liczba zmiennoprzecinkowa (bliska 0)
- **Kontrolka UI**: Pole numeryczne

### num_workers

- **Opis**: Liczba wątków do ładowania danych
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy

### warmup_epochs

- **Opis**: Liczba epok rozgrzewki (warmup)
- **Wartości**: Liczba całkowita (≥0)
- **Kontrolka UI**: Spinner liczbowy

### warmup_lr_init

- **Opis**: Początkowy learning rate dla rozgrzewki
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Pole numeryczne

### mixed_precision

- **Opis**: Czy używać mieszanej precyzji
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### gradient_accumulation_steps

- **Opis**: Liczba kroków akumulacji gradientu
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy

### gradient_clip

- **Opis**: Wartość przycinania gradientu
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Pole numeryczne

### evaluation_freq

- **Opis**: Częstotliwość ewaluacji (co ile epok)
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy

### use_ema

- **Opis**: Czy używać Exponential Moving Average
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### ema_decay

- **Opis**: Współczynnik EMA decay
- **Wartości**: Liczba zmiennoprzecinkowa (0 < x < 1)
- **Kontrolka UI**: Slider liczbowy

### freeze_base_model

- **Opis**: Czy zamrozić wagi bazowego modelu
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### unfreeze_layers

- **Opis**: Które warstwy odmrozić
- **Wartości**: "all", "last_n", lista warstw
- **Kontrolka UI**: Dropdown lub wielowybór

### unfreeze_strategy

- **Opis**: Strategia odmrażania warstw
- **Wartości**: "gradual", "all_at_once"
- **Kontrolka UI**: Dropdown

### unfreeze_after_epochs

- **Opis**: Po ilu epokach odmrozić warstwy
- **Wartości**: Liczba całkowita (≥0)
- **Kontrolka UI**: Spinner liczbowy

### frozen_lr

- **Opis**: Learning rate dla zamrożonych warstw
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Pole numeryczne

### unfrozen_lr

- **Opis**: Learning rate dla odmrożonych warstw
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Pole numeryczne

### validation_split

- **Opis**: Część danych do walidacji
- **Wartości**: Liczba zmiennoprzecinkowa (0 < x < 1)
- **Kontrolka UI**: Slider liczbowy

## Parametry regularyzacji (regularization)

### weight_decay

- **Opis**: Współczynnik weight decay
- **Wartości**: Liczba zmiennoprzecinkowa (≥0)
- **Kontrolka UI**: Pole numeryczne

### label_smoothing

- **Opis**: Współczynnik wygładzania etykiet
- **Wartości**: Liczba zmiennoprzecinkowa (0 ≤ x < 1)
- **Kontrolka UI**: Slider liczbowy

### dropout_rate

- **Opis**: Współczynnik dropoutu
- **Wartości**: Liczba zmiennoprzecinkowa (0 ≤ x < 1)
- **Kontrolka UI**: Slider liczbowy

### drop_connect_rate

- **Opis**: Współczynnik drop connect
- **Wartości**: Liczba zmiennoprzecinkowa (0 ≤ x < 1)
- **Kontrolka UI**: Slider liczbowy

### momentum

- **Opis**: Współczynnik momentum (dla SGD)
- **Wartości**: Liczba zmiennoprzecinkowa (0 < x < 1)
- **Kontrolka UI**: Slider liczbowy

### epsilon

- **Opis**: Epsilon dla optymalizatora
- **Wartości**: Liczba zmiennoprzecinkowa (>0, bliska 0)
- **Kontrolka UI**: Pole numeryczne

### stochastic_depth.use

- **Opis**: Czy używać stochastic depth
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### stochastic_depth.survival_probability

- **Opis**: Prawdopodobieństwo przetrwania dla stochastic depth
- **Wartości**: Liczba zmiennoprzecinkowa (0 < x ≤ 1)
- **Kontrolka UI**: Slider liczbowy

### swa.use

- **Opis**: Czy używać Stochastic Weight Averaging
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### swa.start_epoch

- **Opis**: Od której epoki rozpocząć SWA
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy

### swa.lr_swa

- **Opis**: Learning rate dla SWA
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Pole numeryczne

## Parametry augmentacji danych (augmentation)

### basic.use

- **Opis**: Czy używać podstawowych augmentacji
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### basic.rotation

- **Opis**: Maksymalny kąt rotacji (stopnie)
- **Wartości**: Liczba zmiennoprzecinkowa (≥0)
- **Kontrolka UI**: Slider liczbowy

### basic.brightness

- **Opis**: Zakres zmiany jasności
- **Wartości**: Liczba zmiennoprzecinkowa (≥0)
- **Kontrolka UI**: Slider liczbowy

### basic.contrast

- **Opis**: Zakres zmiany kontrastu
- **Wartości**: Liczba zmiennoprzecinkowa (≥0)
- **Kontrolka UI**: Slider liczbowy

### basic.saturation

- **Opis**: Zakres zmiany nasycenia
- **Wartości**: Liczba zmiennoprzecinkowa (≥0)
- **Kontrolka UI**: Slider liczbowy

### basic.hue

- **Opis**: Zakres zmiany odcienia
- **Wartości**: Liczba zmiennoprzecinkowa (≥0)
- **Kontrolka UI**: Slider liczbowy

### basic.shift

- **Opis**: Maksymalne przesunięcie (piksele lub %)
- **Wartości**: Liczba zmiennoprzecinkowa (≥0)
- **Kontrolka UI**: Slider liczbowy

### basic.zoom

- **Opis**: Zakres powiększenia/zmniejszenia
- **Wartości**: Liczba zmiennoprzecinkowa (≥0)
- **Kontrolka UI**: Slider liczbowy

### basic.horizontal_flip

- **Opis**: Czy stosować odbicia poziome
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### basic.vertical_flip

- **Opis**: Czy stosować odbicia pionowe
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### mixup.use

- **Opis**: Czy używać augmentacji Mixup
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### mixup.alpha

- **Opis**: Parametr alpha dla Mixup
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Slider liczbowy

### cutmix.use

- **Opis**: Czy używać augmentacji CutMix
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### cutmix.alpha

- **Opis**: Parametr alpha dla CutMix
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Slider liczbowy

### autoaugment.use

- **Opis**: Czy używać AutoAugment
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### autoaugment.policy

- **Opis**: Polityka AutoAugment
- **Wartości**: "imagenet", "cifar", "svhn"
- **Kontrolka UI**: Dropdown

### randaugment.use

- **Opis**: Czy używać RandAugment
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### randaugment.n

- **Opis**: Liczba operacji RandAugment
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy

### randaugment.m

- **Opis**: Intensywność operacji RandAugment
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy

### trivialaugment.use

- **Opis**: Czy używać TrivialAugment
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### random_erase.use

- **Opis**: Czy używać Random Erase
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### random_erase.probability

- **Opis**: Prawdopodobieństwo Random Erase
- **Wartości**: Liczba zmiennoprzecinkowa (0 ≤ x ≤ 1)
- **Kontrolka UI**: Slider liczbowy

### random_erase.scale

- **Opis**: Zakres skali dla Random Erase
- **Wartości**: [min, max] gdzie min, max to liczby zmiennoprzecinkowe
- **Kontrolka UI**: Podwójny slider (range slider)

### random_erase.ratio

- **Opis**: Zakres proporcji dla Random Erase
- **Wartości**: [min, max] gdzie min, max to liczby zmiennoprzecinkowe
- **Kontrolka UI**: Podwójny slider (range slider)

### grid_distortion.enabled

- **Opis**: Czy używać zniekształcenia siatki
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### grid_distortion.probability

- **Opis**: Prawdopodobieństwo zniekształcenia siatki
- **Wartości**: Liczba zmiennoprzecinkowa (0 ≤ x ≤ 1)
- **Kontrolka UI**: Slider liczbowy

### grid_distortion.distort_limit

- **Opis**: Limit zniekształcenia siatki
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Slider liczbowy

## Parametry przetwarzania wstępnego (preprocessing)

### normalization.mean

- **Opis**: Średnie wartości dla normalizacji (RGB)
- **Wartości**: [r, g, b] gdzie r,g,b to liczby zmiennoprzecinkowe
- **Kontrolka UI**: Trzy pola numeryczne

### normalization.std

- **Opis**: Odchylenia standardowe dla normalizacji (RGB)
- **Wartości**: [r, g, b] gdzie r,g,b to liczby zmiennoprzecinkowe
- **Kontrolka UI**: Trzy pola numeryczne

### resize_mode

- **Opis**: Metoda zmiany rozmiaru obrazów
- **Wartości**: "bicubic", "bilinear", "nearest"
- **Kontrolka UI**: Dropdown

### cache_dataset

- **Opis**: Czy cachować zestaw danych
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### scaling.method

- **Opis**: Metoda skalowania obrazów
- **Wartości**: "Bicubic", "Bilinear", "Nearest"
- **Kontrolka UI**: Dropdown

### scaling.maintain_aspect_ratio

- **Opis**: Czy zachować proporcje obrazu
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### scaling.pad_to_square

- **Opis**: Czy dopełniać obraz do kwadratu
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### scaling.pad_mode

- **Opis**: Typ dopełnienia
- **Wartości**: "reflection", "constant", "edge"
- **Kontrolka UI**: Dropdown

### scaling.pad_value

- **Opis**: Wartość dopełnienia (dla "constant")
- **Wartości**: Liczba całkowita (0-255)
- **Kontrolka UI**: Spinner liczbowy

### random_resize_crop.enabled

- **Opis**: Czy używać losowego przycinania przy zmianie rozmiaru
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### random_resize_crop.size

- **Opis**: Docelowy rozmiar po przycięciu
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy

### random_resize_crop.scale

- **Opis**: Zakres skali dla losowego przycinania
- **Wartości**: [min, max] gdzie min, max to liczby zmiennoprzecinkowe
- **Kontrolka UI**: Podwójny slider (range slider)

### random_resize_crop.ratio

- **Opis**: Zakres proporcji dla losowego przycinania
- **Wartości**: [min, max] gdzie min, max to liczby zmiennoprzecinkowe
- **Kontrolka UI**: Podwójny slider (range slider)

## Parametry monitorowania (monitoring)

### metrics.accuracy

- **Opis**: Czy obliczać dokładność
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### metrics.precision

- **Opis**: Czy obliczać precyzję
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### metrics.recall

- **Opis**: Czy obliczać recall
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### metrics.f1

- **Opis**: Czy obliczać F1-score
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### metrics.topk

- **Opis**: Lista k dla top-k accuracy
- **Wartości**: Lista liczb całkowitych
- **Kontrolka UI**: Wielowybór lub pole tagów

### metrics.confusion_matrix

- **Opis**: Czy generować macierz pomyłek
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### metrics.auc

- **Opis**: Czy obliczać AUC-ROC
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### metrics.gpu_utilization

- **Opis**: Czy monitorować wykorzystanie GPU
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### metrics.memory_usage

- **Opis**: Czy monitorować zużycie pamięci
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### logging.use_tensorboard

- **Opis**: Czy używać TensorBoard
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### logging.tensorboard_log_dir

- **Opis**: Katalog logów TensorBoard
- **Wartości**: Ścieżka
- **Kontrolka UI**: Pole tekstowe lub wybór katalogu

### logging.use_wandb

- **Opis**: Czy używać Weights & Biases
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### logging.save_to_csv

- **Opis**: Czy zapisywać metryki do CSV
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### logging.csv_log_path

- **Opis**: Ścieżka do pliku CSV z logami
- **Wartości**: Ścieżka
- **Kontrolka UI**: Pole tekstowe lub wybór pliku

### logging.logging_freq

- **Opis**: Częstotliwość logowania
- **Wartości**: "epoch", "batch", "step"
- **Kontrolka UI**: Dropdown

### visualization.use_gradcam

- **Opis**: Czy używać GradCAM do wizualizacji
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### visualization.use_feature_maps

- **Opis**: Czy wizualizować mapy cech
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### visualization.use_pred_samples

- **Opis**: Czy wizualizować przykłady predykcji
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### visualization.num_samples

- **Opis**: Liczba przykładów do wizualizacji
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy

### early_stopping.use

- **Opis**: Czy używać early stopping
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### early_stopping.monitor

- **Opis**: Metryka do monitorowania
- **Wartości**: "val_loss", "val_accuracy", itp.
- **Kontrolka UI**: Dropdown

### early_stopping.patience

- **Opis**: Cierpliwość (liczba epok)
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy

### early_stopping.min_delta

- **Opis**: Minimalna różnica uznawana za poprawę
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Pole numeryczne

### early_stopping.mode

- **Opis**: Tryb monitorowania
- **Wartości**: "min", "max"
- **Kontrolka UI**: Dropdown lub przełącznik

### early_stopping.enabled

- **Opis**: Czy early stopping jest włączone
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### checkpointing.use

- **Opis**: Czy używać checkpointów
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### checkpointing.save_dir

- **Opis**: Katalog zapisu checkpointów
- **Wartości**: Ścieżka
- **Kontrolka UI**: Pole tekstowe lub wybór katalogu

### checkpointing.filename

- **Opis**: Format nazwy pliku checkpointu
- **Wartości**: Ciąg znaków z formatowaniem
- **Kontrolka UI**: Pole tekstowe

### checkpointing.monitor

- **Opis**: Metryka do monitorowania
- **Wartości**: "val_loss", "val_accuracy", itp.
- **Kontrolka UI**: Dropdown

### checkpointing.save_best_only

- **Opis**: Czy zapisywać tylko najlepszy model
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### checkpointing.mode

- **Opis**: Tryb monitorowania
- **Wartości**: "min", "max"
- **Kontrolka UI**: Dropdown lub przełącznik

### checkpointing.save_frequency

- **Opis**: Częstotliwość zapisu checkpointów
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy

### checkpointing.top_k

- **Opis**: Liczba najlepszych modeli do zachowania
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy

### checkpointing.metric

- **Opis**: Metryka używana do rankingu modeli
- **Wartości**: "val_loss", "val_accuracy", itp.
- **Kontrolka UI**: Dropdown

### checkpointing.best_only

- **Opis**: Czy zachować tylko najlepszy model
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

## Parametry zaawansowane (advanced)

### seed

- **Opis**: Ziarno losowości
- **Wartości**: Liczba całkowita
- **Kontrolka UI**: Spinner liczbowy

### deterministic

- **Opis**: Czy używać deterministycznych algorytmów
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### class_weights

- **Opis**: Strategie ważenia klas
- **Wartości**: "balanced", "none", lub słownik wag
- **Kontrolka UI**: Dropdown lub edytor JSON

### sampler

- **Opis**: Sampler do ładowania danych
- **Wartości**: "weighted_random", "random", "sequential"
- **Kontrolka UI**: Dropdown

### image_channels

- **Opis**: Liczba kanałów obrazu
- **Wartości**: Liczba całkowita (zazwyczaj 1 lub 3)
- **Kontrolka UI**: Spinner liczbowy

### tta.use

- **Opis**: Czy używać Test Time Augmentation
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### tta.num_augmentations

- **Opis**: Liczba augmentacji podczas testów
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy

### cross_validation.use

- **Opis**: Czy używać walidacji krzyżowej
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### cross_validation.k_folds

- **Opis**: Liczba foldów walidacji krzyżowej
- **Wartości**: Liczba całkowita (>1)
- **Kontrolka UI**: Spinner liczbowy

### distributed.use

- **Opis**: Czy używać treningu rozproszonego
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### distributed.backend

- **Opis**: Backend dla treningu rozproszonego
- **Wartości**: "nccl", "gloo"
- **Kontrolka UI**: Dropdown

### distributed.world_size

- **Opis**: Liczba procesów/urządzeń
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy

### distributed.rank

- **Opis**: Ranga procesu
- **Wartości**: Liczba całkowita (≥0)
- **Kontrolka UI**: Spinner liczbowy

### export_onnx

- **Opis**: Czy eksportować model do ONNX
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### quantization.use

- **Opis**: Czy używać kwantyzacji
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### quantization.precision

- **Opis**: Precyzja kwantyzacji
- **Wartości**: "int8", "float16"
- **Kontrolka UI**: Dropdown

### catastrophic_forgetting_prevention.enable

- **Opis**: Czy włączyć zapobieganie katastroficznemu zapominaniu
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### catastrophic_forgetting_prevention.preserve_original_classes

- **Opis**: Czy zachować oryginalne klasy
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### catastrophic_forgetting_prevention.rehearsal.use

- **Opis**: Czy używać techniki rehearsal
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### catastrophic_forgetting_prevention.rehearsal.samples_per_class

- **Opis**: Liczba przykładów na klasę do zapamiętania
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy

### catastrophic_forgetting_prevention.rehearsal.synthetic_samples

- **Opis**: Czy używać syntetycznych przykładów
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### catastrophic_forgetting_prevention.knowledge_distillation.use

- **Opis**: Czy używać destylacji wiedzy
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### catastrophic_forgetting_prevention.knowledge_distillation.temperature

- **Opis**: Temperatura dla destylacji wiedzy
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Slider liczbowy

### catastrophic_forgetting_prevention.knowledge_distillation.alpha

- **Opis**: Waga dla destylacji wiedzy
- **Wartości**: Liczba zmiennoprzecinkowa (0 ≤ x ≤ 1)
- **Kontrolka UI**: Slider liczbowy

### catastrophic_forgetting_prevention.ewc_regularization.use

- **Opis**: Czy używać EWC regularyzacji
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### catastrophic_forgetting_prevention.ewc_regularization.lambda

- **Opis**: Współczynnik lambda dla EWC
- **Wartości**: Liczba zmiennoprzecinkowa (>0)
- **Kontrolka UI**: Pole numeryczne

### catastrophic_forgetting_prevention.ewc_regularization.fisher_sample_size

- **Opis**: Rozmiar próbki do estymacji macierzy Fishera
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy

### catastrophic_forgetting_prevention.ewc_regularization.adaptive_lambda

- **Opis**: Czy używać adaptacyjnej wartości lambda
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### catastrophic_forgetting_prevention.layer_freezing.strategy

- **Opis**: Strategia zamrażania warstw
- **Wartości**: "gradual", "selective"
- **Kontrolka UI**: Dropdown

### catastrophic_forgetting_prevention.layer_freezing.freeze_ratio

- **Opis**: Stosunek warstw do zamrożenia
- **Wartości**: Liczba zmiennoprzecinkowa (0 ≤ x ≤ 1)
- **Kontrolka UI**: Slider liczbowy

## Parametry optymalizacji (optimization)

### batch_size

- **Opis**: Rozmiar batcha
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy

### num_workers

- **Opis**: Liczba wątków do ładowania danych
- **Wartości**: Liczba całkowita (≥0)
- **Kontrolka UI**: Spinner liczbowy

### use_mixed_precision

- **Opis**: Czy używać mieszanej precyzji
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### memory_efficient

- **Opis**: Czy optymalizować zużycie pamięci
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### cudnn_benchmark

- **Opis**: Czy włączyć benchmarking cuDNN
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### pin_memory

- **Opis**: Czy przypinać pamięć (dla GPU)
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### dataloader.shuffle

- **Opis**: Czy mieszać dane
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### dataloader.prefetch_factor

- **Opis**: Współczynnik prefetch dla dataloaderów
- **Wartości**: Liczba całkowita (>0)
- **Kontrolka UI**: Spinner liczbowy

### dataloader.persistent_workers

- **Opis**: Czy używać trwałych wątków roboczych
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### dataloader.drop_last

- **Opis**: Czy pomijać ostatni (niepełny) batch
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik

### dataloader.pin_memory

- **Opis**: Czy przypinać pamięć w dataloader
- **Wartości**: true/false
- **Kontrolka UI**: Przełącznik
