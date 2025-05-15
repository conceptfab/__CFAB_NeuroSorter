# Parametry konfiguracyjne fine-tuningu modelu

## Informacje ogólne

### type

- **Opis**: Typ konfiguracji
- **Wartości**: "fine_tuning"
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

- **Opis**: Informacja o wymaganych danych
- **Wartości**: Dowolny tekst
- **Kontrolka UI**: Obszar tekstowy

### hardware_required

- **Opis**: Wymagania sprzętowe
- **Wartości**: Dowolny tekst
- **Kontrolka UI**: Pole tekstowe

## Model

### architecture

- **Opis**: Architektura modelu
- **Wartości**: "EfficientNet"
- **Kontrolka UI**: Lista rozwijana

### variant

- **Opis**: Wariant architektury
- **Wartości**: "EfficientNet-B0", "EfficientNet-B1", "EfficientNet-B2", "EfficientNet-B3", itd.
- **Kontrolka UI**: Lista rozwijana

### input_size

- **Opis**: Rozmiar wejściowy obrazu
- **Wartości**: Liczba całkowita (np. 260)
- **Kontrolka UI**: Pole numeryczne

### num_classes

- **Opis**: Liczba klas do klasyfikacji
- **Wartości**: Liczba całkowita (np. 32)
- **Kontrolka UI**: Pole numeryczne

### pretrained

- **Opis**: Czy używać wstępnie wytrenowanych wag
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### pretrained_weights

- **Opis**: Źródło wstępnie wytrenowanych wag
- **Wartości**: "imagenet", inne źródła
- **Kontrolka UI**: Lista rozwijana

### feature_extraction_only

- **Opis**: Czy używać tylko ekstrakcji cech
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### activation

- **Opis**: Funkcja aktywacji
- **Wartości**: "swish", "relu", "leaky_relu", "sigmoid"
- **Kontrolka UI**: Lista rozwijana

### dropout_at_inference

- **Opis**: Czy używać dropout podczas wnioskowania
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### global_pool

- **Opis**: Metoda globalnego poolingu
- **Wartości**: "avg", "max", "concat"
- **Kontrolka UI**: Lista rozwijana

### last_layer_activation

- **Opis**: Aktywacja ostatniej warstwy
- **Wartości**: "softmax", "sigmoid", "none"
- **Kontrolka UI**: Lista rozwijana

## Trening

### epochs

- **Opis**: Liczba epok treningu
- **Wartości**: Liczba całkowita (np. 100)
- **Kontrolka UI**: Pole numeryczne

### batch_size

- **Opis**: Rozmiar partii danych
- **Wartości**: Liczba całkowita (np. 48)
- **Kontrolka UI**: Pole numeryczne

### learning_rate

- **Opis**: Współczynnik uczenia
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.0002)
- **Kontrolka UI**: Pole numeryczne

### optimizer

- **Opis**: Optymalizator
- **Wartości**: "AdamW", "Adam", "SGD", "RMSprop"
- **Kontrolka UI**: Lista rozwijana

### scheduler.type

- **Opis**: Typ schedulera
- **Wartości**: "CosineAnnealingWarmRestarts", "StepLR", "ReduceLROnPlateau", "OneCycleLR"
- **Kontrolka UI**: Lista rozwijana

### scheduler.T_0

- **Opis**: Parametr T_0 dla CosineAnnealingWarmRestarts
- **Wartości**: Liczba całkowita (np. 10)
- **Kontrolka UI**: Pole numeryczne

### scheduler.T_mult

- **Opis**: Parametr T_mult dla CosineAnnealingWarmRestarts
- **Wartości**: Liczba całkowita (np. 2)
- **Kontrolka UI**: Pole numeryczne

### scheduler.eta_min

- **Opis**: Minimalna wartość learning rate
- **Wartości**: Liczba zmiennoprzecinkowa (np. 1e-7)
- **Kontrolka UI**: Pole numeryczne

### num_workers

- **Opis**: Liczba wątków do ładowania danych
- **Wartości**: Liczba całkowita (np. 4)
- **Kontrolka UI**: Pole numeryczne

### warmup_epochs

- **Opis**: Liczba epok rozgrzewki
- **Wartości**: Liczba całkowita (np. 5)
- **Kontrolka UI**: Pole numeryczne

### warmup_lr_init

- **Opis**: Początkowy learning rate dla rozgrzewki
- **Wartości**: Liczba zmiennoprzecinkowa (np. 5e-7)
- **Kontrolka UI**: Pole numeryczne

### mixed_precision

- **Opis**: Czy używać mieszanej precyzji
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### gradient_accumulation_steps

- **Opis**: Liczba kroków do akumulacji gradientu
- **Wartości**: Liczba całkowita (np. 2)
- **Kontrolka UI**: Pole numeryczne

### gradient_clip

- **Opis**: Wartość przycinania gradientu
- **Wartości**: Liczba zmiennoprzecinkowa (np. 1.0)
- **Kontrolka UI**: Pole numeryczne

### evaluation_freq

- **Opis**: Częstotliwość ewaluacji
- **Wartości**: Liczba całkowita (np. 1)
- **Kontrolka UI**: Pole numeryczne

### use_ema

- **Opis**: Czy używać Exponential Moving Average
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### ema_decay

- **Opis**: Współczynnik zaniku EMA
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.9999)
- **Kontrolka UI**: Pole numeryczne

### unfreeze_strategy

- **Opis**: Strategia odmrażania warstw
- **Wartości**: "gradual_with_lr_scaling", "all_at_once", "none"
- **Kontrolka UI**: Lista rozwijana

### unfreeze_after_epochs

- **Opis**: Po ilu epokach odmrozić warstwy
- **Wartości**: Liczba całkowita (np. 5)
- **Kontrolka UI**: Pole numeryczne

### unfreeze_layers

- **Opis**: Liczba warstw do odmrożenia
- **Wartości**: Liczba całkowita (np. 20)
- **Kontrolka UI**: Pole numeryczne

### frozen_lr

- **Opis**: Learning rate dla zamrożonych warstw
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.0001)
- **Kontrolka UI**: Pole numeryczne

### unfrozen_lr

- **Opis**: Learning rate dla odmrożonych warstw
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.001)
- **Kontrolka UI**: Pole numeryczne

### layer_specific_lr

- **Opis**: Czy używać specyficznych learning rate dla warstw
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### lr_scaling_factor

- **Opis**: Współczynnik skalowania learning rate
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.1)
- **Kontrolka UI**: Pole numeryczne

### loss_function

- **Opis**: Funkcja straty
- **Wartości**: "focal_loss", "cross_entropy", "binary_cross_entropy"
- **Kontrolka UI**: Lista rozwijana

### focal_loss_gamma

- **Opis**: Parametr gamma dla focal loss
- **Wartości**: Liczba zmiennoprzecinkowa (np. 2.0)
- **Kontrolka UI**: Pole numeryczne

### focal_loss_alpha

- **Opis**: Parametr alpha dla focal loss
- **Wartości**: "auto", wartość zmiennoprzecinkowa lub lista wartości
- **Kontrolka UI**: Pole tekstowe/numeryczne

## Regularyzacja

### weight_decay

- **Opis**: Współczynnik regularyzacji wag
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.00015)
- **Kontrolka UI**: Pole numeryczne

### label_smoothing

- **Opis**: Współczynnik wygładzania etykiet
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.1)
- **Kontrolka UI**: Pole numeryczne

### dropout_rate

- **Opis**: Współczynnik dropout
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.3)
- **Kontrolka UI**: Pole numeryczne

### drop_connect_rate

- **Opis**: Współczynnik drop connect
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.2)
- **Kontrolka UI**: Pole numeryczne

### momentum

- **Opis**: Współczynnik momentum
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.9)
- **Kontrolka UI**: Pole numeryczne

### epsilon

- **Opis**: Epsilon do stabilności numerycznej
- **Wartości**: Liczba zmiennoprzecinkowa (np. 1e-6)
- **Kontrolka UI**: Pole numeryczne

### swa.use

- **Opis**: Czy używać Stochastic Weight Averaging
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### swa.start_epoch

- **Opis**: Epoka rozpoczęcia SWA
- **Wartości**: Liczba całkowita (np. 80)
- **Kontrolka UI**: Pole numeryczne

### stochastic_depth.use

- **Opis**: Czy używać stochastic depth
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### stochastic_depth.drop_rate

- **Opis**: Współczynnik drop rate dla stochastic depth
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.2)
- **Kontrolka UI**: Pole numeryczne

### stochastic_depth.survival_probability

- **Opis**: Prawdopodobieństwo przetrwania warstwy
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.8)
- **Kontrolka UI**: Pole numeryczne

### random_erase.use

- **Opis**: Czy używać random erase
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### random_erase.probability

- **Opis**: Prawdopodobieństwo random erase
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.25)
- **Kontrolka UI**: Pole numeryczne

### random_erase.mode

- **Opis**: Tryb random erase
- **Wartości**: "pixel", "block"
- **Kontrolka UI**: Lista rozwijana

## Augmentacja

### augmentation.image_size

- **Opis**: Rozmiar obrazu po augmentacji
- **Wartości**: Lista dwóch liczb całkowitych [szerokość, wysokość]
- **Kontrolka UI**: Dwa pola numeryczne

### basic.use

- **Opis**: Czy używać podstawowej augmentacji
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### basic.rotation

- **Opis**: Zakres rotacji (stopnie)
- **Wartości**: Liczba całkowita (np. 30)
- **Kontrolka UI**: Pole numeryczne

### basic.brightness

- **Opis**: Zakres zmiany jasności
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.3)
- **Kontrolka UI**: Pole numeryczne

### basic.shift

- **Opis**: Zakres przesunięcia
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.15)
- **Kontrolka UI**: Pole numeryczne

### basic.zoom

- **Opis**: Zakres zoomu
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.2)
- **Kontrolka UI**: Pole numeryczne

### basic.horizontal_flip

- **Opis**: Czy używać odbicia poziomego
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### basic.vertical_flip

- **Opis**: Czy używać odbicia pionowego
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### mixup.use

- **Opis**: Czy używać augmentacji mixup
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### mixup.alpha

- **Opis**: Parametr alpha dla mixup
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.3)
- **Kontrolka UI**: Pole numeryczne

### cutmix.use

- **Opis**: Czy używać augmentacji cutmix
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### cutmix.alpha

- **Opis**: Parametr alpha dla cutmix
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.4)
- **Kontrolka UI**: Pole numeryczne

### autoaugment.use

- **Opis**: Czy używać autoaugment
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### randaugment.use

- **Opis**: Czy używać randaugment
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### randaugment.n

- **Opis**: Liczba operacji do zastosowania
- **Wartości**: Liczba całkowita (np. 2)
- **Kontrolka UI**: Pole numeryczne

### randaugment.m

- **Opis**: Siła operacji
- **Wartości**: Liczba całkowita (np. 7)
- **Kontrolka UI**: Pole numeryczne

### advanced.contrast

- **Opis**: Zakres zmiany kontrastu
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.2)
- **Kontrolka UI**: Pole numeryczne

### advanced.saturation

- **Opis**: Zakres zmiany nasycenia
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.2)
- **Kontrolka UI**: Pole numeryczne

### advanced.hue

- **Opis**: Zakres zmiany odcienia
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.1)
- **Kontrolka UI**: Pole numeryczne

### advanced.shear

- **Opis**: Zakres shear
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.1)
- **Kontrolka UI**: Pole numeryczne

### advanced.channel_shift

- **Opis**: Zakres przesunięcia kanałów
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.0)
- **Kontrolka UI**: Pole numeryczne

## Preprocessing

### preprocessing.image_size

- **Opis**: Rozmiar obrazu do preprocessingu
- **Wartości**: Lista dwóch liczb całkowitych [szerokość, wysokość]
- **Kontrolka UI**: Dwa pola numeryczne

### normalization.mean

- **Opis**: Średnie dla normalizacji kanałów RGB
- **Wartości**: Lista trzech liczb zmiennoprzecinkowych
- **Kontrolka UI**: Trzy pola numeryczne

### normalization.std

- **Opis**: Odchylenia standardowe dla normalizacji kanałów RGB
- **Wartości**: Lista trzech liczb zmiennoprzecinkowych
- **Kontrolka UI**: Trzy pola numeryczne

### resize_mode

- **Opis**: Metoda zmiany rozmiaru
- **Wartości**: "bilinear", "bicubic", "nearest"
- **Kontrolka UI**: Lista rozwijana

### cache_dataset

- **Opis**: Czy buforować zbiór danych
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

## Monitoring

### metrics.accuracy

- **Opis**: Czy mierzyć dokładność
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### metrics.precision

- **Opis**: Czy mierzyć precyzję
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### metrics.recall

- **Opis**: Czy mierzyć recall
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### metrics.f1

- **Opis**: Czy mierzyć F1 score
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### metrics.topk

- **Opis**: Czy mierzyć top-k accuracy
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### metrics.confusion_matrix

- **Opis**: Czy generować macierz pomyłek
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### metrics.auc

- **Opis**: Czy mierzyć AUC
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### metrics.balanced_accuracy

- **Opis**: Czy mierzyć zbalansowaną dokładność
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### metrics.specificity

- **Opis**: Czy mierzyć specyficzność
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### metrics.kappa

- **Opis**: Czy mierzyć współczynnik kappa
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### metrics.per_class_metrics

- **Opis**: Czy mierzyć metryki per klasa
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### metrics.class_distribution_monitoring

- **Opis**: Czy monitorować rozkład klas
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### logging.use_tensorboard

- **Opis**: Czy używać TensorBoard do logowania
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### logging.use_wandb

- **Opis**: Czy używać Weights & Biases do logowania
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### logging.save_to_csv

- **Opis**: Czy zapisywać metryki do CSV
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### logging.logging_freq

- **Opis**: Częstotliwość logowania
- **Wartości**: "epoch", "batch", "step"
- **Kontrolka UI**: Lista rozwijana

### visualization.use_gradcam

- **Opis**: Czy używać GradCAM do wizualizacji
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### visualization.use_feature_maps

- **Opis**: Czy używać map cech do wizualizacji
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### visualization.use_pred_samples

- **Opis**: Czy wizualizować przykłady predykcji
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### visualization.num_samples

- **Opis**: Liczba przykładów do wizualizacji
- **Wartości**: Liczba całkowita (np. 10)
- **Kontrolka UI**: Pole numeryczne

### early_stopping.patience

- **Opis**: Cierpliwość dla early stopping
- **Wartości**: Liczba całkowita (np. 15)
- **Kontrolka UI**: Pole numeryczne

### early_stopping.min_delta

- **Opis**: Minimalna zmiana uznawana za poprawę
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.0005)
- **Kontrolka UI**: Pole numeryczne

### early_stopping.monitor

- **Opis**: Metrika do monitorowania
- **Wartości**: "val_loss", "val_accuracy", "val_balanced_accuracy"
- **Kontrolka UI**: Lista rozwijana

### checkpointing.best_only

- **Opis**: Czy zapisywać tylko najlepszy model
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### checkpointing.save_frequency

- **Opis**: Częstotliwość zapisywania modelu
- **Wartości**: Liczba całkowita (np. 1)
- **Kontrolka UI**: Pole numeryczne

### checkpointing.metric

- **Opis**: Metrika do monitorowania
- **Wartości**: "val_loss", "val_accuracy", "val_balanced_accuracy"
- **Kontrolka UI**: Lista rozwijana

## Zaawansowane

### seed

- **Opis**: Ziarno losowości
- **Wartości**: Liczba całkowita (np. 42)
- **Kontrolka UI**: Pole numeryczne

### deterministic

- **Opis**: Czy używać deterministycznych operacji
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### class_weights

- **Opis**: Wagi klas
- **Wartości**: "auto", "balanced", lista wartości
- **Kontrolka UI**: Lista rozwijana/pole tekstowe

### sampler

- **Opis**: Metoda próbkowania
- **Wartości**: "balanced_weighted_random", "random"
- **Kontrolka UI**: Lista rozwijana

### image_channels

- **Opis**: Liczba kanałów obrazu
- **Wartości**: Liczba całkowita (np. 3)
- **Kontrolka UI**: Pole numeryczne

### tta.use

- **Opis**: Czy używać Test Time Augmentation
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### tta.num_augmentations

- **Opis**: Liczba augmentacji dla TTA
- **Wartości**: Liczba całkowita (np. 3)
- **Kontrolka UI**: Pole numeryczne

### export_onnx

- **Opis**: Czy eksportować model do ONNX
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### quantization.use

- **Opis**: Czy kwantyzować model
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### quantization.precision

- **Opis**: Precyzja kwantyzacji
- **Wartości**: "int8", "fp16"
- **Kontrolka UI**: Lista rozwijana

### imbalanced_data_handling.enable

- **Opis**: Czy włączyć obsługę niezbalansowanych danych
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### imbalanced_data_handling.strategy

- **Opis**: Strategia obsługi niezbalansowanych danych
- **Wartości**: "oversampling", "undersampling", "hybrid"
- **Kontrolka UI**: Lista rozwijana

### imbalanced_data_handling.oversampling_ratio

- **Opis**: Współczynnik oversamplingu
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.8)
- **Kontrolka UI**: Pole numeryczne

### imbalanced_data_handling.undersampling_threshold

- **Opis**: Próg undersamplingu
- **Wartości**: Liczba całkowita (np. 500)
- **Kontrolka UI**: Pole numeryczne

### imbalanced_data_handling.dynamic_class_weights

- **Opis**: Czy używać dynamicznych wag klas
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### imbalanced_data_handling.focal_loss.use

- **Opis**: Czy używać focal loss dla niezbalansowanych danych
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### imbalanced_data_handling.focal_loss.gamma

- **Opis**: Parametr gamma dla focal loss
- **Wartości**: Liczba zmiennoprzecinkowa (np. 2.0)
- **Kontrolka UI**: Pole numeryczne

### imbalanced_data_handling.focal_loss.alpha

- **Opis**: Parametr alpha dla focal loss
- **Wartości**: "auto", wartości numeryczne
- **Kontrolka UI**: Lista rozwijana/pole tekstowe

### catastrophic_forgetting_prevention.enable

- **Opis**: Czy włączyć zapobieganie katastroficznemu zapominaniu
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### catastrophic_forgetting_prevention.preserve_original_classes

- **Opis**: Czy zachować oryginalne klasy
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### catastrophic_forgetting_prevention.rehearsal.use

- **Opis**: Czy używać rehearsal
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### catastrophic_forgetting_prevention.rehearsal.samples_per_class

- **Opis**: Liczba próbek na klasę
- **Wartości**: Liczba całkowita (np. 25)
- **Kontrolka UI**: Pole numeryczne

### catastrophic_forgetting_prevention.rehearsal.synthetic_samples

- **Opis**: Czy używać syntetycznych próbek
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### catastrophic_forgetting_prevention.knowledge_distillation.use

- **Opis**: Czy używać destylacji wiedzy
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### catastrophic_forgetting_prevention.knowledge_distillation.temperature

- **Opis**: Temperatura dla destylacji wiedzy
- **Wartości**: Liczba zmiennoprzecinkowa (np. 2.0)
- **Kontrolka UI**: Pole numeryczne

### catastrophic_forgetting_prevention.knowledge_distillation.alpha

- **Opis**: Waga dla destylacji wiedzy
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.4)
- **Kontrolka UI**: Pole numeryczne

### catastrophic_forgetting_prevention.ewc_regularization.use

- **Opis**: Czy używać regularyzacji EWC
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### catastrophic_forgetting_prevention.ewc_regularization.lambda

- **Opis**: Waga regularyzacji EWC
- **Wartości**: Liczba zmiennoprzecinkowa (np. 5000.0)
- **Kontrolka UI**: Pole numeryczne

### catastrophic_forgetting_prevention.ewc_regularization.fisher_sample_size

- **Opis**: Liczba próbek do obliczenia macierzy Fishera
- **Wartości**: Liczba całkowita (np. 200)
- **Kontrolka UI**: Pole numeryczne

### catastrophic_forgetting_prevention.ewc_regularization.adaptive_lambda

- **Opis**: Czy używać adaptacyjnego lambda
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### catastrophic_forgetting_prevention.layer_freezing.strategy

- **Opis**: Strategia zamrażania warstw
- **Wartości**: "gradual", "fixed", "none"
- **Kontrolka UI**: Lista rozwijana

### catastrophic_forgetting_prevention.layer_freezing.freeze_ratio

- **Opis**: Współczynnik zamrożenia warstw
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.7)
- **Kontrolka UI**: Pole numeryczne

### evaluation_on_original_classes.enable

- **Opis**: Czy włączyć ewaluację na oryginalnych klasach
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### evaluation_on_original_classes.frequency

- **Opis**: Częstotliwość ewaluacji
- **Wartości**: Liczba całkowita (np. 5)
- **Kontrolka UI**: Pole numeryczne

### evaluation_on_original_classes.report_metrics

- **Opis**: Czy raportować metryki
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### model_merging.enable

- **Opis**: Czy włączyć łączenie modeli
- **Wartości**: true, false
- **Kontrolka UI**: Przełącznik

### model_merging.method

- **Opis**: Metoda łączenia modeli
- **Wartości**: "fisher_weighted", "average", "max"
- **Kontrolka UI**: Lista rozwijana

### model_merging.interpolation_factor

- **Opis**: Współczynnik interpolacji
- **Wartości**: Liczba zmiennoprzecinkowa (np. 0.5)
- **Kontrolka UI**: Pole numeryczne
