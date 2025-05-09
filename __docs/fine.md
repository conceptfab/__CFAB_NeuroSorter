Zestawienie możliwych wartości parametrów do doszkalania modeli
Poniżej przedstawiam szczegółowe zestawienie poszczególnych parametrów i ich możliwych wartości dla procesu doszkalania modeli:
1. Parametry podstawowe
ParametrMożliwe wartościUwaginameDowolny stringZazwyczaj autogenerowany na podstawie architektury i czasutype"doszkalanie"Stała wartość dla zadań doszkalaniastatus"Nowy", "W kolejce", "W trakcie", "Zakończony", "Błąd"Stan zadaniapriority0-100Priorytet wykonania, gdzie wyższa liczba oznacza wyższy priorytet
2. Ścieżki danych
ParametrMożliwe wartościUwagitrain_dirŚcieżka do kataloguŚcieżka do katalogu z danymi treningowymidata_dirŚcieżka do kataloguZazwyczaj taka sama jak train_dirval_dirŚcieżka do kataloguŚcieżka do katalogu z danymi walidacyjnymi (opcjonalna)
3. Konfiguracja modelu
ParametrMożliwe wartościUwagimodel_pathŚcieżka do plikuŚcieżka do modelu, który ma być doszkalany (.pt, .pth)architecture"EfficientNet", "ResNet", "DenseNet", "MobileNet", "ConvNeXt"Architektura modeluvariantZależne od architektury, np. "EfficientNet-B0" do "EfficientNet-B7"Wariant modelu w ramach wybranej architekturyinput_size32-1024Rozmiar wejściowy obrazu, zazwyczaj wielokrotność 32num_classes2-1000Liczba klas wyjściowychpretrainedtrue, falseCzy używać wag pretrainedpretrained_weights"imagenet", "imagenet21k", "noisy-student"Źródło wag pretrainedfeature_extraction_onlytrue, falseCzy tylko warstwy wyciągające cechy będą trenowaneactivation"swish", "relu", "silu", "mish", "gelu"Funkcja aktywacjidropout_at_inferencetrue, falseCzy używać dropout podczas wnioskowaniaglobal_pool"avg", "max", "token", "none"Metoda poolingu globalnegolast_layer_activation"softmax", "sigmoid", "none"Aktywacja warstwy wyjściowej
4. Parametry treningu
ParametrMożliwe wartościUwagiepochs1-1000Liczba epok treningubatch_size1-1024Rozmiar wsadu (mini-batch)learning_rate0.000001-1.0Współczynnik uczeniaoptimizer"Adam", "AdamW", "SGD", "RMSprop"Algorytm optymalizacjischeduler"None", "StepLR", "ReduceLROnPlateau", "OneCycleLR", "CosineAnnealingLR", "CosineAnnealingWarmRestarts"Harmonogram zmiany współczynnika uczenianum_workers0-32Liczba wątków do ładowania danychwarmup_epochs0-100Liczba epok rozgrzewkimixed_precisiontrue, falseCzy używać mieszanej precyzji (fp16/bf16)warmup_lr_init0.000001-0.1Początkowy współczynnik uczenia podczas rozgrzewkigradient_accumulation_steps1-32Liczba kroków akumulacji gradientuvalidation_split0.1-0.5Proporcja danych użytych do walidacjievaluation_freq1-100Częstotliwość ewaluacji (co ile epok)use_ematrue, falseCzy używać wykładniczej średniej ruchomej parametrówema_decay0.9-0.9999Współczynnik zaniku EMA
5. Regularyzacja
ParametrMożliwe wartościUwagiweight_decay0.0-1.0Współczynnik regularyzacji L2gradient_clip0.0-10.0Maksymalna norma gradientulabel_smoothing0.0-0.5Współczynnik wygładzania etykietdrop_connect_rate0.0-0.5Współczynnik drop connectdropout_rate0.0-0.5Współczynnik dropoutmomentum0.0-1.0Współczynnik momentum dla optymalizatorówepsilon1e-8-1e-3Epsilon dla numerycznej stabilności
5.1. SWA (Stochastic Weight Averaging)
ParametrMożliwe wartościUwagiuse_swatrue, falseCzy używać stochastycznego uśredniania wagstart_epoch1-1000Epoka rozpoczęcia SWA
5.2. Stochastic Depth
ParametrMożliwe wartościUwagiuse_stochastic_depthtrue, falseCzy używać stochastycznej głębokościdrop_rate0.0-0.5Współczynnik porzucania warstwsurvival_probability0.5-1.0Prawdopodobieństwo przetrwania warstwy
5.3. Random Erase
ParametrMożliwe wartościUwagiuse_random_erasetrue, falseCzy używać losowego wymazywaniaprobability0.0-1.0Prawdopodobieństwo wymazaniamode"pixel", "block"Tryb wymazywania
6. Augmentacja
ParametrMożliwe wartościUwagicontrast0.0-1.0Zakres zmiany kontrastusaturation0.0-1.0Zakres zmiany nasyceniahue0.0-0.5Zakres zmiany odcieniashear0.0-1.0Zakres ścinania obrazuchannel_shift_range0.0-1.0Zakres przesunięcia kanałówresize_mode"bilinear", "bicubic", "nearest", "area"Tryb zmiany rozmiaru
6.1. Normalizacja
ParametrMożliwe wartościUwagimean[0.0-1.0, 0.0-1.0, 0.0-1.0]Średnie dla RGB do normalizacjistd[0.0-1.0, 0.0-1.0, 0.0-1.0]Odchylenia standardowe dla RGB
7. Monitorowanie
7.1. Metryki
ParametrMożliwe wartościUwagiaccuracytrue, falseCzy mierzyć dokładnośćprecisiontrue, falseCzy mierzyć precyzjęrecalltrue, falseCzy mierzyć czułośćf1true, falseCzy mierzyć wynik F1top_k_accuracytrue, falseCzy mierzyć top-k accuracyconfusion_matrixtrue, falseCzy generować macierz pomyłekauctrue, falseCzy mierzyć AUC (obszar pod krzywą ROC)
7.2. Logging
ParametrMożliwe wartościUwagiuse_tensorboardtrue, falseCzy używać TensorBoarduse_wandbtrue, falseCzy używać Weights & Biasessave_to_csvtrue, falseCzy zapisywać metryki do CSVlogging_freq"epoch", "batch"Częstotliwość logowania
7.3. Wizualizacja
ParametrMożliwe wartościUwagiuse_gradcamtrue, falseCzy generować mapy aktywacji Grad-CAMuse_feature_mapstrue, falseCzy wizualizować mapy cechuse_prediction_samplestrue, falseCzy pokazywać próbki predykcjinum_samples1-100Liczba próbek do wizualizacji
8. Dane
ParametrMożliwe wartościUwagiclass_weights"balanced", "none"Sposób ważenia klassampler"weighted_random", "uniform", "none"Sampler dla danychimage_channels1-4Liczba kanałów obrazucache_datasettrue, falseCzy cache'ować dataset w pamięci
9. Wnioskowanie
9.1. TTA (Test Time Augmentation)
ParametrMożliwe wartościUwagiuse_ttatrue, falseCzy używać augmentacji w czasie testunum_augmentations1-20Liczba augmentacji na obraz
9.2. Eksport i kwantyzacja
ParametrMożliwe wartościUwagiexport_onnxtrue, falseCzy eksportować model do formatu ONNXuse_quantizationtrue, falseCzy używać kwantyzacjiprecision"int8", "fp16", "bf16"Precyzja po kwantyzacji
10. Inne
ParametrMożliwe wartościUwagiseed0-999999Ziarno generatora liczb losowychdeterministictrue, falseCzy zapewnić deterministyczne działanie
Te parametry dają pełną kontrolę nad procesem doszkalania modeli, pozwalając na dostosowanie go do specyficznych potrzeb i wymagań danego zadania. Pamiętaj, że nie wszystkie parametry będą miały równy wpływ na wydajność modelu - kluczowe są zwykle współczynnik uczenia, rozmiar wsadu, regularyzacja i dobór optymalizatora.


