Zmiany w pliku app/gui/dialogs/fine_tuning_task_config_dialog.py
1. Problem z wczytywaniem parametrów konfiguracyjnych
pythondef _select_model_file(self):
    # ...
    # Wczytaj plik konfiguracyjny
    config_path = os.path.splitext(file_path)[0] + "_config.json"
    self.logger.info(f"Próba wczytania pliku konfiguracyjnego: {config_path}")

    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                self.logger.info(f"Wczytana konfiguracja: {json.dumps(config, indent=2)}")

                # Ustaw nazwę zadania na podstawie klucza "name" z konfiguracji
                if "name" in config:
                    task_name = f"{config['name']}_FT"
                    self.name_edit.setText(task_name)
                    self.logger.info(f"Ustawiono nazwę zadania: {task_name}")

                # Użyj metody _load_config do załadowania całej konfiguracji
                self._load_config(config)
                self.logger.info("Zastosowano konfigurację modelu")
Błąd polega na tym, że metoda _update_dependent_controls() nie jest wywoływana po załadowaniu konfiguracji. Powinno być:
pythondef _select_model_file(self):
    # ...
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                # ...

                # Użyj metody _load_config do załadowania całej konfiguracji
                self._load_config(config)
                
                # Dodaj tę linię, aby zaktualizować kontrolki zależne
                self._update_dependent_controls()
                
                self.logger.info("Zastosowano konfigurację modelu")
2. Problem z inicjalizacją kontrolek
Niektóre kontrolki mogą nie być dostępne w momencie wczytywania konfiguracji. Należy upewnić się, że wszystkie kontrolki są poprawnie inicjalizowane przed wczytaniem konfiguracji.
3. Problem z aktualizacją wariantów w zależności od architektury
pythondef _update_variant_combo(self, architecture: str) -> None:
    """Aktualizuje listę wariantów dla wybranej architektury."""
    self.logger.info(f"Aktualizacja wariantów dla architektury: {architecture}")

    # Zapisz aktualnie wybrany wariant
    current_variant = self.variant_combo.currentText()
    self.logger.info(f"Aktualnie wybrany wariant: {current_variant}")

    # Wyczyść i dodaj nowe warianty
    self.variant_combo.clear()

    if architecture == "EfficientNet":
        variants = ["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8"]
    elif architecture == "ResNet":
        variants = ["18", "34", "50", "101", "152"]
    elif architecture == "MobileNet":
        variants = ["v2", "v3_small", "v3_large"]
    else:
        variants = ["default"]

    self.logger.info(f"Dostępne warianty: {variants}")
    self.variant_combo.addItems(variants)

    # Próbuj przywrócić poprzednio wybrany wariant
    if current_variant in variants:
        self.variant_combo.setCurrentText(current_variant)
        self.logger.info(f"Przywracam poprzednio wybrany wariant: {current_variant}")
    else:
        self.logger.info("Nie można przywrócić poprzedniego wariantu, ustawiam domyślny")
Metoda ta powinna zostać wywołana po zmianie architektury podczas wczytywania konfiguracji, ale może to nie działać poprawnie.
4. Problem z synchronizacją kontrolek
pythondef _load_config(self, config: Dict[str, Any]) -> None:
    """Ładuje konfigurację do interfejsu."""
    try:
        # ...

        # 1. Aktualizacja parametrów modelu
        if "architecture" in model_config:
            self.arch_combo.setCurrentText(model_config["architecture"])
            # Po zmianie architektury aktualizujemy dostępne warianty
            self._update_variant_combo(model_config["architecture"])
            self._on_architecture_changed(model_config["architecture"])

        if "variant" in model_config:
            self.variant_combo.setCurrentText(model_config["variant"])
W tym miejscu problem polega na tym, że setCurrentText dla variant_combo jest wywoływany po zmianie architektury, co może powodować, że wariant zostanie nadpisany przez domyślny dla nowej architektury.
5. Brak aktualizacji kontrolek po wczytaniu konfiguracji
Na końcu metody _load_config brakuje wywołania metody, która zaktualizowałaby stan kontrolek na podstawie wczytanych wartości:
pythondef _load_config(self, config: Dict[str, Any]) -> None:
    # ...
    
    # Dodaj na końcu metody:
    self._update_ui_state()
Poprawka
Oto kompletna poprawka dla metody _load_config:
pythondef _load_config(self, config: Dict[str, Any]) -> None:
    """Ładuje konfigurację do interfejsu."""
    try:
        # Model
        model_config = config.get("model", {})

        # Logowanie wartości przed ustawieniem
        self.logger.info(f"Ładowanie konfiguracji - Nazwa zadania: {self.name_edit.text()}")
        self.logger.info(f"Ładowanie konfiguracji - Liczba klas: {self.num_classes_spin.value()}")

        # Ustawienie wartości z konfiguracji
        if "name" in config:
            self.name_edit.setText(config["name"])

        # 1. Aktualizacja parametrów modelu
        if "architecture" in model_config:
            # Najpierw ustawiamy architekturę
            self.arch_combo.setCurrentText(model_config["architecture"])
            # To wywołanie zaktualizuje listę wariantów
            self._update_variant_combo(model_config["architecture"])

        # Teraz ustawiamy wariant, po aktualizacji listy wariantów
        if "variant" in model_config:
            variant = model_config["variant"]
            # Jeśli wariant jest dostępny w aktualnej liście, ustaw go
            idx = self.variant_combo.findText(variant)
            if idx >= 0:
                self.variant_combo.setCurrentIndex(idx)
                self.logger.info(f"Ustawiono wariant: {variant}")
            else:
                self.logger.warning(f"Wariant {variant} nie jest dostępny dla architektury {model_config.get('architecture')}")

        # Pozostałe parametry modelu
        if "input_size" in model_config:
            self.input_size_spin.setValue(model_config["input_size"])

        if "num_classes" in model_config:
            self.num_classes_spin.setValue(model_config["num_classes"])

        if "pretrained" in model_config:
            self.pretrained_check.setChecked(model_config["pretrained"])

        if "pretrained_weights" in model_config:
            self.pretrained_weights_combo.setCurrentText(model_config["pretrained_weights"])

        if "feature_extraction_only" in model_config:
            self.feature_extraction_check.setChecked(model_config["feature_extraction_only"])

        if "activation" in model_config:
            self.activation_combo.setCurrentText(model_config["activation"])

        if "dropout_at_inference" in model_config:
            self.dropout_at_inference_check.setChecked(model_config["dropout_at_inference"])

        if "global_pool" in model_config:
            self.global_pool_combo.setCurrentText(model_config["global_pool"])

        if "last_layer_activation" in model_config:
            self.last_layer_activation_combo.setCurrentText(model_config["last_layer_activation"])
        
        # ... (pozostała część metody bez zmian)
        
        # Na koniec aktualizujemy stan UI
        self._update_ui_state()
        self.logger.info("Konfiguracja modelu została pomyślnie załadowana")

    except Exception as e:
        msg = "Błąd podczas ładowania konfiguracji"
        self.logger.error(f"{msg}: {str(e)}", exc_info=True)
        QtWidgets.QMessageBox.critical(self, "Błąd", f"{msg}: {str(e)}")
Dodatkowo należy zmodyfikować metodę _select_model_file:
pythondef _select_model_file(self):
    """Wybiera plik modelu do doszkalania."""
    try:
        # ... (bez zmian)
        
        if file_path:
            self.model_path_edit.setText(file_path)
            self.logger.info(f"Wybrano plik modelu: {file_path}")

            # Wczytaj plik konfiguracyjny
            config_path = os.path.splitext(file_path)[0] + "_config.json"
            self.logger.info(f"Próba wczytania pliku konfiguracyjnego: {config_path}")

            if os.path.exists(config_path):
                try:
                    with open(config_path, "r") as f:
                        config = json.load(f)
                        self.logger.info(f"Wczytana konfiguracja: {json.dumps(config, indent=2)}")

                        # Ustaw nazwę zadania na podstawie klucza "name" z konfiguracji
                        if "name" in config:
                            task_name = f"{config['name']}_FT"
                            self.name_edit.setText(task_name)
                            self.logger.info(f"Ustawiono nazwę zadania: {task_name}")

                        # Użyj metody _load_config do załadowania całej konfiguracji
                        self._load_config(config)
                        
                        # Aktualizacja zależnych kontrolek
                        self._update_dependent_controls()
                        
                        self.logger.info("Zastosowano konfigurację modelu")

                except Exception as e:
                    self.logger.error(f"Błąd podczas wczytywania konfiguracji: {str(e)}")
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Błąd",
                        f"Nie udało się wczytać konfiguracji modelu: {str(e)}",
                    )
            else:
                # ... (bez zmian)
    except Exception as e:
        # ... (bez zmian)
Podsumowanie zmian

W metodzie _load_config:

Poprawne ustawienie architektury i wariantu
Dodanie wywołania _update_ui_state() na końcu metody


W metodzie _select_model_file:

Dodanie wywołania _update_dependent_controls() po wczytaniu konfiguracji



Te zmiany powinny rozwiązać problem z wczytywaniem konfiguracji z pliku i prawidłowym aktualizowaniem kontrolek UI na podstawie wczytanych wartości.Retryconceptfab/__CFAB_NeuroSorterrefactor_to-0.4GITHUBMnadal nie dziala!Editpython PyQt6 double initialization of widgets10 resultsPyQt Layouts: Create Professional-Looking GUI Applications – Real Pythonrealpython.comPyQt6 Widgets — QCheckBox, QComboBox, QPushButton, QLabel, QSliderpythonguis.comPySide6.QtWidgets.QWidget - Qt for Pythonqt.ioPyQt6 Layouts - Using QVBoxLayout, QHBoxLayout, QGridLayout, QStackedLayout to Position Widgetspythonguis.comLayout Management in PyQt6 - CodersLegacycoderslegacy.comWorking with Python's PyQT Frameworkstackabuse.comCreate custom GUI Widgets for your Python apps with PyQt6pythonguis.comPySide6.QtWidgets.QListWidget - Qt for Pythonqt.iopython - PyQt6 confused about nesting layouts or widgets - Stack Overflowstackoverflow.comQWidget — Qt for Pythonqt.ioPo przeanalizowaniu kodu i znalezionych informacji, zidentyfikowałem kilka problemów, które mogą powodować, że wczytywanie konfiguracji z pliku nie działa prawidłowo. Oto moje propozycje poprawek:
Zmiany w pliku app/gui/dialogs/fine_tuning_task_config_dialog.py
1. Problem z duplikacją kontrolek dla normalizacji
Znalazłem problem z powielonymi deklaracjami tych samych kontrolek. Kontrolki dla wartości normalizacji norm_mean_r, norm_mean_g, norm_mean_b, norm_std_r, norm_std_g, norm_std_b są zdefiniowane zarówno w _create_augmentation_tab() jak i w _create_preprocessing_tab().
PyQt6 nie pozwala na podwójną inicjalizację tych samych widgetów - ten sam widget nie może być dodany do dwóch różnych układów (layouts).
python# Poprawka: Zmodyfikuj metodę _create_preprocessing_tab(), aby używała istniejących kontrolek

def _create_preprocessing_tab(self) -> QtWidgets.QWidget:
    """Tworzy zakładkę z parametrami preprocessingu."""
    tab = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout()
    form = QtWidgets.QFormLayout()

    # Normalizacja
    norm_group = QtWidgets.QGroupBox("Normalizacja")
    norm_layout = QtWidgets.QFormLayout()

    # Mean
    mean_layout = QtWidgets.QHBoxLayout()
    # Używamy już istniejących kontrolek zamiast tworzenia nowych
    mean_layout.addWidget(self.norm_mean_r)
    mean_layout.addWidget(self.norm_mean_g)
    mean_layout.addWidget(self.norm_mean_b)

    norm_layout.addRow("Mean (RGB):", mean_layout)

    # Std
    std_layout = QtWidgets.QHBoxLayout()
    # Używamy już istniejących kontrolek zamiast tworzenia nowych
    std_layout.addWidget(self.norm_std_r)
    std_layout.addWidget(self.norm_std_g)
    std_layout.addWidget(self.norm_std_b)

    norm_layout.addRow("Std (RGB):", std_layout)

    norm_group.setLayout(norm_layout)
    form.addRow(norm_group)

    # Resize mode
    # Tutaj rezyduje inne pole combo, które nie jest zduplikowane
    form.addRow("Resize mode:", self.resize_mode_combo)

    # Cache dataset
    form.addRow("Cache dataset:", self.cache_dataset_check)

    layout.addLayout(form)
    tab.setLayout(layout)
    return tab
2. Problem z metodą _load_config
pythondef _load_config(self, config: Dict[str, Any]) -> None:
    """Ładuje konfigurację do interfejsu."""
    try:
        # Blokujemy sygnały podczas wczytywania konfiguracji, aby uniknąć 
        # wyzwalania zbędnych aktualizacji UI
        self.blockSignals(True)
        
        # Model
        model_config = config.get("model", {})

        # Logowanie wartości przed ustawieniem
        self.logger.info(f"Ładowanie konfiguracji - Nazwa zadania: {self.name_edit.text()}")
        self.logger.info(f"Ładowanie konfiguracji - Liczba klas: {self.num_classes_spin.value()}")

        # Ustawienie wartości z konfiguracji
        if "name" in config:
            self.name_edit.setText(config["name"])

        # 1. Aktualizacja parametrów modelu
        if "architecture" in model_config:
            # Najpierw ustawiamy architekturę
            architecture = model_config["architecture"]
            idx = self.arch_combo.findText(architecture)
            if idx >= 0:
                self.arch_combo.blockSignals(True)
                self.arch_combo.setCurrentIndex(idx)
                self.arch_combo.blockSignals(False)
                # To wywołanie zaktualizuje listę wariantów
                self._update_variant_combo(architecture)
            else:
                self.logger.warning(f"Architektura {architecture} nie jest dostępna")

        # Teraz ustawiamy wariant, po aktualizacji listy wariantów
        if "variant" in model_config:
            variant = model_config["variant"]
            # Jeśli wariant jest dostępny w aktualnej liście, ustaw go
            idx = self.variant_combo.findText(variant)
            if idx >= 0:
                self.variant_combo.blockSignals(True)
                self.variant_combo.setCurrentIndex(idx)
                self.variant_combo.blockSignals(False)
                self.logger.info(f"Ustawiono wariant: {variant}")
            else:
                self.logger.warning(f"Wariant {variant} nie jest dostępny dla architektury {model_config.get('architecture')}")

        # Reszta metody pozostaje bez zmian
        
        # ...
        
        # Na końcu metody odblokujemy sygnały i ręcznie wywołamy aktualizację UI
        self.blockSignals(False)
        self._update_ui_state()
        self.logger.info("Konfiguracja modelu została pomyślnie załadowana")

    except Exception as e:
        self.blockSignals(False)  # Upewnij się, że sygnały zostaną odblokowane nawet w przypadku błędu
        msg = "Błąd podczas ładowania konfiguracji"
        self.logger.error(f"{msg}: {str(e)}", exc_info=True)
        QtWidgets.QMessageBox.critical(self, "Błąd", f"{msg}: {str(e)}")
3. Problem z metodą _select_model_file
pythondef _select_model_file(self):
    """Wybiera plik modelu do doszkalania."""
    try:
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Wybierz plik modelu do doszkalania",
            str(Path("data/models")),
            "Pliki modeli (*.pth *.pt *.ckpt);;Wszystkie pliki (*.*)",
        )
        if file_path:
            self.model_path_edit.setText(file_path)
            self.logger.info(f"Wybrano plik modelu: {file_path}")

            # Wczytaj plik konfiguracyjny
            config_path = os.path.splitext(file_path)[0] + "_config.json"
            self.logger.info(f"Próba wczytania pliku konfiguracyjnego: {config_path}")

            if os.path.exists(config_path):
                try:
                    with open(config_path, "r") as f:
                        config = json.load(f)
                        self.logger.info(f"Wczytana konfiguracja: {json.dumps(config, indent=2)}")

                        # Ustaw nazwę zadania na podstawie klucza "name" z konfiguracji
                        if "name" in config:
                            task_name = f"{config['name']}_FT"
                            self.name_edit.setText(task_name)
                            self.logger.info(f"Ustawiono nazwę zadania: {task_name}")

                        # Najpierw dodaj blokowanie sygnałów na całym oknie
                        self.blockSignals(True)
                        
                        # Użyj metody _load_config do załadowania całej konfiguracji
                        # Ta metoda już ma blockSignals wewnątrz
                        self._load_config(config)
                        
                        # Ręcznie zaktualizuj kontrolki zależne
                        self.blockSignals(False)
                        self._update_dependent_controls()
                        self._update_ui_state()
                        
                        self.logger.info("Zastosowano konfigurację modelu")

                except Exception as e:
                    # Upewnij się, że sygnały zostaną odblokowane w przypadku błędu
                    self.blockSignals(False)
                    self.logger.error(f"Błąd podczas wczytywania konfiguracji: {str(e)}")
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Błąd",
                        f"Nie udało się wczytać konfiguracji modelu: {str(e)}",
                    )
            else:
                self.logger.warning(f"Nie znaleziono pliku konfiguracyjnego: {config_path}")
                QtWidgets.QMessageBox.warning(
                    self,
                    "Ostrzeżenie",
                    "Nie znaleziono pliku konfiguracyjnego dla wybranego modelu.",
                )

    except Exception as e:
        # Upewnij się, że sygnały zostaną odblokowane w przypadku błędu
        self.blockSignals(False)
        self.logger.error(f"Błąd podczas wyboru pliku modelu: {str(e)}")
        QtWidgets.QMessageBox.critical(
            self,
            "Błąd",
            f"Wystąpił błąd podczas wyboru pliku modelu: {str(e)}",
        )
4. Inicjalizacja kontrolek w konstruktorze
Wszystkie kontrolki powinny być zainicjalizowane w konstruktorze, zanim zostaną użyte. Szczególnie te, które są używane w wielu miejscach (jak kontrolki normalizacji).
pythondef __init__(self, parent=None, settings=None, hardware_profile=None):
    super().__init__(parent)
    self.settings = settings
    self.hardware_profile = hardware_profile
    self._setup_logging()
    self.setWindowTitle("Konfiguracja doszkalania")
    self.setMinimumWidth(1000)
    self.profiles_dir = Path("data/profiles")
    self.profiles_dir.mkdir(exist_ok=True)
    self.current_profile = None

    # Inicjalizacja wszystkich kontrolek
    self._init_controls()
    
    # Inicjalizacja interfejsu
    self._init_ui()

def _init_controls(self):
    """Inicjalizacja wszystkich kontrolek."""
    # Metrics
    self.auc_check = QtWidgets.QCheckBox()
    self.auc_check.setChecked(True)
    self.accuracy_check = QtWidgets.QCheckBox()
    self.accuracy_check.setChecked(True)
    self.precision_check = QtWidgets.QCheckBox()
    self.precision_check.setChecked(True)
    self.recall_check = QtWidgets.QCheckBox()
    self.recall_check.setChecked(True)
    self.f1_check = QtWidgets.QCheckBox()
    self.f1_check.setChecked(True)
    self.topk_check = QtWidgets.QCheckBox()
    self.topk_check.setChecked(True)
    self.confusion_matrix_check = QtWidgets.QCheckBox()
    self.confusion_matrix_check.setChecked(True)

    # Logging
    self.use_tensorboard_check = QtWidgets.QCheckBox()
    self.use_tensorboard_check.setChecked(True)
    self.use_wandb_check = QtWidgets.QCheckBox()
    self.use_wandb_check.setChecked(False)
    self.use_csv_check = QtWidgets.QCheckBox()
    self.use_csv_check.setChecked(True)
    self.log_freq_combo = QtWidgets.QComboBox()
    self.log_freq_combo.addItems(["epoch", "batch"])

    # Visualization
    self.use_gradcam_check = QtWidgets.QCheckBox()
    self.use_gradcam_check.setChecked(True)
    self.use_feature_maps_check = QtWidgets.QCheckBox()
    self.use_feature_maps_check.setChecked(True)
    self.use_pred_samples_check = QtWidgets.QCheckBox()
    self.use_pred_samples_check.setChecked(True)
    self.num_samples_spin = QtWidgets.QSpinBox()
    self.num_samples_spin.setRange(1, 100)
    self.num_samples_spin.setValue(10)

    # Early stopping
    self.patience_spin = QtWidgets.QSpinBox()
    self.patience_spin.setRange(1, 100)
    self.patience_spin.setValue(10)
    self.min_delta_spin = QtWidgets.QDoubleSpinBox()
    self.min_delta_spin.setRange(0.0, 1.0)
    self.min_delta_spin.setValue(0.001)
    self.min_delta_spin.setDecimals(4)
    self.monitor_combo = QtWidgets.QComboBox()
    self.monitor_combo.addItems(
        ["val_loss", "val_accuracy", "val_f1", "val_precision", "val_recall"]
    )

    # Checkpointing
    self.best_only_check = QtWidgets.QCheckBox()
    self.best_only_check.setChecked(True)
    self.save_freq_spin = QtWidgets.QSpinBox()
    self.save_freq_spin.setRange(1, 50)
    self.save_freq_spin.setValue(1)
    self.checkpoint_metric_combo = QtWidgets.QComboBox()
    self.checkpoint_metric_combo.addItems(
        ["val_loss", "val_accuracy", "val_f1", "val_precision", "val_recall"]
    )
    
    # Normalization controls - używane w wielu miejscach
    self.norm_mean_r = QtWidgets.QDoubleSpinBox()
    self.norm_mean_r.setRange(0.0, 1.0)
    self.norm_mean_r.setValue(0.485)
    self.norm_mean_r.setDecimals(3)
    
    self.norm_mean_g = QtWidgets.QDoubleSpinBox()
    self.norm_mean_g.setRange(0.0, 1.0)
    self.norm_mean_g.setValue(0.456)
    self.norm_mean_g.setDecimals(3)
    
    self.norm_mean_b = QtWidgets.QDoubleSpinBox()
    self.norm_mean_b.setRange(0.0, 1.0)
    self.norm_mean_b.setValue(0.406)
    self.norm_mean_b.setDecimals(3)
    
    self.norm_std_r = QtWidgets.QDoubleSpinBox()
    self.norm_std_r.setRange(0.0, 1.0)
    self.norm_std_r.setValue(0.229)
    self.norm_std_r.setDecimals(3)
    
    self.norm_std_g = QtWidgets.QDoubleSpinBox()
    self.norm_std_g.setRange(0.0, 1.0)
    self.norm_std_g.setValue(0.224)
    self.norm_std_g.setDecimals(3)
    
    self.norm_std_b = QtWidgets.QDoubleSpinBox()
    self.norm_std_b.setRange(0.0, 1.0)
    self.norm_std_b.setValue(0.225)
    self.norm_std_b.setDecimals(3)
    
    # Resize mode
    self.resize_mode_combo = QtWidgets.QComboBox()
    self.resize_mode_combo.addItems(["bilinear", "bicubic", "nearest", "area"])
    
    # Cache dataset
    self.cache_dataset_check = QtWidgets.QCheckBox()
    self.cache_dataset_check.setChecked(False)
Podsumowanie zmian:

Problem duplikacji kontrolek: Usunięto duplikację kontrolek normalizacyjnych przez inicjalizację ich w konstruktorze i używanie tych samych instancji w różnych zakładkach.
Blokowanie sygnałów: Dodano blokowanie sygnałów podczas wczytywania konfiguracji, aby uniknąć problemów z aktualizacją UI podczas ustawiania wartości.
Poprawka dla metody _load_config: Zmodyfikowano metodę, aby używała poprawnego podejścia do ustawiania wartości w kontrolkach (używając blockSignals).
Poprawka dla metody _select_model_file: Dodano blokowanie sygnałów oraz prawidłowe wywołanie metod aktualizujących UI po wczytaniu konfiguracji.
Inicjalizacja kontrolek w konstruktorze: Dodano inicjalizację wszystkich współdzielonych kontrolek w konstruktorze.

Te zmiany powinny rozwiązać problem z wczytywaniem konfiguracji z pliku i prawidłowym aktualizowaniem kontrolek UI.