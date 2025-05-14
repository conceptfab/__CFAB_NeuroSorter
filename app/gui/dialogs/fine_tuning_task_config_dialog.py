import datetime  # Dodany import
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

from PyQt6 import QtWidgets

from app.utils.file_utils import (
    validate_training_directory,
    validate_validation_directory,
)


class FineTuningTaskConfigDialog(QtWidgets.QDialog):
    """Dialog konfiguracji zadania doszkalania."""

    # Strategie odmrażania warstw
    UNFREEZE_ALL = "unfreeze_all"
    UNFREEZE_GRADUAL_END = "unfreeze_gradual_end"
    UNFREEZE_GRADUAL_START = "unfreeze_gradual_start"
    UNFREEZE_AFTER_EPOCHS = "unfreeze_after_epochs"

    def __init__(self, parent=None, settings=None, hardware_profile=None):
        super().__init__(parent)
        self.settings = settings
        # Kluczowa zmiana - sprawdzenie czy hardware_profile jest inicjalizowany poprawnie
        if not hardware_profile:
            from app.profiler import (
                HardwareProfiler,
            )  # Import lokalny aby uniknąć cyklicznych zależności

            profiler = HardwareProfiler()
            self.hardware_profile = profiler.get_optimal_parameters()
        else:
            self.hardware_profile = hardware_profile
        self._setup_logging()
        # Dodaj logowanie profilu sprzętowego
        self.logger.info(f"Profil sprzętowy: {self.hardware_profile}")
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
            [
                "val_loss",
                "val_accuracy",
                "val_f1",
                "val_precision",
                "val_recall",
            ]
        )

        # Checkpointing
        self.best_only_check = QtWidgets.QCheckBox()
        self.best_only_check.setChecked(True)
        self.save_freq_spin = QtWidgets.QSpinBox()
        self.save_freq_spin.setRange(1, 50)
        self.save_freq_spin.setValue(1)
        self.checkpoint_metric_combo = QtWidgets.QComboBox()
        self.checkpoint_metric_combo.addItems(
            [
                "val_loss",
                "val_accuracy",
                "val_f1",
                "val_precision",
                "val_recall",
            ]
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

        # Dodanie brakujących kontrolek
        self.scaling_method = QtWidgets.QComboBox()
        self.scaling_method.addItems(["resize", "crop", "pad"])

        self.maintain_aspect_ratio = QtWidgets.QCheckBox()
        self.maintain_aspect_ratio.setChecked(True)

        self.pad_to_square = QtWidgets.QCheckBox()
        self.pad_to_square.setChecked(False)

        self.pad_mode = QtWidgets.QComboBox()
        self.pad_mode.addItems(["constant", "reflect", "replicate"])

        self.pad_value = QtWidgets.QSpinBox()
        self.pad_value.setRange(0, 255)
        self.pad_value.setValue(0)

        self.tensorboard_dir_edit = QtWidgets.QLineEdit()
        self.tensorboard_dir_edit.setText("logs/tensorboard")

        self.model_dir_edit = QtWidgets.QLineEdit()
        self.model_dir_edit.setText("models")

        self.save_logs_check = QtWidgets.QCheckBox()
        self.save_logs_check.setChecked(True)

        # Normalization
        self.normalization_combo = QtWidgets.QComboBox()
        self.normalization_combo.addItems(["RGB", "BGR"])

        # AutoAugment
        self.autoaugment_check = QtWidgets.QCheckBox("Użyj AutoAugment")
        self.autoaugment_check.setChecked(False)

        # RandAugment
        self.randaugment_check = QtWidgets.QCheckBox("Użyj RandAugment")
        self.randaugment_check.setChecked(False)
        self.randaugment_n_spin = QtWidgets.QSpinBox()
        self.randaugment_n_spin.setRange(1, 10)
        self.randaugment_n_spin.setValue(2)
        self.randaugment_m_spin = QtWidgets.QSpinBox()
        self.randaugment_m_spin.setRange(0, 30)
        self.randaugment_m_spin.setValue(9)

        # Kontrolki zapobiegania katastrofalnemu zapominaniu
        self.prevent_forgetting_check = QtWidgets.QCheckBox(
            "Włącz zapobieganie zapominaniu"
        )

        # Inicjalizacja seed_spin
        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(0, 2147483647)
        self.seed_spin.setValue(42)

        self.preserve_classes_check = QtWidgets.QCheckBox("Zachowaj oryginalne klasy")
        self.preserve_classes_check.setChecked(True)

        self.rehearsal_check = QtWidgets.QCheckBox("Użyj rehearsal")
        self.rehearsal_check.setChecked(True)

        self.samples_per_class_spin = QtWidgets.QSpinBox()
        self.samples_per_class_spin.setRange(5, 100)
        self.samples_per_class_spin.setValue(20)

        self.synthetic_samples_check = QtWidgets.QCheckBox("Użyj próbek syntetycznych")
        self.synthetic_samples_check.setChecked(True)

        self.knowledge_distillation_check = QtWidgets.QCheckBox(
            "Użyj knowledge distillation"
        )
        self.knowledge_distillation_check.setChecked(True)

        self.kd_temperature_spin = QtWidgets.QDoubleSpinBox()
        self.kd_temperature_spin.setRange(1.0, 10.0)
        self.kd_temperature_spin.setValue(2.0)
        self.kd_temperature_spin.setDecimals(1)

        self.kd_alpha_spin = QtWidgets.QDoubleSpinBox()
        self.kd_alpha_spin.setRange(0.1, 0.9)
        self.kd_alpha_spin.setValue(0.4)
        self.kd_alpha_spin.setDecimals(2)

        self.ewc_check = QtWidgets.QCheckBox("Użyj EWC regularization")
        self.ewc_check.setChecked(True)

        self.ewc_lambda_spin = QtWidgets.QDoubleSpinBox()
        self.ewc_lambda_spin.setRange(
            100.0, 10000.0
        )  # ZMIANA: Zwiększenie zakresu Lambda
        self.ewc_lambda_spin.setValue(
            5000.0
        )  # ZMIANA: Zwiększenie domyślnej wartości Lambda
        self.ewc_lambda_spin.setDecimals(1)
        self.ewc_lambda_spin.setToolTip(
            "Współczynnik lambda dla EWC. Wyższe wartości oznaczają silniejszą ochronę poprzedniej wiedzy."
        )

        self.fisher_sample_size_spin = QtWidgets.QSpinBox()
        self.fisher_sample_size_spin.setRange(50, 1000)
        self.fisher_sample_size_spin.setValue(200)  # Dodana linia - domyślna wartość

        self.layer_freezing_combo = QtWidgets.QComboBox()
        self.layer_freezing_combo.addItems(["gradual", "selective", "progressive"])

        self.freeze_ratio_spin = QtWidgets.QDoubleSpinBox()
        self.freeze_ratio_spin.setRange(0.0, 0.9)
        self.freeze_ratio_spin.setValue(0.7)
        self.freeze_ratio_spin.setDecimals(2)

        # Inicjalizacja deterministic_check
        self.deterministic_check = QtWidgets.QCheckBox()
        self.deterministic_check.setChecked(False)

        # --- DODANE: Zaawansowane kontrolki ---
        # class_weights
        self.class_weights_combo = QtWidgets.QComboBox()
        self.class_weights_combo.addItems(["none", "balanced", "custom"])
        self.class_weights_combo.setCurrentText("none")

        # sampler
        self.sampler_combo = QtWidgets.QComboBox()
        self.sampler_combo.addItems(["none", "random", "weighted_random"])
        self.sampler_combo.setCurrentText("none")

        # image_channels
        self.image_channels_spin = QtWidgets.QSpinBox()
        self.image_channels_spin.setRange(1, 4)
        self.image_channels_spin.setValue(3)

        # TTA (Test Time Augmentation)
        self.use_tta_check = QtWidgets.QCheckBox()
        self.use_tta_check.setChecked(False)
        self.tta_num_samples_spin = QtWidgets.QSpinBox()
        self.tta_num_samples_spin.setRange(1, 100)
        self.tta_num_samples_spin.setValue(5)

        # export_onnx
        self.export_onnx_check = QtWidgets.QCheckBox()
        self.export_onnx_check.setChecked(False)

        # quantization
        self.quantization_check = QtWidgets.QCheckBox()
        self.quantization_check.setChecked(False)
        self.quantization_precision_combo = QtWidgets.QComboBox()
        self.quantization_precision_combo.addItems(["int8", "float16", "float32"])
        self.quantization_precision_combo.setCurrentText("int8")

        # Stochastic Depth
        self.use_stoch_depth_check = QtWidgets.QCheckBox("Użyj Stochastic Depth")
        self.stoch_depth_drop_rate = QtWidgets.QDoubleSpinBox()
        self.stoch_depth_drop_rate.setRange(0.0, 0.5)
        self.stoch_depth_drop_rate.setDecimals(3)
        self.stoch_depth_drop_rate.setValue(0.2)
        self.stoch_depth_survival_prob = QtWidgets.QDoubleSpinBox()
        self.stoch_depth_survival_prob.setRange(0.5, 1.0)
        self.stoch_depth_survival_prob.setDecimals(3)
        self.stoch_depth_survival_prob.setValue(0.8)

        # Random Erase
        self.use_random_erase_check = QtWidgets.QCheckBox("Użyj Random Erase")
        self.random_erase_prob = QtWidgets.QDoubleSpinBox()
        self.random_erase_prob.setRange(0.0, 1.0)
        self.random_erase_prob.setDecimals(3)
        self.random_erase_prob.setValue(0.25)
        self.random_erase_mode_combo = QtWidgets.QComboBox()
        self.random_erase_mode_combo.addItems(["pixel", "block"])

        # Optymalizacja parametrów EWC
        self.adaptive_ewc_lambda_check = QtWidgets.QCheckBox("Adaptacyjna Lambda")
        self.adaptive_ewc_lambda_check.setChecked(True)
        self.adaptive_ewc_lambda_check.setToolTip(
            "Jeśli zaznaczone, Lambda będzie dynamicznie zwiększana podczas treningu."
        )

        # Dodaj do layoutu
        ewc_layout = QtWidgets.QHBoxLayout()
        ewc_layout.addWidget(self.adaptive_ewc_lambda_check)

    def _setup_logging(self):
        """Konfiguracja logowania dla okna dialogowego."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # Handler do pliku
        fh = logging.FileHandler("training_dialog.log")
        fh.setLevel(logging.DEBUG)

        # Handler do konsoli
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Format logów
        log_fmt = "%(asctime)s - %(name)s - " "%(levelname)s - %(message)s"
        formatter = logging.Formatter(log_fmt)
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        self.logger.info("Inicjalizacja okna")

    def _init_ui(self):
        """Inicjalizuje interfejs użytkownika."""
        try:
            self.logger.debug("Rozpoczęcie inicjalizacji UI")
            layout = QtWidgets.QVBoxLayout(self)

            # Dodaj pasek narzędzi
            toolbar = QtWidgets.QToolBar("Narzędzia")
            (
                self.addToolBar(QtWidgets.Qt.ToolBarArea.TopToolBarArea, toolbar)
                if hasattr(self, "addToolBar")
                else None
            )

            # Przycisk: Pokaż profil sprzętowy
            show_hw_profile_btn = QtWidgets.QPushButton("Pokaż profil sprzętowy")
            show_hw_profile_btn.clicked.connect(self._show_hardware_profile)
            layout.addWidget(show_hw_profile_btn)

            # Przycisk: Otwórz plik logu
            open_log_btn = QtWidgets.QPushButton("Otwórz plik logu")
            open_log_btn.clicked.connect(self._open_log_file)
            layout.addWidget(open_log_btn)

            # Utworzenie zakładek
            self.tabs = QtWidgets.QTabWidget()

            # 1. Zakładka: Dane i Model
            tab = self._create_data_model_tab()
            self.tabs.addTab(tab, "Dane i Model")

            # 2. Zakładka: Parametry Fine-tuningu
            tab = self._create_fine_tuning_params_tab()
            self.tabs.addTab(tab, "Parametry")

            # 3. Zakładka: Regularyzacja i Optymalizacja
            tab = self._create_regularization_tab()
            name = "Regularyzacja"
            self.tabs.addTab(tab, name)

            # 4. Zakładka: Augmentacja Danych
            tab = self._create_augmentation_tab()
            self.tabs.addTab(tab, "Augmentacja")

            # 5. Zakładka: Preprocessing
            tab = self._create_preprocessing_tab()
            self.tabs.addTab(tab, "Preprocessing")

            # 6. Zakładka: Monitorowanie i Zapis
            tab = self._create_monitoring_tab()
            self.tabs.addTab(tab, "Monitorowanie")

            # 7. Zakładka: Zaawansowane
            tab = self._create_advanced_tab()
            self.tabs.addTab(tab, "Zaawansowane")

            # 8. NOWA ZAKŁADKA: Optymalizacja treningu
            tab = self._create_optimization_tab()
            self.tabs.addTab(tab, "Optymalizacja treningu")

            layout.addWidget(self.tabs)

            # Przyciski
            buttons_layout = QtWidgets.QHBoxLayout()

            # Przycisk "Dodaj zadanie"
            add_task_btn = QtWidgets.QPushButton("Dodaj zadanie")
            add_task_btn.clicked.connect(self._on_accept)
            buttons_layout.addWidget(add_task_btn)

            # Przycisk "Zamknij"
            close_btn = QtWidgets.QPushButton("Zamknij")
            close_btn.clicked.connect(self.accept)
            buttons_layout.addWidget(close_btn)

            layout.addLayout(buttons_layout)

            self.logger.debug("Zakończono inicjalizację UI")

        except Exception as e:
            msg = "Błąd podczas inicjalizacji UI"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            raise

        # Połączenia sygnałów dla aktualizacji UI
        self.arch_combo.currentTextChanged.connect(
            self._update_architecture_dependent_controls
        )
        self.optimizer_combo.currentTextChanged.connect(
            self._update_optimizer_dependent_controls
        )
        self.scheduler_combo.currentTextChanged.connect(
            self._update_scheduler_dependent_controls
        )
        self.unfreeze_strategy_combo.currentTextChanged.connect(
            self._update_training_dependent_controls
        )

        # Sygnały dla augmentacji
        self.basic_aug_check.stateChanged.connect(
            self._update_augmentation_dependent_controls
        )
        self.mixup_check.stateChanged.connect(
            self._update_augmentation_dependent_controls
        )
        self.cutmix_check.stateChanged.connect(
            self._update_augmentation_dependent_controls
        )
        self.autoaugment_check.stateChanged.connect(
            self._update_augmentation_dependent_controls
        )
        self.randaugment_check.stateChanged.connect(
            self._update_augmentation_dependent_controls
        )

        # Sygnały dla preprocessingu
        self.norm_mean_r.valueChanged.connect(
            self._update_preprocessing_dependent_controls
        )
        self.norm_mean_g.valueChanged.connect(
            self._update_preprocessing_dependent_controls
        )
        self.norm_mean_b.valueChanged.connect(
            self._update_preprocessing_dependent_controls
        )
        self.norm_std_r.valueChanged.connect(
            self._update_preprocessing_dependent_controls
        )
        self.norm_std_g.valueChanged.connect(
            self._update_preprocessing_dependent_controls
        )
        self.norm_std_b.valueChanged.connect(
            self._update_preprocessing_dependent_controls
        )

        # Sygnały dla monitorowania
        self.patience_spin.valueChanged.connect(
            self._update_monitoring_dependent_controls
        )
        self.monitor_combo.currentTextChanged.connect(
            self._update_monitoring_dependent_controls
        )
        self.checkpoint_metric_combo.currentTextChanged.connect(
            self._update_monitoring_dependent_controls
        )
        self.use_tensorboard_check.stateChanged.connect(
            self._update_monitoring_dependent_controls
        )
        self.use_pred_samples_check.stateChanged.connect(
            self._update_monitoring_dependent_controls
        )
        self.best_only_check.stateChanged.connect(
            self._update_monitoring_dependent_controls
        )

    def _create_data_model_tab(self) -> QtWidgets.QWidget:
        """Tworzenie zakładki Dane i Model."""
        try:
            self.logger.debug("Tworzenie zakładki")
            tab = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(tab)
            form = QtWidgets.QFormLayout()

            # Model do doszkalania
            model_path_layout = QtWidgets.QHBoxLayout()
            model_path_label = QtWidgets.QLabel("Wybór modelu:")
            self.model_path_edit = QtWidgets.QLineEdit()
            model_path_btn = QtWidgets.QPushButton("Przeglądaj...")
            model_path_btn.clicked.connect(self._select_model_file)
            self.create_profile_from_config_btn = QtWidgets.QPushButton(
                "Utwórz profil z konfiguracji modelu"
            )
            self.create_profile_from_config_btn.clicked.connect(
                self._create_profile_from_model_config
            )
            self.create_profile_from_config_btn.setEnabled(
                False
            )  # Początkowo nieaktywny

            model_path_layout.addWidget(model_path_label, 10)  # 10% szerokości
            model_path_layout.addWidget(self.model_path_edit, 60)  # 60% szerokości
            model_path_layout.addWidget(model_path_btn, 15)  # 15% szerokości
            model_path_layout.addWidget(
                self.create_profile_from_config_btn, 15
            )  # 15% szerokości

            form.addRow(
                "", model_path_layout
            )  # Pusty label, aby wyrównać z poprzednim wierszem

            # Architektura modelu
            self.arch_combo = QtWidgets.QComboBox()
            self.arch_combo.addItems(
                [
                    "EfficientNet",
                    "ResNet",
                    "MobileNet",
                    "VGG",
                    "DenseNet",
                    "ConvNeXt",
                    "ConvNeXtV2",
                    "InceptionV3",
                    "Xception",
                ]
            )
            form.addRow("Architektura:", self.arch_combo)

            # Wariant modelu
            self.variant_combo = QtWidgets.QComboBox()
            self._update_variant_combo("EfficientNet")
            form.addRow("Wariant:", self.variant_combo)
            self.arch_combo.currentTextChanged.connect(self._on_architecture_changed)

            # Nazwa zadania (obowiązkowe pole)
            self.name_edit = QtWidgets.QLineEdit()
            self.name_edit.setPlaceholderText("Nazwa zadania (wymagana)")
            self.name_edit.setText("Nowe zadanie")  # Domyślna nazwa
            form.addRow("Nazwa zadania:", self.name_edit)

            # Katalog danych treningowych
            train_dir_layout = QtWidgets.QHBoxLayout()
            self.train_dir_edit = QtWidgets.QLineEdit()
            train_dir_btn = QtWidgets.QPushButton("Przeglądaj...")
            train_dir_btn.clicked.connect(self._select_train_dir)
            train_dir_layout.addWidget(self.train_dir_edit)
            train_dir_layout.addWidget(train_dir_btn)

            train_label = "Katalog treningowy:"
            form.addRow(train_label, train_dir_layout)

            # Katalog danych walidacyjnych
            val_dir_layout = QtWidgets.QHBoxLayout()
            self.val_dir_edit = QtWidgets.QLineEdit()
            val_dir_btn = QtWidgets.QPushButton("Przeglądaj...")
            val_dir_btn.clicked.connect(self._select_val_dir)
            val_dir_layout.addWidget(self.val_dir_edit)
            val_dir_layout.addWidget(val_dir_btn)

            val_label = "Katalog walidacyjny:"
            form.addRow(val_label, val_dir_layout)

            # Rozmiar obrazu wejściowego
            self.input_size_spin = QtWidgets.QSpinBox()
            self.input_size_spin.setRange(32, 1024)
            self.input_size_spin.setValue(224)
            self.input_size_spin.setSingleStep(32)
            size_label = "Rozmiar obrazu:"
            form.addRow(size_label, self.input_size_spin)

            # Liczba klas
            self.num_classes_spin = QtWidgets.QSpinBox()
            self.num_classes_spin.setRange(2, 1000)
            self.num_classes_spin.setValue(2)  # Domyślna wartość
            form.addRow("Liczba klas:", self.num_classes_spin)

            # Logowanie wartości po inicjalizacji
            self.logger.info(
                f"Zainicjalizowano UI - Nazwa zadania: {self.name_edit.text()}"
            )
            self.logger.info(
                f"Zainicjalizowano UI - Liczba klas: {self.num_classes_spin.value()}"
            )

            # Pretrained
            self.pretrained_check = QtWidgets.QCheckBox()
            self.pretrained_check.setChecked(True)
            form.addRow("Użyj pretrained:", self.pretrained_check)

            # Pretrained weights
            self.pretrained_weights_combo = QtWidgets.QComboBox()
            self.pretrained_weights_combo.addItems(
                ["imagenet", "imagenet21k", "noisy-student", "none"]
            )
            form.addRow("Pretrained weights:", self.pretrained_weights_combo)

            # Feature extraction only
            self.feature_extraction_check = QtWidgets.QCheckBox()
            self.feature_extraction_check.setChecked(False)
            form.addRow("Feature extraction only:", self.feature_extraction_check)

            # Activation function
            self.activation_combo = QtWidgets.QComboBox()
            self.activation_combo.addItems(["relu", "swish", "mish", "gelu"])
            form.addRow("Activation:", self.activation_combo)

            # Dropout at inference
            self.dropout_at_inference_check = QtWidgets.QCheckBox()
            self.dropout_at_inference_check.setChecked(False)
            form.addRow("Dropout at inference:", self.dropout_at_inference_check)

            # Global pooling
            self.global_pool_combo = QtWidgets.QComboBox()
            self.global_pool_combo.addItems(["avg", "max"])
            form.addRow("Global pooling:", self.global_pool_combo)

            # Last layer activation
            self.last_layer_activation_combo = QtWidgets.QComboBox()
            self.last_layer_activation_combo.addItems(["softmax", "sigmoid"])
            form.addRow("Last layer activation:", self.last_layer_activation_combo)

            # Grupa profili
            profile_group = QtWidgets.QGroupBox("Dostępne profile")
            profile_layout = QtWidgets.QVBoxLayout()

            # Lista profili
            self.profile_list = QtWidgets.QListWidget()
            self.profile_list.currentItemChanged.connect(self._on_profile_selected)
            self._refresh_profile_list()
            profile_layout.addWidget(self.profile_list)

            # Informacje o profilu
            info_group = QtWidgets.QGroupBox("Informacje o profilu")
            info_layout = QtWidgets.QFormLayout()

            self.profile_info = QtWidgets.QTextEdit()
            self.profile_info.setReadOnly(True)
            self.profile_info.setMaximumHeight(60)
            info_layout.addRow("Info:", self.profile_info)

            self.profile_description = QtWidgets.QTextEdit()
            self.profile_description.setReadOnly(True)
            self.profile_description.setMaximumHeight(60)
            info_layout.addRow("Opis:", self.profile_description)

            self.profile_data_required = QtWidgets.QTextEdit()
            self.profile_data_required.setReadOnly(True)
            self.profile_data_required.setMaximumHeight(60)
            info_layout.addRow("Wymagane dane:", self.profile_data_required)

            self.profile_hardware_required = QtWidgets.QTextEdit()
            self.profile_hardware_required.setReadOnly(True)
            self.profile_hardware_required.setMaximumHeight(20)
            info_layout.addRow("Wymagany sprzęt:", self.profile_hardware_required)

            info_group.setLayout(info_layout)
            profile_layout.addWidget(info_group)

            # Przyciski profilu
            buttons_layout = QtWidgets.QHBoxLayout()

            self.edit_profile_btn = QtWidgets.QPushButton("Edytuj profil")
            self.edit_profile_btn.clicked.connect(self._edit_profile)
            buttons_layout.addWidget(self.edit_profile_btn)

            self.apply_profile_btn = QtWidgets.QPushButton("Zastosuj profil")
            self.apply_profile_btn.clicked.connect(self._apply_profile)
            buttons_layout.addWidget(self.apply_profile_btn)

            self.clone_profile_btn = QtWidgets.QPushButton("Klonuj profil")
            self.clone_profile_btn.clicked.connect(self._clone_profile)
            buttons_layout.addWidget(self.clone_profile_btn)

            self.save_profile_btn = QtWidgets.QPushButton("Zapisz profil")
            self.save_profile_btn.clicked.connect(self._save_profile)
            buttons_layout.addWidget(self.save_profile_btn)

            self.delete_profile_btn = QtWidgets.QPushButton("Usuń profil")
            self.delete_profile_btn.clicked.connect(self._delete_profile)
            buttons_layout.addWidget(self.delete_profile_btn)

            profile_layout.addLayout(buttons_layout)
            profile_group.setLayout(profile_layout)

            layout.addLayout(form)
            layout.addWidget(profile_group)
            return tab

        except Exception as e:
            msg = "Błąd podczas tworzenia zakładki"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            raise

    def _refresh_profile_list(self):
        """Odświeża listę dostępnych profili."""
        self.profile_list.clear()
        self.logger.debug("Rozpoczynam odświeżanie listy profili")
        for profile_file in self.profiles_dir.glob("*.json"):
            try:
                with open(profile_file, "r", encoding="utf-8") as f:
                    profile_data = json.load(f)
                    if profile_data.get("type") == "fine_tuning":
                        self.profile_list.addItem(profile_file.stem)
                        self.logger.debug(f"Dodano profil {profile_file.stem} do listy")
            except Exception as e:
                self.logger.error(
                    f"Błąd podczas wczytywania profilu {profile_file}: {str(e)}",
                    exc_info=True,
                )
        self.logger.debug(
            "Zakończono odświeżanie listy profili. Liczba profili: "
            f"{self.profile_list.count()}"
        )

    def _on_profile_selected(self, current, previous):
        """Obsługa wyboru profilu z listy."""
        if current is None:
            return
        try:
            profile_name = current.text()
            profile_path = self.profiles_dir / f"{profile_name}.json"
            with open(profile_path, "r", encoding="utf-8") as f:
                profile_data = json.load(f)
            self.current_profile = profile_data
            self.profile_info.setText(profile_data.get("info", ""))
            self.profile_description.setText(profile_data.get("description", ""))
            self.profile_data_required.setText(profile_data.get("data_required", ""))
            self.profile_hardware_required.setText(
                profile_data.get("hardware_required", "")
            )
        except Exception as e:
            self.logger.error(
                f"Błąd podczas ładowania profilu: {str(e)}", exc_info=True
            )
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Nie można załadować profilu: {str(e)}"
            )

    def _edit_profile(self):
        """Otwiera profil w edytorze tekstowym."""
        if not self.current_profile:
            QtWidgets.QMessageBox.warning(
                self, "Ostrzeżenie", "Najpierw wybierz profil do edycji."
            )
            return
        try:
            profile_name = self.profile_list.currentItem().text()
            profile_path = self.profiles_dir / f"{profile_name}.json"
            os.startfile(str(profile_path))  # Dla Windows
        except Exception as e:
            self.logger.error(
                f"Błąd podczas otwierania profilu: {str(e)}", exc_info=True
            )
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Nie można otworzyć profilu: {str(e)}"
            )

    def _apply_profile(self):
        """Stosuje wybrany profil konfiguracji."""
        if not self.current_profile:
            QtWidgets.QMessageBox.warning(
                self, "Ostrzeżenie", "Najpierw wybierz profil do zastosowania."
            )
            return

        try:
            self.logger.info(
                f"Zastosowywanie profilu: {self.profile_list.currentItem().text()}"
            )
            config_to_load = self.current_profile.get("config", {})
            if not config_to_load:
                self.logger.warning(
                    "Wybrany profil nie zawiera sekcji 'config' lub jest ona pusta."
                )
                QtWidgets.QMessageBox.warning(
                    self,
                    "Ostrzeżenie",
                    "Profil nie zawiera danych konfiguracyjnych.",
                )
                return

            # Dane i Model
            if "model" in config_to_load:
                model_config = config_to_load["model"]
                if "architecture" in model_config:
                    self.arch_combo.setCurrentText(model_config["architecture"])
                if "variant" in model_config:
                    self.variant_combo.setCurrentText(model_config["variant"])
                if "input_size" in model_config:
                    self.input_size_spin.setValue(model_config["input_size"])
                if "num_classes" in model_config:
                    self.num_classes_spin.setValue(model_config["num_classes"])
                if "pretrained" in model_config:
                    self.pretrained_check.setChecked(model_config["pretrained"])
                if "pretrained_weights" in model_config:
                    self.pretrained_weights_combo.setCurrentText(
                        model_config["pretrained_weights"]
                    )
                if "feature_extraction_only" in model_config:
                    self.feature_extraction_check.setChecked(
                        model_config["feature_extraction_only"]
                    )
                if "activation" in model_config:
                    self.activation_combo.setCurrentText(model_config["activation"])
                if "dropout_at_inference" in model_config:
                    self.dropout_at_inference_check.setChecked(
                        model_config["dropout_at_inference"]
                    )
                if "global_pool" in model_config:
                    self.global_pool_combo.setCurrentText(model_config["global_pool"])
                if "last_layer_activation" in model_config:
                    self.last_layer_activation_combo.setCurrentText(
                        model_config["last_layer_activation"]
                    )

            # Parametry Treningu
            if "training" in config_to_load:
                training_config = config_to_load["training"]
                if "epochs" in training_config:
                    self.epochs_spin.setValue(training_config["epochs"])
                if "batch_size" in training_config:
                    self.batch_size_spin.setValue(training_config["batch_size"])
                if "learning_rate" in training_config:
                    self.lr_spin.setValue(training_config["learning_rate"])
                if "optimizer" in training_config:
                    self.optimizer_combo.setCurrentText(training_config["optimizer"])
                if "scheduler" in training_config:
                    self.scheduler_combo.setCurrentText(training_config["scheduler"])
                if "num_workers" in training_config:
                    self.num_workers_spin.setValue(training_config["num_workers"])
                if "warmup_epochs" in training_config:
                    self.warmup_epochs_spin.setValue(training_config["warmup_epochs"])
                if "warmup_lr_init" in training_config:
                    self.warmup_lr_init_spin.setValue(training_config["warmup_lr_init"])
                if "mixed_precision" in training_config:
                    self.mixed_precision_check.setChecked(
                        training_config["mixed_precision"]
                    )
                if "gradient_accumulation_steps" in training_config:
                    self.grad_accum_steps_spin.setValue(
                        training_config["gradient_accumulation_steps"]
                    )
                if "gradient_clip" in training_config:
                    self.gradient_clip_spin.setValue(training_config["gradient_clip"])
                if "validation_split" in training_config:
                    self.validation_split_spin.setValue(
                        training_config["validation_split"]
                    )
                if "evaluation_freq" in training_config:
                    self.eval_freq_spin.setValue(training_config["evaluation_freq"])
                if "use_ema" in training_config:
                    self.use_ema_check.setChecked(training_config["use_ema"])
                if "ema_decay" in training_config:
                    self.ema_decay_spin.setValue(training_config["ema_decay"])
                if "unfreeze_strategy" in training_config:
                    self.unfreeze_strategy_combo.setCurrentText(
                        training_config["unfreeze_strategy"]
                    )
                if "unfreeze_after_epochs" in training_config:
                    self.unfreeze_after_epochs_spin.setValue(
                        training_config["unfreeze_after_epochs"]
                    )
                if "unfreeze_layers" in training_config:
                    self.unfreeze_layers_spin.setValue(
                        training_config["unfreeze_layers"]
                    )
                if "frozen_lr" in training_config:
                    self.frozen_lr_spin.setValue(training_config["frozen_lr"])
                if "unfrozen_lr" in training_config:
                    self.unfrozen_lr_spin.setValue(training_config["unfrozen_lr"])

            # Regularyzacja
            if "regularization" in config_to_load:
                reg_config = config_to_load["regularization"]
                if "weight_decay" in reg_config:
                    self.weight_decay_spin.setValue(reg_config["weight_decay"])
                if "gradient_clip" in reg_config:
                    self.gradient_clip_spin.setValue(reg_config["gradient_clip"])
                if "label_smoothing" in reg_config:
                    self.label_smoothing_spin.setValue(reg_config["label_smoothing"])
                if "drop_connect_rate" in reg_config:
                    self.drop_connect_spin.setValue(reg_config["drop_connect_rate"])
                if "dropout_rate" in reg_config:
                    self.dropout_spin.setValue(reg_config["dropout_rate"])
                if "momentum" in reg_config:
                    self.momentum_spin.setValue(reg_config["momentum"])
                if "epsilon" in reg_config:
                    self.epsilon_spin.setValue(reg_config["epsilon"])

                # SWA (Stochastic Weight Averaging)
                if "swa" in reg_config:
                    swa_config = reg_config["swa"]
                    if "use" in swa_config:
                        self.use_swa_check.setChecked(swa_config["use"])
                    if "start_epoch" in swa_config:
                        self.swa_start_epoch_spin.setValue(swa_config["start_epoch"])

            # Augmentacja
            if "augmentation" in config_to_load:
                aug_config = config_to_load["augmentation"]

                # Basic augmentation
                if "basic" in aug_config:
                    basic_config = aug_config["basic"]
                    if "use" in basic_config:
                        self.basic_aug_check.setChecked(basic_config["use"])
                    if "rotation" in basic_config:
                        self.rotation_spin.setValue(basic_config["rotation"])
                    if "brightness" in basic_config:
                        self.brightness_spin.setValue(basic_config["brightness"])
                    if "shift" in basic_config:
                        self.shift_spin.setValue(basic_config["shift"])
                    if "zoom" in basic_config:
                        self.zoom_spin.setValue(basic_config["zoom"])
                    if "horizontal_flip" in basic_config:
                        self.horizontal_flip_check.setChecked(
                            basic_config["horizontal_flip"]
                        )
                    if "vertical_flip" in basic_config:
                        self.vertical_flip_check.setChecked(
                            basic_config["vertical_flip"]
                        )

                # Mixup
                if "mixup" in aug_config:
                    mixup_config = aug_config["mixup"]
                    if "use" in mixup_config:
                        self.mixup_check.setChecked(mixup_config["use"])
                    if "alpha" in mixup_config:
                        self.mixup_alpha_spin.setValue(mixup_config["alpha"])

                # CutMix
                if "cutmix" in aug_config:
                    cutmix_config = aug_config["cutmix"]
                    if "use" in cutmix_config:
                        self.cutmix_check.setChecked(cutmix_config["use"])
                    if "alpha" in cutmix_config:
                        self.cutmix_alpha_spin.setValue(cutmix_config["alpha"])

                # AutoAugment i RandAugment
                if "autoaugment" in aug_config:
                    autoaug_config = aug_config["autoaugment"]
                    if "use" in autoaug_config:
                        self.autoaugment_check.setChecked(autoaug_config["use"])

                if "randaugment" in aug_config:
                    randaug_config = aug_config["randaugment"]
                    if "use" in randaug_config:
                        self.randaugment_check.setChecked(randaug_config["use"])
                    if "n" in randaug_config:
                        self.randaugment_n_spin.setValue(randaug_config["n"])
                    if "m" in randaug_config:
                        self.randaugment_m_spin.setValue(randaug_config["m"])

            # Preprocessing
            if "preprocessing" in config_to_load:
                preproc_config = config_to_load["preprocessing"]

                # Normalization
                if "normalization" in preproc_config:
                    norm_config = preproc_config["normalization"]
                    if "mean" in norm_config and len(norm_config["mean"]) == 3:
                        self.norm_mean_r.setValue(norm_config["mean"][0])
                        self.norm_mean_g.setValue(norm_config["mean"][1])
                        self.norm_mean_b.setValue(norm_config["mean"][2])
                    if "std" in norm_config and len(norm_config["std"]) == 3:
                        self.norm_std_r.setValue(norm_config["std"][0])
                        self.norm_std_g.setValue(norm_config["std"][1])
                        self.norm_std_b.setValue(norm_config["std"][2])

                if "resize_mode" in preproc_config:
                    self.resize_mode_combo.setCurrentText(preproc_config["resize_mode"])
                if "cache_dataset" in preproc_config:
                    self.cache_dataset_check.setChecked(preproc_config["cache_dataset"])

            # Monitorowanie
            if "monitoring" in config_to_load:
                monitor_config = config_to_load["monitoring"]

                # Metrics
                if "metrics" in monitor_config:
                    metrics_config = monitor_config["metrics"]
                    if "accuracy" in metrics_config:
                        self.accuracy_check.setChecked(metrics_config["accuracy"])
                    if "precision" in metrics_config:
                        self.precision_check.setChecked(metrics_config["precision"])
                    if "recall" in metrics_config:
                        self.recall_check.setChecked(metrics_config["recall"])
                    if "f1" in metrics_config:
                        self.f1_check.setChecked(metrics_config["f1"])
                    if "topk" in metrics_config:
                        self.topk_check.setChecked(metrics_config["topk"])
                    if "confusion_matrix" in metrics_config:
                        self.confusion_matrix_check.setChecked(
                            metrics_config["confusion_matrix"]
                        )
                    if "auc" in metrics_config:
                        self.auc_check.setChecked(metrics_config["auc"])

                # Logging
                if "logging" in monitor_config:
                    logging_config = monitor_config["logging"]
                    if "use_tensorboard" in logging_config:
                        self.use_tensorboard_check.setChecked(
                            logging_config["use_tensorboard"]
                        )
                    if "use_wandb" in logging_config:
                        self.use_wandb_check.setChecked(logging_config["use_wandb"])
                    if "save_to_csv" in logging_config:
                        self.use_csv_check.setChecked(logging_config["save_to_csv"])
                    if "logging_freq" in logging_config:
                        self.log_freq_combo.setCurrentText(
                            logging_config["logging_freq"]
                        )

                # Visualization
                if "visualization" in monitor_config:
                    viz_config = monitor_config["visualization"]
                    if "use_gradcam" in viz_config:
                        self.use_gradcam_check.setChecked(viz_config["use_gradcam"])
                    if "use_feature_maps" in viz_config:
                        self.use_feature_maps_check.setChecked(
                            viz_config["use_feature_maps"]
                        )
                    if "use_pred_samples" in viz_config:
                        self.use_pred_samples_check.setChecked(
                            viz_config["use_pred_samples"]
                        )
                    if "num_samples" in viz_config:
                        self.num_samples_spin.setValue(viz_config["num_samples"])

                # Early stopping
                if "early_stopping" in monitor_config:
                    es_config = monitor_config["early_stopping"]
                    if "patience" in es_config:
                        self.patience_spin.setValue(es_config["patience"])
                    if "min_delta" in es_config:
                        self.min_delta_spin.setValue(es_config["min_delta"])
                    if "monitor" in es_config:
                        self.monitor_combo.setCurrentText(es_config["monitor"])

                # Checkpointing
                if "checkpointing" in monitor_config:
                    cp_config = monitor_config["checkpointing"]
                    if "best_only" in cp_config:
                        self.best_only_check.setChecked(cp_config["best_only"])
                    if "save_frequency" in cp_config:
                        self.save_freq_spin.setValue(cp_config["save_frequency"])
                    if "metric" in cp_config:
                        self.checkpoint_metric_combo.setCurrentText(cp_config["metric"])

            # Zaawansowane
            if "advanced" in config_to_load:
                adv_config = config_to_load["advanced"]
                if "seed" in adv_config:
                    self.seed_spin.setValue(adv_config["seed"])
                if "deterministic" in adv_config:
                    self.deterministic_check.setChecked(adv_config["deterministic"])
                if "class_weights" in adv_config:
                    self.class_weights_combo.setCurrentText(adv_config["class_weights"])
                if "sampler" in adv_config:
                    self.sampler_combo.setCurrentText(adv_config["sampler"])
                if "image_channels" in adv_config:
                    self.image_channels_spin.setValue(adv_config["image_channels"])

                # TTA (Test Time Augmentation)
                if "tta" in adv_config:
                    tta_config = adv_config["tta"]
                    if "use" in tta_config:
                        self.use_tta_check.setChecked(tta_config["use"])
                    if "num_augmentations" in tta_config:
                        self.tta_num_samples_spin.setValue(
                            tta_config["num_augmentations"]
                        )

                # Export to ONNX
                if "export_onnx" in adv_config:
                    self.export_onnx_check.setChecked(adv_config["export_onnx"])

                # Quantization
                if "quantization" in adv_config:
                    quant_config = adv_config["quantization"]
                    if "use" in quant_config:
                        self.quantization_check.setChecked(quant_config["use"])
                    if "precision" in quant_config:
                        self.quantization_precision_combo.setCurrentText(
                            quant_config["precision"]
                        )

                # Zapobieganie katastrofalnemu zapominaniu
                if "catastrophic_forgetting_prevention" in adv_config:
                    cfp_config = adv_config["catastrophic_forgetting_prevention"]
                    if "enable" in cfp_config:
                        self.prevent_forgetting_check.setChecked(cfp_config["enable"])
                    if "preserve_original_classes" in cfp_config:
                        self.preserve_classes_check.setChecked(
                            cfp_config["preserve_original_classes"]
                        )

                    # Rehearsal
                    if "rehearsal" in cfp_config:
                        rehearsal_config = cfp_config["rehearsal"]
                        if "use" in rehearsal_config:
                            self.rehearsal_check.setChecked(rehearsal_config["use"])
                        if "samples_per_class" in rehearsal_config:
                            self.samples_per_class_spin.setValue(
                                rehearsal_config["samples_per_class"]
                            )
                        if "synthetic_samples" in rehearsal_config:
                            self.synthetic_samples_check.setChecked(
                                rehearsal_config["synthetic_samples"]
                            )

                    # Knowledge Distillation
                    if "knowledge_distillation" in cfp_config:
                        kd_config = cfp_config["knowledge_distillation"]
                        if "use" in kd_config:
                            self.knowledge_distillation_check.setChecked(
                                kd_config["use"]
                            )
                        if "temperature" in kd_config:
                            self.kd_temperature_spin.setValue(kd_config["temperature"])
                        if "alpha" in kd_config:
                            self.kd_alpha_spin.setValue(kd_config["alpha"])

                    # EWC Regularization
                    if "ewc_regularization" in cfp_config:
                        ewc_config = cfp_config["ewc_regularization"]
                        if "use" in ewc_config:
                            self.ewc_check.setChecked(ewc_config["use"])
                        if "lambda" in ewc_config:
                            self.ewc_lambda_spin.setValue(ewc_config["lambda"])
                        if "fisher_sample_size" in ewc_config:
                            self.fisher_sample_size_spin.setValue(
                                ewc_config["fisher_sample_size"]
                            )
                        if "adaptive_lambda" in ewc_config:
                            self.adaptive_ewc_lambda_check.setChecked(
                                ewc_config["adaptive_lambda"]
                            )

                    # Layer Freezing
                    if "layer_freezing" in cfp_config:
                        lf_config = cfp_config["layer_freezing"]
                        if "strategy" in lf_config:
                            self.layer_freezing_combo.setCurrentText(
                                lf_config["strategy"]
                            )
                        if "freeze_ratio" in lf_config:
                            self.freeze_ratio_spin.setValue(lf_config["freeze_ratio"])

            # Aktualizacja zależnych kontrolek
            self._update_ui_state()

            QtWidgets.QMessageBox.information(
                self, "Sukces", "Profil został pomyślnie zastosowany."
            )
            self.logger.info(
                f"Profil {self.profile_list.currentItem().text()} "
                "został pomyślnie zastosowany."
            )

        except Exception as e:
            self.logger.error(
                f"Błąd podczas stosowania profilu: {str(e)}", exc_info=True
            )
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Nie można zastosować profilu: {str(e)}"
            )

    def _clone_profile(self):
        """Klonuje wybrany profil."""
        if not self.current_profile:
            QtWidgets.QMessageBox.warning(
                self, "Ostrzeżenie", "Najpierw wybierz profil do sklonowania."
            )
            return

        try:
            current_name = self.profile_list.currentItem().text()
            new_name, ok = QtWidgets.QInputDialog.getText(
                self,
                "Klonuj profil",
                "Podaj nazwę dla nowego profilu:",
                QtWidgets.QLineEdit.EchoMode.Normal,
                f"{current_name}_clone",
            )

            if ok and new_name:
                new_profile = self.current_profile.copy()
                new_profile["info"] = f"Klon profilu {current_name}"
                new_profile["description"] = f"Klon profilu {current_name}"
                new_profile["type"] = (
                    "fine_tuning"  # Upewniamy się, że typ jest ustawiony
                )

                new_path = self.profiles_dir / f"{new_name}.json"
                with open(new_path, "w", encoding="utf-8") as f:
                    json.dump(new_profile, f, indent=4, ensure_ascii=False)

                self._refresh_profile_list()
                QtWidgets.QMessageBox.information(
                    self, "Sukces", "Profil został pomyślnie sklonowany."
                )

        except Exception as e:
            self.logger.error(
                f"Błąd podczas klonowania profilu: {str(e)}", exc_info=True
            )
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Nie można sklonować profilu: {str(e)}"
            )

    def _save_profile(self):
        """Zapisuje aktualną konfigurację jako profil."""
        try:
            name, ok = QtWidgets.QInputDialog.getText(
                self,
                "Zapisz profil",
                "Podaj nazwę dla nowego profilu:",
                QtWidgets.QLineEdit.EchoMode.Normal,
                f"{self.arch_combo.currentText()}_{self.variant_combo.currentText()}",
            )

            if ok and name:
                config = self._get_complete_config()

                profile_data = {
                    "type": "fine_tuning",
                    "info": f"Profil dla {self.arch_combo.currentText()} {self.variant_combo.currentText()}",
                    "description": "Profil utworzony przez użytkownika",
                    "data_required": "Standardowe dane do doszkalania",
                    "hardware_required": "Standardowy sprzęt",
                    "config": config,
                }

                profile_path = self.profiles_dir / f"{name}.json"
                with open(profile_path, "w", encoding="utf-8") as f:
                    json.dump(profile_data, f, indent=4, ensure_ascii=False)

                self._refresh_profile_list()
                QtWidgets.QMessageBox.information(
                    self, "Sukces", "Profil został pomyślnie zapisany."
                )

        except Exception as e:
            self.logger.error(
                f"Błąd podczas zapisywania profilu: {str(e)}", exc_info=True
            )
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Nie można zapisać profilu: {str(e)}"
            )

    def _get_complete_config(self):
        """
        Zbiera pełną konfigurację ze wszystkich kontrolek UI.

        Returns:
            Słownik z pełną konfiguracją
        """
        return {
            "model": {
                "architecture": self.arch_combo.currentText(),
                "variant": self.variant_combo.currentText(),
                "input_size": self.input_size_spin.value(),
                "num_classes": self.num_classes_spin.value(),
                "pretrained": self.pretrained_check.isChecked(),
                "pretrained_weights": self.pretrained_weights_combo.currentText(),
                "feature_extraction_only": self.feature_extraction_check.isChecked(),
                "activation": self.activation_combo.currentText(),
                "dropout_at_inference": self.dropout_at_inference_check.isChecked(),
                "global_pool": self.global_pool_combo.currentText(),
                "last_layer_activation": self.last_layer_activation_combo.currentText(),
            },
            "training": {
                "epochs": self.epochs_spin.value(),
                "batch_size": self.batch_size_spin.value(),
                "learning_rate": float(self.lr_spin.value()),
                "optimizer": self.optimizer_combo.currentText(),
                "scheduler": self.scheduler_combo.currentText(),
                "num_workers": self.num_workers_spin.value(),
                "warmup_epochs": self.warmup_epochs_spin.value(),
                "warmup_lr_init": self.warmup_lr_init_spin.value(),
                "mixed_precision": self.mixed_precision_check.isChecked(),
                "gradient_accumulation_steps": self.grad_accum_steps_spin.value(),
                "gradient_clip": self.gradient_clip_spin.value(),
                "validation_split": self.validation_split_spin.value(),
                "evaluation_freq": self.eval_freq_spin.value(),
                "use_ema": self.use_ema_check.isChecked(),
                "ema_decay": self.ema_decay_spin.value(),
                "unfreeze_strategy": self.unfreeze_strategy_combo.currentText(),
                "unfreeze_after_epochs": self.unfreeze_after_epochs_spin.value(),
                "unfreeze_layers": self.unfreeze_layers_spin.value(),
                "frozen_lr": self.frozen_lr_spin.value(),
                "unfrozen_lr": self.unfrozen_lr_spin.value(),
            },
        }

    def get_task_config(self):
        """Zwraca konfigurację zadania lub None, jeśli nie dodano zadania."""
        return getattr(self, "task_config", None)

    def closeEvent(self, event):
        """Obsługa zamknięcia okna."""
        self.logger.info("Zamykanie okna dialogowego")
        self.accept()
        event.accept()

    def _get_config_value(self, config, key_path, default_value=None):
        """
        Bezpiecznie pobiera wartość z zagnieżdżonego słownika konfiguracji.

        Args:
            config: Słownik konfiguracji
            key_path: Lista kluczy do nawigacji w zagnieżdżonym słowniku
            default_value: Wartość domyślna, jeśli klucz nie istnieje

        Returns:
            Wartość z konfiguracji lub wartość domyślna
        """
        current = config
        for key in key_path:
            if not isinstance(current, dict) or key not in current:
                self.logger.warning(f"Klucz '{key}' nie istnieje w ścieżce {key_path}")
                return default_value
            current = current[key]
        return current

    def _with_blocked_signals(self, func):
        """
        Wykonuje funkcję z zablokowanymi sygnałami, gwarantując ich odblokowanie.

        Args:
            func: Funkcja do wykonania

        Returns:
            Wynik funkcji
        """
        was_blocked = self.signalsBlocked()
        self.blockSignals(True)
        try:
            return func()
        finally:
            self.blockSignals(was_blocked)

    def _load_config(self, config: Dict[str, Any]) -> None:
        """Ładuje konfigurację do interfejsu."""

        def load_config_impl():
            try:
                model_config = config.get("model", {})
                # ... existing code ...
                self._update_ui_state()
                self.logger.info("Konfiguracja modelu została pomyślnie załadowana")
            except Exception as e:
                raise e

        try:
            self._with_blocked_signals(load_config_impl)
        except Exception as e:
            msg = "Błąd podczas ładowania konfiguracji"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            QtWidgets.QMessageBox.critical(self, "Błąd", f"{msg}: {str(e)}")
        # ... existing code ...

    def _update_ui_state(self):
        """Aktualizuje stan UI po zmianie konfiguracji."""
        # Aktualizacja stanu kontrolek w zależności od wybranych opcji
        self._update_architecture_dependent_controls()
        self._update_training_dependent_controls()
        self._update_augmentation_dependent_controls()
        self._update_preprocessing_dependent_controls()
        self._update_monitoring_dependent_controls()

    def _update_dependent_controls(self):
        """Aktualizuje zależne kontrolki po zmianie konfiguracji."""
        # Aktualizacja kontrolek zależnych od architektury
        self._update_variant_combo(self.arch_combo.currentText())

        # Aktualizacja kontrolek zależnych od treningu
        self._update_optimizer_dependent_controls()
        self._update_scheduler_dependent_controls()

        # Aktualizacja kontrolek zależnych od augmentacji
        self._update_augmentation_dependent_controls()

        # Aktualizacja kontrolek zależnych od preprocessingu
        self._update_preprocessing_dependent_controls()

        # Aktualizacja kontrolek zależnych od monitorowania
        self._update_monitoring_dependent_controls()

    def _select_train_dir(self):
        """Wybiera katalog z danymi treningowymi."""
        try:
            dir_path = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                "Wybierz katalog z danymi treningowymi",
                str(Path.home()),
                QtWidgets.QFileDialog.Option.ShowDirsOnly,
            )
            if dir_path:
                if validate_training_directory(dir_path):
                    self.train_dir_edit.setText(dir_path)
                    self.logger.info(f"Wybrano katalog treningowy: {dir_path}")
                else:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Błąd walidacji",
                        "Wybrany katalog nie spełnia wymagań dla danych treningowych.",
                    )
        except Exception as e:
            self.logger.error(f"Błąd podczas wyboru katalogu treningowego: {str(e)}")
            QtWidgets.QMessageBox.critical(
                self,
                "Błąd",
                f"Nie można wybrać katalogu treningowego: {str(e)}",
            )

    def _select_val_dir(self):
        """Wybiera katalog z danymi walidacyjnymi."""
        try:
            dir_path = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                "Wybierz katalog z danymi walidacyjnymi",
                str(Path.home()),
                QtWidgets.QFileDialog.Option.ShowDirsOnly,
            )
            if dir_path:
                if validate_validation_directory(dir_path):
                    self.val_dir_edit.setText(dir_path)
                    self.logger.info(f"Wybrano katalog walidacyjny: {dir_path}")
                else:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Błąd walidacji",
                        "Wybrany katalog nie spełnia wymagań dla danych walidacyjnych.",
                    )
        except Exception as e:
            self.logger.error(f"Błąd podczas wyboru katalogu walidacyjnego: {str(e)}")
            QtWidgets.QMessageBox.critical(
                self,
                "Błąd",
                f"Nie można wybrać katalogu walidacyjnego: {str(e)}",
            )

    def _select_model_file(self):
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

                # Domyślnie deaktywuj przycisk tworzenia profilu
                self.create_profile_from_config_btn.setEnabled(False)
                self.config = None  # Wyczyść poprzednią konfigurację

                # Wczytaj plik konfiguracyjny
                config_path = os.path.splitext(file_path)[0] + "_config.json"
                self.logger.info(
                    f"Próba wczytania pliku konfiguracyjnego: {config_path}"
                )

                if os.path.exists(config_path):
                    try:
                        # Użyj funkcji read_config do odczytu konfiguracji
                        from app.gui.dialogs.read_config import read_config_file

                        temp_config_path = (
                            os.path.splitext(file_path)[0] + "_temp_config.json"
                        )
                        read_config_file(config_path, temp_config_path)

                        # Wczytaj wyodrębnioną konfigurację
                        with open(temp_config_path, "r") as f:
                            self.config = json.load(f)
                            self.logger.info(
                                f"Wczytana konfiguracja: {json.dumps(self.config, indent=2)}"
                            )
                            # Aktywuj przycisk tworzenia profilu
                            self.create_profile_from_config_btn.setEnabled(True)

                        # Usuń tymczasowy plik
                        os.remove(temp_config_path)

                    except Exception as e:
                        self.logger.error(
                            f"Błąd podczas wczytywania konfiguracji: {str(e)}"
                        )
                        QtWidgets.QMessageBox.warning(
                            self,
                            "Błąd",
                            f"Nie udało się wczytać konfiguracji modelu: {str(e)}",
                        )
                        # Upewnij się, że przycisk jest nieaktywny w razie błędu
                        self.create_profile_from_config_btn.setEnabled(False)
                        self.config = None
                else:
                    self.logger.warning(
                        f"Nie znaleziono pliku konfiguracyjnego: {config_path}"
                    )
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Ostrzeżenie",
                        "Nie znaleziono pliku konfiguracyjnego dla wybranego modelu.",
                    )
                    # Upewnij się, że przycisk jest nieaktywny
                    self.create_profile_from_config_btn.setEnabled(False)
                    self.config = None

        except Exception as e:
            self.logger.error(f"Błąd podczas wyboru pliku modelu: {str(e)}")
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Nie można wybrać pliku modelu: {str(e)}"
            )

    def _update_variant_combo(self, architecture: str) -> None:
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
            self.logger.info(
                f"Przywracam poprzednio wybrany wariant: {current_variant}"
            )
        else:
            self.logger.info(
                "Nie można przywrócić poprzedniego wariantu, ustawiam domyślny"
            )

    def _on_architecture_changed(self, architecture: str):
        """Obsługa zmiany architektury modelu."""
        self._update_variant_combo(architecture)

    def _create_augmentation_tab(self) -> QtWidgets.QWidget:
        """Tworzy zakładkę z parametrami augmentacji."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        # Augmentation
        aug_group = QtWidgets.QGroupBox("Augmentation")
        aug_layout = QtWidgets.QFormLayout()

        # Basic augmentation
        basic_group = QtWidgets.QGroupBox("Basic")
        basic_layout = QtWidgets.QFormLayout()

        self.basic_aug_check = QtWidgets.QCheckBox()
        self.basic_aug_check.setChecked(False)
        basic_layout.addRow("Use basic augmentation:", self.basic_aug_check)

        # Rotation
        self.rotation_spin = QtWidgets.QSpinBox()
        self.rotation_spin.setRange(0, 360)
        self.rotation_spin.setValue(30)
        basic_layout.addRow("Rotation:", self.rotation_spin)

        # Brightness
        self.brightness_spin = QtWidgets.QDoubleSpinBox()
        self.brightness_spin.setRange(0.0, 1.0)
        self.brightness_spin.setValue(0.2)
        self.brightness_spin.setDecimals(2)
        basic_layout.addRow("Brightness:", self.brightness_spin)

        # Shift
        self.shift_spin = QtWidgets.QDoubleSpinBox()
        self.shift_spin.setRange(0.0, 1.0)
        self.shift_spin.setValue(0.1)
        self.shift_spin.setDecimals(2)
        basic_layout.addRow("Shift:", self.shift_spin)

        # Zoom
        self.zoom_spin = QtWidgets.QDoubleSpinBox()
        self.zoom_spin.setRange(0.0, 1.0)
        self.zoom_spin.setValue(0.1)
        self.zoom_spin.setDecimals(2)
        basic_layout.addRow("Zoom:", self.zoom_spin)

        # Horizontal flip
        self.horizontal_flip_check = QtWidgets.QCheckBox()
        self.horizontal_flip_check.setChecked(True)
        basic_layout.addRow("Horizontal flip:", self.horizontal_flip_check)

        # Vertical flip
        self.vertical_flip_check = QtWidgets.QCheckBox()
        self.vertical_flip_check.setChecked(False)
        basic_layout.addRow("Vertical flip:", self.vertical_flip_check)

        basic_group.setLayout(basic_layout)
        aug_layout.addRow(basic_group)

        # Mixup
        mixup_group = QtWidgets.QGroupBox("Mixup")
        mixup_layout = QtWidgets.QFormLayout()

        self.mixup_check = QtWidgets.QCheckBox()
        self.mixup_check.setChecked(False)
        mixup_layout.addRow("Use Mixup:", self.mixup_check)

        self.mixup_alpha_spin = QtWidgets.QDoubleSpinBox()
        self.mixup_alpha_spin.setRange(0.0, 1.0)
        self.mixup_alpha_spin.setValue(0.2)
        self.mixup_alpha_spin.setDecimals(2)
        mixup_layout.addRow("Alpha:", self.mixup_alpha_spin)

        mixup_group.setLayout(mixup_layout)
        aug_layout.addRow(mixup_group)

        # CutMix
        cutmix_group = QtWidgets.QGroupBox("CutMix")
        cutmix_layout = QtWidgets.QFormLayout()

        self.cutmix_check = QtWidgets.QCheckBox()
        self.cutmix_check.setChecked(False)
        cutmix_layout.addRow("Use CutMix:", self.cutmix_check)

        self.cutmix_alpha_spin = QtWidgets.QDoubleSpinBox()
        self.cutmix_alpha_spin.setRange(0.0, 1.0)
        self.cutmix_alpha_spin.setValue(0.2)
        self.cutmix_alpha_spin.setDecimals(2)
        cutmix_layout.addRow("Alpha:", self.cutmix_alpha_spin)

        cutmix_group.setLayout(cutmix_layout)
        aug_layout.addRow(cutmix_group)

        # AutoAugment
        autoaugment_group = QtWidgets.QGroupBox("AutoAugment")
        autoaugment_layout = QtWidgets.QFormLayout()
        autoaugment_layout.addRow("Use AutoAugment:", self.autoaugment_check)
        autoaugment_group.setLayout(autoaugment_layout)
        aug_layout.addRow(autoaugment_group)

        # RandAugment
        randaugment_group = QtWidgets.QGroupBox("RandAugment")
        randaugment_layout = QtWidgets.QFormLayout()
        randaugment_layout.addRow("Use RandAugment:", self.randaugment_check)
        randaugment_layout.addRow("N (num_ops):", self.randaugment_n_spin)
        randaugment_layout.addRow("M (magnitude):", self.randaugment_m_spin)
        randaugment_group.setLayout(randaugment_layout)
        aug_layout.addRow(randaugment_group)

        # Advanced augmentation
        advanced_group = QtWidgets.QGroupBox("Advanced")
        advanced_layout = QtWidgets.QFormLayout()

        # Contrast
        self.contrast_spin = QtWidgets.QDoubleSpinBox()
        self.contrast_spin.setRange(0.0, 1.0)
        self.contrast_spin.setValue(0.2)
        self.contrast_spin.setDecimals(2)
        advanced_layout.addRow("Contrast:", self.contrast_spin)

        # Saturation
        self.saturation_spin = QtWidgets.QDoubleSpinBox()
        self.saturation_spin.setRange(0.0, 1.0)
        self.saturation_spin.setValue(0.2)
        self.saturation_spin.setDecimals(2)
        advanced_layout.addRow("Saturation:", self.saturation_spin)

        # Hue
        self.hue_spin = QtWidgets.QDoubleSpinBox()
        self.hue_spin.setRange(0.0, 0.5)
        self.hue_spin.setValue(0.1)
        self.hue_spin.setDecimals(2)
        advanced_layout.addRow("Hue:", self.hue_spin)

        # Shear
        self.shear_spin = QtWidgets.QDoubleSpinBox()
        self.shear_spin.setRange(0.0, 1.0)
        self.shear_spin.setValue(0.1)
        self.shear_spin.setDecimals(2)
        advanced_layout.addRow("Shear:", self.shear_spin)

        # Channel shift
        self.channel_shift_spin = QtWidgets.QDoubleSpinBox()
        self.channel_shift_spin.setRange(0.0, 1.0)
        self.channel_shift_spin.setValue(0.0)
        self.channel_shift_spin.setDecimals(2)
        advanced_layout.addRow("Channel shift:", self.channel_shift_spin)

        advanced_group.setLayout(advanced_layout)
        aug_layout.addRow(advanced_group)

        # Resize mode
        self.resize_mode_combo = QtWidgets.QComboBox()
        self.resize_mode_combo.addItems(["bilinear", "bicubic", "nearest", "area"])
        aug_layout.addRow("Resize mode:", self.resize_mode_combo)

        # Normalization
        norm_group = QtWidgets.QGroupBox("Normalization")
        norm_layout = QtWidgets.QFormLayout()

        # Mean
        mean_layout = QtWidgets.QHBoxLayout()
        mean_layout.addWidget(self.norm_mean_r)
        mean_layout.addWidget(self.norm_mean_g)
        mean_layout.addWidget(self.norm_mean_b)
        norm_layout.addRow("Mean (RGB):", mean_layout)

        # Std
        std_layout = QtWidgets.QHBoxLayout()
        std_layout.addWidget(self.norm_std_r)
        std_layout.addWidget(self.norm_std_g)
        std_layout.addWidget(self.norm_std_b)
        norm_layout.addRow("Std (RGB):", std_layout)

        norm_group.setLayout(norm_layout)
        aug_layout.addRow(norm_group)

        aug_group.setLayout(aug_layout)
        layout.addWidget(aug_group)

        tab.setLayout(layout)
        return tab

    def _create_preprocessing_tab(self) -> QtWidgets.QWidget:
        """Tworzy zakładkę z parametrami preprocessingu."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        form = QtWidgets.QFormLayout()

        # Normalizacja
        form.addRow("Normalizacja:", self.normalization_combo)

        norm_group = QtWidgets.QGroupBox("Normalizacja")
        norm_layout = QtWidgets.QFormLayout()

        # Mean
        mean_layout = QtWidgets.QHBoxLayout()
        self.norm_mean_r = QtWidgets.QDoubleSpinBox()
        self.norm_mean_r.setRange(0.0, 1.0)
        self.norm_mean_r.setValue(0.485)
        self.norm_mean_r.setDecimals(3)
        mean_layout.addWidget(self.norm_mean_r)

        self.norm_mean_g = QtWidgets.QDoubleSpinBox()
        self.norm_mean_g.setRange(0.0, 1.0)
        self.norm_mean_g.setValue(0.456)
        self.norm_mean_g.setDecimals(3)
        mean_layout.addWidget(self.norm_mean_g)

        self.norm_mean_b = QtWidgets.QDoubleSpinBox()
        self.norm_mean_b.setRange(0.0, 1.0)
        self.norm_mean_b.setValue(0.406)
        self.norm_mean_b.setDecimals(3)
        mean_layout.addWidget(self.norm_mean_b)

        norm_layout.addRow("Mean (RGB):", mean_layout)

        # Std
        std_layout = QtWidgets.QHBoxLayout()
        self.norm_std_r = QtWidgets.QDoubleSpinBox()
        self.norm_std_r.setRange(0.0, 1.0)
        self.norm_std_r.setValue(0.229)
        self.norm_std_r.setDecimals(3)
        std_layout.addWidget(self.norm_std_r)

        self.norm_std_g = QtWidgets.QDoubleSpinBox()
        self.norm_std_g.setRange(0.0, 1.0)
        self.norm_std_g.setValue(0.224)
        self.norm_std_g.setDecimals(3)
        std_layout.addWidget(self.norm_std_g)

        self.norm_std_b = QtWidgets.QDoubleSpinBox()
        self.norm_std_b.setRange(0.0, 1.0)
        self.norm_std_b.setValue(0.225)
        self.norm_std_b.setDecimals(3)
        std_layout.addWidget(self.norm_std_b)

        norm_layout.addRow("Std (RGB):", std_layout)

        norm_group.setLayout(norm_layout)
        form.addRow(norm_group)

        # Resize mode
        self.resize_mode_combo = QtWidgets.QComboBox()
        self.resize_mode_combo.addItems(["bilinear", "bicubic", "nearest", "area"])
        form.addRow("Resize mode:", self.resize_mode_combo)

        # Cache dataset
        self.cache_dataset_check = QtWidgets.QCheckBox()
        self.cache_dataset_check.setChecked(False)
        form.addRow("Cache dataset:", self.cache_dataset_check)

        layout.addLayout(form)
        tab.setLayout(layout)
        return tab

    def _create_monitoring_tab(self) -> QtWidgets.QWidget:
        """Tworzy zakładkę z parametrami monitorowania."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        form = QtWidgets.QFormLayout()

        # Metrics
        metrics_group = QtWidgets.QGroupBox("Metrics")
        metrics_layout = QtWidgets.QFormLayout()

        metrics_layout.addRow("Accuracy:", self.accuracy_check)
        metrics_layout.addRow("Precision:", self.precision_check)
        metrics_layout.addRow("Recall:", self.recall_check)
        metrics_layout.addRow("F1:", self.f1_check)
        metrics_layout.addRow("Top-K accuracy:", self.topk_check)
        metrics_layout.addRow("Confusion matrix:", self.confusion_matrix_check)
        metrics_layout.addRow("AUC:", self.auc_check)

        metrics_group.setLayout(metrics_layout)
        form.addRow(metrics_group)

        # Logging
        logging_group = QtWidgets.QGroupBox("Logging")
        logging_layout = QtWidgets.QFormLayout()

        logging_layout.addRow("Use TensorBoard:", self.use_tensorboard_check)
        logging_layout.addRow("Use Weights & Biases:", self.use_wandb_check)
        logging_layout.addRow("Save to CSV:", self.use_csv_check)
        logging_layout.addRow("Logging frequency:", self.log_freq_combo)

        logging_group.setLayout(logging_layout)
        form.addRow(logging_group)

        # Visualization
        visualization_group = QtWidgets.QGroupBox("Visualization")
        visualization_layout = QtWidgets.QFormLayout()

        self.use_gradcam_check = QtWidgets.QCheckBox()
        self.use_gradcam_check.setChecked(True)
        visualization_layout.addRow("Use GradCAM:", self.use_gradcam_check)

        self.use_feature_maps_check = QtWidgets.QCheckBox()
        self.use_feature_maps_check.setChecked(True)
        visualization_layout.addRow("Use Feature Maps:", self.use_feature_maps_check)

        self.use_pred_samples_check = QtWidgets.QCheckBox()
        self.use_pred_samples_check.setChecked(True)
        visualization_layout.addRow(
            "Use Prediction Samples:", self.use_pred_samples_check
        )

        self.num_samples_spin = QtWidgets.QSpinBox()
        self.num_samples_spin.setRange(1, 100)
        self.num_samples_spin.setValue(10)
        visualization_layout.addRow("Number of samples:", self.num_samples_spin)

        visualization_group.setLayout(visualization_layout)
        form.addRow(visualization_group)

        # Early stopping
        early_stop_group = QtWidgets.QGroupBox("Early stopping")
        early_stop_layout = QtWidgets.QFormLayout()

        self.patience_spin = QtWidgets.QSpinBox()
        self.patience_spin.setRange(1, 100)
        self.patience_spin.setValue(10)
        early_stop_layout.addRow("Patience:", self.patience_spin)

        self.min_delta_spin = QtWidgets.QDoubleSpinBox()
        self.min_delta_spin.setRange(0.0, 1.0)
        self.min_delta_spin.setValue(0.001)
        self.min_delta_spin.setDecimals(4)
        early_stop_layout.addRow("Min delta:", self.min_delta_spin)

        self.monitor_combo = QtWidgets.QComboBox()
        self.monitor_combo.addItems(["val_loss", "val_accuracy"])
        early_stop_layout.addRow("Monitor:", self.monitor_combo)

        early_stop_group.setLayout(early_stop_layout)
        form.addRow(early_stop_group)

        # Checkpointing
        checkpoint_group = QtWidgets.QGroupBox("Checkpointing")
        checkpoint_layout = QtWidgets.QFormLayout()

        self.best_only_check = QtWidgets.QCheckBox()
        self.best_only_check.setChecked(True)
        checkpoint_layout.addRow("Save best only:", self.best_only_check)

        self.save_freq_spin = QtWidgets.QSpinBox()
        self.save_freq_spin.setRange(1, 100)
        self.save_freq_spin.setValue(1)
        checkpoint_layout.addRow("Save frequency:", self.save_freq_spin)

        self.checkpoint_metric_combo = QtWidgets.QComboBox()
        self.checkpoint_metric_combo.addItems(["val_loss", "val_accuracy"])
        checkpoint_layout.addRow("Metric:", self.checkpoint_metric_combo)

        checkpoint_group.setLayout(checkpoint_layout)
        form.addRow(checkpoint_group)

        layout.addLayout(form)
        tab.setLayout(layout)
        return tab

    def _create_advanced_tab(self) -> QtWidgets.QWidget:
        """Tworzy zakładkę z zaawansowanymi parametrami."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        # Nowa grupa: Parametry zaawansowane
        advanced_group = QtWidgets.QGroupBox("Parametry zaawansowane")
        advanced_layout = QtWidgets.QFormLayout()
        advanced_layout.addRow("Seed:", self.seed_spin)
        advanced_layout.addRow("Deterministyczny trening:", self.deterministic_check)
        advanced_layout.addRow("Wagi klas:", self.class_weights_combo)
        advanced_layout.addRow("Sampler:", self.sampler_combo)
        advanced_layout.addRow("Liczba kanałów obrazu:", self.image_channels_spin)
        advanced_layout.addRow("Test Time Augmentation (TTA):", self.use_tta_check)
        advanced_layout.addRow("Liczba augmentacji TTA:", self.tta_num_samples_spin)
        advanced_layout.addRow("Eksportuj do ONNX:", self.export_onnx_check)
        advanced_layout.addRow("Kwantyzacja:", self.quantization_check)
        advanced_layout.addRow(
            "Precyzja kwantyzacji:", self.quantization_precision_combo
        )
        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)

        # Grupa: Zapobieganie katastrofalnemu zapominaniu (istniejąca)
        forgetting_group = QtWidgets.QGroupBox(
            "Zapobieganie katastrofalnemu zapominaniu"
        )
        forgetting_layout = QtWidgets.QFormLayout()
        self.prevent_forgetting_check.stateChanged.connect(
            self._update_forgetting_controls
        )
        forgetting_layout.addRow(
            "Włącz zapobieganie zapominaniu:", self.prevent_forgetting_check
        )
        forgetting_layout.addRow(
            "Zachowaj oryginalne klasy:", self.preserve_classes_check
        )
        forgetting_layout.addRow("Użyj rehearsal:", self.rehearsal_check)
        forgetting_layout.addRow(
            "Liczba przykładów na klasę:", self.samples_per_class_spin
        )
        forgetting_layout.addRow(
            "Użyj próbek syntetycznych:", self.synthetic_samples_check
        )
        forgetting_layout.addRow(
            "Użyj knowledge distillation:", self.knowledge_distillation_check
        )
        forgetting_layout.addRow("Temperatura:", self.kd_temperature_spin)
        forgetting_layout.addRow("Alpha:", self.kd_alpha_spin)
        forgetting_layout.addRow("Użyj EWC regularization:", self.ewc_check)
        forgetting_layout.addRow("Lambda:", self.ewc_lambda_spin)
        forgetting_layout.addRow("Fisher Sample Size:", self.fisher_sample_size_spin)
        forgetting_layout.addRow("Strategia zamrażania:", self.layer_freezing_combo)
        forgetting_layout.addRow("Współczynnik zamrażania:", self.freeze_ratio_spin)
        forgetting_group.setLayout(forgetting_layout)
        layout.addWidget(forgetting_group)

        # Dodanie nowej kontrolki do układu
        forgetting_layout.addRow(
            "Adaptacyjna Lambda EWC:", self.adaptive_ewc_lambda_check
        )

        tab.setLayout(layout)
        return tab

    def _update_forgetting_controls(self):
        """Aktualizuje stan kontrolek związanych z zapobieganiem katastrofalnemu zapominaniu."""
        enabled = self.prevent_forgetting_check.isChecked()

        # Włącz/wyłącz wszystkie kontrolki w zależności od stanu głównej opcji
        self.preserve_classes_check.setEnabled(enabled)
        self.rehearsal_check.setEnabled(enabled)
        self.samples_per_class_spin.setEnabled(
            enabled and self.rehearsal_check.isChecked()
        )
        self.synthetic_samples_check.setEnabled(
            enabled and self.rehearsal_check.isChecked()
        )
        self.knowledge_distillation_check.setEnabled(enabled)
        self.kd_temperature_spin.setEnabled(
            enabled and self.knowledge_distillation_check.isChecked()
        )
        self.kd_alpha_spin.setEnabled(
            enabled and self.knowledge_distillation_check.isChecked()
        )
        self.ewc_check.setEnabled(enabled)
        self.ewc_lambda_spin.setEnabled(enabled and self.ewc_check.isChecked())
        self.fisher_sample_size_spin.setEnabled(enabled and self.ewc_check.isChecked())
        self.layer_freezing_combo.setEnabled(enabled)
        self.freeze_ratio_spin.setEnabled(enabled)

    def _create_fine_tuning_params_tab(self) -> QtWidgets.QWidget:
        """Tworzy zakładkę z parametrami fine-tuningu."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        form = QtWidgets.QFormLayout()

        # Podstawowe parametry treningu
        basic_group = QtWidgets.QGroupBox("Podstawowe parametry")
        basic_layout = QtWidgets.QFormLayout()

        # Liczba epok
        self.epochs_spin = QtWidgets.QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(100)
        basic_layout.addRow("Liczba epok:", self.epochs_spin)

        # Learning rate
        self.lr_spin = QtWidgets.QDoubleSpinBox()
        self.lr_spin.setRange(0.000001, 1.0)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setDecimals(6)
        basic_layout.addRow("Learning rate:", self.lr_spin)

        # Optimizer
        self.optimizer_combo = QtWidgets.QComboBox()
        self.optimizer_combo.addItems(["Adam", "AdamW", "SGD", "RMSprop"])
        basic_layout.addRow("Optimizer:", self.optimizer_combo)

        # Scheduler
        self.scheduler_combo = QtWidgets.QComboBox()
        self.scheduler_combo.addItems(
            ["None", "CosineAnnealingLR", "ReduceLROnPlateau", "OneCycleLR"]
        )
        basic_layout.addRow("Scheduler:", self.scheduler_combo)

        # Warmup epochs
        self.warmup_epochs_spin = QtWidgets.QSpinBox()
        self.warmup_epochs_spin.setRange(0, 100)
        self.warmup_epochs_spin.setValue(5)
        basic_layout.addRow("Warmup epochs:", self.warmup_epochs_spin)

        # Warmup learning rate init
        self.warmup_lr_init_spin = QtWidgets.QDoubleSpinBox()
        self.warmup_lr_init_spin.setRange(0.000001, 0.1)
        self.warmup_lr_init_spin.setValue(0.000001)
        self.warmup_lr_init_spin.setDecimals(6)
        basic_layout.addRow("Warmup LR init:", self.warmup_lr_init_spin)

        # Validation split
        self.validation_split_spin = QtWidgets.QDoubleSpinBox()
        self.validation_split_spin.setRange(0.0, 0.5)
        self.validation_split_spin.setValue(0.2)
        self.validation_split_spin.setDecimals(2)
        basic_layout.addRow("Validation split:", self.validation_split_spin)

        # Evaluation frequency
        self.eval_freq_spin = QtWidgets.QSpinBox()
        self.eval_freq_spin.setRange(1, 100)
        self.eval_freq_spin.setValue(1)
        basic_layout.addRow("Evaluation frequency:", self.eval_freq_spin)

        # EMA
        self.use_ema_check = QtWidgets.QCheckBox()
        self.use_ema_check.setChecked(False)
        basic_layout.addRow("Use EMA:", self.use_ema_check)

        # EMA decay
        self.ema_decay_spin = QtWidgets.QDoubleSpinBox()
        self.ema_decay_spin.setRange(0.0, 1.0)
        self.ema_decay_spin.setValue(0.9999)
        self.ema_decay_spin.setDecimals(4)
        basic_layout.addRow("EMA decay:", self.ema_decay_spin)

        basic_group.setLayout(basic_layout)
        form.addRow(basic_group)

        # Strategia odmrażania warstw
        unfreeze_group = QtWidgets.QGroupBox("Strategia odmrażania")
        unfreeze_layout = QtWidgets.QFormLayout()

        self.unfreeze_strategy_combo = QtWidgets.QComboBox()
        self.unfreeze_strategy_combo.addItems(
            [
                self.UNFREEZE_ALL,
                self.UNFREEZE_GRADUAL_END,
                self.UNFREEZE_GRADUAL_START,
                self.UNFREEZE_AFTER_EPOCHS,
            ]
        )
        unfreeze_layout.addRow("Strategia odmrażania:", self.unfreeze_strategy_combo)

        # Liczba epok przed odmrożeniem
        self.unfreeze_after_epochs_spin = QtWidgets.QSpinBox()
        self.unfreeze_after_epochs_spin.setRange(1, 100)
        self.unfreeze_after_epochs_spin.setValue(5)
        unfreeze_layout.addRow("Odmroź po epokach:", self.unfreeze_after_epochs_spin)

        # Liczba warstw do odmrożenia
        self.unfreeze_layers_spin = QtWidgets.QSpinBox()
        self.unfreeze_layers_spin.setRange(1, 100)
        self.unfreeze_layers_spin.setValue(3)
        unfreeze_layout.addRow(
            "Liczba warstw do odmrożenia:", self.unfreeze_layers_spin
        )

        # Learning rate dla zamrożonych warstw
        self.frozen_lr_spin = QtWidgets.QDoubleSpinBox()
        self.frozen_lr_spin.setRange(0.0, 0.1)
        self.frozen_lr_spin.setValue(0.0001)
        self.frozen_lr_spin.setDecimals(6)
        unfreeze_layout.addRow("LR dla zamrożonych warstw:", self.frozen_lr_spin)

        # Learning rate dla odmrożonych warstw
        self.unfrozen_lr_spin = QtWidgets.QDoubleSpinBox()
        self.unfrozen_lr_spin.setRange(0.0, 0.1)
        self.unfrozen_lr_spin.setValue(0.001)
        self.unfrozen_lr_spin.setDecimals(6)
        unfreeze_layout.addRow("LR dla odmrożonych warstw:", self.unfrozen_lr_spin)

        unfreeze_group.setLayout(unfreeze_layout)
        form.addRow(unfreeze_group)

        # Zaawansowane parametry
        advanced_group = QtWidgets.QGroupBox("Zaawansowane parametry")
        advanced_layout = QtWidgets.QFormLayout()

        # Gradient clipping
        self.gradient_clip_spin = QtWidgets.QDoubleSpinBox()
        self.gradient_clip_spin.setRange(0.0, 10.0)
        self.gradient_clip_spin.setValue(1.0)
        self.gradient_clip_spin.setDecimals(2)
        advanced_layout.addRow("Gradient clipping:", self.gradient_clip_spin)

        advanced_group.setLayout(advanced_layout)
        form.addRow(advanced_group)

        layout.addLayout(form)
        tab.setLayout(layout)
        return tab

    def _create_regularization_tab(self) -> QtWidgets.QWidget:
        """Tworzy zakładkę z parametrami regularyzacji."""
        try:
            self.logger.debug("Tworzenie zakładki regularyzacji")
            tab = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(tab)
            form = QtWidgets.QFormLayout()

            # Weight decay
            self.weight_decay_spin = QtWidgets.QDoubleSpinBox()
            self.weight_decay_spin.setDecimals(6)
            self.weight_decay_spin.setRange(0.0, 1.0)
            self.weight_decay_spin.setSingleStep(0.0001)
            self.weight_decay_spin.setValue(0.0001)
            form.addRow("Weight Decay:", self.weight_decay_spin)

            # Gradient clipping
            self.gradient_clip_spin = QtWidgets.QDoubleSpinBox()
            self.gradient_clip_spin.setRange(0.0, 10.0)
            self.gradient_clip_spin.setDecimals(3)
            self.gradient_clip_spin.setValue(1.0)
            form.addRow("Gradient Clipping:", self.gradient_clip_spin)

            # Label smoothing
            self.label_smoothing_spin = QtWidgets.QDoubleSpinBox()
            self.label_smoothing_spin.setRange(0.0, 0.5)
            self.label_smoothing_spin.setDecimals(3)
            self.label_smoothing_spin.setValue(0.1)
            form.addRow("Label Smoothing:", self.label_smoothing_spin)

            # Drop connect rate
            self.drop_connect_spin = QtWidgets.QDoubleSpinBox()
            self.drop_connect_spin.setRange(0.0, 0.5)
            self.drop_connect_spin.setDecimals(3)
            self.drop_connect_spin.setValue(0.2)
            form.addRow("Drop Connect Rate:", self.drop_connect_spin)

            # Dropout rate
            self.dropout_spin = QtWidgets.QDoubleSpinBox()
            self.dropout_spin.setRange(0.0, 0.5)
            self.dropout_spin.setDecimals(3)
            self.dropout_spin.setValue(0.2)
            form.addRow("Dropout Rate:", self.dropout_spin)

            # Momentum
            self.momentum_spin = QtWidgets.QDoubleSpinBox()
            self.momentum_spin.setRange(0.0, 1.0)
            self.momentum_spin.setDecimals(3)
            self.momentum_spin.setValue(0.9)
            form.addRow("Momentum:", self.momentum_spin)

            # Epsilon
            self.epsilon_spin = QtWidgets.QDoubleSpinBox()
            self.epsilon_spin.setRange(1e-8, 1e-3)
            self.epsilon_spin.setDecimals(8)
            self.epsilon_spin.setValue(1e-6)
            form.addRow("Epsilon:", self.epsilon_spin)

            # SWA
            swa_group = QtWidgets.QGroupBox("Stochastic Weight Averaging")
            swa_layout = QtWidgets.QFormLayout()

            self.use_swa_check = QtWidgets.QCheckBox("Używaj SWA")
            self.swa_start_epoch_spin = QtWidgets.QSpinBox()
            self.swa_start_epoch_spin.setRange(1, 1000)
            self.swa_start_epoch_spin.setValue(10)

            swa_layout.addRow("", self.use_swa_check)
            swa_layout.addRow("Epoka rozpoczęcia:", self.swa_start_epoch_spin)
            swa_group.setLayout(swa_layout)

            # Stochastic Depth
            stoch_depth_group = QtWidgets.QGroupBox("Stochastic Depth")
            stoch_depth_layout = QtWidgets.QFormLayout()

            self.use_stoch_depth_check = QtWidgets.QCheckBox("Używaj Stochastic Depth")
            self.stoch_depth_drop_rate = QtWidgets.QDoubleSpinBox()
            self.stoch_depth_drop_rate.setRange(0.0, 0.5)
            self.stoch_depth_drop_rate.setDecimals(3)
            self.stoch_depth_drop_rate.setValue(0.2)
            self.stoch_depth_survival_prob = QtWidgets.QDoubleSpinBox()
            self.stoch_depth_survival_prob.setRange(0.5, 1.0)
            self.stoch_depth_survival_prob.setDecimals(3)
            self.stoch_depth_survival_prob.setValue(0.8)

            stoch_depth_layout.addRow("", self.use_stoch_depth_check)
            stoch_depth_layout.addRow("Drop rate:", self.stoch_depth_drop_rate)
            stoch_depth_layout.addRow(
                "Survival probability:", self.stoch_depth_survival_prob
            )
            stoch_depth_group.setLayout(stoch_depth_layout)

            # Random Erase
            random_erase_group = QtWidgets.QGroupBox("Random Erase")
            random_erase_layout = QtWidgets.QFormLayout()

            self.use_random_erase_check = QtWidgets.QCheckBox("Używaj Random Erase")
            self.random_erase_prob = QtWidgets.QDoubleSpinBox()
            self.random_erase_prob.setRange(0.0, 1.0)
            self.random_erase_prob.setDecimals(3)
            self.random_erase_prob.setValue(0.25)
            self.random_erase_mode_combo = QtWidgets.QComboBox()
            self.random_erase_mode_combo.addItems(["pixel", "block"])

            random_erase_layout.addRow("", self.use_random_erase_check)
            random_erase_layout.addRow("Probability:", self.random_erase_prob)
            random_erase_layout.addRow("Mode:", self.random_erase_mode_combo)
            random_erase_group.setLayout(random_erase_layout)

            layout.addLayout(form)
            layout.addWidget(swa_group)
            layout.addWidget(stoch_depth_group)
            layout.addWidget(random_erase_group)
            return tab

        except Exception as e:
            msg = "Błąd podczas tworzenia zakładki"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            raise

    def _validate_directory(self, dir_path, is_training=True):
        """
        Waliduje katalog z danymi treningowymi lub walidacyjnymi.

        Args:
            dir_path: Ścieżka do katalogu
            is_training: Czy to katalog treningowy (True) czy walidacyjny (False)

        Returns:
            Tuple: (is_valid, message, subdirs) - czy katalog jest poprawny,
            ewentualny komunikat błędu, lista podkatalogów
        """
        if not os.path.isdir(dir_path):
            return False, f"Katalog nie istnieje: {dir_path}", []

        subdirs = [
            d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))
        ]

        if not subdirs:
            dir_type = "treningowy" if is_training else "walidacyjny"
            return (
                False,
                f"Katalog {dir_type} nie zawiera żadnych podfolderów (klas)",
                [],
            )

        return True, "", subdirs

    def _on_accept(self):
        """Obsługa zatwierdzenia konfiguracji."""
        try:
            # Sprawdź czy nazwa zadania jest pusta
            task_name = self.name_edit.text().strip()
            if not task_name:
                QtWidgets.QMessageBox.warning(
                    self, "Błąd", "Nazwa zadania nie może być pusta."
                )
                return

            # Sprawdź czy model bazowy jest wybrany
            base_model_path = self.model_path_edit.text().strip()
            if not base_model_path:
                QtWidgets.QMessageBox.warning(
                    self, "Błąd", "Musisz wybrać model bazowy."
                )
                return

            # Sprawdź czy katalog treningowy jest ustawiony
            training_dir = self.train_dir_edit.text().strip()
            if not training_dir:
                QtWidgets.QMessageBox.warning(
                    self, "Błąd", "Katalog treningowy nie może być pusty."
                )
                return
            # WALIDACJA katalogu treningowego
            is_valid, error_msg, train_subdirs = self._validate_directory(
                training_dir, True
            )
            if not is_valid:
                QtWidgets.QMessageBox.critical(self, "Błąd", error_msg)
                return

            # Sprawdź czy katalog walidacyjny jest ustawiony
            validation_dir = self.val_dir_edit.text().strip()
            if not validation_dir:
                QtWidgets.QMessageBox.warning(
                    self, "Błąd", "Katalog walidacyjny nie może być pusty."
                )
                return
            # WALIDACJA katalogu walidacyjnego
            is_valid, error_msg, val_subdirs = self._validate_directory(
                validation_dir, False
            )
            if not is_valid:
                QtWidgets.QMessageBox.critical(self, "Błąd", error_msg)
                return

            # Sprawdź zgodność liczby klas z liczbą katalogów
            num_classes = self.num_classes_spin.value()
            if len(train_subdirs) != num_classes:
                self.logger.warning(
                    f"Liczba katalogów ({len(train_subdirs)}) nie zgadza się z podaną liczbą klas ({num_classes})"
                )
                result = QtWidgets.QMessageBox.warning(
                    self,
                    "Niezgodność liczby klas",
                    f"Liczba podkatalogów w katalogu treningowym ({len(train_subdirs)}) nie zgadza się z "
                    f"podaną liczbą klas ({num_classes}). Czy chcesz kontynuować?",
                    QtWidgets.QMessageBox.StandardButton.Yes
                    | QtWidgets.QMessageBox.StandardButton.No,
                    QtWidgets.QMessageBox.StandardButton.No,
                )
                if result == QtWidgets.QMessageBox.StandardButton.No:
                    return

            # Dodaj logi
            self.logger.info("=== TWORZENIE NOWEGO ZADANIA FINE-TUNINGU ===")
            self.logger.info(f"Nazwa zadania: {task_name}")

            config = {
                "base_model": base_model_path,
                "train_dir": training_dir,
                "val_dir": validation_dir,
                "model": {
                    "architecture": self.arch_combo.currentText(),
                    "variant": self.variant_combo.currentText(),
                    "input_size": self.input_size_spin.value(),
                    "num_classes": self.num_classes_spin.value(),
                },
                "training": {
                    "epochs": self.epochs_spin.value(),
                    "learning_rate": float(self.lr_spin.value()),
                    "optimizer": self.optimizer_combo.currentText(),
                    "scheduler": self.scheduler_combo.currentText(),
                    "warmup_epochs": self.warmup_epochs_spin.value(),
                    "warmup_lr_init": self.warmup_lr_init_spin.value(),
                    "gradient_clip": self.gradient_clip_spin.value(),
                    "validation_split": self.validation_split_spin.value(),
                    "evaluation_freq": self.eval_freq_spin.value(),
                    "use_ema": self.use_ema_check.isChecked(),
                    "ema_decay": self.ema_decay_spin.value(),
                    "unfreeze_strategy": self.unfreeze_strategy_combo.currentText(),
                    "unfreeze_after_epochs": self.unfreeze_after_epochs_spin.value(),
                    "unfreeze_layers": self.unfreeze_layers_spin.value(),
                    "frozen_lr": self.frozen_lr_spin.value(),
                    "unfrozen_lr": self.unfrozen_lr_spin.value(),
                },
                "regularization": {
                    "weight_decay": float(self.weight_decay_spin.value()),
                    "gradient_clip": self.gradient_clip_spin.value(),
                    "label_smoothing": self.label_smoothing_spin.value(),
                    "drop_connect_rate": self.drop_connect_spin.value(),
                    "dropout_rate": self.dropout_spin.value(),
                    "momentum": self.momentum_spin.value(),
                    "epsilon": self.epsilon_spin.value(),
                    "swa": {
                        "use": self.use_swa_check.isChecked(),
                        "start_epoch": self.swa_start_epoch_spin.value(),
                    },
                },
                "augmentation": {
                    "basic": {
                        "use": self.basic_aug_check.isChecked(),
                        "rotation": self.rotation_spin.value(),
                        "brightness": self.brightness_spin.value(),
                        "shift": self.shift_spin.value(),
                        "zoom": self.zoom_spin.value(),
                        "horizontal_flip": self.horizontal_flip_check.isChecked(),
                        "vertical_flip": self.vertical_flip_check.isChecked(),
                    },
                    "mixup": {
                        "use": self.mixup_check.isChecked(),
                        "alpha": self.mixup_alpha_spin.value(),
                    },
                    "cutmix": {
                        "use": self.cutmix_check.isChecked(),
                        "alpha": self.cutmix_alpha_spin.value(),
                    },
                    "autoaugment": {
                        "use": self.autoaugment_check.isChecked(),
                    },
                },
                "preprocessing": {
                    "normalization": self.normalization_combo.currentText(),
                    "norm_mean": [
                        self.norm_mean_r.value(),
                        self.norm_mean_g.value(),
                        self.norm_mean_b.value(),
                    ],
                    "norm_std": [
                        self.norm_std_r.value(),
                        self.norm_std_g.value(),
                        self.norm_std_b.value(),
                    ],
                    "resize_mode": self.resize_mode_combo.currentText(),
                    "cache_dataset": self.cache_dataset_check.isChecked(),
                },
                "monitoring": {
                    "metrics": {
                        "accuracy": self.accuracy_check.isChecked(),
                        "precision": self.precision_check.isChecked(),
                        "recall": self.recall_check.isChecked(),
                        "f1": self.f1_check.isChecked(),
                        "topk": self.topk_check.isChecked(),
                        "confusion_matrix": self.confusion_matrix_check.isChecked(),
                        "auc": self.auc_check.isChecked(),
                    },
                    "logging": {
                        "use_tensorboard": self.use_tensorboard_check.isChecked(),
                        "use_wandb": self.use_wandb_check.isChecked(),
                        "use_csv": self.use_csv_check.isChecked(),
                        "logging_freq": self.log_freq_combo.currentText(),
                    },
                    "visualization": {
                        "use_gradcam": self.use_gradcam_check.isChecked(),
                        "use_feature_maps": self.use_feature_maps_check.isChecked(),
                        "use_pred_samples": self.use_pred_samples_check.isChecked(),
                        "num_samples": self.num_samples_spin.value(),
                    },
                    "early_stopping": {
                        "patience": self.patience_spin.value(),
                        "min_delta": self.min_delta_spin.value(),
                        "monitor": self.monitor_combo.currentText(),
                    },
                    "checkpointing": {
                        "best_only": self.best_only_check.isChecked(),
                        "save_freq": self.save_freq_spin.value(),
                        "metric": self.checkpoint_metric_combo.currentText(),
                    },
                },
                "advanced": {
                    "seed": self.seed_spin.value(),
                    "deterministic": self.deterministic_check.isChecked(),
                    "class_weights": self.class_weights_combo.currentText(),
                    "sampler": self.sampler_combo.currentText(),
                    "image_channels": self.image_channels_spin.value(),
                    "tta": {
                        "use": self.use_tta_check.isChecked(),
                        "num_augmentations": self.tta_num_samples_spin.value(),
                    },
                    "export_onnx": self.export_onnx_check.isChecked(),
                    "quantization": {
                        "use": self.quantization_check.isChecked(),
                        "precision": self.quantization_precision_combo.currentText(),
                    },
                    "catastrophic_forgetting_prevention": {
                        "enable": self.prevent_forgetting_check.isChecked(),
                        "preserve_original_classes": self.preserve_classes_check.isChecked(),
                        "rehearsal": {
                            "use": self.rehearsal_check.isChecked(),
                            "samples_per_class": self.samples_per_class_spin.value(),
                            "synthetic_samples": self.synthetic_samples_check.isChecked(),
                        },
                        "knowledge_distillation": {
                            "use": self.knowledge_distillation_check.isChecked(),
                            "temperature": self.kd_temperature_spin.value(),
                            "alpha": self.kd_alpha_spin.value(),
                        },
                        "ewc_regularization": {
                            "use": self.ewc_check.isChecked(),
                            "lambda": self.ewc_lambda_spin.value(),
                            "fisher_sample_size": self.fisher_sample_size_spin.value(),
                            "adaptive_lambda": self.adaptive_ewc_lambda_check.isChecked(),
                        },
                        "layer_freezing": {
                            "strategy": self.layer_freezing_combo.currentText(),
                            "freeze_ratio": self.freeze_ratio_spin.value(),
                        },
                    },
                },
            }

            # Pobieranie wartości z zakładki Optymalizacja treningu
            optimization_config = {}
            if hasattr(self, "optimization_params"):
                for param in self.optimization_params:
                    param_key = param["param_key"]
                    hardware_radio = param["hardware_radio"]
                    value_widget = param["value_widget"]

                    # Pobieranie wartości w zależności od typu widgetu
                    if hardware_radio.isChecked() and param["hw_value"] is not None:
                        param_value = param["hw_value"]
                    else:
                        if isinstance(value_widget, QtWidgets.QSpinBox) or isinstance(
                            value_widget, QtWidgets.QDoubleSpinBox
                        ):
                            param_value = value_widget.value()
                        elif isinstance(value_widget, QtWidgets.QCheckBox):
                            param_value = value_widget.isChecked()
                        else:
                            param_value = value_widget.text()

                    optimization_config[param_key] = param_value

            # Aktualizacja konfiguracji z wartościami z Optymalizacji treningu
            if "recommended_batch_size" in optimization_config:
                config["training"]["batch_size"] = optimization_config[
                    "recommended_batch_size"
                ]
            if "recommended_workers" in optimization_config:
                config["training"]["num_workers"] = optimization_config[
                    "recommended_workers"
                ]
            if "use_mixed_precision" in optimization_config:
                config["training"]["mixed_precision"] = optimization_config[
                    "use_mixed_precision"
                ]
            if "gradient_accumulation_steps" in optimization_config:
                config["training"]["gradient_accumulation_steps"] = optimization_config[
                    "gradient_accumulation_steps"
                ]

            # Dodajemy sekcję optymalizacji do głównej konfiguracji
            config["optimization"] = optimization_config

            self.task_config = {
                "name": self.name_edit.text().strip(),
                "type": "fine_tuning",
                "status": "Nowy",
                "config": config,
                "created_at": datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),  # Dodany klucz
                "training_time": 0,
                "training_time_str": "0:00:00",
                "train_accuracy": 0.0,
                "train_loss": 0.0,
                "validation_accuracy": 0.0,
                "validation_loss": 0.0,
                "model_filename": "",
                "accuracy": 0.0,
                "epochs_trained": 0,
            }

            # Dodaj logi
            self.logger.info(f"Utworzono konfigurację zadania: {task_name}")
            self.logger.info(f"Typ zadania: {self.task_config['type']}")
            self.logger.info(
                f"Pełna konfiguracja: {json.dumps(self.task_config, indent=2, ensure_ascii=False)}"
            )

            self.accept()

        except Exception as e:
            self.logger.error(
                f"Błąd podczas dodawania zadania: {str(e)}", exc_info=True
            )
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Nie można dodać zadania: {str(e)}"
            )

    def _delete_profile(self):
        """Usuwa wybrany profil."""
        if not self.current_profile:
            QtWidgets.QMessageBox.warning(
                self, "Ostrzeżenie", "Najpierw wybierz profil do usunięcia."
            )
            return

        try:
            profile_name = self.profile_list.currentItem().text()
            profile_path = self.profiles_dir / f"{profile_name}.json"
            if profile_path.exists():
                profile_path.unlink()
                self._refresh_profile_list()
                self.current_profile = None
                QtWidgets.QMessageBox.information(
                    self, "Sukces", "Profil został pomyślnie usunięty."
                )
            else:
                self.logger.warning(f"Plik profilu nie istnieje: {profile_path}")
                QtWidgets.QMessageBox.warning(
                    self, "Ostrzeżenie", f"Plik profilu nie istnieje: {profile_name}"
                )
                self._refresh_profile_list()  # Odśwież listę, aby usunąć nieistniejące referencje

        except Exception as e:
            self.logger.error(f"Błąd podczas usuwania profilu: {str(e)}", exc_info=True)
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Nie można usunąć profilu: {str(e)}"
            )

    def _get_model_config(self) -> Dict[str, Any]:
        """Zwraca konfigurację modelu."""
        config = {
            "architecture": self.arch_combo.currentText(),
            "variant": self.variant_combo.currentText(),
            "input_size": self.input_size_spin.value(),
            "num_classes": self.num_classes_spin.value(),
            "pretrained": self.pretrained_check.isChecked(),
            "pretrained_weights": self.pretrained_weights_combo.currentText(),
            "feature_extraction_only": self.feature_extraction_check.isChecked(),
            "activation": self.activation_combo.currentText(),
            "dropout_at_inference": self.dropout_at_inference_check.isChecked(),
            "global_pool": self.global_pool_combo.currentText(),
            "last_layer_activation": self.last_layer_activation_combo.currentText(),
        }
        self.logger.info(
            f"Zapisuję konfigurację modelu: {json.dumps(config, indent=2)}"
        )
        return config

    def _update_architecture_dependent_controls(self):
        """Aktualizuje kontrolki zależne od architektury."""
        architecture = self.arch_combo.currentText()
        self._update_variant_combo(architecture)

    def _update_training_dependent_controls(self):
        """Aktualizuje kontrolki zależne od parametrów treningu."""
        # Aktualizacja kontrolek zależnych od optymalizatora
        self._update_optimizer_dependent_controls()

        # Aktualizacja kontrolek zależnych od schedulera
        self._update_scheduler_dependent_controls()

        # Aktualizacja kontrolek zależnych od mixed precision
        self.mixed_precision_check.setEnabled(True)

        # Aktualizacja kontrolek zależnych od unfreeze strategy
        self.unfreeze_strategy_combo.setEnabled(True)
        self.unfreeze_layers_spin.setEnabled(
            self.unfreeze_strategy_combo.currentText() != self.UNFREEZE_ALL
        )

    def _update_optimizer_dependent_controls(self):
        """Aktualizuje kontrolki zależne od optymalizatora."""
        optimizer = self.optimizer_combo.currentText()

        # Włącz/wyłącz kontrolki w zależności od wybranego optymalizatora
        self.weight_decay_spin.setEnabled(
            True
        )  # Dostępne dla wszystkich optymalizatorów

        # Specyficzne ustawienia dla różnych optymalizatorów
        if optimizer == "AdamW":
            self.weight_decay_spin.setEnabled(True)
        elif optimizer == "SGD":
            self.weight_decay_spin.setEnabled(True)
        elif optimizer == "Adam":
            self.weight_decay_spin.setEnabled(False)

    def _update_scheduler_dependent_controls(self):
        """Aktualizuje kontrolki zależne od schedulera."""
        scheduler = self.scheduler_combo.currentText()

        # Włącz/wyłącz kontrolki w zależności od wybranego schedulera
        self.warmup_epochs_spin.setEnabled(True)
        self.warmup_lr_init_spin.setEnabled(True)

    def _update_augmentation_dependent_controls(self):
        """Aktualizuje kontrolki zależne od augmentacji."""
        # Basic augmentation
        basic_enabled = self.basic_aug_check.isChecked()
        self.rotation_spin.setEnabled(basic_enabled)
        self.brightness_spin.setEnabled(basic_enabled)
        self.shift_spin.setEnabled(basic_enabled)
        self.zoom_spin.setEnabled(basic_enabled)
        self.horizontal_flip_check.setEnabled(basic_enabled)
        self.vertical_flip_check.setEnabled(basic_enabled)

        # Mixup
        mixup_enabled = self.mixup_check.isChecked()
        self.mixup_alpha_spin.setEnabled(mixup_enabled)

        # CutMix
        cutmix_enabled = self.cutmix_check.isChecked()
        self.cutmix_alpha_spin.setEnabled(cutmix_enabled)

        # AutoAugment
        # self.autoaugment_check - stan zarządzany przez użytkownika

        # RandAugment
        randaugment_enabled = self.randaugment_check.isChecked()
        self.randaugment_n_spin.setEnabled(randaugment_enabled)
        self.randaugment_m_spin.setEnabled(randaugment_enabled)

    def _update_preprocessing_dependent_controls(self):
        """Aktualizuje kontrolki zależne od preprocessingu."""
        # Normalizacja
        self.norm_mean_r.setEnabled(True)
        self.norm_mean_g.setEnabled(True)
        self.norm_mean_b.setEnabled(True)
        self.norm_std_r.setEnabled(True)
        self.norm_std_g.setEnabled(True)
        self.norm_std_b.setEnabled(True)

    def _update_monitoring_dependent_controls(self):
        """Aktualizuje kontrolki zależne od monitorowania."""
        # Metrics
        self.accuracy_check.setEnabled(True)
        self.precision_check.setEnabled(True)
        self.recall_check.setEnabled(True)
        self.f1_check.setEnabled(True)
        self.topk_check.setEnabled(True)
        self.confusion_matrix_check.setEnabled(True)
        self.auc_check.setEnabled(True)

        # Logging
        self.use_tensorboard_check.setEnabled(True)
        self.use_wandb_check.setEnabled(True)
        self.use_csv_check.setEnabled(True)
        self.log_freq_combo.setEnabled(True)

        # Visualization
        self.use_gradcam_check.setEnabled(True)
        self.use_feature_maps_check.setEnabled(True)
        self.use_pred_samples_check.setEnabled(True)
        self.num_samples_spin.setEnabled(True)

        # Early stopping
        self.patience_spin.setEnabled(True)
        self.min_delta_spin.setEnabled(True)
        self.monitor_combo.setEnabled(True)

        # Checkpointing
        self.best_only_check.setEnabled(True)
        self.save_freq_spin.setEnabled(True)
        self.checkpoint_metric_combo.setEnabled(True)

        # Aktualizacja zależności między kontrolkami
        if self.use_tensorboard_check.isChecked():
            self.log_freq_combo.setEnabled(True)
        else:
            self.log_freq_combo.setEnabled(False)

        if self.use_pred_samples_check.isChecked():
            self.num_samples_spin.setEnabled(True)
        else:
            self.num_samples_spin.setEnabled(False)

        if self.best_only_check.isChecked():
            self.save_freq_spin.setEnabled(False)
        else:
            self.save_freq_spin.setEnabled(True)

    def _create_profile_from_model_config(self):
        """Tworzy nowy plik profilu na podstawie wczytanej konfiguracji modelu."""
        if not hasattr(self, "config") or not self.config:
            QtWidgets.QMessageBox.warning(
                self, "Ostrzeżenie", "Najpierw wczytaj model z prawidłową konfiguracją."
            )
            self.logger.warning(
                "_create_profile_from_model_config: Brak self.config lub jest pusty."
            )
            return

        try:
            # Zaproponuj nazwę na podstawie architektury i wariantu z self.config
            model_cfg = self.config.get("model", {})
            arch = model_cfg.get("architecture", "unknown_arch")
            variant = model_cfg.get("variant", "unknown_variant")
            default_profile_name = f"{arch}_{variant}_from_config"

            name, ok = QtWidgets.QInputDialog.getText(
                self,
                "Utwórz profil z konfiguracji",
                "Podaj nazwę dla nowego profilu:",
                QtWidgets.QLineEdit.EchoMode.Normal,
                default_profile_name,
            )

            if ok and name:
                if not name.strip():
                    QtWidgets.QMessageBox.warning(
                        self, "Błąd", "Nazwa profilu nie może być pusta."
                    )
                    return

                # Ścieżki do plików
                default_profile_path = self.profiles_dir / "default_profile.json"
                extracted_config_path = self.profiles_dir / "extracted_config.json"
                output_path = str(self.profiles_dir / f"{name.strip()}.json")
                temp_base_profile_path = self.profiles_dir / "temp_base_profile.json"

                # Sprawdź czy plik domyślnego profilu istnieje
                if not default_profile_path.exists():
                    self.logger.error(
                        f"Nie znaleziono pliku default_profile.json pod ścieżką: {default_profile_path}"
                    )
                    QtWidgets.QMessageBox.critical(
                        self,
                        "Błąd",
                        "Nie znaleziono pliku default_profile.json. Skontaktuj się z administratorem.",
                    )
                    return

                try:
                    # Wczytaj domyślny profil
                    with open(default_profile_path, "r", encoding="utf-8") as f:
                        profile_data = json.load(f)

                    # Zaktualizuj 'info' i 'description'
                    profile_data["info"] = name.strip()
                    # Upewnij się, że self.config i odpowiednie klucze istnieją
                    model_details_cfg = self.config.get("model", {})
                    arch = model_details_cfg.get("architecture", "N/A")
                    variant = model_details_cfg.get("variant", "N/A")
                    base_model_filename = (
                        Path(self.model_path_edit.text()).name
                        if self.model_path_edit.text()
                        else "N/A"
                    )
                    profile_data["description"] = (
                        f"Profil z konfiguracji dla modelu: {arch} {variant} "
                        f"(plik bazowy: {base_model_filename})"
                    )

                    # Zapisz zmodyfikowany profil do pliku tymczasowego
                    with open(temp_base_profile_path, "w", encoding="utf-8") as f:
                        json.dump(profile_data, f, indent=4, ensure_ascii=False)

                    # Zapisz wczytaną konfigurację (self.config) do pliku tymczasowego extracted_config.json
                    with open(extracted_config_path, "w", encoding="utf-8") as f:
                        json.dump(self.config, f, indent=4, ensure_ascii=False)

                    # Użyj funkcji merge_configs do połączenia konfiguracji
                    from app.gui.dialogs.config_merger import merge_configs

                    self.logger.info(
                        f"Przekazuję do merge_configs - temp_base: {temp_base_profile_path}, "
                        f"extracted_config: {extracted_config_path}, output: {output_path}"
                    )

                    # Użyj str() dla ścieżek Path, jeśli merge_configs tego oczekuje
                    merge_configs(
                        str(temp_base_profile_path),
                        str(extracted_config_path),
                        output_path,
                    )

                    QtWidgets.QMessageBox.information(
                        self,
                        "Sukces",
                        f"Profil '{name.strip()}' został pomyślnie utworzony i zapisany.",
                    )
                    self.logger.info(
                        f"Utworzono nowy profil '{name.strip()}' z konfiguracji modelu."
                    )

                except FileNotFoundError:
                    self.logger.error(
                        f"Nie znaleziono pliku default_profile.json pod ścieżką: "
                        f"{default_profile_path}"
                    )
                    QtWidgets.QMessageBox.critical(
                        self,
                        "Błąd",
                        "Nie znaleziono pliku default_profile.json. "
                        "Skontaktuj się z administratorem.",
                    )
                    return
                except Exception as e_merge:
                    self.logger.error(
                        f"Błąd podczas procesu tworzenia profilu (merge lub zapis): "
                        f"{str(e_merge)}",
                        exc_info=True,
                    )
                    QtWidgets.QMessageBox.critical(
                        self, "Błąd", f"Nie można utworzyć profilu: {str(e_merge)}"
                    )
                    return

                finally:
                    # Usuń pliki tymczasowe
                    if temp_base_profile_path.exists():
                        os.remove(temp_base_profile_path)
                        self.logger.info(
                            f"Usunięto tymczasowy plik: {temp_base_profile_path}"
                        )
                    if (
                        extracted_config_path.exists()
                    ):  # Sprawdzenie, czy extracted_config_path jest Path
                        os.remove(extracted_config_path)
                        self.logger.info(
                            f"Usunięto tymczasowy plik: {extracted_config_path}"
                        )

                # Odśwież listę profili
                self._refresh_profile_list()

        except Exception as e:
            self.logger.error(
                f"Błąd podczas tworzenia profilu z konfiguracji modelu: {str(e)}",
                exc_info=True,
            )
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Nie można utworzyć profilu z konfiguracji: {str(e)}"
            )

    def _create_spinbox(self, min_val, max_val, default_val, decimals=0, step=1):
        """
        Tworzy i konfiguruje QSpinBox lub QDoubleSpinBox.

        Args:
            min_val: Minimalna wartość
            max_val: Maksymalna wartość
            default_val: Domyślna wartość
            decimals: Liczba miejsc po przecinku (0 dla QSpinBox)
            step: Wartość kroku

        Returns:
            Skonfigurowany QSpinBox lub QDoubleSpinBox
        """
        if decimals > 0:
            spin = QtWidgets.QDoubleSpinBox()
            spin.setDecimals(decimals)
            spin.setSingleStep(0.1 ** (decimals - 1))
        else:
            spin = QtWidgets.QSpinBox()
            spin.setSingleStep(step)

        spin.setRange(min_val, max_val)
        spin.setValue(default_val)
        return spin

    def _create_group_box(self, title, layout_type=QtWidgets.QFormLayout):
        """
        Tworzy QGroupBox z określonym typem layoutu.

        Args:
            title: Tytuł grupy
            layout_type: Typ layoutu (domyślnie QFormLayout)

        Returns:
            Tuple: (group_box, layout)
        """
        group = QtWidgets.QGroupBox(title)
        layout = layout_type()
        group.setLayout(layout)
        return group, layout

    def _add_form_row(self, layout, label, widget):
        """
        Dodaje wiersz do layoutu formularza, obsługując różne typy widgetów.

        Args:
            layout: Layout formularza
            label: Etykieta
            widget: Widget do dodania (może być pojedynczy widget, lista lub layout)
        """
        if isinstance(widget, list):
            hlayout = QtWidgets.QHBoxLayout()
            for w in widget:
                hlayout.addWidget(w)
            layout.addRow(label, hlayout)
        elif isinstance(widget, QtWidgets.QLayout):
            layout.addRow(label, widget)
        else:
            layout.addRow(label, widget)

    def _create_optimization_tab(self) -> QtWidgets.QWidget:
        """Tworzenie zakładki Optymalizacja treningu."""
        try:
            self.logger.debug("Tworzenie zakładki optymalizacji treningu")
            tab = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(tab)

            # Grupa parametrów optymalizacyjnych
            params_group = QtWidgets.QGroupBox("Parametry optymalizacyjne")
            params_layout = QtWidgets.QFormLayout()

            # Dodaj parametry optymalizacyjne
            params = [
                ("Batch size", "batch_size", 32, "int", 1, 1024, 1),
                ("Learning rate", "learning_rate", 0.001, "float", 0.0001, 1.0, 0.0001),
                ("Epochs", "epochs", 100, "int", 1, 1000, 1),
                ("Workers", "num_workers", 4, "int", 0, 32, 1),
                (
                    "Gradient Accumulation",
                    "gradient_accumulation_steps",
                    1,
                    "int",
                    1,
                    16,
                    1,
                ),
                (
                    "Mixed Precision",
                    "use_mixed_precision",
                    True,
                    "bool",
                    None,
                    None,
                    None,
                ),
            ]

            self.optimization_params = []
            for name, key, default, type_, min_, max_, step in params:
                row = self._create_parameter_row(
                    name, key, default, type_, min_, max_, step
                )
                params_layout.addRow(name + ":", row)
                if hasattr(self, "parameter_rows") and key in self.parameter_rows:
                    self.optimization_params.append(self.parameter_rows[key])

            params_group.setLayout(params_layout)
            layout.addWidget(params_group)

            # Przycisk do zastosowania wszystkich optymalizacji
            apply_all_btn = QtWidgets.QPushButton("Zastosuj wszystkie optymalizacje")
            apply_all_btn.clicked.connect(self._apply_all_hardware_optimizations)
            layout.addWidget(apply_all_btn)

            return tab

        except Exception as e:
            self.logger.error(
                f"Błąd podczas tworzenia zakładki: {str(e)}", exc_info=True
            )
            raise

    def _create_parameter_row(
        self,
        name,
        param_key,
        default_value,
        widget_type,
        min_val=None,
        max_val=None,
        step=None,
    ):
        layout = QtWidgets.QHBoxLayout()

        # Wartość użytkownika
        if widget_type == "int":
            value_widget = QtWidgets.QSpinBox()
            value_widget.setRange(min_val or -999999, max_val or 999999)
            value_widget.setValue(default_value)
            if step:
                value_widget.setSingleStep(step)
        elif widget_type == "float":
            value_widget = QtWidgets.QDoubleSpinBox()
            value_widget.setRange(min_val or -999999.0, max_val or 999999.0)
            value_widget.setValue(default_value)
            if step:
                value_widget.setSingleStep(step)
        elif widget_type == "bool":
            value_widget = QtWidgets.QCheckBox()
            value_widget.setChecked(default_value)
        else:
            value_widget = QtWidgets.QLineEdit(str(default_value))

        # Checkbox "Użytkownika"
        user_checkbox = QtWidgets.QCheckBox("Użytkownika")
        user_checkbox.setChecked(True)

        # Mapowanie kluczy profilu sprzętowego
        profile_key = {
            "batch_size": "recommended_batch_size",
            "learning_rate": "learning_rate",
            "epochs": "max_epochs",
            "num_workers": "recommended_workers",
            "use_mixed_precision": "use_mixed_precision",
        }.get(param_key, param_key)
        # Pobierz wartość z profilu sprzętowego lub wyświetl "Brak"
        hw_value_actual = self.hardware_profile.get(profile_key)
        hw_value_text = str(hw_value_actual) if hw_value_actual is not None else "Brak"
        hw_value = QtWidgets.QLabel(hw_value_text)

        # Checkbox "Profil sprzętowy"
        hw_checkbox = QtWidgets.QCheckBox("Profil sprzętowy")
        hw_checkbox.setChecked(False)

        # Grupa przycisków (checkboxów) do wyboru źródła wartości
        source_group = QtWidgets.QButtonGroup()
        source_group.addButton(user_checkbox)
        source_group.addButton(hw_checkbox)
        source_group.setExclusive(True)

        # Dodaj widgety do layoutu
        layout.addWidget(value_widget)
        layout.addWidget(user_checkbox)
        layout.addWidget(QtWidgets.QLabel("Profil sprzętowy:"))
        layout.addWidget(hw_value)
        layout.addWidget(hw_checkbox)

        # Zapamiętanie referencji do widgetów i grupy przycisków
        row_widgets = {
            "param_key": param_key,
            "value_widget": value_widget,
            "user_checkbox": user_checkbox,
            "hw_value_label": None,  # niepotrzebne
            "hw_value": hw_value,
            "hw_checkbox": hw_checkbox,
            "button_group": source_group,
            "hw_value_actual": hw_value_actual,
        }

        user_checkbox.toggled.connect(
            lambda checked: self._on_source_toggle(row_widgets, checked)
        )
        hw_checkbox.toggled.connect(
            lambda checked: self._on_hw_toggle(row_widgets, checked)
        )

        if not hasattr(self, "parameter_rows"):
            self.parameter_rows = {}
        self.parameter_rows[param_key] = row_widgets

        return layout

    def _on_source_toggle(self, row_widgets, is_user_selected):
        """Obsługuje przełączanie między wartościami użytkownika a profilami sprzętowymi."""
        value_widget = row_widgets["value_widget"]
        hw_checkbox = row_widgets["hw_checkbox"]

        if is_user_selected:
            value_widget.setEnabled(True)
            hw_checkbox.setChecked(False)
        else:
            # Ten kod prawdopodobnie nie zostanie wywołany, ponieważ checkboxy są wzajemnie wykluczające
            value_widget.setEnabled(False)

    def _on_hw_toggle(self, row_widgets, is_hw_selected):
        """Obsługuje przełączanie na profil sprzętowy."""
        value_widget = row_widgets["value_widget"]
        user_checkbox = row_widgets["user_checkbox"]
        hw_value_actual = row_widgets["hw_value_actual"]

        if is_hw_selected:
            user_checkbox.setChecked(False)
            value_widget.setEnabled(False)

            # Ustaw wartość z profilu sprzętowego, jeśli jest dostępna
            if hw_value_actual is not None:
                if isinstance(value_widget, QtWidgets.QSpinBox) or isinstance(
                    value_widget, QtWidgets.QDoubleSpinBox
                ):
                    value_widget.setValue(hw_value_actual)
                elif isinstance(value_widget, QtWidgets.QCheckBox):
                    value_widget.setChecked(hw_value_actual)
                else:
                    value_widget.setText(str(hw_value_actual))
        else:
            # Ten kod prawdopodobnie nie zostanie wywołany, ponieważ checkboxy są wzajemnie wykluczające
            value_widget.setEnabled(True)

    def _apply_all_hardware_optimizations(self):
        """Zastosowuje wszystkie optymalne ustawienia z profilu sprzętowego."""
        try:
            count = 0
            for param in self.parameter_rows.values():
                param_key = param["param_key"]
                if param_key in self.hardware_profile:
                    # Włącz checkbox profilu sprzętowego, wyłącz checkbox użytkownika
                    param["hw_checkbox"].setChecked(True)
                    param["user_checkbox"].setChecked(False)

                    # Kontrolka wartości jest już obsługiwana przez _on_hw_toggle
                    count += 1

            QtWidgets.QMessageBox.information(
                self,
                "Sukces",
                f"Zastosowano {count} optymalnych ustawień z profilu sprzętowego.",
            )
        except Exception as e:
            self.logger.error(
                f"Błąd podczas stosowania optymalizacji: {str(e)}", exc_info=True
            )
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Nie można zastosować optymalizacji: {str(e)}"
            )

    def _update_optimization_state(self, state):
        """Aktualizuje stan kontrolek optymalizacji na podstawie stanu checkboxa."""
        enabled = bool(state)

        # Aktualizacja dostępności opcji "z profilu sprzętowego" we wszystkich parametrach
        if hasattr(self, "parameter_rows"):
            for param in self.parameter_rows.values():
                hw_checkbox = param["hw_checkbox"]  # Poprawna nazwa
                hw_value_label = param["hw_value_label"]
                hw_value = param["hw_value"]

                hw_checkbox.setEnabled(enabled)  # Poprawna nazwa
                hw_value_label.setEnabled(enabled)
                hw_value.setEnabled(enabled)

                # Jeśli optymalizacja jest wyłączona, przełącz na "Użytkownika"
                if not enabled and hw_checkbox.isChecked():  # Poprawna nazwa
                    param["user_checkbox"].setChecked(True)  # Poprawna nazwa

    def _show_hardware_profile(self):
        """Wyświetla okno z aktualnym profilem sprzętowym."""
        import pprint

        msg = pprint.pformat(self.hardware_profile, indent=2, width=80)
        QtWidgets.QMessageBox.information(self, "Profil sprzętowy", msg)

    def _open_log_file(self):
        """Otwiera plik logu w domyślnym edytorze tekstu."""
        import os

        log_path = os.path.abspath("training_dialog.log")
        try:
            os.startfile(log_path)
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Błąd", f"Nie można otworzyć pliku logu: {e}"
            )
