import datetime
import json
import logging
import os
from pathlib import Path

from PyQt6 import QtCore, QtWidgets

from app.gui.dialogs.hardware_profile_dialog import HardwareProfileDialog
from app.utils.file_utils import (validate_training_directory,
                                  validate_validation_directory)


class TrainingTaskConfigDialog(QtWidgets.QDialog):
    """Dialog konfiguracji zadania treningu."""

    # Strategie odmrażania warstw
    UNFREEZE_ALL = "unfreeze_all"
    UNFREEZE_GRADUAL_END = "unfreeze_gradual_end"
    UNFREEZE_GRADUAL_START = "unfreeze_gradual_start"
    UNFREEZE_AFTER_EPOCHS = "unfreeze_after_epochs"  # Poprawiona literówka

    def __init__(self, parent=None, settings=None, hardware_profile=None):
        super().__init__(parent)
        self.settings = settings
        if not hardware_profile:
            self.hardware_profile = {}
        else:
            self.hardware_profile = hardware_profile
        self._setup_logging()
        self.logger.info(f"Profil sprzętowy: {self.hardware_profile}")
        self.setWindowTitle("Konfiguracja treningu")
        self.setMinimumWidth(1000)  # Zwiększono szerokość dla lepszego układu
        self.profiles_dir = Path("data/profiles")
        self.profiles_dir.mkdir(exist_ok=True)
        self.current_profile = None
        self.setWindowFlags(
            self.windowFlags() | QtCore.Qt.WindowType.WindowCloseButtonHint
        )
        self._init_ui()
        self._connect_signals()  # Podłączanie sygnałów po inicjalizacji UI

    def _setup_logging(self):
        self.logger = logging.getLogger(__name__)
        # ... (reszta konfiguracji logowania bez zmian)

    def _init_ui(self):
        try:
            self.logger.debug("Rozpoczęcie inicjalizacji UI")
            layout = QtWidgets.QVBoxLayout(self)

            show_hw_profile_btn = QtWidgets.QPushButton("Pokaż profil sprzętowy")
            show_hw_profile_btn.clicked.connect(self._show_hardware_profile)
            layout.addWidget(show_hw_profile_btn)

            self.tabs = QtWidgets.QTabWidget()

            tab_data_model = self._create_data_model_tab()
            self.tabs.addTab(tab_data_model, "Dane i Model")

            tab_training_params = self._create_training_params_tab()
            self.tabs.addTab(
                tab_training_params, "Parametry Treningu"
            )  # Zmieniona nazwa

            tab_regularization = self._create_regularization_tab()
            self.tabs.addTab(tab_regularization, "Regularyzacja")

            tab_augmentation = self._create_augmentation_tab()
            self.tabs.addTab(tab_augmentation, "Augmentacja")

            tab_preprocessing = self._create_preprocessing_tab()
            self.tabs.addTab(tab_preprocessing, "Preprocessing")

            tab_monitoring = self._create_monitoring_tab()
            self.tabs.addTab(
                tab_monitoring, "Monitorowanie i Logowanie"
            )  # Zmieniona nazwa

            tab_advanced = self._create_advanced_tab()
            self.tabs.addTab(tab_advanced, "Zaawansowane")

            tab_optimization = self._create_optimization_tab()
            self.tabs.addTab(
                tab_optimization, "Optymalizacja Sprzętowa"
            )  # Zmieniona nazwa

            layout.addWidget(self.tabs)

            buttons_layout = QtWidgets.QHBoxLayout()
            add_task_btn = QtWidgets.QPushButton("Dodaj zadanie")
            add_task_btn.clicked.connect(self._on_accept)
            buttons_layout.addWidget(add_task_btn)

            close_btn = QtWidgets.QPushButton("Zamknij")
            close_btn.clicked.connect(self.accept)
            buttons_layout.addWidget(close_btn)
            layout.addLayout(buttons_layout)

            self.logger.debug("Zakończono inicjalizację UI")

        except Exception as e:
            msg = "Błąd podczas inicjalizacji UI"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            raise

    def _connect_signals(self):
        # Połączenie sygnału zmiany strategii odmrażania z aktualizacją stanu kontrolki unfreeze_after_epochs_spin
        if hasattr(self, "unfreeze_strategy_combo") and hasattr(
            self, "unfreeze_after_epochs_spin"
        ):
            self.unfreeze_strategy_combo.currentTextChanged.connect(
                self._toggle_unfreeze_after_epochs_spin
            )
            self._toggle_unfreeze_after_epochs_spin(
                self.unfreeze_strategy_combo.currentText()
            )  # Inicjalne ustawienie

        if hasattr(self, "use_early_stopping_check"):
            self.use_early_stopping_check.toggled.connect(
                self._toggle_early_stopping_controls
            )
            self._toggle_early_stopping_controls(
                self.use_early_stopping_check.isChecked()
            )

        if hasattr(self, "reduce_lr_enabled_check"):
            self.reduce_lr_enabled_check.toggled.connect(
                self._toggle_reduce_lr_controls
            )
            self._toggle_reduce_lr_controls(self.reduce_lr_enabled_check.isChecked())

        if hasattr(self, "stochastic_depth_use_check"):  # Używamy nowej nazwy
            self.stochastic_depth_use_check.toggled.connect(
                self._toggle_stochastic_depth_controls
            )
            self._toggle_stochastic_depth_controls(
                self.stochastic_depth_use_check.isChecked()
            )

        if hasattr(self, "use_swa_check"):
            self.use_swa_check.toggled.connect(self._toggle_swa_controls)
            self._toggle_swa_controls(self.use_swa_check.isChecked())

        # Augmentacje
        if hasattr(self, "basic_aug_check"):
            self.basic_aug_check.toggled.connect(self._toggle_basic_aug_controls)
            self._toggle_basic_aug_controls(self.basic_aug_check.isChecked())
        if hasattr(self, "mixup_check"):
            self.mixup_check.toggled.connect(
                lambda checked: (
                    self.mixup_alpha_spin.setEnabled(checked)
                    if hasattr(self, "mixup_alpha_spin")
                    else None
                )
            )
            if hasattr(self, "mixup_alpha_spin"):  # Check before use
                self.mixup_alpha_spin.setEnabled(self.mixup_check.isChecked())
        if hasattr(self, "cutmix_check"):
            self.cutmix_check.toggled.connect(
                lambda checked: (
                    self.cutmix_alpha_spin.setEnabled(checked)
                    if hasattr(self, "cutmix_alpha_spin")
                    else None
                )
            )
            if hasattr(self, "cutmix_alpha_spin"):  # Check before use
                self.cutmix_alpha_spin.setEnabled(self.cutmix_check.isChecked())
        if hasattr(self, "autoaugment_check"):
            self.autoaugment_check.toggled.connect(
                lambda checked: (
                    self.autoaugment_policy_combo.setEnabled(checked)
                    if hasattr(self, "autoaugment_policy_combo")
                    else None
                )
            )
            if hasattr(self, "autoaugment_policy_combo"):  # Check before use
                self.autoaugment_policy_combo.setEnabled(
                    self.autoaugment_check.isChecked()
                )
        if hasattr(self, "randaugment_check"):
            self.randaugment_check.toggled.connect(self._toggle_randaugment_controls)
            self._toggle_randaugment_controls(self.randaugment_check.isChecked())
        if hasattr(self, "random_erase_check"):
            self.random_erase_check.toggled.connect(self._toggle_random_erase_controls)
            self._toggle_random_erase_controls(self.random_erase_check.isChecked())
        if hasattr(self, "grid_distortion_check"):
            self.grid_distortion_check.toggled.connect(
                self._toggle_grid_distortion_controls
            )
            self._toggle_grid_distortion_controls(
                self.grid_distortion_check.isChecked()
            )

        # Preprocessing
        if hasattr(self, "preprocess_resize_enabled_check"):
            self.preprocess_resize_enabled_check.toggled.connect(
                self._toggle_preprocess_resize_controls
            )
            self._toggle_preprocess_resize_controls(
                self.preprocess_resize_enabled_check.isChecked()
            )
        if hasattr(self, "preprocess_normalize_enabled_check"):
            self.preprocess_normalize_enabled_check.toggled.connect(
                self._toggle_preprocess_normalize_controls
            )
            self._toggle_preprocess_normalize_controls(
                self.preprocess_normalize_enabled_check.isChecked()
            )
        if hasattr(self, "preprocess_grayscale_enabled_check"):
            self.preprocess_grayscale_enabled_check.toggled.connect(
                lambda checked: (
                    self.preprocess_grayscale_num_output_channels_spin.setEnabled(
                        checked
                    )
                    if hasattr(self, "preprocess_grayscale_num_output_channels_spin")
                    else None
                )  # Check before use
            )
            if hasattr(
                self, "preprocess_grayscale_num_output_channels_spin"
            ):  # Sprawdzenie przed użyciem
                self.preprocess_grayscale_num_output_channels_spin.setEnabled(
                    self.preprocess_grayscale_enabled_check.isChecked()
                )

        if hasattr(self, "preprocess_color_jitter_enabled_check"):
            self.preprocess_color_jitter_enabled_check.toggled.connect(
                self._toggle_preprocess_color_jitter_controls
            )
            self._toggle_preprocess_color_jitter_controls(
                self.preprocess_color_jitter_enabled_check.isChecked()
            )

        if hasattr(self, "preprocess_gaussian_blur_enabled_check"):
            self.preprocess_gaussian_blur_enabled_check.toggled.connect(
                self._toggle_preprocess_gaussian_blur_controls
            )
            self._toggle_preprocess_gaussian_blur_controls(
                self.preprocess_gaussian_blur_enabled_check.isChecked()
            )

        if hasattr(self, "preprocess_random_resize_crop_enabled_check"):
            self.preprocess_random_resize_crop_enabled_check.toggled.connect(
                self._toggle_preprocess_random_resize_crop_controls
            )
            self._toggle_preprocess_random_resize_crop_controls(
                self.preprocess_random_resize_crop_enabled_check.isChecked()
            )

        # Monitoring
        if hasattr(self, "tensorboard_enabled_check"):
            self.tensorboard_enabled_check.toggled.connect(
                self._toggle_tensorboard_controls
            )
            self._toggle_tensorboard_controls(
                self.tensorboard_enabled_check.isChecked()
            )

        if hasattr(self, "wandb_enabled_check"):
            self.wandb_enabled_check.toggled.connect(self._toggle_wandb_controls)
            self._toggle_wandb_controls(self.wandb_enabled_check.isChecked())

        if hasattr(self, "checkpoint_enabled_check"):
            self.checkpoint_enabled_check.toggled.connect(
                self._toggle_checkpoint_controls
            )
            self._toggle_checkpoint_controls(self.checkpoint_enabled_check.isChecked())

        # Advanced
        if hasattr(self, "advanced_gradient_clip_agc_check"):
            self.advanced_gradient_clip_agc_check.toggled.connect(
                self._toggle_agc_controls
            )
            self._toggle_agc_controls(self.advanced_gradient_clip_agc_check.isChecked())

    def _create_data_model_tab(self):
        try:
            self.logger.debug("Tworzenie zakładki Dane i Model")
            tab = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(tab)
            form_layout = QtWidgets.QFormLayout()  # Zmieniono nazwę dla jasności

            # Katalog danych treningowych
            train_dir_layout = QtWidgets.QHBoxLayout()
            self.train_dir_edit = QtWidgets.QLineEdit()
            train_dir_btn = QtWidgets.QPushButton("Przeglądaj...")
            train_dir_btn.clicked.connect(self._select_train_dir)
            train_dir_layout.addWidget(self.train_dir_edit)
            train_dir_layout.addWidget(train_dir_btn)
            form_layout.addRow("Katalog treningowy:", train_dir_layout)

            # Katalog danych walidacyjnych
            val_dir_layout = QtWidgets.QHBoxLayout()
            self.val_dir_edit = QtWidgets.QLineEdit()
            val_dir_btn = QtWidgets.QPushButton("Przeglądaj...")
            val_dir_btn.clicked.connect(self._select_val_dir)
            val_dir_layout.addWidget(self.val_dir_edit)
            val_dir_layout.addWidget(val_dir_btn)
            form_layout.addRow("Katalog walidacyjny:", val_dir_layout)

            # Architektura modelu
            self.arch_combo = QtWidgets.QComboBox()
            self.arch_combo.addItems(["EfficientNet", "ConvNeXt"])
            form_layout.addRow("Architektura:", self.arch_combo)

            # Wariant modelu
            self.variant_combo = QtWidgets.QComboBox()
            self._update_variant_combo("EfficientNet")  # Inicjalizacja
            form_layout.addRow("Wariant:", self.variant_combo)
            self.arch_combo.currentTextChanged.connect(self._on_architecture_changed)

            # Rozmiar obrazu wejściowego
            self.input_size_spin = QtWidgets.QSpinBox()
            self.input_size_spin.setRange(32, 1024)
            self.input_size_spin.setValue(224)
            self.input_size_spin.setSingleStep(32)
            form_layout.addRow("Rozmiar obrazu (model):", self.input_size_spin)

            # Liczba klas
            self.num_classes_spin = QtWidgets.QSpinBox()
            self.num_classes_spin.setRange(2, 1000)
            self.num_classes_spin.setValue(2)
            form_layout.addRow("Liczba klas:", self.num_classes_spin)

            layout.addLayout(form_layout)

            # Grupa profili (bez zmian)
            profile_group = QtWidgets.QGroupBox("Dostępne profile")
            profile_layout_widget = (
                QtWidgets.QWidget()
            )  # Użyj widgetu jako kontenera dla layoutu
            profile_main_layout = QtWidgets.QVBoxLayout(
                profile_layout_widget
            )  # Główny layout dla grupy profili

            self.profile_list = QtWidgets.QListWidget()
            self.profile_list.currentItemChanged.connect(self._on_profile_selected)
            self._refresh_profile_list()
            profile_main_layout.addWidget(self.profile_list)

            info_group = QtWidgets.QGroupBox("Informacje o profilu")
            info_form_layout = QtWidgets.QFormLayout()  # Zmieniono nazwę

            self.profile_info = QtWidgets.QTextEdit()
            self.profile_info.setReadOnly(True)
            self.profile_info.setMaximumHeight(60)
            info_form_layout.addRow("Info:", self.profile_info)

            self.profile_description = QtWidgets.QTextEdit()
            self.profile_description.setReadOnly(True)
            self.profile_description.setMaximumHeight(60)
            info_form_layout.addRow("Opis:", self.profile_description)

            self.profile_data_required = QtWidgets.QTextEdit()
            self.profile_data_required.setReadOnly(True)
            self.profile_data_required.setMaximumHeight(60)
            info_form_layout.addRow("Wymagane dane:", self.profile_data_required)

            self.profile_hardware_required = QtWidgets.QTextEdit()
            self.profile_hardware_required.setReadOnly(True)
            self.profile_hardware_required.setMaximumHeight(60)
            info_form_layout.addRow("Wymagany sprzęt:", self.profile_hardware_required)

            info_group.setLayout(info_form_layout)
            profile_main_layout.addWidget(info_group)

            profile_buttons_layout = QtWidgets.QHBoxLayout()  # Zmieniono nazwę
            self.edit_profile_btn = QtWidgets.QPushButton("Edytuj profil")
            self.edit_profile_btn.clicked.connect(self._edit_profile)
            profile_buttons_layout.addWidget(self.edit_profile_btn)
            self.apply_profile_btn = QtWidgets.QPushButton("Zastosuj profil")
            self.apply_profile_btn.clicked.connect(self._apply_profile)
            profile_buttons_layout.addWidget(self.apply_profile_btn)
            self.clone_profile_btn = QtWidgets.QPushButton("Klonuj profil")
            self.clone_profile_btn.clicked.connect(self._clone_profile)
            profile_buttons_layout.addWidget(self.clone_profile_btn)
            self.save_profile_btn = QtWidgets.QPushButton("Zapisz profil")
            self.save_profile_btn.clicked.connect(self._save_profile)
            profile_buttons_layout.addWidget(self.save_profile_btn)
            self.delete_profile_btn = QtWidgets.QPushButton("Usuń profil")
            self.delete_profile_btn.clicked.connect(self._delete_profile)
            profile_buttons_layout.addWidget(self.delete_profile_btn)

            profile_main_layout.addLayout(profile_buttons_layout)
            profile_group.setLayout(profile_main_layout)
            layout.addWidget(profile_group)

            layout.addStretch()  # Dodaje elastyczną przestrzeń na dole

            return tab

        except Exception as e:
            msg = "Błąd podczas tworzenia zakładki Dane i Model"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            raise

    def _create_training_params_tab(self):
        try:
            self.logger.debug("Tworzenie zakładki Parametry Treningu")
            tab = QtWidgets.QWidget()
            main_layout = QtWidgets.QVBoxLayout(tab)  # Główny layout dla zakładki

            scroll_area = QtWidgets.QScrollArea()  # Dodajemy ScrollArea
            scroll_area.setWidgetResizable(True)
            scroll_content_widget = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(
                scroll_content_widget
            )  # Layout dla zawartości scrollowanej

            # Model Parameters (moved from Data & Model as per MD)
            model_params_group = QtWidgets.QGroupBox("Parametry Architektury Modelu")
            model_params_form = QtWidgets.QFormLayout()

            self.model_pretrained_check = QtWidgets.QCheckBox(
                "Użyj wstępnie wytrenowanych wag"
            )
            self.model_pretrained_check.setChecked(True)
            model_params_form.addRow(self.model_pretrained_check)

            self.model_pretrained_weights_combo = QtWidgets.QComboBox()
            self.model_pretrained_weights_combo.addItems(
                ["imagenet", "custom"]
            )  # Dodaj więcej opcji jeśli trzeba
            model_params_form.addRow(
                "Źródło wag pretrenowanych:", self.model_pretrained_weights_combo
            )

            self.model_feature_extraction_check = QtWidgets.QCheckBox(
                "Tylko ekstrakcja cech (zamroź wszystkie warstwy poza klasyfikatorem)"
            )
            model_params_form.addRow(self.model_feature_extraction_check)

            self.model_activation_combo = QtWidgets.QComboBox()
            self.model_activation_combo.addItems(
                ["swish", "relu", "gelu", "sigmoid", "tanh"]
            )
            model_params_form.addRow(
                "Funkcja aktywacji w modelu:", self.model_activation_combo
            )

            self.model_dropout_at_inference_check = QtWidgets.QCheckBox(
                "Używaj dropoutu podczas inferencji"
            )
            model_params_form.addRow(self.model_dropout_at_inference_check)

            self.model_global_pool_combo = QtWidgets.QComboBox()
            self.model_global_pool_combo.addItems(["avg", "max", "gem"])
            model_params_form.addRow(
                "Typ global pooling:", self.model_global_pool_combo
            )

            self.model_last_layer_activation_combo = QtWidgets.QComboBox()
            self.model_last_layer_activation_combo.addItems(
                ["softmax", "sigmoid", "none"]
            )
            model_params_form.addRow(
                "Aktywacja ostatniej warstwy:", self.model_last_layer_activation_combo
            )

            model_params_group.setLayout(model_params_form)
            layout.addWidget(model_params_group)

            # Training Hyperparameters
            training_hyperparams_group = QtWidgets.QGroupBox("Hiperparametry Treningu")
            training_hyperparams_form = QtWidgets.QFormLayout()

            self.training_epochs_spin = QtWidgets.QSpinBox()
            self.training_epochs_spin.setRange(1, 10000)
            self.training_epochs_spin.setValue(100)
            training_hyperparams_form.addRow("Liczba epok:", self.training_epochs_spin)

            self.training_batch_size_spin = QtWidgets.QSpinBox()
            self.training_batch_size_spin.setRange(1, 1024)  # Dostosuj zakres
            self.training_batch_size_spin.setValue(32)
            training_hyperparams_form.addRow(
                "Rozmiar batcha:", self.training_batch_size_spin
            )

            self.training_learning_rate_spin = QtWidgets.QDoubleSpinBox()
            self.training_learning_rate_spin.setDecimals(6)
            self.training_learning_rate_spin.setRange(0.000001, 1.0)
            self.training_learning_rate_spin.setSingleStep(0.0001)
            self.training_learning_rate_spin.setValue(0.001)
            training_hyperparams_form.addRow(
                "Współczynnik uczenia (główny):", self.training_learning_rate_spin
            )

            self.training_optimizer_combo = QtWidgets.QComboBox()
            self.training_optimizer_combo.addItems(
                [
                    "AdamW",
                    "Adam",
                    "SGD",
                    "RMSprop",
                    " compréhension",
                    "Adadelta",
                ]  # "compréhension" is likely a typo, but keeping it as per original
            )
            training_hyperparams_form.addRow(
                "Optymalizator:", self.training_optimizer_combo
            )

            # Scheduler
            scheduler_group = QtWidgets.QGroupBox(
                "Harmonogram Współczynnika Uczenia (Scheduler)"
            )
            scheduler_form = QtWidgets.QFormLayout()
            self.training_scheduler_type_combo = QtWidgets.QComboBox()
            self.training_scheduler_type_combo.addItems(
                [
                    "None",
                    "CosineAnnealingWarmRestarts",
                    "StepLR",
                    "OneCycleLR",
                    "ReduceLROnPlateau",
                    "CosineAnnealingLR",
                ]
            )
            scheduler_form.addRow(
                "Typ harmonogramu:", self.training_scheduler_type_combo
            )

            self.training_scheduler_t0_spin = QtWidgets.QSpinBox()
            self.training_scheduler_t0_spin.setRange(1, 1000)
            self.training_scheduler_t0_spin.setValue(10)
            scheduler_form.addRow(
                "T_0 (CosineAnnealingWarmRestarts):", self.training_scheduler_t0_spin
            )

            self.training_scheduler_tmult_spin = QtWidgets.QSpinBox()
            self.training_scheduler_tmult_spin.setRange(1, 10)
            self.training_scheduler_tmult_spin.setValue(2)
            scheduler_form.addRow(
                "T_mult (CosineAnnealingWarmRestarts):",
                self.training_scheduler_tmult_spin,
            )

            self.training_scheduler_eta_min_spin = QtWidgets.QDoubleSpinBox()
            self.training_scheduler_eta_min_spin.setDecimals(7)
            self.training_scheduler_eta_min_spin.setRange(0.0, 0.1)
            self.training_scheduler_eta_min_spin.setSingleStep(0.000001)
            self.training_scheduler_eta_min_spin.setValue(0.000001)
            scheduler_form.addRow(
                "Minimalny LR (eta_min):", self.training_scheduler_eta_min_spin
            )
            scheduler_group.setLayout(scheduler_form)
            training_hyperparams_form.addRow(scheduler_group)

            self.training_num_workers_spin = QtWidgets.QSpinBox()
            self.training_num_workers_spin.setRange(
                0, os.cpu_count() or 4
            )  # Dostosuj zakres
            self.training_num_workers_spin.setValue(min(4, os.cpu_count() or 4))
            training_hyperparams_form.addRow(
                "Liczba wątków (data loader):", self.training_num_workers_spin
            )

            # Warmup
            warmup_group = QtWidgets.QGroupBox("Rozgrzewka (Warmup)")
            warmup_form = QtWidgets.QFormLayout()
            self.training_warmup_epochs_spin = QtWidgets.QSpinBox()
            self.training_warmup_epochs_spin.setRange(0, 100)
            self.training_warmup_epochs_spin.setValue(5)
            warmup_form.addRow(
                "Liczba epok rozgrzewki:", self.training_warmup_epochs_spin
            )

            self.training_warmup_lr_init_spin = QtWidgets.QDoubleSpinBox()
            self.training_warmup_lr_init_spin.setDecimals(7)
            self.training_warmup_lr_init_spin.setRange(0.0000001, 0.1)
            self.training_warmup_lr_init_spin.setSingleStep(0.00001)
            self.training_warmup_lr_init_spin.setValue(0.00001)
            warmup_form.addRow(
                "Początkowy LR dla rozgrzewki:", self.training_warmup_lr_init_spin
            )
            warmup_group.setLayout(warmup_form)
            training_hyperparams_form.addRow(warmup_group)

            self.training_mixed_precision_check = QtWidgets.QCheckBox(
                "Użyj mieszanej precyzji (AMP)"
            )
            training_hyperparams_form.addRow(self.training_mixed_precision_check)

            self.training_grad_accum_steps_spin = QtWidgets.QSpinBox()
            self.training_grad_accum_steps_spin.setRange(1, 64)
            self.training_grad_accum_steps_spin.setValue(1)
            training_hyperparams_form.addRow(
                "Kroki akumulacji gradientu:", self.training_grad_accum_steps_spin
            )

            # Nowy parametr: training.gradient_clip (jeśli różny od regularization.gradient_clip)
            self.training_gradient_clip_value_spin = QtWidgets.QDoubleSpinBox()
            self.training_gradient_clip_value_spin.setRange(
                0.0, 100.0
            )  # 0.0 oznacza brak clip'a
            self.training_gradient_clip_value_spin.setDecimals(2)
            self.training_gradient_clip_value_spin.setValue(0.0)  # Domyślnie wyłączone
            self.training_gradient_clip_value_spin.setToolTip(
                "Wartość przycinania gradientu (0.0 = wyłączone)."
            )
            training_hyperparams_form.addRow(
                "Przycinanie gradientu (wartość):",
                self.training_gradient_clip_value_spin,
            )

            self.training_evaluation_freq_spin = QtWidgets.QSpinBox()
            self.training_evaluation_freq_spin.setRange(1, 100)
            self.training_evaluation_freq_spin.setValue(1)
            training_hyperparams_form.addRow(
                "Częstotliwość ewaluacji (co epok):", self.training_evaluation_freq_spin
            )

            # EMA
            ema_group = QtWidgets.QGroupBox("Exponential Moving Average (EMA)")
            ema_form = QtWidgets.QFormLayout()
            self.training_use_ema_check = QtWidgets.QCheckBox("Użyj EMA")
            ema_form.addRow(self.training_use_ema_check)
            self.training_ema_decay_spin = QtWidgets.QDoubleSpinBox()
            self.training_ema_decay_spin.setRange(0.9, 0.9999)
            self.training_ema_decay_spin.setDecimals(4)
            self.training_ema_decay_spin.setValue(0.999)
            ema_form.addRow("Współczynnik zanikania EMA:", self.training_ema_decay_spin)
            ema_group.setLayout(ema_form)
            training_hyperparams_form.addRow(ema_group)

            # Freeze/Unfreeze
            freeze_unfreeze_group = QtWidgets.QGroupBox(
                "Zamrażanie / Odmrażanie Warstw"
            )
            freeze_unfreeze_form = QtWidgets.QFormLayout()

            self.training_freeze_base_model_check = QtWidgets.QCheckBox(
                "Zamroź wagi bazowego modelu (poza klasyfikatorem) na początku"
            )
            self.training_freeze_base_model_check.setChecked(
                True
            )  # Domyślnie zamrożone dla transfer learning
            freeze_unfreeze_form.addRow(self.training_freeze_base_model_check)

            self.training_unfreeze_layers_edit = QtWidgets.QLineEdit()
            self.training_unfreeze_layers_edit.setPlaceholderText(
                "np. 'all', 'last_3', 'layer_name1,layer_name2'"
            )
            freeze_unfreeze_form.addRow(
                "Warstwy do odmrożenia (nazwy lub 'all'/'last_n'):",
                self.training_unfreeze_layers_edit,
            )

            self.training_unfreeze_strategy_combo = QtWidgets.QComboBox()
            self.training_unfreeze_strategy_combo.addItems(
                [
                    "Wszystkie na raz (unfreeze_all)",
                    "Stopniowo od końca (unfreeze_gradual_end)",
                    "Stopniowo od początku (unfreeze_gradual_start)",
                    f"Po określonej liczbie epok ({self.UNFREEZE_AFTER_EPOCHS})",
                ]
            )
            freeze_unfreeze_form.addRow(
                "Strategia odmrażania:", self.training_unfreeze_strategy_combo
            )

            self.training_unfreeze_after_epochs_spin = QtWidgets.QSpinBox()
            self.training_unfreeze_after_epochs_spin.setRange(0, 1000)
            self.training_unfreeze_after_epochs_spin.setValue(10)
            self.training_unfreeze_after_epochs_spin.setEnabled(
                False
            )  # Domyślnie wyłączone
            freeze_unfreeze_form.addRow(
                "Odmroź po epokach (dla strategii 'Po określonej liczbie epok'):",
                self.training_unfreeze_after_epochs_spin,
            )

            self.unfreeze_strategy_combo = (
                self.training_unfreeze_strategy_combo
            )  # Alias dla _connect_signals
            self.unfreeze_after_epochs_spin = (
                self.training_unfreeze_after_epochs_spin
            )  # Alias dla _connect_signals

            self.training_frozen_lr_spin = QtWidgets.QDoubleSpinBox()
            self.training_frozen_lr_spin.setDecimals(7)
            self.training_frozen_lr_spin.setRange(0.0000001, 0.1)
            self.training_frozen_lr_spin.setValue(0.0001)
            freeze_unfreeze_form.addRow(
                "LR dla zamrożonych warstw (jeśli dotyczy):",
                self.training_frozen_lr_spin,
            )

            self.training_unfrozen_lr_spin = QtWidgets.QDoubleSpinBox()
            self.training_unfrozen_lr_spin.setDecimals(7)
            self.training_unfrozen_lr_spin.setRange(0.0000001, 0.1)
            self.training_unfrozen_lr_spin.setValue(
                0.0001
            )  # Często taki sam jak główny LR lub mniejszy
            freeze_unfreeze_form.addRow(
                "LR dla odmrożonych warstw (jeśli strategia tego wymaga):",
                self.training_unfrozen_lr_spin,
            )

            freeze_unfreeze_group.setLayout(freeze_unfreeze_form)
            training_hyperparams_form.addRow(freeze_unfreeze_group)

            self.training_validation_split_spin = QtWidgets.QDoubleSpinBox()
            self.training_validation_split_spin.setRange(
                0.0, 0.9
            )  # 0.0 oznacza brak wydzielania z treningowego
            self.training_validation_split_spin.setDecimals(2)
            self.training_validation_split_spin.setValue(
                0.0
            )  # Domyślnie nie używaj, jeśli jest osobny val_dir
            self.training_validation_split_spin.setToolTip(
                "Część danych treningowych użyta jako walidacyjne (0.0-0.9). Używane, jeśli katalog walidacyjny nie jest podany lub jest pusty."
            )
            training_hyperparams_form.addRow(
                "Podział na walidację z danych treningowych:",
                self.training_validation_split_spin,
            )

            training_hyperparams_group.setLayout(training_hyperparams_form)
            layout.addWidget(training_hyperparams_group)

            layout.addStretch()
            scroll_content_widget.setLayout(layout)
            scroll_area.setWidget(scroll_content_widget)
            main_layout.addWidget(scroll_area)

            return tab
        except Exception as e:
            msg = "Błąd podczas tworzenia zakładki Parametry Treningu"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            raise

    def _create_regularization_tab(self):
        try:
            self.logger.debug("Tworzenie zakładki Regularyzacja")
            tab = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(
                tab
            )  # Zmieniono na QVBoxLayout dla prostoty, można przywrócić kolumny jeśli trzeba

            reg_form = QtWidgets.QFormLayout()

            self.reg_weight_decay_spin = QtWidgets.QDoubleSpinBox()
            self.reg_weight_decay_spin.setDecimals(6)
            self.reg_weight_decay_spin.setRange(0.0, 1.0)
            self.reg_weight_decay_spin.setSingleStep(0.00001)
            self.reg_weight_decay_spin.setValue(0.0001)
            reg_form.addRow("Weight Decay:", self.reg_weight_decay_spin)

            self.reg_label_smoothing_spin = QtWidgets.QDoubleSpinBox()
            self.reg_label_smoothing_spin.setRange(0.0, 0.5)
            self.reg_label_smoothing_spin.setDecimals(3)
            self.reg_label_smoothing_spin.setValue(0.1)
            reg_form.addRow(
                "Wygładzanie etykiet (Label Smoothing):", self.reg_label_smoothing_spin
            )

            self.reg_dropout_rate_spin = QtWidgets.QDoubleSpinBox()
            self.reg_dropout_rate_spin.setRange(0.0, 0.9)  # Zwiększony zakres
            self.reg_dropout_rate_spin.setDecimals(2)
            self.reg_dropout_rate_spin.setValue(0.2)
            reg_form.addRow(
                "Współczynnik Dropout (dla warstw Dropout):", self.reg_dropout_rate_spin
            )

            self.reg_drop_connect_rate_spin = QtWidgets.QDoubleSpinBox()
            self.reg_drop_connect_rate_spin.setRange(0.0, 0.9)  # Zwiększony zakres
            self.reg_drop_connect_rate_spin.setDecimals(2)
            self.reg_drop_connect_rate_spin.setValue(
                0.2
            )  # Dla modeli typu EfficientNet
            reg_form.addRow(
                "Współczynnik Drop Connect (jeśli model wspiera):",
                self.reg_drop_connect_rate_spin,
            )

            # Gradient Clipping (w Regularyzacji zgodnie z MD)
            self.reg_gradient_clip_spin = QtWidgets.QDoubleSpinBox()
            self.reg_gradient_clip_spin.setRange(0.0, 100.0)  # 0.0 oznacza brak
            self.reg_gradient_clip_spin.setDecimals(2)
            self.reg_gradient_clip_spin.setValue(1.0)  # Popularna wartość
            self.reg_gradient_clip_spin.setToolTip(
                "Maksymalna norma gradientu (0.0 = wyłączone). Różne od gradient_clip_val w Zaawansowanych."
            )
            reg_form.addRow(
                "Przycinanie gradientu (norma):", self.reg_gradient_clip_spin
            )

            # Optimizer specific params (moved from original advanced, if applicable)
            self.reg_momentum_spin = QtWidgets.QDoubleSpinBox()  # Dla SGD, RMSprop
            self.reg_momentum_spin.setRange(0.0, 0.99)
            self.reg_momentum_spin.setDecimals(3)
            self.reg_momentum_spin.setValue(0.9)
            reg_form.addRow("Momentum (dla SGD, RMSprop):", self.reg_momentum_spin)

            self.reg_epsilon_spin = QtWidgets.QDoubleSpinBox()  # Dla Adam, RMSprop
            self.reg_epsilon_spin.setRange(1e-9, 1e-3)
            self.reg_epsilon_spin.setDecimals(9)
            self.reg_epsilon_spin.setValue(1e-8)  # Adam default to 1e-8
            reg_form.addRow("Epsilon (dla Adam, RMSprop):", self.reg_epsilon_spin)

            # Stochastic Depth
            stoch_depth_group = QtWidgets.QGroupBox("Stochastic Depth")
            stoch_depth_form = QtWidgets.QFormLayout()
            self.stochastic_depth_use_check = QtWidgets.QCheckBox(
                "Używaj Stochastic Depth"
            )  # Zmieniono nazwę atrybutu
            stoch_depth_form.addRow(self.stochastic_depth_use_check)
            self.stochastic_depth_survival_prob_spin = QtWidgets.QDoubleSpinBox()
            self.stochastic_depth_survival_prob_spin.setRange(0.5, 1.0)
            self.stochastic_depth_survival_prob_spin.setDecimals(3)
            self.stochastic_depth_survival_prob_spin.setValue(0.8)
            self.stochastic_depth_survival_prob_spin.setEnabled(False)
            stoch_depth_form.addRow(
                "Prawdopodobieństwo przetrwania:",
                self.stochastic_depth_survival_prob_spin,
            )
            stoch_depth_group.setLayout(stoch_depth_form)
            reg_form.addRow(stoch_depth_group)

            # SWA
            swa_group = QtWidgets.QGroupBox("Stochastic Weight Averaging (SWA)")
            swa_form = QtWidgets.QFormLayout()
            self.use_swa_check = QtWidgets.QCheckBox(
                "Używaj SWA"
            )  # Nazwa z oryginalnego kodu
            swa_form.addRow(self.use_swa_check)
            self.swa_start_epoch_spin = QtWidgets.QSpinBox()
            self.swa_start_epoch_spin.setRange(1, 10000)
            self.swa_start_epoch_spin.setValue(10)
            self.swa_start_epoch_spin.setEnabled(False)
            swa_form.addRow("Epoka rozpoczęcia SWA:", self.swa_start_epoch_spin)
            self.swa_lr_spin = QtWidgets.QDoubleSpinBox()
            self.swa_lr_spin.setRange(1e-7, 1e-2)
            self.swa_lr_spin.setDecimals(7)
            self.swa_lr_spin.setValue(5e-5)
            self.swa_lr_spin.setEnabled(False)
            swa_form.addRow("Learning rate dla SWA:", self.swa_lr_spin)
            swa_group.setLayout(swa_form)
            reg_form.addRow(swa_group)

            layout.addLayout(reg_form)
            layout.addStretch()
            return tab

        except Exception as e:
            msg = "Błąd podczas tworzenia zakładki Regularyzacja"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            raise

    def _create_augmentation_tab(self):
        try:
            self.logger.debug("Tworzenie zakładki Augmentacja")
            tab = QtWidgets.QWidget()
            main_layout = QtWidgets.QVBoxLayout(tab)

            scroll_area = QtWidgets.QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_content_widget = QtWidgets.QWidget()
            layout = QtWidgets.QGridLayout(
                scroll_content_widget
            )  # Użyjemy Grida dla 2 kolumn

            # Basic Augmentations
            basic_aug_group = QtWidgets.QGroupBox("Podstawowe Augmentacje")
            basic_aug_form = QtWidgets.QFormLayout()
            self.basic_aug_check = QtWidgets.QCheckBox("Użyj podstawowych augmentacji")
            basic_aug_form.addRow(self.basic_aug_check)

            # Zmiana nazw atrybutów z aug_basic_* na basic_aug_* (Poprawka 8)
            self.basic_aug_rotation_spin = QtWidgets.QSpinBox()
            self.basic_aug_rotation_spin.setRange(0, 180)
            self.basic_aug_rotation_spin.setValue(15)
            basic_aug_form.addRow(
                "Maksymalny kąt rotacji:", self.basic_aug_rotation_spin
            )
            self.basic_aug_brightness_spin = QtWidgets.QDoubleSpinBox()
            self.basic_aug_brightness_spin.setRange(0.0, 1.0)
            self.basic_aug_brightness_spin.setValue(0.2)
            self.basic_aug_brightness_spin.setDecimals(2)
            basic_aug_form.addRow(
                "Zakres zmiany jasności:", self.basic_aug_brightness_spin
            )
            self.basic_aug_contrast_spin = QtWidgets.QDoubleSpinBox()
            self.basic_aug_contrast_spin.setRange(0.0, 1.0)
            self.basic_aug_contrast_spin.setValue(0.2)
            self.basic_aug_contrast_spin.setDecimals(2)
            basic_aug_form.addRow(
                "Zakres zmiany kontrastu:", self.basic_aug_contrast_spin
            )
            self.basic_aug_saturation_spin = QtWidgets.QDoubleSpinBox()
            self.basic_aug_saturation_spin.setRange(0.0, 1.0)
            self.basic_aug_saturation_spin.setValue(0.2)
            self.basic_aug_saturation_spin.setDecimals(2)
            basic_aug_form.addRow(
                "Zakres zmiany nasycenia:", self.basic_aug_saturation_spin
            )
            self.basic_aug_hue_spin = QtWidgets.QDoubleSpinBox()
            self.basic_aug_hue_spin.setRange(0.0, 0.5)
            self.basic_aug_hue_spin.setValue(0.1)
            self.basic_aug_hue_spin.setDecimals(2)
            basic_aug_form.addRow(
                "Zakres zmiany odcienia (hue):", self.basic_aug_hue_spin
            )
            self.basic_aug_shift_spin = QtWidgets.QDoubleSpinBox()  # Max % przesunięcia
            self.basic_aug_shift_spin.setRange(0.0, 0.5)
            self.basic_aug_shift_spin.setValue(0.1)
            self.basic_aug_shift_spin.setDecimals(2)
            basic_aug_form.addRow(
                "Maksymalne przesunięcie (% obrazu):", self.basic_aug_shift_spin
            )
            self.basic_aug_zoom_spin = QtWidgets.QDoubleSpinBox()  # Zakres zoomu
            self.basic_aug_zoom_spin.setRange(0.0, 0.5)
            self.basic_aug_zoom_spin.setValue(0.1)
            self.basic_aug_zoom_spin.setDecimals(2)
            basic_aug_form.addRow(
                "Zakres powiększenia/zmniejszenia:", self.basic_aug_zoom_spin
            )
            self.basic_aug_horizontal_flip_check = QtWidgets.QCheckBox(
                "Odbicie poziome"
            )
            self.basic_aug_horizontal_flip_check.setChecked(True)
            basic_aug_form.addRow(self.basic_aug_horizontal_flip_check)
            self.basic_aug_vertical_flip_check = QtWidgets.QCheckBox("Odbicie pionowe")
            basic_aug_form.addRow(self.basic_aug_vertical_flip_check)
            basic_aug_group.setLayout(basic_aug_form)
            layout.addWidget(basic_aug_group, 0, 0)

            # Mixup
            mixup_group = QtWidgets.QGroupBox("Mixup")
            mixup_form = QtWidgets.QFormLayout()
            self.mixup_check = QtWidgets.QCheckBox("Użyj Mixup")
            mixup_form.addRow(self.mixup_check)
            self.mixup_alpha_spin = QtWidgets.QDoubleSpinBox()
            self.mixup_alpha_spin.setRange(0.0, 5.0)
            self.mixup_alpha_spin.setValue(0.4)
            self.mixup_alpha_spin.setDecimals(2)  # Typowe wartości 0.1-0.4
            self.mixup_alpha_spin.setEnabled(False)
            mixup_form.addRow("Parametr Alpha:", self.mixup_alpha_spin)
            mixup_group.setLayout(mixup_form)
            layout.addWidget(mixup_group, 1, 0)

            # CutMix
            cutmix_group = QtWidgets.QGroupBox("CutMix")
            cutmix_form = QtWidgets.QFormLayout()
            self.cutmix_check = QtWidgets.QCheckBox("Użyj CutMix")
            cutmix_form.addRow(self.cutmix_check)
            self.cutmix_alpha_spin = QtWidgets.QDoubleSpinBox()
            self.cutmix_alpha_spin.setRange(0.0, 5.0)
            self.cutmix_alpha_spin.setValue(1.0)
            self.cutmix_alpha_spin.setDecimals(2)  # Typowa wartość 1.0
            self.cutmix_alpha_spin.setEnabled(False)
            cutmix_form.addRow("Parametr Alpha:", self.cutmix_alpha_spin)
            cutmix_group.setLayout(cutmix_form)
            layout.addWidget(cutmix_group, 2, 0)

            # AutoAugment
            autoaugment_group = QtWidgets.QGroupBox("AutoAugment")
            autoaugment_form = QtWidgets.QFormLayout()
            self.autoaugment_check = QtWidgets.QCheckBox("Użyj AutoAugment")
            autoaugment_form.addRow(self.autoaugment_check)
            self.autoaugment_policy_combo = QtWidgets.QComboBox()
            self.autoaugment_policy_combo.addItems(
                ["imagenet", "cifar10", "svhn"]
            )  # Zgodnie z torchvision
            self.autoaugment_policy_combo.setEnabled(False)
            autoaugment_form.addRow("Polityka:", self.autoaugment_policy_combo)
            autoaugment_group.setLayout(autoaugment_form)
            layout.addWidget(autoaugment_group, 0, 1)

            # RandAugment
            randaugment_group = QtWidgets.QGroupBox("RandAugment")
            randaugment_form = QtWidgets.QFormLayout()
            self.randaugment_check = QtWidgets.QCheckBox("Użyj RandAugment")
            randaugment_form.addRow(self.randaugment_check)
            self.randaugment_n_spin = QtWidgets.QSpinBox()  # Num ops
            self.randaugment_n_spin.setRange(1, 10)
            self.randaugment_n_spin.setValue(2)
            self.randaugment_n_spin.setEnabled(False)
            randaugment_form.addRow("Liczba operacji (N):", self.randaugment_n_spin)
            self.randaugment_m_spin = QtWidgets.QSpinBox()  # Magnitude
            self.randaugment_m_spin.setRange(1, 30)
            self.randaugment_m_spin.setValue(9)
            self.randaugment_m_spin.setEnabled(False)
            randaugment_form.addRow(
                "Intensywność operacji (M):", self.randaugment_m_spin
            )
            randaugment_group.setLayout(randaugment_form)
            layout.addWidget(randaugment_group, 1, 1)

            # TrivialAugment
            trivialaugment_group = QtWidgets.QGroupBox(
                "TrivialAugmentWide"
            )  # torchvision.transforms.TrivialAugmentWide
            trivialaugment_form = QtWidgets.QFormLayout()
            self.trivialaugment_check = QtWidgets.QCheckBox("Użyj TrivialAugmentWide")
            trivialaugment_form.addRow(self.trivialaugment_check)
            trivialaugment_group.setLayout(trivialaugment_form)
            layout.addWidget(trivialaugment_group, 2, 1)

            # Random Erase
            random_erase_group = QtWidgets.QGroupBox("Random Erase")
            random_erase_form = QtWidgets.QFormLayout()
            self.random_erase_check = QtWidgets.QCheckBox("Użyj Random Erase")
            random_erase_form.addRow(self.random_erase_check)
            self.random_erase_prob_spin = QtWidgets.QDoubleSpinBox()
            self.random_erase_prob_spin.setRange(0.0, 1.0)
            self.random_erase_prob_spin.setValue(0.5)
            self.random_erase_prob_spin.setDecimals(2)
            self.random_erase_prob_spin.setEnabled(False)
            random_erase_form.addRow("Prawdopodobieństwo:", self.random_erase_prob_spin)
            # Scale
            scale_layout = QtWidgets.QHBoxLayout()
            self.random_erase_scale_min_spin = QtWidgets.QDoubleSpinBox()
            self.random_erase_scale_min_spin.setRange(0.01, 1.0)
            self.random_erase_scale_min_spin.setValue(0.02)
            self.random_erase_scale_min_spin.setDecimals(3)
            self.random_erase_scale_max_spin = QtWidgets.QDoubleSpinBox()
            self.random_erase_scale_max_spin.setRange(0.01, 1.0)
            self.random_erase_scale_max_spin.setValue(0.33)
            self.random_erase_scale_max_spin.setDecimals(3)
            scale_layout.addWidget(QtWidgets.QLabel("Min:"))
            scale_layout.addWidget(self.random_erase_scale_min_spin)
            scale_layout.addWidget(QtWidgets.QLabel("Max:"))
            scale_layout.addWidget(self.random_erase_scale_max_spin)
            random_erase_form.addRow("Zakres skali wycinanego obszaru:", scale_layout)
            # Ratio
            ratio_layout = QtWidgets.QHBoxLayout()
            self.random_erase_ratio_min_spin = QtWidgets.QDoubleSpinBox()
            self.random_erase_ratio_min_spin.setRange(0.1, 10.0)
            self.random_erase_ratio_min_spin.setValue(0.3)
            self.random_erase_ratio_min_spin.setDecimals(2)
            self.random_erase_ratio_max_spin = QtWidgets.QDoubleSpinBox()
            self.random_erase_ratio_max_spin.setRange(0.1, 10.0)
            self.random_erase_ratio_max_spin.setValue(3.3)
            self.random_erase_ratio_max_spin.setDecimals(2)
            ratio_layout.addWidget(QtWidgets.QLabel("Min:"))
            ratio_layout.addWidget(self.random_erase_ratio_min_spin)
            ratio_layout.addWidget(QtWidgets.QLabel("Max:"))
            ratio_layout.addWidget(self.random_erase_ratio_max_spin)
            random_erase_form.addRow(
                "Zakres proporcji wycinanego obszaru:", ratio_layout
            )
            random_erase_group.setLayout(random_erase_form)
            layout.addWidget(random_erase_group, 3, 0, 1, 2)  # Rozciągnij na 2 kolumny

            # Grid Distortion
            grid_distortion_group = QtWidgets.QGroupBox(
                "Grid Distortion (Albumentations)"
            )
            grid_distortion_form = QtWidgets.QFormLayout()
            self.grid_distortion_check = QtWidgets.QCheckBox(
                "Użyj zniekształcenia siatki"
            )
            grid_distortion_form.addRow(self.grid_distortion_check)
            self.grid_distortion_prob_spin = QtWidgets.QDoubleSpinBox()
            self.grid_distortion_prob_spin.setRange(0.0, 1.0)
            self.grid_distortion_prob_spin.setValue(0.5)
            self.grid_distortion_prob_spin.setDecimals(2)
            self.grid_distortion_prob_spin.setEnabled(False)
            grid_distortion_form.addRow(
                "Prawdopodobieństwo:", self.grid_distortion_prob_spin
            )
            self.grid_distortion_limit_spin = QtWidgets.QDoubleSpinBox()
            self.grid_distortion_limit_spin.setRange(0.0, 1.0)
            self.grid_distortion_limit_spin.setValue(0.3)
            self.grid_distortion_limit_spin.setDecimals(2)
            self.grid_distortion_limit_spin.setEnabled(False)
            grid_distortion_form.addRow(
                "Limit zniekształcenia:", self.grid_distortion_limit_spin
            )
            grid_distortion_group.setLayout(grid_distortion_form)
            layout.addWidget(grid_distortion_group, 4, 0, 1, 2)

            # Kontrolka resize z augmentacji (jeśli jest inna niż w preprocessingu)
            # MD: augmentation.resize.enabled. Kontrolka: self.resize_check.
            # W kodzie była tworzona i w aug i w preproc. Użyjemy self.aug_resize_check
            aug_resize_group = QtWidgets.QGroupBox(
                "Zmiana rozmiaru (w ramach augmentacji)"
            )
            aug_resize_form = QtWidgets.QFormLayout()
            self.aug_resize_enabled_check = QtWidgets.QCheckBox(
                "Włącz zmianę rozmiaru w potoku augmentacji"
            )
            self.aug_resize_enabled_check.setToolTip(
                "Może być użyte np. przed RandomResizedCrop, jeśli obrazy wejściowe są bardzo różne."
            )
            # Tutaj można dodać parametry rozmiaru, jeśli są inne niż w preprocessingu
            aug_resize_form.addRow(self.aug_resize_enabled_check)
            aug_resize_group.setLayout(aug_resize_form)
            layout.addWidget(aug_resize_group, 5, 0, 1, 2)

            layout.setRowStretch(layout.rowCount(), 1)  # Dodaje stretch na końcu siatki
            scroll_content_widget.setLayout(layout)
            scroll_area.setWidget(scroll_content_widget)
            main_layout.addWidget(scroll_area)

            return tab
        except Exception as e:
            msg = "Błąd podczas tworzenia zakładki Augmentacja"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            raise

    def _create_preprocessing_tab(self):
        try:
            self.logger.debug("Tworzenie zakładki Preprocessing")
            tab = QtWidgets.QWidget()
            main_layout = QtWidgets.QVBoxLayout(tab)

            scroll_area = QtWidgets.QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_content_widget = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(scroll_content_widget)

            # Resize
            resize_group = QtWidgets.QGroupBox("Zmiana Rozmiaru Obrazu (Główna)")
            resize_form = QtWidgets.QFormLayout()
            self.preprocess_resize_enabled_check = QtWidgets.QCheckBox(
                "Włącz zmianę rozmiaru"
            )
            self.preprocess_resize_enabled_check.setChecked(True)
            resize_form.addRow(self.preprocess_resize_enabled_check)

            size_layout = QtWidgets.QHBoxLayout()
            self.preprocess_resize_width_spin = QtWidgets.QSpinBox()
            self.preprocess_resize_width_spin.setRange(32, 4096)
            self.preprocess_resize_width_spin.setValue(256)
            self.preprocess_resize_height_spin = QtWidgets.QSpinBox()
            self.preprocess_resize_height_spin.setRange(32, 4096)
            self.preprocess_resize_height_spin.setValue(256)
            size_layout.addWidget(QtWidgets.QLabel("Szerokość:"))
            size_layout.addWidget(self.preprocess_resize_width_spin)
            size_layout.addWidget(QtWidgets.QLabel("Wysokość:"))
            size_layout.addWidget(self.preprocess_resize_height_spin)
            resize_form.addRow("Docelowy rozmiar [szer, wys]:", size_layout)

            self.preprocess_resize_mode_combo = QtWidgets.QComboBox()
            self.preprocess_resize_mode_combo.addItems(
                ["bilinear", "bicubic", "nearest", "lanczos"]
            )  # torchvision.transforms.InterpolationMode
            resize_form.addRow(
                "Tryb zmiany rozmiaru (interpolacja):",
                self.preprocess_resize_mode_combo,
            )
            resize_group.setLayout(resize_form)
            layout.addWidget(resize_group)

            # Normalization
            norm_group = QtWidgets.QGroupBox("Normalizacja")
            norm_form = QtWidgets.QFormLayout()
            self.preprocess_normalize_enabled_check = QtWidgets.QCheckBox(
                "Włącz normalizację"
            )
            self.preprocess_normalize_enabled_check.setChecked(True)
            norm_form.addRow(self.preprocess_normalize_enabled_check)

            mean_layout = QtWidgets.QHBoxLayout()
            self.preprocess_normalize_mean_r_spin = QtWidgets.QDoubleSpinBox()
            self.preprocess_normalize_mean_r_spin.setDecimals(3)
            self.preprocess_normalize_mean_r_spin.setValue(0.485)
            self.preprocess_normalize_mean_g_spin = QtWidgets.QDoubleSpinBox()
            self.preprocess_normalize_mean_g_spin.setDecimals(3)
            self.preprocess_normalize_mean_g_spin.setValue(0.456)
            self.preprocess_normalize_mean_b_spin = QtWidgets.QDoubleSpinBox()
            self.preprocess_normalize_mean_b_spin.setDecimals(3)
            self.preprocess_normalize_mean_b_spin.setValue(0.406)
            mean_layout.addWidget(QtWidgets.QLabel("R:"))
            mean_layout.addWidget(self.preprocess_normalize_mean_r_spin)
            mean_layout.addWidget(QtWidgets.QLabel("G:"))
            mean_layout.addWidget(self.preprocess_normalize_mean_g_spin)
            mean_layout.addWidget(QtWidgets.QLabel("B:"))
            mean_layout.addWidget(self.preprocess_normalize_mean_b_spin)
            norm_form.addRow("Średnie [R, G, B]:", mean_layout)

            std_layout = QtWidgets.QHBoxLayout()
            self.preprocess_normalize_std_r_spin = QtWidgets.QDoubleSpinBox()
            self.preprocess_normalize_std_r_spin.setDecimals(3)
            self.preprocess_normalize_std_r_spin.setValue(0.229)
            self.preprocess_normalize_std_g_spin = QtWidgets.QDoubleSpinBox()
            self.preprocess_normalize_std_g_spin.setDecimals(3)
            self.preprocess_normalize_std_g_spin.setValue(0.224)
            self.preprocess_normalize_std_b_spin = QtWidgets.QDoubleSpinBox()
            self.preprocess_normalize_std_b_spin.setDecimals(3)
            self.preprocess_normalize_std_b_spin.setValue(0.225)
            std_layout.addWidget(QtWidgets.QLabel("R:"))
            std_layout.addWidget(self.preprocess_normalize_std_r_spin)
            std_layout.addWidget(QtWidgets.QLabel("G:"))
            std_layout.addWidget(self.preprocess_normalize_std_g_spin)
            std_layout.addWidget(QtWidgets.QLabel("B:"))
            std_layout.addWidget(self.preprocess_normalize_std_b_spin)
            norm_form.addRow("Odch. std. [R, G, B]:", std_layout)
            norm_group.setLayout(norm_form)
            layout.addWidget(norm_group)

            # Grayscale
            gray_group = QtWidgets.QGroupBox("Konwersja do Skali Szarości")
            gray_form = QtWidgets.QFormLayout()
            self.preprocess_grayscale_enabled_check = QtWidgets.QCheckBox(
                "Konwertuj do skali szarości"
            )
            gray_form.addRow(self.preprocess_grayscale_enabled_check)
            self.preprocess_grayscale_num_output_channels_spin = QtWidgets.QSpinBox()
            self.preprocess_grayscale_num_output_channels_spin.setRange(1, 3)
            self.preprocess_grayscale_num_output_channels_spin.setValue(1)  # 1 or 3
            self.preprocess_grayscale_num_output_channels_spin.setEnabled(False)
            gray_form.addRow(
                "Liczba kanałów wyjściowych (1 lub 3):",
                self.preprocess_grayscale_num_output_channels_spin,
            )
            gray_group.setLayout(gray_form)
            layout.addWidget(gray_group)

            # Color Jitter (w Preprocessingu, jeśli MD to sugeruje jako osobne)
            cj_group = QtWidgets.QGroupBox("Modyfikacja Kolorów (Color Jitter)")
            cj_form = QtWidgets.QFormLayout()
            self.preprocess_color_jitter_enabled_check = QtWidgets.QCheckBox(
                "Włącz Color Jitter"
            )
            cj_form.addRow(self.preprocess_color_jitter_enabled_check)
            self.preprocess_color_jitter_brightness_spin = QtWidgets.QDoubleSpinBox()
            self.preprocess_color_jitter_brightness_spin.setRange(0, 2)
            self.preprocess_color_jitter_brightness_spin.setValue(0.2)
            self.preprocess_color_jitter_brightness_spin.setDecimals(2)
            cj_form.addRow(
                "Jasność (zakres [max(0, 1-br), 1+br]):",
                self.preprocess_color_jitter_brightness_spin,
            )
            # ... analogicznie contrast, saturation, hue
            self.preprocess_color_jitter_contrast_spin = QtWidgets.QDoubleSpinBox()
            self.preprocess_color_jitter_contrast_spin.setRange(0, 2)
            self.preprocess_color_jitter_contrast_spin.setValue(0.2)
            self.preprocess_color_jitter_contrast_spin.setDecimals(2)
            cj_form.addRow(
                "Kontrast (zakres):", self.preprocess_color_jitter_contrast_spin
            )
            self.preprocess_color_jitter_saturation_spin = QtWidgets.QDoubleSpinBox()
            self.preprocess_color_jitter_saturation_spin.setRange(0, 2)
            self.preprocess_color_jitter_saturation_spin.setValue(0.2)
            self.preprocess_color_jitter_saturation_spin.setDecimals(2)
            cj_form.addRow(
                "Nasycenie (zakres):", self.preprocess_color_jitter_saturation_spin
            )
            self.preprocess_color_jitter_hue_spin = QtWidgets.QDoubleSpinBox()
            self.preprocess_color_jitter_hue_spin.setRange(-0.5, 0.5)
            self.preprocess_color_jitter_hue_spin.setValue(0.1)
            self.preprocess_color_jitter_hue_spin.setDecimals(2)
            cj_form.addRow(
                "Odcień (zakres [-hue, hue]):", self.preprocess_color_jitter_hue_spin
            )
            cj_group.setLayout(cj_form)
            layout.addWidget(cj_group)

            # Gaussian Blur
            blur_group = QtWidgets.QGroupBox("Rozmycie Gaussa")
            blur_form = QtWidgets.QFormLayout()
            self.preprocess_gaussian_blur_enabled_check = QtWidgets.QCheckBox(
                "Włącz rozmycie Gaussa"
            )
            blur_form.addRow(self.preprocess_gaussian_blur_enabled_check)
            self.preprocess_gaussian_blur_kernel_size_spin = (
                QtWidgets.QSpinBox()
            )  # Musi być nieparzyste
            self.preprocess_gaussian_blur_kernel_size_spin.setRange(1, 99)
            self.preprocess_gaussian_blur_kernel_size_spin.setValue(3)
            self.preprocess_gaussian_blur_kernel_size_spin.setSingleStep(2)
            blur_form.addRow(
                "Rozmiar jądra (nieparzysty):",
                self.preprocess_gaussian_blur_kernel_size_spin,
            )
            self.preprocess_gaussian_blur_sigma_spin = QtWidgets.QDoubleSpinBox()
            self.preprocess_gaussian_blur_sigma_spin.setRange(0.1, 10.0)
            self.preprocess_gaussian_blur_sigma_spin.setValue(1.0)
            self.preprocess_gaussian_blur_sigma_spin.setDecimals(2)
            blur_form.addRow(
                "Sigma (odch. std.):", self.preprocess_gaussian_blur_sigma_spin
            )
            blur_group.setLayout(blur_form)
            layout.addWidget(blur_group)

            # Scaling options from MD (previously in _create_advanced_tab wrongly)
            scaling_group = QtWidgets.QGroupBox("Zaawansowane Skalowanie/Padding")
            scaling_form = QtWidgets.QFormLayout()
            self.preprocess_scaling_method_combo = QtWidgets.QComboBox()
            self.preprocess_scaling_method_combo.addItems(
                ["Bilinear", "Bicubic", "Nearest", "Lanczos", "Area"]
            )  # Zgodne z kodem
            scaling_form.addRow(
                "Metoda skalowania (jeśli inna niż resize_mode):",
                self.preprocess_scaling_method_combo,
            )
            self.preprocess_scaling_maintain_aspect_ratio_check = QtWidgets.QCheckBox(
                "Zachowaj proporcje obrazu"
            )
            scaling_form.addRow(self.preprocess_scaling_maintain_aspect_ratio_check)
            self.preprocess_scaling_pad_to_square_check = QtWidgets.QCheckBox(
                "Dopełnij do kwadratu (po skalowaniu z zach. proporcji)"
            )
            scaling_form.addRow(self.preprocess_scaling_pad_to_square_check)
            self.preprocess_scaling_pad_mode_combo = QtWidgets.QComboBox()
            self.preprocess_scaling_pad_mode_combo.addItems(
                ["constant", "edge", "reflect", "symmetric"]
            )  # Zgodne z kodem
            scaling_form.addRow(
                "Tryb dopełnienia (paddingu):", self.preprocess_scaling_pad_mode_combo
            )
            self.preprocess_scaling_pad_value_spin = QtWidgets.QSpinBox()
            self.preprocess_scaling_pad_value_spin.setRange(0, 255)
            self.preprocess_scaling_pad_value_spin.setValue(0)
            scaling_form.addRow(
                "Wartość dopełnienia (dla 'constant'):",
                self.preprocess_scaling_pad_value_spin,
            )
            scaling_group.setLayout(scaling_form)
            layout.addWidget(scaling_group)

            # RandomResizedCrop
            rrc_group = QtWidgets.QGroupBox(
                "Losowe Przycinanie ze Zmianą Rozmiaru (RandomResizedCrop)"
            )
            rrc_form = QtWidgets.QFormLayout()
            self.preprocess_random_resize_crop_enabled_check = QtWidgets.QCheckBox(
                "Użyj RandomResizedCrop"
            )
            rrc_form.addRow(self.preprocess_random_resize_crop_enabled_check)
            self.preprocess_random_resize_crop_size_spin = (
                QtWidgets.QSpinBox()
            )  # Docelowy rozmiar
            self.preprocess_random_resize_crop_size_spin.setRange(32, 1024)
            self.preprocess_random_resize_crop_size_spin.setValue(224)
            rrc_form.addRow(
                "Docelowy rozmiar po przycięciu:",
                self.preprocess_random_resize_crop_size_spin,
            )
            # Scale for RRC
            rrc_scale_layout = QtWidgets.QHBoxLayout()
            self.preprocess_random_resize_crop_scale_min_spin = (
                QtWidgets.QDoubleSpinBox()
            )
            self.preprocess_random_resize_crop_scale_min_spin.setRange(0.01, 1.0)
            self.preprocess_random_resize_crop_scale_min_spin.setValue(0.08)
            self.preprocess_random_resize_crop_scale_min_spin.setDecimals(3)
            self.preprocess_random_resize_crop_scale_max_spin = (
                QtWidgets.QDoubleSpinBox()
            )
            self.preprocess_random_resize_crop_scale_max_spin.setRange(0.01, 1.0)
            self.preprocess_random_resize_crop_scale_max_spin.setValue(1.0)
            self.preprocess_random_resize_crop_scale_max_spin.setDecimals(3)
            rrc_scale_layout.addWidget(QtWidgets.QLabel("Min:"))
            rrc_scale_layout.addWidget(
                self.preprocess_random_resize_crop_scale_min_spin
            )
            rrc_scale_layout.addWidget(QtWidgets.QLabel("Max:"))
            rrc_scale_layout.addWidget(
                self.preprocess_random_resize_crop_scale_max_spin
            )
            rrc_form.addRow(
                "Zakres skali przycinania (proporcja oryginału):", rrc_scale_layout
            )
            # Ratio for RRC
            rrc_ratio_layout = QtWidgets.QHBoxLayout()
            self.preprocess_random_resize_crop_ratio_min_spin = (
                QtWidgets.QDoubleSpinBox()
            )
            self.preprocess_random_resize_crop_ratio_min_spin.setRange(0.1, 10.0)
            self.preprocess_random_resize_crop_ratio_min_spin.setValue(0.75)
            self.preprocess_random_resize_crop_ratio_min_spin.setDecimals(3)  # 3./4.
            self.preprocess_random_resize_crop_ratio_max_spin = (
                QtWidgets.QDoubleSpinBox()
            )
            self.preprocess_random_resize_crop_ratio_max_spin.setRange(0.1, 10.0)
            self.preprocess_random_resize_crop_ratio_max_spin.setValue(1.33)
            self.preprocess_random_resize_crop_ratio_max_spin.setDecimals(3)  # 4./3.
            rrc_ratio_layout.addWidget(QtWidgets.QLabel("Min:"))
            rrc_ratio_layout.addWidget(
                self.preprocess_random_resize_crop_ratio_min_spin
            )
            rrc_ratio_layout.addWidget(QtWidgets.QLabel("Max:"))
            rrc_ratio_layout.addWidget(
                self.preprocess_random_resize_crop_ratio_max_spin
            )
            rrc_form.addRow("Zakres proporcji przycinania:", rrc_ratio_layout)
            rrc_group.setLayout(rrc_form)
            layout.addWidget(rrc_group)

            self.preprocess_cache_dataset_check = QtWidgets.QCheckBox(
                "Cachuj przetworzony zestaw danych w pamięci RAM"
            )
            layout.addWidget(self.preprocess_cache_dataset_check)

            layout.addStretch()
            scroll_content_widget.setLayout(layout)
            scroll_area.setWidget(scroll_content_widget)
            main_layout.addWidget(scroll_area)

            return tab
        except Exception as e:
            msg = "Błąd podczas tworzenia zakładki Preprocessing"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            raise

    def _create_monitoring_tab(self):
        try:
            self.logger.debug("Tworzenie zakładki Monitorowanie i Logowanie")
            tab = QtWidgets.QWidget()
            main_layout = QtWidgets.QVBoxLayout(tab)

            scroll_area = QtWidgets.QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_content_widget = QtWidgets.QWidget()
            layout = QtWidgets.QGridLayout(scroll_content_widget)  # Grid dla 2 kolumn

            # Metrics
            metrics_group = QtWidgets.QGroupBox("Obliczane Metryki")
            metrics_form = QtWidgets.QFormLayout()
            self.metrics_accuracy_check = QtWidgets.QCheckBox("Dokładność (Accuracy)")
            self.metrics_accuracy_check.setChecked(True)
            metrics_form.addRow(self.metrics_accuracy_check)
            self.metrics_precision_check = QtWidgets.QCheckBox("Precyzja (Precision)")
            self.metrics_precision_check.setChecked(True)
            metrics_form.addRow(self.metrics_precision_check)
            self.metrics_recall_check = QtWidgets.QCheckBox("Czułość (Recall)")
            self.metrics_recall_check.setChecked(True)
            metrics_form.addRow(self.metrics_recall_check)
            self.metrics_f1_check = QtWidgets.QCheckBox("F1-Score")
            self.metrics_f1_check.setChecked(True)
            metrics_form.addRow(self.metrics_f1_check)
            self.metrics_topk_check = QtWidgets.QCheckBox(
                "Top-k Accuracy (k zostanie określone przez silnik, np. [2,5])"
            )  # Uproszczone do bool, lista k jest poza UI
            metrics_form.addRow(self.metrics_topk_check)
            self.metrics_confusion_matrix_check = QtWidgets.QCheckBox(
                "Macierz Pomyłek (Confusion Matrix)"
            )
            metrics_form.addRow(self.metrics_confusion_matrix_check)
            self.metrics_auc_check = QtWidgets.QCheckBox("AUC-ROC")
            metrics_form.addRow(self.metrics_auc_check)
            # Nowe metryki systemowe
            self.metrics_gpu_utilization_check = QtWidgets.QCheckBox(
                "Monitoruj wykorzystanie GPU"
            )
            metrics_form.addRow(self.metrics_gpu_utilization_check)
            self.metrics_memory_usage_check = QtWidgets.QCheckBox(
                "Monitoruj zużycie pamięci (RAM/VRAM)"
            )
            metrics_form.addRow(self.metrics_memory_usage_check)
            metrics_group.setLayout(metrics_form)
            layout.addWidget(metrics_group, 0, 0)

            # TensorBoard
            tensorboard_group = QtWidgets.QGroupBox("Logowanie do TensorBoard")
            tensorboard_form = QtWidgets.QFormLayout()
            self.tensorboard_enabled_check = QtWidgets.QCheckBox(
                "Włącz logowanie do TensorBoard"
            )  # Zmiana nazwy atrybutu
            self.tensorboard_enabled_check.setChecked(True)
            tensorboard_form.addRow(self.tensorboard_enabled_check)
            self.tensorboard_log_dir_edit = QtWidgets.QLineEdit(
                "logs/tensorboard"
            )  # Zmiana nazwy atrybutu
            self.tensorboard_log_dir_edit.setEnabled(
                self.tensorboard_enabled_check.isChecked()
            )
            tensorboard_form.addRow(
                "Katalog logów TensorBoard:", self.tensorboard_log_dir_edit
            )
            self.tensorboard_update_freq_spin = QtWidgets.QSpinBox()  # steps or epochs
            self.tensorboard_update_freq_spin.setRange(1, 1000)
            self.tensorboard_update_freq_spin.setValue(100)  # Np. co 100 kroków
            self.tensorboard_update_freq_spin.setEnabled(
                self.tensorboard_enabled_check.isChecked()
            )
            tensorboard_form.addRow(
                "Częstotliwość aktualizacji logów (kroki):",
                self.tensorboard_update_freq_spin,
            )
            tensorboard_group.setLayout(tensorboard_form)
            layout.addWidget(tensorboard_group, 1, 0)

            # Weights & Biases (W&B)
            wandb_group = QtWidgets.QGroupBox("Logowanie do Weights & Biases")
            wandb_form = QtWidgets.QFormLayout()
            self.wandb_enabled_check = QtWidgets.QCheckBox("Włącz logowanie do W&B")
            wandb_form.addRow(self.wandb_enabled_check)
            self.wandb_project_edit = QtWidgets.QLineEdit()
            self.wandb_project_edit.setPlaceholderText("Nazwa projektu W&B")
            self.wandb_project_edit.setEnabled(False)
            wandb_form.addRow("Nazwa projektu:", self.wandb_project_edit)
            self.wandb_entity_edit = QtWidgets.QLineEdit()
            self.wandb_entity_edit.setPlaceholderText(
                "Nazwa użytkownika/zespołu W&B (opcjonalnie)"
            )
            self.wandb_entity_edit.setEnabled(False)
            wandb_form.addRow("Encja (użytkownik/zespół):", self.wandb_entity_edit)
            self.wandb_tags_edit = QtWidgets.QLineEdit()
            self.wandb_tags_edit.setPlaceholderText("tag1,tag2,tag3")
            self.wandb_tags_edit.setEnabled(False)
            wandb_form.addRow("Tagi (oddzielone przecinkami):", self.wandb_tags_edit)
            wandb_group.setLayout(wandb_form)
            layout.addWidget(wandb_group, 2, 0)

            # Checkpoints
            checkpoint_group = QtWidgets.QGroupBox("Zapisywanie Modeli (Checkpointy)")
            checkpoint_form = QtWidgets.QFormLayout()
            self.checkpoint_enabled_check = QtWidgets.QCheckBox(
                "Włącz zapisywanie checkpointów"
            )  # Zmiana nazwy
            self.checkpoint_enabled_check.setChecked(True)
            checkpoint_form.addRow(self.checkpoint_enabled_check)
            self.checkpoint_dir_edit = QtWidgets.QLineEdit(
                "checkpoints"
            )  # Zmiana nazwy
            self.checkpoint_dir_edit.setEnabled(
                self.checkpoint_enabled_check.isChecked()
            )
            checkpoint_form.addRow("Katalog na checkpointy:", self.checkpoint_dir_edit)
            self.checkpoint_save_best_only_check = QtWidgets.QCheckBox(
                "Zapisuj tylko najlepszy model"
            )  # Zmiana nazwy
            self.checkpoint_save_best_only_check.setChecked(True)
            self.checkpoint_save_best_only_check.setEnabled(
                self.checkpoint_enabled_check.isChecked()
            )
            checkpoint_form.addRow(self.checkpoint_save_best_only_check)
            self.checkpoint_monitor_combo = QtWidgets.QComboBox()  # Zmiana nazwy
            self.checkpoint_monitor_combo.addItems(
                ["val_loss", "val_accuracy", "train_loss", "train_accuracy"]
            )  # Dodaj więcej
            self.checkpoint_monitor_combo.setEnabled(
                self.checkpoint_enabled_check.isChecked()
            )
            checkpoint_form.addRow(
                "Metryka do monitorowania (dla najlepszego):",
                self.checkpoint_monitor_combo,
            )
            self.checkpoint_mode_combo = QtWidgets.QComboBox()
            self.checkpoint_mode_combo.addItems(["min", "max"])
            self.checkpoint_mode_combo.setEnabled(
                self.checkpoint_enabled_check.isChecked()
            )
            checkpoint_form.addRow(
                "Tryb monitorowania (min/max):", self.checkpoint_mode_combo
            )
            self.checkpoint_save_freq_spin = QtWidgets.QSpinBox()  # Zmiana nazwy
            self.checkpoint_save_freq_spin.setRange(1, 100)
            self.checkpoint_save_freq_spin.setValue(1)  # Co epokę
            self.checkpoint_save_freq_spin.setEnabled(
                self.checkpoint_enabled_check.isChecked()
            )
            checkpoint_form.addRow(
                "Częstotliwość zapisu (co epok, jeśli nie tylko najlepszy):",
                self.checkpoint_save_freq_spin,
            )
            checkpoint_group.setLayout(checkpoint_form)
            layout.addWidget(checkpoint_group, 0, 1)

            # Early Stopping
            early_stop_group = QtWidgets.QGroupBox(
                "Wczesne Zatrzymywanie (Early Stopping)"
            )
            early_stop_form = QtWidgets.QFormLayout()
            self.use_early_stopping_check = QtWidgets.QCheckBox(
                "Użyj wczesnego zatrzymywania"
            )  # Zostawiam oryginalną nazwę
            self.use_early_stopping_check.setChecked(True)
            early_stop_form.addRow(self.use_early_stopping_check)
            self.early_stopping_monitor_combo = (
                QtWidgets.QComboBox()
            )  # Zmiana nazwy atrybutu
            self.early_stopping_monitor_combo.addItems(
                ["val_loss", "val_accuracy", "train_loss"]
            )  # Dodaj więcej
            self.early_stopping_monitor_combo.setEnabled(
                self.use_early_stopping_check.isChecked()
            )
            early_stop_form.addRow(
                "Metryka do monitorowania:", self.early_stopping_monitor_combo
            )
            self.early_stopping_mode_combo = QtWidgets.QComboBox()
            self.early_stopping_mode_combo.addItems(["min", "max"])
            self.early_stopping_mode_combo.setEnabled(
                self.use_early_stopping_check.isChecked()
            )
            early_stop_form.addRow(
                "Tryb monitorowania (min/max):", self.early_stopping_mode_combo
            )
            self.early_stopping_patience_spin = (
                QtWidgets.QSpinBox()
            )  # Zmiana nazwy atrybutu
            self.early_stopping_patience_spin.setRange(1, 100)
            self.early_stopping_patience_spin.setValue(10)
            self.early_stopping_patience_spin.setEnabled(
                self.use_early_stopping_check.isChecked()
            )
            early_stop_form.addRow(
                "Cierpliwość (liczba epok bez poprawy):",
                self.early_stopping_patience_spin,
            )
            self.early_stopping_min_delta_spin = (
                QtWidgets.QDoubleSpinBox()
            )  # Zmiana nazwy atrybutu
            self.early_stopping_min_delta_spin.setRange(0.0, 0.1)
            self.early_stopping_min_delta_spin.setValue(0.001)
            self.early_stopping_min_delta_spin.setDecimals(5)
            self.early_stopping_min_delta_spin.setEnabled(
                self.use_early_stopping_check.isChecked()
            )
            early_stop_form.addRow(
                "Minimalna zmiana uznawana za poprawę:",
                self.early_stopping_min_delta_spin,
            )
            early_stop_group.setLayout(early_stop_form)
            layout.addWidget(early_stop_group, 1, 1)

            # Reduce LR on Plateau
            reduce_lr_group = QtWidgets.QGroupBox("Redukcja LR przy Plateau")
            reduce_lr_form = QtWidgets.QFormLayout()
            self.reduce_lr_enabled_check = QtWidgets.QCheckBox(
                "Włącz redukcję LR przy plateau"
            )
            reduce_lr_form.addRow(self.reduce_lr_enabled_check)
            self.reduce_lr_monitor_combo = QtWidgets.QComboBox()
            self.reduce_lr_monitor_combo.addItems(
                ["val_loss", "val_accuracy", "train_loss"]
            )
            reduce_lr_form.addRow("Metryka monitorowana:", self.reduce_lr_monitor_combo)
            self.reduce_lr_mode_combo = QtWidgets.QComboBox()
            self.reduce_lr_mode_combo.addItems(["min", "max"])
            reduce_lr_form.addRow("Tryb (min/max):", self.reduce_lr_mode_combo)
            self.reduce_lr_factor_spin = QtWidgets.QDoubleSpinBox()
            self.reduce_lr_factor_spin.setRange(0.01, 0.99)
            self.reduce_lr_factor_spin.setValue(0.1)
            self.reduce_lr_factor_spin.setDecimals(2)
            reduce_lr_form.addRow(
                "Współczynnik redukcji LR:", self.reduce_lr_factor_spin
            )
            self.reduce_lr_patience_spin = QtWidgets.QSpinBox()
            self.reduce_lr_patience_spin.setRange(1, 50)
            self.reduce_lr_patience_spin.setValue(5)
            reduce_lr_form.addRow("Cierpliwość (epoki):", self.reduce_lr_patience_spin)
            self.reduce_lr_min_delta_spin = QtWidgets.QDoubleSpinBox()
            self.reduce_lr_min_delta_spin.setRange(0.0, 0.1)
            self.reduce_lr_min_delta_spin.setValue(0.0001)
            self.reduce_lr_min_delta_spin.setDecimals(5)
            reduce_lr_form.addRow("Min. zmiana (delta):", self.reduce_lr_min_delta_spin)
            self.reduce_lr_min_lr_spin = QtWidgets.QDoubleSpinBox()
            self.reduce_lr_min_lr_spin.setRange(0.0, 0.01)
            self.reduce_lr_min_lr_spin.setValue(0.0)
            self.reduce_lr_min_lr_spin.setDecimals(8)
            reduce_lr_form.addRow("Minimalny LR:", self.reduce_lr_min_lr_spin)
            reduce_lr_group.setLayout(reduce_lr_form)
            layout.addWidget(reduce_lr_group, 2, 1)

            # Aliases for original signal connections
            self.accuracy_check = self.metrics_accuracy_check
            self.precision_check = self.metrics_precision_check
            self.recall_check = self.metrics_recall_check
            self.f1_check = self.metrics_f1_check
            self.topk_check = self.metrics_topk_check
            self.confusion_matrix_check = self.metrics_confusion_matrix_check

            self.best_only_check = self.checkpoint_save_best_only_check
            self.save_freq_spin = self.checkpoint_save_freq_spin
            self.checkpoint_metric_combo = self.checkpoint_monitor_combo  # alias
            self.monitor_combo = self.early_stopping_monitor_combo  # alias
            self.patience_spin = self.early_stopping_patience_spin  # alias
            self.min_delta_spin = self.early_stopping_min_delta_spin  # alias

            scroll_content_widget.setLayout(layout)
            scroll_area.setWidget(scroll_content_widget)
            main_layout.addWidget(scroll_area)
            return tab

        except Exception as e:
            msg = "Błąd podczas tworzenia zakładki Monitorowanie i Logowanie"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            raise

    def _create_advanced_tab(self):
        try:
            self.logger.debug("Tworzenie zakładki Zaawansowane")
            tab = QtWidgets.QWidget()
            main_layout = QtWidgets.QVBoxLayout(tab)

            scroll_area = QtWidgets.QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_content_widget = QtWidgets.QWidget()
            layout = QtWidgets.QFormLayout(
                scroll_content_widget
            )  # Użyjemy FormLayout dla tej zakładki

            # General Advanced
            self.advanced_seed_spin = QtWidgets.QSpinBox()
            self.advanced_seed_spin.setRange(0, 999999)
            self.advanced_seed_spin.setValue(42)  # 0 dla losowego
            layout.addRow("Ziarno losowości (0 dla losowego):", self.advanced_seed_spin)

            self.advanced_deterministic_check = QtWidgets.QCheckBox(
                "Użyj deterministycznych operacji (może spowolnić)"
            )
            layout.addRow(self.advanced_deterministic_check)

            self.advanced_benchmark_check = QtWidgets.QCheckBox(
                "Włącz benchmark CUDA (cudnn.benchmark = True)"
            )
            self.advanced_benchmark_check.setChecked(True)
            layout.addRow(self.advanced_benchmark_check)

            # Pamiętaj, że num_workers, pin_memory, prefetch_factor, persistent_workers są teraz w Optimization (Hardware) lub Training Params
            # Jeśli MD sugeruje je *również* tutaj pod `config.advanced`, to jest to duplikacja lub inna konfiguracja.
            # Na razie pomijam je tutaj, aby uniknąć redundancji, zakładając, że te w innych miejscach są właściwe.

            # Gradient Clipping (Advanced - może inne typy niż w Regularyzacji)
            adv_grad_clip_group = QtWidgets.QGroupBox(
                "Zaawansowane Przycinanie Gradientów"
            )
            adv_grad_clip_form = QtWidgets.QFormLayout()

            self.advanced_gradient_clip_val_spin = (
                QtWidgets.QDoubleSpinBox()
            )  # Dla clip_by_value
            self.advanced_gradient_clip_val_spin.setRange(0.0, 100.0)
            self.advanced_gradient_clip_val_spin.setValue(0.0)  # 0.0 = off
            self.advanced_gradient_clip_val_spin.setToolTip(
                "Wartość do przycinania gradientów (clip_by_value, 0.0 = wyłączone)."
            )
            adv_grad_clip_form.addRow(
                "Przycinanie gradientu (wartość):", self.advanced_gradient_clip_val_spin
            )

            self.advanced_gradient_clip_norm_spin = (
                QtWidgets.QDoubleSpinBox()
            )  # Dla clip_by_norm (jeśli inny niż w reg.)
            self.advanced_gradient_clip_norm_spin.setRange(0.0, 100.0)
            self.advanced_gradient_clip_norm_spin.setValue(0.0)  # 0.0 = off
            self.advanced_gradient_clip_norm_spin.setToolTip(
                "Maksymalna norma L2 gradientów (clip_by_norm, 0.0 = wyłączone)."
            )
            adv_grad_clip_form.addRow(
                "Przycinanie gradientu (norma L2):",
                self.advanced_gradient_clip_norm_spin,
            )

            self.advanced_gradient_clip_algorithm_combo = QtWidgets.QComboBox()
            self.advanced_gradient_clip_algorithm_combo.addItems(
                ["norm", "value"]
            )  # Zgodnie z MD
            adv_grad_clip_form.addRow(
                "Algorytm przycinania gradientów:",
                self.advanced_gradient_clip_algorithm_combo,
            )

            adv_grad_clip_group.setLayout(adv_grad_clip_form)
            layout.addRow(adv_grad_clip_group)

            # Accumulate grad batches (jeśli różni się od training.gradient_accumulation_steps)
            # MD sugeruje advanced.accumulate_grad_batches, więc dodajemy jako osobny
            self.advanced_accumulate_grad_batches_spin = QtWidgets.QSpinBox()
            self.advanced_accumulate_grad_batches_spin.setRange(1, 128)
            self.advanced_accumulate_grad_batches_spin.setValue(1)
            layout.addRow(
                "Akumulacja gradientów (zaawansowane, jeśli > 1):",
                self.advanced_accumulate_grad_batches_spin,
            )

            self.advanced_sync_batchnorm_check = QtWidgets.QCheckBox(
                "Synchronizuj BatchNorm (dla DDP/wielu GPU)"
            )
            layout.addRow(self.advanced_sync_batchnorm_check)

            # Precision (Advanced, może bardziej szczegółowe niż training.mixed_precision)
            self.advanced_precision_combo = QtWidgets.QComboBox()
            self.advanced_precision_combo.addItems(
                ["32-true", "16-mixed", "bf16-mixed", "64-true"]
            )  # Pytorch Lightning style
            layout.addRow(
                "Precyzja obliczeń (Pytorch Lightning):", self.advanced_precision_combo
            )

            self.advanced_amp_level_combo = QtWidgets.QComboBox()  # Dla NVIDIA Apex AMP
            self.advanced_amp_level_combo.addItems(["O0", "O1", "O2", "O3"])
            layout.addRow(
                "Poziom AMP (NVIDIA Apex, jeśli używane):",
                self.advanced_amp_level_combo,
            )

            # Adaptive Gradient Clipping (AGC)
            agc_group = QtWidgets.QGroupBox("Adaptive Gradient Clipping (AGC)")
            agc_form = QtWidgets.QFormLayout()
            self.advanced_gradient_clip_agc_check = QtWidgets.QCheckBox("Użyj AGC")
            agc_form.addRow(self.advanced_gradient_clip_agc_check)
            self.advanced_gradient_clip_agc_clipping_spin = QtWidgets.QDoubleSpinBox()
            self.advanced_gradient_clip_agc_clipping_spin.setRange(0.001, 1.0)
            self.advanced_gradient_clip_agc_clipping_spin.setValue(0.01)
            self.advanced_gradient_clip_agc_clipping_spin.setDecimals(3)
            agc_form.addRow(
                "Współczynnik przycinania dla AGC:",
                self.advanced_gradient_clip_agc_clipping_spin,
            )
            self.advanced_gradient_clip_agc_eps_spin = QtWidgets.QDoubleSpinBox()
            self.advanced_gradient_clip_agc_eps_spin.setRange(1e-6, 1e-1)
            self.advanced_gradient_clip_agc_eps_spin.setValue(1e-3)
            self.advanced_gradient_clip_agc_eps_spin.setDecimals(6)
            agc_form.addRow(
                "Epsilon dla AGC:", self.advanced_gradient_clip_agc_eps_spin
            )
            # Pozostałe parametry AGC są bardzo szczegółowe, można dodać w razie potrzeby
            agc_group.setLayout(agc_form)
            layout.addRow(agc_group)

            scroll_content_widget.setLayout(layout)  # Set layout to the content widget
            scroll_area.setWidget(
                scroll_content_widget
            )  # Set content widget to scroll area
            main_layout.addWidget(
                scroll_area
            )  # Add scroll area to the main layout of the tab

            return tab

        except Exception as e:
            msg = "Błąd podczas tworzenia zakładki Zaawansowane"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            raise

    def _create_optimization_tab(self):
        try:
            self.logger.debug("Tworzenie zakładki Optymalizacja Sprzętowa")
            tab = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(tab)

            params_group = QtWidgets.QGroupBox(
                "Parametry Optymalizacji Sprzętowej (wpływają na wydajność i zużycie pamięci)"
            )
            params_layout = QtWidgets.QFormLayout()

            # Lista parametrów dla tej zakładki
            # klucz, domyślna wartość, typ widgetu, [opcjonalnie: min, max, step]
            params = [
                ("CUDNN Benchmark", "cudnn_benchmark", True, "bool"),
                ("Pin Memory (Dataloader)", "pin_memory", True, "bool"),
                ("Shuffle (Dataloader)", "shuffle", True, "bool"),
                ("Prefetch Factor (Dataloader)", "prefetch_factor", 2, "int", 0, 16, 1),
                (
                    "Persistent Workers (Dataloader)",
                    "persistent_workers",
                    False,
                    "bool",
                ),
                ("Drop Last Batch (Dataloader)", "drop_last", False, "bool"),
            ]

            if not hasattr(self, "parameter_rows"):
                self.parameter_rows = {}

            for name, key, default, type_, *args in params:
                row_layout = self._create_parameter_row(
                    name, key, default, type_, *args
                )
                params_layout.addRow(name + ":", row_layout)

            params_group.setLayout(params_layout)
            layout.addWidget(params_group)

            apply_all_btn = QtWidgets.QPushButton(
                "Zastosuj optymalizacje z profilu sprzętowego (dla tej zakładki)"
            )
            apply_all_btn.clicked.connect(self._apply_hardware_optimizations_for_tab)
            layout.addWidget(apply_all_btn)

            layout.addStretch()
            return tab

        except Exception as e:
            self.logger.error(
                f"Błąd podczas tworzenia zakładki optymalizacji: {e!s}", exc_info=True
            )
            raise

    def _create_parameter_row(self, name, param_key, default_value, widget_type, *args):
        min_val, max_val, step = None, None, None
        if args:
            if widget_type == "int" or widget_type == "float":
                if len(args) == 3:
                    min_val, max_val, step = args
                elif len(args) == 2:
                    min_val, max_val = args
                elif len(args) == 1:
                    min_val = args[0]

        layout = QtWidgets.QHBoxLayout()

        if widget_type == "int":
            value_widget = QtWidgets.QSpinBox()
            if min_val is not None:
                value_widget.setMinimum(min_val)
            if max_val is not None:
                value_widget.setMaximum(max_val)
            value_widget.setValue(default_value)
            if step is not None:
                value_widget.setSingleStep(step)
        elif widget_type == "float":
            value_widget = QtWidgets.QDoubleSpinBox()
            if min_val is not None:
                value_widget.setMinimum(min_val)
            if max_val is not None:
                value_widget.setMaximum(max_val)
            value_widget.setValue(default_value)
            if step is not None:
                value_widget.setSingleStep(step)
        elif widget_type == "bool":
            value_widget = QtWidgets.QCheckBox()
            value_widget.setChecked(default_value)
        else:
            value_widget = QtWidgets.QLineEdit(str(default_value))

        user_checkbox = QtWidgets.QCheckBox("Użytkownika")
        user_checkbox.setChecked(True)

        profile_key_map = {
            "cudnn_benchmark": "cudnn_benchmark",
            "pin_memory": "pin_memory",
            "shuffle": "shuffle",  # Assuming shuffle is in hardware profile directly
            "prefetch_factor": "prefetch_factor",
            "persistent_workers": "persistent_workers",
            "drop_last": "drop_last",  # Assuming drop_last is in hardware profile directly
        }
        effective_profile_key = profile_key_map.get(param_key, param_key)

        hw_value_actual = self.hardware_profile.get(effective_profile_key)
        hw_value_text = str(hw_value_actual) if hw_value_actual is not None else "Brak"
        hw_value_label = QtWidgets.QLabel(hw_value_text)

        hw_checkbox = QtWidgets.QCheckBox("Profil sprzętowy")
        hw_checkbox.setChecked(False)

        source_group = QtWidgets.QButtonGroup(self)
        source_group.addButton(user_checkbox)
        source_group.addButton(hw_checkbox)
        source_group.setExclusive(True)

        layout.addWidget(value_widget)
        layout.addWidget(user_checkbox)
        layout.addWidget(QtWidgets.QLabel("Profil:"))
        layout.addWidget(hw_value_label)
        layout.addWidget(hw_checkbox)

        row_widgets = {
            "param_key": param_key,
            "value_widget": value_widget,
            "user_checkbox": user_checkbox,
            "hw_value_label": hw_value_label,
            "hw_checkbox": hw_checkbox,
            "button_group": source_group,
            "hw_value_actual": hw_value_actual,
        }

        user_checkbox.toggled.connect(
            lambda checked, rw=row_widgets: self._on_source_toggle(rw, checked)
        )
        hw_checkbox.toggled.connect(
            lambda checked, rw=row_widgets: self._on_hw_toggle(rw, checked)
        )

        self.parameter_rows[param_key] = row_widgets
        return layout

    def _on_source_toggle(self, row_widgets, is_user_selected):
        value_widget = row_widgets["value_widget"]
        hw_checkbox = row_widgets["hw_checkbox"]
        if is_user_selected:
            value_widget.setEnabled(True)
            if hw_checkbox.isChecked():
                hw_checkbox.setChecked(False)

    def _on_hw_toggle(self, row_widgets, is_hw_selected):
        value_widget = row_widgets["value_widget"]
        user_checkbox = row_widgets["user_checkbox"]
        hw_value_actual = row_widgets["hw_value_actual"]

        if is_hw_selected:
            if user_checkbox.isChecked():
                user_checkbox.setChecked(False)
            value_widget.setEnabled(False)
            if hw_value_actual is not None:
                if isinstance(
                    value_widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)
                ):
                    value_widget.setValue(hw_value_actual)
                elif isinstance(value_widget, QtWidgets.QCheckBox):
                    value_widget.setChecked(bool(hw_value_actual))
                else:
                    value_widget.setText(str(hw_value_actual))
        else:
            if (
                not user_checkbox.isChecked()
            ):  # If HW is deselected and user is not selected, enable widget for manual input
                value_widget.setEnabled(True)

    def _apply_hardware_optimizations_for_tab(self):
        count = 0
        opt_tab_keys = [
            "cudnn_benchmark",
            "pin_memory",
            "shuffle",
            "prefetch_factor",
            "persistent_workers",
            "drop_last",
        ]
        for key in opt_tab_keys:
            if key in self.parameter_rows:
                param_widgets = self.parameter_rows[key]
                if param_widgets["hw_value_actual"] is not None:
                    param_widgets["hw_checkbox"].setChecked(
                        True
                    )  # This will trigger _on_hw_toggle
                    count += 1
        if count > 0:
            QtWidgets.QMessageBox.information(
                self,
                "Sukces",
                f"Zastosowano {count} ustawień z profilu sprzętowego dla tej zakładki.",
            )
        else:
            QtWidgets.QMessageBox.information(
                self,
                "Informacja",
                "Brak dostępnych optymalizacji z profilu sprzętowego dla tej zakładki lub już zastosowane.",
            )

    def _refresh_profile_list(self):
        # ... (bez zmian)
        self.profile_list.clear()
        for profile_file in self.profiles_dir.glob("*.json"):
            try:
                with open(profile_file, "r", encoding="utf-8") as f:
                    profile_data = json.load(f)
                    if (
                        profile_data.get("type") == "training"
                    ):  # Upewnij się, że to profil treningowy
                        self.profile_list.addItem(profile_file.stem)
            except Exception as e:
                self.logger.error(
                    f"Błąd wczytywania profilu {profile_file}: {e}", exc_info=True
                )

    def _on_profile_selected(self, current, previous):
        # ... (bez zmian)
        if current is None:
            self.current_profile = None  # Clear current profile if nothing is selected
            self.profile_info.clear()
            self.profile_description.clear()
            self.profile_data_required.clear()
            self.profile_hardware_required.clear()
            return
        try:
            profile_path = self.profiles_dir / f"{current.text()}.json"
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
            self.logger.error(f"Błąd ładowania profilu: {e}", exc_info=True)
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Nie można załadować profilu: {e}"
            )
            self.current_profile = None  # Reset on error

    def _edit_profile(self):
        # ... (bez zmian)
        if not self.current_profile or not self.profile_list.currentItem():
            QtWidgets.QMessageBox.warning(
                self, "Ostrzeżenie", "Najpierw wybierz profil do edycji."
            )
            return
        try:
            profile_path = (
                self.profiles_dir / f"{self.profile_list.currentItem().text()}.json"
            )
            from PyQt6.QtCore import QUrl
            from PyQt6.QtGui import QDesktopServices

            QDesktopServices.openUrl(QUrl.fromLocalFile(str(profile_path)))
        except Exception as e:
            self.logger.error(f"Błąd otwierania profilu: {e}", exc_info=True)
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Nie można otworzyć profilu: {e}"
            )

    def _apply_profile(self):
        if not self.current_profile:
            QtWidgets.QMessageBox.warning(
                self, "Ostrzeżenie", "Najpierw wybierz profil do zastosowania."
            )
            return
        try:
            config = self.current_profile.get("config", {})

            # Dane i Model (zakładka: Dane i Model)
            self.train_dir_edit.setText(config.get("train_dir", ""))
            self.val_dir_edit.setText(config.get("val_dir", ""))
            self.arch_combo.setCurrentText(
                config.get("model", {}).get("architecture", "EfficientNet")
            )
            # _on_architecture_changed will update variant_combo, then set current text
            self._on_architecture_changed(self.arch_combo.currentText())
            self.variant_combo.setCurrentText(
                config.get("model", {}).get("variant", "EfficientNet-B0")
            )
            self.input_size_spin.setValue(
                config.get("model", {}).get("input_size", 224)
            )
            self.num_classes_spin.setValue(
                config.get("model", {}).get("num_classes", 2)
            )

            # Parametry Architektury Modelu (zakładka: Parametry Treningu)
            model_config = config.get("model", {})
            self.model_pretrained_check.setChecked(model_config.get("pretrained", True))
            self.model_pretrained_weights_combo.setCurrentText(
                model_config.get("pretrained_weights", "imagenet")
            )
            self.model_feature_extraction_check.setChecked(
                model_config.get("feature_extraction_only", False)
            )
            self.model_activation_combo.setCurrentText(
                model_config.get("activation", "swish")
            )
            self.model_dropout_at_inference_check.setChecked(
                model_config.get("dropout_at_inference", False)
            )
            self.model_global_pool_combo.setCurrentText(
                model_config.get("global_pool", "avg")
            )
            self.model_last_layer_activation_combo.setCurrentText(
                model_config.get("last_layer_activation", "softmax")
            )

            # Hiperparametry Treningu (zakładka: Parametry Treningu)
            training_config = config.get("training", {})
            self.training_epochs_spin.setValue(training_config.get("epochs", 100))
            self.training_batch_size_spin.setValue(
                training_config.get("batch_size", 32)
            )
            self.training_learning_rate_spin.setValue(
                training_config.get("learning_rate", 0.001)
            )
            self.training_optimizer_combo.setCurrentText(
                training_config.get("optimizer", "AdamW")
            )

            scheduler_cfg = training_config.get("scheduler", {})
            if isinstance(scheduler_cfg, dict):
                self.training_scheduler_type_combo.setCurrentText(
                    scheduler_cfg.get("type", "None")
                )
                self.training_scheduler_t0_spin.setValue(scheduler_cfg.get("T_0", 10))
                self.training_scheduler_tmult_spin.setValue(
                    scheduler_cfg.get("T_mult", 2)
                )
                self.training_scheduler_eta_min_spin.setValue(
                    scheduler_cfg.get("eta_min", 0.000001)
                )
            elif isinstance(scheduler_cfg, str):  # Starszy format
                self.training_scheduler_type_combo.setCurrentText(scheduler_cfg)
            else:  # Default if scheduler_cfg is not dict or str
                self.training_scheduler_type_combo.setCurrentText("None")

            self.training_num_workers_spin.setValue(
                training_config.get("num_workers", min(4, os.cpu_count() or 4))
            )
            self.training_warmup_epochs_spin.setValue(
                training_config.get("warmup_epochs", 0)
            )
            self.training_warmup_lr_init_spin.setValue(
                training_config.get("warmup_lr_init", 1e-6)
            )
            self.training_mixed_precision_check.setChecked(
                training_config.get(
                    "mixed_precision", training_config.get("use_mixed_precision", False)
                )
            )
            self.training_grad_accum_steps_spin.setValue(
                training_config.get("gradient_accumulation_steps", 1)
            )
            self.training_gradient_clip_value_spin.setValue(
                training_config.get("gradient_clip", 0.0)
            )
            self.training_evaluation_freq_spin.setValue(
                training_config.get("evaluation_freq", 1)
            )
            self.training_use_ema_check.setChecked(
                training_config.get("use_ema", False)
            )
            self.training_ema_decay_spin.setValue(
                training_config.get("ema_decay", 0.999)
            )

            self.training_freeze_base_model_check.setChecked(
                training_config.get("freeze_base_model", True)
            )
            unfreeze_layers_val = training_config.get("unfreeze_layers", "all")
            if isinstance(unfreeze_layers_val, list):
                self.training_unfreeze_layers_edit.setText(
                    ",".join(unfreeze_layers_val)
                )
            else:
                self.training_unfreeze_layers_edit.setText(str(unfreeze_layers_val))

            unfreeze_strategy_map_load = {
                self.UNFREEZE_ALL: "Wszystkie na raz (unfreeze_all)",
                self.UNFREEZE_GRADUAL_END: "Stopniowo od końca (unfreeze_gradual_end)",
                self.UNFREEZE_GRADUAL_START: "Stopniowo od początku (unfreeze_gradual_start)",
                self.UNFREEZE_AFTER_EPOCHS: f"Po określonej liczbie epok ({self.UNFREEZE_AFTER_EPOCHS})",
            }
            # Find the display text corresponding to the loaded strategy value
            loaded_strategy_value = training_config.get(
                "unfreeze_strategy", self.UNFREEZE_ALL
            )
            display_text_to_set = "Wszystkie na raz (unfreeze_all)"  # Default
            for val, display_text in unfreeze_strategy_map_load.items():
                if val == loaded_strategy_value:
                    display_text_to_set = display_text
                    break
            self.training_unfreeze_strategy_combo.setCurrentText(display_text_to_set)

            self.training_unfreeze_after_epochs_spin.setValue(
                training_config.get("unfreeze_after_epochs", 10)
            )
            self.training_frozen_lr_spin.setValue(
                training_config.get("frozen_lr", 1e-5)
            )
            self.training_unfrozen_lr_spin.setValue(
                training_config.get("unfrozen_lr", 1e-4)
            )
            self.training_validation_split_spin.setValue(
                training_config.get("validation_split", 0.0)
            )

            # Regularyzacja
            reg_config = config.get("regularization", {})
            self.reg_weight_decay_spin.setValue(reg_config.get("weight_decay", 0.0001))
            self.reg_label_smoothing_spin.setValue(
                reg_config.get("label_smoothing", 0.1)
            )
            self.reg_dropout_rate_spin.setValue(reg_config.get("dropout_rate", 0.2))
            self.reg_drop_connect_rate_spin.setValue(
                reg_config.get(
                    "drop_connect_rate",
                    0.2 if self.arch_combo.currentText() == "EfficientNet" else 0.0,
                )
            )
            self.reg_gradient_clip_spin.setValue(reg_config.get("gradient_clip", 1.0))
            self.reg_momentum_spin.setValue(reg_config.get("momentum", 0.9))
            self.reg_epsilon_spin.setValue(reg_config.get("epsilon", 1e-8))

            sd_conf = reg_config.get("stochastic_depth", {})
            self.stochastic_depth_use_check.setChecked(sd_conf.get("use", False))
            self.stochastic_depth_survival_prob_spin.setValue(
                sd_conf.get("survival_probability", 0.8)
            )

            swa_conf = reg_config.get("swa", {})
            self.use_swa_check.setChecked(swa_conf.get("use", False))
            self.swa_start_epoch_spin.setValue(swa_conf.get("start_epoch", 10))
            self.swa_lr_spin.setValue(swa_conf.get("lr_swa", 5e-5))

            # Augmentacja
            aug_config = config.get("augmentation", {})
            basic_aug_conf = aug_config.get("basic", {})
            self.basic_aug_check.setChecked(basic_aug_conf.get("use", False))
            # Using corrected names (Poprawka 8)
            self.basic_aug_rotation_spin.setValue(basic_aug_conf.get("rotation", 15))
            self.basic_aug_brightness_spin.setValue(
                basic_aug_conf.get("brightness", 0.2)
            )
            self.basic_aug_contrast_spin.setValue(basic_aug_conf.get("contrast", 0.2))
            self.basic_aug_saturation_spin.setValue(
                basic_aug_conf.get("saturation", 0.2)
            )
            self.basic_aug_hue_spin.setValue(basic_aug_conf.get("hue", 0.1))
            self.basic_aug_shift_spin.setValue(basic_aug_conf.get("shift", 0.1))
            self.basic_aug_zoom_spin.setValue(basic_aug_conf.get("zoom", 0.1))
            self.basic_aug_horizontal_flip_check.setChecked(
                basic_aug_conf.get("horizontal_flip", True)
            )
            self.basic_aug_vertical_flip_check.setChecked(
                basic_aug_conf.get("vertical_flip", False)
            )

            self.mixup_check.setChecked(aug_config.get("mixup", {}).get("use", False))
            self.mixup_alpha_spin.setValue(
                aug_config.get("mixup", {}).get("alpha", 0.4)
            )
            self.cutmix_check.setChecked(aug_config.get("cutmix", {}).get("use", False))
            self.cutmix_alpha_spin.setValue(
                aug_config.get("cutmix", {}).get("alpha", 1.0)
            )
            self.autoaugment_check.setChecked(
                aug_config.get("autoaugment", {}).get("use", False)
            )
            self.autoaugment_policy_combo.setCurrentText(
                aug_config.get("autoaugment", {}).get("policy", "imagenet")
            )
            self.randaugment_check.setChecked(
                aug_config.get("randaugment", {}).get("use", False)
            )
            self.randaugment_n_spin.setValue(
                aug_config.get("randaugment", {}).get("n", 2)
            )
            self.randaugment_m_spin.setValue(
                aug_config.get("randaugment", {}).get("m", 9)
            )
            self.trivialaugment_check.setChecked(
                aug_config.get("trivialaugment", {}).get("use", False)
            )

            re_conf = aug_config.get("random_erase", {})
            self.random_erase_check.setChecked(re_conf.get("use", False))
            self.random_erase_prob_spin.setValue(re_conf.get("probability", 0.5))
            scale = re_conf.get("scale", [0.02, 0.33])
            self.random_erase_scale_min_spin.setValue(
                scale[0] if isinstance(scale, list) and len(scale) > 0 else 0.02
            )
            self.random_erase_scale_max_spin.setValue(
                scale[1] if isinstance(scale, list) and len(scale) > 1 else 0.33
            )
            ratio = re_conf.get("ratio", [0.3, 3.3])
            self.random_erase_ratio_min_spin.setValue(
                ratio[0] if isinstance(ratio, list) and len(ratio) > 0 else 0.3
            )
            self.random_erase_ratio_max_spin.setValue(
                ratio[1] if isinstance(ratio, list) and len(ratio) > 1 else 3.3
            )

            gd_conf = aug_config.get("grid_distortion", {})
            self.grid_distortion_check.setChecked(
                gd_conf.get("enabled", False)
            )  # Ensure key matches save
            self.grid_distortion_prob_spin.setValue(gd_conf.get("probability", 0.5))
            self.grid_distortion_limit_spin.setValue(
                gd_conf.get("distort_limit", 0.3)
            )  # Ensure key matches save

            self.aug_resize_enabled_check.setChecked(
                aug_config.get("resize", {}).get("enabled", False)
            )

            # Preprocessing
            pre_config = config.get("preprocessing", {})
            resize_conf = pre_config.get("resize", {})
            self.preprocess_resize_enabled_check.setChecked(
                resize_conf.get("enabled", True)
            )
            size = resize_conf.get("size", [256, 256])
            self.preprocess_resize_width_spin.setValue(
                size[0] if isinstance(size, list) and len(size) > 0 else 256
            )
            self.preprocess_resize_height_spin.setValue(
                size[1] if isinstance(size, list) and len(size) > 1 else 256
            )
            self.preprocess_resize_mode_combo.setCurrentText(
                resize_conf.get("mode", "bilinear")
            )

            norm_conf = pre_config.get("normalize", {})
            self.preprocess_normalize_enabled_check.setChecked(
                norm_conf.get("enabled", True)
            )
            mean = norm_conf.get("mean", [0.485, 0.456, 0.406])
            self.preprocess_normalize_mean_r_spin.setValue(
                mean[0] if isinstance(mean, list) and len(mean) > 0 else 0.485
            )
            self.preprocess_normalize_mean_g_spin.setValue(
                mean[1] if isinstance(mean, list) and len(mean) > 1 else 0.456
            )
            self.preprocess_normalize_mean_b_spin.setValue(
                mean[2] if isinstance(mean, list) and len(mean) > 2 else 0.406
            )
            std = norm_conf.get("std", [0.229, 0.224, 0.225])
            self.preprocess_normalize_std_r_spin.setValue(
                std[0] if isinstance(std, list) and len(std) > 0 else 0.229
            )
            self.preprocess_normalize_std_g_spin.setValue(
                std[1] if isinstance(std, list) and len(std) > 1 else 0.224
            )
            self.preprocess_normalize_std_b_spin.setValue(
                std[2] if isinstance(std, list) and len(std) > 2 else 0.225
            )

            gray_conf = pre_config.get("grayscale", {})
            self.preprocess_grayscale_enabled_check.setChecked(
                gray_conf.get("enabled", False)
            )
            self.preprocess_grayscale_num_output_channels_spin.setValue(
                gray_conf.get("num_output_channels", 1)
            )

            cj_conf = pre_config.get("color_jitter", {})
            self.preprocess_color_jitter_enabled_check.setChecked(
                cj_conf.get("enabled", False)
            )
            self.preprocess_color_jitter_brightness_spin.setValue(
                cj_conf.get("brightness", 0.2)
            )
            self.preprocess_color_jitter_contrast_spin.setValue(
                cj_conf.get("contrast", 0.2)
            )
            self.preprocess_color_jitter_saturation_spin.setValue(
                cj_conf.get("saturation", 0.2)
            )
            self.preprocess_color_jitter_hue_spin.setValue(cj_conf.get("hue", 0.1))

            blur_conf = pre_config.get("gaussian_blur", {})
            self.preprocess_gaussian_blur_enabled_check.setChecked(
                blur_conf.get("enabled", False)
            )
            self.preprocess_gaussian_blur_kernel_size_spin.setValue(
                blur_conf.get("kernel_size", 3)
            )
            self.preprocess_gaussian_blur_sigma_spin.setValue(
                blur_conf.get("sigma", 1.0)
            )

            self.preprocess_cache_dataset_check.setChecked(bool(config.get("cache_dataset", False)))

            scaling_conf = pre_config.get("scaling", {})
            self.preprocess_scaling_method_combo.setCurrentText(
                scaling_conf.get("method", "Bilinear")
            )
            self.preprocess_scaling_maintain_aspect_ratio_check.setChecked(
                scaling_conf.get("maintain_aspect_ratio", False)
            )
            self.preprocess_scaling_pad_to_square_check.setChecked(
                scaling_conf.get("pad_to_square", False)
            )
            self.preprocess_scaling_pad_mode_combo.setCurrentText(
                scaling_conf.get("pad_mode", "constant")
            )
            self.preprocess_scaling_pad_value_spin.setValue(
                scaling_conf.get("pad_value", 0)
            )

            rrc_conf = pre_config.get("random_resize_crop", {})
            self.preprocess_random_resize_crop_enabled_check.setChecked(
                rrc_conf.get("enabled", False)
            )
            self.preprocess_random_resize_crop_size_spin.setValue(
                rrc_conf.get("size", 224)
            )
            rrc_scale = rrc_conf.get("scale", [0.08, 1.0])
            self.preprocess_random_resize_crop_scale_min_spin.setValue(
                rrc_scale[0]
                if isinstance(rrc_scale, list) and len(rrc_scale) > 0
                else 0.08
            )
            self.preprocess_random_resize_crop_scale_max_spin.setValue(
                rrc_scale[1]
                if isinstance(rrc_scale, list) and len(rrc_scale) > 1
                else 1.0
            )
            rrc_ratio = rrc_conf.get("ratio", [0.75, 1.33])
            self.preprocess_random_resize_crop_ratio_min_spin.setValue(
                rrc_ratio[0]
                if isinstance(rrc_ratio, list) and len(rrc_ratio) > 0
                else 0.75
            )
            self.preprocess_random_resize_crop_ratio_max_spin.setValue(
                rrc_ratio[1]
                if isinstance(rrc_ratio, list) and len(rrc_ratio) > 1
                else 1.33
            )

            # Monitorowanie i Logowanie
            mon_config = config.get("monitoring", {})
            met_conf = mon_config.get("metrics", {})
            self.metrics_accuracy_check.setChecked(met_conf.get("accuracy", True))
            self.metrics_precision_check.setChecked(met_conf.get("precision", True))
            self.metrics_recall_check.setChecked(met_conf.get("recall", True))
            self.metrics_f1_check.setChecked(met_conf.get("f1", True))
            self.metrics_topk_check.setChecked(
                bool(met_conf.get("topk", []))
            )  # Check if list is non-empty
            self.metrics_confusion_matrix_check.setChecked(
                met_conf.get("confusion_matrix", False)
            )
            self.metrics_auc_check.setChecked(met_conf.get("auc", False))
            self.metrics_gpu_utilization_check.setChecked(
                met_conf.get("gpu_utilization", False)
            )
            self.metrics_memory_usage_check.setChecked(
                met_conf.get("memory_usage", False)
            )

            tb_conf = mon_config.get("tensorboard", {})
            self.tensorboard_enabled_check.setChecked(tb_conf.get("enabled", True))
            self.tensorboard_log_dir_edit.setText(
                tb_conf.get("log_dir", "logs/tensorboard")
            )
            self.tensorboard_update_freq_spin.setValue(tb_conf.get("update_freq", 100))

            wb_conf = mon_config.get("wandb", {})
            self.wandb_enabled_check.setChecked(wb_conf.get("enabled", False))
            self.wandb_project_edit.setText(wb_conf.get("project", ""))
            self.wandb_entity_edit.setText(wb_conf.get("entity", ""))
            self.wandb_tags_edit.setText(",".join(wb_conf.get("tags", [])))

            cp_conf = mon_config.get("checkpoint", {})
            self.checkpoint_enabled_check.setChecked(cp_conf.get("enabled", True))
            self.checkpoint_dir_edit.setText(cp_conf.get("dir", "checkpoints"))
            self.checkpoint_save_best_only_check.setChecked(
                cp_conf.get("save_best_only", True)
            )
            self.checkpoint_monitor_combo.setCurrentText(
                cp_conf.get("monitor", "val_loss")
            )
            self.checkpoint_mode_combo.setCurrentText(cp_conf.get("mode", "min"))
            self.checkpoint_save_freq_spin.setValue(cp_conf.get("save_freq", 1))

            es_conf = mon_config.get("early_stopping", {})
            self.use_early_stopping_check.setChecked(es_conf.get("enabled", True))
            self.early_stopping_monitor_combo.setCurrentText(
                es_conf.get("monitor", "val_loss")
            )
            self.early_stopping_mode_combo.setCurrentText(es_conf.get("mode", "min"))
            self.early_stopping_patience_spin.setValue(es_conf.get("patience", 10))
            self.early_stopping_min_delta_spin.setValue(es_conf.get("min_delta", 0.001))

            rlr_conf = mon_config.get("reduce_lr", {})
            self.reduce_lr_enabled_check.setChecked(rlr_conf.get("enabled", False))
            self.reduce_lr_monitor_combo.setCurrentText(
                rlr_conf.get("monitor", "val_loss")
            )
            self.reduce_lr_mode_combo.setCurrentText(rlr_conf.get("mode", "min"))
            self.reduce_lr_factor_spin.setValue(rlr_conf.get("factor", 0.1))
            self.reduce_lr_patience_spin.setValue(rlr_conf.get("patience", 5))
            self.reduce_lr_min_delta_spin.setValue(rlr_conf.get("min_delta", 0.0001))
            self.reduce_lr_min_lr_spin.setValue(rlr_conf.get("min_lr", 0.0))

            # Zaawansowane
            adv_config = config.get("advanced", {})
            self.advanced_seed_spin.setValue(adv_config.get("seed", 42))
            self.advanced_deterministic_check.setChecked(
                adv_config.get("deterministic", False)
            )
            self.advanced_benchmark_check.setChecked(adv_config.get("benchmark", True))
            self.advanced_gradient_clip_val_spin.setValue(
                adv_config.get("gradient_clip_val", 0.0)
            )
            self.advanced_gradient_clip_norm_spin.setValue(
                adv_config.get("gradient_clip_norm", 0.0)
            )
            self.advanced_gradient_clip_algorithm_combo.setCurrentText(
                adv_config.get("gradient_clip_algorithm", "norm")
            )
            self.advanced_accumulate_grad_batches_spin.setValue(
                adv_config.get("accumulate_grad_batches", 1)
            )
            self.advanced_sync_batchnorm_check.setChecked(
                adv_config.get("sync_batchnorm", False)
            )
            self.advanced_precision_combo.setCurrentText(
                adv_config.get("precision", "32-true")
            )
            self.advanced_amp_level_combo.setCurrentText(
                adv_config.get("amp_level", "O1")
            )
            agc_conf = adv_config.get("gradient_clip_agc", {})
            self.advanced_gradient_clip_agc_check.setChecked(agc_conf.get("use", False))
            self.advanced_gradient_clip_agc_clipping_spin.setValue(
                agc_conf.get("clipping", 0.01)
            )
            self.advanced_gradient_clip_agc_eps_spin.setValue(agc_conf.get("eps", 1e-3))

            # Optymalizacja Sprzętowa
            opt_config_main = config.get("optimization", {})
            opt_config_dataloader = opt_config_main.get(
                "dataloader", {}
            )  # For nested structure

            for key, row_data in self.parameter_rows.items():
                # Prioritize value from flat structure, then nested, then default if not found
                value_to_set = None
                found = False
                if key in opt_config_main:
                    value_to_set = opt_config_main[key]
                    found = True
                elif key in opt_config_dataloader:  # Check nested if not in flat
                    value_to_set = opt_config_dataloader[key]
                    found = True

                if found:
                    widget = row_data["value_widget"]
                    # If a value is found in profile, switch to "Użytkownika" and set it.
                    # Don't automatically switch to "Profil sprzętowy" unless user clicks the button.
                    row_data["user_checkbox"].setChecked(
                        True
                    )  # Ensures widget is enabled for setting value
                    if isinstance(
                        widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)
                    ):
                        widget.setValue(value_to_set)
                    elif isinstance(widget, QtWidgets.QCheckBox):
                        widget.setChecked(bool(value_to_set))
                    else:  # QLineEdit
                        widget.setText(str(value_to_set))
                # else: param not in profile, keep widget's current/default value

            QtWidgets.QMessageBox.information(
                self, "Sukces", "Profil został pomyślnie zastosowany."
            )
            self._update_dependent_controls()

        except Exception as e:
            self.logger.error(f"Błąd podczas stosowania profilu: {e}", exc_info=True)
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Nie można zastosować profilu: {e}"
            )

    def _clone_profile(self):
        if not self.current_profile or not self.profile_list.currentItem():
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
                # Klonowanie powinno bazować na *zapisanej* konfiguracji klonowanego profilu,
                # a nie na bieżących ustawieniach UI, chyba że intencją jest "Zapisz jako" na bazie UI.
                # Tutaj _get_current_config_as_dict() bierze z UI.
                # Jeśli chcemy klonować *wybrany* profil, to: new_profile_data = self.current_profile.copy()
                # Ale opis w kodzie sugeruje "na bazie bieżących ustawień UI", więc _get_current_config_as_dict() jest OK.

                new_profile_data = self._get_current_config_as_dict()

                new_profile_data["type"] = "training"
                new_profile_data["info"] = (
                    f"Klon profilu '{current_name}' (na bazie bieżących ustawień UI, {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')})"
                )
                # Zachowaj opis, jeśli jest, inaczej domyślny.
                cloned_description = self.current_profile.get(
                    "description", "Brak opisu."
                )
                new_profile_data["description"] = (
                    f"Klon z '{current_name}'. Oryginalny opis: {cloned_description}"
                )
                cloned_data_req = self.current_profile.get(
                    "data_required", "Nie zdefiniowano."
                )
                new_profile_data["data_required"] = (
                    f"Klon z '{current_name}'. Oryginalne wymagania: {cloned_data_req}"
                )
                cloned_hw_req = self.current_profile.get(
                    "hardware_required", "Nie zdefiniowano."
                )
                new_profile_data["hardware_required"] = (
                    f"Klon z '{current_name}'. Oryginalne wymagania: {cloned_hw_req}"
                )

                new_path = self.profiles_dir / f"{new_name}.json"
                with open(new_path, "w", encoding="utf-8") as f:
                    json.dump(new_profile_data, f, indent=4, ensure_ascii=False)
                self._refresh_profile_list()
                QtWidgets.QMessageBox.information(
                    self,
                    "Sukces",
                    f"Profil '{new_name}' został pomyślnie sklonowany na podstawie bieżących ustawień UI.",
                )
        except Exception as e:
            self.logger.error(f"Błąd klonowania profilu: {e}", exc_info=True)
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Nie można sklonować profilu: {e}"
            )

    def _save_profile(self):
        try:
            default_profile_name = f"{self.arch_combo.currentText()}_{self.variant_combo.currentText()}_custom"
            if (
                self.profile_list.currentItem()
            ):  # If a profile is selected, suggest its name for overwriting or modifying
                default_profile_name = self.profile_list.currentItem().text()

            name, ok = QtWidgets.QInputDialog.getText(
                self,
                "Zapisz profil",
                "Podaj nazwę dla profilu:",
                QtWidgets.QLineEdit.EchoMode.Normal,
                default_profile_name,
            )
            if ok and name:
                profile_data = self._get_current_config_as_dict()

                profile_data["type"] = "training"
                profile_data["info"] = (
                    f"Profil dla {self.arch_combo.currentText()} {self.variant_combo.currentText()} "
                    f"(zapisany {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')})"
                )

                current_description = ""
                if (
                    self.current_profile
                    and self.profile_list.currentItem()
                    and self.profile_list.currentItem().text() == name
                ):
                    current_description = self.current_profile.get(
                        "description",
                        "Utworzony przez użytkownika na podstawie bieżących ustawień.",
                    )
                else:
                    current_description = (
                        "Utworzony przez użytkownika na podstawie bieżących ustawień."
                    )

                description, desc_ok = QtWidgets.QInputDialog.getMultiLineText(
                    self,
                    "Opis Profilu",
                    "Wprowadź opis dla profilu:",
                    current_description,
                )
                if desc_ok:
                    profile_data["description"] = description
                else:  # User cancelled description input
                    profile_data["description"] = (
                        current_description  # Keep old or default
                    )

                profile_data["data_required"] = (
                    "Dane zgodnie z konfiguracją (liczba klas, rozmiar wejścia)."
                )
                profile_data["hardware_required"] = (
                    "Zależne od konfiguracji (batch size, model, precyzja)."
                )

                profile_path = self.profiles_dir / f"{name}.json"
                with open(profile_path, "w", encoding="utf-8") as f:
                    json.dump(profile_data, f, indent=4, ensure_ascii=False)
                self._refresh_profile_list()
                # Try to reselect the saved profile
                for i in range(self.profile_list.count()):
                    if self.profile_list.item(i).text() == name:
                        self.profile_list.setCurrentRow(i)
                        break
                QtWidgets.QMessageBox.information(
                    self, "Sukces", "Profil został pomyślnie zapisany."
                )
        except Exception as e:
            self.logger.error(f"Błąd zapisywania profilu: {e}", exc_info=True)
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Nie można zapisać profilu: {e}"
            )

    def _delete_profile(self):
        if not self.profile_list.currentItem():
            QtWidgets.QMessageBox.warning(
                self, "Ostrzeżenie", "Najpierw wybierz profil do usunięcia."
            )
            return
        try:
            current_name = self.profile_list.currentItem().text()
            reply = QtWidgets.QMessageBox.question(
                self,
                "Potwierdzenie",
                f"Czy na pewno chcesz usunąć profil '{current_name}'?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if reply == QtWidgets.QMessageBox.StandardButton.Yes:
                profile_path = self.profiles_dir / f"{current_name}.json"
                if profile_path.exists():
                    profile_path.unlink()
                    self._refresh_profile_list()  # This will also clear selection if item is gone
                    # _on_profile_selected will be called if selection changes to another item or None
                    # No need to manually clear here as _on_profile_selected(None, ...) will handle it.
                    QtWidgets.QMessageBox.information(
                        self, "Sukces", "Profil został pomyślnie usunięty."
                    )
                else:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Błąd",
                        f"Plik profilu '{current_name}' nie został znaleziony.",
                    )
        except Exception as e:
            self.logger.error(f"Błąd usuwania profilu: {e}", exc_info=True)
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Nie można usunąć profilu: {e}"
            )

    def _on_architecture_changed(self, arch_name):
        self._update_variant_combo(arch_name)

    def _update_variant_combo(self, arch_name):
        current_variant = self.variant_combo.currentText()
        self.variant_combo.clear()
        variants = []
        if arch_name == "EfficientNet":
            variants = [f"EfficientNet-B{i}" for i in range(8)]
        elif arch_name == "ConvNeXt":
            variants = [
                "ConvNeXt-Tiny",
                "ConvNeXt-Small",
                "ConvNeXt-Base",
                "ConvNeXt-Large",
                "ConvNeXt-XL",  # Added from original
                "ConvNeXt-Huge",  # Added from original
            ]
        self.variant_combo.addItems(variants)
        if current_variant in variants:
            self.variant_combo.setCurrentText(current_variant)
        elif variants:
            self.variant_combo.setCurrentIndex(0)

    def _select_train_dir(self):
        # ... (bez zmian)
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Wybierz katalog treningowy",
            self.train_dir_edit.text() or str(Path.home()),
        )
        if dir_path:
            # Simple validation: check if directory exists. More complex validation can be added.
            if Path(
                dir_path
            ).is_dir():  # Replaced validate_training_directory for simplicity
                self.train_dir_edit.setText(dir_path)
            else:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Błąd",
                    "Nieprawidłowy katalog treningowy. Katalog nie istnieje.",
                )

    def _select_val_dir(self):
        # ... (bez zmian)
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Wybierz katalog walidacyjny",
            self.val_dir_edit.text() or str(Path.home()),
        )
        if dir_path:
            if Path(
                dir_path
            ).is_dir():  # Replaced validate_validation_directory for simplicity
                self.val_dir_edit.setText(dir_path)
            else:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Błąd",
                    "Nieprawidłowy katalog walidacyjny. Katalog nie istnieje.",
                )

    def _get_unfreeze_strategy_value(self, display_text):
        if "unfreeze_all" in display_text or "Wszystkie na raz" in display_text:
            return self.UNFREEZE_ALL
        if (
            "unfreeze_gradual_end" in display_text
            or "Stopniowo od końca" in display_text
        ):
            return self.UNFREEZE_GRADUAL_END
        if (
            "unfreeze_gradual_start" in display_text
            or "Stopniowo od początku" in display_text
        ):
            return self.UNFREEZE_GRADUAL_START
        if (
            self.UNFREEZE_AFTER_EPOCHS in display_text
            or "Po określonej liczbie epok" in display_text
        ):
            return self.UNFREEZE_AFTER_EPOCHS
        self.logger.warning(
            f"Nieznana strategia odmrażania w UI: {display_text}, domyślnie {self.UNFREEZE_ALL}"
        )
        return self.UNFREEZE_ALL

    def _get_unfreeze_layers_value(self, text_value: str):
        text_value = text_value.strip()
        if not text_value:
            return "all"
        if text_value.lower() == "all":
            return "all"
        if text_value.lower().startswith("last_"):
            try:
                num = int(text_value.split("_")[1])
                return f"last_{num}"
            except (IndexError, ValueError):
                self.logger.warning(
                    f"Nieprawidłowy format 'last_n' dla unfreeze_layers: {text_value}. Zwracam jako tekst."
                )
                return text_value
        if "," in text_value:
            return [layer.strip() for layer in text_value.split(",") if layer.strip()]
        return text_value

    def _get_scheduler_value(self, display_text):
        scheduler_map = {
            "None": "None",
            "CosineAnnealingWarmRestarts": "CosineAnnealingWarmRestarts",
            "StepLR": "StepLR",
            "OneCycleLR": "OneCycleLR",
            "ReduceLROnPlateau": "ReduceLROnPlateau",
            "CosineAnnealingLR": "CosineAnnealingLR",
        }
        return scheduler_map.get(display_text, "None")

    def _toggle_unfreeze_after_epochs_spin(self, strategy_text):
        if hasattr(self, "training_unfreeze_after_epochs_spin"):
            is_enabled = (
                self.UNFREEZE_AFTER_EPOCHS in strategy_text
                or "Po określonej liczbie epok" in strategy_text
            )
            self.training_unfreeze_after_epochs_spin.setEnabled(is_enabled)

    def _toggle_early_stopping_controls(self, state):
        enabled = bool(state)
        for attr_name in [
            "early_stopping_monitor_combo",
            "early_stopping_mode_combo",
            "early_stopping_patience_spin",
            "early_stopping_min_delta_spin",
        ]:
            if hasattr(self, attr_name):
                getattr(self, attr_name).setEnabled(enabled)

    def _toggle_reduce_lr_controls(self, state):
        enabled = bool(state)
        for attr_name in [
            "reduce_lr_monitor_combo",
            "reduce_lr_mode_combo",
            "reduce_lr_factor_spin",
            "reduce_lr_patience_spin",
            "reduce_lr_min_delta_spin",
            "reduce_lr_min_lr_spin",
        ]:
            if hasattr(self, attr_name):
                getattr(self, attr_name).setEnabled(enabled)

    def _toggle_stochastic_depth_controls(self, state):
        if hasattr(self, "stochastic_depth_survival_prob_spin"):
            self.stochastic_depth_survival_prob_spin.setEnabled(bool(state))

    def _toggle_swa_controls(self, state):
        enabled = bool(state)
        if hasattr(self, "swa_start_epoch_spin"):
            self.swa_start_epoch_spin.setEnabled(enabled)
        if hasattr(self, "swa_lr_spin"):
            self.swa_lr_spin.setEnabled(enabled)

    def _toggle_basic_aug_controls(self, state):
        enabled = bool(state)
        # Using corrected names (Poprawka 8)
        controls = [
            "basic_aug_rotation_spin",
            "basic_aug_brightness_spin",
            "basic_aug_contrast_spin",
            "basic_aug_saturation_spin",
            "basic_aug_hue_spin",
            "basic_aug_shift_spin",
            "basic_aug_zoom_spin",
            "basic_aug_horizontal_flip_check",
            "basic_aug_vertical_flip_check",
        ]
        for ctrl_name in controls:
            if hasattr(self, ctrl_name):
                getattr(self, ctrl_name).setEnabled(enabled)

    def _toggle_randaugment_controls(self, state):
        enabled = bool(state)
        if hasattr(self, "randaugment_n_spin"):
            self.randaugment_n_spin.setEnabled(enabled)
        if hasattr(self, "randaugment_m_spin"):
            self.randaugment_m_spin.setEnabled(enabled)

    def _toggle_random_erase_controls(self, state):
        enabled = bool(state)
        controls = [
            "random_erase_prob_spin",
            "random_erase_scale_min_spin",
            "random_erase_scale_max_spin",
            "random_erase_ratio_min_spin",
            "random_erase_ratio_max_spin",
        ]
        for ctrl_name in controls:
            if hasattr(self, ctrl_name):
                getattr(self, ctrl_name).setEnabled(enabled)

    def _toggle_grid_distortion_controls(self, state):
        enabled = bool(state)
        if hasattr(self, "grid_distortion_prob_spin"):
            self.grid_distortion_prob_spin.setEnabled(enabled)
        if hasattr(self, "grid_distortion_limit_spin"):
            self.grid_distortion_limit_spin.setEnabled(enabled)

    def _toggle_preprocess_resize_controls(self, state):
        enabled = bool(state)
        controls = [
            "preprocess_resize_width_spin",
            "preprocess_resize_height_spin",
            "preprocess_resize_mode_combo",
        ]
        for ctrl_name in controls:
            if hasattr(self, ctrl_name):
                getattr(self, ctrl_name).setEnabled(enabled)

    def _toggle_preprocess_normalize_controls(self, state):
        enabled = bool(state)
        controls = [
            "preprocess_normalize_mean_r_spin",
            "preprocess_normalize_mean_g_spin",
            "preprocess_normalize_mean_b_spin",
            "preprocess_normalize_std_r_spin",
            "preprocess_normalize_std_g_spin",
            "preprocess_normalize_std_b_spin",
        ]
        for ctrl_name in controls:
            if hasattr(self, ctrl_name):
                getattr(self, ctrl_name).setEnabled(enabled)

    def _toggle_preprocess_color_jitter_controls(self, state):
        enabled = bool(state)
        controls = [
            "preprocess_color_jitter_brightness_spin",
            "preprocess_color_jitter_contrast_spin",
            "preprocess_color_jitter_saturation_spin",
            "preprocess_color_jitter_hue_spin",
        ]
        for ctrl_name in controls:
            if hasattr(self, ctrl_name):
                getattr(self, ctrl_name).setEnabled(enabled)

    def _toggle_preprocess_gaussian_blur_controls(self, state):
        enabled = bool(state)
        controls = [
            "preprocess_gaussian_blur_kernel_size_spin",
            "preprocess_gaussian_blur_sigma_spin",
        ]
        for ctrl_name in controls:
            if hasattr(self, ctrl_name):
                getattr(self, ctrl_name).setEnabled(enabled)

    def _toggle_preprocess_random_resize_crop_controls(self, state):
        enabled = bool(state)
        controls = [
            "preprocess_random_resize_crop_size_spin",
            "preprocess_random_resize_crop_scale_min_spin",
            "preprocess_random_resize_crop_scale_max_spin",
            "preprocess_random_resize_crop_ratio_min_spin",
            "preprocess_random_resize_crop_ratio_max_spin",
        ]
        for ctrl_name in controls:
            if hasattr(self, ctrl_name):
                getattr(self, ctrl_name).setEnabled(enabled)

    def _toggle_tensorboard_controls(self, state):
        enabled = bool(state)
        if hasattr(self, "tensorboard_log_dir_edit"):
            self.tensorboard_log_dir_edit.setEnabled(enabled)
        if hasattr(self, "tensorboard_update_freq_spin"):
            self.tensorboard_update_freq_spin.setEnabled(enabled)

    def _toggle_wandb_controls(self, state):
        enabled = bool(state)
        controls = ["wandb_project_edit", "wandb_entity_edit", "wandb_tags_edit"]
        for ctrl_name in controls:
            if hasattr(self, ctrl_name):
                getattr(self, ctrl_name).setEnabled(enabled)

    def _toggle_checkpoint_controls(self, state):
        enabled = bool(state)
        controls = [
            "checkpoint_dir_edit",
            "checkpoint_save_best_only_check",
            "checkpoint_monitor_combo",
            "checkpoint_mode_combo",
            "checkpoint_save_freq_spin",
        ]
        for ctrl_name in controls:
            if hasattr(self, ctrl_name):
                getattr(self, ctrl_name).setEnabled(enabled)

    def _toggle_agc_controls(self, state):
        enabled = bool(state)
        if hasattr(self, "advanced_gradient_clip_agc_clipping_spin"):
            self.advanced_gradient_clip_agc_clipping_spin.setEnabled(enabled)
        if hasattr(self, "advanced_gradient_clip_agc_eps_spin"):
            self.advanced_gradient_clip_agc_eps_spin.setEnabled(enabled)

    def _update_dependent_controls(self):
        """Wywołuje wszystkie funkcje _toggle_* aby zaktualizować stan kontrolek po załadowaniu profilu."""
        if hasattr(self, "training_unfreeze_strategy_combo"):
            self._toggle_unfreeze_after_epochs_spin(
                self.training_unfreeze_strategy_combo.currentText()
            )
        if hasattr(self, "use_early_stopping_check"):
            self._toggle_early_stopping_controls(
                self.use_early_stopping_check.isChecked()
            )
        if hasattr(self, "reduce_lr_enabled_check"):
            self._toggle_reduce_lr_controls(self.reduce_lr_enabled_check.isChecked())
        if hasattr(self, "stochastic_depth_use_check"):
            self._toggle_stochastic_depth_controls(
                self.stochastic_depth_use_check.isChecked()
            )
        if hasattr(self, "use_swa_check"):
            self._toggle_swa_controls(self.use_swa_check.isChecked())

        if hasattr(self, "basic_aug_check"):
            self._toggle_basic_aug_controls(self.basic_aug_check.isChecked())
        if hasattr(self, "mixup_check") and hasattr(self, "mixup_alpha_spin"):
            self.mixup_alpha_spin.setEnabled(self.mixup_check.isChecked())
        if hasattr(self, "cutmix_check") and hasattr(self, "cutmix_alpha_spin"):
            self.cutmix_alpha_spin.setEnabled(self.cutmix_check.isChecked())
        if hasattr(self, "autoaugment_check") and hasattr(
            self, "autoaugment_policy_combo"
        ):
            self.autoaugment_policy_combo.setEnabled(self.autoaugment_check.isChecked())
        if hasattr(self, "randaugment_check"):
            self._toggle_randaugment_controls(self.randaugment_check.isChecked())
        if hasattr(self, "random_erase_check"):
            self._toggle_random_erase_controls(self.random_erase_check.isChecked())
        if hasattr(self, "grid_distortion_check"):
            self._toggle_grid_distortion_controls(
                self.grid_distortion_check.isChecked()
            )

        if hasattr(self, "preprocess_resize_enabled_check"):
            self._toggle_preprocess_resize_controls(
                self.preprocess_resize_enabled_check.isChecked()
            )
        if hasattr(self, "preprocess_normalize_enabled_check"):
            self._toggle_preprocess_normalize_controls(
                self.preprocess_normalize_enabled_check.isChecked()
            )
        if hasattr(self, "preprocess_grayscale_enabled_check") and hasattr(
            self, "preprocess_grayscale_num_output_channels_spin"
        ):
            self.preprocess_grayscale_num_output_channels_spin.setEnabled(
                self.preprocess_grayscale_enabled_check.isChecked()
            )
        if hasattr(self, "preprocess_color_jitter_enabled_check"):
            self._toggle_preprocess_color_jitter_controls(
                self.preprocess_color_jitter_enabled_check.isChecked()
            )
        if hasattr(self, "preprocess_gaussian_blur_enabled_check"):
            self._toggle_preprocess_gaussian_blur_controls(
                self.preprocess_gaussian_blur_enabled_check.isChecked()
            )
        if hasattr(self, "preprocess_random_resize_crop_enabled_check"):
            self._toggle_preprocess_random_resize_crop_controls(
                self.preprocess_random_resize_crop_enabled_check.isChecked()
            )

        if hasattr(self, "tensorboard_enabled_check"):
            self._toggle_tensorboard_controls(
                self.tensorboard_enabled_check.isChecked()
            )
        if hasattr(self, "wandb_enabled_check"):
            self._toggle_wandb_controls(self.wandb_enabled_check.isChecked())
        if hasattr(self, "checkpoint_enabled_check"):
            self._toggle_checkpoint_controls(self.checkpoint_enabled_check.isChecked())
        if hasattr(self, "advanced_gradient_clip_agc_check"):
            self._toggle_agc_controls(self.advanced_gradient_clip_agc_check.isChecked())

    def _get_current_config_as_dict(self):
        """Zbiera wszystkie ustawienia z UI i zwraca jako słownik."""

        optimization_params_values = {}
        opt_tab_keys = [
            "cudnn_benchmark",
            "pin_memory",
            "shuffle",
            "prefetch_factor",
            "persistent_workers",
            "drop_last",
        ]
        for key in opt_tab_keys:
            if key in self.parameter_rows:
                row_data = self.parameter_rows[key]
                widget = row_data["value_widget"]
                if isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                    optimization_params_values[key] = widget.value()
                elif isinstance(widget, QtWidgets.QCheckBox):
                    optimization_params_values[key] = widget.isChecked()
                else:  # QLineEdit
                    optimization_params_values[key] = widget.text()
            else:  # Fallback if widget somehow not in parameter_rows (should not happen)
                default_map = {
                    "cudnn_benchmark": True,
                    "pin_memory": True,
                    "shuffle": True,
                    "prefetch_factor": 2,
                    "persistent_workers": False,
                    "drop_last": False,
                }
                optimization_params_values[key] = default_map.get(key)

        # Constructing the optimization part of the config, including nested dataloader (Poprawka 14)
        final_optimization_config = {
            "cudnn_benchmark": optimization_params_values.get("cudnn_benchmark", True),
            "pin_memory": optimization_params_values.get("pin_memory", True),
            "shuffle": optimization_params_values.get("shuffle", True),
            "prefetch_factor": optimization_params_values.get("prefetch_factor", 2),
            "persistent_workers": optimization_params_values.get(
                "persistent_workers", False
            ),
            "drop_last": optimization_params_values.get("drop_last", False),
            "dataloader": {
                "pin_memory": optimization_params_values.get(
                    "pin_memory", True
                ),  # Added pin_memory here too
                "shuffle": optimization_params_values.get("shuffle", True),
                "prefetch_factor": optimization_params_values.get("prefetch_factor", 2),
                "persistent_workers": optimization_params_values.get(
                    "persistent_workers", False
                ),
                "drop_last": optimization_params_values.get("drop_last", False),
            },
        }

        config_data = {
            "type": "training",
            "info": "",
            "description": "",
            "data_required": "",
            "hardware_required": "",
            "config": {
                "train_dir": self.train_dir_edit.text().strip(),
                "val_dir": self.val_dir_edit.text().strip(),
                "model": {
                    "architecture": self.arch_combo.currentText(),
                    "variant": self.variant_combo.currentText(),
                    "input_size": self.input_size_spin.value(),
                    "num_classes": self.num_classes_spin.value(),
                    "pretrained": self.model_pretrained_check.isChecked(),
                    "pretrained_weights": self.model_pretrained_weights_combo.currentText(),
                    "feature_extraction_only": self.model_feature_extraction_check.isChecked(),
                    "activation": self.model_activation_combo.currentText(),
                    "dropout_at_inference": self.model_dropout_at_inference_check.isChecked(),
                    "global_pool": self.model_global_pool_combo.currentText(),
                    "last_layer_activation": self.model_last_layer_activation_combo.currentText(),
                },
                "training": {
                    "epochs": self.training_epochs_spin.value(),
                    "batch_size": self.training_batch_size_spin.value(),
                    "learning_rate": self.training_learning_rate_spin.value(),
                    "optimizer": self.training_optimizer_combo.currentText(),
                    "scheduler": {
                        "type": self._get_scheduler_value(
                            self.training_scheduler_type_combo.currentText()
                        ),
                        "T_0": self.training_scheduler_t0_spin.value(),
                        "T_mult": self.training_scheduler_tmult_spin.value(),
                        "eta_min": self.training_scheduler_eta_min_spin.value(),
                    },
                    "num_workers": self.training_num_workers_spin.value(),
                    "warmup_epochs": self.training_warmup_epochs_spin.value(),
                    "warmup_lr_init": self.training_warmup_lr_init_spin.value(),
                    "mixed_precision": self.training_mixed_precision_check.isChecked(),
                    "gradient_accumulation_steps": self.training_grad_accum_steps_spin.value(),
                    "gradient_clip": self.training_gradient_clip_value_spin.value(),
                    "evaluation_freq": self.training_evaluation_freq_spin.value(),
                    "use_ema": self.training_use_ema_check.isChecked(),
                    "ema_decay": self.training_ema_decay_spin.value(),
                    "freeze_base_model": self.training_freeze_base_model_check.isChecked(),
                    "unfreeze_layers": self._get_unfreeze_layers_value(
                        self.training_unfreeze_layers_edit.text()
                    ),
                    "unfreeze_strategy": self._get_unfreeze_strategy_value(
                        self.training_unfreeze_strategy_combo.currentText()
                    ),
                    "unfreeze_after_epochs": self.training_unfreeze_after_epochs_spin.value(),
                    "frozen_lr": self.training_frozen_lr_spin.value(),
                    "unfrozen_lr": self.training_unfrozen_lr_spin.value(),
                    "validation_split": self.training_validation_split_spin.value(),
                },
                "regularization": {
                    "weight_decay": self.reg_weight_decay_spin.value(),
                    "label_smoothing": self.reg_label_smoothing_spin.value(),
                    "dropout_rate": self.reg_dropout_rate_spin.value(),
                    "drop_connect_rate": self.reg_drop_connect_rate_spin.value(),
                    "gradient_clip": self.reg_gradient_clip_spin.value(),
                    "momentum": self.reg_momentum_spin.value(),
                    "epsilon": self.reg_epsilon_spin.value(),
                    "stochastic_depth": {
                        "use": self.stochastic_depth_use_check.isChecked(),
                        "survival_probability": self.stochastic_depth_survival_prob_spin.value(),
                    },
                    "swa": {
                        "use": self.use_swa_check.isChecked(),
                        "start_epoch": self.swa_start_epoch_spin.value(),
                        "lr_swa": self.swa_lr_spin.value(),
                    },
                },
                "augmentation": {
                    "basic": {  # Using corrected names (Poprawka 8)
                        "use": self.basic_aug_check.isChecked(),
                        "rotation": self.basic_aug_rotation_spin.value(),
                        "brightness": self.basic_aug_brightness_spin.value(),
                        "contrast": self.basic_aug_contrast_spin.value(),
                        "saturation": self.basic_aug_saturation_spin.value(),
                        "hue": self.basic_aug_hue_spin.value(),
                        "shift": self.basic_aug_shift_spin.value(),
                        "zoom": self.basic_aug_zoom_spin.value(),
                        "horizontal_flip": self.basic_aug_horizontal_flip_check.isChecked(),
                        "vertical_flip": self.basic_aug_vertical_flip_check.isChecked(),
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
                        "policy": self.autoaugment_policy_combo.currentText(),
                    },
                    "randaugment": {
                        "use": self.randaugment_check.isChecked(),
                        "n": self.randaugment_n_spin.value(),
                        "m": self.randaugment_m_spin.value(),
                    },
                    "trivialaugment": {"use": self.trivialaugment_check.isChecked()},
                    "random_erase": {
                        "use": self.random_erase_check.isChecked(),
                        "probability": self.random_erase_prob_spin.value(),
                        "scale": [
                            self.random_erase_scale_min_spin.value(),
                            self.random_erase_scale_max_spin.value(),
                        ],
                        "ratio": [
                            self.random_erase_ratio_min_spin.value(),
                            self.random_erase_ratio_max_spin.value(),
                        ],
                    },
                    "grid_distortion": {
                        "enabled": self.grid_distortion_check.isChecked(),
                        "probability": self.grid_distortion_prob_spin.value(),
                        "distort_limit": self.grid_distortion_limit_spin.value(),
                    },
                    "resize": {"enabled": self.aug_resize_enabled_check.isChecked()},
                },
                "preprocessing": {
                    "resize": {
                        "enabled": self.preprocess_resize_enabled_check.isChecked(),
                        "size": [
                            self.preprocess_resize_width_spin.value(),
                            self.preprocess_resize_height_spin.value(),
                        ],
                        "mode": self.preprocess_resize_mode_combo.currentText(),
                    },
                    "normalize": {
                        "enabled": self.preprocess_normalize_enabled_check.isChecked(),
                        "mean": [
                            self.preprocess_normalize_mean_r_spin.value(),
                            self.preprocess_normalize_mean_g_spin.value(),
                            self.preprocess_normalize_mean_b_spin.value(),
                        ],
                        "std": [
                            self.preprocess_normalize_std_r_spin.value(),
                            self.preprocess_normalize_std_g_spin.value(),
                            self.preprocess_normalize_std_b_spin.value(),
                        ],
                    },
                    "grayscale": {
                        "enabled": self.preprocess_grayscale_enabled_check.isChecked(),
                        "num_output_channels": self.preprocess_grayscale_num_output_channels_spin.value(),
                    },
                    "color_jitter": {
                        "enabled": self.preprocess_color_jitter_enabled_check.isChecked(),
                        "brightness": self.preprocess_color_jitter_brightness_spin.value(),
                        "contrast": self.preprocess_color_jitter_contrast_spin.value(),
                        "saturation": self.preprocess_color_jitter_saturation_spin.value(),
                        "hue": self.preprocess_color_jitter_hue_spin.value(),
                    },
                    "gaussian_blur": {
                        "enabled": self.preprocess_gaussian_blur_enabled_check.isChecked(),
                        "kernel_size": self.preprocess_gaussian_blur_kernel_size_spin.value(),
                        "sigma": self.preprocess_gaussian_blur_sigma_spin.value(),
                    },
                    "cache_dataset": self.preprocess_cache_dataset_check.isChecked(),
                    "scaling": {
                        "method": self.preprocess_scaling_method_combo.currentText(),
                        "maintain_aspect_ratio": self.preprocess_scaling_maintain_aspect_ratio_check.isChecked(),
                        "pad_to_square": self.preprocess_scaling_pad_to_square_check.isChecked(),
                        "pad_mode": self.preprocess_scaling_pad_mode_combo.currentText(),
                        "pad_value": self.preprocess_scaling_pad_value_spin.value(),
                    },
                    "random_resize_crop": {
                        "enabled": self.preprocess_random_resize_crop_enabled_check.isChecked(),
                        "size": self.preprocess_random_resize_crop_size_spin.value(),
                        "scale": [
                            self.preprocess_random_resize_crop_scale_min_spin.value(),
                            self.preprocess_random_resize_crop_scale_max_spin.value(),
                        ],
                        "ratio": [
                            self.preprocess_random_resize_crop_ratio_min_spin.value(),
                            self.preprocess_random_resize_crop_ratio_max_spin.value(),
                        ],
                    },
                },
                "monitoring": {
                    "metrics": {
                        "accuracy": self.metrics_accuracy_check.isChecked(),
                        "precision": self.metrics_precision_check.isChecked(),
                        "recall": self.metrics_recall_check.isChecked(),
                        "f1": self.metrics_f1_check.isChecked(),
                        "topk": [2, 5] if self.metrics_topk_check.isChecked() else [],
                        "confusion_matrix": self.metrics_confusion_matrix_check.isChecked(),
                        "auc": self.metrics_auc_check.isChecked(),
                        "gpu_utilization": self.metrics_gpu_utilization_check.isChecked(),
                        "memory_usage": self.metrics_memory_usage_check.isChecked(),
                    },
                    "tensorboard": {
                        "enabled": self.tensorboard_enabled_check.isChecked(),
                        "log_dir": self.tensorboard_log_dir_edit.text(),
                        "update_freq": self.tensorboard_update_freq_spin.value(),
                    },
                    "wandb": {
                        "enabled": self.wandb_enabled_check.isChecked(),
                        "project": self.wandb_project_edit.text(),
                        "entity": self.wandb_entity_edit.text(),
                        "tags": [
                            tag.strip()
                            for tag in self.wandb_tags_edit.text().split(",")
                            if tag.strip()
                        ],
                    },
                    "checkpoint": {
                        "enabled": self.checkpoint_enabled_check.isChecked(),
                        "dir": self.checkpoint_dir_edit.text(),
                        "save_best_only": self.checkpoint_save_best_only_check.isChecked(),
                        "monitor": self.checkpoint_monitor_combo.currentText(),
                        "mode": self.checkpoint_mode_combo.currentText(),
                        "save_freq": self.checkpoint_save_freq_spin.value(),
                    },
                    "early_stopping": {
                        "enabled": self.use_early_stopping_check.isChecked(),
                        "monitor": self.early_stopping_monitor_combo.currentText(),
                        "mode": self.early_stopping_mode_combo.currentText(),
                        "patience": self.early_stopping_patience_spin.value(),
                        "min_delta": self.early_stopping_min_delta_spin.value(),
                    },
                    "reduce_lr": {
                        "enabled": self.reduce_lr_enabled_check.isChecked(),
                        "monitor": self.reduce_lr_monitor_combo.currentText(),
                        "mode": self.reduce_lr_mode_combo.currentText(),
                        "factor": self.reduce_lr_factor_spin.value(),
                        "patience": self.reduce_lr_patience_spin.value(),
                        "min_delta": self.reduce_lr_min_delta_spin.value(),
                        "min_lr": self.reduce_lr_min_lr_spin.value(),
                    },
                },
                "advanced": {
                    "seed": self.advanced_seed_spin.value(),
                    "deterministic": self.advanced_deterministic_check.isChecked(),
                    "benchmark": self.advanced_benchmark_check.isChecked(),
                    "gradient_clip_val": self.advanced_gradient_clip_val_spin.value(),
                    "gradient_clip_norm": self.advanced_gradient_clip_norm_spin.value(),
                    "gradient_clip_algorithm": self.advanced_gradient_clip_algorithm_combo.currentText(),
                    "accumulate_grad_batches": self.advanced_accumulate_grad_batches_spin.value(),
                    "sync_batchnorm": self.advanced_sync_batchnorm_check.isChecked(),
                    "precision": self.advanced_precision_combo.currentText(),
                    "amp_level": self.advanced_amp_level_combo.currentText(),
                    "gradient_clip_agc": {
                        "use": self.advanced_gradient_clip_agc_check.isChecked(),
                        "clipping": self.advanced_gradient_clip_agc_clipping_spin.value(),
                        "eps": self.advanced_gradient_clip_agc_eps_spin.value(),
                    },
                },
                "optimization": final_optimization_config,  # Using the structured dict
            },
        }
        return config_data

    def _on_accept(self):
        try:
            variant = self.variant_combo.currentText()
            num_classes = self.num_classes_spin.value()
            now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            task_name = f"{variant}_{num_classes}cls_{now}"

            train_dir = self.train_dir_edit.text().strip()
            if not train_dir:
                QtWidgets.QMessageBox.warning(
                    self, "Błąd", "Katalog treningowy nie może być pusty."
                )
                return
            if not Path(train_dir).is_dir():
                QtWidgets.QMessageBox.warning(
                    self, "Błąd", f"Katalog treningowy '{train_dir}' nie istnieje."
                )
                return

            val_dir = self.val_dir_edit.text().strip()
            if not val_dir and self.training_validation_split_spin.value() == 0.0:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Błąd",
                    "Katalog walidacyjny nie może być pusty, jeśli nie używasz podziału z danych treningowych.",
                )
                return
            if val_dir and not Path(val_dir).is_dir():
                QtWidgets.QMessageBox.warning(
                    self, "Błąd", f"Katalog walidacyjny '{val_dir}' nie istnieje."
                )
                return

            self.logger.info("=== TWORZENIE NOWEGO ZADANIA TRENINGOWEGO ===")
            self.logger.info(f"Nazwa zadania: {task_name}")

            full_config_data = self._get_current_config_as_dict()

            self.task_config = {
                "name": task_name,
                "type": "training",
                "status": "Nowy",
                "priority": 0,
                "created_at": datetime.datetime.now().isoformat(),
                "config": full_config_data["config"],
            }

            self.logger.info(f"Utworzono konfigurację zadania: {task_name}")
            # config_str = json.dumps(self.task_config, indent=2, ensure_ascii=False) # Keep for debugging if needed
            # self.logger.debug(f"Pełna konfiguracja zadania: {config_str}")

            self.accept()

        except Exception as e:
            self.logger.error(f"Błąd podczas dodawania zadania: {e}", exc_info=True)
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Nie można dodać zadania: {e}"
            )

    def get_task_config(self):
        return getattr(self, "task_config", None)

    def closeEvent(self, event):
        self.logger.info("Zamykanie okna dialogowego konfiguracji treningu")
        super().closeEvent(event)

    def _show_hardware_profile(self):
        # Ensure hardware_profile is a dict, even if empty
        current_hw_profile = (
            self.hardware_profile if isinstance(self.hardware_profile, dict) else {}
        )
        dialog = HardwareProfileDialog(current_hw_profile, self)
        dialog.exec()


if __name__ == "__main__":
    import sys

    mock_hardware_profile = {
        "recommended_batch_size": 16,  # This key is not directly used by _create_parameter_row's map
        "recommended_workers": 2,  # This key is not directly used by _create_parameter_row's map
        "use_mixed_precision": True,  # This key is not directly used by _create_parameter_row's map
        "cudnn_benchmark": True,
        "pin_memory": True,
        "prefetch_factor": 4,  # Example different value
        "persistent_workers": True,  # Example different value
        "shuffle": False,  # Example different value for dataloader shuffle
        "drop_last": True,  # Example
    }

    app = QtWidgets.QApplication(sys.argv)
    # Setup basic logging to see messages
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    dialog = TrainingTaskConfigDialog(hardware_profile=mock_hardware_profile)

    # Example: Try to load a default profile to test _apply_profile more thoroughly
    # Create a dummy profile for testing
    # test_profile_name = "test_profile.json"
    # test_profile_content = {
    #     "type": "training",
    #     "info": "Test Profile",
    #     "description": "A profile for testing apply.",
    #     "config": {
    #         "model": {"architecture": "ConvNeXt", "variant": "ConvNeXt-Small", "input_size": 256},
    #         "training": {"epochs": 10, "learning_rate": 0.0005},
    #         "optimization": {
    #             "cudnn_benchmark": False,
    #             "pin_memory": False,
    #             "dataloader": {"prefetch_factor": 3}
    #          }
    #     }
    # }
    # with open(dialog.profiles_dir / test_profile_name, "w") as f:
    #     json.dump(test_profile_content, f)
    # dialog._refresh_profile_list()
    # for i in range(dialog.profile_list.count()):
    #     if dialog.profile_list.item(i).text() == test_profile_name.replace(".json",""):
    #         dialog.profile_list.setCurrentRow(i)
    #         # dialog._apply_profile() # _on_profile_selected should trigger loading if we want to auto-apply
    #         break

    if dialog.exec():
        task_conf = dialog.get_task_config()
        if task_conf:
            print("Dodano zadanie:")
            print(json.dumps(task_conf, indent=4, ensure_ascii=False))
        else:
            print("Anulowano dodawanie zadania.")
    else:
        print("Okno dialogowe zostało zamknięte bez dodawania zadania.")

    # sys.exit(app.exec()) # Not needed with QDialog.exec()
