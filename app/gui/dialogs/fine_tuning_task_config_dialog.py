import json
import logging
import os
from datetime import datetime
from pathlib import Path

from PyQt6 import QtWidgets

from app.utils.config import DEFAULT_TRAINING_PARAMS
from app.utils.file_utils import (
    validate_training_directory,
    validate_validation_directory,
)


class FineTuningTaskConfigDialog(QtWidgets.QDialog):
    """Okno dialogowe konfiguracji doszkalania modelu."""

    # Stałe dla strategii odblokowywania warstw
    UNFREEZE_ALL = "Odblokuj wszystkie warstwy"
    UNFREEZE_GRADUAL_END = "Odblokuj stopniowo od końca"
    UNFREEZE_GRADUAL_START = "Odblokuj stopniowo od początku"
    UNFREEZE_AFTER_EPOCHS = "Odblokuj po określonej liczbie epok"

    def __init__(self, parent=None):
        """Inicjalizacja okna dialogowego konfiguracji doszkalania."""
        try:
            super().__init__(parent)
            self.setWindowTitle("Konfiguracja doszkalania")
            self.setMinimumWidth(1000)

            # Konfiguracja logowania
            self._setup_logging()

            # Katalog profili
            self.profiles_dir = Path("profiles")
            self.profiles_dir.mkdir(exist_ok=True)

            # Inicjalizacja interfejsu
            self._init_ui()

            self.logger.debug("Zainicjalizowano okno dialogowe")
        except Exception as e:
            msg = "Błąd podczas inicjalizacji okna dialogowego"
            print(f"{msg}: {str(e)}")
            raise

    def _setup_logging(self):
        """Konfiguracja logowania."""
        try:
            self.logger = logging.getLogger("FineTuningTaskConfigDialog")
            self.logger.setLevel(logging.DEBUG)

            # Handler do pliku
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            file_handler = logging.FileHandler(log_dir / "fine_tuning_dialog.log")
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

            # Handler do konsoli
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter("%(levelname)s: %(message)s")
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

            self.logger.debug("Zainicjalizowano logowanie")
        except Exception as e:
            print(f"Błąd podczas konfiguracji logowania: {str(e)}")
            raise

    def _init_ui(self):
        """Inicjalizacja interfejsu użytkownika z zakładkami."""
        try:
            self.logger.debug("Rozpoczęcie inicjalizacji UI")
            layout = QtWidgets.QVBoxLayout(self)

            # Utworzenie zakładek
            self.tabs = QtWidgets.QTabWidget()

            # 1. Zakładka: Model i Dane
            tab = self._create_model_data_tab()
            self.tabs.addTab(tab, "Model i Dane")

            # 2. Zakładka: Parametry Doszkalania
            tab = self._create_fine_tuning_params_tab()
            self.tabs.addTab(tab, "Parametry")

            # 3. Zakładka: Regularyzacja
            tab = self._create_regularization_tab()
            self.tabs.addTab(tab, "Regularyzacja")

            # 4. Zakładka: Augmentacja
            tab = self._create_augmentation_tab()
            self.tabs.addTab(tab, "Augmentacja")

            # 5. Zakładka: Monitorowanie
            tab = self._create_monitoring_tab()
            self.tabs.addTab(tab, "Monitorowanie")

            # 6. Zakładka: Zaawansowane
            tab = self._create_advanced_tab()
            self.tabs.addTab(tab, "Zaawansowane")

            layout.addWidget(self.tabs)

            # Przyciski OK i Anuluj
            buttons = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.StandardButton.Ok
                | QtWidgets.QDialogButtonBox.StandardButton.Cancel
            )
            buttons.accepted.connect(self.accept)
            buttons.rejected.connect(self.reject)
            layout.addWidget(buttons)

            self.logger.debug("Zakończono inicjalizację UI")

        except Exception as e:
            msg = "Błąd podczas inicjalizacji UI"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            raise

    def _create_model_data_tab(self):
        """Tworzenie zakładki Model i Dane."""
        try:
            self.logger.debug("Tworzenie zakładki Model i Dane")
            tab = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(tab)
            form = QtWidgets.QFormLayout()

            # Wybór modelu bazowego
            base_model_layout = QtWidgets.QHBoxLayout()
            self.base_model_edit = QtWidgets.QLineEdit()
            base_model_btn = QtWidgets.QPushButton("Przeglądaj...")
            base_model_btn.clicked.connect(self._select_base_model)
            base_model_layout.addWidget(self.base_model_edit)
            base_model_layout.addWidget(base_model_btn)
            form.addRow("Model bazowy:", base_model_layout)

            # Katalog danych treningowych
            train_dir_layout = QtWidgets.QHBoxLayout()
            self.train_dir_edit = QtWidgets.QLineEdit()
            train_dir_btn = QtWidgets.QPushButton("Przeglądaj...")
            train_dir_btn.clicked.connect(self._select_train_dir)
            train_dir_layout.addWidget(self.train_dir_edit)
            train_dir_layout.addWidget(train_dir_btn)
            form.addRow("Katalog treningowy:", train_dir_layout)

            # Katalog danych walidacyjnych
            val_dir_layout = QtWidgets.QHBoxLayout()
            self.val_dir_edit = QtWidgets.QLineEdit()
            val_dir_btn = QtWidgets.QPushButton("Przeglądaj...")
            val_dir_btn.clicked.connect(self._select_val_dir)
            val_dir_layout.addWidget(self.val_dir_edit)
            val_dir_layout.addWidget(val_dir_btn)
            form.addRow("Katalog walidacyjny:", val_dir_layout)

            # Architektura modelu
            self.arch_combo = QtWidgets.QComboBox()
            self.arch_combo.addItems(["EfficientNet", "ConvNeXt"])
            form.addRow("Architektura:", self.arch_combo)

            # Wariant modelu
            self.variant_combo = QtWidgets.QComboBox()
            self._update_variant_combo("EfficientNet")
            form.addRow("Wariant:", self.variant_combo)
            self.arch_combo.currentTextChanged.connect(self._on_architecture_changed)

            # Rozmiar obrazu wejściowego
            self.input_size_spin = QtWidgets.QSpinBox()
            self.input_size_spin.setRange(32, 1024)
            self.input_size_spin.setValue(224)
            self.input_size_spin.setSingleStep(32)
            form.addRow("Rozmiar obrazu:", self.input_size_spin)

            # Liczba klas
            self.num_classes_spin = QtWidgets.QSpinBox()
            self.num_classes_spin.setRange(2, 1000)
            self.num_classes_spin.setValue(38)
            form.addRow("Liczba klas:", self.num_classes_spin)

            # Grupa profili
            profile_group = QtWidgets.QGroupBox("Dostępne profile")
            profile_layout = QtWidgets.QVBoxLayout()

            # Informacje o profilu
            info_group = QtWidgets.QGroupBox("Informacje o profilu")
            info_layout = QtWidgets.QFormLayout()

            # Nowe pola informacyjne
            self.profile_name_edit = QtWidgets.QLineEdit()
            info_layout.addRow("Nazwa profilu:", self.profile_name_edit)

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
            self.profile_hardware_required.setMaximumHeight(60)
            info_layout.addRow("Wymagany sprzęt:", self.profile_hardware_required)

            info_group.setLayout(info_layout)
            profile_layout.addWidget(info_group)

            # Lista profili
            self.profile_list = QtWidgets.QListWidget()
            self.profile_list.currentItemChanged.connect(self._on_profile_selected)
            self._refresh_profile_list()
            profile_layout.addWidget(self.profile_list)

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

            profile_layout.addLayout(buttons_layout)
            profile_group.setLayout(profile_layout)

            layout.addLayout(form)
            layout.addWidget(profile_group)
            return tab

        except Exception as e:
            msg = "Błąd podczas tworzenia zakładki"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            raise

    def _create_fine_tuning_params_tab(self):
        """Tworzenie zakładki Parametry doszkalania."""
        try:
            self.logger.debug("Tworzenie zakładki Parametry doszkalania")
            tab = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(tab)
            form = QtWidgets.QFormLayout()

            # Liczba epok
            self.epochs_spin = QtWidgets.QSpinBox()
            self.epochs_spin.setRange(1, 1000)
            self.epochs_spin.setValue(40)
            form.addRow("Liczba epok:", self.epochs_spin)

            # Rozmiar batcha
            self.batch_size_spin = QtWidgets.QSpinBox()
            self.batch_size_spin.setRange(1, 1024)
            self.batch_size_spin.setValue(16)
            self.batch_size_spin.setSingleStep(8)
            form.addRow("Rozmiar batcha:", self.batch_size_spin)

            # Learning rate
            self.learning_rate_spin = QtWidgets.QDoubleSpinBox()
            self.learning_rate_spin.setRange(0.000001, 1.0)
            self.learning_rate_spin.setValue(0.00005)
            self.learning_rate_spin.setSingleStep(0.00001)
            self.learning_rate_spin.setDecimals(6)
            form.addRow("Learning rate:", self.learning_rate_spin)

            # Optymalizator
            self.optimizer_combo = QtWidgets.QComboBox()
            self.optimizer_combo.addItems(["Adam", "AdamW", "SGD", "RMSprop"])
            self.optimizer_combo.setCurrentText("AdamW")
            form.addRow("Optymalizator:", self.optimizer_combo)

            # Strategia odblokowywania warstw
            self.unfreeze_strategy_combo = QtWidgets.QComboBox()
            self.unfreeze_strategy_combo.addItems(
                [
                    "Odblokuj wszystkie warstwy",
                    "Odblokuj stopniowo od końca",
                    "Odblokuj stopniowo od początku",
                    "Odblokuj po określonej liczbie epok",
                ]
            )
            form.addRow("Strategia odblokowywania:", self.unfreeze_strategy_combo)

            # Liczba epok przed odblokowaniem
            self.unfreeze_after_epochs_spin = QtWidgets.QSpinBox()
            self.unfreeze_after_epochs_spin.setRange(1, 1000)
            self.unfreeze_after_epochs_spin.setValue(5)
            form.addRow("Odblokuj po epokach:", self.unfreeze_after_epochs_spin)

            # Liczba warstw do odblokowania
            self.unfreeze_layers_spin = QtWidgets.QSpinBox()
            self.unfreeze_layers_spin.setRange(1, 100)
            self.unfreeze_layers_spin.setValue(3)
            form.addRow("Liczba warstw do odblokowania:", self.unfreeze_layers_spin)

            # Scheduler
            self.scheduler_combo = QtWidgets.QComboBox()
            self.scheduler_combo.addItems(
                ["Brak", "CosineAnnealingLR", "ReduceLROnPlateau", "OneCycleLR"]
            )
            form.addRow("Scheduler:", self.scheduler_combo)

            # Parametry schedulera
            scheduler_group = QtWidgets.QGroupBox("Parametry schedulera")
            scheduler_layout = QtWidgets.QFormLayout()

            # CosineAnnealingLR
            self.t_max_spin = QtWidgets.QSpinBox()
            self.t_max_spin.setRange(1, 1000)
            self.t_max_spin.setValue(10)
            scheduler_layout.addRow("T_max:", self.t_max_spin)

            self.eta_min_spin = QtWidgets.QDoubleSpinBox()
            self.eta_min_spin.setRange(0.000001, 1.0)
            self.eta_min_spin.setValue(0.00001)
            self.eta_min_spin.setDecimals(6)
            scheduler_layout.addRow("Eta min:", self.eta_min_spin)

            # ReduceLROnPlateau
            self.patience_spin = QtWidgets.QSpinBox()
            self.patience_spin.setRange(1, 100)
            self.patience_spin.setValue(5)
            scheduler_layout.addRow("Patience:", self.patience_spin)

            self.factor_spin = QtWidgets.QDoubleSpinBox()
            self.factor_spin.setRange(0.1, 1.0)
            self.factor_spin.setValue(0.5)
            self.factor_spin.setSingleStep(0.1)
            scheduler_layout.addRow("Factor:", self.factor_spin)

            # OneCycleLR
            self.max_lr_spin = QtWidgets.QDoubleSpinBox()
            self.max_lr_spin.setRange(0.000001, 1.0)
            self.max_lr_spin.setValue(0.01)
            self.max_lr_spin.setDecimals(6)
            scheduler_layout.addRow("Max LR:", self.max_lr_spin)

            self.pct_start_spin = QtWidgets.QDoubleSpinBox()
            self.pct_start_spin.setRange(0.1, 0.9)
            self.pct_start_spin.setValue(0.3)
            self.pct_start_spin.setSingleStep(0.1)
            scheduler_layout.addRow("Pct start:", self.pct_start_spin)

            scheduler_group.setLayout(scheduler_layout)
            form.addRow(scheduler_group)

            layout.addLayout(form)
            return tab

        except Exception as e:
            msg = "Błąd podczas tworzenia zakładki"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            raise

    def _create_regularization_tab(self):
        """Tworzenie zakładki Regularizacja."""
        try:
            self.logger.debug("Tworzenie zakładki Regularizacja")
            tab = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(tab)
            form = QtWidgets.QFormLayout()

            # Weight decay
            self.weight_decay_spin = QtWidgets.QDoubleSpinBox()
            self.weight_decay_spin.setRange(0.0, 1.0)
            self.weight_decay_spin.setValue(0.001)
            self.weight_decay_spin.setDecimals(6)
            form.addRow("Weight decay:", self.weight_decay_spin)

            # Dropout
            self.dropout_spin = QtWidgets.QDoubleSpinBox()
            self.dropout_spin.setRange(0.0, 0.9)
            self.dropout_spin.setValue(0.2)
            self.dropout_spin.setSingleStep(0.1)
            form.addRow("Dropout:", self.dropout_spin)

            # Gradient clipping
            self.gradient_clip_spin = QtWidgets.QDoubleSpinBox()
            self.gradient_clip_spin.setRange(0.0, 10.0)
            self.gradient_clip_spin.setValue(0.5)
            self.gradient_clip_spin.setSingleStep(0.1)
            form.addRow("Gradient clipping:", self.gradient_clip_spin)

            # Label smoothing
            self.label_smoothing_spin = QtWidgets.QDoubleSpinBox()
            self.label_smoothing_spin.setRange(0.0, 0.5)
            self.label_smoothing_spin.setValue(0.05)
            self.label_smoothing_spin.setSingleStep(0.05)
            form.addRow("Label smoothing:", self.label_smoothing_spin)

            # Mixup
            self.mixup_check = QtWidgets.QCheckBox()
            form.addRow("Mixup:", self.mixup_check)

            # Mixup alpha
            self.mixup_alpha_spin = QtWidgets.QDoubleSpinBox()
            self.mixup_alpha_spin.setRange(0.1, 10.0)
            self.mixup_alpha_spin.setValue(0.2)
            self.mixup_alpha_spin.setSingleStep(0.1)
            form.addRow("Mixup alpha:", self.mixup_alpha_spin)

            # CutMix
            self.cutmix_check = QtWidgets.QCheckBox()
            form.addRow("CutMix:", self.cutmix_check)

            # CutMix alpha
            self.cutmix_alpha_spin = QtWidgets.QDoubleSpinBox()
            self.cutmix_alpha_spin.setRange(0.1, 10.0)
            self.cutmix_alpha_spin.setValue(1.0)
            self.cutmix_alpha_spin.setSingleStep(0.1)
            form.addRow("CutMix alpha:", self.cutmix_alpha_spin)

            # Stochastic depth
            self.stochastic_depth_spin = QtWidgets.QDoubleSpinBox()
            self.stochastic_depth_spin.setRange(0.0, 0.9)
            self.stochastic_depth_spin.setValue(0.1)
            self.stochastic_depth_spin.setSingleStep(0.1)
            form.addRow("Stochastic depth:", self.stochastic_depth_spin)

            layout.addLayout(form)
            return tab

        except Exception as e:
            msg = "Błąd podczas tworzenia zakładki"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            raise

    def _create_augmentation_tab(self):
        """Tworzenie zakładki Augmentacja."""
        try:
            self.logger.debug("Tworzenie zakładki Augmentacja")
            tab = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(tab)
            form = QtWidgets.QFormLayout()

            # Random horizontal flip
            self.horizontal_flip_check = QtWidgets.QCheckBox()
            form.addRow("Random horizontal flip:", self.horizontal_flip_check)

            # Random vertical flip
            self.vertical_flip_check = QtWidgets.QCheckBox()
            form.addRow("Random vertical flip:", self.vertical_flip_check)

            # Random rotation
            self.rotation_check = QtWidgets.QCheckBox()
            form.addRow("Random rotation:", self.rotation_check)

            # Rotation degrees
            self.rotation_degrees_spin = QtWidgets.QSpinBox()
            self.rotation_degrees_spin.setRange(0, 180)
            self.rotation_degrees_spin.setValue(10)
            form.addRow("Rotation degrees:", self.rotation_degrees_spin)

            # Random brightness
            self.brightness_check = QtWidgets.QCheckBox()
            form.addRow("Random brightness:", self.brightness_check)

            # Brightness factor
            self.brightness_factor_spin = QtWidgets.QDoubleSpinBox()
            self.brightness_factor_spin.setRange(0.1, 2.0)
            self.brightness_factor_spin.setValue(0.2)
            self.brightness_factor_spin.setSingleStep(0.1)
            form.addRow("Brightness factor:", self.brightness_factor_spin)

            # Random contrast
            self.contrast_check = QtWidgets.QCheckBox()
            form.addRow("Random contrast:", self.contrast_check)

            # Contrast factor
            self.contrast_factor_spin = QtWidgets.QDoubleSpinBox()
            self.contrast_factor_spin.setRange(0.1, 2.0)
            self.contrast_factor_spin.setValue(0.2)
            self.contrast_factor_spin.setSingleStep(0.1)
            form.addRow("Contrast factor:", self.contrast_factor_spin)

            # Random saturation
            self.saturation_check = QtWidgets.QCheckBox()
            form.addRow("Random saturation:", self.saturation_check)

            # Saturation factor
            self.saturation_factor_spin = QtWidgets.QDoubleSpinBox()
            self.saturation_factor_spin.setRange(0.1, 2.0)
            self.saturation_factor_spin.setValue(0.1)
            self.saturation_factor_spin.setSingleStep(0.1)
            form.addRow("Saturation factor:", self.saturation_factor_spin)

            # Random hue
            self.hue_check = QtWidgets.QCheckBox()
            form.addRow("Random hue:", self.hue_check)

            # Hue factor
            self.hue_factor_spin = QtWidgets.QDoubleSpinBox()
            self.hue_factor_spin.setRange(0.0, 0.5)
            self.hue_factor_spin.setValue(0.05)
            self.hue_factor_spin.setSingleStep(0.05)
            form.addRow("Hue factor:", self.hue_factor_spin)

            # Random erasing
            self.random_erasing_check = QtWidgets.QCheckBox()
            form.addRow("Random erasing:", self.random_erasing_check)

            # Erasing probability
            self.erasing_prob_spin = QtWidgets.QDoubleSpinBox()
            self.erasing_prob_spin.setRange(0.0, 1.0)
            self.erasing_prob_spin.setValue(0.5)
            self.erasing_prob_spin.setSingleStep(0.1)
            form.addRow("Erasing probability:", self.erasing_prob_spin)

            # Erasing scale
            self.erasing_scale_spin = QtWidgets.QDoubleSpinBox()
            self.erasing_scale_spin.setRange(0.02, 0.33)
            self.erasing_scale_spin.setValue(0.1)
            self.erasing_scale_spin.setSingleStep(0.01)
            form.addRow("Erasing scale:", self.erasing_scale_spin)

            # Erasing ratio
            self.erasing_ratio_spin = QtWidgets.QDoubleSpinBox()
            self.erasing_ratio_spin.setRange(0.3, 3.3)
            self.erasing_ratio_spin.setValue(0.3)
            self.erasing_ratio_spin.setSingleStep(0.1)
            form.addRow("Erasing ratio:", self.erasing_ratio_spin)

            layout.addLayout(form)
            return tab

        except Exception as e:
            msg = "Błąd podczas tworzenia zakładki"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            raise

    def _create_monitoring_tab(self):
        """Tworzenie zakładki Monitorowanie."""
        try:
            self.logger.debug("Tworzenie zakładki Monitorowanie")
            tab = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(tab)
            form = QtWidgets.QFormLayout()

            # Early stopping
            self.early_stopping_check = QtWidgets.QCheckBox()
            self.early_stopping_check.setChecked(True)
            form.addRow("Early stopping:", self.early_stopping_check)

            # Patience
            self.early_stopping_patience_spin = QtWidgets.QSpinBox()
            self.early_stopping_patience_spin.setRange(1, 100)
            self.early_stopping_patience_spin.setValue(10)
            form.addRow("Patience:", self.early_stopping_patience_spin)

            # Min delta
            self.early_stopping_delta_spin = QtWidgets.QDoubleSpinBox()
            self.early_stopping_delta_spin.setRange(0.0, 1.0)
            self.early_stopping_delta_spin.setValue(0.001)
            self.early_stopping_delta_spin.setDecimals(6)
            form.addRow("Min delta:", self.early_stopping_delta_spin)

            # Monitorowanie metryk
            metrics_group = QtWidgets.QGroupBox("Metryki do monitorowania")
            metrics_layout = QtWidgets.QVBoxLayout()

            # Accuracy
            self.monitor_accuracy_check = QtWidgets.QCheckBox("Accuracy")
            self.monitor_accuracy_check.setChecked(True)
            metrics_layout.addWidget(self.monitor_accuracy_check)

            # Precision
            self.monitor_precision_check = QtWidgets.QCheckBox("Precision")
            self.monitor_precision_check.setChecked(True)
            metrics_layout.addWidget(self.monitor_precision_check)

            # Recall
            self.monitor_recall_check = QtWidgets.QCheckBox("Recall")
            self.monitor_recall_check.setChecked(True)
            metrics_layout.addWidget(self.monitor_recall_check)

            # F1-score
            self.monitor_f1_check = QtWidgets.QCheckBox("F1-score")
            self.monitor_f1_check.setChecked(True)
            metrics_layout.addWidget(self.monitor_f1_check)

            # Confusion matrix
            self.monitor_confusion_matrix_check = QtWidgets.QCheckBox(
                "Confusion matrix"
            )
            self.monitor_confusion_matrix_check.setChecked(True)
            metrics_layout.addWidget(self.monitor_confusion_matrix_check)

            # ROC curve
            self.monitor_roc_check = QtWidgets.QCheckBox("ROC curve")
            self.monitor_roc_check.setChecked(True)
            metrics_layout.addWidget(self.monitor_roc_check)

            # PR curve
            self.monitor_pr_check = QtWidgets.QCheckBox("PR curve")
            self.monitor_pr_check.setChecked(True)
            metrics_layout.addWidget(self.monitor_pr_check)

            metrics_group.setLayout(metrics_layout)
            form.addRow(metrics_group)

            # Zapisywanie modelu
            save_group = QtWidgets.QGroupBox("Zapisywanie modelu")
            save_layout = QtWidgets.QFormLayout()

            # Save best only
            self.save_best_only_check = QtWidgets.QCheckBox()
            self.save_best_only_check.setChecked(True)
            save_layout.addRow("Zapisz tylko najlepszy:", self.save_best_only_check)

            # Save frequency
            self.save_frequency_spin = QtWidgets.QSpinBox()
            self.save_frequency_spin.setRange(1, 100)
            self.save_frequency_spin.setValue(1)
            save_layout.addRow("Częstotliwość zapisu:", self.save_frequency_spin)

            # Save format
            self.save_format_combo = QtWidgets.QComboBox()
            self.save_format_combo.addItems(["pth", "onnx", "torchscript"])
            save_layout.addRow("Format zapisu:", self.save_format_combo)

            save_group.setLayout(save_layout)
            form.addRow(save_group)

            layout.addLayout(form)
            return tab

        except Exception as e:
            msg = "Błąd podczas tworzenia zakładki"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            raise

    def _create_advanced_tab(self):
        """Tworzenie zakładki Zaawansowane."""
        try:
            self.logger.debug("Tworzenie zakładki Zaawansowane")
            tab = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(tab)
            form = QtWidgets.QFormLayout()

            # Mixed precision training
            self.mixed_precision_check = QtWidgets.QCheckBox()
            self.mixed_precision_check.setChecked(True)
            form.addRow("Mixed precision training:", self.mixed_precision_check)

            # Gradient accumulation
            self.gradient_accumulation_spin = QtWidgets.QSpinBox()
            self.gradient_accumulation_spin.setRange(1, 100)
            self.gradient_accumulation_spin.setValue(2)
            form.addRow("Gradient accumulation:", self.gradient_accumulation_spin)

            # Number of workers
            self.num_workers_spin = QtWidgets.QSpinBox()
            self.num_workers_spin.setRange(0, 32)
            self.num_workers_spin.setValue(14)
            form.addRow("Number of workers:", self.num_workers_spin)

            # Pin memory
            self.pin_memory_check = QtWidgets.QCheckBox()
            self.pin_memory_check.setChecked(True)
            form.addRow("Pin memory:", self.pin_memory_check)

            # Prefetch factor
            self.prefetch_factor_spin = QtWidgets.QSpinBox()
            self.prefetch_factor_spin.setRange(1, 10)
            self.prefetch_factor_spin.setValue(2)
            form.addRow("Prefetch factor:", self.prefetch_factor_spin)

            # Persistent workers
            self.persistent_workers_check = QtWidgets.QCheckBox()
            self.persistent_workers_check.setChecked(True)
            form.addRow("Persistent workers:", self.persistent_workers_check)

            # Device
            self.device_combo = QtWidgets.QComboBox()
            self.device_combo.addItems(["cuda", "cpu"])
            form.addRow("Device:", self.device_combo)

            # Seed
            self.seed_spin = QtWidgets.QSpinBox()
            self.seed_spin.setRange(0, 999999)
            self.seed_spin.setValue(42)
            form.addRow("Seed:", self.seed_spin)

            # Deterministic
            self.deterministic_check = QtWidgets.QCheckBox()
            form.addRow("Deterministic:", self.deterministic_check)

            # Benchmark
            self.benchmark_check = QtWidgets.QCheckBox()
            self.benchmark_check.setChecked(True)
            form.addRow("Benchmark:", self.benchmark_check)

            # Logging
            logging_group = QtWidgets.QGroupBox("Logging")
            logging_layout = QtWidgets.QFormLayout()

            # Log level
            self.log_level_combo = QtWidgets.QComboBox()
            self.log_level_combo.addItems(
                ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            )
            self.log_level_combo.setCurrentText("INFO")
            logging_layout.addRow("Log level:", self.log_level_combo)

            # Log interval
            self.log_interval_spin = QtWidgets.QSpinBox()
            self.log_interval_spin.setRange(1, 1000)
            self.log_interval_spin.setValue(10)
            logging_layout.addRow("Log interval:", self.log_interval_spin)

            # Tensorboard
            self.tensorboard_check = QtWidgets.QCheckBox()
            self.tensorboard_check.setChecked(True)
            logging_layout.addRow("Tensorboard:", self.tensorboard_check)

            # Wandb
            self.wandb_check = QtWidgets.QCheckBox()
            logging_layout.addRow("Weights & Biases:", self.wandb_check)

            logging_group.setLayout(logging_layout)
            form.addRow(logging_group)

            layout.addLayout(form)
            return tab

        except Exception as e:
            msg = "Błąd podczas tworzenia zakładki"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            raise

    def _select_base_model(self):
        """Wybór modelu bazowego."""
        try:
            file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Wybierz model bazowy",
                "",
                "Model files (*.pth *.pt *.onnx);;All files (*.*)",
            )
            if file_name:
                self.base_model_edit.setText(file_name)
                self.logger.debug(f"Wybrano model bazowy: {file_name}")
        except Exception as e:
            msg = "Błąd podczas wyboru modelu bazowego"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            QtWidgets.QMessageBox.critical(self, "Błąd", f"{msg}: {str(e)}")

    def _select_train_dir(self):
        """Wybór katalogu treningowego."""
        try:
            dir_name = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Wybierz katalog treningowy", ""
            )
            if dir_name:
                self.train_dir_edit.setText(dir_name)
                self.logger.debug(f"Wybrano katalog treningowy: {dir_name}")
        except Exception as e:
            msg = "Błąd podczas wyboru katalogu treningowego"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            QtWidgets.QMessageBox.critical(self, "Błąd", f"{msg}: {str(e)}")

    def _select_val_dir(self):
        """Wybór katalogu walidacyjnego."""
        try:
            dir_name = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Wybierz katalog walidacyjny", ""
            )
            if dir_name:
                self.val_dir_edit.setText(dir_name)
                self.logger.debug(f"Wybrano katalog walidacyjny: {dir_name}")
        except Exception as e:
            msg = "Błąd podczas wyboru katalogu walidacyjnego"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            QtWidgets.QMessageBox.critical(self, "Błąd", f"{msg}: {str(e)}")

    def _on_architecture_changed(self, architecture):
        """Obsługa zmiany architektury modelu."""
        try:
            self._update_variant_combo(architecture)
            self.logger.debug(f"Zmieniono architekturę na: {architecture}")
        except Exception as e:
            msg = "Błąd podczas zmiany architektury"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            QtWidgets.QMessageBox.critical(self, "Błąd", f"{msg}: {str(e)}")

    def _update_variant_combo(self, architecture):
        """Aktualizacja listy wariantów modelu."""
        try:
            self.variant_combo.clear()
            if architecture == "EfficientNet":
                self.variant_combo.addItems(
                    [
                        "efficientnet_b0",
                        "efficientnet_b1",
                        "efficientnet_b2",
                        "efficientnet_b3",
                        "efficientnet_b4",
                        "efficientnet_b5",
                        "efficientnet_b6",
                        "efficientnet_b7",
                    ]
                )
            elif architecture == "ConvNeXt":
                self.variant_combo.addItems(
                    [
                        "convnext_tiny",
                        "convnext_small",
                        "convnext_base",
                        "convnext_large",
                    ]
                )
            self.logger.debug(
                f"Zaktualizowano listę wariantów dla architektury: {architecture}"
            )
        except Exception as e:
            msg = "Błąd podczas aktualizacji listy wariantów"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            QtWidgets.QMessageBox.critical(self, "Błąd", f"{msg}: {str(e)}")

    def _refresh_profile_list(self):
        """Odświeżenie listy profili."""
        try:
            self.profile_list.clear()
            if self.profiles_dir.exists():
                for profile_file in self.profiles_dir.glob("*.json"):
                    self.profile_list.addItem(profile_file.stem)
            self.logger.debug("Odświeżono listę profili")
        except Exception as e:
            msg = "Błąd podczas odświeżania listy profili"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            QtWidgets.QMessageBox.critical(self, "Błąd", f"{msg}: {str(e)}")

    def _on_profile_selected(self, current, previous):
        """Obsługa wyboru profilu."""
        try:
            if current is None:
                return

            profile_path = self.profiles_dir / f"{current.text()}.json"
            if not profile_path.exists():
                return

            with open(profile_path, "r") as f:
                profile_data = json.load(f)

            self.profile_name_edit.setText(profile_data.get("name", ""))
            self.profile_info.setText(profile_data.get("info", ""))
            self.profile_description.setText(profile_data.get("description", ""))
            self.profile_data_required.setText(profile_data.get("data_required", ""))
            self.profile_hardware_required.setText(
                profile_data.get("hardware_required", "")
            )

            self.logger.debug(f"Wybrano profil: {current.text()}")
        except Exception as e:
            msg = "Błąd podczas wyboru profilu"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            QtWidgets.QMessageBox.critical(self, "Błąd", f"{msg}: {str(e)}")

    def _edit_profile(self):
        """Edycja profilu."""
        try:
            current_item = self.profile_list.currentItem()
            if current_item is None:
                QtWidgets.QMessageBox.warning(
                    self, "Ostrzeżenie", "Wybierz profil do edycji"
                )
                return

            profile_name = current_item.text()
            profile_path = self.profiles_dir / f"{profile_name}.json"

            if not profile_path.exists():
                QtWidgets.QMessageBox.warning(
                    self, "Ostrzeżenie", "Nie znaleziono profilu"
                )
                return

            with open(profile_path, "r") as f:
                profile_data = json.load(f)

            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle("Edycja profilu")
            layout = QtWidgets.QFormLayout(dialog)

            info_edit = QtWidgets.QTextEdit()
            info_edit.setText(profile_data.get("info", ""))
            layout.addRow("Info:", info_edit)

            description_edit = QtWidgets.QTextEdit()
            description_edit.setText(profile_data.get("description", ""))
            layout.addRow("Opis:", description_edit)

            data_required_edit = QtWidgets.QTextEdit()
            data_required_edit.setText(profile_data.get("data_required", ""))
            layout.addRow("Wymagane dane:", data_required_edit)

            hardware_required_edit = QtWidgets.QTextEdit()
            hardware_required_edit.setText(profile_data.get("hardware_required", ""))
            layout.addRow("Wymagany sprzęt:", hardware_required_edit)

            buttons = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.StandardButton.Ok
                | QtWidgets.QDialogButtonBox.StandardButton.Cancel
            )
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            layout.addRow(buttons)

            if dialog.exec_() == QtWidgets.QDialog.Accepted:
                profile_data["info"] = info_edit.toPlainText()
                profile_data["description"] = description_edit.toPlainText()
                profile_data["data_required"] = data_required_edit.toPlainText()
                profile_data["hardware_required"] = hardware_required_edit.toPlainText()

                with open(profile_path, "w") as f:
                    json.dump(profile_data, f, indent=4)

                self._on_profile_selected(current_item, None)
                self.logger.debug(f"Zedytowano profil: {profile_name}")

        except Exception as e:
            msg = "Błąd podczas edycji profilu"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            QtWidgets.QMessageBox.critical(self, "Błąd", f"{msg}: {str(e)}")

    def _apply_profile(self):
        """Zastosowanie profilu."""
        try:
            current_item = self.profile_list.currentItem()
            if current_item is None:
                QtWidgets.QMessageBox.warning(
                    self, "Ostrzeżenie", "Wybierz profil do zastosowania"
                )
                return

            profile_name = current_item.text()
            profile_path = self.profiles_dir / f"{profile_name}.json"

            if not profile_path.exists():
                QtWidgets.QMessageBox.warning(
                    self, "Ostrzeżenie", "Nie znaleziono profilu"
                )
                return

            with open(profile_path, "r") as f:
                profile_data = json.load(f)

            # Zastosowanie parametrów z profilu
            config = profile_data.get("config", {})

            # Model i dane
            self.base_model_edit.setText(config.get("base_model", ""))
            self.train_dir_edit.setText(config.get("train_dir", ""))
            self.val_dir_edit.setText(config.get("val_dir", ""))
            self.arch_combo.setCurrentText(config.get("architecture", "EfficientNet"))
            self.variant_combo.setCurrentText(config.get("variant", "efficientnet_b0"))
            self.input_size_spin.setValue(config.get("input_size", 224))
            self.num_classes_spin.setValue(config.get("num_classes", 38))

            # Parametry doszkalania
            self.epochs_spin.setValue(config.get("epochs", 40))
            self.batch_size_spin.setValue(config.get("batch_size", 16))
            self.learning_rate_spin.setValue(config.get("learning_rate", 0.00005))
            self.optimizer_combo.setCurrentText(config.get("optimizer", "AdamW"))
            self.unfreeze_strategy_combo.setCurrentText(
                config.get("unfreeze_strategy", "Odblokuj wszystkie warstwy")
            )
            self.unfreeze_after_epochs_spin.setValue(
                config.get("unfreeze_after_epochs", 5)
            )
            self.unfreeze_layers_spin.setValue(config.get("unfreeze_layers", 3))
            self.scheduler_combo.setCurrentText(config.get("scheduler", "Brak"))

            # Regularizacja
            self.weight_decay_spin.setValue(config.get("weight_decay", 0.001))
            self.dropout_spin.setValue(config.get("dropout", 0.2))
            self.gradient_clip_spin.setValue(config.get("gradient_clip", 0.5))
            self.label_smoothing_spin.setValue(config.get("label_smoothing", 0.05))
            self.mixup_check.setChecked(config.get("mixup", False))
            self.mixup_alpha_spin.setValue(config.get("mixup_alpha", 0.2))
            self.cutmix_check.setChecked(config.get("cutmix", False))
            self.cutmix_alpha_spin.setValue(config.get("cutmix_alpha", 1.0))
            self.stochastic_depth_spin.setValue(config.get("stochastic_depth", 0.1))

            # Augmentacja
            self.horizontal_flip_check.setChecked(config.get("horizontal_flip", True))
            self.vertical_flip_check.setChecked(config.get("vertical_flip", False))
            self.rotation_check.setChecked(config.get("rotation", True))
            self.rotation_degrees_spin.setValue(config.get("rotation_degrees", 10))
            self.brightness_check.setChecked(config.get("brightness", True))
            self.brightness_factor_spin.setValue(config.get("brightness_factor", 0.2))
            self.contrast_check.setChecked(config.get("contrast", True))
            self.contrast_factor_spin.setValue(config.get("contrast_factor", 0.2))
            self.saturation_check.setChecked(config.get("saturation", True))
            self.saturation_factor_spin.setValue(config.get("saturation_factor", 0.1))
            self.hue_check.setChecked(config.get("hue", True))
            self.hue_factor_spin.setValue(config.get("hue_factor", 0.05))
            self.random_erasing_check.setChecked(config.get("random_erasing", True))
            self.erasing_prob_spin.setValue(config.get("erasing_prob", 0.5))
            self.erasing_scale_spin.setValue(config.get("erasing_scale", 0.1))
            self.erasing_ratio_spin.setValue(config.get("erasing_ratio", 0.3))

            # Monitorowanie
            self.early_stopping_check.setChecked(config.get("early_stopping", True))
            self.early_stopping_patience_spin.setValue(
                config.get("early_stopping_patience", 10)
            )
            self.early_stopping_delta_spin.setValue(
                config.get("early_stopping_delta", 0.001)
            )
            self.monitor_accuracy_check.setChecked(config.get("monitor_accuracy", True))
            self.monitor_precision_check.setChecked(
                config.get("monitor_precision", True)
            )
            self.monitor_recall_check.setChecked(config.get("monitor_recall", True))
            self.monitor_f1_check.setChecked(config.get("monitor_f1", True))
            self.monitor_confusion_matrix_check.setChecked(
                config.get("monitor_confusion_matrix", True)
            )
            self.monitor_roc_check.setChecked(config.get("monitor_roc", True))
            self.monitor_pr_check.setChecked(config.get("monitor_pr", True))
            self.save_best_only_check.setChecked(config.get("save_best_only", True))
            self.save_frequency_spin.setValue(config.get("save_frequency", 1))
            self.save_format_combo.setCurrentText(config.get("save_format", "pth"))

            # Zaawansowane
            self.mixed_precision_check.setChecked(config.get("mixed_precision", True))
            self.gradient_accumulation_spin.setValue(
                config.get("gradient_accumulation", 2)
            )
            self.num_workers_spin.setValue(config.get("num_workers", 14))
            self.pin_memory_check.setChecked(config.get("pin_memory", True))
            self.prefetch_factor_spin.setValue(config.get("prefetch_factor", 2))
            self.persistent_workers_check.setChecked(
                config.get("persistent_workers", True)
            )
            self.device_combo.setCurrentText(config.get("device", "cuda"))
            self.seed_spin.setValue(config.get("seed", 42))
            self.deterministic_check.setChecked(config.get("deterministic", False))
            self.benchmark_check.setChecked(config.get("benchmark", True))
            self.log_level_combo.setCurrentText(config.get("log_level", "INFO"))
            self.log_interval_spin.setValue(config.get("log_interval", 10))
            self.tensorboard_check.setChecked(config.get("tensorboard", True))
            self.wandb_check.setChecked(config.get("wandb", False))

            self.logger.debug(f"Zastosowano profil: {profile_name}")

        except Exception as e:
            msg = "Błąd podczas stosowania profilu"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            QtWidgets.QMessageBox.critical(self, "Błąd", f"{msg}: {str(e)}")

    def _clone_profile(self):
        """Klonowanie profilu."""
        try:
            current_item = self.profile_list.currentItem()
            if current_item is None:
                QtWidgets.QMessageBox.warning(
                    self, "Ostrzeżenie", "Wybierz profil do sklonowania"
                )
                return

            profile_name = current_item.text()
            profile_path = self.profiles_dir / f"{profile_name}.json"

            if not profile_path.exists():
                QtWidgets.QMessageBox.warning(
                    self, "Ostrzeżenie", "Nie znaleziono profilu"
                )
                return

            new_name, ok = QtWidgets.QInputDialog.getText(
                self,
                "Klonowanie profilu",
                "Podaj nazwę nowego profilu:",
                QtWidgets.QLineEdit.EchoMode.Normal,
                f"{profile_name}_copy",
            )

            if ok and new_name:
                new_path = self.profiles_dir / f"{new_name}.json"
                if new_path.exists():
                    QtWidgets.QMessageBox.warning(
                        self, "Ostrzeżenie", "Profil o takiej nazwie już istnieje"
                    )
                    return

                with open(profile_path, "r") as f:
                    profile_data = json.load(f)

                profile_data["name"] = new_name
                profile_data["created_at"] = datetime.now().isoformat()
                profile_data["modified_at"] = datetime.now().isoformat()

                with open(new_path, "w") as f:
                    json.dump(profile_data, f, indent=4)

                self._refresh_profile_list()
                self.logger.debug(f"Sklonowano profil {profile_name} jako {new_name}")

        except Exception as e:
            msg = "Błąd podczas klonowania profilu"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            QtWidgets.QMessageBox.critical(self, "Błąd", f"{msg}: {str(e)}")

    def _save_profile(self):
        """Zapisanie aktualnej konfiguracji jako profil."""
        try:
            name, ok = QtWidgets.QInputDialog.getText(
                self,
                "Zapisywanie profilu",
                "Podaj nazwę profilu:",
                QtWidgets.QLineEdit.EchoMode.Normal,
            )

            if ok and name:
                profile_path = self.profiles_dir / f"{name}.json"
                if profile_path.exists():
                    reply = QtWidgets.QMessageBox.question(
                        self,
                        "Potwierdzenie",
                        "Profil o takiej nazwie już istnieje. Czy chcesz go nadpisać?",
                        QtWidgets.QMessageBox.StandardButton.Yes
                        | QtWidgets.QMessageBox.StandardButton.No,
                    )
                    if reply == QtWidgets.QMessageBox.StandardButton.No:
                        return

                # Zbieranie aktualnej konfiguracji
                config = {
                    # Model i dane
                    "base_model": self.base_model_edit.text(),
                    "train_dir": self.train_dir_edit.text(),
                    "val_dir": self.val_dir_edit.text(),
                    "architecture": self.arch_combo.currentText(),
                    "variant": self.variant_combo.currentText(),
                    "input_size": self.input_size_spin.value(),
                    "num_classes": self.num_classes_spin.value(),
                    # Parametry doszkalania
                    "epochs": self.epochs_spin.value(),
                    "batch_size": self.batch_size_spin.value(),
                    "learning_rate": self.learning_rate_spin.value(),
                    "optimizer": self.optimizer_combo.currentText(),
                    "unfreeze_strategy": self.unfreeze_strategy_combo.currentText(),
                    "unfreeze_after_epochs": self.unfreeze_after_epochs_spin.value(),
                    "unfreeze_layers": self.unfreeze_layers_spin.value(),
                    "scheduler": self.scheduler_combo.currentText(),
                    # Regularizacja
                    "weight_decay": self.weight_decay_spin.value(),
                    "dropout": self.dropout_spin.value(),
                    "gradient_clip": self.gradient_clip_spin.value(),
                    "label_smoothing": self.label_smoothing_spin.value(),
                    "mixup": self.mixup_check.isChecked(),
                    "mixup_alpha": self.mixup_alpha_spin.value(),
                    "cutmix": self.cutmix_check.isChecked(),
                    "cutmix_alpha": self.cutmix_alpha_spin.value(),
                    "stochastic_depth": self.stochastic_depth_spin.value(),
                    # Augmentacja
                    "horizontal_flip": self.horizontal_flip_check.isChecked(),
                    "vertical_flip": self.vertical_flip_check.isChecked(),
                    "rotation": self.rotation_check.isChecked(),
                    "rotation_degrees": self.rotation_degrees_spin.value(),
                    "brightness": self.brightness_check.isChecked(),
                    "brightness_factor": self.brightness_factor_spin.value(),
                    "contrast": self.contrast_check.isChecked(),
                    "contrast_factor": self.contrast_factor_spin.value(),
                    "saturation": self.saturation_check.isChecked(),
                    "saturation_factor": self.saturation_factor_spin.value(),
                    "hue": self.hue_check.isChecked(),
                    "hue_factor": self.hue_factor_spin.value(),
                    "random_erasing": self.random_erasing_check.isChecked(),
                    "erasing_prob": self.erasing_prob_spin.value(),
                    "erasing_scale": self.erasing_scale_spin.value(),
                    "erasing_ratio": self.erasing_ratio_spin.value(),
                    # Monitorowanie
                    "early_stopping": self.early_stopping_check.isChecked(),
                    "early_stopping_patience": self.early_stopping_patience_spin.value(),
                    "early_stopping_delta": self.early_stopping_delta_spin.value(),
                    "monitor_accuracy": self.monitor_accuracy_check.isChecked(),
                    "monitor_precision": self.monitor_precision_check.isChecked(),
                    "monitor_recall": self.monitor_recall_check.isChecked(),
                    "monitor_f1": self.monitor_f1_check.isChecked(),
                    "monitor_confusion_matrix": self.monitor_confusion_matrix_check.isChecked(),
                    "monitor_roc": self.monitor_roc_check.isChecked(),
                    "monitor_pr": self.monitor_pr_check.isChecked(),
                    "save_best_only": self.save_best_only_check.isChecked(),
                    "save_frequency": self.save_frequency_spin.value(),
                    "save_format": self.save_format_combo.currentText(),
                    # Zaawansowane
                    "mixed_precision": self.mixed_precision_check.isChecked(),
                    "gradient_accumulation": self.gradient_accumulation_spin.value(),
                    "num_workers": self.num_workers_spin.value(),
                    "pin_memory": self.pin_memory_check.isChecked(),
                    "prefetch_factor": self.prefetch_factor_spin.value(),
                    "persistent_workers": self.persistent_workers_check.isChecked(),
                    "device": self.device_combo.currentText(),
                    "seed": self.seed_spin.value(),
                    "deterministic": self.deterministic_check.isChecked(),
                    "benchmark": self.benchmark_check.isChecked(),
                    "log_level": self.log_level_combo.currentText(),
                    "log_interval": self.log_interval_spin.value(),
                    "tensorboard": self.tensorboard_check.isChecked(),
                    "wandb": self.wandb_check.isChecked(),
                }

                profile_data = {
                    "name": name,
                    "created_at": datetime.now().isoformat(),
                    "modified_at": datetime.now().isoformat(),
                    "config": config,
                }

                with open(profile_path, "w") as f:
                    json.dump(profile_data, f, indent=4)

                self._refresh_profile_list()
                self.logger.debug(f"Zapisano profil: {name}")

        except Exception as e:
            msg = "Błąd podczas zapisywania profilu"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            QtWidgets.QMessageBox.critical(self, "Błąd", f"{msg}: {str(e)}")

    def _validate_parameters(self):
        """Walidacja parametrów zgodnie z dokumentacją."""
        try:
            # Walidacja podstawowych parametrów
            if self.input_size_spin.value() != 224:
                QtWidgets.QMessageBox.warning(
                    self, "Ostrzeżenie", "Zalecany rozmiar wejścia to 224x224 pikseli"
                )

            if self.num_classes_spin.value() != 38:
                QtWidgets.QMessageBox.warning(
                    self, "Ostrzeżenie", "Zalecana liczba klas to 38"
                )

            # Walidacja parametrów treningu
            if self.epochs_spin.value() != 40:
                QtWidgets.QMessageBox.warning(
                    self, "Ostrzeżenie", "Zalecana liczba epok to 40"
                )

            if self.batch_size_spin.value() != 16:
                QtWidgets.QMessageBox.warning(
                    self, "Ostrzeżenie", "Zalecany rozmiar batcha to 16"
                )

            if abs(self.learning_rate_spin.value() - 0.00005) > 0.000001:
                QtWidgets.QMessageBox.warning(
                    self, "Ostrzeżenie", "Zalecany learning rate to 0.00005"
                )

            if self.optimizer_combo.currentText() != "AdamW":
                QtWidgets.QMessageBox.warning(
                    self, "Ostrzeżenie", "Zalecany optymalizator to AdamW"
                )

            # Walidacja parametrów regularyzacji
            if abs(self.weight_decay_spin.value() - 0.001) > 0.000001:
                QtWidgets.QMessageBox.warning(
                    self, "Ostrzeżenie", "Zalecany weight decay to 0.001"
                )

            if abs(self.dropout_spin.value() - 0.2) > 0.000001:
                QtWidgets.QMessageBox.warning(
                    self, "Ostrzeżenie", "Zalecany dropout to 0.2"
                )

            if abs(self.gradient_clip_spin.value() - 0.5) > 0.000001:
                QtWidgets.QMessageBox.warning(
                    self, "Ostrzeżenie", "Zalecany gradient clip to 0.5"
                )

            if abs(self.label_smoothing_spin.value() - 0.05) > 0.000001:
                QtWidgets.QMessageBox.warning(
                    self, "Ostrzeżenie", "Zalecane label smoothing to 0.05"
                )

            # Walidacja parametrów augmentacji
            if self.rotation_degrees_spin.value() != 10:
                QtWidgets.QMessageBox.warning(
                    self, "Ostrzeżenie", "Zalecana rotacja to 10 stopni"
                )

            if abs(self.brightness_factor_spin.value() - 0.2) > 0.000001:
                QtWidgets.QMessageBox.warning(
                    self, "Ostrzeżenie", "Zalecany współczynnik jasności to 0.2"
                )

            if abs(self.contrast_factor_spin.value() - 0.2) > 0.000001:
                QtWidgets.QMessageBox.warning(
                    self, "Ostrzeżenie", "Zalecany współczynnik kontrastu to 0.2"
                )

            if abs(self.saturation_factor_spin.value() - 0.1) > 0.000001:
                QtWidgets.QMessageBox.warning(
                    self, "Ostrzeżenie", "Zalecany współczynnik nasycenia to 0.1"
                )

            if abs(self.hue_factor_spin.value() - 0.05) > 0.000001:
                QtWidgets.QMessageBox.warning(
                    self, "Ostrzeżenie", "Zalecany współczynnik odcienia to 0.05"
                )

            # Walidacja parametrów zaawansowanych
            if self.gradient_accumulation_spin.value() != 2:
                QtWidgets.QMessageBox.warning(
                    self, "Ostrzeżenie", "Zalecana akumulacja gradientu to 2"
                )

            if self.num_workers_spin.value() != 14:
                QtWidgets.QMessageBox.warning(
                    self, "Ostrzeżenie", "Zalecana liczba workerów to 14"
                )

            return True

        except Exception as e:
            msg = "Błąd podczas walidacji parametrów"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            QtWidgets.QMessageBox.critical(self, "Błąd", f"{msg}: {str(e)}")
            return False

    def accept(self):
        """Obsługa przycisku OK."""
        try:
            # Sprawdzenie wymaganych pól
            if not self.base_model_edit.text():
                QtWidgets.QMessageBox.warning(
                    self, "Ostrzeżenie", "Wybierz model bazowy"
                )
                return

            if not self.train_dir_edit.text():
                QtWidgets.QMessageBox.warning(
                    self, "Ostrzeżenie", "Wybierz katalog treningowy"
                )
                return

            if not self.val_dir_edit.text():
                QtWidgets.QMessageBox.warning(
                    self, "Ostrzeżenie", "Wybierz katalog walidacyjny"
                )
                return

            # Walidacja parametrów
            if not self._validate_parameters():
                return

            # Zbieranie konfiguracji
            config = {
                # Model i dane
                "base_model": self.base_model_edit.text(),
                "train_dir": self.train_dir_edit.text(),
                "val_dir": self.val_dir_edit.text(),
                "architecture": self.arch_combo.currentText(),
                "variant": self.variant_combo.currentText(),
                "input_size": self.input_size_spin.value(),
                "num_classes": self.num_classes_spin.value(),
                # Parametry doszkalania
                "epochs": self.epochs_spin.value(),
                "batch_size": self.batch_size_spin.value(),
                "learning_rate": self.learning_rate_spin.value(),
                "optimizer": self.optimizer_combo.currentText(),
                "unfreeze_strategy": self.unfreeze_strategy_combo.currentText(),
                "unfreeze_after_epochs": self.unfreeze_after_epochs_spin.value(),
                "unfreeze_layers": self.unfreeze_layers_spin.value(),
                "scheduler": self.scheduler_combo.currentText(),
                # Regularizacja
                "weight_decay": self.weight_decay_spin.value(),
                "dropout": self.dropout_spin.value(),
                "gradient_clip": self.gradient_clip_spin.value(),
                "label_smoothing": self.label_smoothing_spin.value(),
                "mixup": self.mixup_check.isChecked(),
                "mixup_alpha": self.mixup_alpha_spin.value(),
                "cutmix": self.cutmix_check.isChecked(),
                "cutmix_alpha": self.cutmix_alpha_spin.value(),
                "stochastic_depth": self.stochastic_depth_spin.value(),
                # Augmentacja
                "horizontal_flip": self.horizontal_flip_check.isChecked(),
                "vertical_flip": self.vertical_flip_check.isChecked(),
                "rotation": self.rotation_check.isChecked(),
                "rotation_degrees": self.rotation_degrees_spin.value(),
                "brightness": self.brightness_check.isChecked(),
                "brightness_factor": self.brightness_factor_spin.value(),
                "contrast": self.contrast_check.isChecked(),
                "contrast_factor": self.contrast_factor_spin.value(),
                "saturation": self.saturation_check.isChecked(),
                "saturation_factor": self.saturation_factor_spin.value(),
                "hue": self.hue_check.isChecked(),
                "hue_factor": self.hue_factor_spin.value(),
                "random_erasing": self.random_erasing_check.isChecked(),
                "erasing_prob": self.erasing_prob_spin.value(),
                "erasing_scale": self.erasing_scale_spin.value(),
                "erasing_ratio": self.erasing_ratio_spin.value(),
                # Monitorowanie
                "early_stopping": self.early_stopping_check.isChecked(),
                "early_stopping_patience": self.early_stopping_patience_spin.value(),
                "early_stopping_delta": self.early_stopping_delta_spin.value(),
                "monitor_accuracy": self.monitor_accuracy_check.isChecked(),
                "monitor_precision": self.monitor_precision_check.isChecked(),
                "monitor_recall": self.monitor_recall_check.isChecked(),
                "monitor_f1": self.monitor_f1_check.isChecked(),
                "monitor_confusion_matrix": self.monitor_confusion_matrix_check.isChecked(),
                "monitor_roc": self.monitor_roc_check.isChecked(),
                "monitor_pr": self.monitor_pr_check.isChecked(),
                "save_best_only": self.save_best_only_check.isChecked(),
                "save_frequency": self.save_frequency_spin.value(),
                "save_format": self.save_format_combo.currentText(),
                # Zaawansowane
                "mixed_precision": self.mixed_precision_check.isChecked(),
                "gradient_accumulation": self.gradient_accumulation_spin.value(),
                "num_workers": self.num_workers_spin.value(),
                "pin_memory": self.pin_memory_check.isChecked(),
                "prefetch_factor": self.prefetch_factor_spin.value(),
                "persistent_workers": self.persistent_workers_check.isChecked(),
                "device": self.device_combo.currentText(),
                "seed": self.seed_spin.value(),
                "deterministic": self.deterministic_check.isChecked(),
                "benchmark": self.benchmark_check.isChecked(),
                "log_level": self.log_level_combo.currentText(),
                "log_interval": self.log_interval_spin.value(),
                "tensorboard": self.tensorboard_check.isChecked(),
                "wandb": self.wandb_check.isChecked(),
            }

            self.config = config
            super().accept()
            self.logger.debug("Zatwierdzono konfigurację")

        except Exception as e:
            msg = "Błąd podczas zatwierdzania konfiguracji"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            QtWidgets.QMessageBox.critical(self, "Błąd", f"{msg}: {str(e)}")

    def reject(self):
        """Obsługa przycisku Anuluj."""
        try:
            super().reject()
            self.logger.debug("Anulowano konfigurację")
        except Exception as e:
            msg = "Błąd podczas anulowania konfiguracji"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            QtWidgets.QMessageBox.critical(self, "Błąd", f"{msg}: {str(e)}")
