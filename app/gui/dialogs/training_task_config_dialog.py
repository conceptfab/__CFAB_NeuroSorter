import datetime
import json
import logging
import os
from pathlib import Path

from PyQt6 import QtCore, QtWidgets

from app.gui.dialogs.hardware_profile_dialog import HardwareProfileDialog
from app.utils.config import (
    DEFAULT_TRAINING_PARAMS,
)  # Assuming this exists and has relevant keys
from app.utils.file_utils import (
    validate_training_directory,
    validate_validation_directory,
)


class TrainingTaskConfigDialog(QtWidgets.QDialog):
    """Dialog konfiguracji zadania treningu."""

    # Strategie odmrażania warstw
    UNFREEZE_ALL = "unfreeze_all"
    UNFREEZE_GRADUAL_END = "unfreeze_gradual_end"
    UNFREEZE_GRADUAL_START = "unfreeze_gradual_start"
    UNFREEZE_AFTER_EPOCHS = "unfreeze_after_epochs"  # Corrected typo

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
        self.setMinimumWidth(
            1000
        )  # Consider increasing if many new fields make it crowded
        self.profiles_dir = Path("data/profiles")
        self.profiles_dir.mkdir(exist_ok=True)
        self.current_profile = None
        self.setWindowFlags(
            self.windowFlags() | QtCore.Qt.WindowType.WindowCloseButtonHint
        )
        self.parameter_rows = {}  # Initialize here
        self._init_ui()

    def _setup_logging(self):
        """Konfiguracja logowania dla okna dialogowego."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        fh = logging.FileHandler("training_dialog.log")
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(log_fmt)
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        if not self.logger.hasHandlers():  # Avoid duplicate handlers on re-init
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

        self.logger.info("Inicjalizacja okna")

    def _init_ui(self):
        """Inicjalizacja interfejsu użytkownika z zakładkami."""
        try:
            self.logger.debug("Rozpoczęcie inicjalizacji UI")
            layout = QtWidgets.QVBoxLayout(self)

            toolbar = QtWidgets.QToolBar("Narzędzia")
            if hasattr(self, "addToolBar"):  # QDialog doesn't have addToolBar directly
                # For QMainWindow, this is fine. For QDialog, we add it to layout
                pass  # Toolbar not directly added to QDialog like this.
                # If needed, add buttons directly or use a QMainWindow.
                # For now, let's keep show_hw_profile_btn as a direct button.

            show_hw_profile_btn = QtWidgets.QPushButton("Pokaż profil sprzętowy")
            show_hw_profile_btn.clicked.connect(self._show_hardware_profile)
            layout.addWidget(show_hw_profile_btn)

            self.tabs = QtWidgets.QTabWidget()

            tab_data_model = self._create_data_model_tab()
            self.tabs.addTab(tab_data_model, "Dane i Model")

            tab_training_params = self._create_training_params_tab()
            self.tabs.addTab(tab_training_params, "Parametry")

            tab_regularization = self._create_regularization_tab()
            self.tabs.addTab(tab_regularization, "Regularyzacja")

            tab_augmentation = self._create_augmentation_tab()
            self.tabs.addTab(tab_augmentation, "Augmentacja")

            tab_preprocessing = self._create_preprocessing_tab()
            self.tabs.addTab(tab_preprocessing, "Preprocessing")

            tab_monitoring = self._create_monitoring_tab()
            self.tabs.addTab(tab_monitoring, "Monitorowanie")

            tab_advanced = self._create_advanced_tab()
            self.tabs.addTab(tab_advanced, "Zaawansowane")

            tab_optimization = self._create_optimization_tab()
            self.tabs.addTab(tab_optimization, "Optymalizacja treningu")

            layout.addWidget(self.tabs)

            buttons_layout = QtWidgets.QHBoxLayout()
            add_task_btn = QtWidgets.QPushButton("Dodaj zadanie")
            add_task_btn.clicked.connect(self._on_accept)
            buttons_layout.addWidget(add_task_btn)

            close_btn = QtWidgets.QPushButton("Zamknij")
            close_btn.clicked.connect(
                self.reject
            )  # Changed to reject for typical dialog behavior
            buttons_layout.addWidget(close_btn)

            layout.addLayout(buttons_layout)
            self.logger.debug("Zakończono inicjalizację UI")

        except Exception as e:
            msg = "Błąd podczas inicjalizacji UI"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            QtWidgets.QMessageBox.critical(self, "Błąd UI", f"{msg}: {str(e)}")

    def _create_data_model_tab(self):
        """Tworzenie zakładki Dane i Model."""
        try:
            self.logger.debug("Tworzenie zakładki Dane i Model")
            tab = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(tab)
            form = QtWidgets.QFormLayout()

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
            self.arch_combo.addItems(
                ["EfficientNet", "ConvNeXt", "ResNet", "ViT"]
            )  # Added more
            form.addRow("Architektura:", self.arch_combo)

            # Wariant modelu
            self.variant_combo = QtWidgets.QComboBox()
            self._update_variant_combo(
                self.arch_combo.currentText()
            )  # Init with current arch
            form.addRow("Wariant:", self.variant_combo)
            self.arch_combo.currentTextChanged.connect(self._on_architecture_changed)

            # Rozmiar obrazu wejściowego
            self.input_size_spin = QtWidgets.QSpinBox()
            self.input_size_spin.setRange(32, 2048)  # Increased max
            self.input_size_spin.setValue(224)
            self.input_size_spin.setSingleStep(32)
            form.addRow("Rozmiar obrazu:", self.input_size_spin)

            # Liczba klas
            self.num_classes_spin = QtWidgets.QSpinBox()
            self.num_classes_spin.setRange(
                1, 10000
            )  # Increased max, min to 1 for some tasks
            self.num_classes_spin.setValue(2)
            form.addRow("Liczba klas:", self.num_classes_spin)

            # Pretrained
            self.pretrained_check = QtWidgets.QCheckBox("Użyj wag pretrenowanych")
            self.pretrained_check.setChecked(True)
            form.addRow(self.pretrained_check)

            # Pretrained weights source
            self.pretrained_weights_combo = QtWidgets.QComboBox()
            self.pretrained_weights_combo.addItems(
                ["imagenet", "ImageNet-21k", "custom"]
            )
            # TODO: Add QLineEdit for custom path if "custom" is selected
            form.addRow("Źródło wag pretrenowanych:", self.pretrained_weights_combo)

            # Feature extraction only
            self.feature_extraction_only_check = QtWidgets.QCheckBox(
                "Tylko ekstrakcja cech (zamrożony szkielet)"
            )
            form.addRow(self.feature_extraction_only_check)

            # Activation
            self.activation_combo = QtWidgets.QComboBox()
            self.activation_combo.addItems(["swish", "relu", "gelu", "sigmoid", "mish"])
            self.activation_combo.setCurrentText("swish")
            form.addRow("Funkcja aktywacji w szkielecie:", self.activation_combo)

            # Dropout at inference
            self.dropout_at_inference_check = QtWidgets.QCheckBox(
                "Użyj dropoutu podczas inferencji (MC Dropout)"
            )
            form.addRow(self.dropout_at_inference_check)

            # Global pool
            self.global_pool_combo = QtWidgets.QComboBox()
            self.global_pool_combo.addItems(["avg", "max", "gem"])
            self.global_pool_combo.setCurrentText("avg")
            form.addRow("Typ global pooling:", self.global_pool_combo)

            # Last layer activation
            self.last_layer_activation_combo = QtWidgets.QComboBox()
            self.last_layer_activation_combo.addItems(["softmax", "sigmoid", "none"])
            self.last_layer_activation_combo.setCurrentText(
                "softmax"
            )  # Default for classification
            form.addRow(
                "Aktywacja ostatniej warstwy:", self.last_layer_activation_combo
            )

            layout.addLayout(form)

            # Profile Management Group (as before)
            profile_group = QtWidgets.QGroupBox("Dostępne profile")
            profile_layout = QtWidgets.QVBoxLayout()
            self.profile_list = QtWidgets.QListWidget()
            self.profile_list.currentItemChanged.connect(self._on_profile_selected)
            self._refresh_profile_list()
            profile_layout.addWidget(self.profile_list)
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
            self.profile_hardware_required.setMaximumHeight(60)
            info_layout.addRow("Wymagany sprzęt:", self.profile_hardware_required)
            info_group.setLayout(info_layout)
            profile_layout.addWidget(info_group)
            buttons_layout_profile = (
                QtWidgets.QHBoxLayout()
            )  # Renamed to avoid conflict
            self.edit_profile_btn = QtWidgets.QPushButton("Edytuj profil")
            self.edit_profile_btn.clicked.connect(self._edit_profile)
            buttons_layout_profile.addWidget(self.edit_profile_btn)
            self.apply_profile_btn = QtWidgets.QPushButton("Zastosuj profil")
            self.apply_profile_btn.clicked.connect(self._apply_profile)
            buttons_layout_profile.addWidget(self.apply_profile_btn)
            self.clone_profile_btn = QtWidgets.QPushButton("Klonuj profil")
            self.clone_profile_btn.clicked.connect(self._clone_profile)
            buttons_layout_profile.addWidget(self.clone_profile_btn)
            self.save_profile_btn = QtWidgets.QPushButton("Zapisz profil")
            self.save_profile_btn.clicked.connect(self._save_profile)
            buttons_layout_profile.addWidget(self.save_profile_btn)
            self.delete_profile_btn = QtWidgets.QPushButton("Usuń profil")
            self.delete_profile_btn.clicked.connect(self._delete_profile)
            buttons_layout_profile.addWidget(self.delete_profile_btn)
            profile_layout.addLayout(buttons_layout_profile)
            profile_group.setLayout(profile_layout)
            layout.addWidget(profile_group)

            return tab

        except Exception as e:
            msg = "Błąd podczas tworzenia zakładki Dane i Model"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            QtWidgets.QMessageBox.critical(self, "Błąd UI", f"{msg}: {str(e)}")
            return QtWidgets.QWidget()  # Return empty widget on error

    def _refresh_profile_list(self):
        self.profile_list.clear()
        for profile_file in self.profiles_dir.glob("*.json"):
            try:
                with open(profile_file, "r", encoding="utf-8") as f:
                    profile_data = json.load(f)
                    if profile_data.get("type") == "training":
                        self.profile_list.addItem(profile_file.stem)
            except Exception as e:
                self.logger.error(
                    f"Błąd podczas wczytywania profilu {profile_file}: {str(e)}",
                    exc_info=True,
                )

    def _on_profile_selected(self, current, previous):
        if current is None:
            self.current_profile = None
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
            self.logger.error(
                f"Błąd podczas ładowania profilu: {str(e)}", exc_info=True
            )
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Nie można załadować profilu: {str(e)}"
            )
            self.current_profile = None  # Reset on error

    def _edit_profile(self):
        if not self.current_profile or self.profile_list.currentItem() is None:
            QtWidgets.QMessageBox.warning(
                self, "Ostrzeżenie", "Najpierw wybierz profil do edycji."
            )
            return
        try:
            profile_path = (
                self.profiles_dir / f"{self.profile_list.currentItem().text()}.json"
            )
            # For cross-platform, use QDesktopServices
            url = QtCore.QUrl.fromLocalFile(str(profile_path))
            if not QtWidgets.QDesktopServices.openUrl(url):
                QtWidgets.QMessageBox.warning(
                    self, "Błąd", f"Nie można otworzyć pliku: {profile_path}"
                )
        except Exception as e:
            self.logger.error(
                f"Błąd podczas otwierania profilu: {str(e)}", exc_info=True
            )
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Nie można otworzyć profilu: {str(e)}"
            )

    def _safe_set_spinbox_value(self, spinbox, value, default):
        if value is not None:
            if isinstance(spinbox, QtWidgets.QDoubleSpinBox):
                spinbox.setValue(float(value))
            else:
                spinbox.setValue(int(value))
        else:
            spinbox.setValue(default)

    def _safe_set_checkbox_value(self, checkbox, value, default):
        if value is not None:
            checkbox.setChecked(bool(value))
        else:
            checkbox.setChecked(default)

    def _safe_set_combo_value(self, combobox, value, default_text=None):
        if value is not None:
            idx = combobox.findText(str(value))
            if idx != -1:
                combobox.setCurrentIndex(idx)
            elif default_text:
                idx_default = combobox.findText(str(default_text))
                if idx_default != -1:
                    combobox.setCurrentIndex(idx_default)
            # else: self.logger.warning(f"Value '{value}' not found in ComboBox {combobox.objectName()} and no default_text provided.")
        elif default_text:
            idx_default = combobox.findText(str(default_text))
            if idx_default != -1:
                combobox.setCurrentIndex(idx_default)

    def _apply_profile(self):
        if not self.current_profile:
            QtWidgets.QMessageBox.warning(
                self, "Ostrzeżenie", "Najpierw wybierz profil do zastosowania."
            )
            return

        try:
            config = self.current_profile.get("config", {})

            # Dane i Model
            model_config = config.get("model", {})
            self._safe_set_combo_value(
                self.arch_combo, model_config.get("architecture"), "EfficientNet"
            )
            self._update_variant_combo(
                self.arch_combo.currentText()
            )  # Important after arch_combo change
            self._safe_set_combo_value(self.variant_combo, model_config.get("variant"))
            self._safe_set_spinbox_value(
                self.input_size_spin, model_config.get("input_size"), 224
            )
            self._safe_set_spinbox_value(
                self.num_classes_spin, model_config.get("num_classes"), 2
            )
            self._safe_set_checkbox_value(
                self.pretrained_check, model_config.get("pretrained"), True
            )
            self._safe_set_combo_value(
                self.pretrained_weights_combo,
                model_config.get("pretrained_weights"),
                "imagenet",
            )
            self._safe_set_checkbox_value(
                self.feature_extraction_only_check,
                model_config.get("feature_extraction_only"),
                False,
            )
            self._safe_set_combo_value(
                self.activation_combo, model_config.get("activation"), "swish"
            )
            self._safe_set_checkbox_value(
                self.dropout_at_inference_check,
                model_config.get("dropout_at_inference"),
                False,
            )
            self._safe_set_combo_value(
                self.global_pool_combo, model_config.get("global_pool"), "avg"
            )
            self._safe_set_combo_value(
                self.last_layer_activation_combo,
                model_config.get("last_layer_activation"),
                "softmax",
            )

            # Parametry Treningu
            training_config = config.get("training", {})
            optimization_profile_config = config.get(
                "optimization", {}
            )  # For batch_size etc. if there

            self._safe_set_spinbox_value(
                self.epochs_spin, training_config.get("epochs"), 100
            )
            self._safe_set_spinbox_value(
                self.lr_spin, training_config.get("learning_rate"), 0.001
            )
            self._safe_set_combo_value(
                self.optimizer_combo, training_config.get("optimizer"), "AdamW"
            )

            scheduler_config = training_config.get("scheduler", {})
            if isinstance(scheduler_config, dict):
                self._safe_set_combo_value(
                    self.scheduler_combo, scheduler_config.get("type"), "None"
                )
                self._safe_set_spinbox_value(
                    self.scheduler_T0_spin, scheduler_config.get("T_0"), 10
                )
                self._safe_set_spinbox_value(
                    self.scheduler_Tmult_spin, scheduler_config.get("T_mult"), 1
                )
                self._safe_set_spinbox_value(
                    self.scheduler_eta_min_spin, scheduler_config.get("eta_min"), 1e-6
                )
                # Add for StepLR, OneCycleLR etc.
                self._safe_set_spinbox_value(
                    self.scheduler_step_size_spin, scheduler_config.get("step_size"), 30
                )
                self._safe_set_spinbox_value(
                    self.scheduler_gamma_spin, scheduler_config.get("gamma"), 0.1
                )
                self._safe_set_spinbox_value(
                    self.scheduler_max_lr_spin, scheduler_config.get("max_lr"), 0.01
                )
                self._safe_set_spinbox_value(
                    self.scheduler_pct_start_spin,
                    scheduler_config.get("pct_start"),
                    0.3,
                )

            else:  # old format
                self._safe_set_combo_value(
                    self.scheduler_combo, str(scheduler_config), "None"
                )

            self._safe_set_spinbox_value(
                self.warmup_epochs_spin, training_config.get("warmup_epochs"), 5
            )
            self._safe_set_spinbox_value(
                self.warmup_lr_init_spin, training_config.get("warmup_lr_init"), 1e-5
            )
            self._safe_set_spinbox_value(
                self.grad_accum_steps_spin,
                training_config.get("gradient_accumulation_steps"),
                1,
            )
            self._safe_set_spinbox_value(
                self.evaluation_freq_spin, training_config.get("evaluation_freq"), 1
            )
            self._safe_set_checkbox_value(
                self.use_ema_check, training_config.get("use_ema"), False
            )
            self._safe_set_spinbox_value(
                self.ema_decay_spin, training_config.get("ema_decay"), 0.999
            )

            # Parameters from optimization tab / training section
            batch_size_val = training_config.get(
                "batch_size", optimization_profile_config.get("batch_size", 32)
            )
            num_workers_val = training_config.get(
                "num_workers", optimization_profile_config.get("num_workers", 4)
            )
            mixed_precision_val = training_config.get(
                "mixed_precision",
                optimization_profile_config.get("use_mixed_precision", True),
            )

            if hasattr(self, "parameter_rows"):
                if "batch_size" in self.parameter_rows:
                    self._safe_set_spinbox_value(
                        self.parameter_rows["batch_size"]["value_widget"],
                        batch_size_val,
                        32,
                    )
                if "num_workers" in self.parameter_rows:
                    self._safe_set_spinbox_value(
                        self.parameter_rows["num_workers"]["value_widget"],
                        num_workers_val,
                        4,
                    )
                if "use_mixed_precision" in self.parameter_rows:
                    self._safe_set_checkbox_value(
                        self.parameter_rows["use_mixed_precision"]["value_widget"],
                        mixed_precision_val,
                        True,
                    )

            self._safe_set_spinbox_value(
                self.unfreeze_after_epochs_spin,
                training_config.get("unfreeze_after_epochs"),
                0,
            )
            self._safe_set_spinbox_value(
                self.frozen_lr_spin, training_config.get("frozen_lr"), 1e-5
            )
            self._safe_set_spinbox_value(
                self.unfrozen_lr_spin, training_config.get("unfrozen_lr"), 1e-4
            )
            self._safe_set_spinbox_value(
                self.validation_split_spin, training_config.get("validation_split"), 0.2
            )

            # Regularyzacja
            reg_config = config.get("regularization", {})
            self._safe_set_spinbox_value(
                self.weight_decay_spin, reg_config.get("weight_decay"), 0.0001
            )
            self._safe_set_spinbox_value(
                self.gradient_clip_spin, reg_config.get("gradient_clip"), 1.0
            )  # Also in advanced, ensure one source
            self._safe_set_spinbox_value(
                self.label_smoothing_spin, reg_config.get("label_smoothing"), 0.1
            )
            drop_connect_val = reg_config.get("drop_connect_rate")
            self._safe_set_spinbox_value(
                self.drop_connect_spin,
                drop_connect_val,
                0.2 if drop_connect_val is None else drop_connect_val,
            )  # Keep original logic for this one if it was special
            self._safe_set_spinbox_value(
                self.dropout_spin, reg_config.get("dropout_rate"), 0.2
            )
            self._safe_set_spinbox_value(
                self.momentum_spin, reg_config.get("momentum"), 0.9
            )
            self._safe_set_spinbox_value(
                self.epsilon_spin, reg_config.get("epsilon"), 1e-6
            )

            swa_config = reg_config.get("swa", {})
            self._safe_set_checkbox_value(
                self.use_swa_check, swa_config.get("use"), False
            )
            self._safe_set_spinbox_value(
                self.swa_start_epoch_spin, swa_config.get("start_epoch"), 10
            )
            self._safe_set_spinbox_value(
                self.swa_lr_swa_spin, swa_config.get("lr_swa"), 0.0005
            )

            stochastic_depth_config = reg_config.get("stochastic_depth", {})
            self._safe_set_checkbox_value(
                self.stochastic_depth_check, stochastic_depth_config.get("use"), False
            )
            self._safe_set_spinbox_value(
                self.stochastic_depth_prob_spin,
                stochastic_depth_config.get("survival_probability"),
                0.8,
            )

            # Augmentacja
            aug_config = config.get("augmentation", {})
            basic_config = aug_config.get("basic", {})
            self._safe_set_checkbox_value(
                self.basic_aug_check, basic_config.get("use"), False
            )
            self._safe_set_spinbox_value(
                self.rotation_spin, basic_config.get("rotation"), 30
            )
            self._safe_set_spinbox_value(
                self.brightness_spin, basic_config.get("brightness"), 0.2
            )
            self._safe_set_spinbox_value(
                self.contrast_spin, basic_config.get("contrast"), 0.2
            )
            self._safe_set_spinbox_value(
                self.saturation_spin, basic_config.get("saturation"), 0.2
            )
            self._safe_set_spinbox_value(self.hue_spin, basic_config.get("hue"), 0.1)
            self._safe_set_spinbox_value(
                self.shift_spin, basic_config.get("shift"), 0.1
            )
            self._safe_set_spinbox_value(self.zoom_spin, basic_config.get("zoom"), 0.1)
            self._safe_set_checkbox_value(
                self.horizontal_flip_check, basic_config.get("horizontal_flip"), True
            )
            self._safe_set_checkbox_value(
                self.vertical_flip_check, basic_config.get("vertical_flip"), False
            )

            mixup_config = aug_config.get("mixup", {})
            self._safe_set_checkbox_value(
                self.mixup_check, mixup_config.get("use"), False
            )
            self._safe_set_spinbox_value(
                self.mixup_alpha_spin, mixup_config.get("alpha"), 0.2
            )

            cutmix_config = aug_config.get("cutmix", {})
            self._safe_set_checkbox_value(
                self.cutmix_check, cutmix_config.get("use"), False
            )
            self._safe_set_spinbox_value(
                self.cutmix_alpha_spin, cutmix_config.get("alpha"), 1.0
            )

            autoaug_config = aug_config.get("autoaugment", {})
            self._safe_set_checkbox_value(
                self.autoaugment_check, autoaug_config.get("use"), False
            )
            self._safe_set_combo_value(
                self.autoaugment_policy_combo, autoaug_config.get("policy"), "imagenet"
            )

            randaug_config = aug_config.get("randaugment", {})
            self._safe_set_checkbox_value(
                self.randaugment_check, randaug_config.get("use"), False
            )
            self._safe_set_spinbox_value(
                self.randaugment_n_spin, randaug_config.get("n"), 2
            )
            self._safe_set_spinbox_value(
                self.randaugment_m_spin, randaug_config.get("m"), 9
            )

            self._safe_set_checkbox_value(
                self.trivialaugment_check,
                aug_config.get("trivialaugment", {}).get("use"),
                False,
            )

            random_erase_config = aug_config.get("random_erase", {})
            self._safe_set_checkbox_value(
                self.random_erase_check, random_erase_config.get("use"), False
            )
            self._safe_set_spinbox_value(
                self.random_erase_prob_spin, random_erase_config.get("probability"), 0.5
            )
            re_scale = random_erase_config.get("scale", [0.02, 0.33])
            if isinstance(re_scale, list) and len(re_scale) == 2:
                self.random_erase_scale_min_spin.setValue(re_scale[0])
                self.random_erase_scale_max_spin.setValue(re_scale[1])
            re_ratio = random_erase_config.get("ratio", [0.3, 3.3])
            if isinstance(re_ratio, list) and len(re_ratio) == 2:
                self.random_erase_ratio_min_spin.setValue(re_ratio[0])
                self.random_erase_ratio_max_spin.setValue(re_ratio[1])

            grid_dist_config = aug_config.get("grid_distortion", {})
            self._safe_set_checkbox_value(
                self.grid_distortion_check, grid_dist_config.get("enabled"), False
            )  # MD uses 'enabled'
            self._safe_set_spinbox_value(
                self.grid_distortion_prob_spin, grid_dist_config.get("probability"), 0.5
            )
            self._safe_set_spinbox_value(
                self.grid_distortion_limit_spin,
                grid_dist_config.get("distort_limit"),
                0.3,
            )

            # Preprocessing
            prep_config = config.get("preprocessing", {})
            norm_config = prep_config.get("normalization", {})
            if isinstance(norm_config, str):  # Old format with just "RGB" / "BGR"
                self._safe_set_combo_value(
                    self.normalization_rgb_bgr_combo, norm_config, "RGB"
                )
            else:  # New format with mean/std
                self._safe_set_combo_value(
                    self.normalization_rgb_bgr_combo,
                    norm_config.get("color_order", "RGB"),
                    "RGB",
                )  # Assuming you add color_order
                mean_values = norm_config.get("mean", [0.485, 0.456, 0.406])
                std_values = norm_config.get("std", [0.229, 0.224, 0.225])
                if len(mean_values) == 3:
                    self.norm_mean_r_spin.setValue(mean_values[0])
                    self.norm_mean_g_spin.setValue(mean_values[1])
                    self.norm_mean_b_spin.setValue(mean_values[2])
                if len(std_values) == 3:
                    self.norm_std_r_spin.setValue(std_values[0])
                    self.norm_std_g_spin.setValue(std_values[1])
                    self.norm_std_b_spin.setValue(std_values[2])

            self._safe_set_combo_value(
                self.scaling_method,
                prep_config.get(
                    "resize_mode", prep_config.get("scaling", {}).get("method")
                ),
                "Bilinear",
            )  # Handle both keys
            self._safe_set_checkbox_value(
                self.cache_dataset_check, prep_config.get("cache_dataset"), False
            )

            scaling_config = prep_config.get(
                "scaling", {}
            )  # This was the previous structure for these
            self._safe_set_checkbox_value(
                self.maintain_aspect_ratio,
                scaling_config.get("maintain_aspect_ratio"),
                True,
            )
            self._safe_set_checkbox_value(
                self.pad_to_square, scaling_config.get("pad_to_square"), False
            )
            self._safe_set_combo_value(
                self.pad_mode, scaling_config.get("pad_mode"), "constant"
            )
            self._safe_set_spinbox_value(
                self.pad_value, scaling_config.get("pad_value"), 0
            )

            rrc_config = prep_config.get("random_resize_crop", {})
            self._safe_set_checkbox_value(
                self.rrc_check, rrc_config.get("enabled"), False
            )
            self._safe_set_spinbox_value(
                self.rrc_size_spin, rrc_config.get("size"), 224
            )
            rrc_scale = rrc_config.get("scale", [0.08, 1.0])
            if isinstance(rrc_scale, list) and len(rrc_scale) == 2:
                self.rrc_scale_min_spin.setValue(rrc_scale[0])
                self.rrc_scale_max_spin.setValue(rrc_scale[1])
            rrc_ratio = rrc_config.get("ratio", [0.75, 1.33])
            if isinstance(rrc_ratio, list) and len(rrc_ratio) == 2:
                self.rrc_ratio_min_spin.setValue(rrc_ratio[0])
                self.rrc_ratio_max_spin.setValue(rrc_ratio[1])

            # Monitorowanie
            monitor_config = config.get("monitoring", {})
            metrics_config = monitor_config.get("metrics", {})
            self._safe_set_checkbox_value(
                self.accuracy_check, metrics_config.get("accuracy"), True
            )
            self._safe_set_checkbox_value(
                self.precision_check, metrics_config.get("precision"), True
            )
            self._safe_set_checkbox_value(
                self.recall_check, metrics_config.get("recall"), True
            )
            self._safe_set_checkbox_value(self.f1_check, metrics_config.get("f1"), True)

            topk_val = metrics_config.get(
                "topk", ""
            )  # Expecting string from QLineEdit now
            if isinstance(topk_val, list):  # Handle old format if present
                self.topk_edit.setText(",".join(map(str, topk_val)))
            elif isinstance(topk_val, (bool, int)):  # Handle very old format
                self.topk_edit.setText(
                    str(topk_val)
                    if isinstance(topk_val, int)
                    else ("5" if topk_val else "")
                )
            else:
                self.topk_edit.setText(str(topk_val))

            self._safe_set_checkbox_value(
                self.confusion_matrix_check,
                metrics_config.get("confusion_matrix"),
                False,
            )
            self._safe_set_checkbox_value(
                self.auc_check, metrics_config.get("auc"), False
            )
            self._safe_set_checkbox_value(
                self.gpu_util_check, metrics_config.get("gpu_utilization"), False
            )
            self._safe_set_checkbox_value(
                self.memory_usage_check, metrics_config.get("memory_usage"), False
            )

            logging_config = monitor_config.get("logging", {})
            self._safe_set_checkbox_value(
                self.use_tensorboard_check, logging_config.get("use_tensorboard"), True
            )
            self.tensorboard_dir_edit.setText(
                logging_config.get("tensorboard_log_dir", "logs/tensorboard")
            )
            self._safe_set_checkbox_value(
                self.use_wandb_check, logging_config.get("use_wandb"), False
            )
            self.wandb_project_edit.setText(logging_config.get("wandb_project", ""))
            self.wandb_entity_edit.setText(logging_config.get("wandb_entity", ""))
            self._safe_set_checkbox_value(
                self.save_logs_check, logging_config.get("save_to_csv"), True
            )  # save_logs_check is existing
            self._safe_set_combo_value(
                self.logging_freq_combo, logging_config.get("logging_freq"), "epoch"
            )

            viz_config = monitor_config.get("visualization", {})
            self._safe_set_checkbox_value(
                self.viz_gradcam_check, viz_config.get("use_gradcam"), False
            )
            self._safe_set_checkbox_value(
                self.viz_feature_maps_check, viz_config.get("use_feature_maps"), False
            )
            self._safe_set_checkbox_value(
                self.viz_pred_samples_check, viz_config.get("use_pred_samples"), False
            )
            self._safe_set_spinbox_value(
                self.viz_num_samples_spin, viz_config.get("num_samples"), 16
            )

            early_stop_config = monitor_config.get("early_stopping", {})
            # use_early_stopping_check is 'enabled' in MD and Python's save
            self._safe_set_checkbox_value(
                self.use_early_stopping_check,
                early_stop_config.get("use", early_stop_config.get("enabled")),
                True,
            )
            self._safe_set_spinbox_value(
                self.patience_spin, early_stop_config.get("patience"), 10
            )
            self._safe_set_spinbox_value(
                self.min_delta_spin, early_stop_config.get("min_delta"), 0.001
            )
            self._safe_set_combo_value(
                self.monitor_combo, early_stop_config.get("monitor"), "val_loss"
            )
            self._safe_set_combo_value(
                self.early_stopping_mode_combo, early_stop_config.get("mode"), "min"
            )

            checkpoint_config = monitor_config.get("checkpointing", {})
            self._safe_set_checkbox_value(
                self.use_checkpointing_check, checkpoint_config.get("use", True), True
            )  # Assume true if section exists
            self.checkpoint_save_dir_edit.setText(
                checkpoint_config.get("save_dir", "models/checkpoints")
            )
            self.checkpoint_filename_edit.setText(
                checkpoint_config.get("filename", "{epoch}-{val_loss:.2f}")
            )
            self._safe_set_checkbox_value(
                self.best_only_check,
                checkpoint_config.get(
                    "save_best_only", checkpoint_config.get("best_only")
                ),
                True,
            )
            self._safe_set_spinbox_value(
                self.save_freq_spin, checkpoint_config.get("save_frequency"), 1
            )
            self._safe_set_combo_value(
                self.checkpoint_metric_combo,
                checkpoint_config.get("monitor", checkpoint_config.get("metric")),
                "val_loss",
            )
            self._safe_set_combo_value(
                self.checkpoint_mode_combo, checkpoint_config.get("mode"), "min"
            )
            self._safe_set_spinbox_value(
                self.checkpoint_top_k_spin, checkpoint_config.get("top_k"), 1
            )

            # Advanced Tab settings
            adv_config = config.get("advanced", {})
            self._safe_set_spinbox_value(self.seed_spin, adv_config.get("seed"), 42)
            self._safe_set_checkbox_value(
                self.deterministic_check, adv_config.get("deterministic"), False
            )
            # class_weights can be dict, handle carefully
            cw_val = adv_config.get("class_weights", "none")
            if isinstance(cw_val, str):
                self._safe_set_combo_value(self.class_weights_combo, cw_val, "none")
                self.class_weights_custom_edit.clear()
            elif isinstance(cw_val, dict):
                self.class_weights_combo.setCurrentText("custom")
                self.class_weights_custom_edit.setText(json.dumps(cw_val))

            self._safe_set_combo_value(
                self.sampler_combo, adv_config.get("sampler"), "random"
            )
            self._safe_set_spinbox_value(
                self.image_channels_spin, adv_config.get("image_channels"), 3
            )

            tta_config = adv_config.get("tta", {})
            self._safe_set_checkbox_value(self.tta_check, tta_config.get("use"), False)
            self._safe_set_spinbox_value(
                self.tta_num_aug_spin, tta_config.get("num_augmentations"), 5
            )

            cv_config = adv_config.get(
                "cross_validation", {}
            )  # 'cross_validation' in MD
            self._safe_set_checkbox_value(
                self.use_cv_check, cv_config.get("use"), False
            )  # use_cv is existing widget
            self._safe_set_spinbox_value(
                self.cv_folds_spin, cv_config.get("k_folds"), 5
            )  # cv_folds existing widget

            dist_config = adv_config.get("distributed", {})
            self._safe_set_checkbox_value(
                self.use_dist_check, dist_config.get("use"), False
            )  # use_dist existing
            self._safe_set_combo_value(
                self.dist_backend_combo, dist_config.get("backend"), "nccl"
            )  # dist_backend existing
            # self.dist_strategy - this was python specific, MD does not list it. Assuming DDP is implied.
            self._safe_set_spinbox_value(
                self.dist_world_size_spin, dist_config.get("world_size"), 1
            )
            self._safe_set_spinbox_value(
                self.dist_rank_spin, dist_config.get("rank"), 0
            )

            self._safe_set_checkbox_value(
                self.export_onnx_check, adv_config.get("export_onnx"), False
            )
            quant_config = adv_config.get("quantization", {})
            self._safe_set_checkbox_value(
                self.quantization_check, quant_config.get("use"), False
            )
            self._safe_set_combo_value(
                self.quantization_precision_combo, quant_config.get("precision"), "int8"
            )

            # Transfer learning (already in advanced tab in python)
            # freeze_base_model, unfreeze_layers, unfreeze_strategy are existing widgets
            self._safe_set_checkbox_value(
                self.freeze_base_model_check,
                training_config.get("freeze_base_model"),
                True,
            )
            unfreeze_layers_val = training_config.get("unfreeze_layers", "")
            self.unfreeze_layers_edit.setText(str(unfreeze_layers_val))

            internal_strategy_value = training_config.get(
                "unfreeze_strategy", self.UNFREEZE_ALL
            )
            for i in range(self.unfreeze_strategy_combo.count()):
                item_text = self.unfreeze_strategy_combo.itemText(i)
                # Use a helper to parse strategy from display text if needed, or ensure display text is unique
                # For simplicity, assuming display text contains the key for now
                if (
                    self._get_unfreeze_strategy_value(item_text)
                    == internal_strategy_value
                ):
                    self.unfreeze_strategy_combo.setCurrentIndex(i)
                    break

            # Catastrophic Forgetting Prevention
            cfp_config = adv_config.get("catastrophic_forgetting_prevention", {})
            self._safe_set_checkbox_value(
                self.cfp_enable_check, cfp_config.get("enable"), False
            )
            # ... and so on for all CFP parameters ...

            # Optimization Tab parameters (from profile's "optimization" or "training" section)
            opt_tab_config = config.get("optimization", {})
            # If not in "optimization", try "training" for these common params
            if not opt_tab_config:
                opt_tab_config = training_config

            if hasattr(self, "parameter_rows"):
                params_to_set = {
                    "memory_efficient": opt_tab_config.get("memory_efficient", False),
                    "cudnn_benchmark": opt_tab_config.get("cudnn_benchmark", True),
                    "pin_memory": opt_tab_config.get("pin_memory", True),
                    "dataloader_shuffle": opt_tab_config.get("dataloader", {}).get(
                        "shuffle", True
                    ),
                    "dataloader_prefetch_factor": opt_tab_config.get(
                        "dataloader", {}
                    ).get("prefetch_factor", 2),
                    "dataloader_persistent_workers": opt_tab_config.get(
                        "dataloader", {}
                    ).get("persistent_workers", False),
                    "dataloader_drop_last": opt_tab_config.get("dataloader", {}).get(
                        "drop_last", False
                    ),
                }
                for key, val in params_to_set.items():
                    if key in self.parameter_rows:
                        widget = self.parameter_rows[key]["value_widget"]
                        if isinstance(widget, QtWidgets.QCheckBox):
                            self._safe_set_checkbox_value(widget, val, False)
                        elif isinstance(
                            widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)
                        ):
                            self._safe_set_spinbox_value(widget, val, 0)

            QtWidgets.QMessageBox.information(
                self, "Sukces", "Profil został pomyślnie zastosowany."
            )

        except Exception as e:
            self.logger.error(
                f"Błąd podczas stosowania profilu: {str(e)}", exc_info=True
            )
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Nie można zastosować profilu: {str(e)}"
            )

    def _clone_profile(self):
        if not self.current_profile or self.profile_list.currentItem() is None:
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
                new_profile_path = self.profiles_dir / f"{new_name}.json"
                if new_profile_path.exists():
                    QtWidgets.QMessageBox.warning(
                        self, "Błąd", f"Profil '{new_name}' już istnieje."
                    )
                    return

                # Deep copy the current profile data
                new_profile_data = json.loads(json.dumps(self.current_profile))
                new_profile_data["info"] = f"Klon profilu {current_name} ({new_name})"
                new_profile_data["description"] = (
                    f"Sklonowany z '{current_name}'. "
                    f"{new_profile_data.get('description', '')}"
                )
                new_profile_data["type"] = "training"

                with open(new_profile_path, "w", encoding="utf-8") as f:
                    json.dump(new_profile_data, f, indent=4, ensure_ascii=False)

                self._refresh_profile_list()
                # Optionally select the new profile
                for i in range(self.profile_list.count()):
                    if self.profile_list.item(i).text() == new_name:
                        self.profile_list.setCurrentRow(i)
                        break
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

    def _get_current_config_as_dict(self):
        """Helper function to gather all current UI settings into a config dict."""

        # Gather optimization params first as they might be used in training section
        optimization_config_data = {}
        if hasattr(self, "parameter_rows"):
            for key, row_widgets in self.parameter_rows.items():
                value_widget = row_widgets["value_widget"]
                if isinstance(
                    value_widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)
                ):
                    optimization_config_data[key] = value_widget.value()
                elif isinstance(value_widget, QtWidgets.QCheckBox):
                    optimization_config_data[key] = value_widget.isChecked()
                else:  # QLineEdit
                    optimization_config_data[key] = value_widget.text()

        # Scheduler config
        scheduler_config_data = {"type": self.scheduler_combo.currentText()}
        if scheduler_config_data["type"] == "CosineAnnealingWarmRestarts":
            scheduler_config_data["T_0"] = self.scheduler_T0_spin.value()
            scheduler_config_data["T_mult"] = self.scheduler_Tmult_spin.value()
            scheduler_config_data["eta_min"] = self.scheduler_eta_min_spin.value()
        elif scheduler_config_data["type"] == "StepLR":
            scheduler_config_data["step_size"] = self.scheduler_step_size_spin.value()
            scheduler_config_data["gamma"] = self.scheduler_gamma_spin.value()
        elif scheduler_config_data["type"] == "OneCycleLR":
            scheduler_config_data["max_lr"] = self.scheduler_max_lr_spin.value()
            scheduler_config_data["pct_start"] = self.scheduler_pct_start_spin.value()
            # total_steps might need calculation based on epochs and dataset size, or be a direct input
            # scheduler_config_data["total_steps"] = self.scheduler_total_steps_spin.value()
        # ... other schedulers

        topk_text = self.topk_edit.text().strip()
        topk_list = []
        if topk_text:
            try:
                topk_list = [int(k.strip()) for k in topk_text.split(",") if k.strip()]
            except ValueError:
                self.logger.warning(
                    f"Nieprawidłowa wartość Top-K: {topk_text}. Używam pustej listy."
                )

        config_data = {
            "model": {
                "architecture": self.arch_combo.currentText(),
                "variant": self.variant_combo.currentText(),
                "input_size": self.input_size_spin.value(),
                "num_classes": self.num_classes_spin.value(),
                "pretrained": self.pretrained_check.isChecked(),
                "pretrained_weights": self.pretrained_weights_combo.currentText(),
                "feature_extraction_only": self.feature_extraction_only_check.isChecked(),
                "activation": self.activation_combo.currentText(),
                "dropout_at_inference": self.dropout_at_inference_check.isChecked(),
                "global_pool": self.global_pool_combo.currentText(),
                "last_layer_activation": self.last_layer_activation_combo.currentText(),
            },
            "training": {
                "epochs": self.epochs_spin.value(),
                "batch_size": optimization_config_data.get("batch_size", 32),
                "learning_rate": float(self.lr_spin.value()),
                "optimizer": self.optimizer_combo.currentText(),
                "scheduler": scheduler_config_data,
                "num_workers": optimization_config_data.get("num_workers", 4),
                "warmup_epochs": self.warmup_epochs_spin.value(),
                "warmup_lr_init": self.warmup_lr_init_spin.value(),
                "mixed_precision": optimization_config_data.get(
                    "use_mixed_precision", True
                ),
                "gradient_accumulation_steps": self.grad_accum_steps_spin.value(),
                "evaluation_freq": self.evaluation_freq_spin.value(),
                "use_ema": self.use_ema_check.isChecked(),
                "ema_decay": self.ema_decay_spin.value(),
                "freeze_base_model": self.freeze_base_model_check.isChecked(),
                "unfreeze_layers": self._get_unfreeze_layers_value(
                    self.unfreeze_layers_edit.text()
                ),
                "unfreeze_strategy": self._get_unfreeze_strategy_value(
                    self.unfreeze_strategy_combo.currentText()
                ),
                "unfreeze_after_epochs": self.unfreeze_after_epochs_spin.value(),
                "frozen_lr": self.frozen_lr_spin.value(),
                "unfrozen_lr": self.unfrozen_lr_spin.value(),
                "validation_split": self.validation_split_spin.value(),
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
                    "lr_swa": self.swa_lr_swa_spin.value(),
                },
                "stochastic_depth": {
                    "use": self.stochastic_depth_check.isChecked(),
                    "survival_probability": self.stochastic_depth_prob_spin.value(),
                },
            },
            "augmentation": {
                "basic": {
                    "use": self.basic_aug_check.isChecked(),
                    "rotation": self.rotation_spin.value(),
                    "brightness": self.brightness_spin.value(),
                    "contrast": self.contrast_spin.value(),
                    "saturation": self.saturation_spin.value(),
                    "hue": self.hue_spin.value(),
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
            },
            "preprocessing": {
                "normalization": {  # Changed to dict
                    "color_order": self.normalization_rgb_bgr_combo.currentText(),  # Renamed widget
                    "mean": [
                        self.norm_mean_r_spin.value(),
                        self.norm_mean_g_spin.value(),
                        self.norm_mean_b_spin.value(),
                    ],
                    "std": [
                        self.norm_std_r_spin.value(),
                        self.norm_std_g_spin.value(),
                        self.norm_std_b_spin.value(),
                    ],
                },
                "resize_mode": self.scaling_method.currentText(),  # scaling_method is existing widget for this
                "cache_dataset": self.cache_dataset_check.isChecked(),
                "scaling": {  # Keep this structure if old profiles use it, or merge into top level of preprocessing
                    "method": self.scaling_method.currentText(),  # Duplicate of resize_mode, choose one
                    "maintain_aspect_ratio": self.maintain_aspect_ratio.isChecked(),
                    "pad_to_square": self.pad_to_square.isChecked(),
                    "pad_mode": self.pad_mode.currentText(),
                    "pad_value": self.pad_value.value(),
                },
                "random_resize_crop": {
                    "enabled": self.rrc_check.isChecked(),
                    "size": self.rrc_size_spin.value(),
                    "scale": [
                        self.rrc_scale_min_spin.value(),
                        self.rrc_scale_max_spin.value(),
                    ],
                    "ratio": [
                        self.rrc_ratio_min_spin.value(),
                        self.rrc_ratio_max_spin.value(),
                    ],
                },
            },
            "monitoring": {
                "metrics": {
                    "accuracy": self.accuracy_check.isChecked(),
                    "precision": self.precision_check.isChecked(),
                    "recall": self.recall_check.isChecked(),
                    "f1": self.f1_check.isChecked(),
                    "topk": topk_list,  # Save as list
                    "confusion_matrix": self.confusion_matrix_check.isChecked(),
                    "auc": self.auc_check.isChecked(),
                    "gpu_utilization": self.gpu_util_check.isChecked(),
                    "memory_usage": self.memory_usage_check.isChecked(),
                },
                "logging": {
                    "use_tensorboard": self.use_tensorboard_check.isChecked(),
                    "tensorboard_log_dir": self.tensorboard_dir_edit.text(),
                    "use_wandb": self.use_wandb_check.isChecked(),
                    "wandb_project": self.wandb_project_edit.text(),
                    "wandb_entity": self.wandb_entity_edit.text(),
                    "save_to_csv": self.save_logs_check.isChecked(),  # save_logs_check is existing
                    "logging_freq": self.logging_freq_combo.currentText(),
                },
                "visualization": {
                    "use_gradcam": self.viz_gradcam_check.isChecked(),
                    "use_feature_maps": self.viz_feature_maps_check.isChecked(),
                    "use_pred_samples": self.viz_pred_samples_check.isChecked(),
                    "num_samples": self.viz_num_samples_spin.value(),
                },
                "early_stopping": {
                    "enabled": self.use_early_stopping_check.isChecked(),  # MD uses 'enabled'
                    "monitor": self.monitor_combo.currentText(),
                    "patience": self.patience_spin.value(),
                    "min_delta": self.min_delta_spin.value(),
                    "mode": self.early_stopping_mode_combo.currentText(),
                },
                "checkpointing": {
                    "use": self.use_checkpointing_check.isChecked(),
                    "save_dir": self.checkpoint_save_dir_edit.text(),
                    "filename": self.checkpoint_filename_edit.text(),
                    "monitor": self.checkpoint_metric_combo.currentText(),  # metric is existing widget
                    "save_best_only": self.best_only_check.isChecked(),
                    "mode": self.checkpoint_mode_combo.currentText(),
                    "save_frequency": self.save_freq_spin.value(),
                    "top_k": self.checkpoint_top_k_spin.value(),
                },
            },
            "advanced": {
                "seed": self.seed_spin.value(),
                "deterministic": self.deterministic_check.isChecked(),
                "class_weights": (
                    self.class_weights_custom_edit.text()
                    if self.class_weights_combo.currentText() == "custom"
                    else self.class_weights_combo.currentText()
                ),  # Parse custom to dict if needed
                "sampler": self.sampler_combo.currentText(),
                "image_channels": self.image_channels_spin.value(),
                "tta": {
                    "use": self.tta_check.isChecked(),
                    "num_augmentations": self.tta_num_aug_spin.value(),
                },
                "cross_validation": {  # use_cv, cv_folds are existing
                    "use": self.use_cv_check.isChecked(),
                    "k_folds": self.cv_folds_spin.value(),
                },
                "distributed": {  # use_dist, dist_backend are existing
                    "use": self.use_dist_check.isChecked(),
                    "backend": self.dist_backend_combo.currentText(),
                    "world_size": self.dist_world_size_spin.value(),
                    "rank": self.dist_rank_spin.value(),
                },
                "export_onnx": self.export_onnx_check.isChecked(),
                "quantization": {
                    "use": self.quantization_check.isChecked(),
                    "precision": self.quantization_precision_combo.currentText(),
                },
                "catastrophic_forgetting_prevention": {
                    "enable": self.cfp_enable_check.isChecked(),
                    # ... other CFP fields ...
                },
                # Existing advanced fields not in MD (scheduler params, init_weights etc.)
                "scheduler_params": {  # Group existing scheduler params if they are for specific types
                    "patience": self.scheduler_patience_spin.value(),
                    "factor": self.scheduler_factor_spin.value(),
                    "min_lr": self.min_lr_spin.value(),
                    "cooldown": self.scheduler_cooldown_spin.value(),
                },
                "weights_init": self.init_weights_combo.currentText(),
                "freeze_cnn_layers": self.freeze_layers_check.isChecked(),  # freeze_layers is existing
                "online_validation": {  # use_online_val, online_val_freq are existing
                    "use": self.use_online_val_check.isChecked(),
                    "frequency": self.online_val_freq_spin.value(),
                },
            },
            "optimization": optimization_config_data,  # Save the full optimization tab state
        }
        if config_data["advanced"]["class_weights"] == "custom":
            try:
                config_data["advanced"]["class_weights"] = json.loads(
                    self.class_weights_custom_edit.text()
                )
            except json.JSONDecodeError:
                self.logger.warning(
                    "Invalid JSON for custom class weights. Saving as string."
                )
                # Keep as string or set to None/error
        return config_data

    def _save_profile(self):
        try:
            name, ok = QtWidgets.QInputDialog.getText(
                self,
                "Zapisz profil",
                "Podaj nazwę dla nowego profilu:",
                QtWidgets.QLineEdit.EchoMode.Normal,
                f"{self.arch_combo.currentText()}_{self.variant_combo.currentText()}",
            )

            if ok and name:
                if not name.strip():
                    QtWidgets.QMessageBox.warning(
                        self, "Błąd", "Nazwa profilu nie może być pusta."
                    )
                    return

                profile_path = self.profiles_dir / f"{name}.json"
                if profile_path.exists():
                    reply = QtWidgets.QMessageBox.question(
                        self,
                        "Nadpisać profil?",
                        f"Profil '{name}' już istnieje. Czy chcesz go nadpisać?",
                        QtWidgets.QMessageBox.StandardButton.Yes
                        | QtWidgets.QMessageBox.StandardButton.No,
                        QtWidgets.QMessageBox.StandardButton.No,
                    )
                    if reply == QtWidgets.QMessageBox.StandardButton.No:
                        return

                current_config_dict = self._get_current_config_as_dict()

                profile_data = {
                    "type": "training",
                    "info": f"Profil dla {self.arch_combo.currentText()} {self.variant_combo.currentText()}",
                    "description": "Profil utworzony przez użytkownika",
                    "data_required": "Standardowe dane treningowe",  # Placeholder
                    "hardware_required": "Standardowy sprzęt",  # Placeholder
                    "config": current_config_dict,
                }

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

    def _delete_profile(self):
        if not self.current_profile or self.profile_list.currentItem() is None:
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
                    self._refresh_profile_list()
                    self.current_profile = None
                    self.profile_info.clear()
                    self.profile_description.clear()
                    self.profile_data_required.clear()
                    self.profile_hardware_required.clear()
                    QtWidgets.QMessageBox.information(
                        self, "Sukces", "Profil został pomyślnie usunięty."
                    )
                else:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Błąd",
                        f"Plik profilu '{current_name}.json' nie istnieje.",
                    )

        except Exception as e:
            self.logger.error(f"Błąd podczas usuwania profilu: {str(e)}", exc_info=True)
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Nie można usunąć profilu: {str(e)}"
            )

    def _on_architecture_changed(self, arch_name):
        self._update_variant_combo(arch_name)

    def _update_variant_combo(self, arch_name):
        self.variant_combo.clear()
        # This should be data-driven, e.g., from a config file or a more extensive dict
        if arch_name == "EfficientNet":
            self.variant_combo.addItems(
                [f"EfficientNet-B{i}" for i in range(8)]
                + [f"EfficientNet-L{i}" for i in range(2, 3)]
            )
        elif arch_name == "ConvNeXt":
            self.variant_combo.addItems(
                [
                    "ConvNeXt-Tiny",
                    "ConvNeXt-Small",
                    "ConvNeXt-Base",
                    "ConvNeXt-Large",
                    "ConvNeXt-XLarge",
                ]
            )
        elif arch_name == "ResNet":
            self.variant_combo.addItems(
                ["ResNet-18", "ResNet-34", "ResNet-50", "ResNet-101", "ResNet-152"]
            )
        elif arch_name == "ViT":
            self.variant_combo.addItems(
                [
                    "ViT-Base-patch16-224",
                    "ViT-Large-patch16-224",
                    "ViT-Huge-patch14-224",
                ]
            )
        # Add more architectures and variants as needed

    def _select_train_dir(self):
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Wybierz katalog treningowy", self.train_dir_edit.text()
        )
        if dir_path:
            if validate_training_directory(
                dir_path
            ):  # Assuming this function exists and is correct
                self.train_dir_edit.setText(dir_path)
            else:
                QtWidgets.QMessageBox.warning(
                    self, "Błąd", "Nieprawidłowy katalog treningowy."
                )

    def _select_val_dir(self):
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Wybierz katalog walidacyjny", self.val_dir_edit.text()
        )
        if dir_path:
            if validate_validation_directory(
                dir_path
            ):  # Assuming this function exists and is correct
                self.val_dir_edit.setText(dir_path)
            else:
                QtWidgets.QMessageBox.warning(
                    self, "Błąd", "Nieprawidłowy katalog walidacyjny."
                )

    def _create_training_params_tab(self):
        try:
            self.logger.debug("Tworzenie zakładki Parametry Treningu")
            tab = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(tab)
            form = QtWidgets.QFormLayout()

            # Liczba epok
            self.epochs_spin = QtWidgets.QSpinBox()
            self.epochs_spin.setRange(1, 10000)
            default_epochs = (
                DEFAULT_TRAINING_PARAMS.get("max_epochs", 100)
                if isinstance(DEFAULT_TRAINING_PARAMS, dict)
                else 100
            )
            self.epochs_spin.setValue(default_epochs)
            form.addRow("Liczba epok:", self.epochs_spin)

            # Współczynnik uczenia
            self.lr_spin = QtWidgets.QDoubleSpinBox()
            self.lr_spin.setDecimals(7)  # Increased precision
            self.lr_spin.setRange(1e-7, 1.0)
            self.lr_spin.setSingleStep(1e-4)
            self.lr_spin.setValue(0.001)
            form.addRow("Współczynnik uczenia:", self.lr_spin)

            # Optymalizator
            self.optimizer_combo = QtWidgets.QComboBox()
            optimizers = ["AdamW", "Adam", "SGD", "RMSprop", "Lion", "Adafactor"]
            self.optimizer_combo.addItems(optimizers)
            self.optimizer_combo.setCurrentText("AdamW")
            form.addRow("Optymalizator:", self.optimizer_combo)

            # --- Scheduler Group ---
            scheduler_group = QtWidgets.QGroupBox("Harmonogram uczenia")
            scheduler_layout = QtWidgets.QFormLayout()

            self.scheduler_combo = QtWidgets.QComboBox()
            schedulers = [
                "None",
                "StepLR",
                "ReduceLROnPlateau",
                "CosineAnnealingLR",
                "OneCycleLR",
                "CosineAnnealingWarmRestarts",
                "LinearWarmupCosineAnnealing",
            ]
            self.scheduler_combo.addItems(schedulers)
            scheduler_layout.addRow("Typ harmonogramu:", self.scheduler_combo)

            # CosineAnnealingWarmRestarts params
            self.scheduler_T0_spin = QtWidgets.QSpinBox()
            self.scheduler_T0_spin.setRange(1, 1000)
            self.scheduler_T0_spin.setValue(10)
            scheduler_layout.addRow("T_0 (CosineAnnealingWR):", self.scheduler_T0_spin)

            self.scheduler_Tmult_spin = QtWidgets.QSpinBox()
            self.scheduler_Tmult_spin.setRange(1, 10)
            self.scheduler_Tmult_spin.setValue(1)
            scheduler_layout.addRow(
                "T_mult (CosineAnnealingWR):", self.scheduler_Tmult_spin
            )

            self.scheduler_eta_min_spin = QtWidgets.QDoubleSpinBox()
            self.scheduler_eta_min_spin.setDecimals(7)
            self.scheduler_eta_min_spin.setRange(0, 0.1)
            self.scheduler_eta_min_spin.setValue(1e-6)
            scheduler_layout.addRow(
                "Eta Min (CosineAnnealingWR/LR):", self.scheduler_eta_min_spin
            )

            # StepLR params
            self.scheduler_step_size_spin = QtWidgets.QSpinBox()
            self.scheduler_step_size_spin.setRange(1, 1000)
            self.scheduler_step_size_spin.setValue(30)
            scheduler_layout.addRow(
                "Step Size (StepLR):", self.scheduler_step_size_spin
            )

            self.scheduler_gamma_spin = QtWidgets.QDoubleSpinBox()
            self.scheduler_gamma_spin.setDecimals(3)
            self.scheduler_gamma_spin.setRange(0.001, 1.0)
            self.scheduler_gamma_spin.setValue(0.1)
            scheduler_layout.addRow("Gamma (StepLR):", self.scheduler_gamma_spin)

            # OneCycleLR params
            self.scheduler_max_lr_spin = QtWidgets.QDoubleSpinBox()
            self.scheduler_max_lr_spin.setDecimals(7)
            self.scheduler_max_lr_spin.setRange(1e-7, 1.0)
            self.scheduler_max_lr_spin.setValue(0.01)  # Should be higher than base LR
            scheduler_layout.addRow("Max LR (OneCycleLR):", self.scheduler_max_lr_spin)

            self.scheduler_pct_start_spin = QtWidgets.QDoubleSpinBox()
            self.scheduler_pct_start_spin.setDecimals(2)
            self.scheduler_pct_start_spin.setRange(0.01, 0.99)
            self.scheduler_pct_start_spin.setValue(0.3)
            scheduler_layout.addRow(
                "Pct Start (OneCycleLR):", self.scheduler_pct_start_spin
            )

            # TODO: Add total_steps for OneCycleLR if not calculated automatically
            # self.scheduler_total_steps_spin = QtWidgets.QSpinBox() ...
            # scheduler_layout.addRow("Total Steps (OneCycleLR, if needed):", self.scheduler_total_steps_spin)

            scheduler_group.setLayout(scheduler_layout)
            form.addRow(scheduler_group)

            # Liczba epok rozgrzewki
            self.warmup_epochs_spin = QtWidgets.QSpinBox()
            self.warmup_epochs_spin.setRange(0, 100)
            self.warmup_epochs_spin.setValue(5)
            form.addRow("Epoki rozgrzewki:", self.warmup_epochs_spin)

            # Początkowy LR dla rozgrzewki
            self.warmup_lr_init_spin = QtWidgets.QDoubleSpinBox()
            self.warmup_lr_init_spin.setDecimals(7)
            self.warmup_lr_init_spin.setRange(1e-8, 1e-2)
            self.warmup_lr_init_spin.setValue(1e-5)
            form.addRow("LR startowy rozgrzewki:", self.warmup_lr_init_spin)

            # Kroki akumulacji gradientu
            self.grad_accum_steps_spin = QtWidgets.QSpinBox()
            self.grad_accum_steps_spin.setRange(1, 64)
            self.grad_accum_steps_spin.setValue(1)
            form.addRow("Kroki akumulacji gradientu:", self.grad_accum_steps_spin)

            # Częstotliwość ewaluacji
            self.evaluation_freq_spin = QtWidgets.QSpinBox()
            self.evaluation_freq_spin.setRange(1, 100)
            self.evaluation_freq_spin.setValue(1)
            form.addRow("Częstotliwość ewaluacji (epoki):", self.evaluation_freq_spin)

            # EMA
            self.use_ema_check = QtWidgets.QCheckBox(
                "Używaj Exponential Moving Average (EMA)"
            )
            form.addRow(self.use_ema_check)
            self.ema_decay_spin = QtWidgets.QDoubleSpinBox()
            self.ema_decay_spin.setDecimals(4)
            self.ema_decay_spin.setRange(0.9, 0.9999)
            self.ema_decay_spin.setValue(0.999)
            form.addRow("Współczynnik zanikania EMA:", self.ema_decay_spin)

            # Validation split
            self.validation_split_spin = QtWidgets.QDoubleSpinBox()
            self.validation_split_spin.setDecimals(3)
            self.validation_split_spin.setRange(
                0.0, 0.9
            )  # 0 means use separate val_dir
            self.validation_split_spin.setValue(0.2)  # Default to 20%
            self.validation_split_spin.setToolTip(
                "Ułamek danych treningowych użyty do walidacji (jeśli katalog walidacyjny nie jest podany, lub jako dodatkowy podział). 0 aby nie dzielić."
            )
            form.addRow(
                "Podział na walidację (z tr. danych):", self.validation_split_spin
            )

            layout.addLayout(form)
            return tab
        except Exception as e:
            self.logger.error(
                f"Błąd tworzenia zakładki Parametry Treningu: {e}", exc_info=True
            )
            return QtWidgets.QWidget()

    def _create_regularization_tab(self):
        try:
            self.logger.debug("Tworzenie zakładki Regularyzacja")
            tab = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(tab)
            form = QtWidgets.QFormLayout()

            self.weight_decay_spin = QtWidgets.QDoubleSpinBox()
            self.weight_decay_spin.setDecimals(7)
            self.weight_decay_spin.setRange(0.0, 1.0)
            self.weight_decay_spin.setSingleStep(1e-5)
            self.weight_decay_spin.setValue(1e-4)  # Common default for AdamW
            form.addRow("Weight Decay:", self.weight_decay_spin)

            self.gradient_clip_spin = QtWidgets.QDoubleSpinBox()
            self.gradient_clip_spin.setRange(0.0, 100.0)  # Increased range
            self.gradient_clip_spin.setDecimals(2)
            self.gradient_clip_spin.setValue(1.0)
            self.gradient_clip_spin.setToolTip("0.0 aby wyłączyć")
            form.addRow("Gradient Clipping (max norm):", self.gradient_clip_spin)

            self.label_smoothing_spin = QtWidgets.QDoubleSpinBox()
            self.label_smoothing_spin.setRange(0.0, 0.5)
            self.label_smoothing_spin.setDecimals(3)
            self.label_smoothing_spin.setValue(0.1)
            form.addRow("Label Smoothing:", self.label_smoothing_spin)

            self.drop_connect_spin = (
                QtWidgets.QDoubleSpinBox()
            )  # For EfficientNet-like models
            self.drop_connect_spin.setRange(0.0, 0.5)
            self.drop_connect_spin.setDecimals(3)
            self.drop_connect_spin.setValue(0.2)
            form.addRow("Drop Connect Rate (jeśli dotyczy):", self.drop_connect_spin)

            self.dropout_spin = (
                QtWidgets.QDoubleSpinBox()
            )  # General dropout for classifier head
            self.dropout_spin.setRange(0.0, 0.9)  # Increased range
            self.dropout_spin.setDecimals(3)
            self.dropout_spin.setValue(0.2)
            form.addRow("Dropout Rate (klasyfikator):", self.dropout_spin)

            self.momentum_spin = QtWidgets.QDoubleSpinBox()  # For SGD, RMSprop
            self.momentum_spin.setRange(0.0, 0.999)
            self.momentum_spin.setDecimals(3)
            self.momentum_spin.setValue(0.9)
            form.addRow("Momentum (dla SGD/RMSprop):", self.momentum_spin)

            self.epsilon_spin = QtWidgets.QDoubleSpinBox()  # For Adam, RMSprop
            self.epsilon_spin.setDecimals(8)
            self.epsilon_spin.setRange(1e-9, 1e-3)
            self.epsilon_spin.setValue(1e-6)
            form.addRow("Epsilon (dla Adam/RMSprop):", self.epsilon_spin)

            # Stochastic Weight Averaging (SWA)
            swa_group = QtWidgets.QGroupBox("Stochastic Weight Averaging (SWA)")
            swa_layout = QtWidgets.QFormLayout()
            self.use_swa_check = QtWidgets.QCheckBox("Używaj SWA")
            swa_layout.addRow(self.use_swa_check)
            self.swa_start_epoch_spin = QtWidgets.QSpinBox()
            self.swa_start_epoch_spin.setRange(1, 1000)
            self.swa_start_epoch_spin.setValue(10)
            swa_layout.addRow("Epoka rozpoczęcia SWA:", self.swa_start_epoch_spin)
            self.swa_lr_swa_spin = QtWidgets.QDoubleSpinBox()
            self.swa_lr_swa_spin.setDecimals(7)
            self.swa_lr_swa_spin.setRange(1e-7, 0.1)
            self.swa_lr_swa_spin.setValue(0.0005)  # Often smaller than main LR
            swa_layout.addRow("Learning Rate dla SWA:", self.swa_lr_swa_spin)
            swa_group.setLayout(swa_layout)
            form.addRow(swa_group)

            # Stochastic Depth
            sd_group = QtWidgets.QGroupBox("Stochastic Depth (jeśli dotyczy)")
            sd_layout = QtWidgets.QFormLayout()
            self.stochastic_depth_check = QtWidgets.QCheckBox("Używaj Stochastic Depth")
            sd_layout.addRow(self.stochastic_depth_check)
            self.stochastic_depth_prob_spin = QtWidgets.QDoubleSpinBox()
            self.stochastic_depth_prob_spin.setDecimals(2)
            self.stochastic_depth_prob_spin.setRange(0.5, 1.0)  # Survival probability
            self.stochastic_depth_prob_spin.setValue(0.8)
            sd_layout.addRow(
                "Prawd. przetrwania (survival prob.):", self.stochastic_depth_prob_spin
            )
            sd_group.setLayout(sd_layout)
            form.addRow(sd_group)

            layout.addLayout(form)
            return tab
        except Exception as e:
            self.logger.error(
                f"Błąd tworzenia zakładki Regularyzacja: {e}", exc_info=True
            )
            return QtWidgets.QWidget()

    def _create_augmentation_tab(self):
        try:
            self.logger.debug("Tworzenie zakładki Augmentacja Danych")
            tab = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(tab)

            # Basic Augmentation Group
            basic_group = QtWidgets.QGroupBox("Podstawowa augmentacja")
            basic_layout = QtWidgets.QFormLayout()
            self.basic_aug_check = QtWidgets.QCheckBox("Używaj podstawowej")
            basic_layout.addRow(self.basic_aug_check)
            self.rotation_spin = QtWidgets.QSpinBox()
            self.rotation_spin.setRange(0, 180)
            self.rotation_spin.setValue(30)
            basic_layout.addRow("Kąt rotacji (± stopnie):", self.rotation_spin)
            self.brightness_spin = QtWidgets.QDoubleSpinBox()
            self.brightness_spin.setRange(0.0, 1.0)
            self.brightness_spin.setValue(0.2)  # e.g. (1-0.2, 1+0.2)
            basic_layout.addRow("Zakres jasności (±):", self.brightness_spin)
            self.contrast_spin = QtWidgets.QDoubleSpinBox()
            self.contrast_spin.setRange(0.0, 1.0)
            self.contrast_spin.setValue(0.2)
            basic_layout.addRow("Zakres kontrastu (±):", self.contrast_spin)
            self.saturation_spin = QtWidgets.QDoubleSpinBox()
            self.saturation_spin.setRange(0.0, 1.0)
            self.saturation_spin.setValue(0.2)
            basic_layout.addRow("Zakres nasycenia (±):", self.saturation_spin)
            self.hue_spin = QtWidgets.QDoubleSpinBox()
            self.hue_spin.setRange(0.0, 0.5)
            self.hue_spin.setValue(0.1)  # e.g. (-0.1, 0.1)
            basic_layout.addRow("Zakres odcienia (±):", self.hue_spin)
            self.shift_spin = QtWidgets.QDoubleSpinBox()  # As fraction of image size
            self.shift_spin.setRange(0.0, 0.5)
            self.shift_spin.setValue(0.1)
            basic_layout.addRow("Przesunięcie (frakcja obrazu, ±):", self.shift_spin)
            self.zoom_spin = QtWidgets.QDoubleSpinBox()  # e.g., (1-0.1, 1+0.1)
            self.zoom_spin.setRange(0.0, 0.9)
            self.zoom_spin.setValue(0.1)
            basic_layout.addRow("Zakres zoomu (±):", self.zoom_spin)
            self.horizontal_flip_check = QtWidgets.QCheckBox("Odwrócenie poziome")
            self.horizontal_flip_check.setChecked(True)
            basic_layout.addRow(self.horizontal_flip_check)
            self.vertical_flip_check = QtWidgets.QCheckBox("Odwrócenie pionowe")
            basic_layout.addRow(self.vertical_flip_check)
            basic_group.setLayout(basic_layout)
            layout.addWidget(basic_group)

            # Advanced Augmentation Group (Mixing techniques, AutoAugment, etc.)
            advanced_group = QtWidgets.QGroupBox("Zaawansowane techniki augmentacji")
            advanced_layout = QtWidgets.QFormLayout()

            # Mixup
            self.mixup_check = QtWidgets.QCheckBox("Używaj Mixup")
            advanced_layout.addRow(self.mixup_check)
            self.mixup_alpha_spin = QtWidgets.QDoubleSpinBox()
            self.mixup_alpha_spin.setRange(0.0, 5.0)
            self.mixup_alpha_spin.setValue(0.2)  # Alpha > 0
            advanced_layout.addRow("Alpha (Mixup):", self.mixup_alpha_spin)

            # CutMix
            self.cutmix_check = QtWidgets.QCheckBox("Używaj CutMix")
            advanced_layout.addRow(self.cutmix_check)
            self.cutmix_alpha_spin = QtWidgets.QDoubleSpinBox()
            self.cutmix_alpha_spin.setRange(0.0, 5.0)
            self.cutmix_alpha_spin.setValue(1.0)  # Alpha > 0
            advanced_layout.addRow("Alpha (CutMix):", self.cutmix_alpha_spin)

            # AutoAugment
            self.autoaugment_check = QtWidgets.QCheckBox("Używaj AutoAugment")
            advanced_layout.addRow(self.autoaugment_check)
            self.autoaugment_policy_combo = QtWidgets.QComboBox()
            self.autoaugment_policy_combo.addItems(["imagenet", "cifar10", "svhn"])
            advanced_layout.addRow(
                "Polityka AutoAugment:", self.autoaugment_policy_combo
            )

            # RandAugment
            self.randaugment_check = QtWidgets.QCheckBox("Używaj RandAugment")
            advanced_layout.addRow(self.randaugment_check)
            self.randaugment_n_spin = QtWidgets.QSpinBox()  # Num ops
            self.randaugment_n_spin.setRange(1, 10)
            self.randaugment_n_spin.setValue(2)
            advanced_layout.addRow(
                "N (RandAugment - liczba operacji):", self.randaugment_n_spin
            )
            self.randaugment_m_spin = QtWidgets.QSpinBox()  # Magnitude
            self.randaugment_m_spin.setRange(1, 30)
            self.randaugment_m_spin.setValue(9)
            advanced_layout.addRow(
                "M (RandAugment - intensywność):", self.randaugment_m_spin
            )

            # TrivialAugment
            self.trivialaugment_check = QtWidgets.QCheckBox("Używaj TrivialAugmentWide")
            advanced_layout.addRow(self.trivialaugment_check)

            # Random Erase
            re_group = QtWidgets.QGroupBox("Random Erasing")
            re_layout = QtWidgets.QFormLayout()
            self.random_erase_check = QtWidgets.QCheckBox("Używaj Random Erasing")
            re_layout.addRow(self.random_erase_check)
            self.random_erase_prob_spin = QtWidgets.QDoubleSpinBox()
            self.random_erase_prob_spin.setRange(0.0, 1.0)
            self.random_erase_prob_spin.setValue(0.5)
            re_layout.addRow("Prawdopodobieństwo:", self.random_erase_prob_spin)
            self.random_erase_scale_min_spin = QtWidgets.QDoubleSpinBox()
            self.random_erase_scale_min_spin.setRange(0.01, 0.5)
            self.random_erase_scale_min_spin.setValue(0.02)
            re_layout.addRow("Skala min:", self.random_erase_scale_min_spin)
            self.random_erase_scale_max_spin = QtWidgets.QDoubleSpinBox()
            self.random_erase_scale_max_spin.setRange(0.01, 0.5)
            self.random_erase_scale_max_spin.setValue(0.33)
            re_layout.addRow("Skala max:", self.random_erase_scale_max_spin)
            self.random_erase_ratio_min_spin = QtWidgets.QDoubleSpinBox()
            self.random_erase_ratio_min_spin.setRange(0.1, 10.0)
            self.random_erase_ratio_min_spin.setValue(0.3)
            re_layout.addRow("Proporcje min:", self.random_erase_ratio_min_spin)
            self.random_erase_ratio_max_spin = QtWidgets.QDoubleSpinBox()
            self.random_erase_ratio_max_spin.setRange(0.1, 10.0)
            self.random_erase_ratio_max_spin.setValue(3.3)
            re_layout.addRow("Proporcje max:", self.random_erase_ratio_max_spin)
            re_group.setLayout(re_layout)
            advanced_layout.addRow(re_group)

            # Grid Distortion
            gd_group = QtWidgets.QGroupBox("Grid Distortion")
            gd_layout = QtWidgets.QFormLayout()
            self.grid_distortion_check = QtWidgets.QCheckBox("Używaj Grid Distortion")
            gd_layout.addRow(self.grid_distortion_check)
            self.grid_distortion_prob_spin = QtWidgets.QDoubleSpinBox()
            self.grid_distortion_prob_spin.setRange(0.0, 1.0)
            self.grid_distortion_prob_spin.setValue(0.5)
            gd_layout.addRow("Prawdopodobieństwo:", self.grid_distortion_prob_spin)
            self.grid_distortion_limit_spin = QtWidgets.QDoubleSpinBox()
            self.grid_distortion_limit_spin.setRange(0.0, 1.0)
            self.grid_distortion_limit_spin.setValue(0.3)
            gd_layout.addRow("Limit zniekształcenia:", self.grid_distortion_limit_spin)
            gd_group.setLayout(gd_layout)
            advanced_layout.addRow(gd_group)

            advanced_group.setLayout(advanced_layout)
            layout.addWidget(advanced_group)
            return tab
        except Exception as e:
            self.logger.error(
                f"Błąd tworzenia zakładki Augmentacja: {e}", exc_info=True
            )
            return QtWidgets.QWidget()

    def _create_preprocessing_tab(self):
        try:
            self.logger.debug("Tworzenie zakładki Preprocessing")
            tab = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(tab)

            # Normalization Group
            norm_group = QtWidgets.QGroupBox("Normalizacja")
            norm_layout = QtWidgets.QFormLayout()
            self.normalization_rgb_bgr_combo = (
                QtWidgets.QComboBox()
            )  # Renamed from normalization_combo
            self.normalization_rgb_bgr_combo.addItems(["RGB", "BGR"])
            norm_layout.addRow(
                "Kolejność kanałów (dla wag):", self.normalization_rgb_bgr_combo
            )

            mean_layout = QtWidgets.QHBoxLayout()
            self.norm_mean_r_spin = QtWidgets.QDoubleSpinBox()
            self.norm_mean_r_spin.setValue(0.485)
            self.norm_mean_g_spin = QtWidgets.QDoubleSpinBox()
            self.norm_mean_g_spin.setValue(0.456)
            self.norm_mean_b_spin = QtWidgets.QDoubleSpinBox()
            self.norm_mean_b_spin.setValue(0.406)
            mean_layout.addWidget(QtWidgets.QLabel("R:"))
            mean_layout.addWidget(self.norm_mean_r_spin)
            mean_layout.addWidget(QtWidgets.QLabel("G:"))
            mean_layout.addWidget(self.norm_mean_g_spin)
            mean_layout.addWidget(QtWidgets.QLabel("B:"))
            mean_layout.addWidget(self.norm_mean_b_spin)
            norm_layout.addRow("Średnie (Mean):", mean_layout)

            std_layout = QtWidgets.QHBoxLayout()
            self.norm_std_r_spin = QtWidgets.QDoubleSpinBox()
            self.norm_std_r_spin.setValue(0.229)
            self.norm_std_g_spin = QtWidgets.QDoubleSpinBox()
            self.norm_std_g_spin.setValue(0.224)
            self.norm_std_b_spin = QtWidgets.QDoubleSpinBox()
            self.norm_std_b_spin.setValue(0.225)
            std_layout.addWidget(QtWidgets.QLabel("R:"))
            std_layout.addWidget(self.norm_std_r_spin)
            std_layout.addWidget(QtWidgets.QLabel("G:"))
            std_layout.addWidget(self.norm_std_g_spin)
            std_layout.addWidget(QtWidgets.QLabel("B:"))
            std_layout.addWidget(self.norm_std_b_spin)
            norm_layout.addRow("Odchylenia Std (Std):", std_layout)
            norm_group.setLayout(norm_layout)
            layout.addWidget(norm_group)

            # Scaling Group (existing)
            scaling_group = QtWidgets.QGroupBox("Skalowanie obrazu (Resize)")
            scaling_layout = QtWidgets.QFormLayout()
            self.scaling_method = QtWidgets.QComboBox()  # Interpolation method
            self.scaling_method.addItems(
                ["Bilinear", "Bicubic", "Lanczos", "Nearest", "Area"]
            )
            self.scaling_method.setCurrentText(
                "Bicubic"
            )  # Often better for downscaling
            scaling_layout.addRow("Metoda interpolacji:", self.scaling_method)
            self.maintain_aspect_ratio = QtWidgets.QCheckBox(
                "Zachowaj proporcje (przy resize)"
            )
            self.maintain_aspect_ratio.setChecked(True)
            scaling_layout.addRow(self.maintain_aspect_ratio)
            self.pad_to_square = QtWidgets.QCheckBox(
                "Uzupełnij do kwadratu (po resize, jeśli proporcje zachowane)"
            )
            scaling_layout.addRow(self.pad_to_square)
            self.pad_mode = QtWidgets.QComboBox()
            self.pad_mode.addItems(["constant", "edge", "reflect", "symmetric"])
            self.pad_mode.setCurrentText("constant")
            scaling_layout.addRow("Tryb uzupełniania (padding):", self.pad_mode)
            self.pad_value = QtWidgets.QSpinBox()
            self.pad_value.setRange(0, 255)
            self.pad_value.setValue(0)
            scaling_layout.addRow(
                "Wartość uzupełniania (dla 'constant'):", self.pad_value
            )
            scaling_group.setLayout(scaling_layout)
            layout.addWidget(scaling_group)

            # Random Resize Crop Group
            rrc_group = QtWidgets.QGroupBox("Random Resized Crop")
            rrc_layout = QtWidgets.QFormLayout()
            self.rrc_check = QtWidgets.QCheckBox("Używaj RandomResizedCrop")
            rrc_layout.addRow(self.rrc_check)
            self.rrc_size_spin = QtWidgets.QSpinBox()  # Target size after crop
            self.rrc_size_spin.setRange(32, 2048)
            self.rrc_size_spin.setValue(224)
            self.rrc_size_spin.setSingleStep(32)
            rrc_layout.addRow("Rozmiar docelowy:", self.rrc_size_spin)
            self.rrc_scale_min_spin = QtWidgets.QDoubleSpinBox()
            self.rrc_scale_min_spin.setRange(0.05, 1.0)
            self.rrc_scale_min_spin.setValue(0.08)
            rrc_layout.addRow("Min skala przycięcia:", self.rrc_scale_min_spin)
            self.rrc_scale_max_spin = QtWidgets.QDoubleSpinBox()
            self.rrc_scale_max_spin.setRange(0.05, 1.0)
            self.rrc_scale_max_spin.setValue(1.0)
            rrc_layout.addRow("Max skala przycięcia:", self.rrc_scale_max_spin)
            self.rrc_ratio_min_spin = QtWidgets.QDoubleSpinBox()
            self.rrc_ratio_min_spin.setRange(0.1, 10.0)
            self.rrc_ratio_min_spin.setValue(0.75)  # 3./4.
            rrc_layout.addRow("Min proporcje przycięcia:", self.rrc_ratio_min_spin)
            self.rrc_ratio_max_spin = QtWidgets.QDoubleSpinBox()
            self.rrc_ratio_max_spin.setRange(0.1, 10.0)
            self.rrc_ratio_max_spin.setValue(1.33)  # 4./3.
            rrc_layout.addRow("Max proporcje przycięcia:", self.rrc_ratio_max_spin)
            rrc_group.setLayout(rrc_layout)
            layout.addWidget(rrc_group)

            # Dataset Caching
            self.cache_dataset_check = QtWidgets.QCheckBox(
                "Cachuj przetworzony zestaw danych w pamięci RAM (jeśli wystarczająco miejsca)"
            )
            layout.addWidget(self.cache_dataset_check)

            return tab
        except Exception as e:
            self.logger.error(
                f"Błąd tworzenia zakładki Preprocessing: {e}", exc_info=True
            )
            return QtWidgets.QWidget()

    def _create_monitoring_tab(self):
        try:
            self.logger.debug("Tworzenie zakładki Monitorowanie")
            tab = QtWidgets.QWidget()
            scroll_area = QtWidgets.QScrollArea()
            scroll_area.setWidgetResizable(True)
            content_widget = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(content_widget)  # Main layout for content

            # Metrics Group
            metrics_group = QtWidgets.QGroupBox("Metryki monitorowania")
            metrics_layout = (
                QtWidgets.QGridLayout()
            )  # Use GridLayout for better alignment
            self.accuracy_check = QtWidgets.QCheckBox("Accuracy")
            self.accuracy_check.setChecked(True)
            self.precision_check = QtWidgets.QCheckBox("Precision (macro)")
            self.precision_check.setChecked(True)
            self.recall_check = QtWidgets.QCheckBox("Recall (macro)")
            self.recall_check.setChecked(True)
            self.f1_check = QtWidgets.QCheckBox("F1-score (macro)")
            self.f1_check.setChecked(True)
            self.topk_edit = QtWidgets.QLineEdit()  # Changed from QCheckBox
            self.topk_edit.setPlaceholderText("np. 1,3,5 (puste = brak)")
            self.confusion_matrix_check = QtWidgets.QCheckBox(
                "Macierz pomyłek (na koniec)"
            )
            self.auc_check = QtWidgets.QCheckBox(
                "AUC-ROC (macro, jeśli binarne/multiclass)"
            )
            self.gpu_util_check = QtWidgets.QCheckBox("Monitoruj użycie GPU (%)")
            self.memory_usage_check = QtWidgets.QCheckBox(
                "Monitoruj użycie pamięci GPU (MiB)"
            )
            metrics_layout.addWidget(self.accuracy_check, 0, 0)
            metrics_layout.addWidget(self.precision_check, 0, 1)
            metrics_layout.addWidget(self.recall_check, 1, 0)
            metrics_layout.addWidget(self.f1_check, 1, 1)
            metrics_layout.addWidget(
                QtWidgets.QLabel("Top-K Accuracy (wartości K):"), 2, 0
            )
            metrics_layout.addWidget(self.topk_edit, 2, 1)
            metrics_layout.addWidget(self.confusion_matrix_check, 3, 0)
            metrics_layout.addWidget(self.auc_check, 3, 1)
            metrics_layout.addWidget(self.gpu_util_check, 4, 0)
            metrics_layout.addWidget(self.memory_usage_check, 4, 1)
            metrics_group.setLayout(metrics_layout)
            layout.addWidget(metrics_group)

            # Logging Group (TensorBoard, W&B, CSV)
            logging_group = QtWidgets.QGroupBox("Logowanie zewnętrzne i częstotliwość")
            logging_layout = QtWidgets.QFormLayout()
            self.use_tensorboard_check = QtWidgets.QCheckBox("Używaj TensorBoard")
            self.use_tensorboard_check.setChecked(True)
            logging_layout.addRow(self.use_tensorboard_check)
            self.tensorboard_dir_edit = QtWidgets.QLineEdit("logs/tensorboard")
            logging_layout.addRow(
                "Katalog logów TensorBoard:", self.tensorboard_dir_edit
            )

            self.use_wandb_check = QtWidgets.QCheckBox("Używaj Weights & Biases (W&B)")
            logging_layout.addRow(self.use_wandb_check)
            self.wandb_project_edit = QtWidgets.QLineEdit()
            self.wandb_project_edit.setPlaceholderText("Nazwa projektu W&B")
            logging_layout.addRow("Projekt W&B:", self.wandb_project_edit)
            self.wandb_entity_edit = QtWidgets.QLineEdit()
            self.wandb_entity_edit.setPlaceholderText(
                "Nazwa użytkownika/teamu W&B (opcjonalnie)"
            )
            logging_layout.addRow("Encja W&B:", self.wandb_entity_edit)

            self.save_logs_check = QtWidgets.QCheckBox("Zapisuj metryki do pliku CSV")
            self.save_logs_check.setChecked(True)  # Existing
            logging_layout.addRow(self.save_logs_check)
            # self.csv_log_path_edit = QtWidgets.QLineEdit("logs/training_metrics.csv") # If specific path needed
            # logging_layout.addRow("Ścieżka pliku CSV:", self.csv_log_path_edit)

            self.logging_freq_combo = QtWidgets.QComboBox()
            self.logging_freq_combo.addItems(
                ["epoch", "step"]
            )  # "batch" is often same as "step"
            logging_layout.addRow(
                "Częstotliwość logowania (metryk):", self.logging_freq_combo
            )
            logging_group.setLayout(logging_layout)
            layout.addWidget(logging_group)

            # Visualization Group
            viz_group = QtWidgets.QGroupBox("Wizualizacje (pod koniec treningu)")
            viz_layout = QtWidgets.QFormLayout()
            self.viz_gradcam_check = QtWidgets.QCheckBox("Generuj mapy Grad-CAM")
            viz_layout.addRow(self.viz_gradcam_check)
            self.viz_feature_maps_check = QtWidgets.QCheckBox(
                "Wizualizuj mapy cech (wybrane warstwy)"
            )
            viz_layout.addRow(self.viz_feature_maps_check)
            self.viz_pred_samples_check = QtWidgets.QCheckBox(
                "Wizualizuj przykłady predykcji (poprawne/błędne)"
            )
            viz_layout.addRow(self.viz_pred_samples_check)
            self.viz_num_samples_spin = QtWidgets.QSpinBox()
            self.viz_num_samples_spin.setRange(1, 100)
            self.viz_num_samples_spin.setValue(16)
            viz_layout.addRow(
                "Liczba przykładów do wizualizacji:", self.viz_num_samples_spin
            )
            viz_group.setLayout(viz_layout)
            layout.addWidget(viz_group)

            # Early Stopping Group
            early_stop_group = QtWidgets.QGroupBox("Early Stopping")
            early_stop_layout = QtWidgets.QFormLayout()
            self.use_early_stopping_check = QtWidgets.QCheckBox("Włącz early stopping")
            self.use_early_stopping_check.setChecked(True)
            self.use_early_stopping_check.stateChanged.connect(
                self._toggle_early_stopping_controls
            )
            early_stop_layout.addRow(self.use_early_stopping_check)
            self.monitor_combo = QtWidgets.QComboBox()
            self.monitor_combo.addItems(
                [
                    "val_loss",
                    "val_accuracy",
                    "val_f1",
                    "val_precision",
                    "val_recall",
                    "train_loss",
                ]
            )
            early_stop_layout.addRow("Monitorowana metryka:", self.monitor_combo)
            self.patience_spin = QtWidgets.QSpinBox()
            self.patience_spin.setRange(1, 100)
            self.patience_spin.setValue(10)
            early_stop_layout.addRow("Cierpliwość (epoki):", self.patience_spin)
            self.min_delta_spin = QtWidgets.QDoubleSpinBox()
            self.min_delta_spin.setDecimals(5)
            self.min_delta_spin.setRange(0.0, 0.1)
            self.min_delta_spin.setValue(0.0001)
            early_stop_layout.addRow("Minimalna delta:", self.min_delta_spin)
            self.early_stopping_mode_combo = QtWidgets.QComboBox()
            self.early_stopping_mode_combo.addItems(
                ["min", "max"]
            )  # min for loss, max for accuracy
            early_stop_layout.addRow("Tryb (min/max):", self.early_stopping_mode_combo)
            early_stop_group.setLayout(early_stop_layout)
            layout.addWidget(early_stop_group)

            # Checkpointing Group
            checkpoint_group = QtWidgets.QGroupBox("Checkpointowanie")
            checkpoint_layout = QtWidgets.QFormLayout()
            self.use_checkpointing_check = QtWidgets.QCheckBox(
                "Zapisuj checkpointy modelu"
            )
            self.use_checkpointing_check.setChecked(True)
            checkpoint_layout.addRow(self.use_checkpointing_check)

            # model_dir_edit (existing, in this tab as per original Python) for general save path
            self.model_dir_edit = QtWidgets.QLineEdit(
                "output_models"
            )  # Define here if it was not before
            model_dir_btn = QtWidgets.QPushButton("Przeglądaj...")
            # model_dir_btn.clicked.connect(lambda: self._select_dir_for_lineedit(self.model_dir_edit, "Wybierz katalog zapisu modeli"))
            model_dir_layout = QtWidgets.QHBoxLayout()
            model_dir_layout.addWidget(self.model_dir_edit)
            model_dir_layout.addWidget(model_dir_btn)
            checkpoint_layout.addRow(
                "Główny katalog zapisu modeli:", model_dir_layout
            )  # This might be general output

            self.checkpoint_save_dir_edit = QtWidgets.QLineEdit(
                "checkpoints"
            )  # Relative to main model_dir or absolute
            checkpoint_layout.addRow(
                "Podkatalog dla checkpointów:", self.checkpoint_save_dir_edit
            )
            self.checkpoint_filename_edit = QtWidgets.QLineEdit(
                "{epoch:02d}-{val_loss:.4f}"
            )
            self.checkpoint_filename_edit.setToolTip(
                "Dostępne zmienne: {epoch}, {step}, oraz metryki np. {val_loss}"
            )
            checkpoint_layout.addRow(
                "Format nazwy pliku checkpointu:", self.checkpoint_filename_edit
            )

            self.checkpoint_metric_combo = (
                QtWidgets.QComboBox()
            )  # Same as monitor_combo items
            self.checkpoint_metric_combo.addItems(
                ["val_loss", "val_accuracy", "val_f1", "train_loss"]
            )
            checkpoint_layout.addRow(
                "Metryka do monitorowania:", self.checkpoint_metric_combo
            )
            self.best_only_check = QtWidgets.QCheckBox(
                "Zapisuj tylko najlepszy model wg. metryki"
            )
            self.best_only_check.setChecked(True)
            checkpoint_layout.addRow(self.best_only_check)
            self.checkpoint_mode_combo = QtWidgets.QComboBox()
            self.checkpoint_mode_combo.addItems(["min", "max"])
            checkpoint_layout.addRow(
                "Tryb zapisu (min/max):", self.checkpoint_mode_combo
            )
            self.save_freq_spin = (
                QtWidgets.QSpinBox()
            )  # Save every N epochs/steps (depends on trainer)
            self.save_freq_spin.setRange(1, 100)
            self.save_freq_spin.setValue(1)
            checkpoint_layout.addRow(
                "Częstość zapisu (epoki/kroki):", self.save_freq_spin
            )
            self.checkpoint_top_k_spin = QtWidgets.QSpinBox()
            self.checkpoint_top_k_spin.setRange(1, 10)
            self.checkpoint_top_k_spin.setValue(1)
            checkpoint_layout.addRow(
                "Liczba najlepszych modeli do zachowania:", self.checkpoint_top_k_spin
            )
            checkpoint_group.setLayout(checkpoint_layout)
            layout.addWidget(checkpoint_group)

            # Final part of original tab (Zapis modelu i logów)
            # self.save_logs_check is now in logging_group

            scroll_area.setWidget(content_widget)
            main_tab_layout = QtWidgets.QVBoxLayout(tab)
            main_tab_layout.addWidget(scroll_area)

            return tab
        except Exception as e:
            self.logger.error(
                f"Błąd tworzenia zakładki Monitorowanie: {e}", exc_info=True
            )
            return QtWidgets.QWidget()

    def _create_advanced_tab(self):
        try:
            self.logger.debug("Tworzenie zakładki Zaawansowane")
            tab = QtWidgets.QWidget()
            scroll_area = QtWidgets.QScrollArea()
            scroll_area.setWidgetResizable(True)
            content_widget = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(content_widget)  # Main layout for content

            # General Advanced Settings
            general_adv_group = QtWidgets.QGroupBox("Ogólne ustawienia zaawansowane")
            general_adv_layout = QtWidgets.QFormLayout()
            self.seed_spin = QtWidgets.QSpinBox()
            self.seed_spin.setRange(0, 2147483647)
            self.seed_spin.setValue(42)
            general_adv_layout.addRow(
                "Ziarno losowości (0 dla losowego):", self.seed_spin
            )
            self.deterministic_check = QtWidgets.QCheckBox(
                "Używaj algorytmów deterministycznych (wolniej)"
            )
            general_adv_layout.addRow(self.deterministic_check)

            class_weights_layout = QtWidgets.QHBoxLayout()
            self.class_weights_combo = QtWidgets.QComboBox()
            self.class_weights_combo.addItems(["none", "balanced", "custom"])
            self.class_weights_custom_edit = QtWidgets.QLineEdit()
            self.class_weights_custom_edit.setPlaceholderText(
                'np. {"klasa0": 0.5, "klasa1": 2.0}'
            )
            self.class_weights_custom_edit.setVisible(False)  # Show when "custom"
            self.class_weights_combo.currentTextChanged.connect(
                lambda text: self.class_weights_custom_edit.setVisible(text == "custom")
            )
            class_weights_layout.addWidget(self.class_weights_combo)
            class_weights_layout.addWidget(self.class_weights_custom_edit)
            general_adv_layout.addRow("Wagi klas:", class_weights_layout)

            self.sampler_combo = QtWidgets.QComboBox()
            self.sampler_combo.addItems(
                ["random", "weighted_random", "sequential"]
            )  # default random/shuffle for train
            general_adv_layout.addRow(
                "Sampler danych treningowych:", self.sampler_combo
            )
            self.image_channels_spin = QtWidgets.QSpinBox()
            self.image_channels_spin.setRange(1, 4)
            self.image_channels_spin.setValue(
                3
            )  # 1 for grayscale, 3 for RGB, 4 for RGBA
            general_adv_layout.addRow(
                "Liczba kanałów obrazu:", self.image_channels_spin
            )
            general_adv_group.setLayout(general_adv_layout)
            layout.addWidget(general_adv_group)

            # Scheduler advanced params (ReduceLROnPlateau specific)
            adv_scheduler_group = QtWidgets.QGroupBox(
                "Zaawansowane parametry harmonogramu (np. dla ReduceLROnPlateau)"
            )
            adv_scheduler_layout = QtWidgets.QFormLayout()
            self.scheduler_patience_spin = (
                QtWidgets.QSpinBox()
            )  # Renamed from scheduler_patience
            self.scheduler_patience_spin.setRange(1, 50)
            self.scheduler_patience_spin.setValue(10)
            adv_scheduler_layout.addRow(
                "Patience (ReduceLROnPlateau):", self.scheduler_patience_spin
            )
            self.scheduler_factor_spin = (
                QtWidgets.QDoubleSpinBox()
            )  # Renamed from scheduler_factor
            self.scheduler_factor_spin.setRange(0.01, 0.9)
            self.scheduler_factor_spin.setValue(0.1)
            adv_scheduler_layout.addRow(
                "Factor (ReduceLROnPlateau):", self.scheduler_factor_spin
            )
            self.min_lr_spin = QtWidgets.QDoubleSpinBox()  # Renamed from min_lr
            self.min_lr_spin.setDecimals(8)
            self.min_lr_spin.setRange(1e-9, 1e-3)
            self.min_lr_spin.setValue(1e-7)
            adv_scheduler_layout.addRow("Min LR (ReduceLROnPlateau):", self.min_lr_spin)
            self.scheduler_cooldown_spin = (
                QtWidgets.QSpinBox()
            )  # Renamed from scheduler_cooldown
            self.scheduler_cooldown_spin.setRange(0, 50)
            self.scheduler_cooldown_spin.setValue(0)
            adv_scheduler_layout.addRow(
                "Cooldown (ReduceLROnPlateau):", self.scheduler_cooldown_spin
            )
            adv_scheduler_group.setLayout(adv_scheduler_layout)
            layout.addWidget(adv_scheduler_group)

            # Transfer Learning Group (existing widgets)
            transfer_group = QtWidgets.QGroupBox("Transfer Learning i Fine-tuning")
            transfer_layout = QtWidgets.QFormLayout()
            self.freeze_base_model_check = QtWidgets.QCheckBox(
                "Zamroź wagi modelu bazowego (szkieletu)"
            )
            self.freeze_base_model_check.setChecked(True)  # Renamed
            transfer_layout.addRow(self.freeze_base_model_check)
            self.unfreeze_layers_edit = QtWidgets.QLineEdit()  # Renamed
            self.unfreeze_layers_edit.setPlaceholderText(
                "np. last_2, specific_layer_name, all, none"
            )
            transfer_layout.addRow(
                "Warstwy do odmrożenia (lub 'all', 'last_N'):",
                self.unfreeze_layers_edit,
            )
            self.unfreeze_strategy_combo = QtWidgets.QComboBox()  # Renamed
            self.unfreeze_strategy_combo.addItems(
                [
                    f"Wszystkie na raz ({self.UNFREEZE_ALL})",
                    f"Stopniowo od końca ({self.UNFREEZE_GRADUAL_END})",
                    f"Stopniowo od początku ({self.UNFREEZE_GRADUAL_START})",
                    f"Po określonej liczbie epok ({self.UNFREEZE_AFTER_EPOCHS})",  # Corrected typo in display
                ]
            )
            transfer_layout.addRow(
                "Strategia odmrażania:", self.unfreeze_strategy_combo
            )
            self.unfreeze_after_epochs_spin = (
                QtWidgets.QSpinBox()
            )  # Added from Training Params MD
            self.unfreeze_after_epochs_spin.setRange(0, 1000)
            self.unfreeze_after_epochs_spin.setValue(
                0
            )  # 0 means not used or unfreeze from start if strategy is 'all'
            transfer_layout.addRow(
                "Odmroź po N epokach (dla strategii 'po epokach'):",
                self.unfreeze_after_epochs_spin,
            )
            self.frozen_lr_spin = (
                QtWidgets.QDoubleSpinBox()
            )  # Added from Training Params MD
            self.frozen_lr_spin.setDecimals(7)
            self.frozen_lr_spin.setRange(0, 0.1)
            self.frozen_lr_spin.setValue(1e-5)
            transfer_layout.addRow(
                "LR dla zamrożonych warstw (jeśli różne):", self.frozen_lr_spin
            )
            self.unfrozen_lr_spin = (
                QtWidgets.QDoubleSpinBox()
            )  # Added from Training Params MD
            self.unfrozen_lr_spin.setDecimals(7)
            self.unfrozen_lr_spin.setRange(0, 0.1)
            self.unfrozen_lr_spin.setValue(1e-4)
            transfer_layout.addRow("LR dla odmrożonych warstw:", self.unfrozen_lr_spin)
            transfer_group.setLayout(transfer_layout)
            layout.addWidget(transfer_group)

            # Weights Initialization (existing)
            weights_group = QtWidgets.QGroupBox(
                "Inicjalizacja wag (dla nowej głowicy/warstw)"
            )
            weights_layout = QtWidgets.QFormLayout()
            self.init_weights_combo = QtWidgets.QComboBox()  # Renamed
            self.init_weights_combo.addItems(
                [
                    "kaiming_normal",
                    "kaiming_uniform",
                    "xavier_normal",
                    "xavier_uniform",
                    "orthogonal",
                    "default_pytorch",
                ]
            )
            weights_layout.addRow("Metoda inicjalizacji:", self.init_weights_combo)
            self.freeze_layers_check = QtWidgets.QCheckBox(
                "Zamroź dodatkowe warstwy CNN (jeśli niestandardowe)"
            )  # freeze_layers is existing
            weights_layout.addRow(
                self.freeze_layers_check
            )  # This seems very specific, ensure it's used
            weights_group.setLayout(weights_layout)
            layout.addWidget(weights_group)

            # Cross-validation (existing)
            cv_group = QtWidgets.QGroupBox("Walidacja krzyżowa (Cross-validation)")
            cv_layout = QtWidgets.QFormLayout()
            self.use_cv_check = QtWidgets.QCheckBox(
                "Używaj walidacji krzyżowej"
            )  # use_cv is existing
            cv_layout.addRow(self.use_cv_check)
            self.cv_folds_spin = QtWidgets.QSpinBox()  # cv_folds is existing
            self.cv_folds_spin.setRange(2, 100)
            self.cv_folds_spin.setValue(5)
            cv_layout.addRow("Liczba foldów (K-folds):", self.cv_folds_spin)
            cv_group.setLayout(cv_layout)
            layout.addWidget(cv_group)

            # Distributed Training (existing + new)
            dist_group = QtWidgets.QGroupBox("Trening dystrybuowany")
            dist_layout = QtWidgets.QFormLayout()
            self.use_dist_check = QtWidgets.QCheckBox(
                "Używaj treningu dystrybuowanego (DDP)"
            )  # use_dist is existing
            dist_layout.addRow(self.use_dist_check)
            self.dist_backend_combo = QtWidgets.QComboBox()  # dist_backend is existing
            self.dist_backend_combo.addItems(["nccl", "gloo", "mpi"])
            dist_layout.addRow("Backend:", self.dist_backend_combo)
            # self.dist_strategy (removed as DDP is common, or add if needed)
            self.dist_world_size_spin = QtWidgets.QSpinBox()
            self.dist_world_size_spin.setRange(1, 128)
            self.dist_world_size_spin.setValue(1)  # Num GPUs/processes
            dist_layout.addRow(
                "World Size (liczba procesów):", self.dist_world_size_spin
            )
            self.dist_rank_spin = QtWidgets.QSpinBox()
            self.dist_rank_spin.setRange(0, 127)
            self.dist_rank_spin.setValue(0)  # Rank of current process
            dist_layout.addRow(
                "Rank procesu (dla manualnego startu):", self.dist_rank_spin
            )
            dist_group.setLayout(dist_layout)
            layout.addWidget(dist_group)

            # Test Time Augmentation (TTA)
            tta_group = QtWidgets.QGroupBox("Test Time Augmentation (TTA)")
            tta_layout = QtWidgets.QFormLayout()
            self.tta_check = QtWidgets.QCheckBox(
                "Używaj TTA podczas ewaluacji/predykcji"
            )
            tta_layout.addRow(self.tta_check)
            self.tta_num_aug_spin = QtWidgets.QSpinBox()
            self.tta_num_aug_spin.setRange(1, 64)
            self.tta_num_aug_spin.setValue(5)
            tta_layout.addRow("Liczba augmentacji TTA:", self.tta_num_aug_spin)
            tta_group.setLayout(tta_layout)
            layout.addWidget(tta_group)

            # Export and Quantization
            export_quant_group = QtWidgets.QGroupBox(
                "Eksport i Kwantyzacja Modelu (po treningu)"
            )
            export_quant_layout = QtWidgets.QFormLayout()
            self.export_onnx_check = QtWidgets.QCheckBox("Eksportuj model do ONNX")
            export_quant_layout.addRow(self.export_onnx_check)
            self.quantization_check = QtWidgets.QCheckBox(
                "Przeprowadź kwantyzację (np. PTQ)"
            )
            export_quant_layout.addRow(self.quantization_check)
            self.quantization_precision_combo = QtWidgets.QComboBox()
            self.quantization_precision_combo.addItems(["int8", "float16"])
            export_quant_layout.addRow(
                "Precyzja kwantyzacji:", self.quantization_precision_combo
            )
            export_quant_group.setLayout(export_quant_layout)
            layout.addWidget(export_quant_group)

            # Catastrophic Forgetting Prevention (stub, very large)
            cfp_group = QtWidgets.QGroupBox(
                "Zapobieganie Katastroficznemu Zapominaniu (Continual Learning)"
            )
            cfp_layout = QtWidgets.QFormLayout()
            self.cfp_enable_check = QtWidgets.QCheckBox(
                "Włącz mechanizmy zapobiegania zapominaniu"
            )
            cfp_layout.addRow(self.cfp_enable_check)
            # ... Many more fields here based on MD ...
            # For brevity, I'll skip fully implementing all CFP sub-fields in this response
            # but they would follow the pattern of creating checkboxes, spinboxes, etc.
            cfp_group.setLayout(cfp_layout)
            cfp_group.setToolTip(
                "Zaawansowane ustawienia dla scenariuszy uczenia ciągłego."
            )
            layout.addWidget(cfp_group)

            # Online validation (existing)
            online_val_group = QtWidgets.QGroupBox("Walidacja online (w trakcie epoki)")
            online_val_layout = QtWidgets.QFormLayout()
            self.use_online_val_check = QtWidgets.QCheckBox(
                "Używaj walidacji online (co N kroków)"
            )  # use_online_val existing
            online_val_layout.addRow(self.use_online_val_check)
            self.online_val_freq_spin = QtWidgets.QSpinBox()  # online_val_freq existing
            self.online_val_freq_spin.setRange(10, 10000)
            self.online_val_freq_spin.setValue(100)
            online_val_layout.addRow(
                "Częstość walidacji online (kroki):", self.online_val_freq_spin
            )
            online_val_group.setLayout(online_val_layout)
            layout.addWidget(online_val_group)

            scroll_area.setWidget(content_widget)
            main_tab_layout = QtWidgets.QVBoxLayout(tab)
            main_tab_layout.addWidget(scroll_area)

            return tab
        except Exception as e:
            self.logger.error(
                f"Błąd tworzenia zakładki Zaawansowane: {e}", exc_info=True
            )
            return QtWidgets.QWidget()

    def _create_optimization_tab(self):
        try:
            self.logger.debug("Tworzenie zakładki Optymalizacja treningu")
            tab = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(tab)
            params_group = QtWidgets.QGroupBox(
                "Parametry optymalizacyjne (sprzętowo-zależne)"
            )
            params_layout = QtWidgets.QFormLayout()

            # Parameters for _create_parameter_row: (name, key, default, type, min, max, step)
            # Standard ones from original code: batch_size, num_workers, use_mixed_precision
            # Added from MD:
            opt_params_definitions = [
                ("Batch size", "batch_size", 32, "int", 1, 2048, 1),
                (
                    "Liczba wątków (Workers)",
                    "num_workers",
                    4,
                    "int",
                    0,
                    os.cpu_count() or 32,
                    1,
                ),
                (
                    "Mieszana precyzja (Mixed Precision)",
                    "use_mixed_precision",
                    True,
                    "bool",
                    None,
                    None,
                    None,
                ),
                (
                    "Optymalizacja zużycia pamięci",
                    "memory_efficient",
                    False,
                    "bool",
                    None,
                    None,
                    None,
                ),
                ("Benchmark cuDNN", "cudnn_benchmark", True, "bool", None, None, None),
                (
                    "Przypinanie pamięci (Pin Memory)",
                    "pin_memory",
                    True,
                    "bool",
                    None,
                    None,
                    None,
                ),
                (
                    "Mieszaj dane (Dataloader Shuffle)",
                    "dataloader_shuffle",
                    True,
                    "bool",
                    None,
                    None,
                    None,
                ),
                (
                    "Współczynnik prefetch (Dataloader)",
                    "dataloader_prefetch_factor",
                    2,
                    "int",
                    0,
                    16,
                    1,
                ),
                (
                    "Trwałe wątki (Dataloader Persistent Workers)",
                    "dataloader_persistent_workers",
                    False,
                    "bool",
                    None,
                    None,
                    None,
                ),
                (
                    "Pomiń ostatni batch (Dataloader Drop Last)",
                    "dataloader_drop_last",
                    False,
                    "bool",
                    None,
                    None,
                    None,
                ),
            ]

            self.optimization_params_list = []  # Store keys for easier access if needed
            for (
                name,
                key,
                default,
                type_,
                min_v,
                max_v,
                step_v,
            ) in opt_params_definitions:
                row_layout = self._create_parameter_row(
                    name, key, default, type_, min_v, max_v, step_v
                )
                params_layout.addRow(name + ":", row_layout)
                self.optimization_params_list.append(key)

            params_group.setLayout(params_layout)
            layout.addWidget(params_group)

            apply_all_btn = QtWidgets.QPushButton(
                "Zastosuj wszystkie optymalizacje z profilu sprzętowego"
            )
            apply_all_btn.clicked.connect(self._apply_all_hardware_optimizations)
            layout.addWidget(apply_all_btn)
            return tab
        except Exception as e:
            self.logger.error(
                f"Błąd tworzenia zakładki Optymalizacja: {e}", exc_info=True
            )
            return QtWidgets.QWidget()

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
        value_widget = None

        if widget_type == "int":
            value_widget = QtWidgets.QSpinBox()
            value_widget.setRange(
                min_val if min_val is not None else -999999,
                max_val if max_val is not None else 999999,
            )
            value_widget.setValue(default_value)
            if step is not None:
                value_widget.setSingleStep(step)
        elif widget_type == "float":
            value_widget = QtWidgets.QDoubleSpinBox()
            value_widget.setRange(
                min_val if min_val is not None else -999999.0,
                max_val if max_val is not None else 999999.0,
            )
            value_widget.setValue(default_value)
            if step is not None:
                value_widget.setSingleStep(step)
        elif widget_type == "bool":
            value_widget = QtWidgets.QCheckBox()
            value_widget.setChecked(default_value)
        else:  # Default to QLineEdit for string or other types
            value_widget = QtWidgets.QLineEdit(str(default_value))

        value_widget.setObjectName(f"opt_{param_key}_value_widget")
        layout.addWidget(value_widget)

        # User choice (default)
        user_checkbox = QtWidgets.QCheckBox("Użytkownika")
        user_checkbox.setChecked(True)
        user_checkbox.setObjectName(f"opt_{param_key}_user_checkbox")
        layout.addWidget(user_checkbox)

        # Hardware profile suggestion
        layout.addWidget(QtWidgets.QLabel("Profil sprzętowy:"))

        # Map internal param_key to potential hardware_profile keys
        hw_profile_key_map = {
            "batch_size": "recommended_batch_size",
            "num_workers": "recommended_workers",
            "use_mixed_precision": "use_mixed_precision",
            # Add other mappings if hardware profile uses different names
        }
        actual_hw_key = hw_profile_key_map.get(param_key, param_key)
        hw_value_from_profile = self.hardware_profile.get(actual_hw_key)

        hw_value_display_text = (
            str(hw_value_from_profile) if hw_value_from_profile is not None else "Brak"
        )
        hw_value_label = QtWidgets.QLabel(hw_value_display_text)
        hw_value_label.setObjectName(f"opt_{param_key}_hw_label")
        layout.addWidget(hw_value_label)

        hw_checkbox = QtWidgets.QCheckBox("Użyj z profilu")
        hw_checkbox.setObjectName(f"opt_{param_key}_hw_checkbox")
        hw_checkbox.setEnabled(
            hw_value_from_profile is not None
        )  # Enable only if value exists
        layout.addWidget(hw_checkbox)

        source_group = QtWidgets.QButtonGroup(
            self
        )  # Pass parent to avoid early collection
        source_group.addButton(user_checkbox)
        source_group.addButton(hw_checkbox)
        source_group.setExclusive(True)

        row_data = {
            "param_key": param_key,
            "value_widget": value_widget,
            "user_checkbox": user_checkbox,
            "hw_value_label": hw_value_label,  # For potential updates if profile changes
            "hw_value_actual": hw_value_from_profile,  # Store the actual value
            "hw_checkbox": hw_checkbox,
            "button_group": source_group,
        }

        user_checkbox.toggled.connect(
            lambda checked, rd=row_data: self._on_source_toggle(rd, checked, "user")
        )
        hw_checkbox.toggled.connect(
            lambda checked, rd=row_data: self._on_source_toggle(rd, checked, "hw")
        )

        self.parameter_rows[param_key] = row_data
        return layout

    def _on_source_toggle(self, row_data, is_selected, source_type):
        value_widget = row_data["value_widget"]
        user_checkbox = row_data["user_checkbox"]
        hw_checkbox = row_data["hw_checkbox"]
        hw_value_actual = row_data["hw_value_actual"]

        if source_type == "user" and is_selected:
            value_widget.setEnabled(True)
            if (
                hw_checkbox.isChecked()
            ):  # Ensure hw_checkbox is unchecked without re-triggering
                hw_checkbox.blockSignals(True)
                hw_checkbox.setChecked(False)
                hw_checkbox.blockSignals(False)
        elif source_type == "hw" and is_selected:
            value_widget.setEnabled(False)  # Value comes from hw_profile
            if user_checkbox.isChecked():  # Ensure user_checkbox is unchecked
                user_checkbox.blockSignals(True)
                user_checkbox.setChecked(False)
                user_checkbox.blockSignals(False)

            # Apply hw_value_actual to value_widget
            if hw_value_actual is not None:
                if isinstance(
                    value_widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)
                ):
                    value_widget.setValue(hw_value_actual)
                elif isinstance(value_widget, QtWidgets.QCheckBox):
                    value_widget.setChecked(bool(hw_value_actual))
                elif isinstance(value_widget, QtWidgets.QLineEdit):
                    value_widget.setText(str(hw_value_actual))

        # If user unchecks their box, and hw is not checked, re-enable widget (or decide behavior)
        if not user_checkbox.isChecked() and not hw_checkbox.isChecked():
            value_widget.setEnabled(
                True
            )  # Default to user editable if nothing is explicitly selected

    def _apply_all_hardware_optimizations(self):
        count = 0
        for param_key, row_widgets in self.parameter_rows.items():
            if (
                row_widgets["hw_value_actual"] is not None
                and row_widgets["hw_checkbox"].isEnabled()
            ):
                # Simulate clicking the hw_checkbox
                if not row_widgets["hw_checkbox"].isChecked():
                    row_widgets["hw_checkbox"].setChecked(
                        True
                    )  # This will trigger _on_source_toggle
                else:  # If already checked, ensure value is applied (e.g. if default was different)
                    self._on_source_toggle(row_widgets, True, "hw")
                count += 1
        if count > 0:
            QtWidgets.QMessageBox.information(
                self,
                "Sukces",
                f"Zastosowano {count} optymalnych ustawień z profilu sprzętowego.",
            )
        else:
            QtWidgets.QMessageBox.information(
                self,
                "Informacja",
                "Brak dostępnych optymalizacji w profilu sprzętowym lub już zastosowano.",
            )

    def _get_unfreeze_strategy_value(self, display_text):
        # Assumes display_text contains the key like "(unfreeze_all)"
        if f"({self.UNFREEZE_ALL})" in display_text:
            return self.UNFREEZE_ALL
        if f"({self.UNFREEZE_GRADUAL_END})" in display_text:
            return self.UNFREEZE_GRADUAL_END
        if f"({self.UNFREEZE_GRADUAL_START})" in display_text:
            return self.UNFREEZE_GRADUAL_START
        if f"({self.UNFREEZE_AFTER_EPOCHS})" in display_text:
            return self.UNFREEZE_AFTER_EPOCHS
        self.logger.warning(f"Nieznana strategia odmrażania w tekście: {display_text}")
        return self.UNFREEZE_ALL

    def _get_unfreeze_layers_value(self, value_str):
        value_str = value_str.strip()
        if not value_str or value_str.lower() == "all":
            return "all"
        if value_str.lower() == "none":
            return "none"
        if value_str.lower().startswith("last_"):
            try:
                num = int(value_str.split("_")[1])
                return f"last_{num}"
            except:
                return value_str  # Return as string if parsing fails
        # Could also parse comma-separated layer names/indices here if needed
        return value_str  # As fallback, return the string itself (e.g. for specific layer names)

    def _toggle_early_stopping_controls(self, state):
        enabled = bool(state)
        self.monitor_combo.setEnabled(enabled)
        self.patience_spin.setEnabled(enabled)
        self.min_delta_spin.setEnabled(enabled)
        self.early_stopping_mode_combo.setEnabled(enabled)

    def _on_accept(self):
        try:
            train_dir = self.train_dir_edit.text().strip()
            if not train_dir:
                QtWidgets.QMessageBox.warning(
                    self, "Błąd", "Katalog treningowy nie może być pusty."
                )
                self.tabs.setCurrentWidget(
                    self.tabs.widget(0)
                )  # Switch to Data/Model tab
                self.train_dir_edit.setFocus()
                return
            if not Path(train_dir).is_dir():
                QtWidgets.QMessageBox.warning(
                    self, "Błąd", "Podany katalog treningowy nie istnieje."
                )
                self.tabs.setCurrentWidget(self.tabs.widget(0))
                self.train_dir_edit.setFocus()
                return

            val_dir = self.val_dir_edit.text().strip()
            if (
                not val_dir and self.validation_split_spin.value() == 0.0
            ):  # If no val_dir and no split from train
                QtWidgets.QMessageBox.warning(
                    self,
                    "Błąd",
                    "Katalog walidacyjny nie może być pusty, jeśli podział z danych treningowych jest 0.",
                )
                self.tabs.setCurrentWidget(self.tabs.widget(0))
                self.val_dir_edit.setFocus()
                return
            if val_dir and not Path(val_dir).is_dir():
                QtWidgets.QMessageBox.warning(
                    self,
                    "Błąd",
                    "Podany katalog walidacyjny nie istnieje (jeśli podany).",
                )
                self.tabs.setCurrentWidget(self.tabs.widget(0))
                self.val_dir_edit.setFocus()
                return

            variant = (
                self.variant_combo.currentText() or self.arch_combo.currentText()
            )  # Fallback if variant empty
            num_classes = self.num_classes_spin.value()
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            task_name = (
                f"{variant.replace('-', '_')}_cls{num_classes}_{now}"  # Sanitize name
            )

            self.logger.info(
                f"=== TWORZENIE NOWEGO ZADANIA TRENINGOWEGO: {task_name} ==="
            )

            # Get full configuration from UI state
            current_task_config_dict = self._get_current_config_as_dict()

            self.task_config = {
                "name": task_name,
                "type": "training",
                "status": "Nowy",
                "priority": 0,  # Placeholder, could be set by user
                "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "train_dir": train_dir,  # Add these top-level for convenience if your system uses them
                "val_dir": val_dir,
                "config": current_task_config_dict,
            }

            # Log the generated config
            config_str = json.dumps(self.task_config, indent=2, ensure_ascii=False)
            self.logger.info(f"Pełna konfiguracja zadania:\n{config_str}")

            super().accept()  # Call QDialog's accept method

        except Exception as e:
            self.logger.error(
                f"Błąd podczas dodawania zadania: {str(e)}", exc_info=True
            )
            QtWidgets.QMessageBox.critical(
                self, "Błąd krytyczny", f"Nie można przygotować zadania: {str(e)}"
            )

    def get_task_config(self):
        return getattr(self, "task_config", None)

    def closeEvent(self, event):
        self.logger.info("Zamykanie okna dialogowego przez przycisk zamknięcia okna.")
        # Standard QDialog behavior on close button is reject.
        # If accept is desired, call self.accept() but usually it's reject.
        # self.reject() # This will be called by QDialog's default closeEvent if not overridden to accept.
        super().closeEvent(event)  # Call base class closeEvent

    def _show_hardware_profile(self):
        if not self.hardware_profile:
            QtWidgets.QMessageBox.information(
                self, "Profil sprzętowy", "Brak załadowanego profilu sprzętowego."
            )
            return
        dialog = HardwareProfileDialog(self.hardware_profile, self)
        dialog.exec()


if __name__ == "__main__":
    # Example usage:
    import sys

    # Mock DEFAULT_TRAINING_PARAMS if not available globally
    if "DEFAULT_TRAINING_PARAMS" not in globals():
        DEFAULT_TRAINING_PARAMS = {"max_epochs": 50}  # Example mock

    app = QtWidgets.QApplication(sys.argv)

    # Example hardware profile
    mock_hardware_profile = {
        "gpu_name": "NVIDIA GeForce RTX 3080",
        "gpu_memory_gb": 10,
        "cpu_cores": 8,
        "ram_gb": 32,
        "recommended_batch_size": 64,
        "recommended_workers": 4,
        "use_mixed_precision": True,
        "cudnn_benchmark": True,
        "pin_memory": True,
    }

    dialog = TrainingTaskConfigDialog(hardware_profile=mock_hardware_profile)
    if (
        dialog.exec()
    ):  # For QDialog, exec() returns True if accepted, False if rejected.
        task_cfg = dialog.get_task_config()
        if task_cfg:
            print("--- Konfiguracja zadania ---")
            print(json.dumps(task_cfg, indent=2, ensure_ascii=False))
            # Save to a file for inspection
            with open("last_task_config.json", "w", encoding="utf-8") as f:
                json.dump(task_cfg, f, indent=2, ensure_ascii=False)
            print("\nKonfiguracja zapisana do last_task_config.json")
    else:
        print("Dialog został zamknięty lub odrzucony.")
    sys.exit()
