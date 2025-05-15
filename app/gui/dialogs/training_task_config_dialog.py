import datetime
import json
import logging
import os
from pathlib import Path

from PyQt6 import QtCore, QtWidgets

from app.gui.dialogs.hardware_profile_dialog import HardwareProfileDialog
from app.utils.config import DEFAULT_TRAINING_PARAMS
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
    UNFREEZE_AFTER_EPOCHS = "unfreeze_after_epoochs"

    def __init__(self, parent=None, settings=None, hardware_profile=None):
        super().__init__(parent)
        self.settings = settings
        if not hardware_profile:
            # from app.profiler import HardwareProfiler  # Komentuję, bo brak modułu
            # profiler = HardwareProfiler()
            # self.hardware_profile = profiler.get_optimal_parameters()
            self.hardware_profile = {}  # Tymczasowo pusta konfiguracja
        else:
            self.hardware_profile = hardware_profile
        self._setup_logging()
        self.logger.info(f"Profil sprzętowy: {self.hardware_profile}")
        self.setWindowTitle("Konfiguracja treningu")
        self.setMinimumWidth(1000)
        self.profiles_dir = Path("data/profiles")
        self.profiles_dir.mkdir(exist_ok=True)
        self.current_profile = None
        self.setWindowFlags(
            self.windowFlags() | QtCore.Qt.WindowType.WindowCloseButtonHint
        )
        self._init_ui()

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
        """Inicjalizacja interfejsu użytkownika z zakładkami."""
        try:
            self.logger.debug("Rozpoczęcie inicjalizacji UI")
            layout = QtWidgets.QVBoxLayout(self)

            # Dodaj pasek narzędzi
            toolbar = QtWidgets.QToolBar("Narzędzia")
            if hasattr(self, "addToolBar"):
                self.addToolBar(QtWidgets.Qt.ToolBarArea.TopToolBarArea, toolbar)

            # Przycisk: Pokaż profil sprzętowy
            show_hw_profile_btn = QtWidgets.QPushButton("Pokaż profil sprzętowy")
            show_hw_profile_btn.clicked.connect(self._show_hardware_profile)
            layout.addWidget(show_hw_profile_btn)

            # Utworzenie zakładek
            self.tabs = QtWidgets.QTabWidget()

            # 1. Zakładka: Dane i Model
            tab = self._create_data_model_tab()
            self.tabs.addTab(tab, "Dane i Model")

            # 2. Zakładka: Parametry Treningu
            tab = self._create_training_params_tab()
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

    def _create_data_model_tab(self):
        """Tworzenie zakładki Dane i Model."""
        try:
            self.logger.debug("Tworzenie zakładki")
            self.optimization_params = []
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
            size_label = "Rozmiar obrazu:"
            form.addRow(size_label, self.input_size_spin)

            # Liczba klas
            self.num_classes_spin = QtWidgets.QSpinBox()
            self.num_classes_spin.setRange(2, 1000)
            self.num_classes_spin.setValue(2)
            form.addRow("Liczba klas:", self.num_classes_spin)

            # Pretrained
            self.pretrained_check = QtWidgets.QCheckBox(
                "Użyj wstępnie wytrenowanych wag"
            )
            self.pretrained_check.setChecked(True)
            form.addRow("", self.pretrained_check)

            # Pretrained weights
            self.pretrained_weights_combo = QtWidgets.QComboBox()
            self.pretrained_weights_combo.addItems(["imagenet", "custom"])
            form.addRow("Źródło wag:", self.pretrained_weights_combo)

            # Feature extraction only
            self.feature_extraction_check = QtWidgets.QCheckBox("Tylko ekstrakcja cech")
            self.feature_extraction_check.setChecked(False)
            form.addRow("", self.feature_extraction_check)

            # Activation
            self.activation_combo = QtWidgets.QComboBox()
            self.activation_combo.addItems(["swish", "relu", "sigmoid", "tanh"])
            form.addRow("Funkcja aktywacji:", self.activation_combo)

            # Dropout at inference
            self.dropout_at_inference_check = QtWidgets.QCheckBox(
                "Dropout podczas inferencji"
            )
            self.dropout_at_inference_check.setChecked(False)
            form.addRow("", self.dropout_at_inference_check)

            # Global pool
            self.global_pool_combo = QtWidgets.QComboBox()
            self.global_pool_combo.addItems(["avg", "max"])
            form.addRow("Global pooling:", self.global_pool_combo)

            # Last layer activation
            self.last_layer_activation_combo = QtWidgets.QComboBox()
            self.last_layer_activation_combo.addItems(["softmax", "sigmoid", "none"])
            form.addRow(
                "Aktywacja ostatniej warstwy:", self.last_layer_activation_combo
            )

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
            self.profile_hardware_required.setMaximumHeight(60)
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

            # --- Dodajemy parametry na końcu formularza jako zwykłe widgety ---
            self.lr_spin = QtWidgets.QDoubleSpinBox()
            self.lr_spin.setDecimals(6)
            self.lr_spin.setRange(0.000001, 1.0)
            self.lr_spin.setSingleStep(0.0001)
            self.lr_spin.setValue(0.001)
            form.addRow("Learning rate:", self.lr_spin)

            self.epochs_spin = QtWidgets.QSpinBox()
            self.epochs_spin.setRange(1, 1000)
            self.epochs_spin.setValue(100)
            form.addRow("Epochs:", self.epochs_spin)

            self.grad_accum_steps_spin = QtWidgets.QSpinBox()
            self.grad_accum_steps_spin.setRange(1, 16)
            self.grad_accum_steps_spin.setValue(1)
            form.addRow("Gradient Accumulation:", self.grad_accum_steps_spin)

            return tab

        except Exception as e:
            msg = "Błąd podczas tworzenia zakładki"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            raise

    def _refresh_profile_list(self):
        """Odświeża listę dostępnych profili."""
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
        """Obsługa wyboru profilu z listy."""
        if current is None:
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

    def _edit_profile(self):
        """Otwiera profil w edytorze tekstowym."""
        if not self.current_profile:
            QtWidgets.QMessageBox.warning(
                self, "Ostrzeżenie", "Najpierw wybierz profil do edycji."
            )
            return

        try:
            profile_path = (
                self.profiles_dir / f"{self.profile_list.currentItem().text()}.json"
            )
            os.startfile(str(profile_path))  # Dla Windows
        except Exception as e:
            self.logger.error(
                f"Błąd podczas otwierania profilu: {str(e)}", exc_info=True
            )
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Nie można otworzyć profilu: {str(e)}"
            )

    def _apply_profile(self):
        """Zastosowuje wybrany profil do konfiguracji."""
        try:
            config = self.current_profile.get("config", {})

            # Dane i Model
            if "model" in config:
                model_config = config["model"]
                self.arch_combo.setCurrentText(
                    model_config.get("architecture", "EfficientNet")
                )
                self.variant_combo.setCurrentText(
                    model_config.get("variant", "EfficientNet-B0")
                )
                self.input_size_spin.setValue(model_config.get("input_size", 224))
                self.num_classes_spin.setValue(model_config.get("num_classes", 2))
                self.pretrained_check.setChecked(model_config.get("pretrained", True))
                self.pretrained_weights_combo.setCurrentText(
                    model_config.get("pretrained_weights", "imagenet")
                )
                self.feature_extraction_check.setChecked(
                    model_config.get("feature_extraction_only", False)
                )
                self.activation_combo.setCurrentText(
                    model_config.get("activation", "swish")
                )
                self.dropout_at_inference_check.setChecked(
                    model_config.get("dropout_at_inference", False)
                )
                self.global_pool_combo.setCurrentText(
                    model_config.get("global_pool", "avg")
                )
                self.last_layer_activation_combo.setCurrentText(
                    model_config.get("last_layer_activation", "softmax")
                )

            # Parametry Treningu
            if "training" in config:
                training_config = config["training"]
                self.epochs_spin.setValue(training_config.get("epochs", 100))
                self.lr_spin.setValue(training_config.get("learning_rate", 0.001))
                self.frozen_lr_spin.setValue(training_config.get("frozen_lr", 0.0001))
                self.unfrozen_lr_spin.setValue(
                    training_config.get("unfrozen_lr", 0.001)
                )
                self.optimizer_combo.setCurrentText(
                    training_config.get("optimizer", "Adam")
                )
                if "scheduler" in training_config:
                    scheduler_config = training_config["scheduler"]
                    if isinstance(scheduler_config, dict):
                        self.scheduler_combo.setCurrentText(
                            scheduler_config.get("type", "None")
                        )
                    else:
                        self.scheduler_combo.setCurrentText(scheduler_config)
                self.scheduler_t0_spin.setValue(training_config.get("T_0", 10))
                self.scheduler_tmult_spin.setValue(training_config.get("T_mult", 2))
                self.scheduler_eta_min_spin.setValue(
                    training_config.get("eta_min", 0.000001)
                )
                self.warmup_epochs_spin.setValue(
                    training_config.get("warmup_epochs", 5)
                )
                self.warmup_lr_init_spin.setValue(
                    training_config.get("warmup_lr_init", 0.00001)
                )
                self.freeze_base_model.setChecked(
                    training_config.get("freeze_base_model", True)
                )
                self.evaluation_freq_spin.setValue(
                    training_config.get("evaluation_freq", 1)
                )
                self.unfreeze_layers.setText(training_config.get("unfreeze_layers", ""))
                self.use_ema_check.setChecked(training_config.get("use_ema", False))
                self.ema_decay_spin.setValue(training_config.get("ema_decay", 0.999))

                # Konwersja strategii odmrażania
                strategy = training_config.get("unfreeze_strategy", "")
                if "Po" in strategy and "epokach" in strategy:
                    self.unfreeze_strategy.setCurrentText(
                        "Po określonej liczbie epok (unfreeze_after_epoochs)"
                    )
                elif "Stopniowo" in strategy and "końca" in strategy:
                    self.unfreeze_strategy.setCurrentText(
                        "Stopniowo od końca (unfreeze_gradual_end)"
                    )
                elif "Stopniowo" in strategy and "początku" in strategy:
                    self.unfreeze_strategy.setCurrentText(
                        "Stopniowo od początku (unfreeze_gradual_start)"
                    )
                else:
                    self.unfreeze_strategy.setCurrentText(
                        "Wszystkie na raz (unfreeze_all)"
                    )

                if "unfreeze_after_epochs" in training_config:
                    self.unfreeze_after_epochs_spin.setValue(
                        training_config["unfreeze_after_epochs"]
                    )
                if "frozen_lr" in training_config:
                    self.frozen_lr_spin.setValue(training_config["frozen_lr"])
                if "unfrozen_lr" in training_config:
                    self.unfrozen_lr_spin.setValue(training_config["unfrozen_lr"])

            # Regularyzacja
            if "regularization" in config:
                reg_config = config["regularization"]
                self.weight_decay_spin.setValue(reg_config.get("weight_decay", 0.0001))
                self.gradient_clip_spin.setValue(reg_config.get("gradient_clip", 1.0))
                self.label_smoothing_spin.setValue(
                    reg_config.get("label_smoothing", 0.1)
                )
                # Poprawka: Sprawdź, czy wartość nie jest None
                drop_connect_rate = reg_config.get("drop_connect_rate")
                if drop_connect_rate is None:
                    drop_connect_rate = 0.2  # Domyślna wartość, jeśli None
                self.drop_connect_spin.setValue(drop_connect_rate)
                self.dropout_spin.setValue(reg_config.get("dropout_rate", 0.2))
                self.momentum_spin.setValue(reg_config.get("momentum", 0.9))
                self.epsilon_spin.setValue(reg_config.get("epsilon", 1e-6))
                self.use_swa_check.setChecked(
                    reg_config.get("swa", {}).get("use", False)
                )
                self.swa_start_epoch_spin.setValue(
                    reg_config.get("swa", {}).get("start_epoch", 10)
                )

            # Augmentacja
            if "augmentation" in config:
                aug_config = config["augmentation"]
                basic_config = aug_config.get("basic", {})
                self.basic_aug_check.setChecked(basic_config.get("use", False))
                self.rotation_spin.setValue(basic_config.get("rotation", 30))
                self.brightness_spin.setValue(basic_config.get("brightness", 0.2))
                self.contrast_spin.setValue(basic_config.get("contrast", 0.2))
                self.saturation_spin.setValue(basic_config.get("saturation", 0.2))
                self.hue_spin.setValue(basic_config.get("hue", 0.1))
                self.shift_spin.setValue(basic_config.get("shift", 0.1))
                self.zoom_spin.setValue(basic_config.get("zoom", 0.1))
                self.horizontal_flip_check.setChecked(
                    basic_config.get("horizontal_flip", True)
                )
                self.vertical_flip_check.setChecked(
                    basic_config.get("vertical_flip", False)
                )

                mixup_config = aug_config.get("mixup", {})
                self.mixup_check.setChecked(mixup_config.get("use", False))
                self.mixup_alpha_spin.setValue(mixup_config.get("alpha", 0.2))

                cutmix_config = aug_config.get("cutmix", {})
                self.cutmix_check.setChecked(cutmix_config.get("use", False))
                self.cutmix_alpha_spin.setValue(cutmix_config.get("alpha", 1.0))

            # Monitorowanie
            if "monitoring" in config:
                monitor_config = config["monitoring"]
                metrics_config = monitor_config.get("metrics", {})
                self.accuracy_check.setChecked(metrics_config.get("accuracy", True))
                self.precision_check.setChecked(metrics_config.get("precision", True))
                self.recall_check.setChecked(metrics_config.get("recall", True))
                self.f1_check.setChecked(metrics_config.get("f1", True))

                # Poprawiona obsługa topk
                topk_value = metrics_config.get("topk", False)
                if isinstance(topk_value, list):
                    self.topk_check.setChecked(len(topk_value) > 0)
                else:
                    self.topk_check.setChecked(bool(topk_value))

                self.confusion_matrix_check.setChecked(
                    metrics_config.get("confusion_matrix", False)
                )

                early_stop_config = monitor_config.get("early_stopping", {})
                self.patience_spin.setValue(early_stop_config.get("patience", 10))
                self.min_delta_spin.setValue(early_stop_config.get("min_delta", 0.001))
                self.monitor_combo.setCurrentText(
                    early_stop_config.get("monitor", "val_loss")
                )

                checkpoint_config = monitor_config.get("checkpointing", {})
                self.best_only_check.setChecked(
                    checkpoint_config.get("best_only", True)
                )
                self.save_freq_spin.setValue(checkpoint_config.get("save_frequency", 1))
                self.checkpoint_metric_combo.setCurrentText(
                    checkpoint_config.get("metric", "val_loss")
                )

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
                new_profile["type"] = "training"  # Upewniamy się, że typ jest ustawiony

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
        """Zapisuje aktualną konfigurację jako nowy profil."""
        try:
            name, ok = QtWidgets.QInputDialog.getText(
                self,
                "Zapisz profil",
                "Podaj nazwę dla nowego profilu:",
                QtWidgets.QLineEdit.EchoMode.Normal,
                f"{self.arch_combo.currentText()}_"
                f"{self.variant_combo.currentText()}",
            )

            if ok and name:
                profile_data = {
                    "type": "training",
                    "info": (
                        f"Profil dla {self.arch_combo.currentText()} "
                        f"{self.variant_combo.currentText()}"
                    ),
                    "description": "Profil utworzony przez użytkownika",
                    "data_required": "Standardowe dane treningowe",
                    "hardware_required": "Standardowy sprzęt",
                    "config": {
                        "model": {
                            "architecture": self.arch_combo.currentText(),
                            "variant": self.variant_combo.currentText(),
                            "input_size": self.input_size_spin.value(),
                            "num_classes": self.num_classes_spin.value(),
                            "pretrained": self.pretrained_check.isChecked(),
                            "pretrained_weights": (
                                self.pretrained_weights_combo.currentText()
                            ),
                            "feature_extraction_only": (
                                self.feature_extraction_check.isChecked()
                            ),
                            "activation": self.activation_combo.currentText(),
                            "dropout_at_inference": (
                                self.dropout_at_inference_check.isChecked()
                            ),
                            "global_pool": self.global_pool_combo.currentText(),
                            "last_layer_activation": (
                                self.last_layer_activation_combo.currentText()
                            ),
                        },
                        "training": {
                            "epochs": self.epochs_spin.value(),
                            "batch_size": (
                                self.parameter_rows["batch_size"][
                                    "value_widget"
                                ].value()
                            ),
                            "learning_rate": float(self.lr_spin.value()),
                            "optimizer": self.optimizer_combo.currentText(),
                            "scheduler": {
                                "type": self.scheduler_combo.currentText(),
                                "T_0": self.scheduler_t0_spin.value(),
                                "T_mult": self.scheduler_tmult_spin.value(),
                                "eta_min": self.scheduler_eta_min_spin.value(),
                            },
                            "num_workers": (
                                self.parameter_rows["num_workers"][
                                    "value_widget"
                                ].value()
                            ),
                            "warmup_epochs": self.warmup_epochs_spin.value(),
                            "warmup_lr_init": self.warmup_lr_init_spin.value(),
                            "mixed_precision": (
                                self.parameter_rows["use_mixed_precision"][
                                    "value_widget"
                                ].isChecked()
                            ),
                            "evaluation_freq": self.evaluation_freq_spin.value(),
                            "use_ema": self.use_ema_check.isChecked(),
                            "ema_decay": self.ema_decay_spin.value(),
                        },
                        "regularization": {
                            "weight_decay": float(self.weight_decay_spin.value()),
                            "gradient_clip": self.gradient_clip_spin.value(),
                            "label_smoothing": (self.label_smoothing_spin.value()),
                            "drop_connect_rate": (self.drop_connect_spin.value()),
                            "dropout_rate": self.dropout_spin.value(),
                            "momentum": self.momentum_spin.value(),
                            "epsilon": self.epsilon_spin.value(),
                            "swa": {
                                "use": self.use_swa_check.isChecked(),
                                "start_epoch": (self.swa_start_epoch_spin.value()),
                                "lr_swa": self.swa_lr_spin.value(),
                            },
                            "stochastic_depth": {
                                "use": self.use_stoch_depth_check.isChecked(),
                                "survival_probability": self.stoch_depth_survival_prob.value(),
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
                                "horizontal_flip": (
                                    self.horizontal_flip_check.isChecked()
                                ),
                                "vertical_flip": (self.vertical_flip_check.isChecked()),
                            },
                            "mixup": {
                                "use": self.mixup_check.isChecked(),
                                "alpha": self.mixup_alpha_spin.value(),
                            },
                            "cutmix": {
                                "use": self.cutmix_check.isChecked(),
                                "alpha": self.cutmix_alpha_spin.value(),
                            },
                        },
                        "monitoring": {
                            "metrics": {
                                "accuracy": self.accuracy_check.isChecked(),
                                "precision": self.precision_check.isChecked(),
                                "recall": self.recall_check.isChecked(),
                                "f1": self.f1_check.isChecked(),
                                "topk": self.topk_check.isChecked(),
                                "confusion_matrix": (
                                    self.confusion_matrix_check.isChecked()
                                ),
                            },
                            "early_stopping": {
                                "patience": self.patience_spin.value(),
                                "min_delta": self.min_delta_spin.value(),
                                "monitor": self.monitor_combo.currentText(),
                            },
                            "checkpointing": {
                                "best_only": self.best_only_check.isChecked(),
                                "save_frequency": self.save_freq_spin.value(),
                                "metric": (self.checkpoint_metric_combo.currentText()),
                            },
                        },
                    },
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

    def _delete_profile(self):
        """Usuwa wybrany profil."""
        if not self.current_profile:
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

        except Exception as e:
            self.logger.error(f"Błąd podczas usuwania profilu: {str(e)}", exc_info=True)
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Nie można usunąć profilu: {str(e)}"
            )

    def _on_architecture_changed(self, arch_name):
        self._update_variant_combo(arch_name)

    def _update_variant_combo(self, arch_name):
        self.variant_combo.clear()
        if arch_name == "EfficientNet":
            self.variant_combo.addItems(
                [
                    "EfficientNet-B0",
                    "EfficientNet-B1",
                    "EfficientNet-B2",
                    "EfficientNet-B3",
                    "EfficientNet-B4",
                    "EfficientNet-B5",
                    "EfficientNet-B6",
                    "EfficientNet-B7",
                ]
            )
        elif arch_name == "ConvNeXt":
            self.variant_combo.addItems(
                [
                    "ConvNeXt-Tiny",
                    "ConvNeXt-Small",
                    "ConvNeXt-Base",
                    "ConvNeXt-Large",
                ]
            )

    def _select_train_dir(self):
        """Wybór katalogu danych treningowych."""
        try:
            title = "Wybierz katalog treningowy"
            dir_path = QtWidgets.QFileDialog.getExistingDirectory(self, title)

            if dir_path:
                if validate_training_directory(dir_path):
                    self.train_dir_edit.setText(dir_path)
                else:
                    title = "Błąd"
                    msg = "Nieprawidłowy katalog treningowy"
                    QtWidgets.QMessageBox.warning(self, title, msg)

        except Exception as e:
            msg = "Błąd wyboru katalogu"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)

    def _select_val_dir(self):
        """Wybór katalogu danych walidacyjnych."""
        try:
            title = "Wybierz katalog walidacyjny"
            dir_path = QtWidgets.QFileDialog.getExistingDirectory(self, title)

            if dir_path:
                if validate_validation_directory(dir_path):
                    self.val_dir_edit.setText(dir_path)
                else:
                    title = "Błąd"
                    msg = "Nieprawidłowy katalog walidacyjny"
                    QtWidgets.QMessageBox.warning(self, title, msg)

        except Exception as e:
            msg = "Błąd wyboru katalogu"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)

    def _create_training_params_tab(self):
        """Tworzenie zakładki Parametry Treningu."""
        try:
            self.logger.debug("Tworzenie zakładki parametrów")
            tab = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(tab)
            form = QtWidgets.QFormLayout()

            # Pretrained
            self.pretrained_check = QtWidgets.QCheckBox(
                "Użyj wstępnie wytrenowanych wag"
            )
            self.pretrained_check.setChecked(True)
            form.addRow("", self.pretrained_check)

            # Pretrained weights
            self.pretrained_weights_combo = QtWidgets.QComboBox()
            self.pretrained_weights_combo.addItems(["imagenet", "custom"])
            form.addRow("Źródło wag:", self.pretrained_weights_combo)

            # Feature extraction only
            self.feature_extraction_check = QtWidgets.QCheckBox("Tylko ekstrakcja cech")
            self.feature_extraction_check.setChecked(False)
            form.addRow("", self.feature_extraction_check)

            # Activation
            self.activation_combo = QtWidgets.QComboBox()
            self.activation_combo.addItems(["swish", "relu", "sigmoid", "tanh"])
            form.addRow("Funkcja aktywacji:", self.activation_combo)

            # Dropout at inference
            self.dropout_at_inference_check = QtWidgets.QCheckBox(
                "Dropout podczas inferencji"
            )
            self.dropout_at_inference_check.setChecked(False)
            form.addRow("", self.dropout_at_inference_check)

            # Global pool
            self.global_pool_combo = QtWidgets.QComboBox()
            self.global_pool_combo.addItems(["avg", "max"])
            form.addRow("Global pooling:", self.global_pool_combo)

            # Liczba epok
            self.epochs_spin = QtWidgets.QSpinBox()
            self.epochs_spin.setRange(1, 1000)
            self.epochs_spin.setValue(DEFAULT_TRAINING_PARAMS["max_epochs"])
            form.addRow("Liczba epok:", self.epochs_spin)

            # Współczynnik uczenia
            self.lr_spin = QtWidgets.QDoubleSpinBox()
            self.lr_spin.setDecimals(6)
            self.lr_spin.setRange(0.000001, 1.0)
            self.lr_spin.setSingleStep(0.0001)
            self.lr_spin.setValue(0.001)
            form.addRow("Współczynnik uczenia:", self.lr_spin)

            # Optymalizator
            self.optimizer_combo = QtWidgets.QComboBox()
            optimizers = ["Adam", "AdamW", "SGD", "RMSprop"]
            self.optimizer_combo.addItems(optimizers)
            form.addRow("Optymalizator:", self.optimizer_combo)

            # Harmonogram uczenia
            self.scheduler_combo = QtWidgets.QComboBox()
            schedulers = [
                "None",
                "StepLR",
                "ReduceLROnPlateau",
                "CosineAnnealingLR",
                "OneCycleLR",
                "CosineAnnealingWarmRestarts",
            ]
            self.scheduler_combo.addItems(schedulers)
            form.addRow("Harmonogram uczenia:", self.scheduler_combo)

            # Liczba epok rozgrzewki
            self.warmup_epochs_spin = QtWidgets.QSpinBox()
            self.warmup_epochs_spin.setRange(0, 50)
            self.warmup_epochs_spin.setValue(5)
            form.addRow("Epoki rozgrzewki:", self.warmup_epochs_spin)

            # Parametry CosineAnnealingWarmRestarts
            self.scheduler_t0_spin = QtWidgets.QSpinBox()
            self.scheduler_t0_spin.setRange(1, 1000)
            self.scheduler_t0_spin.setValue(10)  # Domyślna wartość
            form.addRow("Scheduler T_0:", self.scheduler_t0_spin)

            self.scheduler_tmult_spin = QtWidgets.QSpinBox()
            self.scheduler_tmult_spin.setRange(1, 10)
            self.scheduler_tmult_spin.setValue(2)  # Domyślna wartość
            form.addRow("Scheduler T_mult:", self.scheduler_tmult_spin)

            self.scheduler_eta_min_spin = QtWidgets.QDoubleSpinBox()
            self.scheduler_eta_min_spin.setDecimals(6)
            self.scheduler_eta_min_spin.setRange(0.000000, 0.1)
            self.scheduler_eta_min_spin.setSingleStep(0.000001)
            self.scheduler_eta_min_spin.setValue(0.000001)  # Domyślna wartość
            form.addRow("Scheduler Eta_min:", self.scheduler_eta_min_spin)

            # Warmup LR init
            self.warmup_lr_init_spin = QtWidgets.QDoubleSpinBox()
            self.warmup_lr_init_spin.setDecimals(6)
            self.warmup_lr_init_spin.setRange(0.000000, 0.1)
            self.warmup_lr_init_spin.setSingleStep(0.000001)
            self.warmup_lr_init_spin.setValue(0.00001)  # Domyślna wartość
            form.addRow("Warmup LR Init:", self.warmup_lr_init_spin)

            # Evaluation frequency
            self.evaluation_freq_spin = QtWidgets.QSpinBox()
            self.evaluation_freq_spin.setRange(1, 100)  # Przykładowy zakres
            self.evaluation_freq_spin.setValue(1)  # Domyślna wartość
            form.addRow("Częstotliwość ewaluacji (epoki):", self.evaluation_freq_spin)

            # Use EMA
            self.use_ema_check = QtWidgets.QCheckBox("Używaj EMA")
            self.use_ema_check.setChecked(False)  # Domyślnie wyłączone
            form.addRow("", self.use_ema_check)

            # EMA decay
            self.ema_decay_spin = QtWidgets.QDoubleSpinBox()
            self.ema_decay_spin.setDecimals(4)  # Precyzja
            self.ema_decay_spin.setRange(0.9, 0.9999)  # Typowy zakres
            self.ema_decay_spin.setSingleStep(0.001)
            self.ema_decay_spin.setValue(0.999)  # Domyślna wartość
            form.addRow("EMA Decay:", self.ema_decay_spin)

            # Learning rate dla zamrożonych warstw
            self.frozen_lr_spin = QtWidgets.QDoubleSpinBox()
            self.frozen_lr_spin.setDecimals(6)
            self.frozen_lr_spin.setRange(0.000001, 1.0)
            self.frozen_lr_spin.setSingleStep(0.0001)
            self.frozen_lr_spin.setValue(0.0001)  # Domyślna wartość
            form.addRow("Learning rate (zamrożone warstwy):", self.frozen_lr_spin)

            # Learning rate dla odmrożonych warstw
            self.unfrozen_lr_spin = QtWidgets.QDoubleSpinBox()
            self.unfrozen_lr_spin.setDecimals(6)
            self.unfrozen_lr_spin.setRange(0.000001, 1.0)
            self.unfrozen_lr_spin.setSingleStep(0.0001)
            self.unfrozen_lr_spin.setValue(0.001)  # Domyślna wartość
            form.addRow("Learning rate (odmrożone warstwy):", self.unfrozen_lr_spin)

            layout.addLayout(form)
            return tab

        except Exception as e:
            msg = "Błąd podczas tworzenia zakładki"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            raise

    def _create_regularization_tab(self):
        """Tworzenie zakładki Regularyzacja i Optymalizacja."""
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

            # Dodanie SWA learning rate
            self.swa_lr_spin = QtWidgets.QDoubleSpinBox()
            self.swa_lr_spin.setRange(1e-6, 1e-3)
            self.swa_lr_spin.setDecimals(6)
            self.swa_lr_spin.setValue(5e-5)

            swa_layout.addRow("", self.use_swa_check)
            swa_layout.addRow("Epoka rozpoczęcia:", self.swa_start_epoch_spin)
            swa_layout.addRow("Learning rate SWA:", self.swa_lr_spin)
            swa_group.setLayout(swa_layout)

            # Stochastic Depth
            stoch_depth_group = QtWidgets.QGroupBox("Stochastic Depth")
            stoch_depth_layout = QtWidgets.QFormLayout()

            self.use_stoch_depth_check = QtWidgets.QCheckBox("Używaj Stochastic Depth")
            self.stoch_depth_survival_prob = QtWidgets.QDoubleSpinBox()
            self.stoch_depth_survival_prob.setRange(0.5, 1.0)
            self.stoch_depth_survival_prob.setDecimals(3)
            self.stoch_depth_survival_prob.setValue(0.8)

            stoch_depth_layout.addRow("", self.use_stoch_depth_check)
            stoch_depth_layout.addRow(
                "Prawdopodobieństwo przetrwania:", self.stoch_depth_survival_prob
            )
            stoch_depth_group.setLayout(stoch_depth_layout)

            layout.addLayout(form)
            layout.addWidget(swa_group)
            layout.addWidget(stoch_depth_group)
            return tab

        except Exception as e:
            msg = "Błąd podczas tworzenia zakładki"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            raise

    def _create_augmentation_tab(self):
        """Tworzy zakładkę z parametrami augmentacji."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        # Podstawowe augmentacje
        basic_group = QtWidgets.QGroupBox("Podstawowe augmentacje")
        basic_layout = QtWidgets.QFormLayout()

        self.basic_aug_check = QtWidgets.QCheckBox()
        self.basic_aug_check.setChecked(False)
        basic_layout.addRow("Użyj podstawowych augmentacji:", self.basic_aug_check)

        self.rotation_spin = QtWidgets.QSpinBox()
        self.rotation_spin.setRange(0, 180)
        self.rotation_spin.setValue(30)
        basic_layout.addRow("Rotacja (stopnie):", self.rotation_spin)

        self.brightness_spin = QtWidgets.QDoubleSpinBox()
        self.brightness_spin.setRange(0.0, 1.0)
        self.brightness_spin.setDecimals(2)
        self.brightness_spin.setValue(0.2)
        basic_layout.addRow("Jasność:", self.brightness_spin)

        self.contrast_spin = QtWidgets.QDoubleSpinBox()
        self.contrast_spin.setRange(0.0, 1.0)
        self.contrast_spin.setDecimals(2)
        self.contrast_spin.setValue(0.2)
        basic_layout.addRow("Kontrast:", self.contrast_spin)

        self.saturation_spin = QtWidgets.QDoubleSpinBox()
        self.saturation_spin.setRange(0.0, 1.0)
        self.saturation_spin.setDecimals(2)
        self.saturation_spin.setValue(0.2)
        basic_layout.addRow("Nasycenie:", self.saturation_spin)

        self.hue_spin = QtWidgets.QDoubleSpinBox()
        self.hue_spin.setRange(0.0, 0.5)
        self.hue_spin.setDecimals(2)
        self.hue_spin.setValue(0.1)
        basic_layout.addRow("Hue:", self.hue_spin)

        self.shift_spin = QtWidgets.QDoubleSpinBox()
        self.shift_spin.setRange(0.0, 1.0)
        self.shift_spin.setDecimals(2)
        self.shift_spin.setValue(0.1)
        basic_layout.addRow("Przesunięcie:", self.shift_spin)

        self.zoom_spin = QtWidgets.QDoubleSpinBox()
        self.zoom_spin.setRange(0.0, 1.0)
        self.zoom_spin.setDecimals(2)
        self.zoom_spin.setValue(0.1)
        basic_layout.addRow("Zoom:", self.zoom_spin)

        self.horizontal_flip_check = QtWidgets.QCheckBox()
        self.horizontal_flip_check.setChecked(True)
        basic_layout.addRow("Odbicia poziome:", self.horizontal_flip_check)

        self.vertical_flip_check = QtWidgets.QCheckBox()
        self.vertical_flip_check.setChecked(False)
        basic_layout.addRow("Odbicia pionowe:", self.vertical_flip_check)

        basic_group.setLayout(basic_layout)
        layout.addWidget(basic_group)

        # Mixup
        mixup_group = QtWidgets.QGroupBox("Mixup")
        mixup_layout = QtWidgets.QFormLayout()

        self.mixup_check = QtWidgets.QCheckBox()
        self.mixup_check.setChecked(False)
        mixup_layout.addRow("Użyj Mixup:", self.mixup_check)

        self.mixup_alpha_spin = QtWidgets.QDoubleSpinBox()
        self.mixup_alpha_spin.setRange(0.0, 1.0)
        self.mixup_alpha_spin.setValue(0.2)
        self.mixup_alpha_spin.setDecimals(2)
        mixup_layout.addRow("Alpha:", self.mixup_alpha_spin)

        mixup_group.setLayout(mixup_layout)
        layout.addWidget(mixup_group)

        # CutMix
        cutmix_group = QtWidgets.QGroupBox("CutMix")
        cutmix_layout = QtWidgets.QFormLayout()

        self.cutmix_check = QtWidgets.QCheckBox()
        self.cutmix_check.setChecked(False)
        cutmix_layout.addRow("Użyj CutMix:", self.cutmix_check)

        self.cutmix_alpha_spin = QtWidgets.QDoubleSpinBox()
        self.cutmix_alpha_spin.setRange(0.0, 1.0)
        self.cutmix_alpha_spin.setValue(0.2)
        self.cutmix_alpha_spin.setDecimals(2)
        cutmix_layout.addRow("Alpha:", self.cutmix_alpha_spin)

        cutmix_group.setLayout(cutmix_layout)
        layout.addWidget(cutmix_group)

        tab.setLayout(layout)
        return tab

    def _create_preprocessing_tab(self):
        """Tworzenie zakładki Preprocessing."""
        try:
            self.logger.debug("Tworzenie zakładki preprocessing")
            tab = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(tab)
            form = QtWidgets.QFormLayout()

            # Normalizacja
            self.normalization_combo = QtWidgets.QComboBox()
            self.normalization_combo.addItems(["RGB", "BGR"])
            form.addRow("Normalizacja:", self.normalization_combo)

            # Skalowanie obrazu
            scaling_group = QtWidgets.QGroupBox("Skalowanie obrazu")
            scaling_layout = QtWidgets.QFormLayout()

            self.scaling_method = QtWidgets.QComboBox()
            self.scaling_method.addItems(
                ["Bilinear", "Bicubic", "Lanczos", "Nearest", "Area"]
            )
            self.scaling_method.setCurrentText("Bilinear")

            self.maintain_aspect_ratio = QtWidgets.QCheckBox("Zachowaj proporcje")
            self.maintain_aspect_ratio.setChecked(True)

            self.pad_to_square = QtWidgets.QCheckBox("Uzupełnij do kwadratu")
            self.pad_to_square.setChecked(False)

            self.pad_mode = QtWidgets.QComboBox()
            self.pad_mode.addItems(["constant", "edge", "reflect", "symmetric"])
            self.pad_mode.setCurrentText("constant")

            self.pad_value = QtWidgets.QSpinBox()
            self.pad_value.setRange(0, 255)
            self.pad_value.setValue(0)

            scaling_layout.addRow("Metoda skalowania:", self.scaling_method)
            scaling_layout.addRow("", self.maintain_aspect_ratio)
            scaling_layout.addRow("", self.pad_to_square)
            scaling_layout.addRow("Tryb uzupełniania:", self.pad_mode)
            scaling_layout.addRow("Wartość uzupełniania:", self.pad_value)
            scaling_group.setLayout(scaling_layout)

            layout.addLayout(form)
            layout.addWidget(scaling_group)
            return tab

        except Exception as e:
            msg = "Błąd podczas tworzenia zakładki"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            raise

    def _create_monitoring_tab(self):
        """Tworzenie zakładki Monitorowanie i Zapis."""
        try:
            self.logger.debug("Tworzenie zakładki monitorowania")
            tab = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(tab)
            form = QtWidgets.QFormLayout()

            # --- Metryki ---
            self.accuracy_check = QtWidgets.QCheckBox("Accuracy")
            self.precision_check = QtWidgets.QCheckBox("Precision")
            self.recall_check = QtWidgets.QCheckBox("Recall")
            self.f1_check = QtWidgets.QCheckBox("F1")
            self.topk_check = QtWidgets.QCheckBox("Top-K")
            self.confusion_matrix_check = QtWidgets.QCheckBox("Macierz pomyłek")

            metrics_group = QtWidgets.QGroupBox("Metryki monitorowania")
            metrics_layout = QtWidgets.QVBoxLayout()
            metrics_layout.addWidget(self.accuracy_check)
            metrics_layout.addWidget(self.precision_check)
            metrics_layout.addWidget(self.recall_check)
            metrics_layout.addWidget(self.f1_check)
            metrics_layout.addWidget(self.topk_check)
            metrics_layout.addWidget(self.confusion_matrix_check)
            metrics_group.setLayout(metrics_layout)
            form.addRow(metrics_group)

            # Early stopping
            early_stop_group = QtWidgets.QGroupBox("Early stopping")
            early_stop_layout = QtWidgets.QFormLayout()

            self.use_early_stopping_check = QtWidgets.QCheckBox("Włącz early stopping")
            self.use_early_stopping_check.setChecked(True)
            self.use_early_stopping_check.stateChanged.connect(
                self._toggle_early_stopping_controls
            )
            early_stop_layout.addRow("", self.use_early_stopping_check)

            self.patience_spin = QtWidgets.QSpinBox()
            self.patience_spin.setRange(1, 100)
            self.patience_spin.setValue(10)
            self.patience_spin.setToolTip(
                "Liczba epok bez poprawy, po której trening zostanie zatrzymany"
            )
            early_stop_layout.addRow("Patience:", self.patience_spin)

            self.min_delta_spin = QtWidgets.QDoubleSpinBox()
            self.min_delta_spin.setRange(0.0, 1.0)
            self.min_delta_spin.setValue(0.001)
            self.min_delta_spin.setDecimals(4)
            self.min_delta_spin.setToolTip(
                "Minimalna zmiana metryki, uznawana za poprawę"
            )
            early_stop_layout.addRow("Min delta:", self.min_delta_spin)

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
            self.monitor_combo.setToolTip("Metryka używana do monitorowania poprawy")
            early_stop_layout.addRow("Monitor:", self.monitor_combo)

            early_stop_group.setLayout(early_stop_layout)
            form.addRow(early_stop_group)

            # Checkpointowanie
            checkpoint_group = QtWidgets.QGroupBox("Checkpointowanie")
            checkpoint_layout = QtWidgets.QFormLayout()

            self.best_only_check = QtWidgets.QCheckBox("Tylko najlepszy model")
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

            checkpoint_layout.addRow("", self.best_only_check)
            checkpoint_layout.addRow("Częstość zapisu:", self.save_freq_spin)
            checkpoint_layout.addRow("Metryka:", self.checkpoint_metric_combo)
            checkpoint_group.setLayout(checkpoint_layout)

            # TensorBoard
            tensorboard_group = QtWidgets.QGroupBox("TensorBoard")
            tensorboard_layout = QtWidgets.QFormLayout()

            self.use_tensorboard_check = QtWidgets.QCheckBox("Używaj TensorBoard")
            self.tensorboard_dir_edit = QtWidgets.QLineEdit()
            self.tensorboard_dir_edit.setPlaceholderText(
                "Katalog dla logów TensorBoard"
            )

            tensorboard_layout.addRow("", self.use_tensorboard_check)
            tensorboard_layout.addRow("Katalog:", self.tensorboard_dir_edit)
            tensorboard_group.setLayout(tensorboard_layout)

            # Katalog zapisu i logi
            save_group = QtWidgets.QGroupBox("Zapis modelu i logów")
            save_layout = QtWidgets.QFormLayout()

            self.model_dir_edit = QtWidgets.QLineEdit()
            model_dir_btn = QtWidgets.QPushButton("Przeglądaj...")
            model_dir_layout = QtWidgets.QHBoxLayout()
            model_dir_layout.addWidget(self.model_dir_edit)
            model_dir_layout.addWidget(model_dir_btn)

            self.save_logs_check = QtWidgets.QCheckBox("Zapisuj logi")

            save_layout.addRow("Katalog modelu:", model_dir_layout)
            save_layout.addRow("", self.save_logs_check)
            save_group.setLayout(save_layout)

            layout.addWidget(early_stop_group)
            layout.addWidget(checkpoint_group)
            layout.addWidget(tensorboard_group)
            layout.addWidget(save_group)
            return tab

        except Exception as e:
            msg = "Błąd podczas tworzenia zakładki"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            raise

    def _create_advanced_tab(self):
        """Tworzenie zakładki Zaawansowane."""
        try:
            self.logger.debug("Tworzenie zakładki zaawansowanej")
            tab = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(tab)

            # Parametry harmonogramu uczenia
            scheduler_group = QtWidgets.QGroupBox("Harmonogram uczenia")
            scheduler_layout = QtWidgets.QFormLayout()

            self.scheduler_patience = QtWidgets.QSpinBox()
            self.scheduler_patience.setRange(1, 50)
            self.scheduler_patience.setValue(5)

            self.scheduler_factor = QtWidgets.QDoubleSpinBox()
            self.scheduler_factor.setRange(0.1, 0.9)
            self.scheduler_factor.setValue(0.1)

            self.min_lr = QtWidgets.QDoubleSpinBox()
            self.min_lr.setRange(1e-6, 1e-2)
            self.min_lr.setValue(1e-6)

            self.scheduler_cooldown = QtWidgets.QSpinBox()
            self.scheduler_cooldown.setRange(0, 10)
            self.scheduler_cooldown.setValue(0)

            scheduler_layout.addRow("Patience:", self.scheduler_patience)
            scheduler_layout.addRow("Factor:", self.scheduler_factor)
            scheduler_layout.addRow("Min LR:", self.min_lr)
            scheduler_layout.addRow("Cooldown:", self.scheduler_cooldown)
            scheduler_group.setLayout(scheduler_layout)

            # Transfer learning
            transfer_group = QtWidgets.QGroupBox("Transfer Learning")
            transfer_layout = QtWidgets.QFormLayout()

            self.freeze_base_model = QtWidgets.QCheckBox("Zamroź model bazowy")
            self.unfreeze_layers = QtWidgets.QLineEdit()
            self.unfreeze_layers.setPlaceholderText("np. 2,3,4")

            self.unfreeze_strategy = QtWidgets.QComboBox()
            self.unfreeze_strategy.addItems(
                [
                    "Wszystkie na raz (unfreeze_all)",
                    "Stopniowo od końca (unfreeze_gradual_end)",
                    "Stopniowo od początku (unfreeze_gradual_start)",
                    "Po określonej liczbie epok (unfreeze_after_epoochs)",
                ]
            )

            self.unfreeze_after_epochs_spin = QtWidgets.QSpinBox()
            self.unfreeze_after_epochs_spin.setRange(0, 1000)  # Zakres epok
            self.unfreeze_after_epochs_spin.setValue(0)  # Domyślnie 0
            self.unfreeze_after_epochs_spin.setEnabled(False)  # Domyślnie wyłączone

            # Powiąż aktywność unfreeze_after_epochs_spin ze strategią
            self.unfreeze_strategy.currentTextChanged.connect(
                self._toggle_unfreeze_after_epochs_spin
            )

            transfer_layout.addRow("", self.freeze_base_model)
            transfer_layout.addRow("Warstwy do odmrożenia:", self.unfreeze_layers)
            transfer_layout.addRow("Strategia:", self.unfreeze_strategy)
            transfer_layout.addRow(
                "Odmroź po epokach:", self.unfreeze_after_epochs_spin
            )

            # Dodanie kontrolek dla learning rate
            self.frozen_lr_spin = QtWidgets.QDoubleSpinBox()
            self.frozen_lr_spin.setRange(0.0, 0.1)
            self.frozen_lr_spin.setValue(0.0001)
            self.frozen_lr_spin.setDecimals(6)
            transfer_layout.addRow("LR dla zamrożonych warstw:", self.frozen_lr_spin)

            self.unfrozen_lr_spin = QtWidgets.QDoubleSpinBox()
            self.unfrozen_lr_spin.setRange(0.0, 0.1)
            self.unfrozen_lr_spin.setValue(0.001)
            self.unfrozen_lr_spin.setDecimals(6)
            transfer_layout.addRow("LR dla odmrożonych warstw:", self.unfrozen_lr_spin)

            transfer_group.setLayout(transfer_layout)

            # Inicjalizacja wag
            weights_group = QtWidgets.QGroupBox("Inicjalizacja wag")
            weights_layout = QtWidgets.QFormLayout()

            self.init_weights = QtWidgets.QComboBox()
            init_methods = [
                "kaiming_normal",
                "kaiming_uniform",
                "xavier_normal",
                "xavier_uniform",
            ]
            self.init_weights.addItems(init_methods)

            self.freeze_layers = QtWidgets.QCheckBox("Zamroź warstwy CNN")

            weights_layout.addRow("Metoda:", self.init_weights)
            weights_layout.addRow("", self.freeze_layers)
            weights_group.setLayout(weights_layout)

            # Walidacja krzyżowa
            cv_group = QtWidgets.QGroupBox("Walidacja krzyżowa")
            cv_layout = QtWidgets.QFormLayout()

            self.use_cv = QtWidgets.QCheckBox("Używaj walidacji krzyżowej")
            self.cv_folds = QtWidgets.QSpinBox()
            self.cv_folds.setRange(2, 10)
            self.cv_folds.setValue(5)

            cv_layout.addRow("", self.use_cv)
            cv_layout.addRow("Liczba foldów:", self.cv_folds)
            cv_group.setLayout(cv_layout)

            # Trening dystrybuowany
            dist_group = QtWidgets.QGroupBox("Trening dystrybuowany")
            dist_layout = QtWidgets.QFormLayout()

            self.use_dist = QtWidgets.QCheckBox("Używaj treningu dystrybuowanego")
            self.dist_backend = QtWidgets.QComboBox()
            self.dist_backend.addItems(["nccl", "gloo"])
            self.dist_strategy = QtWidgets.QComboBox()
            self.dist_strategy.addItems(["ddp", "dp"])

            dist_layout.addRow("", self.use_dist)
            dist_layout.addRow("Backend:", self.dist_backend)
            dist_layout.addRow("Strategia:", self.dist_strategy)
            dist_group.setLayout(dist_layout)

            # Gradienty
            grad_group = QtWidgets.QGroupBox("Gradienty")
            grad_layout = QtWidgets.QFormLayout()

            self.grad_clip = QtWidgets.QDoubleSpinBox()
            self.grad_clip.setRange(0.0, 10.0)
            self.grad_clip.setValue(1.0)
            self.grad_clip.setDecimals(3)

            grad_layout.addRow("Gradient Clipping:", self.grad_clip)
            grad_group.setLayout(grad_layout)

            # Walidacja online
            online_val_group = QtWidgets.QGroupBox("Walidacja online")
            online_val_layout = QtWidgets.QFormLayout()

            self.use_online_val = QtWidgets.QCheckBox("Używaj walidacji online")
            self.online_val_freq = QtWidgets.QSpinBox()
            self.online_val_freq.setRange(1, 100)
            self.online_val_freq.setValue(10)

            online_val_layout.addRow("", self.use_online_val)
            online_val_layout.addRow("Częstość:", self.online_val_freq)
            online_val_group.setLayout(online_val_layout)

            layout.addWidget(scheduler_group)
            layout.addWidget(transfer_group)
            layout.addWidget(weights_group)
            layout.addWidget(cv_group)
            layout.addWidget(dist_group)
            layout.addWidget(grad_group)
            layout.addWidget(online_val_group)
            return tab

        except Exception as e:
            msg = "Błąd podczas tworzenia zakładki"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            raise

    def _create_optimization_tab(self):
        """Tworzenie zakładki Optymalizacja treningu."""
        try:
            self.logger.debug("Tworzenie zakładki optymalizacji treningu")
            tab = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(tab)

            # Grupa parametrów optymalizacyjnych
            params_group = QtWidgets.QGroupBox("Parametry optymalizacyjne")
            params_layout = QtWidgets.QFormLayout()

            # Dodaj parametry optymalizacyjne (bez learning_rate, epochs, gradient_accumulation_steps)
            params = [
                ("Batch size", "batch_size", 32, "int", 1, 1024, 1),
                ("Workers", "num_workers", 4, "int", 0, 32, 1),
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
                if key in self.parameter_rows:
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
        """Obsługuje przełączanie na źródło użytkownika."""
        value_widget = row_widgets["value_widget"]
        hw_checkbox = row_widgets["hw_checkbox"]

        if is_user_selected:
            value_widget.setEnabled(True)
            hw_checkbox.setChecked(False)

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

    def _apply_all_hardware_optimizations(self):
        """Stosuje wszystkie optymalizacje sprzętowe."""
        count = 0
        for param in self.parameter_rows.values():
            param_key = param["param_key"]
            # Mapowanie na klucz profilu sprzętowego
            profile_key = {
                "batch_size": "recommended_batch_size",
                "num_workers": "recommended_workers",
                "use_mixed_precision": "use_mixed_precision",
            }.get(param_key, param_key)
            if profile_key in self.hardware_profile:
                # Wymuś ustawienie checkboxa profilu sprzętowego (to wywoła _on_hw_toggle)
                param["hw_checkbox"].blockSignals(True)
                param["hw_checkbox"].setChecked(True)
                param["hw_checkbox"].blockSignals(False)
                # Ustaw wartość z profilu sprzętowego bezpośrednio
                value_widget = param["value_widget"]
                value = self.hardware_profile[profile_key]
                if isinstance(value_widget, QtWidgets.QSpinBox) or isinstance(
                    value_widget, QtWidgets.QDoubleSpinBox
                ):
                    value_widget.setValue(value)
                elif isinstance(value_widget, QtWidgets.QCheckBox):
                    value_widget.setChecked(bool(value))
                else:
                    value_widget.setText(str(value))
                count += 1
        QtWidgets.QMessageBox.information(
            self,
            "Sukces",
            f"Zastosowano {count} optymalnych ustawień z profilu sprzętowego.",
        )

    def _get_unfreeze_strategy_value(self, display_text):
        """Konwertuje wyświetlaną wartość strategii odmrażania na wartość wewnętrzną."""
        if "unfreeze_all" in display_text:
            return self.UNFREEZE_ALL
        elif "unfreeze_gradual_end" in display_text:
            return self.UNFREEZE_GRADUAL_END
        elif "unfreeze_gradual_start" in display_text:
            return self.UNFREEZE_GRADUAL_START
        elif "unfreeze_after_epoochs" in display_text:
            return self.UNFREEZE_AFTER_EPOCHS
        return self.UNFREEZE_ALL  # domyślna wartość

    def _get_unfreeze_layers_value(self, value):
        """Konwertuje wartość unfreeze_layers na odpowiedni format."""
        if not value:
            return "all"
        try:
            # Próba konwersji na int
            return int(value)
        except ValueError:
            # Jeśli nie da się przekonwertować, zwracamy string
            return value

    def _get_scheduler_value(self, display_text):
        """Konwertuje wyświetlaną wartość schedulera na wartość wewnętrzną."""
        if "OneCycleLR" in display_text:
            return "OneCycleLR"
        elif "CosineAnnealingWarmRestarts" in display_text:
            return "CosineAnnealingWarmRestarts"
        elif "StepLR" in display_text:
            return "StepLR"
        elif "ReduceLROnPlateau" in display_text:
            return "ReduceLROnPlateau"
        elif "CosineAnnealingLR" in display_text:
            return "CosineAnnealingLR"
        return "None"  # domyślna wartość

    def _toggle_early_stopping_controls(self, state):
        enabled = bool(state)
        self.patience_spin.setEnabled(enabled)
        self.min_delta_spin.setEnabled(enabled)
        self.monitor_combo.setEnabled(enabled)

    def _on_accept(self):
        """Obsługa przycisku OK."""
        try:
            # Generowanie nazwy zadania automatycznie
            variant = self.variant_combo.currentText()
            num_classes = self.num_classes_spin.value()
            now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            task_name = f"{variant}_{num_classes}_{now}"

            # Sprawdź czy katalog treningowy jest ustawiony
            train_dir = self.train_dir_edit.text().strip()
            if not train_dir:
                QtWidgets.QMessageBox.warning(
                    self, "Błąd", "Katalog treningowy nie może być pusty."
                )
                return

            # Sprawdź czy katalog walidacyjny jest ustawiony
            val_dir = self.val_dir_edit.text().strip()
            if not val_dir:
                QtWidgets.QMessageBox.warning(
                    self, "Błąd", "Katalog walidacyjny nie może być pusty."
                )
                return

            # Dodaj logi
            self.logger.info("=== TWORZENIE NOWEGO ZADANIA TRENINGOWEGO ===")
            self.logger.info(f"Nazwa zadania: {task_name}")

            # Podstawowe augmentacje
            augmentation = {
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
            }

            # Pobranie konfiguracji optymalizacyjnej
            optimization_config = {}
            if hasattr(self, "parameter_rows"):
                for param in self.parameter_rows.values():
                    param_key = param["param_key"]
                    hw_checkbox = param["hw_checkbox"]
                    value_widget = param["value_widget"]

                    if hw_checkbox.isChecked() and param["hw_value_actual"] is not None:
                        param_value = param["hw_value_actual"]
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

            # Pobierz wartości bezpośrednio z widgetów dla learning_rate, epochs, gradient_accumulation_steps
            learning_rate = self.lr_spin.value()
            epochs = self.epochs_spin.value()
            grad_accum_steps = self.grad_accum_steps_spin.value()

            # --- POPRAWKA: UZUPEŁNIENIE SEKCJI MONITORING I PREPROCESSING ---
            config = {
                "train_dir": train_dir,
                "data_dir": train_dir,
                "val_dir": val_dir,
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
                    "epochs": epochs,
                    "batch_size": optimization_config.get("batch_size", 32),
                    "learning_rate": float(learning_rate),
                    "frozen_lr": self.frozen_lr_spin.value(),
                    "unfrozen_lr": self.unfrozen_lr_spin.value(),
                    "optimizer": self.optimizer_combo.currentText(),
                    "scheduler": {
                        "type": self._get_scheduler_value(
                            self.scheduler_combo.currentText()
                        ),
                        "T_0": self.scheduler_t0_spin.value(),
                        "T_mult": self.scheduler_tmult_spin.value(),
                        "eta_min": self.scheduler_eta_min_spin.value(),
                    },
                    "num_workers": optimization_config.get("num_workers", 4),
                    "warmup_epochs": self.warmup_epochs_spin.value(),
                    "warmup_lr_init": self.warmup_lr_init_spin.value(),
                    "mixed_precision": optimization_config.get(
                        "use_mixed_precision", True
                    ),
                    "evaluation_freq": self.evaluation_freq_spin.value(),
                    "use_ema": self.use_ema_check.isChecked(),
                    "ema_decay": self.ema_decay_spin.value(),
                    "freeze_base_model": (
                        self.freeze_base_model.isChecked()
                        if hasattr(self, "freeze_base_model")
                        else False
                    ),
                    "unfreeze_layers": self._get_unfreeze_layers_value(
                        self.unfreeze_layers.text()
                    ),
                    "unfreeze_strategy": self._get_unfreeze_strategy_value(
                        self.unfreeze_strategy.currentText()
                    ),
                    "unfreeze_after_epochs": self.unfreeze_after_epochs_spin.value(),
                    "gradient_accumulation_steps": grad_accum_steps,
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
                        "start_epoch": (self.swa_start_epoch_spin.value()),
                        "lr_swa": self.swa_lr_spin.value(),
                    },
                    "stochastic_depth": {
                        "use": self.use_stoch_depth_check.isChecked(),
                        "survival_probability": self.stoch_depth_survival_prob.value(),
                    },
                },
                "augmentation": augmentation,
                "optimization": optimization_config,
                "monitoring": {
                    "metrics": {
                        "accuracy": self.accuracy_check.isChecked(),
                        "precision": self.precision_check.isChecked(),
                        "recall": self.recall_check.isChecked(),
                        "f1": self.f1_check.isChecked(),
                        "topk": self.topk_check.isChecked(),
                        "confusion_matrix": self.confusion_matrix_check.isChecked(),
                    },
                    "early_stopping": {
                        "enabled": self.use_early_stopping_check.isChecked(),
                        "patience": self.patience_spin.value(),
                        "min_delta": self.min_delta_spin.value(),
                        "monitor": self.monitor_combo.currentText(),
                    },
                    "checkpointing": {
                        "best_only": self.best_only_check.isChecked(),
                        "save_frequency": self.save_freq_spin.value(),
                        "metric": (self.checkpoint_metric_combo.currentText()),
                    },
                },
            }
            # Dodaj sekcję preprocessingu jeśli nie ma
            if "preprocessing" not in config:
                config["preprocessing"] = {
                    "normalization": "RGB",  # Wartość domyślna
                    "resize_mode": "bilinear",
                    "cache_dataset": False,
                }
            # --- KONIEC POPRAWKI ---

            self.task_config = {
                "name": task_name,
                "type": "training",
                "status": "Nowy",
                "priority": 0,
                "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "config": config,
            }

            # Dodaj logi
            self.logger.info(f"Utworzono konfigurację zadania: {task_name}")
            self.logger.info(f"Typ zadania: {self.task_config['type']}")
            config_str = json.dumps(self.task_config, indent=2, ensure_ascii=False)
            self.logger.info(f"Pełna konfiguracja: {config_str}")

            self.accept()

        except Exception as e:
            self.logger.error(
                f"Błąd podczas dodawania zadania: {str(e)}", exc_info=True
            )
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Nie można dodać zadania: {str(e)}"
            )

    def get_task_config(self):
        """Zwraca konfigurację zadania lub None, jeśli nie dodano zadania."""
        return getattr(self, "task_config", None)

    def closeEvent(self, event):
        """Obsługa zamknięcia okna."""
        self.logger.info("Zamykanie okna dialogowego")
        self.accept()
        event.accept()

    def _show_hardware_profile(self):
        """Wyświetla okno z aktualnym profilem sprzętowym."""
        dialog = HardwareProfileDialog(self.hardware_profile, self)
        dialog.exec()

    def _toggle_unfreeze_after_epochs_spin(self, strategy_text):
        """Włącza/wyłącza kontrolkę unfreeze_after_epochs_spin w zależności od wybranej strategii."""
        is_enabled = "unfreeze_after_epoochs" in strategy_text
        self.unfreeze_after_epochs_spin.setEnabled(is_enabled)
