import datetime
import json
import logging
import os
from pathlib import Path

from PyQt6 import QtCore, QtWidgets

from app.gui.dialogs.hardware_profile_dialog import HardwareProfileDialog
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
                # Przygotuj konfigurację optymalizacji
                optimization_config = {
                    "batch_size": self.parameter_rows["batch_size"][
                        "value_widget"
                    ].value(),
                    "num_workers": self.parameter_rows["num_workers"][
                        "value_widget"
                    ].value(),
                    "use_mixed_precision": self.parameter_rows["use_mixed_precision"][
                        "value_widget"
                    ].isChecked(),
                    "memory_efficient": self.parameter_rows["memory_efficient"][
                        "value_widget"
                    ].isChecked(),
                    "cudnn_benchmark": self.parameter_rows["cudnn_benchmark"][
                        "value_widget"
                    ].isChecked(),
                    "pin_memory": self.parameter_rows["pin_memory"][
                        "value_widget"
                    ].isChecked(),
                    "dataloader": {
                        "shuffle": self.parameter_rows["shuffle"][
                            "value_widget"
                        ].isChecked(),
                        "prefetch_factor": self.parameter_rows["prefetch_factor"][
                            "value_widget"
                        ].value(),
                        "persistent_workers": self.parameter_rows["persistent_workers"][
                            "value_widget"
                        ].isChecked(),
                        "drop_last": self.parameter_rows["drop_last"][
                            "value_widget"
                        ].isChecked(),
                    },
                }

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
                            "autoaugment": {
                                "use": self.autoaugment_check.isChecked(),
                                "policy": self.autoaugment_policy_combo.currentText(),
                            },
                            "randaugment": {
                                "use": self.randaugment_check.isChecked(),
                                "n": self.randaugment_n_spin.value(),
                                "m": self.randaugment_m_spin.value(),
                            },
                            "trivialaugment": {
                                "use": self.trivialaugment_check.isChecked()
                            },
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
                            "resize": {"enabled": self.resize_check.isChecked()},
                        },
                        "optimization": optimization_config,
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

            # Kontener na dwie kolumny
            columns_layout = QtWidgets.QHBoxLayout()

            # Lewa kolumna
            left_column = QtWidgets.QWidget()
            left_layout = QtWidgets.QVBoxLayout(left_column)
            left_form = QtWidgets.QFormLayout()

            # Pretrained
            self.pretrained_check = QtWidgets.QCheckBox(
                "Użyj wstępnie wytrenowanych wag"
            )
            self.pretrained_check.setChecked(True)
            left_form.addRow("", self.pretrained_check)

            # Pretrained weights
            self.pretrained_weights_combo = QtWidgets.QComboBox()
            self.pretrained_weights_combo.addItems(["imagenet", "custom"])
            left_form.addRow("Źródło wag:", self.pretrained_weights_combo)

            # Feature extraction only
            self.feature_extraction_check = QtWidgets.QCheckBox("Tylko ekstrakcja cech")
            self.feature_extraction_check.setChecked(False)
            left_form.addRow("", self.feature_extraction_check)

            # Activation
            self.activation_combo = QtWidgets.QComboBox()
            self.activation_combo.addItems(["swish", "relu", "sigmoid", "tanh"])
            left_form.addRow("Funkcja aktywacji:", self.activation_combo)

            # Dropout at inference
            self.dropout_at_inference_check = QtWidgets.QCheckBox(
                "Dropout podczas inferencji"
            )
            self.dropout_at_inference_check.setChecked(False)
            left_form.addRow("", self.dropout_at_inference_check)

            # Global pool
            self.global_pool_combo = QtWidgets.QComboBox()
            self.global_pool_combo.addItems(["avg", "max"])
            left_form.addRow("Global pooling:", self.global_pool_combo)

            # Last layer activation
            self.last_layer_activation_combo = QtWidgets.QComboBox()
            self.last_layer_activation_combo.addItems(["softmax", "sigmoid", "none"])
            left_form.addRow(
                "Aktywacja ostatniej warstwy:", self.last_layer_activation_combo
            )

            left_layout.addLayout(left_form)

            # Prawa kolumna (pusta na razie)
            right_column = QtWidgets.QWidget()
            right_layout = QtWidgets.QVBoxLayout(right_column)

            # Dodaj kolumny do głównego layoutu
            columns_layout.addWidget(left_column)
            columns_layout.addWidget(right_column)

            layout.addLayout(columns_layout)
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

            # Kontener na dwie kolumny
            columns_layout = QtWidgets.QHBoxLayout()

            # Lewa kolumna
            left_column = QtWidgets.QWidget()
            left_layout = QtWidgets.QVBoxLayout(left_column)
            left_form = QtWidgets.QFormLayout()

            # Weight decay
            self.weight_decay_spin = QtWidgets.QDoubleSpinBox()
            self.weight_decay_spin.setDecimals(6)
            self.weight_decay_spin.setRange(0.0, 1.0)
            self.weight_decay_spin.setSingleStep(0.0001)
            self.weight_decay_spin.setValue(0.0001)
            left_form.addRow("Weight Decay:", self.weight_decay_spin)

            # Gradient clipping
            self.gradient_clip_spin = QtWidgets.QDoubleSpinBox()
            self.gradient_clip_spin.setRange(0.0, 10.0)
            self.gradient_clip_spin.setDecimals(3)
            self.gradient_clip_spin.setValue(1.0)
            left_form.addRow("Gradient Clipping:", self.gradient_clip_spin)

            # Label smoothing
            self.label_smoothing_spin = QtWidgets.QDoubleSpinBox()
            self.label_smoothing_spin.setRange(0.0, 0.5)
            self.label_smoothing_spin.setDecimals(3)
            self.label_smoothing_spin.setValue(0.1)
            left_form.addRow("Label Smoothing:", self.label_smoothing_spin)

            # Drop connect rate
            self.drop_connect_spin = QtWidgets.QDoubleSpinBox()
            self.drop_connect_spin.setRange(0.0, 0.5)
            self.drop_connect_spin.setDecimals(3)
            self.drop_connect_spin.setValue(0.2)
            left_form.addRow("Drop Connect Rate:", self.drop_connect_spin)

            # Dropout rate
            self.dropout_spin = QtWidgets.QDoubleSpinBox()
            self.dropout_spin.setRange(0.0, 0.5)
            self.dropout_spin.setDecimals(3)
            self.dropout_spin.setValue(0.2)
            left_form.addRow("Dropout Rate:", self.dropout_spin)

            # Momentum
            self.momentum_spin = QtWidgets.QDoubleSpinBox()
            self.momentum_spin.setRange(0.0, 1.0)
            self.momentum_spin.setDecimals(3)
            self.momentum_spin.setValue(0.9)
            left_form.addRow("Momentum:", self.momentum_spin)

            # Epsilon
            self.epsilon_spin = QtWidgets.QDoubleSpinBox()
            self.epsilon_spin.setRange(1e-8, 1e-3)
            self.epsilon_spin.setDecimals(8)
            self.epsilon_spin.setValue(1e-6)
            left_form.addRow("Epsilon:", self.epsilon_spin)

            left_layout.addLayout(left_form)

            # Prawa kolumna
            right_column = QtWidgets.QWidget()
            right_layout = QtWidgets.QVBoxLayout(right_column)

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

            right_layout.addWidget(swa_group)
            right_layout.addWidget(stoch_depth_group)

            # Dodaj kolumny do głównego layoutu
            columns_layout.addWidget(left_column)
            columns_layout.addWidget(right_column)

            layout.addLayout(columns_layout)
            return tab

        except Exception as e:
            msg = "Błąd podczas tworzenia zakładki"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            raise

    def _create_augmentation_tab(self):
        """Tworzy zakładkę z parametrami augmentacji."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        # Kontener na dwie kolumny
        columns_layout = QtWidgets.QHBoxLayout()

        # Lewa kolumna
        left_column = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_column)

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
        left_layout.addWidget(basic_group)

        # Prawa kolumna
        right_column = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_column)

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
        right_layout.addWidget(mixup_group)

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
        right_layout.addWidget(cutmix_group)

        # AutoAugment
        autoaugment_group = QtWidgets.QGroupBox("AutoAugment")
        autoaugment_layout = QtWidgets.QFormLayout()

        self.autoaugment_check = QtWidgets.QCheckBox()
        self.autoaugment_check.setChecked(False)
        autoaugment_layout.addRow("Użyj AutoAugment:", self.autoaugment_check)

        self.autoaugment_policy_combo = QtWidgets.QComboBox()
        self.autoaugment_policy_combo.addItems(["imagenet", "cifar", "svhn"])
        autoaugment_layout.addRow("Polityka:", self.autoaugment_policy_combo)

        autoaugment_group.setLayout(autoaugment_layout)
        right_layout.addWidget(autoaugment_group)

        # TrivialAugment
        trivialaugment_group = QtWidgets.QGroupBox("TrivialAugment")
        trivialaugment_layout = QtWidgets.QFormLayout()

        self.trivialaugment_check = QtWidgets.QCheckBox()
        self.trivialaugment_check.setChecked(False)
        trivialaugment_layout.addRow("Użyj TrivialAugment:", self.trivialaugment_check)

        trivialaugment_group.setLayout(trivialaugment_layout)
        right_layout.addWidget(trivialaugment_group)

        # Random Erase
        random_erase_group = QtWidgets.QGroupBox("Random Erase")
        random_erase_layout = QtWidgets.QFormLayout()

        self.random_erase_check = QtWidgets.QCheckBox()
        self.random_erase_check.setChecked(False)
        random_erase_layout.addRow("Użyj Random Erase:", self.random_erase_check)

        self.random_erase_prob_spin = QtWidgets.QDoubleSpinBox()
        self.random_erase_prob_spin.setRange(0.0, 1.0)
        self.random_erase_prob_spin.setValue(0.5)
        self.random_erase_prob_spin.setDecimals(2)
        random_erase_layout.addRow("Prawdopodobieństwo:", self.random_erase_prob_spin)

        random_erase_group.setLayout(random_erase_layout)
        right_layout.addWidget(random_erase_group)

        # Grid Distortion
        grid_distortion_group = QtWidgets.QGroupBox("Grid Distortion")
        grid_distortion_layout = QtWidgets.QFormLayout()

        self.grid_distortion_check = QtWidgets.QCheckBox()
        self.grid_distortion_check.setChecked(False)
        grid_distortion_layout.addRow(
            "Użyj Grid Distortion:", self.grid_distortion_check
        )

        self.grid_distortion_prob_spin = QtWidgets.QDoubleSpinBox()
        self.grid_distortion_prob_spin.setRange(0.0, 1.0)
        self.grid_distortion_prob_spin.setValue(0.2)
        self.grid_distortion_prob_spin.setDecimals(2)
        grid_distortion_layout.addRow(
            "Prawdopodobieństwo:", self.grid_distortion_prob_spin
        )

        self.grid_distortion_limit_spin = QtWidgets.QDoubleSpinBox()
        self.grid_distortion_limit_spin.setRange(0.0, 1.0)
        self.grid_distortion_limit_spin.setValue(0.1)
        self.grid_distortion_limit_spin.setDecimals(2)
        grid_distortion_layout.addRow(
            "Limit zniekształcenia:", self.grid_distortion_limit_spin
        )

        grid_distortion_group.setLayout(grid_distortion_layout)
        right_layout.addWidget(grid_distortion_group)

        # RandAugment
        randaugment_group = QtWidgets.QGroupBox("RandAugment")
        randaugment_layout = QtWidgets.QFormLayout()

        self.randaugment_check = QtWidgets.QCheckBox()
        self.randaugment_check.setChecked(False)
        randaugment_layout.addRow("Użyj RandAugment:", self.randaugment_check)

        self.randaugment_n_spin = QtWidgets.QSpinBox()
        self.randaugment_n_spin.setRange(1, 10)
        self.randaugment_n_spin.setValue(2)
        randaugment_layout.addRow("Liczba operacji (n):", self.randaugment_n_spin)

        self.randaugment_m_spin = QtWidgets.QSpinBox()
        self.randaugment_m_spin.setRange(1, 30)
        self.randaugment_m_spin.setValue(10)
        randaugment_layout.addRow("Intensywność (m):", self.randaugment_m_spin)

        randaugment_group.setLayout(randaugment_layout)
        right_layout.addWidget(randaugment_group)

        # Dodaj kolumny do głównego layoutu
        columns_layout.addWidget(left_column)
        columns_layout.addWidget(right_column)

        layout.addLayout(columns_layout)
        tab.setLayout(layout)
        return tab

    def _create_preprocessing_tab(self):
        """Tworzy zakładkę z parametrami preprocessingu."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        # Kontener na dwie kolumny
        columns_layout = QtWidgets.QHBoxLayout()

        # Lewa kolumna
        left_column = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_column)

        # Normalizacja
        normalization_group = QtWidgets.QGroupBox("Normalizacja")
        normalization_layout = QtWidgets.QFormLayout()

        self.normalize_check = QtWidgets.QCheckBox()
        self.normalize_check.setChecked(True)
        normalization_layout.addRow("Włącz normalizację:", self.normalize_check)

        self.normalize_mean_spin = QtWidgets.QDoubleSpinBox()
        self.normalize_mean_spin.setRange(0.0, 1.0)
        self.normalize_mean_spin.setValue(0.485)
        self.normalize_mean_spin.setDecimals(3)
        normalization_layout.addRow("Średnia:", self.normalize_mean_spin)

        self.normalize_std_spin = QtWidgets.QDoubleSpinBox()
        self.normalize_std_spin.setRange(0.0, 1.0)
        self.normalize_std_spin.setValue(0.229)
        self.normalize_std_spin.setDecimals(3)
        normalization_layout.addRow("Odchylenie standardowe:", self.normalize_std_spin)

        normalization_group.setLayout(normalization_layout)
        left_layout.addWidget(normalization_group)

        # Zmiana rozmiaru
        resize_group = QtWidgets.QGroupBox("Zmiana rozmiaru")
        resize_layout = QtWidgets.QFormLayout()

        self.resize_check = QtWidgets.QCheckBox()
        self.resize_check.setChecked(True)
        resize_layout.addRow("Włącz zmianę rozmiaru:", self.resize_check)

        self.resize_width_spin = QtWidgets.QSpinBox()
        self.resize_width_spin.setRange(32, 1024)
        self.resize_width_spin.setValue(224)
        resize_layout.addRow("Szerokość:", self.resize_width_spin)

        self.resize_height_spin = QtWidgets.QSpinBox()
        self.resize_height_spin.setRange(32, 1024)
        self.resize_height_spin.setValue(224)
        resize_layout.addRow("Wysokość:", self.resize_height_spin)

        resize_group.setLayout(resize_layout)
        left_layout.addWidget(resize_group)

        # Prawa kolumna
        right_column = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_column)

        # Konwersja kolorów
        color_group = QtWidgets.QGroupBox("Konwersja kolorów")
        color_layout = QtWidgets.QFormLayout()

        self.grayscale_check = QtWidgets.QCheckBox()
        self.grayscale_check.setChecked(False)
        color_layout.addRow("Konwertuj do skali szarości:", self.grayscale_check)

        color_group.setLayout(color_layout)
        right_layout.addWidget(color_group)

        # Filtrowanie
        filter_group = QtWidgets.QGroupBox("Filtrowanie")
        filter_layout = QtWidgets.QFormLayout()

        self.blur_check = QtWidgets.QCheckBox()
        self.blur_check.setChecked(False)
        filter_layout.addRow("Włącz rozmycie:", self.blur_check)

        self.blur_kernel_spin = QtWidgets.QSpinBox()
        self.blur_kernel_spin.setRange(1, 15)
        self.blur_kernel_spin.setValue(3)
        self.blur_kernel_spin.setSingleStep(2)
        filter_layout.addRow("Rozmiar jądra rozmycia:", self.blur_kernel_spin)

        filter_group.setLayout(filter_layout)
        right_layout.addWidget(filter_group)

        # Dodaj kolumny do głównego layoutu
        columns_layout.addWidget(left_column)
        columns_layout.addWidget(right_column)

        layout.addLayout(columns_layout)
        tab.setLayout(layout)
        return tab

    def _create_monitoring_tab(self):
        """Tworzy zakładkę z parametrami monitorowania."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        # Kontener na dwie kolumny
        columns_layout = QtWidgets.QHBoxLayout()

        # Lewa kolumna
        left_column = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_column)

        # TensorBoard
        tensorboard_group = QtWidgets.QGroupBox("TensorBoard")
        tensorboard_layout = QtWidgets.QFormLayout()

        self.tensorboard_check = QtWidgets.QCheckBox()
        self.tensorboard_check.setChecked(True)
        tensorboard_layout.addRow("Włącz TensorBoard:", self.tensorboard_check)

        self.tensorboard_log_dir = QtWidgets.QLineEdit()
        self.tensorboard_log_dir.setText("logs/tensorboard")
        tensorboard_layout.addRow("Katalog logów:", self.tensorboard_log_dir)

        tensorboard_group.setLayout(tensorboard_layout)
        left_layout.addWidget(tensorboard_group)

        # Checkpointy
        checkpoint_group = QtWidgets.QGroupBox("Checkpointy")
        checkpoint_layout = QtWidgets.QFormLayout()

        self.save_checkpoints_check = QtWidgets.QCheckBox()
        self.save_checkpoints_check.setChecked(True)
        checkpoint_layout.addRow("Zapisuj checkpointy:", self.save_checkpoints_check)

        self.checkpoint_dir = QtWidgets.QLineEdit()
        self.checkpoint_dir.setText("checkpoints")
        checkpoint_layout.addRow("Katalog checkpointów:", self.checkpoint_dir)

        self.save_best_only_check = QtWidgets.QCheckBox()
        self.save_best_only_check.setChecked(True)
        checkpoint_layout.addRow(
            "Zapisuj tylko najlepszy model:", self.save_best_only_check
        )

        checkpoint_group.setLayout(checkpoint_layout)
        left_layout.addWidget(checkpoint_group)

        # Prawa kolumna
        right_column = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_column)

        # Metryki
        metrics_group = QtWidgets.QGroupBox("Metryki")
        metrics_layout = QtWidgets.QFormLayout()

        self.metrics_interval_spin = QtWidgets.QSpinBox()
        self.metrics_interval_spin.setRange(1, 1000)
        self.metrics_interval_spin.setValue(10)
        metrics_layout.addRow("Interwał zapisu metryk:", self.metrics_interval_spin)

        self.metrics_dir = QtWidgets.QLineEdit()
        self.metrics_dir.setText("metrics")
        metrics_layout.addRow("Katalog metryk:", self.metrics_dir)

        metrics_group.setLayout(metrics_layout)
        right_layout.addWidget(metrics_group)

        # Wczesne zatrzymanie
        early_stopping_group = QtWidgets.QGroupBox("Wczesne zatrzymanie")
        early_stopping_layout = QtWidgets.QFormLayout()

        self.early_stopping_check = QtWidgets.QCheckBox()
        self.early_stopping_check.setChecked(True)
        early_stopping_layout.addRow(
            "Włącz wczesne zatrzymanie:", self.early_stopping_check
        )

        self.early_stopping_patience_spin = QtWidgets.QSpinBox()
        self.early_stopping_patience_spin.setRange(1, 100)
        self.early_stopping_patience_spin.setValue(10)
        early_stopping_layout.addRow("Cierpliwość:", self.early_stopping_patience_spin)

        self.early_stopping_min_delta_spin = QtWidgets.QDoubleSpinBox()
        self.early_stopping_min_delta_spin.setRange(0.0, 1.0)
        self.early_stopping_min_delta_spin.setValue(0.001)
        self.early_stopping_min_delta_spin.setDecimals(4)
        early_stopping_layout.addRow(
            "Minimalna zmiana:", self.early_stopping_min_delta_spin
        )

        early_stopping_group.setLayout(early_stopping_layout)
        right_layout.addWidget(early_stopping_group)

        # Dodaj kolumny do głównego layoutu
        columns_layout.addWidget(left_column)
        columns_layout.addWidget(right_column)

        layout.addLayout(columns_layout)
        tab.setLayout(layout)
        return tab

    def _create_advanced_tab(self):
        """Tworzy zakładkę z zaawansowanymi parametrami."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        # Kontener na dwie kolumny
        columns_layout = QtWidgets.QHBoxLayout()

        # Lewa kolumna
        left_column = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_column)

        # Optymalizator
        optimizer_group = QtWidgets.QGroupBox("Optymalizator")
        optimizer_layout = QtWidgets.QFormLayout()

        self.optimizer_combo = QtWidgets.QComboBox()
        self.optimizer_combo.addItems(["Adam", "AdamW", "SGD", "RMSprop"])
        optimizer_layout.addRow("Typ optymalizatora:", self.optimizer_combo)

        self.learning_rate_spin = QtWidgets.QDoubleSpinBox()
        self.learning_rate_spin.setRange(1e-6, 1.0)
        self.learning_rate_spin.setValue(0.001)
        self.learning_rate_spin.setDecimals(6)
        optimizer_layout.addRow("Learning rate:", self.learning_rate_spin)

        self.weight_decay_spin = QtWidgets.QDoubleSpinBox()
        self.weight_decay_spin.setRange(0.0, 1.0)
        self.weight_decay_spin.setValue(0.0001)
        self.weight_decay_spin.setDecimals(4)
        optimizer_layout.addRow("Weight decay:", self.weight_decay_spin)

        optimizer_group.setLayout(optimizer_layout)
        left_layout.addWidget(optimizer_group)

        # Scheduler
        scheduler_group = QtWidgets.QGroupBox("Scheduler")
        scheduler_layout = QtWidgets.QFormLayout()

        self.scheduler_combo = QtWidgets.QComboBox()
        self.scheduler_combo.addItems(
            ["CosineAnnealingLR", "ReduceLROnPlateau", "OneCycleLR"]
        )
        scheduler_layout.addRow("Typ schedulera:", self.scheduler_combo)

        self.warmup_epochs_spin = QtWidgets.QSpinBox()
        self.warmup_epochs_spin.setRange(0, 100)
        self.warmup_epochs_spin.setValue(5)
        scheduler_layout.addRow("Epoki rozgrzewki:", self.warmup_epochs_spin)

        scheduler_group.setLayout(scheduler_layout)
        left_layout.addWidget(scheduler_group)

        # Prawa kolumna
        right_column = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_column)

        # Gradient
        gradient_group = QtWidgets.QGroupBox("Gradient")
        gradient_layout = QtWidgets.QFormLayout()

        self.gradient_clip_val_spin = QtWidgets.QDoubleSpinBox()
        self.gradient_clip_val_spin.setRange(0.0, 10.0)
        self.gradient_clip_val_spin.setValue(1.0)
        self.gradient_clip_val_spin.setDecimals(2)
        gradient_layout.addRow("Wartość przycinania:", self.gradient_clip_val_spin)

        self.accumulate_grad_batches_spin = QtWidgets.QSpinBox()
        self.accumulate_grad_batches_spin.setRange(1, 100)
        self.accumulate_grad_batches_spin.setValue(1)
        gradient_layout.addRow(
            "Akumulacja gradientów:", self.accumulate_grad_batches_spin
        )

        gradient_group.setLayout(gradient_layout)
        right_layout.addWidget(gradient_group)

        # Zaawansowane
        advanced_group = QtWidgets.QGroupBox("Zaawansowane")
        advanced_layout = QtWidgets.QFormLayout()

        self.precision_combo = QtWidgets.QComboBox()
        self.precision_combo.addItems(["32", "16", "bf16"])
        advanced_layout.addRow("Precyzja:", self.precision_combo)

        self.deterministic_check = QtWidgets.QCheckBox()
        self.deterministic_check.setChecked(False)
        advanced_layout.addRow("Tryb deterministyczny:", self.deterministic_check)

        self.benchmark_check = QtWidgets.QCheckBox()
        self.benchmark_check.setChecked(True)
        advanced_layout.addRow("Benchmark CUDA:", self.benchmark_check)

        advanced_group.setLayout(advanced_layout)
        right_layout.addWidget(advanced_group)

        # Dodaj kolumny do głównego layoutu
        columns_layout.addWidget(left_column)
        columns_layout.addWidget(right_column)

        layout.addLayout(columns_layout)
        tab.setLayout(layout)
        return tab

    def _create_optimization_tab(self):
        """Tworzenie zakładki Optymalizacja treningu."""
        try:
            self.logger.debug("Tworzenie zakładki optymalizacji treningu")
            tab = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(tab)

            # Grupa parametrów optymalizacyjnych
            params_group = QtWidgets.QGroupBox("Parametry optymalizacyjne")
            params_layout = QtWidgets.QFormLayout()

            # Dodaj parametry optymalizacyjne (bez learning_rate, epochs,
            # gradient_accumulation_steps)
            params = [
                ("Batch size", "batch_size", 32, "int", 1, 1024, 1),
                ("Workers", "num_workers", 4, "int", 0, 32, 1),
                ("Mixed Precision", "use_mixed_precision", True, "bool"),
                ("Memory Efficient", "memory_efficient", False, "bool"),
                # Dodane parametry:
                ("CUDNN Benchmark", "cudnn_benchmark", True, "bool"),
                ("Pin Memory", "pin_memory", True, "bool"),
                ("Shuffle (Dataloader)", "shuffle", True, "bool"),
                ("Prefetch Factor (Dataloader)", "prefetch_factor", 4, "int", 0, 16, 1),
                ("Persistent Workers (Dataloader)", "persistent_workers", True, "bool"),
                ("Drop Last Batch (Dataloader)", "drop_last", True, "bool"),
            ]

            self.optimization_params = []
            if not hasattr(self, "parameter_rows"):
                self.parameter_rows = {}

            for name, key, default, type_, *args in params:
                min_val, max_val, step = None, None, None
                if args:
                    if type_ == "int" or type_ == "float":  # Mimo że float nie ma tu
                        if len(args) == 3:
                            min_val, max_val, step = args
                        elif len(args) == 2:
                            min_val, max_val = args
                        elif len(args) == 1:
                            min_val = args[0]

                row = self._create_parameter_row(
                    name, key, default, type_, min_val, max_val, step
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
                f"Błąd podczas tworzenia zakładki optymalizacji: {e!s}", exc_info=True
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
            "memory_efficient": "memory_efficient",
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
                "memory_efficient": "memory_efficient",
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

            # Konfiguracja augmentacji
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
                "resize": {"enabled": self.resize_check.isChecked()},
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

        self.unfreeze_after_epochs_spin.setEnabled(is_enabled)

        self.unfreeze_after_epochs_spin.setEnabled(is_enabled)
