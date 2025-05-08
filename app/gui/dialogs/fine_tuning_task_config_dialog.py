import datetime
import json
import logging
import os
from pathlib import Path

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt

from app.utils.file_utils import (
    validate_training_directory,
    validate_validation_directory,
)


class FineTuningTaskConfigDialog(QtWidgets.QDialog):
    """Okno dialogowe do konfiguracji zadania fine-tuningu."""

    # Strategie odmrażania warstw
    UNFREEZE_ALL = "unfreeze_all"
    UNFREEZE_GRADUAL_END = "unfreeze_gradual_end"
    UNFREEZE_GRADUAL_START = "unfreeze_gradual_start"
    UNFREEZE_AFTER_EPOCHS = "unfreeze_after_epoochs"

    def __init__(self, parent=None, settings=None, hardware_profile=None):
        """Inicjalizacja okna dialogowego."""
        super().__init__(parent)
        self.settings = settings
        self.hardware_profile = hardware_profile
        self.logger = self._setup_logging()
        self.logger.info("Inicjalizacja okna konfiguracji fine-tuningu")
        self.setWindowTitle("Konfiguracja zadania fine-tuningu")
        self.setMinimumWidth(1000)
        self.profiles_dir = Path("data/profiles")
        self.profiles_dir.mkdir(exist_ok=True)
        self.current_profile = None
        self.config = None
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowCloseButtonHint)
        self._init_ui()

    def _setup_logging(self):
        """Konfiguracja logowania dla okna dialogowego."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        # Handler do pliku
        fh = logging.FileHandler("fine_tuning_dialog.log")
        fh.setLevel(logging.DEBUG)

        # Handler do konsoli
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Format logów
        log_fmt = "%(asctime)s - %(name)s - " "%(levelname)s - %(message)s"
        formatter = logging.Formatter(log_fmt)
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

        logger.info("Inicjalizacja okna")
        return logger

    def _init_ui(self):
        """Inicjalizacja interfejsu użytkownika z zakładkami."""
        try:
            self.logger.debug("Rozpoczęcie inicjalizacji UI")
            layout = QtWidgets.QVBoxLayout(self)

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

            info_group.setLayout(info_layout)
            profile_layout.addWidget(info_group)

            # Przyciski do obsługi profilu
            profile_buttons = QtWidgets.QHBoxLayout()

            edit_profile_btn = QtWidgets.QPushButton("Edytuj profil")
            edit_profile_btn.clicked.connect(self._edit_profile)
            profile_buttons.addWidget(edit_profile_btn)

            apply_profile_btn = QtWidgets.QPushButton("Zastosuj profil")
            apply_profile_btn.clicked.connect(self._apply_profile)
            profile_buttons.addWidget(apply_profile_btn)

            clone_profile_btn = QtWidgets.QPushButton("Klonuj profil")
            clone_profile_btn.clicked.connect(self._clone_profile)
            profile_buttons.addWidget(clone_profile_btn)

            save_profile_btn = QtWidgets.QPushButton("Zapisz profil")
            save_profile_btn.clicked.connect(self._save_profile)
            profile_buttons.addWidget(save_profile_btn)

            delete_profile_btn = QtWidgets.QPushButton("Usuń profil")
            delete_profile_btn.clicked.connect(self._delete_profile)
            profile_buttons.addWidget(delete_profile_btn)

            profile_layout.addLayout(profile_buttons)
            profile_group.setLayout(profile_layout)

            layout.addLayout(form)
            layout.addWidget(profile_group)

            return tab

        except Exception as e:
            msg = "Błąd podczas tworzenia zakładki"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            raise

    def _create_training_params_tab(self):
        """Tworzenie zakładki Parametry Treningu."""
        try:
            self.logger.debug("Tworzenie zakładki parametrów treningu")
            tab = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(tab)
            form = QtWidgets.QFormLayout()

            # Liczba epok
            self.epochs_spin = QtWidgets.QSpinBox()
            self.epochs_spin.setRange(1, 1000)
            self.epochs_spin.setValue(100)
            form.addRow("Liczba epok:", self.epochs_spin)

            # Rozmiar wsadu
            self.batch_size_spin = QtWidgets.QSpinBox()
            self.batch_size_spin.setRange(1, 512)
            self.batch_size_spin.setValue(32)
            form.addRow("Rozmiar wsadu:", self.batch_size_spin)

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

            # Liczba wątków
            self.num_workers_spin = QtWidgets.QSpinBox()
            self.num_workers_spin.setRange(0, 32)
            self.num_workers_spin.setValue(4)
            form.addRow("Liczba wątków:", self.num_workers_spin)

            # Epoki rozgrzewki
            self.warmup_epochs_spin = QtWidgets.QSpinBox()
            self.warmup_epochs_spin.setRange(0, 100)
            self.warmup_epochs_spin.setValue(5)
            form.addRow("Epoki rozgrzewki:", self.warmup_epochs_spin)

            # Mixed precision
            self.mixed_precision_check = QtWidgets.QCheckBox("Używaj mixed precision")
            self.mixed_precision_check.setChecked(True)
            form.addRow("", self.mixed_precision_check)

            # Zamrożenie modelu bazowego
            self.freeze_base_model = QtWidgets.QCheckBox("Zamroź model bazowy")
            self.freeze_base_model.setChecked(True)
            form.addRow("", self.freeze_base_model)

            # Warstwy do odmrożenia
            self.unfreeze_layers = QtWidgets.QLineEdit()
            self.unfreeze_layers.setPlaceholderText(
                "all lub numery warstw oddzielone przecinkami"
            )
            form.addRow("Warstwy do odmrożenia:", self.unfreeze_layers)

            # Strategia odmrażania
            self.unfreeze_strategy = QtWidgets.QComboBox()
            strategies = [
                "Wszystkie na raz (unfreeze_all)",
                "Stopniowo od końca (unfreeze_gradual_end)",
                "Stopniowo od początku (unfreeze_gradual_start)",
                "Po określonej liczbie epok (unfreeze_after_epoochs)",
            ]
            self.unfreeze_strategy.addItems(strategies)
            form.addRow("Strategia odmrażania:", self.unfreeze_strategy)

            layout.addLayout(form)
            return tab

        except Exception as e:
            msg = "Błąd podczas tworzenia zakładki"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            raise

    def _select_model_dir(self):
        """Wybór katalogu z modelem bazowym."""
        try:
            title = "Wybierz katalog z modelem bazowym"
            dir_path = QtWidgets.QFileDialog.getExistingDirectory(self, title)

            if dir_path:
                self.model_dir_edit.setText(dir_path)

        except Exception as e:
            msg = "Błąd wyboru katalogu"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)

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

    def _refresh_profile_list(self):
        """Odświeża listę dostępnych profili."""
        self.profile_list.clear()
        for profile_file in self.profiles_dir.glob("*.json"):
            self.profile_list.addItem(profile_file.stem)

    def _on_profile_selected(self, current, previous):
        """Obsługa wyboru profilu."""
        try:
            if not current:
                return

            profile_path = self.profiles_dir / f"{current.text()}.json"
            with open(profile_path, "r", encoding="utf-8") as f:
                profile_data = json.load(f)

            self.current_profile = profile_data
            self.profile_info.setText(profile_data.get("info", ""))
            self.profile_description.setText(profile_data.get("description", ""))
            self.profile_data_required.setText(profile_data.get("data_required", ""))

        except Exception as e:
            msg = "Błąd podczas ładowania profilu"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)

    def _edit_profile(self):
        """Otwarcie profilu w edytorze."""
        try:
            if not self.profile_list.currentItem():
                QtWidgets.QMessageBox.warning(
                    self, "Ostrzeżenie", "Najpierw wybierz profil!"
                )
                return

            item = self.profile_list.currentItem()
            profile_path = self.profiles_dir / f"{item.text()}.json"
            os.startfile(str(profile_path))  # Dla Windows

        except Exception as e:
            msg = "Błąd podczas otwierania profilu"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            QtWidgets.QMessageBox.critical(
                self,
                "Błąd",
                f"Nie można otworzyć profilu: {str(e)}",
            )

    def _apply_profile(self):
        """Zastosowanie wybranego profilu."""
        try:
            if not self.profile_list.currentItem():
                QtWidgets.QMessageBox.warning(
                    self, "Ostrzeżenie", "Najpierw wybierz profil!"
                )
                return

            item = self.profile_list.currentItem()
            profile_path = self.profiles_dir / f"{item.text()}.json"
            with open(profile_path, "r", encoding="utf-8") as f:
                profile_data = json.load(f)

            ft_config = profile_data.get("fine_tuning", {})

            # Podstawowe parametry
            self.epochs_spin.setValue(ft_config.get("epochs", 10))
            self.batch_size_spin.setValue(ft_config.get("batch_size", 32))
            self.lr_spin.setValue(ft_config.get("learning_rate", 0.001))
            self.optimizer_combo.setCurrentText(ft_config.get("optimizer", "Adam"))
            self.scheduler_combo.setCurrentText(
                ft_config.get("scheduler", "ReduceLROnPlateau")
            )
            self.num_workers_spin.setValue(ft_config.get("num_workers", 4))
            self.mixed_precision_check.setChecked(
                ft_config.get("mixed_precision", True)
            )

            # Strategia odmrażania
            unfreeze_strategy = ft_config.get("unfreeze_strategy", "unfreeze_all")
            self.unfreeze_strategy.setCurrentText(
                self._get_unfreeze_strategy_text(unfreeze_strategy)
            )
            self.unfreeze_layers.setText(str(ft_config.get("unfreeze_layers", "all")))

            # Regularyzacja
            reg_config = ft_config.get("regularization", {})
            self.weight_decay_spin.setValue(reg_config.get("weight_decay", 0.0))
            self.gradient_clip_spin.setValue(reg_config.get("gradient_clipping", 1.0))
            self.label_smoothing_spin.setValue(reg_config.get("label_smoothing", 0.0))
            self.dropout_spin.setValue(reg_config.get("dropout_rate", 0.0))

            # Zaawansowana regularyzacja
            stochastic_depth = reg_config.get("stochastic_depth", {})
            self.stochastic_depth_check.setChecked(
                stochastic_depth.get("enabled", False)
            )
            self.stochastic_depth_rate.setValue(stochastic_depth.get("drop_rate", 0.1))

            swa = reg_config.get("swa", {})
            self.swa_check.setChecked(swa.get("enabled", False))
            self.swa_start_epoch.setValue(swa.get("start_epoch", 10))
            self.swa_lr.setValue(swa.get("lr", 0.05))

            ema = reg_config.get("ema", {})
            self.ema_check.setChecked(ema.get("enabled", False))
            self.ema_decay.setValue(ema.get("decay", 0.9999))

            # PEFT
            peft_config = ft_config.get("peft", {})
            self.peft_technique.setCurrentText(peft_config.get("technique", "none"))

            lora = peft_config.get("lora", {})
            self.lora_rank.setValue(lora.get("rank", 8))
            self.lora_alpha.setValue(lora.get("alpha", 16))
            self.lora_dropout.setValue(lora.get("dropout", 0.1))
            self.lora_target_modules.setText(
                ",".join(lora.get("target_modules", ["query", "key", "value"]))
            )

            adapter = peft_config.get("adapter", {})
            self.adapter_hidden_size.setValue(adapter.get("hidden_size", 64))
            self.adapter_type.setCurrentText(adapter.get("adapter_type", "houlsby"))
            self.adapter_activation.setCurrentText(
                adapter.get("adapter_activation", "relu")
            )

            prompt_tuning = peft_config.get("prompt_tuning", {})
            self.num_virtual_tokens.setValue(
                prompt_tuning.get("num_virtual_tokens", 20)
            )
            self.prompt_init.setCurrentText(prompt_tuning.get("prompt_init", "random"))

            # Augmentacja
            aug_config = ft_config.get("augmentation", {})
            self.basic_aug_check.setChecked(aug_config.get("enabled", True))
            self.rotation_spin.setValue(aug_config.get("rotation", 0))
            self.brightness_spin.setValue(aug_config.get("brightness", 0.0))
            self.shift_spin.setValue(aug_config.get("shift", 0.0))
            self.zoom_spin.setValue(aug_config.get("zoom", 0.0))
            self.horizontal_flip_check.setChecked(
                aug_config.get("horizontal_flip", True)
            )
            self.vertical_flip_check.setChecked(aug_config.get("vertical_flip", False))

            # Preprocessing
            preproc_config = ft_config.get("preprocessing", {})
            self.normalization_combo.setCurrentText(
                preproc_config.get("normalization", "RGB")
            )

            scaling_config = preproc_config.get("scaling", {})
            self.scaling_method.setCurrentText(scaling_config.get("method", "Bilinear"))
            self.maintain_aspect_ratio.setChecked(
                scaling_config.get("maintain_aspect_ratio", True)
            )
            self.pad_to_square.setChecked(scaling_config.get("pad_to_square", False))
            self.pad_mode.setCurrentText(scaling_config.get("pad_mode", "constant"))
            self.pad_value.setValue(scaling_config.get("pad_value", 0))

            # Monitorowanie
            monitoring_config = ft_config.get("monitoring", {})
            self._apply_metrics_config(monitoring_config.get("metrics", []))

            viz_config = monitoring_config.get("visualization", {})
            self.tensorboard_check.setChecked(viz_config.get("tensorboard", False))
            self.wandb_check.setChecked(viz_config.get("wandb", False))
            self.log_histograms_check.setChecked(
                viz_config.get("log_histograms", False)
            )
            self.log_gradients_check.setChecked(viz_config.get("log_gradients", False))
            self.log_weights_check.setChecked(viz_config.get("log_weights", False))

            self._apply_early_stopping_config(
                monitoring_config.get("early_stopping", {})
            )
            self._apply_checkpointing_config(monitoring_config.get("checkpointing", {}))

            # Zaawansowane
            adv_config = ft_config.get("advanced", {})

            # Scheduler
            scheduler_config = adv_config.get("scheduler", {})
            self.scheduler_patience.setValue(scheduler_config.get("patience", 5))
            self.scheduler_factor.setValue(scheduler_config.get("factor", 0.1))
            self.min_lr.setValue(scheduler_config.get("min_lr", 1e-6))
            self.scheduler_cooldown.setValue(scheduler_config.get("cooldown", 0))

            # Transfer learning
            transfer_config = adv_config.get("transfer", {})
            self.unfreeze_after_epoch.setValue(
                transfer_config.get("unfreeze_after_epoch", 5)
            )
            self.gradual_unfreeze_rate.setValue(
                transfer_config.get("gradual_unfreeze_rate", 2)
            )
            self.feature_extraction_only.setChecked(
                transfer_config.get("feature_extraction_only", False)
            )
            self.custom_head.setChecked(transfer_config.get("custom_head", False))

            # Gradienty
            gradients_config = adv_config.get("gradients", {})
            self.grad_accum.setValue(gradients_config.get("accumulation", 1))
            self.freeze_bn.setChecked(gradients_config.get("freeze_bn", True))

            QtWidgets.QMessageBox.information(
                self, "Sukces", "Profil został pomyślnie zastosowany."
            )

        except Exception as e:
            error_msg = "Błąd podczas stosowania profilu: " + str(e)
            self.logger.error(error_msg, exc_info=True)
            QtWidgets.QMessageBox.critical(self, "Błąd", error_msg)

    def _save_profile(self):
        """Zapisanie aktualnej konfiguracji jako profil."""
        try:
            name, ok = QtWidgets.QInputDialog.getText(
                self,
                "Zapisz profil",
                "Podaj nazwę dla nowego profilu:",
                QtWidgets.QLineEdit.EchoMode.Normal,
                f"{self.arch_combo.currentText()}_{self.variant_combo.currentText()}",
            )

            if ok and name:
                profile_data = {
                    "info": f"Profil dla {self.arch_combo.currentText()} {self.variant_combo.currentText()}",
                    "description": "Profil utworzony przez użytkownika",
                    "data_required": "Standardowe dane treningowe",
                    "hardware_required": "Standardowy sprzęt",
                    "config": {
                        "model": {
                            "architecture": self.arch_combo.currentText(),
                            "variant": self.variant_combo.currentText(),
                            "input_size": self.input_size_spin.value(),
                            "num_classes": self.num_classes_spin.value(),
                        },
                        "training": {
                            "epochs": self.epochs_spin.value(),
                            "batch_size": self.batch_size_spin.value(),
                            "learning_rate": float(self.lr_spin.value()),
                            "optimizer": self.optimizer_combo.currentText(),
                            "scheduler": self.scheduler_combo.currentText(),
                            "num_workers": self.num_workers_spin.value(),
                            "warmup_epochs": self.warmup_epochs_spin.value(),
                            "mixed_precision": self.mixed_precision_check.isChecked(),
                        },
                        "regularization": {
                            "weight_decay": float(self.weight_decay_spin.value()),
                            "gradient_clip": self.gradient_clip_spin.value(),
                            "label_smoothing": self.label_smoothing_spin.value(),
                            "drop_connect_rate": self.drop_connect_spin.value(),
                            "dropout_rate": self.dropout_spin.value(),
                            "momentum": self.momentum_spin.value(),
                            "epsilon": self.epsilon_spin.value(),
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
                            }
                        },
                        "monitoring": {
                            "metrics": {
                                "accuracy": self.accuracy_check.isChecked(),
                                "precision": self.precision_check.isChecked(),
                                "recall": self.recall_check.isChecked(),
                                "f1": self.f1_check.isChecked(),
                            }
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

    def _get_unfreeze_strategy_text(self, strategy):
        """Konwertuje wartość wewnętrzną strategii odmrażania na tekst."""
        if strategy == self.UNFREEZE_ALL:
            return "Odmroź wszystkie warstwy"
        elif strategy == self.UNFREEZE_GRADUAL_END:
            return "Odmrażaj stopniowo od końca"
        elif strategy == self.UNFREEZE_GRADUAL_START:
            return "Odmrażaj stopniowo od początku"
        elif strategy == self.UNFREEZE_AFTER_EPOCHS:
            return "Po określonej liczbie epok"
        return "Odmroź wszystkie warstwy"

    def _get_unfreeze_strategy_value(self, display_text):
        """Konwertuje wyświetlaną wartość strategii odmrażania na wartość."""
        if "Odmroź wszystkie warstwy" in display_text:
            return self.UNFREEZE_ALL
        elif "Odmrażaj stopniowo od końca" in display_text:
            return self.UNFREEZE_GRADUAL_END
        elif "Odmrażaj stopniowo od początku" in display_text:
            return self.UNFREEZE_GRADUAL_START
        elif "Po określonej liczbie epok" in display_text:
            return self.UNFREEZE_AFTER_EPOCHS
        return self.UNFREEZE_ALL

    def _on_accept(self):
        """Obsługa akceptacji konfiguracji."""
        try:
            if not self._validate_basic_params():
                return

            # Przygotowanie konfiguracji
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = (
                f"{self.arch_combo.currentText()}_{self.variant_combo.currentText()}"
            )
            task_name = f"{model_name}_{timestamp}.json"

            self.config = {
                "name": task_name,
                "type": "Fine-tuning",
                "status": "Nowy",
                "priority": 0,
                "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "config": {
                    "train_dir": str(self.train_dir_edit.text()),
                    "data_dir": str(self.train_dir_edit.text()),
                    "val_dir": str(self.val_dir_edit.text()),
                    "model": {
                        "architecture": self.arch_combo.currentText(),
                        "variant": self.variant_combo.currentText(),
                        "input_size": self.input_size_spin.value(),
                        "num_classes": self.num_classes_spin.value(),
                    },
                    "training": {
                        "epochs": self.epochs_spin.value(),
                        "batch_size": self.batch_size_spin.value(),
                        "learning_rate": float(self.lr_spin.value()),
                        "optimizer": self.optimizer_combo.currentText(),
                        "scheduler": self._get_scheduler_value(
                            self.scheduler_combo.currentText()
                        ),
                        "num_workers": self.num_workers_spin.value(),
                        "warmup_epochs": self.warmup_epochs_spin.value(),
                        "mixed_precision": self.mixed_precision_check.isChecked(),
                        "freeze_base_model": self.freeze_base_model.isChecked(),
                        "unfreeze_layers": self._get_unfreeze_layers_value(
                            self.unfreeze_layers.text()
                        ),
                        "unfreeze_strategy": self._get_unfreeze_strategy_value(
                            self.unfreeze_strategy.currentText()
                        ),
                    },
                    "regularization": {
                        "weight_decay": float(self.weight_decay_spin.value()),
                        "gradient_clip": self.gradient_clip_spin.value(),
                        "label_smoothing": self.label_smoothing_spin.value(),
                        "drop_connect_rate": self.drop_connect_spin.value(),
                        "dropout_rate": self.dropout_spin.value(),
                        "momentum": self.momentum_spin.value(),
                        "epsilon": self.epsilon_spin.value(),
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
                        "randaugment": {
                            "use": self.randaugment_check.isChecked(),
                            "n": self.randaugment_n_spin.value(),
                            "m": self.randaugment_m_spin.value(),
                        },
                    },
                    "preprocessing": {
                        "normalization": self.normalization_combo.currentText(),
                        "scaling": {
                            "method": self.scaling_method.currentText(),
                            "maintain_aspect_ratio": self.maintain_aspect_ratio.isChecked(),
                            "pad_to_square": self.pad_to_square.isChecked(),
                            "pad_mode": self.pad_mode.currentText(),
                            "pad_value": self.pad_value.value(),
                        },
                    },
                    "monitoring": {
                        "metrics": {
                            "accuracy": self.accuracy_check.isChecked(),
                            "precision": self.precision_check.isChecked(),
                            "recall": self.recall_check.isChecked(),
                            "f1": self.f1_check.isChecked(),
                        },
                        "early_stopping": {
                            "patience": self.patience_spin.value(),
                            "min_delta": self.min_delta_spin.value(),
                            "monitor": self.monitor_combo.currentText(),
                        },
                        "checkpointing": {
                            "best_only": self.best_only_check.isChecked(),
                            "save_frequency": self.save_freq_spin.value(),
                            "metric": self.checkpoint_metric_combo.currentText(),
                        },
                        "tensorboard": {
                            "use": self.use_tensorboard_check.isChecked(),
                            "log_dir": self.tensorboard_dir_edit.text(),
                        },
                        "save_dir": self.model_dir_edit.text(),
                        "save_logs": self.save_logs_check.isChecked(),
                    },
                    "advanced": {
                        "scheduler": {
                            "patience": self.scheduler_patience.value(),
                            "factor": self.scheduler_factor.value(),
                            "min_lr": self.min_lr.value(),
                            "cooldown": self.scheduler_cooldown.value(),
                        },
                        "weights": {
                            "init_method": self.init_weights.currentText(),
                            "freeze_cnn": self.freeze_layers.isChecked(),
                        },
                        "cross_validation": {
                            "use": self.use_cv.isChecked(),
                            "folds": self.cv_folds.value(),
                        },
                        "distributed": {
                            "use": self.use_dist.isChecked(),
                            "backend": self.dist_backend.currentText(),
                            "strategy": self.dist_strategy.currentText(),
                        },
                        "gradients": {
                            "clip": self.grad_clip.value(),
                            "accumulation": self.grad_accum.value(),
                        },
                        "online_validation": {
                            "use": self.use_online_val.isChecked(),
                            "frequency": self.online_val_freq.value(),
                        },
                    },
                },
            }

            self.logger.info(f"Utworzono konfigurację zadania: {task_name}")
            QtWidgets.QMessageBox.information(
                self, "Sukces", "Zadanie zostało pomyślnie dodane."
            )

        except Exception as e:
            self.logger.error("Błąd podczas zapisywania konfiguracji", exc_info=True)
            QtWidgets.QMessageBox.critical(
                self,
                "Błąd",
                f"Wystąpił błąd podczas zapisywania konfiguracji: {str(e)}",
            )

    def get_task_config(self):
        """Zwraca konfigurację zadania lub None, jeśli nie dodano zadania."""
        return getattr(self, "task_config", None)

    def closeEvent(self, event):
        """Obsługa zamknięcia okna."""
        self.logger.info("Zamykanie okna dialogowego")
        self.accept()
        event.accept()

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
            self.epsilon_spin.setRange(0.000001, 0.1)
            self.epsilon_spin.setDecimals(6)
            self.epsilon_spin.setValue(0.000001)
            form.addRow("Epsilon:", self.epsilon_spin)

            # Stochastic Depth
            stochastic_depth_group = QtWidgets.QGroupBox("Stochastic Depth")
            stochastic_depth_layout = QtWidgets.QFormLayout()

            self.stochastic_depth_check = QtWidgets.QCheckBox("Używaj Stochastic Depth")
            self.stochastic_depth_rate = QtWidgets.QDoubleSpinBox()
            self.stochastic_depth_rate.setRange(0.0, 0.5)
            self.stochastic_depth_rate.setDecimals(3)
            self.stochastic_depth_rate.setValue(0.1)

            stochastic_depth_layout.addRow("", self.stochastic_depth_check)
            stochastic_depth_layout.addRow("Drop rate:", self.stochastic_depth_rate)
            stochastic_depth_group.setLayout(stochastic_depth_layout)

            # SWA
            swa_group = QtWidgets.QGroupBox("Stochastic Weight Averaging (SWA)")
            swa_layout = QtWidgets.QFormLayout()

            self.swa_check = QtWidgets.QCheckBox("Używaj SWA")
            self.swa_start_epoch = QtWidgets.QSpinBox()
            self.swa_start_epoch.setRange(1, 100)
            self.swa_start_epoch.setValue(10)
            self.swa_lr = QtWidgets.QDoubleSpinBox()
            self.swa_lr.setRange(0.000001, 0.1)
            self.swa_lr.setDecimals(6)
            self.swa_lr.setValue(0.05)

            swa_layout.addRow("", self.swa_check)
            swa_layout.addRow("Start epoki:", self.swa_start_epoch)
            swa_layout.addRow("Learning rate:", self.swa_lr)
            swa_group.setLayout(swa_layout)

            # EMA
            ema_group = QtWidgets.QGroupBox("Exponential Moving Average (EMA)")
            ema_layout = QtWidgets.QFormLayout()

            self.ema_check = QtWidgets.QCheckBox("Używaj EMA")
            self.ema_decay = QtWidgets.QDoubleSpinBox()
            self.ema_decay.setRange(0.0, 1.0)
            self.ema_decay.setDecimals(4)
            self.ema_decay.setValue(0.9999)

            ema_layout.addRow("", self.ema_check)
            ema_layout.addRow("Decay:", self.ema_decay)
            ema_group.setLayout(ema_layout)

            layout.addLayout(form)
            layout.addWidget(stochastic_depth_group)
            layout.addWidget(swa_group)
            layout.addWidget(ema_group)

            return tab

        except Exception as e:
            msg = "Błąd podczas tworzenia zakładki"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            raise

    def _create_augmentation_tab(self):
        """Tworzenie zakładki Augmentacja Danych."""
        try:
            self.logger.debug("Tworzenie zakładki augmentacji")
            tab = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(tab)

            # Podstawowa augmentacja
            basic_group = QtWidgets.QGroupBox("Podstawowa augmentacja")
            basic_layout = QtWidgets.QFormLayout()

            self.basic_aug_check = QtWidgets.QCheckBox("Używaj podstawowej augmentacji")
            self.basic_aug_check.setChecked(True)

            self.rotation_spin = QtWidgets.QSpinBox()
            self.rotation_spin.setRange(0, 360)
            self.rotation_spin.setValue(30)

            self.brightness_spin = QtWidgets.QDoubleSpinBox()
            self.brightness_spin.setRange(0.0, 2.0)
            self.brightness_spin.setValue(0.2)
            self.brightness_spin.setDecimals(2)

            self.shift_spin = QtWidgets.QDoubleSpinBox()
            self.shift_spin.setRange(0.0, 1.0)
            self.shift_spin.setValue(0.1)
            self.shift_spin.setDecimals(2)

            self.zoom_spin = QtWidgets.QDoubleSpinBox()
            self.zoom_spin.setRange(0.0, 1.0)
            self.zoom_spin.setValue(0.1)
            self.zoom_spin.setDecimals(2)

            self.horizontal_flip_check = QtWidgets.QCheckBox("Odwrócenie poziome")
            self.horizontal_flip_check.setChecked(True)

            self.vertical_flip_check = QtWidgets.QCheckBox("Odwrócenie pionowe")
            self.vertical_flip_check.setChecked(False)

            basic_layout.addRow("", self.basic_aug_check)
            basic_layout.addRow("Kąt rotacji:", self.rotation_spin)
            basic_layout.addRow("Jasność:", self.brightness_spin)
            basic_layout.addRow("Przesunięcie:", self.shift_spin)
            basic_layout.addRow("Przybliżenie:", self.zoom_spin)
            basic_layout.addRow("", self.horizontal_flip_check)
            basic_layout.addRow("", self.vertical_flip_check)
            basic_group.setLayout(basic_layout)

            # Mixup
            mixup_group = QtWidgets.QGroupBox("Mixup")
            mixup_layout = QtWidgets.QFormLayout()

            self.mixup_check = QtWidgets.QCheckBox("Używaj Mixup")
            self.mixup_alpha_spin = QtWidgets.QDoubleSpinBox()
            self.mixup_alpha_spin.setRange(0.0, 1.0)
            self.mixup_alpha_spin.setValue(0.2)
            self.mixup_alpha_spin.setDecimals(2)

            mixup_layout.addRow("", self.mixup_check)
            mixup_layout.addRow("Alpha:", self.mixup_alpha_spin)
            mixup_group.setLayout(mixup_layout)

            # CutMix
            cutmix_group = QtWidgets.QGroupBox("CutMix")
            cutmix_layout = QtWidgets.QFormLayout()

            self.cutmix_check = QtWidgets.QCheckBox("Używaj CutMix")
            self.cutmix_alpha_spin = QtWidgets.QDoubleSpinBox()
            self.cutmix_alpha_spin.setRange(0.0, 1.0)
            self.cutmix_alpha_spin.setValue(0.2)
            self.cutmix_alpha_spin.setDecimals(2)

            cutmix_layout.addRow("", self.cutmix_check)
            cutmix_layout.addRow("Alpha:", self.cutmix_alpha_spin)
            cutmix_group.setLayout(cutmix_layout)

            # AutoAugment
            autoaugment_group = QtWidgets.QGroupBox("AutoAugment")
            autoaugment_layout = QtWidgets.QFormLayout()

            self.autoaugment_check = QtWidgets.QCheckBox("Używaj AutoAugment")

            autoaugment_layout.addRow("", self.autoaugment_check)
            autoaugment_group.setLayout(autoaugment_layout)

            # RandAugment
            randaugment_group = QtWidgets.QGroupBox("RandAugment")
            randaugment_layout = QtWidgets.QFormLayout()

            self.randaugment_check = QtWidgets.QCheckBox("Używaj RandAugment")
            self.randaugment_n_spin = QtWidgets.QSpinBox()
            self.randaugment_n_spin.setRange(1, 10)
            self.randaugment_n_spin.setValue(2)

            self.randaugment_m_spin = QtWidgets.QSpinBox()
            self.randaugment_m_spin.setRange(1, 30)
            self.randaugment_m_spin.setValue(9)

            randaugment_layout.addRow("", self.randaugment_check)
            randaugment_layout.addRow("N:", self.randaugment_n_spin)
            randaugment_layout.addRow("M:", self.randaugment_m_spin)
            randaugment_group.setLayout(randaugment_layout)

            layout.addWidget(basic_group)
            layout.addWidget(mixup_group)
            layout.addWidget(cutmix_group)
            layout.addWidget(autoaugment_group)
            layout.addWidget(randaugment_group)

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

            # Metryki monitorowania
            metrics_group = QtWidgets.QGroupBox("Metryki")
            metrics_layout = QtWidgets.QVBoxLayout()

            self.accuracy_check = QtWidgets.QCheckBox("Accuracy")
            self.precision_check = QtWidgets.QCheckBox("Precision")
            self.recall_check = QtWidgets.QCheckBox("Recall")
            self.f1_check = QtWidgets.QCheckBox("F1 Score")
            self.topk_check = QtWidgets.QCheckBox("Top-k Accuracy")
            self.confusion_matrix_check = QtWidgets.QCheckBox("Confusion Matrix")

            metrics_layout.addWidget(self.accuracy_check)
            metrics_layout.addWidget(self.precision_check)
            metrics_layout.addWidget(self.recall_check)
            metrics_layout.addWidget(self.f1_check)
            metrics_layout.addWidget(self.topk_check)
            metrics_layout.addWidget(self.confusion_matrix_check)
            metrics_group.setLayout(metrics_layout)

            # Wczesne zatrzymanie
            early_stop_group = QtWidgets.QGroupBox("Wczesne zatrzymanie")
            early_stop_layout = QtWidgets.QFormLayout()

            self.patience_spin = QtWidgets.QSpinBox()
            self.patience_spin.setRange(1, 100)
            self.patience_spin.setValue(10)

            self.min_delta_spin = QtWidgets.QDoubleSpinBox()
            self.min_delta_spin.setRange(0.0, 1.0)
            self.min_delta_spin.setValue(0.001)

            self.monitor_combo = QtWidgets.QComboBox()
            self.monitor_combo.addItems(
                ["val_loss", "val_accuracy", "loss", "accuracy"]
            )

            early_stop_layout.addRow("Cierpliwość:", self.patience_spin)
            early_stop_layout.addRow("Minimalna zmiana:", self.min_delta_spin)
            early_stop_layout.addRow("Monitoruj:", self.monitor_combo)
            early_stop_group.setLayout(early_stop_layout)

            # Checkpointing
            checkpoint_group = QtWidgets.QGroupBox("Checkpointing")
            checkpoint_layout = QtWidgets.QFormLayout()

            self.best_only_check = QtWidgets.QCheckBox("Zapisz tylko najlepszy model")
            self.best_only_check.setChecked(True)

            self.save_freq_spin = QtWidgets.QSpinBox()
            self.save_freq_spin.setRange(1, 100)
            self.save_freq_spin.setValue(1)

            self.checkpoint_metric_combo = QtWidgets.QComboBox()
            self.checkpoint_metric_combo.addItems(["val_loss", "val_accuracy"])

            checkpoint_layout.addRow("", self.best_only_check)
            checkpoint_layout.addRow("Częstość zapisu:", self.save_freq_spin)
            checkpoint_layout.addRow("Metryka:", self.checkpoint_metric_combo)
            checkpoint_group.setLayout(checkpoint_layout)

            # Tensorboard
            tensorboard_group = QtWidgets.QGroupBox("Tensorboard")
            tensorboard_layout = QtWidgets.QFormLayout()

            self.use_tensorboard_check = QtWidgets.QCheckBox("Używaj Tensorboard")
            self.tensorboard_dir_edit = QtWidgets.QLineEdit()
            self.tensorboard_dir_edit.setText("runs/fine_tuning")

            tensorboard_layout.addRow("", self.use_tensorboard_check)
            tensorboard_layout.addRow("Katalog:", self.tensorboard_dir_edit)
            tensorboard_group.setLayout(tensorboard_layout)

            # Logi
            logs_group = QtWidgets.QGroupBox("Logi")
            logs_layout = QtWidgets.QFormLayout()

            self.save_logs_check = QtWidgets.QCheckBox("Zapisz logi")
            self.save_logs_check.setChecked(True)

            logs_layout.addRow("", self.save_logs_check)
            logs_group.setLayout(logs_layout)

            layout.addWidget(metrics_group)
            layout.addWidget(early_stop_group)
            layout.addWidget(checkpoint_group)
            layout.addWidget(tensorboard_group)
            layout.addWidget(logs_group)

            return tab

        except Exception as e:
            msg = "Błąd podczas tworzenia zakładki"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            raise

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

    def _create_advanced_tab(self):
        """Tworzenie zakładki Zaawansowane."""
        try:
            self.logger.debug("Tworzenie zakładki zaawansowanej")
            tab = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(tab)

            # Scheduler
            scheduler_group = QtWidgets.QGroupBox("Scheduler")
            scheduler_layout = QtWidgets.QFormLayout()

            self.scheduler_patience = QtWidgets.QSpinBox()
            self.scheduler_patience.setRange(1, 100)
            self.scheduler_patience.setValue(5)

            self.scheduler_factor = QtWidgets.QDoubleSpinBox()
            self.scheduler_factor.setRange(0.0, 1.0)
            self.scheduler_factor.setValue(0.1)
            self.scheduler_factor.setDecimals(3)

            self.min_lr = QtWidgets.QDoubleSpinBox()
            self.min_lr.setRange(0.000001, 0.1)
            self.min_lr.setValue(0.000001)
            self.min_lr.setDecimals(6)

            self.scheduler_cooldown = QtWidgets.QSpinBox()
            self.scheduler_cooldown.setRange(0, 100)
            self.scheduler_cooldown.setValue(0)

            scheduler_layout.addRow("Cierpliwość:", self.scheduler_patience)
            scheduler_layout.addRow("Faktor:", self.scheduler_factor)
            scheduler_layout.addRow("Min. współczynnik uczenia:", self.min_lr)
            scheduler_layout.addRow("Cooldown:", self.scheduler_cooldown)
            scheduler_group.setLayout(scheduler_layout)

            # Transfer learning
            transfer_group = QtWidgets.QGroupBox("Transfer learning")
            transfer_layout = QtWidgets.QFormLayout()

            self.unfreeze_after_epoch = QtWidgets.QSpinBox()
            self.unfreeze_after_epoch.setRange(0, 100)
            self.unfreeze_after_epoch.setValue(0)

            self.gradual_unfreeze_rate = QtWidgets.QSpinBox()
            self.gradual_unfreeze_rate.setRange(1, 10)
            self.gradual_unfreeze_rate.setValue(2)

            self.feature_extraction_only = QtWidgets.QCheckBox("Tylko ekstrakcja cech")
            self.feature_extraction_only.setChecked(False)

            self.custom_head = QtWidgets.QCheckBox("Używaj niestandardowej głowicy")
            self.custom_head.setChecked(False)

            transfer_layout.addRow("Odmroź po epoce:", self.unfreeze_after_epoch)
            transfer_layout.addRow("Stopień odmrażania:", self.gradual_unfreeze_rate)
            transfer_layout.addRow("", self.feature_extraction_only)
            transfer_layout.addRow("", self.custom_head)
            transfer_group.setLayout(transfer_layout)

            # Wagi
            weights_group = QtWidgets.QGroupBox("Wagi")
            weights_layout = QtWidgets.QFormLayout()

            self.init_weights = QtWidgets.QComboBox()
            self.init_weights.addItems(
                ["kaiming_normal", "kaiming_uniform", "xavier_normal", "xavier_uniform"]
            )
            self.init_weights.setCurrentText("kaiming_normal")

            self.freeze_layers = QtWidgets.QCheckBox("Zamroź warstwy CNN")
            self.freeze_layers.setChecked(True)

            weights_layout.addRow("Metoda inicjalizacji:", self.init_weights)
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

            self.grad_accum = QtWidgets.QSpinBox()
            self.grad_accum.setRange(1, 32)
            self.grad_accum.setValue(1)

            grad_layout.addRow("Gradient Clipping:", self.grad_clip)
            grad_layout.addRow("Gradient Accumulation:", self.grad_accum)
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

    def _get_unfreeze_layers_value(self, value):
        """Konwertuje wartość warstw do odmrożenia na listę."""
        if not value or value.lower() == "all":
            return "all"
        try:
            return [int(x.strip()) for x in value.split(",")]
        except ValueError:
            return "all"

    def _get_scheduler_value(self, display_text):
        """Konwertuje wyświetlaną wartość harmonogramu na wartość."""
        if "None" in display_text:
            return "None"
        elif "StepLR" in display_text:
            return "StepLR"
        elif "ReduceLROnPlateau" in display_text:
            return "ReduceLROnPlateau"
        elif "CosineAnnealingLR" in display_text:
            return "CosineAnnealingLR"
        elif "OneCycleLR" in display_text:
            return "OneCycleLR"
        elif "CosineAnnealingWarmRestarts" in display_text:
            return "CosineAnnealingWarmRestarts"
        return "None"

    def _validate_basic_params(self):
        """Walidacja podstawowych parametrów."""
        try:
            # Walidacja katalogu treningowego
            train_dir = self.train_dir_edit.text()
            if not train_dir.strip():
                self.logger.warning("Nie wybrano katalogu treningowego")
                QtWidgets.QMessageBox.critical(
                    self, "Błąd", "Musisz wybrać katalog danych treningowych!"
                )
                return False

            if not validate_training_directory(train_dir):
                self.logger.error(f"Nieprawidłowy katalog treningowy: {train_dir}")
                return False

            # Walidacja katalogu walidacyjnego
            val_dir = self.val_dir_edit.text()
            if val_dir and not validate_validation_directory(val_dir):
                self.logger.error(f"Nieprawidłowy katalog walidacyjny: {val_dir}")
                return False

            return True

        except Exception as e:
            self.logger.error(
                f"Błąd podczas walidacji parametrów: {str(e)}", exc_info=True
            )
            return False

    def _apply_checkpointing_config(self, config):
        """Stosuje konfigurację checkpointowania."""
        self.best_only_check.setChecked(config.get("best_only", True))
        self.save_freq_spin.setValue(config.get("save_frequency", 1))
        self.checkpoint_metric_combo.setCurrentText(config.get("monitor", "val_loss"))

    def _apply_early_stopping_config(self, config):
        """Stosuje konfigurację wczesnego zatrzymania."""
        self.patience_spin.setValue(config.get("patience", 10))
        self.min_delta_spin.setValue(config.get("min_delta", 0.001))
        self.monitor_combo.setCurrentText(config.get("monitor", "val_loss"))

    def _get_early_stopping_config(self):
        """Zwraca konfigurację wczesnego zatrzymania."""
        return {
            "patience": self.patience_spin.value(),
            "min_delta": self.min_delta_spin.value(),
            "monitor": self.monitor_combo.currentText(),
        }

    def _get_checkpointing_config(self, config):
        """Zwraca konfigurację checkpointowania."""
        return {
            "best_only": self.best_only_check.isChecked(),
            "save_frequency": self.save_freq_spin.value(),
            "monitor": self.checkpoint_metric_combo.currentText(),
        }

    def _get_selected_metrics(self):
        """Zwraca listę wybranych metryk monitorowania."""
        metrics = []
        if self.accuracy_check.isChecked():
            metrics.append("accuracy")
        if self.precision_check.isChecked():
            metrics.append("precision")
        if self.recall_check.isChecked():
            metrics.append("recall")
        if self.f1_check.isChecked():
            metrics.append("f1")
        if self.confusion_matrix_check.isChecked():
            metrics.append("confusion_matrix")
        if self.roc_auc_check.isChecked():
            metrics.append("roc_auc")
        if self.pr_auc_check.isChecked():
            metrics.append("pr_auc")
        if self.top_k_check.isChecked():
            metrics.append("top_k_accuracy")
        return metrics

    def _apply_metrics_config(self, metrics):
        """Stosuje konfigurację metryk."""
        self.accuracy_check.setChecked("accuracy" in metrics)
        self.precision_check.setChecked("precision" in metrics)
        self.recall_check.setChecked("recall" in metrics)
        self.f1_check.setChecked("f1" in metrics)
        self.confusion_matrix_check.setChecked("confusion_matrix" in metrics)
        self.roc_auc_check.setChecked("roc_auc" in metrics)
        self.pr_auc_check.setChecked("pr_auc" in metrics)
        self.top_k_check.setChecked("top_k_accuracy" in metrics)

    def _on_architecture_changed(self, arch_name):
        """Obsługa zmiany architektury modelu."""
        self._update_variant_combo(arch_name)

    def _update_variant_combo(self, arch_name):
        """Aktualizuje listę wariantów modelu w zależności od wybranej architektury."""
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
                ["ConvNeXt-Tiny", "ConvNeXt-Small", "ConvNeXt-Base", "ConvNeXt-Large"]
            )

    def _delete_profile(self):
        """Usuwa wybrany profil."""
        try:
            if not self.profile_list.currentItem():
                QtWidgets.QMessageBox.warning(
                    self, "Ostrzeżenie", "Najpierw wybierz profil!"
                )
                return

            item = self.profile_list.currentItem()
            profile_path = self.profiles_dir / f"{item.text()}.json"

            reply = QtWidgets.QMessageBox.question(
                self,
                "Potwierdzenie",
                f"Czy na pewno chcesz usunąć profil {item.text()}?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )

            if reply == QtWidgets.QMessageBox.StandardButton.Yes:
                profile_path.unlink()
                self._refresh_profile_list()
                QtWidgets.QMessageBox.information(
                    self, "Sukces", "Profil został pomyślnie usunięty."
                )

        except Exception as e:
            msg = "Błąd podczas usuwania profilu"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            QtWidgets.QMessageBox.critical(
                self,
                "Błąd",
                f"Wystąpił błąd podczas usuwania profilu: {str(e)}",
            )

    def _clone_profile(self):
        """Klonuje wybrany profil."""
        try:
            if not self.profile_list.currentItem():
                QtWidgets.QMessageBox.warning(
                    self, "Ostrzeżenie", "Najpierw wybierz profil!"
                )
                return

            item = self.profile_list.currentItem()
            profile_path = self.profiles_dir / f"{item.text()}.json"

            # Pobierz nazwę dla sklonowanego profilu
            new_name, ok = QtWidgets.QInputDialog.getText(
                self, "Klonuj profil", "Nazwa nowego profilu:"
            )
            if not ok or not new_name:
                return

            # Sprawdź czy profil o takiej nazwie już istnieje
            new_profile_path = self.profiles_dir / f"{new_name}.json"
            if new_profile_path.exists():
                QtWidgets.QMessageBox.warning(
                    self, "Ostrzeżenie", "Profil o takiej nazwie już istnieje!"
                )
                return

            # Wczytaj oryginalny profil
            with open(profile_path, "r", encoding="utf-8") as f:
                profile_data = json.load(f)

            # Zapisz sklonowany profil
            with open(new_profile_path, "w", encoding="utf-8") as f:
                json.dump(profile_data, f, indent=4)

            self._refresh_profile_list()
            QtWidgets.QMessageBox.information(
                self, "Sukces", "Profil został pomyślnie sklonowany."
            )

        except Exception as e:
            msg = "Błąd podczas klonowania profilu"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            QtWidgets.QMessageBox.critical(
                self,
                "Błąd",
                f"Wystąpił błąd podczas klonowania profilu: {str(e)}",
            )
