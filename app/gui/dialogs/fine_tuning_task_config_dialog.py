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

            # Model bazowy
            base_model_layout = QtWidgets.QHBoxLayout()
            self.base_model_edit = QtWidgets.QLineEdit()
            base_model_btn = QtWidgets.QPushButton("Przeglądaj...")
            base_model_btn.clicked.connect(self._select_model_dir)
            base_model_layout.addWidget(self.base_model_edit)
            base_model_layout.addWidget(base_model_btn)

            base_model_label = "Model bazowy:"
            form.addRow(base_model_label, base_model_layout)

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
            self.pretrained_check = QtWidgets.QCheckBox("Używaj pretrenowanego modelu")
            self.pretrained_check.setChecked(True)
            form.addRow("", self.pretrained_check)

            # Pretrained weights
            self.pretrained_weights_combo = QtWidgets.QComboBox()
            self.pretrained_weights_combo.addItems(
                ["imagenet", "imagenet21k", "noisy-student", "none"]
            )
            form.addRow("Pretrained weights:", self.pretrained_weights_combo)

            # Feature extraction only
            self.feature_extraction_check = QtWidgets.QCheckBox("Tylko ekstrakcja cech")
            self.feature_extraction_check.setChecked(False)
            form.addRow("", self.feature_extraction_check)

            # Activation
            self.activation_combo = QtWidgets.QComboBox()
            self.activation_combo.addItems(["swish", "relu", "silu", "mish", "gelu"])
            form.addRow("Aktywacja:", self.activation_combo)

            # Dropout at inference
            self.dropout_at_inference_check = QtWidgets.QCheckBox(
                "Dropout podczas inferencji"
            )
            self.dropout_at_inference_check.setChecked(False)
            form.addRow("", self.dropout_at_inference_check)

            # Global pool
            self.global_pool_combo = QtWidgets.QComboBox()
            self.global_pool_combo.addItems(["avg", "max", "token", "none"])
            form.addRow("Global pool:", self.global_pool_combo)

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

    def _select_model_dir(self):
        """Wybór katalogu z modelem bazowym."""
        try:
            title = "Wybierz katalog z modelem bazowym"
            dir_path = QtWidgets.QFileDialog.getExistingDirectory(self, title)

            if dir_path:
                self.base_model_edit.setText(dir_path)

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
        try:
            self.profile_list.clear()
            for profile_file in self.profiles_dir.glob("*.json"):
                try:
                    with open(profile_file, "r", encoding="utf-8") as f:
                        profile_data = json.load(f)
                        if profile_data.get("typ") == "tuning":
                            self.profile_list.addItem(profile_file.stem)
                except Exception as e:
                    self.logger.error(
                        f"Błąd podczas wczytywania profilu {profile_file}: {str(e)}",
                        exc_info=True,
                    )
        except Exception as e:
            self.logger.error(
                f"Błąd podczas odświeżania listy profili: {str(e)}", exc_info=True
            )

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
                    "typ": "tuning",
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
        """Obsługa zatwierdzenia konfiguracji."""
        try:
            # Sprawdź czy wybrano model bazowy
            if not self.base_model_edit.text():
                raise ValueError("Nie wybrano modelu bazowego")

            # Sprawdź czy wybrano katalog treningowy
            if not self.train_dir_edit.text():
                raise ValueError("Nie wybrano katalogu treningowego")

            # Przygotuj nazwę zadania
            base_model = self.base_model_edit.text()
            model_name = os.path.splitext(os.path.basename(base_model))[0]
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            task_name = f"{model_name}_{timestamp}.json"

            # Przygotuj konfigurację
            self.config = {
                "name": task_name,
                "type": "Doszkalanie",
                "status": "Nowy",
                "priority": 0,
                "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "config": {
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
                        "learning_rate": self.lr_spin.value(),
                        "optimizer": self.optimizer_combo.currentText(),
                        "scheduler": self.scheduler_combo.currentText(),
                        "num_workers": self.num_workers_spin.value(),
                        "warmup_epochs": self.warmup_epochs_spin.value(),
                        "mixed_precision": self.mixed_precision_check.isChecked(),
                        "warmup_lr_init": self.warmup_lr_init_spin.value(),
                        "gradient_accumulation_steps": self.grad_accum_steps_spin.value(),
                        "validation_split": self.validation_split_spin.value(),
                        "evaluation_freq": self.eval_freq_spin.value(),
                        "use_ema": self.use_ema_check.isChecked(),
                        "ema_decay": self.ema_decay_spin.value(),
                        "scheduler_params": {
                            "step_size": self.step_size_spin.value(),
                            "gamma": self.gamma_spin.value(),
                            "milestones": [
                                int(x.strip())
                                for x in self.milestones_edit.text().split(",")
                            ],
                            "T_max": self.t_max_spin.value(),
                            "eta_min": self.eta_min_spin.value(),
                            "patience": self.scheduler_patience_spin.value(),
                            "factor": self.scheduler_factor_spin.value(),
                            "threshold": self.scheduler_threshold_spin.value(),
                            "monitor": self.scheduler_monitor_combo.currentText(),
                        },
                    },
                    "regularization": {
                        "weight_decay": self.weight_decay_spin.value(),
                        "gradient_clip": self.gradient_clip_spin.value(),
                        "label_smoothing": self.label_smoothing_spin.value(),
                        "drop_connect_rate": self.drop_connect_spin.value(),
                        "dropout_rate": self.dropout_spin.value(),
                        "momentum": self.momentum_spin.value(),
                        "epsilon": self.epsilon_spin.value(),
                        "stochastic_depth": {
                            "enabled": self.stochastic_depth_check.isChecked(),
                            "drop_rate": self.stochastic_depth_rate.value(),
                            "survival_probability": self.stochastic_depth_survival.value(),
                        },
                        "mixup": {
                            "enabled": self.mixup_check.isChecked(),
                            "alpha": self.mixup_alpha.value(),
                            "prob": self.mixup_prob.value(),
                            "mode": self.mixup_mode.currentText(),
                        },
                        "cutmix": {
                            "enabled": self.cutmix_check.isChecked(),
                            "alpha": self.cutmix_alpha.value(),
                            "prob": self.cutmix_prob.value(),
                        },
                        "random_erase": {
                            "enabled": self.random_erase_check.isChecked(),
                            "prob": self.random_erase_prob.value(),
                            "mode": self.random_erase_mode.currentText(),
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
                        "advanced": {
                            "random_erasing": {
                                "enabled": self.random_erasing_check.isChecked(),
                                "probability": self.random_erasing_prob.value(),
                                "max_area": self.random_erasing_max_area.value(),
                            }
                        },
                        "resize_mode": self.resize_mode_combo.currentText(),
                        "normalization": {
                            "mean": [
                                float(x.strip())
                                for x in self.normalization_mean_edit.text().split(",")
                            ],
                            "std": [
                                float(x.strip())
                                for x in self.normalization_std_edit.text().split(",")
                            ],
                        },
                    },
                    "monitoring": {
                        "metrics": {
                            "accuracy": self.accuracy_check.isChecked(),
                            "precision": self.precision_check.isChecked(),
                            "recall": self.recall_check.isChecked(),
                            "f1": self.f1_check.isChecked(),
                            "auc": self.auc_check.isChecked(),
                            "confusion_matrix": self.confusion_matrix_check.isChecked(),
                            "top_k_accuracy": {
                                "enabled": self.topk_check.isChecked(),
                                "k": self.topk_value.value(),
                            },
                        },
                        "logging": {
                            "tensorboard": self.tensorboard_check.isChecked(),
                            "wandb": self.wandb_check.isChecked(),
                            "csv": self.csv_check.isChecked(),
                            "log_freq": self.log_freq_combo.currentText(),
                        },
                        "visualization": {
                            "gradcam": self.gradcam_check.isChecked(),
                            "feature_maps": self.feature_maps_check.isChecked(),
                            "prediction_samples": self.prediction_samples_check.isChecked(),
                            "num_samples": self.num_samples_spin.value(),
                        },
                    },
                    "data": {
                        "train_path": self.train_path_edit.text(),
                        "val_path": self.val_path_edit.text(),
                        "test_path": self.test_path_edit.text(),
                        "class_weights": self.class_weights_combo.currentText(),
                        "sampler": self.sampler_combo.currentText(),
                        "image_channels": self.image_channels_spin.value(),
                        "cache_dataset": self.cache_dataset_check.isChecked(),
                    },
                    "distributed": {
                        "use_distributed": self.use_distributed_check.isChecked(),
                        "backend": self.distributed_backend_combo.currentText(),
                        "sync_bn": self.sync_bn_check.isChecked(),
                        "find_unused_parameters": self.find_unused_params_check.isChecked(),
                    },
                    "inference": {
                        "test_time_augmentation": {
                            "enabled": self.use_tta_check.isChecked(),
                            "num_augments": self.num_augments_spin.value(),
                        },
                        "onnx_export": self.export_onnx_check.isChecked(),
                        "quantization": {
                            "enabled": self.use_quantization_check.isChecked(),
                            "precision": self.quantization_precision_combo.currentText(),
                        },
                    },
                    "seed": self.seed_spin.value(),
                    "deterministic": self.deterministic_check.isChecked(),
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

            self.stochastic_depth_survival = QtWidgets.QDoubleSpinBox()
            self.stochastic_depth_survival.setRange(0.0, 1.0)
            self.stochastic_depth_survival.setDecimals(3)
            self.stochastic_depth_survival.setValue(0.8)

            stochastic_depth_layout.addRow("", self.stochastic_depth_check)
            stochastic_depth_layout.addRow("Drop rate:", self.stochastic_depth_rate)
            stochastic_depth_layout.addRow(
                "Survival probability:", self.stochastic_depth_survival
            )
            stochastic_depth_group.setLayout(stochastic_depth_layout)

            # Mixup
            mixup_group = QtWidgets.QGroupBox("Mixup")
            mixup_layout = QtWidgets.QFormLayout()

            self.mixup_check = QtWidgets.QCheckBox("Używaj Mixup")
            self.mixup_alpha = QtWidgets.QDoubleSpinBox()
            self.mixup_alpha.setRange(0.0, 1.0)
            self.mixup_alpha.setDecimals(3)
            self.mixup_alpha.setValue(0.2)

            self.mixup_prob = QtWidgets.QDoubleSpinBox()
            self.mixup_prob.setRange(0.0, 1.0)
            self.mixup_prob.setDecimals(3)
            self.mixup_prob.setValue(1.0)

            self.mixup_mode = QtWidgets.QComboBox()
            self.mixup_mode.addItems(["batch", "pair", "elem"])

            mixup_layout.addRow("", self.mixup_check)
            mixup_layout.addRow("Alpha:", self.mixup_alpha)
            mixup_layout.addRow("Probability:", self.mixup_prob)
            mixup_layout.addRow("Mode:", self.mixup_mode)
            mixup_group.setLayout(mixup_layout)

            # CutMix
            cutmix_group = QtWidgets.QGroupBox("CutMix")
            cutmix_layout = QtWidgets.QFormLayout()

            self.cutmix_check = QtWidgets.QCheckBox("Używaj CutMix")
            self.cutmix_alpha = QtWidgets.QDoubleSpinBox()
            self.cutmix_alpha.setRange(0.0, 1.0)
            self.cutmix_alpha.setDecimals(3)
            self.cutmix_alpha.setValue(1.0)

            self.cutmix_prob = QtWidgets.QDoubleSpinBox()
            self.cutmix_prob.setRange(0.0, 1.0)
            self.cutmix_prob.setDecimals(3)
            self.cutmix_prob.setValue(0.5)

            cutmix_layout.addRow("", self.cutmix_check)
            cutmix_layout.addRow("Alpha:", self.cutmix_alpha)
            cutmix_layout.addRow("Probability:", self.cutmix_prob)
            cutmix_group.setLayout(cutmix_layout)

            # Random Erase
            random_erase_group = QtWidgets.QGroupBox("Random Erase")
            random_erase_layout = QtWidgets.QFormLayout()

            self.random_erase_check = QtWidgets.QCheckBox("Używaj Random Erase")
            self.random_erase_prob = QtWidgets.QDoubleSpinBox()
            self.random_erase_prob.setRange(0.0, 1.0)
            self.random_erase_prob.setDecimals(3)
            self.random_erase_prob.setValue(0.25)

            self.random_erase_mode = QtWidgets.QComboBox()
            self.random_erase_mode.addItems(["pixel", "block"])

            random_erase_layout.addRow("", self.random_erase_check)
            random_erase_layout.addRow("Probability:", self.random_erase_prob)
            random_erase_layout.addRow("Mode:", self.random_erase_mode)
            random_erase_group.setLayout(random_erase_layout)

            layout.addLayout(form)
            layout.addWidget(stochastic_depth_group)
            layout.addWidget(mixup_group)
            layout.addWidget(cutmix_group)
            layout.addWidget(random_erase_group)

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

            # Advanced augmentation
            advanced_group = QtWidgets.QGroupBox("Zaawansowana augmentacja")
            advanced_layout = QtWidgets.QFormLayout()

            # Random erasing
            self.random_erasing_check = QtWidgets.QCheckBox("Używaj Random Erasing")
            self.random_erasing_prob = QtWidgets.QDoubleSpinBox()
            self.random_erasing_prob.setRange(0.0, 1.0)
            self.random_erasing_prob.setDecimals(3)
            self.random_erasing_prob.setValue(0.5)

            self.random_erasing_max_area = QtWidgets.QDoubleSpinBox()
            self.random_erasing_max_area.setRange(0.0, 1.0)
            self.random_erasing_max_area.setDecimals(3)
            self.random_erasing_max_area.setValue(0.4)

            advanced_layout.addRow("", self.random_erasing_check)
            advanced_layout.addRow("Probability:", self.random_erasing_prob)
            advanced_layout.addRow("Max area:", self.random_erasing_max_area)

            # Resize mode
            self.resize_mode_combo = QtWidgets.QComboBox()
            self.resize_mode_combo.addItems(["bilinear", "bicubic", "nearest", "area"])
            advanced_layout.addRow("Resize mode:", self.resize_mode_combo)

            # Normalization
            normalization_group = QtWidgets.QGroupBox("Normalizacja")
            normalization_layout = QtWidgets.QFormLayout()

            self.normalization_mean_edit = QtWidgets.QLineEdit()
            self.normalization_mean_edit.setText("0.485,0.456,0.406")
            normalization_layout.addRow("Mean:", self.normalization_mean_edit)

            self.normalization_std_edit = QtWidgets.QLineEdit()
            self.normalization_std_edit.setText("0.229,0.224,0.225")
            normalization_layout.addRow("Std:", self.normalization_std_edit)

            normalization_group.setLayout(normalization_layout)
            advanced_layout.addWidget(normalization_group)

            advanced_group.setLayout(advanced_layout)
            layout.addWidget(basic_group)
            layout.addWidget(mixup_group)
            layout.addWidget(cutmix_group)
            layout.addWidget(autoaugment_group)
            layout.addWidget(randaugment_group)
            layout.addWidget(advanced_group)

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
            self.auc_check = QtWidgets.QCheckBox("AUC")
            self.confusion_matrix_check = QtWidgets.QCheckBox("Confusion Matrix")

            # Top-k accuracy
            topk_group = QtWidgets.QGroupBox("Top-k Accuracy")
            topk_layout = QtWidgets.QFormLayout()

            self.topk_check = QtWidgets.QCheckBox("Używaj Top-k Accuracy")
            self.topk_value = QtWidgets.QSpinBox()
            self.topk_value.setRange(1, 10)
            self.topk_value.setValue(5)

            topk_layout.addRow("", self.topk_check)
            topk_layout.addRow("k:", self.topk_value)
            topk_group.setLayout(topk_layout)

            metrics_layout.addWidget(self.accuracy_check)
            metrics_layout.addWidget(self.precision_check)
            metrics_layout.addWidget(self.recall_check)
            metrics_layout.addWidget(self.f1_check)
            metrics_layout.addWidget(self.auc_check)
            metrics_layout.addWidget(self.confusion_matrix_check)
            metrics_layout.addWidget(topk_group)
            metrics_group.setLayout(metrics_layout)

            # Logging
            logging_group = QtWidgets.QGroupBox("Logowanie")
            logging_layout = QtWidgets.QFormLayout()

            self.tensorboard_check = QtWidgets.QCheckBox("Używaj Tensorboard")
            self.wandb_check = QtWidgets.QCheckBox("Używaj Weights & Biases")
            self.csv_check = QtWidgets.QCheckBox("Zapisz do CSV")
            self.csv_check.setChecked(True)

            self.log_freq_combo = QtWidgets.QComboBox()
            self.log_freq_combo.addItems(["epoch", "batch"])
            self.log_freq_combo.setCurrentText("epoch")

            logging_layout.addRow("", self.tensorboard_check)
            logging_layout.addRow("", self.wandb_check)
            logging_layout.addRow("", self.csv_check)
            logging_layout.addRow("Częstość logowania:", self.log_freq_combo)
            logging_group.setLayout(logging_layout)

            # Wizualizacja
            visualization_group = QtWidgets.QGroupBox("Wizualizacja")
            visualization_layout = QtWidgets.QFormLayout()

            self.gradcam_check = QtWidgets.QCheckBox("Używaj GradCAM")
            self.feature_maps_check = QtWidgets.QCheckBox("Pokaż mapy cech")
            self.prediction_samples_check = QtWidgets.QCheckBox(
                "Pokaż przykłady predykcji"
            )

            self.num_samples_spin = QtWidgets.QSpinBox()
            self.num_samples_spin.setRange(1, 100)
            self.num_samples_spin.setValue(10)

            visualization_layout.addRow("", self.gradcam_check)
            visualization_layout.addRow("", self.feature_maps_check)
            visualization_layout.addRow("", self.prediction_samples_check)
            visualization_layout.addRow("Liczba próbek:", self.num_samples_spin)
            visualization_group.setLayout(visualization_layout)

            layout.addWidget(metrics_group)
            layout.addWidget(logging_group)
            layout.addWidget(visualization_group)

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

            # Data
            data_group = QtWidgets.QGroupBox("Dane")
            data_layout = QtWidgets.QFormLayout()

            self.train_path_edit = QtWidgets.QLineEdit()
            self.val_path_edit = QtWidgets.QLineEdit()
            self.test_path_edit = QtWidgets.QLineEdit()

            self.class_weights_combo = QtWidgets.QComboBox()
            self.class_weights_combo.addItems(["balanced", "none"])

            self.sampler_combo = QtWidgets.QComboBox()
            self.sampler_combo.addItems(["weighted_random", "uniform", "none"])

            self.image_channels_spin = QtWidgets.QSpinBox()
            self.image_channels_spin.setRange(1, 4)
            self.image_channels_spin.setValue(3)

            self.cache_dataset_check = QtWidgets.QCheckBox("Cache dataset")
            self.cache_dataset_check.setChecked(False)

            data_layout.addRow("Ścieżka treningowa:", self.train_path_edit)
            data_layout.addRow("Ścieżka walidacyjna:", self.val_path_edit)
            data_layout.addRow("Ścieżka testowa:", self.test_path_edit)
            data_layout.addRow("Wagi klas:", self.class_weights_combo)
            data_layout.addRow("Sampler:", self.sampler_combo)
            data_layout.addRow("Kanały obrazu:", self.image_channels_spin)
            data_layout.addRow("", self.cache_dataset_check)

            data_group.setLayout(data_layout)

            # Distributed
            distributed_group = QtWidgets.QGroupBox("Trening dystrybuowany")
            distributed_layout = QtWidgets.QFormLayout()

            self.use_distributed_check = QtWidgets.QCheckBox(
                "Używaj treningu dystrybuowanego"
            )
            self.use_distributed_check.setChecked(False)

            self.distributed_backend_combo = QtWidgets.QComboBox()
            self.distributed_backend_combo.addItems(["nccl", "gloo"])

            self.sync_bn_check = QtWidgets.QCheckBox("Synchronizuj BatchNorm")
            self.sync_bn_check.setChecked(True)

            self.find_unused_params_check = QtWidgets.QCheckBox(
                "Znajdź nieużywane parametry"
            )
            self.find_unused_params_check.setChecked(False)

            distributed_layout.addRow("", self.use_distributed_check)
            distributed_layout.addRow("Backend:", self.distributed_backend_combo)
            distributed_layout.addRow("", self.sync_bn_check)
            distributed_layout.addRow("", self.find_unused_params_check)

            distributed_group.setLayout(distributed_layout)

            # Inference
            inference_group = QtWidgets.QGroupBox("Inferencja")
            inference_layout = QtWidgets.QFormLayout()

            # Test time augmentation
            self.use_tta_check = QtWidgets.QCheckBox("Używaj TTA")
            self.use_tta_check.setChecked(False)

            self.num_augments_spin = QtWidgets.QSpinBox()
            self.num_augments_spin.setRange(1, 10)
            self.num_augments_spin.setValue(5)

            # ONNX export
            self.export_onnx_check = QtWidgets.QCheckBox("Eksportuj do ONNX")
            self.export_onnx_check.setChecked(False)

            # Quantization
            self.use_quantization_check = QtWidgets.QCheckBox("Używaj kwantyzacji")
            self.use_quantization_check.setChecked(False)

            self.quantization_precision_combo = QtWidgets.QComboBox()
            self.quantization_precision_combo.addItems(["int8", "fp16", "bf16"])

            inference_layout.addRow("", self.use_tta_check)
            inference_layout.addRow("Liczba augmentacji:", self.num_augments_spin)
            inference_layout.addRow("", self.export_onnx_check)
            inference_layout.addRow("", self.use_quantization_check)
            inference_layout.addRow("Precyzja:", self.quantization_precision_combo)

            inference_group.setLayout(inference_layout)

            # Seed i determinizm
            seed_group = QtWidgets.QGroupBox("Seed i determinizm")
            seed_layout = QtWidgets.QFormLayout()

            self.seed_spin = QtWidgets.QSpinBox()
            self.seed_spin.setRange(0, 1000000)
            self.seed_spin.setValue(42)

            self.deterministic_check = QtWidgets.QCheckBox("Tryb deterministyczny")
            self.deterministic_check.setChecked(True)

            seed_layout.addRow("Seed:", self.seed_spin)
            seed_layout.addRow("", self.deterministic_check)

            seed_group.setLayout(seed_layout)

            layout.addWidget(data_group)
            layout.addWidget(distributed_group)
            layout.addWidget(inference_group)
            layout.addWidget(seed_group)

            return tab

        except Exception as e:
            msg = "Błąd podczas tworzenia zakładki"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            raise

    def _get_unfreeze_layers_value(self, text):
        """Konwertuje tekst warstw do odmrożenia na wartość."""
        if text.lower() == "all":
            return "all"
        try:
            return [int(x.strip()) for x in text.split(",")]
        except ValueError:
            return "all"

    def _get_scheduler_value(self, display_text):
        """Konwertuje wyświetlaną wartość harmonogramu uczenia na wartość wewnętrzną."""
        if display_text == "None":
            return None
        return display_text

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
        """Zwraca listę wybranych metryk."""
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

            # Upewnij się, że typ jest ustawiony na "tuning"
            profile_data["typ"] = "tuning"
            profile_data["info"] = f"Klon profilu {item.text()}"
            profile_data["description"] = f"Klon profilu {item.text()}"

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

    def _create_training_params_tab(self):
        """Tworzenie zakładki Parametry Treningu."""
        try:
            self.logger.debug("Tworzenie zakładki parametrów treningu")
            tab = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(tab)
            form = QtWidgets.QFormLayout()

            # Warmup epochs
            self.warmup_epochs_spin = QtWidgets.QSpinBox()
            self.warmup_epochs_spin.setRange(0, 100)
            self.warmup_epochs_spin.setValue(5)
            form.addRow("Epoki rozgrzewki:", self.warmup_epochs_spin)

            # Warmup learning rate init
            self.warmup_lr_init_spin = QtWidgets.QDoubleSpinBox()
            self.warmup_lr_init_spin.setRange(0.000001, 0.1)
            self.warmup_lr_init_spin.setValue(0.000001)
            self.warmup_lr_init_spin.setDecimals(6)
            form.addRow("Początkowy learning rate:", self.warmup_lr_init_spin)

            # Gradient accumulation steps
            self.grad_accum_steps_spin = QtWidgets.QSpinBox()
            self.grad_accum_steps_spin.setRange(1, 32)
            self.grad_accum_steps_spin.setValue(1)
            form.addRow("Kroki akumulacji gradientu:", self.grad_accum_steps_spin)

            # Validation split
            self.validation_split_spin = QtWidgets.QDoubleSpinBox()
            self.validation_split_spin.setRange(0.1, 0.5)
            self.validation_split_spin.setValue(0.2)
            self.validation_split_spin.setDecimals(2)
            form.addRow("Podział walidacyjny:", self.validation_split_spin)

            # Evaluation frequency
            self.eval_freq_spin = QtWidgets.QSpinBox()
            self.eval_freq_spin.setRange(1, 100)
            self.eval_freq_spin.setValue(1)
            form.addRow("Częstość ewaluacji:", self.eval_freq_spin)

            # EMA
            self.use_ema_check = QtWidgets.QCheckBox("Używaj EMA")
            self.use_ema_check.setChecked(False)
            form.addRow("", self.use_ema_check)

            self.ema_decay_spin = QtWidgets.QDoubleSpinBox()
            self.ema_decay_spin.setRange(0.9, 0.9999)
            self.ema_decay_spin.setValue(0.9999)
            self.ema_decay_spin.setDecimals(4)
            form.addRow("EMA decay:", self.ema_decay_spin)

            # Scheduler parameters
            scheduler_params_group = QtWidgets.QGroupBox("Parametry schedulera")
            scheduler_params_layout = QtWidgets.QFormLayout()

            self.step_size_spin = QtWidgets.QSpinBox()
            self.step_size_spin.setRange(1, 100)
            self.step_size_spin.setValue(30)
            scheduler_params_layout.addRow("Step size:", self.step_size_spin)

            self.gamma_spin = QtWidgets.QDoubleSpinBox()
            self.gamma_spin.setRange(0.0, 1.0)
            self.gamma_spin.setValue(0.1)
            self.gamma_spin.setDecimals(3)
            scheduler_params_layout.addRow("Gamma:", self.gamma_spin)

            self.milestones_edit = QtWidgets.QLineEdit()
            self.milestones_edit.setText("30,60,90")
            scheduler_params_layout.addRow("Milestones:", self.milestones_edit)

            self.t_max_spin = QtWidgets.QSpinBox()
            self.t_max_spin.setRange(1, 1000)
            self.t_max_spin.setValue(100)
            scheduler_params_layout.addRow("T_max:", self.t_max_spin)

            self.eta_min_spin = QtWidgets.QDoubleSpinBox()
            self.eta_min_spin.setRange(0.000001, 0.1)
            self.eta_min_spin.setValue(0.000001)
            self.eta_min_spin.setDecimals(6)
            scheduler_params_layout.addRow("Eta min:", self.eta_min_spin)

            self.scheduler_patience_spin = QtWidgets.QSpinBox()
            self.scheduler_patience_spin.setRange(1, 100)
            self.scheduler_patience_spin.setValue(10)
            scheduler_params_layout.addRow("Patience:", self.scheduler_patience_spin)

            self.scheduler_factor_spin = QtWidgets.QDoubleSpinBox()
            self.scheduler_factor_spin.setRange(0.0, 1.0)
            self.scheduler_factor_spin.setValue(0.1)
            self.scheduler_factor_spin.setDecimals(3)
            scheduler_params_layout.addRow("Factor:", self.scheduler_factor_spin)

            self.scheduler_threshold_spin = QtWidgets.QDoubleSpinBox()
            self.scheduler_threshold_spin.setRange(0.0, 1.0)
            self.scheduler_threshold_spin.setValue(0.01)
            self.scheduler_threshold_spin.setDecimals(3)
            scheduler_params_layout.addRow("Threshold:", self.scheduler_threshold_spin)

            self.scheduler_monitor_combo = QtWidgets.QComboBox()
            self.scheduler_monitor_combo.addItems(["val_loss", "val_accuracy"])
            scheduler_params_layout.addRow("Monitor:", self.scheduler_monitor_combo)

            scheduler_params_group.setLayout(scheduler_params_layout)
            layout.addWidget(scheduler_params_group)

            return tab

        except Exception as e:
            msg = "Błąd podczas tworzenia zakładki"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            raise
