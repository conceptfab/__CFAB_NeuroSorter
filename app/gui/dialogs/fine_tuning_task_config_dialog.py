import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt

from app.utils.config import DEFAULT_TRAINING_PARAMS
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
    UNFREEZE_AFTER_EPOCHS = "unfreeze_after_epoochs"

    def __init__(self, parent=None, settings=None, hardware_profile=None):
        super().__init__(parent)
        self.settings = settings
        self.hardware_profile = hardware_profile
        self._setup_logging()
        self.setWindowTitle("Konfiguracja doszkalania")
        self.setMinimumWidth(1000)
        self.profiles_dir = Path("data/profiles")
        self.profiles_dir.mkdir(exist_ok=True)
        self.current_profile = None
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowCloseButtonHint)
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

    def _create_data_model_tab(self) -> QtWidgets.QWidget:
        """Tworzy zakładkę z konfiguracją danych i modelu."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        # Model
        model_group = QtWidgets.QGroupBox("Model")
        model_layout = QtWidgets.QFormLayout()

        # Architektura
        self.arch_combo = QtWidgets.QComboBox()
        self.arch_combo.addItems(["EfficientNet", "ResNet", "DenseNet", "MobileNet"])
        model_layout.addRow("Architektura:", self.arch_combo)

        # Wariant
        self.variant_combo = QtWidgets.QComboBox()
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
        model_layout.addRow("Wariant:", self.variant_combo)

        # Rozmiar wejścia
        self.input_size_spin = QtWidgets.QSpinBox()
        self.input_size_spin.setRange(32, 1024)
        self.input_size_spin.setValue(224)
        self.input_size_spin.setSingleStep(32)
        model_layout.addRow("Rozmiar wejścia:", self.input_size_spin)

        # Liczba klas
        self.num_classes_spin = QtWidgets.QSpinBox()
        self.num_classes_spin.setRange(2, 1000)
        self.num_classes_spin.setValue(2)
        model_layout.addRow("Liczba klas:", self.num_classes_spin)

        # Pretrained
        self.pretrained_check = QtWidgets.QCheckBox()
        self.pretrained_check.setChecked(True)
        model_layout.addRow("Użyj pretrained:", self.pretrained_check)

        # Pretrained weights
        self.pretrained_weights_combo = QtWidgets.QComboBox()
        self.pretrained_weights_combo.addItems(
            ["imagenet", "imagenet21k", "noisy-student"]
        )
        model_layout.addRow("Pretrained weights:", self.pretrained_weights_combo)

        # Feature extraction only
        self.feature_extraction_check = QtWidgets.QCheckBox()
        self.feature_extraction_check.setChecked(False)
        model_layout.addRow("Feature extraction only:", self.feature_extraction_check)

        # Activation
        self.activation_combo = QtWidgets.QComboBox()
        self.activation_combo.addItems(["swish", "relu", "silu", "mish", "gelu"])
        model_layout.addRow("Activation:", self.activation_combo)

        # Dropout at inference
        self.dropout_at_inference_check = QtWidgets.QCheckBox()
        self.dropout_at_inference_check.setChecked(False)
        model_layout.addRow("Dropout at inference:", self.dropout_at_inference_check)

        # Global pool
        self.global_pool_combo = QtWidgets.QComboBox()
        self.global_pool_combo.addItems(["avg", "max", "token", "none"])
        model_layout.addRow("Global pool:", self.global_pool_combo)

        # Last layer activation
        self.last_layer_activation_combo = QtWidgets.QComboBox()
        self.last_layer_activation_combo.addItems(["softmax", "sigmoid", "none"])
        model_layout.addRow("Last layer activation:", self.last_layer_activation_combo)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Dane
        data_group = QtWidgets.QGroupBox("Dane")
        data_layout = QtWidgets.QFormLayout()

        # Katalog treningowy
        train_dir_layout = QtWidgets.QHBoxLayout()
        self.train_dir_edit = QtWidgets.QLineEdit()
        self.train_dir_edit.setReadOnly(True)
        train_dir_button = QtWidgets.QPushButton("Wybierz")
        train_dir_button.clicked.connect(self._select_train_dir)
        train_dir_layout.addWidget(self.train_dir_edit)
        train_dir_layout.addWidget(train_dir_button)
        data_layout.addRow("Katalog treningowy:", train_dir_layout)

        # Katalog walidacyjny
        val_dir_layout = QtWidgets.QHBoxLayout()
        self.val_dir_edit = QtWidgets.QLineEdit()
        self.val_dir_edit.setReadOnly(True)
        val_dir_button = QtWidgets.QPushButton("Wybierz")
        val_dir_button.clicked.connect(self._select_val_dir)
        val_dir_layout.addWidget(self.val_dir_edit)
        val_dir_layout.addWidget(val_dir_button)
        data_layout.addRow("Katalog walidacyjny:", val_dir_layout)

        # Model do doszkalania
        model_path_layout = QtWidgets.QHBoxLayout()
        self.model_path_edit = QtWidgets.QLineEdit()
        self.model_path_edit.setReadOnly(True)
        model_path_button = QtWidgets.QPushButton("Wybierz")
        model_path_button.clicked.connect(self._select_model_path)
        model_path_layout.addWidget(self.model_path_edit)
        model_path_layout.addWidget(model_path_button)
        data_layout.addRow("Model do doszkalania:", model_path_layout)

        data_group.setLayout(data_layout)
        layout.addWidget(data_group)

        tab.setLayout(layout)
        return tab

    def _refresh_profile_list(self):
        """Odświeża listę dostępnych profili."""
        self.profile_list.clear()
        self.logger.debug("Rozpoczynam odświeżanie listy profili")
        for profile_file in self.profiles_dir.glob("*.json"):
            try:
                with open(profile_file, "r", encoding="utf-8") as f:
                    profile_data = json.load(f)
                    profile_type = profile_data.get("typ")
                    self.logger.debug(
                        f"Znaleziono profil {profile_file.stem} typu: {profile_type}"
                    )
                    if profile_type == "doszkalanie":
                        profile_name = profile_file.stem
                        self.profile_list.addItem(profile_name)
                        self.logger.debug(f"Dodano profil {profile_name} do listy")
                    else:
                        self.logger.debug(
                            f"Pominięto profil {profile_file.stem} - nieprawidłowy typ"
                        )
            except Exception as e:
                self.logger.error(
                    f"Błąd podczas wczytywania profilu {profile_file}: {str(e)}",
                    exc_info=True,
                )
        self.logger.debug(
            f"Zakończono odświeżanie listy profili. Liczba profili: {self.profile_list.count()}"
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

    def _apply_profile(self, profile_name: str) -> None:
        """Stosuje wybrany profil konfiguracji."""
        try:
            if profile_name == "default":
                config = {
                    "model": {
                        "pretrained": True,
                        "pretrained_weights": "imagenet",
                        "feature_extraction_only": False,
                        "activation": "swish",
                        "dropout_at_inference": False,
                        "global_pool": "avg",
                        "last_layer_activation": "softmax",
                    },
                    "training": {
                        "warmup_lr_init": 0.000001,
                        "gradient_accumulation_steps": 1,
                        "validation_split": 0.2,
                        "evaluation_freq": 1,
                        "use_ema": False,
                        "ema_decay": 0.9999,
                    },
                    "regularization": {
                        "swa": {
                            "use_swa": False,
                            "start_epoch": 10,
                        },
                        "stochastic_depth": {
                            "use_stochastic_depth": False,
                            "drop_rate": 0.2,
                            "survival_probability": 0.8,
                        },
                        "random_erase": {
                            "use_random_erase": False,
                            "probability": 0.25,
                            "mode": "pixel",
                        },
                    },
                    "augmentation": {
                        "contrast": 0.2,
                        "saturation": 0.2,
                        "hue": 0.1,
                        "shear": 0.1,
                        "channel_shift_range": 0.0,
                        "resize_mode": "bilinear",
                        "normalization": {
                            "mean": [0.485, 0.456, 0.406],
                            "std": [0.229, 0.224, 0.225],
                        },
                    },
                    "monitoring": {
                        "metrics": {
                            "accuracy": True,
                            "precision": True,
                            "recall": True,
                            "f1": True,
                            "top_k_accuracy": True,
                            "confusion_matrix": True,
                            "auc": True,
                        },
                        "logging": {
                            "use_tensorboard": True,
                            "use_wandb": False,
                            "save_to_csv": True,
                            "logging_freq": "epoch",
                        },
                        "visualization": {
                            "use_gradcam": True,
                            "use_feature_maps": True,
                            "use_prediction_samples": True,
                            "num_samples": 10,
                        },
                    },
                    "data": {
                        "class_weights": "balanced",
                        "sampler": "weighted_random",
                        "image_channels": 3,
                        "cache_dataset": False,
                    },
                    "inference": {
                        "tta": {
                            "use_tta": False,
                            "num_augmentations": 5,
                        },
                        "export_onnx": False,
                        "quantization": {
                            "use_quantization": False,
                            "precision": "int8",
                        },
                    },
                    "seed": 42,
                    "deterministic": True,
                }
            elif profile_name == "high_accuracy":
                config = {
                    "model": {
                        "pretrained": True,
                        "pretrained_weights": "imagenet21k",
                        "feature_extraction_only": False,
                        "activation": "swish",
                        "dropout_at_inference": True,
                        "global_pool": "avg",
                        "last_layer_activation": "softmax",
                    },
                    "training": {
                        "warmup_lr_init": 0.000001,
                        "gradient_accumulation_steps": 2,
                        "validation_split": 0.2,
                        "evaluation_freq": 1,
                        "use_ema": True,
                        "ema_decay": 0.9999,
                    },
                    "regularization": {
                        "swa": {
                            "use_swa": True,
                            "start_epoch": 10,
                        },
                        "stochastic_depth": {
                            "use_stochastic_depth": True,
                            "drop_rate": 0.2,
                            "survival_probability": 0.8,
                        },
                        "random_erase": {
                            "use_random_erase": True,
                            "probability": 0.25,
                            "mode": "pixel",
                        },
                    },
                    "augmentation": {
                        "contrast": 0.3,
                        "saturation": 0.3,
                        "hue": 0.2,
                        "shear": 0.2,
                        "channel_shift_range": 0.1,
                        "resize_mode": "bilinear",
                        "normalization": {
                            "mean": [0.485, 0.456, 0.406],
                            "std": [0.229, 0.224, 0.225],
                        },
                    },
                    "monitoring": {
                        "metrics": {
                            "accuracy": True,
                            "precision": True,
                            "recall": True,
                            "f1": True,
                            "top_k_accuracy": True,
                            "confusion_matrix": True,
                            "auc": True,
                        },
                        "logging": {
                            "use_tensorboard": True,
                            "use_wandb": True,
                            "save_to_csv": True,
                            "logging_freq": "batch",
                        },
                        "visualization": {
                            "use_gradcam": True,
                            "use_feature_maps": True,
                            "use_prediction_samples": True,
                            "num_samples": 20,
                        },
                    },
                    "data": {
                        "class_weights": "balanced",
                        "sampler": "weighted_random",
                        "image_channels": 3,
                        "cache_dataset": True,
                    },
                    "inference": {
                        "tta": {
                            "use_tta": True,
                            "num_augmentations": 10,
                        },
                        "export_onnx": True,
                        "quantization": {
                            "use_quantization": True,
                            "precision": "int8",
                        },
                    },
                    "seed": 42,
                    "deterministic": True,
                }
            elif profile_name == "fast_training":
                config = {
                    "model": {
                        "pretrained": True,
                        "pretrained_weights": "imagenet",
                        "feature_extraction_only": True,
                        "activation": "relu",
                        "dropout_at_inference": False,
                        "global_pool": "avg",
                        "last_layer_activation": "softmax",
                    },
                    "training": {
                        "warmup_lr_init": 0.0001,
                        "gradient_accumulation_steps": 1,
                        "validation_split": 0.1,
                        "evaluation_freq": 5,
                        "use_ema": False,
                        "ema_decay": 0.9999,
                    },
                    "regularization": {
                        "swa": {
                            "use_swa": False,
                            "start_epoch": 10,
                        },
                        "stochastic_depth": {
                            "use_stochastic_depth": False,
                            "drop_rate": 0.2,
                            "survival_probability": 0.8,
                        },
                        "random_erase": {
                            "use_random_erase": False,
                            "probability": 0.25,
                            "mode": "pixel",
                        },
                    },
                    "augmentation": {
                        "contrast": 0.1,
                        "saturation": 0.1,
                        "hue": 0.05,
                        "shear": 0.05,
                        "channel_shift_range": 0.0,
                        "resize_mode": "bilinear",
                        "normalization": {
                            "mean": [0.485, 0.456, 0.406],
                            "std": [0.229, 0.224, 0.225],
                        },
                    },
                    "monitoring": {
                        "metrics": {
                            "accuracy": True,
                            "precision": False,
                            "recall": False,
                            "f1": False,
                            "top_k_accuracy": False,
                            "confusion_matrix": False,
                            "auc": False,
                        },
                        "logging": {
                            "use_tensorboard": True,
                            "use_wandb": False,
                            "save_to_csv": True,
                            "logging_freq": "epoch",
                        },
                        "visualization": {
                            "use_gradcam": False,
                            "use_feature_maps": False,
                            "use_prediction_samples": True,
                            "num_samples": 5,
                        },
                    },
                    "data": {
                        "class_weights": "none",
                        "sampler": "uniform",
                        "image_channels": 3,
                        "cache_dataset": False,
                    },
                    "inference": {
                        "tta": {
                            "use_tta": False,
                            "num_augmentations": 5,
                        },
                        "export_onnx": False,
                        "quantization": {
                            "use_quantization": False,
                            "precision": "int8",
                        },
                    },
                    "seed": 42,
                    "deterministic": True,
                }
            else:
                raise ValueError(f"Nieznany profil: {profile_name}")

            self._load_config(config)

        except Exception as e:
            msg = "Błąd podczas stosowania profilu"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            QtWidgets.QMessageBox.critical(self, "Błąd", f"{msg}: {str(e)}")

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
                new_profile["typ"] = "doszkalanie"
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
                f"{self.arch_combo.currentText()}_{self.variant_combo.currentText()}",
            )
            if ok and name:
                profile_name = name
                profile_data = {
                    "typ": "doszkalanie",
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
                        },
                        "training": {
                            "epochs": self.epochs_spin.value(),
                            "batch_size": self.batch_size_spin.value(),
                            "learning_rate": float(self.lr_spin.value()),
                            "optimizer": self.optimizer_combo.currentText(),
                            "scheduler": self.scheduler_combo.currentText(),
                            "num_workers": self.num_workers_spin.value(),
                            "warmup_epochs": self.warmup_epochs_spin.value(),
                            "mixed_precision": (self.mixed_precision_check.isChecked()),
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
                            },
                        },
                        "stochastic_depth": {
                            "use": self.use_stoch_depth_check.isChecked(),
                            "drop_rate": self.stoch_depth_drop_rate.value(),
                            "survival_prob": self.stoch_depth_survival_prob.value(),
                        },
                        "random_erase": {
                            "use": self.use_random_erase_check.isChecked(),
                            "prob": self.random_erase_prob.value(),
                            "mode": self.random_erase_mode.currentText(),
                        },
                        "augmentation": {
                            "basic": {
                                "use": self.basic_aug_check.isChecked(),
                                "rotation": self.rotation_spin.value(),
                                "brightness": self.brightness_spin.value(),
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
                                "auc": self.auc_check.isChecked(),
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
                profile_path = self.profiles_dir / f"{profile_name}.json"
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
            profile_path = self.profiles_dir / f"{current_name}.json"
            reply = QtWidgets.QMessageBox.question(
                self,
                "Potwierdzenie",
                f"Czy na pewno chcesz usunąć profil '{current_name}'?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if reply == QtWidgets.QMessageBox.StandardButton.Yes:
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

    def _select_fine_tuning_dir(self):
        """Wybór katalogu danych do fine-tuningu."""
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

    def _select_model_file(self):
        """Wybór pliku modelu do doszkalania."""
        try:
            title = "Wybierz model do doszkalania"
            file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                title,
                "",
                "Pliki modeli PyTorch (*.pt *.pth);;Wszystkie pliki (*.*)",
            )

            if file_path:
                if file_path.lower().endswith((".pt", ".pth")):
                    self.model_path_edit.setText(file_path)
                else:
                    title = "Błąd"
                    msg = "Wybrany plik nie jest plikiem modelu PyTorch (*.pt, *.pth)"
                    QtWidgets.QMessageBox.warning(self, title, msg)

        except Exception as e:
            msg = "Błąd wyboru pliku modelu"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)

    def _create_fine_tuning_params_tab(self) -> QtWidgets.QWidget:
        """Tworzy zakładkę z parametrami fine-tuningu."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        # Training
        training_group = QtWidgets.QGroupBox("Training")
        training_layout = QtWidgets.QFormLayout()

        # Epochs
        self.epochs_spin = QtWidgets.QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(100)
        training_layout.addRow("Epochs:", self.epochs_spin)

        # Batch size
        self.batch_size_spin = QtWidgets.QSpinBox()
        self.batch_size_spin.setRange(1, 1024)
        self.batch_size_spin.setValue(32)
        training_layout.addRow("Batch size:", self.batch_size_spin)

        # Learning rate
        self.lr_spin = QtWidgets.QDoubleSpinBox()
        self.lr_spin.setRange(0.000001, 1.0)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setDecimals(6)
        training_layout.addRow("Learning rate:", self.lr_spin)

        # Optimizer
        self.optimizer_combo = QtWidgets.QComboBox()
        self.optimizer_combo.addItems(["Adam", "AdamW", "SGD", "RMSprop"])
        training_layout.addRow("Optimizer:", self.optimizer_combo)

        # Scheduler
        self.scheduler_combo = QtWidgets.QComboBox()
        self.scheduler_combo.addItems(
            ["None", "CosineAnnealingLR", "ReduceLROnPlateau", "OneCycleLR"]
        )
        training_layout.addRow("Scheduler:", self.scheduler_combo)

        # Number of workers
        self.num_workers_spin = QtWidgets.QSpinBox()
        self.num_workers_spin.setRange(0, 32)
        self.num_workers_spin.setValue(4)
        training_layout.addRow("Number of workers:", self.num_workers_spin)

        # Warmup epochs
        self.warmup_epochs_spin = QtWidgets.QSpinBox()
        self.warmup_epochs_spin.setRange(0, 100)
        self.warmup_epochs_spin.setValue(5)
        training_layout.addRow("Warmup epochs:", self.warmup_epochs_spin)

        # Mixed precision
        self.mixed_precision_check = QtWidgets.QCheckBox()
        self.mixed_precision_check.setChecked(True)
        training_layout.addRow("Mixed precision:", self.mixed_precision_check)

        # Freeze base model
        self.freeze_base_model = QtWidgets.QCheckBox()
        self.freeze_base_model.setChecked(True)
        training_layout.addRow("Freeze base model:", self.freeze_base_model)

        # Unfreeze layers
        self.unfreeze_layers = QtWidgets.QLineEdit()
        training_layout.addRow("Unfreeze layers:", self.unfreeze_layers)

        # Warmup learning rate init
        self.warmup_lr_init_spin = QtWidgets.QDoubleSpinBox()
        self.warmup_lr_init_spin.setRange(0.000001, 1.0)
        self.warmup_lr_init_spin.setValue(0.000001)
        self.warmup_lr_init_spin.setDecimals(6)
        training_layout.addRow("Warmup LR init:", self.warmup_lr_init_spin)

        # Gradient accumulation steps
        self.grad_accum_steps_spin = QtWidgets.QSpinBox()
        self.grad_accum_steps_spin.setRange(1, 32)
        self.grad_accum_steps_spin.setValue(1)
        training_layout.addRow(
            "Gradient accumulation steps:", self.grad_accum_steps_spin
        )

        # Validation split
        self.validation_split_spin = QtWidgets.QDoubleSpinBox()
        self.validation_split_spin.setRange(0.1, 0.5)
        self.validation_split_spin.setValue(0.2)
        self.validation_split_spin.setDecimals(2)
        training_layout.addRow("Validation split:", self.validation_split_spin)

        # Evaluation frequency
        self.eval_freq_spin = QtWidgets.QSpinBox()
        self.eval_freq_spin.setRange(1, 100)
        self.eval_freq_spin.setValue(1)
        training_layout.addRow("Evaluation frequency:", self.eval_freq_spin)

        # EMA
        ema_group = QtWidgets.QGroupBox("Exponential Moving Average (EMA)")
        ema_layout = QtWidgets.QFormLayout()

        self.use_ema_check = QtWidgets.QCheckBox()
        self.use_ema_check.setChecked(False)
        ema_layout.addRow("Use EMA:", self.use_ema_check)

        self.ema_decay_spin = QtWidgets.QDoubleSpinBox()
        self.ema_decay_spin.setRange(0.9, 0.9999)
        self.ema_decay_spin.setValue(0.9999)
        self.ema_decay_spin.setDecimals(4)
        ema_layout.addRow("EMA decay:", self.ema_decay_spin)

        ema_group.setLayout(ema_layout)
        training_layout.addRow(ema_group)

        training_group.setLayout(training_layout)
        layout.addWidget(training_group)

        tab.setLayout(layout)
        return tab

    def _create_regularization_tab(self) -> QtWidgets.QWidget:
        """Tworzy zakładkę z parametrami regularyzacji."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        # Regularization
        reg_group = QtWidgets.QGroupBox("Regularization")
        reg_layout = QtWidgets.QFormLayout()

        # Weight decay
        self.weight_decay_spin = QtWidgets.QDoubleSpinBox()
        self.weight_decay_spin.setRange(0.0, 1.0)
        self.weight_decay_spin.setValue(0.0001)
        self.weight_decay_spin.setDecimals(6)
        reg_layout.addRow("Weight decay:", self.weight_decay_spin)

        # Gradient clip
        self.gradient_clip_spin = QtWidgets.QDoubleSpinBox()
        self.gradient_clip_spin.setRange(0.0, 10.0)
        self.gradient_clip_spin.setValue(1.0)
        self.gradient_clip_spin.setDecimals(2)
        reg_layout.addRow("Gradient clip:", self.gradient_clip_spin)

        # Label smoothing
        self.label_smoothing_spin = QtWidgets.QDoubleSpinBox()
        self.label_smoothing_spin.setRange(0.0, 1.0)
        self.label_smoothing_spin.setValue(0.1)
        self.label_smoothing_spin.setDecimals(2)
        reg_layout.addRow("Label smoothing:", self.label_smoothing_spin)

        # Drop connect rate
        self.drop_connect_spin = QtWidgets.QDoubleSpinBox()
        self.drop_connect_spin.setRange(0.0, 1.0)
        self.drop_connect_spin.setValue(0.2)
        self.drop_connect_spin.setDecimals(2)
        reg_layout.addRow("Drop connect rate:", self.drop_connect_spin)

        # Dropout rate
        self.dropout_spin = QtWidgets.QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 1.0)
        self.dropout_spin.setValue(0.2)
        self.dropout_spin.setDecimals(2)
        reg_layout.addRow("Dropout rate:", self.dropout_spin)

        # Momentum
        self.momentum_spin = QtWidgets.QDoubleSpinBox()
        self.momentum_spin.setRange(0.0, 1.0)
        self.momentum_spin.setValue(0.9)
        self.momentum_spin.setDecimals(2)
        reg_layout.addRow("Momentum:", self.momentum_spin)

        # Epsilon
        self.epsilon_spin = QtWidgets.QDoubleSpinBox()
        self.epsilon_spin.setRange(1e-8, 1e-4)
        self.epsilon_spin.setValue(1e-6)
        self.epsilon_spin.setDecimals(8)
        reg_layout.addRow("Epsilon:", self.epsilon_spin)

        # SWA
        swa_group = QtWidgets.QGroupBox("Stochastic Weight Averaging (SWA)")
        swa_layout = QtWidgets.QFormLayout()

        self.use_swa_check = QtWidgets.QCheckBox()
        self.use_swa_check.setChecked(False)
        swa_layout.addRow("Use SWA:", self.use_swa_check)

        self.swa_start_epoch_spin = QtWidgets.QSpinBox()
        self.swa_start_epoch_spin.setRange(1, 1000)
        self.swa_start_epoch_spin.setValue(10)
        swa_layout.addRow("Start epoch:", self.swa_start_epoch_spin)

        swa_group.setLayout(swa_layout)
        reg_layout.addRow(swa_group)

        # Stochastic Depth
        stoch_depth_group = QtWidgets.QGroupBox("Stochastic Depth")
        stoch_depth_layout = QtWidgets.QFormLayout()

        self.use_stoch_depth_check = QtWidgets.QCheckBox()
        self.use_stoch_depth_check.setChecked(False)
        stoch_depth_layout.addRow("Use Stochastic Depth:", self.use_stoch_depth_check)

        self.stoch_depth_drop_rate = QtWidgets.QDoubleSpinBox()
        self.stoch_depth_drop_rate.setRange(0.0, 1.0)
        self.stoch_depth_drop_rate.setValue(0.2)
        self.stoch_depth_drop_rate.setDecimals(2)
        stoch_depth_layout.addRow("Drop rate:", self.stoch_depth_drop_rate)

        self.stoch_depth_survival_prob = QtWidgets.QDoubleSpinBox()
        self.stoch_depth_survival_prob.setRange(0.0, 1.0)
        self.stoch_depth_survival_prob.setValue(0.8)
        self.stoch_depth_survival_prob.setDecimals(2)
        stoch_depth_layout.addRow(
            "Survival probability:", self.stoch_depth_survival_prob
        )

        stoch_depth_group.setLayout(stoch_depth_layout)
        reg_layout.addRow(stoch_depth_group)

        # Random Erase
        random_erase_group = QtWidgets.QGroupBox("Random Erase")
        random_erase_layout = QtWidgets.QFormLayout()

        self.use_random_erase_check = QtWidgets.QCheckBox()
        self.use_random_erase_check.setChecked(False)
        random_erase_layout.addRow("Use Random Erase:", self.use_random_erase_check)

        self.random_erase_prob = QtWidgets.QDoubleSpinBox()
        self.random_erase_prob.setRange(0.0, 1.0)
        self.random_erase_prob.setValue(0.25)
        self.random_erase_prob.setDecimals(2)
        random_erase_layout.addRow("Probability:", self.random_erase_prob)

        self.random_erase_mode = QtWidgets.QComboBox()
        self.random_erase_mode.addItems(["pixel", "block"])
        random_erase_layout.addRow("Mode:", self.random_erase_mode)

        random_erase_group.setLayout(random_erase_layout)
        reg_layout.addRow(random_erase_group)

        reg_group.setLayout(reg_layout)
        layout.addWidget(reg_group)

        tab.setLayout(layout)
        return tab

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

        # Contrast
        self.contrast_spin = QtWidgets.QDoubleSpinBox()
        self.contrast_spin.setRange(0.0, 1.0)
        self.contrast_spin.setValue(0.2)
        self.contrast_spin.setDecimals(2)
        basic_layout.addRow("Contrast:", self.contrast_spin)

        # Saturation
        self.saturation_spin = QtWidgets.QDoubleSpinBox()
        self.saturation_spin.setRange(0.0, 1.0)
        self.saturation_spin.setValue(0.2)
        self.saturation_spin.setDecimals(2)
        basic_layout.addRow("Saturation:", self.saturation_spin)

        # Hue
        self.hue_spin = QtWidgets.QDoubleSpinBox()
        self.hue_spin.setRange(0.0, 0.5)
        self.hue_spin.setValue(0.1)
        self.hue_spin.setDecimals(2)
        basic_layout.addRow("Hue:", self.hue_spin)

        # Shear
        self.shear_spin = QtWidgets.QDoubleSpinBox()
        self.shear_spin.setRange(0.0, 1.0)
        self.shear_spin.setValue(0.1)
        self.shear_spin.setDecimals(2)
        basic_layout.addRow("Shear:", self.shear_spin)

        # Channel shift
        self.channel_shift_spin = QtWidgets.QDoubleSpinBox()
        self.channel_shift_spin.setRange(0.0, 1.0)
        self.channel_shift_spin.setValue(0.0)
        self.channel_shift_spin.setDecimals(2)
        basic_layout.addRow("Channel shift:", self.channel_shift_spin)

        basic_group.setLayout(basic_layout)
        aug_layout.addRow(basic_group)

        # Resize mode
        self.resize_mode_combo = QtWidgets.QComboBox()
        self.resize_mode_combo.addItems(["bilinear", "bicubic", "nearest", "area"])
        aug_layout.addRow("Resize mode:", self.resize_mode_combo)

        # Normalization
        norm_group = QtWidgets.QGroupBox("Normalization")
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
        aug_layout.addRow(norm_group)

        aug_group.setLayout(aug_layout)
        layout.addWidget(aug_group)

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

    def _create_monitoring_tab(self) -> QtWidgets.QWidget:
        """Tworzy zakładkę z parametrami monitorowania."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        # Monitoring
        monitor_group = QtWidgets.QGroupBox("Monitoring")
        monitor_layout = QtWidgets.QFormLayout()

        # Metrics
        metrics_group = QtWidgets.QGroupBox("Metrics")
        metrics_layout = QtWidgets.QVBoxLayout()

        self.accuracy_check = QtWidgets.QCheckBox("Accuracy")
        self.accuracy_check.setChecked(True)
        metrics_layout.addWidget(self.accuracy_check)

        self.precision_check = QtWidgets.QCheckBox("Precision")
        self.precision_check.setChecked(True)
        metrics_layout.addWidget(self.precision_check)

        self.recall_check = QtWidgets.QCheckBox("Recall")
        self.recall_check.setChecked(True)
        metrics_layout.addWidget(self.recall_check)

        self.f1_check = QtWidgets.QCheckBox("F1 Score")
        self.f1_check.setChecked(True)
        metrics_layout.addWidget(self.f1_check)

        self.topk_check = QtWidgets.QCheckBox("Top-K Accuracy")
        self.topk_check.setChecked(False)
        metrics_layout.addWidget(self.topk_check)

        self.confusion_matrix_check = QtWidgets.QCheckBox("Confusion Matrix")
        self.confusion_matrix_check.setChecked(False)
        metrics_layout.addWidget(self.confusion_matrix_check)

        self.auc_check = QtWidgets.QCheckBox("AUC")
        self.auc_check.setChecked(False)
        metrics_layout.addWidget(self.auc_check)

        metrics_group.setLayout(metrics_layout)
        monitor_layout.addRow(metrics_group)

        # Logging
        logging_group = QtWidgets.QGroupBox("Logging")
        logging_layout = QtWidgets.QFormLayout()

        self.use_tensorboard_check = QtWidgets.QCheckBox()
        self.use_tensorboard_check.setChecked(False)
        logging_layout.addRow("Use TensorBoard:", self.use_tensorboard_check)

        self.use_wandb_check = QtWidgets.QCheckBox()
        self.use_wandb_check.setChecked(False)
        logging_layout.addRow("Use Weights & Biases:", self.use_wandb_check)

        self.use_csv_check = QtWidgets.QCheckBox()
        self.use_csv_check.setChecked(True)
        logging_layout.addRow("Save to CSV:", self.use_csv_check)

        self.log_freq_combo = QtWidgets.QComboBox()
        self.log_freq_combo.addItems(["epoch", "batch"])
        logging_layout.addRow("Logging frequency:", self.log_freq_combo)

        logging_group.setLayout(logging_layout)
        monitor_layout.addRow(logging_group)

        # Visualization
        vis_group = QtWidgets.QGroupBox("Visualization")
        vis_layout = QtWidgets.QFormLayout()

        self.use_gradcam_check = QtWidgets.QCheckBox()
        self.use_gradcam_check.setChecked(False)
        vis_layout.addRow("Use GradCAM:", self.use_gradcam_check)

        self.use_feature_maps_check = QtWidgets.QCheckBox()
        self.use_feature_maps_check.setChecked(False)
        vis_layout.addRow("Use Feature Maps:", self.use_feature_maps_check)

        self.use_pred_samples_check = QtWidgets.QCheckBox()
        self.use_pred_samples_check.setChecked(False)
        vis_layout.addRow("Use Prediction Samples:", self.use_pred_samples_check)

        self.num_samples_spin = QtWidgets.QSpinBox()
        self.num_samples_spin.setRange(1, 100)
        self.num_samples_spin.setValue(10)
        vis_layout.addRow("Number of samples:", self.num_samples_spin)

        vis_group.setLayout(vis_layout)
        monitor_layout.addRow(vis_group)

        monitor_group.setLayout(monitor_layout)
        layout.addWidget(monitor_group)

        tab.setLayout(layout)
        return tab

    def _create_advanced_tab(self) -> QtWidgets.QWidget:
        """Tworzy zakładkę z zaawansowanymi parametrami."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        # Seed i determinizm
        seed_group = QtWidgets.QGroupBox("Seed i determinizm")
        seed_layout = QtWidgets.QFormLayout()

        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(42)
        seed_layout.addRow("Seed:", self.seed_spin)

        self.deterministic_check = QtWidgets.QCheckBox()
        self.deterministic_check.setChecked(True)
        seed_layout.addRow("Deterministic:", self.deterministic_check)

        seed_group.setLayout(seed_layout)
        layout.addWidget(seed_group)

        # Data
        data_group = QtWidgets.QGroupBox("Data")
        data_layout = QtWidgets.QFormLayout()

        self.class_weights_combo = QtWidgets.QComboBox()
        self.class_weights_combo.addItems(["balanced", "none"])
        data_layout.addRow("Class weights:", self.class_weights_combo)

        self.sampler_combo = QtWidgets.QComboBox()
        self.sampler_combo.addItems(["weighted_random", "uniform", "none"])
        data_layout.addRow("Sampler:", self.sampler_combo)

        self.image_channels_spin = QtWidgets.QSpinBox()
        self.image_channels_spin.setRange(1, 4)
        self.image_channels_spin.setValue(3)
        data_layout.addRow("Image channels:", self.image_channels_spin)

        self.cache_dataset_check = QtWidgets.QCheckBox()
        self.cache_dataset_check.setChecked(False)
        data_layout.addRow("Cache dataset:", self.cache_dataset_check)

        data_group.setLayout(data_layout)
        layout.addWidget(data_group)

        # Inference
        inference_group = QtWidgets.QGroupBox("Inference")
        inference_layout = QtWidgets.QFormLayout()

        # Test Time Augmentation
        tta_group = QtWidgets.QGroupBox("Test Time Augmentation (TTA)")
        tta_layout = QtWidgets.QFormLayout()

        self.use_tta_check = QtWidgets.QCheckBox()
        self.use_tta_check.setChecked(False)
        tta_layout.addRow("Use TTA:", self.use_tta_check)

        self.num_tta_spin = QtWidgets.QSpinBox()
        self.num_tta_spin.setRange(1, 10)
        self.num_tta_spin.setValue(5)
        tta_layout.addRow("Number of augmentations:", self.num_tta_spin)

        tta_group.setLayout(tta_layout)
        inference_layout.addRow(tta_group)

        # ONNX Export
        self.export_onnx_check = QtWidgets.QCheckBox()
        self.export_onnx_check.setChecked(False)
        inference_layout.addRow("Export to ONNX:", self.export_onnx_check)

        # Quantization
        quant_group = QtWidgets.QGroupBox("Quantization")
        quant_layout = QtWidgets.QFormLayout()

        self.quantization_check = QtWidgets.QCheckBox()
        self.quantization_check.setChecked(False)
        quant_layout.addRow("Use quantization:", self.quantization_check)

        self.quantization_precision_combo = QtWidgets.QComboBox()
        self.quantization_precision_combo.addItems(["int8", "fp16", "bf16"])
        quant_layout.addRow("Precision:", self.quantization_precision_combo)

        quant_group.setLayout(quant_layout)
        inference_layout.addRow(quant_group)

        inference_group.setLayout(inference_layout)
        layout.addWidget(inference_group)

        tab.setLayout(layout)
        return tab

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

    def _on_accept(self):
        """Obsługa akceptacji konfiguracji."""
        try:
            # Walidacja katalogu treningowego
            train_dir = self.train_dir_edit.text()
            if not train_dir.strip():
                self.logger.warning("Nie wybrano katalogu treningowego")
                QtWidgets.QMessageBox.critical(
                    self, "Błąd", "Musisz wybrać katalog danych treningowych!"
                )
                return

            if not validate_training_directory(train_dir):
                self.logger.error(f"Nieprawidłowy katalog treningowy: {train_dir}")
                return

            # Walidacja katalogu walidacyjnego
            val_dir = self.val_dir_edit.text()
            if val_dir and not validate_validation_directory(val_dir):
                self.logger.error(f"Nieprawidłowy katalog walidacyjny: {val_dir}")
                return

            # Przygotowanie konfiguracji
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = (
                f"{self.arch_combo.currentText()}_"
                f"{self.variant_combo.currentText()}"
            )
            task_name = f"{model_name}_{timestamp}.json"

            config = {
                "name": task_name,
                "typ": "doszkalanie",
                "status": "Nowy",
                "priority": 0,
                "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "config": {
                    "train_dir": train_dir,
                    "data_dir": train_dir,
                    "val_dir": val_dir,
                    "model": {
                        "pretrained": self.pretrained_check.isChecked(),
                        "pretrained_weights": self.pretrained_weights_combo.currentText(),
                        "feature_extraction_only": self.feature_extraction_check.isChecked(),
                        "activation": self.activation_combo.currentText(),
                        "dropout_at_inference": self.dropout_at_inference_check.isChecked(),
                        "global_pool": self.global_pool_combo.currentText(),
                        "last_layer_activation": self.last_layer_activation_combo.currentText(),
                    },
                    "training": {
                        "warmup_lr_init": self.warmup_lr_init_spin.value(),
                        "gradient_accumulation_steps": self.grad_accum_steps_spin.value(),
                        "validation_split": self.validation_split_spin.value(),
                        "evaluation_freq": self.eval_freq_spin.value(),
                        "use_ema": self.use_ema_check.isChecked(),
                        "ema_decay": self.ema_decay_spin.value(),
                    },
                    "regularization": {
                        "swa": {
                            "use_swa": self.use_swa_check.isChecked(),
                            "start_epoch": self.swa_start_epoch_spin.value(),
                        },
                        "stochastic_depth": {
                            "use_stochastic_depth": self.use_stoch_depth_check.isChecked(),
                            "drop_rate": self.stoch_depth_drop_rate.value(),
                            "survival_probability": self.stoch_depth_survival_prob.value(),
                        },
                        "random_erase": {
                            "use_random_erase": self.use_random_erase_check.isChecked(),
                            "probability": self.random_erase_prob.value(),
                            "mode": self.random_erase_mode.currentText(),
                        },
                    },
                    "augmentation": {
                        "contrast": self.contrast_spin.value(),
                        "saturation": self.saturation_spin.value(),
                        "hue": self.hue_spin.value(),
                        "shear": self.shear_spin.value(),
                        "channel_shift_range": self.channel_shift_spin.value(),
                        "resize_mode": self.resize_mode_combo.currentText(),
                        "normalization": {
                            "mean": [
                                self.norm_mean_r.value(),
                                self.norm_mean_g.value(),
                                self.norm_mean_b.value(),
                            ],
                            "std": [
                                self.norm_std_r.value(),
                                self.norm_std_g.value(),
                                self.norm_std_b.value(),
                            ],
                        },
                    },
                    "monitoring": {
                        "metrics": {
                            "accuracy": self.accuracy_check.isChecked(),
                            "precision": self.precision_check.isChecked(),
                            "recall": self.recall_check.isChecked(),
                            "f1": self.f1_check.isChecked(),
                            "top_k_accuracy": self.topk_check.isChecked(),
                            "confusion_matrix": self.confusion_matrix_check.isChecked(),
                            "auc": self.auc_check.isChecked(),
                        },
                        "logging": {
                            "use_tensorboard": self.use_tensorboard_check.isChecked(),
                            "use_wandb": self.use_wandb_check.isChecked(),
                            "save_to_csv": self.use_csv_check.isChecked(),
                            "logging_freq": self.log_freq_combo.currentText(),
                        },
                        "visualization": {
                            "use_gradcam": self.use_gradcam_check.isChecked(),
                            "use_feature_maps": self.use_feature_maps_check.isChecked(),
                            "use_prediction_samples": self.use_pred_samples_check.isChecked(),
                            "num_samples": self.num_samples_spin.value(),
                        },
                    },
                    "data": {
                        "class_weights": self.class_weights_combo.currentText(),
                        "sampler": self.sampler_combo.currentText(),
                        "image_channels": self.image_channels_spin.value(),
                        "cache_dataset": self.cache_dataset_check.isChecked(),
                    },
                    "inference": {
                        "tta": {
                            "use_tta": self.use_tta_check.isChecked(),
                            "num_augmentations": self.num_tta_spin.value(),
                        },
                        "export_onnx": self.export_onnx_check.isChecked(),
                        "quantization": {
                            "use_quantization": self.quantization_check.isChecked(),
                            "precision": self.quantization_precision_combo.currentText(),
                        },
                    },
                    "seed": self.seed_spin.value(),
                    "deterministic": self.deterministic_check.isChecked(),
                },
            }

            self.task_config = config

            # Zapisz konfigurację do pliku
            task_file = os.path.join("data", "tasks", task_name)
            os.makedirs(os.path.dirname(task_file), exist_ok=True)
            with open(task_file, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)

            self.accept()

        except Exception as e:
            msg = "Błąd podczas zapisywania konfiguracji"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            QtWidgets.QMessageBox.critical(self, "Błąd", f"{msg}: {str(e)}")

    def get_task_config(self):
        """Zwraca konfigurację zadania lub None, jeśli nie dodano zadania."""
        return getattr(self, "task_config", None)

    def closeEvent(self, event):
        """Obsługa zamknięcia okna."""
        self.logger.info("Zamykanie okna dialogowego")
        self.accept()
        event.accept()

    def _load_config(self, config: Dict[str, Any]) -> None:
        """Ładuje konfigurację do interfejsu."""
        try:
            # Model
            model_config = config.get("model", {})
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

            # Training
            training_config = config.get("training", {})
            self.warmup_lr_init_spin.setValue(
                training_config.get("warmup_lr_init", 0.000001)
            )
            self.gradient_accumulation_steps_spin.setValue(
                training_config.get("gradient_accumulation_steps", 1)
            )
            self.validation_split_spin.setValue(
                training_config.get("validation_split", 0.2)
            )
            self.evaluation_freq_spin.setValue(
                training_config.get("evaluation_freq", 1)
            )
            self.use_ema_check.setChecked(training_config.get("use_ema", False))
            self.ema_decay_spin.setValue(training_config.get("ema_decay", 0.9999))

            # Regularization
            regularization_config = config.get("regularization", {})

            # SWA
            swa_config = regularization_config.get("swa", {})
            self.use_swa_check.setChecked(swa_config.get("use_swa", False))
            self.swa_start_epoch_spin.setValue(swa_config.get("start_epoch", 10))

            # Stochastic Depth
            stoch_depth_config = regularization_config.get("stochastic_depth", {})
            self.use_stochastic_depth_check.setChecked(
                stoch_depth_config.get("use_stochastic_depth", False)
            )
            self.stochastic_depth_drop_rate_spin.setValue(
                stoch_depth_config.get("drop_rate", 0.2)
            )
            self.stochastic_depth_survival_prob_spin.setValue(
                stoch_depth_config.get("survival_probability", 0.8)
            )

            # Random Erase
            random_erase_config = regularization_config.get("random_erase", {})
            self.use_random_erase_check.setChecked(
                random_erase_config.get("use_random_erase", False)
            )
            self.random_erase_prob_spin.setValue(
                random_erase_config.get("probability", 0.25)
            )
            self.random_erase_mode_combo.setCurrentText(
                random_erase_config.get("mode", "pixel")
            )

            # Augmentation
            augmentation_config = config.get("augmentation", {})
            self.contrast_spin.setValue(augmentation_config.get("contrast", 0.2))
            self.saturation_spin.setValue(augmentation_config.get("saturation", 0.2))
            self.hue_spin.setValue(augmentation_config.get("hue", 0.1))
            self.shear_spin.setValue(augmentation_config.get("shear", 0.1))
            self.channel_shift_spin.setValue(
                augmentation_config.get("channel_shift_range", 0.0)
            )
            self.resize_mode_combo.setCurrentText(
                augmentation_config.get("resize_mode", "bilinear")
            )

            # Normalization
            normalization_config = augmentation_config.get("normalization", {})
            mean = normalization_config.get("mean", [0.485, 0.456, 0.406])
            std = normalization_config.get("std", [0.229, 0.224, 0.225])
            self.normalization_mean_r_spin.setValue(mean[0])
            self.normalization_mean_g_spin.setValue(mean[1])
            self.normalization_mean_b_spin.setValue(mean[2])
            self.normalization_std_r_spin.setValue(std[0])
            self.normalization_std_g_spin.setValue(std[1])
            self.normalization_std_b_spin.setValue(std[2])

            # Monitoring
            monitoring_config = config.get("monitoring", {})

            # Metrics
            metrics_config = monitoring_config.get("metrics", {})
            self.accuracy_check.setChecked(metrics_config.get("accuracy", True))
            self.precision_check.setChecked(metrics_config.get("precision", True))
            self.recall_check.setChecked(metrics_config.get("recall", True))
            self.f1_check.setChecked(metrics_config.get("f1", True))
            self.top_k_accuracy_check.setChecked(
                metrics_config.get("top_k_accuracy", True)
            )
            self.confusion_matrix_check.setChecked(
                metrics_config.get("confusion_matrix", True)
            )
            self.auc_check.setChecked(metrics_config.get("auc", True))

            # Logging
            logging_config = monitoring_config.get("logging", {})
            self.use_tensorboard_check.setChecked(
                logging_config.get("use_tensorboard", True)
            )
            self.use_wandb_check.setChecked(logging_config.get("use_wandb", False))
            self.save_to_csv_check.setChecked(logging_config.get("save_to_csv", True))
            self.logging_freq_combo.setCurrentText(
                logging_config.get("logging_freq", "epoch")
            )

            # Visualization
            visualization_config = monitoring_config.get("visualization", {})
            self.use_gradcam_check.setChecked(
                visualization_config.get("use_gradcam", True)
            )
            self.use_feature_maps_check.setChecked(
                visualization_config.get("use_feature_maps", True)
            )
            self.use_pred_samples_check.setChecked(
                visualization_config.get("use_prediction_samples", True)
            )
            self.num_samples_spin.setValue(visualization_config.get("num_samples", 10))

            # Data
            data_config = config.get("data", {})
            self.class_weights_combo.setCurrentText(
                data_config.get("class_weights", "balanced")
            )
            self.sampler_combo.setCurrentText(
                data_config.get("sampler", "weighted_random")
            )
            self.image_channels_spin.setValue(data_config.get("image_channels", 3))
            self.cache_dataset_check.setChecked(data_config.get("cache_dataset", False))

            # Inference
            inference_config = config.get("inference", {})

            # TTA
            tta_config = inference_config.get("tta", {})
            self.use_tta_check.setChecked(tta_config.get("use_tta", False))
            self.num_tta_spin.setValue(tta_config.get("num_augmentations", 5))

            # ONNX Export
            self.export_onnx_check.setChecked(
                inference_config.get("export_onnx", False)
            )

            # Quantization
            quantization_config = inference_config.get("quantization", {})
            self.quantization_check.setChecked(
                quantization_config.get("use_quantization", False)
            )
            self.quantization_precision_combo.setCurrentText(
                quantization_config.get("precision", "int8")
            )

            # Seed i determinizm
            self.seed_spin.setValue(config.get("seed", 42))
            self.deterministic_check.setChecked(config.get("deterministic", True))

            self.logger.info("=== REKOMENDACJE SPRZĘTOWE ===")
            self.logger.info(
                f"Zalecany rozmiar batcha: {self.hardware_profile.get('recommended_batch_size', 'N/A')}"
            )
            self.logger.info(
                f"Zalecana liczba workerów: {self.hardware_profile.get('recommended_workers', 'N/A')}"
            )
            self.logger.info(
                f"Zalecane użycie mixed precision: {self.hardware_profile.get('use_mixed_precision', 'N/A')}"
            )
            self.logger.info(
                f"Zalecana architektura: {self.hardware_profile.get('additional_recommendations', {}).get('recommended_model', 'N/A')}"
            )
            self.logger.info(
                f"Zalecana precyzja: {self.hardware_profile.get('additional_recommendations', {}).get('recommended_precision', 'N/A')}"
            )
            self.logger.info(
                f"Zalecany poziom augmentacji: {self.hardware_profile.get('additional_recommendations', {}).get('recommended_augmentation', 'N/A')}"
            )
            self.logger.info("=============================")

        except Exception as e:
            msg = "Błąd podczas ładowania konfiguracji"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            QtWidgets.QMessageBox.critical(self, "Błąd", f"{msg}: {str(e)}")
