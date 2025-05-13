import datetime
import json
import logging
import os
from pathlib import Path

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt

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
        self.hardware_profile = hardware_profile
        self._setup_logging()
        self.setWindowTitle("Konfiguracja treningu")
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
        if not self.current_profile:
            QtWidgets.QMessageBox.warning(
                self, "Ostrzeżenie", "Najpierw wybierz profil do zastosowania."
            )
            return

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

            # Parametry Treningu
            if "training" in config:
                training_config = config["training"]
                self.epochs_spin.setValue(training_config.get("epochs", 100))
                self.batch_size_spin.setValue(training_config.get("batch_size", 32))
                self.lr_spin.setValue(training_config.get("learning_rate", 0.001))
                self.optimizer_combo.setCurrentText(
                    training_config.get("optimizer", "Adam")
                )

                # Poprawiona obsługa schedulera
                scheduler_config = training_config.get("scheduler", {})
                if isinstance(scheduler_config, dict):
                    scheduler_type = scheduler_config.get("type", "None")
                    self.scheduler_combo.setCurrentText(scheduler_type)
                else:
                    self.scheduler_combo.setCurrentText(str(scheduler_config))

                self.num_workers_spin.setValue(training_config.get("num_workers", 4))
                self.warmup_epochs_spin.setValue(
                    training_config.get("warmup_epochs", 5)
                )
                self.mixed_precision_check.setChecked(
                    training_config.get("mixed_precision", True)
                )
                self.freeze_base_model.setChecked(
                    training_config.get("freeze_base_model", True)
                )
                self.unfreeze_layers.setText(
                    str(training_config.get("unfreeze_layers", ""))
                )

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

            # Liczba epok
            self.epochs_spin = QtWidgets.QSpinBox()
            self.epochs_spin.setRange(1, 1000)
            self.epochs_spin.setValue(DEFAULT_TRAINING_PARAMS["max_epochs"])
            form.addRow("Liczba epok:", self.epochs_spin)

            # Rozmiar wsadu
            self.batch_size_spin = QtWidgets.QSpinBox()
            self.batch_size_spin.setRange(1, 512)
            batch_size = DEFAULT_TRAINING_PARAMS["batch_size"]
            self.batch_size_spin.setValue(batch_size)
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
            workers = DEFAULT_TRAINING_PARAMS["num_workers"]
            self.num_workers_spin.setValue(workers)
            form.addRow("Liczba wątków:", self.num_workers_spin)

            # Liczba epok rozgrzewki
            self.warmup_epochs_spin = QtWidgets.QSpinBox()
            self.warmup_epochs_spin.setRange(0, 50)
            self.warmup_epochs_spin.setValue(5)
            form.addRow("Epoki rozgrzewki:", self.warmup_epochs_spin)

            # Mixed precision
            self.mixed_precision_check = QtWidgets.QCheckBox("Używaj mixed precision")
            self.mixed_precision_check.setChecked(True)
            form.addRow("", self.mixed_precision_check)

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

            swa_layout.addRow("", self.use_swa_check)
            swa_layout.addRow("Epoka rozpoczęcia:", self.swa_start_epoch_spin)
            swa_group.setLayout(swa_layout)

            layout.addLayout(form)
            layout.addWidget(swa_group)
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

            self.basic_aug_check = QtWidgets.QCheckBox("Używaj podstawowej")
            self.rotation_spin = QtWidgets.QSpinBox()
            self.rotation_spin.setRange(0, 180)
            self.rotation_spin.setValue(30)

            self.brightness_spin = QtWidgets.QDoubleSpinBox()
            self.brightness_spin.setRange(0.0, 1.0)
            self.brightness_spin.setValue(0.2)

            self.shift_spin = QtWidgets.QDoubleSpinBox()
            self.shift_spin.setRange(0.0, 1.0)
            self.shift_spin.setValue(0.1)

            self.zoom_spin = QtWidgets.QDoubleSpinBox()
            self.zoom_spin.setRange(0.0, 1.0)
            self.zoom_spin.setValue(0.1)

            # Dodanie nowych parametrów
            self.horizontal_flip_check = QtWidgets.QCheckBox("Odwrócenie poziome")
            self.vertical_flip_check = QtWidgets.QCheckBox("Odwrócenie pionowe")

            basic_layout.addRow("", self.basic_aug_check)
            basic_layout.addRow("Kąt rotacji:", self.rotation_spin)
            basic_layout.addRow("Jasność:", self.brightness_spin)
            basic_layout.addRow("Przesunięcie:", self.shift_spin)
            basic_layout.addRow("Przybliżenie:", self.zoom_spin)
            basic_layout.addRow("", self.horizontal_flip_check)
            basic_layout.addRow("", self.vertical_flip_check)
            basic_group.setLayout(basic_layout)

            # Zaawansowana augmentacja
            advanced_group = QtWidgets.QGroupBox("Techniki mieszania")
            advanced_layout = QtWidgets.QFormLayout()

            self.mixup_check = QtWidgets.QCheckBox("Używaj Mixup")
            self.mixup_alpha_spin = QtWidgets.QDoubleSpinBox()
            self.mixup_alpha_spin.setRange(0.0, 1.0)
            self.mixup_alpha_spin.setValue(0.2)

            self.cutmix_check = QtWidgets.QCheckBox("Używaj CutMix")
            self.cutmix_alpha_spin = QtWidgets.QDoubleSpinBox()
            self.cutmix_alpha_spin.setRange(0.0, 1.0)
            self.cutmix_alpha_spin.setValue(1.0)

            # Dodanie AutoAugment i RandAugment
            self.autoaugment_check = QtWidgets.QCheckBox("Używaj AutoAugment")
            self.randaugment_check = QtWidgets.QCheckBox("Używaj RandAugment")
            self.randaugment_n_spin = QtWidgets.QSpinBox()
            self.randaugment_n_spin.setRange(1, 10)
            self.randaugment_n_spin.setValue(2)
            self.randaugment_m_spin = QtWidgets.QSpinBox()
            self.randaugment_m_spin.setRange(1, 30)
            self.randaugment_m_spin.setValue(9)

            advanced_layout.addRow("", self.mixup_check)
            advanced_layout.addRow("Alpha:", self.mixup_alpha_spin)
            advanced_layout.addRow("", self.cutmix_check)
            advanced_layout.addRow("Alpha:", self.cutmix_alpha_spin)
            advanced_layout.addRow("", self.autoaugment_check)
            advanced_layout.addRow("", self.randaugment_check)
            advanced_layout.addRow("N:", self.randaugment_n_spin)
            advanced_layout.addRow("M:", self.randaugment_m_spin)
            advanced_group.setLayout(advanced_layout)

            layout.addWidget(basic_group)
            layout.addWidget(advanced_group)
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
            self.min_delta_spin.setDecimals(4)

            self.monitor_combo = QtWidgets.QComboBox()
            metrics = [
                "val_loss",
                "val_accuracy",
                "val_f1",
                "val_precision",
                "val_recall",
            ]
            self.monitor_combo.addItems(metrics)

            early_stop_layout.addRow("Epoki bez poprawy:", self.patience_spin)
            early_stop_layout.addRow("Minimalna poprawa:", self.min_delta_spin)
            early_stop_layout.addRow("Metryka:", self.monitor_combo)
            early_stop_group.setLayout(early_stop_layout)

            # Checkpointowanie
            checkpoint_group = QtWidgets.QGroupBox("Checkpointowanie")
            checkpoint_layout = QtWidgets.QFormLayout()

            self.best_only_check = QtWidgets.QCheckBox("Tylko najlepszy model")
            self.save_freq_spin = QtWidgets.QSpinBox()
            self.save_freq_spin.setRange(1, 50)
            self.save_freq_spin.setValue(1)

            self.checkpoint_metric_combo = QtWidgets.QComboBox()
            self.checkpoint_metric_combo.addItems(metrics)

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

            layout.addWidget(metrics_group)
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

            transfer_layout.addRow("", self.freeze_base_model)
            transfer_layout.addRow("Warstwy do odmrożenia:", self.unfreeze_layers)
            transfer_layout.addRow("Strategia:", self.unfreeze_strategy)
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

    def _create_optimization_tab(self):
        """Tworzenie zakładki Optymalizacja treningu."""
        try:
            self.logger.debug("Tworzenie zakładki optymalizacji treningu")
            tab = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(tab)

            # Informacja o profilu sprzętowym
            if self.hardware_profile:
                hardware_info = QtWidgets.QLabel(
                    f"Używany profil sprzętowy: {self.hardware_profile.get('device_name', 'Nieznany')}"
                )
            else:
                hardware_info = QtWidgets.QLabel(
                    "Brak załadowanego profilu sprzętowego"
                )
            hardware_info.setStyleSheet("font-weight: bold; color: #333;")
            layout.addWidget(hardware_info)

            # Separator
            line = QtWidgets.QFrame()
            line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
            line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
            layout.addWidget(line)

            # Tworzenie grup parametrów
            form_layout = QtWidgets.QFormLayout()

            # 1. Rozmiar batch'a
            batch_size_layout = self._create_parameter_row(
                name="Rozmiar batch'a",
                param_key="recommended_batch_size",
                default_value=32,
                widget_type="spinbox",
                min_val=1,
                max_val=512,
                step=1,
            )
            form_layout.addRow("Rozmiar batch'a:", batch_size_layout)

            # 2. Liczba workerów
            num_workers_layout = self._create_parameter_row(
                name="Liczba workerów",
                param_key="recommended_workers",
                default_value=4,
                widget_type="spinbox",
                min_val=0,
                max_val=32,
                step=1,
            )
            form_layout.addRow("Liczba workerów:", num_workers_layout)

            # 3. Mixed precision
            mixed_precision_layout = self._create_parameter_row(
                name="Mixed precision",
                param_key="use_mixed_precision",
                default_value=True,
                widget_type="checkbox",
            )
            form_layout.addRow("Mixed precision:", mixed_precision_layout)

            # 4. Prefetch factor
            prefetch_layout = self._create_parameter_row(
                name="Prefetch factor",
                param_key="prefetch_factor",
                default_value=2,
                widget_type="spinbox",
                min_val=1,
                max_val=10,
                step=1,
            )
            form_layout.addRow("Prefetch factor:", prefetch_layout)

            # 5. Pin memory
            pin_memory_layout = self._create_parameter_row(
                name="Pin memory",
                param_key="pin_memory",
                default_value=True,
                widget_type="checkbox",
            )
            form_layout.addRow("Pin memory:", pin_memory_layout)

            # 6. Persistent workers
            persistent_workers_layout = self._create_parameter_row(
                name="Persistent workers",
                param_key="persistent_workers",
                default_value=False,
                widget_type="checkbox",
            )
            form_layout.addRow("Persistent workers:", persistent_workers_layout)

            # 7. CUDA streaming
            cuda_stream_layout = self._create_parameter_row(
                name="CUDA streaming",
                param_key="cuda_streaming",
                default_value=True,
                widget_type="checkbox",
            )
            form_layout.addRow("CUDA streaming:", cuda_stream_layout)

            # 8. Benchmark CUDNN
            benchmark_cudnn_layout = self._create_parameter_row(
                name="Benchmark CUDNN",
                param_key="benchmark_cudnn",
                default_value=True,
                widget_type="checkbox",
            )
            form_layout.addRow("Benchmark CUDNN:", benchmark_cudnn_layout)

            # 9. Garbage collector
            gc_layout = self._create_parameter_row(
                name="Wyłącz garbage collector",
                param_key="disable_gc",
                default_value=False,
                widget_type="checkbox",
            )
            form_layout.addRow("Wyłącz garbage collector:", gc_layout)

            # 10. Gradient accumulation steps
            grad_accum_layout = self._create_parameter_row(
                name="Gradient accumulation steps",
                param_key="gradient_accumulation_steps",
                default_value=1,
                widget_type="spinbox",
                min_val=1,
                max_val=32,
                step=1,
            )
            form_layout.addRow("Gradient accumulation steps:", grad_accum_layout)

            # 11. Channels last memory format
            channels_last_layout = self._create_parameter_row(
                name="Channels last memory format",
                param_key="channels_last",
                default_value=False,
                widget_type="checkbox",
            )
            form_layout.addRow("Channels last:", channels_last_layout)

            # Dodanie całego layoutu do zakładki
            layout.addLayout(form_layout)

            # Dodanie przycisku do załadowania wszystkich optymalnych ustawień
            load_all_btn = QtWidgets.QPushButton(
                "Zastosuj wszystkie optymalne ustawienia z profilu sprzętowego"
            )
            load_all_btn.clicked.connect(self._apply_all_hardware_optimizations)
            layout.addWidget(load_all_btn)

            # Dodanie rozciągliwego elementu na końcu (spacer)
            layout.addStretch(1)

            tab.setLayout(layout)
            return tab

        except Exception as e:
            msg = "Błąd podczas tworzenia zakładki optymalizacji"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
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
        """
        Tworzy wiersz parametru z opcją wyboru źródła wartości.

        Args:
            name: Nazwa parametru
            param_key: Klucz parametru w profilu sprzętowym
            default_value: Wartość domyślna
            widget_type: Typ widgetu ('spinbox', 'checkbox', etc.)
            min_val: Minimalna wartość (dla spinbox)
            max_val: Maksymalna wartość (dla spinbox)
            step: Wartość kroku (dla spinbox)

        Returns:
            QLayout: Layout z kontrolkami parametru
        """
        layout = QtWidgets.QHBoxLayout()

        # Źródło wartości
        source_group = QtWidgets.QButtonGroup()

        # Przycisk opcji dla wartości z UI/profilu
        profile_radio = QtWidgets.QRadioButton("Z profilu")
        profile_radio.setChecked(True)
        source_group.addButton(profile_radio, 1)

        # Przycisk opcji dla wartości z profilu sprzętowego
        hardware_radio = QtWidgets.QRadioButton("Z profilu sprzętowego")
        source_group.addButton(hardware_radio, 2)

        # Wartość z profilu
        profile_value = default_value

        # Wartość z profilu sprzętowego
        hw_value = None
        if self.hardware_profile and param_key in self.hardware_profile:
            hw_value = self.hardware_profile[param_key]
            hardware_radio.setEnabled(True)
        else:
            hardware_radio.setEnabled(False)
            hardware_radio.setText("Z profilu sprzętowego (niedostępne)")

        # Widget edycji wartości
        if widget_type == "spinbox":
            value_widget = QtWidgets.QSpinBox()
            if min_val is not None:
                value_widget.setMinimum(min_val)
            if max_val is not None:
                value_widget.setMaximum(max_val)
            if step is not None:
                value_widget.setSingleStep(step)
            value_widget.setValue(profile_value)
        elif widget_type == "checkbox":
            value_widget = QtWidgets.QCheckBox()
            value_widget.setChecked(profile_value)
        else:
            value_widget = QtWidgets.QLineEdit(str(profile_value))

        # Etykieta z wartością z profilu sprzętowego
        hw_value_label = QtWidgets.QLabel("Niedostępne")
        if hw_value is not None:
            hw_value_label.setText(str(hw_value))

        # Dodanie widgetów do layoutu
        layout.addWidget(profile_radio)
        layout.addWidget(value_widget)
        layout.addWidget(hardware_radio)
        layout.addWidget(hw_value_label)

        # Zapamiętanie referencji do widgetów
        row_widgets = {
            "param_key": param_key,
            "profile_radio": profile_radio,
            "hardware_radio": hardware_radio,
            "value_widget": value_widget,
            "hw_value_label": hw_value_label,
            "hw_value": hw_value,
        }

        # Dodanie do listy parametrów
        if not hasattr(self, "optimization_params"):
            self.optimization_params = []
        self.optimization_params.append(row_widgets)

        # Obsługa zmiany źródła wartości
        def on_source_changed():
            if profile_radio.isChecked():  # Profil
                value_widget.setEnabled(True)
            else:  # Profil sprzętowy
                value_widget.setEnabled(False)
                if hw_value is not None:
                    if widget_type == "spinbox":
                        value_widget.setValue(hw_value)
                    elif widget_type == "checkbox":
                        value_widget.setChecked(hw_value)
                    else:
                        value_widget.setText(str(hw_value))

        # Podłącz do sygnałów toggled dla obu przycisków
        profile_radio.toggled.connect(on_source_changed)
        hardware_radio.toggled.connect(on_source_changed)

        return layout

    def _apply_all_hardware_optimizations(self):
        """Zastosowuje wszystkie optymalne ustawienia z profilu sprzętowego."""
        if not hasattr(self, "optimization_params") or not self.hardware_profile:
            QtWidgets.QMessageBox.warning(
                self,
                "Ostrzeżenie",
                "Brak dostępnego profilu sprzętowego lub parametrów do zastosowania.",
            )
            return

        count = 0
        for param in self.optimization_params:
            param_key = param["param_key"]
            if param_key in self.hardware_profile:
                param["hardware_radio"].setChecked(True)
                hw_value = self.hardware_profile[param_key]
                value_widget = param["value_widget"]
                value_widget.setEnabled(False)

                if isinstance(value_widget, QtWidgets.QSpinBox) or isinstance(
                    value_widget, QtWidgets.QDoubleSpinBox
                ):
                    value_widget.setValue(hw_value)
                elif isinstance(value_widget, QtWidgets.QCheckBox):
                    value_widget.setChecked(hw_value)
                else:
                    value_widget.setText(str(hw_value))

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

    def _on_accept(self):
        """Obsługa zatwierdzenia konfiguracji."""
        try:
            # Generowanie nazwy zadania automatycznie
            variant = self.variant_combo.currentText()
            num_classes = self.num_classes_spin.value()
            now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
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

            # Pobranie konfiguracji optymalizacyjnej
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

            self.task_config = {
                "name": task_name,
                "type": "training",
                "status": "Nowy",
                "priority": 0,
                "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "config": {
                    "train_dir": train_dir,
                    "data_dir": train_dir,
                    "val_dir": val_dir,
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
                        "randaugment": {
                            "use": self.randaugment_check.isChecked(),
                            "n": self.randaugment_n_spin.value(),
                            "m": self.randaugment_m_spin.value(),
                        },
                    },
                    "optimization": optimization_config,  # Dodajemy sekcję optymalizacji
                },
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

    def get_task_config(self):
        """Zwraca konfigurację zadania lub None, jeśli nie dodano zadania."""
        return getattr(self, "task_config", None)

    def closeEvent(self, event):
        """Obsługa zamknięcia okna."""
        self.logger.info("Zamykanie okna dialogowego")
        self.accept()
        event.accept()
