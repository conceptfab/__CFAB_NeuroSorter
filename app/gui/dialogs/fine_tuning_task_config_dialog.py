import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

from PyQt6 import QtWidgets

from app.utils.file_utils import (
    validate_training_directory,
)  # validate_validation_directory, # Usunięty nieużywany import


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
        self.hardware_profile = hardware_profile
        self._setup_logging()
        self.setWindowTitle("Konfiguracja doszkalania")
        self.setMinimumWidth(1000)
        self.profiles_dir = Path("data/profiles")
        self.profiles_dir.mkdir(exist_ok=True)
        self.current_profile = None

        self._define_dependencies()  # Definicja słownika zależności

        # Inicjalizacja wszystkich kontrolek
        self._init_controls()

        # Inicjalizacja interfejsu
        self._init_ui()

    def _define_dependencies(self):
        """Definiuje słownik zależności między kontrolkami."""
        self.dependencies = {
            "arch_combo": {"update_method": self._update_variant_combo},
            "optimizer_combo": {
                "controls_to_update": {
                    "weight_decay_spin": lambda opt_text: opt_text != "Adam",
                    "momentum_spin": lambda opt_text: opt_text == "SGD",
                }
            },
            "scheduler_combo": {
                "controls_to_update": {
                    "warmup_epochs_spin": lambda sch_text: sch_text != "None",
                    "warmup_lr_init_spin": lambda sch_text: sch_text != "None",
                }
            },
            "unfreeze_strategy_combo": {
                "controls_to_update": {
                    "unfreeze_layers_spin": lambda strategy_text: strategy_text
                    != self.UNFREEZE_ALL,
                    "unfreeze_after_epochs_spin": lambda strategy_text: strategy_text
                    == self.UNFREEZE_AFTER_EPOCHS,
                    "frozen_lr_spin": lambda strategy_text: strategy_text
                    != self.UNFREEZE_ALL,
                    "unfrozen_lr_spin": lambda strategy_text: strategy_text
                    != self.UNFREEZE_ALL,
                }
            },
            "basic_aug_check": {
                "controls_to_update": {
                    "rotation_spin": lambda checked: checked,
                    "brightness_spin": lambda checked: checked,
                    "shift_spin": lambda checked: checked,
                    "zoom_spin": lambda checked: checked,
                    "horizontal_flip_check": lambda checked: checked,
                    "vertical_flip_check": lambda checked: checked,
                }
            },
            "mixup_check": {
                "controls_to_update": {"mixup_alpha_spin": lambda checked: checked}
            },
            "cutmix_check": {
                "controls_to_update": {"cutmix_alpha_spin": lambda checked: checked}
            },
            "randaugment_check": {
                "controls_to_update": {
                    "randaugment_n_spin": lambda checked: checked,
                    "randaugment_m_spin": lambda checked: checked,
                }
            },
            "prevent_forgetting_check": {  # Główny przełącznik dla zapobiegania zapominaniu
                "controls_to_update": {
                    "preserve_classes_check": lambda checked: checked,
                    "rehearsal_check": lambda checked: checked,
                    "knowledge_distillation_check": lambda checked: checked,
                    "ewc_check": lambda checked: checked,
                    "layer_freezing_combo": lambda checked: checked,
                    "freeze_ratio_spin": lambda checked: checked,
                }
            },
            "rehearsal_check": {  # Zależne od prevent_forgetting_check
                "controls_to_update": {
                    "samples_per_class_spin": lambda rehearsal_checked: rehearsal_checked,
                    "synthetic_samples_check": lambda rehearsal_checked: rehearsal_checked,
                },
                "parent_condition_control": "prevent_forgetting_check",
            },
            "knowledge_distillation_check": {  # Zależne od prevent_forgetting_check
                "controls_to_update": {
                    "kd_temperature_spin": lambda kd_checked: kd_checked,
                    "kd_alpha_spin": lambda kd_checked: kd_checked,
                },
                "parent_condition_control": "prevent_forgetting_check",
            },
            "ewc_check": {  # Zależne od prevent_forgetting_check
                "controls_to_update": {
                    "ewc_lambda_spin": lambda ewc_checked: ewc_checked,
                    "adaptive_ewc_lambda_check": lambda ewc_checked: ewc_checked,
                    "fisher_sample_size_spin": lambda ewc_checked: ewc_checked,
                },
                "parent_condition_control": "prevent_forgetting_check",
            },
            "use_tensorboard_check": {
                "controls_to_update": {"log_freq_combo": lambda checked: checked}
            },
            "use_pred_samples_check": {
                "controls_to_update": {"num_samples_spin": lambda checked: checked}
            },
            "best_only_check": {
                "controls_to_update": {"save_freq_spin": lambda checked: not checked}
            },
        }

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

        # Połączenia sygnałów dla aktualizacji UI
        self.arch_combo.currentTextChanged.connect(self._update_ui_state)
        self.optimizer_combo.currentTextChanged.connect(self._update_ui_state)
        self.scheduler_combo.currentTextChanged.connect(self._update_ui_state)
        self.unfreeze_strategy_combo.currentTextChanged.connect(self._update_ui_state)

        # Sygnały dla augmentacji
        self.basic_aug_check.stateChanged.connect(self._update_ui_state)
        self.mixup_check.stateChanged.connect(self._update_ui_state)
        self.cutmix_check.stateChanged.connect(self._update_ui_state)
        self.autoaugment_check.stateChanged.connect(
            self._update_ui_state
        )  # autoaugment nie ma kontrolek zależnych w self.dependencies, ale może wpływać na inne aspekty
        self.randaugment_check.stateChanged.connect(self._update_ui_state)

        # Sygnały dla monitorowania
        self.patience_spin.valueChanged.connect(
            self._update_ui_state
        )  # Może wpływać na logikę, ale nie ma kontrolek zależnych w self.dependencies
        self.monitor_combo.currentTextChanged.connect(
            self._update_ui_state
        )  # Jak wyżej
        self.checkpoint_metric_combo.currentTextChanged.connect(
            self._update_ui_state
        )  # Jak wyżej
        self.use_tensorboard_check.stateChanged.connect(self._update_ui_state)
        self.use_pred_samples_check.stateChanged.connect(self._update_ui_state)
        self.best_only_check.stateChanged.connect(self._update_ui_state)

        # Sygnały dla zapobiegania katastrofalnemu zapominaniu
        self.prevent_forgetting_check.stateChanged.connect(self._update_ui_state)
        self.rehearsal_check.stateChanged.connect(self._update_ui_state)
        self.knowledge_distillation_check.stateChanged.connect(self._update_ui_state)
        self.ewc_check.stateChanged.connect(self._update_ui_state)

        self._update_ui_state()  # Ustawienie początkowego stanu kontrolek

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
                    "Wybrany profil nie zawiera sekcji 'config' " "lub jest ona pusta."
                )
                QtWidgets.QMessageBox.warning(
                    self,
                    "Ostrzeżenie",
                    "Profil nie zawiera danych konfiguracyjnych.",
                )
                return

            # Dodajemy nazwę zadania do config_to_load, jeśli _load_config
            # tego oczekuje (Oryginalnie _load_config może próbować ustawić
            # self.name_edit z config['name']) Tutaj zakładamy, że nazwa
            # zadania jest zarządzana osobno lub nie jest częścią 'config'
            # profilu. Jeśli nazwa zadania z profilu powinna być zastosowana,
            # można ją dodać do config_to_load:
            # if "info" in self.current_profile:
            # # Lub inna odpowiednia nazwa z profilu
            #    config_to_load["name"] = self.current_profile["info"]
            # # Przykładowo

            self._load_config(config_to_load)  # Wywołanie _load_config

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
                profile_data = {
                    "type": "fine_tuning",
                    "info": f"Profil dla {self.arch_combo.currentText()} {self.variant_combo.currentText()}",
                    "description": "Profil utworzony przez użytkownika",
                    "data_required": "Standardowe dane do doszkalania",
                    "hardware_required": "Standardowy sprzęt",
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
                        "regularization": {
                            "weight_decay": float(self.weight_decay_spin.value()),
                            "label_smoothing": float(self.label_smoothing_spin.value()),
                            "dropout_rate": float(self.dropout_spin.value()),
                            "drop_connect_rate": float(self.drop_connect_spin.value()),
                            "momentum": float(self.momentum_spin.value()),
                            "epsilon": float(self.epsilon_spin.value()),
                            "swa": {
                                "use": self.use_swa_check.isChecked(),
                                "start_epoch": int(self.swa_start_epoch_spin.value()),
                            },
                            "stochastic_depth": {
                                "use": self.use_stoch_depth_check.isChecked(),
                                "drop_rate": float(self.stoch_depth_drop_rate.value()),
                                "survival_probability": float(
                                    self.stoch_depth_survival_prob.value()
                                ),
                            },
                            "random_erase": {
                                "use": self.use_random_erase_check.isChecked(),
                                "probability": float(self.random_erase_prob.value()),
                                "mode": self.random_erase_mode_combo.currentText(),
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
                            "advanced": {
                                "contrast": self.contrast_spin.value(),
                                "saturation": self.saturation_spin.value(),
                                "hue": self.hue_spin.value(),
                                "shear": self.shear_spin.value(),
                                "channel_shift": self.channel_shift_spin.value(),
                            },
                        },
                        "preprocessing": {
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
                                "save_to_csv": self.use_csv_check.isChecked(),
                                "logging_freq": self.log_freq_combo.currentText(),
                            },
                            "visualization": {
                                "use_gradcam": self.use_gradcam_check.isChecked(),
                                "use_feature_maps": self.use_feature_maps_check.isChecked(),
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
            # Blokujemy sygnały podczas wczytywania konfiguracji, aby uniknąć
            # wyzwalania zbędnych aktualizacji UI
            self.blockSignals(
                True
            )  # Zmienione z self.arch_combo.blockSignals itp. na globalne

            # Model
            model_config = config.get("model", {})

            # Logowanie wartości przed ustawieniem
            self.logger.info(
                f"Ładowanie konfiguracji - Nazwa zadania: {self.name_edit.text()}"
            )
            self.logger.info(
                f"Ładowanie konfiguracji - Liczba klas: {self.num_classes_spin.value()}"
            )

            # Ustawienie wartości z konfiguracji
            if "name" in config:
                self.name_edit.setText(config["name"])

            # 1. Aktualizacja parametrów modelu
            if "architecture" in model_config:
                # Najpierw ustawiamy architekturę
                architecture = model_config["architecture"]
                idx = self.arch_combo.findText(architecture)
                if idx >= 0:
                    self.arch_combo.blockSignals(True)
                    self.arch_combo.setCurrentIndex(idx)
                    self.arch_combo.blockSignals(False)
                    # To wywołanie zaktualizuje listę wariantów
                    self._update_variant_combo(architecture)
                else:
                    self.logger.warning(
                        f"Architektura {architecture} nie jest dostępna"
                    )

            # Teraz ustawiamy wariant, po aktualizacji listy wariantów
            if "variant" in model_config:
                variant = model_config["variant"]
                # Jeśli wariant jest dostępny w aktualnej liście, ustaw go
                idx = self.variant_combo.findText(variant)
                if idx >= 0:
                    self.variant_combo.blockSignals(True)
                    self.variant_combo.setCurrentIndex(idx)
                    self.variant_combo.blockSignals(False)
                    self.logger.info(f"Ustawiono wariant: {variant}")
                else:
                    self.logger.warning(
                        f"Wariant {variant} nie jest dostępny dla architektury {model_config.get('architecture')}"
                    )

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

            # Logowanie wartości po ustawieniu
            self.logger.info(
                f"Zaktualizowano wartości - Nazwa zadania: {self.name_edit.text()}"
            )
            self.logger.info(
                f"Zaktualizowano wartości - Liczba klas: {self.num_classes_spin.value()}"
            )

            # 2. Aktualizacja parametrów treningu
            training_config = config.get("training", {})

            if "batch_size" in training_config:
                self.batch_size_spin.setValue(training_config["batch_size"])

            if "learning_rate" in training_config:
                self.lr_spin.setValue(training_config["learning_rate"])

            if "optimizer" in training_config:
                self.optimizer_combo.setCurrentText(training_config["optimizer"])

            if "scheduler" in training_config:
                self.scheduler_combo.setCurrentText(training_config["scheduler"])

            if "warmup_epochs" in training_config:
                self.warmup_epochs_spin.setValue(training_config["warmup_epochs"])

            if "mixed_precision" in training_config:
                self.mixed_precision_check.setChecked(
                    training_config["mixed_precision"]
                )

            if "unfreeze_strategy" in training_config:
                self.unfreeze_strategy_combo.setCurrentText(
                    training_config["unfreeze_strategy"]
                )

            if "unfreeze_layers" in training_config:
                self.unfreeze_layers_spin.setValue(training_config["unfreeze_layers"])

            if "warmup_lr_init" in training_config:
                self.warmup_lr_init_spin.setValue(training_config["warmup_lr_init"])

            if "gradient_accumulation_steps" in training_config:
                self.grad_accum_steps_spin.setValue(
                    training_config["gradient_accumulation_steps"]
                )

            if "validation_split" in training_config:
                self.validation_split_spin.setValue(training_config["validation_split"])

            if "evaluation_freq" in training_config:
                self.eval_freq_spin.setValue(training_config["evaluation_freq"])

            if "use_ema" in training_config:
                self.use_ema_check.setChecked(training_config["use_ema"])

            if "ema_decay" in training_config:
                self.ema_decay_spin.setValue(training_config["ema_decay"])

            # 3. Aktualizacja parametrów regularyzacji
            regularization_config = config.get("regularization", {})

            if "weight_decay" in regularization_config:
                self.weight_decay_spin.setValue(regularization_config["weight_decay"])

            if "drop_connect_rate" in regularization_config:
                self.drop_connect_spin.setValue(
                    regularization_config["drop_connect_rate"]
                )

            if "dropout_rate" in regularization_config:
                self.dropout_spin.setValue(regularization_config["dropout_rate"])

            if "label_smoothing" in regularization_config:
                self.label_smoothing_spin.setValue(
                    regularization_config["label_smoothing"]
                )

            # SWA
            swa_config = regularization_config.get("swa", {})
            if "use" in swa_config:
                self.use_swa_check.setChecked(swa_config["use"])

            if "start_epoch" in swa_config:
                self.swa_start_epoch_spin.setValue(swa_config["start_epoch"])

            # 4. Aktualizacja parametrów augmentacji
            augmentation_config = config.get("augmentation", {})

            # Basic augmentation
            basic_config = augmentation_config.get("basic", {})
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
                self.horizontal_flip_check.setChecked(basic_config["horizontal_flip"])

            if "vertical_flip" in basic_config:
                self.vertical_flip_check.setChecked(basic_config["vertical_flip"])

            # Mixup
            mixup_config = augmentation_config.get("mixup", {})
            if "use" in mixup_config:
                self.mixup_check.setChecked(mixup_config["use"])

            if "alpha" in mixup_config:
                self.mixup_alpha_spin.setValue(mixup_config["alpha"])

            # CutMix
            cutmix_config = augmentation_config.get("cutmix", {})
            if "use" in cutmix_config:
                self.cutmix_check.setChecked(cutmix_config["use"])

            if "alpha" in cutmix_config:
                self.cutmix_alpha_spin.setValue(cutmix_config["alpha"])

            # AutoAugment
            autoaugment_config = augmentation_config.get("autoaugment", {})
            if "use" in autoaugment_config:
                self.autoaugment_check.setChecked(autoaugment_config["use"])

            # RandAugment
            randaugment_config = augmentation_config.get("randaugment", {})
            if "use" in randaugment_config:
                self.randaugment_check.setChecked(randaugment_config["use"])
            if "n" in randaugment_config:
                self.randaugment_n_spin.setValue(randaugment_config["n"])
            if "m" in randaugment_config:
                self.randaugment_m_spin.setValue(randaugment_config["m"])

            # Advanced augmentation
            advanced_config = augmentation_config.get("advanced", {})
            if "contrast" in advanced_config:
                self.contrast_spin.setValue(advanced_config["contrast"])
            if "saturation" in advanced_config:
                self.saturation_spin.setValue(advanced_config["saturation"])
            if "hue" in advanced_config:
                self.hue_spin.setValue(advanced_config["hue"])
            if "shear" in advanced_config:
                self.shear_spin.setValue(advanced_config["shear"])
            if "channel_shift" in advanced_config:
                self.channel_shift_spin.setValue(advanced_config["channel_shift"])

            # 5. Aktualizacja parametrów preprocessingu
            preprocessing_config = config.get("preprocessing", {})

            # Normalization
            normalization_config = preprocessing_config.get("normalization", {})
            if (
                "mean" in normalization_config
                and len(normalization_config["mean"]) == 3
            ):
                self.norm_mean_r.setValue(normalization_config["mean"][0])
                self.norm_mean_g.setValue(normalization_config["mean"][1])
                self.norm_mean_b.setValue(normalization_config["mean"][2])

            if "std" in normalization_config and len(normalization_config["std"]) == 3:
                self.norm_std_r.setValue(normalization_config["std"][0])
                self.norm_std_g.setValue(normalization_config["std"][1])
                self.norm_std_b.setValue(normalization_config["std"][2])

            # 6. Aktualizacja parametrów monitorowania
            monitoring_config = config.get("monitoring", {})

            # Metrics
            metrics_config = monitoring_config.get("metrics", {})
            if "accuracy" in metrics_config:
                self.accuracy_check.setChecked(metrics_config["accuracy"])
            if "precision" in metrics_config:
                self.precision_check.setChecked(metrics_config["precision"])
            if "recall" in metrics_config:
                self.recall_check.setChecked(metrics_config["recall"])
            if "f1" in metrics_config:
                self.f1_check.setChecked(metrics_config["f1"])
            if "top_k_accuracy" in metrics_config:
                self.topk_check.setChecked(metrics_config["top_k_accuracy"])
            if "confusion_matrix" in metrics_config:
                self.confusion_matrix_check.setChecked(
                    metrics_config["confusion_matrix"]
                )
            if "auc" in metrics_config:
                self.auc_check.setChecked(metrics_config["auc"])

            # Logging
            logging_config = monitoring_config.get("logging", {})
            if "use_tensorboard" in logging_config:
                self.use_tensorboard_check.setChecked(logging_config["use_tensorboard"])
            if "use_wandb" in logging_config:
                self.use_wandb_check.setChecked(logging_config["use_wandb"])
            if "save_to_csv" in logging_config:
                self.use_csv_check.setChecked(logging_config["save_to_csv"])
            if "logging_freq" in logging_config:
                self.log_freq_combo.setCurrentText(logging_config["logging_freq"])

            # Visualization
            visualization_config = monitoring_config.get("visualization", {})
            if "use_gradcam" in visualization_config:
                self.use_gradcam_check.setChecked(visualization_config["use_gradcam"])
            if "use_feature_maps" in visualization_config:
                self.use_feature_maps_check.setChecked(
                    visualization_config["use_feature_maps"]
                )

            # Early stopping
            early_stopping_config = monitoring_config.get("early_stopping", {})
            if "patience" in early_stopping_config:
                self.patience_spin.setValue(early_stopping_config["patience"])
            if "min_delta" in early_stopping_config:
                self.min_delta_spin.setValue(early_stopping_config["min_delta"])
            if "monitor" in early_stopping_config:
                self.monitor_combo.setCurrentText(early_stopping_config["monitor"])

            # Checkpointing
            checkpointing_config = monitoring_config.get("checkpointing", {})
            if "best_only" in checkpointing_config:
                self.best_only_check.setChecked(checkpointing_config["best_only"])
            if "save_frequency" in checkpointing_config:
                self.save_freq_spin.setValue(checkpointing_config["save_frequency"])
            if "metric" in checkpointing_config:
                self.checkpoint_metric_combo.setCurrentText(
                    checkpointing_config["metric"]
                )

            # Aktualizacja zależnych kontrolek
            self._update_dependent_controls()

            # Wczytaj ustawienia zapobiegania katastrofalnemu zapominaniu
            forgetting_config = config.get("advanced", {}).get(
                "catastrophic_forgetting_prevention", {}
            )

            if forgetting_config:
                # Główna opcja włączająca
                if "enable" in forgetting_config:
                    self.prevent_forgetting_check.setChecked(
                        forgetting_config["enable"]
                    )

                # Zachowanie oryginalnych klas
                if "preserve_original_classes" in forgetting_config:
                    self.preserve_classes_check.setChecked(
                        forgetting_config["preserve_original_classes"]
                    )

                # Rehearsal
                rehearsal_config = forgetting_config.get("rehearsal", {})
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
                kd_config = forgetting_config.get("knowledge_distillation", {})
                if "use" in kd_config:
                    self.knowledge_distillation_check.setChecked(kd_config["use"])

                if "temperature" in kd_config:
                    self.kd_temperature_spin.setValue(kd_config["temperature"])

                if "alpha" in kd_config:
                    self.kd_alpha_spin.setValue(kd_config["alpha"])

                # EWC Regularization
                ewc_config = forgetting_config.get("ewc_regularization", {})
                if "use" in ewc_config:
                    self.ewc_check.setChecked(ewc_config["use"])

                if "lambda" in ewc_config:
                    self.ewc_lambda_spin.setValue(ewc_config["lambda"])

                if "fisher_sample_size" in ewc_config:
                    self.fisher_sample_size_spin.setValue(
                        ewc_config["fisher_sample_size"]
                    )

                # Layer Freezing
                layer_freezing_config = forgetting_config.get("layer_freezing", {})
                if "strategy" in layer_freezing_config:
                    index = self.layer_freezing_combo.findText(
                        layer_freezing_config["strategy"]
                    )
                    if index >= 0:
                        self.layer_freezing_combo.setCurrentIndex(index)

                if "freeze_ratio" in layer_freezing_config:
                    self.freeze_ratio_spin.setValue(
                        layer_freezing_config["freeze_ratio"]
                    )

            # Aktualizacja zależnych kontrolek po wczytaniu wszystkich wartości
            # Usunięto _update_forgetting_controls(), bo logika jest w _update_ui_state()

            # Na koniec metody odblokujemy sygnały i ręcznie wywołamy aktualizację UI
            self.blockSignals(False)  # Globalne odblokowanie
            self._update_ui_state()
            self.logger.info("Konfiguracja modelu została pomyślnie załadowana")

        except Exception as e:
            self.blockSignals(
                False
            )  # Upewnij się, że sygnały zostaną odblokowane nawet w przypadku błędu
            msg = "Błąd podczas ładowania konfiguracji"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            QtWidgets.QMessageBox.critical(self, "Błąd", f"{msg}: {str(e)}")

    def _update_ui_state(self):
        """Aktualizuje stan UI na podstawie zdefiniowanych zależności."""
        for main_control_attr, dep_info in self.dependencies.items():
            if not hasattr(self, main_control_attr):
                self.logger.debug(
                    f"_update_ui_state: Main control {main_control_attr} not found."
                )
                continue

            main_control_widget = getattr(self, main_control_attr)
            current_value = None

            if isinstance(main_control_widget, QtWidgets.QComboBox):
                current_value = main_control_widget.currentText()
            elif isinstance(main_control_widget, QtWidgets.QCheckBox):
                current_value = main_control_widget.isChecked()
            elif isinstance(
                main_control_widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)
            ):
                current_value = (
                    main_control_widget.value()
                )  # Dla spinboxów, jeśli będą primary controllers

            if "update_method" in dep_info:
                # Specjalne metody aktualizacji, np. _update_variant_combo
                if (
                    current_value is not None
                ):  # Upewnijmy się, że mamy wartość dla metody
                    dep_info["update_method"](current_value)
                else:  # Jeśli current_value to None, a metoda oczekuje wartości (np. text z QComboBox)
                    if isinstance(
                        main_control_widget, QtWidgets.QComboBox
                    ):  # Domyślnie dla ComboBox
                        dep_info["update_method"](main_control_widget.currentText())

            if "controls_to_update" in dep_info:
                parent_active = True  # Domyślnie traktujemy jako aktywne
                parent_control_attr = dep_info.get("parent_condition_control")

                if parent_control_attr:
                    if hasattr(self, parent_control_attr):
                        parent_widget = getattr(self, parent_control_attr)
                        if isinstance(parent_widget, QtWidgets.QCheckBox):
                            parent_active = (
                                parent_widget.isChecked() and parent_widget.isEnabled()
                            )
                        else:  # Domyślnie, jeśli rodzic nie jest checkboxem, zakładamy, że nie blokuje
                            parent_active = parent_widget.isEnabled()
                    else:  # Rodzic nie istnieje, więc dzieci nie powinny być aktywowane
                        parent_active = False
                        self.logger.warning(
                            f"Parent control {parent_control_attr} not found for {main_control_attr}"
                        )

                for control_to_update_attr, condition_fn in dep_info[
                    "controls_to_update"
                ].items():
                    if hasattr(self, control_to_update_attr):
                        control_to_update_widget = getattr(self, control_to_update_attr)
                        # Kontrolka zależna jest włączana jeśli jej rodzic jest aktywny ORAZ spełnia swój warunek
                        is_condition_met = (
                            condition_fn(current_value)
                            if current_value is not None
                            else False
                        )
                        control_to_update_widget.setEnabled(
                            parent_active and is_condition_met
                        )
                    else:
                        self.logger.debug(
                            f"_update_ui_state: Dependent control {control_to_update_attr} for {main_control_attr} not found."
                        )

    # Usunięcie starych metod aktualizacji UI
    # def _update_dependent_controls(self):
    #     pass # Już niepotrzebne

    # def _update_architecture_dependent_controls(self):
    #     pass # Logika przeniesiona do _update_ui_state i _update_variant_combo

    # def _update_training_dependent_controls(self):
    #     pass # Logika przeniesiona

    # def _update_optimizer_dependent_controls(self):
    #     pass # Logika przeniesiona

    # def _update_scheduler_dependent_controls(self):
    #     pass # Logika przeniesiona

    # def _update_augmentation_dependent_controls(self):
    #     pass # Logika przeniesiona

    # def _update_preprocessing_dependent_controls(self):
    #     pass # Była pusta lub redundantna

    # def _update_monitoring_dependent_controls(self):
    #     pass # Logika przeniesiona

    # def _update_forgetting_controls(self):
    #     pass # Logika przeniesiona do _update_ui_state

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
        forgetting_layout.addRow("", self.prevent_forgetting_check)
        forgetting_layout.addRow("", self.preserve_classes_check)
        forgetting_layout.addRow("", self.rehearsal_check)
        forgetting_layout.addRow(
            "Liczba przykładów na klasę:", self.samples_per_class_spin
        )
        forgetting_layout.addRow("", self.synthetic_samples_check)
        forgetting_layout.addRow("", self.knowledge_distillation_check)
        forgetting_layout.addRow("Temperatura:", self.kd_temperature_spin)
        forgetting_layout.addRow("Alpha:", self.kd_alpha_spin)
        forgetting_layout.addRow("", self.ewc_check)
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

        # Rozmiar batcha
        self.batch_size_spin = QtWidgets.QSpinBox()
        self.batch_size_spin.setRange(1, 1024)
        self.batch_size_spin.setValue(32)
        basic_layout.addRow("Rozmiar batcha:", self.batch_size_spin)

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

        # Liczba workerów
        self.num_workers_spin = QtWidgets.QSpinBox()
        self.num_workers_spin.setRange(0, 32)
        self.num_workers_spin.setValue(4)
        basic_layout.addRow("Liczba workerów:", self.num_workers_spin)

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

        # Gradient accumulation steps
        self.grad_accum_steps_spin = QtWidgets.QSpinBox()
        self.grad_accum_steps_spin.setRange(1, 100)
        self.grad_accum_steps_spin.setValue(1)
        advanced_layout.addRow(
            "Gradient accumulation steps:", self.grad_accum_steps_spin
        )

        # Mixed precision
        self.mixed_precision_check = QtWidgets.QCheckBox()
        self.mixed_precision_check.setChecked(True)
        advanced_layout.addRow("Mixed precision:", self.mixed_precision_check)

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

    def _on_accept(self):
        """Obsługa zatwierdzenia konfiguracji."""
        import os

        from PyQt6 import QtWidgets

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
            if not os.path.isdir(training_dir):
                QtWidgets.QMessageBox.critical(
                    self, "Błąd", f"Katalog treningowy nie istnieje:\n{training_dir}"
                )
                return
            subdirs = [
                d
                for d in os.listdir(training_dir)
                if os.path.isdir(os.path.join(training_dir, d))
            ]
            if not subdirs:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Błąd",
                    f"Katalog treningowy nie zawiera żadnych podfolderów (klas):\n{training_dir}",
                )
                return

            # Sprawdź czy katalog walidacyjny jest ustawiony
            validation_dir = self.val_dir_edit.text().strip()
            if not validation_dir:
                QtWidgets.QMessageBox.warning(
                    self, "Błąd", "Katalog walidacyjny nie może być pusty."
                )
                return
            # WALIDACJA katalogu walidacyjnego
            if not os.path.isdir(validation_dir):
                QtWidgets.QMessageBox.critical(
                    self, "Błąd", f"Katalog walidacyjny nie istnieje:\n{validation_dir}"
                )
                return
            val_subdirs = [
                d
                for d in os.listdir(validation_dir)
                if os.path.isdir(os.path.join(validation_dir, d))
            ]
            if not val_subdirs:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Błąd",
                    f"Katalog walidacyjny nie zawiera żadnych podfolderów (klas):\n{validation_dir}",
                )
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
                    "advanced": {
                        "contrast": self.contrast_spin.value(),
                        "saturation": self.saturation_spin.value(),
                        "hue": self.hue_spin.value(),
                        "shear": self.shear_spin.value(),
                        "channel_shift": self.channel_shift_spin.value(),
                    },
                },
                "preprocessing": {
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
                        "save_to_csv": self.use_csv_check.isChecked(),
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
                        "save_frequency": self.save_freq_spin.value(),
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

            self.task_config = {
                "name": self.name_edit.text().strip(),
                "type": "fine_tuning",  # Upewniamy się, że typ jest ustawiony na "fine_tuning"
                "config": config,
                "training_time": 0,
                "training_time_str": "0:00:00",
                "status": "Oczekujące",
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
            reply = QtWidgets.QMessageBox.question(
                self,
                "Potwierdzenie",
                f"Czy na pewno chcesz usunąć profil {profile_name}?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )

            if reply == QtWidgets.QMessageBox.StandardButton.Yes:
                profile_path = self.profiles_dir / f"{profile_name}.json"
                profile_path.unlink()
                self._refresh_profile_list()
                self.current_profile = None
                QtWidgets.QMessageBox.information(
                    self, "Sukces", "Profil został pomyślnie usunięty."
                )

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
                # Użyj self.profiles_dir do tworzenia ścieżki
                output_path = str(self.profiles_dir / f"{name.strip()}.json")
                temp_base_profile_path = self.profiles_dir / "temp_base_profile.json"

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
