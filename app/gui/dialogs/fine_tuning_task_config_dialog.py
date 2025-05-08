import datetime
import json
import logging
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

            # 8. Zakładka: PEFT
            tab = self._create_peft_tab()
            self.tabs.addTab(tab, "PEFT")

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

    def _select_train_dir(self):
        """Wybór katalogu z danymi treningowymi."""
        try:
            dir_path = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Wybierz katalog z danymi treningowymi"
            )
            if dir_path:
                if validate_training_directory(dir_path):
                    self.train_dir_edit.setText(dir_path)
                    self.logger.info(f"Wybrano katalog treningowy: {dir_path}")
                else:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Ostrzeżenie",
                        "Wybrany katalog nie zawiera poprawnych danych treningowych.",
                    )
        except Exception as e:
            self.logger.error(f"Błąd podczas wyboru katalogu treningowego: {str(e)}")
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Nie udało się wybrać katalogu treningowego: {str(e)}"
            )

    def _select_val_dir(self):
        """Wybór katalogu z danymi walidacyjnymi."""
        try:
            dir_path = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Wybierz katalog z danymi walidacyjnymi"
            )
            if dir_path:
                if validate_validation_directory(dir_path):
                    self.val_dir_edit.setText(dir_path)
                    self.logger.info(f"Wybrano katalog walidacyjny: {dir_path}")
                else:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Ostrzeżenie",
                        "Wybrany katalog nie zawiera poprawnych danych walidacyjnych.",
                    )
        except Exception as e:
            self.logger.error(f"Błąd podczas wyboru katalogu walidacyjnego: {str(e)}")
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Nie udało się wybrać katalogu walidacyjnego: {str(e)}"
            )

    def _refresh_profile_list(self):
        """Odświeża listę dostępnych profili."""
        try:
            self.profile_list.clear()
            for profile_file in self.profiles_dir.glob("*.json"):
                try:
                    with open(profile_file, "r", encoding="utf-8") as f:
                        profile_data = json.load(f)
                        name = profile_data.get("name", profile_file.stem)
                        item = QtWidgets.QListWidgetItem(name)
                        item.setData(Qt.ItemDataRole.UserRole, profile_file)
                        self.profile_list.addItem(item)
                except Exception as e:
                    self.logger.error(
                        f"Błąd podczas wczytywania profilu {profile_file}: {str(e)}"
                    )
        except Exception as e:
            self.logger.error(f"Błąd podczas odświeżania listy profili: {str(e)}")

    def _on_profile_selected(self, current, previous):
        """Obsługa wyboru profilu z listy."""
        try:
            if current is None:
                self.profile_info.clear()
                self.current_profile = None
                return

            profile_file = current.data(Qt.ItemDataRole.UserRole)
            if not profile_file.exists():
                self.logger.error(f"Plik profilu nie istnieje: {profile_file}")
                return

            with open(profile_file, "r", encoding="utf-8") as f:
                profile_data = json.load(f)
                self.current_profile = profile_data
                info_text = f"Profil: {profile_data.get('name', '')}\n\n"
                info_text += f"Opis: {profile_data.get('description', '')}\n\n"
                info_text += f"Parametry:\n{json.dumps(profile_data.get('parameters', {}), indent=2)}"
                self.profile_info.setText(info_text)

        except Exception as e:
            self.logger.error(f"Błąd podczas wyboru profilu: {str(e)}")
            self.profile_info.clear()
            self.current_profile = None

    def _edit_profile(self):
        """Edycja wybranego profilu."""
        try:
            if not self.current_profile:
                QtWidgets.QMessageBox.warning(
                    self, "Ostrzeżenie", "Najpierw wybierz profil do edycji."
                )
                return

            # TODO: Implementacja edycji profilu
            self.logger.info("Edycja profilu - funkcjonalność w trakcie implementacji")

        except Exception as e:
            self.logger.error(f"Błąd podczas edycji profilu: {str(e)}")
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Nie udało się edytować profilu: {str(e)}"
            )

    def _apply_profile(self):
        """Zastosowanie wybranego profilu do konfiguracji."""
        try:
            if not self.current_profile:
                QtWidgets.QMessageBox.warning(
                    self, "Ostrzeżenie", "Najpierw wybierz profil do zastosowania."
                )
                return

            # TODO: Implementacja zastosowania profilu
            self.logger.info(
                "Zastosowanie profilu - funkcjonalność w trakcie implementacji"
            )

        except Exception as e:
            self.logger.error(f"Błąd podczas zastosowania profilu: {str(e)}")
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Nie udało się zastosować profilu: {str(e)}"
            )

    def _save_profile(self):
        """Zapisanie aktualnej konfiguracji jako profil."""
        try:
            # TODO: Implementacja zapisywania profilu
            self.logger.info(
                "Zapisywanie profilu - funkcjonalność w trakcie implementacji"
            )

        except Exception as e:
            self.logger.error(f"Błąd podczas zapisywania profilu: {str(e)}")
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Nie udało się zapisać profilu: {str(e)}"
            )

    def _delete_profile(self):
        """Usunięcie wybranego profilu."""
        try:
            if not self.current_profile:
                QtWidgets.QMessageBox.warning(
                    self, "Ostrzeżenie", "Najpierw wybierz profil do usunięcia."
                )
                return

            # TODO: Implementacja usuwania profilu
            self.logger.info(
                "Usuwanie profilu - funkcjonalność w trakcie implementacji"
            )

        except Exception as e:
            self.logger.error(f"Błąd podczas usuwania profilu: {str(e)}")
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Nie udało się usunąć profilu: {str(e)}"
            )

    def _clone_profile(self):
        """Klonowanie wybranego profilu."""
        try:
            if not self.current_profile:
                QtWidgets.QMessageBox.warning(
                    self, "Ostrzeżenie", "Najpierw wybierz profil do sklonowania."
                )
                return

            # TODO: Implementacja klonowania profilu
            self.logger.info(
                "Klonowanie profilu - funkcjonalność w trakcie implementacji"
            )

        except Exception as e:
            self.logger.error(f"Błąd podczas klonowania profilu: {str(e)}")
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Nie udało się sklonować profilu: {str(e)}"
            )

    def _on_architecture_changed(self, arch_name):
        """Obsługa zmiany architektury modelu."""
        try:
            self._update_variant_combo(arch_name)
        except Exception as e:
            self.logger.error(f"Błąd podczas zmiany architektury: {str(e)}")

    def _update_variant_combo(self, arch_name):
        """Aktualizuje listę dostępnych wariantów dla wybranej architektury."""
        try:
            self.variant_combo.clear()
            if arch_name == "EfficientNet":
                self.variant_combo.addItems(
                    ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"]
                )
            elif arch_name == "ConvNeXt":
                self.variant_combo.addItems(
                    ["Tiny", "Small", "Base", "Large", "XLarge"]
                )
        except Exception as e:
            self.logger.error(f"Błąd podczas aktualizacji wariantów: {str(e)}")

    def _create_monitoring_tab(self):
        """Tworzenie zakładki Monitorowanie i Zapis."""
        try:
            self.logger.debug("Tworzenie zakładki Monitorowanie")
            tab = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(tab)

            # Metryki
            metrics_group = QtWidgets.QGroupBox("Metryki")
            metrics_layout = QtWidgets.QVBoxLayout()

            # Podstawowe metryki
            self.accuracy_check = QtWidgets.QCheckBox("Accuracy")
            self.precision_check = QtWidgets.QCheckBox("Precision")
            self.recall_check = QtWidgets.QCheckBox("Recall")
            self.f1_check = QtWidgets.QCheckBox("F1")
            self.confusion_matrix_check = QtWidgets.QCheckBox("Confusion Matrix")

            # Zaawansowane metryki
            self.roc_auc_check = QtWidgets.QCheckBox("ROC AUC")
            self.pr_auc_check = QtWidgets.QCheckBox("PR AUC")
            self.top_k_check = QtWidgets.QCheckBox("Top-K Accuracy")

            metrics_layout.addWidget(self.accuracy_check)
            metrics_layout.addWidget(self.precision_check)
            metrics_layout.addWidget(self.recall_check)
            metrics_layout.addWidget(self.f1_check)
            metrics_layout.addWidget(self.confusion_matrix_check)
            metrics_layout.addWidget(self.roc_auc_check)
            metrics_layout.addWidget(self.pr_auc_check)
            metrics_layout.addWidget(self.top_k_check)

            metrics_group.setLayout(metrics_layout)
            layout.addWidget(metrics_group)

            # Katalog zapisu i logi
            save_group = QtWidgets.QGroupBox("Katalog zapisu i logi")
            save_layout = QtWidgets.QFormLayout()

            self.model_dir_edit = QtWidgets.QLineEdit()
            model_dir_btn = QtWidgets.QPushButton("Przeglądaj...")
            model_dir_btn.clicked.connect(self._select_model_dir)
            model_dir_layout = QtWidgets.QHBoxLayout()
            model_dir_layout.addWidget(self.model_dir_edit)
            model_dir_layout.addWidget(model_dir_btn)
            save_layout.addRow("Katalog zapisu:", model_dir_layout)

            save_group.setLayout(save_layout)
            layout.addWidget(save_group)

            return tab

        except Exception as e:
            msg = "Błąd podczas tworzenia zakładki Monitorowanie"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            raise

    def _select_model_dir(self):
        """Wybór katalogu do zapisu modelu."""
        try:
            dir_path = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Wybierz katalog do zapisu modelu"
            )
            if dir_path:
                self.model_dir_edit.setText(dir_path)
                self.logger.info(f"Wybrano katalog modelu: {dir_path}")
        except Exception as e:
            self.logger.error(f"Błąd podczas wyboru katalogu modelu: {str(e)}")
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Nie udało się wybrać katalogu modelu: {str(e)}"
            )

    def _create_peft_tab(self):
        """Tworzenie zakładki Parameter-Efficient Fine-Tuning (PEFT)."""
        try:
            self.logger.debug("Tworzenie zakładki PEFT")
            tab = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(tab)

            # Wybór techniki PEFT
            peft_group = QtWidgets.QGroupBox("Technika PEFT")
            peft_layout = QtWidgets.QFormLayout()

            self.peft_technique = QtWidgets.QComboBox()
            self.peft_technique.addItems(
                ["none", "lora", "prefix_tuning", "adapter", "prompt_tuning"]
            )
            self.peft_technique.currentTextChanged.connect(
                self._on_peft_technique_changed
            )

            peft_layout.addRow("Technika:", self.peft_technique)
            peft_group.setLayout(peft_layout)
            layout.addWidget(peft_group)

            # Konfiguracja LoRA
            lora_group = QtWidgets.QGroupBox("LoRA")
            lora_layout = QtWidgets.QFormLayout()

            self.lora_rank = QtWidgets.QSpinBox()
            self.lora_rank.setRange(1, 64)
            self.lora_rank.setValue(8)

            self.lora_alpha = QtWidgets.QSpinBox()
            self.lora_alpha.setRange(1, 64)
            self.lora_alpha.setValue(16)

            self.lora_dropout = QtWidgets.QDoubleSpinBox()
            self.lora_dropout.setRange(0.0, 0.5)
            self.lora_dropout.setValue(0.1)
            self.lora_dropout.setDecimals(2)

            self.lora_target_modules = QtWidgets.QLineEdit()
            self.lora_target_modules.setText("query,key,value")

            lora_layout.addRow("Rank:", self.lora_rank)
            lora_layout.addRow("Alpha:", self.lora_alpha)
            lora_layout.addRow("Dropout:", self.lora_dropout)
            lora_layout.addRow("Target Modules:", self.lora_target_modules)
            lora_group.setLayout(lora_layout)
            layout.addWidget(lora_group)

            # Konfiguracja Adaptera
            adapter_group = QtWidgets.QGroupBox("Adapter")
            adapter_layout = QtWidgets.QFormLayout()

            self.adapter_hidden_size = QtWidgets.QSpinBox()
            self.adapter_hidden_size.setRange(1, 256)
            self.adapter_hidden_size.setValue(64)

            self.adapter_type = QtWidgets.QComboBox()
            self.adapter_type.addItems(["houlsby", "pfeiffer"])

            self.adapter_activation = QtWidgets.QComboBox()
            self.adapter_activation.addItems(["relu", "gelu", "sigmoid", "tanh"])

            adapter_layout.addRow("Hidden Size:", self.adapter_hidden_size)
            adapter_layout.addRow("Typ:", self.adapter_type)
            adapter_layout.addRow("Aktywacja:", self.adapter_activation)
            adapter_group.setLayout(adapter_layout)
            layout.addWidget(adapter_group)

            # Konfiguracja Prompt Tuning
            prompt_group = QtWidgets.QGroupBox("Prompt Tuning")
            prompt_layout = QtWidgets.QFormLayout()

            self.num_virtual_tokens = QtWidgets.QSpinBox()
            self.num_virtual_tokens.setRange(1, 100)
            self.num_virtual_tokens.setValue(20)

            self.prompt_init = QtWidgets.QComboBox()
            self.prompt_init.addItems(["random", "text", "embedding"])

            prompt_layout.addRow("Liczba tokenów:", self.num_virtual_tokens)
            prompt_layout.addRow("Inicjalizacja:", self.prompt_init)
            prompt_group.setLayout(prompt_layout)
            layout.addWidget(prompt_group)

            return tab

        except Exception as e:
            msg = "Błąd podczas tworzenia zakładki PEFT"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            raise

    def _on_peft_technique_changed(self, technique):
        """Obsługa zmiany techniki PEFT."""
        try:
            # TODO: Implementacja obsługi zmiany techniki PEFT
            self.logger.info(f"Zmieniono technikę PEFT na: {technique}")
        except Exception as e:
            self.logger.error(f"Błąd podczas zmiany techniki PEFT: {str(e)}")

    def _apply_metrics_config(self, metrics):
        """Stosuje konfigurację metryk."""
        try:
            # Upewnij się, że wszystkie pola formularza istnieją
            if not hasattr(self, "accuracy_check"):
                self.accuracy_check = QtWidgets.QCheckBox("Accuracy")
            if not hasattr(self, "precision_check"):
                self.precision_check = QtWidgets.QCheckBox("Precision")
            if not hasattr(self, "recall_check"):
                self.recall_check = QtWidgets.QCheckBox("Recall")
            if not hasattr(self, "f1_check"):
                self.f1_check = QtWidgets.QCheckBox("F1")
            if not hasattr(self, "confusion_matrix_check"):
                self.confusion_matrix_check = QtWidgets.QCheckBox("Confusion Matrix")
            if not hasattr(self, "roc_auc_check"):
                self.roc_auc_check = QtWidgets.QCheckBox("ROC AUC")
            if not hasattr(self, "pr_auc_check"):
                self.pr_auc_check = QtWidgets.QCheckBox("PR AUC")
            if not hasattr(self, "top_k_check"):
                self.top_k_check = QtWidgets.QCheckBox("Top-K Accuracy")

            self.accuracy_check.setChecked("accuracy" in metrics)
            self.precision_check.setChecked("precision" in metrics)
            self.recall_check.setChecked("recall" in metrics)
            self.f1_check.setChecked("f1" in metrics)
            self.confusion_matrix_check.setChecked("confusion_matrix" in metrics)
            self.roc_auc_check.setChecked("roc_auc" in metrics)
            self.pr_auc_check.setChecked("pr_auc" in metrics)
            self.top_k_check.setChecked("top_k_accuracy" in metrics)

        except Exception as e:
            self.logger.error(f"Błąd podczas stosowania konfiguracji metryk: {str(e)}")

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

            self.task_config = {
                "name": task_name,
                "typ": "doszkalanie",
                "status": "Nowy",
                "priority": 0,
                "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "config": {
                    "train_dir": str(self.train_dir_edit.text()),
                    "data_dir": str(self.train_dir_edit.text()),
                    "val_dir": str(self.val_dir_edit.text()),
                    "base_model": str(self.base_model_edit.text()),
                    "architecture": self.arch_combo.currentText(),
                    "variant": self.variant_combo.currentText(),
                    "input_size": self.input_size_spin.value(),
                    "num_classes": self.num_classes_spin.value(),
                    "pretrained": self.pretrained_check.isChecked(),
                    "pretrained_weights": self.pretrained_weights_combo.currentText(),
                    "feature_extraction": self.feature_extraction_check.isChecked(),
                    "activation": self.activation_combo.currentText(),
                    "dropout_at_inference": self.dropout_at_inference_check.isChecked(),
                    "global_pool": self.global_pool_combo.currentText(),
                    "last_layer_activation": self.last_layer_activation_combo.currentText(),
                    "peft": {
                        "technique": (
                            self.peft_technique.currentText()
                            if hasattr(self, "peft_technique")
                            else "none"
                        ),
                        "lora": {
                            "rank": (
                                self.lora_rank.value()
                                if hasattr(self, "lora_rank")
                                else 8
                            ),
                            "alpha": (
                                self.lora_alpha.value()
                                if hasattr(self, "lora_alpha")
                                else 16
                            ),
                            "dropout": (
                                self.lora_dropout.value()
                                if hasattr(self, "lora_dropout")
                                else 0.1
                            ),
                            "target_modules": (
                                self.lora_target_modules.text().split(",")
                                if hasattr(self, "lora_target_modules")
                                else ["query", "key", "value"]
                            ),
                        },
                        "adapter": {
                            "hidden_size": (
                                self.adapter_hidden_size.value()
                                if hasattr(self, "adapter_hidden_size")
                                else 64
                            ),
                            "adapter_type": (
                                self.adapter_type.currentText()
                                if hasattr(self, "adapter_type")
                                else "houlsby"
                            ),
                            "adapter_activation": (
                                self.adapter_activation.currentText()
                                if hasattr(self, "adapter_activation")
                                else "relu"
                            ),
                        },
                        "prompt_tuning": {
                            "num_virtual_tokens": (
                                self.num_virtual_tokens.value()
                                if hasattr(self, "num_virtual_tokens")
                                else 20
                            ),
                            "prompt_init": (
                                self.prompt_init.currentText()
                                if hasattr(self, "prompt_init")
                                else "random"
                            ),
                        },
                    },
                },
            }

            self.logger.info(f"Utworzono konfigurację zadania: {task_name}")
            QtWidgets.QMessageBox.information(
                self, "Sukces", "Zadanie zostało pomyślnie dodane."
            )
            self.accept()

        except Exception as e:
            self.logger.error("Błąd podczas zapisywania konfiguracji", exc_info=True)
            QtWidgets.QMessageBox.critical(
                self,
                "Błąd",
                f"Wystąpił błąd podczas zapisywania konfiguracji: {str(e)}",
            )

    def _validate_basic_params(self):
        """Walidacja podstawowych parametrów."""
        try:
            if not self.base_model_edit.text():
                QtWidgets.QMessageBox.warning(
                    self, "Ostrzeżenie", "Wybierz model bazowy."
                )
                return False

            if not self.train_dir_edit.text():
                QtWidgets.QMessageBox.warning(
                    self, "Ostrzeżenie", "Wybierz katalog z danymi treningowymi."
                )
                return False

            if not self.val_dir_edit.text():
                QtWidgets.QMessageBox.warning(
                    self, "Ostrzeżenie", "Wybierz katalog z danymi walidacyjnymi."
                )
                return False

            return True

        except Exception as e:
            self.logger.error(f"Błąd podczas walidacji parametrów: {str(e)}")
            return False

    def get_task_config(self):
        """Zwraca konfigurację zadania."""
        return self.task_config

    def closeEvent(self, event):
        """Obsługa zamknięcia okna."""
        self.logger.info("Zamykanie okna konfiguracji")
        event.accept()

    def _create_training_params_tab(self):
        """Tworzenie zakładki Parametry Treningu."""
        try:
            self.logger.debug("Tworzenie zakładki Parametry Treningu")
            tab = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(tab)
            form = QtWidgets.QFormLayout()

            # Batch size
            self.batch_size_spin = QtWidgets.QSpinBox()
            self.batch_size_spin.setRange(1, 1024)
            self.batch_size_spin.setValue(32)
            form.addRow("Batch size:", self.batch_size_spin)

            # Liczba epok
            self.epochs_spin = QtWidgets.QSpinBox()
            self.epochs_spin.setRange(1, 1000)
            self.epochs_spin.setValue(100)
            form.addRow("Liczba epok:", self.epochs_spin)

            # Learning rate
            self.learning_rate_spin = QtWidgets.QDoubleSpinBox()
            self.learning_rate_spin.setRange(0.000001, 1.0)
            self.learning_rate_spin.setValue(0.001)
            self.learning_rate_spin.setDecimals(6)
            form.addRow("Learning rate:", self.learning_rate_spin)

            # Optimizer
            self.optimizer_combo = QtWidgets.QComboBox()
            self.optimizer_combo.addItems(["adam", "sgd", "adamw", "rmsprop"])
            form.addRow("Optimizer:", self.optimizer_combo)

            layout.addLayout(form)
            return tab

        except Exception as e:
            msg = "Błąd podczas tworzenia zakładki Parametry Treningu"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            raise

    def _create_regularization_tab(self):
        """Tworzenie zakładki Regularyzacja i Optymalizacja."""
        try:
            self.logger.debug("Tworzenie zakładki Regularyzacja")
            tab = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(tab)
            form = QtWidgets.QFormLayout()

            # Weight decay
            self.weight_decay_spin = QtWidgets.QDoubleSpinBox()
            self.weight_decay_spin.setRange(0.0, 1.0)
            self.weight_decay_spin.setValue(0.0001)
            self.weight_decay_spin.setDecimals(6)
            form.addRow("Weight decay:", self.weight_decay_spin)

            # Dropout
            self.dropout_spin = QtWidgets.QDoubleSpinBox()
            self.dropout_spin.setRange(0.0, 0.9)
            self.dropout_spin.setValue(0.2)
            self.dropout_spin.setDecimals(2)
            form.addRow("Dropout:", self.dropout_spin)

            layout.addLayout(form)
            return tab

        except Exception as e:
            msg = "Błąd podczas tworzenia zakładki Regularyzacja"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            raise

    def _create_augmentation_tab(self):
        """Tworzenie zakładki Augmentacja Danych."""
        try:
            self.logger.debug("Tworzenie zakładki Augmentacja")
            tab = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(tab)
            form = QtWidgets.QFormLayout()

            # Random horizontal flip
            self.horizontal_flip_check = QtWidgets.QCheckBox("Random horizontal flip")
            self.horizontal_flip_check.setChecked(True)
            form.addRow("", self.horizontal_flip_check)

            # Random vertical flip
            self.vertical_flip_check = QtWidgets.QCheckBox("Random vertical flip")
            self.vertical_flip_check.setChecked(False)
            form.addRow("", self.vertical_flip_check)

            # Random rotation
            self.rotation_check = QtWidgets.QCheckBox("Random rotation")
            self.rotation_check.setChecked(True)
            form.addRow("", self.rotation_check)

            layout.addLayout(form)
            return tab

        except Exception as e:
            msg = "Błąd podczas tworzenia zakładki Augmentacja"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            raise

    def _create_preprocessing_tab(self):
        """Tworzenie zakładki Preprocessing."""
        try:
            self.logger.debug("Tworzenie zakładki Preprocessing")
            tab = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(tab)
            form = QtWidgets.QFormLayout()

            # Normalizacja
            self.normalize_check = QtWidgets.QCheckBox("Normalizacja")
            self.normalize_check.setChecked(True)
            form.addRow("", self.normalize_check)

            # Resize
            self.resize_check = QtWidgets.QCheckBox("Resize")
            self.resize_check.setChecked(True)
            form.addRow("", self.resize_check)

            layout.addLayout(form)
            return tab

        except Exception as e:
            msg = "Błąd podczas tworzenia zakładki Preprocessing"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            raise

    def _create_advanced_tab(self):
        """Tworzenie zakładki Zaawansowane."""
        try:
            self.logger.debug("Tworzenie zakładki Zaawansowane")
            tab = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(tab)
            form = QtWidgets.QFormLayout()

            # Mixed precision
            self.mixed_precision_check = QtWidgets.QCheckBox("Mixed precision")
            self.mixed_precision_check.setChecked(True)
            form.addRow("", self.mixed_precision_check)

            # Gradient accumulation
            self.gradient_accumulation_spin = QtWidgets.QSpinBox()
            self.gradient_accumulation_spin.setRange(1, 32)
            self.gradient_accumulation_spin.setValue(1)
            form.addRow("Gradient accumulation:", self.gradient_accumulation_spin)

            layout.addLayout(form)
            return tab

        except Exception as e:
            msg = "Błąd podczas tworzenia zakładki Zaawansowane"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            raise
