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

    # Stałe dla strategii odmrażania
    UNFREEZE_ALL = "unfreeze_all"
    UNFREEZE_GRADUAL_END = "unfreeze_gradual_end"
    UNFREEZE_GRADUAL_START = "unfreeze_gradual_start"
    UNFREEZE_GRADUAL_BOTH = "unfreeze_gradual_both"
    UNFREEZE_SPECIFIC = "unfreeze_specific"

    def __init__(self, parent=None, settings=None, hardware_profile=None):
        """Inicjalizacja okna dialogowego."""
        super().__init__(parent)
        self.settings = settings
        self.hardware_profile = hardware_profile
        self.logger = self._setup_logging()
        self.logger.info("Inicjalizacja okna konfiguracji fine-tuningu")
        self.setWindowTitle("Konfiguracja zadania fine-tuningu")
        self.setMinimumWidth(800)
        self.profiles_dir = Path("profiles/fine_tuning")
        self.profiles_dir.mkdir(exist_ok=True)
        self.current_profile = None
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

            # 1. Zakładka: Model i Dane
            tab = self._create_model_data_tab()
            self.tabs.addTab(tab, "Model i Dane")

            # 2. Zakładka: Parametry Fine-tuningu
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

    def _create_model_data_tab(self):
        """Tworzenie zakładki Model i Dane."""
        try:
            self.logger.debug("Tworzenie zakładki Model i Dane")
            tab = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(tab)
            form = QtWidgets.QFormLayout()

            # Wybór modelu bazowego
            model_dir_layout = QtWidgets.QHBoxLayout()
            self.model_dir_edit = QtWidgets.QLineEdit()
            model_dir_btn = QtWidgets.QPushButton("Przeglądaj...")
            model_dir_btn.clicked.connect(self._select_model_dir)
            model_dir_layout.addWidget(self.model_dir_edit)
            model_dir_layout.addWidget(model_dir_btn)

            form.addRow("Model bazowy:", model_dir_layout)

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
        """Tworzenie zakładki Parametry."""
        try:
            self.logger.debug("Tworzenie zakładki parametrów")
            tab = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(tab)
            form = QtWidgets.QFormLayout()

            # Liczba epok
            self.epochs_spin = QtWidgets.QSpinBox()
            self.epochs_spin.setRange(1, 1000)
            # Mniejsza liczba epok dla fine-tuningu
            self.epochs_spin.setValue(10)
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
            # Mniejszy learning rate dla fine-tuningu
            self.lr_spin.setValue(0.0001)
            form.addRow("Współczynnik uczenia:", self.lr_spin)

            # Optymalizator
            self.optimizer_combo = QtWidgets.QComboBox()
            optimizers = ["Adam", "AdamW", "SGD"]
            self.optimizer_combo.addItems(optimizers)
            form.addRow("Optymalizator:", self.optimizer_combo)

            # Scheduler
            self.scheduler_combo = QtWidgets.QComboBox()
            schedulers = ["None", "ReduceLROnPlateau", "CosineAnnealingLR"]
            self.scheduler_combo.addItems(schedulers)
            form.addRow("Scheduler:", self.scheduler_combo)

            # Liczba workerów
            self.num_workers_spin = QtWidgets.QSpinBox()
            self.num_workers_spin.setRange(0, 32)
            self.num_workers_spin.setValue(4)
            form.addRow("Liczba workerów:", self.num_workers_spin)

            # Mixed precision
            self.mixed_precision_check = QtWidgets.QCheckBox("Używaj mixed precision")
            self.mixed_precision_check.setChecked(True)
            form.addRow("", self.mixed_precision_check)

            # Strategia odmrażania
            self.unfreeze_strategy = QtWidgets.QComboBox()
            strategies = [
                "Odmroź wszystkie warstwy",
                "Odmrażaj stopniowo od końca",
                "Odmrażaj stopniowo od początku",
                "Odmrażaj stopniowo z obu stron",
                "Odmroź wybrane warstwy",
            ]
            self.unfreeze_strategy.addItems(strategies)
            form.addRow("Strategia odmrażania:", self.unfreeze_strategy)

            # Warstwy do odmrożenia
            self.unfreeze_layers = QtWidgets.QLineEdit()
            self.unfreeze_layers.setPlaceholderText("np. 1,2,3 lub 1-3")
            form.addRow("Warstwy do odmrożenia:", self.unfreeze_layers)

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
            desc = profile_data.get("description", "")
            self.profile_description.setText(desc)

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

            # Zastosowanie podstawowych parametrów
            ft_config = profile_data.get("fine_tuning", {})
            self.epochs_spin.setValue(ft_config.get("epochs", 10))
            batch_size = ft_config.get("batch_size", 32)
            self.batch_size_spin.setValue(batch_size)
            learning_rate = ft_config.get("learning_rate", 0.0001)
            self.lr_spin.setValue(learning_rate)
            optimizer = ft_config.get("optimizer", "Adam")
            self.optimizer_combo.setCurrentText(optimizer)
            scheduler = ft_config.get("scheduler", "None")
            self.scheduler_combo.setCurrentText(scheduler)
            num_workers = ft_config.get("num_workers", 4)
            self.num_workers_spin.setValue(num_workers)
            mixed_precision = ft_config.get("mixed_precision", True)
            self.mixed_precision_check.setChecked(mixed_precision)
            strategy = ft_config.get("unfreeze_strategy", "unfreeze_all")
            self.unfreeze_strategy.setCurrentText(strategy)
            layers = str(ft_config.get("unfreeze_layers", ""))
            self.unfreeze_layers.setText(layers)

            # Zastosowanie parametrów regularyzacji
            reg_config = ft_config.get("regularization", {})
            self.weight_decay_spin.setValue(reg_config.get("weight_decay", 0.0001))
            self.gradient_clip_spin.setValue(reg_config.get("gradient_clipping", 1.0))
            self.label_smoothing_spin.setValue(reg_config.get("label_smoothing", 0.1))
            self.dropout_spin.setValue(reg_config.get("dropout_rate", 0.2))

            # Zastosowanie parametrów augmentacji
            aug_config = ft_config.get("augmentation", {})
            self.basic_aug_check.setChecked(aug_config.get("enabled", True))
            self.rotation_spin.setValue(aug_config.get("rotation", 30))
            self.brightness_spin.setValue(aug_config.get("brightness", 0.2))
            self.shift_spin.setValue(aug_config.get("shift", 0.1))
            self.zoom_spin.setValue(aug_config.get("zoom", 0.1))
            self.horizontal_flip_check.setChecked(
                aug_config.get("horizontal_flip", True)
            )
            self.vertical_flip_check.setChecked(aug_config.get("vertical_flip", False))

            # Zastosowanie parametrów monitorowania
            mon_config = ft_config.get("monitoring", {})
            metrics = mon_config.get("metrics", [])
            self.accuracy_check.setChecked("accuracy" in metrics)
            self.precision_check.setChecked("precision" in metrics)
            self.recall_check.setChecked("recall" in metrics)
            self.f1_check.setChecked("f1" in metrics)
            self.confusion_matrix_check.setChecked("confusion_matrix" in metrics)

            # Zastosowanie parametrów wczesnego zatrzymania
            es_config = mon_config.get("early_stopping", {})
            self.patience_spin.setValue(es_config.get("patience", 10))
            self.min_delta_spin.setValue(es_config.get("min_delta", 0.001))
            self.monitor_combo.setCurrentText(es_config.get("monitor", "val_loss"))

            # Zastosowanie parametrów checkpointowania
            cp_config = mon_config.get("checkpointing", {})
            self.best_only_check.setChecked(cp_config.get("best_only", True))
            self.save_freq_spin.setValue(cp_config.get("save_frequency", 1))
            self.checkpoint_metric_combo.setCurrentText(
                cp_config.get("monitor", "val_loss")
            )

            QtWidgets.QMessageBox.information(
                self, "Sukces", "Profil został pomyślnie zastosowany."
            )

        except Exception as e:
            self.logger.error(
                f"Błąd podczas stosowania profilu: {str(e)}", exc_info=True
            )
            QtWidgets.QMessageBox.critical(
                self,
                "Błąd",
                f"Wystąpił błąd podczas stosowania profilu: {str(e)}",
            )

    def _save_profile(self):
        """Zapisanie aktualnej konfiguracji jako profil."""
        try:
            name, ok = QtWidgets.QInputDialog.getText(
                self, "Zapisz profil", "Nazwa profilu:"
            )
            if not ok or not name.strip():
                return

            # Przygotowanie metryk monitorowania
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

            # Przygotowanie parametrów augmentacji
            augmentation = {
                "enabled": self.basic_aug_check.isChecked(),
                "rotation": self.rotation_spin.value(),
                "brightness": self.brightness_spin.value(),
                "shift": self.shift_spin.value(),
                "zoom": self.zoom_spin.value(),
                "horizontal_flip": self.horizontal_flip_check.isChecked(),
                "vertical_flip": self.vertical_flip_check.isChecked(),
            }

            # Przygotowanie parametrów wczesnego zatrzymania
            early_stopping = {
                "enabled": True,
                "patience": self.patience_spin.value(),
                "min_delta": self.min_delta_spin.value(),
                "monitor": self.monitor_combo.currentText(),
            }

            # Przygotowanie parametrów checkpointowania
            checkpointing = {
                "best_only": self.best_only_check.isChecked(),
                "save_frequency": self.save_freq_spin.value(),
                "monitor": self.checkpoint_metric_combo.currentText(),
            }

            profile_data = {
                "name": name,
                "type": "Fine-tuning",
                "info": "Profil konfiguracji fine-tuningu",
                "description": "Zapisany profil konfiguracji fine-tuningu",
                "fine_tuning": {
                    "epochs": self.epochs_spin.value(),
                    "batch_size": self.batch_size_spin.value(),
                    "learning_rate": float(self.lr_spin.value()),
                    "optimizer": self.optimizer_combo.currentText(),
                    "scheduler": self.scheduler_combo.currentText(),
                    "num_workers": self.num_workers_spin.value(),
                    "mixed_precision": self.mixed_precision_check.isChecked(),
                    "unfreeze_strategy": self._get_unfreeze_strategy_value(
                        self.unfreeze_strategy.currentText()
                    ),
                    "unfreeze_layers": self._get_unfreeze_layers_value(
                        self.unfreeze_layers.text()
                    ),
                    "regularization": {
                        "weight_decay": float(self.weight_decay_spin.value()),
                        "gradient_clipping": float(self.gradient_clip_spin.value()),
                        "label_smoothing": float(self.label_smoothing_spin.value()),
                        "dropout_rate": float(self.dropout_spin.value()),
                    },
                    "augmentation": augmentation,
                    "monitoring": {
                        "metrics": metrics,
                        "early_stopping": early_stopping,
                        "checkpointing": checkpointing,
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
                self,
                "Błąd",
                f"Wystąpił błąd podczas zapisywania profilu: {str(e)}",
            )

    def _get_unfreeze_strategy_value(self, display_text):
        """Konwertuje wyświetlaną wartość strategii odmrażania na wartość wewnętrzną."""
        if "Odmroź wszystkie warstwy" in display_text:
            return self.UNFREEZE_ALL
        elif "Odmrażaj stopniowo od końca" in display_text:
            return self.UNFREEZE_GRADUAL_END
        elif "Odmrażaj stopniowo od początku" in display_text:
            return self.UNFREEZE_GRADUAL_START
        elif "Odmrażaj stopniowo z obu stron" in display_text:
            return self.UNFREEZE_GRADUAL_BOTH
        elif "Odmroź wybrane warstwy" in display_text:
            return self.UNFREEZE_SPECIFIC
        else:
            return self.UNFREEZE_ALL  # Domyślna wartość

    def _get_unfreeze_layers_value(self, value):
        """Konwertuje wartość warstw do odmrożenia na odpowiedni format."""
        if not value.strip():
            return []

        try:
            # Obsługa zakresów (np. "1-3")
            if "-" in value:
                start, end = map(int, value.split("-"))
                return list(range(start, end + 1))

            # Obsługa pojedynczych warstw (np. "1,2,3")
            return [int(x.strip()) for x in value.split(",")]
        except ValueError:
            self.logger.warning(
                f"Nieprawidłowy format warstw: {value}. " "Używam domyślnej wartości."
            )
            return []

    def _on_accept(self):
        """Obsługa zatwierdzenia konfiguracji."""
        try:
            self.logger.info("Rozpoczęcie walidacji i zapisu konfiguracji")

            # Walidacja modelu bazowego
            model_dir = self.model_dir_edit.text()
            if not model_dir.strip():
                self.logger.warning("Nie wybrano modelu bazowego")
                QtWidgets.QMessageBox.critical(
                    self, "Błąd", "Musisz wybrać model bazowy!"
                )
                return

            # Walidacja katalogu treningowego
            train_dir = self.train_dir_edit.text()
            if not train_dir.strip():
                self.logger.warning("Nie wybrano katalogu treningowego")
                msg = "Musisz wybrać katalog danych treningowych!"
                QtWidgets.QMessageBox.critical(self, "Błąd", msg)
                return

            if not validate_training_directory(train_dir):
                msg = f"Nieprawidłowy katalog treningowy: {train_dir}"
                self.logger.error(msg)
                return

            # Walidacja katalogu walidacyjnego
            val_dir = self.val_dir_edit.text()
            if val_dir and not validate_validation_directory(val_dir):
                msg = f"Nieprawidłowy katalog walidacyjny: {val_dir}"
                self.logger.error(msg)
                return

            # Przygotowanie konfiguracji
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            task_name = f"fine_tuning_{timestamp}.json"

            # Przygotowanie metryk monitorowania
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

            # Przygotowanie parametrów augmentacji
            augmentation = {
                "enabled": self.basic_aug_check.isChecked(),
                "rotation": self.rotation_spin.value(),
                "brightness": self.brightness_spin.value(),
                "shift": self.shift_spin.value(),
                "zoom": self.zoom_spin.value(),
                "horizontal_flip": self.horizontal_flip_check.isChecked(),
                "vertical_flip": self.vertical_flip_check.isChecked(),
            }

            # Przygotowanie parametrów wczesnego zatrzymania
            early_stopping = {
                "enabled": True,
                "patience": self.patience_spin.value(),
                "min_delta": self.min_delta_spin.value(),
                "monitor": self.monitor_combo.currentText(),
            }

            # Przygotowanie parametrów checkpointowania
            checkpointing = {
                "best_only": self.best_only_check.isChecked(),
                "save_frequency": self.save_freq_spin.value(),
                "monitor": self.checkpoint_metric_combo.currentText(),
            }

            self.task_config = {
                "name": task_name,
                "type": "Fine-tuning",
                "status": "Nowy",
                "priority": 0,
                "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "config": {
                    "model_dir": model_dir,
                    "train_dir": train_dir,
                    "val_dir": val_dir,
                    "num_classes": self.num_classes_spin.value(),
                    "fine_tuning": {
                        "epochs": self.epochs_spin.value(),
                        "batch_size": self.batch_size_spin.value(),
                        "learning_rate": float(self.lr_spin.value()),
                        "optimizer": self.optimizer_combo.currentText(),
                        "scheduler": self.scheduler_combo.currentText(),
                        "num_workers": self.num_workers_spin.value(),
                        "mixed_precision": self.mixed_precision_check.isChecked(),
                        "unfreeze_strategy": self._get_unfreeze_strategy_value(
                            self.unfreeze_strategy.currentText()
                        ),
                        "unfreeze_layers": self._get_unfreeze_layers_value(
                            self.unfreeze_layers.text()
                        ),
                        "regularization": {
                            "weight_decay": float(self.weight_decay_spin.value()),
                            "gradient_clipping": float(self.gradient_clip_spin.value()),
                            "label_smoothing": float(self.label_smoothing_spin.value()),
                            "dropout_rate": float(self.dropout_spin.value()),
                        },
                        "augmentation": augmentation,
                        "monitoring": {
                            "metrics": metrics,
                            "early_stopping": early_stopping,
                            "checkpointing": checkpointing,
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
        """Tworzenie zakładki Regularyzacja."""
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

            # Dropout rate
            self.dropout_spin = QtWidgets.QDoubleSpinBox()
            self.dropout_spin.setRange(0.0, 0.5)
            self.dropout_spin.setDecimals(3)
            self.dropout_spin.setValue(0.2)
            form.addRow("Dropout Rate:", self.dropout_spin)

            layout.addLayout(form)
            return tab

        except Exception as e:
            msg = "Błąd podczas tworzenia zakładki"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            raise

    def _create_augmentation_tab(self):
        """Tworzenie zakładki Augmentacja."""
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

            layout.addWidget(basic_group)
            return tab

        except Exception as e:
            msg = "Błąd podczas tworzenia zakładki"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            raise

    def _create_monitoring_tab(self):
        """Tworzenie zakładki Monitorowanie."""
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
            self.confusion_matrix_check = QtWidgets.QCheckBox("Confusion Matrix")

            metrics_layout.addWidget(self.accuracy_check)
            metrics_layout.addWidget(self.precision_check)
            metrics_layout.addWidget(self.recall_check)
            metrics_layout.addWidget(self.f1_check)
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
            metrics = ["val_loss", "val_accuracy", "val_f1"]
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

            layout.addWidget(metrics_group)
            layout.addWidget(early_stop_group)
            layout.addWidget(checkpoint_group)
            return tab

        except Exception as e:
            msg = "Błąd podczas tworzenia zakładki"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            raise
