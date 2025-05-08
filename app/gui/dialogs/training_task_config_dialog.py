import datetime
import logging

from PyQt6 import QtWidgets

from app.utils.config import DEFAULT_TRAINING_PARAMS
from app.utils.file_utils import (
    validate_training_directory,
    validate_validation_directory,
)


class TrainingTaskConfigDialog(QtWidgets.QDialog):
    """Dialog konfiguracji zadania treningu."""

    def __init__(self, parent=None, settings=None, hardware_profile=None):
        super().__init__(parent)
        self.settings = settings
        self.hardware_profile = hardware_profile
        self._setup_logging()
        self.setWindowTitle("Konfiguracja treningu")
        self.setMinimumWidth(1000)
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

            # 5. Zakładka: Monitorowanie i Zapis
            tab = self._create_monitoring_tab()
            self.tabs.addTab(tab, "Monitorowanie")

            # 6. Zakładka: Zaawansowane
            tab = self._create_advanced_tab()
            self.tabs.addTab(tab, "Zaawansowane")

            layout.addWidget(self.tabs)

            # Przyciski OK i Anuluj
            btn_ok = QtWidgets.QDialogButtonBox.StandardButton.Ok
            btn_cancel = QtWidgets.QDialogButtonBox.StandardButton.Cancel
            buttons = QtWidgets.QDialogButtonBox(btn_ok | btn_cancel)
            buttons.accepted.connect(self._on_accept)
            buttons.rejected.connect(self.reject)
            layout.addWidget(buttons)

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

            layout.addLayout(form)
            return tab

        except Exception as e:
            msg = "Błąd podczas tworzenia zakładki"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            raise

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

            basic_layout.addRow("", self.basic_aug_check)
            basic_layout.addRow("Kąt rotacji:", self.rotation_spin)
            basic_layout.addRow("Jasność:", self.brightness_spin)
            basic_layout.addRow("Przesunięcie:", self.shift_spin)
            basic_layout.addRow("Przybliżenie:", self.zoom_spin)
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

            advanced_layout.addRow("", self.mixup_check)
            advanced_layout.addRow("Alpha:", self.mixup_alpha_spin)
            advanced_layout.addRow("", self.cutmix_check)
            advanced_layout.addRow("Alpha:", self.cutmix_alpha_spin)
            advanced_group.setLayout(advanced_layout)

            layout.addWidget(basic_group)
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
            self.topk_check = QtWidgets.QCheckBox("Top-k Accuracy")

            metrics_layout.addWidget(self.accuracy_check)
            metrics_layout.addWidget(self.precision_check)
            metrics_layout.addWidget(self.recall_check)
            metrics_layout.addWidget(self.f1_check)
            metrics_layout.addWidget(self.topk_check)
            metrics_group.setLayout(metrics_layout)

            # Wczesne zatrzymanie
            early_stop_group = QtWidgets.QGroupBox("Wczesne zatrzymanie")
            early_stop_layout = QtWidgets.QFormLayout()

            self.patience_spin = QtWidgets.QSpinBox()
            self.patience_spin.setRange(1, 100)
            self.patience_spin.setValue(10)

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
            early_stop_layout.addRow("Metryka:", self.monitor_combo)
            early_stop_group.setLayout(early_stop_layout)

            # Checkpointowanie
            checkpoint_group = QtWidgets.QGroupBox("Checkpointowanie")
            checkpoint_layout = QtWidgets.QFormLayout()

            self.best_only_check = QtWidgets.QCheckBox("Tylko najlepszy model")
            self.save_freq_spin = QtWidgets.QSpinBox()
            self.save_freq_spin.setRange(1, 50)
            self.save_freq_spin.setValue(1)

            checkpoint_layout.addRow("", self.best_only_check)
            checkpoint_layout.addRow("Częstość zapisu:", self.save_freq_spin)
            checkpoint_group.setLayout(checkpoint_layout)

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

            scheduler_layout.addRow("Patience:", self.scheduler_patience)
            scheduler_layout.addRow("Factor:", self.scheduler_factor)
            scheduler_layout.addRow("Min LR:", self.min_lr)
            scheduler_group.setLayout(scheduler_layout)

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

            # Dodatkowe parametry
            extra_group = QtWidgets.QGroupBox("Parametry specyficzne")
            extra_layout = QtWidgets.QFormLayout()

            self.extra_params = QtWidgets.QLineEdit()
            placeholder = '{"param1": value1, "param2": value2}'
            self.extra_params.setPlaceholderText(placeholder)

            extra_layout.addRow("JSON:", self.extra_params)
            extra_group.setLayout(extra_layout)

            layout.addWidget(scheduler_group)
            layout.addWidget(weights_group)
            layout.addWidget(extra_group)
            return tab

        except Exception as e:
            msg = "Błąd podczas tworzenia zakładki"
            self.logger.error(f"{msg}: {str(e)}", exc_info=True)
            raise

    def _on_accept(self):
        """Obsługa zatwierdzenia konfiguracji."""
        try:
            self.logger.info("Rozpoczęcie walidacji i zapisu konfiguracji")

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

            self.task_config = {
                "name": task_name,
                "type": "Trening",
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
                        },
                        "early_stopping": {
                            "patience": self.patience_spin.value(),
                            "monitor": self.monitor_combo.currentText(),
                        },
                        "checkpointing": {
                            "best_only": self.best_only_check.isChecked(),
                            "save_frequency": self.save_freq_spin.value(),
                        },
                        "save_dir": self.model_dir_edit.text(),
                        "save_logs": self.save_logs_check.isChecked(),
                    },
                    "advanced": {
                        "scheduler": {
                            "patience": self.scheduler_patience.value(),
                            "factor": self.scheduler_factor.value(),
                            "min_lr": self.min_lr.value(),
                        },
                        "weights": {
                            "init_method": self.init_weights.currentText(),
                            "freeze_cnn": self.freeze_layers.isChecked(),
                        },
                        "extra_params": self.extra_params.text(),
                    },
                },
            }

            self.logger.info(f"Utworzono konfigurację zadania: {task_name}")
            self.accept()

        except Exception as e:
            self.logger.error("Błąd podczas zapisywania konfiguracji", exc_info=True)
            QtWidgets.QMessageBox.critical(
                self,
                "Błąd",
                f"Wystąpił błąd podczas zapisywania konfiguracji: {str(e)}",
            )

    def get_task_config(self):
        """Zwraca konfigurację zadania."""
        return self.task_config
