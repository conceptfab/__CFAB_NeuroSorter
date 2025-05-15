Zmiany w pliku training_task_config_dialog.py:
python# Zmiana 1: Zamień import PySide6 na PyQt6
# zmień:
from PySide6 import QtCore, QtWidgets
# na:
from PyQt6 import QtCore, QtWidgets
python# Zmiana 2: Dodaj brakujące komponenty do _init_ui, których oczekuje _create_training_params_tab
# Dodaj po zainicjalizowaniu tab_widget:
self.epochs_spin = QtWidgets.QSpinBox()
self.epochs_spin.setRange(1, 1000)
self.epochs_spin.setValue(self.default_params.get("max_epochs", 100))

self.lr_spin = QtWidgets.QDoubleSpinBox()
self.lr_spin.setRange(0.000001, 1.0)
self.lr_spin.setDecimals(6)
self.lr_spin.setValue(self.default_params.get("learning_rate", 0.001))

self.optimizer_combo = QtWidgets.QComboBox()
self.optimizer_combo.addItems(["AdamW", "SGD", "Adam", "RMSprop"])
self.optimizer_combo.setCurrentText(self.default_params.get("optimizer", "AdamW"))

self.scheduler_combo = QtWidgets.QComboBox()
self.scheduler_combo.addItems(["cosine", "step", "exponential", "none"])
self.scheduler_combo.setCurrentText(self.default_params.get("scheduler", "cosine"))

self.weight_decay_spin = QtWidgets.QDoubleSpinBox()
self.weight_decay_spin.setRange(0.0, 1.0)
self.weight_decay_spin.setDecimals(6)
self.weight_decay_spin.setValue(self.default_params.get("weight_decay", 0.0001))

self.validation_split_spin = QtWidgets.QDoubleSpinBox()
self.validation_split_spin.setRange(0.0, 0.5)
self.validation_split_spin.setDecimals(2)
self.validation_split_spin.setValue(self.default_params.get("validation_split", 0.2))
python# Zmiana 3: Implementuj brakującą metodę _create_training_params_tab
def _create_training_params_tab(self):
    """Tworzy zakładkę z parametrami treningu."""
    try:
        self.logger.debug("Tworzenie zakładki parametrów treningu")

        # Tworzenie widgetu zakładki
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout()

        # Liczba epok
        layout.addRow("Liczba epok:", self.epochs_spin)

        # Learning rate
        layout.addRow("Learning rate:", self.lr_spin)

        # Optymalizator
        layout.addRow("Optymalizator:", self.optimizer_combo)

        # Scheduler
        layout.addRow("Scheduler:", self.scheduler_combo)

        # Weight decay
        layout.addRow("Weight decay:", self.weight_decay_spin)

        # Podział walidacyjny
        layout.addRow("Podział walidacyjny:", self.validation_split_spin)

        # Mixed precision
        self.mixed_precision_check = QtWidgets.QCheckBox()
        self.mixed_precision_check.setChecked(
            self.default_params.get("use_mixed_precision", True)
        )
        layout.addRow("Mixed precision:", self.mixed_precision_check)

        # Gradient clipping
        self.gradient_clip_spin = QtWidgets.QDoubleSpinBox()
        self.gradient_clip_spin.setRange(0.0, 10.0)
        self.gradient_clip_spin.setDecimals(2)
        self.gradient_clip_spin.setValue(
            self.default_params.get("gradient_clip_val", 1.0)
        )
        layout.addRow("Gradient clipping:", self.gradient_clip_spin)

        # Ustawienie layoutu
        tab.setLayout(layout)
        return tab

    except Exception as e:
        self.logger.error(
            f"Błąd podczas tworzenia zakładki parametrów treningu: {str(e)}",
            exc_info=True,
        )
        QtWidgets.QMessageBox.critical(
            self,
            "Błąd",
            f"Nie można utworzyć zakładki parametrów treningu: {str(e)}",
        )
        return QtWidgets.QWidget()
python# Zmiana 4: Dodaj brakującą implementację _create_regularization_tab
def _create_regularization_tab(self):
    """Tworzy zakładkę z opcjami regularyzacji."""
    try:
        self.logger.debug("Tworzenie zakładki regularyzacji")

        # Tworzenie widgetu zakładki
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout()

        # Label smoothing
        self.label_smoothing_spin = QtWidgets.QDoubleSpinBox()
        self.label_smoothing_spin.setRange(0.0, 0.5)
        self.label_smoothing_spin.setDecimals(2)
        self.label_smoothing_spin.setValue(self.default_params.get("label_smoothing", 0.1))
        layout.addRow("Label smoothing:", self.label_smoothing_spin)

        # Dropout
        self.dropout_spin = QtWidgets.QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 0.9)
        self.dropout_spin.setDecimals(2)
        self.dropout_spin.setValue(self.default_params.get("dropout", 0.2))
        layout.addRow("Dropout:", self.dropout_spin)

        # Drop connect
        self.drop_connect_spin = QtWidgets.QDoubleSpinBox()
        self.drop_connect_spin.setRange(0.0, 0.9)
        self.drop_connect_spin.setDecimals(2)
        self.drop_connect_spin.setValue(self.default_params.get("drop_connect", 0.2))
        layout.addRow("Drop connect:", self.drop_connect_spin)

        # Stochastic depth probability
        self.stochastic_depth_prob_spin = QtWidgets.QDoubleSpinBox()
        self.stochastic_depth_prob_spin.setRange(0.0, 1.0)
        self.stochastic_depth_prob_spin.setDecimals(2)
        self.stochastic_depth_prob_spin.setValue(self.default_params.get("stochastic_depth_prob", 0.8))
        layout.addRow("Stochastic depth probability:", self.stochastic_depth_prob_spin)

        # Ustawienie layoutu
        tab.setLayout(layout)
        return tab

    except Exception as e:
        self.logger.error(
            f"Błąd podczas tworzenia zakładki regularyzacji: {str(e)}",
            exc_info=True,
        )
        QtWidgets.QMessageBox.critical(
            self,
            "Błąd",
            f"Nie można utworzyć zakładki regularyzacji: {str(e)}",
        )
        return QtWidgets.QWidget()
python# Zmiana 5: Dodaj brakujące implementacje pozostałych zakładek
def _create_augmentation_tab(self):
    """Tworzy zakładkę z opcjami augmentacji danych."""
    try:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout()

        # Rotacja
        self.rotation_spin = QtWidgets.QSpinBox()
        self.rotation_spin.setRange(0, 180)
        self.rotation_spin.setValue(self.default_params.get("rotation", 30))
        layout.addRow("Rotacja (stopnie):", self.rotation_spin)

        # Jasność
        self.brightness_spin = QtWidgets.QDoubleSpinBox()
        self.brightness_spin.setRange(0.0, 1.0)
        self.brightness_spin.setDecimals(2)
        self.brightness_spin.setValue(self.default_params.get("brightness", 0.2))
        layout.addRow("Jasność:", self.brightness_spin)

        # Kontrast
        self.contrast_spin = QtWidgets.QDoubleSpinBox()
        self.contrast_spin.setRange(0.0, 1.0)
        self.contrast_spin.setDecimals(2)
        self.contrast_spin.setValue(self.default_params.get("contrast", 0.2))
        layout.addRow("Kontrast:", self.contrast_spin)

        # Nasycenie
        self.saturation_spin = QtWidgets.QDoubleSpinBox()
        self.saturation_spin.setRange(0.0, 1.0)
        self.saturation_spin.setDecimals(2)
        self.saturation_spin.setValue(self.default_params.get("saturation", 0.2))
        layout.addRow("Nasycenie:", self.saturation_spin)

        # Hue
        self.hue_spin = QtWidgets.QDoubleSpinBox()
        self.hue_spin.setRange(0.0, 0.5)
        self.hue_spin.setDecimals(2)
        self.hue_spin.setValue(self.default_params.get("hue", 0.1))
        layout.addRow("Hue:", self.hue_spin)

        # Horizontal flip
        self.horizontal_flip_check = QtWidgets.QCheckBox()
        self.horizontal_flip_check.setChecked(self.default_params.get("horizontal_flip", True))
        layout.addRow("Odbicie poziome:", self.horizontal_flip_check)

        # MixUp alpha
        self.mixup_alpha_spin = QtWidgets.QDoubleSpinBox()
        self.mixup_alpha_spin.setRange(0.0, 1.0)
        self.mixup_alpha_spin.setDecimals(2)
        self.mixup_alpha_spin.setValue(self.default_params.get("mixup_alpha", 0.2))
        layout.addRow("MixUp alpha:", self.mixup_alpha_spin)

        # CutMix alpha
        self.cutmix_alpha_spin = QtWidgets.QDoubleSpinBox()
        self.cutmix_alpha_spin.setRange(0.0, 1.0)
        self.cutmix_alpha_spin.setDecimals(2)
        self.cutmix_alpha_spin.setValue(self.default_params.get("cutmix_alpha", 1.0))
        layout.addRow("CutMix alpha:", self.cutmix_alpha_spin)

        # RandAugment n
        self.randaugment_n_spin = QtWidgets.QSpinBox()
        self.randaugment_n_spin.setRange(0, 10)
        self.randaugment_n_spin.setValue(self.default_params.get("randaugment_n", 2))
        layout.addRow("RandAugment n:", self.randaugment_n_spin)

        # RandAugment m
        self.randaugment_m_spin = QtWidgets.QSpinBox()
        self.randaugment_m_spin.setRange(0, 30)
        self.randaugment_m_spin.setValue(self.default_params.get("randaugment_m", 9))
        layout.addRow("RandAugment m:", self.randaugment_m_spin)

        tab.setLayout(layout)
        return tab

    except Exception as e:
        self.logger.error(f"Błąd podczas tworzenia zakładki augmentacji: {str(e)}", exc_info=True)
        return QtWidgets.QWidget()
python# Zmiana 6: Dodaj brakujące implementacje processingTaba i monitoringTaba
def _create_preprocessing_tab(self):
    """Tworzy zakładkę z opcjami preprocessingu."""
    try:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout()

        # Normalizacja mean R
        self.norm_mean_r_spin = QtWidgets.QDoubleSpinBox()
        self.norm_mean_r_spin.setRange(0.0, 1.0)
        self.norm_mean_r_spin.setDecimals(3)
        self.norm_mean_r_spin.setValue(self.default_params.get("norm_mean_r", 0.485))
        layout.addRow("Normalizacja mean R:", self.norm_mean_r_spin)

        # Normalizacja mean G
        self.norm_mean_g_spin = QtWidgets.QDoubleSpinBox()
        self.norm_mean_g_spin.setRange(0.0, 1.0)
        self.norm_mean_g_spin.setDecimals(3)
        self.norm_mean_g_spin.setValue(self.default_params.get("norm_mean_g", 0.456))
        layout.addRow("Normalizacja mean G:", self.norm_mean_g_spin)

        # Normalizacja mean B
        self.norm_mean_b_spin = QtWidgets.QDoubleSpinBox()
        self.norm_mean_b_spin.setRange(0.0, 1.0)
        self.norm_mean_b_spin.setDecimals(3)
        self.norm_mean_b_spin.setValue(self.default_params.get("norm_mean_b", 0.406))
        layout.addRow("Normalizacja mean B:", self.norm_mean_b_spin)

        # Normalizacja std R
        self.norm_std_r_spin = QtWidgets.QDoubleSpinBox()
        self.norm_std_r_spin.setRange(0.001, 1.0)
        self.norm_std_r_spin.setDecimals(3)
        self.norm_std_r_spin.setValue(self.default_params.get("norm_std_r", 0.229))
        layout.addRow("Normalizacja std R:", self.norm_std_r_spin)

        # Normalizacja std G
        self.norm_std_g_spin = QtWidgets.QDoubleSpinBox()
        self.norm_std_g_spin.setRange(0.001, 1.0)
        self.norm_std_g_spin.setDecimals(3)
        self.norm_std_g_spin.setValue(self.default_params.get("norm_std_g", 0.224))
        layout.addRow("Normalizacja std G:", self.norm_std_g_spin)

        # Normalizacja std B
        self.norm_std_b_spin = QtWidgets.QDoubleSpinBox()
        self.norm_std_b_spin.setRange(0.001, 1.0)
        self.norm_std_b_spin.setDecimals(3)
        self.norm_std_b_spin.setValue(self.default_params.get("norm_std_b", 0.225))
        layout.addRow("Normalizacja std B:", self.norm_std_b_spin)

        # Metoda skalowania
        self.scaling_method = QtWidgets.QComboBox()
        self.scaling_method.addItems(["bilinear", "bicubic", "nearest", "area"])
        self.scaling_method.setCurrentText(self.default_params.get("scaling_method", "bilinear"))
        layout.addRow("Metoda skalowania:", self.scaling_method)

        # Zachowanie proporcji
        self.maintain_aspect_ratio = QtWidgets.QCheckBox()
        self.maintain_aspect_ratio.setChecked(self.default_params.get("maintain_aspect_ratio", True))
        layout.addRow("Zachowanie proporcji:", self.maintain_aspect_ratio)

        # Tryb padding
        self.pad_mode = QtWidgets.QComboBox()
        self.pad_mode.addItems(["constant", "reflect", "replicate"])
        self.pad_mode.setCurrentText(self.default_params.get("pad_mode", "constant"))
        layout.addRow("Tryb padding:", self.pad_mode)

        # Wartość padding
        self.pad_value = QtWidgets.QSpinBox()
        self.pad_value.setRange(0, 255)
        self.pad_value.setValue(self.default_params.get("pad_value", 0))
        layout.addRow("Wartość padding:", self.pad_value)

        tab.setLayout(layout)
        return tab

    except Exception as e:
        self.logger.error(f"Błąd podczas tworzenia zakładki preprocessingu: {str(e)}", exc_info=True)
        return QtWidgets.QWidget()

def _create_monitoring_tab(self):
    """Tworzy zakładkę z opcjami monitorowania."""
    try:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout()

        # Monitorowanie dokładności
        self.accuracy_check = QtWidgets.QCheckBox()
        self.accuracy_check.setChecked(self.default_params.get("monitor_accuracy", True))
        layout.addRow("Monitorowanie dokładności:", self.accuracy_check)

        # Monitorowanie precyzji
        self.precision_check = QtWidgets.QCheckBox()
        self.precision_check.setChecked(self.default_params.get("monitor_precision", True))
        layout.addRow("Monitorowanie precyzji:", self.precision_check)

        # Monitorowanie czułości
        self.recall_check = QtWidgets.QCheckBox()
        self.recall_check.setChecked(self.default_params.get("monitor_recall", True))
        layout.addRow("Monitorowanie czułości:", self.recall_check)

        # Monitorowanie F1
        self.f1_check = QtWidgets.QCheckBox()
        self.f1_check.setChecked(self.default_params.get("monitor_f1", True))
        layout.addRow("Monitorowanie F1:", self.f1_check)

        # Użycie TensorBoard
        self.use_tensorboard_check = QtWidgets.QCheckBox()
        self.use_tensorboard_check.setChecked(self.default_params.get("use_tensorboard", True))
        layout.addRow("Użycie TensorBoard:", self.use_tensorboard_check)

        # Zapisywanie logów
        self.save_logs_check = QtWidgets.QCheckBox()
        self.save_logs_check.setChecked(self.default_params.get("save_logs", True))
        layout.addRow("Zapisywanie logów:", self.save_logs_check)

        # Wizualizacja: liczba próbek
        self.viz_num_samples_spin = QtWidgets.QSpinBox()
        self.viz_num_samples_spin.setRange(0, 100)
        self.viz_num_samples_spin.setValue(self.default_params.get("viz_num_samples", 16))
        layout.addRow("Liczba próbek do wizualizacji:", self.viz_num_samples_spin)

        # Early stopping
        self.use_early_stopping_check = QtWidgets.QCheckBox()
        self.use_early_stopping_check.setChecked(self.default_params.get("use_early_stopping", True))
        self.use_early_stopping_check.toggled.connect(self._toggle_early_stopping_controls)
        layout.addRow("Użycie early stopping:", self.use_early_stopping_check)

        # Monitorowana wartość
        self.monitor_combo = QtWidgets.QComboBox()
        self.monitor_combo.addItems(["val_loss", "val_accuracy", "val_f1"])
        self.monitor_combo.setCurrentText(self.default_params.get("monitor", "val_loss"))
        layout.addRow("Monitorowana wartość:", self.monitor_combo)

        # Cierpliwość
        self.patience_spin = QtWidgets.QSpinBox()
        self.patience_spin.setRange(1, 100)
        self.patience_spin.setValue(self.default_params.get("early_stopping_patience", 10))
        layout.addRow("Cierpliwość:", self.patience_spin)

        # Minimalna delta
        self.min_delta_spin = QtWidgets.QDoubleSpinBox()
        self.min_delta_spin.setRange(0.0, 0.1)
        self.min_delta_spin.setDecimals(5)
        self.min_delta_spin.setValue(self.default_params.get("early_stopping_min_delta", 0.0001))
        layout.addRow("Minimalna delta:", self.min_delta_spin)

        # Tryb
        self.early_stopping_mode_combo = QtWidgets.QComboBox()
        self.early_stopping_mode_combo.addItems(["min", "max"])
        self.early_stopping_mode_combo.setCurrentText(self.default_params.get("early_stopping_mode", "min"))
        layout.addRow("Tryb:", self.early_stopping_mode_combo)

        # Używanie checkpointów
        self.use_checkpointing_check = QtWidgets.QCheckBox()
        self.use_checkpointing_check.setChecked(self.default_params.get("use_checkpointing", True))
        layout.addRow("Używanie checkpointów:", self.use_checkpointing_check)

        # Zapisywanie tylko najlepszych
        self.best_only_check = QtWidgets.QCheckBox()
        self.best_only_check.setChecked(self.default_params.get("save_best_only", True))
        layout.addRow("Zapisywanie tylko najlepszych:", self.best_only_check)

        # Częstotliwość zapisywania
        self.save_freq_spin = QtWidgets.QSpinBox()
        self.save_freq_spin.setRange(1, 100)
        self.save_freq_spin.setValue(self.default_params.get("save_frequency", 1))
        layout.addRow("Częstotliwość zapisywania (epoki):", self.save_freq_spin)

        # Top K checkpointów
        self.checkpoint_top_k_spin = QtWidgets.QSpinBox()
        self.checkpoint_top_k_spin.setRange(1, 10)
        self.checkpoint_top_k_spin.setValue(self.default_params.get("checkpoint_top_k", 1))
        layout.addRow("Top K checkpointów:", self.checkpoint_top_k_spin)

        # Początkowy stan kontrolek early stopping
        self._toggle_early_stopping_controls(self.use_early_stopping_check.isChecked())

        tab.setLayout(layout)
        return tab

    except Exception as e:
        self.logger.error(f"Błąd podczas tworzenia zakładki monitorowania: {str(e)}", exc_info=True)
        return QtWidgets.QWidget()
python# Zmiana 7: Dodaj brakującą metodę do aktualizacji wariantów modelu
def _update_variant_combo(self, architecture):
    """Aktualizuje listę wariantów w zależności od wybranej architektury."""
    self.variant_combo.clear()
    if architecture == "EfficientNet":
        self.variant_combo.addItems(["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7"])
    elif architecture == "ResNet":
        self.variant_combo.addItems(["18", "34", "50", "101", "152"])
    elif architecture == "DenseNet":
        self.variant_combo.addItems(["121", "169", "201", "264"])
    elif architecture == "MobileNet":
        self.variant_combo.addItems(["v2", "v3_small", "v3_large"])
    else:
        self.variant_combo.addItems(["Standardowy"])
python# Zmiana 8: Dodaj brakującą klasę HardwareProfileDialog
class HardwareProfileDialog(QtWidgets.QDialog):
    """Dialog wyświetlający szczegóły profilu sprzętowego."""
    
    def __init__(self, hardware_profile, parent=None):
        super().__init__(parent)
        self.hardware_profile = hardware_profile
        self.setWindowTitle("Profil sprzętowy")
        self.resize(600, 400)
        self._init_ui()
        
    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout()
        
        # Utwórz widget tekstowy do wyświetlania danych profilu
        self.text_edit = QtWidgets.QTextEdit()
        self.text_edit.setReadOnly(True)
        
        if self.hardware_profile:
            html = "<h3>Szczegóły profilu sprzętowego</h3>"
            html += "<table border='0' style='margin-left:10px'>"
            
            # Grupuj dane profilu w sekcje
            sections = {
                "Ogólne": ["name", "id", "created_at"],
                "CPU": ["cpu_info", "cpu_cores", "cpu_threads"],
                "GPU": ["gpu_info", "gpu_memory", "cuda_version"],
                "Pamięć": ["ram_total", "ram_available"],
                "Rekomendacje": ["recommended_batch_size", "recommended_workers", 
                               "use_mixed_precision", "cudnn_benchmark", "pin_memory"]
            }
            
            for section, keys in sections.items():
                html += f"<tr><td colspan='2'><b>{section}</b></td></tr>"
                
                for key in keys:
                    if key in self.hardware_profile:
                        value = self.hardware_profile[key]
                        
                        # Formatowanie wartości w zależności od typu
                        if isinstance(value, bool):
                            value_str = "Tak" if value else "Nie"
                        elif isinstance(value, (dict, list)):
                            try:
                                import json
                                value_str = json.dumps(value, indent=2, ensure_ascii=False)
                            except:
                                value_str = str(value)
                        else:
                            value_str = str(value)
                            
                        # Jeśli to długi tekst, ograniczamy długość w tabeli
                        display_value = value_str
                        if len(value_str) > 100:  # Limit dla wyświetlania w tabeli
                            display_value = value_str[:100] + "..."
                            
                        html += f"<tr><td style='padding-left:20px'>{key}</td><td>{display_value}</td></tr>"
            
            html += "</table>"
            self.text_edit.setHtml(html)
        else:
            self.text_edit.setPlainText("Brak dostępnego profilu sprzętowego.")
            
        layout.addWidget(self.text_edit)
        
        # Przycisk zamknięcia
        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok)
        button_box.accepted.connect(self.accept)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
python# Zmiana 9: Dodaj brakujące pola dla DialogTaskConfig
# Dodaj do konstruktora lub metody _init_ui:
self.train_dir_edit = QtWidgets.QLineEdit()
self.val_dir_edit = QtWidgets.QLineEdit()
python# Zmiana 10: Dodaj metodę _get_current_config_as_dict
def _get_current_config_as_dict(self):
    """Zbiera stan UI do pliku konfiguracyjnego."""
    try:
        config = {}
        
        # Zbieranie stanu dla każdej sekcji
        config["model"] = self._collect_model_state()
        config["training"] = self._collect_training_state()
        config["regularization"] = self._collect_regularization_state()
        config["augmentation"] = self._collect_augmentation_state()
        config["preprocessing"] = self._collect_preprocessing_state()
        config["monitoring"] = self._collect_monitoring_state()
        config["advanced"] = self._collect_advanced_state()
        config["optimization"] = self._collect_optimization_state()
        
        return config
    
    except Exception as e:
        self.logger.error(f"Błąd podczas zbierania stanu UI: {str(e)}", exc_info=True)
        raise
Zmiany w pliku training_manager.py:
python# Zmiana 1: W metodzie _add_training_task, dodaj obsługę braku katalogów
# W metodzie _add_training_task, po utwozreniu dialog = TrainingTaskConfigDialog(...) dodaj:

# Utwórz katalogi dla zadań, jeśli nie istnieją
os.makedirs(os.path.join("data", "tasks"), exist_ok=True)
Podsumowanie głównych problemów i rozwiązań:

Niezgodność bibliotek: Dialog używa PySide6, a TrainingManager używa PyQt6
Brakujące implementacje metod tworzących zakładki w dialogu
Brakujące kontrolki UI wymagane przez metody
Brak poprawnej implementacji niektórych metod używanych przez dialog

Powyższe zmiany powinny rozwiązać problemy z uruchamianiem dialogu tworzącego nowe zadanie treningowe.