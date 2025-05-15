python# Poprawka 1: Korekta literówki w stałej UNFREEZE_AFTER_EPOCHS
# Linia ~35-38 w pliku training_task_config_dialog.py

class TrainingTaskConfigDialog(QtWidgets.QDialog):
    """Dialog konfiguracji zadania treningu."""
    
    # Strategie odmrażania warstw
    UNFREEZE_ALL = "unfreeze_all"
    UNFREEZE_GRADUAL_END = "unfreeze_gradual_end"
    UNFREEZE_GRADUAL_START = "unfreeze_gradual_start"
    UNFREEZE_AFTER_EPOCHS = "unfreeze_after_epochs"  # Poprawiono literówkę z "unfreeze_after_epoochs"
Ujednolicenie nazw parametrów w metodzie _get_current_config_as_dict
python# Poprawka 2: Ujednolicenie nazw parametrów dla mixed_precision
# W metodzie _get_current_config_as_dict

# Zmienić:
"mixed_precision": self.training_mixed_precision_check.isChecked(),

# Na:
"use_mixed_precision": self.training_mixed_precision_check.isChecked(),
python# Poprawka 3: Ujednolicenie nazw parametrów dla unfreeze_strategy między profilem a zadaniem
# W metodzie _get_unfreeze_strategy_value

def _get_unfreeze_strategy_value(self, display_text):
    if "unfreeze_all" in display_text:
        return self.UNFREEZE_ALL
    elif "unfreeze_gradual_end" in display_text:
        return self.UNFREEZE_GRADUAL_END
    elif "unfreeze_gradual_start" in display_text:
        return self.UNFREEZE_GRADUAL_START
    elif self.UNFREEZE_AFTER_EPOCHS in display_text:
        return self.UNFREEZE_AFTER_EPOCHS
    
    # Mapuj między formatami
    strategy_mapping = {
        "gradual": self.UNFREEZE_GRADUAL_END  # Ujednolicenie wartości "gradual" z profilu na "unfreeze_gradual_end"
    }
    if display_text in strategy_mapping:
        return strategy_mapping[display_text]
    
    return self.UNFREEZE_ALL  # Domyślna wartość
2. Poprawki związane z kontrolkami UI - przeniesienie między zakładkami
A. Przeniesienie kontrolki batch_size z Optymalizacji do Parametrów Treningu
python# Poprawka 4: Przeniesienie parametru batch_size z Optymalizacji do Parametrów Treningu
# W metodzie _create_training_params_tab, po self.training_epochs_spin

self.training_batch_size_spin = QtWidgets.QSpinBox()
self.training_batch_size_spin.setRange(1, 1024)
self.training_batch_size_spin.setValue(32)
training_hyperparams_form.addRow("Rozmiar batcha:", self.training_batch_size_spin)

# W metodzie _on_accept dodać wczytywanie tej wartości zamiast z parameter_rows:
batch_size = self.training_batch_size_spin.value()
self.task_config["config"]["training"]["batch_size"] = batch_size
self.task_config["config"]["optimization"]["batch_size"] = batch_size  # Duplikacja dla zgodności
B. Przeniesienie kontrolki num_workers z Optymalizacji do Parametrów Treningu
python# Poprawka 5: Przeniesienie parametru num_workers z Optymalizacji do Parametrów Treningu
# W metodzie _create_training_params_tab

self.training_num_workers_spin = QtWidgets.QSpinBox()
self.training_num_workers_spin.setRange(0, os.cpu_count() or 4)
self.training_num_workers_spin.setValue(min(4, os.cpu_count() or 4))
training_hyperparams_form.addRow("Liczba wątków (data loader):", self.training_num_workers_spin)

# W metodzie _on_accept dodać wczytywanie tej wartości:
num_workers = self.training_num_workers_spin.value()
self.task_config["config"]["training"]["num_workers"] = num_workers
self.task_config["config"]["optimization"]["num_workers"] = num_workers  # Duplikacja dla zgodności
C. Przeniesienie kontrolki mixed_precision z Optymalizacji do Parametrów Treningu
python# Poprawka 6: Przeniesienie parametru mixed_precision z Optymalizacji do Parametrów Treningu
# W metodzie _create_training_params_tab

self.training_mixed_precision_check = QtWidgets.QCheckBox("Użyj mieszanej precyzji (AMP)")
training_hyperparams_form.addRow(self.training_mixed_precision_check)

# W metodzie _on_accept dodać wczytywanie tej wartości:
use_mixed_precision = self.training_mixed_precision_check.isChecked()
self.task_config["config"]["training"]["use_mixed_precision"] = use_mixed_precision
self.task_config["config"]["optimization"]["use_mixed_precision"] = use_mixed_precision  # Duplikacja dla zgodności
3. Implementacja brakujących kontrolek UI
A. Dodanie brakującej kontrolki dla preprocessing.resize.mode
python# Poprawka 7: Dodanie kontrolki UI dla preprocessing.resize.mode
# W metodzie _create_preprocessing_tab, w sekcji resize_group

self.preprocess_resize_mode_combo = QtWidgets.QComboBox()
self.preprocess_resize_mode_combo.addItems(["bilinear", "bicubic", "nearest", "lanczos"])
resize_form.addRow("Tryb zmiany rozmiaru (interpolacja):", self.preprocess_resize_mode_combo)

# W metodzie _apply_profile, dodać:
self.preprocess_resize_mode_combo.setCurrentText(
    config.get("preprocessing", {}).get("resize", {}).get("mode", "bilinear")
)

# W metodzie _get_current_config_as_dict, zaktualizować sekcję preprocessing:
"preprocessing": {
    "resize": {
        "enabled": self.preprocess_resize_enabled_check.isChecked(),
        "size": [
            self.preprocess_resize_width_spin.value(),
            self.preprocess_resize_height_spin.value(),
        ],
        "mode": self.preprocess_resize_mode_combo.currentText(),
    },
    # pozostałe parametry...
}
B. Dodanie kontrolki dla training.gradient_clip
python# Poprawka 8: Dodanie kontrolki UI dla training.gradient_clip
# W metodzie _create_training_params_tab, dodać w sekcji training_hyperparams_form:

self.training_gradient_clip_value_spin = QtWidgets.QDoubleSpinBox()
self.training_gradient_clip_value_spin.setRange(0.0, 100.0)  # 0.0 oznacza brak clip'a
self.training_gradient_clip_value_spin.setDecimals(2)
self.training_gradient_clip_value_spin.setValue(0.0)  # Domyślnie wyłączone
self.training_gradient_clip_value_spin.setToolTip("Wartość przycinania gradientu (0.0 = wyłączone).")
training_hyperparams_form.addRow("Przycinanie gradientu (wartość):", self.training_gradient_clip_value_spin)

# W metodzie _apply_profile, dodać:
self.training_gradient_clip_value_spin.setValue(
    training_config.get("gradient_clip", 0.0)
)
4. Poprawki w mechanizmach wczytywania i zapisu
A. Poprawienie wczytywania parametrów stochastic_depth w _apply_profile
python# Poprawka 9: Poprawienie wczytywania parametrów stochastic_depth w _apply_profile
# W metodzie _apply_profile, w sekcji Regularyzacja:

# Zastąpić istniejący kod:
sd_conf = reg_config.get("stochastic_depth", {})
self.stochastic_depth_use_check.setChecked(sd_conf.get("use", False))
self.stochastic_depth_survival_prob_spin.setValue(sd_conf.get("survival_probability", 0.8))

# Dodać aktywację kontrolek zależnych:
self.stochastic_depth_survival_prob_spin.setEnabled(self.stochastic_depth_use_check.isChecked())
B. Poprawienie wczytywania parametrów SWA w _apply_profile
python# Poprawka 10: Poprawienie wczytywania parametrów SWA w _apply_profile
# W metodzie _apply_profile, w sekcji Regularyzacja:

# Zastąpić istniejący kod:
swa_conf = reg_config.get("swa", {})
self.use_swa_check.setChecked(swa_conf.get("use", False))
self.swa_start_epoch_spin.setValue(swa_conf.get("start_epoch", 10))
self.swa_lr_spin.setValue(swa_conf.get("lr_swa", 5e-5))

# Dodać aktywację kontrolek zależnych:
self.swa_start_epoch_spin.setEnabled(self.use_swa_check.isChecked())
self.swa_lr_spin.setEnabled(self.use_swa_check.isChecked())
C. Dodanie zapisu do profilu dla parametrów zamrażania warstw
python# Poprawka 11: Dodanie zapisu do profilu dla parametrów zamrażania warstw
# W metodzie _save_profile, w sekcji "training":

profile_data["config"]["training"]["freeze_base_model"] = self.training_freeze_base_model_check.isChecked()
profile_data["config"]["training"]["unfreeze_layers"] = self._get_unfreeze_layers_value(self.training_unfreeze_layers_edit.text())
profile_data["config"]["training"]["unfreeze_strategy"] = self._get_unfreeze_strategy_value(self.training_unfreeze_strategy_combo.currentText())
profile_data["config"]["training"]["unfreeze_after_epochs"] = self.training_unfreeze_after_epochs_spin.value()
profile_data["config"]["training"]["frozen_lr"] = self.training_frozen_lr_spin.value()
profile_data["config"]["training"]["unfrozen_lr"] = self.training_unfrozen_lr_spin.value()
D. Dodanie wczytywania parametrów AutoAugment i RandAugment
python# Poprawka 12: Dodanie wczytywania parametrów AutoAugment i RandAugment w _apply_profile
# W metodzie _apply_profile, w sekcji Augmentacja:

# Dodać po istniejącym kodzie dla cutmix:
# AutoAugment
self.autoaugment_check.setChecked(aug_config.get("autoaugment", {}).get("use", False))
self.autoaugment_policy_combo.setCurrentText(aug_config.get("autoaugment", {}).get("policy", "imagenet"))
self.autoaugment_policy_combo.setEnabled(self.autoaugment_check.isChecked())

# RandAugment
self.randaugment_check.setChecked(aug_config.get("randaugment", {}).get("use", False))
self.randaugment_n_spin.setValue(aug_config.get("randaugment", {}).get("n", 2))
self.randaugment_m_spin.setValue(aug_config.get("randaugment", {}).get("m", 9))
self.randaugment_n_spin.setEnabled(self.randaugment_check.isChecked())
self.randaugment_m_spin.setEnabled(self.randaugment_check.isChecked())
5. Implementacja brakujących parametrów monitorowania
A. Dodanie kontrolki metrics.auc
python# Poprawka 13: Dodanie kontrolki metrics.auc w _create_monitoring_tab
# W metodzie _create_monitoring_tab, w sekcji metrics_form:

self.metrics_auc_check = QtWidgets.QCheckBox("AUC-ROC")
metrics_form.addRow(self.metrics_auc_check)

# W metodzie _apply_profile, dodać:
self.metrics_auc_check.setChecked(met_conf.get("auc", False))

# W metodzie _get_current_config_as_dict, zaktualizować sekcję metrics:
"metrics": {
    # pozostałe parametry...
    "auc": self.metrics_auc_check.isChecked(),
}
B. Dodanie kontrolek dla monitorowania GPU i pamięci
python# Poprawka 14: Dodanie kontrolek dla monitorowania GPU i pamięci
# W metodzie _create_monitoring_tab, w sekcji metrics_form:

self.metrics_gpu_utilization_check = QtWidgets.QCheckBox("Monitoruj wykorzystanie GPU")
metrics_form.addRow(self.metrics_gpu_utilization_check)
self.metrics_memory_usage_check = QtWidgets.QCheckBox("Monitoruj zużycie pamięci (RAM/VRAM)")
metrics_form.addRow(self.metrics_memory_usage_check)

# W metodzie _apply_profile, dodać:
self.metrics_gpu_utilization_check.setChecked(met_conf.get("gpu_utilization", False))
self.metrics_memory_usage_check.setChecked(met_conf.get("memory_usage", False))

# W metodzie _get_current_config_as_dict, zaktualizować sekcję metrics:
"metrics": {
    # pozostałe parametry...
    "gpu_utilization": self.metrics_gpu_utilization_check.isChecked(),
    "memory_usage": self.metrics_memory_usage_check.isChecked(),
}
6. Implementacja brakujących parametrów zaawansowanych
A. Dodanie kontrolki dla parametru seed
python# Poprawka 15: Dodanie kontrolki dla parametru seed
# W metodzie _create_advanced_tab, na początku layout:

self.advanced_seed_spin = QtWidgets.QSpinBox()
self.advanced_seed_spin.setRange(0, 999999)
self.advanced_seed_spin.setValue(42)  # 0 dla losowego
layout.addRow("Ziarno losowości (0 dla losowego):", self.advanced_seed_spin)

# W metodzie _apply_profile, dodać:
self.advanced_seed_spin.setValue(adv_config.get("seed", 42))

# W metodzie _get_current_config_as_dict, zaktualizować sekcję advanced:
"advanced": {
    "seed": self.advanced_seed_spin.value(),
    # pozostałe parametry...
}
B. Dodanie kontrolki dla parametru deterministic
python# Poprawka 16: Dodanie kontrolki dla parametru deterministic
# W metodzie _create_advanced_tab, po parametrze seed:

self.advanced_deterministic_check = QtWidgets.QCheckBox("Użyj deterministycznych operacji (może spowolnić)")
layout.addRow(self.advanced_deterministic_check)

# W metodzie _apply_profile, dodać:
self.advanced_deterministic_check.setChecked(adv_config.get("deterministic", False))

# W metodzie _get_current_config_as_dict, zaktualizować sekcję advanced:
"advanced": {
    # pozostałe parametry...
    "deterministic": self.advanced_deterministic_check.isChecked(),
    # pozostałe parametry...
}
7. Ujednolicenie struktur JSON między profilem a zadaniem
A. Ujednolicenie formatu zapisywania scheduler
python# Poprawka 17: Ujednolicenie formatu zapisywania scheduler
# W metodzie _get_current_config_as_dict, zaktualizować sekcję scheduler:

scheduler_type = self._get_scheduler_value(self.training_scheduler_type_combo.currentText())
if scheduler_type == "None":
    "scheduler": "None",  # Prosty string dla 'None'
else:
    "scheduler": {
        "type": scheduler_type,
        "T_0": self.training_scheduler_t0_spin.value(),
        "T_mult": self.training_scheduler_tmult_spin.value(),
        "eta_min": self.training_scheduler_eta_min_spin.value(),
    },
B. Ujednolicenie formatu parametrów preprocessing
python# Poprawka 18: Ujednolicenie formatu parametrów preprocessing
# W metodzie _get_current_config_as_dict, zaktualizować sekcję preprocessing:

"preprocessing": {
    "resize": {
        "enabled": self.preprocess_resize_enabled_check.isChecked(),
        "size": [
            self.preprocess_resize_width_spin.value(),
            self.preprocess_resize_height_spin.value(),
        ],
        "mode": self.preprocess_resize_mode_combo.currentText(),
    },
    "normalize": {
        "enabled": self.preprocess_normalize_enabled_check.isChecked(),
        "mean": [
            self.preprocess_normalize_mean_r_spin.value(),
            self.preprocess_normalize_mean_g_spin.value(),
            self.preprocess_normalize_mean_b_spin.value(),
        ],
        "std": [
            self.preprocess_normalize_std_r_spin.value(),
            self.preprocess_normalize_std_g_spin.value(),
            self.preprocess_normalize_std_b_spin.value(),
        ],
    },
    # pozostałe parametry...
    # Dodać brakujące parametry z profilu dla spójności
    "cache_dataset": self.preprocess_cache_dataset_check.isChecked(),
},
8. Refaktoryzacja i optymalizacje kodu
A. Dodanie metody do aktywacji/deaktywacji wielu kontrolek
python# Poprawka 19: Dodanie pomocniczej metody do aktywacji/deaktywacji wielu kontrolek
# Dodać nową metodę:

def _toggle_controls_by_names(self, enabled, control_names):
    """
    Aktywuje lub deaktywuje kontrolki o podanych nazwach.
    
    Args:
        enabled (bool): Czy kontrolki mają być aktywne
        control_names (list): Lista nazw atrybutów kontrolek
    """
    for ctrl_name in control_names:
        if hasattr(self, ctrl_name):
            getattr(self, ctrl_name).setEnabled(enabled)

# Zmodyfikować istniejące metody toggle, np. z:
def _toggle_basic_aug_controls(self, state):
    enabled = bool(state)
    controls = [
        "aug_basic_rotation_spin",
        "aug_basic_brightness_spin",
        "aug_basic_contrast_spin",
        # ...
    ]
    for ctrl_name in controls:
        if hasattr(self, ctrl_name):
            getattr(self, ctrl_name).setEnabled(enabled)

# Na:
def _toggle_basic_aug_controls(self, state):
    self._toggle_controls_by_names(bool(state), [
        "aug_basic_rotation_spin",
        "aug_basic_brightness_spin",
        "aug_basic_contrast_spin",
        # ...
    ])
B. Refaktoryzacja metody _update_dependent_controls
python# Poprawka 20: Refaktoryzacja metody _update_dependent_controls
# Przerobić metodę, aby używała słownika kontrolek i ich callbacków:

def _update_dependent_controls(self):
    """Wywołuje wszystkie funkcje _toggle_* aby zaktualizować stan kontrolek."""
    toggle_pairs = [
        (self.training_unfreeze_strategy_combo.currentText(), self._toggle_unfreeze_after_epochs_spin),
        (self.use_early_stopping_check.isChecked(), self._toggle_early_stopping_controls),
        (self.reduce_lr_enabled_check.isChecked(), self._toggle_reduce_lr_controls),
        (self.stochastic_depth_use_check.isChecked(), self._toggle_stochastic_depth_controls),
        (self.use_swa_check.isChecked(), self._toggle_swa_controls),
        (self.basic_aug_check.isChecked(), self._toggle_basic_aug_controls),
        # Dodać pozostałe kontrolki...
    ]
    
    for value, callback in toggle_pairs:
        callback(value)
        
    # Proste przełączniki które nie mają dedykowanych metod toggle
    simple_toggles = [
        (self.mixup_check, self.mixup_alpha_spin),
        (self.cutmix_check, self.cutmix_alpha_spin),
        (self.autoaugment_check, self.autoaugment_policy_combo),
        # Dodać pozostałe...
    ]
    
    for check, widget in simple_toggles:
        if hasattr(self, check.objectName()) and hasattr(self, widget.objectName()):
            widget.setEnabled(check.isChecked())