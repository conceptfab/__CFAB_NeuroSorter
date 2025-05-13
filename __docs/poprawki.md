Zdublowane kontrolki:

Rozmiar batch'a

W zakładce "Parametry" (_create_fine_tuning_params_tab): self.batch_size_spin
W zakładce "Optymalizacja treningu" (_create_optimization_tab): kontrolka dla "recommended_batch_size"


Liczba workerów

W zakładce "Parametry" (_create_fine_tuning_params_tab): self.num_workers_spin
W zakładce "Optymalizacja treningu" (_create_optimization_tab): kontrolka dla "recommended_workers"


Mixed precision

W zakładce "Parametry" (_create_fine_tuning_params_tab): self.mixed_precision_check
W zakładce "Optymalizacja treningu" (_create_optimization_tab): kontrolka dla "use_mixed_precision"


Gradient accumulation steps

W zakładce "Parametry" (_create_fine_tuning_params_tab): self.grad_accum_steps_spin
W zakładce "Optymalizacja treningu" (_create_optimization_tab): kontrolka dla "gradient_accumulation_steps"



Proponowane zmiany w kodzie:
Zgodnie z wymaganiem, kontrolki z zakładki "Optymalizacja treningu" powinny pozostać, a duplikaty z innych zakładek należy usunąć. Poniżej przedstawiam zmiany, które należy wprowadzić w metodzie _create_fine_tuning_params_tab:
pythondef _create_fine_tuning_params_tab(self) -> QtWidgets.QWidget:
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

    # USUNIĘTO: Rozmiar batcha (przeniesiony do zakładki Optymalizacja treningu)

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

    # USUNIĘTO: Liczba workerów (przeniesiony do zakładki Optymalizacja treningu)

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

    # USUNIĘTO: Gradient accumulation steps (przeniesiony do zakładki Optymalizacja treningu)

    # USUNIĘTO: Mixed precision (przeniesiony do zakładki Optymalizacja treningu)

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
Dodatkowo, konieczne jest zmodyfikowanie metody _on_accept, aby korzystała z wartości z odpowiednich zakładek:
pythondef _on_accept(self):
    # ... istniejący kod ...
    
    # Używamy wartości z zakładki Optymalizacja treningu zamiast z zakładki Parametry
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
    
    # Aktualizacja konfiguracji z wartościami z Optymalizacji treningu
    if "recommended_batch_size" in optimization_config:
        config["training"]["batch_size"] = optimization_config["recommended_batch_size"]
    if "recommended_workers" in optimization_config:
        config["training"]["num_workers"] = optimization_config["recommended_workers"]
    if "use_mixed_precision" in optimization_config:
        config["training"]["mixed_precision"] = optimization_config["use_mixed_precision"]
    if "gradient_accumulation_steps" in optimization_config:
        config["training"]["gradient_accumulation_steps"] = optimization_config["gradient_accumulation_steps"]
    
    # Dodajemy sekcję optymalizacji do głównej konfiguracji
    config["optimization"] = optimization_config
    
    # ... pozostały kod ...
Te zmiany usuwają zdublowane kontrolki z zakładki "Parametry", pozostawiając ich odpowiedniki w zakładce "Optymalizacja treningu", zgodnie z wymaganiem.