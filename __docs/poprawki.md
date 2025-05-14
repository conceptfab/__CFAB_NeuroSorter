Poprawki dla interfejsu użytkownika (UI)
Kontrolki dla Early Stopping istniejące w kodzie
W pliku fine_tuning_task_config_dialog.py już istnieją kontrolki UI dla early stopping w metodzie _create_monitoring_tab():
python# Early stopping
self.patience_spin = QtWidgets.QSpinBox()
self.patience_spin.setRange(1, 100)
self.patience_spin.setValue(10)
self.min_delta_spin = QtWidgets.QDoubleSpinBox()
self.min_delta_spin.setRange(0.0, 1.0)
self.min_delta_spin.setValue(0.001)
self.min_delta_spin.setDecimals(4)
self.monitor_combo = QtWidgets.QComboBox()
self.monitor_combo.addItems(
    [
        "val_loss",
        "val_accuracy",
        "val_f1",
        "val_precision",
        "val_recall",
    ]
)
Podobnie w training_task_config_dialog.py już istnieją te elementy.
Kluczowa poprawka - upewnienie się, że dane są zbierane z UI
Główny problem polega na tym, że te parametry nie są prawidłowo zbierane podczas tworzenia konfiguracji zadania. Oto konkretne zmiany potrzebne, aby parametry early stopping były w pełni dostępne w UI:
1. Modyfikacja w fine_tuning_task_config_dialog.py - metoda _on_accept()
pythondef _on_accept(self):
    # ... istniejący kod ...
    
    config = {
        # ... istniejące sekcje konfiguracji ...
        
        # Dodaj sekcję monitoring, jeśli nie istnieje
        "monitoring": {
            "metrics": {
                "accuracy": self.accuracy_check.isChecked(),
                "precision": self.precision_check.isChecked(),
                "recall": self.recall_check.isChecked(),
                "f1": self.f1_check.isChecked(),
                "topk": self.topk_check.isChecked(),
                "confusion_matrix": self.confusion_matrix_check.isChecked(),
                "auc": self.auc_check.isChecked()
            },
            "early_stopping": {
                "patience": self.patience_spin.value(),
                "min_delta": self.min_delta_spin.value(),
                "monitor": self.monitor_combo.currentText()
            },
            "checkpointing": {
                "best_only": self.best_only_check.isChecked(),
                "save_frequency": self.save_freq_spin.value(),
                "metric": self.checkpoint_metric_combo.currentText()
            },
            "logging": {
                "use_tensorboard": self.use_tensorboard_check.isChecked(),
                "use_wandb": self.use_wandb_check.isChecked(),
                "save_to_csv": self.use_csv_check.isChecked(),
                "logging_freq": self.log_freq_combo.currentText()
            }
        },
        
        # ... inne sekcje ...
    }
    
    self.task_config = {
        "name": self.name_edit.text().strip(),
        "type": "fine_tuning",
        "status": "Nowy",
        "config": config,
        # ... reszta kodu ...
    }
2. Podobna poprawka dla training_task_config_dialog.py
pythondef _on_accept(self):
    # ... istniejący kod ...
    
    self.task_config = {
        "name": task_name,
        "type": "training",
        "status": "Nowy",
        "priority": 0,
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            # ... istniejące sekcje ...
            
            "monitoring": {
                "metrics": {
                    "accuracy": self.accuracy_check.isChecked(),
                    "precision": self.precision_check.isChecked(),
                    "recall": self.recall_check.isChecked(),
                    "f1": self.f1_check.isChecked(),
                    "topk": self.topk_check.isChecked(),
                    "confusion_matrix": self.confusion_matrix_check.isChecked()
                },
                "early_stopping": {
                    "patience": self.patience_spin.value(),
                    "min_delta": self.min_delta_spin.value(),
                    "monitor": self.monitor_combo.currentText()
                },
                "checkpointing": {
                    "best_only": self.best_only_check.isChecked(),
                    "save_frequency": self.save_freq_spin.value(),
                    "metric": self.checkpoint_metric_combo.currentText()
                }
            },
            
            # ... inne sekcje ...
        }
    }
3. Opcjonalnie: Dodanie checkboxa włączającego/wyłączającego Early Stopping
Jeśli chcesz mieć możliwość całkowitego wyłączenia early stopping, możesz dodać checkbox:
python# W metodzie _create_monitoring_tab():

# Early stopping
early_stop_group = QtWidgets.QGroupBox("Early stopping")
early_stop_layout = QtWidgets.QFormLayout()

# Dodaj checkbox do włączania/wyłączania early stopping
self.use_early_stopping_check = QtWidgets.QCheckBox("Włącz early stopping")
self.use_early_stopping_check.setChecked(True)
self.use_early_stopping_check.stateChanged.connect(self._toggle_early_stopping_controls)
early_stop_layout.addRow("", self.use_early_stopping_check)

# ... istniejące kontrolki ...

# Metoda do przełączania dostępności kontrolek early stopping
def _toggle_early_stopping_controls(self, state):
    enabled = bool(state)
    self.patience_spin.setEnabled(enabled)
    self.min_delta_spin.setEnabled(enabled)
    self.monitor_combo.setEnabled(enabled)
4. Dodanie tooltipów wyjaśniających działanie early stopping
python# W metodzie _create_monitoring_tab():

self.patience_spin.setToolTip("Liczba epok bez poprawy, po której trening zostanie zatrzymany")
self.min_delta_spin.setToolTip("Minimalna zmiana metryki, uznawana za poprawę")
self.monitor_combo.setToolTip("Metryka używana do monitorowania poprawy")
5. Wizualizacja stanu early stopping w interfejsie podczas treningu
Do okna monitorującego postęp treningu można dodać wskaźnik pokazujący aktualny stan early stopping:
python# W oknie monitorującym postęp:

# Tworzenie kontrolek wskaźnika early stopping
self.early_stopping_group = QtWidgets.QGroupBox("Early Stopping")
self.early_stopping_layout = QtWidgets.QVBoxLayout()
self.early_stopping_label = QtWidgets.QLabel("Czekam na dane...")
self.early_stopping_progress = QtWidgets.QProgressBar()
self.early_stopping_progress.setRange(0, 10)  # Domyślny zakres, aktualizowany później
self.early_stopping_progress.setValue(0)
self.early_stopping_layout.addWidget(self.early_stopping_label)
self.early_stopping_layout.addWidget(self.early_stopping_progress)
self.early_stopping_group.setLayout(self.early_stopping_layout)
self.main_layout.addWidget(self.early_stopping_group)

# W metodzie aktualizującej interfejs podczas treningu:
def update_ui(self, epoch, epochs, train_loss, train_acc, val_loss, val_acc, patience_counter=0, patience_max=10):
    # ... istniejący kod aktualizacji wykresu i innych elementów ...
    
    # Aktualizacja wskaźnika early stopping
    self.early_stopping_progress.setMaximum(patience_max)
    self.early_stopping_progress.setValue(patience_counter)
    
    if patience_counter == 0:
        self.early_stopping_label.setText("Early stopping: Poprawa w tej epoce")
        self.early_stopping_progress.setStyleSheet("QProgressBar::chunk { background-color: green; }")
    else:
        self.early_stopping_label.setText(f"Early stopping: {patience_counter}/{patience_max}")
        
        # Zmień kolor w zależności od stanu
        if patience_counter >= patience_max - 1:
            self.early_stopping_progress.setStyleSheet("QProgressBar::chunk { background-color: red; }")
        elif patience_counter >= patience_max // 2:
            self.early_stopping_progress.setStyleSheet("QProgressBar::chunk { background-color: orange; }")
        else:
            self.early_stopping_progress.setStyleSheet("QProgressBar::chunk { background-color: yellow; }")
Podsumowanie zmian w UI
Z powyższymi zmianami parametry early stopping będą:

Widoczne w interfejsie użytkownika
Odpowiednio dodawane do konfiguracji zadania
Przekazywane do kodu wykonującego trening
Wizualizowane podczas treningu

Większość elementów UI dla early stopping już istnieje w kodzie, główna poprawka polega na upewnieniu się, że są one prawidłowo zbierane podczas tworzenia konfiguracji zadania i przekazywane do funkcji trenujących.tak