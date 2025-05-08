Zmiany w pliku app/gui/dialogs/fine_tuning_task_config_dialog.py
1. Poprawa obsługi zmiennej task_config
pythondef _on_accept(self):
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

        self.task_config = {  # Zmiana: Użycie self.task_config zamiast self.config
            "name": task_name,
            "type": "Fine-tuning",
            "status": "Nowy",
            "priority": 0,
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                # Pozostała zawartość konfiguracji...
            }
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
2. Brakujące inicjalizacje elementów UI
Dodanie inicjalizacji brakujących elementów w metodzie _create_monitoring_tab():
pythondef _create_monitoring_tab(self):
    # Dodanie brakujących elementów UI:
    self.roc_auc_check = QtWidgets.QCheckBox("ROC AUC")
    self.pr_auc_check = QtWidgets.QCheckBox("PR AUC")
    self.top_k_check = QtWidgets.QCheckBox("Top-K Accuracy")
    
    # Dodanie tych elementów do layoutu:
    metrics_layout.addWidget(self.roc_auc_check)
    metrics_layout.addWidget(self.pr_auc_check)
    metrics_layout.addWidget(self.top_k_check)
    
    # Pozostała część kodu...
3. Poprawa w metodzie _create_data_model_tab()
Dodanie brakującego parametru self.profile_hardware_required:
pythondef _create_data_model_tab(self):
    # W sekcji "Informacje o profilu":
    self.profile_hardware_required = QtWidgets.QTextEdit()
    self.profile_hardware_required.setReadOnly(True)
    self.profile_hardware_required.setMaximumHeight(60)
    info_layout.addRow("Wymagany sprzęt:", self.profile_hardware_required)
    
    # Pozostała część kodu...
4. Poprawa w metodzie _on_profile_selected()
Dodanie uzupełnienia pola profile_hardware_required:
pythondef _on_profile_selected(self, current, previous):
    """Obsługa wyboru profilu."""
    try:
        if not current:
            return

        profile_path = self.profiles_dir / f"{current.text()}.json"
        with open(profile_path, "r", encoding="utf-8") as f:
            profile_data = json.load(f)

        self.current_profile = profile_data
        self.profile_info.setText(profile_data.get("info", ""))
        self.profile_description.setText(profile_data.get("description", ""))
        self.profile_data_required.setText(profile_data.get("data_required", ""))
        self.profile_hardware_required.setText(profile_data.get("hardware_required", ""))

    except Exception as e:
        msg = "Błąd podczas ładowania profilu"
        self.logger.error(f"{msg}: {str(e)}", exc_info=True)
5. Dodanie brakujących pól formularza w metodzie _apply_profile()
pythondef _apply_profile(self):
    # Dodanie inicjalizacji brakujących pól
    if not hasattr(self, 'roc_auc_check'):
        self.roc_auc_check = QtWidgets.QCheckBox("ROC AUC")
    if not hasattr(self, 'pr_auc_check'):
        self.pr_auc_check = QtWidgets.QCheckBox("PR AUC")
    if not hasattr(self, 'top_k_check'):
        self.top_k_check = QtWidgets.QCheckBox("Top-K Accuracy")
    
    # Pozostała część kodu...
6. Poprawka dla metody _select_model_dir()
Dodanie brakującego pola:
pythondef _create_monitoring_tab(self):
    # W sekcji "Katalog zapisu i logi":
    self.model_dir_edit = QtWidgets.QLineEdit()
    model_dir_btn = QtWidgets.QPushButton("Przeglądaj...")
    model_dir_btn.clicked.connect(self._select_model_dir)
    model_dir_layout = QtWidgets.QHBoxLayout()
    model_dir_layout.addWidget(self.model_dir_edit)
    model_dir_layout.addWidget(model_dir_btn)
    
    # Pozostała część kodu...
7. Synchronizacja struktury danych zadań
Synchronizacja struktury JSON zapisywanej w _on_accept():
pythondef _on_accept(self):
    # Upewnij się, że struktura danych jest zgodna z tą z training_task_config_dialog.py
    self.task_config = {
        "name": task_name,
        "type": "Fine-tuning",  # Zachowanie typu specyficznego dla fine-tuningu
        "status": "Nowy",
        "priority": 0,
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "train_dir": str(self.train_dir_edit.text()),
            "data_dir": str(self.train_dir_edit.text()),
            "val_dir": str(self.val_dir_edit.text()),
            # Pozostała część konfiguracji...
        }
    }
8. Dodanie parametrów specyficznych dla fine-tuningu w strukturze zadania
pythondef _on_accept(self):
    # W sekcji "config" dodanie parametrów specyficznych dla fine-tuningu:
    "peft": {
        "technique": self.peft_technique.currentText() if hasattr(self, "peft_technique") else "none",
        "lora": {
            "rank": self.lora_rank.value() if hasattr(self, "lora_rank") else 8,
            "alpha": self.lora_alpha.value() if hasattr(self, "lora_alpha") else 16,
            "dropout": self.lora_dropout.value() if hasattr(self, "lora_dropout") else 0.1,
            "target_modules": self.lora_target_modules.text().split(",") if hasattr(self, "lora_target_modules") else ["query", "key", "value"]
        },
        "adapter": {
            "hidden_size": self.adapter_hidden_size.value() if hasattr(self, "adapter_hidden_size") else 64,
            "adapter_type": self.adapter_type.currentText() if hasattr(self, "adapter_type") else "houlsby",
            "adapter_activation": self.adapter_activation.currentText() if hasattr(self, "adapter_activation") else "relu"
        },
        "prompt_tuning": {
            "num_virtual_tokens": self.num_virtual_tokens.value() if hasattr(self, "num_virtual_tokens") else 20,
            "prompt_init": self.prompt_init.currentText() if hasattr(self, "prompt_init") else "random"
        }
    }
9. Dodanie metody _apply_metrics_config() jeśli brakuje
pythondef _apply_metrics_config(self, metrics):
    """Stosuje konfigurację metryk."""
    # Upewnij się, że wszystkie pola formularza istnieją
    if not hasattr(self, 'accuracy_check'):
        self.accuracy_check = QtWidgets.QCheckBox("Accuracy")
    if not hasattr(self, 'precision_check'):
        self.precision_check = QtWidgets.QCheckBox("Precision")
    if not hasattr(self, 'recall_check'):
        self.recall_check = QtWidgets.QCheckBox("Recall")
    if not hasattr(self, 'f1_check'):
        self.f1_check = QtWidgets.QCheckBox("F1")
    if not hasattr(self, 'confusion_matrix_check'):
        self.confusion_matrix_check = QtWidgets.QCheckBox("Confusion Matrix")
    if not hasattr(self, 'roc_auc_check'):
        self.roc_auc_check = QtWidgets.QCheckBox("ROC AUC")
    if not hasattr(self, 'pr_auc_check'):
        self.pr_auc_check = QtWidgets.QCheckBox("PR AUC")
    if not hasattr(self, 'top_k_check'):
        self.top_k_check = QtWidgets.QCheckBox("Top-K Accuracy")
    
    self.accuracy_check.setChecked("accuracy" in metrics)
    self.precision_check.setChecked("precision" in metrics)
    self.recall_check.setChecked("recall" in metrics)
    self.f1_check.setChecked("f1" in metrics)
    self.confusion_matrix_check.setChecked("confusion_matrix" in metrics)
    self.roc_auc_check.setChecked("roc_auc" in metrics)
    self.pr_auc_check.setChecked("pr_auc" in metrics)
    self.top_k_check.setChecked("top_k_accuracy" in metrics)
10. Dodatkowo stworzenie zakładki PEFT dla technik fine-tuningu
pythondef _create_peft_tab(self):
    """Tworzenie zakładki Parameter-Efficient Fine-Tuning (PEFT)."""
    try:
        self.logger.debug("Tworzenie zakładki PEFT")
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        
        # Wybór techniki PEFT
        peft_group = QtWidgets.QGroupBox("Technika PEFT")
        peft_layout = QtWidgets.QFormLayout()
        
        self.peft_technique = QtWidgets.QComboBox()
        self.peft_technique.addItems(["none", "lora", "prefix_tuning", "adapter", "prompt_tuning"])
        self.peft_technique.currentTextChanged.connect(self._on_peft_technique_changed)
        
        peft_layout.addRow("Technika:", self.peft_technique)
        peft_group.setLayout(peft_layout)
        
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
        
        layout.addWidget(peft_group)
        layout.addWidget(lora_group)
        layout.addWidget(adapter_group)
        layout.addWidget(prompt_group)
        
        return tab
    
    except Exception as e:
        msg = "Błąd podczas tworzenia zakładki PEFT"
        self.logger.error(f"{msg}: {str(e)}", exc_info=True)
        raise
        
def _on_peft_technique_changed(self, technique):
    """Obsługa zmiany techniki PEFT."""
    # Włączanie/wyłączanie odpowiednich grup w zależności od wybranej techniki
    # Ta metoda może być zaimplementowana później
    pass
11. Dodanie zakładki PEFT do głównego UI
pythondef _init_ui(self):
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

        # 6. Zakładka: PEFT (Nowa zakładka)
        tab = self._create_peft_tab()
        self.tabs.addTab(tab, "PEFT")

        # 7. Zakładka: Monitorowanie i Zapis
        tab = self._create_monitoring_tab()
        self.tabs.addTab(tab, "Monitorowanie")

        # 8. Zakładka: Zaawansowane
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
Podsumowanie wprowadzonych zmian:

Poprawiono nazwę zmiennej task_config (zamiast config) w metodzie _on_accept(), aby była zgodna z tą używaną w training_task_config_dialog.py.
Dodano brakujące pola formularza, szczególnie w sekcji monitorowania i metrykach.
Dodano pole profile_hardware_required i jego obsługę w metodach związanych z profilem.
Naprawiono inicjalizację wszystkich elementów UI, które są używane później w kodzie.
Dodano zakładkę PEFT specyficzną dla fine-tuningu, która nie występuje w training_task_config_dialog.py.
Poprawiono strukturę zadania zapisywanego w _on_accept(), aby była zgodna ze strukturą w training_task_config_dialog.py.
Dodano obsługę parametrów specyficznych dla fine-tuningu w strukturze zadania.
Implementacja _apply_metrics_config() jeśli brakuje.

Te zmiany powinny sprawić, że fine_tuning_task_config_dialog.py będzie działał podobnie do training_task_config_dialog.py, zachowując jednocześnie funkcjonalność specyficzną dla fine-tuningu.