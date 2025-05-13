Dodanie zakładki Optymalizacja treningu
Na podstawie analizy dostarczonych plików, dodam zakładkę "Optymalizacja treningu" do obu dialogów konfiguracyjnych: TrainingTaskConfigDialog oraz FineTuningTaskConfigDialog. Zakładka ta będzie zawierać parametry, które mogą być wczytywane z profilu sprzętowego, z możliwością wyboru między wartościami domyślnymi a tymi z profilu.
Zmiany w pliku fine_tuning_task_config_dialog.py:
pythondef _init_ui(self):
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
Dodajmy teraz implementację nowej metody _create_optimization_tab():
pythondef _create_optimization_tab(self) -> QtWidgets.QWidget:
    """Tworzy zakładkę z parametrami optymalizacji treningu."""
    try:
        self.logger.debug("Tworzenie zakładki optymalizacji treningu")
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        
        # Informacja o profilu sprzętowym
        if self.hardware_profile:
            hardware_info = QtWidgets.QLabel(f"Używany profil sprzętowy: {self.hardware_profile.get('device_name', 'Nieznany')}")
        else:
            hardware_info = QtWidgets.QLabel("Brak załadowanego profilu sprzętowego")
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
            step=1
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
            step=1
        )
        form_layout.addRow("Liczba workerów:", num_workers_layout)
        
        # 3. Mixed precision
        mixed_precision_layout = self._create_parameter_row(
            name="Mixed precision", 
            param_key="use_mixed_precision",
            default_value=True,
            widget_type="checkbox"
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
            step=1
        )
        form_layout.addRow("Prefetch factor:", prefetch_layout)
        
        # 5. Pin memory
        pin_memory_layout = self._create_parameter_row(
            name="Pin memory", 
            param_key="pin_memory",
            default_value=True,
            widget_type="checkbox"
        )
        form_layout.addRow("Pin memory:", pin_memory_layout)
        
        # 6. Persistent workers
        persistent_workers_layout = self._create_parameter_row(
            name="Persistent workers", 
            param_key="persistent_workers",
            default_value=False,
            widget_type="checkbox"
        )
        form_layout.addRow("Persistent workers:", persistent_workers_layout)
        
        # 7. CUDA streaming
        cuda_stream_layout = self._create_parameter_row(
            name="CUDA streaming", 
            param_key="cuda_streaming",
            default_value=True,
            widget_type="checkbox"
        )
        form_layout.addRow("CUDA streaming:", cuda_stream_layout)
        
        # 8. Benchmark CUDNN
        benchmark_cudnn_layout = self._create_parameter_row(
            name="Benchmark CUDNN", 
            param_key="benchmark_cudnn",
            default_value=True,
            widget_type="checkbox"
        )
        form_layout.addRow("Benchmark CUDNN:", benchmark_cudnn_layout)
        
        # 9. Garbage collector
        gc_layout = self._create_parameter_row(
            name="Wyłącz garbage collector", 
            param_key="disable_gc",
            default_value=False,
            widget_type="checkbox"
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
            step=1
        )
        form_layout.addRow("Gradient accumulation steps:", grad_accum_layout)
        
        # 11. Channels last memory format
        channels_last_layout = self._create_parameter_row(
            name="Channels last memory format", 
            param_key="channels_last",
            default_value=False,
            widget_type="checkbox"
        )
        form_layout.addRow("Channels last:", channels_last_layout)
        
        # Dodanie całego layoutu do zakładki
        layout.addLayout(form_layout)
        
        # Dodanie przycisku do załadowania wszystkich optymalnych ustawień
        load_all_btn = QtWidgets.QPushButton("Zastosuj wszystkie optymalne ustawienia z profilu sprzętowego")
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

def _create_parameter_row(self, name, param_key, default_value, widget_type, min_val=None, max_val=None, step=None):
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
        'param_key': param_key,
        'profile_radio': profile_radio,
        'hardware_radio': hardware_radio,
        'value_widget': value_widget,
        'hw_value_label': hw_value_label,
        'hw_value': hw_value
    }
    
    # Dodanie do listy parametrów
    if not hasattr(self, 'optimization_params'):
        self.optimization_params = []
    self.optimization_params.append(row_widgets)
    
    # Obsługa zmiany źródła wartości
    def on_source_changed(id):
        if id == 1:  # Profil
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
    
    source_group.idClicked.connect(on_source_changed)
    
    return layout

def _apply_all_hardware_optimizations(self):
    """Zastosowuje wszystkie optymalne ustawienia z profilu sprzętowego."""
    if not hasattr(self, 'optimization_params') or not self.hardware_profile:
        QtWidgets.QMessageBox.warning(
            self, 
            "Ostrzeżenie", 
            "Brak dostępnego profilu sprzętowego lub parametrów do zastosowania."
        )
        return
    
    count = 0
    for param in self.optimization_params:
        param_key = param['param_key']
        if param_key in self.hardware_profile:
            param['hardware_radio'].setChecked(True)
            hw_value = self.hardware_profile[param_key]
            value_widget = param['value_widget']
            value_widget.setEnabled(False)
            
            if isinstance(value_widget, QtWidgets.QSpinBox) or isinstance(value_widget, QtWidgets.QDoubleSpinBox):
                value_widget.setValue(hw_value)
            elif isinstance(value_widget, QtWidgets.QCheckBox):
                value_widget.setChecked(hw_value)
            else:
                value_widget.setText(str(hw_value))
            
            count += 1
    
    QtWidgets.QMessageBox.information(
        self, 
        "Sukces", 
        f"Zastosowano {count} optymalnych ustawień z profilu sprzętowego."
    )
Teraz dodajmy podobne zmiany do metody _on_accept() w pliku fine_tuning_task_config_dialog.py, aby uwzględnić nowe parametry optymalizacji:
pythondef _on_accept(self):
    """Obsługa zatwierdzenia konfiguracji."""
    try:
        # Istniejący kod...
        
        # Dodajemy sekcję optymalizacji do konfiguracji
        optimization_config = {}
        
        if hasattr(self, 'optimization_params'):
            for param in self.optimization_params:
                param_key = param['param_key']
                hardware_radio = param['hardware_radio']
                value_widget = param['value_widget']
                
                # Pobieranie wartości w zależności od typu widgetu
                if hardware_radio.isChecked() and param['hw_value'] is not None:
                    param_value = param['hw_value']
                else:
                    if isinstance(value_widget, QtWidgets.QSpinBox) or isinstance(value_widget, QtWidgets.QDoubleSpinBox):
                        param_value = value_widget.value()
                    elif isinstance(value_widget, QtWidgets.QCheckBox):
                        param_value = value_widget.isChecked()
                    else:
                        param_value = value_widget.text()
                
                optimization_config[param_key] = param_value
        
        # Dodajemy sekcję optymalizacji do głównej konfiguracji
        config["optimization"] = optimization_config
        
        # Istniejący kod dla tworzenia pełnej konfiguracji...
        self.task_config = {
            "name": task_name,
            "type": "fine_tuning",
            "status": "Nowy",
            "config": config,
            # ...
        }
        
        # Pozostały istniejący kod...
    except Exception as e:
        self.logger.error(f"Błąd podczas dodawania zadania: {str(e)}", exc_info=True)
        QtWidgets.QMessageBox.critical(self, "Błąd", f"Nie można dodać zadania: {str(e)}")
Zmiany w pliku training_task_config_dialog.py:
W podobny sposób dodajemy nową zakładkę do training_task_config_dialog.py. Najpierw dodajemy nową zakładkę w metodzie _init_ui():
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

        # Pozostały kod...
Następnie implementujemy nowe metody, które są identyczne jak w przypadku fine_tuning_task_config_dialog.py:
pythondef _create_optimization_tab(self):
    """Tworzy zakładkę z parametrami optymalizacji treningu."""
    try:
        self.logger.debug("Tworzenie zakładki optymalizacji treningu")
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        
        # Informacja o profilu sprzętowym
        if self.hardware_profile:
            hardware_info = QtWidgets.QLabel(f"Używany profil sprzętowy: {self.hardware_profile.get('device_name', 'Nieznany')}")
        else:
            hardware_info = QtWidgets.QLabel("Brak załadowanego profilu sprzętowego")
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
            step=1
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
            step=1
        )
        form_layout.addRow("Liczba workerów:", num_workers_layout)
        
        # 3. Mixed precision
        mixed_precision_layout = self._create_parameter_row(
            name="Mixed precision", 
            param_key="use_mixed_precision",
            default_value=True,
            widget_type="checkbox"
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
            step=1
        )
        form_layout.addRow("Prefetch factor:", prefetch_layout)
        
        # 5. Pin memory
        pin_memory_layout = self._create_parameter_row(
            name="Pin memory", 
            param_key="pin_memory",
            default_value=True,
            widget_type="checkbox"
        )
        form_layout.addRow("Pin memory:", pin_memory_layout)
        
        # 6. Persistent workers
        persistent_workers_layout = self._create_parameter_row(
            name="Persistent workers", 
            param_key="persistent_workers",
            default_value=False,
            widget_type="checkbox"
        )
        form_layout.addRow("Persistent workers:", persistent_workers_layout)
        
        # 7. CUDA streaming
        cuda_stream_layout = self._create_parameter_row(
            name="CUDA streaming", 
            param_key="cuda_streaming",
            default_value=True,
            widget_type="checkbox"
        )
        form_layout.addRow("CUDA streaming:", cuda_stream_layout)
        
        # 8. Benchmark CUDNN
        benchmark_cudnn_layout = self._create_parameter_row(
            name="Benchmark CUDNN", 
            param_key="benchmark_cudnn",
            default_value=True,
            widget_type="checkbox"
        )
        form_layout.addRow("Benchmark CUDNN:", benchmark_cudnn_layout)
        
        # 9. Garbage collector
        gc_layout = self._create_parameter_row(
            name="Wyłącz garbage collector", 
            param_key="disable_gc",
            default_value=False,
            widget_type="checkbox"
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
            step=1
        )
        form_layout.addRow("Gradient accumulation steps:", grad_accum_layout)
        
        # 11. Channels last memory format
        channels_last_layout = self._create_parameter_row(
            name="Channels last memory format", 
            param_key="channels_last",
            default_value=False,
            widget_type="checkbox"
        )
        form_layout.addRow("Channels last:", channels_last_layout)
        
        # Dodanie całego layoutu do zakładki
        layout.addLayout(form_layout)
        
        # Dodanie przycisku do załadowania wszystkich optymalnych ustawień
        load_all_btn = QtWidgets.QPushButton("Zastosuj wszystkie optymalne ustawienia z profilu sprzętowego")
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

def _create_parameter_row(self, name, param_key, default_value, widget_type, min_val=None, max_val=None, step=None):
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
        'param_key': param_key,
        'profile_radio': profile_radio,
        'hardware_radio': hardware_radio,
        'value_widget': value_widget,
        'hw_value_label': hw_value_label,
        'hw_value': hw_value
    }
    
    # Dodanie do listy parametrów
    if not hasattr(self, 'optimization_params'):
        self.optimization_params = []
    self.optimization_params.append(row_widgets)
    
    # Obsługa zmiany źródła wartości
    def on_source_changed(id):
        if id == 1:  # Profil
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
    
    source_group.idClicked.connect(on_source_changed)
    
    return layout

def _apply_all_hardware_optimizations(self):
    """Zastosowuje wszystkie optymalne ustawienia z profilu sprzętowego."""
    if not hasattr(self, 'optimization_params') or not self.hardware_profile:
        QtWidgets.QMessageBox.warning(
            self, 
            "Ostrzeżenie", 
            "Brak dostępnego profilu sprzętowego lub parametrów do zastosowania."
        )
        return
    
    count = 0
    for param in self.optimization_params:
        param_key = param['param_key']
        if param_key in self.hardware_profile:
            param['hardware_radio'].setChecked(True)
            hw_value = self.hardware_profile[param_key]
            value_widget = param['value_widget']
            value_widget.setEnabled(False)
            
            if isinstance(value_widget, QtWidgets.QSpinBox) or isinstance(value_widget, QtWidgets.QDoubleSpinBox):
                value_widget.setValue(hw_value)
            elif isinstance(value_widget, QtWidgets.QCheckRetryClaude hit the max length for a message and has paused its response. You can write Continue to keep the chat going.MContinueEditKontynuacja kodu dla training_task_config_dialog.py:
pythondef _apply_all_hardware_optimizations(self):
    """Zastosowuje wszystkie optymalne ustawienia z profilu sprzętowego."""
    if not hasattr(self, 'optimization_params') or not self.hardware_profile:
        QtWidgets.QMessageBox.warning(
            self, 
            "Ostrzeżenie", 
            "Brak dostępnego profilu sprzętowego lub parametrów do zastosowania."
        )
        return
    
    count = 0
    for param in self.optimization_params:
        param_key = param['param_key']
        if param_key in self.hardware_profile:
            param['hardware_radio'].setChecked(True)
            hw_value = self.hardware_profile[param_key]
            value_widget = param['value_widget']
            value_widget.setEnabled(False)
            
            if isinstance(value_widget, QtWidgets.QSpinBox) or isinstance(value_widget, QtWidgets.QDoubleSpinBox):
                value_widget.setValue(hw_value)
            elif isinstance(value_widget, QtWidgets.QCheckBox):
                value_widget.setChecked(hw_value)
            else:
                value_widget.setText(str(hw_value))
            
            count += 1
    
    QtWidgets.QMessageBox.information(
        self, 
        "Sukces", 
        f"Zastosowano {count} optymalnych ustawień z profilu sprzętowego."
    )
Teraz aktualizujemy metodę _on_accept() w pliku training_task_config_dialog.py, aby uwzględnić nowe parametry optymalizacji:
pythondef _on_accept(self):
    """Obsługa zatwierdzenia konfiguracji."""
    try:
        # Istniejący kod...
        
        # Dodajemy sekcję optymalizacji do konfiguracji
        optimization_config = {}
        
        if hasattr(self, 'optimization_params'):
            for param in self.optimization_params:
                param_key = param['param_key']
                hardware_radio = param['hardware_radio']
                value_widget = param['value_widget']
                
                # Pobieranie wartości w zależności od typu widgetu
                if hardware_radio.isChecked() and param['hw_value'] is not None:
                    param_value = param['hw_value']
                else:
                    if isinstance(value_widget, QtWidgets.QSpinBox) or isinstance(value_widget, QtWidgets.QDoubleSpinBox):
                        param_value = value_widget.value()
                    elif isinstance(value_widget, QtWidgets.QCheckBox):
                        param_value = value_widget.isChecked()
                    else:
                        param_value = value_widget.text()
                
                optimization_config[param_key] = param_value
        
        # Uzupełniamy konfigurację o sekcję optymalizacji
        self.task_config["config"]["optimization"] = optimization_config
        
        # Pozostały istniejący kod...
        
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
Importy w training_task_config_dialog.py i fine_tuning_task_config_dialog.py
Upewnijmy się, że na początku obu plików są wszystkie potrzebne importy:
pythonimport datetime
import json
import logging
import os
from pathlib import Path

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt
Integracja nowych parametrów optymalizacji z istniejącym kodem
Na końcu metody _on_accept() w obu plikach musimy dostosować sposób integracji nowych parametrów z istniejącą konfiguracją. W pliku fine_tuning_task_config_dialog.py należy dodać następujące modyfikacje:
pythondef _on_accept(self):
    """Obsługa zatwierdzenia konfiguracji."""
    import os
    from PyQt6 import QtWidgets

    try:
        # Istniejący kod...
        
        # Pobranie konfiguracji optymalizacyjnej
        optimization_config = {}
        
        if hasattr(self, 'optimization_params'):
            for param in self.optimization_params:
                param_key = param['param_key']
                hardware_radio = param['hardware_radio']
                value_widget = param['value_widget']
                
                # Pobieranie wartości w zależności od typu widgetu
                if hardware_radio.isChecked() and param['hw_value'] is not None:
                    param_value = param['hw_value']
                else:
                    if isinstance(value_widget, QtWidgets.QSpinBox) or isinstance(value_widget, QtWidgets.QDoubleSpinBox):
                        param_value = value_widget.value()
                    elif isinstance(value_widget, QtWidgets.QCheckBox):
                        param_value = value_widget.isChecked()
                    else:
                        param_value = value_widget.text()
                
                optimization_config[param_key] = param_value
        
        # Istniejący kod dla tworzenia konfiguracji...
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
            # ... inne sekcje konfiguracji ...
            "optimization": optimization_config  # Dodajemy sekcję optymalizacji
        }
        
        # Pozostały istniejący kod...
        self.task_config = {
            "name": self.name_edit.text().strip(),
            "type": "fine_tuning",
            "status": "Nowy",
            "config": config,
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            # ... inne pola ...
        }
        
        # Dodaj logi
        self.logger.info(f"Utworzono konfigurację zadania: {task_name}")
        self.logger.info(f"Typ zadania: {self.task_config['type']}")
        self.logger.info(
            f"Pełna konfiguracja: {json.dumps(self.task_config, indent=2, ensure_ascii=False)}"
        )
        
        self.accept()
        
    except Exception as e:
        self.logger.error(f"Błąd podczas dodawania zadania: {str(e)}", exc_info=True)
        QtWidgets.QMessageBox.critical(self, "Błąd", f"Nie można dodać zadania: {str(e)}")
Poprawki dla obsługi radiobutton w nowszych wersjach PyQt6
W nowszych wersjach PyQt6 metoda idClicked mogła być zastąpiona inną metodą. Jeśli napotkasz problemy, zaktualizuj kod następująco:
pythondef _create_parameter_row(self, name, param_key, default_value, widget_type, min_val=None, max_val=None, step=None):
    # ... istniejący kod ...
    
    # Obsługa zmiany źródła wartości - alternatywna wersja dla nowszej PyQt6
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
Podsumowanie zmian
Wprowadzone zmiany dodają nową zakładkę "Optymalizacja treningu" do obu dialogów konfiguracyjnych (FineTuningTaskConfigDialog i TrainingTaskConfigDialog). Zakładka ta pozwala użytkownikowi na:

Wyświetlenie dostępnych parametrów optymalizacyjnych z profilu sprzętowego
Wybór źródła wartości dla każdego parametru (profil zadania lub profil sprzętowy)
Zobaczenie wartości parametrów z profilu sprzętowego
Zastosowanie wszystkich optymalnych ustawień z profilu sprzętowego jednym kliknięciem

Ponadto, konfiguracja optymalizacji jest uwzględniana przy zapisywaniu zadania do konfiguracji i może być później wykorzystana przez algorytmy trenujące.
Parametry optymalizacyjne, które zostały dodane:

Rozmiar batch'a
Liczba workerów
Mixed precision
Prefetch factor
Pin memory
Persistent workers
CUDA streaming
Benchmark CUDNN
Wyłączenie garbage collectora
Gradient accumulation steps
Channels last memory format

Te parametry powinny pokrywać większość ustawień, które mogą wpływać na wydajność treningu na konkretnym sprzęcie.