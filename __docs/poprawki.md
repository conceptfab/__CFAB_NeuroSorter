1. Problem z radiobutton w obu oknach dialogowych
W obu klasach (TrainingTaskConfigDialog i FineTuningTaskConfigDialog) w funkcji _create_parameter_row tworzone są button grupy, które nie są prawidłowo przechowywane jako atrybuty klasy. To powoduje, że tylko ostatnio wybrana opcja jest zapamiętywana, a reszta traci stan.
2. Problem z wygaszaniem kontrolek
Funkcja _update_optimization_state nie jest nigdzie podłączana do kontrolki use_optimization_checkbox, przez co zmiany stanu checkboxa nie wywołują aktualizacji interfejsu.
Proponowane zmiany w kodzie
Zmiany w pliku training_task_config_dialog.py:
python# Zmiana 1: Poprawka funkcji _create_parameter_row
def _create_parameter_row(
    self,
    name,
    param_key,
    default_value,
    widget_type,
    min_val=None,
    max_val=None,
    step=None,
):
    """
    Tworzy wiersz parametru z opcją wyboru źródła wartości.
    """
    layout = QtWidgets.QHBoxLayout()

    # Źródło wartości - Tworzymy unikalną grupę dla każdego wiersza
    source_group = QtWidgets.QButtonGroup()
    
    # Przycisk opcji dla wartości z UI/profilu
    profile_radio = QtWidgets.QRadioButton("Z profilu")
    profile_radio.setChecked(True)
    source_group.addButton(profile_radio, 1)
    
    # Przycisk opcji dla wartości z profilu sprzętowego
    hardware_radio = QtWidgets.QRadioButton("Z profilu sprzętowego")
    source_group.addButton(hardware_radio, 2)
    
    # Sprawdź czy optymalizacja jest włączona
    optimization_enabled = True
    if hasattr(self, "use_optimization_checkbox"):
        optimization_enabled = self.use_optimization_checkbox.isChecked()
    
    # [reszta kodu bez zmian]
    
    # Zapamiętanie referencji do widgetów i grupy przycisków
    row_widgets = {
        "param_key": param_key,
        "profile_radio": profile_radio,
        "hardware_radio": hardware_radio,
        "value_widget": value_widget,
        "hw_value_label": hw_value_label,
        "hw_value": hw_value,
        "button_group": source_group,  # Zapisujemy grupę by nie została usunięta przez GC
    }
    
    # [reszta kodu bez zmian]
    
    return layout

# Zmiana 2: Dodanie inicjalizacji use_optimization_checkbox i podłączenie sygnału
def _create_optimization_tab(self):
    """Tworzenie zakładki Optymalizacja treningu."""
    try:
        self.logger.debug("Tworzenie zakładki optymalizacji treningu")
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        
        # Dodanie checkbox'a do włączania/wyłączania optymalizacji
        self.use_optimization_checkbox = QtWidgets.QCheckBox("Użyj optymalizacji sprzętowej")
        self.use_optimization_checkbox.setChecked(True)
        self.use_optimization_checkbox.stateChanged.connect(self._update_optimization_state)
        layout.addWidget(self.use_optimization_checkbox)
        
        # [reszta kodu bez zmian]
Zmiany w pliku fine_tuning_task_config_dialog.py:
python# Zmiana 1: Poprawka funkcji _create_parameter_row (analogiczna jak wyżej)
def _create_parameter_row(
    self,
    name,
    param_key,
    default_value,
    widget_type,
    min_val=None,
    max_val=None,
    step=None,
):
    """
    Tworzy wiersz parametru z opcją wyboru źródła wartości.
    """
    layout = QtWidgets.QHBoxLayout()

    # Źródło wartości - Tworzymy unikalną grupę dla każdego wiersza
    source_group = QtWidgets.QButtonGroup()
    
    # Przycisk opcji dla wartości z UI/profilu
    profile_radio = QtWidgets.QRadioButton("Z profilu")
    profile_radio.setChecked(True)
    source_group.addButton(profile_radio, 1)
    
    # Przycisk opcji dla wartości z profilu sprzętowego
    hardware_radio = QtWidgets.QRadioButton("Z profilu sprzętowego")
    source_group.addButton(hardware_radio, 2)
    
    # Sprawdź czy optymalizacja jest włączona
    optimization_enabled = True
    if hasattr(self, "use_optimization_checkbox"):
        optimization_enabled = self.use_optimization_checkbox.isChecked()
    
    # [reszta kodu bez zmian]
    
    # Zapamiętanie referencji do widgetów i grupy przycisków
    row_widgets = {
        "param_key": param_key,
        "profile_radio": profile_radio,
        "hardware_radio": hardware_radio,
        "value_widget": value_widget,
        "hw_value_label": hw_value_label,
        "hw_value": hw_value,
        "button_group": source_group,  # Zapisujemy grupę by nie została usunięta przez GC
    }
    
    # [reszta kodu bez zmian]
    
    return layout

# Zmiana 2: Dodanie inicjalizacji use_optimization_checkbox i podłączenie sygnału
def _create_optimization_tab(self):
    """Tworzenie zakładki Optymalizacja treningu."""
    try:
        self.logger.debug("Tworzenie zakładki optymalizacji treningu")
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        
        # Dodanie checkbox'a do włączania/wyłączania optymalizacji
        self.use_optimization_checkbox = QtWidgets.QCheckBox("Użyj optymalizacji sprzętowej")
        self.use_optimization_checkbox.setChecked(True)
        self.use_optimization_checkbox.stateChanged.connect(self._update_optimization_state)
        layout.addWidget(self.use_optimization_checkbox)
        
        # [reszta kodu bez zmian]
Podsumowanie zmian w plikach
Zmiany w pliku training_task_config_dialog.py:

Dodanie checkboxa use_optimization_checkbox w funkcji _create_optimization_tab
Podłączenie sygnału stateChanged checkboxa do funkcji _update_optimization_state
Zachowanie grupy przycisków w słowniku row_widgets jako atrybut button_group

Zmiany w pliku fine_tuning_task_config_dialog.py:

Dodanie checkboxa use_optimization_checkbox w funkcji _create_optimization_tab
Podłączenie sygnału stateChanged checkboxa do funkcji _update_optimization_state
Zachowanie grupy przycisków w słowniku row_widgets jako atrybut button_group

Te zmiany pozwolą na:

Niezależne wybieranie opcji "Z profilu sprzętowego" dla różnych parametrów
Prawidłowe działanie wygaszania kontrolek przy wyłączeniu opcji "Użyj optymalizacji sprzętowej"