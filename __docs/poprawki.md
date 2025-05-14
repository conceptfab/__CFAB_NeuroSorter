Analiza problemu z wartościami w zakładce Optymalizacja treningu
Analizując kod i zrzuty ekranu aplikacji, widzę problem z wyświetlaniem sugerowanych wartości z profilu sprzętowego w zakładce "Optymalizacja treningu". Zamiast wartości z profilu sprzętowego, widoczne są wartości "None" lub "Brak".
Zidentyfikowane problemy
Problemy występują w następujących plikach:
Zmiany w pliku training_task_config_dialog.py:
python# W metodzie _create_parameter_row:
# Brak przypisania wartości z profilu sprzętowego podczas inicjalizacji

# Obecna implementacja
hw_value = QtWidgets.QLabel(str(self.hardware_profile.get(param_key, "Brak")))

# Należy dodać pobranie i wyświetlenie rzeczywistej wartości:
hw_value_actual = self.hardware_profile.get(param_key)
hw_value_text = str(hw_value_actual) if hw_value_actual is not None else "Brak"
hw_value = QtWidgets.QLabel(hw_value_text)
Zmiany w pliku fine_tuning_task_config_dialog.py:
python# Podobny problem w metodzie _create_parameter_row
# Brakuje poprawnego przekazania wartości z profilu sprzętowego:

hw_value_actual = self.hardware_profile.get(param_key)
hw_value_text = str(hw_value_actual) if hw_value_actual is not None else "Brak"
hw_value = QtWidgets.QLabel(hw_value_text)
Rozwiązanie - zmiany w kodzie
1. Zmiany w training_task_config_dialog.py
pythondef _create_parameter_row(self, name, param_key, default_value, widget_type, min_val=None, max_val=None, step=None):
    """
    Tworzy wiersz parametru z kontrolkami dla ustawień użytkownika i profilu sprzętowego.
    """
    layout = QtWidgets.QHBoxLayout()

    # Wartość użytkownika
    if widget_type == "int":
        value_widget = QtWidgets.QSpinBox()
        value_widget.setRange(min_val or -999999, max_val or 999999)
        value_widget.setValue(default_value)
        if step:
            value_widget.setSingleStep(step)
    elif widget_type == "float":
        value_widget = QtWidgets.QDoubleSpinBox()
        value_widget.setRange(min_val or -999999.0, max_val or 999999.0)
        value_widget.setValue(default_value)
        if step:
            value_widget.setSingleStep(step)
    elif widget_type == "bool":
        value_widget = QtWidgets.QCheckBox()
        value_widget.setChecked(default_value)
    else:
        value_widget = QtWidgets.QLineEdit(str(default_value))

    # Checkbox "Użytkownika"
    user_checkbox = QtWidgets.QCheckBox("Użytkownika")
    user_checkbox.setChecked(True)

    # Etykieta i wartość z profilu sprzętowego
    hw_value_label = QtWidgets.QLabel("Profil sprzętowy:")
    # Pobierz wartość z profilu sprzętowego lub wyświetl "Brak"
    hw_value_actual = self.hardware_profile.get(param_key)
    hw_value_text = str(hw_value_actual) if hw_value_actual is not None else "Brak"
    hw_value = QtWidgets.QLabel(hw_value_text)

    # Checkbox "Profil sprzętowy"
    hw_checkbox = QtWidgets.QCheckBox("Profil sprzętowy")
    hw_checkbox.setChecked(False)

    # Grupa przycisków (checkboxów) do wyboru źródła wartości
    source_group = QtWidgets.QButtonGroup()
    source_group.addButton(user_checkbox)
    source_group.addButton(hw_checkbox)
    source_group.setExclusive(True)  # Tylko jeden z checkboxów może być zaznaczony

    # Dodaj widgety do layoutu
    layout.addWidget(value_widget)
    layout.addWidget(user_checkbox)
    layout.addWidget(hw_value_label)
    layout.addWidget(hw_value)
    layout.addWidget(hw_checkbox)

    # Zapamiętanie referencji do widgetów i grupy przycisków
    row_widgets = {
        "param_key": param_key,
        "value_widget": value_widget,
        "user_checkbox": user_checkbox,
        "hw_value_label": hw_value_label,
        "hw_value": hw_value,
        "hw_checkbox": hw_checkbox,
        "button_group": source_group,
        "hw_value_actual": hw_value_actual,
    }

    # Podpięcie zdarzeń do kontrolek
    user_checkbox.toggled.connect(lambda checked: self._on_source_toggle(row_widgets, checked))
    hw_checkbox.toggled.connect(lambda checked: self._on_hw_toggle(row_widgets, checked))

    # Zapisz referencje jako atrybut klasy
    if not hasattr(self, "parameter_rows"):
        self.parameter_rows = {}
    self.parameter_rows[param_key] = row_widgets

    return layout
2. Zmiany w fine_tuning_task_config_dialog.py
pythondef _create_parameter_row(self, name, param_key, default_value, widget_type, min_val=None, max_val=None, step=None):
    """
    Tworzy wiersz parametru z opcją wyboru źródła wartości.
    """
    layout = QtWidgets.QHBoxLayout()

    # Wartość użytkownika
    if widget_type == "int":
        value_widget = QtWidgets.QSpinBox()
        value_widget.setRange(min_val or -999999, max_val or 999999)
        value_widget.setValue(default_value)
        if step:
            value_widget.setSingleStep(step)
    elif widget_type == "float":
        value_widget = QtWidgets.QDoubleSpinBox()
        value_widget.setRange(min_val or -999999.0, max_val or 999999.0)
        value_widget.setValue(default_value)
        if step:
            value_widget.setSingleStep(step)
    else:
        value_widget = QtWidgets.QLineEdit(str(default_value))

    # Checkbox "Użytkownika"
    user_checkbox = QtWidgets.QCheckBox("Użytkownika")
    user_checkbox.setChecked(True)

    # Etykieta i wartość z profilu sprzętowego
    hw_value_label = QtWidgets.QLabel("Profil sprzętowy:")
    hw_value_actual = self.hardware_profile.get(param_key)
    hw_value_text = str(hw_value_actual) if hw_value_actual is not None else "Brak"
    hw_value = QtWidgets.QLabel(hw_value_text)

    # Checkbox "Profil sprzętowy"
    hw_checkbox = QtWidgets.QCheckBox("Profil sprzętowy")
    hw_checkbox.setChecked(False)

    # Grupa przycisków (checkboxów) do wyboru źródła wartości
    source_group = QtWidgets.QButtonGroup()
    source_group.addButton(user_checkbox)
    source_group.addButton(hw_checkbox)
    source_group.setExclusive(True)

    # Dodaj widgety do layoutu
    layout.addWidget(value_widget)
    layout.addWidget(user_checkbox)
    layout.addWidget(hw_value_label)
    layout.addWidget(hw_value)
    layout.addWidget(hw_checkbox)

    # Zapamiętanie referencji do widgetów i grupy przycisków
    row_widgets = {
        "param_key": param_key,
        "value_widget": value_widget,
        "user_checkbox": user_checkbox,
        "hw_value_label": hw_value_label,
        "hw_value": hw_value,
        "hw_checkbox": hw_checkbox,
        "button_group": source_group,
        "hw_value_actual": hw_value_actual,
    }

    # Podpięcie zdarzeń do kontrolek
    user_checkbox.toggled.connect(lambda checked: self._on_source_toggle(row_widgets, checked))
    hw_checkbox.toggled.connect(lambda checked: self._on_hw_toggle(row_widgets, checked))

    # Zapisz referencje jako atrybut klasy
    if not hasattr(self, "parameter_rows"):
        self.parameter_rows = {}
    self.parameter_rows[param_key] = row_widgets

    return layout
3. Problem z inicjalizacją hardware_profile w konstruktorach obu klas
Należy również upewnić się, że hardware_profile jest prawidłowo inicjalizowany w konstruktorach obu klas dialogowych. Prawdopodobnie brakuje również poprawnej inicjalizacji profilu sprzętowego lub inicjalizacja jest niepoprawna.
python# Sprawdź inicjalizację w konstruktorze
def __init__(self, parent=None, settings=None, hardware_profile=None):
    super().__init__(parent)
    self.settings = settings
    
    # Kluczowa zmiana - sprawdzenie czy hardware_profile jest inicjalizowany poprawnie
    if not hardware_profile:
        from app.profiler import HardwareProfiler  # Import lokalny aby uniknąć cyklicznych zależności
        profiler = HardwareProfiler()
        self.hardware_profile = profiler.get_optimal_parameters()
    else:
        self.hardware_profile = hardware_profile
Dodatkowe uwagi

W pliku profiler.py metoda get_optimal_parameters() zwraca słownik z optymalnymi parametrami lub wartościami domyślnymi, jeśli profil nie istnieje. Ta metoda działa prawidłowo, ale wartości nie są poprawnie przekazywane do interfejsu.
Upewnij się, że parametry w hardwareProfile są mapowane na dokładnie te same nazwy kluczy, których oczekują dialogi konfiguracyjne.
Sprawdź spójność nazw kluczy między profilowaniem a oknem dialogowym, gdyż mogą istnieć różnice powodujące, że hardware_profile nie zawiera wartości dla oczekiwanego klucza.

Implementując powyższe zmiany, interfejs powinien wyświetlać poprawne wartości pobrane z profilu sprzętowego zamiast "None" lub "Brak".