Przeanalizowałem kod źródłowy dla obu okien dialogowych konfiguracji zadań treningowych. Znalazłem kilka problemów, które należy rozwiązać, aby przy wyłączonej opcji "Użyj optymalizacji sprzętowej" wszystkie opcje "z profilu sprzętowego" były niedostępne.
Zmiany w kodzie
Plik fine_tuning_task_config_dialog.py
python# Dodajemy brakujące połączenie między checkboxem optymalizacji a funkcją aktualizującą dostępność opcji
def _init_ui(self):
    # [istniejący kod...]
    
    # Dodać po zdefiniowaniu use_optimization_checkbox:
    self.use_optimization_checkbox = QtWidgets.QCheckBox("Używaj optymalizacji sprzętowej")
    self.use_optimization_checkbox.setChecked(True)
    self.use_optimization_checkbox.stateChanged.connect(self._update_optimization_state)
    
    # [istniejący kod...]

# Dodajemy nową metodę do aktualizacji dostępności opcji z profilu sprzętowego
def _update_optimization_state(self, state):
    """Aktualizuje dostępność opcji z profilu sprzętowego w zależności od stanu checkboxa."""
    enabled = state == Qt.CheckState.Checked
    
    # Aktualizacja dostępności opcji "z profilu sprzętowego" we wszystkich parametrach
    if hasattr(self, "optimization_params"):
        for param in self.optimization_params:
            hardware_radio = param["hardware_radio"]
            hardware_radio.setEnabled(enabled)
            
            # Jeśli optymalizacja jest wyłączona, zmieniamy na "Z profilu"
            if not enabled and hardware_radio.isChecked():
                param["profile_radio"].setChecked(True)
Plik training_task_config_dialog.py
python# Dodajemy brakujące połączenie między checkboxem optymalizacji a funkcją aktualizującą dostępność opcji
def _init_ui(self):
    # [istniejący kod...]
    
    # Dodać po zdefiniowaniu use_optimization_checkbox:
    self.use_optimization_checkbox = QtWidgets.QCheckBox("Używaj optymalizacji sprzętowej")
    self.use_optimization_checkbox.setChecked(True)
    self.use_optimization_checkbox.stateChanged.connect(self._update_optimization_state)
    
    # [istniejący kod...]

# Dodajemy nową metodę do aktualizacji dostępności opcji z profilu sprzętowego
def _update_optimization_state(self, state):
    """Aktualizuje dostępność opcji z profilu sprzętowego w zależności od stanu checkboxa."""
    enabled = state == Qt.CheckState.Checked
    
    # Aktualizacja dostępności opcji "z profilu sprzętowego" we wszystkich parametrach
    if hasattr(self, "optimization_params"):
        for param in self.optimization_params:
            hardware_radio = param["hardware_radio"]
            hardware_radio.setEnabled(enabled)
            
            # Jeśli optymalizacja jest wyłączona, zmieniamy na "Z profilu"
            if not enabled and hardware_radio.isChecked():
                param["profile_radio"].setChecked(True)
Zmiany w metodzie _create_parameter_row w obu plikach
Dodatkowo, w obu klasach należy zmodyfikować metodę _create_parameter_row, aby sprawdzała stan checkboxa optymalizacji przy tworzeniu parametrów:
pythondef _create_parameter_row(self, name, param_key, default_value, widget_type, min_val=None, max_val=None, step=None):
    # [istniejący kod...]
    
    # Przycisk opcji dla wartości z profilu sprzętowego
    hardware_radio = QtWidgets.QRadioButton("Z profilu sprzętowego")
    source_group.addButton(hardware_radio, 2)
    
    # Sprawdź czy optymalizacja jest włączona
    optimization_enabled = True
    if hasattr(self, "use_optimization_checkbox"):
        optimization_enabled = self.use_optimization_checkbox.isChecked()
    
    # Wartość z profilu sprzętowego
    hw_value = None
    if self.hardware_profile and param_key in self.hardware_profile:
        hw_value = self.hardware_profile[param_key]
        hardware_radio.setEnabled(optimization_enabled)  # Dodana zależność od stanu checkboxa
    else:
        hardware_radio.setEnabled(False)
        hardware_radio.setText("Z profilu sprzętowego (niedostępne)")
    
    # [pozostały kod...]
Zmiany w metodzie _apply_all_hardware_optimizations w obu plikach
Należy również zmodyfikować metodę, która aktywuje wszystkie optymalizacje sprzętowe, aby sprawdzała, czy checkbox optymalizacji jest włączony:
pythondef _apply_all_hardware_optimizations(self):
    """Zastosowuje wszystkie optymalne ustawienia z profilu sprzętowego."""
    # Sprawdź czy optymalizacja jest włączona
    if hasattr(self, "use_optimization_checkbox") and not self.use_optimization_checkbox.isChecked():
        QtWidgets.QMessageBox.warning(
            self,
            "Optymalizacja wyłączona",
            "Optymalizacja sprzętowa jest obecnie wyłączona. Włącz ją, aby zastosować ustawienia z profilu sprzętowego.",
        )
        return
        
    if not hasattr(self, "optimization_params") or not self.hardware_profile:
        QtWidgets.QMessageBox.warning(
            self,
            "Ostrzeżenie",
            "Brak dostępnego profilu sprzętowego lub parametrów do zastosowania.",
        )
        return
    
    # [pozostały kod...]
Podsumowanie zmian

Dodano połączenie między checkboxem optymalizacji a funkcją aktualizującą dostępność opcji z profilu sprzętowego.
Dodano nową metodę _update_optimization_state, która aktualizuje dostępność opcji w zależności od stanu checkboxa.
Zmodyfikowano metodę _create_parameter_row, aby przy tworzeniu parametrów uwzględniała stan checkboxa optymalizacji.
Zmodyfikowano metodę _apply_all_hardware_optimizations, aby sprawdzała, czy optymalizacja jest włączona.

Te zmiany zapewnią, że przy wyłączonej opcji "Użyj optymalizacji sprzętowej" wszystkie opcje "z profilu sprzętowego" będą niedostępne, a jeśli były już wybrane, to zostaną przełączone na opcje "Z profilu".