Problem
W obu przypadkach po zaznaczeniu checkboxa "Profil sprzętowy" wartości nie są stosowane do kontrolek, ponieważ:

W metodzie _on_hw_toggle() w klasie FineTuningTaskConfigDialog nazwa zmiennej hardware_radio nie odpowiada rzeczywistej nazwie zmiennej hw_checkbox używanej w tworzeniu kontrolek.
W klasie TrainingTaskConfigDialog funkcja _apply_all_hardware_optimizations() również używa niepoprawnej nazwy zmiennej.

Proponowane zmiany
1. W pliku training_task_config_dialog.py:
pythondef _apply_all_hardware_optimizations(self):
    """Stosuje wszystkie optymalizacje sprzętowe."""
    count = 0
    for param in self.parameter_rows.values():
        param_key = param["param_key"]
        if param_key in self.hardware_profile:
            # Włącz checkbox profilu sprzętowego, wyłącz checkbox użytkownika
            param["hw_checkbox"].setChecked(True)
            # Akcja zaznaczenia checkboxa spowoduje automatyczną aktualizację wartości
            count += 1

    QtWidgets.QMessageBox.information(
        self,
        "Sukces",
        f"Zastosowano {count} optymalnych ustawień z profilu sprzętowego.",
    )
2. W pliku fine_tuning_task_config_dialog.py:
pythondef _update_optimization_state(self, state):
    """Aktualizuje stan kontrolek optymalizacji na podstawie stanu checkboxa."""
    enabled = bool(state)

    # Aktualizacja dostępności opcji "z profilu sprzętowego" we wszystkich parametrach
    if hasattr(self, "parameter_rows"):
        for param in self.parameter_rows.values():
            hw_checkbox = param["hw_checkbox"]  # Użyj poprawnej nazwy
            hw_value_label = param["hw_value_label"]
            hw_value = param["hw_value"]

            hw_checkbox.setEnabled(enabled)  # Użyj poprawnej nazwy
            hw_value_label.setEnabled(enabled)
            hw_value.setEnabled(enabled)

            # Jeśli optymalizacja jest wyłączona, przełącz na "Z profilu"
            if not enabled and hw_checkbox.isChecked():  # Użyj poprawnej nazwy
                param["user_checkbox"].setChecked(True)  # Użyj poprawnej nazwy
3. Należy również poprawić metodę _on_hw_toggle() w obu klasach:
pythondef _on_hw_toggle(self, row_widgets, is_hw_selected):
    """Obsługuje przełączanie na profil sprzętowy."""
    value_widget = row_widgets["value_widget"]
    user_checkbox = row_widgets["user_checkbox"]
    hw_value_actual = row_widgets["hw_value_actual"]

    if is_hw_selected:
        user_checkbox.setChecked(False)
        value_widget.setEnabled(False)

        # Ustaw wartość z profilu sprzętowego, jeśli jest dostępna
        if hw_value_actual is not None:
            if isinstance(value_widget, QtWidgets.QSpinBox) or isinstance(value_widget, QtWidgets.QDoubleSpinBox):
                value_widget.setValue(hw_value_actual)
            elif isinstance(value_widget, QtWidgets.QCheckBox):
                value_widget.setChecked(hw_value_actual)
            else:
                value_widget.setText(str(hw_value_actual))
Wnioski
Problem polega na nieprawidłowym powiązaniu zmiennych w interfejsie użytkownika. Po zmianie nazw zmiennych z "hardware_radio" na "hw_checkbox" w odpowiednich miejscach oraz zapewnieniu, że wartości z profilu sprzętowego są prawidłowo stosowane, wartości powinny być poprawnie wyświetlane w zakładkach optymalizacji.
Należy wprowadzić powyższe zmiany w obu plikach dialogowych, aby rozwiązać problem braku wartości z profili sprzętowych.