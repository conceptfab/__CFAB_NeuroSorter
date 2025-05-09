Analiza różnic między plikami
Analizując plik fine_tuning_task_config_dialog.py oraz default_profile.json, zauważyłem następujące różnice w obsługiwanych parametrach:
Zmiany w pliku fine_tuning_task_config_dialog.py
1. Sekcja augmentacji (augmentation)
W kodzie interfejsu użytkownika dodano obsługę autoaugment i randaugment, których nie ma w domyślnym profilu:
python# AutoAugment
self.autoaugment_check = QtWidgets.QCheckBox()
self.autoaugment_check.setChecked(False)

# RandAugment
self.randaugment_check = QtWidgets.QCheckBox()
self.randaugment_check.setChecked(False)
self.randaugment_n_spin = QtWidgets.QSpinBox()
self.randaugment_n_spin.setRange(1, 10)
self.randaugment_n_spin.setValue(2)  # Domyślna wartość N
self.randaugment_m_spin = QtWidgets.QSpinBox()
self.randaugment_m_spin.setRange(1, 15)  # Zakres dla M
self.randaugment_m_spin.setValue(9)  # Domyślna wartość M
2. Sekcja zaawansowanych kontrolek augmentacji
W interfejsie dodano dodatkowe kontrolki dla zaawansowanej augmentacji, których nie ma w profilu domyślnym:
python# Advanced augmentation
advanced_group = QtWidgets.QGroupBox("Advanced")
advanced_layout = QtWidgets.QFormLayout()

# Contrast
self.contrast_spin = QtWidgets.QDoubleSpinBox()
# Saturation
self.saturation_spin = QtWidgets.QDoubleSpinBox()
# Hue
self.hue_spin = QtWidgets.QDoubleSpinBox()
# Shear
self.shear_spin = QtWidgets.QDoubleSpinBox()
# Channel shift
self.channel_shift_spin = QtWidgets.QDoubleSpinBox()
3. Błąd w strategii odmrażania warstw
W kodzie zdefiniowano stałą UNFREEZE_AFTER_EPOCHS z literówką:
pythonUNFREEZE_AFTER_EPOCHS = "unfreeze_after_epoochs"  # Literówka: epoochs zamiast epochs
4. Brak reprezentacji w JSON
W interfejsie istnieją elementy, które nie są zapisywane w konfiguracji w metodzie _on_accept():

Nie widzę zapisywania wartości dla contrast_spin, saturation_spin, hue_spin, shear_spin, channel_shift_spin w finałowej konfiguracji

5. Elementy profilu niepobierane w metodzie _load_config
Nie wszystkie elementy konfiguracji są wczytywane w metodzie _load_config:

Brak wczytywania wartości dla autoaugment, randaugment oraz zaawansowanych kontrolek augmentacji
Brak obsługi wartości stochastic_depth, survival_probability itd.

Proponowane poprawki
1. Naprawić literówkę w stałej UNFREEZE_AFTER_EPOCHS
Zmiana w pliku fine_tuning_task_config_dialog.py:
python# Zmienić
UNFREEZE_AFTER_EPOCHS = "unfreeze_after_epoochs"
# Na
UNFREEZE_AFTER_EPOCHS = "unfreeze_after_epochs"
2. Dodać brakujące parametry do domyślnego profilu default_profile.json
Dodać sekcje dla autoaugment i randaugment w profilu domyślnym:
json"augmentation": {
  "basic": {
    "use": false,
    "rotation": 30,
    "brightness": 0.2,
    "shift": 0.1,
    "zoom": 0.1,
    "horizontal_flip": true,
    "vertical_flip": false
  },
  "mixup": {
    "use": false,
    "alpha": 0.2
  },
  "cutmix": {
    "use": false,
    "alpha": 0.2
  },
  "autoaugment": {
    "use": false
  },
  "randaugment": {
    "use": false,
    "n": 2,
    "m": 9
  },
  "advanced": {
    "contrast": 0.2,
    "saturation": 0.2,
    "hue": 0.1,
    "shear": 0.1,
    "channel_shift": 0.0
  }
}
3. Uzupełnić zapisywanie wszystkich parametrów w metodzie _on_accept()
Dodać brakujące parametry do zapisywanej konfiguracji:
python"augmentation": {
    # Istniejące parametry...
    "autoaugment": {
        "use": self.autoaugment_check.isChecked(),
    },
    "randaugment": {
        "use": self.randaugment_check.isChecked(),
        "n": self.randaugment_n_spin.value(),
        "m": self.randaugment_m_spin.value(),
    },
    "advanced": {
        "contrast": self.contrast_spin.value(),
        "saturation": self.saturation_spin.value(),
        "hue": self.hue_spin.value(),
        "shear": self.shear_spin.value(),
        "channel_shift": self.channel_shift_spin.value(),
    },
},
4. Uzupełnić wczytywanie wszystkich parametrów w metodzie _load_config
Dodać brakujące fragmenty do wczytywania konfiguracji:
python# Dodać obsługę autoaugment/randaugment
autoaugment_config = augmentation_config.get("autoaugment", {})
if "use" in autoaugment_config:
    self.autoaugment_check.setChecked(autoaugment_config["use"])

randaugment_config = augmentation_config.get("randaugment", {})
if "use" in randaugment_config:
    self.randaugment_check.setChecked(randaugment_config["use"])
if "n" in randaugment_config:
    self.randaugment_n_spin.setValue(randaugment_config["n"])
if "m" in randaugment_config:
    self.randaugment_m_spin.setValue(randaugment_config["m"])

# Dodać obsługę zaawansowanych parametrów augmentacji
advanced_config = augmentation_config.get("advanced", {})
if "contrast" in advanced_config:
    self.contrast_spin.setValue(advanced_config["contrast"])
if "saturation" in advanced_config:
    self.saturation_spin.setValue(advanced_config["saturation"])
if "hue" in advanced_config:
    self.hue_spin.setValue(advanced_config["hue"])
if "shear" in advanced_config:
    self.shear_spin.setValue(advanced_config["shear"])
if "channel_shift" in advanced_config:
    self.channel_shift_spin.setValue(advanced_config["channel_shift"])
Te zmiany zapewnią pełną zgodność parametrów między interfejsem użytkownika a strukturą domyślnego profilu, co ułatwi zapisywanie i wczytywanie konfiguracji bez utraty danych.