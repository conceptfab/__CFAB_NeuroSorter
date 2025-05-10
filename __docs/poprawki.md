Precyzyjne poprawki do pliku fine_tuning_task_config_dialog.py
Oto konkretne poprawki do kodu, które rozwiązują zidentyfikowane problemy. Każda poprawka jest opisana wraz z dokładnym kodem do wprowadzenia.
Poprawka 1: Usunięcie duplikacji kontrolek normalizacji
Należy usunąć redundantne tworzenie kontrolek normalizacji w \_create_augmentation_tab() i \_create_preprocessing_tab() i zamiast tego wykorzystać kontrolki już utworzone w \_init_controls().
python# Zmiana w \_create_augmentation_tab() - ok. linia 813

# Usuń cały blok tworzący kontrolki norm*mean*_ i norm*std*_ i zastąp go tym kodem:

# Mean

mean_layout = QtWidgets.QHBoxLayout()
mean_layout.addWidget(self.norm_mean_r)
mean_layout.addWidget(self.norm_mean_g)
mean_layout.addWidget(self.norm_mean_b)
norm_layout.addRow("Mean (RGB):", mean_layout)

# Std

std_layout = QtWidgets.QHBoxLayout()
std_layout.addWidget(self.norm_std_r)
std_layout.addWidget(self.norm_std_g)
std_layout.addWidget(self.norm_std_b)
norm_layout.addRow("Std (RGB):", std_layout)
python# Zmiana w \_create_preprocessing_tab() - ok. linia 878

# Usuń cały blok tworzący kontrolki norm*mean*_ i norm*std*_ i zastąp go tym kodem:

# Mean

mean_layout = QtWidgets.QHBoxLayout()
mean_layout.addWidget(self.norm_mean_r)
mean_layout.addWidget(self.norm_mean_g)
mean_layout.addWidget(self.norm_mean_b)
norm_layout.addRow("Mean (RGB):", mean_layout)

# Std

std_layout = QtWidgets.QHBoxLayout()
std_layout.addWidget(self.norm_std_r)
std_layout.addWidget(self.norm_std_g)
std_layout.addWidget(self.norm_std_b)
norm_layout.addRow("Std (RGB):", std_layout)
Poprawka 2: Naprawienie layoutu EWC dla adaptive_ewc_lambda_check
Dodanie brakującego kodu, który dokończy implementację layoutu dla adaptive_ewc_lambda_check w metodzie \_init_controls().
python# Na końcu metody \_init_controls() dodaj:

# Uzupełnienie konfiguracji layoutu EWC (brakująca część)

ewc_layout = QtWidgets.QHBoxLayout()
ewc_layout.addWidget(self.adaptive_ewc_lambda_check)
self.adaptive_ewc_layout = ewc_layout # Zapisanie referencji do późniejszego użycia
Poprawka 3: Bezpieczne usuwanie profili
Dodanie obsługi przypadku, gdy plik profilu nie istnieje w metodzie \_delete_profile().
python# Zastąp fragment w \_delete_profile() dotyczący usuwania pliku tym kodem:

profile_path = self.profiles_dir / f"{profile_name}.json"
if profile_path.exists():
profile_path.unlink()
self.\_refresh_profile_list()
self.current_profile = None
QtWidgets.QMessageBox.information(
self, "Sukces", "Profil został pomyślnie usunięty."
)
else:
self.logger.warning(f"Plik profilu nie istnieje: {profile_path}")
QtWidgets.QMessageBox.warning(
self, "Ostrzeżenie", f"Plik profilu nie istnieje: {profile_name}"
)
self.\_refresh_profile_list() # Odśwież listę, aby usunąć nieistniejące referencje
Poprawka 4: Bezpieczne ładowanie konfiguracji
Dodanie bezpiecznego dostępu do kluczy w słowniku konfiguracji w metodzie \_load_config().
python# Dodaj nową metodę pomocniczą po \_load_config():

def \_get_config_value(self, config, key_path, default_value=None):
"""
Bezpiecznie pobiera wartość z zagnieżdżonego słownika konfiguracji.

    Args:
        config: Słownik konfiguracji
        key_path: Lista kluczy do nawigacji w zagnieżdżonym słowniku
        default_value: Wartość domyślna, jeśli klucz nie istnieje

    Returns:
        Wartość z konfiguracji lub wartość domyślna
    """
    current = config
    for key in key_path:
        if not isinstance(current, dict) or key not in current:
            self.logger.warning(f"Klucz '{key}' nie istnieje w ścieżce {key_path}")
            return default_value
        current = current[key]
    return current

python# Przykład zastosowania - zamień fragment kodu w \_load_config():

# Zamiast:

if "architecture" in model_config:
architecture = model_config["architecture"] # ...

# Użyj:

architecture = self.\_get_config_value(config, ["model", "architecture"])
if architecture:
idx = self.arch_combo.findText(architecture)
if idx >= 0:
self.arch_combo.blockSignals(True)
self.arch_combo.setCurrentIndex(idx)
self.arch_combo.blockSignals(False)
self.\_update_variant_combo(architecture)
else:
self.logger.warning(f"Architektura {architecture} nie jest dostępna")
Poprawka 5: Bezpieczne blokowanie sygnałów
Dodanie metody zapewniającej odblokowanie sygnałów nawet w przypadku wyjątków.
python# Dodaj nową metodę pomocniczą:

def \_with_blocked_signals(self, func):
"""
Wykonuje funkcję z zablokowanymi sygnałami, gwarantując ich odblokowanie.

    Args:
        func: Funkcja do wykonania

    Returns:
        Wynik funkcji
    """
    was_blocked = self.signalsBlocked()
    self.blockSignals(True)
    try:
        return func()
    finally:
        self.blockSignals(was_blocked)

python# Zastosowanie w \_load_config():

# Zamiast:

self.blockSignals(True)
try: # ... kod wczytywania konfiguracji ...
except Exception as e:
self.blockSignals(False)
msg = "Błąd podczas ładowania konfiguracji"
self.logger.error(f"{msg}: {str(e)}", exc_info=True)
QtWidgets.QMessageBox.critical(self, "Błąd", f"{msg}: {str(e)}")

# Użyj:

def load_config_impl(): # ... kod wczytywania konfiguracji ...
self.\_update_ui_state()
self.logger.info("Konfiguracja modelu została pomyślnie załadowana")

try:
self.\_with_blocked_signals(load_config_impl)
except Exception as e:
msg = "Błąd podczas ładowania konfiguracji"
self.logger.error(f"{msg}: {str(e)}", exc_info=True)
QtWidgets.QMessageBox.critical(self, "Błąd", f"{msg}: {str(e)}")
Poprawka 6: Walidacja zgodności liczby klas
Dodanie walidacji liczby klas w katalogu treningowym w metodzie \_on_accept().
python# Dodaj tę walidację w \_on_accept() po sprawdzeniu, czy katalogi istnieją:

# Sprawdź zgodność liczby klas z liczbą katalogów

subdirs = [
d
for d in os.listdir(training_dir)
if os.path.isdir(os.path.join(training_dir, d))
]
num_classes = self.num_classes_spin.value()
if len(subdirs) != num_classes:
self.logger.warning(
f"Liczba katalogów ({len(subdirs)}) nie zgadza się z podaną liczbą klas ({num_classes})"
)
result = QtWidgets.QMessageBox.warning(
self,
"Niezgodność liczby klas",
f"Liczba podkatalogów w katalogu treningowym ({len(subdirs)}) nie zgadza się z "
f"podaną liczbą klas ({num_classes}). Czy chcesz kontynuować?",
QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
QtWidgets.QMessageBox.StandardButton.No,
)
if result == QtWidgets.QMessageBox.StandardButton.No:
return
Poprawka 7: Refaktoryzacja tworzenia kontrolek
Dodanie pomocniczych metod do tworzenia kontrolek, aby zmniejszyć duplikację kodu.
python# Dodaj te metody pomocnicze po **init**():

def \_create_spinbox(self, min_val, max_val, default_val, decimals=0, step=1):
"""
Tworzy i konfiguruje QSpinBox lub QDoubleSpinBox.

    Args:
        min_val: Minimalna wartość
        max_val: Maksymalna wartość
        default_val: Domyślna wartość
        decimals: Liczba miejsc po przecinku (0 dla QSpinBox)
        step: Wartość kroku

    Returns:
        Skonfigurowany QSpinBox lub QDoubleSpinBox
    """
    if decimals > 0:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setSingleStep(0.1 ** (decimals - 1))
    else:
        spin = QtWidgets.QSpinBox()
        spin.setSingleStep(step)

    spin.setRange(min_val, max_val)
    spin.setValue(default_val)
    return spin

def \_create_group_box(self, title, layout_type=QtWidgets.QFormLayout):
"""
Tworzy QGroupBox z określonym typem layoutu.

    Args:
        title: Tytuł grupy
        layout_type: Typ layoutu (domyślnie QFormLayout)

    Returns:
        Tuple: (group_box, layout)
    """
    group = QtWidgets.QGroupBox(title)
    layout = layout_type()
    group.setLayout(layout)
    return group, layout

def \_add_form_row(self, layout, label, widget):
"""
Dodaje wiersz do layoutu formularza, obsługując różne typy widgetów.

    Args:
        layout: Layout formularza
        label: Etykieta
        widget: Widget do dodania (może być pojedynczy widget, lista lub layout)
    """
    if isinstance(widget, list):
        hlayout = QtWidgets.QHBoxLayout()
        for w in widget:
            hlayout.addWidget(w)
        layout.addRow(label, hlayout)
    elif isinstance(widget, QtWidgets.QLayout):
        layout.addRow(label, widget)
    else:
        layout.addRow(label, widget)

Poprawka 8: Sprawdzenie istnienia pliku default_profile.json
Sprawdzenie czy plik default_profile.json istnieje przed próbą jego wczytania w metodzie \_create_profile_from_model_config().
python# Zastąp fragment kodu dotyczący wczytywania domyślnego profilu w \_create_profile_from_model_config():

# Ścieżki do plików

default_profile_path = self.profiles_dir / "default_profile.json"
extracted_config_path = self.profiles_dir / "extracted_config.json"
output_path = str(self.profiles_dir / f"{name.strip()}.json")
temp_base_profile_path = self.profiles_dir / "temp_base_profile.json"

# Sprawdź czy plik domyślnego profilu istnieje

if not default_profile_path.exists():
self.logger.error(
f"Nie znaleziono pliku default_profile.json pod ścieżką: {default_profile_path}"
)
QtWidgets.QMessageBox.critical(
self,
"Błąd",
"Nie znaleziono pliku default_profile.json. Skontaktuj się z administratorem.",
)
return

try: # Wczytaj domyślny profil
with open(default_profile_path, "r", encoding="utf-8") as f:
profile_data = json.load(f)

    # Reszta kodu...

Poprawka 9: Redukcja złożoności metody \_save_profile()
Wydzielenie logiki tworzenia konfiguracji do osobnej metody, aby zmniejszyć złożoność metody \_save_profile().
python# Dodaj nową metodę do zbierania konfiguracji:

def \_get_complete_config(self):
"""
Zbiera pełną konfigurację ze wszystkich kontrolek UI.

    Returns:
        Słownik z pełną konfiguracją
    """
    return {
        "model": {
            "architecture": self.arch_combo.currentText(),
            "variant": self.variant_combo.currentText(),
            "input_size": self.input_size_spin.value(),
            "num_classes": self.num_classes_spin.value(),
            "pretrained": self.pretrained_check.isChecked(),
            "pretrained_weights": self.pretrained_weights_combo.currentText(),
            "feature_extraction_only": self.feature_extraction_check.isChecked(),
            "activation": self.activation_combo.currentText(),
            "dropout_at_inference": self.dropout_at_inference_check.isChecked(),
            "global_pool": self.global_pool_combo.currentText(),
            "last_layer_activation": self.last_layer_activation_combo.currentText(),
        },
        "training": {
            "epochs": self.epochs_spin.value(),
            "batch_size": self.batch_size_spin.value(),
            "learning_rate": float(self.lr_spin.value()),
            "optimizer": self.optimizer_combo.currentText(),
            "scheduler": self.scheduler_combo.currentText(),
            "num_workers": self.num_workers_spin.value(),
            "warmup_epochs": self.warmup_epochs_spin.value(),
            "warmup_lr_init": self.warmup_lr_init_spin.value(),
            "mixed_precision": self.mixed_precision_check.isChecked(),
            "gradient_accumulation_steps": self.grad_accum_steps_spin.value(),
            "gradient_clip": self.gradient_clip_spin.value(),
            "validation_split": self.validation_split_spin.value(),
            "evaluation_freq": self.eval_freq_spin.value(),
            "use_ema": self.use_ema_check.isChecked(),
            "ema_decay": self.ema_decay_spin.value(),
            "unfreeze_strategy": self.unfreeze_strategy_combo.currentText(),
            "unfreeze_after_epochs": self.unfreeze_after_epochs_spin.value(),
            "unfreeze_layers": self.unfreeze_layers_spin.value(),
            "frozen_lr": self.frozen_lr_spin.value(),
            "unfrozen_lr": self.unfrozen_lr_spin.value(),
        },
        # ... kontynuacja dla pozostałych sekcji ...
    }

python# Uproszczona metoda \_save_profile():

def _save_profile(self):
"""Zapisuje aktualną konfigurację jako profil."""
try:
name, ok = QtWidgets.QInputDialog.getText(
self,
"Zapisz profil",
"Podaj nazwę dla nowego profilu:",
QtWidgets.QLineEdit.EchoMode.Normal,
f"{self.arch_combo.currentText()}_{self.variant_combo.currentText()}",
)

        if ok and name:
            config = self._get_complete_config()

            profile_data = {
                "type": "fine_tuning",
                "info": f"Profil dla {self.arch_combo.currentText()} {self.variant_combo.currentText()}",
                "description": "Profil utworzony przez użytkownika",
                "data_required": "Standardowe dane do doszkalania",
                "hardware_required": "Standardowy sprzęt",
                "config": config,
            }

            profile_path = self.profiles_dir / f"{name}.json"
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(profile_data, f, indent=4, ensure_ascii=False)

            self._refresh_profile_list()
            QtWidgets.QMessageBox.information(
                self, "Sukces", "Profil został pomyślnie zapisany."
            )

    except Exception as e:
        self.logger.error(
            f"Błąd podczas zapisywania profilu: {str(e)}", exc_info=True
        )
        QtWidgets.QMessageBox.critical(
            self, "Błąd", f"Nie można zapisać profilu: {str(e)}"
        )

Poprawka 10: Dodanie metody do walidacji katalogów
Wydzielenie logiki walidacji katalogów do osobnej metody, aby zmniejszyć duplikację kodu.
python# Dodaj nową metodę do walidacji katalogów:

def \_validate_directory(self, dir_path, is_training=True):
"""
Waliduje katalog z danymi treningowymi lub walidacyjnymi.

    Args:
        dir_path: Ścieżka do katalogu
        is_training: Czy to katalog treningowy (True) czy walidacyjny (False)

    Returns:
        Tuple: (is_valid, message, subdirs) - czy katalog jest poprawny,
        ewentualny komunikat błędu, lista podkatalogów
    """
    if not os.path.isdir(dir_path):
        return False, f"Katalog nie istnieje: {dir_path}", []

    subdirs = [
        d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))
    ]

    if not subdirs:
        dir_type = "treningowy" if is_training else "walidacyjny"
        return False, f"Katalog {dir_type} nie zawiera żadnych podfolderów (klas)", []

    return True, "", subdirs

python# Zastosowanie w \_on_accept():

# Zamiast:

if not os.path.isdir(training_dir):
QtWidgets.QMessageBox.critical(
self, "Błąd", f"Katalog treningowy nie istnieje:\n{training_dir}"
)
return
subdirs = [
d
for d in os.listdir(training_dir)
if os.path.isdir(os.path.join(training_dir, d))
]
if not subdirs:
QtWidgets.QMessageBox.critical(
self,
"Błąd",
f"Katalog treningowy nie zawiera żadnych podfolderów (klas):\n{training_dir}",
)
return

# Użyj:

is_valid, error_msg, train_subdirs = self.\_validate_directory(training_dir, True)
if not is_valid:
QtWidgets.QMessageBox.critical(self, "Błąd", error_msg)
return

is_valid, error_msg, val_subdirs = self.\_validate_directory(validation_dir, False)
if not is_valid:
QtWidgets.QMessageBox.critical(self, "Błąd", error_msg)
return

# Sprawdź zgodność liczby klas

num_classes = self.num_classes_spin.value()
if len(train_subdirs) != num_classes: # ... (reszta kodu z poprawki 6) ...
Te poprawki adresują główne problemy znalezione w kodzie, poprawiając jego jakość, czytelność i stabilność.
