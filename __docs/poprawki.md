Poprawki do klasy FineTuningTaskConfigDialog
1. Eliminacja duplikacji kontrolek
python# Usunięcie duplikacji kontrolek w zakładkach augmentacji i preprocessingu

# W metodzie _init_controls() zostawiamy wszystkie definicje kontrolek
def _init_controls(self):
    """Inicjalizacja wszystkich kontrolek."""
    # ... pozostały kod ...
    
    # Normalization controls - używane w wielu miejscach
    self.norm_mean_r = QtWidgets.QDoubleSpinBox()
    self.norm_mean_r.setRange(0.0, 1.0)
    self.norm_mean_r.setValue(0.485)
    self.norm_mean_r.setDecimals(3)
    
    # ... pozostałe kontrolki ...

# W metodzie _create_augmentation_tab() usuwamy duplikowane definicje
def _create_augmentation_tab(self) -> QtWidgets.QWidget:
    """Tworzy zakładkę z parametrami augmentacji."""
    tab = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout()
    
    # ... pozostały kod ...
    
    # Normalization
    norm_group = QtWidgets.QGroupBox("Normalization")
    norm_layout = QtWidgets.QFormLayout()
    
    # Mean - używamy już zdefiniowanych kontrolek
    mean_layout = QtWidgets.QHBoxLayout()
    mean_layout.addWidget(self.norm_mean_r)
    mean_layout.addWidget(self.norm_mean_g)
    mean_layout.addWidget(self.norm_mean_b)
    
    norm_layout.addRow("Mean (RGB):", mean_layout)
    
    # ... podobnie dla pozostałych kontrolek ...
2. Usprawniony mechanizm aktualizacji UI
pythondef _update_ui_state(self):
    """Aktualizuje stan UI po zmianie konfiguracji."""
    
    # Słownik określający zależności między kontrolkami
    dependencies = {
        # Dla kontrolek architektury
        'architecture': {
            'enabled': True,
            'update_method': self._update_variant_combo
        },
        # Dla optymalizatora
        'optimizer': {
            'weight_decay_spin': lambda opt: opt != "Adam",
            'momentum_spin': lambda opt: opt == "SGD"
        },
        # Podobnie dla innych kontrolek...
    }
    
    # Aktualizacja kontrolek architektury
    architecture = self.arch_combo.currentText()
    if 'architecture' in dependencies:
        if 'update_method' in dependencies['architecture']:
            dependencies['architecture']['update_method'](architecture)
    
    # Aktualizacja kontrolek optymalizatora
    optimizer = self.optimizer_combo.currentText()
    if 'optimizer' in dependencies:
        for control, condition in dependencies['optimizer'].items():
            if hasattr(self, control):
                getattr(self, control).setEnabled(condition(optimizer))
    
    # ... podobnie dla innych grup kontrolek ...
3. Poprawione zarządzanie pamięcią
pythondef closeEvent(self, event):
    """Obsługa zamknięcia okna."""
    try:
        self.logger.info("Zamykanie okna dialogowego")
        
        # Odłączenie wszystkich połączonych sygnałów
        self.arch_combo.currentTextChanged.disconnect()
        self.optimizer_combo.currentTextChanged.disconnect()
        self.scheduler_combo.currentTextChanged.disconnect()
        self.unfreeze_strategy_combo.currentTextChanged.disconnect()
        
        # ... odłączenie pozostałych sygnałów ...
        
        # Czyszczenie zasobów logowania
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
        
        self.accept()
        event.accept()
        
    except Exception as e:
        self.logger.error(f"Błąd podczas zamykania okna dialogowego: {str(e)}")
        event.accept()  # Zawsze akceptujemy zamknięcie okna
4. Usprawniona walidacja danych
pythondef _validate_config(self) -> bool:
    """Sprawdza poprawność danych konfiguracyjnych."""
    errors = []
    
    # Sprawdzenie nazwy zadania
    task_name = self.name_edit.text().strip()
    if not task_name:
        errors.append("Nazwa zadania nie może być pusta.")
    
    # Sprawdzenie ścieżki modelu
    model_path = self.model_path_edit.text().strip()
    if not model_path:
        errors.append("Musisz wybrać model bazowy.")
    elif not os.path.exists(model_path):
        errors.append(f"Wybrany plik modelu nie istnieje: {model_path}")
    
    # Sprawdzenie katalogów danych
    training_dir = self.train_dir_edit.text().strip()
    if not training_dir:
        errors.append("Katalog treningowy nie może być pusty.")
    elif not os.path.isdir(training_dir):
        errors.append(f"Katalog treningowy nie istnieje: {training_dir}")
    else:
        subdirs = [d for d in os.listdir(training_dir) 
                  if os.path.isdir(os.path.join(training_dir, d))]
        if not subdirs:
            errors.append("Katalog treningowy nie zawiera podfolderów (klas).")
    
    # ... podobnie dla katalogu walidacyjnego i innych parametrów ...
    
    # Wyświetlenie wszystkich błędów na raz
    if errors:
        QtWidgets.QMessageBox.critical(
            self,
            "Błędy walidacji",
            "\n".join(errors)
        )
        return False
    
    return True

def _on_accept(self):
    """Obsługa zatwierdzenia konfiguracji."""
    try:
        # Najpierw walidacja danych
        if not self._validate_config():
            return
            
        # ... pozostały kod tworzący konfigurację ...
        
    except Exception as e:
        self.logger.error(f"Błąd podczas dodawania zadania: {str(e)}", exc_info=True)
        QtWidgets.QMessageBox.critical(
            self, "Błąd", f"Nie można dodać zadania: {str(e)}"
        )
5. Usprawniona obsługa przycisków dialogu
pythondef _init_ui(self):
    """Inicjalizuje interfejs użytkownika."""
    try:
        self.logger.debug("Rozpoczęcie inicjalizacji UI")
        layout = QtWidgets.QVBoxLayout(self)
        
        # ... kod tworzący zakładki ...
        
        layout.addWidget(self.tabs)
        
        # Przyciski - użycie QDialogButtonBox dla lepszej kompatybilności z platformą
        button_box = QtWidgets.QDialogButtonBox()
        
        # Przycisk "Dodaj zadanie"
        add_task_btn = button_box.addButton("Dodaj zadanie", 
                                            QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        add_task_btn.clicked.connect(self._on_accept)
        
        # Przycisk "Zamknij"
        close_btn = button_box.addButton("Zamknij", 
                                        QtWidgets.QDialogButtonBox.ButtonRole.RejectRole)
        close_btn.clicked.connect(self.reject)
        
        layout.addWidget(button_box)
        
        self.logger.debug("Zakończono inicjalizację UI")
        
    except Exception as e:
        msg = "Błąd podczas inicjalizacji UI"
        self.logger.error(f"{msg}: {str(e)}", exc_info=True)
        raise
6. Uproszczenie zarządzania profilami
pythondef _save_profile(self):
    """Zapisuje aktualną konfigurację jako profil."""
    try:
        # Pobranie nazwy profilu od użytkownika
        suggested_name = f"{self.arch_combo.currentText()}_{self.variant_combo.currentText()}"
        name, ok = QtWidgets.QInputDialog.getText(
            self,
            "Zapisz profil",
            "Podaj nazwę dla nowego profilu:",
            QtWidgets.QLineEdit.EchoMode.Normal,
            suggested_name,
        )
        
        if not (ok and name.strip()):
            return
            
        # Sprawdzenie, czy profil już istnieje
        profile_path = self.profiles_dir / f"{name.strip()}.json"
        if profile_path.exists():
            confirm = QtWidgets.QMessageBox.question(
                self,
                "Potwierdzenie",
                f"Profil o nazwie '{name.strip()}' już istnieje. Czy chcesz go nadpisać?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No
            )
            if confirm != QtWidgets.QMessageBox.StandardButton.Yes:
                return
                
        # Utworzenie konfiguracji profilu
        profile_data = self._create_profile_data(name.strip())
        
        # Zapisanie profilu
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=4, ensure_ascii=False)
            
        self._refresh_profile_list()
        QtWidgets.QMessageBox.information(
            self, "Sukces", "Profil został pomyślnie zapisany."
        )
        
    except Exception as e:
        self.logger.error(f"Błąd podczas zapisywania profilu: {str(e)}", exc_info=True)
        QtWidgets.QMessageBox.critical(
            self, "Błąd", f"Nie można zapisać profilu: {str(e)}"
        )
        
def _create_profile_data(self, name):
    """Tworzy dane profilu na podstawie aktualnej konfiguracji."""
    return {
        "type": "fine_tuning",
        "info": name,
        "description": f"Profil dla {self.arch_combo.currentText()} {self.variant_combo.currentText()}",
        "data_required": "Standardowe dane do doszkalania",
        "hardware_required": "Standardowy sprzęt",
        "config": {
            "model": self._get_model_config(),
            "training": self._get_training_config(),
            "regularization": self._get_regularization_config(),
            # ... podobnie dla pozostałych sekcji ...
        }
    }
    
def _get_model_config(self):
    """Zwraca konfigurację modelu."""
    return {
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
    }
    
# ... podobne metody dla innych sekcji konfiguracji ...
7. Refaktoryzacja struktur danych i zarządzania stanem
python# Dodaj na początku klasy FineTuningTaskConfigDialog
def __init__(self, parent=None, settings=None, hardware_profile=None):
    super().__init__(parent)
    self.settings = settings
    self.hardware_profile = hardware_profile
    
    # Inicjalizacja stanu
    self.state = {
        "current_profile": None,
        "config": None,
        "model_path": "",
        "train_dir": "",
        "val_dir": ""
    }
    
    self._setup_logging()
    self.setWindowTitle("Konfiguracja doszkalania")
    self.setMinimumWidth(1000)
    self.profiles_dir = Path("data/profiles")
    self.profiles_dir.mkdir(exist_ok=True)

    # Inicjalizacja wszystkich kontrolek
    self._init_controls()

    # Inicjalizacja interfejsu
    self._init_ui()
    
# Aktualizacja metod dostępu do stanu:
def _on_profile_selected(self, current, previous):
    """Obsługa wyboru profilu z listy."""
    if current is None:
        return
    try:
        profile_name = current.text()
        profile_path = self.profiles_dir / f"{profile_name}.json"
        with open(profile_path, "r", encoding="utf-8") as f:
            profile_data = json.load(f)
        
        # Aktualizacja stanu
        self.state["current_profile"] = profile_data
        
        # Aktualizacja UI
        self.profile_info.setText(profile_data.get("info", ""))
        self.profile_description.setText(profile_data.get("description", ""))
        self.profile_data_required.setText(profile_data.get("data_required", ""))
        self.profile_hardware_required.setText(
            profile_data.get("hardware_required", "")
        )
    except Exception as e:
        self.logger.error(
            f"Błąd podczas ładowania profilu: {str(e)}", exc_info=True
        )
        QtWidgets.QMessageBox.critical(
            self, "Błąd", f"Nie można załadować profilu: {str(e)}"
        )
8. Usprawniona implementacja logowania
pythondef _setup_logging(self):
    """Konfiguracja logowania dla okna dialogowego."""
    # Użycie unikalnego identyfikatora dla logera, aby uniknąć konfliktów
    logger_name = f"{__name__}.{id(self)}"
    self.logger = logging.getLogger(logger_name)
    self.logger.setLevel(logging.DEBUG)

    # Sprawdzenie, czy logger już ma handlery, aby uniknąć duplikacji
    if not self.logger.handlers:
        # Handler do pliku
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "training_dialog.log"
        
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)

        # Handler do konsoli
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Format logów
        log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(log_fmt)
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    self.logger.info("Inicjalizacja okna dialogowego")
Powyższe propozycje poprawek koncentrują się na eliminacji nadmiarowego kodu, poprawie zarządzania zasobami, usprawnieniu logiki aplikacji oraz zwiększeniu stabilności i utrzymywalności kodu. Wykorzystanie dobrych praktyk projektowania interfejsów PyQt6 i Qt Designer może znacząco usprawnić proces tworzenia i utrzymania dialogów w aplikacji Creating Dialogs With Qt Designer - PyQt6.