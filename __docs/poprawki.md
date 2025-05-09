Zmiany w pliku app/gui/dialogs/fine_tuning_task_config_dialog.py
1. Problem z wczytywaniem parametrów konfiguracyjnych
pythondef _select_model_file(self):
    # ...
    # Wczytaj plik konfiguracyjny
    config_path = os.path.splitext(file_path)[0] + "_config.json"
    self.logger.info(f"Próba wczytania pliku konfiguracyjnego: {config_path}")

    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                self.logger.info(f"Wczytana konfiguracja: {json.dumps(config, indent=2)}")

                # Ustaw nazwę zadania na podstawie klucza "name" z konfiguracji
                if "name" in config:
                    task_name = f"{config['name']}_FT"
                    self.name_edit.setText(task_name)
                    self.logger.info(f"Ustawiono nazwę zadania: {task_name}")

                # Użyj metody _load_config do załadowania całej konfiguracji
                self._load_config(config)
                self.logger.info("Zastosowano konfigurację modelu")
Błąd polega na tym, że metoda _update_dependent_controls() nie jest wywoływana po załadowaniu konfiguracji. Powinno być:
pythondef _select_model_file(self):
    # ...
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                # ...

                # Użyj metody _load_config do załadowania całej konfiguracji
                self._load_config(config)
                
                # Dodaj tę linię, aby zaktualizować kontrolki zależne
                self._update_dependent_controls()
                
                self.logger.info("Zastosowano konfigurację modelu")
2. Problem z inicjalizacją kontrolek
Niektóre kontrolki mogą nie być dostępne w momencie wczytywania konfiguracji. Należy upewnić się, że wszystkie kontrolki są poprawnie inicjalizowane przed wczytaniem konfiguracji.
3. Problem z aktualizacją wariantów w zależności od architektury
pythondef _update_variant_combo(self, architecture: str) -> None:
    """Aktualizuje listę wariantów dla wybranej architektury."""
    self.logger.info(f"Aktualizacja wariantów dla architektury: {architecture}")

    # Zapisz aktualnie wybrany wariant
    current_variant = self.variant_combo.currentText()
    self.logger.info(f"Aktualnie wybrany wariant: {current_variant}")

    # Wyczyść i dodaj nowe warianty
    self.variant_combo.clear()

    if architecture == "EfficientNet":
        variants = ["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8"]
    elif architecture == "ResNet":
        variants = ["18", "34", "50", "101", "152"]
    elif architecture == "MobileNet":
        variants = ["v2", "v3_small", "v3_large"]
    else:
        variants = ["default"]

    self.logger.info(f"Dostępne warianty: {variants}")
    self.variant_combo.addItems(variants)

    # Próbuj przywrócić poprzednio wybrany wariant
    if current_variant in variants:
        self.variant_combo.setCurrentText(current_variant)
        self.logger.info(f"Przywracam poprzednio wybrany wariant: {current_variant}")
    else:
        self.logger.info("Nie można przywrócić poprzedniego wariantu, ustawiam domyślny")
Metoda ta powinna zostać wywołana po zmianie architektury podczas wczytywania konfiguracji, ale może to nie działać poprawnie.
4. Problem z synchronizacją kontrolek
pythondef _load_config(self, config: Dict[str, Any]) -> None:
    """Ładuje konfigurację do interfejsu."""
    try:
        # ...

        # 1. Aktualizacja parametrów modelu
        if "architecture" in model_config:
            self.arch_combo.setCurrentText(model_config["architecture"])
            # Po zmianie architektury aktualizujemy dostępne warianty
            self._update_variant_combo(model_config["architecture"])
            self._on_architecture_changed(model_config["architecture"])

        if "variant" in model_config:
            self.variant_combo.setCurrentText(model_config["variant"])
W tym miejscu problem polega na tym, że setCurrentText dla variant_combo jest wywoływany po zmianie architektury, co może powodować, że wariant zostanie nadpisany przez domyślny dla nowej architektury.
5. Brak aktualizacji kontrolek po wczytaniu konfiguracji
Na końcu metody _load_config brakuje wywołania metody, która zaktualizowałaby stan kontrolek na podstawie wczytanych wartości:
pythondef _load_config(self, config: Dict[str, Any]) -> None:
    # ...
    
    # Dodaj na końcu metody:
    self._update_ui_state()
Poprawka
Oto kompletna poprawka dla metody _load_config:
pythondef _load_config(self, config: Dict[str, Any]) -> None:
    """Ładuje konfigurację do interfejsu."""
    try:
        # Model
        model_config = config.get("model", {})

        # Logowanie wartości przed ustawieniem
        self.logger.info(f"Ładowanie konfiguracji - Nazwa zadania: {self.name_edit.text()}")
        self.logger.info(f"Ładowanie konfiguracji - Liczba klas: {self.num_classes_spin.value()}")

        # Ustawienie wartości z konfiguracji
        if "name" in config:
            self.name_edit.setText(config["name"])

        # 1. Aktualizacja parametrów modelu
        if "architecture" in model_config:
            # Najpierw ustawiamy architekturę
            self.arch_combo.setCurrentText(model_config["architecture"])
            # To wywołanie zaktualizuje listę wariantów
            self._update_variant_combo(model_config["architecture"])

        # Teraz ustawiamy wariant, po aktualizacji listy wariantów
        if "variant" in model_config:
            variant = model_config["variant"]
            # Jeśli wariant jest dostępny w aktualnej liście, ustaw go
            idx = self.variant_combo.findText(variant)
            if idx >= 0:
                self.variant_combo.setCurrentIndex(idx)
                self.logger.info(f"Ustawiono wariant: {variant}")
            else:
                self.logger.warning(f"Wariant {variant} nie jest dostępny dla architektury {model_config.get('architecture')}")

        # Pozostałe parametry modelu
        if "input_size" in model_config:
            self.input_size_spin.setValue(model_config["input_size"])

        if "num_classes" in model_config:
            self.num_classes_spin.setValue(model_config["num_classes"])

        if "pretrained" in model_config:
            self.pretrained_check.setChecked(model_config["pretrained"])

        if "pretrained_weights" in model_config:
            self.pretrained_weights_combo.setCurrentText(model_config["pretrained_weights"])

        if "feature_extraction_only" in model_config:
            self.feature_extraction_check.setChecked(model_config["feature_extraction_only"])

        if "activation" in model_config:
            self.activation_combo.setCurrentText(model_config["activation"])

        if "dropout_at_inference" in model_config:
            self.dropout_at_inference_check.setChecked(model_config["dropout_at_inference"])

        if "global_pool" in model_config:
            self.global_pool_combo.setCurrentText(model_config["global_pool"])

        if "last_layer_activation" in model_config:
            self.last_layer_activation_combo.setCurrentText(model_config["last_layer_activation"])
        
        # ... (pozostała część metody bez zmian)
        
        # Na koniec aktualizujemy stan UI
        self._update_ui_state()
        self.logger.info("Konfiguracja modelu została pomyślnie załadowana")

    except Exception as e:
        msg = "Błąd podczas ładowania konfiguracji"
        self.logger.error(f"{msg}: {str(e)}", exc_info=True)
        QtWidgets.QMessageBox.critical(self, "Błąd", f"{msg}: {str(e)}")
Dodatkowo należy zmodyfikować metodę _select_model_file:
pythondef _select_model_file(self):
    """Wybiera plik modelu do doszkalania."""
    try:
        # ... (bez zmian)
        
        if file_path:
            self.model_path_edit.setText(file_path)
            self.logger.info(f"Wybrano plik modelu: {file_path}")

            # Wczytaj plik konfiguracyjny
            config_path = os.path.splitext(file_path)[0] + "_config.json"
            self.logger.info(f"Próba wczytania pliku konfiguracyjnego: {config_path}")

            if os.path.exists(config_path):
                try:
                    with open(config_path, "r") as f:
                        config = json.load(f)
                        self.logger.info(f"Wczytana konfiguracja: {json.dumps(config, indent=2)}")

                        # Ustaw nazwę zadania na podstawie klucza "name" z konfiguracji
                        if "name" in config:
                            task_name = f"{config['name']}_FT"
                            self.name_edit.setText(task_name)
                            self.logger.info(f"Ustawiono nazwę zadania: {task_name}")

                        # Użyj metody _load_config do załadowania całej konfiguracji
                        self._load_config(config)
                        
                        # Aktualizacja zależnych kontrolek
                        self._update_dependent_controls()
                        
                        self.logger.info("Zastosowano konfigurację modelu")

                except Exception as e:
                    self.logger.error(f"Błąd podczas wczytywania konfiguracji: {str(e)}")
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Błąd",
                        f"Nie udało się wczytać konfiguracji modelu: {str(e)}",
                    )
            else:
                # ... (bez zmian)
    except Exception as e:
        # ... (bez zmian)
Podsumowanie zmian

W metodzie _load_config:

Poprawne ustawienie architektury i wariantu
Dodanie wywołania _update_ui_state() na końcu metody


W metodzie _select_model_file:

Dodanie wywołania _update_dependent_controls() po wczytaniu konfiguracji



Te zmiany powinny rozwiązać problem z wczytywaniem konfiguracji z pliku i prawidłowym aktualizowaniem kontrolek UI na podstawie wczytanych wartości.