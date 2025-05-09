1. Dodanie przycisku aktualizacji (Reset UI)
Dodanie jawnego przycisku aktualizacji może być dobrym rozwiązaniem. Modyfikacja metody _select_model_file:
pythondef _select_model_file(self):
    """Wybiera plik modelu do doszkalania."""
    try:
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Wybierz plik modelu do doszkalania",
            str(Path("data/models")),
            "Pliki modeli (*.pth *.pt *.ckpt);;Wszystkie pliki (*.*)",
        )
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
                        self.config = config  # Zapisz konfigurację jako atrybut obiektu
                        self.logger.info(f"Wczytana konfiguracja: {json.dumps(config, indent=2)}")

                        # Ustaw nazwę zadania na podstawie klucza "name" z konfiguracji
                        if "name" in config:
                            task_name = f"{config['name']}_FT"
                            self.name_edit.setText(task_name)
                            self.logger.info(f"Ustawiono nazwę zadania: {task_name}")

                        # Dodaj pytanie czy zaaplikować konfigurację
                        reply = QtWidgets.QMessageBox.question(
                            self, 
                            "Wczytano konfigurację",
                            "Czy chcesz zastosować wczytaną konfigurację?",
                            QtWidgets.QMessageBox.StandardButton.Yes | 
                            QtWidgets.QMessageBox.StandardButton.No,
                            QtWidgets.QMessageBox.StandardButton.Yes
                        )
                        
                        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
                            self._apply_config()

                except Exception as e:
                    self.logger.error(f"Błąd podczas wczytywania konfiguracji: {str(e)}")
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Błąd",
                        f"Nie udało się wczytać konfiguracji modelu: {str(e)}",
                    )
            else:
                self.logger.warning(f"Nie znaleziono pliku konfiguracyjnego: {config_path}")
                QtWidgets.QMessageBox.warning(
                    self,
                    "Ostrzeżenie",
                    "Nie znaleziono pliku konfiguracyjnego dla wybranego modelu.",
                )

    except Exception as e:
        self.logger.error(f"Błąd podczas wyboru pliku modelu: {str(e)}")
        QtWidgets.QMessageBox.critical(
            self,
            "Błąd",
            f"Wystąpił błąd podczas wyboru pliku modelu: {str(e)}",
        )
2. Dodanie nowej metody _apply_config
Ta metoda będzie odpowiedzialna za aktualizację kontrolek UI na podstawie wczytanej konfiguracji:
pythondef _apply_config(self):
    """Aplikuje wczytaną konfigurację do UI."""
    try:
        if not hasattr(self, 'config'):
            QtWidgets.QMessageBox.warning(
                self,
                "Ostrzeżenie",
                "Brak wczytanej konfiguracji."
            )
            return
            
        config = self.config
        
        # Najpierw zablokuj wszystkie sygnały w całym dialogu
        # Zapisz stany blokowania sygnałów wszystkich kontrolek
        widgets = self.findChildren(QtWidgets.QWidget)
        blocked_states = {}
        
        for widget in widgets:
            blocked_states[widget] = widget.signalsBlocked()
            widget.blockSignals(True)
            
        try:
            # Ustawienie parametrów modelu
            model_config = config.get("model", {})
            
            # 1. Najpierw architektura i wariant
            if "architecture" in model_config:
                architecture = model_config["architecture"]
                idx = self.arch_combo.findText(architecture)
                if idx >= 0:
                    self.arch_combo.setCurrentIndex(idx)
                    self._update_variant_combo(architecture)  # Aktualizuje listę wariantów
                else:
                    self.logger.warning(f"Architektura {architecture} nie jest dostępna")
            
            if "variant" in model_config:
                variant = model_config["variant"]
                idx = self.variant_combo.findText(variant)
                if idx >= 0:
                    self.variant_combo.setCurrentIndex(idx)
                    self.logger.info(f"Ustawiono wariant: {variant}")
                else:
                    self.logger.warning(f"Wariant {variant} nie jest dostępny")
                    
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
            
            # 2. Parametry treningu
            training_config = config.get("training", {})
            
            if "batch_size" in training_config:
                self.batch_size_spin.setValue(training_config["batch_size"])
                
            if "learning_rate" in training_config:
                self.lr_spin.setValue(training_config["learning_rate"])
                
            if "optimizer" in training_config:
                self.optimizer_combo.setCurrentText(training_config["optimizer"])
                
            if "scheduler" in training_config:
                self.scheduler_combo.setCurrentText(training_config["scheduler"])
                
            if "warmup_epochs" in training_config:
                self.warmup_epochs_spin.setValue(training_config["warmup_epochs"])
                
            if "mixed_precision" in training_config:
                self.mixed_precision_check.setChecked(training_config["mixed_precision"])
                
            if "unfreeze_strategy" in training_config:
                self.unfreeze_strategy_combo.setCurrentText(training_config["unfreeze_strategy"])
                
            if "unfreeze_layers" in training_config:
                self.unfreeze_layers_spin.setValue(training_config["unfreeze_layers"])
                
            if "warmup_lr_init" in training_config:
                self.warmup_lr_init_spin.setValue(training_config["warmup_lr_init"])
                
            if "gradient_accumulation_steps" in training_config:
                self.grad_accum_steps_spin.setValue(training_config["gradient_accumulation_steps"])
                
            if "validation_split" in training_config:
                self.validation_split_spin.setValue(training_config["validation_split"])
                
            if "evaluation_freq" in training_config:
                self.eval_freq_spin.setValue(training_config["evaluation_freq"])
                
            if "use_ema" in training_config:
                self.use_ema_check.setChecked(training_config["use_ema"])
                
            if "ema_decay" in training_config:
                self.ema_decay_spin.setValue(training_config["ema_decay"])
            
            # 3. Parametry regularyzacji
            regularization_config = config.get("regularization", {})
            
            if "weight_decay" in regularization_config:
                self.weight_decay_spin.setValue(regularization_config["weight_decay"])
                
            if "drop_connect_rate" in regularization_config:
                self.drop_connect_spin.setValue(regularization_config["drop_connect_rate"])
                
            if "dropout_rate" in regularization_config:
                self.dropout_spin.setValue(regularization_config["dropout_rate"])
                
            if "label_smoothing" in regularization_config:
                self.label_smoothing_spin.setValue(regularization_config["label_smoothing"])
                
            # SWA
            swa_config = regularization_config.get("swa", {})
            if "use" in swa_config:
                self.use_swa_check.setChecked(swa_config["use"])
                
            if "start_epoch" in swa_config:
                self.swa_start_epoch_spin.setValue(swa_config["start_epoch"])
                
            # 4. Parametry augmentacji i preprocessingu
            augmentation_config = config.get("augmentation", {})
            
            # Basic augmentation
            basic_config = augmentation_config.get("basic", {})
            if "use" in basic_config:
                self.basic_aug_check.setChecked(basic_config["use"])
                
            # Pozostałe parametry augmentacji...
            
            # 5. Normalizacja
            preprocessing_config = config.get("preprocessing", {})
            
            # Normalization
            normalization_config = preprocessing_config.get("normalization", {})
            if "mean" in normalization_config and len(normalization_config["mean"]) == 3:
                self.norm_mean_r.setValue(normalization_config["mean"][0])
                self.norm_mean_g.setValue(normalization_config["mean"][1])
                self.norm_mean_b.setValue(normalization_config["mean"][2])
                
            if "std" in normalization_config and len(normalization_config["std"]) == 3:
                self.norm_std_r.setValue(normalization_config["std"][0])
                self.norm_std_g.setValue(normalization_config["std"][1])
                self.norm_std_b.setValue(normalization_config["std"][2])
            
            # Na końcu ręcznie aktualizujemy UI
            self._update_architecture_dependent_controls()
            self._update_training_dependent_controls()
            self._update_augmentation_dependent_controls()
            self._update_preprocessing_dependent_controls()
            self._update_monitoring_dependent_controls()
            
            QtWidgets.QMessageBox.information(
                self,
                "Sukces",
                "Konfiguracja została pomyślnie zastosowana."
            )
            
        finally:
            # Przywracamy stany blokowania sygnałów
            for widget, was_blocked in blocked_states.items():
                widget.blockSignals(was_blocked)
                
    except Exception as e:
        self.logger.error(f"Błąd podczas stosowania konfiguracji: {str(e)}", exc_info=True)
        QtWidgets.QMessageBox.critical(
            self,
            "Błąd",
            f"Nie można zastosować konfiguracji: {str(e)}"
        )
3. Dodanie przycisku "Reset" do UI
W metodzie _create_data_model_tab, dodajmy przycisk do resetu UI:
pythondef _create_data_model_tab(self) -> QtWidgets.QWidget:
    # ... istniejący kod ...
    
    # Dodaj przycisk aktualizacji UI
    reset_btn = QtWidgets.QPushButton("Zastosuj konfigurację")
    reset_btn.clicked.connect(self._apply_config)
    form.addRow("", reset_btn)
    
    # ... reszta istniejącego kodu ...
4. Alternatywne podejście - użycie QSignalBlocker
Zamiast ręcznego blokowania i odblokowywania sygnałów, możemy użyć QSignalBlocker, który automatycznie odblokowuje sygnały po wyjściu z kontekstu:
pythonfrom PyQt6.QtCore import QSignalBlocker

def _apply_config(self):
    """Aplikuje wczytaną konfigurację do UI."""
    try:
        if not hasattr(self, 'config'):
            QtWidgets.QMessageBox.warning(
                self,
                "Ostrzeżenie",
                "Brak wczytanej konfiguracji."
            )
            return
            
        config = self.config
        
        # Użyj QSignalBlocker dla całego dialogu
        with QSignalBlocker(self):
            # Ustawienie parametrów modelu
            model_config = config.get("model", {})
            
            # 1. Najpierw architektura i wariant
            # ... i tak dalej, taki sam kod jak wcześniej ...
            
            # Na końcu ręcznie aktualizujemy UI
            self._update_architecture_dependent_controls()
            self._update_training_dependent_controls()
            self._update_augmentation_dependent_controls()
            self._update_preprocessing_dependent_controls()
            self._update_monitoring_dependent_controls()
            
        QtWidgets.QMessageBox.information(
            self,
            "Sukces",
            "Konfiguracja została pomyślnie zastosowana."
        )
                
    except Exception as e:
        self.logger.error(f"Błąd podczas stosowania konfiguracji: {str(e)}", exc_info=True)
        QtWidgets.QMessageBox.critical(
            self,
            "Błąd",
            f"Nie można zastosować konfiguracji: {str(e)}"
        )
5. Problem z dublowaniem kontrolek
Jednym z głównych problemów w oryginalnym kodzie było dublowanie kontrolek dla normalizacji norm_mean_r, norm_mean_g, itp. W aktualnym kodzie zostało to już poprawione przez dodanie inicjalizacji wszystkich kontrolek w metodzie _init_controls(). Upewnijmy się jednak, że w metodach _create_augmentation_tab() i _create_preprocessing_tab() nie są tworzone nowe instancje tych kontrolek, a używane są już istniejące.
Podsumowanie
Proponowane zmiany:

Dodanie przycisku "Zastosuj konfigurację" do UI
Dodanie metody _apply_config() do ręcznej aktualizacji UI na podstawie wczytanej konfiguracji
Modyfikacja metody _select_model_file(), aby pytała użytkownika, czy chce zastosować wczytaną konfigurację
Użycie QSignalBlocker lub ręcznego blokowania sygnałów dla wszystkich kontrolek podczas aktualizacji UI

Te zmiany powinny rozwiązać problem z wczytywaniem konfiguracji z pliku, ponieważ dają użytkownikowi większą kontrolę nad procesem i zapewniają, że sygnały są prawidłowo blokowane podczas aktualizacji.