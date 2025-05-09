Zmiany w pliku app/gui/dialogs/fine_tuning_task_config_dialog.py
Problem
Po wybraniu modelu dialog powinien automatycznie wczytywać odpowiedni plik konfiguracyjny i aktualizować interfejs użytkownika o wartości z tego pliku. Z informacji wynika, że to nie działa prawidłowo - aplikacja nie aktualizuje parametrów UI po wczytaniu modelu i jego konfiguracji.
Analiza przyczyny
Analizując kod metody \_select_model_file(), znalazłem kilka problemów:

Metoda poprawnie wczytuje plik konfiguracyjny, ale nie wszystkie elementy interfejsu są aktualizowane
Część parametrów wymienionych w liście wymaganych do aktualizacji (np. unfreeze_strategy, mixed_precision) nie jest poprawnie ustawiana
Metoda \_load_config() nie obsługuje wszystkich wymaganych pól

Proponowane rozwiązanie
python# Zmiany w metodzie \_load_config w pliku app/gui/dialogs/fine_tuning_task_config_dialog.py

def \_load_config(self, config: Dict[str, Any]) -> None:
"""Ładuje konfigurację do interfejsu."""
try: # Model
model_config = config.get("model", {})

        # Logowanie wartości przed ustawieniem
        self.logger.info(f"Ładowanie konfiguracji - Nazwa zadania: {self.name_edit.text()}")
        self.logger.info(f"Ładowanie konfiguracji - Liczba klas: {self.num_classes_spin.value()}")

        # Ustawienie wartości z konfiguracji
        if "name" in config:
            self.name_edit.setText(config["name"])

        # 1. Aktualizacja parametrów modelu
        if "architecture" in model_config:
            self.arch_combo.setCurrentText(model_config["architecture"])
            # Po zmianie architektury aktualizujemy dostępne warianty
            self._update_variant_combo(model_config["architecture"])

        if "variant" in model_config:
            self.variant_combo.setCurrentText(model_config["variant"])

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

        # Logowanie wartości po ustawieniu
        self.logger.info(f"Zaktualizowano wartości - Nazwa zadania: {self.name_edit.text()}")
        self.logger.info(f"Zaktualizowano wartości - Liczba klas: {self.num_classes_spin.value()}")

        # 2. Aktualizacja parametrów treningu
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

        # 3. Aktualizacja parametrów regularyzacji
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

        # 4. Aktualizacja parametrów augmentacji
        augmentation_config = config.get("augmentation", {})

        # Basic augmentation
        basic_config = augmentation_config.get("basic", {})
        if "use" in basic_config:
            self.basic_aug_check.setChecked(basic_config["use"])

        if "rotation" in basic_config:
            self.rotation_spin.setValue(basic_config["rotation"])

        if "brightness" in basic_config:
            self.brightness_spin.setValue(basic_config["brightness"])

        if "shift" in basic_config:
            self.shift_spin.setValue(basic_config["shift"])

        if "zoom" in basic_config:
            self.zoom_spin.setValue(basic_config["zoom"])

        if "horizontal_flip" in basic_config:
            self.horizontal_flip_check.setChecked(basic_config["horizontal_flip"])

        if "vertical_flip" in basic_config:
            self.vertical_flip_check.setChecked(basic_config["vertical_flip"])

        # Mixup
        mixup_config = augmentation_config.get("mixup", {})
        if "use" in mixup_config:
            self.mixup_check.setChecked(mixup_config["use"])

        if "alpha" in mixup_config:
            self.mixup_alpha_spin.setValue(mixup_config["alpha"])

        # CutMix
        cutmix_config = augmentation_config.get("cutmix", {})
        if "use" in cutmix_config:
            self.cutmix_check.setChecked(cutmix_config["use"])

        if "alpha" in cutmix_config:
            self.cutmix_alpha_spin.setValue(cutmix_config["alpha"])

        # 5. Aktualizacja parametrów preprocessingu
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

        # 6. Aktualizacja parametrów monitorowania
        monitoring_config = config.get("monitoring", {})

        # Early stopping
        early_stopping_config = monitoring_config.get("early_stopping", {})
        if "patience" in early_stopping_config:
            self.patience_spin.setValue(early_stopping_config["patience"])

        if "monitor" in early_stopping_config:
            self.monitor_combo.setCurrentText(early_stopping_config["monitor"])

        # Checkpointing
        checkpointing_config = monitoring_config.get("checkpointing", {})
        if "metric" in checkpointing_config:
            self.checkpoint_metric_combo.setCurrentText(checkpointing_config["metric"])

        self.logger.info("Konfiguracja modelu została pomyślnie załadowana")

    except Exception as e:
        msg = "Błąd podczas ładowania konfiguracji"
        self.logger.error(f"{msg}: {str(e)}", exc_info=True)
        QtWidgets.QMessageBox.critical(self, "Błąd", f"{msg}: {str(e)}")

Wyjaśnienie zmian

Dodano brakujące warunki sprawdzające obecność parametrów w konfiguracji przed próbą ich ustawienia
Uporządkowano i dodano brakujące kategorie parametrów zgodnie z wymaganiami:

Parametry modelu (architecture, variant, input_size, num_classes, pretrained)
Parametry treningu (batch_size, learning_rate, optimizer, scheduler, unfreeze_layers, unfreeze_strategy, mixed_precision)
Parametry regularyzacji (weight_decay, drop_connect_rate, dropout_rate, label_smoothing, swa)
Parametry augmentacji (basic, mixup, cutmix)
Parametry preprocessingu (normalization)
Parametry monitorowania (early_stopping, checkpointing)

Dodano aktualizację combo box-a z wariantami po zmianie architektury
Poprawiono obsługę zaawansowanych parametrów jak normalizacja, gdzie wartości są przechowywane w listach

Po wprowadzeniu tych zmian, dialog powinien poprawnie wczytywać wszystkie wymagane parametry z pliku konfiguracyjnego modelu i aktualizować interfejs użytkownika.
