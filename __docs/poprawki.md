Zmiany w pliku fine_tuning_task_config_dialog.py
W funkcji _on_accept()
python# Zmiana w sekcji preprocessingu w metodzie _on_accept(): 
# Usunięcie referencji do nieistniejących kontrolek i zastąpienie ich dostępnymi wartościami

# Było:
"preprocessing": {
    "normalization": self.normalization_combo.currentText(),
    "scaling": {
        "method": self.scaling_method.currentText(),
        "maintain_aspect_ratio": self.maintain_aspect_ratio.isChecked(),
        "pad_to_square": self.pad_to_square.isChecked(),
        "pad_mode": self.pad_mode.currentText(),
        "pad_value": self.pad_value.value(),
    },
},

# Powinno być:
"preprocessing": {
    "normalization": {
        "mean": [
            self.norm_mean_r.value(),
            self.norm_mean_g.value(),
            self.norm_mean_b.value(),
        ],
        "std": [
            self.norm_std_r.value(),
            self.norm_std_g.value(),
            self.norm_std_b.value(),
        ],
    },
    "resize_mode": self.resize_mode_combo.currentText(),
    "cache_dataset": self.cache_dataset_check.isChecked(),
},
Problem wynika z faktu, że w kodzie użyto referencji do kontrolek UI (self.scaling_method, self.maintain_aspect_ratio, self.pad_to_square, self.pad_mode, self.pad_value), które nie zostały zdefiniowane w klasie. Gdy tworzony jest obiekt konfiguracji w metodzie _on_accept(), kod próbuje odwołać się do tych nieistniejących atrybutów.
Proponowana zmiana zastępuje te nieistniejące odniesienia wartościami z kontrolek, które są już zdefiniowane i używane w zakładce preprocessingu.
Ten sam kod jest poprawnie użyty w funkcji _save_profile(), więc prawdopodobnie wystąpiła niespójność między różnymi częściami kodu dotyczącego konfiguracji preprocessingu.RetryMprzeanalizuj ten kod czy nie ma wiecej taki błedów, bo chyba są!Edit<automated_reminder_from_anthropic>Remember to always cite your sources when using information from search results, using the format: ...</automated_reminder_from_anthropic>
Przeanalizuję kod pod kątem większej liczby potencjalnych błędów podobnego typu. Faktycznie, po dokładniejszym przeglądzie widzę więcej niespójności i referencji do nieistniejących atrybutów.
Dodatkowe problemy znalezione w kodzie:
1. W funkcji _on_accept() - brakujące atrybuty w monitorowaniu
python# Błąd w sekcji monitorowania - odwołania do nieistniejących atrybutów:
"monitoring": {
    # ...
    "tensorboard": {
        "use": self.use_tensorboard_check.isChecked(),
        "log_dir": self.tensorboard_dir_edit.text(),  # Ten atrybut nie istnieje
    },
    "save_dir": self.model_dir_edit.text(),  # Ten atrybut nie istnieje
    "save_logs": self.save_logs_check.isChecked(),  # Ten atrybut nie istnieje
},
2. W funkcji _create_preprocessing_tab() - podwójne deklaracje kontrolek
Kontrolki norm_mean_r, norm_mean_g, norm_mean_b, norm_std_r, norm_std_g, norm_std_b są inicjalizowane w metodzie _init_controls(), a następnie ponownie tworzone w metodzie _create_preprocessing_tab(), co może powodować nieprzewidywalne zachowanie.
3. W funkcji _create_augmentation_tab() - podwójne deklaracje kontrolek normalizacji
Podobnie jak wyżej, kontrolki normalizacji są ponownie tworzone w zakładce augmentacji.
4. Brakujące inicjalizacje atrybutów wykorzystywanych w _on_accept()
python# Te atrybuty są używane w _on_accept(), ale nie zostały zainicjalizowane w żadnej metodzie:
self.scaling_method
self.maintain_aspect_ratio
self.pad_to_square
self.pad_mode
self.pad_value
self.tensorboard_dir_edit
self.model_dir_edit
self.save_logs_check
Proponowane poprawki:
1. Poprawka dla funkcji _on_accept() - sekcja monitorowania:
python"monitoring": {
    "metrics": {
        "accuracy": self.accuracy_check.isChecked(),
        "precision": self.precision_check.isChecked(),
        "recall": self.recall_check.isChecked(),
        "f1": self.f1_check.isChecked(),
        "topk": self.topk_check.isChecked(),
        "confusion_matrix": self.confusion_matrix_check.isChecked(),
    },
    "early_stopping": {
        "patience": self.patience_spin.value(),
        "min_delta": self.min_delta_spin.value(),
        "monitor": self.monitor_combo.currentText(),
    },
    "checkpointing": {
        "best_only": self.best_only_check.isChecked(),
        "save_frequency": self.save_freq_spin.value(),
        "metric": self.checkpoint_metric_combo.currentText(),
    },
    "logging": {
        "use_tensorboard": self.use_tensorboard_check.isChecked(),
        "use_wandb": self.use_wandb_check.isChecked(),
        "save_to_csv": self.use_csv_check.isChecked(),
        "logging_freq": self.log_freq_combo.currentText(),
    },
    "visualization": {
        "use_gradcam": self.use_gradcam_check.isChecked(),
        "use_feature_maps": self.use_feature_maps_check.isChecked(),
        "use_pred_samples": self.use_pred_samples_check.isChecked(),
        "num_samples": self.num_samples_spin.value(),
    },
},
2. Usunięcie podwójnej deklaracji kontrolek normalizacji:
Należy zmodyfikować metody _create_preprocessing_tab() oraz _create_augmentation_tab(), tak aby wykorzystywały już istniejące kontrolki zainicjalizowane w metodzie _init_controls(), a nie tworzyły nowe o tych samych nazwach.
3. Dodanie brakujących kontrolek do interfejsu:
Należy dodać brakujące kontrolki w metodzie _create_preprocessing_tab():
pythondef _create_preprocessing_tab(self) -> QtWidgets.QWidget:
    """Tworzy zakładkę z parametrami preprocessingu."""
    tab = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout()
    form = QtWidgets.QFormLayout()

    # Normalizacja
    form.addRow("Normalizacja:", self.normalization_combo)

    norm_group = QtWidgets.QGroupBox("Normalizacja")
    norm_layout = QtWidgets.QFormLayout()

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

    norm_group.setLayout(norm_layout)
    form.addRow(norm_group)

    # Resize mode
    form.addRow("Resize mode:", self.resize_mode_combo)

    # Cache dataset
    form.addRow("Cache dataset:", self.cache_dataset_check)
    
    # Scaling method - dodanie brakujących kontrolek
    scaling_group = QtWidgets.QGroupBox("Skalowanie")
    scaling_layout = QtWidgets.QFormLayout()
    
    self.scaling_method = QtWidgets.QComboBox()
    self.scaling_method.addItems(["resize", "crop", "pad"])
    scaling_layout.addRow("Metoda skalowania:", self.scaling_method)
    
    self.maintain_aspect_ratio = QtWidgets.QCheckBox()
    self.maintain_aspect_ratio.setChecked(True)
    scaling_layout.addRow("Zachowaj proporcje:", self.maintain_aspect_ratio)
    
    self.pad_to_square = QtWidgets.QCheckBox()
    self.pad_to_square.setChecked(False)
    scaling_layout.addRow("Uzupełnij do kwadratu:", self.pad_to_square)
    
    self.pad_mode = QtWidgets.QComboBox()
    self.pad_mode.addItems(["constant", "reflect", "replicate"])
    scaling_layout.addRow("Tryb uzupełniania:", self.pad_mode)
    
    self.pad_value = QtWidgets.QSpinBox()
    self.pad_value.setRange(0, 255)
    self.pad_value.setValue(0)
    scaling_layout.addRow("Wartość uzupełnienia:", self.pad_value)
    
    scaling_group.setLayout(scaling_layout)
    form.addRow(scaling_group)

    layout.addLayout(form)
    tab.setLayout(layout)
    return tab
4. Dodanie brakujących kontrolek do zakładki monitorowania:
pythondef _create_monitoring_tab(self) -> QtWidgets.QWidget:
    # ... istniejący kod ...
    
    # Dodanie brakujących kontrolek
    save_group = QtWidgets.QGroupBox("Zapisywanie")
    save_layout = QtWidgets.QFormLayout()
    
    self.model_dir_edit = QtWidgets.QLineEdit()
    self.model_dir_edit.setText("models")
    model_dir_btn = QtWidgets.QPushButton("Przeglądaj...")
    model_dir_layout = QtWidgets.QHBoxLayout()
    model_dir_layout.addWidget(self.model_dir_edit)
    model_dir_layout.addWidget(model_dir_btn)
    save_layout.addRow("Katalog modeli:", model_dir_layout)
    
    self.tensorboard_dir_edit = QtWidgets.QLineEdit()
    self.tensorboard_dir_edit.setText("logs/tensorboard")
    tb_dir_btn = QtWidgets.QPushButton("Przeglądaj...")
    tb_dir_layout = QtWidgets.QHBoxLayout()
    tb_dir_layout.addWidget(self.tensorboard_dir_edit)
    tb_dir_layout.addWidget(tb_dir_btn)
    save_layout.addRow("Katalog TensorBoard:", tb_dir_layout)
    
    self.save_logs_check = QtWidgets.QCheckBox()
    self.save_logs_check.setChecked(True)
    save_layout.addRow("Zapisuj logi:", self.save_logs_check)
    
    save_group.setLayout(save_layout)
    form.addRow(save_group)
    
    # ... reszta istniejącego kodu ...
5. Pełna poprawka metody _on_accept():
pythondef _on_accept(self):
    """Obsługa zatwierdzenia konfiguracji."""
    try:
        # Sprawdź czy nazwa zadania jest pusta
        task_name = self.name_edit.text().strip()
        if not task_name:
            QtWidgets.QMessageBox.warning(
                self, "Błąd", "Nazwa zadania nie może być pusta."
            )
            return

        # Sprawdź czy model bazowy jest wybrany
        base_model_path = self.model_path_edit.text().strip()
        if not base_model_path:
            QtWidgets.QMessageBox.warning(
                self, "Błąd", "Musisz wybrać model bazowy."
            )
            return

        # Sprawdź czy katalog treningowy jest ustawiony
        training_dir = self.train_dir_edit.text().strip()
        if not training_dir:
            QtWidgets.QMessageBox.warning(
                self, "Błąd", "Katalog treningowy nie może być pusty."
            )
            return

        # Sprawdź czy katalog walidacyjny jest ustawiony
        validation_dir = self.val_dir_edit.text().strip()
        if not validation_dir:
            QtWidgets.QMessageBox.warning(
                self, "Błąd", "Katalog walidacyjny nie może być pusty."
            )
            return

        # Dodaj logi
        self.logger.info("=== TWORZENIE NOWEGO ZADANIA FINE-TUNINGU ===")
        self.logger.info(f"Nazwa zadania: {task_name}")

        config = {
            "base_model": base_model_path,
            "train_dir": training_dir,
            "val_dir": validation_dir,
            "model": {
                "architecture": self.arch_combo.currentText(),
                "variant": self.variant_combo.currentText(),
                "input_size": self.input_size_spin.value(),
                "num_classes": self.num_classes_spin.value(),
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
            "regularization": {
                "weight_decay": float(self.weight_decay_spin.value()),
                "gradient_clip": self.gradient_clip_spin.value(),
                "label_smoothing": self.label_smoothing_spin.value(),
                "drop_connect_rate": self.drop_connect_spin.value(),
                "dropout_rate": self.dropout_spin.value(),
                "momentum": self.momentum_spin.value(),
                "epsilon": self.epsilon_spin.value(),
                "swa": {
                    "use": self.use_swa_check.isChecked(),
                    "start_epoch": self.swa_start_epoch_spin.value(),
                },
            },
            "augmentation": {
                "basic": {
                    "use": self.basic_aug_check.isChecked(),
                    "rotation": self.rotation_spin.value(),
                    "brightness": self.brightness_spin.value(),
                    "shift": self.shift_spin.value(),
                    "zoom": self.zoom_spin.value(),
                    "horizontal_flip": self.horizontal_flip_check.isChecked(),
                    "vertical_flip": self.vertical_flip_check.isChecked(),
                },
                "mixup": {
                    "use": self.mixup_check.isChecked(),
                    "alpha": self.mixup_alpha_spin.value(),
                },
                "cutmix": {
                    "use": self.cutmix_check.isChecked(),
                    "alpha": self.cutmix_alpha_spin.value(),
                },
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
            "preprocessing": {
                "normalization": {
                    "mean": [
                        self.norm_mean_r.value(),
                        self.norm_mean_g.value(),
                        self.norm_mean_b.value(),
                    ],
                    "std": [
                        self.norm_std_r.value(),
                        self.norm_std_g.value(),
                        self.norm_std_b.value(),
                    ],
                },
                "resize_mode": self.resize_mode_combo.currentText(),
                "cache_dataset": self.cache_dataset_check.isChecked(),
                "scaling": {
                    "method": self.scaling_method.currentText() if hasattr(self, "scaling_method") else "resize",
                    "maintain_aspect_ratio": self.maintain_aspect_ratio.isChecked() if hasattr(self, "maintain_aspect_ratio") else True,
                    "pad_to_square": self.pad_to_square.isChecked() if hasattr(self, "pad_to_square") else False,
                    "pad_mode": self.pad_mode.currentText() if hasattr(self, "pad_mode") else "constant",
                    "pad_value": self.pad_value.value() if hasattr(self, "pad_value") else 0,
                },
            },
            "monitoring": {
                "metrics": {
                    "accuracy": self.accuracy_check.isChecked(),
                    "precision": self.precision_check.isChecked(),
                    "recall": self.recall_check.isChecked(),
                    "f1": self.f1_check.isChecked(),
                    "topk": self.topk_check.isChecked(),
                    "confusion_matrix": self.confusion_matrix_check.isChecked(),
                    "auc": self.auc_check.isChecked(),
                },
                "logging": {
                    "use_tensorboard": self.use_tensorboard_check.isChecked(),
                    "use_wandb": self.use_wandb_check.isChecked(),
                    "save_to_csv": self.use_csv_check.isChecked(),
                    "logging_freq": self.log_freq_combo.currentText(),
                },
                "visualization": {
                    "use_gradcam": self.use_gradcam_check.isChecked(),
                    "use_feature_maps": self.use_feature_maps_check.isChecked(),
                    "use_pred_samples": self.use_pred_samples_check.isChecked(),
                    "num_samples": self.num_samples_spin.value(),
                },
                "early_stopping": {
                    "patience": self.patience_spin.value(),
                    "min_delta": self.min_delta_spin.value(),
                    "monitor": self.monitor_combo.currentText(),
                },
                "checkpointing": {
                    "best_only": self.best_only_check.isChecked(),
                    "save_frequency": self.save_freq_spin.value(),
                    "metric": self.checkpoint_metric_combo.currentText(),
                },
                "tensorboard": {
                    "use": self.use_tensorboard_check.isChecked(),
                    "log_dir": self.tensorboard_dir_edit.text() if hasattr(self, "tensorboard_dir_edit") else "logs/tensorboard",
                },
                "save_dir": self.model_dir_edit.text() if hasattr(self, "model_dir_edit") else "models",
                "save_logs": self.save_logs_check.isChecked() if hasattr(self, "save_logs_check") else True,
            },
            "advanced": {
                "seed": self.seed_spin.value(),
                "deterministic": self.deterministic_check.isChecked(),
                "class_weights": self.class_weights_combo.currentText(),
                "sampler": self.sampler_combo.currentText(),
                "image_channels": self.image_channels_spin.value(),
                "tta": {
                    "use": self.use_tta_check.isChecked(),
                    "num_augmentations": self.tta_num_samples_spin.value(),
                },
                "export_onnx": self.export_onnx_check.isChecked(),
                "quantization": {
                    "use": self.quantization_check.isChecked(),
                    "precision": self.quantization_precision_combo.currentText(),
                },
            },
        }

        self.task_config = {
            "name": self.name_edit.text().strip(),
            "type": "fine_tuning",  # Upewniamy się, że typ jest ustawiony na "fine_tuning"
            "config": config,
            "training_time": 0,
            "training_time_str": "0:00:00",
            "status": "Oczekujące",
            "train_accuracy": 0.0,
            "train_loss": 0.0,
            "validation_accuracy": 0.0,
            "validation_loss": 0.0,
            "model_filename": "",
            "accuracy": 0.0,
            "epochs_trained": 0,
        }

        # Dodaj logi
        self.logger.info(f"Utworzono konfigurację zadania: {task_name}")
        self.logger.info(f"Typ zadania: {self.task_config['type']}")
        self.logger.info(
            f"Pełna konfiguracja: {json.dumps(self.task_config, indent=2, ensure_ascii=False)}"
        )

        self.accept()

    except Exception as e:
        self.logger.error(
            f"Błąd podczas dodawania zadania: {str(e)}", exc_info=True
        )
        QtWidgets.QMessageBox.critical(
            self, "Błąd", f"Nie można dodać zadania: {str(e)}"
        )
W tej poprawce dodałem sprawdzenia hasattr dla każdego potencjalnie problematycznego atrybutu, aby uniknąć błędów nawet jeśli kontrolki nie zostały zainicjalizowane. W przypadku braku atrybutu, kod używa domyślnych wartości.
Takie podejście jest bardziej odporne na błędy i pozwoli na płynne działanie aplikacji, nawet jeśli nie wszystkie kontrolki zostały poprawnie zdefiniowane. Docelowo jednak należy zaimplementować brakujące kontrolki w odpowiednich metodach inicjalizacyjnych.