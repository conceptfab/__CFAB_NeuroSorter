Analiza zmian w pliku fine_tuning_task_config_dialog.py
Przeprowadziłem analizę obu plików i zidentyfikowałem zmiany, które należy wprowadzić w pliku fine_tuning_task_config_dialog.py, aby zapewnić jego prawidłowe działanie przy zachowaniu zbliżonej funkcjonalności do training_task_config_dialog.py.

1. Poprawki w funkcji \_select_train_dir()
   python# Plik: app/gui/dialogs/fine_tuning_task_config_dialog.py

# Funkcja: \_select_train_dir()

def \_select_train_dir(self):
"""Wybiera katalog z danymi treningowymi."""
try:
dir_path = QtWidgets.QFileDialog.getExistingDirectory(
self,
"Wybierz katalog z danymi treningowymi",
str(Path.home()),
QtWidgets.QFileDialog.Option.ShowDirsOnly,
)
if dir_path:
if validate_training_directory(dir_path):
self.train_dir_edit.setText(dir_path)
self.logger.info(f"Wybrano katalog treningowy: {dir_path}")
else:
QtWidgets.QMessageBox.warning(
self,
"Błąd walidacji",
"Wybrany katalog nie spełnia wymagań dla danych treningowych.",
)
except Exception as e:
self.logger.error(f"Błąd podczas wyboru katalogu treningowego: {str(e)}")
QtWidgets.QMessageBox.critical(
self,
"Błąd",
f"Nie można wybrać katalogu treningowego: {str(e)}",
) 2. Dodanie funkcji \_apply_profile()
Brakuje prawidłowo zaimplementowanej funkcji \_apply_profile(). Aktualny kod wymaga argumentu profile_name, ale nie używa go do wczytania profilu, tylko ma sztywno zdefiniowane szablony profili:
python# Plik: app/gui/dialogs/fine_tuning_task_config_dialog.py

# Funkcja: \_apply_profile()

def \_apply_profile(self):
"""Stosuje wybrany profil konfiguracji."""
if not self.current_profile:
QtWidgets.QMessageBox.warning(
self, "Ostrzeżenie", "Najpierw wybierz profil do zastosowania."
)
return

    try:
        config = self.current_profile.get("config", {})

        # Model
        if "model" in config:
            model_config = config["model"]
            self.pretrained_check.setChecked(model_config.get("pretrained", True))
            self.pretrained_weights_combo.setCurrentText(
                model_config.get("pretrained_weights", "imagenet")
            )
            self.feature_extraction_check.setChecked(
                model_config.get("feature_extraction_only", False)
            )
            self.activation_combo.setCurrentText(
                model_config.get("activation", "swish")
            )
            self.dropout_at_inference_check.setChecked(
                model_config.get("dropout_at_inference", False)
            )
            self.global_pool_combo.setCurrentText(
                model_config.get("global_pool", "avg")
            )
            self.last_layer_activation_combo.setCurrentText(
                model_config.get("last_layer_activation", "softmax")
            )

        # Training
        if "training" in config:
            training_config = config["training"]
            self.warmup_lr_init_spin.setValue(
                training_config.get("warmup_lr_init", 0.000001)
            )
            self.grad_accum_steps_spin.setValue(
                training_config.get("gradient_accumulation_steps", 1)
            )
            self.validation_split_spin.setValue(
                training_config.get("validation_split", 0.2)
            )
            self.eval_freq_spin.setValue(
                training_config.get("evaluation_freq", 1)
            )
            self.use_ema_check.setChecked(training_config.get("use_ema", False))
            self.ema_decay_spin.setValue(training_config.get("ema_decay", 0.9999))

            # Parametry treningu
            self.epochs_spin.setValue(training_config.get("epochs", 100))
            self.batch_size_spin.setValue(training_config.get("batch_size", 32))
            self.lr_spin.setValue(training_config.get("learning_rate", 0.001))
            self.optimizer_combo.setCurrentText(
                training_config.get("optimizer", "Adam")
            )
            self.scheduler_combo.setCurrentText(
                training_config.get("scheduler", "None")
            )
            self.num_workers_spin.setValue(training_config.get("num_workers", 4))
            self.warmup_epochs_spin.setValue(
                training_config.get("warmup_epochs", 5)
            )

        # Regularization
        if "regularization" in config:
            reg_config = config["regularization"]
            self.weight_decay_spin.setValue(reg_config.get("weight_decay", 0.0001))
            self.gradient_clip_spin.setValue(reg_config.get("gradient_clip", 1.0))
            self.label_smoothing_spin.setValue(
                reg_config.get("label_smoothing", 0.1)
            )
            self.drop_connect_spin.setValue(
                reg_config.get("drop_connect_rate", 0.2)
            )
            self.dropout_spin.setValue(reg_config.get("dropout_rate", 0.2))
            self.momentum_spin.setValue(reg_config.get("momentum", 0.9))
            self.epsilon_spin.setValue(reg_config.get("epsilon", 1e-6))

            # SWA
            swa_config = reg_config.get("swa", {})
            self.use_swa_check.setChecked(swa_config.get("use", False))
            self.swa_start_epoch_spin.setValue(swa_config.get("start_epoch", 10))

            # Stochastic Depth
            stoch_depth_config = reg_config.get("stochastic_depth", {})
            self.use_stoch_depth_check.setChecked(
                stoch_depth_config.get("use_stochastic_depth", False)
            )
            self.stoch_depth_drop_rate.setValue(
                stoch_depth_config.get("drop_rate", 0.2)
            )
            self.stoch_depth_survival_prob.setValue(
                stoch_depth_config.get("survival_probability", 0.8)
            )

            # Random Erase
            random_erase_config = reg_config.get("random_erase", {})
            self.use_random_erase_check.setChecked(
                random_erase_config.get("use_random_erase", False)
            )
            self.random_erase_prob.setValue(
                random_erase_config.get("probability", 0.25)
            )
            self.random_erase_mode.setCurrentText(
                random_erase_config.get("mode", "pixel")
            )

        # Augmentation
        if "augmentation" in config:
            aug_config = config["augmentation"]
            self.contrast_spin.setValue(aug_config.get("contrast", 0.2))
            self.saturation_spin.setValue(aug_config.get("saturation", 0.2))
            self.hue_spin.setValue(aug_config.get("hue", 0.1))
            self.shear_spin.setValue(aug_config.get("shear", 0.1))
            self.channel_shift_spin.setValue(
                aug_config.get("channel_shift_range", 0.0)
            )
            self.resize_mode_combo.setCurrentText(
                aug_config.get("resize_mode", "bilinear")
            )

            # Normalization
            norm_config = aug_config.get("normalization", {})
            mean = norm_config.get("mean", [0.485, 0.456, 0.406])
            std = norm_config.get("std", [0.229, 0.224, 0.225])
            self.norm_mean_r.setValue(mean[0])
            self.norm_mean_g.setValue(mean[1])
            self.norm_mean_b.setValue(mean[2])
            self.norm_std_r.setValue(std[0])
            self.norm_std_g.setValue(std[1])
            self.norm_std_b.setValue(std[2])

        # Monitoring
        if "monitoring" in config:
            monitor_config = config["monitoring"]
            metrics_config = monitor_config.get("metrics", {})
            self.accuracy_check.setChecked(metrics_config.get("accuracy", True))
            self.precision_check.setChecked(metrics_config.get("precision", True))
            self.recall_check.setChecked(metrics_config.get("recall", True))
            self.f1_check.setChecked(metrics_config.get("f1", True))
            self.topk_check.setChecked(metrics_config.get("top_k_accuracy", True))
            self.confusion_matrix_check.setChecked(
                metrics_config.get("confusion_matrix", True)
            )
            self.auc_check.setChecked(metrics_config.get("auc", True))

            # Logging
            logging_config = monitor_config.get("logging", {})
            self.use_tensorboard_check.setChecked(
                logging_config.get("use_tensorboard", True)
            )
            self.use_wandb_check.setChecked(logging_config.get("use_wandb", False))
            self.use_csv_check.setChecked(logging_config.get("save_to_csv", True))
            self.log_freq_combo.setCurrentText(
                logging_config.get("logging_freq", "epoch")
            )

        QtWidgets.QMessageBox.information(
            self, "Sukces", "Profil został pomyślnie zastosowany."
        )

    except Exception as e:
        self.logger.error(
            f"Błąd podczas stosowania profilu: {str(e)}", exc_info=True
        )
        QtWidgets.QMessageBox.critical(
            self, "Błąd", f"Nie można zastosować profilu: {str(e)}"
        )

3. Poprawienie funkcji \_on_accept()
   Funkcja \_on_accept() w pliku fine_tuning_task_config_dialog.py wymaga poprawienia, aby była zgodna z funkcją w pliku training_task_config_dialog.py:
   python# Plik: app/gui/dialogs/fine_tuning_task_config_dialog.py

# Funkcja: \_on_accept()

def \_on_accept(self):
"""Obsługa akceptacji konfiguracji."""
try: # Walidacja katalogu treningowego
train_dir = self.train_dir_edit.text()
if not train_dir.strip():
self.logger.warning("Nie wybrano katalogu treningowego")
QtWidgets.QMessageBox.critical(
self, "Błąd", "Musisz wybrać katalog danych treningowych!"
)
return

        if not validate_training_directory(train_dir):
            self.logger.error(f"Nieprawidłowy katalog treningowy: {train_dir}")
            return

        # Walidacja katalogu walidacyjnego
        val_dir = self.val_dir_edit.text()
        if val_dir and not validate_validation_directory(val_dir):
            self.logger.error(f"Nieprawidłowy katalog walidacyjny: {val_dir}")
            return

        # Walidacja ścieżki modelu do doszkalania
        model_path = self.model_path_edit.text()
        if not model_path.strip():
            self.logger.warning("Nie wybrano pliku modelu do doszkalania")
            QtWidgets.QMessageBox.critical(
                self, "Błąd", "Musisz wybrać plik modelu do doszkalania!"
            )
            return

        # Przygotowanie konfiguracji
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{self.arch_combo.currentText()}_{self.variant_combo.currentText()}"
        task_name = f"{model_name}_finetune_{timestamp}"

        # Sprawdź czy zadanie już istnieje
        if self._check_task_exists(task_name):
            QtWidgets.QMessageBox.warning(
                self,
                "Ostrzeżenie",
                f"Zadanie o nazwie {task_name} już istnieje. Wybierz inną nazwę.",
            )
            return

        config = {
            "name": task_name,
            "type": "doszkalanie",
            "status": "Nowy",
            "priority": 0,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "train_dir": train_dir,
                "data_dir": train_dir,
                "val_dir": val_dir,
                "model": {
                    "model_path": model_path,
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
                    "mixed_precision": self.mixed_precision_check.isChecked(),
                    "warmup_lr_init": self.warmup_lr_init_spin.value(),
                    "gradient_accumulation_steps": self.grad_accum_steps_spin.value(),
                    "validation_split": self.validation_split_spin.value(),
                    "evaluation_freq": self.eval_freq_spin.value(),
                    "use_ema": self.use_ema_check.isChecked(),
                    "ema_decay": self.ema_decay_spin.value(),
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
                        "use_swa": self.use_swa_check.isChecked(),
                        "start_epoch": self.swa_start_epoch_spin.value(),
                    },
                    "stochastic_depth": {
                        "use_stochastic_depth": self.use_stoch_depth_check.isChecked(),
                        "drop_rate": self.stoch_depth_drop_rate.value(),
                        "survival_probability": self.stoch_depth_survival_prob.value(),
                    },
                    "random_erase": {
                        "use_random_erase": self.use_random_erase_check.isChecked(),
                        "probability": self.random_erase_prob.value(),
                        "mode": self.random_erase_mode.currentText(),
                    },
                },
                "augmentation": {
                    "contrast": self.contrast_spin.value(),
                    "saturation": self.saturation_spin.value(),
                    "hue": self.hue_spin.value(),
                    "shear": self.shear_spin.value(),
                    "channel_shift_range": self.channel_shift_spin.value(),
                    "resize_mode": self.resize_mode_combo.currentText(),
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
                },
                "monitoring": {
                    "metrics": {
                        "accuracy": self.accuracy_check.isChecked(),
                        "precision": self.precision_check.isChecked(),
                        "recall": self.recall_check.isChecked(),
                        "f1": self.f1_check.isChecked(),
                        "top_k_accuracy": self.topk_check.isChecked(),
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
                        "use_prediction_samples": self.use_pred_samples_check.isChecked(),
                        "num_samples": self.num_samples_spin.value(),
                    },
                },
                "data": {
                    "class_weights": self.class_weights_combo.currentText(),
                    "sampler": self.sampler_combo.currentText(),
                    "image_channels": self.image_channels_spin.value(),
                    "cache_dataset": self.cache_dataset_check.isChecked(),
                },
                "inference": {
                    "tta": {
                        "use_tta": self.use_tta_check.isChecked(),
                        "num_augmentations": self.num_tta_spin.value(),
                    },
                    "export_onnx": self.export_onnx_check.isChecked(),
                    "quantization": {
                        "use_quantization": self.quantization_check.isChecked(),
                        "precision": self.quantization_precision_combo.currentText(),
                    },
                },
                "seed": self.seed_spin.value(),
                "deterministic": self.deterministic_check.isChecked(),
            },
        }

        # Walidacja konfiguracji
        is_valid, error_msg = self._validate_config(config)
        if not is_valid:
            QtWidgets.QMessageBox.critical(self, "Błąd walidacji", error_msg)
            return

        self.task_config = config

        # Zapisz konfigurację do pliku
        task_file = os.path.join("data", "tasks", f"{task_name}.json")
        os.makedirs(os.path.dirname(task_file), exist_ok=True)
        with open(task_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)

        QtWidgets.QMessageBox.information(
            self, "Sukces", "Zadanie zostało pomyślnie dodane."
        )
        self.accept()

    except Exception as e:
        msg = "Błąd podczas zapisywania konfiguracji"
        self.logger.error(f"{msg}: {str(e)}", exc_info=True)
        QtWidgets.QMessageBox.critical(self, "Błąd", f"{msg}: {str(e)}")

4. Poprawki w \_refresh_profile_list()
   Funkcja \_refresh_profile_list() powinna być rozszerzona o lepsze logowanie:
   python# Plik: app/gui/dialogs/fine_tuning_task_config_dialog.py

# Funkcja: \_refresh_profile_list()

def \_refresh_profile_list(self):
"""Odświeża listę dostępnych profili."""
self.profile_list.clear()
self.logger.debug("Rozpoczynam odświeżanie listy profili")
for profile_file in self.profiles_dir.glob("\*.json"):
try:
with open(profile_file, "r", encoding="utf-8") as f:
profile_data = json.load(f)
profile_type = profile_data.get("typ")
self.logger.debug(
f"Znaleziono profil {profile_file.stem} typu: {profile_type}"
)
if profile_type == "doszkalanie":
profile_name = profile_file.stem
self.profile_list.addItem(profile_name)
self.logger.debug(f"Dodano profil {profile_name} do listy")
else:
self.logger.debug(
f"Pominięto profil {profile_file.stem} - nieprawidłowy typ"
)
except Exception as e:
self.logger.error(
f"Błąd podczas wczytywania profilu {profile_file}: {str(e)}",
exc_info=True,
)
self.logger.debug(
f"Zakończono odświeżanie listy profili. Liczba profili: {self.profile_list.count()}"
) 5. Główne różnice i zalecane zmiany

Plik fine_tuning_task_config_dialog.py zawiera funkcje, które nie są w pełni zaimplementowane (np. \_apply_profile przyjmuje argument, ale nie jest on używany).
W fine_tuning_task_config_dialog.py znajduje się funkcja \_load_config, której nie ma w training_task_config_dialog.py. Ta funkcja powinna być dostosowana do obu plików.
Funkcja \_on_accept w fine_tuning_task_config_dialog.py nie zapisuje pliku zadania w odpowiednim miejscu i formacie.
Funkcja \_validate_config istnieje w fine_tuning_task_config_dialog.py, ale nie w training_task_config_dialog.py - należy rozważyć dodanie jej do drugiego pliku.
W fine_tuning_task_config_dialog.py brakuje odpowiednika funkcji \_check_task_exists, która została zaimplementowana tylko w \_on_accept.
Funkcja \_select_model_file jest specyficzna dla trybu doszkalania i jest poprawnie zaimplementowana.

Podsumowanie
Proponowane zmiany umożliwią prawidłowe działanie obu dialogów konfiguracyjnych, z zachowaniem ich specyficznych funkcji, ale ze zbliżonym interfejsem i logiką działania. Główne zmiany obejmują:

Poprawienie funkcji \_apply_profile, aby nie wymagała argumentu i korzystała z aktualnie wybranego profilu.
Uzupełnienie funkcji \_on_accept o prawidłowe zapisywanie pliku zadania.
Uspójnienie funkcji \_refresh_profile_list o lepsze logowanie.
Doprecyzowanie walidacji w funkcji \_select_train_dir i innych funkcjach wyboru plików/katalogów.
Rozważenie dodania funkcji \_validate_config do training_task_config_dialog.py.

Po wprowadzeniu tych zmian, oba dialogi powinny działać poprawnie, umożliwiając zarówno tworzenie nowych modeli, jak i doszkalanie istniejących, z zachowaniem podobnego interfejsu użytkownika i logiki działania.
