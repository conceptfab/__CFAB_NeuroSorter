Zmiany w pliku app/gui/dialogs/fine_tuning_task_config_dialog.py
Zidentyfikowałem kilka problemów w pliku fine_tuning_task_config_dialog.py w porównaniu z poprawnie działającym plikiem training_task_config_dialog.py. Poniżej przedstawiam potrzebne poprawki, które zapewnią prawidłowe działanie funkcji związanych z profilami:

1.  Zmiana funkcji \_refresh_profile_list
    pythondef \_refresh_profile_list(self):
    """Odświeża listę dostępnych profili."""
    try:
    self.profile_list.clear()
    for profile_file in self.profiles_dir.glob("\*.json"):
    try:
    with open(profile_file, "r", encoding="utf-8") as f:
    profile_data = json.load(f)
    if profile_data.get("typ") == "doszkalanie":
    name = profile_data.get("name", profile_file.stem)
    item = QtWidgets.QListWidgetItem(name)
    item.setData(Qt.ItemDataRole.UserRole, profile_file)
    self.profile_list.addItem(item)
    except Exception as e:
    self.logger.error(
    f"Błąd podczas wczytywania profilu {profile_file}: {str(e)}",
    exc_info=True,
    )
    except Exception as e:
    self.logger.error(f"Błąd podczas odświeżania listy profili: {str(e)}")
2.  Zmiana funkcji \_on_profile_selected
    pythondef \_on_profile_selected(self, current, previous):
    """Obsługa wyboru profilu z listy."""
    try:
    if current is None:
    self.profile_info.clear()
    self.profile_description.clear()
    self.profile_data_required.clear()
    self.profile_hardware_required.clear()
    self.current_profile = None
    return

            profile_file = current.data(Qt.ItemDataRole.UserRole)
            if not profile_file.exists():
                self.logger.error(f"Plik profilu nie istnieje: {profile_file}")
                return

            with open(profile_file, "r", encoding="utf-8") as f:
                profile_data = json.load(f)
                self.current_profile = profile_data
                self.profile_info.setText(profile_data.get("info", ""))
                self.profile_description.setText(profile_data.get("description", ""))
                self.profile_data_required.setText(profile_data.get("data_required", ""))
                self.profile_hardware_required.setText(
                    profile_data.get("hardware_required", "")
                )

        except Exception as e:
            self.logger.error(f"Błąd podczas wyboru profilu: {str(e)}", exc_info=True)
            self.profile_info.clear()
            self.profile_description.clear()
            self.profile_data_required.clear()
            self.profile_hardware_required.clear()
            self.current_profile = None

3.  Implementacja funkcji \_edit_profile
    pythondef \_edit_profile(self):
    """Edycja wybranego profilu."""
    if not self.current_profile:
    QtWidgets.QMessageBox.warning(
    self, "Ostrzeżenie", "Najpierw wybierz profil do edycji."
    )
    return

        try:
            profile_path = self.profiles_dir / f"{self.profile_list.currentItem().text()}.json"
            os.startfile(str(profile_path))  # Dla Windows
        except Exception as e:
            self.logger.error(
                f"Błąd podczas otwierania profilu: {str(e)}", exc_info=True
            )
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Nie można otworzyć profilu: {str(e)}"
            )

4.  Implementacja funkcji \_apply_profile
    pythondef \_apply_profile(self):
    """Zastosowanie wybranego profilu do konfiguracji."""
    if not self.current_profile:
    QtWidgets.QMessageBox.warning(
    self, "Ostrzeżenie", "Najpierw wybierz profil do zastosowania."
    )
    return

        try:
            config = self.current_profile.get("config", {})

            # Dane i Model
            if "model" in config:
                model_config = config["model"]
                self.arch_combo.setCurrentText(
                    model_config.get("architecture", "EfficientNet")
                )
                self.variant_combo.setCurrentText(
                    model_config.get("variant", "B0")
                )
                self.input_size_spin.setValue(model_config.get("input_size", 224))
                self.num_classes_spin.setValue(model_config.get("num_classes", 2))

            # Parametry Treningu
            if "training" in config:
                training_config = config["training"]
                self.epochs_spin.setValue(training_config.get("epochs", 100))
                self.batch_size_spin.setValue(training_config.get("batch_size", 32))
                self.learning_rate_spin.setValue(training_config.get("learning_rate", 0.001))
                self.optimizer_combo.setCurrentText(
                    training_config.get("optimizer", "adam")
                )

            # Regularyzacja
            if "regularization" in config:
                reg_config = config["regularization"]
                self.weight_decay_spin.setValue(reg_config.get("weight_decay", 0.0001))
                self.dropout_spin.setValue(reg_config.get("dropout_rate", 0.2))

            # Augmentacja
            if "augmentation" in config:
                aug_config = config["augmentation"]
                basic_config = aug_config.get("basic", {})
                self.horizontal_flip_check.setChecked(
                    basic_config.get("horizontal_flip", True)
                )
                self.vertical_flip_check.setChecked(
                    basic_config.get("vertical_flip", False)
                )
                self.rotation_check.setChecked(
                    basic_config.get("rotation", True)
                )

            # Monitorowanie
            if "monitoring" in config:
                monitor_config = config["monitoring"]
                metrics_config = monitor_config.get("metrics", {})
                self.accuracy_check.setChecked(metrics_config.get("accuracy", True))
                self.precision_check.setChecked(metrics_config.get("precision", True))
                self.recall_check.setChecked(metrics_config.get("recall", True))
                self.f1_check.setChecked(metrics_config.get("f1", True))
                self.confusion_matrix_check.setChecked(
                    metrics_config.get("confusion_matrix", False)
                )
                self.roc_auc_check.setChecked(metrics_config.get("roc_auc", False))
                self.pr_auc_check.setChecked(metrics_config.get("pr_auc", False))
                self.top_k_check.setChecked(metrics_config.get("top_k", False))

            # PEFT
            if "peft" in config:
                peft_config = config["peft"]
                self.peft_technique.setCurrentText(
                    peft_config.get("technique", "none")
                )

                lora_config = peft_config.get("lora", {})
                self.lora_rank.setValue(lora_config.get("rank", 8))
                self.lora_alpha.setValue(lora_config.get("alpha", 16))
                self.lora_dropout.setValue(lora_config.get("dropout", 0.1))
                self.lora_target_modules.setText(",".join(lora_config.get("target_modules", ["query","key","value"])))

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

5.  Implementacja funkcji \_clone_profile
    pythondef \_clone_profile(self):
    """Klonuje wybrany profil."""
    if not self.current_profile:
    QtWidgets.QMessageBox.warning(
    self, "Ostrzeżenie", "Najpierw wybierz profil do sklonowania."
    )
    return

        try:
            current_name = self.profile_list.currentItem().text()
            new_name, ok = QtWidgets.QInputDialog.getText(
                self,
                "Klonuj profil",
                "Podaj nazwę dla nowego profilu:",
                QtWidgets.QLineEdit.EchoMode.Normal,
                f"{current_name}_clone",
            )

            if ok and new_name:
                new_profile = self.current_profile.copy()
                new_profile["info"] = f"Klon profilu {current_name}"
                new_profile["description"] = f"Klon profilu {current_name}"
                new_profile["typ"] = "doszkalanie"  # Upewniamy się, że typ jest ustawiony

                new_path = self.profiles_dir / f"{new_name}.json"
                with open(new_path, "w", encoding="utf-8") as f:
                    json.dump(new_profile, f, indent=4, ensure_ascii=False)

                self._refresh_profile_list()
                QtWidgets.QMessageBox.information(
                    self, "Sukces", "Profil został pomyślnie sklonowany."
                )

        except Exception as e:
            self.logger.error(
                f"Błąd podczas klonowania profilu: {str(e)}", exc_info=True
            )
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Nie można sklonować profilu: {str(e)}"
            )

6.  Implementacja funkcji _save_profile
    pythondef \_save_profile(self):
    """Zapisuje aktualną konfigurację jako nowy profil."""
    try:
    name, ok = QtWidgets.QInputDialog.getText(
    self,
    "Zapisz profil",
    "Podaj nazwę dla nowego profilu:",
    QtWidgets.QLineEdit.EchoMode.Normal,
    f"{self.arch_combo.currentText()}_{self.variant_combo.currentText()}",
    )

            if ok and name:
                profile_data = {
                    "typ": "doszkalanie",
                    "info": (
                        f"Profil dla {self.arch_combo.currentText()} "
                        f"{self.variant_combo.currentText()}"
                    ),
                    "description": "Profil utworzony przez użytkownika",
                    "data_required": "Standardowe dane treningowe",
                    "hardware_required": "Standardowy sprzęt",
                    "config": {
                        "model": {
                            "architecture": self.arch_combo.currentText(),
                            "variant": self.variant_combo.currentText(),
                            "input_size": self.input_size_spin.value(),
                            "num_classes": self.num_classes_spin.value(),
                        },
                        "training": {
                            "epochs": self.epochs_spin.value(),
                            "batch_size": self.batch_size_spin.value(),
                            "learning_rate": float(self.learning_rate_spin.value()),
                            "optimizer": self.optimizer_combo.currentText(),
                        },
                        "regularization": {
                            "weight_decay": float(self.weight_decay_spin.value()),
                            "dropout_rate": self.dropout_spin.value(),
                        },
                        "augmentation": {
                            "basic": {
                                "horizontal_flip": self.horizontal_flip_check.isChecked(),
                                "vertical_flip": self.vertical_flip_check.isChecked(),
                                "rotation": self.rotation_check.isChecked(),
                            },
                        },
                        "monitoring": {
                            "metrics": {
                                "accuracy": self.accuracy_check.isChecked(),
                                "precision": self.precision_check.isChecked(),
                                "recall": self.recall_check.isChecked(),
                                "f1": self.f1_check.isChecked(),
                                "confusion_matrix": self.confusion_matrix_check.isChecked(),
                                "roc_auc": self.roc_auc_check.isChecked(),
                                "pr_auc": self.pr_auc_check.isChecked(),
                                "top_k": self.top_k_check.isChecked(),
                            },
                        },
                        "peft": {
                            "technique": self.peft_technique.currentText(),
                            "lora": {
                                "rank": self.lora_rank.value(),
                                "alpha": self.lora_alpha.value(),
                                "dropout": self.lora_dropout.value(),
                                "target_modules": self.lora_target_modules.text().split(","),
                            },
                        },
                    },
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

7.  Implementacja funkcji \_delete_profile
    pythondef \_delete_profile(self):
    """Usuwa wybrany profil."""
    if not self.current_profile:
    QtWidgets.QMessageBox.warning(
    self, "Ostrzeżenie", "Najpierw wybierz profil do usunięcia."
    )
    return

        try:
            current_name = self.profile_list.currentItem().text()
            reply = QtWidgets.QMessageBox.question(
                self,
                "Potwierdzenie",
                f"Czy na pewno chcesz usunąć profil '{current_name}'?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )

            if reply == QtWidgets.QMessageBox.StandardButton.Yes:
                profile_path = self.profiles_dir / f"{current_name}.json"
                if profile_path.exists():
                    profile_path.unlink()
                    self._refresh_profile_list()
                    self.current_profile = None
                    self.profile_info.clear()
                    self.profile_description.clear()
                    self.profile_data_required.clear()
                    self.profile_hardware_required.clear()
                    QtWidgets.QMessageBox.information(
                        self, "Sukces", "Profil został pomyślnie usunięty."
                    )

        except Exception as e:
            self.logger.error(f"Błąd podczas usuwania profilu: {str(e)}", exc_info=True)
            QtWidgets.QMessageBox.critical(
                self, "Błąd", f"Nie można usunąć profilu: {str(e)}"
            )

8.  Dodanie importu os
    Dodaj import modułu os na początku pliku:
    pythonimport os
9.  Uzupełnienie funkcji \_validate_basic_params
    pythondef \_validate_basic_params(self):
    """Walidacja podstawowych parametrów."""
    try:
    if not self.train_dir_edit.text():
    QtWidgets.QMessageBox.warning(
    self, "Ostrzeżenie", "Wybierz katalog z danymi treningowymi."
    )
    return False

            if not self.val_dir_edit.text():
                QtWidgets.QMessageBox.warning(
                    self, "Ostrzeżenie", "Wybierz katalog z danymi walidacyjnymi."
                )
                return False

            return True

        except Exception as e:
            self.logger.error(f"Błąd podczas walidacji parametrów: {str(e)}")
            return False

    Te zmiany poprawią funkcjonalność zarządzania profilami w dialogu konfiguracji zadania fine-tuningu. Główne problemy w oryginalnym pliku były związane z brakiem implementacji tych funkcji lub niewłaściwym użyciem mechanizmu przechowywania danych w elementach QListWidgetItem.
