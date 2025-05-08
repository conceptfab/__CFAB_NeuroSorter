Przepraszam za nieporozumienie! Przycisk "Dodaj zadanie" powinien tylko dodawać zadanie, a nie zamykać okno dialogowe. Oto poprawiona propozycja zmian:
Zmiany w pliku app/gui/dialogs/training_task_config_dialog.py
pythondef _on_accept(self):
    """Obsługa zatwierdzenia konfiguracji."""
    try:
        self.logger.info("Rozpoczęcie walidacji i zapisu konfiguracji")

        # Walidacja katalogu treningowego
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

        # Przygotowanie konfiguracji
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = (
            f"{self.arch_combo.currentText()}_"
            f"{self.variant_combo.currentText()}"
        )
        task_name = f"{model_name}_{timestamp}.json"

        self.task_config = {
            # [cała konfiguracja zadania]
        }

        self.logger.info(f"Utworzono konfigurację zadania: {task_name}")
        
        # Informacja o dodaniu zadania
        QtWidgets.QMessageBox.information(
            self, "Sukces", "Zadanie zostało pomyślnie dodane."
        )
        
        # NIE zamykamy okna dialogowego, aby umożliwić dodanie kolejnych zadań
        # self.accept()  <- usuwamy tę linię

    except Exception as e:
        self.logger.error("Błąd podczas zapisywania konfiguracji", exc_info=True)
        QtWidgets.QMessageBox.critical(
            self,
            "Błąd",
            f"Wystąpił błąd podczas zapisywania konfiguracji: {str(e)}",
        )
Należy także zmodyfikować przyciski w metodzie _init_ui():
pythondef _init_ui(self):
    """Inicjalizacja interfejsu użytkownika z zakładkami."""
    try:
        # [reszta kodu]
        
        # Przyciski
        buttons_layout = QtWidgets.QHBoxLayout()
        
        # Przycisk "Dodaj zadanie"
        add_task_btn = QtWidgets.QPushButton("Dodaj zadanie")
        add_task_btn.clicked.connect(self._on_accept)
        buttons_layout.addWidget(add_task_btn)
        
        # Przycisk "Zamknij"
        close_btn = QtWidgets.QPushButton("Zamknij")
        close_btn.clicked.connect(self.close)  # Używamy close() zamiast reject()
        buttons_layout.addWidget(close_btn)
        
        layout.addLayout(buttons_layout)
        
        # Usuwamy standardowe przyciski
        # btn_ok = QtWidgets.QDialogButtonBox.StandardButton.Ok
        # btn_cancel = QtWidgets.QDialogButtonBox.StandardButton.Cancel
        # buttons = QtWidgets.QDialogButtonBox(btn_ok | btn_cancel)
        # buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText(
        #     "Dodaj zadanie"
        # )
        # buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Cancel).setText(
        #     "Zamknij"
        # )
        # buttons.accepted.connect(self._on_accept)
        # buttons.rejected.connect(self.reject)
        # layout.addWidget(buttons)
        
        self.logger.debug("Zakończono inicjalizację UI")
        
    except Exception as e:
        msg = "Błąd podczas inicjalizacji UI"
        self.logger.error(f"{msg}: {str(e)}", exc_info=True)
        raise
Dodatkowo, aby upewnić się, że krzyżyk w narożniku okna działa poprawnie:
pythondef closeEvent(self, event):
    """Obsługa zamknięcia okna."""
    self.logger.info("Zamykanie okna dialogowego")
    event.accept()
Te zmiany spowodują, że:

Przycisk "Dodaj zadanie" będzie tylko dodawał zadanie, bez zamykania okna
Przycisk "Zamknij" oraz krzyżyk w narożniku będą prawidłowo zamykać okno
Użytkownik będzie mógł dodawać wiele zadań bez potrzeby ponownego otwierania okna dialogowego