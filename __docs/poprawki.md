Zmiany w pliku tools/data_splitter_gui.py
Zmiana w klasie FileSplitter
Potrzebujemy dodać parametr move_files do klasy FileSplitter, aby określić, czy pliki mają być kopiowane czy przenoszone:
pythonclass FileSplitter:
    def __init__(
        self,
        input_dir,
        output_dir,
        split_mode,
        split_value,
        use_validation=True,
        selected_categories=None,
        move_files=False,  # Nowy parametr
    ):
        ds_logger.info("Inicjalizacja FileSplitter")
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.split_mode = split_mode
        self.split_value = split_value
        self.use_validation = use_validation
        self.selected_categories = selected_categories if selected_categories else []
        self.stats = {"train": {}, "valid": {}}
        self.json_report = {}
        self.min_files_in_selection_for_report = 0
        self.folders_with_min_for_report = []
        self.move_files = move_files  # Zapisanie parametru
        ds_logger.info(
            f"FileSplitter zainicjalizowany: tryb={split_mode}, wartość={split_value}, walidacja={use_validation}, przenoszenie={move_files}"
        )
Modyfikacja metody process_files w klasie FileSplitter
Zmodyfikuj metodę process_files, aby używała shutil.move zamiast shutil.copy2 gdy self.move_files jest True:
python# Modyfikacja w sekcji kopiowania plików treningowych - zmiana na odpowiednią funkcję
for file_path in train_files_to_copy:
    if cancel_check and cancel_check():
        break
    try:
        # Wybierz funkcję w zależności od tego, czy przenosimy czy kopiujemy
        file_operation = shutil.move if self.move_files else shutil.copy2
        file_operation(file_path, current_train_path / file_path.name)
        self.stats["train"][str(relative_path)] += 1
        self.json_report[str(relative_path)]["train"].append(
            file_path.name
        )
        processed_files_count += 1
    except Exception as e:
        ds_logger.error(f"Błąd {'przenoszenia' if self.move_files else 'kopiowania'} {file_path} (trening): {e}")
        return None, f"Błąd {'przenoszenia' if self.move_files else 'kopiowania'} {file_path} (trening): {e}"

# Modyfikacja w sekcji kopiowania plików walidacyjnych
for file_path in valid_files_to_copy:
    if cancel_check and cancel_check():
        break
    try:
        if current_valid_path:  # Ensure path exists
            # Wybierz funkcję w zależności od tego, czy przenosimy czy kopiujemy
            file_operation = shutil.move if self.move_files else shutil.copy2
            file_operation(file_path, current_valid_path / file_path.name)
            if (
                self.use_validation
            ):  # Double check, though num_valid > 0 implies this
                self.stats["valid"][str(relative_path)] += 1
                self.json_report[str(relative_path)]["valid"].append(
                    file_path.name
                )
            processed_files_count += 1
    except Exception as e:
        ds_logger.error(f"Błąd {'przenoszenia' if self.move_files else 'kopiowania'} {file_path} (walidacja): {e}")
        return None, f"Błąd {'przenoszenia' if self.move_files else 'kopiowania'} {file_path} (walidacja): {e}"
Modyfikacja metody _generate_report w klasie FileSplitter
Zaktualizuj metodę generowania raportu, aby uwzględniała, czy pliki były kopiowane czy przenoszone:
pythondef _generate_report(self):
    operation_type = "przenoszenia" if self.move_files else "kopiowania"
    report = [f"=== RAPORT {operation_type.upper()} ===", ""]
    # Pozostała część metody bez zmian
    # ...
    
    report.append(f"Łącznie {'przeniesiono' if self.move_files else 'skopiowano'}: {total_train + total_valid} plików")
    report.append(f"  - {TRAIN_FOLDER_NAME}: {total_train} plików")
    report.append(f"  - {VALID_FOLDER_NAME}: {total_valid} plików")
    return "\n".join(report)
Modyfikacja klasy DS_Worker
Dodajemy parametr move_files do klasy DS_Worker i przekazujemy go do klasy FileSplitter:
pythonclass DS_Worker(QThread):  # Renamed to DS_Worker
    progress_updated = pyqtSignal(int, str)
    finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        input_dir,
        output_dir,
        split_mode,
        split_value,
        use_validation=True,
        selected_categories=None,
        move_files=False,  # Nowy parametr
    ):
        super().__init__()
        ds_logger.info("Inicjalizacja wątku DS_Worker")
        self.splitter = FileSplitter(
            input_dir,
            output_dir,
            split_mode,
            split_value,
            use_validation,
            selected_categories,
            move_files,  # Przekazanie parametru
        )
        self.is_cancelled = False
Modyfikacja klasy DataSplitterApp
Teraz zmodyfikujemy klasę DataSplitterApp, aby powiązać przycisk "Przycisk" z funkcją przenoszenia plików:
pythondef initUI(self):
    # ... istniejący kod ...
    
    control_buttons_layout = QHBoxLayout()
    self.start_button = QPushButton("Rozpocznij kopiowanie")
    self.start_button.setProperty("action", "success")
    self.start_button.clicked.connect(self.start_processing)
    self.empty_button = QPushButton("Przenieś pliki")  # Zmiana nazwy przycisku
    self.empty_button.clicked.connect(self.start_moving)  # Powiązanie z nową funkcją
    self.cancel_button = QPushButton("Anuluj")
    self.cancel_button.clicked.connect(self.cancel_processing)
    self.cancel_button.setEnabled(False)
    control_buttons_layout.addWidget(self.start_button)
    control_buttons_layout.addWidget(self.empty_button)
    control_buttons_layout.addWidget(self.cancel_button)
    layout.addLayout(control_buttons_layout)  # Add QHBoxLayout directly
    
    # ... pozostały kod ...
Dodanie nowej metody start_moving do klasy DataSplitterApp
Dodajemy nową metodę do klasy DataSplitterApp, która będzie analogiczna do start_processing, ale z parametrem move_files=True:
pythondef start_moving(self):
    """Funkcja analogiczna do start_processing, ale przenosi pliki zamiast kopiować"""
    self.log_message("Rozpoczynam przenoszenie plików")
    if not self.input_dir or not Path(self.input_dir).is_dir():
        QMessageBox.warning(
            self, "Brak folderu", "Wybierz prawidłowy folder źródłowy."
        )
        return
    if not self.output_dir:
        QMessageBox.warning(self, "Brak folderu", "Wybierz folder docelowy.")
        return

    selected_categories = self.get_selected_categories_names()
    if not selected_categories:
        QMessageBox.warning(
            self,
            "Brak kategorii",
            "Wybierz przynajmniej jedną kategorię do przetworzenia.",
        )
        return

    self.log_message(f"Wybrane kategorie do przeniesienia: {selected_categories}")

    input_path = Path(self.input_dir)
    output_path = Path(self.output_dir)
    if input_path == output_path or output_path.is_relative_to(input_path):
        reply = QMessageBox.question(
            self,
            "Potwierdzenie ścieżki",
            f"Folder docelowy ('{output_path}') jest taki sam jak źródłowy lub znajduje się wewnątrz niego. "
            f"Spowoduje to utworzenie '{TRAIN_FOLDER_NAME}' i '{VALID_FOLDER_NAME}' w '{output_path}'.\n"
            "Kontynuować?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.No:
            self.log_message(
                "Użytkownik anulował operację ze względu na ścieżki.",
                level=logging.INFO,
            )
            return

    # Dodatkowe ostrzeżenie, ponieważ przenoszenie jest operacją nieodwracalną
    reply = QMessageBox.question(
        self,
        "Potwierdzenie przenoszenia",
        "UWAGA: Przenoszenie plików jest operacją nieodwracalną i spowoduje usunięcie oryginałów."
        "\nCzy na pewno chcesz kontynuować?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    if reply == QMessageBox.StandardButton.No:
        self.log_message("Użytkownik anulował operację przenoszenia.", level=logging.INFO)
        return

    self._set_controls_enabled(False)
    self.cancel_button.setEnabled(True)
    self.progress_bar.setValue(0)

    split_mode_str = "percent" if self.mode_combo.currentIndex() == 0 else "files"
    split_val = (
        self.split_slider.value()
        if split_mode_str == "percent"
        else self.files_spin.value()
    )
    use_val_check = self.validation_check.isChecked()

    self.log_message(
        f"Parametry przenoszenia: tryb={split_mode_str}, wartość={split_val}, walidacja={use_val_check}"
    )

    self.processing_thread = DS_Worker(  # Use DS_Worker
        self.input_dir,
        self.output_dir,
        split_mode_str,
        split_val,
        use_val_check,
        selected_categories,
        move_files=True,  # Parametr move_files=True
    )
    self.processing_thread.progress_updated.connect(self.update_progress_display)
    self.processing_thread.finished.connect(self.processing_finished_display)
    self.processing_thread.error_occurred.connect(self.processing_error_display)
    self.log_message("Uruchamiam wątek przenoszenia DataSplitter")
    self.processing_thread.start()
Zmiana klasy DS_ReportDialog
Zaktualizujmy też tytuł okna raportu, aby uwzględniał, czy pliki były kopiowane czy przenoszone:
pythonclass DS_ReportDialog(QDialog):  # Renamed
    def __init__(self, report_text, parent=None):
        super().__init__(parent)
        self.log_to_main_console = (
            parent.log_to_main_console
            if hasattr(parent, "log_to_main_console")
            else None
        )
        if self.log_to_main_console:
            self.log_to_main_console.emit("Inicjalizacja okna raportu DataSplitter")

        # Ustaw tytuł okna na podstawie zawartości raportu
        if "RAPORT PRZENOSZENIA" in report_text:
            self.setWindowTitle("Raport przenoszenia (Data Splitter)")
        else:
            self.setWindowTitle("Raport kopiowania (Data Splitter)")
            
        self.setMinimumSize(700, 500)  # Adjusted size
        # ... reszta metody bez zmian
Zmiana metody _format_report_to_html w klasie DS_ReportDialog
pythondef _format_report_to_html(self, report_text):
    html = report_text.replace("\n", "<br>")
    # Basic formatting for headers
    html = html.replace("=== RAPORT KOPIOWANIA ===", "<h2>RAPORT KOPIOWANIA</h2>")
    html = html.replace("=== RAPORT PRZENOSZENIA ===", "<h2>RAPORT PRZENOSZENIA</h2>")
    html = html.replace(
        "=== PODSUMOWANIE OGÓLNE ===", "<h2>PODSUMOWANIE OGÓLNE</h2>"
    )
    # Highlight folder names or key stats if needed, e.g. by wrapping with <strong>
    return f"<body style='color:{DS_TEXT_COLOR}; background-color:{DS_BACKGROUND};'>{html}</body>"
To są wszystkie niezbędne zmiany, które należy wprowadzić w pliku tools/data_splitter_gui.py, aby dodać funkcjonalność przenoszenia plików do przycisku wcześniej opisanego jako "Przycisk".