Oto proponowane zmiany:

Wyłączenie logowania do pliku:

python# Plik: data_splitter_gui.py
# Linia ~2775-2798 (metoda _setup_main_logger_and_qt_handler)

def _setup_main_logger_and_qt_handler(self):
    self.global_logger = logging.getLogger("CombinedApp")  # Use the global one
    self.global_logger.handlers.clear()  # Clear any existing handlers from previous runs if any
    self.global_logger.setLevel(logging.DEBUG)
    self.global_logger.propagate = False

    # QtLogHandler for UI console
    self.qt_log_handler = QtLogHandler(self)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"
    )
    self.qt_log_handler.setFormatter(formatter)
    self.qt_log_handler.log_signal.connect(self._append_log_to_console)
    self.global_logger.addHandler(self.qt_log_handler)

    # Wyłączenie logowania do pliku
    # Optional: Add a file handler for persistent logs
    # try:
    #     log_dir = Path("logs")
    #     log_dir.mkdir(exist_ok=True)
    #     file_handler = logging.FileHandler(
    #         log_dir / "combined_app.log", mode="a", encoding="utf-8"
    #     )
    #     file_handler.setFormatter(formatter)
    #     self.global_logger.addHandler(file_handler)
    #     self.global_logger.info("File logging to combined_app.log enabled.")
    # except Exception as e:
    #     self.global_logger.error(f"Could not set up file logger: {e}")

Ustalenie stałej wysokości dla grupy konsoli:

python# Plik: data_splitter_gui.py
# Linia ~2872-2886 (metoda _create_console_panel)

def _create_console_panel(self, parent_layout):
    # Simplified from main_window.py for brevity, can be expanded
    console_group = QGroupBox("Konsola Aplikacji")
    console_group.setFixedHeight(125)  # MODIFIED: Ustawienie stałej wysokości dla całej grupy
    console_layout_internal = QVBoxLayout(console_group)

    self.console_text = QTextEdit()
    self.console_text.setReadOnly(True)
    # Usunięto: self.console_text.setMinimumHeight(75)
    # Usunięto: self.console_text.setMaximumHeight(100)
    self.console_text.setStyleSheet(
        "font-family: 'Consolas', 'Courier New', monospace; font-size: 10px;"
    )
    console_layout_internal.addWidget(self.console_text)

    button_row_layout = QHBoxLayout()
    clear_btn = QPushButton("Wyczyść konsolę")
    clear_btn.clicked.connect(self.console_text.clear)
    button_row_layout.addWidget(clear_btn)
    button_row_layout.addStretch(1)
    console_layout_internal.addLayout(button_row_layout)

    parent_layout.addWidget(console_group)
Te zmiany zapewnią, że:

Logowanie do pliku jest wyłączone (zakomentowane)
Grupa konsoli ma ustaloną stałą wysokość 125px, co obejmuje zarówno obszar konsoli jak i przycisk "Wyczyść konsolę"
Usunięte zostały niepotrzebne ograniczenia wysokości dla samego pola konsoli, ponieważ teraz cała grupa ma stałą wysokość

Proponowana wartość 125px powinna zapewnić odpowiednią wysokość dla grupy, ale można ją dostosować w zależności od potrzeb interfejsu.