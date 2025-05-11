import logging
import os
import traceback

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.gui.tab_interface import TabInterface
from app.gui.widgets.class_list import ClassList
from app.metadata.metadata_manager import MetadataManager
from app.sorter.image_sorter import ImageSorter


# Definicja nowego wątku sortującego
class SortingThread(QThread):
    progress_updated = pyqtSignal(int, int)
    category_processed = pyqtSignal(
        str, int, str
    )  # Nazwa kategorii, liczba plików, status (np. "Przetworzono")
    finished_successfully = pyqtSignal(dict)  # Statystyki końcowe
    error_occurred = pyqtSignal(str)
    status_changed = pyqtSignal(str)  # Sygnał zmiany statusu głównej etykiety

    def __init__(self, sorter_instance, method_name, params, parent=None):
        super().__init__(parent)
        self.sorter = sorter_instance
        # Nazwa metody do wywołania w ImageSorter
        # ('sort_images' lub 'sort_directory')
        self.method_name = method_name
        self.params = params  # Słownik z parametrami dla tej metody
        self._is_running = True

    def run(self):
        try:
            self.status_changed.emit("Rozpoczęto sortowanie...")

            # Callback do przekazania do metod sortujących w ImageSorter
            # Będzie emitował sygnał progress_updated z tego wątku
            def progress_callback(current, total):
                if self._is_running:
                    self.progress_updated.emit(current, total)

            # Callback dla przetwarzanych kategorii
            def category_processed_callback(category_name, count, status_msg):
                if self._is_running:
                    self.category_processed.emit(category_name, count, status_msg)

            self.params["callback"] = progress_callback
            self.params["category_callback"] = (
                category_processed_callback  # Dodajemy nowy callback
            )

            # Wywołanie odpowiedniej metody sortującej
            if self.method_name == "sort_images":
                # Metoda sort_images może potrzebować output_directory
                # ustawionego w instancji sortera lub przekazanego w params,
                # jeśli jej sygnatura to dopuszcza.
                # Zakładamy, że output_directory jest już w
                # self.sorter.output_directory lub jest częścią
                # self.params dla sort_images.
                stats = self.sorter.sort_images(**self.params)
            elif self.method_name == "sort_directory":
                stats = self.sorter.sort_directory(**self.params)
            else:
                msg = f"Nieznana metoda sortowania: {self.method_name}"
                raise ValueError(msg)

            if self._is_running:  # Sprawdź, czy nie przerwano w międzyczasie
                self.finished_successfully.emit(stats)
                self.status_changed.emit("Sortowanie zakończone pomyślnie.")
        except Exception as e:
            # Tylko jeśli błąd nie jest skutkiem zatrzymania
            if self._is_running:
                error_main = f"Błąd podczas sortowania: {str(e)}"
                detailed_error = f"{error_main}\n{traceback.format_exc()}"
                self.error_occurred.emit(detailed_error)
                self.status_changed.emit("Błąd podczas sortowania.")
        finally:
            self._is_running = False  # Upewnij się, że flaga jest resetowana

    def stop_thread(self):
        self.status_changed.emit("Przerywanie sortowania...")
        self._is_running = False
        if self.sorter:
            self.sorter.stop()  # Wywołaj metodę stop na instancji sortera


class BatchProcessor(QWidget, TabInterface):
    """Klasa zarządzająca zakładką klasyfikacji wsadowej."""

    def __init__(self, parent=None, settings=None):
        super().__init__(parent)
        self.parent = parent
        self.settings = settings
        self.metadata_manager = MetadataManager()
        self.image_sorter = None
        self.sorting_thread = None
        self.is_processing = False
        self.setup_ui()
        self.connect_signals()
        self.logger = logging.getLogger("BatchProcessor")
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            log_format = "%(asctime)s [%(levelname)s] %(message)s"
            formatter = logging.Formatter(log_format)
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def setup_ui(self):
        """Konfiguruje interfejs użytkownika zakładki."""
        main_layout = QVBoxLayout(self)

        # Sekcja wyboru katalogów
        dir_group = QGroupBox("Wybór katalogów")
        dir_layout = QHBoxLayout()
        self.input_dir_label = QLabel("Katalog źródłowy: -")
        self.input_dir_label.setWordWrap(True)
        select_input_btn = QPushButton("Wybierz")
        select_input_btn.clicked.connect(self._select_input_dir)
        self.output_dir_label = QLabel("Katalog docelowy: -")
        self.output_dir_label.setWordWrap(True)
        select_output_btn = QPushButton("Wybierz")
        select_output_btn.clicked.connect(self._select_output_dir)
        dir_layout.addWidget(self.input_dir_label, 1)
        dir_layout.addWidget(select_input_btn)
        dir_layout.addWidget(self.output_dir_label, 1)
        dir_layout.addWidget(select_output_btn)
        dir_group.setLayout(dir_layout)
        main_layout.addWidget(dir_group)

        # Sekcja opcji
        options_group = QGroupBox("Opcje sortowania")
        options_layout = QHBoxLayout()

        # 1. Checkbox "Kopiuj pliki" (lewa strona)
        copy_files_text = "Kopiuj pliki (zamiast przenosić)"
        self.copy_files_checkbox = QCheckBox(copy_files_text)
        self.copy_files_checkbox.setChecked(True)
        options_layout.addWidget(self.copy_files_checkbox)

        options_layout.addStretch(1)  # Odpycha suwak od lewego checkboxa

        # 2. Kontrolki progu pewności (środek)
        # Zmieniamy na układ pionowy dla dwóch suwaków
        confidence_sliders_layout = QVBoxLayout()

        # Minimalny próg pewności
        min_confidence_layout = QHBoxLayout()
        self.min_confidence_label = QLabel("Pewność OD:")
        min_confidence_layout.addWidget(self.min_confidence_label)

        self.min_confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_confidence_slider.setRange(0, 100)  # 0.00 do 1.00
        self.min_confidence_slider.setValue(50)  # Domyślnie 0.50
        self.min_confidence_slider.setToolTip(
            "Ustaw minimalny próg pewności dla klasyfikacji (0.00 - 1.00)"
        )
        self.min_confidence_slider.setMinimumWidth(768)
        min_confidence_layout.addWidget(self.min_confidence_slider, 2)

        self.min_confidence_value_label = QLabel(
            f"{self.min_confidence_slider.value() / 100.0:.2f}"
        )
        min_confidence_layout.addWidget(self.min_confidence_value_label)
        confidence_sliders_layout.addLayout(min_confidence_layout)

        # Maksymalny próg pewności
        max_confidence_layout = QHBoxLayout()
        self.max_confidence_label = QLabel("Pewność DO:")
        max_confidence_layout.addWidget(self.max_confidence_label)

        self.max_confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_confidence_slider.setRange(0, 100)  # 0.00 do 1.00
        self.max_confidence_slider.setValue(100)  # Domyślnie 1.00
        self.max_confidence_slider.setToolTip(
            "Ustaw maksymalny próg pewności dla klasyfikacji (0.00 - 1.00)"
        )
        self.max_confidence_slider.setMinimumWidth(768)
        max_confidence_layout.addWidget(self.max_confidence_slider, 2)

        self.max_confidence_value_label = QLabel(
            f"{self.max_confidence_slider.value() / 100.0:.2f}"
        )
        max_confidence_layout.addWidget(self.max_confidence_value_label)
        confidence_sliders_layout.addLayout(max_confidence_layout)

        options_layout.addLayout(
            confidence_sliders_layout
        )  # Dodajemy układ suwaków do głównego układu opcji

        # Podłączenie sygnałów suwaków
        self.min_confidence_slider.valueChanged.connect(
            self._update_min_confidence_display
        )
        self.max_confidence_slider.valueChanged.connect(
            self._update_max_confidence_display
        )

        options_layout.addStretch(1)  # Odpycha prawy checkbox od grupy suwaka

        # 3. Nowy checkbox "Użyj osobnego folderu..." (prawa strona)
        uncategorized_text = "Użyj osobnego folderu dla plików bez kategorii"
        self.uncategorized_folder_checkbox = QCheckBox(uncategorized_text)
        # Domyślnie zaznaczony
        self.uncategorized_folder_checkbox.setChecked(True)
        tooltip_text = (
            "Jeśli zaznaczone, pliki niezakwalifikowane do żadnej kategorii \n"
            "zostaną skopiowane do podfolderu '__bez_kategorii__' "
            "w katalogu docelowym."
        )
        self.uncategorized_folder_checkbox.setToolTip(tooltip_text)
        options_layout.addWidget(self.uncategorized_folder_checkbox)

        options_group.setLayout(options_layout)
        main_layout.addWidget(options_group)

        # Sekcja kontrolna
        control_group = QGroupBox("Kontrola")
        main_control_layout = QVBoxLayout()

        # Górny rząd: Status i Pasek Postępu
        progress_layout = QHBoxLayout()
        self.status_label = QLabel("Gotowy")
        progress_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(18)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar, 1)

        main_control_layout.addLayout(progress_layout)

        # Dolny rząd: Przyciski
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch(1)

        self.start_button = QPushButton("Rozpocznij sortowanie")
        start_button_style = (
            "background-color: #4CAF50; color: white; "
            "border-radius: 4px; padding: 6px;"
        )
        self.start_button.setStyleSheet(start_button_style)
        self.start_button.clicked.connect(self._start_processing)
        buttons_layout.addWidget(self.start_button)

        buttons_layout.addSpacing(20)  # Szerszy separator

        self.sort_selected_btn = QPushButton("Sortuj wybrane kategorie")
        sort_selected_style = (
            "background-color: #2196F3; color: white; "
            "border-radius: 4px; padding: 6px;"
        )
        self.sort_selected_btn.setStyleSheet(sort_selected_style)
        self.sort_selected_btn.clicked.connect(self._show_class_list)
        buttons_layout.addWidget(self.sort_selected_btn)

        buttons_layout.addSpacing(20)  # Szerszy separator

        self.stop_btn = QPushButton("Przerwij sortowanie")
        stop_btn_style = (
            "background-color: #f44336; color: white; "
            "border-radius: 4px; padding: 6px;"
        )
        self.stop_btn.setStyleSheet(stop_btn_style)
        self.stop_btn.clicked.connect(self._stop_processing)
        self.stop_btn.setEnabled(False)
        buttons_layout.addWidget(self.stop_btn)

        buttons_layout.addSpacing(20)  # Szerszy separator

        self.clear_history_btn = QPushButton("Wyczyść historię")
        self.clear_history_btn.clicked.connect(self._clear_history)
        buttons_layout.addWidget(self.clear_history_btn)

        buttons_layout.addStretch(1)
        main_control_layout.addLayout(buttons_layout)

        control_group.setLayout(main_control_layout)
        main_layout.addWidget(control_group)

        # Sekcja wyników
        results_group = QGroupBox("Wyniki sortowania")
        results_layout = QVBoxLayout()
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(
            ["Typ", "Nazwa", "Liczba plików", "Status"]
        )
        self.results_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.results_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.ResizeToContents
        )
        # Ustawienie polityki rozmiaru dla tabeli
        self.results_table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.results_table.setMinimumHeight(200)  # Minimalna wysokość tabeli
        results_layout.addWidget(self.results_table)
        results_group.setLayout(results_layout)
        main_layout.addWidget(results_group)

    def connect_signals(self):
        """Podłącza sygnały (obecnie puste)."""
        pass

    def refresh(self):
        """Odświeża widok (obecnie puste)."""
        pass

    def update_settings(self, settings):
        """Aktualizuje ustawienia."""
        self.settings = settings
        last_input = self.settings.get("last_batch_input_dir")
        if last_input:
            self.input_dir = last_input
            self.input_dir_label.setText(f"Katalog źródłowy: {last_input}")
        last_output = self.settings.get("last_batch_output_dir")
        if last_output:
            self.output_dir = last_output
            self.output_dir_label.setText(f"Katalog docelowy: {last_output}")

        last_min_confidence = float(
            self.settings.get("last_min_confidence_threshold", 0.50)
        )
        self.min_confidence_slider.setValue(int(last_min_confidence * 100))

        last_max_confidence = float(
            self.settings.get("last_max_confidence_threshold", 1.00)
        )
        self.max_confidence_slider.setValue(int(last_max_confidence * 100))

        last_use_uncategorized = self.settings.get(
            "last_use_uncategorized_folder", True
        )
        uncategorized_cb = self.uncategorized_folder_checkbox
        uncategorized_cb.setChecked(bool(last_use_uncategorized))

    def save_state(self):
        """Zapisuje stan zakładki."""
        state = {}
        if hasattr(self, "input_dir"):
            state["last_batch_input_dir"] = self.input_dir
        if hasattr(self, "output_dir"):
            state["last_batch_output_dir"] = self.output_dir
        min_confidence = self.min_confidence_slider.value() / 100.0
        state["last_min_confidence_threshold"] = min_confidence
        max_confidence = self.max_confidence_slider.value() / 100.0
        state["last_max_confidence_threshold"] = max_confidence
        state["last_use_uncategorized_folder"] = (
            self.uncategorized_folder_checkbox.isChecked()
        )
        return state

    def restore_state(self, state):
        """Przywraca stan zakładki."""
        if state:
            input_dir = state.get("last_batch_input_dir")
            if input_dir:
                self.input_dir = input_dir
                self.input_dir_label.setText(f"Katalog źródłowy: {input_dir}")
            output_dir = state.get("last_batch_output_dir")
            if output_dir:
                self.output_dir = output_dir
                text = f"Katalog docelowy: {output_dir}"
                self.output_dir_label.setText(text)

            min_confidence = state.get("last_min_confidence_threshold")
            if min_confidence is not None:
                self.min_confidence_slider.setValue(int(float(min_confidence) * 100))
                # Etykieta z wartością zaktualizuje się przez
                # sygnał valueChanged

            max_confidence = state.get("last_max_confidence_threshold")
            if max_confidence is not None:
                self.max_confidence_slider.setValue(int(float(max_confidence) * 100))
                # Etykieta z wartością zaktualizuje się przez
                # sygnał valueChanged

            use_uncategorized = state.get("last_use_uncategorized_folder")
            if use_uncategorized is not None:
                uncategorized_cb = self.uncategorized_folder_checkbox
                uncategorized_cb.setChecked(bool(use_uncategorized))

    def _select_input_dir(self):
        """Wybiera katalog źródłowy."""
        default_dir = os.path.expanduser("~")
        last_dir = self.settings.get("last_batch_input_dir", default_dir)
        dir_path = QFileDialog.getExistingDirectory(
            self, "Wybierz katalog źródłowy", last_dir
        )
        if dir_path:
            self.input_dir = dir_path
            self.input_dir_label.setText(f"Katalog źródłowy: {dir_path}")
            self.settings["last_batch_input_dir"] = dir_path

    def _select_output_dir(self):
        """Wybiera katalog docelowy."""
        default_dir = os.path.expanduser("~")
        last_dir = self.settings.get("last_batch_output_dir", default_dir)
        dir_path = QFileDialog.getExistingDirectory(
            self, "Wybierz katalog docelowy", last_dir
        )
        if dir_path:
            self.output_dir = dir_path
            self.output_dir_label.setText(f"Katalog docelowy: {dir_path}")
            self.settings["last_batch_output_dir"] = dir_path

    def _clear_results(self):
        """Czyści wyniki."""
        self.results_table.setRowCount(0)

    def _start_processing(self):
        """Rozpoczyna przetwarzanie (sortowanie wszystkich obrazów)
        w osobnym wątku."""
        if self.is_processing:
            QMessageBox.information(
                self, "Informacja", "Przetwarzanie jest już w toku."
            )
            return

        no_classifier = (
            not hasattr(self.parent, "classifier") or not self.parent.classifier
        )
        if no_classifier:
            msg = (
                "Nie załadowano modelu klasyfikacji. "
                "Proszę najpierw załadować model."
            )
            QMessageBox.warning(self, "Brak modelu", msg)
            return

        if not hasattr(self, "input_dir") or not self.input_dir:
            QMessageBox.warning(self, "Brak katalogu", "Wybierz katalog źródłowy.")
            return
        if not hasattr(self, "output_dir") or not self.output_dir:
            QMessageBox.warning(self, "Brak katalogu", "Wybierz katalog docelowy.")
            return

        try:
            model_path = self.parent.classifier.get_weights_path()
            if not model_path:
                QMessageBox.critical(
                    self,
                    "Błąd krytyczny",
                    "Nie udało się uzyskać ścieżki do modelu. "
                    "Upewnij się, że model jest załadowany.",
                )
                return

            # Utwórz instancję ImageSorter tuż przed uruchomieniem wątku
            current_image_sorter = ImageSorter(
                model_path=model_path,
                output_directory=self.output_dir,
                preserve_original_classes=(self.copy_files_checkbox.isChecked()),
                logger=self.logger,
            )

        except AttributeError:
            error_msg = "Klasyfikator nie ma metody 'get_weights_path'."
            QMessageBox.critical(
                self,
                "Błąd krytyczny",
                error_msg,
            )
            return
        except Exception as e:
            QMessageBox.critical(self, "Błąd inicjalizacji sortera", str(e))
            log_msg_prefix = "Błąd inicjalizacji ImageSorter"
            log_msg = (
                f"{log_msg_prefix} dla _start_processing: {e}\n"
                f"{traceback.format_exc()}"
            )
            self.logger.error(log_msg)
            return

        self.is_processing = True
        self.stop_btn.setEnabled(True)
        self._clear_results()
        self._setup_progress_bar()
        self.status_label.setText("Przygotowywanie do sortowania...")

        params = {
            "input_directory": self.input_dir,
            "batch_size": 16,  # Można to uczynić konfigurowalnym
            "min_confidence_threshold": self.min_confidence_slider.value() / 100.0,
            "max_confidence_threshold": self.max_confidence_slider.value() / 100.0,
            "use_uncategorized_folder": (
                self.uncategorized_folder_checkbox.isChecked()
            ),
            # output_directory dla sort_images jest brany
            # z instancji ImageSorter
            # category_callback zostanie dodany przez SortingThread
        }

        # Zmieniony log dla dwóch progów
        min_thresh = params["min_confidence_threshold"]
        max_thresh = params["max_confidence_threshold"]
        self.logger.info(
            f"DEBUG: Przekazywane prógi pewności do sort_images: "
            f"{min_thresh} - {max_thresh}"
        )

        thread = SortingThread(current_image_sorter, "sort_images", params)
        self.sorting_thread = thread

        # Podłączanie sygnałów wątku do slotów GUI
        self.sorting_thread.progress_updated.connect(self._update_progress)
        thread = self.sorting_thread
        thread.category_processed.connect(self._handle_category_processed)
        thread.finished_successfully.connect(self._handle_sorting_finished)
        self.sorting_thread.error_occurred.connect(self._handle_sorting_error)
        self.sorting_thread.status_changed.connect(self._update_status_label)
        self.sorting_thread.finished.connect(self._on_thread_finished)

        self.sorting_thread.start()
        self.status_label.setText("Sortowanie w toku...")

    def _update_status_label(self, message):
        self.status_label.setText(message)

    def _on_thread_finished(self):
        self.is_processing = False
        self.stop_btn.setEnabled(False)
        self.sorting_thread = None  # Usuń referencję do zakończonego wątku
        if hasattr(self, "progress_bar"):
            self.progress_bar.setVisible(False)
            self.status_label.setText("Gotowy")

    def _handle_category_processed(self, class_name, count, status_msg):
        # Aktualizuje tabelę wyników na bieżąco
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        self.results_table.setItem(row, 0, QTableWidgetItem("Kategoria"))
        self.results_table.setItem(row, 1, QTableWidgetItem(class_name))
        self.results_table.setItem(row, 2, QTableWidgetItem(str(count)))
        self.results_table.setItem(row, 3, QTableWidgetItem(status_msg))
        # Przewiń na dół, aby widzieć ostatnie wpisy
        self.results_table.scrollToBottom()

    def _handle_sorting_finished(self, stats):
        # Ta metoda zastąpi część logiki z oryginalnego
        # _start_processing po pętli sortowania
        if stats:  # Upewnijmy się, że stats nie jest None
            total_processed = stats.get("total_processed", "N/A")
            total_moved = stats.get("total_moved_or_copied", "N/A")
            total_skipped_conf = stats.get("total_skipped_confidence", "N/A")
            total_skipped_err = stats.get("total_skipped_errors", "N/A")
            total_uncategorized = stats.get(
                "total_uncategorized", "N/A"
            )  # Nowa statystyka
            line_uncategorized = (
                f"Skopiowano do '__bez_kategorii__': {total_uncategorized}"
            )

            summary_msg = (
                f"Sortowanie plików zakończone pomyślnie.\n\n"
                f"Przetworzono obrazów: {total_processed}\n"
                f"Przeniesiono/skopiowano plików: {total_moved}\n"
                f"Pominięto (niska pewność): {total_skipped_conf}\n"
                f"Pominięto (błędy): {total_skipped_err}\n"
                f"{line_uncategorized}"
            )
            QMessageBox.information(self, "Sukces", summary_msg)
            self.logger.info(summary_msg)
        else:
            QMessageBox.information(
                self,
                "Informacja",
                "Sortowanie zakończone, ale brak szczegółowych statystyk.",
            )
            self.logger.info("Sortowanie zakończone, brak statystyk.")
        # _on_thread_finished zajmie się resztą (is_processing, stop_btn)

    def _handle_sorting_error(self, error_message):
        QMessageBox.critical(self, "Błąd sortowania", error_message)
        log_msg = f"Błąd zgłoszony przez SortingThread: {error_message}"
        self.logger.error(log_msg)
        # _on_thread_finished zajmie się resztą (is_processing, stop_btn)

    def _setup_progress_bar(self):
        """Inicjalizuje i pokazuje pasek postępu."""
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

    def _update_progress(self, current, total):
        percentage = int((current / total) * 100) if total > 0 else 0
        self.progress_bar.setValue(percentage)

    def _stop_processing(self):
        if self.sorting_thread and self.sorting_thread.isRunning():
            self.sorting_thread.stop_thread()
        else:
            msg = "Brak aktywnego procesu sortowania do przerwania."
            QMessageBox.warning(self, "Informacja", msg)

    def _clear_history(self):
        """Czyści historię wyników sortowania po potwierdzeniu użytkownika."""
        reply = QMessageBox.question(
            self,
            "Potwierdzenie",
            "Czy na pewno chcesz wyczyścić historię wyników?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self._clear_results()
            self.status_label.setText("Historia wyczyszczona")
            log_msg = "Historia wyników sortowania została wyczyszczona"
            self.logger.info(log_msg)

    def _update_min_confidence_display(self, value):
        """Aktualizuje wyświetlanie minimalnego progu pewności i dostosowuje suwaki."""
        self.min_confidence_value_label.setText(f"{value / 100.0:.2f}")
        # Zapobiegaj sytuacji, w której min > max
        if value > self.max_confidence_slider.value():
            self.max_confidence_slider.setValue(value)

    def _update_max_confidence_display(self, value):
        """Aktualizuje wyświetlanie maksymalnego progu pewności i dostosowuje suwaki."""
        self.max_confidence_value_label.setText(f"{value / 100.0:.2f}")
        # Zapobiegaj sytuacji, w której max < min
        if value < self.min_confidence_slider.value():
            self.min_confidence_slider.setValue(value)

    def _show_class_list(self):
        """Pokazuje okno z listą klas do wyboru i uruchamia
        sortowanie dla wybranych."""
        if self.is_processing:
            QMessageBox.information(
                self, "Informacja", "Inny proces sortowania jest już w toku."
            )
            return

        no_classifier = (
            not hasattr(self.parent, "classifier") or not self.parent.classifier
        )
        if no_classifier:
            msg_text = (
                "Nie załadowano modelu klasyfikacji. "
                "Proszę najpierw załadować model."
            )
            QMessageBox.warning(
                self,
                "Brak modelu",
                msg_text,
            )
            return

        class_mapping = self.parent.classifier.get_class_mapping()
        dialog = ClassList(self)
        dialog.set_items(class_mapping)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_classes = dialog.get_selected_items()
            if not selected_classes:
                QMessageBox.warning(
                    self,
                    "Brak wybranych klas",
                    "Proszę wybrać przynajmniej jedną klasę do sortowania.",
                )
                return

            if not hasattr(self, "input_dir") or not self.input_dir:
                QMessageBox.warning(self, "Brak katalogu", "Wybierz katalog źródłowy.")
                return
            if not hasattr(self, "output_dir") or not self.output_dir:
                QMessageBox.warning(self, "Brak katalogu", "Wybierz katalog docelowy.")
                return

            try:
                model_path = self.parent.classifier.get_weights_path()
                if not model_path:
                    QMessageBox.critical(
                        self,
                        "Błąd krytyczny",
                        "Nie udało się uzyskać ścieżki do modelu.",
                    )
                    return

                current_image_sorter = ImageSorter(
                    model_path=model_path,
                    output_directory=self.output_dir,
                    preserve_original_classes=(self.copy_files_checkbox.isChecked()),
                    logger=self.logger,
                )
            except AttributeError:
                QMessageBox.critical(
                    self,
                    "Błąd krytyczny",
                    "Klasyfikator nie ma metody 'get_weights_path'.",
                )
                return
            except Exception as e:
                QMessageBox.critical(self, "Błąd inicjalizacji sortera", str(e))
                log_msg_prefix = "Błąd inicjalizacji ImageSorter"
                log_msg = (
                    f"{log_msg_prefix} dla _show_class_list: {e}\n"
                    f"{traceback.format_exc()}"
                )
                self.logger.error(log_msg)
                return

            self.is_processing = True
            self.stop_btn.setEnabled(True)
            self._clear_results()
            self._setup_progress_bar()
            self.status_label.setText(
                "Przygotowywanie do sortowania wybranych kategorii..."
            )

            params = {
                "input_dir": self.input_dir,
                "output_dir": self.output_dir,
                "min_confidence_threshold": self.min_confidence_slider.value() / 100.0,
                "max_confidence_threshold": self.max_confidence_slider.value() / 100.0,
                "selected_classes": selected_classes,
                # Klucz 'use_uncategorized_folder' został usunięty,
                # ponieważ metoda 'sort_directory' go nie oczekuje.
                # 'callback' zostanie dodany przez SortingThread
                # 'category_callback' zostanie dodany przez SortingThread
            }
            # Zmieniony log dla dwóch progów
            min_thresh = params["min_confidence_threshold"]
            max_thresh = params["max_confidence_threshold"]
            self.logger.info(
                f"DEBUG: Przekazywane prógi pewności do sort_directory: "
                f"{min_thresh} - {max_thresh}"
            )
            self.logger.info(
                f"DEBUG: Wybrane klasy do sort_directory: {selected_classes}"
            )

            self.sorting_thread = SortingThread(
                current_image_sorter, "sort_directory", params
            )

            self.sorting_thread.progress_updated.connect(self._update_progress)
            self.sorting_thread.category_processed.connect(
                self._handle_category_processed
            )
            self.sorting_thread.finished_successfully.connect(
                self._handle_sorting_finished
            )
            thread = self.sorting_thread
            thread.error_occurred.connect(self._handle_sorting_error)
            thread.status_changed.connect(self._update_status_label)
            self.sorting_thread.finished.connect(self._on_thread_finished)

            self.sorting_thread.start()
            status_text = "Sortowanie wybranych kategorii w toku..."
            self.status_label.setText(status_text)
