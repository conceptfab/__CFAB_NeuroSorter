import logging
import os
import traceback

from PyQt6.QtCore import QCoreApplication, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
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
    status_changed = pyqtSignal(str)  # Do aktualizacji głównej etykiety statusu

    def __init__(self, sorter_instance, method_name, params, parent=None):
        super().__init__(parent)
        self.sorter = sorter_instance
        self.method_name = method_name  # Nazwa metody do wywołania w ImageSorter ('sort_images' lub 'sort_directory')
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

            self.params["callback"] = progress_callback

            # Wywołanie odpowiedniej metody sortującej
            if self.method_name == "sort_images":
                # Metoda sort_images może potrzebować output_directory ustawionego w instancji sortera
                # lub przekazanego w params, jeśli jej sygnatura to dopuszcza.
                # Zakładamy, że output_directory jest już w self.sorter.output_directory
                # lub jest częścią self.params dla sort_images.
                stats = self.sorter.sort_images(**self.params)
            elif self.method_name == "sort_directory":
                stats = self.sorter.sort_directory(**self.params)
            else:
                raise ValueError(f"Nieznana metoda sortowania: {self.method_name}")

            if self._is_running:  # Sprawdź, czy nie przerwano w międzyczasie
                self.finished_successfully.emit(stats)
                self.status_changed.emit("Sortowanie zakończone pomyślnie.")
        except Exception as e:
            if self._is_running:  # Tylko jeśli błąd nie jest wynikiem zatrzymania
                detailed_error = (
                    f"Błąd podczas sortowania: {str(e)}\n{traceback.format_exc()}"
                )
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
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
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
        self.copy_files_checkbox = QCheckBox("Kopiuj pliki (zamiast przenosić)")
        self.copy_files_checkbox.setChecked(True)
        options_layout.addWidget(self.copy_files_checkbox)
        options_layout.addStretch()
        options_group.setLayout(options_layout)
        main_layout.addWidget(options_group)

        # Sekcja kontrolna
        control_group = QGroupBox("Kontrola")
        self.control_layout = QHBoxLayout()
        start_btn = QPushButton("Rozpocznij sortowanie")
        start_btn.clicked.connect(self._start_processing)
        self.sort_selected_btn = QPushButton("Sortuj wybrane kategorie")
        self.sort_selected_btn.clicked.connect(self._show_class_list)
        self.stop_btn = QPushButton("Przerwij sortowanie")
        self.stop_btn.clicked.connect(self._stop_processing)
        self.stop_btn.setEnabled(False)  # Domyślnie wyłączony
        self.clear_history_btn = QPushButton("Wyczyść historię")
        self.clear_history_btn.clicked.connect(self._clear_history)
        self.status_label = QLabel("Gotowy")
        self.control_layout.addWidget(start_btn)
        self.control_layout.addWidget(self.sort_selected_btn)
        self.control_layout.addWidget(self.stop_btn)
        self.control_layout.addWidget(self.clear_history_btn)
        self.control_layout.addStretch()
        self.control_layout.addWidget(self.status_label)
        control_group.setLayout(self.control_layout)
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

    def save_state(self):
        """Zapisuje stan zakładki (obecnie tylko ścieżki)."""
        state = {}
        if hasattr(self, "input_dir"):
            state["last_batch_input_dir"] = self.input_dir
        if hasattr(self, "output_dir"):
            state["last_batch_output_dir"] = self.output_dir
        return state

    def restore_state(self, state):
        """Przywraca stan zakładki (obecnie tylko ścieżki)."""
        if state:
            input_dir = state.get("last_batch_input_dir")
            if input_dir:
                self.input_dir = input_dir
                self.input_dir_label.setText(f"Katalog źródłowy: {input_dir}")
            output_dir = state.get("last_batch_output_dir")
            if output_dir:
                self.output_dir = output_dir
                self.output_dir_label.setText(f"Katalog docelowy: {output_dir}")

    def _select_input_dir(self):
        """Wybiera katalog źródłowy."""
        last_dir = self.settings.get("last_batch_input_dir", os.path.expanduser("~"))
        dir_path = QFileDialog.getExistingDirectory(
            self, "Wybierz katalog źródłowy", last_dir
        )
        if dir_path:
            self.input_dir = dir_path
            self.input_dir_label.setText(f"Katalog źródłowy: {dir_path}")
            self.settings["last_batch_input_dir"] = dir_path

    def _select_output_dir(self):
        """Wybiera katalog docelowy."""
        last_dir = self.settings.get("last_batch_output_dir", os.path.expanduser("~"))
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
        """Rozpoczyna przetwarzanie (sortowanie wszystkich obrazów) w osobnym wątku."""
        if self.is_processing:
            QMessageBox.information(
                self, "Informacja", "Przetwarzanie jest już w toku."
            )
            return

        if not hasattr(self.parent, "classifier") or not self.parent.classifier:
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
                output_directory=self.output_dir,  # Ważne dla metody sort_images
                preserve_original_classes=self.copy_files_checkbox.isChecked(),
                logger=self.logger,
            )

        except AttributeError:
            QMessageBox.critical(
                self, "Błąd krytyczny", "Klasyfikator nie ma metody 'get_weights_path'."
            )
            return
        except Exception as e:
            QMessageBox.critical(self, "Błąd inicjalizacji sortera", str(e))
            self.logger.error(
                f"Błąd inicjalizacji ImageSorter dla _start_processing: {e}\n{traceback.format_exc()}"
            )
            return

        self.is_processing = True
        self.stop_btn.setEnabled(True)
        self._clear_results()
        self._setup_progress_bar()
        self.status_label.setText("Przygotowywanie do sortowania...")

        params = {
            "input_directory": self.input_dir,
            "batch_size": 16,  # Można to uczynić konfigurowalnym
            "confidence_threshold": 0.5,  # Można to uczynić konfigurowalnym
            # output_directory dla sort_images jest brany z instancji ImageSorter
        }

        self.sorting_thread = SortingThread(current_image_sorter, "sort_images", params)

        # Podłączanie sygnałów wątku do slotów GUI
        self.sorting_thread.progress_updated.connect(self._update_progress)
        self.sorting_thread.category_processed.connect(self._handle_category_processed)
        self.sorting_thread.finished_successfully.connect(self._handle_sorting_finished)
        self.sorting_thread.error_occurred.connect(self._handle_sorting_error)
        self.sorting_thread.status_changed.connect(
            self._update_status_label
        )  # Nowy slot
        self.sorting_thread.finished.connect(
            self._on_thread_finished
        )  # Do resetowania stanu

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

    def _handle_category_processed(self, class_name, count, status_msg):
        # Aktualizuje tabelę wyników na bieżąco
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        self.results_table.setItem(row, 0, QTableWidgetItem("Kategoria"))
        self.results_table.setItem(row, 1, QTableWidgetItem(class_name))
        self.results_table.setItem(row, 2, QTableWidgetItem(str(count)))
        self.results_table.setItem(row, 3, QTableWidgetItem(status_msg))
        self.results_table.scrollToBottom()  # Przewiń na dół, aby widzieć ostatnie wpisy

    def _handle_sorting_finished(self, stats):
        # Ta metoda zastąpi część logiki z oryginalnego _start_processing po pętli sortowania
        if stats:  # Upewnijmy się, że stats nie jest None
            summary_msg = (
                f"Sortowanie plików zakończone pomyślnie.\n\n"
                f"Przetworzono obrazów: {stats.get('total_processed', 'N/A')}\n"
                f"Przeniesiono/skopiowano plików: {stats.get('total_moved_or_copied', 'N/A')}\n"
                f"Pominięto (niska pewność): {stats.get('total_skipped_confidence', 'N/A')}\n"
                f"Pominięto (błędy): {stats.get('total_skipped_errors', 'N/A')}"
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
        self.logger.error(f"Błąd zgłoszony przez SortingThread: {error_message}")
        # _on_thread_finished zajmie się resztą (is_processing, stop_btn)

    def _setup_progress_bar(self):
        """Inicjalizuje i pokazuje pasek postępu, jeśli nie istnieje."""
        if not hasattr(self, "progress_bar"):
            self.progress_bar = QProgressBar()
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.progress_bar.setFixedHeight(18)
            self.progress_bar.setTextVisible(False)
            # Dodaj pasek postępu do layoutu kontrolnego
            self.control_layout.insertWidget(
                1, self.progress_bar
            )  # Wstaw przed status_label

        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

    def _update_progress(self, current, total):
        if not hasattr(self, "progress_bar"):
            self._setup_progress_bar()

        percentage = int((current / total) * 100) if total > 0 else 0
        self.progress_bar.setValue(percentage)
        QCoreApplication.processEvents()  # To może nie być potrzebne, jeśli UI jest responsywne

    def _stop_processing(self):
        if self.sorting_thread and self.sorting_thread.isRunning():
            self.sorting_thread.stop_thread()
        else:
            QMessageBox.warning(
                self, "Informacja", "Brak aktywnego procesu sortowania do przerwania."
            )

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
            self.logger.info("Historia wyników sortowania została wyczyszczona")

    def _show_class_list(self):
        """Pokazuje okno z listą klas do wyboru i uruchamia sortowanie dla wybranych."""
        if self.is_processing:
            QMessageBox.information(
                self, "Informacja", "Inny proces sortowania jest już w toku."
            )
            return

        if not hasattr(self.parent, "classifier") or not self.parent.classifier:
            QMessageBox.warning(
                self,
                "Brak modelu",
                "Nie załadowano modelu klasyfikacji. Proszę najpierw załadować model.",
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
                    output_directory=self.output_dir,  # Potrzebne, jeśli sort_directory nie ustawia go samo
                    preserve_original_classes=self.copy_files_checkbox.isChecked(),
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
                self.logger.error(
                    f"Błąd inicjalizacji ImageSorter dla _show_class_list: {e}\n{traceback.format_exc()}"
                )
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
                "confidence_threshold": 0.5,  # Można to uczynić konfigurowalnym
                "selected_classes": selected_classes,
                # 'callback' zostanie dodany przez SortingThread
            }

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
            self.sorting_thread.error_occurred.connect(self._handle_sorting_error)
            self.sorting_thread.status_changed.connect(self._update_status_label)
            self.sorting_thread.finished.connect(self._on_thread_finished)

            self.sorting_thread.start()
            self.status_label.setText("Sortowanie wybranych kategorii w toku...")
