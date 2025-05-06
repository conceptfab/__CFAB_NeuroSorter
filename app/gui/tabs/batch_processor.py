import os
from typing import List

from PyQt6.QtCore import QCoreApplication
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.gui.tab_interface import TabInterface
from app.metadata.metadata_manager import MetadataManager
from app.sorter.image_sorter import ImageSorter


class BatchProcessor(QWidget, TabInterface):
    """Klasa zarządzająca zakładką klasyfikacji wsadowej."""

    def __init__(self, parent=None, settings=None):
        super().__init__(parent)
        self.parent = parent
        self.settings = settings
        self.metadata_manager = MetadataManager()
        self.image_sorter = None
        self.setup_ui()
        self.connect_signals()

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
        self.status_label = QLabel("Gotowy")
        self.control_layout.addWidget(start_btn)
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
        results_layout.addWidget(self.results_table)
        results_group.setLayout(results_layout)
        main_layout.addWidget(results_group)

        main_layout.addStretch()

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
        """Rozpoczyna przetwarzanie plików."""
        if not hasattr(self, "input_dir") or not self.input_dir:
            QMessageBox.warning(
                self,
                "Brak katalogu",
                "Wybierz katalog źródłowy przed rozpoczęciem sortowania.",
            )
            return
        if not hasattr(self, "output_dir") or not self.output_dir:
            QMessageBox.warning(
                self,
                "Brak katalogu",
                "Wybierz katalog docelowy przed rozpoczęciem sortowania.",
            )
            return

        # Sprawdź, czy katalog wejściowy i wyjściowy są różne, jeśli przenosimy
        if (
            not self.copy_files_checkbox.isChecked()
            and self.input_dir == self.output_dir
        ):
            QMessageBox.critical(
                self,
                "Błąd konfiguracji",
                "Katalog źródłowy i docelowy muszą być różne, jeśli wybrano "
                "opcję przenoszenia plików.",
            )
            return

        # Sprawdź, czy klasyfikator jest załadowany w obiekcie nadrzędnym
        if not hasattr(self.parent, "classifier") or not self.parent.classifier:
            QMessageBox.warning(
                self,
                "Brak modelu",
                "Nie załadowano modelu klasyfikacji. Proszę najpierw załadować model.",
            )
            return

        # Użyj istniejącego klasyfikatora z obiektu nadrzędnego
        classifier = self.parent.classifier

        # Inicjalizacja sortera z poprawnym klasyfikatorem
        self.image_sorter = ImageSorter(
            classifier=classifier, copy_files=self.copy_files_checkbox.isChecked()
        )

        # Rozpocznij sortowanie
        try:
            self.status_label.setText("Sortowanie w toku...")
            self._clear_results()  # Wyczyść poprzednie wyniki
            self._setup_progress_bar()  # Ustaw/pokaż pasek postępu
            QApplication.processEvents()  # Odśwież UI

            stats = self.image_sorter.sort_directory(
                input_dir=self.input_dir,
                output_dir=self.output_dir,
                confidence_threshold=0.5,  # Można dodać do UI
                callback=self._update_progress,  # Callback do paska postępu
            )

            # Aktualizuj tabelę wyników
            self.results_table.setRowCount(0)
            for category, count in stats["categories"].items():
                row = self.results_table.rowCount()
                self.results_table.insertRow(row)
                self.results_table.setItem(row, 0, QTableWidgetItem("Kategoria"))
                self.results_table.setItem(row, 1, QTableWidgetItem(category))
                self.results_table.setItem(row, 2, QTableWidgetItem(str(count)))
                self.results_table.setItem(row, 3, QTableWidgetItem("Zakończono"))

            # Dodaj informację o plikach bez kategorii
            if stats["uncategorized"] > 0:
                row = self.results_table.rowCount()
                self.results_table.insertRow(row)
                self.results_table.setItem(row, 0, QTableWidgetItem("Bez kategorii"))
                self.results_table.setItem(
                    row, 1, QTableWidgetItem(self.image_sorter.uncategorized_dir)
                )
                self.results_table.setItem(
                    row, 2, QTableWidgetItem(str(stats["uncategorized"]))
                )
                self.results_table.setItem(row, 3, QTableWidgetItem("Zakończono"))

            self.status_label.setText("Sortowanie zakończone")
            QMessageBox.information(
                self,
                "Sukces",
                f"Sortowanie plików zostało zakończone pomyślnie.\n\n"
                f"Przetworzono: {stats['processed']}\n"
                f"Przeniesiono/skopiowano: {stats['moved']}\n"
                f"Pliki bez kategorii: {stats['uncategorized']}\n"
                f"Pominięto: {stats['skipped']}",
            )
        except ValueError as ve:
            # Obsługa błędu walidacji katalogów z ImageSorter
            QMessageBox.critical(self, "Błąd konfiguracji", str(ve))
            self.status_label.setText("Błąd konfiguracji")
        except Exception as e:
            QMessageBox.critical(
                self, "Błąd sortowania", f"Wystąpił nieoczekiwany błąd: {str(e)}"
            )
            self.status_label.setText("Błąd sortowania")
        finally:
            # Ukryj pasek postępu po zakończeniu (lub błędzie)
            if hasattr(self, "progress_bar"):
                self.progress_bar.setVisible(False)

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
        """Aktualizuje pasek postępu."""
        if not hasattr(self, "progress_bar"):
            self._setup_progress_bar()  # Spróbuj utworzyć, jeśli jeszcze nie ma

        percentage = int((current / total) * 100) if total > 0 else 0
        self.progress_bar.setValue(percentage)
        self.status_label.setText(f"Sortowanie: {current}/{total} ({percentage}%)")
        QCoreApplication.processEvents()  # Użyj QCoreApplication
