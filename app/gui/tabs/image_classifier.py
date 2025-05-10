import datetime
import json
import os

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
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


class ClassificationHistoryThread(QThread):
    """Wątek do obsługi historii klasyfikacji."""

    history_loaded = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    def __init__(self, history_file):
        super().__init__()
        self.history_file = history_file

    def run(self):
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
                self.history_loaded.emit(history)
            else:
                self.history_loaded.emit([])
        except Exception as e:
            self.error_occurred.emit(str(e))


class BatchClassificationThread(QThread):
    """Wątek do obsługi wsadowej klasyfikacji."""

    progress_updated = pyqtSignal(int)
    result_ready = pyqtSignal(dict)
    finished = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, files, classifier):
        super().__init__()
        self.files = files
        self.classifier = classifier
        self._stop_flag = False

    def run(self):
        try:
            total = len(self.files)
            for i, file_path in enumerate(self.files):
                if self._stop_flag:
                    break

                try:
                    if self.classifier and hasattr(self.classifier, "predict"):
                        result = self.classifier.predict(file_path)
                        result["file_path"] = file_path
                        self.result_ready.emit(result)
                    else:
                        self.error_occurred.emit(
                            f"Błąd: Klasyfikator nie jest gotowy dla "
                            f"{os.path.basename(file_path)}"
                        )
                except Exception as e:
                    self.error_occurred.emit(
                        f"Błąd klasyfikacji {os.path.basename(file_path)}: " f"{str(e)}"
                    )

                progress = int((i + 1) / total * 100) if total > 0 else 0
                self.progress_updated.emit(progress)

        except Exception as e:
            self.error_occurred.emit(f"Błąd wątku wsadowego: {str(e)}")
        finally:
            self.finished.emit()

    def stop(self):
        self._stop_flag = True


class ImageClassifierTab(QWidget, TabInterface):
    """Klasa zarządzająca zakładką klasyfikacji pojedynczych obrazów."""

    def __init__(self, parent=None, settings=None):
        super().__init__(parent)
        self.parent = parent
        self.settings = settings
        self.current_image_path = None
        self.classification_history = []
        self.history_file = os.path.join("data", "classification_history.json")
        self.batch_thread = None
        self.setAcceptDrops(True)  # Umożliwia drag and drop dla całej zakładki
        self.setup_ui()
        self.connect_signals()
        self.load_classification_history()

    def setup_ui(self):
        """Tworzy i konfiguruje elementy interfejsu zakładki."""
        # Główny układ poziomy dla dwóch kolumn
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Lewa kolumna - Obraz i Historia klasyfikacji
        left_column = QWidget()
        left_layout = QVBoxLayout(left_column)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Nagłówek sekcji klasyfikacji
        classify_header = QLabel("KLASYFIKACJA OBRAZU")
        classify_header.setStyleSheet(
            "font-weight: bold; color: #CCCCCC; "
            "font-size: 11px; padding-bottom: 4px;"
        )
        left_layout.addWidget(classify_header)

        # Podgląd obrazu
        self.image_preview = QLabel("Wybierz obraz do klasyfikacji")
        self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview.setMinimumHeight(200)
        self.image_preview.setStyleSheet(
            "background-color: #1C1C1C; "
            "border: 1px solid #3F3F46; color: #AAA;"
        )
        left_layout.addWidget(self.image_preview, 1)

        # Przyciski operacji
        buttons_layout = QHBoxLayout()

        select_image_btn = QPushButton("Wybierz obraz")
        select_image_btn.clicked.connect(self._select_image)
        select_image_btn.setFixedHeight(24)
        buttons_layout.addWidget(select_image_btn)

        classify_btn = QPushButton("Klasyfikuj")
        classify_btn.clicked.connect(self._classify_image)
        classify_btn.setFixedHeight(24)
        buttons_layout.addWidget(classify_btn)

        buttons_layout.addStretch()
        left_layout.addLayout(buttons_layout)

        # Nagłówek sekcji historii
        history_header = QLabel("HISTORIA KLASYFIKACJI")
        history_header.setStyleSheet(
            "font-weight: bold; color: #CCCCCC; "
            "font-size: 11px; padding-bottom: 4px;"
        )
        left_layout.addWidget(history_header)

        # Tabela historii
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(4)
        self.history_table.setHorizontalHeaderLabels(
            ["Data", "Obraz", "Klasa", "Pewność"]
        )
        self.history_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.history_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.history_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self.history_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.ResizeToContents
        )
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setAlternatingRowColors(True)
        self.history_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.history_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.history_table.itemDoubleClicked.connect(self._show_history_item)
        left_layout.addWidget(self.history_table)

        # Przyciski historii
        history_buttons = QHBoxLayout()

        clear_history_btn = QPushButton("Wyczyść historię")
        clear_history_btn.clicked.connect(self._clear_history)
        clear_history_btn.setFixedHeight(24)
        history_buttons.addWidget(clear_history_btn)

        export_history_btn = QPushButton("Eksportuj historię")
        export_history_btn.clicked.connect(self._export_history)
        export_history_btn.setFixedHeight(24)
        history_buttons.addWidget(export_history_btn)

        history_buttons.addStretch()
        left_layout.addLayout(history_buttons)

        # Prawa kolumna - Klasyfikacja kategorii
        right_column = QWidget()
        right_layout = QVBoxLayout(right_column)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Nagłówek sekcji kategorii
        categories_header = QLabel("KLASYFIKACJA KATEGORII")
        categories_header.setStyleSheet(
            "font-weight: bold; color: #CCCCCC; "
            "font-size: 11px; padding-bottom: 4px;"
        )
        right_layout.addWidget(categories_header)

        # Wyniki klasyfikacji (dla pojedynczego obrazu)
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(["Kategoria", "Pewność"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setAlternatingRowColors(True)
        right_layout.addWidget(self.results_table, 1)

        # Pasek postępu dla klasyfikacji wsadowej
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        right_layout.addWidget(self.progress_bar)

        # Dodaj kolumny do głównego układu
        main_layout.addWidget(left_column, 1)
        main_layout.addWidget(right_column, 1)

    def connect_signals(self):
        """Podłącza sygnały do slotów."""
        pass

    def refresh(self):
        """Odświeża zawartość zakładki."""
        self.load_classification_history()

    def update_settings(self, settings):
        """Aktualizuje ustawienia zakładki."""
        self.settings = settings

    def save_state(self):
        """Zapisuje stan zakładki."""
        state = {}
        if self.current_image_path:
            state["current_image_path"] = self.current_image_path
        return state

    def restore_state(self, state):
        """Przywraca zapisany stan zakładki."""
        if state:
            self.current_image_path = state.get("current_image_path")
            if self.current_image_path and os.path.exists(self.current_image_path):
                self._show_image_preview(self.current_image_path)
            else:
                self.current_image_path = None
                self.image_preview.setText("Wybierz obraz do klasyfikacji")
                self.image_preview.setPixmap(QPixmap())

    def dragEnterEvent(self, event):
        """Obsługuje zdarzenie wejścia przeciąganego obiektu."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Obsługuje zdarzenie upuszczenia obiektu."""
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.DropAction.CopyAction)
            # Pobierz pierwszą ścieżkę pliku (zakładamy pojedynczy plik)
            url = event.mimeData().urls()[0]
            file_path = url.toLocalFile()

            # Sprawdź, czy plik jest obrazem (opcjonalne, ale zalecane)
            # Możesz dodać bardziej zaawansowaną walidację typów plików
            if file_path.lower().endswith(
                ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
            ):
                self.current_image_path = file_path
                self._show_image_preview(file_path)
                self._classify_image()  # Automatyczna klasyfikacja
                event.acceptProposedAction()
            else:
                QMessageBox.warning(
                    self, "Błąd",
                    "Upuszczony plik nie jest obsługiwanym obrazem."
                )
                event.ignore()
        else:
            event.ignore()

    def _select_image(self):
        """Wybiera obraz do klasyfikacji i wyświetla jego podgląd."""
        last_dir = self.settings.get("last_single_image_dir", os.path.expanduser("~"))
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Wybierz obraz do klasyfikacji",
            last_dir,
            "Obrazy (*.png *.jpg *.jpeg *.bmp *.gif);;Wszystkie pliki (*.*)",
        )

        if file_path:
            self.current_image_path = file_path
            self.settings["last_single_image_dir"] = os.path.dirname(file_path)
            self._show_image_preview(file_path)
            self.results_table.setRowCount(0)

    def _show_image_preview(self, image_path):
        """Wyświetla podgląd wybranego obrazu."""
        try:
            image = QImage(image_path)
            if image.isNull():
                raise ValueError("Nie można wczytać obrazu")

            pixmap = QPixmap.fromImage(image)
            scaled_pixmap = pixmap.scaled(
                self.image_preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.image_preview.setPixmap(scaled_pixmap)
        except Exception as e:
            self.current_image_path = None
            self.image_preview.setText(f"Błąd wczytania podglądu:\n{str(e)}")
            self.image_preview.setPixmap(QPixmap())
            QMessageBox.critical(
                self, "Błąd", f"Nie udało się wyświetlić podglądu obrazu: {str(e)}"
            )

    def _classify_image(self, image_path=None):
        """Klasyfikuje wybrany obraz."""
        try:
            if not image_path:
                image_path = self.current_image_path

            if not image_path:
                QMessageBox.warning(
                    self, "Ostrzeżenie", "Najpierw wybierz obraz do " "klasyfikacji."
                )
                return

            # Sprawdź, czy klasyfikator jest załadowany
            if not hasattr(self.parent, "classifier") or not self.parent.classifier:
                QMessageBox.information(
                    self,
                    "Brak modelu",
                    "Najpierw załaduj model klasyfikacji w zakładce "
                    "'Zarządzanie modelami'.",
                )
                self.parent.tab_widget.setCurrentWidget(self.parent.model_manager_tab)
                return

            # Jeśli klasyfikator jest załadowany, kontynuuj klasyfikację
            self.parent.logger.info(f"Klasyfikacja obrazu: {image_path}")
            # Wyczyść poprzednie wyniki dla pojedynczej klasyfikacji
            self.results_table.setRowCount(0)

            # Klasyfikuj obraz (używamy metody predict, zakładając że zwraca odpowiedni słownik)
            results = self.parent.classifier.predict(image_path, return_ranking=True)
            self._display_classification_results(results)
            self._add_to_history(image_path, results)

            self.parent.logger.info("Klasyfikacja zakończona.")
            # Nie przełączaj automatycznie zakładek po klasyfikacji
            # self.parent.tab_widget.setCurrentWidget(self.parent.model_manager_tab)

        except Exception as e:
            self._display_classification_error(str(e))

    def _display_classification_results(self, results):
        """Wyświetla wyniki klasyfikacji."""
        if not results or "class_name" not in results or not results["class_name"]:
            self.parent.logger.info("Nie można sklasyfikować obrazu.")
            return

        # Aktualizacja tabeli wyników
        if "class_ranking" in results:
            # Wyświetl pełny ranking klas
            ranking = results["class_ranking"]
            self.results_table.setRowCount(len(ranking))

            for i, cls in enumerate(ranking):
                self.results_table.setItem(i, 0, QTableWidgetItem(cls["class_name"]))
                self.results_table.setItem(
                    i, 1, QTableWidgetItem(f"{cls['confidence']:.2%}")
                )
        else:
            # Tryb kompatybilności wstecznej - tylko najlepsza klasa
            self.results_table.setRowCount(1)
            self.results_table.setItem(0, 0, QTableWidgetItem(results["class_name"]))
            self.results_table.setItem(
                0, 1, QTableWidgetItem(f"{results['confidence']:.2%}")
            )

        self.parent.logger.info("Klasyfikacja zakończona.")

    def _display_classification_error(self, error_message):
        """Wyświetla błąd klasyfikacji."""
        self.parent.logger.info(f"Błąd klasyfikacji: {error_message}")
        self.parent.statusBar().showMessage(f"Błąd klasyfikacji: {error_message}")
        QMessageBox.critical(
            self,
            "Błąd klasyfikacji",
            f"Wystąpił błąd podczas klasyfikacji: {error_message}",
        )

    def _start_batch_classification(self):
        """Rozpoczyna wsadową klasyfikację obrazów."""
        try:
            # Wybierz pliki do klasyfikacji
            file_paths, _ = QFileDialog.getOpenFileNames(
                self,
                "Wybierz obrazy do klasyfikacji",
                "",
                "Obrazy (*.png *.jpg *.jpeg *.bmp *.gif);;Wszystkie pliki (*.*)",
            )

            if not file_paths:
                return

            if not hasattr(self.parent, "classifier") or not self.parent.classifier:
                QMessageBox.warning(
                    self, "Ostrzeżenie", "Najpierw załaduj model klasyfikacji."
                )
                return

            # Wyczyść poprzednie wyniki
            self.results_table.setRowCount(0)
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)

            # Utwórz i uruchom wątek klasyfikacji
            self.batch_thread = BatchClassificationThread(
                file_paths, self.parent.classifier
            )
            self.batch_thread.progress_updated.connect(self._update_batch_progress)
            self.batch_thread.result_ready.connect(self._handle_batch_result)
            self.batch_thread.error_occurred.connect(self._handle_batch_error)
            self.batch_thread.finished.connect(self._batch_classification_finished)
            self.batch_thread.start()

        except Exception as e:
            QMessageBox.critical(
                self, "Błąd", f"Nie udało się rozpocząć klasyfikacji wsadowej: {str(e)}"
            )

    def _update_batch_progress(self, progress):
        """Aktualizuje pasek postępu klasyfikacji wsadowej."""
        self.progress_bar.setValue(progress)

    def _handle_batch_result(self, result):
        """Obsługuje wynik klasyfikacji wsadowej."""
        # Dodaj wynik do tabeli
        row = self.results_table.rowCount()

        if "class_ranking" in result:
            # Dodaj wszystkie klasy z rankingu
            for cls in result["class_ranking"]:
                self.results_table.insertRow(row)
                self.results_table.setItem(row, 0, QTableWidgetItem(cls["class_name"]))
                self.results_table.setItem(
                    row, 1, QTableWidgetItem(f"{cls['confidence']:.2%}")
                )
                row += 1
        else:
            # Tryb kompatybilności wstecznej - tylko najlepsza klasa
            self.results_table.insertRow(row)
            self.results_table.setItem(row, 0, QTableWidgetItem(result["class_name"]))
            self.results_table.setItem(
                row, 1, QTableWidgetItem(f"{result['confidence']:.2%}")
            )

        # Dodaj do historii (tylko najlepsza klasa)
        self._add_to_history(result["file_path"], result)

    def _handle_batch_error(self, error_message):
        """Obsługuje błąd klasyfikacji wsadowej."""
        self.parent.logger.info(f"Błąd klasyfikacji wsadowej: {error_message}")

    def _batch_classification_finished(self):
        """Obsługuje zakończenie klasyfikacji wsadowej."""
        self.progress_bar.setVisible(False)
        QMessageBox.information(
            self, "Sukces", "Klasyfikacja wsadowa została zakończona."
        )

    def load_classification_history(self):
        """Wczytuje historię klasyfikacji z pliku."""
        self.history_thread = ClassificationHistoryThread(self.history_file)
        self.history_thread.history_loaded.connect(self._display_history)
        self.history_thread.error_occurred.connect(
            lambda msg: self.parent.logger.info(
                f"Błąd wczytywania historii: {msg}"
            )
        )
        self.history_thread.start()

    def _display_history(self, history):
        """Wyświetla historię klasyfikacji w tabeli."""
        self.classification_history = history
        self.history_table.setRowCount(0)

        for item in history:
            row = self.history_table.rowCount()
            self.history_table.insertRow(row)

            # Data
            date = datetime.datetime.fromisoformat(item["date"]).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            self.history_table.setItem(row, 0, QTableWidgetItem(date))

            # Nazwa pliku
            filename = os.path.basename(item["image_path"])
            self.history_table.setItem(row, 1, QTableWidgetItem(filename))

            # Klasa
            self.history_table.setItem(row, 2, QTableWidgetItem(item["class_name"]))

            # Pewność
            self.history_table.setItem(
                row, 3, QTableWidgetItem(f"{item['confidence']:.2%}")
            )

    def _add_to_history(self, image_path, result):
        """Dodaje wynik klasyfikacji do historii."""
        history_item = {
            "date": datetime.datetime.now().isoformat(),
            "image_path": image_path,
            "class_name": result["class_name"],
            "confidence": result["confidence"],
        }

        self.classification_history.append(history_item)
        self._save_history()
        self._display_history(self.classification_history)  # Odśwież widok

    def _save_history(self):
        """Zapisuje historię klasyfikacji do pliku."""
        try:
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self.classification_history, f, indent=4, ensure_ascii=False)
        except Exception as e:
            self.parent.logger.info(f"Błąd zapisywania historii: {str(e)}")

    def _clear_history(self):
        """Czyści historię klasyfikacji."""
        reply = QMessageBox.question(
            self,
            "Potwierdzenie",
            "Czy na pewno chcesz wyczyścić historię klasyfikacji?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.classification_history = []
            self._save_history()
            self._display_history([])

    def _export_history(self):
        """Eksportuje historię klasyfikacji do pliku CSV."""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Zapisz historię klasyfikacji",
                "",
                "Pliki CSV (*.csv)",
            )

            if not file_path:
                return

            with open(file_path, "w", encoding="utf-8") as f:
                f.write("Data,Obraz,Klasa,Pewność\n")
                for item in self.classification_history:
                    date = datetime.datetime.fromisoformat(item["date"]).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    filename = os.path.basename(item["image_path"])
                    f.write(
                        f"{date},{filename},{item['class_name']},"
                        f"{item['confidence']:.2%}\n"
                    )

            QMessageBox.information(
                self,
                "Sukces",
                f"Historia klasyfikacji została wyeksportowana do:\n{file_path}",
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Błąd", f"Nie udało się wyeksportować historii: {str(e)}"
            )

    def _show_history_item(self, item):
        """Wyświetla wybrany element z historii."""
        try:
            row = item.row()
            image_path = self.classification_history[row]["image_path"]

            if os.path.exists(image_path):
                self.current_image_path = image_path
                self._show_image_preview(image_path)
                self._classify_image(image_path)
            else:
                QMessageBox.warning(
                    self,
                    "Ostrzeżenie",
                    f"Nie znaleziono pliku obrazu:\n{image_path}",
                )

        except Exception as e:
            QMessageBox.critical(
                self, "Błąd", f"Nie udało się wyświetlić elementu historii: {str(e)}"
            )
