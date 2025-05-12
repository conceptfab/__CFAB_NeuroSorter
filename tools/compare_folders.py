import os
import sys
from collections import defaultdict

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


class FolderLoader(QThread):
    progress_updated = pyqtSignal(int)
    folder_info_ready = pyqtSignal(dict)
    tree_item_ready = pyqtSignal(str, str, str, int, bool)
    error_occurred = pyqtSignal(str)
    finished_loading = pyqtSignal()

    def __init__(self, root_path):
        super().__init__()
        self.root_path = root_path
        self.folder_names = defaultdict(list)
        self.total_items = 0
        self.processed_items = 0

    def run(self):
        try:
            # Jeden przebieg - zbieramy wszystkie informacje
            for dirpath, dirnames, filenames in os.walk(self.root_path):
                # Dodajemy foldery do licznika
                self.total_items += len(dirnames) + len(filenames)

                # Zbieramy informacje o folderach
                for dirname in dirnames:
                    full_path = os.path.join(dirpath, dirname)
                    self.folder_names[dirname].append(full_path)
                    self.processed_items += 1
                    self.progress_updated.emit(
                        int((self.processed_items / self.total_items) * 30)
                    )

            # Emitujemy informacje o folderach
            self.folder_info_ready.emit(dict(self.folder_names))

            # Budujemy drzewo
            self._build_tree(self.root_path, "", 0)

            self.finished_loading.emit()

        except Exception as e:
            self.error_occurred.emit(str(e))

    def _build_tree(self, path, parent_path, level):
        try:
            items = os.listdir(path)
            dirs = sorted(d for d in items if os.path.isdir(os.path.join(path, d)))
            files = [f for f in items if os.path.isfile(os.path.join(path, f))]

            for dirname in dirs:
                full_path = os.path.join(path, dirname)
                is_duplicate = (
                    dirname in self.folder_names and len(self.folder_names[dirname]) > 1
                )

                self.tree_item_ready.emit(
                    dirname, full_path, parent_path, len(files), is_duplicate
                )

                self.processed_items += 1
                self.progress_updated.emit(
                    30 + int((self.processed_items / self.total_items) * 70)
                )
                self._build_tree(full_path, full_path, level + 1)

        except PermissionError:
            self.error_occurred.emit(f"Brak dostępu do folderu: {path}")
        except Exception as e:
            self.error_occurred.emit(str(e))


class FolderViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Przeglądarka folderów")
        self.setGeometry(100, 100, 1200, 800)

        # Kolory zgodne z Material Design i VS Code
        self.primary_color = "#007ACC"  # Niebieski VS Code
        self.success_color = "#10B981"  # Zielony
        self.warning_color = "#DC2626"  # Czerwony
        self.background = "#1E1E1E"  # Ciemne tło
        self.surface = "#252526"  # Lekko jaśniejsze tło dla paneli
        self.border_color = "#3F3F46"  # Kolor obramowania
        self.text_color = "#CCCCCC"  # Kolor tekstu

        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)

        self.create_folder_column(main_layout, "Lewa strona")
        self.create_folder_column(main_layout, "Prawa strona")

        self.setCentralWidget(central_widget)

        self.folder_colors = {}
        self.color_index = 0
        self.colors = [
            QColor(255, 200, 200),  # jasny czerwony
            QColor(200, 255, 200),  # jasny zielony
            QColor(200, 200, 255),  # jasny niebieski
            QColor(255, 255, 200),  # jasny żółty
            QColor(255, 200, 255),  # jasny fioletowy
            QColor(200, 255, 255),  # jasny turkusowy
            QColor(255, 230, 180),  # jasny pomarańczowy
            QColor(220, 220, 220),  # jasny szary
        ]

        self.loaders = {}

        # Zastosuj style
        self._apply_styles()

    def _apply_styles(self):
        """Aplikuje style Material Design do aplikacji."""
        self.setStyleSheet(
            f"""
            QMainWindow, QDialog {{
                background-color: {self.background};
                color: {self.text_color};
            }}
            QPushButton {{
                background-color: {self.surface};
                color: {self.text_color};
                border: 1px solid {self.border_color};
                border-radius: 2px;
                padding: 4px 12px;
                min-height: 24px;
                max-height: 24px;
            }}
            QPushButton:hover {{
                background-color: #2A2D2E;
            }}
            QPushButton:pressed {{
                background-color: #3E3E40;
            }}
            QTreeWidget {{
                background-color: {self.surface};
                color: {self.text_color};
                border: 1px solid {self.border_color};
                border-radius: 2px;
            }}
            QTreeWidget::item {{
                padding: 2px;
            }}
            QTreeWidget::item:selected {{
                background-color: #264F78;
                color: white;
            }}
            QProgressBar {{
                border: 1px solid {self.border_color};
                background-color: {self.surface};
                text-align: center;
                height: 16px;
            }}
            QProgressBar::chunk {{
                background-color: {self.primary_color};
            }}
            QLabel {{
                color: {self.text_color};
            }}
            QScrollArea {{
                background-color: {self.surface};
                border: 1px solid {self.border_color};
                border-radius: 2px;
            }}
            """
        )

    def create_folder_column(self, parent_layout, title):
        column_widget = QWidget()
        column_layout = QVBoxLayout(column_widget)

        progress_bar = QProgressBar()
        progress_bar.setMinimum(0)
        progress_bar.setMaximum(100)
        progress_bar.setValue(0)
        progress_bar.setTextVisible(True)
        progress_bar.setFormat("%p%")
        column_layout.addWidget(progress_bar)

        select_folder_button = QPushButton("Wybierz folder")
        select_folder_button.clicked.connect(
            lambda: self.select_folder_for_column(column_widget)
        )
        column_layout.addWidget(select_folder_button)

        splitter = QSplitter(Qt.Orientation.Vertical)

        folder_tree = QTreeWidget()
        folder_tree.setHeaderLabels(["Nazwa folderu", "Ilość plików", "Rozmiar [MB]"])
        folder_tree.setColumnWidth(0, 300)
        folder_tree.setColumnWidth(1, 100)
        folder_tree.setColumnWidth(2, 120)
        splitter.addWidget(folder_tree)

        stats_container = QWidget()
        stats_layout = QVBoxLayout(stats_container)
        stats_layout.setContentsMargins(0, 0, 0, 0)

        duplicates_area = QScrollArea()
        duplicates_area.setWidgetResizable(True)
        duplicates_widget = QWidget()
        duplicates_layout = QVBoxLayout(duplicates_widget)
        duplicates_area.setWidget(duplicates_widget)
        stats_layout.addWidget(duplicates_area)

        splitter.addWidget(stats_container)
        splitter.setSizes([700, 300])
        column_layout.addWidget(splitter)
        parent_layout.addWidget(column_widget)

        column_widget.folder_tree = folder_tree
        column_widget.duplicates_area = duplicates_area
        column_widget.duplicates_layout = duplicates_layout
        column_widget.folder_colors = {}
        column_widget.color_index = 0
        column_widget.progress_bar = progress_bar
        column_widget.items = {}

    def select_folder_for_column(self, column_widget):
        folder_path = QFileDialog.getExistingDirectory(self, "Wybierz folder")
        if folder_path:
            self.display_folder_structure(folder_path, column_widget)

    def display_folder_structure(self, root_path, column_widget):
        column_widget.progress_bar.setValue(0)
        column_widget.folder_tree.clear()
        column_widget.folder_colors = {}
        column_widget.color_index = 0
        column_widget.items = {}

        for i in reversed(range(column_widget.duplicates_layout.count())):
            column_widget.duplicates_layout.itemAt(i).widget().setParent(None)

        loader = FolderLoader(root_path)
        self.loaders[column_widget] = loader

        loader.progress_updated.connect(column_widget.progress_bar.setValue)
        loader.folder_info_ready.connect(
            lambda folders: self._handle_folder_info(folders, column_widget, root_path)
        )
        loader.tree_item_ready.connect(
            lambda name, path, parent, files, is_duplicate: self._handle_tree_item(
                name, path, parent, files, is_duplicate, column_widget
            )
        )
        loader.error_occurred.connect(
            lambda msg: QMessageBox.warning(self, "Błąd", msg)
        )
        loader.finished_loading.connect(
            lambda: self._handle_loading_finished(column_widget)
        )

        loader.start()

    def _handle_folder_info(self, folders, column_widget, root_path):
        for name, paths in folders.items():
            if len(paths) > 1:
                if column_widget.color_index < len(self.colors):
                    column_widget.folder_colors[name] = self.colors[
                        column_widget.color_index
                    ]
                    column_widget.color_index += 1
                else:
                    column_widget.color_index = 0
                    column_widget.folder_colors[name] = self.colors[
                        column_widget.color_index
                    ]
                    column_widget.color_index += 1
                # Oblicz sumaryczny rozmiar plików dla wszystkich wystąpień folderu (rekurencyjnie)
                total_size = 0
                for p in paths:
                    total_size += self._get_folder_size(p)
                size_mb = total_size / (1024 * 1024)
                label = QLabel(
                    f"Folder: {name} (występuje {len(paths)} razy, {size_mb:.2f} MB)"
                )
                label.setStyleSheet(
                    f"background-color: rgb({column_widget.folder_colors[name].red()}, "
                    f"{column_widget.folder_colors[name].green()}, "
                    f"{column_widget.folder_colors[name].blue()}); "
                    "padding: 2px; margin: 1px; "
                    "color: black; font-weight: bold; "
                    "font-size: 12px; "
                    "min-height: 16px; max-height: 16px;"
                )
                column_widget.duplicates_layout.addWidget(label)

    def _handle_tree_item(
        self, name, path, parent_path, files, is_duplicate, column_widget
    ):
        if parent_path == "":
            item = QTreeWidgetItem(column_widget.folder_tree)
        else:
            parent_item = column_widget.items.get(parent_path)
            if parent_item:
                item = QTreeWidgetItem(parent_item)
            else:
                return

        item.setText(0, name)
        item.setText(1, str(files))
        # Oblicz rozmiar plików w tym folderze (rekurencyjnie)
        try:
            size = self._get_folder_size(path)
            size_mb = size / (1024 * 1024)
        except Exception:
            size_mb = 0.0
        item.setText(2, f"{size_mb:.2f}")

        if is_duplicate:
            for column in range(item.columnCount()):
                item.setBackground(column, QBrush(column_widget.folder_colors[name]))
                item.setForeground(column, QBrush(QColor(0, 0, 0)))

        column_widget.items[path] = item

        if parent_path == "":
            item.setExpanded(True)

    def _handle_loading_finished(self, column_widget):
        if column_widget in self.loaders:
            del self.loaders[column_widget]
        column_widget.progress_bar.setValue(100)

    def _get_folder_size(self, path):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total_size += os.path.getsize(fp)
                except Exception:
                    pass
        return total_size


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FolderViewer()
    window.show()
    sys.exit(app.exec())
