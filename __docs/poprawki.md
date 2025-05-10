Przeanalizowałem kod pliku app/utils/file_tools/data_splitter_gui.py i przygotowałem listę zmian mających na celu naprawę potencjalnych problemów oraz przygotowanie go na rozbudowę. Oto sugerowane zmiany:
Zmiany w pliku data_splitter_gui.py

1. Problemy z importami PyQt6
   markdownPlik: app/utils/file_tools/data_splitter_gui.py
   Funkcja: import
   Proponowany kod:

````python
# Zamienić
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QIcon
from PyQt6.QtWidgets import (...)

# Na
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
2. Refaktoryzacja klasy Worker
Klasa Worker zawiera zbyt wiele odpowiedzialności. Należy wydzielić logikę przetwarzania plików do osobnej klasy:
markdownPlik: app/utils/file_tools/data_splitter_gui.py
Funkcja: Worker
Proponowany kod:
```python
class FileSplitter:
    """Klasa odpowiedzialna za logikę podziału plików."""

    def __init__(self, input_dir, output_dir, split_mode, split_value, use_validation=True, selected_categories=None):
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

    # Tutaj przenieść metody z Worker związane z przetwarzaniem plików
    # get_min_files_in_selected_categories_for_processing, _generate_report

    def process_files(self, progress_callback=None, cancel_check=None):
        """
        Główna metoda procesująca pliki z obsługą postępu

        :param progress_callback: funkcja wywoływana do aktualizacji postępu (value, message)
        :param cancel_check: funkcja sprawdzająca czy operacja została anulowana
        :return: (status_string, error_message)
        """
        # Implementacja logiki obecnie znajdującej się w metodzie run() klasy Worker
markdownPlik: app/utils/file_tools/data_splitter_gui.py
Funkcja: Worker po zmianach
Proponowany kod:
```python
class Worker(QThread):
    """Wątek do przetwarzania danych w tle"""
    progress_updated = pyqtSignal(int, str)
    finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, input_dir, output_dir, split_mode, split_value, use_validation=True, selected_categories=None):
        super().__init__()
        self.splitter = FileSplitter(input_dir, output_dir, split_mode, split_value,
                                     use_validation, selected_categories)
        self.is_cancelled = False

    def run(self):
        try:
            self.progress_updated.emit(0, "Rozpoczynanie przetwarzania...")

            result, error = self.splitter.process_files(
                progress_callback=self.progress_updated.emit,
                cancel_check=lambda: self.is_cancelled
            )

            if self.is_cancelled:
                self.finished.emit("Anulowano.")
            elif error:
                self.error_occurred.emit(f"Błąd: {error}")
                self.finished.emit(f"Błąd: {error}")
            else:
                self.finished.emit(result)

        except ValueError as ve:
            self.error_occurred.emit(f"Błąd konfiguracji: {ve}")
            self.finished.emit(f"Błąd: {ve}")
        except Exception as e:
            self.error_occurred.emit(f"Niespodziewany błąd: {e}")
            self.finished.emit(f"Niespodziewany błąd: {e}")

    def cancel(self):
        self.is_cancelled = True
        self.progress_updated.emit(0, "Anulowanie...")
3. Optymalizacja metody update_folder_tree
Metoda update_folder_tree w klasie DataSplitterApp wykonuje wiele operacji na głównym wątku, co może prowadzić do blokowania interfejsu. Powinno się ją zoptymalizować:
markdownPlik: app/utils/file_tools/data_splitter_gui.py
Funkcja: update_folder_tree
Proponowany kod:
```python
def update_folder_tree(self):
    self.folder_tree.clear()
    if not self.input_dir:
        self.update_files_limit_and_validation_based_on_selection()
        return

    root_path = Path(self.input_dir)
    tree_root_item = QTreeWidgetItem(self.folder_tree, [root_path.name])
    self.folder_tree.addTopLevelItem(tree_root_item)
    tree_root_item.setExpanded(True)

    # Odpinanie sygnału
    try:
        self.folder_tree.itemChanged.disconnect(self.on_folder_tree_item_changed)
    except TypeError:
        pass

    # Użycie QApplication.processEvents aby uniknąć zamrożenia UI
    QApplication.processEvents()

    # Znajdowanie minimum w osobnej metodzie
    folder_data = self.calculate_folder_statistics(root_path)
    min_files_folder = folder_data.get('min_files_folder')
    folder_counts = folder_data.get('folder_counts', {})

    # Dodawanie folderów do drzewa
    for category_dir in root_path.iterdir():
        if category_dir.is_dir():
            direct_file_count = folder_counts.get(category_dir.name, 0)
            display_text = f"{category_dir.name} ({direct_file_count} plików)"
            item = QTreeWidgetItem(tree_root_item, [display_text])
            item.setData(0, Qt.ItemDataRole.UserRole, category_dir.name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, Qt.CheckState.Checked)

            # Wyróżnienie minimalnego folderu
            if category_dir.name == min_files_folder and folder_counts.get(category_dir.name, 0) > 0:
                item.setForeground(0, QColor(HIGHLIGHT_COLOR))

        # Co kilka iteracji przetwarzaj zdarzenia UI
        if category_dir.name.endswith('0'):
            QApplication.processEvents()

    # Ponowne podłączenie sygnału
    self.folder_tree.itemChanged.connect(self.on_folder_tree_item_changed)
    self.update_files_limit_and_validation_based_on_selection()

def calculate_folder_statistics(self, root_path):
    """Oblicza statystyki folderów w tle"""
    min_files_count = float("inf")
    min_files_folder = None
    folder_counts = {}

    for category_dir in root_path.iterdir():
        if category_dir.is_dir():
            direct_file_count = sum(
                1
                for f in category_dir.iterdir()
                if f.is_file() and f.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
            )
            folder_counts[category_dir.name] = direct_file_count
            if direct_file_count > 0 and direct_file_count < min_files_count:
                min_files_count = direct_file_count
                min_files_folder = category_dir.name

    return {
        'min_files_count': min_files_count if min_files_count != float("inf") else 0,
        'min_files_folder': min_files_folder,
        'folder_counts': folder_counts
    }
4. Przebudowa systemu aktualizacji UI w update_files_limit_and_validation_based_on_selection
Metoda update_files_limit_and_validation_based_on_selection jest zbyt długa i złożona, co utrudnia jej utrzymanie:
markdownPlik: app/utils/file_tools/data_splitter_gui.py
Funkcja: update_files_limit_and_validation_based_on_selection
Proponowany kod:
```python
def update_files_limit_and_validation_based_on_selection(self):
    """Aktualizuje stan kontrolek na podstawie wybranych kategorii"""
    selected_cats = self.get_selected_categories_names()
    min_files_in_selection, folders_with_min = self.get_min_files_in_selected_categories(selected_cats)

    # Aktualizuj stan SpinBoxa dla plików
    self._update_files_spin_state(selected_cats, min_files_in_selection)

    # Aktualizuj stan checkboxa walidacji
    self._update_validation_checkbox_state(selected_cats, min_files_in_selection)

def _update_files_spin_state(self, selected_cats, min_files_in_selection):
    """Aktualizuje stan kontrolki files_spin"""
    if not selected_cats:
        self.files_spin.setEnabled(False)
        self.files_spin.setValue(1)
        self.log_message("Wybierz kategorie, aby dostosować opcje podziału.")
    else:
        self.files_spin.setEnabled(self.mode_combo.currentIndex() == 1)
        if min_files_in_selection > 0:
            self.files_spin.setMaximum(min_files_in_selection)
            if self.files_spin.value() > min_files_in_selection:
                self.files_spin.setValue(min_files_in_selection)
        else:
            self.files_spin.setMaximum(1)
            self.files_spin.setValue(1)

def _update_validation_checkbox_state(self, selected_cats, min_files_in_selection):
    """Aktualizuje stan checkboxa walidacji i jego etykiety"""
    current_files_limit = self.files_spin.value()

    if self.mode_combo.currentIndex() == 0:  # Tryb procentowy
        self._update_validation_for_percent_mode(selected_cats)
    else:  # Tryb limitu plików
        self._update_validation_for_files_mode(selected_cats, min_files_in_selection, current_files_limit)

def _update_validation_for_percent_mode(self, selected_cats):
    """Aktualizuje stan walidacji dla trybu procentowego"""
    can_create_validation = self.split_slider.value() < 100
    self.validation_check.setEnabled(can_create_validation and bool(selected_cats))

    if not can_create_validation:
        self.validation_check.setChecked(False)
        self.validation_label.setText("(100% tren.)")
    else:
        self.validation_label.setText(f"({100 - self.split_slider.value()}% walid.)")

def _update_validation_for_files_mode(self, selected_cats, min_files_in_selection, current_files_limit):
    """Aktualizuje stan walidacji dla trybu limitu plików"""
    can_create_validation = (
        current_files_limit < min_files_in_selection
        and min_files_in_selection > 0
        and bool(selected_cats)
    )
    self.validation_check.setEnabled(can_create_validation)

    if not can_create_validation:
        self.validation_check.setChecked(False)
        self._set_validation_label_for_files_mode(selected_cats, min_files_in_selection)
    else:
        max_valid = min_files_in_selection - current_files_limit
        self.validation_label.setText(f"(do {max_valid} walid.)")

def _set_validation_label_for_files_mode(self, selected_cats, min_files_in_selection):
    """Ustawia tekst etykiety walidacji dla trybu plików gdy walidacja jest niedostępna"""
    if not selected_cats:
        self.validation_label.setText("(wybierz kat.)")
    elif min_files_in_selection == 0:
        self.validation_label.setText("(brak plików)")
    else:
        self.validation_label.setText("(limit >= min.)")
5. Dodanie systemu logowania
Dla lepszego debugowania i rozbudowy, dodajmy profesjonalny system logowania:
markdownPlik: app/utils/file_tools/data_splitter_gui.py
Dodanie na początku pliku:
```python
import logging
from datetime import datetime

# Konfiguracja logowania
def setup_logger():
    log_file = Path("logs") / f"data_splitter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file.parent.mkdir(exist_ok=True)

    logger = logging.getLogger("DataSplitter")
    logger.setLevel(logging.DEBUG)

    # Handler pliku
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Handler konsoli
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Dodaj handlery
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logger()
markdownPlik: app/utils/file_tools/data_splitter_gui.py
Modyfikacja metody log_message:
```python
def log_message(self, message, level=logging.INFO):
    """Loguje wiadomość w interfejsie i w systemie logowania"""
    self.log_edit.append(message)
    self.log_edit.verticalScrollBar().setValue(
        self.log_edit.verticalScrollBar().maximum()
    )
    QApplication.processEvents()

    # Logowanie do pliku
    if level == logging.DEBUG:
        logger.debug(message)
    elif level == logging.INFO:
        logger.info(message)
    elif level == logging.WARNING:
        logger.warning(message)
    elif level == logging.ERROR:
        logger.error(message)
    elif level == logging.CRITICAL:
        logger.critical(message)
6. Dodanie obsługi konfiguracji
Zamiast sztywnych stałych konfiguracyjnych, zaimplementujmy system konfiguracji:
markdownPlik: app/utils/file_tools/config.py
Nowy plik:
```python
import json
from pathlib import Path

DEFAULT_CONFIG = {
    "folders": {
        "train_folder_name": "__dane_treningowe",
        "valid_folder_name": "__dane_walidacyjne",
    },
    "defaults": {
        "train_split_percent": 80,
        "files_per_category": 100,
    },
    "extensions": {
        "allowed_image_extensions": [
            ".png", ".webp", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".tif"
        ]
    },
    "ui": {
        "colors": {
            "primary_color": "#007ACC",
            "background": "#1E1E1E",
            "surface": "#252526",
            "border_color": "#3F3F46",
            "text_color": "#CCCCCC",
            "highlight_color": "#FF0000"
        }
    }
}

class Config:
    """Klasa przechowująca konfigurację aplikacji"""

    def __init__(self):
        self.config_path = Path("config") / "data_splitter_config.json"
        self.config = DEFAULT_CONFIG
        self.load()

    def load(self):
        """Ładuje konfigurację z pliku"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    # Aktualizuj tylko istniejące klucze
                    self._update_config_recursive(self.config, loaded_config)
            except Exception as e:
                print(f"Błąd ładowania konfiguracji: {e}")

    def save(self):
        """Zapisuje konfigurację do pliku"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Błąd zapisywania konfiguracji: {e}")

    def _update_config_recursive(self, target, source):
        """Aktualizuje konfigurację rekurencyjnie"""
        for key, value in source.items():
            if key in target:
                if isinstance(value, dict) and isinstance(target[key], dict):
                    self._update_config_recursive(target[key], value)
                else:
                    target[key] = value

    def get(self, section, key, default=None):
        """Pobiera wartość z konfiguracji"""
        try:
            return self.config[section][key]
        except KeyError:
            return default

    def set(self, section, key, value):
        """Ustawia wartość w konfiguracji"""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        self.save()

# Singleton konfiguracji
config = Config()
markdownPlik: app/utils/file_tools/data_splitter_gui.py
Zastąpienie stałych:
```python
# Importowanie konfiguracji
from app.utils.file_tools.config import config

# Zastąpienie stałych
TRAIN_FOLDER_NAME = config.get("folders", "train_folder_name")
VALID_FOLDER_NAME = config.get("folders", "valid_folder_name")
DEFAULT_TRAIN_SPLIT_PERCENT = config.get("defaults", "train_split_percent")
DEFAULT_FILES_PER_CATEGORY = config.get("defaults", "files_per_category")
ALLOWED_IMAGE_EXTENSIONS = tuple(config.get("extensions", "allowed_image_extensions"))

# Style
PRIMARY_COLOR = config.get("ui", "colors")["primary_color"]
BACKGROUND = config.get("ui", "colors")["background"]
SURFACE = config.get("ui", "colors")["surface"]
BORDER_COLOR = config.get("ui", "colors")["border_color"]
TEXT_COLOR = config.get("ui", "colors")["text_color"]
HIGHLIGHT_COLOR = config.get("ui", "colors")["highlight_color"]
7. Poprawa obsługi wyjątków
Warto poprawić obsługę wyjątków w kodzie, aby zapewnić lepszą obsługę błędów:
markdownPlik: app/utils/file_tools/data_splitter_gui.py
Dodanie klasy niestandardowego wyjątku:
```python
class DataSplitterError(Exception):
    """Bazowa klasa dla wyjątków w aplikacji Data Splitter"""
    pass

class ConfigurationError(DataSplitterError):
    """Błędy związane z konfiguracją (ścieżki, wartości, etc.)"""
    pass

class ProcessingError(DataSplitterError):
    """Błędy występujące podczas przetwarzania plików"""
    pass
markdownPlik: app/utils/file_tools/data_splitter_gui.py
Zastąpienie zwykłych raises w procesie kopiowania plików:
```python
# Zamiast
if not self.input_dir.is_dir():
    raise ValueError(f"Folder wejściowy nie istnieje: {self.input_dir}")

# Użyj
if not self.input_dir.is_dir():
    raise ConfigurationError(f"Folder wejściowy nie istnieje: {self.input_dir}")
8. Refaktoryzacja ReportDialog dla lepszej prezentacji
Dialog raportu można usprawnić dla lepszej czytelności:
markdownPlik: app/utils/file_tools/data_splitter_gui.py
Funkcja: ReportDialog._format_report_to_html
Proponowany kod:
```python
def _format_report_to_html(self, report_text):
    """Formatuje raport tekstowy na HTML z lepszym stylowaniem"""
    # Podziel na linie
    lines = report_text.split('\n')
    html_parts = []

    in_section = False
    section_content = []

    for line in lines:
        # Nagłówki
        if line.startswith('==='):
            if in_section:
                # Zamknij poprzednią sekcję
                html_parts.append(f"<div class='section'><ul>{''.join(section_content)}</ul></div>")
                section_content = []

            # Usuń znaki === z początku i końca linii
            header_text = line.replace('===', '').strip()
            html_parts.append(f"<h3>{header_text}</h3>")
            in_section = True

        # Puste linie
        elif not line.strip():
            continue

        # Elementy listy (kategorie)
        elif line.startswith('├──') or line.startswith('└──'):
            indent = '&nbsp;&nbsp;&nbsp;&nbsp;'
            item_text = line.replace('├──', '').replace('└──', '').strip()
            section_content.append(f"<li>{indent}{item_text}</li>")

        # Nazwy kategorii
        elif in_section and not (line.startswith(' ') or line.startswith('-')):
            if section_content:  # Jeśli mamy zawartość poprzedniej kategorii
                html_parts.append(f"<div class='section'><ul>{''.join(section_content)}</ul></div>")
                section_content = []
            section_content.append(f"<li><strong>{line}</strong></li>")

        # Normalne linie
        else:
            if in_section:
                section_content.append(f"<li>{line}</li>")
            else:
                html_parts.append(f"<p>{line}</p>")

    # Dodaj ostatnią sekcję jeśli istnieje
    if in_section and section_content:
        html_parts.append(f"<div class='section'><ul>{''.join(section_content)}</ul></div>")

    # Style CSS
    css = f"""
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; color: {TEXT_COLOR}; background-color: {BACKGROUND}; padding: 10px; }}
        h3 {{ color: {PRIMARY_COLOR}; border-bottom: 1px solid {BORDER_COLOR}; padding-bottom: 5px; margin-top: 15px; }}
        .section {{ margin-bottom: 15px; }}
        ul {{ list-style-type: none; padding-left: 10px; }}
        li {{ margin-bottom: 2px; }}
        strong {{ color: {PRIMARY_COLOR}; }}
    </style>
    """

    full_html = f"<html><head>{css}</head><body>{''.join(html_parts)}</body></html>"
    return full_html
9. Dodanie klasy ułatwiającej rozbudowę programu
Propozycja dodania wzorca Model-View-Controller (MVC), co ułatwi rozbudowę aplikacji:
markdownPlik: app/utils/file_tools/models.py
Nowy plik:
```python
from pathlib import Path

class DataCategory:
    """Reprezentuje kategorię danych (folder) w strukturze danych"""

    def __init__(self, name, path):
        self.name = name
        self.path = Path(path)
        self.file_count = 0
        self.is_selected = True
        self.files = []
        self.scan()

    def scan(self, allowed_extensions=None):
        """Skanuje folder w poszukiwaniu plików"""
        if not allowed_extensions:
            from app.utils.file_tools.config import config
            allowed_extensions = tuple(config.get("extensions", "allowed_image_extensions"))

        self.files = [
            f for f in self.path.iterdir()
            if f.is_file() and f.suffix.lower() in allowed_extensions
        ]
        self.file_count = len(self.files)
        return self.file_count

    def __str__(self):
        return f"{self.name} ({self.file_count} plików)"


class DataStructure:
    """Model reprezentujący strukturę danych wejściowych"""

    def __init__(self):
        self.root_path = None
        self.categories = []
        self.min_files_category = None

    def load(self, root_path):
        """Ładuje strukturę kategorii z podanego folderu"""
        self.root_path = Path(root_path)
        self.categories = []

        if not self.root_path.exists() or not self.root_path.is_dir():
            return False

        # Znajdź wszystkie podfoldery (kategorie)
        for folder in self.root_path.iterdir():
            if folder.is_dir():
                category = DataCategory(folder.name, folder)
                self.categories.append(category)

        # Znajdź kategorię z najmniejszą liczbą plików
        self._find_min_category()
        return True

    def _find_min_category(self):
        """Znajduje kategorię z najmniejszą niezerową liczbą plików"""
        min_files = float('inf')
        min_category = None

        for category in self.categories:
            if 0 < category.file_count < min_files:
                min_files = category.file_count
                min_category = category

        self.min_files_category = min_category

    def get_selected_categories(self):
        """Zwraca listę wybranych kategorii"""
        return [cat for cat in self.categories if cat.is_selected]

    def get_min_files_in_selected(self):
        """Zwraca minimalną liczbę plików wśród wybranych kategorii"""
        selected = self.get_selected_categories()
        if not selected:
            return 0

        min_files = min((cat.file_count for cat in selected if cat.file_count > 0), default=0)
        return min_files

    def set_all_selected(self, selected):
        """Zaznacza lub odznacza wszystkie kategorie"""
        for category in self.categories:
            category.is_selected = selected
10. Optymalizacja wczytywania plików
Aktualna metoda update_files_list jest nieefektywna dla dużej liczby plików:
markdownPlik: app/utils/file_tools/data_splitter_gui.py
Funkcja: update_files_list
Proponowany kod:
```python
def update_files_list(self):
    """Aktualizuje listę wszystkich plików w wybranych katalogach"""
    self.files_list_widget.clear()
    self.files_list = []

    if not self.input_dir:
        return

    self.files_list_widget.addItem("Trwa skanowanie plików...")
    QApplication.processEvents()

    root_path = Path(self.input_dir)

    # Używamy oddzielnego wątku do skanowania plików
    class FileScanner(QThread):
        finished = pyqtSignal(list)
        progress = pyqtSignal(int)

        def __init__(self, root_path, allowed_extensions):
            super().__init__()
            self.root_path = root_path
            self.allowed_extensions = allowed_extensions
            self.files = []

        def run(self):
            total_found = 0
            for item in self.root_path.rglob("*"):
                if item.is_file() and item.suffix.lower() in self.allowed_extensions:
                    relative_path = item.relative_to(self.root_path)
                    self.files.append((item, str(relative_path)))
                    total_found += 1
                    if total_found % 100 == 0:
                        self.progress.emit(total_found)
            self.finished.emit(self.files)

    # Funkcje callback
    def on_files_found(files):
        self.files_list_widget.clear()
        self.files_list = [f[0] for f in files]  # Zapisz pełne ścieżki

        # Dodaj pierwsze 1000 plików do widoku (lub mniej, jeśli jest ich mniej)RetryClaude hit the max length for a message and has paused its response. You can write Continue to keep the chat going.MContinueEditPyQt6 thread worker pattern optimization10 resultsUse PyQt's QThread to Prevent Freezing GUIs – Real Pythonrealpython.comMultithreading PyQt6 applications with QThreadPoolpythonguis.comPySide6.QtCore.QThread - Qt for Pythonqt.ioNo More Freezing using Qt Multithreading: Learn with Examples | Tech Art Learningxingyulei.comPyQt6: Multithreadingcoderscratchpad.compython - Direct replacement for threading.Thread() in pyQT6, to run functions in thread without subclassing - Stack Overflowstackoverflow.compython - Background thread with QThread in PyQt - Stack Overflowstackoverflow.comThreads & Processes in PyQt6 &mdash; Run concurrent tasks without impacting your PyQt6 UIpythonguis.comSimple PyQt6 and OpenCV VideoCapture viewer using a worker thread · GitHubgithub.comMultithreading PyQt5 applications with QThreadPoolpythonguis.comKontynuując proponowane zmiany w kodzie pliku app/utils/file_tools/data_splitter_gui.py:
10. Optymalizacja wczytywania plików (dokończenie)
markdownPlik: app/utils/file_tools/data_splitter_gui.py
Funkcja: update_files_list
Proponowany kod:
```python
def update_files_list(self):
    """Aktualizuje listę wszystkich plików w wybranych katalogach"""
    self.files_list_widget.clear()
    self.files_list = []

    if not self.input_dir:
        return

    # Dodaj element informujący o skanowaniu
    self.files_list_widget.addItem("Trwa skanowanie plików...")
    QApplication.processEvents()

    # Skaner plików w osobnym wątku
    class FileScanner(QThread):
        finished = pyqtSignal(list)
        progress = pyqtSignal(int)

        def __init__(self, root_path, allowed_extensions):
            super().__init__()
            self.root_path = root_path
            self.allowed_extensions = allowed_extensions
            self.files = []

        def run(self):
            for item in self.root_path.rglob("*"):
                if item.is_file() and item.suffix.lower() in self.allowed_extensions:
                    relative_path = item.relative_to(self.root_path)
                    self.files.append((item, str(relative_path)))
                    if len(self.files) % 100 == 0:
                        self.progress.emit(len(self.files))
            self.finished.emit(self.files)

    # Funkcje zwrotne dla skanera
    def on_scan_progress(count):
        self.files_list_widget.item(0).setText(f"Skanowanie w toku... Znaleziono {count} plików")
        QApplication.processEvents()

    def on_scan_complete(files):
        self.files_list_widget.clear()
        self.files_list = [f[0] for f in files]  # Zapisz pełne ścieżki

        # Pokaż tylko pierwsze 1000 plików w widoku
        display_limit = 1000
        for i, (_, file_path_str) in enumerate(files[:display_limit]):
            self.files_list_widget.addItem(file_path_str)

        # Dodaj informację o limicie
        if len(files) > display_limit:
            self.files_list_widget.addItem(f"... oraz {len(files) - display_limit} więcej plików")

        self.log_message(f"Znaleziono łącznie {len(files)} plików.", logging.INFO)

    # Uruchom skaner
    scanner = FileScanner(Path(self.input_dir), ALLOWED_IMAGE_EXTENSIONS)
    scanner.progress.connect(on_scan_progress)
    scanner.finished.connect(on_scan_complete)
    scanner.start()
11. Implementacja wzorca QThreadPool dla operacji skanowania folderów
Zgodnie z nowszymi zaleceniami dla PyQt6, lepiej jest używać QThreadPool do wielu zadań w tle zamiast bezpośrednio QThread. QThreadPool zarządza kolejką zadań i automatycznie przydziela dostępne wątki. Multithreading PyQt6 applications with QThreadPool
markdownPlik: app/utils/file_tools/data_splitter_gui.py
Nowa implementacja skanera folderów:
```python
from PyQt6.QtCore import QRunnable, QThreadPool, pyqtSlot, QObject, pyqtSignal

# Sygnały dla skanera folderów
class ScannerSignals(QObject):
    """Sygnały emitowane przez skaner katalogów"""
    finished = pyqtSignal(dict)  # Przekazuje słownik z danymi skanowania
    progress = pyqtSignal(int, str)  # Postęp (procent, wiadomość)
    error = pyqtSignal(str)  # Komunikat błędu

class FolderScanner(QRunnable):
    """Zadanie skanowania katalogów w tle używające QRunnable"""

    def __init__(self, root_path, allowed_extensions):
        super().__init__()
        self.root_path = Path(root_path)
        self.allowed_extensions = allowed_extensions
        self.signals = ScannerSignals()

    @pyqtSlot()
    def run(self):
        """Wykonuje skanowanie katalogów"""
        try:
            result = {
                'folder_counts': {},
                'min_files_count': float('inf'),
                'min_files_folder': None,
                'total_files': 0
            }

            # Określ całkowitą liczbę folderów
            folders = [f for f in self.root_path.iterdir() if f.is_dir()]
            total_folders = len(folders)

            for i, category_dir in enumerate(folders):
                # Aktualizuj postęp
                progress_percent = int((i / total_folders) * 100) if total_folders > 0 else 0
                self.signals.progress.emit(
                    progress_percent,
                    f"Skanowanie: {category_dir.name} ({i+1}/{total_folders})"
                )

                # Policz pliki w kategorii
                file_count = sum(
                    1 for f in category_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in self.allowed_extensions
                )

                result['folder_counts'][category_dir.name] = file_count
                result['total_files'] += file_count

                # Aktualizuj minimum, jeśli kategoria ma pliki
                if 0 < file_count < result['min_files_count']:
                    result['min_files_count'] = file_count
                    result['min_files_folder'] = category_dir.name

            # Zakończ, jeśli nie znaleziono minimum
            if result['min_files_count'] == float('inf'):
                result['min_files_count'] = 0
                result['min_files_folder'] = None

            self.signals.finished.emit(result)

        except Exception as e:
            self.signals.error.emit(str(e))
markdownPlik: app/utils/file_tools/data_splitter_gui.py
Modyfikacja DataSplitterApp.__init__:
```python
def __init__(self):
    super().__init__()
    self.input_dir = ""
    self.output_dir = ""
    self.processing_thread = None
    self.files_list = []

    # Inicjalizacja puli wątków
    self.threadpool = QThreadPool()
    logger.info(f"Używam puli wątków z maksymalnie {self.threadpool.maxThreadCount()} wątkami")

    # Inicjalizacja interfejsu
    icon_path = Path("resources/img/icon.png")
    if icon_path.exists():
        self.setWindowIcon(QIcon(str(icon_path)))
    else:
        logger.warning(f"Nie znaleziono pliku ikony: {icon_path}")

    self.initUI()
    self._apply_material_theme()
    self.update_files_limit_and_validation_based_on_selection()
markdownPlik: app/utils/file_tools/data_splitter_gui.py
Modyfikacja DataSplitterApp.update_folder_tree:
```python
def update_folder_tree(self):
    """Aktualizuje drzewo folderów z wykorzystaniem QThreadPool"""
    self.folder_tree.clear()
    if not self.input_dir:
        self.update_files_limit_and_validation_based_on_selection()
        return

    root_path = Path(self.input_dir)
    tree_root_item = QTreeWidgetItem(self.folder_tree, [root_path.name])
    self.folder_tree.addTopLevelItem(tree_root_item)
    tree_root_item.setExpanded(True)

    # Odłącz sygnał zmiany elementu
    try:
        self.folder_tree.itemChanged.disconnect(self.on_folder_tree_item_changed)
    except TypeError:
        pass

    # Uruchom skaner folderów w tle
    scanner = FolderScanner(root_path, ALLOWED_IMAGE_EXTENSIONS)

    # Funkcje obsługi sygnałów
    def on_scan_progress(percent, message):
        self.log_message(message)
        self.progress_bar.setValue(percent)
        QApplication.processEvents()

    def on_scan_finished(result):
        folder_counts = result['folder_counts']
        min_files_folder = result['min_files_folder']

        # Dodaj foldery do drzewa
        for category_dir in root_path.iterdir():
            if category_dir.is_dir():
                file_count = folder_counts.get(category_dir.name, 0)
                display_text = f"{category_dir.name} ({file_count} plików)"
                item = QTreeWidgetItem(tree_root_item, [display_text])
                item.setData(0, Qt.ItemDataRole.UserRole, category_dir.name)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(0, Qt.CheckState.Checked)

                # Wyróżnij folder z najmniejszą liczbą plików
                if category_dir.name == min_files_folder and file_count > 0:
                    item.setForeground(0, QColor(HIGHLIGHT_COLOR))

                QApplication.processEvents()  # Zapobiega blokowaniu UI

        # Podłącz ponownie sygnał i aktualizuj kontrolki
        self.folder_tree.itemChanged.connect(self.on_folder_tree_item_changed)
        self.update_files_limit_and_validation_based_on_selection()
        self.progress_bar.setValue(0)  # Zresetuj pasek postępu
        self.log_message(f"Znaleziono {result['total_files']} plików w {len(folder_counts)} kategoriach.")

    def on_scan_error(error_message):
        self.log_message(f"Błąd podczas skanowania: {error_message}", logging.ERROR)
        self.progress_bar.setValue(0)
        # Podłącz ponownie sygnał
        self.folder_tree.itemChanged.connect(self.on_folder_tree_item_changed)

    # Podłącz sygnały i rozpocznij skanowanie
    scanner.signals.progress.connect(on_scan_progress)
    scanner.signals.finished.connect(on_scan_finished)
    scanner.signals.error.connect(on_scan_error)

    self.threadpool.start(scanner)
12. Implementacja przetwarzania plików z użyciem QThreadPool
Zamiast używać klasy Worker opartej na QThread, zaimplementujmy rozwiązanie oparte na QRunnable, które lepiej pasuje do modelu wielowątkowości w PyQt6. Multithreading PyQt6 applications with QThreadPool
markdownPlik: app/utils/file_tools/data_splitter_gui.py
Nowa implementacja procesora plików:
```python
class ProcessorSignals(QObject):
    """Sygnały emitowane przez procesor plików"""
    progress_updated = pyqtSignal(int, str)
    finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

class FileProcessor(QRunnable):
    """Zadanie przetwarzania plików w tle używające QRunnable"""

    def __init__(
        self,
        input_dir,
        output_dir,
        split_mode,
        split_value,
        use_validation=True,
        selected_categories=None,
    ):
        super().__init__()
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.split_mode = split_mode
        self.split_value = split_value
        self.use_validation = use_validation
        self.selected_categories = selected_categories if selected_categories else []
        self.signals = ProcessorSignals()
        self.is_cancelled = False
        self.stats = {"train": {}, "valid": {}}
        self.json_report = {}
        self.min_files_in_selection_for_report = 0
        self.folders_with_min_for_report = []

    def get_min_files_in_selected_categories_for_processing(self):
        """Znajduje minimalną liczbę plików w wybranych kategoriach"""
        if not self.input_dir or not self.selected_categories:
            return 0, []

        min_files_val = float("inf")
        folders_with_min = []
        root_path = Path(self.input_dir)

        for category_name in self.selected_categories:
            category_dir = root_path / category_name
            if category_dir.is_dir():
                count = sum(
                    1
                    for f in category_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
                )
                if 0 < count < min_files_val:
                    min_files_val = count
                    folders_with_min = [category_name]
                elif count == min_files_val and min_files_val != float("inf"):
                    folders_with_min.append(category_name)

        if min_files_val == float("inf"):
            return 0, []
        return min_files_val, folders_with_min

    @pyqtSlot()
    def run(self):
        """Wykonuje przetwarzanie plików w osobnym wątku"""
        try:
            self.signals.progress_updated.emit(0, "Rozpoczynanie przetwarzania...")

            # Tutaj cała logika z metody run() starej klasy Worker
            # Kod jest taki sam, ale używa self.signals zamiast bezpośrednich sygnałów
            # oraz sprawdza self.is_cancelled zamiast is_cancelled

            # Przykładowo:
            if not self.input_dir.is_dir():
                raise ConfigurationError(f"Folder wejściowy nie istnieje: {self.input_dir}")

            # ...reszta kodu przetwarzania...

            if self.is_cancelled:
                self.signals.finished.emit("Anulowano.")
            else:
                self.signals.progress_updated.emit(100, "Zakończono kopiowanie plików.")
                report = self._generate_report()
                json_path = self.output_dir / "raport_kopiowania.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(self.json_report, f, ensure_ascii=False, indent=2)
                self.signals.finished.emit(
                    f"Przetwarzanie zakończone pomyślnie!\n\n{report}\n\nZapisano raport JSON: {json_path}"
                )

        except ConfigurationError as ce:
            self.signals.error_occurred.emit(f"Błąd konfiguracji: {ce}")
            self.signals.finished.emit(f"Błąd: {ce}")
        except Exception as e:
            self.signals.error_occurred.emit(f"Niespodziewany błąd: {e}")
            self.signals.finished.emit(f"Niespodziewany błąd: {e}")

    def cancel(self):
        """Anuluje przetwarzanie"""
        self.is_cancelled = True
        self.signals.progress_updated.emit(0, "Anulowanie...")

    def _generate_report(self):
        """Generuje raport z przetwarzania (ten sam kod co w starej klasie Worker)"""
        # Bez zmian
markdownPlik: app/utils/file_tools/data_splitter_gui.py
Modyfikacja metody start_processing:
```python
def start_processing(self):
    """Rozpoczyna przetwarzanie plików z użyciem QThreadPool"""
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

    # Sprawdzenie ścieżek
    input_path = Path(self.input_dir)
    output_path = Path(self.output_dir)
    try:
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
                return
    except ValueError:
        pass

    # Zablokuj interfejs
    self.start_button.setEnabled(False)
    self.cancel_button.setEnabled(True)
    self.mode_combo.setEnabled(False)
    self.split_slider.setEnabled(False)
    self.files_spin.setEnabled(False)
    self.validation_check.setEnabled(False)
    self.progress_bar.setValue(0)
    self.log_edit.clear()

    split_mode = "percent" if self.mode_combo.currentIndex() == 0 else "files"
    split_value = (
        self.split_slider.value()
        if split_mode == "percent"
        else self.files_spin.value()
    )
    use_validation = self.validation_check.isChecked()

    self.log_message("=" * 30)
    self.log_message(
        f"Rozpoczynanie (tryb: {split_mode}, wartość: {split_value}, walidacja: {use_validation})"
    )
    self.log_message(f"Wybrane kategorie: {', '.join(selected_categories)}")

    # Utwórz procesor plików
    self.file_processor = FileProcessor(
        self.input_dir,
        self.output_dir,
        split_mode,
        split_value,
        use_validation,
        selected_categories,
    )

    # Podłącz sygnały
    self.file_processor.signals.progress_updated.connect(self.update_progress)
    self.file_processor.signals.finished.connect(self.processing_finished)
    self.file_processor.signals.error_occurred.connect(self.processing_error)

    # Przechowaj procesor jako pole klasy, aby mieć możliwość anulowania
    self.processing_thread = self.file_processor

    # Uruchom w puli wątków
    self.threadpool.start(self.file_processor)
markdownPlik: app/utils/file_tools/data_splitter_gui.py
Modyfikacja metody cancel_processing:
```python
def cancel_processing(self):
    """Anuluje przetwarzanie plików"""
    if self.processing_thread:
        self.log_message("Wysyłanie sygnału anulowania...")
        self.processing_thread.cancel()
        self.cancel_button.setEnabled(False)
13. Podsumowanie proponowanych zmian
Przeprowadzone zmiany w kodzie koncentrują się na następujących ulepszeniach:

Optymalizacja wielowątkowości - zamiana QThread na bardziej wydajny QThreadPool i QRunnable
Modularyzacja kodu - wydzielenie logiki do osobnych klas i metod
Poprawa odpowiedzialności pojedynczych klas (Single Responsibility Principle)
Dodanie profesjonalnego systemu logowania
Wprowadzenie konfiguracji zamiast sztywno zakodowanych stałych
Udoskonalenie obsługi błędów z niestandardowymi wyjątkami
Optymalizacja interfejsu użytkownika dla lepszej responsywności
Przygotowanie do dalszej rozbudowy przez wprowadzenie modelu danych

Wprowadzone zmiany znacznie poprawiają zarówno wydajność, jak i jakość kodu, przygotowując go na przyszłe rozszerzenia. Zastosowanie wzorców projektowych takich jak Signals/Slots czy QThreadPool zgodnie z najnowszymi zaleceniami dla PyQt6 sprawia, że aplikacja będzie działać płynniej nawet przy dużej liczbie plików Multithreading PyQt6 applications with QThreadPool.
````
