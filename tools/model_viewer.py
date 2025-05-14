import collections
import json
import logging
import os
import sys

import matplotlib

matplotlib.use("qtagg")  # Standardowy backend Qt
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
matplotlib.rcParams["font.serif"] = ["DejaVu Sans"]
matplotlib.rcParams["font.monospace"] = ["DejaVu Sans"]
matplotlib.rcParams["font.cursive"] = ["DejaVu Sans"]
matplotlib.rcParams["font.fantasy"] = ["DejaVu Sans"]
matplotlib.rcParams["mathtext.fontset"] = "dejavusans"
logging.getLogger("matplotlib").setLevel(logging.WARNING)
import numpy as np
import torch
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

# Konfiguracja loggera
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class ModelAnalyzerThread(QThread):
    analysis_complete = pyqtSignal(str)
    analysis_error = pyqtSignal(str)
    progress_update = pyqtSignal(str)

    def __init__(self, model):
        super().__init__()
        self.model = model

    def run(self):
        try:
            total_params = 0
            layers_info = []
            model_type = type(self.model).__name__

            self.progress_update.emit(f"Rozpoczęcie analizy modelu typu: {model_type}")

            def analyze_module(module, name=""):
                nonlocal total_params
                params = sum(p.numel() for p in module.parameters())
                total_params += params

                if params > 0:
                    layer_info = {
                        "name": name or module.__class__.__name__,
                        "parameters": params,
                        "type": module.__class__.__name__,
                    }
                    layers_info.append(layer_info)
                    self.progress_update.emit(
                        f"Znaleziono warstwę: {layer_info['name']} "
                        f"({layer_info['type']}) z {params:,} parametrami"
                    )

                for child_name, child in module.named_children():
                    analyze_module(
                        child, f"{name}.{child_name}" if name else child_name
                    )

            def analyze_dict(d, prefix=""):
                nonlocal total_params
                layer_params = {}

                for key, value in d.items():
                    current_path = f"{prefix}.{key}" if prefix else key
                    self.progress_update.emit(
                        f"Analizuję klucz: {current_path}, typ wartości: {type(value)}"
                    )

                    if isinstance(value, torch.Tensor):
                        self.progress_update.emit(
                            f"Tensor {current_path}: shape={value.shape}, "
                            f"dtype={value.dtype}, numel={value.numel()}"
                        )
                        layer_name = current_path.split(".")[0]
                        if layer_name not in layer_params:
                            layer_params[layer_name] = {"parameters": 0, "tensors": []}
                        layer_params[layer_name]["parameters"] += value.numel()
                        layer_params[layer_name]["tensors"].append(current_path)
                        total_params += value.numel()
                    elif isinstance(value, (dict, collections.OrderedDict)):
                        self.progress_update.emit(
                            f"Znaleziono zagnieżdżony słownik: {current_path}"
                        )
                        analyze_dict(value, current_path)
                    else:
                        self.progress_update.emit(
                            f"Pominięto nie-tensor: {current_path} ({type(value)})"
                        )

                return layer_params

            if isinstance(self.model, torch.nn.Module):
                self.progress_update.emit("Analizuję model typu torch.nn.Module")
                analyze_module(self.model)
            elif isinstance(self.model, dict):
                self.progress_update.emit("Analizuję model typu dict (state_dict)")
                self.progress_update.emit(
                    f"Klucze w state_dict: {list(self.model.keys())}"
                )

                layer_params = analyze_dict(self.model)

                self.progress_update.emit(
                    f"Znalezione warstwy: {list(layer_params.keys())}"
                )

                for layer_name, info in layer_params.items():
                    layers_info.append(
                        {
                            "name": layer_name,
                            "parameters": info["parameters"],
                            "type": "state_dict_layer",
                            "tensors": info["tensors"],
                        }
                    )
                    self.progress_update.emit(
                        f"Znaleziono warstwę state_dict: {layer_name} "
                        f"z {info['parameters']:,} parametrami"
                    )
            else:
                raise ValueError(f"Nieobsługiwany typ modelu: {model_type}")

            # Przygotuj raport
            report = "Analiza modelu:\n\n"
            report += f"Typ modelu: {model_type}\n"
            report += f"Całkowita liczba parametrów: {total_params:,}\n"
            report += f"Liczba warstw: {len(layers_info)}\n\n"
            report += "Szczegóły warstw:\n"

            for layer in layers_info:
                report += (
                    f"- {layer['name']}: "
                    f"{layer['parameters']:,} parametrów "
                    f"({layer['type']})\n"
                )
                if "tensors" in layer:
                    for tensor in layer["tensors"]:
                        report += f"  * {tensor}\n"

            self.analysis_complete.emit(report)

        except Exception as e:
            self.analysis_error.emit(str(e))


class ModelViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        logger.info("Uruchomiono Przeglądarkę Modeli PyTorch (UI)")
        self.setWindowTitle("Przeglądarka Modeli PyTorch")
        self.setGeometry(100, 100, 1200, 800)

        # Kolory zgodne z Material Design i VS Code
        self.primary_color = "#007ACC"  # Niebieski VS Code
        self.success_color = "#10B981"  # Zielony
        self.warning_color = "#DC2626"  # Czerwony
        self.background = "#1E1E1E"  # Ciemne tło
        self.surface = "#252526"  # Lekko jaśniejsze tło dla paneli
        self.border_color = "#3F3F46"  # Kolor obramowania
        self.text_color = "#CCCCCC"  # Kolor tekstu

        # Ścieżka do folderu z modelami
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(base_dir, "..", "data", "models")
        self.models_dir = os.path.abspath(self.models_dir)
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

        # Główny widget i layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Pasek narzędzi
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # Przyciski w pasku narzędzi
        save_button = QPushButton("Zapisz")
        save_button.clicked.connect(self.save_model)
        save_button.setEnabled(False)
        self.save_button = save_button

        export_structure_button = QPushButton("Eksportuj strukturę")
        export_structure_button.clicked.connect(self.export_structure)
        export_structure_button.setEnabled(False)
        self.export_structure_button = export_structure_button

        analyze_button = QPushButton("Analizuj")
        analyze_button.clicked.connect(self.analyze_model)
        analyze_button.setEnabled(False)
        self.analyze_button = analyze_button

        toolbar.addWidget(save_button)
        toolbar.addWidget(export_structure_button)
        toolbar.addWidget(analyze_button)
        toolbar.addSeparator()

        export_onnx_button = QPushButton("ONNX")
        export_onnx_button.clicked.connect(self.export_to_onnx)
        export_onnx_button.setEnabled(False)
        self.export_onnx_button = export_onnx_button

        export_torchscript_button = QPushButton("TorchScript")
        export_torchscript_button.clicked.connect(self.export_to_torchscript)
        export_torchscript_button.setEnabled(False)
        self.export_torchscript_button = export_torchscript_button

        toolbar.addWidget(export_onnx_button)
        toolbar.addWidget(export_torchscript_button)
        toolbar.addSeparator()

        visualize_params_button = QPushButton("Wizualizuj")
        visualize_params_button.clicked.connect(self.visualize_parameters)
        visualize_params_button.setEnabled(False)
        self.visualize_params_button = visualize_params_button

        compare_models_button = QPushButton("Porównaj")
        compare_models_button.clicked.connect(self.compare_models)
        compare_models_button.setEnabled(False)
        self.compare_models_button = compare_models_button

        toolbar.addWidget(visualize_params_button)
        toolbar.addWidget(compare_models_button)

        # Splitter dla listy modeli, drzewa i szczegółów
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(8)  # Grubszy uchwyt do przeciągania
        splitter.setChildrenCollapsible(False)  # Zapobiega całkowitemu zwinięciu sekcji

        # Panel po lewej: lista modeli + drzewo
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Splitter dla lewego panelu (lista modeli + drzewo)
        left_splitter = QSplitter(Qt.Orientation.Vertical)
        left_splitter.setHandleWidth(8)
        left_splitter.setChildrenCollapsible(False)

        # Lista modeli ze scrollbarem
        model_list_scroll = QScrollArea()
        model_list_scroll.setWidgetResizable(True)
        model_list_widget = QWidget()
        model_list_layout = QVBoxLayout(model_list_widget)
        label_models = QLabel("Modele:")
        label_models.setStyleSheet(f"color: {self.text_color}; font-weight: bold;")
        self.model_list = QListWidget()
        self.model_list.itemClicked.connect(self.on_model_selected)
        model_list_layout.addWidget(label_models)
        model_list_layout.addWidget(self.model_list)
        model_list_scroll.setWidget(model_list_widget)
        left_splitter.addWidget(model_list_scroll)

        # Drzewo modelu ze scrollbarem
        tree_scroll = QScrollArea()
        tree_scroll.setWidgetResizable(True)
        tree_widget = QWidget()
        tree_layout = QVBoxLayout(tree_widget)

        # Nagłówek i przyciski dla drzewa
        tree_header = QWidget()
        tree_header_layout = QHBoxLayout(tree_header)
        tree_header_layout.setContentsMargins(0, 0, 0, 0)

        tree_label = QLabel("Struktura Modelu")
        tree_label.setStyleSheet(f"color: {self.text_color}; font-weight: bold;")
        tree_header_layout.addWidget(tree_label)

        # Przycisk zwijania/rozwijania
        toggle_button = QPushButton("Zwiń/Rozwiń")
        toggle_button.clicked.connect(self.toggle_tree_expansion)
        tree_header_layout.addWidget(toggle_button)

        tree_header_layout.addStretch()
        tree_layout.addWidget(tree_header)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)  # Ukrywamy domyślny nagłówek
        self.tree.itemClicked.connect(self.show_details)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.show_context_menu)
        tree_layout.addWidget(self.tree)
        tree_scroll.setWidget(tree_widget)
        left_splitter.addWidget(tree_scroll)

        # Ustaw proporcje dla lewego splittera (30% lista, 70% drzewo)
        left_splitter.setSizes([100, 300])

        left_layout.addWidget(left_splitter)
        splitter.addWidget(left_panel)

        # Panel szczegółów ze scrollbarem
        details_scroll = QScrollArea()
        details_scroll.setWidgetResizable(True)
        details_widget = QWidget()
        details_layout = QVBoxLayout(details_widget)
        self.details_label = QLabel("Wybierz element z drzewa aby zobaczyć szczegóły")
        self.details_label.setWordWrap(True)
        details_layout.addWidget(self.details_label)
        details_scroll.setWidget(details_widget)
        splitter.addWidget(details_scroll)

        # Ustaw proporcje dla głównego splittera (40% lewy panel, 60% szczegóły)
        splitter.setSizes([400, 600])

        layout.addWidget(splitter)

        self.model = None
        self.current_model_path = None
        self.comparison_model = None
        self.analyzer_thread = None

        # Zastosuj style
        self._apply_styles()

        self.load_models_from_folder()

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
            QListWidget {{
                background-color: {self.surface};
                color: {self.text_color};
                border: 1px solid {self.border_color};
                border-radius: 2px;
            }}
            QListWidget::item {{
                padding: 2px;
            }}
            QListWidget::item:selected {{
                background-color: #264F78;
                color: white;
            }}
            QLabel {{
                color: {self.text_color};
            }}
            QScrollArea {{
                background-color: {self.surface};
                border: 1px solid {self.border_color};
                border-radius: 2px;
            }}
            QToolBar {{
                background-color: {self.surface};
                border-bottom: 1px solid {self.border_color};
                spacing: 4px;
                padding: 4px;
            }}
            QSplitter::handle {{
                background-color: {self.border_color};
            }}
            """
        )

    def load_models_from_folder(self):
        logger.info("Ładowanie listy modeli z folderu: %s", self.models_dir)
        self.model_list.clear()
        model_files = [
            f for f in os.listdir(self.models_dir) if f.endswith((".pt", ".pth"))
        ]
        for fname in model_files:
            self.model_list.addItem(fname)
        if model_files:
            logger.info("Znaleziono modele: %s", ", ".join(model_files))
            # Automatycznie wybierz i załaduj pierwszy model
            self.model_list.setCurrentRow(0)
            self.load_model_from_file(os.path.join(self.models_dir, model_files[0]))
        else:
            logger.info("Nie znaleziono żadnych modeli w folderze.")

    def on_model_selected(self, item):
        logger.info("Użytkownik wybrał model: %s", item.text())
        model_path = os.path.join(self.models_dir, item.text())
        self.load_model_from_file(model_path)

    def load_model_from_file(self, file_path):
        logger.info("Wczytywanie modelu z pliku: %s", file_path)
        try:
            self.model = torch.load(file_path, map_location=torch.device("cpu"))
            self.current_model_path = file_path
            self.populate_tree()
            # Włącz wszystkie przyciski po załadowaniu modelu
            self.save_button.setEnabled(True)
            self.export_structure_button.setEnabled(True)
            self.analyze_button.setEnabled(True)
            self.export_onnx_button.setEnabled(True)
            self.export_torchscript_button.setEnabled(True)
            self.visualize_params_button.setEnabled(True)
            self.compare_models_button.setEnabled(True)
            logger.info("Model został poprawnie wczytany: %s", file_path)
        except Exception as e:
            logger.error("Błąd podczas wczytywania modelu: %s", str(e))
            self.details_label.setText(f"Błąd podczas wczytywania modelu: {str(e)}")

    def show_context_menu(self, position):
        logger.info("Wyświetlenie menu kontekstowego dla drzewa modelu")
        item = self.tree.itemAt(position)
        if not item:
            logger.warning("Brak wybranego elementu w drzewie")
            return

        menu = QMenu()
        edit_action = menu.addAction("Edytuj parametr")
        edit_action.triggered.connect(lambda: self.edit_parameter(item))
        menu.exec(self.tree.mapToGlobal(position))

    def edit_parameter(self, item):
        logger.info("Edycja parametru przez użytkownika")
        path = []
        current = item
        while current is not None:
            path.insert(0, current.text(0))
            current = current.parent()

        # Znajdź parametr w modelu
        param = self._get_parameter_by_path(path)
        if not param or not isinstance(param, torch.Tensor):
            logger.warning(
                "Nie można edytować wybranego elementu: %s", " -> ".join(path)
            )
            self.show_message(
                "Błąd", "Nie można edytować tego elementu", QMessageBox.Icon.Warning
            )
            return

        # Pobierz nową wartość
        value, ok = QInputDialog.getDouble(
            self,
            "Edytuj parametr",
            f"Wprowadź nową wartość dla {path[-1]}:",
            float(param.float().mean().item()),
            -1e10,
            1e10,
            6,
        )

        if ok:
            try:
                # Zaktualizuj wartość parametru
                param.fill_(value)
                self.populate_tree()  # Odśwież widok
                logger.info(
                    "Parametr %s został zaktualizowany na wartość: %s", path[-1], value
                )
                self.show_message("Sukces", "Parametr został zaktualizowany")
            except Exception as e:
                logger.error("Nie udało się zaktualizować parametru: %s", str(e))
                self.show_message(
                    "Błąd",
                    f"Nie udało się zaktualizować parametru: {str(e)}",
                    QMessageBox.Icon.Critical,
                )

    def save_model(self):
        logger.info("Zapis modelu do pliku")
        if not self.model:
            logger.warning("Brak modelu do zapisania")
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Zapisz model",
            self.current_model_path or "",
            "PyTorch Files (*.pt *.pth);;All Files (*)",
        )

        if file_name:
            try:
                torch.save(self.model, file_name)
                self.current_model_path = file_name
                self.show_message("Sukces", "Model został zapisany pomyślnie")
            except Exception as e:
                logger.error(f"Błąd podczas zapisywania modelu: {str(e)}")
                self.show_message(
                    "Błąd",
                    f"Nie udało się zapisać modelu: {str(e)}",
                    QMessageBox.Icon.Critical,
                )

    def export_structure(self):
        if not self.model:
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Eksportuj strukturę",
            "",
            "JSON Files (*.json);;Text Files (*.txt);;All Files (*)",
        )

        if file_name:
            try:
                structure = self._get_model_structure()
                with open(file_name, "w", encoding="utf-8") as f:
                    json.dump(structure, f, indent=2)
                self.show_message("Sukces", "Struktura modelu została wyeksportowana")
            except Exception as e:
                logger.error(f"Błąd podczas eksportowania struktury: {str(e)}")
                self.show_message(
                    "Błąd",
                    f"Nie udało się wyeksportować struktury: {str(e)}",
                    QMessageBox.Icon.Critical,
                )

    def analyze_model(self):
        if not self.model:
            logger.warning("Próba analizy pustego modelu")
            self.details_label.setText("Błąd: Brak modelu do analizy")
            return

        # Wyłącz przycisk analizy podczas analizy
        self.analyze_button.setEnabled(False)
        self.details_label.setText("Analizuję model...")

        # Utwórz i uruchom wątek analizy
        self.analyzer_thread = ModelAnalyzerThread(self.model)
        self.analyzer_thread.analysis_complete.connect(self._on_analysis_complete)
        self.analyzer_thread.analysis_error.connect(self._on_analysis_error)
        self.analyzer_thread.progress_update.connect(self._on_progress_update)
        self.analyzer_thread.start()

    def _on_analysis_complete(self, report):
        self.details_label.setText(report)
        self.analyze_button.setEnabled(True)
        logger.info("Analiza modelu zakończona")

    def _on_analysis_error(self, error_msg):
        self.details_label.setText(f"Błąd podczas analizy: {error_msg}")
        self.analyze_button.setEnabled(True)
        logger.error(f"Błąd podczas analizy modelu: {error_msg}")

    def _on_progress_update(self, message):
        logger.debug(message)
        # Możemy też aktualizować status w interfejsie
        self.statusBar().showMessage(message)

    def _get_model_structure(self):
        structure = {}

        def add_to_structure(obj, path):
            if isinstance(obj, torch.nn.Module):
                structure[path] = {
                    "type": obj.__class__.__name__,
                    "parameters": sum(p.numel() for p in obj.parameters()),
                    "children": {},
                }
                for name, child in obj.named_children():
                    add_to_structure(child, f"{path}.{name}")
            elif isinstance(obj, dict):
                structure[path] = {"type": "dict", "items": len(obj), "children": {}}
                for key, value in obj.items():
                    add_to_structure(value, f"{path}.{key}")
            elif isinstance(obj, torch.Tensor):
                structure[path] = {
                    "type": "tensor",
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                }
            else:
                structure[path] = {"type": type(obj).__name__, "value": str(obj)}

        if isinstance(self.model, torch.nn.Module):
            add_to_structure(self.model, "model")
        else:
            add_to_structure(self.model, "state_dict")

        return structure

    def _get_parameter_by_path(self, path):
        if not path:
            return None

        current = self.model
        for name in path[1:]:  # Pomijamy "Model" lub "State Dict"
            if isinstance(current, torch.nn.Module):
                current = getattr(current, name)
            elif isinstance(current, dict):
                current = current.get(name)
            else:
                return None

        return current

    def populate_tree(self):
        self.tree.clear()
        if isinstance(self.model, torch.nn.Module):
            root = QTreeWidgetItem(self.tree, ["Model"])
            self._add_module_to_tree(root, self.model)
        else:
            root = QTreeWidgetItem(self.tree, ["State Dict"])
            self._add_state_dict_to_tree(root, self.model)
        self.tree.expandAll()

    def _add_module_to_tree(self, parent_item, module):
        for name, child in module.named_children():
            # Dodaj typ warstwy i liczbę parametrów
            layer_type = child.__class__.__name__
            param_count = sum(p.numel() for p in child.parameters())
            item = QTreeWidgetItem(
                parent_item, [f"{name} ({layer_type}, " f"parametry: {param_count})"]
            )
            if isinstance(child, torch.nn.Module):
                self._add_module_to_tree(item, child)
            self._add_parameters_to_tree(item, child)

    def _add_state_dict_to_tree(self, parent_item, state_dict):
        for key, value in state_dict.items():
            if isinstance(value, dict):
                item = QTreeWidgetItem(parent_item, [key])
                self._add_state_dict_to_tree(item, value)
            else:
                self._add_parameters_to_tree(parent_item, {key: value})

    def _add_parameters_to_tree(self, parent_item, module):
        # Obsługa zarówno torch.nn.Module, jak i dict
        if isinstance(module, dict):
            items = module.items()
        else:
            items = module.named_parameters()
        for name, param in items:
            if isinstance(param, torch.Tensor):
                shape = list(param.shape)
                dtype = str(param.dtype)
                mean = float(param.float().mean().item()) if param.numel() > 0 else 0.0
                QTreeWidgetItem(
                    parent_item,
                    [f"{name}: shape={shape}, dtype={dtype}, mean={mean:.4f}"],
                )

    def show_details(self, item, column):
        path = []
        current = item
        while current is not None:
            path.insert(0, current.text(0))
            current = current.parent()

        details = self._get_details(path)
        self.details_label.setText(details)

    def _get_details(self, path):
        if not path:
            return "Brak szczegółów"

        current = self.model
        for name in path[1:]:  # Pomijamy "Model" lub "State Dict"
            # Wyciągnij czystą nazwę (bez typu i liczby parametrów)
            name_clean = name.split(":")[0].split("(")[0].strip()
            if isinstance(current, torch.nn.Module):
                if hasattr(current, name_clean):
                    current = getattr(current, name_clean)
                else:
                    # Może to być parametr
                    try:
                        current = dict(current.named_parameters())[name_clean]
                    except Exception:
                        return f"Nie można znaleźć: {' -> '.join(path)}"
            elif isinstance(current, dict):
                current = current.get(name_clean)
            else:
                return f"Nie można znaleźć: {' -> '.join(path)}"

        if isinstance(current, torch.nn.Module):
            param_count = sum(p.numel() for p in current.parameters())
            return f"Moduł: {current.__class__.__name__}\n" f"Parametry: {param_count}"
        elif isinstance(current, torch.Tensor):
            shape = list(current.shape)
            dtype = str(current.dtype)
            mean = float(current.float().mean().item()) if current.numel() > 0 else 0.0
            min_val = (
                float(current.float().min().item()) if current.numel() > 0 else 0.0
            )
            max_val = (
                float(current.float().max().item()) if current.numel() > 0 else 0.0
            )
            return (
                f"Tensor:\n"
                f"  shape: {shape}\n"
                f"  dtype: {dtype}\n"
                f"  mean: {mean:.4f}\n"
                f"  min: {min_val:.4f}\n"
                f"  max: {max_val:.4f}"
            )
        else:
            return str(current)

    def export_to_onnx(self):
        if not isinstance(self.model, torch.nn.Module):
            logger.warning("Próba eksportu do ONNX nieprawidłowego typu modelu")
            self.show_message(
                "Błąd",
                "Eksport do ONNX wymaga modelu typu torch.nn.Module",
                QMessageBox.Icon.Warning,
            )
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self, "Eksportuj do ONNX", "", "ONNX Files (*.onnx);;All Files (*)"
        )

        if file_name:
            try:
                # Przykładowy input dla modelu
                dummy_input = torch.randn(1, 3, 224, 224)
                torch.onnx.export(
                    self.model,
                    dummy_input,
                    file_name,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={
                        "input": {0: "batch_size"},
                        "output": {0: "batch_size"},
                    },
                )
                logger.info(f"Model został wyeksportowany do ONNX: {file_name}")
                self.show_message(
                    "Sukces", "Model został wyeksportowany do formatu ONNX"
                )
            except Exception as e:
                logger.error(f"Błąd podczas eksportu do ONNX: {str(e)}")
                self.show_message(
                    "Błąd",
                    f"Nie udało się wyeksportować modelu: {str(e)}",
                    QMessageBox.Icon.Critical,
                )

    def export_to_torchscript(self):
        if not isinstance(self.model, torch.nn.Module):
            logger.warning("Próba eksportu do TorchScript nieprawidłowego typu modelu")
            self.show_message(
                "Błąd",
                "Eksport do TorchScript wymaga modelu typu torch.nn.Module",
                QMessageBox.Icon.Warning,
            )
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Eksportuj do TorchScript",
            "",
            "TorchScript Files (*.pt);;All Files (*)",
        )

        if file_name:
            try:
                # Przykładowy input dla modelu
                dummy_input = torch.randn(1, 3, 224, 224)
                traced_model = torch.jit.trace(self.model, dummy_input)
                traced_model.save(file_name)
                logger.info(f"Model został wyeksportowany do TorchScript: {file_name}")
                self.show_message(
                    "Sukces", "Model został wyeksportowany do formatu TorchScript"
                )
            except Exception as e:
                logger.error(f"Błąd podczas eksportu do TorchScript: {str(e)}")
                self.show_message(
                    "Błąd",
                    f"Nie udało się wyeksportować modelu: {str(e)}",
                    QMessageBox.Icon.Critical,
                )

    def visualize_parameters(self):
        """Wizualizacja parametrów modelu z obsługą outlierów i skalą logarytmiczną."""
        logger.info("Wizualizacja parametrów modelu")
        if not self.model:
            logger.warning("Brak modelu do wizualizacji parametrów")
            self.show_message(
                "Błąd", "Brak modelu do wizualizacji", QMessageBox.Icon.Warning
            )
            return

        try:
            dialog = QDialog(self)
            dialog.setWindowTitle("Wizualizacja parametrów")
            dialog.setGeometry(200, 200, 1200, 800)
            layout = QVBoxLayout(dialog)
            tabs = QTabWidget()

            all_params = []
            layer_params = {}

            # Zbieranie parametrów
            if isinstance(self.model, torch.nn.Module):
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        values = param.detach().cpu().numpy().flatten()
                        all_params.extend(values.tolist())
                        layer_name = name.split(".")[0]
                        if layer_name not in layer_params:
                            layer_params[layer_name] = []
                        layer_params[layer_name].extend(values.tolist())
            elif isinstance(self.model, dict):

                def extract_tensors(d, prefix=""):
                    for key, value in d.items():
                        current_path = f"{prefix}.{key}" if prefix else key
                        if isinstance(value, torch.Tensor):
                            values = value.detach().cpu().numpy().flatten()
                            all_params.extend(values.tolist())
                            layer_name = current_path.split(".")[0]
                            if layer_name not in layer_params:
                                layer_params[layer_name] = []
                            layer_params[layer_name].extend(values.tolist())
                        elif isinstance(value, dict):
                            extract_tensors(value, current_path)

                extract_tensors(self.model)

            if not all_params:
                self.show_message(
                    "Błąd", "Brak parametrów do wizualizacji", QMessageBox.Icon.Warning
                )
                return

            all_params = np.array(all_params)
            # Wyznacz percentyle do ucinania outlierów
            p1 = np.percentile(all_params, 1)
            p99 = np.percentile(all_params, 99)
            mask_normal = (all_params >= p1) & (all_params <= p99)
            params_normal = all_params[mask_normal]
            params_outliers = all_params[~mask_normal]
            outlier_info = f"Liczba outlierów: {len(params_outliers)} ({100*len(params_outliers)/len(all_params):.2f}%)"
            ostrzezenie = None
            if len(params_outliers) > 0:
                ostrzezenie = f"UWAGA: wykryto {len(params_outliers)} outlierów poza zakresem 1-99 percentyla!"

            # Przełącznik skali logarytmicznej
            log_checkbox = QCheckBox("Skala logarytmiczna (histogram)")
            layout.addWidget(log_checkbox)

            def draw_histogram(ax, data, log_scale, title, stats=True):
                ax.clear()
                n, bins, patches = ax.hist(
                    data, bins=50, density=True, alpha=0.7, log=log_scale
                )
                ax.set_title(title, fontsize=14, pad=20)
                ax.set_xlabel("Wartość", fontsize=12)
                ax.set_ylabel("Gęstość", fontsize=12)
                ax.grid(True, alpha=0.3)
                if stats:
                    stats_text = (
                        f"Statystyki:\n"
                        f"Min: {data.min():.4f}\n"
                        f"Max: {data.max():.4f}\n"
                        f"Średnia: {data.mean():.4f}\n"
                        f"Mediana: {np.median(data):.4f}\n"
                        f"Std: {data.std():.4f}"
                    )
                    ax.text(
                        0.02,
                        0.98,
                        stats_text,
                        transform=ax.transAxes,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                    )

            # 1. Histogram (tylko normalne wartości)
            hist_tab = QWidget()
            hist_layout = QVBoxLayout(hist_tab)
            hist_fig = Figure(figsize=(12, 8), dpi=100)
            hist_canvas = FigureCanvas(hist_fig)
            hist_layout.addWidget(hist_canvas)
            ax1 = hist_fig.add_subplot(111)
            draw_histogram(
                ax1, params_normal, False, "Histogram wartości (1-99 percentyl)"
            )
            hist_canvas.draw()
            tabs.addTab(hist_tab, "Histogram wartości")

            # 2. Histogram outlierów
            if len(params_outliers) > 0:
                outlier_tab = QWidget()
                outlier_layout = QVBoxLayout(outlier_tab)
                outlier_fig = Figure(figsize=(12, 8), dpi=100)
                outlier_canvas = FigureCanvas(outlier_fig)
                outlier_layout.addWidget(outlier_canvas)
                ax_out = outlier_fig.add_subplot(111)
                draw_histogram(
                    ax_out, params_outliers, False, "Histogram outlierów (poza 1-99%)"
                )
                outlier_canvas.draw()
                tabs.addTab(outlier_tab, "Outliery")

            # 3. Rozkład wartości (tylko normalne)
            dist_tab = QWidget()
            dist_layout = QVBoxLayout(dist_tab)
            dist_fig = Figure(figsize=(12, 8), dpi=100)
            dist_canvas = FigureCanvas(dist_fig)
            dist_layout.addWidget(dist_canvas)
            ax2 = dist_fig.add_subplot(111)
            sorted_params = np.sort(params_normal)
            ax2.plot(sorted_params, np.linspace(0, 1, len(sorted_params)))
            ax2.set_title("Rozkład wartości parametrów (1-99%)", fontsize=14, pad=20)
            ax2.set_xlabel("Wartość", fontsize=12)
            ax2.set_ylabel("Kwantyl", fontsize=12)
            ax2.grid(True, alpha=0.3)
            dist_canvas.draw()
            tabs.addTab(dist_tab, "Rozkład wartości")

            # 4. Wykres warstw (tylko normalne)
            layers_tab = QWidget()
            layers_layout = QVBoxLayout(layers_tab)
            layers_fig = Figure(figsize=(12, 8), dpi=100)
            layers_canvas = FigureCanvas(layers_fig)
            layers_layout.addWidget(layers_canvas)
            ax3 = layers_fig.add_subplot(111)
            layer_names = list(layer_params.keys())
            layer_means = [
                (
                    np.mean([v for v in layer_params[name] if p1 <= v <= p99])
                    if len([v for v in layer_params[name] if p1 <= v <= p99]) > 0
                    else 0
                )
                for name in layer_names
            ]
            layer_stds = [
                (
                    np.std([v for v in layer_params[name] if p1 <= v <= p99])
                    if len([v for v in layer_params[name] if p1 <= v <= p99]) > 0
                    else 0
                )
                for name in layer_names
            ]
            y_pos = np.arange(len(layer_names))
            ax3.barh(y_pos, layer_means, xerr=layer_stds, align="center", alpha=0.7)
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(layer_names)
            ax3.set_xlabel("Średnia wartość parametrów", fontsize=12)
            ax3.set_title(
                "Średnie wartości parametrów dla każdej warstwy (1-99%)",
                fontsize=14,
                pad=20,
            )
            ax3.grid(True, alpha=0.3)
            layers_canvas.draw()
            tabs.addTab(layers_tab, "Parametry warstw")

            # Ostrzeżenie o outlierach
            if ostrzezenie:
                warn_label = QLabel(ostrzezenie + "\n" + outlier_info)
                warn_label.setStyleSheet("color: orange; font-weight: bold;")
                layout.addWidget(warn_label)

            # Obsługa przełącznika logarytmicznego
            def on_log_checkbox(state):
                draw_histogram(
                    ax1,
                    params_normal,
                    log_checkbox.isChecked(),
                    "Histogram wartości (1-99 percentyl)",
                )
                hist_canvas.draw()
                if len(params_outliers) > 0:
                    draw_histogram(
                        ax_out,
                        params_outliers,
                        log_checkbox.isChecked(),
                        "Histogram outlierów (poza 1-99%)",
                    )
                    outlier_canvas.draw()

            log_checkbox.stateChanged.connect(on_log_checkbox)

            layout.addWidget(tabs)
            logger.info("Wizualizacja parametrów zakończona")
            dialog.exec()

        except Exception as e:
            logger.error(f"Błąd podczas wizualizacji parametrów: {str(e)}")
            self.show_message(
                "Błąd",
                f"Nie udało się wyświetlić wizualizacji: {str(e)}",
                QMessageBox.Icon.Critical,
            )

    def compare_models(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Wybierz model do porównania",
            "",
            "PyTorch Files (*.pt *.pth);;All Files (*)",
        )

        if file_name:
            try:
                self.comparison_model = torch.load(
                    file_name, map_location=torch.device("cpu")
                )
                self.show_comparison_dialog()
            except Exception as e:
                QMessageBox.critical(
                    self, "Błąd", f"Nie udało się wczytać modelu: {str(e)}"
                )

    def show_comparison_dialog(self):
        if not self.model or not self.comparison_model:
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Porównanie modeli")
        dialog.setGeometry(200, 200, 1000, 800)

        layout = QVBoxLayout(dialog)

        # Zakładki dla różnych typów porównania
        tabs = QTabWidget()

        # Porównanie struktury
        structure_tab = QWidget()
        structure_layout = QVBoxLayout(structure_tab)
        structure_text = QLabel()
        structure_text.setWordWrap(True)
        structure_scroll = QScrollArea()
        structure_scroll.setWidget(structure_text)
        structure_layout.addWidget(structure_scroll)
        tabs.addTab(structure_tab, "Struktura")

        # Porównanie parametrów
        params_tab = QWidget()
        params_layout = QVBoxLayout(params_tab)
        params_fig = Figure(figsize=(8, 6))
        params_canvas = FigureCanvas(params_fig)
        params_layout.addWidget(params_canvas)
        tabs.addTab(params_tab, "Parametry")

        layout.addWidget(tabs)

        # Porównaj strukturę
        structure_diff = self._compare_structures()
        structure_text.setText(structure_diff)

        # Porównaj parametry
        self._compare_parameters(params_fig)
        params_canvas.draw()

        dialog.exec()

    def _compare_structures(self):
        if not isinstance(self.model, torch.nn.Module) or not isinstance(
            self.comparison_model, torch.nn.Module
        ):
            return "Porównanie struktury wymaga modeli typu torch.nn.Module"

        diff = []

        def compare_modules(m1, m2, path=""):
            m1_params = set(m1.named_parameters())
            m2_params = set(m2.named_parameters())

            # Znajdź różnice w parametrach
            only_in_m1 = m1_params - m2_params
            only_in_m2 = m2_params - m1_params

            if only_in_m1:
                diff.append(f"Parametry tylko w modelu 1 ({path}):")
                for name, _ in only_in_m1:
                    diff.append(f"  - {name}")

            if only_in_m2:
                diff.append(f"Parametry tylko w modelu 2 ({path}):")
                for name, _ in only_in_m2:
                    diff.append(f"  - {name}")

            # Porównaj podmoduły
            m1_children = set(m1.named_children())
            m2_children = set(m2.named_children())

            only_in_m1 = m1_children - m2_children
            only_in_m2 = m2_children - m1_children

            if only_in_m1:
                diff.append(f"Moduły tylko w modelu 1 ({path}):")
                for name, _ in only_in_m1:
                    diff.append(f"  - {name}")

            if only_in_m2:
                diff.append(f"Moduły tylko w modelu 2 ({path}):")
                for name, _ in only_in_m2:
                    diff.append(f"  - {name}")

            # Rekurencyjnie porównaj wspólne moduły
            common = m1_children & m2_children
            for name, child in common:
                compare_modules(
                    getattr(m1, name),
                    getattr(m2, name),
                    f"{path}.{name}" if path else name,
                )

        compare_modules(self.model, self.comparison_model)
        return "\n".join(diff) if diff else "Modele mają identyczną strukturę"

    def _compare_parameters(self, fig):
        if not isinstance(self.model, torch.nn.Module) or not isinstance(
            self.comparison_model, torch.nn.Module
        ):
            return

        ax = fig.add_subplot(111)

        # Zbierz parametry z obu modeli
        model1_params = []
        model2_params = []

        for p1, p2 in zip(self.model.parameters(), self.comparison_model.parameters()):
            model1_params.extend(p1.detach().cpu().numpy().flatten())
            model2_params.extend(p2.detach().cpu().numpy().flatten())

        # Narysuj wykres porównawczy
        ax.scatter(model1_params, model2_params, alpha=0.5)
        ax.plot(
            [min(model1_params), max(model1_params)],
            [min(model1_params), max(model1_params)],
            "r--",
        )
        ax.set_title("Porównanie wartości parametrów")
        ax.set_xlabel("Model 1")
        ax.set_ylabel("Model 2")

    def toggle_tree_expansion(self):
        """Zwijanie/rozwijanie całej struktury drzewa."""
        if self.tree.topLevelItemCount() > 0:
            root = self.tree.topLevelItem(0)
            if root.isExpanded():
                self.tree.collapseAll()
            else:
                self.tree.expandAll()

    def show_message(self, title, message, icon=QMessageBox.Icon.Information):
        """Wyświetla komunikat i loguje go do konsoli."""
        logger.info(f"UI: {title} - {message}")
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setIcon(icon)
        msg_box.exec()


def main():
    app = QApplication(sys.argv)
    viewer = ModelViewer()
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
