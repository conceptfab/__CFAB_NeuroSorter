import collections
import gc
import json
import logging
import os
import sys

import matplotlib
import numpy as np
import torch
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

# Konfiguracja matplotlib
matplotlib.use("qtagg")  # Standardowy backend Qt
matplotlib.rcParams["font.family"] = "DejaVu Sans"
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# Konfiguracja loggera
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Dodaj po importach, przed klasą ModelAnalyzerThread
LAYER_PATTERNS = collections.OrderedDict(
    [
        ("embedding", ["embedding"]),
        ("attention", ["attention", "attn"]),
        ("batch_norm", ["bn", "batch_norm"]),
        (
            "layer_norm",
            ["layernorm", "layer_norm"],
        ),  # bardziej specyficzne niż ogólne "norm"
        ("normalization", ["norm"]),  # ogólne norm, jeśli inne nie pasują
        ("conv", ["conv", "convolutional"]),
        (
            "linear",
            ["fc", "linear", "dense"],
        ),  # "linear" jest bardziej typowe w PyTorch niż "fc"
        ("pooling", ["pool"]),
        (
            "activation",
            ["activation", "relu", "sigmoid", "tanh", "gelu"],
        ),  # dodać więcej aktywacji
    ]
)


class ModelAnalyzerThread(QThread):
    analysis_complete = pyqtSignal(str)
    analysis_error = pyqtSignal(str)
    progress_update = pyqtSignal(str)

    def __init__(self, model, group_depth=2):  # Dodaj argument group_depth
        super().__init__()
        self.model = model
        self.group_depth = group_depth  # Zapisz wartość
        logger.info("Utworzono wątek analizy modelu")

    def run(self):
        try:
            logger.info("Rozpoczęcie analizy w wątku")
            if not self.model:
                raise ValueError("Model nie został załadowany")

            model_type = type(self.model).__name__
            self.progress_update.emit(f"Rozpoczęcie analizy modelu typu: {model_type}")
            logger.info(f"Analizuję model typu: {model_type}")

            total_params = 0
            layers_info = []

            def analyze_module(module, name=""):
                nonlocal total_params
                params = sum(p.numel() for p in module.parameters())
                total_params += params

                # Zbierz informacje o kształtach parametrów
                param_shapes = {}
                param_dtypes = {}
                param_stats = {}

                for param_name, param in module.named_parameters():
                    if param.requires_grad:
                        param_shapes[param_name] = list(param.shape)
                        param_dtypes[param_name] = str(param.dtype)
                        if param.numel() > 0:
                            param_stats[param_name] = {
                                "mean": float(param.float().mean().item()),
                                "std": float(param.float().std().item()),
                                "min": float(param.float().min().item()),
                                "max": float(param.float().max().item()),
                            }

                layer_info = {
                    "name": name or module.__class__.__name__,
                    "parameters": params,
                    "type": module.__class__.__name__,
                    "has_parameters": params > 0,
                    "param_shapes": param_shapes,
                    "param_dtypes": param_dtypes,
                    "param_stats": param_stats,
                }
                layers_info.append(layer_info)

                if params > 0:
                    msg = (
                        f"Znaleziono warstwę: {layer_info['name']} "
                        f"({layer_info['type']}) z {params:,} parametrami"
                    )
                else:
                    msg = (
                        f"Znaleziono moduł: {layer_info['name']} "
                        f"({layer_info['type']}) bez parametrów"
                    )
                logger.info(msg)
                self.progress_update.emit(msg)

                for child_name, child in module.named_children():
                    analyze_module(
                        child, f"{name}.{child_name}" if name else child_name
                    )

            def analyze_dict(d, prefix="", group_depth=2):
                nonlocal total_params
                layer_params = {}
                current_layer = None

                # Dodaj logowanie struktury modelu
                logger.info(f"Analizuję słownik z prefixem: {prefix}")
                logger.info(f"Klucze w słowniku: {list(d.keys())}")

                for key, value in d.items():
                    current_path = f"{prefix}.{key}" if prefix else key
                    self.progress_update.emit(
                        f"Analizuję klucz: {current_path}, typ wartości: {type(value)}"
                    )

                    # Określ nazwę warstwy na podstawie struktury klucza
                    parts = current_path.split(".")
                    logger.info(f"Ścieżka: {current_path}, części: {parts}")

                    # Grupowanie na podstawie konfigurowalnej głębokości
                    if len(parts) >= group_depth:
                        current_layer = ".".join(parts[:group_depth])
                    else:
                        current_layer = parts[0]
                    logger.info(
                        f"Użyto grupowania do głębokości {group_depth}: {current_layer}"
                    )

                    if isinstance(value, torch.Tensor):
                        msg = (
                            f"Tensor {current_path}: shape={value.shape}, "
                            f"dtype={value.dtype}, numel={value.numel()}"
                        )
                        self.progress_update.emit(msg)
                        logger.info(msg)

                        if current_layer not in layer_params:
                            layer_params[current_layer] = {
                                "parameters": 0,
                                "tensors": [],
                                "shapes": set(),
                                "dtypes": set(),
                                "sub_layers": set(),
                                "layer_type": "unknown",
                                "has_parameters": False,
                                "param_shapes": {},
                                "param_dtypes": {},
                                "param_stats": {},
                            }
                            logger.info(f"Utworzono nową warstwę: {current_layer}")

                        # Użyj LAYER_PATTERNS do określenia typu warstwy
                        layer_params[current_layer]["layer_type"] = "unknown"
                        for type_name, patterns in LAYER_PATTERNS.items():
                            if any(
                                pattern in current_path.lower() for pattern in patterns
                            ):
                                layer_params[current_layer]["layer_type"] = type_name
                                break

                        layer_params[current_layer]["parameters"] += value.numel()
                        layer_params[current_layer]["tensors"].append(current_path)
                        layer_params[current_layer]["shapes"].add(str(value.shape))
                        layer_params[current_layer]["dtypes"].add(str(value.dtype))
                        layer_params[current_layer]["has_parameters"] = True

                        # Dodaj statystyki parametrów
                        if value.numel() > 0:
                            layer_params[current_layer]["param_shapes"][
                                current_path
                            ] = list(value.shape)
                            layer_params[current_layer]["param_dtypes"][
                                current_path
                            ] = str(value.dtype)
                            layer_params[current_layer]["param_stats"][current_path] = {
                                "mean": float(value.float().mean().item()),
                                "std": float(value.float().std().item()),
                                "min": float(value.float().min().item()),
                                "max": float(value.float().max().item()),
                            }

                        # Dodaj informację o podwarstwie (pozostała część ścieżki)
                        if len(parts) > group_depth:
                            sub_layer = ".".join(parts[group_depth:])
                            layer_params[current_layer]["sub_layers"].add(sub_layer)
                            logger.info(
                                f"Dodano podwarstwę: {sub_layer} do warstwy {current_layer}"
                            )

                        total_params += value.numel()

                    elif isinstance(value, (dict, collections.OrderedDict)):
                        self.progress_update.emit(f"Analizuję słownik: {current_path}")
                        logger.info(f"Znaleziono zagnieżdżony słownik: {current_path}")
                        sub_params = analyze_dict(value, current_path, group_depth)
                        # Połącz parametry z podwarstwami
                        for layer, info in sub_params.items():
                            if layer not in layer_params:
                                layer_params[layer] = info
                                logger.info(
                                    f"Dodano nową warstwę z podwarstwy: {layer}"
                                )
                            else:
                                layer_params[layer]["parameters"] += info["parameters"]
                                layer_params[layer]["tensors"].extend(info["tensors"])
                                layer_params[layer]["shapes"].update(info["shapes"])
                                layer_params[layer]["dtypes"].update(info["dtypes"])
                                layer_params[layer]["sub_layers"].update(
                                    info["sub_layers"]
                                )
                                # Aktualizacja has_parameters
                                if "has_parameters" in info:
                                    layer_params[layer]["has_parameters"] = (
                                        layer_params[layer]["has_parameters"]
                                        or info["has_parameters"]
                                    )
                                # Aktualizacja statystyk parametrów
                                if "param_shapes" in info:
                                    layer_params[layer]["param_shapes"].update(
                                        info["param_shapes"]
                                    )
                                if "param_dtypes" in info:
                                    layer_params[layer]["param_dtypes"].update(
                                        info["param_dtypes"]
                                    )
                                if "param_stats" in info:
                                    layer_params[layer]["param_stats"].update(
                                        info["param_stats"]
                                    )
                                if info["layer_type"] != "unknown":
                                    layer_params[layer]["layer_type"] = info[
                                        "layer_type"
                                    ]
                                logger.info(f"Zaktualizowano warstwę: {layer}")
                    else:
                        self.progress_update.emit(
                            f"Pominięto nie-tensor: {current_path} ({type(value)})"
                        )
                        logger.info(
                            f"Pominięto nie-tensor: {current_path} ({type(value)})"
                        )

                logger.info(f"Zakończono analizę słownika z prefixem: {prefix}")
                logger.info(f"Znalezione warstwy: {list(layer_params.keys())}")
                return layer_params

            if isinstance(self.model, torch.nn.Module):
                logger.info("Analizuję model typu torch.nn.Module")
                self.progress_update.emit("Analizuję model typu torch.nn.Module")
                analyze_module(self.model)
            elif isinstance(self.model, dict):
                logger.info("Analizuję model typu dict (state_dict)")
                self.progress_update.emit("Analizuję model typu dict (state_dict)")
                layer_params = analyze_dict(self.model, group_depth=self.group_depth)

                # Konwertuj wyniki analizy słownika na format warstw
                for layer_name, info in layer_params.items():
                    layers_info.append(
                        {
                            "name": layer_name,
                            "parameters": info.get("parameters", 0),
                            "type": info.get("layer_type", "unknown"),
                            "tensors": info.get("tensors", []),
                            "shapes": list(info.get("shapes", set())),
                            "dtypes": list(info.get("dtypes", set())),
                            "sub_layers": list(info.get("sub_layers", set())),
                            "has_parameters": info.get("has_parameters", False),
                            "param_shapes": info.get("param_shapes", {}),
                            "param_dtypes": info.get("param_dtypes", {}),
                            "param_stats": info.get("param_stats", {}),
                        }
                    )
            else:
                raise ValueError(f"Nieobsługiwany typ modelu: {type(self.model)}")

            # Przygotuj raport
            report = []
            report.append(f"Typ modelu: {model_type}")
            report.append(f"Całkowita liczba parametrów: {total_params:,}")
            report.append(f"Liczba warstw: {len(layers_info)}")
            report.append("\nSzczegóły warstw:")

            # Sortuj warstwy według nazwy dla lepszej czytelności
            layers_info.sort(key=lambda x: x["name"])

            for layer in layers_info:
                report.append(f"\n{layer['name']}:")
                report.append(f"  Typ: {layer['type']}")
                report.append(f"  Liczba parametrów: {layer['parameters']:,}")

                if layer["has_parameters"]:
                    report.append("  Statystyki parametrów:")
                    for param_name, stats in layer["param_stats"].items():
                        report.append(f"    {param_name}:")
                        report.append(
                            f"      Kształt: {layer['param_shapes'][param_name]}"
                        )
                        report.append(
                            f"      Typ danych: {layer['param_dtypes'][param_name]}"
                        )
                        report.append(f"      Średnia: {stats['mean']:.4f}")
                        report.append(f"      Std: {stats['std']:.4f}")
                        report.append(f"      Min: {stats['min']:.4f}")
                        report.append(f"      Max: {stats['max']:.4f}")

                if "sub_layers" in layer and layer["sub_layers"]:
                    report.append("  Podwarstwy:")
                    for sub_layer in sorted(layer["sub_layers"]):
                        report.append(f"    - {sub_layer}")
                if "shapes" in layer:
                    report.append("  Kształty tensorów:")
                    for shape in layer["shapes"]:
                        report.append(f"    - {shape}")
                if "dtypes" in layer:
                    report.append("  Typy danych:")
                    for dtype in layer["dtypes"]:
                        report.append(f"    - {dtype}")
                if "tensors" in layer:
                    report.append("  Tensory:")
                    for tensor_path in layer["tensors"]:
                        report.append(f"    - {tensor_path}")

            final_report = "\n".join(report)
            logger.info(f"Wygenerowano raport analizy: {final_report[:200]}...")
            self.analysis_complete.emit(final_report)

        except Exception as e:
            error_msg = f"Błąd podczas analizy modelu: {str(e)}"
            logger.error(error_msg)
            self.analysis_error.emit(error_msg)


class ModelViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        logger.info("Uruchomiono Przeglądarkę Modeli PyTorch (UI)")
        self.setWindowTitle("Przeglądarka Modeli PyTorch")
        self.setGeometry(100, 100, 1200, 800)
        # Ustaw ikonę aplikacji
        icon_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "resources",
            "img",
            "icon.png",
        )
        icon_path = os.path.abspath(icon_path)
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
            logger.info(f"Ustawiono ikonę aplikacji: {icon_path}")
        else:
            logger.warning(f"Nie znaleziono pliku ikony: {icon_path}")

        # Kolory zgodne z Material Design i VS Code
        self.primary_color = "#007ACC"  # Niebieski VS Code
        self.success_color = "#10B981"  # Zielony
        self.warning_color = "#DC2626"  # Czerwony
        self.background = "#1E1E1E"  # Ciemne tło
        self.surface = "#252526"  # Lekko jaśniejsze tło dla paneli
        self.border_color = "#3F3F46"  # Kolor obramowania
        self.text_color = "#CCCCCC"  # Kolor tekstu

        # Inicjalizacja paska postępu
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)

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

        # Dodaj pole wyszukiwania do drzewa
        search_layout = QHBoxLayout()
        search_label = QLabel("Szukaj:")
        search_label.setStyleSheet(f"color: {self.text_color};")
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Wpisz nazwę warstwy lub parametru...")
        self.search_input.textChanged.connect(self.filter_tree)
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_input)
        tree_layout.addLayout(search_layout)

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
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setStyleSheet(
            f"background-color: {self.surface}; color: {self.text_color};"
            f"font-family: monospace; border: none;"
        )
        details_layout.addWidget(self.details_text)
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
            # Wyczyść pamięć podręczną i zwolnij poprzedni model
            if hasattr(self, "_details_cache"):
                self._details_cache = {}

            if hasattr(self, "model") and self.model is not None:
                del self.model
            if hasattr(self, "comparison_model") and self.comparison_model is not None:
                del self.comparison_model

            # Wymuś odśmiecanie
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
            error_msg = f"Błąd podczas wczytywania modelu: {str(e)}"
            logger.error(error_msg)
            self.details_text.setPlainText(error_msg)
            self.show_message("Błąd", error_msg, QMessageBox.Icon.Critical)

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
            self.details_text.setPlainText("Błąd: Brak modelu do analizy")
            return

        logger.info(f"Rozpoczynam analizę modelu typu: {type(self.model)}")

        # Wyłącz przycisk analizy i pokaż pasek postępu
        self.analyze_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Tryb nieokreślony
        self.details_text.setPlainText("Analizuję model...")

        # Utwórz i uruchom wątek analizy
        self.analyzer_thread = ModelAnalyzerThread(self.model)
        self.analyzer_thread.analysis_complete.connect(self._on_analysis_complete)
        self.analyzer_thread.analysis_error.connect(self._on_analysis_error)
        self.analyzer_thread.progress_update.connect(self._on_progress_update)
        logger.info("Uruchamiam wątek analizy modelu")
        self.analyzer_thread.start()

    def _on_analysis_complete(self, report):
        logger.info("Otrzymano sygnał zakończenia analizy")
        try:
            # Formatowanie raportu dla lepszej czytelności
            formatted_report = report.replace("\n", "<br>")
            self.details_text.setHtml(formatted_report)
            self.analyze_button.setEnabled(True)
            self.progress_bar.setVisible(False)
            logger.info("Analiza modelu zakończona i wyświetlona")
        except Exception as e:
            logger.error(f"Błąd podczas wyświetlania raportu: {str(e)}")
            self.details_text.setPlainText(
                f"Błąd podczas wyświetlania raportu: {str(e)}"
            )
            self.analyze_button.setEnabled(True)
            self.progress_bar.setVisible(False)

    def _on_analysis_error(self, error_msg):
        logger.error(f"Otrzymano sygnał błędu analizy: {error_msg}")
        self.details_text.setPlainText(f"Błąd podczas analizy: {error_msg}")
        self.analyze_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        logger.error(f"Błąd podczas analizy modelu: {error_msg}")

    def _on_progress_update(self, message):
        logger.debug(f"Otrzymano aktualizację postępu: {message}")
        # Aktualizuj status w interfejsie
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

        try:
            current = self.model
            for name in path[1:]:  # Pomijamy "Model" lub "State Dict"
                if isinstance(current, torch.nn.Module):
                    if hasattr(current, name):
                        current = getattr(current, name)
                    else:
                        try:
                            current = current.get_parameter(name)
                        except AttributeError:
                            raise ValueError(
                                f"Nie można znaleźć parametru: {name} "
                                f"w ścieżce {' -> '.join(path[:-1])}"
                            )
                elif isinstance(current, dict):
                    if name not in current:
                        raise ValueError(
                            f"Nie można znaleźć klucza: {name} "
                            f"w ścieżce {' -> '.join(path[:-1])}"
                        )
                    current = current[name]
                else:
                    raise ValueError(
                        f"Nieprawidłowy typ obiektu: {type(current)} "
                        f"w ścieżce {' -> '.join(path[:-1])}"
                    )
            return current
        except Exception as e:
            logger.error(f"Błąd podczas pobierania parametru: {str(e)}")
            raise

    def populate_tree(self):
        self.tree.clear()
        if isinstance(self.model, torch.nn.Module):
            root = QTreeWidgetItem(self.tree, ["Model"])
            self._add_module_to_tree(root, self.model)
        else:
            root = QTreeWidgetItem(self.tree, ["State Dict"])
            self._add_state_dict_to_tree(root, self.model)
        root.setExpanded(True)  # Rozwiń tylko główny węzeł

    def _add_module_to_tree(self, parent_item, module):
        MAX_CHILDREN = 500  # Maksymalna liczba elementów w drzewie
        count = 0

        for name, child in module.named_children():
            if count >= MAX_CHILDREN:
                item = QTreeWidgetItem(
                    parent_item, ["... i więcej elementów (limit wyświetlania)"]
                )
                break

            # Dodaj typ warstwy i liczbę parametrów
            layer_type = child.__class__.__name__
            param_count = sum(p.numel() for p in child.parameters())
            item_text = f"{name} ({layer_type}, " f"parametry: {param_count})"
            item = QTreeWidgetItem(parent_item, [item_text])
            if isinstance(child, torch.nn.Module):
                self._add_module_to_tree(item, child)
            self._add_parameters_to_tree(item, child)
            count += 1

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

        # Grupowanie parametrów dla lepszej wydajności
        param_batch = []
        for name, param in items:
            if isinstance(param, torch.Tensor):
                shape = list(param.shape)
                dtype = str(param.dtype)
                mean = float(param.float().mean().item()) if param.numel() > 0 else 0.0
                param_text = (
                    f"{name}: shape={shape}, " f"dtype={dtype}, mean={mean:.4f}"
                )
                param_batch.append(param_text)

        # Dodanie wszystkich parametrów naraz
        for param_text in param_batch:
            QTreeWidgetItem(parent_item, [param_text])

    def show_details(self, item, column):
        # Buforowanie ścieżki dla lepszej wydajności
        path = []
        current = item
        while current is not None:
            path.insert(0, current.text(0))
            current = current.parent()

        # Możemy dodać cache dla często oglądanych detali
        # Przykład:
        path_key = "/".join(path)
        if hasattr(self, "_details_cache") and path_key in self._details_cache:
            self.details_text.setPlainText(self._details_cache[path_key])
            return

        details = self._get_details(path)

        # Przechowuj w cache dla przyszłych wywołań
        if not hasattr(self, "_details_cache"):
            self._details_cache = {}
        self._details_cache[path_key] = details

        self.details_text.setPlainText(details)

    def _get_details(self, path):
        if not path:
            return "Brak szczegółów"

        try:
            current = self.model
            for name in path[1:]:  # Pomijamy "Model" lub "State Dict"
                # Wyciągnij czystą nazwę (bez typu i liczby parametrów)
                name_clean = name.split(":")[0].split("(")[0].strip()
                if isinstance(current, torch.nn.Module):
                    if hasattr(current, name_clean):
                        current = getattr(current, name_clean)
                    else:
                        try:
                            current = current.get_parameter(name_clean)
                        except AttributeError:
                            return (
                                f"Nie można znaleźć atrybutu/parametru: {name_clean} "
                                f"w {' -> '.join(path[:-1])}"
                            )
                elif isinstance(current, dict):
                    if name_clean not in current:
                        return f"Nie można znaleźć klucza: {name_clean}"
                    current = current[name_clean]
                else:
                    return f"Nie można znaleźć: {' -> '.join(path)}"

            if isinstance(current, torch.nn.Module):
                param_count = sum(p.numel() for p in current.parameters())
                return (
                    f"Moduł: {current.__class__.__name__}\n" f"Parametry: {param_count}"
                )
            elif isinstance(current, torch.Tensor):
                shape = list(current.shape)
                dtype = str(current.dtype)
                if current.numel() > 0:
                    mean = float(current.float().mean().item())
                    min_val = float(current.float().min().item())
                    max_val = float(current.float().max().item())
                    std = float(current.float().std().item())
                    return (
                        f"Tensor:\n"
                        f"  shape: {shape}\n"
                        f"  dtype: {dtype}\n"
                        f"  mean: {mean:.4f}\n"
                        f"  std: {std:.4f}\n"
                        f"  min: {min_val:.4f}\n"
                        f"  max: {max_val:.4f}"
                    )
                else:
                    return (
                        f"Tensor (pusty):\n" f"  shape: {shape}\n" f"  dtype: {dtype}"
                    )
            else:
                return str(current)
        except Exception as e:
            error_msg = f"Błąd podczas pobierania szczegółów: {str(e)}"
            logger.error(error_msg)
            return error_msg

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
                # Wykrywamy kształt wejścia z modelu jeśli to możliwe
                # zamiast zawsze używać 1x3x224x224
                input_shape = None
                for name, param in self.model.named_parameters():
                    if "weight" in name and len(param.shape) == 4:
                        # Najprawdopodobniej warstwa konwolucyjna, użyj jej kształtu jako wskazówki
                        input_shape = (1, param.shape[1], 224, 224)
                        break

                # Jeśli nie znaleziono, użyj domyślnego kształtu
                if input_shape is None:
                    input_shape = (1, 3, 224, 224)

                dummy_input = torch.randn(*input_shape)

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
            # Inicjalizacja okna dialogowego
            dialog = QDialog(self)
            dialog.setWindowTitle("Wizualizacja parametrów")
            dialog.setGeometry(200, 200, 1200, 800)
            layout = QVBoxLayout(dialog)
            tabs = QTabWidget()

            # Poprawa wydajności: zbieramy maksymalnie 100 tys. parametrów
            # dla szybszej wizualizacji dużych modeli
            MAX_PARAMS = 100000

            all_params = []
            layer_params = {}

            # Zbieranie parametrów - limit do MAX_PARAMS
            if isinstance(self.model, torch.nn.Module):
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        values = param.detach().cpu().numpy().flatten()
                        # Jeśli jest za dużo parametrów, pobierz podpróbkę
                        if len(values) > MAX_PARAMS:
                            indices = np.linspace(
                                0, len(values) - 1, MAX_PARAMS, dtype=int
                            )
                            values = values[indices]
                        all_params.extend(values.tolist())
                        layer_name = name.split(".")[0]
                        if layer_name not in layer_params:
                            layer_params[layer_name] = []
                        layer_params[layer_name].extend(values.tolist())
                        # Limit do MAX_PARAMS parametrów na warstwę
                        if len(layer_params[layer_name]) > MAX_PARAMS:
                            layer_params[layer_name] = layer_params[layer_name][
                                :MAX_PARAMS
                            ]
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

        # Zbierz parametry z obu modeli jako słowniki
        model1_params = {name: p for name, p in self.model.named_parameters()}
        model2_params = {
            name: p for name, p in self.comparison_model.named_parameters()
        }

        # Znajdź wspólne parametry
        common_params = set(model1_params.keys()) & set(model2_params.keys())
        only_in_model1 = set(model1_params.keys()) - set(model2_params.keys())
        only_in_model2 = set(model2_params.keys()) - set(model1_params.keys())

        # Przygotuj dane do wykresu
        model1_values = []
        model2_values = []
        param_names = []

        for name in common_params:
            p1 = model1_params[name].detach().cpu().numpy().flatten()
            p2 = model2_params[name].detach().cpu().numpy().flatten()
            model1_values.extend(p1)
            model2_values.extend(p2)
            param_names.extend([name] * len(p1))

        # Narysuj wykres porównawczy
        ax.scatter(model1_values, model2_values, alpha=0.5)
        ax.plot(
            [min(model1_values), max(model1_values)],
            [min(model1_values), max(model1_values)],
            "r--",
            label="y=x",
        )
        ax.set_title("Porównanie wartości parametrów")
        ax.set_xlabel("Model 1")
        ax.set_ylabel("Model 2")
        ax.legend()

        # Dodaj informacje o różnicach w strukturze
        if only_in_model1 or only_in_model2:
            diff_info = "Różnice w strukturze:\n"
            if only_in_model1:
                diff_info += f"\nParametry tylko w modelu 1:\n"
                for name in sorted(only_in_model1):
                    diff_info += f"- {name}\n"
            if only_in_model2:
                diff_info += f"\nParametry tylko w modelu 2:\n"
                for name in sorted(only_in_model2):
                    diff_info += f"- {name}\n"

            # Dodaj tekst do wykresu
            ax.text(
                0.02,
                0.98,
                diff_info,
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

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

    def filter_tree(self, text):
        """Filtruje drzewo na podstawie tekstu wyszukiwania."""

        def filter_item(item):
            if not text:
                item.setHidden(False)
                return True

            text_lower = text.lower()
            item_text = item.text(0).lower()

            # Sprawdź czy tekst pasuje do elementu
            matches = text_lower in item_text

            # Rekurencyjnie sprawdź dzieci
            has_visible_children = False
            for i in range(item.childCount()):
                child = item.child(i)
                if filter_item(child):
                    has_visible_children = True

            # Pokaż element jeśli pasuje lub ma pasujące dzieci
            item.setHidden(not (matches or has_visible_children))
            return matches or has_visible_children

        # Zastosuj filtrowanie do wszystkich elementów
        root = self.tree.topLevelItem(0)
        if root:
            filter_item(root)


def main():
    app = QApplication(sys.argv)
    viewer = ModelViewer()
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
