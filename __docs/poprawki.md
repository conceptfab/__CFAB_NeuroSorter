Zidentyfikowałem dwie główne przyczyny problemów w dostarczonym kodzie, które mogły prowadzić do błędu `KeyError: 'has_parameters'` oraz potencjalnie niepoprawnego grupowania warstw podczas analizy modeli typu `dict` (np. `state_dict`).

**Problem 1: Brakujące klucze w `layers_info` dla modeli typu `dict`**

Kiedy analizowany model jest słownikiem (np. `state_dict`), informacje o warstwach są zbierane w `layer_params` w funkcji `analyze_dict`. Ta funkcja poprawnie inicjalizuje i wypełnia klucze takie jak `'has_parameters'`, `'param_shapes'`, `'param_dtypes'` i `'param_stats'`.
Jednak podczas konwersji `layer_params` do `layers_info`, te klucze nie były kopiowane. Pętla generująca raport następnie próbowała uzyskać dostęp do `layer["has_parameters"]`, co prowadziło do `KeyError`.

**Rozwiązanie dla Problemu 1:**
Zmodyfikowałem sekcję konwersji `layer_params` na `layers_info` w metodzie `run` klasy `ModelAnalyzerThread`, aby uwzględnić brakujące klucze. Użyłem również metody `.get()` dla bezpieczeństwa, aby zapewnić domyślne wartości, gdyby klucz z jakiegoś powodu nie istniał.

```python
# ... wewnątrz ModelAnalyzerThread.run() ...
            elif isinstance(self.model, dict):
                logger.info("Analizuję model typu dict (state_dict)")
                self.progress_update.emit("Analizuję model typu dict (state_dict)")
                # Poprawka: Przekazuj self.group_depth do analyze_dict
                layer_params = analyze_dict(self.model, group_depth=self.group_depth)

                # Konwertuj wyniki analizy słownika na format warstw
                for layer_name, info in layer_params.items():
                    layers_info.append(
                        {
                            "name": layer_name,
                            "parameters": info.get("parameters", 0), # Użyj .get dla bezpieczeństwa
                            "type": info.get("layer_type", "unknown"),
                            "tensors": info.get("tensors", []),
                            "shapes": list(info.get("shapes", set())),
                            "dtypes": list(info.get("dtypes", set())),
                            "sub_layers": list(info.get("sub_layers", set())),
                            # POPRAWKA: Dodanie brakujących kluczy
                            "has_parameters": info.get("has_parameters", False),
                            "param_shapes": info.get("param_shapes", {}),
                            "param_dtypes": info.get("param_dtypes", {}),
                            "param_stats": info.get("param_stats", {}),
                        }
                    )
# ... reszta kodu ...
```

**Problem 2: Niepoprawne przekazywanie `group_depth` w rekurencyjnych wywołaniach `analyze_dict`**

Funkcja `analyze_dict` miała zdefiniowany parametr `group_depth` z wartością domyślną. Kiedy `analyze_dict` była wywoływana rekurencyjnie dla zagnieżdżonych słowników, nie przekazywała oryginalnej (lub skonfigurowanej przez użytkownika) wartości `group_depth`, lecz polegała na wartości domyślnej. To mogło prowadzić do niespójnego grupowania warstw.

**Rozwiązanie dla Problemu 2:**
Zmodyfikowałem rekurencyjne wywołanie `analyze_dict` tak, aby przekazywało bieżącą wartość `group_depth`.

```python
# ... wewnątrz ModelAnalyzerThread.run() -> def analyze_dict(...) ...
                    elif isinstance(value, (dict, collections.OrderedDict)):
                        self.progress_update.emit(f"Analizuję słownik: {current_path}")
                        logger.info(f"Znaleziono zagnieżdżony słownik: {current_path}")
                        # POPRAWKA: Przekaż group_depth do rekurencyjnego wywołania
                        sub_params = analyze_dict(value, current_path, group_depth)
                        # Połącz parametry z podwarstwami
# ... reszta kodu ...
```

Poniżej znajduje się pełny, poprawiony kod klasy `ModelAnalyzerThread`. Reszta kodu pozostaje bez zmian, ponieważ błąd dotyczył logiki analizy.

```python
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

    def __init__(self, model, group_depth=2):
        super().__init__()
        self.model = model
        self.group_depth = group_depth
        logger.info(f"Utworzono wątek analizy modelu z group_depth={group_depth}")

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
                # Sprawdź, czy moduł ma metodę parameters przed jej wywołaniem
                if not hasattr(module, 'parameters'):
                    logger.debug(f"Moduł {name or module.__class__.__name__} ({module.__class__.__name__}) nie ma metody 'parameters'. Pomijam analizę parametrów.")
                    # Można tu dodać podstawowe informacje o module bez parametrów, jeśli to potrzebne
                    # Na przykład, tylko typ i nazwę
                    layer_info = {
                        "name": name or module.__class__.__name__,
                        "parameters": 0,
                        "type": module.__class__.__name__,
                        "has_parameters": False,
                        "param_shapes": {},
                        "param_dtypes": {},
                        "param_stats": {},
                    }
                    layers_info.append(layer_info)

                    # Nadal rekurencyjnie analizuj dzieci
                    for child_name, child in module.named_children():
                        analyze_module(
                            child, f"{name}.{child_name}" if name else child_name
                        )
                    return

                params = sum(p.numel() for p in module.parameters() if p.requires_grad) # Liczymy tylko parametry z requires_grad

                # Jeśli moduł nie ma własnych parametrów (tylko dzieci mają), to params będzie 0.
                # total_params powinno sumować parametry z `module.parameters()` i rekurencyjnie z dzieci.
                # Poniższa linia sumuje parametry tylko bieżącego modułu (bez dzieci)
                # To jest OK, bo total_params jest nonlocal i aktualizowane w każdym wywołaniu.

                # Zbierz informacje o kształtach parametrów
                param_shapes = {}
                param_dtypes = {}
                param_stats = {}

                # Iteruj po named_parameters() tylko jeśli istnieją
                if hasattr(module, 'named_parameters'):
                    for param_name, param in module.named_parameters():
                        if param.requires_grad: # Analizuj tylko parametry, które się uczą
                            # Sumowanie parametrów powinno się odbywać tylko raz dla każdego parametru
                            # `total_params` jest już aktualizowane o `params` wyżej
                            # Jeśli chcemy sumować parametry rekurencyjnie, to `total_params += param.numel()` powinno być tutaj.
                            # Ale obecna logika `params = sum(...)` i potem `total_params += params`
                            # jest poprawna, jeśli `total_params` jest aktualizowane tylko dla modułów z `module.parameters()`.
                            # Problem może być, jeśli `total_params` jest aktualizowane i tu i przez `params` wyżej.
                            # Dla uproszczenia: `params` to parametry *tego* modułu (nie dzieci).
                            # `total_params` to suma wszystkich parametrów.

                            param_shapes[param_name] = list(param.shape)
                            param_dtypes[param_name] = str(param.dtype)
                            if param.numel() > 0:
                                param_stats[param_name] = {
                                    "mean": float(param.float().mean().item()),
                                    "std": float(param.float().std().item()),
                                    "min": float(param.float().min().item()),
                                    "max": float(param.float().max().item()),
                                }

                # `total_params` powinno być sumą wszystkich parametrów modelu.
                # Obecnie `params` to suma parametrów *bezpośrednio* w `module`.
                # A `total_params` jest aktualizowane przez `total_params += params`.
                # To oznacza, że `total_params` będzie sumą parametrów wszystkich modułów, które *mają* parametry.
                # To jest poprawne.

                total_params_module_level = sum(p.numel() for p in module.parameters() if p.requires_grad)


                layer_info = {
                    "name": name or module.__class__.__name__,
                    "parameters": total_params_module_level, # Użyj parametrów bezpośrednio z tego modułu
                    "type": module.__class__.__name__,
                    "has_parameters": total_params_module_level > 0,
                    "param_shapes": param_shapes,
                    "param_dtypes": param_dtypes,
                    "param_stats": param_stats,
                }
                layers_info.append(layer_info)

                if total_params_module_level > 0:
                    msg = (
                        f"Znaleziono warstwę: {layer_info['name']} "
                        f"({layer_info['type']}) z {total_params_module_level:,} parametrami"
                    )
                    # Aktualizuj total_params tylko raz dla parametrów na tym poziomie
                    # Jeśli jest to kontener, jego parametry to suma parametrów dzieci,
                    # więc nie chcemy podwójnie liczyć.
                    # Prawidłowe sumowanie total_params powinno się odbywać na najniższym poziomie, gdzie parametry są zdefiniowane.
                    # Najprościej jest, jeśli `total_params` jest sumowane na końcu na podstawie `layers_info`.
                    # Alternatywnie, jeśli `analyze_module` jest wywoływane rekurencyjnie,
                    # `total_params` powinno być aktualizowane tylko dla liści (modułów z parametrami, a nie kontenerów).
                    # Obecna logika `params = sum(...)` i potem `total_params += params` (które jest `total_params_module_level`)
                    # zsumuje parametry każdego modułu. Jeśli moduł A zawiera moduł B, a oba mają parametry,
                    # to parametry B będą liczone dwukrotnie (raz w B, raz jako część A).
                    # Aby tego uniknąć, sumuj `p.numel()` dla `p in module.parameters(recurse=False)`.
                    # Lub, po prostu zaufaj, że PyTorch `module.parameters()` (domyślnie recurse=True) da wszystkie.
                    # W takim razie `total_params` powinno być liczone tylko raz na początku dla całego modelu.
                    # Spróbujmy inaczej: `total_params` będzie sumą z `layers_info` na końcu.
                    # A `params` w `layer_info` to parametry *tylko* tego modułu.

                else: # total_params_module_level == 0
                    msg = (
                        f"Znaleziono moduł: {layer_info['name']} "
                        f"({layer_info['type']}) bez własnych parametrów"
                    )
                logger.info(msg)
                self.progress_update.emit(msg)

                for child_name, child in module.named_children():
                    analyze_module(
                        child, f"{name}.{child_name}" if name else child_name
                    )

            # Reset total_params, będzie liczony na podstawie layers_info na końcu
            # lub inaczej:
            # Jeśli model to nn.Module, oblicz total_params raz na początku
            initial_total_params = 0
            if isinstance(self.model, torch.nn.Module):
                initial_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)


            def analyze_dict(d, prefix="", current_group_depth=2): # Zmieniono nazwę argumentu dla jasności
                # `total_params` jest `nonlocal`, więc modyfikuje zmienną z `run`
                nonlocal total_params # To total_params będzie sumą dla state_dict

                layer_params_agg = {} # Zmieniono nazwę, aby uniknąć konfliktu z `layer_params` w `run`

                logger.info(f"Analizuję słownik z prefixem: {prefix}, głębokość grupowania: {current_group_depth}")
                logger.info(f"Klucze w słowniku: {list(d.keys())}")

                for key, value in d.items():
                    current_path = f"{prefix}.{key}" if prefix else key
                    self.progress_update.emit(
                        f"Analizuję klucz: {current_path}, typ wartości: {type(value)}"
                    )

                    parts = current_path.split(".")
                    logger.info(f"Ścieżka: {current_path}, części: {parts}")

                    if len(parts) >= current_group_depth:
                        current_layer_key = ".".join(parts[:current_group_depth])
                    else:
                        current_layer_key = parts[0] # Lub cała ścieżka, jeśli krótsza niż głębokość
                        if len(parts) < current_group_depth and len(parts) > 1: # np. a.b przy depth=3 -> current_layer_key = a.b
                             current_layer_key = ".".join(parts)


                    logger.info(
                        f"Użyto grupowania do głębokości {current_group_depth}: klucz warstwy '{current_layer_key}'"
                    )

                    if isinstance(value, torch.Tensor):
                        num_elements = value.numel()
                        msg = (
                            f"Tensor {current_path}: shape={value.shape}, "
                            f"dtype={value.dtype}, numel={num_elements}"
                        )
                        self.progress_update.emit(msg)
                        logger.info(msg)

                        if current_layer_key not in layer_params_agg:
                            layer_params_agg[current_layer_key] = {
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
                            logger.info(f"Utworzono nową grupę warstw: {current_layer_key}")

                        # Użyj LAYER_PATTERNS do określenia typu warstwy dla grupy
                        # Typ warstwy powinien być określany na podstawie nazwy grupy, a nie każdego tensora
                        if layer_params_agg[current_layer_key]["layer_type"] == "unknown":
                            for type_name, patterns in LAYER_PATTERNS.items():
                                if any(
                                    pattern in current_layer_key.lower() for pattern in patterns
                                ):
                                    layer_params_agg[current_layer_key]["layer_type"] = type_name
                                    break

                        layer_params_agg[current_layer_key]["parameters"] += num_elements
                        layer_params_agg[current_layer_key]["tensors"].append(current_path)
                        layer_params_agg[current_layer_key]["shapes"].add(str(value.shape))
                        layer_params_agg[current_layer_key]["dtypes"].add(str(value.dtype))
                        if num_elements > 0:
                             layer_params_agg[current_layer_key]["has_parameters"] = True

                        if num_elements > 0:
                            layer_params_agg[current_layer_key]["param_shapes"][
                                current_path
                            ] = list(value.shape)
                            layer_params_agg[current_layer_key]["param_dtypes"][
                                current_path
                            ] = str(value.dtype)
                            layer_params_agg[current_layer_key]["param_stats"][current_path] = {
                                "mean": float(value.float().mean().item()),
                                "std": float(value.float().std().item()),
                                "min": float(value.float().min().item()),
                                "max": float(value.float().max().item()),
                            }

                        if len(parts) > current_group_depth:
                            sub_layer_path = ".".join(parts[current_group_depth:])
                            layer_params_agg[current_layer_key]["sub_layers"].add(sub_layer_path)
                            logger.info(
                                f"Dodano podścieżkę tensora: {sub_layer_path} do grupy {current_layer_key}"
                            )

                        total_params += num_elements # Sumuj parametry dla state_dict

                    elif isinstance(value, (dict, collections.OrderedDict)):
                        self.progress_update.emit(f"Analizuję zagnieżdżony słownik: {current_path}")
                        logger.info(f"Znaleziono zagnieżdżony słownik: {current_path}")
                        # POPRAWKA: Przekaż group_depth do rekurencyjnego wywołania
                        sub_results = analyze_dict(value, current_path, current_group_depth)

                        for sub_layer_key, sub_info in sub_results.items():
                            if sub_layer_key not in layer_params_agg:
                                layer_params_agg[sub_layer_key] = sub_info
                                logger.info(
                                    f"Dodano nową grupę warstw z pod-słownika: {sub_layer_key}"
                                )
                            else:
                                # Łączenie informacji
                                layer_params_agg[sub_layer_key]["parameters"] += sub_info.get("parameters", 0)
                                layer_params_agg[sub_layer_key]["tensors"].extend(sub_info.get("tensors", []))
                                layer_params_agg[sub_layer_key]["shapes"].update(sub_info.get("shapes", set()))
                                layer_params_agg[sub_layer_key]["dtypes"].update(sub_info.get("dtypes", set()))
                                layer_params_agg[sub_layer_key]["sub_layers"].update(sub_info.get("sub_layers", set()))
                                if sub_info.get("has_parameters", False):
                                     layer_params_agg[sub_layer_key]["has_parameters"] = True

                                layer_params_agg[sub_layer_key]["param_shapes"].update(sub_info.get("param_shapes", {}))
                                layer_params_agg[sub_layer_key]["param_dtypes"].update(sub_info.get("param_dtypes", {}))
                                layer_params_agg[sub_layer_key]["param_stats"].update(sub_info.get("param_stats", {}))

                                if layer_params_agg[sub_layer_key]["layer_type"] == "unknown" and sub_info.get("layer_type", "unknown") != "unknown":
                                    layer_params_agg[sub_layer_key]["layer_type"] = sub_info["layer_type"]
                                logger.info(f"Zaktualizowano grupę warstw: {sub_layer_key} z pod-słownika")
                    else:
                        self.progress_update.emit(
                            f"Pominięto nie-tensor/słownik: {current_path} ({type(value)})"
                        )
                        logger.info(
                            f"Pominięto nie-tensor/słownik: {current_path} ({type(value)})"
                        )

                logger.info(f"Zakończono analizę słownika z prefixem: {prefix}")
                logger.info(f"Znalezione grupy warstw na tym poziomie: {list(layer_params_agg.keys())}")
                return layer_params_agg

            if isinstance(self.model, torch.nn.Module):
                logger.info("Analizuję model typu torch.nn.Module")
                self.progress_update.emit("Analizuję model typu torch.nn.Module")
                # Wyczyść layers_info przed ponownym wypełnieniem
                layers_info.clear()
                analyze_module(self.model)
                # Ustaw total_params na podstawie sumy parametrów z nn.Module
                total_params = initial_total_params # Użyj wcześniej obliczonej sumy dla nn.Module

            elif isinstance(self.model, dict):
                logger.info("Analizuję model typu dict (state_dict)")
                self.progress_update.emit("Analizuję model typu dict (state_dict)")
                # `total_params` będzie modyfikowane wewnątrz `analyze_dict` przez `nonlocal`
                total_params = 0 # Reset dla state_dict, będzie sumowane w analyze_dict

                # Przekaż self.group_depth
                analyzed_layer_groups = analyze_dict(self.model, current_group_depth=self.group_depth)

                layers_info.clear() # Wyczyść layers_info
                for layer_name, info in analyzed_layer_groups.items():
                    layers_info.append(
                        {
                            "name": layer_name,
                            "parameters": info.get("parameters", 0),
                            "type": info.get("layer_type", "unknown"),
                            "has_parameters": info.get("has_parameters", False),
                            "param_shapes": info.get("param_shapes", {}),
                            "param_dtypes": info.get("param_dtypes", {}),
                            "param_stats": info.get("param_stats", {}),
                            # Dodatkowe pola, jeśli są potrzebne w raporcie
                            "tensors": info.get("tensors", []),
                            "shapes": list(info.get("shapes", set())),
                            "dtypes": list(info.get("dtypes", set())),
                            "sub_layers": list(info.get("sub_layers", set())),
                        }
                    )
            else:
                raise ValueError(f"Nieobsługiwany typ modelu: {type(self.model)}")

            # Przygotuj raport
            report = []
            report.append(f"Typ modelu: {model_type}")
            report.append(f"Całkowita liczba parametrów (uczonych): {total_params:,}") # total_params jest teraz poprawne
            report.append(f"Liczba znalezionych warstw/grup: {len(layers_info)}") # Dla dict to będą grupy
            report.append("\nSzczegóły warstw/grup:")

            layers_info.sort(key=lambda x: x["name"])

            for layer in layers_info:
                report.append(f"\n{layer['name']}:")
                report.append(f"  Typ: {layer['type']}")
                report.append(f"  Liczba parametrów: {layer.get('parameters', 0):,}")

                # Sprawdź 'has_parameters' przed dostępem do statystyk
                if layer.get("has_parameters", False): # Użyj .get dla bezpieczeństwa
                    # Sprawdź, czy 'param_stats' istnieje i nie jest puste
                    if layer.get("param_stats"):
                        report.append("  Statystyki parametrów (dla tensorów w tej grupie):")
                        for param_path, stats in layer["param_stats"].items(): # param_path to ścieżka tensora
                            report.append(f"    Tensor: {param_path}:")
                            if param_path in layer.get("param_shapes", {}):
                                report.append(
                                    f"      Kształt: {layer['param_shapes'][param_path]}"
                                )
                            if param_path in layer.get("param_dtypes", {}):
                                report.append(
                                    f"      Typ danych: {layer['param_dtypes'][param_path]}"
                                )
                            report.append(f"      Średnia: {stats['mean']:.4f}")
                            report.append(f"      Std: {stats['std']:.4f}")
                            report.append(f"      Min: {stats['min']:.4f}")
                            report.append(f"      Max: {stats['max']:.4f}")
                    else:
                        report.append("  Brak szczegółowych statystyk parametrów dla tej grupy.")
                else:
                    report.append("  Brak uczonych parametrów w tej warstwie/grupie.")


                if "sub_layers" in layer and layer["sub_layers"]:
                    report.append("  Zawarte podścieżki (jeśli grupa z state_dict):")
                    for sub_layer in sorted(list(layer["sub_layers"])): # Konwersja set na list dla sortowania
                        report.append(f"    - {sub_layer}")
                if "shapes" in layer and layer.get("shapes"): # Sprawdź czy shapes nie jest puste
                    report.append("  Kształty tensorów w grupie:")
                    for shape in sorted(list(layer["shapes"])):
                        report.append(f"    - {shape}")
                if "dtypes" in layer and layer.get("dtypes"):
                    report.append("  Typy danych tensorów w grupie:")
                    for dtype in sorted(list(layer["dtypes"])):
                        report.append(f"    - {dtype}")
                if "tensors" in layer and layer.get("tensors"):
                    report.append("  Pełne ścieżki tensorów w grupie:")
                    for tensor_path in sorted(layer["tensors"]):
                        report.append(f"    - {tensor_path}")


            final_report = "\n".join(report)
            logger.info(f"Wygenerowano raport analizy (początek): {final_report[:300]}...")
            self.analysis_complete.emit(final_report)

        except Exception as e:
            import traceback
            error_msg = f"Błąd podczas analizy modelu: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.analysis_error.emit(error_msg)

# Reszta klas (ModelViewer, etc.) pozostaje bez zmian.
# Jeśli chcesz, mogę wkleić cały plik z tą poprawioną klasą.

# --- Miejsce na resztę kodu (ModelViewer, main, etc.) ---
# (Pozostała część kodu z pytania)
# ... (cały kod ModelViewer i funkcji main) ...
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
        try:
            model_files = [
                f for f in os.listdir(self.models_dir) if f.endswith((".pt", ".pth"))
            ]
        except FileNotFoundError:
            logger.warning(f"Folder modeli {self.models_dir} nie istnieje. Tworzenie.")
            try:
                os.makedirs(self.models_dir)
                model_files = []
            except Exception as e:
                logger.error(f"Nie udało się utworzyć folderu modeli: {e}")
                self.show_message("Błąd", f"Nie można utworzyć folderu modeli: {self.models_dir}", QMessageBox.Icon.Critical)
                return

        for fname in model_files:
            self.model_list.addItem(fname)
        if model_files:
            logger.info("Znaleziono modele: %s", ", ".join(model_files))
            self.model_list.setCurrentRow(0)
            self.load_model_from_file(os.path.join(self.models_dir, model_files[0]))
        else:
            logger.info("Nie znaleziono żadnych modeli w folderze.")
            self.show_message("Informacja", f"Brak modeli w folderze: {self.models_dir}. Możesz je tam dodać.", QMessageBox.Icon.Information)


    def on_model_selected(self, item):
        logger.info("Użytkownik wybrał model: %s", item.text())
        model_path = os.path.join(self.models_dir, item.text())
        self.load_model_from_file(model_path)

    def load_model_from_file(self, file_path):
        logger.info("Wczytywanie modelu z pliku: %s", file_path)
        try:
            if hasattr(self, "_details_cache"):
                self._details_cache = {}

            if hasattr(self, "model") and self.model is not None:
                del self.model
            if hasattr(self, "comparison_model") and self.comparison_model is not None:
                del self.comparison_model

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.model = torch.load(file_path, map_location=torch.device("cpu"))
            self.current_model_path = file_path

            self.populate_tree()
            self.save_button.setEnabled(True)
            self.export_structure_button.setEnabled(True)
            self.analyze_button.setEnabled(True)
            self.export_onnx_button.setEnabled(isinstance(self.model, torch.nn.Module))
            self.export_torchscript_button.setEnabled(isinstance(self.model, torch.nn.Module))
            self.visualize_params_button.setEnabled(True)
            self.compare_models_button.setEnabled(True)
            logger.info("Model został poprawnie wczytany: %s", file_path)
            self.details_text.setPlainText(f"Załadowano model: {os.path.basename(file_path)}\n"
                                           f"Typ: {type(self.model).__name__}\n"
                                           "Kliknij 'Analizuj' aby zobaczyć szczegóły lub wybierz element z drzewa.")

        except Exception as e:
            import traceback
            error_msg = f"Błąd podczas wczytywania modelu: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.details_text.setPlainText(error_msg)
            self.show_message("Błąd", error_msg, QMessageBox.Icon.Critical)
            # Wyłącz przyciski, jeśli model nie został załadowany
            self.save_button.setEnabled(False)
            self.export_structure_button.setEnabled(False)
            self.analyze_button.setEnabled(False)
            self.export_onnx_button.setEnabled(False)
            self.export_torchscript_button.setEnabled(False)
            self.visualize_params_button.setEnabled(False)
            self.compare_models_button.setEnabled(False)


    def show_context_menu(self, position):
        logger.info("Wyświetlenie menu kontekstowego dla drzewa modelu")
        item = self.tree.itemAt(position)
        if not item:
            logger.warning("Brak wybranego elementu w drzewie")
            return

        menu = QMenu()
        # Opcja edycji (może być ograniczona lub usunięta, jeśli powoduje problemy)
        # edit_action = menu.addAction("Edytuj parametr (eksperymentalne)")
        # edit_action.triggered.connect(lambda: self.edit_parameter(item))
        # menu.addSeparator()
        copy_path_action = menu.addAction("Kopiuj ścieżkę elementu")
        copy_path_action.triggered.connect(lambda: self.copy_item_path(item))

        menu.exec(self.tree.mapToGlobal(position))

    def copy_item_path(self, item):
        path = []
        current = item
        while current is not None:
            path.insert(0, current.text(0).split(" (")[0].split(":")[0].strip()) # Czysta nazwa
            current = current.parent()

        path_str = " -> ".join(path)
        clipboard = QApplication.clipboard()
        clipboard.setText(path_str)
        self.show_message("Skopiowano", f"Ścieżka '{path_str}' została skopiowana do schowka.")


    def edit_parameter(self, item):
        logger.info("Edycja parametru przez użytkownika")
        # Ta funkcja jest ryzykowna i może nie działać dla wszystkich struktur,
        # zwłaszcza dla state_dict. Powinna być używana z ostrożnością.
        path_parts = []
        current = item
        while current is not None:
            # Pobierz czystą nazwę, usuwając dodatkowe informacje w nawiasach lub po dwukropku
            raw_text = current.text(0)
            clean_name = raw_text.split(" (")[0].split(":")[0].strip()
            path_parts.insert(0, clean_name)
            current = current.parent()

        if not path_parts:
            self.show_message("Błąd", "Nie można zidentyfikować ścieżki do elementu.", QMessageBox.Icon.Warning)
            return

        logger.debug(f"Próba edycji elementu o ścieżce: {path_parts}")

        try:
            param_obj = self._get_parameter_by_path(path_parts)
        except ValueError as e:
            logger.warning(f"Nie można pobrać parametru: {e}")
            self.show_message("Błąd", f"Nie można zlokalizować elementu do edycji: {e}", QMessageBox.Icon.Warning)
            return

        if not isinstance(param_obj, torch.Tensor):
            logger.warning(
                "Wybrany element nie jest tensorem i nie można go edytować w ten sposób: %s", " -> ".join(path_parts)
            )
            self.show_message(
                "Informacja", "Można edytować tylko wartości tensorów.", QMessageBox.Icon.Information
            )
            return

        if param_obj.numel() == 0:
            self.show_message("Informacja", "Wybrany tensor jest pusty i nie można go edytować.", QMessageBox.Icon.Information)
            return

        if param_obj.numel() == 1:
            current_value = param_obj.item()
            new_value_str, ok = QInputDialog.getText(
                self,
                "Edytuj wartość tensora (skalar)",
                f"Wprowadź nową wartość dla '{path_parts[-1]}':",
                QLineEdit.EchoMode.Normal,
                str(current_value)
            )
            if ok and new_value_str:
                try:
                    new_value = float(new_value_str) # Próba konwersji na float
                    param_obj.fill_(new_value)
                    self.populate_tree()  # Odśwież widok
                    self.show_details(item, 0) # Odśwież panel szczegółów
                    logger.info(
                        "Tensor %s został zaktualizowany na wartość: %s", path_parts[-1], new_value
                    )
                    self.show_message("Sukces", "Wartość tensora została zaktualizowana.")
                except ValueError:
                    self.show_message("Błąd", "Wprowadzona wartość nie jest liczbą.", QMessageBox.Icon.Warning)
                except Exception as e:
                    logger.error("Nie udało się zaktualizować tensora: %s", str(e))
                    self.show_message("Błąd", f"Nie udało się zaktualizować tensora: {str(e)}", QMessageBox.Icon.Critical)
        else: # Tensor wielowymiarowy
            choice, ok = QInputDialog.getItem(self, "Edycja tensora",
                                               f"Tensor '{path_parts[-1]}' ma {param_obj.numel()} elementów.\nJak chcesz go edytować?",
                                               ["Wypełnij stałą wartością", "Anuluj"], 0, False)
            if ok and choice == "Wypełnij stałą wartością":
                fill_value_str, ok_fill = QInputDialog.getText(
                    self,
                    "Wypełnij tensor",
                    f"Wprowadź wartość, którą wypełnić tensor '{path_parts[-1]}':",
                    QLineEdit.EchoMode.Normal,
                    "0.0"
                )
                if ok_fill and fill_value_str:
                    try:
                        fill_value = float(fill_value_str)
                        param_obj.fill_(fill_value)
                        self.populate_tree()
                        self.show_details(item, 0)
                        logger.info("Tensor %s został wypełniony wartością: %s", path_parts[-1], fill_value)
                        self.show_message("Sukces", f"Tensor został wypełniony wartością {fill_value}.")
                    except ValueError:
                        self.show_message("Błąd", "Wprowadzona wartość do wypełnienia nie jest liczbą.", QMessageBox.Icon.Warning)
                    except Exception as e:
                        logger.error("Nie udało się wypełnić tensora: %s", str(e))
                        self.show_message("Błąd", f"Nie udało się wypełnić tensora: {str(e)}", QMessageBox.Icon.Critical)


    def save_model(self):
        logger.info("Zapis modelu do pliku")
        if not self.model:
            logger.warning("Brak modelu do zapisania")
            self.show_message("Błąd", "Brak załadowanego modelu do zapisania.", QMessageBox.Icon.Warning)
            return

        default_path = self.current_model_path or os.path.join(self.models_dir, "zapisany_model.pth")
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Zapisz model",
            default_path,
            "PyTorch Files (*.pt *.pth);;All Files (*)",
        )

        if file_name:
            try:
                torch.save(self.model, file_name)
                # Aktualizuj current_model_path jeśli zapisano pod nową nazwą lub po raz pierwszy
                self.current_model_path = file_name
                # Dodaj/zaktualizuj na liście modeli, jeśli zapisano w domyślnym folderze
                if os.path.dirname(file_name) == self.models_dir:
                    self.load_models_from_folder() # Odświeży listę
                    # Spróbuj zaznaczyć nowo zapisany model
                    for i in range(self.model_list.count()):
                        if self.model_list.item(i).text() == os.path.basename(file_name):
                            self.model_list.setCurrentRow(i)
                            break

                self.show_message("Sukces", f"Model został zapisany pomyślnie jako:\n{file_name}")
            except Exception as e:
                logger.error(f"Błąd podczas zapisywania modelu: {str(e)}")
                self.show_message(
                    "Błąd",
                    f"Nie udało się zapisać modelu: {str(e)}",
                    QMessageBox.Icon.Critical,
                )

    def export_structure(self):
        if not self.model:
            self.show_message("Błąd", "Brak załadowanego modelu do eksportu struktury.", QMessageBox.Icon.Warning)
            return

        default_filename = "struktura_modelu.json"
        if self.current_model_path:
            base, _ = os.path.splitext(os.path.basename(self.current_model_path))
            default_filename = f"{base}_struktura.json"

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Eksportuj strukturę",
            os.path.join(self.models_dir, default_filename), # Sugeruj folder modeli
            "JSON Files (*.json);;Text Files (*.txt);;All Files (*)",
        )

        if file_name:
            try:
                structure = self._get_model_structure(self.model) # Przekaż model do funkcji
                with open(file_name, "w", encoding="utf-8") as f:
                    json.dump(structure, f, indent=2, ensure_ascii=False)
                self.show_message("Sukces", f"Struktura modelu została wyeksportowana do:\n{file_name}")
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
            self.details_text.setPlainText("Błąd: Brak modelu do analizy. Załaduj model z listy.")
            self.show_message("Błąd", "Brak załadowanego modelu do analizy.", QMessageBox.Icon.Warning)
            return

        logger.info(f"Rozpoczynam analizę modelu typu: {type(self.model)}")

        self.analyze_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.details_text.setPlainText("Analizuję model, proszę czekać...")
        QApplication.processEvents() # Upewnij się, że UI się odświeży

        # Pobierz group_depth, np. z QInputDialog
        group_depth, ok = QInputDialog.getInt(self, "Głębokość grupowania dla state_dict",
                                              "Podaj głębokość grupowania (np. 2 dla 'transformer.h'):",
                                              value=2, min=1, max=10)
        if not ok: # Użytkownik anulował
            self.analyze_button.setEnabled(True)
            self.progress_bar.setVisible(False)
            self.details_text.setPlainText("Analiza anulowana przez użytkownika.")
            return

        self.analyzer_thread = ModelAnalyzerThread(self.model, group_depth=group_depth)
        self.analyzer_thread.analysis_complete.connect(self._on_analysis_complete)
        self.analyzer_thread.analysis_error.connect(self._on_analysis_error)
        self.analyzer_thread.progress_update.connect(self._on_progress_update)
        logger.info("Uruchamiam wątek analizy modelu")
        self.analyzer_thread.start()

    def _on_analysis_complete(self, report):
        logger.info("Otrzymano sygnał zakończenia analizy")
        try:
            # Użyj HTML do formatowania, ale upewnij się, że znaki specjalne są escapowane
            # jeśli raport nie jest HTML. W tym przypadku raport to zwykły tekst z \n.
            # QTextEdit.setHtml() poprawnie zinterpretuje <br> zamiast \n.
            html_report = report.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            html_report = html_report.replace("\n", "<br>")

            # Dodaj styl dla preformatowanego tekstu dla lepszej czytelności
            styled_html_report = (f"<div style='font-family: \"DejaVu Sans Mono\", monospace; "
                                  f"white-space: pre-wrap; color: {self.text_color};'>{html_report}</div>")

            self.details_text.setHtml(styled_html_report)
            self.analyze_button.setEnabled(True)
            self.progress_bar.setVisible(False)
            self.statusBar().showMessage("Analiza zakończona.", 5000)
            logger.info("Analiza modelu zakończona i wyświetlona")
        except Exception as e:
            logger.error(f"Błąd podczas wyświetlania raportu: {str(e)}")
            self.details_text.setPlainText(
                f"Błąd podczas wyświetlania raportu: {str(e)}\n\n--- Surowy raport ---\n{report}"
            )
            self.analyze_button.setEnabled(True)
            self.progress_bar.setVisible(False)

    def _on_analysis_error(self, error_msg):
        logger.error(f"Otrzymano sygnał błędu analizy: {error_msg}")
        self.details_text.setPlainText(f"Błąd podczas analizy:\n{error_msg}")
        self.analyze_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("Błąd analizy.", 5000)
        self.show_message("Błąd Analizy", error_msg, QMessageBox.Icon.Critical)


    def _on_progress_update(self, message):
        logger.debug(f"Otrzymano aktualizację postępu: {message}")
        self.statusBar().showMessage(message)
        # Można też aktualizować details_text, ale może to być zbyt częste
        # self.details_text.append(message) # Jeśli chcemy log na żywo w panelu

    def _get_model_structure(self, model_obj): # Przyjmuje model jako argument
        """Pobiera strukturę modelu (nn.Module lub dict) jako słownik."""
        structure = {}

        # Użyj licznika, aby uniknąć zbyt głębokiej rekursji lub zbyt dużych struktur
        # Można dodać max_depth
        MAX_ITEMS_PER_LEVEL = 100

        def add_to_structure_recursive(obj, current_path_list, current_dict_level, depth=0, max_depth=10):
            if depth > max_depth:
                current_dict_level["_error_"] = "Przekroczono maksymalną głębokość rekursji"
                return

            path_str = ".".join(current_path_list)

            if isinstance(obj, torch.nn.Module):
                children_dict = {}
                current_dict_level[obj.__class__.__name__] = { # Użyj nazwy klasy jako klucza dla modułu
                    "type": obj.__class__.__name__,
                    "parameters": sum(p.numel() for p in obj.parameters(recurse=False) if p.requires_grad), # Tylko własne parametry
                    "children": children_dict
                }
                # Ogranicz liczbę dzieci do przetworzenia
                child_count = 0
                for name, child_module in obj.named_children():
                    if child_count >= MAX_ITEMS_PER_LEVEL:
                        children_dict["... (więcej dzieci)"] = {}
                        break
                    add_to_structure_recursive(child_module, current_path_list + [name], children_dict, depth + 1, max_depth)
                    child_count += 1

            elif isinstance(obj, (dict, collections.OrderedDict)):
                # Jeśli sama struktura to dict (np. state_dict), klucze są ważne
                # Dla uproszczenia, traktujemy klucze jako "nazwy"
                items_dict = {}
                current_dict_level["items"] = items_dict # Użyj "items" jako kontenera dla elementów dict

                item_count = 0
                for key, value in obj.items():
                    if item_count >= MAX_ITEMS_PER_LEVEL:
                        items_dict["... (więcej elementów)"] = {}
                        break
                    # Tworzymy nowy poziom słownika dla każdego klucza
                    key_node = {}
                    items_dict[key] = key_node
                    add_to_structure_recursive(value, current_path_list + [key], key_node, depth + 1, max_depth)
                    item_count += 1

            elif isinstance(obj, torch.Tensor):
                current_dict_level["_tensor_details_"] = { # Specjalny klucz dla liścia-tensora
                    "type": "tensor",
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "numel": obj.numel()
                }
            elif isinstance(obj, (list, tuple)):
                list_items = {}
                current_dict_level["_list_items_"] = list_items
                item_count = 0
                for i, item_val in enumerate(obj):
                    if item_count >= MAX_ITEMS_PER_LEVEL:
                        list_items[f"... (więcej elementów list/tuple)"] = {}
                        break
                    item_node = {}
                    list_items[f"[{i}]"] = item_node # Użyj indeksu jako klucza
                    add_to_structure_recursive(item_val, current_path_list + [f"[{i}]"], item_node, depth + 1, max_depth)
                    item_count += 1
            else:
                 # Dla innych typów, po prostu przechowaj ich reprezentację string
                try:
                    str_val = str(obj)
                    if len(str_val) > 100: # Ogranicz długość stringa
                        str_val = str_val[:100] + "..."
                    current_dict_level["_value_"] = {
                        "type": type(obj).__name__,
                        "value": str_val
                    }
                except Exception:
                     current_dict_level["_value_"] = {
                        "type": type(obj).__name__,
                        "value": "(nie można przekonwertować na string)"
                    }


        root_name = "model_structure"
        if isinstance(model_obj, torch.nn.Module):
            root_name = model_obj.__class__.__name__ # lub "model"
        elif isinstance(model_obj, dict):
            root_name = "state_dict"

        # Struktura zaczyna się od korzenia
        root_structure_node = {}
        structure[root_name] = root_structure_node
        add_to_structure_recursive(model_obj, [root_name], root_structure_node)

        return structure


    def _get_parameter_by_path(self, path_parts):
        if not path_parts:
            return None

        # Pierwszy element ścieżki to zwykle "Model" lub "State Dict" - korzeń drzewa
        # Rzeczywista nawigacja zaczyna się od self.model
        current_obj = self.model

        # Pomijamy pierwszy element path_parts, jeśli to "Model" lub "State Dict"
        # które są nazwami korzeni drzewa, a nie rzeczywistymi atrybutami/kluczami modelu
        start_index = 0
        if path_parts[0] in ["Model", "State Dict"]:
            start_index = 1

        for name_part in path_parts[start_index:]:
            name_clean = name_part # Już powinno być czyste z copy_item_path lub show_details

            if isinstance(current_obj, torch.nn.Module):
                # Spróbuj jako atrybut (podmoduł)
                if hasattr(current_obj, name_clean):
                    current_obj = getattr(current_obj, name_clean)
                else:
                    # Spróbuj jako parametr
                    try:
                        # named_parameters() zwraca generator, musimy znaleźć odpowiedni
                        found_param = False
                        for p_name, param_val in current_obj.named_parameters(recurse=False): # tylko bezpośrednie
                            if p_name == name_clean:
                                current_obj = param_val
                                found_param = True
                                break
                        if not found_param:
                            # Spróbuj jako bufor
                            for b_name, buffer_val in current_obj.named_buffers(recurse=False):
                                if b_name == name_clean:
                                    current_obj = buffer_val
                                    found_param = True
                                    break
                        if not found_param:
                             raise ValueError(f"Nie znaleziono atrybutu, parametru ani bufora '{name_clean}' w module {type(current_obj).__name__}")
                    except AttributeError: # Jeśli .get_parameter nie istnieje lub inny błąd
                        raise ValueError(f"Nie można uzyskać dostępu do '{name_clean}' w module {type(current_obj).__name__}")

            elif isinstance(current_obj, (dict, collections.OrderedDict)):
                if name_clean in current_obj:
                    current_obj = current_obj[name_clean]
                else:
                    raise ValueError(f"Nie znaleziono klucza '{name_clean}' w słowniku")

            elif isinstance(current_obj, (list, tuple)):
                try:
                    # Oczekujemy, że name_clean to indeks w formacie "[i]"
                    if name_clean.startswith("[") and name_clean.endswith("]"):
                        idx = int(name_clean[1:-1])
                        if 0 <= idx < len(current_obj):
                            current_obj = current_obj[idx]
                        else:
                            raise ValueError(f"Indeks '{idx}' poza zakresem dla listy/tupli o długości {len(current_obj)}")
                    else:
                         raise ValueError(f"Element '{name_clean}' nie jest prawidłowym indeksem listy/tupli")
                except (ValueError, IndexError) as e:
                    raise ValueError(f"Błąd dostępu do elementu listy/tupli '{name_clean}': {e}")
            else:
                # Jeśli dotarliśmy do liścia, który nie jest kontenerem, a ścieżka jest dłuższa
                raise ValueError(f"Obiekt typu {type(current_obj).__name__} nie jest kontenerem, ale ścieżka prowadzi dalej przez '{name_clean}'")

        return current_obj


    def populate_tree(self):
        self.tree.clear()
        if not self.model:
            return

        if isinstance(self.model, torch.nn.Module):
            root_item = QTreeWidgetItem(self.tree, [f"Model ({self.model.__class__.__name__})"])
            self._add_to_tree_recursive(root_item, self.model)
        elif isinstance(self.model, (dict, collections.OrderedDict)):
            root_item = QTreeWidgetItem(self.tree, ["State Dict"])
            self._add_to_tree_recursive(root_item, self.model)
        elif isinstance(self.model, (list, tuple)):
            root_item = QTreeWidgetItem(self.tree, ["Lista/Tupla"])
            self._add_to_tree_recursive(root_item, self.model)
        else:
            # Dla pojedynczych tensorów lub innych typów
            root_item = QTreeWidgetItem(self.tree, [f"Obiekt ({type(self.model).__name__})"])
            self._add_to_tree_recursive(root_item, self.model, "value")


        if self.tree.topLevelItemCount() > 0:
            self.tree.topLevelItem(0).setExpanded(True)

    def _add_to_tree_recursive(self, parent_item, obj, name_prefix=""):
        MAX_CHILDREN_DISPLAY = 200 # Ograniczenie dla wydajności
        child_count = 0

        if isinstance(obj, torch.nn.Module):
            # Moduły: pokaż nazwę, typ, liczbę własnych parametrów
            # Najpierw parametry i bufory samego modułu
            for param_name, param in obj.named_parameters(recurse=False): # Tylko bezpośrednie
                if child_count >= MAX_CHILDREN_DISPLAY: break
                text = (f"{param_name} (Param): shape={list(param.shape)}, "
                        f"dtype={param.dtype}, grad={param.requires_grad}")
                QTreeWidgetItem(parent_item, [text])
                child_count +=1
            for buffer_name, buffer_val in obj.named_buffers(recurse=False):
                if child_count >= MAX_CHILDREN_DISPLAY: break
                text = (f"{buffer_name} (Buffer): shape={list(buffer_val.shape)}, "
                        f"dtype={buffer_val.dtype}")
                QTreeWidgetItem(parent_item, [text])
                child_count +=1

            # Potem dzieci modułu
            for child_name, child_module in obj.named_children():
                if child_count >= MAX_CHILDREN_DISPLAY: break
                num_params = sum(p.numel() for p in child_module.parameters(recurse=False) if p.requires_grad)
                item_text = f"{child_name} ({child_module.__class__.__name__}, params: {num_params})"
                child_tree_item = QTreeWidgetItem(parent_item, [item_text])
                self._add_to_tree_recursive(child_tree_item, child_module)
                child_count +=1

        elif isinstance(obj, (dict, collections.OrderedDict)):
            for key, value in obj.items():
                if child_count >= MAX_CHILDREN_DISPLAY: break
                item_text = f"{key} (klucz)"
                child_tree_item = QTreeWidgetItem(parent_item, [item_text])
                self._add_to_tree_recursive(child_tree_item, value, key) # Przekaż klucz jako name_prefix dla tensorów
                child_count +=1

        elif isinstance(obj, (list, tuple)):
            for i, item_val in enumerate(obj):
                if child_count >= MAX_CHILDREN_DISPLAY: break
                item_text = f"[{i}] (element listy/tupli)"
                child_tree_item = QTreeWidgetItem(parent_item, [item_text])
                self._add_to_tree_recursive(child_tree_item, item_val, f"{name_prefix}[{i}]")
                child_count +=1

        elif isinstance(obj, torch.Tensor):
            # Dla tensora bezpośrednio pod rodzicem (np. w state_dict lub jako parametr)
            # Nazwa powinna być już w `parent_item` lub `name_prefix`
            # Tutaj dodajemy szczegóły tensora jako liść
            mean_val_str = "N/A"
            if obj.numel() > 0 and obj.is_floating_point(): # mean tylko dla float i niepustych
                try:
                    mean_val_str = f"{obj.float().mean().item():.4f}"
                except RuntimeError: # np. dla typów nie float
                     mean_val_str = "N/A (not float)"

            tensor_details = (f"Tensor: shape={list(obj.shape)}, dtype={obj.dtype}, "
                              f"numel={obj.numel()}, mean={mean_val_str}")
            # Zamiast tworzyć nowy element, można dodać to jako drugi "kolumnę" (opis) do parent_item
            # lub jeśli parent_item to np. klucz, to ten opis jest dla wartości tego klucza.
            # Jeśli parent_item reprezentuje tensor (np. z named_parameters), to nie rób nic.
            # To jest skomplikowane, bo struktura drzewa i modelu muszą być spójne.
            # W obecnej logice, jeśli `obj` jest tensorem, to jest to liść.
            # `parent_item` to jego "kontener" (moduł, klucz dict).
            # Trzeba zmienić tekst `parent_item` lub dodać nowy element.
            # Załóżmy, że `populate_tree` tworzy elementy dla kontenerów,
            # a `_add_to_tree_recursive` dodaje zawartość.
            # Jeśli `name_prefix` istnieje (czyli jesteśmy wartością w słowniku), to parent_item to klucz.
            # Dodajmy nowy element dla samego tensora.

            # Jeśli name_prefix jest pusty, to znaczy, że tensor jest bezpośrednio pod modułem (param/buffer)
            # lub jest to główny obiekt modelu.
            # W tym przypadku, szczegóły tensora powinny być w parent_item.
            # Dla uproszczenia, zawsze dodawajmy nowy element, jeśli jest to tensor-liść.
            # A `name_prefix` to nazwa tego tensora.

            # Ta funkcja jest wołana z `child_module` lub `value` (z dict) lub `item_val` (z list)
            # Jeśli to tensor, to parent_item reprezentuje kontener. Dodajemy tensor jako dziecko.
            leaf_text = f"{name_prefix if name_prefix else 'Tensor'}: {list(obj.shape)}, {obj.dtype}"
            # QTreeWidgetItem(parent_item, [leaf_text]) # Zamiast tego, pozwól show_details pokazać info

        else: # Inne typy atomowe
            try:
                str_val = str(obj)
                if len(str_val) > 50: str_val = str_val[:50] + "..."
                leaf_text = f"{name_prefix if name_prefix else 'Wartość'}: {str_val} ({type(obj).__name__})"
                QTreeWidgetItem(parent_item, [leaf_text])
            except:
                QTreeWidgetItem(parent_item, [f"{name_prefix if name_prefix else 'Wartość'}: (błąd konwersji) ({type(obj).__name__})"])

        if child_count >= MAX_CHILDREN_DISPLAY:
            QTreeWidgetItem(parent_item, ["... (więcej elementów, ograniczono wyświetlanie)"])


    def show_details(self, item, column):
        path_parts = []
        current = item
        while current is not None:
            raw_text = current.text(0)
            clean_name = raw_text.split(" (")[0].split(":")[0].strip()
            path_parts.insert(0, clean_name)
            current = current.parent()

        path_key = " -> ".join(path_parts)
        logger.debug(f"Wyświetlanie szczegółów dla: {path_key}")

        # Cache
        if hasattr(self, "_details_cache") and path_key in self._details_cache:
            self.details_text.setHtml(self._details_cache[path_key]) # Zakładamy, że cache przechowuje HTML
            return

        try:
            obj_to_display = self._get_parameter_by_path(path_parts)
            details_str = self._format_object_details(obj_to_display, path_parts)

        except ValueError as e:
            details_str = f"Błąd pobierania szczegółów dla '{path_key}':\n{e}"
            logger.warning(details_str)
        except Exception as e:
            import traceback
            details_str = (f"Nieoczekiwany błąd podczas pobierania szczegółów dla '{path_key}':\n{e}\n"
                           f"{traceback.format_exc()}")
            logger.error(details_str)

        # Formatowanie HTML
        html_details = details_str.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html_details = html_details.replace("\n", "<br>")
        styled_html_details = (f"<div style='font-family: \"DejaVu Sans Mono\", monospace; "
                               f"white-space: pre-wrap; color: {self.text_color};'>{html_details}</div>")

        self.details_text.setHtml(styled_html_details)

        if not hasattr(self, "_details_cache"):
            self._details_cache = {}
        self._details_cache[path_key] = styled_html_details # Cache'uj sformatowany HTML

    def _format_object_details(self, obj, path_parts):
        """Formatuje szczegóły obiektu do wyświetlenia."""
        header = f"Szczegóły dla: {' -> '.join(path_parts)}\n"
        header += f"Typ obiektu: {type(obj).__name__}\n"

        content = ""
        if isinstance(obj, torch.nn.Module):
            own_params = sum(p.numel() for p in obj.parameters(recurse=False) if p.requires_grad)
            total_params_children = sum(p.numel() for p in obj.parameters(recurse=True) if p.requires_grad) - own_params
            content += f"Moduł: {obj.__class__.__name__}\n"
            content += f"  Bezpośrednie parametry (uczone): {own_params:,}\n"
            content += f"  Parametry w podmodułach (uczone): {total_params_children:,}\n"
            content += f"  Łącznie parametry (uczone): {own_params + total_params_children:,}\n"
            content += "  Reprezentacja modułu:\n"
            try:
                module_str = str(obj)
                # Ogranicz długość, jeśli jest zbyt duża
                if len(module_str) > 2000:
                    module_str = module_str[:2000] + "\n... (reprezentacja skrócona)"
                content += f"{module_str}\n"
            except Exception as e:
                content += f"  (Błąd podczas generowania reprezentacji modułu: {e})\n"

        elif isinstance(obj, torch.Tensor):
            content += f"Tensor:\n"
            content += f"  Kształt: {list(obj.shape)}\n"
            content += f"  Typ danych: {obj.dtype}\n"
            content += f"  Liczba elementów: {obj.numel():,}\n"
            content += f"  Urządzenie: {obj.device}\n"
            content += f"  Wymaga gradientu: {obj.requires_grad}\n"
            if obj.numel() > 0:
                # Ogranicz liczbę wyświetlanych wartości
                MAX_VALUES_TO_SHOW = 100

                try:
                    # Statystyki tylko dla tensorów zmiennoprzecinkowych
                    if obj.is_floating_point():
                        content += f"  Średnia: {obj.float().mean().item():.6f}\n"
                        content += f"  Odch. std.: {obj.float().std().item():.6f}\n"
                        content += f"  Min: {obj.float().min().item():.6f}\n"
                        content += f"  Max: {obj.float().max().item():.6f}\n"
                    else: # Dla int, bool
                        # Można dodać np. sumę, lub unikalne wartości jeśli jest ich mało
                        unique_vals, counts = torch.unique(obj, return_counts=True)
                        if unique_vals.numel() < 20: # Pokaż unikalne jeśli jest ich mało
                            content += "  Unikalne wartości (liczność):\n"
                            for uv, c in zip(unique_vals, counts):
                                content += f"    {uv.item()}: {c.item()}\n"
                        else:
                             content += "  (Tensor nie jest zmiennoprzecinkowy - statystyki jak mean/std nie są liczone)\n"


                    content += f"  Pierwsze {min(obj.numel(), MAX_VALUES_TO_SHOW)} wartości (spłaszczone):\n"
                    flat_tensor = obj.flatten()
                    for i in range(min(flat_tensor.numel(), MAX_VALUES_TO_SHOW)):
                        content += f"    {flat_tensor[i].item():.4f}" if flat_tensor[i].is_floating_point() else f"    {flat_tensor[i].item()}"
                        if (i + 1) % 10 == 0: content += "\n" # Nowa linia co 10 wartości
                        else: content += "  "
                    if flat_tensor.numel() > MAX_VALUES_TO_SHOW:
                        content += "\n    ... (więcej wartości)"
                    content += "\n"

                except Exception as e: # np. .item() na tensorze z wieloma elementami bez spłaszczenia
                    content += f"  (Błąd podczas obliczania statystyk/wartości: {e})\n"
            else:
                content += "  (Tensor jest pusty)\n"

        elif isinstance(obj, (dict, collections.OrderedDict)):
            content += f"Słownik ({len(obj)} elementów):\n"
            item_count = 0
            MAX_DICT_ITEMS_TO_SHOW = 50
            for k, v in obj.items():
                if item_count >= MAX_DICT_ITEMS_TO_SHOW:
                    content += "  ... (więcej elementów)\n"
                    break
                content += f"  Klucz: '{k}', Typ wartości: {type(v).__name__}\n"
                item_count += 1

        elif isinstance(obj, (list, tuple)):
            content += f"Lista/Tupla ({len(obj)} elementów):\n"
            item_count = 0
            MAX_LIST_ITEMS_TO_SHOW = 50
            for i, v_item in enumerate(obj):
                if item_count >= MAX_LIST_ITEMS_TO_SHOW:
                    content += "  ... (więcej elementów)\n"
                    break
                content += f"  Indeks: [{i}], Typ wartości: {type(v_item).__name__}\n"
                item_count += 1
        else:
            # Inne typy
            try:
                str_repr = str(obj)
                if len(str_repr) > 1000:
                    str_repr = str_repr[:1000] + "... (reprezentacja skrócona)"
                content += f"Wartość:\n  {str_repr}\n"
            except Exception as e:
                 content += f"(Błąd podczas konwersji obiektu na string: {e})\n"

        return header + content


    def export_to_onnx(self):
        if not self.model:
            self.show_message("Błąd", "Brak załadowanego modelu.", QMessageBox.Icon.Warning)
            return
        if not isinstance(self.model, torch.nn.Module):
            logger.warning("Próba eksportu do ONNX nieprawidłowego typu modelu")
            self.show_message(
                "Błąd",
                "Eksport do ONNX wymaga modelu typu torch.nn.Module.\n"
                f"Aktualny typ to: {type(self.model).__name__}",
                QMessageBox.Icon.Warning,
            )
            return

        default_filename = "model.onnx"
        if self.current_model_path:
            base, _ = os.path.splitext(os.path.basename(self.current_model_path))
            default_filename = f"{base}.onnx"

        file_name, _ = QFileDialog.getSaveFileName(
            self, "Eksportuj do ONNX",
            os.path.join(self.models_dir, default_filename),
            "ONNX Files (*.onnx);;All Files (*)"
        )

        if file_name:
            try:
                # Próba odgadnięcia kształtu wejścia (bardzo podstawowa)
                input_shape = None
                # Poszukaj pierwszej warstwy Conv2d lub Linear jako wskazówki
                for m_child in self.model.modules(): # Iteruj po wszystkich modułach, nie tylko dzieciach
                    if isinstance(m_child, torch.nn.Conv2d):
                        # (batch_size, in_channels, H, W)
                        # Domyślny H, W dla obrazów, można to uczynić konfigurowalnym
                        input_shape = (1, m_child.in_channels, 224, 224)
                        logger.info(f"Wykryto Conv2d, sugerowany kształt wejścia: {input_shape}")
                        break
                    elif isinstance(m_child, torch.nn.Linear):
                        # (batch_size, in_features)
                        input_shape = (1, m_child.in_features)
                        logger.info(f"Wykryto Linear, sugerowany kształt wejścia: {input_shape}")
                        break

                if input_shape is None:
                    # Jeśli nie znaleziono, poproś użytkownika lub użyj generycznego
                    shape_str, ok = QInputDialog.getText(self, "Kształt wejścia dla ONNX",
                                                         "Podaj kształt wejścia (np. 1,3,224,224):",
                                                         text="1,3,224,224")
                    if ok and shape_str:
                        try:
                            input_shape = tuple(map(int, shape_str.split(',')))
                        except ValueError:
                            self.show_message("Błąd", "Nieprawidłowy format kształtu wejścia.", QMessageBox.Icon.Warning)
                            return
                    else: # Anulowano lub pusty
                        logger.info("Używam domyślnego kształtu wejścia (1,3,224,224) dla ONNX.")
                        input_shape = (1, 3, 224, 224) # Domyślny, często używany

                logger.info(f"Używam kształtu wejścia {input_shape} dla eksportu ONNX.")
                dummy_input = torch.randn(*input_shape, device='cpu') # Upewnij się, że na CPU, jeśli model tam jest

                # Ustaw model w tryb ewaluacji
                self.model.eval()

                torch.onnx.export(
                    self.model,
                    dummy_input,
                    file_name,
                    export_params=True,
                    opset_version=11, # Można zwiększyć, jeśli potrzeba nowszych operatorów
                    do_constant_folding=True,
                    input_names=["input"],   # Można skonfigurować
                    output_names=["output"], # Można skonfigurować
                    dynamic_axes={ # Umożliwia zmienny batch_size
                        "input": {0: "batch_size"},
                        "output": {0: "batch_size"},
                    },
                )
                logger.info(f"Model został wyeksportowany do ONNX: {file_name}")
                self.show_message(
                    "Sukces", f"Model został wyeksportowany do formatu ONNX:\n{file_name}"
                )
            except Exception as e:
                import traceback
                err_msg = f"Błąd podczas eksportu do ONNX: {str(e)}\n{traceback.format_exc()}"
                logger.error(err_msg)
                self.show_message(
                    "Błąd",
                    err_msg,
                    QMessageBox.Icon.Critical,
                )

    def export_to_torchscript(self):
        if not self.model:
            self.show_message("Błąd", "Brak załadowanego modelu.", QMessageBox.Icon.Warning)
            return
        if not isinstance(self.model, torch.nn.Module):
            logger.warning("Próba eksportu do TorchScript nieprawidłowego typu modelu")
            self.show_message(
                "Błąd",
                "Eksport do TorchScript wymaga modelu typu torch.nn.Module.\n"
                f"Aktualny typ to: {type(self.model).__name__}",
                QMessageBox.Icon.Warning,
            )
            return

        default_filename = "model_script.pt"
        if self.current_model_path:
            base, _ = os.path.splitext(os.path.basename(self.current_model_path))
            default_filename = f"{base}_script.pt"

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Eksportuj do TorchScript",
            os.path.join(self.models_dir, default_filename),
            "TorchScript Files (*.pt);;All Files (*)",
        )

        if file_name:
            try:
                # Ustaw model w tryb ewaluacji
                self.model.eval()

                # Dla TorchScript `trace` potrzebujemy przykładowego wejścia
                # Podobnie jak w ONNX, spróbujmy odgadnąć lub zapytać
                input_shape = None
                for m_child in self.model.modules():
                    if isinstance(m_child, torch.nn.Conv2d):
                        input_shape = (1, m_child.in_channels, 224, 224)
                        break
                    elif isinstance(m_child, torch.nn.Linear):
                        input_shape = (1, m_child.in_features)
                        break

                if input_shape is None:
                    shape_str, ok = QInputDialog.getText(self, "Kształt wejścia dla TorchScript",
                                                         "Podaj kształt wejścia dla śledzenia (np. 1,3,224,224):",
                                                         text="1,3,224,224")
                    if ok and shape_str:
                        try:
                            input_shape = tuple(map(int, shape_str.split(',')))
                        except ValueError:
                            self.show_message("Błąd", "Nieprawidłowy format kształtu wejścia.", QMessageBox.Icon.Warning)
                            return
                    else:
                        input_shape = (1, 3, 224, 224)

                dummy_input = torch.randn(*input_shape, device='cpu')

                # Wybór między trace a script
                method, ok = QInputDialog.getItem(self, "Metoda TorchScript", "Wybierz metodę:", ["trace", "script (eksperymentalne)"], 0, False)
                if not ok: return

                if method == "trace":
                    traced_model = torch.jit.trace(self.model, dummy_input)
                    traced_model.save(file_name)
                else: # script
                    try:
                        scripted_model = torch.jit.script(self.model)
                        scripted_model.save(file_name)
                    except Exception as e_script:
                        logger.error(f"Błąd podczas torch.jit.script: {e_script}")
                        self.show_message("Błąd skryptowania",
                                          f"Nie udało się przekonwertować modelu za pomocą torch.jit.script:\n{e_script}\n"
                                          "Spróbuj metody 'trace' lub upewnij się, że model jest kompatybilny z jit.script.",
                                          QMessageBox.Icon.Warning)
                        return


                logger.info(f"Model został wyeksportowany do TorchScript ({method}): {file_name}")
                self.show_message(
                    "Sukces", f"Model został wyeksportowany do formatu TorchScript ({method}) jako:\n{file_name}"
                )
            except Exception as e:
                import traceback
                err_msg = f"Błąd podczas eksportu do TorchScript: {str(e)}\n{traceback.format_exc()}"
                logger.error(err_msg)
                self.show_message(
                    "Błąd",
                    err_msg,
                    QMessageBox.Icon.Critical,
                )

    def visualize_parameters(self):
        logger.info("Wizualizacja parametrów modelu")
        if not self.model:
            logger.warning("Brak modelu do wizualizacji parametrów")
            self.show_message(
                "Błąd", "Brak modelu do wizualizacji. Załaduj model.", QMessageBox.Icon.Warning
            )
            return

        try:
            dialog = QDialog(self)
            dialog.setWindowTitle("Wizualizacja parametrów")
            # Dostosuj rozmiar okna dialogowego
            dialog.setMinimumSize(800, 600) # Minimalny rozmiar
            dialog.resize(1000, 700) # Domyślny rozmiar przy otwarciu

            layout = QVBoxLayout(dialog)
            tabs = QTabWidget()
            layout.addWidget(tabs) # Dodaj zakładki do layoutu głównego

            # Opcje wizualizacji
            options_layout = QHBoxLayout()
            log_checkbox = QCheckBox("Skala logarytmiczna (histogramy)")
            options_layout.addWidget(log_checkbox)

            outlier_percentile_label = QLabel("Zakres percentyli (usuń outliery):")
            options_layout.addWidget(outlier_percentile_label)

            lower_percentile_input = QLineEdit("1")
            lower_percentile_input.setFixedWidth(40)
            options_layout.addWidget(lower_percentile_input)

            upper_percentile_input = QLineEdit("99")
            upper_percentile_input.setFixedWidth(40)
            options_layout.addWidget(upper_percentile_input)

            max_params_label = QLabel("Max parametrów na warstwę/model (0=wszystkie):")
            options_layout.addWidget(max_params_label)
            max_params_input = QLineEdit("100000") # Domyślnie 100k
            max_params_input.setFixedWidth(80)
            options_layout.addWidget(max_params_input)

            redraw_button = QPushButton("Odśwież wykresy")
            options_layout.addWidget(redraw_button)
            options_layout.addStretch()

            layout.addLayout(options_layout) # Dodaj opcje pod zakładkami (lub nad)

            # Placeholder dla canvasów, które będą tworzone dynamicznie
            # To jest ważne, aby figury i canvasy były tworzone w funkcji rysującej,
            # aby poprawnie reagowały na zmiany opcji

            # --- Struktura danych dla wykresów ---
            # Będziemy je przechowywać, aby móc je odświeżać
            self.viz_data = {
                "all_params_hist_canvas": None, "all_params_hist_fig": None, "all_params_hist_ax": None,
                "outliers_hist_canvas": None, "outliers_hist_fig": None, "outliers_hist_ax": None,
                "dist_canvas": None, "dist_fig": None, "dist_ax": None,
                "layers_bar_canvas": None, "layers_bar_fig": None, "layers_bar_ax": None,
                "heatmap_canvas": None, "heatmap_fig": None, "heatmap_ax": None, # Dla heatmapy wag
            }

            # --- Zakładki ---
            # 1. Histogram wszystkich parametrów
            all_params_hist_tab = QWidget()
            all_params_hist_layout = QVBoxLayout(all_params_hist_tab)
            tabs.addTab(all_params_hist_tab, "Histogram (Wszystkie)")

            # 2. Histogram outlierów
            outliers_hist_tab = QWidget()
            outliers_hist_layout = QVBoxLayout(outliers_hist_tab)
            tabs.addTab(outliers_hist_tab, "Histogram (Outliery)")

            # 3. Rozkład wartości (CDF)
            dist_tab = QWidget()
            dist_layout = QVBoxLayout(dist_tab)
            tabs.addTab(dist_tab, "Rozkład (CDF)")

            # 4. Parametry per warstwa (Box plot lub Bar)
            layers_bar_tab = QWidget()
            layers_bar_layout = QVBoxLayout(layers_bar_tab)
            tabs.addTab(layers_bar_tab, "Parametry Warstw")

            # 5. Heatmapa wag (dla wybranych warstw) - opcjonalnie
            heatmap_tab = QWidget()
            heatmap_layout = QVBoxLayout(heatmap_tab)
            # Można dodać QComboBox do wyboru warstwy dla heatmapy
            self.heatmap_layer_selector = QComboBox()
            heatmap_layout.addWidget(QLabel("Wybierz warstwę dla heatmapy wag (tylko z wagami 2D):"))
            heatmap_layout.addWidget(self.heatmap_layer_selector)
            tabs.addTab(heatmap_tab, "Heatmapa Wag")

            # --- Etykieta informacyjna ---
            self.stats_info_label = QLabel("Statystyki pojawią się po odświeżeniu.")
            self.stats_info_label.setWordWrap(True)
            layout.addWidget(self.stats_info_label)


            # --- Funkcja rysująca/odświeżająca ---
            def redraw_visualizations():
                try:
                    log_scale = log_checkbox.isChecked()
                    p_lower = float(lower_percentile_input.text())
                    p_upper = float(upper_percentile_input.text())
                    max_p_vis = int(max_params_input.text())
                    if max_p_vis == 0: max_p_vis = None # None oznacza wszystkie

                    if not (0 <= p_lower < p_upper <= 100):
                        self.show_message("Błąd", "Nieprawidłowy zakres percentyli.", QMessageBox.Icon.Warning)
                        return

                    # Zbieranie parametrów
                    all_param_values = []
                    layer_param_values = collections.defaultdict(list)
                    param_names_for_heatmap = [] # Dla selektora heatmapy

                    def collect_params_recursive(module_or_dict, prefix=""):
                        if isinstance(module_or_dict, torch.nn.Module):
                            for name, param in module_or_dict.named_parameters(recurse=False):
                                if param.requires_grad and param.numel() > 0:
                                    full_name = f"{prefix}.{name}" if prefix else name
                                    vals = param.detach().cpu().numpy().flatten()
                                    if max_p_vis and len(vals) > max_p_vis: # Próbkowanie
                                        vals = np.random.choice(vals, max_p_vis, replace=False)
                                    all_param_values.extend(vals)
                                    # Grupowanie po nazwie modułu najwyższego poziomu
                                    layer_key = prefix.split('.')[0] if '.' in prefix else prefix if prefix else "model"
                                    if not layer_key and isinstance(self.model, torch.nn.Module): # Jeśli model jest korzeniem
                                        layer_key = name.split('.')[0] # Użyj pierwszej części nazwy parametru

                                    layer_param_values[layer_key].extend(vals)
                                    if param.ndim == 2 and param.shape[0] > 1 and param.shape[1] > 1: # Heurystyka dla sensownej heatmapy
                                        param_names_for_heatmap.append(full_name)


                            for name, child_module in module_or_dict.named_children():
                                collect_params_recursive(child_module, f"{prefix}.{name}" if prefix else name)

                        elif isinstance(module_or_dict, (dict, collections.OrderedDict)):
                            for key, value in module_or_dict.items():
                                current_path = f"{prefix}.{key}" if prefix else key
                                if isinstance(value, torch.Tensor) and value.numel() > 0:
                                    vals = value.detach().cpu().numpy().flatten()
                                    if max_p_vis and len(vals) > max_p_vis:
                                        vals = np.random.choice(vals, max_p_vis, replace=False)
                                    all_param_values.extend(vals)
                                    layer_key = current_path.split('.')[0] # Pierwsza część klucza jako nazwa warstwy
                                    layer_param_values[layer_key].extend(vals)
                                    if value.ndim == 2 and value.shape[0] > 1 and value.shape[1] > 1:
                                        param_names_for_heatmap.append(current_path)

                                elif isinstance(value, (dict, collections.OrderedDict, torch.nn.Module)): # Rekurencja dla zagnieżdżonych
                                    collect_params_recursive(value, current_path)

                    collect_params_recursive(self.model)

                    # Aktualizuj selektor heatmapy
                    current_heatmap_selection = self.heatmap_layer_selector.currentText()
                    self.heatmap_layer_selector.clear()
                    self.heatmap_layer_selector.addItems(sorted(list(set(param_names_for_heatmap))))
                    if current_heatmap_selection in param_names_for_heatmap:
                        self.heatmap_layer_selector.setCurrentText(current_heatmap_selection)


                    if not all_param_values:
                        self.stats_info_label.setText("Brak parametrów do wizualizacji.")
                        return

                    all_param_values_np = np.array(all_param_values, dtype=np.float32) # Użyj float32 dla oszczędności pamięci
                    if max_p_vis and len(all_param_values_np) > max_p_vis * 5: # Próbkuj całość jeśli zbyt duża
                        all_param_values_np = np.random.choice(all_param_values_np, max_p_vis * 5, replace=False)


                    val_min, val_p_lower, val_p_upper, val_max = (
                        np.min(all_param_values_np),
                        np.percentile(all_param_values_np, p_lower),
                        np.percentile(all_param_values_np, p_upper),
                        np.max(all_param_values_np)
                    )

                    params_in_range = all_param_values_np[(all_param_values_np >= val_p_lower) & (all_param_values_np <= val_p_upper)]
                    params_outliers = all_param_values_np[(all_param_values_np < val_p_lower) | (all_param_values_np > val_p_upper)]

                    stats_summary = (
                        f"Łącznie parametrów: {len(all_param_values_np):,}. Min: {val_min:.4f}, Max: {val_max:.4f}.\n"
                        f"Zakres ({p_lower}%-{p_upper}%): [{val_p_lower:.4f}, {val_p_upper:.4f}]. "
                        f"W zakresie: {len(params_in_range):,} ({len(params_in_range)/len(all_param_values_np)*100:.1f}%).\n"
                        f"Outliery: {len(params_outliers):,} ({len(params_outliers)/len(all_param_values_np)*100:.1f}%)."
                    )
                    self.stats_info_label.setText(stats_summary)

                    # --- Rysowanie ---
                    # Helper do (re)tworzenia canvasu
                    def setup_canvas(tab_layout, viz_key_prefix):
                        # Usuń stary canvas, jeśli istnieje
                        if self.viz_data[f"{viz_key_prefix}_canvas"]:
                            self.viz_data[f"{viz_key_prefix}_canvas"].deleteLater()

                        fig = Figure(figsize=(7, 5), dpi=100) # Mniejsze figury dla zakładek
                        canvas = FigureCanvas(fig)
                        ax = fig.add_subplot(111)
                        tab_layout.addWidget(canvas)

                        self.viz_data[f"{viz_key_prefix}_canvas"] = canvas
                        self.viz_data[f"{viz_key_prefix}_fig"] = fig
                        self.viz_data[f"{viz_key_prefix}_ax"] = ax
                        return fig, canvas, ax

                    # 1. Histogram (w zakresie)
                    fig_hist, _, ax_hist = setup_canvas(all_params_hist_layout, "all_params_hist")
                    if len(params_in_range) > 0:
                        ax_hist.hist(params_in_range, bins=50, density=True, alpha=0.75, log=log_scale, color=self.primary_color)
                        ax_hist.set_title(f"Histogram parametrów ({p_lower}-{p_upper} percentyl)", fontsize=10)
                        ax_hist.set_xlabel("Wartość parametru", fontsize=8)
                        ax_hist.set_ylabel("Gęstość", fontsize=8)
                        fig_hist.tight_layout()
                        self.viz_data["all_params_hist_canvas"].draw()
                    else: ax_hist.text(0.5, 0.5, "Brak danych w zakresie", ha='center', va='center')

                    # 2. Histogram (outliery)
                    fig_out, _, ax_out = setup_canvas(outliers_hist_layout, "outliers_hist")
                    if len(params_outliers) > 0:
                        ax_out.hist(params_outliers, bins=50, density=True, alpha=0.75, log=log_scale, color=self.warning_color)
                        ax_out.set_title(f"Histogram outlierów (poza {p_lower}-{p_upper}%)", fontsize=10)
                        ax_out.set_xlabel("Wartość parametru", fontsize=8)
                        ax_out.set_ylabel("Gęstość", fontsize=8)
                        fig_out.tight_layout()
                        self.viz_data["outliers_hist_canvas"].draw()
                    else: ax_out.text(0.5, 0.5, "Brak outlierów", ha='center', va='center')


                    # 3. Rozkład (CDF) dla params_in_range
                    fig_dist, _, ax_dist = setup_canvas(dist_layout, "dist")
                    if len(params_in_range) > 0:
                        sorted_params = np.sort(params_in_range)
                        yvals = np.arange(len(sorted_params)) / float(len(sorted_params) -1 if len(sorted_params) > 1 else 1)
                        ax_dist.plot(sorted_params, yvals, color=self.primary_color)
                        ax_dist.set_title(f"Rozkład dystrybucji (CDF, {p_lower}-{p_upper}%)", fontsize=10)
                        ax_dist.set_xlabel("Wartość parametru", fontsize=8)
                        ax_dist.set_ylabel("Kwantyl", fontsize=8)
                        fig_dist.tight_layout()
                        self.viz_data["dist_canvas"].draw()
                    else: ax_dist.text(0.5, 0.5, "Brak danych w zakresie dla CDF", ha='center', va='center')


                    # 4. Parametry warstw (Box plot)
                    fig_layers, _, ax_layers = setup_canvas(layers_bar_layout, "layers_bar")
                    # Filtruj warstwy z bardzo małą liczbą parametrów
                    valid_layer_data = {k: np.array(v,dtype=np.float32) for k, v in layer_param_values.items() if len(v) > 10} # Minimum 10 parametrów
                    if valid_layer_data:
                        layer_names_sorted = sorted(valid_layer_data.keys())
                        data_to_plot = [valid_layer_data[k] for k in layer_names_sorted]

                        # Przycinanie wartości w danych warstw do zakresu p_lower-p_upper dla lepszej wizualizacji boxplotów
                        data_to_plot_clipped = []
                        for arr in data_to_plot:
                            clipped_arr = arr[(arr >= val_p_lower) & (arr <= val_p_upper)]
                            if len(clipped_arr) > 0:
                                data_to_plot_clipped.append(clipped_arr)
                            # else: data_to_plot_clipped.append(np.array([np.nan])) # Pusta warstwa, jeśli wszystko outlier

                        if data_to_plot_clipped:
                            ax_layers.boxplot(data_to_plot_clipped, vert=False, labels=layer_names_sorted, whis=[p_lower, p_upper], showfliers=False) # whis na percentylach
                            ax_layers.set_title(f"Rozkład parametrów w warstwach ({p_lower}-{p_upper}%)", fontsize=10)
                            ax_layers.set_xlabel("Wartość parametru", fontsize=8)
                            ax_layers.tick_params(axis='y', labelsize=8)
                            fig_layers.tight_layout() # Ważne dla długich nazw warstw
                            self.viz_data["layers_bar_canvas"].draw()
                        else:
                             ax_layers.text(0.5, 0.5, "Brak danych warstw w zakresie", ha='center', va='center')
                    else:
                        ax_layers.text(0.5, 0.5, "Brak wystarczających danych warstw", ha='center', va='center')

                    # 5. Heatmapa (jeśli wybrano warstwę)
                    selected_heatmap_param_name = self.heatmap_layer_selector.currentText()
                    if selected_heatmap_param_name:
                        param_to_heatmap = self._get_parameter_by_path(selected_heatmap_param_name.split('.'))
                        if isinstance(param_to_heatmap, torch.Tensor) and param_to_heatmap.ndim == 2:
                            fig_heat, _, ax_heat = setup_canvas(heatmap_layout, "heatmap") # Heatmap ma własny layout

                            data_for_heatmap = param_to_heatmap.detach().cpu().numpy()
                            # Ogranicz rozmiar heatmapy dla wydajności
                            max_dim_heatmap = 100
                            if data_for_heatmap.shape[0] > max_dim_heatmap or data_for_heatmap.shape[1] > max_dim_heatmap:
                                rows = np.linspace(0, data_for_heatmap.shape[0]-1, min(data_for_heatmap.shape[0], max_dim_heatmap), dtype=int)
                                cols = np.linspace(0, data_for_heatmap.shape[1]-1, min(data_for_heatmap.shape[1], max_dim_heatmap), dtype=int)
                                data_for_heatmap = data_for_heatmap[np.ix_(rows, cols)]
                                ax_heat.set_title(f"Heatmapa wag: {selected_heatmap_param_name} (próbkowana)", fontsize=10)
                            else:
                                ax_heat.set_title(f"Heatmapa wag: {selected_heatmap_param_name}", fontsize=10)

                            cax = ax_heat.matshow(data_for_heatmap, aspect='auto', cmap='viridis')
                            fig_heat.colorbar(cax)
                            ax_heat.set_xlabel("Kolumny", fontsize=8)
                            ax_heat.set_ylabel("Wiersze", fontsize=8)
                            fig_heat.tight_layout()
                            self.viz_data["heatmap_canvas"].draw()
                        elif self.viz_data["heatmap_canvas"]: # Jeśli wcześniej była heatmapa, a teraz nie ma danych
                             # Wyczyść, jeśli jest, a nie ma co rysować
                            if self.viz_data["heatmap_ax"]: self.viz_data["heatmap_ax"].clear()
                            self.viz_data["heatmap_ax"].text(0.5, 0.5, "Wybierz tensor 2D z listy", ha='center', va='center')
                            self.viz_data["heatmap_canvas"].draw()
                    elif self.viz_data["heatmap_canvas"]: # Jeśli nie wybrano nic, a canvas istnieje
                        if self.viz_data["heatmap_ax"]: self.viz_data["heatmap_ax"].clear()
                        self.viz_data["heatmap_ax"].text(0.5, 0.5, "Wybierz tensor 2D z listy", ha='center', va='center')
                        self.viz_data["heatmap_canvas"].draw()


                except Exception as e_draw:
                    import traceback
                    logger.error(f"Błąd podczas rysowania wizualizacji: {e_draw}\n{traceback.format_exc()}")
                    self.stats_info_label.setText(f"Błąd wizualizacji: {e_draw}")

            # Połączenia sygnałów
            redraw_button.clicked.connect(redraw_visualizations)
            self.heatmap_layer_selector.currentIndexChanged.connect(redraw_visualizations) # Odśwież przy zmianie heatmapy

            # Pierwsze rysowanie
            redraw_visualizations()

            logger.info("Wizualizacja parametrów zainicjowana")
            dialog.exec()

        except Exception as e:
            import traceback
            logger.error(f"Błąd podczas inicjalizacji wizualizacji parametrów: {str(e)}\n{traceback.format_exc()}")
            self.show_message(
                "Błąd",
                f"Nie udało się przygotować wizualizacji: {str(e)}",
                QMessageBox.Icon.Critical,
            )

    def compare_models(self):
        if not self.model:
            self.show_message("Błąd", "Najpierw załaduj główny model do porównania.", QMessageBox.Icon.Warning)
            return

        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Wybierz drugi model do porównania",
            self.models_dir, # Zacznij w folderze modeli
            "PyTorch Files (*.pt *.pth);;All Files (*)",
        )

        if file_name:
            try:
                # Użyj innej zmiennej, aby nie nadpisać głównego self.model
                comparison_model_obj = torch.load(
                    file_name, map_location=torch.device("cpu")
                )
                self.show_comparison_dialog(self.model, comparison_model_obj,
                                            os.path.basename(self.current_model_path or "Model 1"),
                                            os.path.basename(file_name))
            except Exception as e:
                import traceback
                QMessageBox.critical(
                    self, "Błąd", f"Nie udało się wczytać modelu do porównania: {str(e)}\n{traceback.format_exc()}"
                )

    def show_comparison_dialog(self, model1, model2, model1_name, model2_name):
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Porównanie: {model1_name} vs {model2_name}")
        dialog.setMinimumSize(800, 600)
        dialog.resize(1000,700)


        layout = QVBoxLayout(dialog)
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # --- Porównanie struktury (tekstowe) ---
        structure_tab = QWidget()
        structure_layout = QVBoxLayout(structure_tab)
        structure_text_edit = QTextEdit()
        structure_text_edit.setReadOnly(True)
        structure_text_edit.setFontFamily("monospace")
        structure_layout.addWidget(structure_text_edit)
        tabs.addTab(structure_tab, "Różnice Struktury")

        structure_diff_report = self._compare_model_structures_text(model1, model2, model1_name, model2_name)
        structure_text_edit.setText(structure_diff_report)

        # --- Porównanie parametrów (wykres i statystyki) ---
        # Tylko jeśli oba są nn.Module
        if isinstance(model1, torch.nn.Module) and isinstance(model2, torch.nn.Module):
            params_tab = QWidget()
            params_layout = QVBoxLayout(params_tab)

            params_fig = Figure(figsize=(8, 6))
            params_canvas = FigureCanvas(params_fig)
            params_layout.addWidget(params_canvas)

            diff_stats_label = QLabel("Statystyki różnic pojawią się tutaj.")
            diff_stats_label.setWordWrap(True)
            params_layout.addWidget(diff_stats_label)

            tabs.addTab(params_tab, "Różnice Parametrów")

            self._plot_parameter_comparison(params_fig, diff_stats_label, model1, model2, model1_name, model2_name)
            params_canvas.draw()
        else:
            not_module_tab = QWidget()
            not_module_layout = QVBoxLayout(not_module_tab)
            not_module_label = QLabel("Porównanie parametrów jest dostępne tylko dla modeli typu torch.nn.Module.")
            not_module_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            not_module_layout.addWidget(not_module_label)
            tabs.addTab(not_module_tab, "Różnice Parametrów")


        dialog.exec()

    def _compare_model_structures_text(self, model1, model2, name1, name2):
        """Generuje tekstowy raport porównujący strukturę dwóch obiektów (modeli, state_dict)."""
        report = [f"Porównanie struktury: '{name1}' vs '{name2}'\n"]

        processed_paths_m1 = set()
        processed_paths_m2 = set()

        def get_paths_and_types(obj, prefix="", current_paths_types=None):
            if current_paths_types is None: current_paths_types = {}

            if isinstance(obj, torch.nn.Module):
                current_paths_types[prefix if prefix else "model"] = f"Moduł ({obj.__class__.__name__})"
                for child_name, child_module in obj.named_children():
                    get_paths_and_types(child_module, f"{prefix}.{child_name}" if prefix else child_name, current_paths_types)
                # Dodajmy też parametry jako część struktury
                for param_name, _ in obj.named_parameters(recurse=False):
                    param_path = f"{prefix}.{param_name}" if prefix else param_name
                    current_paths_types[param_path] = "Parametr"

            elif isinstance(obj, (dict, collections.OrderedDict)):
                current_paths_types[prefix if prefix else "dict_root"] = f"Słownik (len: {len(obj)})"
                for key, value in obj.items():
                    get_paths_and_types(value, f"{prefix}.{key}" if prefix else key, current_paths_types)

            elif isinstance(obj, torch.Tensor):
                 current_paths_types[prefix] = f"Tensor (shape: {list(obj.shape)}, dtype: {obj.dtype})"
            # Można dodać obsługę list/tupli
            else:
                current_paths_types[prefix] = f"Inny ({type(obj).__name__})"
            return current_paths_types

        paths_m1 = get_paths_and_types(model1)
        paths_m2 = get_paths_and_types(model2)

        all_paths = sorted(list(set(paths_m1.keys()) | set(paths_m2.keys())))

        differences_found = False
        for path in all_paths:
            in_m1 = path in paths_m1
            in_m2 = path in paths_m2

            if in_m1 and not in_m2:
                report.append(f"- Tylko w '{name1}': {path} (Typ: {paths_m1[path]})")
                differences_found = True
            elif not in_m1 and in_m2:
                report.append(f"- Tylko w '{name2}': {path} (Typ: {paths_m2[path]})")
                differences_found = True
            elif paths_m1[path] != paths_m2[path]: # Istnieje w obu, ale typ/opis się różni
                report.append(f"- Różnica w '{path}':")
                report.append(f"  '{name1}': {paths_m1[path]}")
                report.append(f"  '{name2}': {paths_m2[path]}")
                differences_found = True

        if not differences_found:
            report.append("Nie znaleziono znaczących różnic w strukturze (na podstawie ścieżek i typów).")

        return "\n".join(report)


    def _plot_parameter_comparison(self, fig, stats_label_widget, model1, model2, name1, name2):
        """Porównuje parametry i rysuje wykres (tylko dla nn.Module)."""
        ax = fig.add_subplot(111)
        ax.clear() # Wyczyść poprzedni wykres

        m1_params_dict = {name: p for name, p in model1.named_parameters() if p.requires_grad}
        m2_params_dict = {name: p for name, p in model2.named_parameters() if p.requires_grad}

        common_param_names = sorted(list(set(m1_params_dict.keys()) & set(m2_params_dict.keys())))

        m1_only_params = sorted(list(set(m1_params_dict.keys()) - set(m2_params_dict.keys())))
        m2_only_params = sorted(list(set(m2_params_dict.keys()) - set(m1_params_dict.keys())))

        report_lines = [f"Porównanie parametrów (tylko uczone): '{name1}' vs '{name2}'\n"]

        if m1_only_params:
            report_lines.append(f"Parametry tylko w '{name1}': {len(m1_only_params)}")
            # for p in m1_only_params[:5]: report_lines.append(f"  - {p}") # Pokaż kilka
            # if len(m1_only_params) > 5: report_lines.append("  ...")
        if m2_only_params:
            report_lines.append(f"Parametry tylko w '{name2}': {len(m2_only_params)}")
            # for p in m2_only_params[:5]: report_lines.append(f"  - {p}")
            # if len(m2_only_params) > 5: report_lines.append("  ...")

        report_lines.append(f"Wspólne parametry: {len(common_param_names)}")

        if not common_param_names:
            ax.text(0.5, 0.5, "Brak wspólnych parametrów do porównania.", ha='center', va='center')
            stats_label_widget.setText("\n".join(report_lines))
            return

        all_diffs_abs = []
        all_diffs_rel = []
        param_names_for_plot = []
        m1_means = []
        m2_means = []

        MAX_PARAMS_TO_PLOT_SCATTER = 50000 # Ograniczenie dla scatter plot

        values_m1_flat = []
        values_m2_flat = []

        for i, p_name in enumerate(common_param_names):
            p1 = m1_params_dict[p_name].detach().cpu().float()
            p2 = m2_params_dict[p_name].detach().cpu().float()

            if p1.shape != p2.shape:
                report_lines.append(f"  - UWAGA: Parametr '{p_name}' ma różne kształty: {p1.shape} vs {p2.shape}. Pomijam.")
                continue

            if p1.numel() == 0: continue

            diff_abs = torch.abs(p1 - p2)
            # Względna różnica, unikaj dzielenia przez zero
            diff_rel = diff_abs / (torch.abs(p1) + 1e-9) # Mały epsilon dla stabilności

            all_diffs_abs.append(diff_abs.mean().item())
            all_diffs_rel.append(diff_rel.mean().item())

            param_names_for_plot.append(p_name if len(p_name) < 30 else p_name[:27]+"...") # Skróć długie nazwy
            m1_means.append(p1.mean().item())
            m2_means.append(p2.mean().item())

            # Do scatter plot - tylko próbka dla wydajności
            if len(values_m1_flat) < MAX_PARAMS_TO_PLOT_SCATTER:
                num_to_add = min(p1.numel(), MAX_PARAMS_TO_PLOT_SCATTER - len(values_m1_flat))
                indices = torch.randperm(p1.numel())[:num_to_add]
                values_m1_flat.extend(p1.flatten()[indices].tolist())
                values_m2_flat.extend(p2.flatten()[indices].tolist())

        if not values_m1_flat: # Jeśli żaden wspólny parametr nie miał wartości
            ax.text(0.5, 0.5, "Brak wartości parametrów do narysowania.", ha='center', va='center')
            stats_label_widget.setText("\n".join(report_lines))
            return

        # Scatter plot wartości
        ax.scatter(values_m1_flat, values_m2_flat, alpha=0.3, s=10, label="Wartości parametrów (próbka)")
        min_val = min(min(values_m1_flat), min(values_m2_flat))
        max_val = max(max(values_m1_flat), max(values_m2_flat))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label="Idealna zgodność (y=x)")

        ax.set_xlabel(f"Wartości w '{name1}'", fontsize=9)
        ax.set_ylabel(f"Wartości w '{name2}'", fontsize=9)
        ax.set_title("Porównanie wartości parametrów", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.7)
        fig.tight_layout()

        # Dodatkowe statystyki
        if all_diffs_abs:
            report_lines.append(f"\nStatystyki różnic dla wspólnych parametrów ({len(all_diffs_abs)}):")
            report_lines.append(f"  Średnia różnica absolutna (Mean Absolute Difference): {np.mean(all_diffs_abs):.4g}")
            report_lines.append(f"  Średnia różnica względna (Mean Relative Difference): {np.mean(all_diffs_rel):.4g} (ostrożnie interpretować)")

            # Znajdź parametry z największymi różnicami
            if len(all_diffs_abs) > 1:
                top_n = min(5, len(all_diffs_abs))
                # Użyj common_param_names, bo param_names_for_plot mogą być skrócone
                # i indices muszą pasować do oryginalnych nazw.
                # Trzeba upewnić się, że `common_param_names` odpowiada `all_diffs_abs`
                # To powinno być OK, bo iterowaliśmy po `common_param_names`

                # indices_abs = np.argsort(all_diffs_abs)[-top_n:][::-1] # Największe abs
                # report_lines.append(f"\nTop {top_n} parametrów z największą śr. różnicą absolutną:")
                # for idx in indices_abs:
                #     p_name_orig = common_param_names[idx] # Znajdź oryginalną, pełną nazwę
                #     report_lines.append(f"  - {p_name_orig}: {all_diffs_abs[idx]:.4g}")

                # Lepiej: użyj `param_names_for_plot` które jest tej samej długości co `all_diffs_abs`
                # ale może być skrócone. Potrzebujemy mapowania z `param_names_for_plot` na `common_param_names`
                # Prościej: sortuj `zip(all_diffs_abs, common_param_names)`

                diff_abs_named = sorted(zip(all_diffs_abs, common_param_names), key=lambda x: x[0], reverse=True)
                report_lines.append(f"\nTop {top_n} parametrów z największą śr. różnicą absolutną:")
                for diff_val, p_name_full in diff_abs_named[:top_n]:
                     report_lines.append(f"  - {p_name_full}: {diff_val:.4g}")


        stats_label_widget.setText("\n".join(report_lines))



    def toggle_tree_expansion(self):
        """Zwijanie/rozwijanie całej struktury drzewa."""
        if self.tree.topLevelItemCount() > 0:
            # Sprawdź, czy pierwszy element jest rozwinięty jako wskaźnik
            # To nie jest idealne, jeśli drzewo ma wiele korzeni lub jest częściowo rozwinięte
            # Lepsze: sprawdź, czy JAKIKOLWIEK element jest rozwinięty.

            is_anything_expanded = False
            iterator = QTreeWidgetItemIterator(self.tree)
            while iterator.value():
                item = iterator.value()
                if item.isExpanded():
                    is_anything_expanded = True
                    break
                iterator += 1

            if is_anything_expanded:
                self.tree.collapseAll()
                logger.debug("Drzewo zwinięte.")
            else:
                self.tree.expandAll() # Rozwiń wszystko
                # Alternatywnie, rozwiń tylko do pewnego poziomu:
                # self.tree.expandToDepth(1) # Rozwiń korzeń i jego bezpośrednie dzieci
                logger.debug("Drzewo rozwinięte.")


    def show_message(self, title, message, icon=QMessageBox.Icon.Information):
        """Wyświetla komunikat i loguje go do konsoli."""
        if icon == QMessageBox.Icon.Information: logger.info(f"UI Info: {title} - {message}")
        elif icon == QMessageBox.Icon.Warning: logger.warning(f"UI Warning: {title} - {message}")
        elif icon == QMessageBox.Icon.Critical: logger.error(f"UI Error: {title} - {message}")
        else: logger.debug(f"UI Message: {title} - {message}")

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(title)
        # Użyj setInformativeText dla dłuższych wiadomości, setText dla krótkich
        if len(message) > 200: # Arbitralny próg
            msg_box.setText(title) # Krótki tekst główny
            msg_box.setInformativeText(message)
        else:
            msg_box.setText(message)

        msg_box.setIcon(icon)
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg_box.exec()

    def filter_tree(self, text):
        """Filtruje drzewo na podstawie tekstu wyszukiwania. (Prosta implementacja)"""
        # Ta implementacja jest podstawowa. Dla dużych drzew może być wolna.
        # Ukrywa/pokazuje elementy, ale nie zwija/rozwija automatycznie.

        search_term = text.lower()

        iterator = QTreeWidgetItemIterator(self.tree)
        while iterator.value():
            item = iterator.value()
            item_text = item.text(0).lower()

            # Sprawdź, czy element pasuje
            matches_item = search_term in item_text

            # Sprawdź, czy którykolwiek z rodziców pasuje (aby pokazać ścieżkę)
            # lub czy którekolwiek z dzieci pasuje (aby pokazać rodzica pasującego dziecka)
            # To jest trudniejsze do zrobienia wydajnie bez pełnego przeszukiwania
            # dla każdego elementu.

            # Prostsze: pokaż element, jeśli pasuje. Jeśli nie, ukryj.
            # Problem: jeśli rodzic jest ukryty, dziecko też nie będzie widoczne.

            # Lepsza logika:
            # 1. Jeśli element pasuje, pokaż go i wszystkich jego rodziców.
            # 2. Jeśli element nie pasuje, ale ma pasujące dziecko, pokaż go i rodziców.
            # 3. Jeśli element nie pasuje i nie ma pasujących dzieci, ukryj go (chyba że rodzic musi być pokazany).

            # Rekurencyjna funkcja pomocnicza
            def check_visibility(current_item):
                # current_item_text = current_item.text(0).lower()
                # item_directly_matches = search_term in current_item_text
                item_directly_matches = search_term in current_item.text(0).lower()


                any_child_matches = False
                for i in range(current_item.childCount()):
                    child = current_item.child(i)
                    if check_visibility(child): # Rekurencyjnie sprawdź dzieci
                        any_child_matches = True

                # Pokaż element, jeśli sam pasuje LUB którekolwiek z jego dzieci (lub wnuków itd.) pasuje
                should_be_visible = item_directly_matches or any_child_matches
                current_item.setHidden(not should_be_visible)

                # Jeśli jest widoczny i ma dzieci, a my szukamy, rozwiń go
                if should_be_visible and any_child_matches and search_term: # Rozwiń, jeśli ma pasujące dzieci
                    current_item.setExpanded(True)
                elif not search_term and current_item.childCount() > 0: # Jeśli czyścimy wyszukiwanie, zwiń (opcjonalne)
                    # Można dodać logikę przywracania poprzedniego stanu rozwinięcia
                    pass # Nie zmieniaj rozwinięcia przy czyszczeniu

                return should_be_visible

            # Przejdź po elementach najwyższego poziomu
            for i in range(self.tree.topLevelItemCount()):
                top_item = self.tree.topLevelItem(i)
                check_visibility(top_item)

            if not search_term: # Jeśli wyszukiwanie jest puste, przywróć domyślne rozwinięcie
                if self.tree.topLevelItemCount() > 0:
                    self.tree.topLevelItem(0).setExpanded(True) # Rozwiń tylko pierwszy poziom
                    # self.tree.collapseAll() # Lub zwiń wszystko i rozwiń korzeń
                    # if self.tree.topLevelItemCount() > 0: self.tree.topLevelItem(0).setExpanded(True)



def main():
    # Ustaw zmienną środowiskową dla lepszego skalowania na HiDPI (opcjonalnie)
    # os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    # QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling)

    app = QApplication(sys.argv)

    # Można ustawić globalny styl aplikacji, np. Fusion dla bardziej nowoczesnego wyglądu
    # app.setStyle("Fusion")

    viewer = ModelViewer()
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```
