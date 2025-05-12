import json
import sys

import numpy as np
import torch
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


class ModelViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Przeglądarka Modeli PyTorch")
        self.setGeometry(100, 100, 1200, 800)

        # Główny widget i layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Przyciski
        button_layout = QHBoxLayout()
        self.load_button = QPushButton("Wczytaj Model")
        self.load_button.clicked.connect(self.load_model)

        # Dodanie przycisku operacji
        self.operations_button = QPushButton("Operacje")
        self.operations_button.clicked.connect(self.show_operations_menu)
        self.operations_button.setEnabled(False)  # Domyślnie wyłączony

        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.operations_button)
        layout.addLayout(button_layout)

        # Splitter dla drzewa i szczegółów
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Drzewo modelu
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Struktura Modelu"])
        self.tree.itemClicked.connect(self.show_details)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.show_context_menu)
        splitter.addWidget(self.tree)

        # Panel szczegółów
        self.details_label = QLabel("Wybierz element z drzewa aby zobaczyć szczegóły")
        self.details_label.setWordWrap(True)
        splitter.addWidget(self.details_label)

        layout.addWidget(splitter)

        self.model = None
        self.current_model_path = None
        self.comparison_model = None

    def show_operations_menu(self):
        if not self.model:
            return

        menu = QMenu(self)

        # Menu podstawowych operacji
        basic_menu = menu.addMenu("Podstawowe operacje")
        save_action = basic_menu.addAction("Zapisz model")
        save_action.triggered.connect(self.save_model)

        export_structure_action = basic_menu.addAction("Eksportuj strukturę")
        export_structure_action.triggered.connect(self.export_structure)

        analyze_action = basic_menu.addAction("Analizuj model")
        analyze_action.triggered.connect(self.analyze_model)

        # Menu eksportu
        export_menu = menu.addMenu("Eksport")
        export_onnx_action = export_menu.addAction("Eksportuj do ONNX")
        export_onnx_action.triggered.connect(self.export_to_onnx)

        export_torchscript_action = export_menu.addAction("Eksportuj do TorchScript")
        export_torchscript_action.triggered.connect(self.export_to_torchscript)

        # Menu analizy
        analysis_menu = menu.addMenu("Analiza")
        visualize_params_action = analysis_menu.addAction("Wizualizuj parametry")
        visualize_params_action.triggered.connect(self.visualize_parameters)

        compare_models_action = analysis_menu.addAction("Porównaj modele")
        compare_models_action.triggered.connect(self.compare_models)

        menu.exec(
            self.operations_button.mapToGlobal(
                self.operations_button.rect().bottomLeft()
            )
        )

    def show_context_menu(self, position):
        item = self.tree.itemAt(position)
        if not item:
            return

        menu = QMenu()
        edit_action = menu.addAction("Edytuj parametr")
        edit_action.triggered.connect(lambda: self.edit_parameter(item))
        menu.exec(self.tree.mapToGlobal(position))

    def edit_parameter(self, item):
        path = []
        current = item
        while current is not None:
            path.insert(0, current.text(0))
            current = current.parent()

        # Znajdź parametr w modelu
        param = self._get_parameter_by_path(path)
        if not param or not isinstance(param, torch.Tensor):
            QMessageBox.warning(self, "Błąd", "Nie można edytować tego elementu")
            return

        # Pobierz nową wartość
        value, ok = QInputDialog.getDouble(
            self,
            "Edytuj parametr",
            f"Wprowadź nową wartość dla {path[-1]}:",
            float(param.mean().item()),
            -1e10,
            1e10,
            6,
        )

        if ok:
            try:
                # Zaktualizuj wartość parametru
                param.fill_(value)
                self.populate_tree()  # Odśwież widok
                QMessageBox.information(
                    self, "Sukces", "Parametr został zaktualizowany"
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Błąd", f"Nie udało się zaktualizować parametru: {str(e)}"
                )

    def save_model(self):
        if not self.model:
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
                QMessageBox.information(
                    self, "Sukces", "Model został zapisany pomyślnie"
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Błąd", f"Nie udało się zapisać modelu: {str(e)}"
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
                QMessageBox.information(
                    self, "Sukces", "Struktura modelu została wyeksportowana"
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Błąd", f"Nie udało się wyeksportować struktury: {str(e)}"
                )

    def analyze_model(self):
        if not self.model:
            return

        total_params = 0
        layers_info = []

        def analyze_module(module, name=""):
            nonlocal total_params
            params = sum(p.numel() for p in module.parameters())
            total_params += params

            if params > 0:  # Tylko warstwy z parametrami
                layers_info.append(
                    {
                        "name": name or module.__class__.__name__,
                        "parameters": params,
                        "type": module.__class__.__name__,
                    }
                )

            for child_name, child in module.named_children():
                analyze_module(child, f"{name}.{child_name}" if name else child_name)

        if isinstance(self.model, torch.nn.Module):
            analyze_module(self.model)
        else:
            total_params = sum(
                p.numel() for p in self.model.values() if isinstance(p, torch.Tensor)
            )

        # Przygotuj raport
        report = f"Analiza modelu:\n\n"
        report += f"Całkowita liczba parametrów: {total_params:,}\n"
        report += f"Liczba warstw: {len(layers_info)}\n\n"
        report += "Szczegóły warstw:\n"
        for layer in layers_info:
            report += f"- {layer['name']}: {layer['parameters']:,} parametrów\n"

        # Wyświetl raport
        QMessageBox.information(self, "Analiza modelu", report)

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

    def load_model(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Wybierz plik modelu PyTorch",
            "",
            "PyTorch Files (*.pt *.pth);;All Files (*)",
        )

        if file_name:
            try:
                self.model = torch.load(file_name, map_location=torch.device("cpu"))
                self.current_model_path = file_name
                self.populate_tree()
                self.operations_button.setEnabled(True)
            except Exception as e:
                self.details_label.setText(f"Błąd podczas wczytywania modelu: {str(e)}")

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
            item = QTreeWidgetItem(parent_item, [name])
            if isinstance(child, torch.nn.Module):
                self._add_module_to_tree(item, child)
            else:
                self._add_parameters_to_tree(item, child)

    def _add_state_dict_to_tree(self, parent_item, state_dict):
        for key, value in state_dict.items():
            item = QTreeWidgetItem(parent_item, [key])
            if isinstance(value, dict):
                self._add_state_dict_to_tree(item, value)
            else:
                self._add_parameters_to_tree(item, value)

    def _add_parameters_to_tree(self, parent_item, module):
        for name, param in module.named_parameters():
            QTreeWidgetItem(parent_item, [f"{name}: {param.shape}"])

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
            if isinstance(current, torch.nn.Module):
                current = getattr(current, name)
            elif isinstance(current, dict):
                current = current.get(name)
            else:
                return f"Nie można znaleźć: {' -> '.join(path)}"

        if isinstance(current, torch.nn.Module):
            return (
                f"Moduł: {current.__class__.__name__}\n"
                + f"Parametry: {sum(p.numel() for p in current.parameters())}"
            )
        elif isinstance(current, torch.Tensor):
            return (
                f"Tensor: {current.shape}\n"
                + f"Typ: {current.dtype}\n"
                + f"Wymiary: {current.dim()}"
            )
        else:
            return str(current)

    def export_to_onnx(self):
        if not isinstance(self.model, torch.nn.Module):
            QMessageBox.warning(
                self, "Błąd", "Eksport do ONNX wymaga modelu typu torch.nn.Module"
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
                QMessageBox.information(
                    self, "Sukces", "Model został wyeksportowany do formatu ONNX"
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Błąd", f"Nie udało się wyeksportować modelu: {str(e)}"
                )

    def export_to_torchscript(self):
        if not isinstance(self.model, torch.nn.Module):
            QMessageBox.warning(
                self,
                "Błąd",
                "Eksport do TorchScript wymaga modelu typu torch.nn.Module",
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
                QMessageBox.information(
                    self, "Sukces", "Model został wyeksportowany do formatu TorchScript"
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Błąd", f"Nie udało się wyeksportować modelu: {str(e)}"
                )

    def visualize_parameters(self):
        if not self.model:
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Wizualizacja parametrów")
        dialog.setGeometry(200, 200, 800, 600)

        layout = QVBoxLayout(dialog)

        # Zakładki dla różnych typów wizualizacji
        tabs = QTabWidget()

        # Histogram wartości
        hist_tab = QWidget()
        hist_layout = QVBoxLayout(hist_tab)
        hist_fig = Figure(figsize=(8, 6))
        hist_canvas = FigureCanvas(hist_fig)
        hist_layout.addWidget(hist_canvas)
        tabs.addTab(hist_tab, "Histogram wartości")

        # Wykres rozkładu
        dist_tab = QWidget()
        dist_layout = QVBoxLayout(dist_tab)
        dist_fig = Figure(figsize=(8, 6))
        dist_canvas = FigureCanvas(dist_fig)
        dist_layout.addWidget(dist_canvas)
        tabs.addTab(dist_tab, "Rozkład wartości")

        layout.addWidget(tabs)

        # Zbierz wszystkie parametry
        all_params = []
        for param in self.model.parameters():
            all_params.extend(param.detach().cpu().numpy().flatten())

        # Rysuj histogram
        ax1 = hist_fig.add_subplot(111)
        ax1.hist(all_params, bins=50)
        ax1.set_title("Rozkład wartości parametrów")
        ax1.set_xlabel("Wartość")
        ax1.set_ylabel("Liczba parametrów")
        hist_canvas.draw()

        # Rysuj rozkład
        ax2 = dist_fig.add_subplot(111)
        ax2.plot(sorted(all_params))
        ax2.set_title("Rozkład wartości parametrów")
        ax2.set_xlabel("Indeks parametru")
        ax2.set_ylabel("Wartość")
        dist_canvas.draw()

        dialog.exec()

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


def main():
    app = QApplication(sys.argv)
    viewer = ModelViewer()
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
