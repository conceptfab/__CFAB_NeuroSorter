1. Zmiany w konfiguracji matplotlib
python# Zbędna nadmiarowa konfiguracja czcionek - można uprościć
matplotlib.use("qtagg")  # Standardowy backend Qt
matplotlib.rcParams["font.family"] = "DejaVu Sans"
# Usunąć te nadmiarowe linie konfiguracji czcionek, wystarczy ustawić font.family
# matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
# matplotlib.rcParams["font.serif"] = ["DejaVu Sans"]
# matplotlib.rcParams["font.monospace"] = ["DejaVu Sans"]
# matplotlib.rcParams["font.cursive"] = ["DejaVu Sans"]
# matplotlib.rcParams["font.fantasy"] = ["DejaVu Sans"]
# matplotlib.rcParams["mathtext.fontset"] = "dejavusans"
2. Optymalizacja klasy ModelAnalyzerThread
pythondef analyze_module(module, name=""):
    nonlocal total_params
    params = sum(p.numel() for p in module.parameters())
    total_params += params

    # Optymalizacja - dodajemy tylko warstwy z parametrami
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
3. Poprawa literówki w komunikacie
python# Literówka w słowie "zagnieżdżony" - powinno być "zagnieżdżony"
self.progress_update.emit(
    f"Znaleziono zagnieżdżony słownik: {current_path}"
)
4. Optymalizacja funkcji populate_tree
pythondef populate_tree(self):
    self.tree.clear()
    if isinstance(self.model, torch.nn.Module):
        root = QTreeWidgetItem(self.tree, ["Model"])
        self._add_module_to_tree(root, self.model)
    else:
        root = QTreeWidgetItem(self.tree, ["State Dict"])
        self._add_state_dict_to_tree(root, self.model)
    # Opcjonalnie: zamiast expandAll() można rozwinąć tylko pierwszy poziom 
    # dla dużych modeli
    # self.tree.expandAll()  # Może być wolne dla dużych modeli
    root.setExpanded(True)  # Rozwiń tylko główny węzeł
5. Optymalizacja funkcji _add_parameters_to_tree
pythondef _add_parameters_to_tree(self, parent_item, module):
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
            param_batch.append(
                (f"{name}: shape={shape}, dtype={dtype}, mean={mean:.4f}")
            )
    
    # Dodanie wszystkich parametrów naraz
    for param_text in param_batch:
        QTreeWidgetItem(parent_item, [param_text])
6. Optymalizacja funkcji show_details
pythondef show_details(self, item, column):
    # Buforowanie ścieżki dla lepszej wydajności
    path = []
    current = item
    while current is not None:
        path.insert(0, current.text(0))
        current = current.parent()

    # Możemy dodać cache dla często oglądanych detali
    # Przykład:
    path_key = "/".join(path)
    if hasattr(self, '_details_cache') and path_key in self._details_cache:
        self.details_label.setText(self._details_cache[path_key])
        return
    
    details = self._get_details(path)
    
    # Przechowuj w cache dla przyszłych wywołań
    if not hasattr(self, '_details_cache'):
        self._details_cache = {}
    self._details_cache[path_key] = details
    
    self.details_label.setText(details)
7. Poprawa funkcji export_to_onnx i export_to_torchscript
pythondef export_to_onnx(self):
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
                if 'weight' in name and len(param.shape) == 4:
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
8. Optymalizacja funkcji visualize_parameters
pythondef visualize_parameters(self):
    """Wizualizacja parametrów modelu z obsługą outlierów i skalą logarytmiczną."""
    logger.info("Wizualizacja parametrów modelu")
    if not self.model:
        logger.warning("Brak modelu do wizualizacji parametrów")
        self.show_message(
            "Błąd", "Brak modelu do wizualizacji", QMessageBox.Icon.Warning
        )
        return

    try:
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
                        indices = np.linspace(0, len(values)-1, MAX_PARAMS, dtype=int)
                        values = values[indices]
                    all_params.extend(values.tolist())
                    layer_name = name.split(".")[0]
                    if layer_name not in layer_params:
                        layer_params[layer_name] = []
                    layer_params[layer_name].extend(values.tolist())
                    # Limit do MAX_PARAMS parametrów na warstwę
                    if len(layer_params[layer_name]) > MAX_PARAMS:
                        layer_params[layer_name] = layer_params[layer_name][:MAX_PARAMS]
        # ... pozostała część funkcji bez zmian ...
9. Dodanie limitu liczby elementów w drzewie
pythondef _add_module_to_tree(self, parent_item, module):
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
        item = QTreeWidgetItem(
            parent_item, [f"{name} ({layer_type}, " f"parametry: {param_count})"]
        )
        if isinstance(child, torch.nn.Module):
            self._add_module_to_tree(item, child)
        self._add_parameters_to_tree(item, child)
        count += 1
10. Czyszczenie pamięci cache
pythondef load_model_from_file(self, file_path):
    logger.info("Wczytywanie modelu z pliku: %s", file_path)
    try:
        # Wyczyść pamięć podręczną przy ładowaniu nowego modelu
        if hasattr(self, '_details_cache'):
            self._details_cache = {}
            
        self.model = torch.load(file_path, map_location=torch.device("cpu"))
        self.current_model_path = file_path
        # Opcjonalnie: wymuś odśmiecanie
        import gc
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
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
Powyższe zmiany powinny znacząco poprawić wydajność, zoptymalizować kod i usunąć kilka potencjalnych błędów. Główne obszary usprawnień to:

Redukcja niepotrzebnej konfiguracji matplotlib
Optymalizacja przetwarzania dużych modeli poprzez limit elementów
Buforowanie wyników dla częstych operacji
Inteligentniejsze wykrywanie kształtu wejścia modelu
Poprawa wydajności przy wizualizacji parametrów dużych modeli
Zarządzanie pamięcią dla dużych modeli
Poprawa błędów literowych i logicznych