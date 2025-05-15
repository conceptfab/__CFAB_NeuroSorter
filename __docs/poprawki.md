Wizualizacja użycia GPU podczas procesów treningowych w PyQt6
Tak, procesy treningowe mogą wizualizować pracę GPU. W dostarczonym kodzie, szczególnie w TrainingVisualization z pliku training_visualization.py, istnieje już infrastruktura do wyświetlania metryk treningu w czasie rzeczywistym. Można ją rozszerzyć, aby monitorować również użycie GPU.
Zmiany w kodzie
Zmiana 1: Dodanie monitorowania GPU w training_visualization.py
python# W pliku training_visualization.py
# Dodajemy import biblioteki do monitorowania GPU
import torch
# Można również użyć pynvml lub GPUtil dla bardziej szczegółowych informacji

def get_theme_colors(self):
    """Zwraca kolory dla ciemnego motywu."""
    colors = {
        # istniejące kolory...
        "background": (45, 45, 45),
        "foreground": (220, 220, 220),
        "grid": (80, 80, 80),
        "train_loss": (0, 150, 255),
        # ...

        # Dodajemy nowe kolory dla metryk GPU
        "gpu_usage": (255, 215, 0),  # Złoty
        "gpu_memory": (255, 69, 0),  # Czerwono-pomarańczowy
    }
    return colors
Zmiana 2: Dodanie nowych pól danych w konstruktorze TrainingVisualization
pythondef __init__(self, parent=None, settings=None):
    super().__init__(parent)
    self.settings = settings if settings is not None else {}
    self.setup_ui()

    # Inicjalizacja danych (istniejące)
    self.train_loss_data = []
    # ...

    # Nowe pola do monitorowania GPU
    self.gpu_usage_data = []
    self.gpu_memory_data = []
Zmiana 3: Dodanie nowych konfiguracji linii do funkcji get_line_configs
pythondef get_line_configs(self):
    """Zwraca konfiguracje dla wszystkich linii wykresu."""
    return [
        # Istniejące metryki...
        
        # Dodaj nowe metryki GPU
        {
            "data": self.gpu_usage_data,
            "color": (255, 215, 0),  # Złoty
            "width": 2,
            "style": Qt.PenStyle.DashLine,
            "name": "Użycie GPU (%)",
            "symbol": "s",
            "symbol_size": 3,
            "dash": [5, 5],
        },
        {
            "data": self.gpu_memory_data,
            "color": (255, 69, 0),  # Czerwono-pomarańczowy
            "width": 2,
            "style": Qt.PenStyle.DashLine,
            "name": "Pamięć GPU (MB)",
            "symbol": "t",
            "symbol_size": 3,
            "dash": [2, 4],
        },
    ]
Zmiana 4: Rozszerzenie metody update_data o parametry GPU
pythondef update_data(
    self,
    epoch,
    train_loss,
    train_acc,
    val_loss=None,
    val_acc=None,
    val_top3=None,
    val_top5=None,
    val_precision=None,
    val_recall=None,
    val_f1=None,
    val_auc=None,
    learning_rate=None,
    gpu_usage=None,  # Nowy parametr
    gpu_memory=None,  # Nowy parametr
):
    """Aktualizuje dane wykresu."""
    print(
        f"[TrainingVisualization] update_data: epoka={epoch}, "
        f"train_loss={train_loss}, train_acc={train_acc}, "
        f"val_loss={val_loss}, val_acc={val_acc}, "
        f"gpu_usage={gpu_usage}, gpu_memory={gpu_memory}"  # Dodane nowe parametry do logowania
    )
    try:
        # Konwersja i walidacja danych
        try:
            epoch = int(epoch)
            # ...istniejące konwersje...
            
            # Nowe konwersje dla GPU
            gpu_usage = float(gpu_usage) if gpu_usage is not None else None
            gpu_memory = float(gpu_memory) if gpu_memory is not None else None
        except (ValueError, TypeError) as e:
            print(f"BŁĄD konwersji danych: {e}")
            return

        # Dodaj nowe dane tylko jeśli epoka jest dodatnia
        if epoch > 0:
            # Sprawdź czy ta epoka już istnieje
            if epoch in self.epochs:
                idx = self.epochs.index(epoch)
                # ...istniejące przypisania...
                
                # Dodanie danych GPU
                if gpu_usage is not None:
                    self.gpu_usage_data[idx] = gpu_usage
                if gpu_memory is not None:
                    self.gpu_memory_data[idx] = gpu_memory
            else:
                self.epochs.append(epoch)
                # ...istniejące dodania...
                
                # Dodanie danych GPU
                self.gpu_usage_data.append(gpu_usage)
                self.gpu_memory_data.append(gpu_memory)

            self.data_updated = True
            self.update_plot()

    except Exception as e:
        print(f"Błąd w update_data: {e}")
        import traceback
        print(traceback.format_exc())
Zmiana 5: Dodanie funkcji pobierania danych GPU w queue_manager.py
pythondef _on_task_progress(self, task_name, progress, details):
    """Obsługa postępu zadania."""
    # Aktualizuj wizualizację
    if self.training_visualization:
        epoch = int(details.get("epoch", 0))
        train_loss = details.get("train_loss")
        train_acc = details.get("train_acc")
        val_loss = details.get("val_loss")
        val_acc = details.get("val_acc")
        val_top3 = details.get("val_top3")
        val_top5 = details.get("val_top5")
        val_precision = details.get("val_precision")
        val_recall = details.get("val_recall")
        val_f1 = details.get("val_f1")
        val_auc = details.get("val_auc")
        
        # Dodaj pobieranie informacji o GPU
        gpu_usage = None
        gpu_memory = None
        if torch.cuda.is_available():
            try:
                # Pobierz wykorzystanie GPU
                gpu_usage = torch.cuda.utilization()  # Procentowe wykorzystanie
                # Pobierz wykorzystanie pamięci w MB
                gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
            except Exception as e:
                print(f"Błąd podczas pobierania danych GPU: {e}")
        
        if epoch > 0:
            self.training_visualization.update_data(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                val_top3=val_top3,
                val_top5=val_top5,
                val_precision=val_precision,
                val_recall=val_recall,
                val_f1=val_f1,
                val_auc=val_auc,
                gpu_usage=gpu_usage,       # Nowy parametr
                gpu_memory=gpu_memory,     # Nowy parametr
            )
Zmiana 6: Dodanie grupy informacyjnej w interfejsie
pythondef setup_ui(self):
    """Konfiguruje interfejs użytkownika."""
    layout = QVBoxLayout(self)
    
    # ...istniejące elementy...
    
    # Dodaj nową grupę informacji o GPU
    self.gpu_info_group = QGroupBox("Informacje o GPU")
    self.gpu_info_layout = QHBoxLayout()
    
    self.gpu_usage_label = QLabel("Wykorzystanie GPU: -")
    self.gpu_memory_label = QLabel("Pamięć GPU: -")
    
    self.gpu_info_layout.addWidget(self.gpu_usage_label)
    self.gpu_info_layout.addWidget(self.gpu_memory_label)
    
    self.gpu_info_group.setLayout(self.gpu_info_layout)
    layout.addWidget(self.gpu_info_group)
    
    # ...istniejące elementy...
Zmiana 7: Aktualizacja etykiet GPU w metodzie update_data
pythondef update_data(self, ...):
    # ...istniejący kod...
    
    # Aktualizuj etykiety GPU
    if gpu_usage is not None:
        self.gpu_usage_label.setText(f"Wykorzystanie GPU: {gpu_usage:.1f}%")
    if gpu_memory is not None:
        self.gpu_memory_label.setText(f"Pamięć GPU: {gpu_memory:.1f} MB")