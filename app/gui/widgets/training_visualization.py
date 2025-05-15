import numpy as np
import pyqtgraph as pg
import torch
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)


class TrainingVisualization(QWidget):
    """Widget do wizualizacji procesu treningu w czasie rzeczywistym."""

    def __init__(self, parent=None, settings=None):
        super().__init__(parent)
        self.settings = settings if settings is not None else {}
        self.setup_ui()

        # Inicjalizacja danych
        self.train_loss_data = []
        self.train_acc_data = []
        self.val_loss_data = []
        self.val_acc_data = []
        self.val_top3_data = []
        self.val_top5_data = []
        self.val_precision_data = []
        self.val_recall_data = []
        self.val_f1_data = []
        self.val_auc_data = []
        self.loss_diff_data = []  # Nowa metryka
        self.learning_rates_data = []  # Nowa metryka
        self.gpu_usage_data = []  # Nowa metryka GPU
        self.gpu_memory_data = []  # Nowa metryka GPU
        self.epochs = []

        # Flaga wskazująca czy dane zostały zaktualizowane
        self.data_updated = False

    def get_theme_colors(self):
        """Zwraca kolory dla ciemnego motywu."""
        return {
            "background": (45, 45, 45),  # Jaśniejszy ciemnoszary
            "foreground": (220, 220, 220),  # Jaśniejszy jasnoszary
            "grid": (80, 80, 80),  # Jaśniejszy średni szary
            "train_loss": (0, 150, 255),  # Jaśniejszy niebieski
            "val_loss": (255, 50, 50),  # Jaśniejszy czerwony
            "train_acc": (50, 220, 50),  # Jaśniejszy zielony
            "val_acc": (220, 50, 220),  # Jaśniejszy fioletowy
            "val_top3": (255, 180, 50),  # Jaśniejszy pomarańczowy
            "val_top5": (255, 160, 50),  # Jaśniejszy ciemny pomarańczowy
            "val_precision": (50, 255, 255),  # Jaśniejszy cyjan
            "val_recall": (255, 50, 255),  # Jaśniejszy magenta
            "val_f1": (255, 255, 50),  # Jaśniejszy żółty
            "val_auc": (180, 50, 180),  # Jaśniejszy fioletowy
            "gpu_usage": (255, 215, 0),  # Złoty
            "gpu_memory": (255, 69, 0),  # Czerwono-pomarańczowy
        }

    def setup_ui(self):
        """Konfiguruje interfejs użytkownika."""
        layout = QVBoxLayout(self)

        # Kontrolki
        controls_layout = QHBoxLayout()
        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Wykres
        self.plot_widget = pg.PlotWidget()

        # Pobierz kolory z motywu
        colors = self.get_theme_colors()

        # Ustaw kolor tła
        self.plot_widget.setBackground(colors["background"])

        # Ustaw kolor siatki
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.getAxis("bottom").setPen(colors["grid"])
        self.plot_widget.getAxis("left").setPen(colors["grid"])

        # Ustaw kolory osi
        self.plot_widget.getAxis("bottom").setTextPen(colors["foreground"])
        self.plot_widget.getAxis("left").setTextPen(colors["foreground"])

        # Ustaw kolory legendy
        self.plot_widget.addLegend(labelTextColor=colors["foreground"])

        # Ustaw etykiety osi
        self.plot_widget.setLabel("left", "Wartość", color=colors["foreground"])
        self.plot_widget.setLabel("bottom", "Epoka", color=colors["foreground"])

        layout.addWidget(self.plot_widget)

        # --- Early Stopping ---
        self.early_stopping_group = QGroupBox("Early Stopping")
        self.early_stopping_layout = QVBoxLayout()
        self.early_stopping_label = QLabel("Czekam na dane...")
        self.early_stopping_progress = QProgressBar()
        self.early_stopping_progress.setRange(0, 10)
        self.early_stopping_progress.setValue(0)
        self.early_stopping_layout.addWidget(self.early_stopping_label)
        self.early_stopping_layout.addWidget(self.early_stopping_progress)
        self.early_stopping_group.setLayout(self.early_stopping_layout)
        layout.addWidget(self.early_stopping_group)

        # --- GPU Info ---
        self.gpu_info_group = QGroupBox("Informacje o GPU")
        self.gpu_info_layout = QHBoxLayout()

        self.gpu_usage_label = QLabel("Wykorzystanie GPU: -")
        self.gpu_memory_label = QLabel("Pamięć GPU: -")

        self.gpu_info_layout.addWidget(self.gpu_usage_label)
        self.gpu_info_layout.addWidget(self.gpu_memory_label)

        self.gpu_info_group.setLayout(self.gpu_info_layout)
        layout.addWidget(self.gpu_info_group)

        # Style dla linii epok
        self.epoch_line_pen = pg.mkPen(
            color=colors["grid"], style=Qt.PenStyle.DashLine, width=1
        )

    def get_line_configs(self):
        """Zwraca konfiguracje dla wszystkich linii wykresu."""
        return [
            # Metryki kluczowe - najgrubsze linie, mocne kolory, linia ciągła
            {
                "data": self.val_loss_data,
                "color": (255, 50, 50),
                "width": 4,
                "style": Qt.PenStyle.SolidLine,
                "name": "Strata walidacyjna",
                "symbol": "o",
                "symbol_size": 6,
            },
            {
                "data": self.val_acc_data,
                "color": (50, 220, 50),
                "width": 4,
                "style": Qt.PenStyle.SolidLine,
                "name": "Dokładność walidacyjna",
                "symbol": "o",
                "symbol_size": 6,
            },
            {
                "data": self.train_loss_data,
                "color": (0, 150, 255),
                "width": 3,
                "style": Qt.PenStyle.SolidLine,
                "name": "Strata treningowa",
                "symbol": "o",
                "symbol_size": 4,
            },
            {
                "data": self.train_acc_data,
                "color": (220, 50, 220),
                "width": 3,
                "style": Qt.PenStyle.SolidLine,
                "name": "Dokładność treningowa",
                "symbol": "o",
                "symbol_size": 4,
            },
            # Nowa metryka - różnica między stratami
            {
                "data": self.loss_diff_data,
                "color": (255, 255, 50),
                "width": 2,
                "style": Qt.PenStyle.DashLine,
                "name": "Różnica strat",
                "symbol": "d",
                "symbol_size": 3,
                "dash": [5, 5],
            },
            # Nowa metryka - learning rate
            {
                "data": self.learning_rates_data,
                "color": (50, 255, 50),
                "width": 2,
                "style": Qt.PenStyle.DotLine,
                "name": "Learning rate",
                "symbol": "x",
                "symbol_size": 3,
                "dash": [2, 2],
            },
            # Pozostałe metryki
            {
                "data": self.val_f1_data,
                "color": (255, 180, 50),
                "width": 2,
                "style": Qt.PenStyle.DashLine,
                "name": "F1-score",
                "symbol": "t",
                "symbol_size": 3,
                "dash": [10, 5],
            },
            {
                "data": self.val_auc_data,
                "color": (180, 50, 180),
                "width": 2,
                "style": Qt.PenStyle.DotLine,
                "name": "AUC",
                "symbol": "t",
                "symbol_size": 3,
                "dash": [2, 4],
            },
            {
                "data": self.val_precision_data,
                "color": (50, 255, 255),
                "width": 1,
                "style": Qt.PenStyle.DashDotLine,
                "name": "Precyzja",
                "symbol": "s",
                "symbol_size": 2,
                "dash": [8, 4, 2, 4],
            },
            {
                "data": self.val_recall_data,
                "color": (255, 50, 255),
                "width": 1,
                "style": Qt.PenStyle.DashDotDotLine,
                "name": "Recall",
                "symbol": "s",
                "symbol_size": 2,
                "dash": [8, 4, 2, 4, 2, 4],
            },
            {
                "data": self.val_top3_data,
                "color": (255, 160, 50),
                "width": 1,
                "style": Qt.PenStyle.DashDotLine,
                "name": "Top-3 dokładność",
                "symbol": "t",
                "symbol_size": 2,
                "dash": [6, 3, 2, 3],
            },
            {
                "data": self.val_top5_data,
                "color": (255, 100, 50),
                "width": 1,
                "style": Qt.PenStyle.DashDotDotLine,
                "name": "Top-5 dokładność",
                "symbol": "t",
                "symbol_size": 2,
                "dash": [6, 3, 2, 3, 2, 3],
            },
            # Nowe metryki GPU
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

    def update_plot(self):
        """Aktualizuje wykres, rysując wszystkie dostępne metryki."""
        try:
            self.plot_widget.clear()
            if not self.epochs or len(self.epochs) == 0:
                return

            x_data = np.array(self.epochs)

            # Rysuj wszystkie metryki według konfiguracji
            for config in self.get_line_configs():
                if len(config["data"]) > 0 and all(
                    v is not None for v in config["data"]
                ):
                    y_data = np.array(config["data"])
                    pen_kwargs = {
                        "color": config["color"],
                        "width": config["width"],
                        "style": config["style"],
                    }
                    # Dodaj parametr dash jeśli jest zdefiniowany
                    if "dash" in config:
                        pen_kwargs["dash"] = config["dash"]

                    self.plot_widget.plot(
                        x_data[: len(y_data)],
                        y_data,
                        pen=pg.mkPen(**pen_kwargs),
                        name=config["name"],
                        symbol=config["symbol"],
                        symbolSize=config["symbol_size"],
                        symbolBrush=config["color"],
                        symbolPen=None,
                    )

            self.plot_widget.autoRange()
            self.data_updated = False

        except Exception as e:
            print(f"Błąd w update_plot: {e}")
            import traceback

            print(traceback.format_exc())

    def update_data(
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
            f"gpu_usage={gpu_usage}, gpu_memory={gpu_memory}"  # Dodane nowe parametry
        )
        try:
            # Konwersja i walidacja danych
            try:
                epoch = int(epoch)
                train_loss = float(train_loss) if train_loss is not None else None
                train_acc = float(train_acc) if train_acc is not None else None
                val_loss = float(val_loss) if val_loss is not None else None
                val_acc = float(val_acc) if val_acc is not None else None
                val_top3 = float(val_top3) if val_top3 is not None else None
                val_top5 = float(val_top5) if val_top5 is not None else None
                val_precision = (
                    float(val_precision) if val_precision is not None else None
                )
                val_recall = float(val_recall) if val_recall is not None else None
                val_f1 = float(val_f1) if val_f1 is not None else None
                val_auc = float(val_auc) if val_auc is not None else None
                learning_rate = (
                    float(learning_rate) if learning_rate is not None else None
                )
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
                    self.train_loss_data[idx] = train_loss
                    self.train_acc_data[idx] = train_acc
                    self.val_loss_data[idx] = val_loss
                    self.val_acc_data[idx] = val_acc
                    self.val_top3_data[idx] = val_top3
                    self.val_top5_data[idx] = val_top5
                    self.val_precision_data[idx] = val_precision
                    self.val_recall_data[idx] = val_recall
                    self.val_f1_data[idx] = val_f1
                    self.val_auc_data[idx] = val_auc
                    self.learning_rates_data[idx] = learning_rate
                    # Dodanie danych GPU
                    if gpu_usage is not None:
                        self.gpu_usage_data[idx] = gpu_usage
                    if gpu_memory is not None:
                        self.gpu_memory_data[idx] = gpu_memory
                else:
                    self.epochs.append(epoch)
                    self.train_loss_data.append(train_loss)
                    self.train_acc_data.append(train_acc)
                    self.val_loss_data.append(val_loss)
                    self.val_acc_data.append(val_acc)
                    self.val_top3_data.append(val_top3)
                    self.val_top5_data.append(val_top5)
                    self.val_precision_data.append(val_precision)
                    self.val_recall_data.append(val_recall)
                    self.val_f1_data.append(val_f1)
                    self.val_auc_data.append(val_auc)
                    self.learning_rates_data.append(learning_rate)
                    # Dodanie danych GPU
                    self.gpu_usage_data.append(gpu_usage)
                    self.gpu_memory_data.append(gpu_memory)

            # Aktualizuj etykiety GPU
            if gpu_usage is not None:
                self.gpu_usage_label.setText(f"Wykorzystanie GPU: {gpu_usage:.1f}%")
            if gpu_memory is not None:
                self.gpu_memory_label.setText(f"Pamięć GPU: {gpu_memory:.1f} MB")

            self.data_updated = True
            self.update_plot()

        except Exception as e:
            print(f"Błąd w update_data: {e}")
            import traceback

            print(traceback.format_exc())

    def clear_data(self):
        """Czyści wszystkie dane wykresu."""
        self.epochs = []
        self.train_loss_data = []
        self.train_acc_data = []
        self.val_loss_data = []
        self.val_acc_data = []
        self.val_top3_data = []
        self.val_top5_data = []
        self.val_precision_data = []
        self.val_recall_data = []
        self.val_f1_data = []
        self.val_auc_data = []
        self.loss_diff_data = []  # Nowa metryka
        self.learning_rates_data = []  # Nowa metryka
        self.gpu_usage_data = []  # Nowa metryka GPU
        self.gpu_memory_data = []  # Nowa metryka GPU
        self.data_updated = False
        self.update_plot()
        self.early_stopping_label.setText("Czekam na dane...")
        self.early_stopping_progress.setRange(0, 10)
        self.early_stopping_progress.setValue(0)

    def reset_plot(self):
        """Resetuje wykres do stanu początkowego."""
        self.plot_widget.clear()
        self.plot_widget.setLabel("left", "Wartość")
        self.plot_widget.setLabel("bottom", "Epoka")
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.autoRange()
        self.data_updated = False
        self.early_stopping_label.setText("Czekam na dane...")
        self.early_stopping_progress.setRange(0, 10)
        self.early_stopping_progress.setValue(0)

    def save_plot(self, filename):
        """Zapisuje wykres do pliku PNG.

        Args:
            filename (str): Ścieżka do pliku, w którym zostanie zapisany wykres.
        """
        try:
            # Upewnij się, że wykres jest aktualny
            if self.data_updated:
                self.update_plot()

            # Zapamiętaj oryginalne ograniczenia szerokości
            original_min_width = self.plot_widget.minimumWidth()
            original_max_width = self.plot_widget.maximumWidth()

            # Zapisz wykres w formacie PNG z minimalną szerokością 3000px
            self.plot_widget.setFixedWidth(3000)  # Ustaw minimalną szerokość
            export_image = self.plot_widget.grab()
            export_image.save(filename)

            # Przywróć oryginalne ograniczenia szerokości
            self.plot_widget.setMinimumWidth(original_min_width)
            self.plot_widget.setMaximumWidth(original_max_width)

            # Upewnij się, że wykres jest nadal widoczny
            self.update_plot()

            return True
        except Exception as e:
            print(f"Błąd podczas zapisywania wykresu: {e}")
            import traceback

            print(traceback.format_exc())
            return False

    def update_early_stopping_status(self, patience_counter=0, patience_max=10):
        self.early_stopping_progress.setMaximum(patience_max)
        self.early_stopping_progress.setValue(patience_counter)
        if patience_counter == 0:
            self.early_stopping_label.setText("Early stopping: Poprawa w tej epoce")
            self.early_stopping_progress.setStyleSheet(
                "QProgressBar::chunk { background-color: green; }"
            )
        else:
            self.early_stopping_label.setText(
                f"Early stopping: {patience_counter}/{patience_max}"
            )
            if patience_counter >= patience_max - 1:
                self.early_stopping_progress.setStyleSheet(
                    "QProgressBar::chunk { background-color: red; }"
                )
            elif patience_counter >= patience_max // 2:
                self.early_stopping_progress.setStyleSheet(
                    "QProgressBar::chunk { background-color: orange; }"
                )
            else:
                self.early_stopping_progress.setStyleSheet(
                    "QProgressBar::chunk { background-color: yellow; }"
                )
