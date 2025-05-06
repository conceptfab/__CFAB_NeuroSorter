import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget


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
        self.epochs = []

        # Flaga wskazująca czy dane zostały zaktualizowane
        self.data_updated = False

    def get_theme_colors(self):
        """Zwraca kolory dla ciemnego motywu."""
        return {
            "background": (30, 30, 30),  # Ciemnoszary
            "foreground": (200, 200, 200),  # Jasnoszary
            "grid": (60, 60, 60),  # Średni szary
            "train_loss": (0, 120, 215),  # Niebieski
            "val_loss": (215, 0, 0),  # Czerwony
            "train_acc": (0, 180, 0),  # Zielony
            "val_acc": (180, 0, 180),  # Fioletowy
        }

    def setup_ui(self):
        """Konfiguruje interfejs użytkownika."""
        layout = QVBoxLayout(self)

        # Nagłówek
        header = QLabel("Wizualizacja treningu")
        header.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(header)

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

        # Style dla linii epok
        self.epoch_line_pen = pg.mkPen(
            color=colors["grid"], style=Qt.PenStyle.DashLine, width=1
        )

    def update_plot(self):
        """Aktualizuje wykres, rysując wszystkie dostępne metryki."""
        try:
            # Wyczyść wykres
            self.plot_widget.clear()

            # Sprawdź czy mamy dane do wyświetlenia
            if not self.epochs or len(self.epochs) == 0:
                return

            # Przygotuj dane X (epoki)
            x_data = np.array(self.epochs)

            # Dodaj legendę
            self.plot_widget.addLegend()

            # Pobierz kolory z motywu
            colors = self.get_theme_colors()

            # Rysuj stratę treningową
            if len(self.train_loss_data) > 0 and all(
                isinstance(v, (int, float)) for v in self.train_loss_data
            ):
                y_data_train_loss = np.array(self.train_loss_data)
                self.plot_widget.plot(
                    x_data[: len(y_data_train_loss)],
                    y_data_train_loss,
                    pen=pg.mkPen(color=colors["train_loss"], width=2),
                    name="Strata treningowa",
                    symbol="o",
                )

            # Rysuj stratę walidacyjną
            if len(self.val_loss_data) > 0 and all(
                (v is None or isinstance(v, (int, float))) for v in self.val_loss_data
            ):
                x_val_loss, y_val_loss = [], []
                for i, val in enumerate(self.val_loss_data):
                    if val is not None and i < len(self.epochs):
                        x_val_loss.append(self.epochs[i])
                        y_val_loss.append(val)
                if len(x_val_loss) > 0:
                    self.plot_widget.plot(
                        np.array(x_val_loss),
                        np.array(y_val_loss),
                        pen=pg.mkPen(color=colors["val_loss"], width=2),
                        name="Strata walidacyjna",
                        symbol="o",
                    )

            # Rysuj dokładność treningową
            if len(self.train_acc_data) > 0 and all(
                isinstance(v, (int, float)) for v in self.train_acc_data
            ):
                y_data_train_acc = np.array(self.train_acc_data)
                self.plot_widget.plot(
                    x_data[: len(y_data_train_acc)],
                    y_data_train_acc,
                    pen=pg.mkPen(color=colors["train_acc"], width=2),
                    name="Dokładność treningowa",
                    symbol="o",
                )

            # Rysuj dokładność walidacyjną
            if len(self.val_acc_data) > 0 and all(
                (v is None or isinstance(v, (int, float))) for v in self.val_acc_data
            ):
                x_val_acc, y_val_acc = [], []
                for i, val in enumerate(self.val_acc_data):
                    if val is not None and i < len(self.epochs):
                        x_val_acc.append(self.epochs[i])
                        y_val_acc.append(val)
                if len(x_val_acc) > 0:
                    self.plot_widget.plot(
                        np.array(x_val_acc),
                        np.array(y_val_acc),
                        pen=pg.mkPen(color=colors["val_acc"], width=2),
                        name="Dokładność walidacyjna",
                        symbol="o",
                    )

            # Dostosuj widok do danych
            self.plot_widget.autoRange()

            # Resetuj flagę aktualizacji
            self.data_updated = False

        except Exception as e:
            import traceback

            print(f"Błąd w update_plot: {e}")
            print(traceback.format_exc())

    def update_data(self, epoch, train_loss, train_acc, val_loss=None, val_acc=None):
        """Aktualizuje dane wykresu."""
        try:
            # Konwersja i walidacja danych
            try:
                epoch = int(epoch)
                train_loss = float(train_loss) if train_loss is not None else None
                train_acc = float(train_acc) if train_acc is not None else None
                val_loss = float(val_loss) if val_loss is not None else None
                val_acc = float(val_acc) if val_acc is not None else None
            except (ValueError, TypeError) as e:
                print(f"BŁĄD konwersji danych: {e}")
                # Nie przerywamy, tylko ustawiamy wartości domyślne
                if train_loss is None or not isinstance(train_loss, (int, float)):
                    train_loss = 1.0  # Wartość domyślna
                if train_acc is None or not isinstance(train_acc, (int, float)):
                    train_acc = 0.5  # Wartość domyślna

            # Dodaj nowe dane tylko jeśli epoka jest dodatnia
            if epoch > 0:
                # Sprawdź czy ta epoka już istnieje
                if epoch in self.epochs:
                    # Znajdź indeks dla tej epoki
                    idx = self.epochs.index(epoch)
                    # Zaktualizuj istniejące dane
                    self.train_loss_data[idx] = train_loss
                    self.train_acc_data[idx] = train_acc
                    if val_loss is not None and idx < len(self.val_loss_data):
                        self.val_loss_data[idx] = val_loss
                    elif val_loss is not None:
                        # Rozszerz listę jeśli potrzeba
                        while len(self.val_loss_data) < idx:
                            self.val_loss_data.append(None)
                        self.val_loss_data.append(val_loss)

                    if val_acc is not None and idx < len(self.val_acc_data):
                        self.val_acc_data[idx] = val_acc
                    elif val_acc is not None:
                        # Rozszerz listę jeśli potrzeba
                        while len(self.val_acc_data) < idx:
                            self.val_acc_data.append(None)
                        self.val_acc_data.append(val_acc)
                else:
                    # Dodaj nowe dane na końcu list
                    self.epochs.append(epoch)
                    self.train_loss_data.append(train_loss)
                    self.train_acc_data.append(train_acc)
                    if val_loss is not None:
                        # Rozszerz listę val_loss_data jeśli potrzeba
                        while len(self.val_loss_data) < len(self.epochs) - 1:
                            self.val_loss_data.append(None)
                        self.val_loss_data.append(val_loss)

                    if val_acc is not None:
                        # Rozszerz listę val_acc_data jeśli potrzeba
                        while len(self.val_acc_data) < len(self.epochs) - 1:
                            self.val_acc_data.append(None)
                        self.val_acc_data.append(val_acc)

            # Oznacz, że dane zostały zaktualizowane
            self.data_updated = True

            # Ręczne wywołanie update_plot
            self.update_plot()

        except Exception as e:
            import traceback

            print(f"Błąd w update_data: {e}")
            print(traceback.format_exc())

    def clear_data(self):
        """Czyści wszystkie dane wykresu."""
        self.epochs = []
        self.train_loss_data = []
        self.train_acc_data = []
        self.val_loss_data = []
        self.val_acc_data = []
        self.data_updated = False
        self.update_plot()
