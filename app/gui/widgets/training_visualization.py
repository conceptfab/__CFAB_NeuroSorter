import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget


class TrainingVisualization(QWidget):
    """Widget do wizualizacji procesu treningu w czasie rzeczywistym."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

        # Inicjalizacja danych
        self.train_loss_data = []
        self.train_acc_data = []
        self.val_loss_data = []
        self.val_acc_data = []
        self.epochs = []

        # Timer do automatycznego odświeżania
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_plot)
        self.update_timer.start(500)  # Odświeżaj co 500ms

        # Debug info
        self.debug_label = QLabel("Brak danych")
        self.layout().addWidget(self.debug_label)

        # Flaga wskazująca czy dane zostały zaktualizowane
        self.data_updated = False

    def setup_ui(self):
        """Konfiguruje interfejs użytkownika."""
        layout = QVBoxLayout(self)

        # Nagłówek
        header = QLabel("Wizualizacja treningu")
        header.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(header)

        # Kontrolki
        controls_layout = QHBoxLayout()

        # Wybór metryki
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["Strata", "Dokładność"])
        self.metric_combo.currentTextChanged.connect(self.update_plot)
        controls_layout.addWidget(QLabel("Metryka:"))
        controls_layout.addWidget(self.metric_combo)

        # Wybór zestawu danych
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["Trening", "Walidacja"])
        self.dataset_combo.currentTextChanged.connect(self.update_plot)
        controls_layout.addWidget(QLabel("Zestaw danych:"))
        controls_layout.addWidget(self.dataset_combo)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Wykres
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.addLegend()
        self.plot_widget.setLabel("left", "Wartość")
        self.plot_widget.setLabel("bottom", "Epoka")
        layout.addWidget(self.plot_widget)

        # Style dla linii epok
        self.epoch_line_pen = pg.mkPen(
            color=(100, 100, 100), style=Qt.PenStyle.DashLine, width=1
        )

    def update_plot(self):
        """Aktualizuje wykres na podstawie wybranej metryki i zestawu danych."""
        if not self.data_updated:
            return

        metric = self.metric_combo.currentText()
        dataset = self.dataset_combo.currentText()

        # Debug info
        debug_text = f"Epoki: {len(self.epochs)}\n"
        debug_text += f"Strata treningowa: {len(self.train_loss_data)} punktów\n"
        debug_text += f"Dokładność treningowa: {len(self.train_acc_data)} punktów\n"
        debug_text += f"Strata walidacyjna: {len(self.val_loss_data)} punktów\n"
        debug_text += f"Dokładność walidacyjna: {len(self.val_acc_data)} punktów\n\n"

        if self.epochs:
            debug_text += f"Ostatnia epoka: {self.epochs[-1]}\n"
            if self.train_loss_data:
                debug_text += (
                    f"Ostatnia strata treningowa: {self.train_loss_data[-1]:.4f}\n"
                )
            if self.train_acc_data:
                debug_text += (
                    f"Ostatnia dokładność treningowa: {self.train_acc_data[-1]:.4f}\n"
                )
            if self.val_loss_data:
                debug_text += (
                    f"Ostatnia strata walidacyjna: {self.val_loss_data[-1]:.4f}\n"
                )
            if self.val_acc_data:
                debug_text += (
                    f"Ostatnia dokładność walidacyjna: {self.val_acc_data[-1]:.4f}\n"
                )

        self.debug_label.setText(debug_text)

        # Sprawdź czy mamy dane do wyświetlenia
        if not self.epochs:
            print("DEBUG: Brak danych do wyświetlenia")
            return

        # Wyczyść wykres
        self.plot_widget.clear()

        # Przygotuj dane do wykresu
        x_data = np.array(self.epochs)

        # Dodaj legendę
        self.plot_widget.addLegend()

        # Rysuj dane treningowe
        if metric == "Strata":
            if self.train_loss_data:
                y_data = np.array(self.train_loss_data)
                print(f"DEBUG: Dane straty treningowej: {y_data}")
                self.plot_widget.plot(
                    x_data,
                    y_data,
                    pen=pg.mkPen(color="b", width=2),
                    name="Strata treningowa",
                    symbol="o",
                )
            if self.val_loss_data:
                y_data = np.array(self.val_loss_data)
                print(f"DEBUG: Dane straty walidacyjnej: {y_data}")
                self.plot_widget.plot(
                    x_data,
                    y_data,
                    pen=pg.mkPen(color="r", width=2),
                    name="Strata walidacyjna",
                    symbol="o",
                )
        else:  # Dokładność
            if self.train_acc_data:
                y_data = np.array(self.train_acc_data)
                print(f"DEBUG: Dane dokładności treningowej: {y_data}")
                self.plot_widget.plot(
                    x_data,
                    y_data,
                    pen=pg.mkPen(color="g", width=2),
                    name="Dokładność treningowa",
                    symbol="o",
                )
            if self.val_acc_data:
                y_data = np.array(self.val_acc_data)
                print(f"DEBUG: Dane dokładności walidacyjnej: {y_data}")
                self.plot_widget.plot(
                    x_data,
                    y_data,
                    pen=pg.mkPen(color="m", width=2),
                    name="Dokładność walidacyjna",
                    symbol="o",
                )

        # Rysuj linie epok
        for epoch in self.epochs:
            self.plot_widget.addItem(
                pg.InfiniteLine(
                    pos=epoch,
                    angle=90,
                    pen=self.epoch_line_pen,
                    label=f"Epoka {epoch}",
                )
            )

        # Dostosuj widok do danych
        self.plot_widget.autoRange()

        # Resetuj flagę aktualizacji
        self.data_updated = False

    def update_data(self, epoch, train_loss, train_acc, val_loss=None, val_acc=None):
        """Aktualizuje dane wykresu."""
        print(f"\nDEBUG update_data:")
        print(f"Epoka: {epoch}")
        print(f"Strata treningowa: {train_loss}")
        print(f"Dokładność treningowa: {train_acc}")
        print(f"Strata walidacyjna: {val_loss}")
        print(f"Dokładność walidacyjna: {val_acc}")

        # Sprawdź poprawność danych
        try:
            epoch = int(epoch)
            train_loss = float(train_loss)
            train_acc = float(train_acc)
            if val_loss is not None:
                val_loss = float(val_loss)
            if val_acc is not None:
                val_acc = float(val_acc)
        except (ValueError, TypeError) as e:
            print(f"BŁĄD konwersji danych: {e}")
            return

        # Sprawdź czy dane są sensowne
        if train_loss <= 0 or train_acc < 0 or train_acc > 1:
            print("BŁĄD: Nieprawidłowe wartości danych treningowych")
            return

        if val_loss is not None and val_loss <= 0:
            print("BŁĄD: Nieprawidłowa wartość straty walidacyjnej")
            return

        if val_acc is not None and (val_acc < 0 or val_acc > 1):
            print("BŁĄD: Nieprawidłowa wartość dokładności walidacyjnej")
            return

        # Dodaj nowe dane
        self.epochs.append(epoch)
        self.train_loss_data.append(train_loss)
        self.train_acc_data.append(train_acc)

        if val_loss is not None:
            self.val_loss_data.append(val_loss)
        if val_acc is not None:
            self.val_acc_data.append(val_acc)

        print("\nDEBUG: Stan danych po aktualizacji:")
        print(f"Epoki: {self.epochs}")
        print(f"Strata treningowa: {self.train_loss_data}")
        print(f"Dokładność treningowa: {self.train_acc_data}")
        print(f"Strata walidacyjna: {self.val_loss_data}")
        print(f"Dokładność walidacyjna: {self.val_acc_data}")

        # Oznacz, że dane zostały zaktualizowane i odśwież wykres
        self.data_updated = True
        self.update_plot()

        # Wymuś odświeżenie wykresu
        self.plot_widget.replot()

    def clear_data(self):
        """Czyści wszystkie dane wykresu."""
        self.epochs = []
        self.train_loss_data = []
        self.train_acc_data = []
        self.val_loss_data = []
        self.val_acc_data = []
        self.data_updated = False
        self.update_plot()

    def closeEvent(self, event):
        """Obsługuje zamknięcie widgetu."""
        # Zatrzymaj timer przed zamknięciem
        self.update_timer.stop()
        super().closeEvent(event)
