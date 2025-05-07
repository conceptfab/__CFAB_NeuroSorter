Tak, w pyqtgraph możliwe jest ustawienie zarówno grubości linii, jak i jej stylu (np. linia przerywana). Biblioteka pyqtgraph korzysta z funkcjonalności Qt do stylizowania linii.
Zmiana grubości linii i typu linii
Aby zmienić styl i grubość linii w pyqtgraph, możesz użyć funkcji mkPen():
pythonimport pyqtgraph as pg
from PyQt5 import QtCore  # lub PySide2.QtCore jeśli używasz PySide2

# Inicjalizacja okna wykresu
plt = pg.plot()

# Przykładowe dane
x = range(10)
y = [2, 4, 6, 8, 6, 4, 2, 4, 6, 8]

# Linia przerywana (DashLine) o szerokości 3 pikseli w kolorze czerwonym
dash_line = plt.plot(x, y, pen=pg.mkPen('r', width=3, style=QtCore.Qt.DashLine))

# Dodanie drugiej linii - kropkowanej (DotLine) o szerokości 2 pikseli w kolorze niebieskim
y2 = [8, 6, 4, 2, 4, 6, 8, 6, 4, 2]
dot_line = plt.plot(x, y2, pen=pg.mkPen(color='b', width=2, style=QtCore.Qt.DotLine))
Dostępne style linii w Qt
W Qt (a więc i w pyqtgraph) dostępne są następujące style linii:

Qt.SolidLine - linia ciągła
Qt.DashLine - linia przerywana (kreski)
Qt.DotLine - linia kropkowana
Qt.DashDotLine - linia kropkowo-kreskowa
Qt.DashDotDotLine - linia z kreską i dwiema kropkami Python GUIs

Przykład z linią nieskończoną (InfiniteLine)
Jeśli chcesz dodać pionową lub poziomą linię przerywaną, możesz użyć InfiniteLine:
python# Pozioma linia przerywana
h_line = pg.InfiniteLine(angle=0, movable=False, 
                        pen=pg.mkPen('y', width=2, style=QtCore.Qt.DotLine))
plot.addItem(h_line)

# Pionowa linia ciągła
v_line = pg.InfiniteLine(angle=90, movable=False, 
                        pen=pg.mkPen('g', width=4, style=QtCore.Qt.SolidLine))
plot.addItem(v_line)
Snyk Advisor
Przykład zmiany stylu istniejącej linii
Możesz również zmienić styl istniejącej linii używając metody addLine():
python# Dodawanie poziomej linii
line = widget.addLine(x=None, y=0.8, pen=pg.mkPen('r', width=3, style=QtCore.Qt.DashLine))
Stack Overflow
Pełny przykład kodu
Oto kompletny przykład pokazujący różne style linii:
pythonimport pyqtgraph as pg
from PyQt5 import QtCore, QtGui
import numpy as np

# Inicjalizacja aplikacji
app = pg.mkQApp("Style linii")

# Tworzenie wykresu
win = pg.GraphLayoutWidget(show=True, title="Przykłady stylów linii")
win.resize(800, 600)

# Dodanie wykresu
plot = win.addPlot(title="Style linii w pyqtgraph")
plot.showGrid(x=True, y=True)
plot.addLegend()

# Dane
x = np.arange(0, 10, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x + np.pi/4)

# Linie z różnymi stylami
line1 = plot.plot(x, y1, pen=pg.mkPen('r', width=2), name="Linia ciągła")
line2 = plot.plot(x, y2, pen=pg.mkPen('b', width=3, style=QtCore.Qt.DashLine), name="Linia przerywana")
line3 = plot.plot(x, y3, pen=pg.mkPen('g', width=2, style=QtCore.Qt.DotLine), name="Linia kropkowana")

# Dodanie linii nieskończonej
hLine = pg.InfiniteLine(pos=0, angle=0, movable=False, 
                       pen=pg.mkPen('y', width=2, style=QtCore.Qt.DashDotLine))
plot.addItem(hLine)

# Uruchomienie aplikacji
if __name__ == '__main__':
    pg.exec()
Warto zauważyć, że grubość linii określa się w pikselach parametrem width, a styl linii ustawia się za pomocą parametru style w funkcji mkPen(). Readthedocs
Jeśli potrzebujesz bardziej zaawansowanych stylów linii, możesz również korzystać z ustawień pen.setDashPattern() do definiowania własnych wzorów linii przerywanych.