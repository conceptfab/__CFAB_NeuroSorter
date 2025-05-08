from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt


class QueueManager(QtWidgets.QDialog):
    def __init__(self, parent=None, settings=None):
        super().__init__(parent)
        self.setMinimumSize(1200, 800)

        # Layout główny
        main_layout = QtWidgets.QVBoxLayout(self)

        # Górna grupa: Lista zadań
        self.top_group = QtWidgets.QGroupBox("Lista zadań")
        top_layout = QtWidgets.QVBoxLayout(self.top_group)
        self.tasks_table = QtWidgets.QTableWidget()
        self.tasks_table.setColumnCount(3)
        self.tasks_table.setHorizontalHeaderLabels(["Nazwa", "Status", "Data"])
        top_layout.addWidget(self.tasks_table)
        main_layout.addWidget(self.top_group, stretch=1)

        # Dolna grupa: Wizualizacja treningu
        self.bottom_group = QtWidgets.QGroupBox("Wizualizacja treningu")
        bottom_layout = QtWidgets.QVBoxLayout(self.bottom_group)
        self.visualization_label = QtWidgets.QLabel(
            "[Tutaj będzie wizualizacja treningu]"
        )
        self.visualization_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bottom_layout.addWidget(self.visualization_label)
        main_layout.addWidget(self.bottom_group, stretch=1)
