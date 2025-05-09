from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ClassList(QDialog):
    """Okno z listą klas i przyciskami do zarządzania zaznaczeniem."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Lista klas")
        self.setFixedSize(400, 800)
        self.setup_ui()

    def setup_ui(self):
        """Konfiguruje interfejs użytkownika."""
        # Główny layout
        main_layout = QVBoxLayout(self)

        # Lista klas
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        main_layout.addWidget(self.list_widget)

        # Przyciski zarządzania zaznaczeniem
        selection_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("Zaznacz wszystkie")
        self.select_all_btn.clicked.connect(self._select_all)
        self.deselect_all_btn = QPushButton("Odznacz wszystkie")
        self.deselect_all_btn.clicked.connect(self._deselect_all)
        selection_layout.addWidget(self.select_all_btn)
        selection_layout.addWidget(self.deselect_all_btn)
        main_layout.addLayout(selection_layout)

        # Przyciski akcji
        action_layout = QHBoxLayout()
        self.ok_btn = QPushButton("OK")
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("Anuluj")
        self.cancel_btn.clicked.connect(self.reject)
        action_layout.addWidget(self.ok_btn)
        action_layout.addWidget(self.cancel_btn)
        main_layout.addLayout(action_layout)

    def _select_all(self):
        """Zaznacza wszystkie elementy na liście."""
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            widget = self.list_widget.itemWidget(item)
            widget.findChild(QCheckBox).setChecked(True)

    def _deselect_all(self):
        """Odznacza wszystkie elementy na liście."""
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            widget = self.list_widget.itemWidget(item)
            widget.findChild(QCheckBox).setChecked(False)

    def set_items(self, items):
        """Ustawia elementy na liście.

        Args:
            items (dict): Słownik z numerami i nazwami klas
        """
        self.list_widget.clear()
        for class_num, class_name in items.items():
            list_item = QListWidgetItem()
            widget = QWidget()
            layout = QHBoxLayout(widget)
            layout.setContentsMargins(2, 2, 2, 2)

            checkbox = QCheckBox()
            checkbox.setText(f"{class_num}: {class_name}")
            layout.addWidget(checkbox)

            widget.setLayout(layout)
            list_item.setSizeHint(widget.sizeHint())

            self.list_widget.addItem(list_item)
            self.list_widget.setItemWidget(list_item, widget)

    def get_selected_items(self):
        """Zwraca listę zaznaczonych elementów.

        Returns:
            list: Lista zaznaczonych elementów
        """
        selected = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            widget = self.list_widget.itemWidget(item)
            checkbox = widget.findChild(QCheckBox)
            if checkbox.isChecked():
                # Usuń numer i dwukropek z początku tekstu
                text = checkbox.text()
                class_name = text[text.find(":") + 2 :].strip()
                selected.append(class_name)
        return selected
