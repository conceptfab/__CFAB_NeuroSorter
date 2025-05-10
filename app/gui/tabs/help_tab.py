from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.gui.tab_interface import TabInterface


class HelpTab(QWidget, TabInterface):
    """Klasa zarządzająca zakładką pomocy."""

    def __init__(self, parent=None, settings=None):
        super().__init__(parent)
        self.parent = parent
        self.settings = settings
        self.setup_ui()
        self.connect_signals()

    def setup_ui(self):
        """Tworzy i konfiguruje elementy interfejsu zakładki."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Nagłówek
        header = QLabel("POMOC")
        header.setStyleSheet(
            "font-weight: bold; color: #CCCCCC; font-size: 11px; padding-bottom: 4px;"
        )
        layout.addWidget(header)

        # Obszar przewijania dla treści pomocy
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("border: none;")

        # Widget zawierający treść pomocy
        help_content = QWidget()
        help_layout = QVBoxLayout(help_content)
        help_layout.setSpacing(16)

        # Dodaj WebView do wyświetlania dokumentacji HTML
        self.web_view = QWebEngineView()
        help_layout.addWidget(self.web_view)

        # Wczytaj dokumentację HTML
        try:
            import os

            doc_path = os.path.join(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                ),
                "resources",
                "help",
                "index.html",
            )
            if os.path.exists(doc_path):
                self.web_view.load(QUrl.fromLocalFile(doc_path))
            else:
                QMessageBox.warning(self, "Błąd", "Nie znaleziono pliku dokumentacji.")
        except Exception as e:
            QMessageBox.critical(
                self, "Błąd", f"Nie udało się załadować dokumentacji: {str(e)}"
            )

        # Ustaw zawartość obszaru przewijania
        scroll_area.setWidget(help_content)
        layout.addWidget(scroll_area)

        # Przyciski na dole
        buttons_layout = QHBoxLayout()

        self.feedback_btn = QPushButton("Zgłoś problem")
        self.feedback_btn.setFixedHeight(24)
        buttons_layout.addWidget(self.feedback_btn)

        buttons_layout.addStretch(1)
        layout.addLayout(buttons_layout)

    def connect_signals(self):
        """Podłącza sygnały do slotów."""
        self.feedback_btn.clicked.connect(self._open_feedback)

    def _open_feedback(self):
        """Otwiera formularz zgłaszania problemów."""
        # TODO: Implementacja formularza zgłaszania problemów
        pass

    def refresh(self):
        """Odświeża zawartość zakładki."""
        pass

    def update_settings(self, settings):
        """Aktualizuje ustawienia zakładki."""
        self.settings = settings

    def save_state(self):
        """Zapisuje stan zakładki."""
        return {}

    def restore_state(self, state):
        """Przywraca zapisany stan zakładki."""
        pass
