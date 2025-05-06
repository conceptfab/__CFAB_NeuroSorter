from PyQt6.QtWidgets import QWidget


class TabInterface:
    """Interfejs dla klas zarządzających zakładkami."""

    def setup_ui(self):
        """Tworzy i konfiguruje elementy interfejsu zakładki."""
        raise NotImplementedError("Metoda musi być zaimplementowana w klasie pochodnej")

    def connect_signals(self):
        """Podłącza sygnały do slotów."""
        raise NotImplementedError("Metoda musi być zaimplementowana w klasie pochodnej")

    def refresh(self):
        """Odświeża zawartość zakładki."""
        raise NotImplementedError("Metoda musi być zaimplementowana w klasie pochodnej")

    def update_settings(self, settings):
        """Aktualizuje ustawienia zakładki."""
        raise NotImplementedError("Metoda musi być zaimplementowana w klasie pochodnej")

    def save_state(self):
        """Zapisuje stan zakładki."""
        raise NotImplementedError("Metoda musi być zaimplementowana w klasie pochodnej")

    def restore_state(self):
        """Przywraca zapisany stan zakładki."""
        raise NotImplementedError("Metoda musi być zaimplementowana w klasie pochodnej")
