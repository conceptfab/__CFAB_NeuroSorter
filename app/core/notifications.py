from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional


class NotificationType(Enum):
    """Typy powiadomień."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"


@dataclass
class Notification:
    """Klasa reprezentująca powiadomienie."""

    message: str
    type: NotificationType
    timestamp: datetime
    details: Optional[Dict] = None


class NotificationManager:
    """Klasa zarządzająca systemem powiadomień."""

    def __init__(self):
        """Inicjalizuje menedżer powiadomień."""
        self._notifications: List[Notification] = []
        self._callbacks: Dict[NotificationType, List[Callable]] = {
            NotificationType.INFO: [],
            NotificationType.WARNING: [],
            NotificationType.ERROR: [],
            NotificationType.SUCCESS: [],
        }

    def add_notification(
        self,
        message: str,
        type: NotificationType = NotificationType.INFO,
        details: Optional[Dict] = None,
    ):
        """Dodaje nowe powiadomienie.

        Args:
            message: Treść powiadomienia
            type: Typ powiadomienia
            details: Dodatkowe szczegóły
        """
        notification = Notification(
            message=message, type=type, timestamp=datetime.now(), details=details
        )

        self._notifications.append(notification)

        # Wywołaj callbacki dla danego typu
        for callback in self._callbacks[type]:
            callback(notification)

    def get_notifications(
        self, type: Optional[NotificationType] = None, limit: Optional[int] = None
    ) -> List[Notification]:
        """Zwraca listę powiadomień.

        Args:
            type: Filtr po typie powiadomienia
            limit: Maksymalna liczba powiadomień

        Returns:
            Lista powiadomień
        """
        notifications = self._notifications

        if type:
            notifications = [n for n in notifications if n.type == type]

        if limit:
            notifications = notifications[-limit:]

        return notifications

    def clear_notifications(self, type: Optional[NotificationType] = None):
        """Czyści powiadomienia.

        Args:
            type: Filtr po typie powiadomienia
        """
        if type:
            self._notifications = [n for n in self._notifications if n.type != type]
        else:
            self._notifications.clear()

    def register_callback(
        self, type: NotificationType, callback: Callable[[Notification], None]
    ):
        """Rejestruje callback dla danego typu powiadomienia.

        Args:
            type: Typ powiadomienia
            callback: Funkcja wywoływana przy nowym powiadomieniu
        """
        if callback not in self._callbacks[type]:
            self._callbacks[type].append(callback)

    def unregister_callback(
        self, type: NotificationType, callback: Callable[[Notification], None]
    ):
        """Usuwa callback dla danego typu powiadomienia.

        Args:
            type: Typ powiadomienia
            callback: Funkcja do usunięcia
        """
        if callback in self._callbacks[type]:
            self._callbacks[type].remove(callback)

    def get_notification_count(self, type: Optional[NotificationType] = None) -> int:
        """Zwraca liczbę powiadomień.

        Args:
            type: Filtr po typie powiadomienia

        Returns:
            Liczba powiadomień
        """
        if type:
            return sum(1 for n in self._notifications if n.type == type)
        return len(self._notifications)
