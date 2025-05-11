from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Classification:
    """Model danych dla klasyfikacji obrazu."""

    id: Optional[int]
    image_path: str
    class_id: int
    class_name: str
    confidence: float
    timestamp: datetime = datetime.now()


@dataclass
class Category:
    """Model danych dla kategorii."""

    id: Optional[int]
    name: str
    description: Optional[str] = None
