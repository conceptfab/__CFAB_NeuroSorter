"""
Pakiet zawierający klasy zakładek interfejsu użytkownika.
"""

from .batch_processor import BatchProcessor
from .help_tab import HelpTab
from .image_classifier import ImageClassifierTab
from .model_manager import ModelManager
from .report_generator import ReportGenerator
from .settings_manager import SettingsManager
from .training_manager import TrainingManager

__all__ = [
    "BatchProcessor",
    "HelpTab",
    "ImageClassifierTab",
    "ModelManager",
    "ReportGenerator",
    "SettingsManager",
    "TrainingManager",
]
