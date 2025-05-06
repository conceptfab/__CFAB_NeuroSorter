import datetime
import glob
import json
import os

import torch
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.core.workers.batch_training_thread import BatchTrainingThread
from app.gui.tab_interface import TabInterface
from app.gui.widgets.training_visualization import TrainingVisualization
from app.utils.config import DEFAULT_TRAINING_PARAMS
from app.utils.file_utils import (
    validate_task_config,
    validate_task_file,
    validate_training_directory,
    validate_validation_directory,
)


class TrainingManager(QWidget, TabInterface):
    """Klasa zarządzająca zakładką treningu."""

    def __init__(self, parent=None, settings=None):
        super().__init__(parent)
        self.parent = parent
        self.settings = settings
        self.training_thread = None
        self.setup_ui()
        self.connect_signals()
        # Automatyczne odświeżenie listy zadań przy starcie
        self.refresh()

    def setup_ui(self):
        """Tworzy i konfiguruje elementy interfejsu zakładki."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Dodaj etykietę informacji o profilu sprzętowym
        profile_layout = QHBoxLayout()

        # Lewa kolumna - informacje o profilu
        profile_info_panel = QWidget()
        profile_info_layout = QVBoxLayout(profile_info_panel)
        profile_info_layout.setContentsMargins(0, 0, 0, 0)

        # Nagłówek sekcji
        optimization_header = QLabel("OPTYMALIZACJA TRENINGU")
        optimization_header.setStyleSheet(
            "font-weight: bold; color: #CCCCCC; "
            "font-size: 11px; padding-bottom: 4px;"
        )
        profile_info_layout.addWidget(optimization_header)

        self.profile_info_label = QLabel("Status profilu: Nieznany")
        self.profile_info_label.setStyleSheet("color: #CCCCCC; padding: 4px;")
        profile_info_layout.addWidget(self.profile_info_label)

        # Prawa kolumna - przyciski i opcje
        profile_controls_panel = QWidget()
        profile_controls_layout = QVBoxLayout(profile_controls_panel)
        profile_controls_layout.setContentsMargins(0, 0, 0, 0)

        # Checkbox optymalizacji
        self.use_optimization_checkbox = QCheckBox("Używaj optymalizacji sprzętowej")
        self.use_optimization_checkbox.setChecked(True)
        self.use_optimization_checkbox.setToolTip(
            "Automatycznie dobiera parametry treningu "
            "na podstawie profilu sprzętowego"
        )
        profile_controls_layout.addWidget(self.use_optimization_checkbox)

        # Przycisk profilowania
        self.run_profiler_btn = QPushButton("Uruchom profilowanie sprzętu")
        self.run_profiler_btn.setFixedHeight(24)
        self.run_profiler_btn.clicked.connect(self._run_profiler)
        profile_controls_layout.addWidget(self.run_profiler_btn)

        # Dodaj obie kolumny do głównego layoutu
        profile_layout.addWidget(profile_info_panel)
        profile_layout.addWidget(profile_controls_panel)
        layout.addLayout(profile_layout)

        self._create_add_task_panel(layout)

        # === Początek zmian: Dodanie QTabWidget ===
        self.tabs = QTabWidget()

        # Zakładka 1: Kolejka zadań treningowych
        queue_panel_widget = self._create_queue_panel_widget()
        self.tabs.addTab(queue_panel_widget, "Kolejka zadań treningowych")

        # Zakładka 2: Wizualizacja treningu
        self.training_visualization = TrainingVisualization(
            parent=self, settings=self.settings
        )
        self.tabs.addTab(self.training_visualization, "Wizualizacja treningu")

        layout.addWidget(
            self.tabs, 1
        )  # Dodajemy QTabWidget do głównego layoutu, rozciągalny
        # === Koniec zmian: Dodanie QTabWidget ===

        # Stara implementacja (przed QTabWidget) - zakomentowana lub usunięta
        # # Panel wizualizacji treningu
        # self.training_visualization = TrainingVisualization()
        # layout.addWidget(self.training_visualization)
        #
        # # Panel kolejki zadań treningowych (skalowalny)
        # queue_panel = self._create_queue_panel_widget()
        # layout.addWidget(queue_panel, 1)  # <- skalowalny na wysokość

    def connect_signals(self):
        """Podłącza sygnały do slotów."""
        self.add_task_btn.clicked.connect(self._add_training_task)
        self.refresh_queue_btn.clicked.connect(self.refresh)
        self.start_queue_btn.clicked.connect(self._start_task_queue)
        self.clear_queue_btn.clicked.connect(self._clear_task_queue)

    def refresh(self):
        """Odświeża listę zadań w kolejce."""
        self._refresh_task_queue()

    def _refresh_task_queue(self):
        """Odświeża listę zadań treningowych w kolejce."""
        try:
            # Wyczyść tabelę
            self.tasks_table.setRowCount(0)

            # Katalog z zadaniami
            tasks_dir = os.path.join("data", "tasks")
            os.makedirs(tasks_dir, exist_ok=True)

            # Pobierz pliki zadań
            task_files = sorted(glob.glob(os.path.join(tasks_dir, "*.json")))

            if not task_files:
                return

            # Dodaj zadania do tabeli
            for task_file in task_files:
                try:
                    with open(task_file, "r", encoding="utf-8") as f:
                        task_data = json.load(f)

                    # Dodaj wiersz do tabeli
                    row = self.tasks_table.rowCount()
                    self.tasks_table.insertRow(row)
                    self.tasks_table.setRowHeight(row, 38)

                    # Nazwa zadania
                    task_name = task_data.get("name", os.path.basename(task_file))
                    self.tasks_table.setItem(row, 0, QTableWidgetItem(task_name))

                    # Typ zadania
                    task_type = task_data.get("type", "Trening")
                    self.tasks_table.setItem(row, 1, QTableWidgetItem(task_type))

                    # Status
                    status = task_data.get("status", "Nowy")
                    self.tasks_table.setItem(row, 2, QTableWidgetItem(status))

                    # Priorytet
                    priority = task_data.get("priority", 0)
                    self.tasks_table.setItem(row, 3, QTableWidgetItem(str(priority)))

                    # Data utworzenia
                    created_at = task_data.get("created_at", "")
                    self.tasks_table.setItem(row, 4, QTableWidgetItem(created_at))

                    # Przyciski akcji
                    action_widget = QWidget()
                    action_layout = QHBoxLayout(action_widget)
                    action_layout.setContentsMargins(0, 0, 0, 0)
                    action_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)

                    # Przycisk uruchomienia
                    run_btn = QPushButton("Uruchom")
                    run_btn.setFixedWidth(80)
                    run_btn.setFixedHeight(20)
                    run_btn.clicked.connect(
                        lambda checked, file=task_file: self._run_task_from_queue(file)
                    )

                    # Przycisk edycji
                    edit_btn = QPushButton("Edytuj")
                    edit_btn.setFixedWidth(80)
                    edit_btn.setFixedHeight(20)
                    edit_btn.clicked.connect(
                        lambda checked, file=task_file: self._edit_task_from_queue(file)
                    )

                    # Przycisk usunięcia
                    delete_btn = QPushButton("Usuń")
                    delete_btn.setFixedWidth(80)
                    delete_btn.setFixedHeight(20)
                    delete_btn.clicked.connect(
                        lambda checked, file=task_file: self._delete_task_from_queue(
                            file
                        )
                    )

                    action_layout.addWidget(run_btn)
                    action_layout.addWidget(edit_btn)
                    action_layout.addWidget(delete_btn)
                    action_layout.addStretch()

                    self.tasks_table.setCellWidget(row, 10, action_widget)

                    # Dodaj wyniki treningu jeśli zadanie jest zakończone
                    if status == "Zakończony":
                        # Czas treningu
                        training_time = task_data.get("training_time_str", "")
                        self.tasks_table.setItem(
                            row, 5, QTableWidgetItem(training_time)
                        )

                        # Dokładność treningu
                        train_acc = task_data.get("train_accuracy", 0)
                        self.tasks_table.setItem(
                            row, 6, QTableWidgetItem(f"{train_acc:.2f}")
                        )

                        # Strata treningu
                        train_loss = task_data.get("train_loss", 0)
                        self.tasks_table.setItem(
                            row, 7, QTableWidgetItem(f"{train_loss:.2f}")
                        )

                        # Dokładność walidacji
                        val_acc = task_data.get("validation_accuracy", 0)
                        self.tasks_table.setItem(
                            row, 8, QTableWidgetItem(f"{val_acc:.2f}")
                        )

                        # Strata walidacji
                        val_loss = task_data.get("validation_loss", 0)
                        self.tasks_table.setItem(
                            row, 9, QTableWidgetItem(f"{val_loss:.2f}")
                        )

                except Exception as e:
                    self.parent.logger.error(
                        f"Błąd podczas wczytywania zadania {task_file}: {str(e)}"
                    )

            # Dostosuj szerokość kolumn
            self.tasks_table.resizeColumnsToContents()
            header = self.tasks_table.horizontalHeader()
            for col in range(self.tasks_table.columnCount()):
                width = self.tasks_table.columnWidth(col)
                if col == 0:  # Kolumna "Nazwa"
                    new_width = int(width * 1.15)
                elif col == 5:  # Kolumna "Czas treningu"
                    new_width = int(width * 1.4)
                elif col in [6, 7, 8, 9]:  # Kolumny z wynikami treningu
                    new_width = int(width * 1.0)
                else:
                    new_width = int(width * 1.6)
                self.tasks_table.setColumnWidth(col, new_width)
            # Zmniejsz kolumnę 'Akcje' (ostatnia) do odpowiedniej szerokości
            last_col = self.tasks_table.columnCount() - 1
            self.tasks_table.setColumnWidth(last_col, 250)
            header.setStretchLastSection(False)

        except Exception as e:
            self.parent.logger.error(
                f"Błąd podczas odświeżania kolejki zadań: {str(e)}"
            )

    def _create_add_task_panel(self, parent_layout):
        """Tworzy panel dodawania nowego zadania treningowego."""
        add_task_panel = QWidget()
        add_task_layout = QVBoxLayout(add_task_panel)
        add_task_layout.setContentsMargins(0, 0, 0, 0)

        # Nagłówek sekcji
        add_task_header = QLabel("DODAJ NOWE ZADANIE TRENINGOWE")
        add_task_header.setStyleSheet(
            "font-weight: bold; color: #CCCCCC; "
            "font-size: 11px; padding-bottom: 4px;"
        )
        add_task_layout.addWidget(add_task_header)

        # Wybór typu zadania
        task_type_layout = QHBoxLayout()
        task_type_label = QLabel("Typ zadania:")
        task_type_label.setFixedWidth(120)
        self.task_type_combo = QComboBox()
        self.task_type_combo.addItems(
            ["Trening nowego modelu", "Doszkalanie istniejącego modelu"]
        )

        task_type_layout.addWidget(task_type_label)
        task_type_layout.addWidget(self.task_type_combo)
        add_task_layout.addLayout(task_type_layout)

        # Model architektury
        self.model_arch_combo = QComboBox()
        self.model_arch_combo.addItems(
            ["resnet50", "efficientnet", "mobilenet", "vit", "convnext"]
        )

        # Liczba epok
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(30)

        # Model bazowy dla doszkalania
        self.base_model_combo = QComboBox()
        models_dir = os.path.join("data", "models")
        if os.path.exists(models_dir):
            for model in os.listdir(models_dir):
                if model.endswith(".h5"):
                    self.base_model_combo.addItem(model)

        # Przycisk dodania zadania
        self.add_task_btn = QPushButton("Dodaj zadanie do kolejki")
        self.add_task_btn.clicked.connect(self._add_training_task)
        self.add_task_btn.setFixedHeight(24)
        add_task_layout.addWidget(self.add_task_btn)

        parent_layout.addWidget(add_task_panel)

    def _create_queue_panel(self, parent_layout):
        """Zachowujemy dla kompatybilności, ale nie używamy już tej metody do dodawania panelu."""
        pass

    def _create_queue_panel_widget(self):
        """Tworzy i zwraca panel kolejki zadań treningowych jako widget."""
        queue_panel = QWidget()
        queue_layout = QVBoxLayout(queue_panel)
        queue_layout.setContentsMargins(0, 0, 0, 0)
        queue_layout.setSpacing(8)

        # Nagłówek sekcji
        queue_header = QLabel("KOLEJKA ZADAŃ TRENINGOWYCH")
        queue_header.setStyleSheet(
            "font-weight: bold; color: #CCCCCC; "
            "font-size: 11px; padding-bottom: 4px;"
        )
        queue_layout.addWidget(queue_header)

        # Tabela z zadaniami
        self.tasks_table = QTableWidget()
        self.tasks_table.setColumnCount(11)
        self.tasks_table.setHorizontalHeaderLabels(
            [
                "Nazwa",
                "Typ",
                "Status",
                "Priorytet",
                "Utworzono",
                "Czas treningu",
                "Dokładność treningu",
                "Strata treningu",
                "Dokładność walidacji",
                "Strata walidacji",
                "Akcje",
            ]
        )
        self.tasks_table.horizontalHeader().setStretchLastSection(True)
        self.tasks_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.tasks_table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        queue_layout.addWidget(self.tasks_table, 1)

        # Przyciski zarządzania kolejką
        buttons_container = QWidget()
        buttons_layout = QHBoxLayout(buttons_container)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(8)

        self.refresh_queue_btn = QPushButton("Odśwież kolejkę")
        self.refresh_queue_btn.clicked.connect(self.refresh)
        self.refresh_queue_btn.setFixedHeight(24)
        buttons_layout.addWidget(self.refresh_queue_btn)

        self.start_queue_btn = QPushButton("Uruchom kolejkę")
        self.start_queue_btn.clicked.connect(self._start_task_queue)
        self.start_queue_btn.setFixedHeight(24)
        buttons_layout.addWidget(self.start_queue_btn)

        self.clear_queue_btn = QPushButton("Wyczyść kolejkę")
        self.clear_queue_btn.clicked.connect(self._clear_task_queue)
        self.clear_queue_btn.setFixedHeight(24)
        buttons_layout.addWidget(self.clear_queue_btn)

        buttons_layout.addStretch(1)
        queue_layout.addWidget(buttons_container)

        return queue_panel

    def _create_progress_panel(self, parent_layout):
        """Tworzy panel postępu zadania."""
        # Usuwamy całą sekcję POSTĘP ZADANIA, ponieważ została przeniesiona do głównego okna
        pass

    def _add_training_task(self):
        """Dodaje nowe zadanie treningowe do kolejki."""
        try:
            # Pobierz typ zadania
            task_type_index = self.task_type_combo.currentIndex()
            task_type = "Trening" if task_type_index == 0 else "Doszkalanie"

            # Wyświetl dialog konfiguracji w zależności od typu zadania
            if task_type == "Trening":
                self._configure_training_task()
            else:
                self._configure_finetuning_task()

        except Exception as e:
            QMessageBox.critical(
                self, "Błąd", f"Nie udało się dodać zadania treningowego: {str(e)}"
            )

    def _configure_training_task(self):
        """Konfiguruje zadanie treningu nowego modelu."""
        try:
            self.parent.logger.info(f"Rozpoczęto konfigurację zadania treningowego")

            # Dialog konfiguracji
            dialog = QDialog(self)
            dialog.setWindowTitle("Konfiguracja zadania treningu")
            dialog.setMinimumWidth(500)
            layout = QVBoxLayout(dialog)
            form_layout = QFormLayout()

            # Wykorzystaj profil sprzętowy do ustawienia domyślnych wartości
            optimal_batch_size = DEFAULT_TRAINING_PARAMS["batch_size"]
            optimal_workers = DEFAULT_TRAINING_PARAMS["num_workers"]
            optimal_mixed_precision = DEFAULT_TRAINING_PARAMS["use_mixed_precision"]
            optimal_learning_rate = DEFAULT_TRAINING_PARAMS["learning_rate"]
            optimal_optimizer = DEFAULT_TRAINING_PARAMS["optimizer"]
            optimal_scheduler = DEFAULT_TRAINING_PARAMS["scheduler"]

            # Jeśli mamy dostęp do profilu sprzętowego, użyj jego wartości
            if (
                hasattr(self.parent, "hardware_profile")
                and self.parent.hardware_profile
            ):
                profile = self.parent.hardware_profile
                optimal_batch_size = profile.get(
                    "recommended_batch_size", optimal_batch_size
                )
                optimal_workers = profile.get("recommended_workers", optimal_workers)
                optimal_mixed_precision = profile.get(
                    "use_mixed_precision", optimal_mixed_precision
                )
                optimal_learning_rate = profile.get(
                    "learning_rate", optimal_learning_rate
                )
                optimal_optimizer = profile.get("optimizer", optimal_optimizer)
                optimal_scheduler = profile.get("scheduler", optimal_scheduler)

                self.parent.logger.info(
                    f"Użyto optymalnych wartości z profilu: batch_size={optimal_batch_size}, "
                    f"workers={optimal_workers}, mixed_precision={optimal_mixed_precision}"
                )

            # 1. Ścieżka do danych treningowych
            data_dir_layout = QHBoxLayout()
            data_dir_edit = QLineEdit()
            data_dir_button = QPushButton("Przeglądaj...")
            data_dir_button.clicked.connect(
                lambda: self._select_directory(data_dir_edit)
            )
            data_dir_layout.addWidget(data_dir_edit)
            data_dir_layout.addWidget(data_dir_button)
            form_layout.addRow("Katalog danych treningowych:", data_dir_layout)

            # 2. Ścieżka do danych walidacyjnych (opcjonalnie)
            val_dir_layout = QHBoxLayout()
            val_dir_edit = QLineEdit()
            val_dir_button = QPushButton("Przeglądaj...")
            val_dir_button.clicked.connect(lambda: self._select_directory(val_dir_edit))
            val_dir_layout.addWidget(val_dir_edit)
            val_dir_layout.addWidget(val_dir_button)
            form_layout.addRow(
                "Katalog danych walidacyjnych (opcjonalnie):", val_dir_layout
            )

            # 3. Architektura modelu
            model_arch_combo = QComboBox()
            model_arch_combo.addItems(
                ["resnet50", "efficientnet", "mobilenet", "vit", "convnext"]
            )

            # Sprawdź rekomendowany model z profilu
            if (
                hasattr(self.parent, "hardware_profile")
                and self.parent.hardware_profile
            ):
                additional_recommendations = self.parent.hardware_profile.get(
                    "additional_recommendations", {}
                )
                if isinstance(additional_recommendations, str):
                    try:
                        additional_recommendations = json.loads(
                            additional_recommendations
                        )
                    except:
                        additional_recommendations = {}

                recommended_model = additional_recommendations.get(
                    "recommended_model", ""
                )

                # Mapowanie rekomendacji na dostępne modele
                if "convnext" in recommended_model.lower():
                    model_arch_combo.setCurrentText("convnext")
                elif "vit" in recommended_model.lower():
                    model_arch_combo.setCurrentText("vit")
                elif "efficient" in recommended_model.lower():
                    model_arch_combo.setCurrentText("efficientnet")
                elif "mobile" in recommended_model.lower():
                    model_arch_combo.setCurrentText("mobilenet")
                else:
                    model_arch_combo.setCurrentText("resnet50")

            form_layout.addRow("Architektura modelu:", model_arch_combo)

            # 4. Liczba epok
            epochs_spin = QSpinBox()
            epochs_spin.setRange(1, 1000)
            epochs_spin.setValue(DEFAULT_TRAINING_PARAMS["max_epochs"])
            form_layout.addRow("Liczba epok:", epochs_spin)

            # 5. Rozmiar wsadu
            batch_size_spin = QSpinBox()
            batch_size_spin.setRange(1, 256)
            batch_size_spin.setValue(optimal_batch_size)
            form_layout.addRow("Rozmiar wsadu:", batch_size_spin)

            # 6. Współczynnik uczenia
            learning_rate_combo = QComboBox()
            learning_rate_combo.addItems(["0.1", "0.01", "0.001", "0.0001"])
            learning_rate_combo.setCurrentText("0.001")  # Domyślna wartość
            self.parent.logger.info("Ustawiono domyślną wartość learning rate: 0.001")
            form_layout.addRow("Współczynnik uczenia:", learning_rate_combo)

            # 7. Optymalizator
            optimizer_combo = QComboBox()
            optimizer_combo.addItems(["Adam", "SGD", "RMSprop", "AdamW"])
            optimizer_combo.setCurrentText(optimal_optimizer)
            form_layout.addRow("Optymalizator:", optimizer_combo)

            # 8. Harmonogram uczenia
            scheduler_combo = QComboBox()
            scheduler_combo.addItems(
                ["None", "StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"]
            )
            scheduler_combo.setCurrentText(optimal_scheduler)
            form_layout.addRow("Harmonogram uczenia:", scheduler_combo)

            # 9. Liczba wątków do ładowania danych
            num_workers_spin = QSpinBox()
            num_workers_spin.setRange(0, 32)

            # Pobierz zalecaną liczbę workerów z profilu
            if (
                hasattr(self.parent, "hardware_profile")
                and self.parent.hardware_profile
            ):
                profile = self.parent.hardware_profile
                recommended_workers = profile.get("recommended_workers", 2)
                num_workers_spin.setValue(recommended_workers)
                num_workers_spin.setToolTip(
                    f"Zalecana wartość z profilu sprzętowego: {recommended_workers}"
                )
            else:
                num_workers_spin.setValue(2)
                num_workers_spin.setToolTip(
                    "Brak profilu sprzętowego - używana wartość domyślna: 2"
                )

            form_layout.addRow("Liczba wątków do ładowania danych:", num_workers_spin)

            # 10. Współczynnik regularyzacji L2
            weight_decay_combo = QComboBox()
            weight_decay_combo.addItems(["1e-3", "1e-4", "1e-5", "1e-6"])
            weight_decay_combo.setCurrentText("1e-4")  # Domyślna wartość
            form_layout.addRow("Współczynnik regularyzacji L2:", weight_decay_combo)

            # 11. Wartość przycinania gradientów
            gradient_clip_spin = QDoubleSpinBox()
            gradient_clip_spin.setRange(0.0, 1.0)
            gradient_clip_spin.setValue(DEFAULT_TRAINING_PARAMS["gradient_clip_val"])
            gradient_clip_spin.setDecimals(2)
            gradient_clip_spin.setSingleStep(0.01)
            form_layout.addRow("Wartość przycinania gradientów:", gradient_clip_spin)

            # 12. Liczba epok bez poprawy przed wczesnym zatrzymaniem
            early_stopping_spin = QSpinBox()
            early_stopping_spin.setRange(1, 50)
            early_stopping_spin.setValue(
                DEFAULT_TRAINING_PARAMS["early_stopping_patience"]
            )
            form_layout.addRow(
                "Liczba epok bez poprawy przed zatrzymaniem:", early_stopping_spin
            )

            # Augmentacja danych
            augmentation_group = QGroupBox("Augmentacja danych")
            augmentation_layout = QFormLayout(augmentation_group)

            # Wykorzystaj rekomendowany poziom augmentacji z profilu
            basic_aug_check = QCheckBox("Podstawowa augmentacja")
            advanced_aug_check = QCheckBox("Zaawansowana augmentacja")

            if (
                hasattr(self.parent, "hardware_profile")
                and self.parent.hardware_profile
            ):
                additional_recommendations = self.parent.hardware_profile.get(
                    "additional_recommendations", {}
                )
                if isinstance(additional_recommendations, str):
                    try:
                        additional_recommendations = json.loads(
                            additional_recommendations
                        )
                    except:
                        additional_recommendations = {}

                aug_level = additional_recommendations.get(
                    "recommended_augmentation", "basic"
                )
                if aug_level == "high":
                    basic_aug_check.setChecked(True)
                    advanced_aug_check.setChecked(True)
                elif aug_level == "medium":
                    basic_aug_check.setChecked(True)
                    advanced_aug_check.setChecked(False)
                else:
                    basic_aug_check.setChecked(False)
                    advanced_aug_check.setChecked(False)
            else:
                basic_aug_check.setChecked(True)  # Domyślnie włączona podstawowa
                advanced_aug_check.setChecked(False)

            augmentation_layout.addRow("", basic_aug_check)
            augmentation_layout.addRow("", advanced_aug_check)

            # 14. Mixed Precision
            mixed_precision_check = QCheckBox("Używaj mixed precision")
            mixed_precision_check.setChecked(optimal_mixed_precision)
            mixed_precision_check.setEnabled(torch.cuda.is_available())
            form_layout.addRow("", mixed_precision_check)

            # Dodaj wszystkie elementy do głównego layoutu
            layout.addLayout(form_layout)
            layout.addWidget(augmentation_group)

            # Przyciski OK i Anuluj
            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok
                | QDialogButtonBox.StandardButton.Cancel
            )
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            layout.addWidget(buttons)

            # Wyświetl dialog i obsłuż tylko jednokrotnie sygnał accepted
            buttons.accepted.disconnect()  # Usuń domyślne połączenie
            buttons.accepted.connect(
                lambda: self._handle_dialog_accept(
                    dialog,
                    data_dir_edit,
                    val_dir_edit,
                    model_arch_combo,
                    epochs_spin,
                    batch_size_spin,
                    learning_rate_combo,
                    optimizer_combo,
                    scheduler_combo,
                    num_workers_spin,
                    weight_decay_combo,
                    gradient_clip_spin,
                    early_stopping_spin,
                    basic_aug_check,
                    advanced_aug_check,
                    mixed_precision_check,
                )
            )

            # Zabezpieczenie przed wielokrotnym wywołaniem tego samego okna dialogowego
            self.dialog_active = True

            result = dialog.exec()
            self.dialog_active = False

            # Metoda nie zwraca konfiguracji, bo jest obsługiwana w _handle_dialog_accept
            return None

        except Exception as e:
            self.parent.logger.error(
                f"Błąd podczas konfiguracji zadania treningowego: {str(e)}"
            )
            QMessageBox.critical(self, "Błąd", f"Wystąpił nieoczekiwany błąd: {str(e)}")
            return None

    def _handle_dialog_accept(
        self,
        dialog,
        data_dir_edit,
        val_dir_edit,
        model_arch_combo,
        epochs_spin,
        batch_size_spin,
        learning_rate_combo,
        optimizer_combo,
        scheduler_combo,
        num_workers_spin,
        weight_decay_combo,
        gradient_clip_spin,
        early_stopping_spin,
        basic_aug_check,
        advanced_aug_check,
        mixed_precision_check,
    ):
        """Obsługuje akceptację dialogu bez ryzyka podwójnego wywołania."""
        self.parent.logger.info("Użytkownik zaakceptował konfigurację zadania")

        # Pobierz wartości z formularza
        data_dir = data_dir_edit.text()
        val_dir = val_dir_edit.text()
        model_arch = model_arch_combo.currentText()
        epochs = epochs_spin.value()
        batch_size = batch_size_spin.value()

        # Loguj konfigurację
        self.parent.logger.info(f"Konfiguracja zadania treningu:")
        self.parent.logger.info(f"- Architektura modelu: {model_arch}")
        self.parent.logger.info(f"- Liczba epok: {epochs}")
        self.parent.logger.info(f"- Katalog treningowy: {data_dir}")
        self.parent.logger.info(f"- Katalog walidacyjny: {val_dir}")

        # Generuj nazwę pliku zadania
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        task_name = f"{model_arch}_{epochs}epok_{timestamp}.json"

        # Przygotuj konfigurację zadania
        task_config = {
            "name": task_name,
            "type": "Trening",
            "status": "Nowy",
            "priority": 0,
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "model_arch": model_arch,
                "data_dir": data_dir,
                "val_dir": val_dir,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": float(learning_rate_combo.currentText()),
                "optimizer": optimizer_combo.currentText(),
                "scheduler": scheduler_combo.currentText(),
                "use_mixed_precision": mixed_precision_check.isChecked(),
                "num_workers": num_workers_spin.value(),
                "weight_decay": float(weight_decay_combo.currentText()),
                "gradient_clip": gradient_clip_spin.value(),
                "early_stopping": early_stopping_spin.value(),
                "augmentation": {
                    "basic": basic_aug_check.isChecked(),
                    "advanced": advanced_aug_check.isChecked(),
                },
            },
        }

        # Loguj pełną konfigurację
        self.parent.logger.info(
            f"Pełna konfiguracja zadania: {json.dumps(task_config, indent=2)}"
        )

        # Zapisz konfigurację zadania
        tasks_dir = os.path.join("data", "tasks")
        os.makedirs(tasks_dir, exist_ok=True)
        task_file = os.path.join(tasks_dir, task_name)

        self.parent.logger.info(f"Zapisywanie pliku zadania: {task_file}")
        try:
            # Sprawdź czy katalog istnieje
            os.makedirs(os.path.dirname(task_file), exist_ok=True)

            # Zapisz plik
            with open(task_file, "w", encoding="utf-8") as f:
                json.dump(task_config, f, indent=4, ensure_ascii=False)

            # Sprawdź czy plik został poprawnie zapisany
            if not os.path.exists(task_file) or os.path.getsize(task_file) == 0:
                raise IOError("Plik zadania nie został poprawnie zapisany")

            # Walidacja zapisanego pliku
            is_valid, error_msg = validate_task_file(task_file)
            if not is_valid:
                self.parent.logger.error(f"Błąd walidacji pliku zadania: {error_msg}")
                os.remove(task_file)  # Usuń nieprawidłowy plik
                QMessageBox.critical(
                    self,
                    "Błąd walidacji",
                    f"Plik zadania nie przeszedł walidacji: {error_msg}",
                )
                return None

            self.parent.logger.info(f"Zapisano plik zadania: {task_file}")

        except Exception as e:
            self.parent.logger.error(
                f"Błąd podczas zapisywania pliku zadania: {str(e)}"
            )
            if os.path.exists(task_file):
                os.remove(task_file)  # Usuń uszkodzony plik
            QMessageBox.critical(
                self, "Błąd zapisu", f"Nie udało się zapisać pliku zadania: {str(e)}"
            )
            return None

        # Wyświetl potwierdzenie
        QMessageBox.information(
            self,
            "Zadanie utworzone",
            f"Zadanie treningowe '{task_name}' zostało dodane do kolejki.",
        )

        # Odśwież listę zadań po zapisaniu
        self.refresh()

        # Zamknij dialog
        dialog.accept()

    def _select_directory(self, line_edit_widget):
        """Otwiera dialog wyboru katalogu i ustawia ścieżkę w QLineEdit."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Wybierz katalog",
            # Użyj bieżącej ścieżki z QLineEdit jako startowej, jeśli istnieje
            (
                line_edit_widget.text()
                if line_edit_widget.text()
                else os.path.expanduser("~")
            ),
            QFileDialog.Option.ShowDirsOnly,
        )
        if directory:
            line_edit_widget.setText(directory)

    def _configure_finetuning_task(self):
        """Konfiguruje zadanie doszkalania istniejącego modelu."""
        try:
            # Dialog konfiguracji
            dialog = QDialog(self)
            dialog.setWindowTitle("Konfiguracja zadania doszkalania")
            dialog.setMinimumWidth(500)
            layout = QVBoxLayout(dialog)

            # Formularz konfiguracji
            form_layout = QFormLayout()

            # Wybór modelu bazowego
            base_model_layout = QHBoxLayout()
            base_model_edit = QLineEdit()
            base_model_button = QPushButton("Przeglądaj...")

            def select_base_model():
                file_path, _ = QFileDialog.getOpenFileName(
                    self,
                    "Wybierz model bazowy",
                    os.path.join("data", "models"),
                    "Modele (*.h5 *.pt);;Wszystkie pliki (*.*)",
                )
                if file_path:
                    base_model_edit.setText(file_path)

            base_model_button.clicked.connect(select_base_model)
            base_model_layout.addWidget(base_model_edit)
            base_model_layout.addWidget(base_model_button)
            form_layout.addRow("Model bazowy:", base_model_layout)

            # Wybór folderu treningowego
            train_dir_layout = QHBoxLayout()
            train_dir_edit = QLineEdit()
            train_dir_button = QPushButton("Przeglądaj...")
            train_dir_button.clicked.connect(
                lambda: self._select_directory(train_dir_edit)
            )
            train_dir_layout.addWidget(train_dir_edit)
            train_dir_layout.addWidget(train_dir_button)
            form_layout.addRow("Katalog danych treningowych:", train_dir_layout)

            # Wybór folderu walidacyjnego (opcjonalnie)
            val_dir_layout = QHBoxLayout()
            val_dir_edit = QLineEdit()
            val_dir_button = QPushButton("Przeglądaj...")
            val_dir_button.clicked.connect(lambda: self._select_directory(val_dir_edit))
            val_dir_layout.addWidget(val_dir_edit)
            val_dir_layout.addWidget(val_dir_button)
            form_layout.addRow(
                "Katalog danych walidacyjnych (opcjonalnie):", val_dir_layout
            )

            # Liczba epok
            epochs_spin = QSpinBox()
            epochs_spin.setRange(1, 1000)
            epochs_spin.setValue(50)
            form_layout.addRow("Liczba epok:", epochs_spin)

            # Typ modelu (architektura)
            model_arch_combo = QComboBox()
            model_arch_combo.addItems(
                ["resnet50", "efficientnet", "mobilenet", "vit", "convnext"]
            )
            form_layout.addRow("Typ modelu:", model_arch_combo)

            # Rozmiar wsadu
            batch_size_spin = QSpinBox()
            batch_size_spin.setRange(1, 256)
            batch_size_spin.setValue(32)
            form_layout.addRow("Rozmiar wsadu:", batch_size_spin)

            # Współczynnik uczenia
            learning_rate_combo = QComboBox()
            learning_rate_combo.addItems(["0.1", "0.01", "0.001", "0.0001"])
            learning_rate_combo.setCurrentText("0.0001")
            form_layout.addRow("Współczynnik uczenia:", learning_rate_combo)

            # Optymalizator
            optimizer_combo = QComboBox()
            optimizer_combo.addItems(["Adam", "SGD", "RMSprop", "AdamW"])
            form_layout.addRow("Optymalizator:", optimizer_combo)

            # Liczba wątków do ładowania danych
            num_workers_spin = QSpinBox()
            num_workers_spin.setRange(0, 16)
            if (
                hasattr(self.parent, "hardware_profile")
                and self.parent.hardware_profile
            ):
                profile = self.parent.hardware_profile
                recommended_workers = profile.get("recommended_workers", 2)
                num_workers_spin.setValue(recommended_workers)
                num_workers_spin.setToolTip(
                    f"Zalecana wartość z profilu sprzętowego: {recommended_workers}"
                )
            else:
                num_workers_spin.setValue(2)
                num_workers_spin.setToolTip(
                    "Brak profilu sprzętowego - używana wartość domyślna: 2"
                )
            form_layout.addRow("Liczba wątków do ładowania danych:", num_workers_spin)

            # Współczynnik regularyzacji L2
            weight_decay_combo = QComboBox()
            weight_decay_combo.addItems(["1e-3", "1e-4", "1e-5", "1e-6"])
            weight_decay_combo.setCurrentText("1e-4")  # Domyślna wartość
            form_layout.addRow("Współczynnik regularyzacji L2:", weight_decay_combo)

            # Wartość przycinania gradientów
            gradient_clip_spin = QDoubleSpinBox()
            gradient_clip_spin.setRange(0.0, 1.0)
            gradient_clip_spin.setValue(DEFAULT_TRAINING_PARAMS["gradient_clip_val"])
            gradient_clip_spin.setDecimals(2)
            gradient_clip_spin.setSingleStep(0.01)
            form_layout.addRow("Wartość przycinania gradientów:", gradient_clip_spin)

            # Liczba epok bez poprawy przed wczesnym zatrzymaniem
            early_stopping_spin = QSpinBox()
            early_stopping_spin.setRange(1, 50)
            early_stopping_spin.setValue(
                DEFAULT_TRAINING_PARAMS["early_stopping_patience"]
            )
            form_layout.addRow(
                "Liczba epok bez poprawy przed zatrzymaniem:", early_stopping_spin
            )

            # Opcje doszkalania
            finetuning_group = QGroupBox("Opcje doszkalania")
            finetuning_layout = QFormLayout(finetuning_group)

            # Zamrożenie warstw
            freeze_layers = QCheckBox("Zamroź warstwy bazowe")
            freeze_layers.setChecked(True)
            finetuning_layout.addRow("", freeze_layers)

            # Liczba warstw do trenowania
            trainable_layers_spin = QSpinBox()
            trainable_layers_spin.setRange(1, 10)
            trainable_layers_spin.setValue(3)
            finetuning_layout.addRow(
                "Liczba warstw do trenowania:", trainable_layers_spin
            )

            # Augmentacja danych
            augmentation_group = QGroupBox("Augmentacja danych")
            augmentation_layout = QFormLayout(augmentation_group)

            # Włącz augmentację
            use_augmentation = QCheckBox("Używaj augmentacji danych")
            augmentation_layout.addRow("", use_augmentation)

            # Rotacja
            rotation_spin = QSpinBox()
            rotation_spin.setRange(0, 360)
            rotation_spin.setValue(15)
            rotation_spin.setSuffix("°")
            augmentation_layout.addRow("Maksymalny kąt rotacji:", rotation_spin)

            # Jasność
            brightness_spin = QSpinBox()
            brightness_spin.setRange(0, 100)
            brightness_spin.setValue(20)
            brightness_spin.setSuffix("%")
            augmentation_layout.addRow("Zmiana jasności:", brightness_spin)

            # Przyciski
            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok
                | QDialogButtonBox.StandardButton.Cancel
            )
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)

            # Dodaj wszystkie elementy do głównego layoutu
            layout.addLayout(form_layout)
            layout.addWidget(finetuning_group)
            layout.addWidget(augmentation_group)
            layout.addWidget(buttons)

            # Wyłącz domyślne połączenie przycisku OK i podłącz własny handler
            buttons.accepted.disconnect()

            # Funkcja obsługująca zaakceptowanie dialogu
            def handle_finetuning_accept():
                try:
                    # Pobierz podstawowe dane
                    base_model = base_model_edit.text()
                    train_dir = train_dir_edit.text()
                    val_dir = val_dir_edit.text()
                    epochs = epochs_spin.value()
                    model_arch = model_arch_combo.currentText()

                    # Walidacja danych
                    if not base_model or not os.path.exists(base_model):
                        QMessageBox.warning(
                            self, "Błąd", "Wybierz poprawny model bazowy."
                        )
                        return

                    is_valid, error_msg = validate_training_directory(train_dir)
                    if not is_valid:
                        QMessageBox.warning(
                            self,
                            "Błąd",
                            f"Nieprawidłowy katalog treningowy: {error_msg}",
                        )
                        return

                    if val_dir:
                        is_valid, error_msg = validate_validation_directory(val_dir)
                        if not is_valid:
                            QMessageBox.warning(
                                self,
                                "Błąd",
                                f"Nieprawidłowy katalog walidacyjny: {error_msg}",
                            )
                            return

                    # Generuj nazwę pliku zadania
                    base_model_name = os.path.splitext(os.path.basename(base_model))[0]
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
                    task_name = (
                        f"{base_model_name}_{model_arch}_{epochs}epok_{timestamp}.json"
                    )

                    # Przygotuj konfigurację zadania
                    task_config = {
                        "name": task_name,
                        "type": "Doszkalanie",
                        "status": "Nowy",
                        "priority": 0,
                        "created_at": datetime.datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "config": {
                            "base_model": base_model,
                            "train_dir": train_dir,
                            "data_dir": train_dir,
                            "val_dir": val_dir,
                            "epochs": epochs,
                            "model_arch": model_arch,
                            "batch_size": batch_size_spin.value(),
                            "learning_rate": float(learning_rate_combo.currentText()),
                            "optimizer": optimizer_combo.currentText(),
                            "num_workers": num_workers_spin.value(),
                            "weight_decay": float(weight_decay_combo.currentText()),
                            "gradient_clip": gradient_clip_spin.value(),
                            "early_stopping": early_stopping_spin.value(),
                            "finetuning": {
                                "freeze_base_layers": freeze_layers.isChecked(),
                                "trainable_layers": trainable_layers_spin.value(),
                            },
                            "augmentation": {
                                "enabled": use_augmentation.isChecked(),
                                "rotation": rotation_spin.value(),
                                "brightness": brightness_spin.value(),
                            },
                        },
                    }

                    # Walidacja konfiguracji
                    is_valid, error_msg = validate_task_config(task_config)
                    if not is_valid:
                        QMessageBox.warning(
                            self,
                            "Błąd walidacji",
                            f"Konfiguracja zadania nieprawidłowa: {error_msg}",
                        )
                        return

                    # Zapisz konfigurację
                    tasks_dir = os.path.join("data", "tasks")
                    os.makedirs(tasks_dir, exist_ok=True)
                    task_file = os.path.join(tasks_dir, task_name)

                    with open(task_file, "w", encoding="utf-8") as f:
                        json.dump(task_config, f, indent=4, ensure_ascii=False)

                    # Sprawdź czy plik został zapisany poprawnie
                    if not os.path.exists(task_file) or os.path.getsize(task_file) == 0:
                        raise IOError("Plik zadania nie został poprawnie zapisany")

                    # Walidacja zapisanego pliku
                    is_valid, error_msg = validate_task_file(task_file)
                    if not is_valid:
                        self.parent.logger.error(
                            f"Błąd walidacji pliku zadania: {error_msg}"
                        )
                        os.remove(task_file)  # Usuń nieprawidłowy plik
                        QMessageBox.critical(
                            self,
                            "Błąd walidacji",
                            f"Plik zadania nie przeszedł walidacji: {error_msg}",
                        )
                        return

                    # Komunikat o sukcesie
                    QMessageBox.information(
                        self,
                        "Zadanie utworzone",
                        f"Zadanie doszkalania '{task_name}' zostało dodane do kolejki.",
                    )

                    # Odśwież listę zadań
                    self.refresh()

                    # Zamknij dialog
                    dialog.accept()

                except Exception as e:
                    QMessageBox.critical(
                        self,
                        "Błąd",
                        f"Wystąpił błąd podczas tworzenia zadania: {str(e)}",
                    )

            # Podłącz handler
            buttons.accepted.connect(handle_finetuning_accept)

            # Wyświetl dialog
            dialog.exec()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Błąd",
                f"Nie udało się otworzyć okna konfiguracji zadania doszkalania: {str(e)}",
            )

    def _start_task_queue(self):
        """Uruchamia wszystkie zadania w kolejce."""
        # Sprawdź, czy wątek treningowy już działa
        if (
            hasattr(self, "training_thread")
            and self.training_thread is not None
            and self.training_thread.isRunning()
        ):
            QMessageBox.warning(
                self,
                "Trening w toku",
                "Proces treningu jest aktywny. Poczekaj na jego zakończenie lub zatrzymaj go.",
            )
            return

        # Katalog z zadaniami
        tasks_dir = os.path.join("data", "tasks")
        os.makedirs(tasks_dir, exist_ok=True)

        # Pobierz listę plików zadań
        task_files = sorted(glob.glob(os.path.join(tasks_dir, "*.json")))

        if not task_files:
            QMessageBox.information(
                self, "Kolejka pusta", "Brak zadań w kolejce do uruchomienia."
            )
            return

        self.parent.logger.info(
            f"Znaleziono {len(task_files)} zadań w kolejce. Uruchamianie..."
        )

        try:
            # Utwórz nowy wątek z listą zadań
            self.training_thread = BatchTrainingThread(task_files)

            # Podłącz sygnały
            self.training_thread.task_started.connect(self._training_task_started)
            self.training_thread.task_progress.connect(self._training_task_progress)
            self.training_thread.task_completed.connect(self._training_task_completed)
            self.training_thread.all_tasks_completed.connect(
                self._all_training_tasks_completed
            )
            self.training_thread.error.connect(self._training_task_error)

            # Uruchom wątek
            self.training_thread.start()

            # Zaktualizuj UI
            self.parent.current_task_info.setText(
                "Rozpoczynanie przetwarzania kolejki..."
            )
            self.parent.logger.info("Uruchomiono przetwarzanie kolejki zadań.")

        except Exception as e:
            QMessageBox.critical(
                self, "Błąd", f"Nie udało się uruchomić kolejki zadań: {str(e)}"
            )
            self.parent.logger.error(f"Błąd podczas uruchamiania kolejki: {str(e)}")

    def _clear_task_queue(self):
        """Czyści kolejkę zadań treningowych."""
        reply = QMessageBox.question(
            self,
            "Potwierdzenie",
            "Czy na pewno chcesz usunąć wszystkie zadania z kolejki?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            # Katalog z zadaniami
            tasks_dir = os.path.join("data", "tasks")

            if not os.path.exists(tasks_dir):
                return

            # Usuń wszystkie pliki zadań
            for task_file in glob.glob(os.path.join(tasks_dir, "*.json")):
                try:
                    os.remove(task_file)
                except Exception as e:
                    self.parent.logger.error(
                        f"Błąd podczas usuwania zadania {task_file}: {str(e)}"
                    )

            # Odśwież listę zadań
            self.refresh()

            # Wyświetl potwierdzenie
            QMessageBox.information(
                self,
                "Kolejka wyczyszczona",
                "Wszystkie zadania zostały usunięte z kolejki.",
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Błąd", f"Wystąpił błąd podczas czyszczenia kolejki: {str(e)}"
            )

    def _training_task_started(self, task_name, task_type):
        """Obsługa rozpoczęcia zadania treningowego."""
        # Logowanie
        self.parent.logger.info(f"Rozpoczęto zadanie {task_type}: {task_name}")

        # Aktualizacja UI w głównym oknie
        self.parent.current_task_info.setText(
            f"Aktywne zadanie: {task_name} ({task_type})"
        )
        self.parent.task_progress_bar.setValue(0)  # Zresetuj pasek postępu
        self.parent.stop_task_btn.setEnabled(True)  # Aktywuj przycisk zatrzymania

    def _training_task_progress(self, task_name, progress, details):
        """Obsługa postępu zadania treningowego."""
        try:
            # Pobierz dane z details i upewnij się, że mają prawidłowe wartości
            epoch = int(details.get("epoch", 0))
            total_epochs = int(details.get("total_epochs", 1))

            # Zabezpieczenie przed dzieleniem przez zero
            if total_epochs <= 0:
                total_epochs = 1

            # Pobierz i weryfikuj wartości loss i accuracy
            train_loss = details.get("train_loss")
            train_acc = details.get("train_acc")
            val_loss = details.get("val_loss")
            val_acc = details.get("val_acc")

            # Aktualizacja paska postępu
            percentage = min(100, max(0, int((epoch / total_epochs) * 100)))
            self.parent.task_progress_bar.setValue(percentage)

            # Aktualizacja opisu postępu
            if epoch > 0:
                loss_text = f"{train_loss:.4f}" if train_loss is not None else "N/A"
                acc_text = f"{train_acc:.2%}" if train_acc is not None else "N/A"
                details_text = (
                    f"Epoka {epoch}/{total_epochs} | Strata: {loss_text}, "
                    f"Dokładność: {acc_text}"
                )
                self.parent.task_progress_details.setText(details_text)
                self.parent.logger.info(details_text)

            # Aktualizacja wizualizacji jeśli istnieje
            if hasattr(self, "training_visualization") and self.training_visualization:
                # Upewnij się, że epoka jest większa od zera
                if epoch > 0:
                    try:
                        self.training_visualization.update_data(
                            epoch=epoch,
                            train_loss=train_loss,
                            train_acc=train_acc,
                            val_loss=val_loss,
                            val_acc=val_acc,
                        )
                    except Exception as vis_error:
                        self.parent.logger.error(
                            f"Błąd aktualizacji wizualizacji: {vis_error}"
                        )

        except Exception as e:
            import traceback

            self.parent.logger.error(f"Błąd w _training_task_progress: {e}")
            self.parent.logger.error(traceback.format_exc())

    def _training_task_completed(self, task_name, result):
        """Obsługuje zakończenie zadania treningowego."""
        try:
            # Odśwież zakładkę modeli
            self.parent.model_manager_tab.refresh()

            model_filename = result.get("model_filename", "")
            accuracy = result.get("accuracy", 0.0)
            epochs_trained = result.get("epochs_trained", 0)
            training_time = result.get("training_time", 0)

            self.parent.logger.info(
                f"Zakończono zadanie {task_name}. Model: {model_filename}, "
                f"Dokładność: {accuracy:.2%}, Epoki: {epochs_trained}, "
                f"Czas: {training_time:.1f}s"
            )

            # Aktualizacja UI w głównym oknie
            self.parent.current_task_info.setText("Brak aktywnego zadania")
            self.parent.task_progress_bar.setValue(0)
            self.parent.task_progress_details.setText("")
            self.parent.stop_task_btn.setEnabled(False)

            # Wyczyść dane wizualizacji
            # self.training_visualization.clear_data() # <--- TYMCZASOWO ZAKOMENTOWANE

            # Zmień status zadania na 'Zakończony'
            self._set_task_status(task_name, "Zakończony")

        except Exception as e:
            QMessageBox.critical(
                self,
                "Błąd",
                f"Nie udało się zakończyć zadania: {str(e)}",
            )

    def _all_training_tasks_completed(self):
        """Obsługa zakończenia wszystkich zadań treningowych."""
        self.parent.logger.info("Wszystkie zadania treningowe zostały zakończone")

        # Informacja dla użytkownika
        QMessageBox.information(
            self,
            "Trening zakończony",
            "Wszystkie zadania treningowe zostały zakończone pomyślnie.",
        )

        # Aktualizacja UI w głównym oknie
        self.parent.current_task_info.setText("Brak aktywnego zadania")
        self.parent.task_progress_bar.setValue(0)
        self.parent.task_progress_details.setText("Oczekiwanie na zadania...")

        # Odśwież kolejkę, aby zaktualizować statusy zadań
        self._refresh_task_queue()

        # Tutaj można dodać logikę, np. opcję zapisania wykresu

    def _training_task_error(self, task_name, error_message):
        """Obsługa błędu zadania treningowego."""
        # Logowanie do konsoli
        print(f"\n[KRYTYCZNY BŁĄD] Zadanie: {task_name}")
        print(f"Treść błędu: {error_message}")
        print("-" * 80)

        self.parent.logger.error(f"Błąd zadania {task_name}: {error_message}")

        # Pokaż komunikat o błędzie
        QMessageBox.critical(
            self,
            "Błąd treningu",
            f"Wystąpił błąd w zadaniu {task_name}: {error_message}",
        )

        # Odświeżamy listę zadań
        self.refresh()

    def update_settings(self, settings):
        """Aktualizuje ustawienia zakładki."""
        self.settings = settings

    def save_state(self):
        """Zapisuje stan zakładki."""
        return {}

    def restore_state(self, state):
        """Przywraca zapisany stan zakładki."""
        pass

    def update_hardware_profile(self, profile):
        """Aktualizuje profil sprzętowy używany do optymalizacji treningu."""
        self.hardware_profile = profile
        self.parent.logger.info("Zaktualizowano profil sprzętowy w zakładce treningu")

        # Opcjonalnie: aktualizacja UI na podstawie nowego profilu
        if hasattr(self, "use_optimization_checkbox"):
            self.use_optimization_checkbox.setChecked(True)

        if profile and hasattr(self, "profile_info_label"):
            cpu_info = profile.get("cpu_info", {})
            if isinstance(cpu_info, str):
                try:
                    cpu_info = json.loads(cpu_info)
                except:
                    cpu_info = {}

            gpu_info = profile.get("gpu_info", {})
            if isinstance(gpu_info, str):
                try:
                    gpu_info = json.loads(gpu_info)
                except:
                    gpu_info = {}

            info_text = "Status profilu: Aktywny\n"
            info_text += f"CPU: {cpu_info.get('name', 'Nieznany')}\n"
            if gpu_info:
                info_text += f"GPU: {gpu_info.get('name', 'Nieznany')}\n"
            info_text += f"RAM: {profile.get('ram_total', 0):.1f} GB"

            self.profile_info_label.setText(info_text)

    def _run_profiler(self):
        """Uruchamia proces profilowania sprzętu."""
        try:
            # Wyświetl dialog potwierdzenia
            reply = QMessageBox.question(
                self,
                "Profilowanie sprzętu",
                "Proces profilowania może potrwać kilka minut i okresowo "
                "obciążyć system.\n\nCzy chcesz kontynuować?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )

            if reply != QMessageBox.StandardButton.Yes:
                return

            # Utwórz i skonfiguruj dialog postępu
            progress_dialog = QProgressDialog(
                "Trwa profilowanie sprzętu...", "Anuluj", 0, 100, self
            )
            progress_dialog.setWindowTitle("Profilowanie")
            progress_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
            progress_dialog.setMinimumDuration(0)
            progress_dialog.show()

            # Aktualizacja interfejsu
            progress_dialog.setValue(10)
            QApplication.processEvents()

            # Uruchom profiler w tle
            from app.utils.profiler import HardwareProfiler

            profiler = HardwareProfiler()

            # Aktualizacja interfejsu
            progress_dialog.setValue(20)
            progress_dialog.setLabelText("Rozpoczynam profilowanie...")
            QApplication.processEvents()

            # Wykonaj pełne profilowanie
            profile = profiler.run_profile()

            # Aktualizacja interfejsu
            progress_dialog.setValue(100)
            progress_dialog.setLabelText("Profilowanie zakończone!")
            QApplication.processEvents()

            # Załaduj wygenerowany profil
            self.parent.hardware_profile = profile

            # Wyświetl podsumowanie profilowania
            QMessageBox.information(
                self,
                "Profilowanie zakończone",
                "Profilowanie sprzętu zostało zakończone pomyślnie.\n\n"
                "Parametry aplikacji zostały zoptymalizowane dla Twojego systemu.",
            )

            # Aktualizuj informacje o profilu
            self._update_profile_info()

        except Exception as e:
            self.parent.logger.error(f"Błąd profilowania: {str(e)}")
            QMessageBox.critical(
                self,
                "Błąd profilowania",
                f"Wystąpił nieoczekiwany błąd: {str(e)}",
            )

    def _update_profile_info(self):
        """Aktualizuje informacje o profilu sprzętowym."""
        try:
            if (
                hasattr(self.parent, "hardware_profile")
                and self.parent.hardware_profile
            ):
                profile = self.parent.hardware_profile
                cpu_info = profile.get("cpu_info", {})
                gpu_info = profile.get("gpu_info", {})
                ram_info = profile.get("ram_info", {})

                info_text = "Status profilu: Aktywny\n"
                info_text += f"CPU: {cpu_info.get('name', 'Nieznany')}\n"
                if gpu_info:
                    info_text += f"GPU: {gpu_info.get('name', 'Nieznany')}\n"
                info_text += f"RAM: {ram_info.get('total_gb', 0):.1f} GB"

                self.profile_info_label.setText(info_text)
            else:
                self.profile_info_label.setText("Status profilu: Brak")
        except Exception as e:
            self.parent._log_message(
                f"Błąd aktualizacji informacji o profilu: {str(e)}"
            )
            self.profile_info_label.setText("Status profilu: Błąd")

    def _run_task_from_queue(self, task_file):
        """Uruchamia pojedyncze zadanie z kolejki."""
        try:
            # Sprawdź, czy wątek treningowy już działa
            if (
                hasattr(self, "training_thread")
                and self.training_thread is not None
                and self.training_thread.isRunning()
            ):
                QMessageBox.warning(
                    self,
                    "Trening w toku",
                    "Inny proces treningu jest już aktywny. "
                    "Poczekaj na jego zakończenie lub zatrzymaj go.",
                )
                return

            # Utwórz nowy wątek z pojedynczym zadaniem
            self.training_thread = BatchTrainingThread([task_file])

            # Podłącz sygnały
            self.training_thread.task_started.connect(self._training_task_started)
            self.training_thread.task_progress.connect(self._training_task_progress)
            self.training_thread.task_completed.connect(self._training_task_completed)
            self.training_thread.all_tasks_completed.connect(
                self._all_training_tasks_completed
            )
            self.training_thread.error.connect(self._training_task_error)

            # Uruchom wątek
            self.training_thread.start()

            # Zaktualizuj UI
            self.parent.current_task_info.setText("Rozpoczynanie zadania...")
            self.parent.logger.info("Uruchomiono pojedyncze zadanie.")

        except Exception as e:
            QMessageBox.critical(
                self, "Błąd", f"Nie udało się uruchomić zadania: {str(e)}"
            )

    def _delete_task_from_queue(self, task_file):
        """Usuwa zadanie z kolejki."""
        try:
            # Pobierz nazwę zadania z pliku
            with open(task_file, "r", encoding="utf-8") as f:
                task_data = json.load(f)
                task_name = task_data.get("name", os.path.basename(task_file))

            # Potwierdzenie usunięcia
            reply = QMessageBox.question(
                self,
                "Potwierdzenie",
                f"Czy na pewno chcesz usunąć zadanie '{task_name}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if reply != QMessageBox.StandardButton.Yes:
                return

            # Usuń plik zadania
            os.remove(task_file)

            # Odśwież listę zadań
            self.refresh()

            # Wyświetl potwierdzenie
            QMessageBox.information(
                self,
                "Zadanie usunięte",
                f"Zadanie '{task_name}' zostało usunięte z kolejki.",
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Błąd", f"Wystąpił błąd podczas usuwania zadania: {str(e)}"
            )
            self.parent.logger.error(f"Błąd podczas usuwania zadania: {str(e)}")

    def _set_task_status(self, task_name, new_status):
        """Zmienia status zadania na nowy i zapisuje do pliku."""
        tasks_dir = os.path.join("data", "tasks")
        task_file = os.path.join(tasks_dir, f"{task_name}")
        if os.path.exists(task_file):
            try:
                with open(task_file, "r", encoding="utf-8") as f:
                    task_data = json.load(f)
                task_data["status"] = new_status
                with open(task_file, "w", encoding="utf-8") as f:
                    json.dump(task_data, f, indent=4, ensure_ascii=False)
            except Exception as e:
                self.parent.logger.error(f"Błąd przy zmianie statusu zadania: {e}")

    def _stop_current_task(self):
        """Bezwzględnie zatrzymuje aktualnie wykonywane zadanie treningu."""
        try:
            if hasattr(self, "training_thread") and self.training_thread is not None:
                # Zatrzymaj wątek treningu
                self.training_thread.stop()

                # Poczekaj na zakończenie wątku (maksymalnie 5 sekund)
                if self.training_thread.isRunning():
                    self.training_thread.wait(5000)

                    # Jeśli wątek nadal działa, wymuś jego zakończenie
                    if self.training_thread.isRunning():
                        self.training_thread.terminate()
                        self.training_thread.wait()

                # Wyczyść referencję do wątku
                self.training_thread = None

                # Zaktualizuj UI
                self.parent.current_task_info.setText("Zadanie zostało zatrzymane")
                self.parent.task_progress_bar.setValue(0)
                self.parent.task_progress_details.setText("")
                self.parent.stop_task_btn.setEnabled(False)

                # Odśwież listę zadań
                self.refresh()

                self.parent.logger.info(
                    "Zadanie treningu zostało bezwzględnie zatrzymane"
                )

        except Exception as e:
            self.parent.logger.error(f"Błąd podczas zatrzymywania zadania: {str(e)}")
            QMessageBox.critical(
                self, "Błąd", f"Wystąpił błąd podczas zatrzymywania zadania:\n{str(e)}"
            )

    def _edit_task_from_queue(self, task_file):
        """Otwiera plik zadania w domyślnym edytorze tekstowym/kodu."""
        try:
            import platform
            import subprocess

            if platform.system() == "Windows":
                os.startfile(task_file)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", task_file])
            else:  # Linux
                subprocess.run(["xdg-open", task_file])

        except Exception as e:
            self.parent.logger.error(f"Błąd podczas otwierania pliku: {str(e)}")
            QMessageBox.critical(
                self, "Błąd", f"Nie udało się otworzyć pliku w edytorze: {str(e)}"
            )
