from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt


class HardwareProfileDialog(QtWidgets.QDialog):
    def __init__(self, hardware_profile: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Profil sprzętowy")
        self.setMinimumWidth(400)
        layout = QtWidgets.QVBoxLayout(self)

        # Wyciągnij czytelne informacje sprzętowe
        cpu_info = hardware_profile.get("cpu_info", {})
        if isinstance(cpu_info, str):
            import json

            try:
                cpu_info = json.loads(cpu_info)
            except Exception:
                cpu_info = {}
        if not isinstance(cpu_info, dict):
            cpu_info = {}
        gpu_info = hardware_profile.get("gpu_info", {})
        if isinstance(gpu_info, str):
            import json

            try:
                gpu_info = json.loads(gpu_info)
            except Exception:
                gpu_info = {}
        if not isinstance(gpu_info, dict):
            gpu_info = {}
        ram_gb = hardware_profile.get("ram_total")
        if ram_gb is None:
            ram_info = hardware_profile.get("ram_info", {})
            ram_gb = ram_info.get("total_gb")
        # Przygotuj dane do tabeli sprzętowej
        rows = []
        rows.append(("CPU", cpu_info.get("name", "Nieznany")))
        if "cores" in cpu_info:
            rows.append(("Rdzeni CPU", cpu_info["cores"]))
        rows.append(("GPU", gpu_info.get("name", "Nieznany")))
        if "memory" in gpu_info:
            rows.append(("VRAM GPU", f"{gpu_info['memory']} GB"))
        if ram_gb is not None:
            rows.append(("RAM", f"{ram_gb:.1f} GB"))

        table = QtWidgets.QTableWidget(self)
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Parametr", "Wartość"])
        table.setRowCount(len(rows))
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        table.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        for row, (key, value) in enumerate(rows):
            key_item = QtWidgets.QTableWidgetItem(str(key))
            value_item = QtWidgets.QTableWidgetItem(str(value))
            key_item.setFlags(key_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            value_item.setFlags(value_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, 0, key_item)
            table.setItem(row, 1, value_item)

        table.resizeColumnsToContents()
        layout.addWidget(table)

        # --- Optymalne parametry treningu ---
        param_keys = [
            ("recommended_batch_size", "Batch size"),
            ("recommended_workers", "Liczba workerów"),
            ("use_mixed_precision", "Mixed precision"),
            ("learning_rate", "Learning rate"),
            ("max_epochs", "Maks. epok"),
            ("gradient_accumulation_steps", "Gradient accumulation"),
        ]
        param_rows = []
        for key, label in param_keys:
            if key in hardware_profile:
                value = hardware_profile[key]
                param_rows.append((label, str(value)))
        if param_rows:
            param_label = QtWidgets.QLabel("Optymalne parametry treningu:")
            param_label.setStyleSheet("font-weight: bold; margin-top: 12px;")
            layout.addWidget(param_label)

            param_table = QtWidgets.QTableWidget(self)
            param_table.setColumnCount(2)
            param_table.setHorizontalHeaderLabels(["Parametr", "Wartość"])
            param_table.setRowCount(len(param_rows))
            param_table.verticalHeader().setVisible(False)
            param_table.setEditTriggers(
                QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
            )
            param_table.setSelectionMode(
                QtWidgets.QAbstractItemView.SelectionMode.NoSelection
            )
            param_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)

            for row, (key, value) in enumerate(param_rows):
                key_item = QtWidgets.QTableWidgetItem(str(key))
                value_item = QtWidgets.QTableWidgetItem(str(value))
                key_item.setFlags(key_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                value_item.setFlags(value_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                param_table.setItem(row, 0, key_item)
                param_table.setItem(row, 1, value_item)

            param_table.resizeColumnsToContents()
            layout.addWidget(param_table)

        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
        )
        btn_box.accepted.connect(self.accept)
        layout.addWidget(btn_box)
