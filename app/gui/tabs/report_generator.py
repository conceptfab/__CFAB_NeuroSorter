import datetime
import json
import os
import webbrowser

import pandas as pd
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.gui.tab_interface import TabInterface
from app.utils.report_utils import generate_report_pdf


class ReportGenerator(QWidget, TabInterface):
    """Klasa zarządzająca zakładką generowania raportów."""

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

        # Panel konfiguracji raportu
        self._create_config_panel(layout)

        # Panel wyników
        self._create_results_panel(layout)

        # Panel eksportu
        self._create_export_panel(layout)

        # Dodaj elastyczną przestrzeń na dole
        layout.addStretch(1)

    def connect_signals(self):
        """Podłącza sygnały do slotów."""
        self.generate_btn.clicked.connect(
            lambda: self._generate_single_report(self.report_type_combo.currentText())
        )
        self.export_pdf_btn.clicked.connect(self._export_to_pdf)
        self.export_excel_btn.clicked.connect(self._export_to_excel)
        self.delete_btn.clicked.connect(self._delete_report)

        # Podłącz sygnały dla przycisków porównania
        self.compare_btn.clicked.connect(self._compare_generated_reports)
        self.preview_btn.clicked.connect(self._preview_comparison_report)
        self.save_btn.clicked.connect(self._save_comparison_report)

    def refresh(self):
        """Odświeża zawartość zakładki."""
        self._refresh_results()

    def update_settings(self, settings):
        """Aktualizuje ustawienia zakładki."""
        self.settings = settings

    def save_state(self):
        """Zapisuje stan zakładki."""
        return {}

    def restore_state(self, state):
        """Przywraca zapisany stan zakładki."""
        pass

    def _create_config_panel(self, parent_layout):
        """Tworzy panel konfiguracji raportu."""
        config_panel = QWidget()
        config_layout = QVBoxLayout(config_panel)
        config_layout.setContentsMargins(0, 0, 0, 0)

        # Nagłówek sekcji
        config_header = QLabel("KONFIGURACJA RAPORTU")
        config_header.setStyleSheet(
            "font-weight: bold; color: #CCCCCC; "
            "font-size: 11px; padding-bottom: 4px;"
        )
        config_layout.addWidget(config_header)

        # Typ raportu
        report_type_layout = QHBoxLayout()
        report_type_label = QLabel("Typ raportu:")
        report_type_label.setFixedWidth(120)
        self.report_type_combo = QComboBox()
        self.report_type_combo.addItems(
            [
                "Raport wydajności modelu",
                "Raport klasyfikacji wsadowej",
                "Raport treningu",
                "Raport porównawczy modeli",
            ]
        )

        report_type_layout.addWidget(report_type_label)
        report_type_layout.addWidget(self.report_type_combo)
        config_layout.addLayout(report_type_layout)

        # Nazwa raportu
        report_name_layout = QHBoxLayout()
        report_name_label = QLabel("Nazwa raportu:")
        report_name_label.setFixedWidth(120)
        self.report_name_edit = QLineEdit()
        self.report_name_edit.setText(
            "Raport-" + datetime.datetime.now().strftime("%Y%m%d-%H%M")
        )

        report_name_layout.addWidget(report_name_label)
        report_name_layout.addWidget(self.report_name_edit)
        config_layout.addLayout(report_name_layout)

        # Opcje raportu
        options_layout = QHBoxLayout()
        self.include_charts_checkbox = QCheckBox("Dołącz wykresy")
        self.include_charts_checkbox.setChecked(True)
        self.include_details_checkbox = QCheckBox("Szczegółowe statystyki")
        self.include_details_checkbox.setChecked(True)
        options_layout.addWidget(self.include_charts_checkbox)
        options_layout.addWidget(self.include_details_checkbox)
        options_layout.addStretch(1)
        config_layout.addLayout(options_layout)

        # Przycisk generowania
        self.generate_btn = QPushButton("Generuj raport")
        self.generate_btn.clicked.connect(self._generate_single_report)
        self.generate_btn.setFixedHeight(24)
        config_layout.addWidget(self.generate_btn)

        parent_layout.addWidget(config_panel)

    def _create_results_panel(self, parent_layout):
        """Tworzy panel wyników."""
        results_panel = QWidget()
        results_layout = QVBoxLayout(results_panel)
        results_layout.setContentsMargins(0, 0, 0, 0)

        # Nagłówek sekcji
        results_header = QLabel("WYNIKI")
        results_header.setStyleSheet(
            "font-weight: bold; color: #CCCCCC; "
            "font-size: 11px; padding-bottom: 4px;"
        )
        results_layout.addWidget(results_header)

        # Tabela wyników
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(
            ["Nazwa raportu", "Typ", "Data utworzenia", "Status"]
        )
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setAlternatingRowColors(True)
        results_layout.addWidget(self.results_table)

        parent_layout.addWidget(results_panel)

    def _create_export_panel(self, parent_layout):
        """Tworzy panel eksportu."""
        self.export_panel = QWidget()
        export_layout = QVBoxLayout(self.export_panel)
        export_layout.setContentsMargins(0, 0, 0, 0)

        # Nagłówek sekcji
        export_header = QLabel("EKSPORT")
        export_header.setStyleSheet(
            "font-weight: bold; color: #CCCCCC; "
            "font-size: 11px; padding-bottom: 4px;"
        )
        export_layout.addWidget(export_header)

        # Przyciski eksportu
        buttons_layout = QHBoxLayout()

        self.export_pdf_btn = QPushButton("Eksportuj do PDF")
        self.export_pdf_btn.clicked.connect(self._export_to_pdf)
        self.export_pdf_btn.setFixedHeight(24)
        buttons_layout.addWidget(self.export_pdf_btn)

        self.export_excel_btn = QPushButton("Eksportuj do Excel")
        self.export_excel_btn.clicked.connect(self._export_to_excel)
        self.export_excel_btn.setFixedHeight(24)
        buttons_layout.addWidget(self.export_excel_btn)

        self.delete_btn = QPushButton("Usuń raport")
        self.delete_btn.clicked.connect(self._delete_report)
        self.delete_btn.setFixedHeight(24)
        buttons_layout.addWidget(self.delete_btn)

        buttons_layout.addStretch(1)

        # Dodaj przyciski do porównywania i podglądu raportów
        self.compare_btn = QPushButton("Porównaj raporty")
        self.compare_btn.setFixedHeight(24)
        buttons_layout.addWidget(self.compare_btn)

        self.preview_btn = QPushButton("Podgląd raportu")
        self.preview_btn.setFixedHeight(24)
        buttons_layout.addWidget(self.preview_btn)

        self.save_btn = QPushButton("Zapisz raport")
        self.save_btn.setFixedHeight(24)
        buttons_layout.addWidget(self.save_btn)

        export_layout.addLayout(buttons_layout)

        parent_layout.addWidget(self.export_panel)

    def _generate_single_report(self):
        """Generuje pojedynczy raport określonego typu."""
        try:
            report_type = self.report_type_combo.currentText()
            self._generate_single_report(report_type)

        except Exception as e:
            QMessageBox.critical(
                self, "Błąd", f"Wystąpił błąd podczas generowania raportu: {str(e)}"
            )

    def _generate_model_performance_report(self, report_name, progress_dialog):
        """Generuje raport wydajności modelu."""
        try:
            # Katalog z modelami
            models_dir = os.path.join("data", "models")
            if not os.path.exists(models_dir):
                raise Exception("Katalog modeli nie istnieje")

            # Pobierz listę modeli
            model_files = [f for f in os.listdir(models_dir) if f.endswith(".h5")]
            if not model_files:
                raise Exception("Brak modeli do analizy")

            # Przygotuj dane do raportu
            report_data = {
                "name": report_name,
                "type": "Raport wydajności modelu",
                "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "Gotowy",
                "data": {
                    "models": [],
                    "statistics": {
                        "total_models": len(model_files),
                        "total_size": 0,
                        "avg_accuracy": 0,
                        "models_by_accuracy": {
                            "high": 0,  # > 90%
                            "medium": 0,  # 70-90%
                            "low": 0,  # < 70%
                        },
                    },
                },
            }

            # Analizuj każdy model
            for i, model_file in enumerate(model_files):
                progress_dialog.setValue(int((i / len(model_files)) * 100))
                if progress_dialog.wasCanceled():
                    return

                model_path = os.path.join(models_dir, model_file)
                config_path = os.path.join(
                    models_dir, model_file.replace(".h5", ".json")
                )

                # Pobierz dane modelu
                model_data = {
                    "name": model_file,
                    "size": os.path.getsize(model_path) / (1024 * 1024),  # MB
                    "created_at": datetime.datetime.fromtimestamp(
                        os.path.getctime(model_path)
                    ).strftime("%Y-%m-%d %H:%M:%S"),
                    "accuracy": None,
                    "classes": [],
                    "parameters": 0,
                }

                # Pobierz dodatkowe dane z pliku konfiguracyjnego
                if os.path.exists(config_path):
                    with open(config_path, "r", encoding="utf-8") as f:
                        config = json.load(f)
                        model_data["accuracy"] = config.get("accuracy")
                        model_data["classes"] = config.get("classes", [])
                        model_data["parameters"] = config.get("parameters", 0)

                # Dodaj do statystyk
                report_data["data"]["models"].append(model_data)
                report_data["data"]["statistics"]["total_size"] += model_data["size"]

                if model_data["accuracy"]:
                    accuracy = float(model_data["accuracy"])
                    if accuracy > 0.9:
                        report_data["data"]["statistics"]["models_by_accuracy"][
                            "high"
                        ] += 1
                    elif accuracy > 0.7:
                        report_data["data"]["statistics"]["models_by_accuracy"][
                            "medium"
                        ] += 1
                    else:
                        report_data["data"]["statistics"]["models_by_accuracy"][
                            "low"
                        ] += 1

            # Oblicz średnią dokładność
            accuracies = [
                m["accuracy"] for m in report_data["data"]["models"] if m["accuracy"]
            ]
            if accuracies:
                report_data["data"]["statistics"]["avg_accuracy"] = sum(
                    accuracies
                ) / len(accuracies)

            # Zapisz raport
            reports_dir = os.path.join("data", "reports")
            os.makedirs(reports_dir, exist_ok=True)
            report_path = os.path.join(reports_dir, f"{report_name}.json")

            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=4, ensure_ascii=False)

            # Wygeneruj raport HTML
            html_report = self._generate_html_performance_report(report_data)
            html_path = os.path.join(reports_dir, f"{report_name}.html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_report)

            return True

        except Exception as e:
            raise Exception(f"Błąd generowania raportu wydajności: {str(e)}")

    def _generate_batch_classification_report(self, report_name, progress_dialog):
        """Generuje raport klasyfikacji wsadowej."""
        QMessageBox.information(
            self,
            "Funkcja w trakcie implementacji",
            "Generowanie raportu klasyfikacji wsadowej jest w trakcie implementacji.",
        )

    def _generate_training_report(self, report_name, progress_dialog):
        """Generuje raport treningu."""
        QMessageBox.information(
            self,
            "Funkcja w trakcie implementacji",
            "Generowanie raportu treningu jest w trakcie implementacji.",
        )

    def _generate_model_comparison_report(self, report_name, progress_dialog):
        """Generuje raport porównawczy modeli."""
        try:
            # Katalog z modelami
            models_dir = os.path.join("data", "models")
            if not os.path.exists(models_dir):
                raise Exception("Katalog modeli nie istnieje")

            # Pobierz listę modeli
            model_files = [f for f in os.listdir(models_dir) if f.endswith(".h5")]
            if not model_files:
                raise Exception("Brak modeli do porównania")

            # Przygotuj dane do raportu
            report_data = {
                "name": report_name,
                "type": "Raport porównawczy modeli",
                "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "Gotowy",
                "data": {
                    "models": [],
                    "comparison": {
                        "accuracy": [],
                        "size": [],
                        "parameters": [],
                        "classes": set(),
                    },
                },
            }

            # Analizuj każdy model
            for i, model_file in enumerate(model_files):
                progress_dialog.setValue(int((i / len(model_files)) * 100))
                if progress_dialog.wasCanceled():
                    return

                model_path = os.path.join(models_dir, model_file)
                config_path = os.path.join(
                    models_dir, model_file.replace(".h5", ".json")
                )

                # Pobierz dane modelu
                model_data = {
                    "name": model_file,
                    "size": os.path.getsize(model_path) / (1024 * 1024),  # MB
                    "created_at": datetime.datetime.fromtimestamp(
                        os.path.getctime(model_path)
                    ).strftime("%Y-%m-%d %H:%M:%S"),
                    "accuracy": None,
                    "classes": [],
                    "parameters": 0,
                }

                # Pobierz dodatkowe dane z pliku konfiguracyjnego
                if os.path.exists(config_path):
                    with open(config_path, "r", encoding="utf-8") as f:
                        config = json.load(f)
                        model_data["accuracy"] = config.get("accuracy")
                        model_data["classes"] = config.get("classes", [])
                        model_data["parameters"] = config.get("parameters", 0)

                # Dodaj do danych porównawczych
                report_data["data"]["models"].append(model_data)
                report_data["data"]["comparison"]["accuracy"].append(
                    model_data["accuracy"]
                )
                report_data["data"]["comparison"]["size"].append(model_data["size"])
                report_data["data"]["comparison"]["parameters"].append(
                    model_data["parameters"]
                )
                report_data["data"]["comparison"]["classes"].update(
                    model_data["classes"]
                )

            # Konwertuj set klas na listę
            report_data["data"]["comparison"]["classes"] = list(
                report_data["data"]["comparison"]["classes"]
            )

            # Zapisz raport
            reports_dir = os.path.join("data", "reports")
            os.makedirs(reports_dir, exist_ok=True)
            report_path = os.path.join(reports_dir, f"{report_name}.json")

            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=4, ensure_ascii=False)

            # Wygeneruj raport HTML
            html_report = self._generate_html_comparison_report(report_data)
            html_path = os.path.join(reports_dir, f"{report_name}.html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_report)

            return True

        except Exception as e:
            raise Exception(f"Błąd generowania raportu porównawczego: {str(e)}")

    def _generate_html_performance_report(self, report_data):
        """Generuje raport HTML z wydajności modeli."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Raport wydajności modeli</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f5f5f5; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .stats {{ display: flex; gap: 20px; margin: 20px 0; }}
                .stat-box {{ 
                    background: #f5f5f5;
                    padding: 15px;
                    border-radius: 5px;
                    flex: 1;
                }}
            </style>
        </head>
        <body>
            <h1>Raport wydajności modeli</h1>
            <p>Data: {report_data['created_at']}</p>
            
            <h2>Statystyki ogólne</h2>
            <div class="stats">
                <div class="stat-box">
                    <h3>Liczba modeli</h3>
                    <p>{report_data['data']['statistics']['total_models']}</p>
                </div>
                <div class="stat-box">
                    <h3>Całkowity rozmiar</h3>
                    <p>{report_data['data']['statistics']['total_size']:.2f} MB</p>
                </div>
                <div class="stat-box">
                    <h3>Średnia dokładność</h3>
                    <p>{report_data['data']['statistics']['avg_accuracy']:.2%}</p>
                </div>
            </div>

            <h2>Podział na dokładność</h2>
            <div class="stats">
                <div class="stat-box">
                    <h3>Wysoka (>90%)</h3>
                    <p>{report_data['data']['statistics']['models_by_accuracy']['high']}</p>
                </div>
                <div class="stat-box">
                    <h3>Średnia (70-90%)</h3>
                    <p>{report_data['data']['statistics']['models_by_accuracy']['medium']}</p>
                </div>
                <div class="stat-box">
                    <h3>Niska (<70%)</h3>
                    <p>{report_data['data']['statistics']['models_by_accuracy']['low']}</p>
                </div>
            </div>

            <h2>Szczegółowe dane modeli</h2>
            <table>
                <tr>
                    <th>Nazwa</th>
                    <th>Rozmiar (MB)</th>
                    <th>Data utworzenia</th>
                    <th>Dokładność</th>
                    <th>Liczba klas</th>
                    <th>Parametry</th>
                </tr>
                {''.join(
                    f"<tr><td>{m['name']}</td><td>{m['size']:.2f}</td>"
                    f"<td>{m['created_at']}</td>"
                    f"<td>{m['accuracy']:.2% if m['accuracy'] else 'Nieznana'}</td>"
                    f"<td>{len(m['classes'])}</td><td>{m['parameters']:,}</td></tr>"
                    for m in report_data['data']['models']
                )}
            </table>
        </body>
        </html>
        """
        return html

    def _generate_html_comparison_report(self, report_data):
        """Generuje raport HTML porównawczy modeli."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Raport porównawczy modeli</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f5f5f5; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .stats {{ display: flex; gap: 20px; margin: 20px 0; }}
                .stat-box {{ 
                    background: #f5f5f5;
                    padding: 15px;
                    border-radius: 5px;
                    flex: 1;
                }}
            </style>
        </head>
        <body>
            <h1>Raport porównawczy modeli</h1>
            <p>Data: {report_data['created_at']}</p>
            
            <h2>Porównanie modeli</h2>
            <table>
                <tr>
                    <th>Nazwa</th>
                    <th>Rozmiar (MB)</th>
                    <th>Data utworzenia</th>
                    <th>Dokładność</th>
                    <th>Liczba klas</th>
                    <th>Parametry</th>
                </tr>
                {''.join(
                    f"<tr><td>{m['name']}</td><td>{m['size']:.2f}</td>"
                    f"<td>{m['created_at']}</td>"
                    f"<td>{m['accuracy']:.2% if m['accuracy'] else 'Nieznana'}</td>"
                    f"<td>{len(m['classes'])}</td><td>{m['parameters']:,}</td></tr>"
                    for m in report_data['data']['models']
                )}
            </table>

            <h2>Wspólne klasy</h2>
            <ul>
                {''.join(f"<li>{cls}</li>" for cls in sorted(report_data['data']['comparison']['classes']))}
            </ul>

            <h2>Statystyki porównawcze</h2>
            <div class="stats">
                <div class="stat-box">
                    <h3>Dokładność</h3>
                    <p>Min: {min(report_data['data']['comparison']['accuracy']):.2%}</p>
                    <p>Max: {max(report_data['data']['comparison']['accuracy']):.2%}</p>
                    <p>Średnia: {sum(report_data['data']['comparison']['accuracy'])/len(report_data['data']['comparison']['accuracy']):.2%}</p>
                </div>
                <div class="stat-box">
                    <h3>Rozmiar</h3>
                    <p>Min: {min(report_data['data']['comparison']['size']):.2f} MB</p>
                    <p>Max: {max(report_data['data']['comparison']['size']):.2f} MB</p>
                    <p>Średnia: {sum(report_data['data']['comparison']['size'])/len(report_data['data']['comparison']['size']):.2f} MB</p>
                </div>
                <div class="stat-box">
                    <h3>Parametry</h3>
                    <p>Min: {min(report_data['data']['comparison']['parameters']):,}</p>
                    <p>Max: {max(report_data['data']['comparison']['parameters']):,}</p>
                    <p>Średnia: {sum(report_data['data']['comparison']['parameters'])/len(report_data['data']['comparison']['parameters']):,.0f}</p>
                </div>
            </div>
        </body>
        </html>
        """
        return html

    def _refresh_results(self):
        """Odświeża listę wygenerowanych raportów."""
        try:
            # Wyczyść tabelę
            self.results_table.setRowCount(0)

            # Katalog z raportami
            reports_dir = os.path.join("data", "reports")
            os.makedirs(reports_dir, exist_ok=True)

            # Pobierz pliki raportów
            report_files = sorted(glob.glob(os.path.join(reports_dir, "*.json")))

            if not report_files:
                return

            # Dodaj raporty do tabeli
            for report_file in report_files:
                try:
                    with open(report_file, "r", encoding="utf-8") as f:
                        report_data = json.load(f)

                    # Dodaj wiersz do tabeli
                    row = self.results_table.rowCount()
                    self.results_table.insertRow(row)

                    # Nazwa raportu
                    report_name = report_data.get("name", os.path.basename(report_file))
                    self.results_table.setItem(row, 0, QTableWidgetItem(report_name))

                    # Typ raportu
                    report_type = report_data.get("type", "Nieznany")
                    self.results_table.setItem(row, 1, QTableWidgetItem(report_type))

                    # Data utworzenia
                    created_at = report_data.get("created_at", "")
                    self.results_table.setItem(row, 2, QTableWidgetItem(created_at))

                    # Status
                    status = report_data.get("status", "Gotowy")
                    self.results_table.setItem(row, 3, QTableWidgetItem(status))

                except Exception as e:
                    self.parent._log_message(
                        f"Błąd podczas wczytywania raportu {report_file}: {str(e)}"
                    )

            # Dostosuj szerokość kolumn
            self.results_table.resizeColumnsToContents()
            self.results_table.horizontalHeader().setStretchLastSection(True)

        except Exception as e:
            self.parent._log_message(
                f"Błąd podczas odświeżania listy raportów: {str(e)}"
            )

    def _export_to_pdf(self):
        """Eksportuje wybrany raport do formatu PDF."""
        try:
            # Sprawdź czy jest wybrany raport
            current_row = self.results_table.currentRow()
            if current_row < 0:
                QMessageBox.warning(
                    self, "Ostrzeżenie", "Wybierz raport do wyeksportowania."
                )
                return

            # Pobierz nazwę raportu
            report_name = self.results_table.item(current_row, 0).text()

            # Wybierz miejsce zapisu
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Zapisz raport PDF",
                report_name + ".pdf",
                "Pliki PDF (*.pdf)",
            )

            if not file_path:
                return

            # Generuj PDF
            generate_report_pdf(report_name, file_path)

            # Wyświetl komunikat o sukcesie
            QMessageBox.information(
                self,
                "Sukces",
                f"Raport został wyeksportowany do pliku:\n{file_path}",
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Błąd", f"Nie udało się wyeksportować raportu: {str(e)}"
            )

    def _export_to_excel(self):
        """Eksportuje wybrany raport do formatu Excel."""
        try:
            # Sprawdź czy jest wybrany raport
            current_row = self.results_table.currentRow()
            if current_row < 0:
                QMessageBox.warning(
                    self, "Ostrzeżenie", "Wybierz raport do wyeksportowania."
                )
                return

            # Pobierz nazwę raportu
            report_name = self.results_table.item(current_row, 0).text()

            # Wybierz miejsce zapisu
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Zapisz raport Excel",
                report_name + ".xlsx",
                "Pliki Excel (*.xlsx)",
            )

            if not file_path:
                return

            # Wczytaj dane raportu
            report_file = os.path.join("data", "reports", report_name + ".json")
            with open(report_file, "r", encoding="utf-8") as f:
                report_data = json.load(f)

            # Konwertuj dane do DataFrame
            df = pd.DataFrame(report_data["data"])

            # Zapisz do Excel
            df.to_excel(file_path, index=False)

            # Wyświetl komunikat o sukcesie
            QMessageBox.information(
                self,
                "Sukces",
                f"Raport został wyeksportowany do pliku:\n{file_path}",
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Błąd", f"Nie udało się wyeksportować raportu: {str(e)}"
            )

    def _delete_report(self):
        """Usuwa wybrany raport."""
        try:
            # Sprawdź czy jest wybrany raport
            current_row = self.results_table.currentRow()
            if current_row < 0:
                QMessageBox.warning(self, "Ostrzeżenie", "Wybierz raport do usunięcia.")
                return

            # Pobierz nazwę raportu
            report_name = self.results_table.item(current_row, 0).text()

            # Potwierdzenie przed usunięciem
            reply = QMessageBox.question(
                self,
                "Potwierdzenie",
                f"Czy na pewno chcesz usunąć raport {report_name}?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if reply != QMessageBox.StandardButton.Yes:
                return

            # Usuń plik raportu
            report_file = os.path.join("data", "reports", report_name + ".json")
            if os.path.exists(report_file):
                os.remove(report_file)

            # Usuń pliki powiązane (PDF, Excel)
            for ext in [".pdf", ".xlsx"]:
                related_file = os.path.splitext(report_file)[0] + ext
                if os.path.exists(related_file):
                    os.remove(related_file)

            # Odśwież listę raportów
            self.refresh()

            # Wyświetl komunikat o sukcesie
            QMessageBox.information(
                self, "Sukces", f"Raport {report_name} został usunięty."
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Błąd", f"Nie udało się usunąć raportu: {str(e)}"
            )

    def _compare_generated_reports(self):
        """Porównuje wygenerowane raporty."""
        try:
            selected_items = self.results_table.selectedItems()
            if len(selected_items) < 2:
                QMessageBox.warning(
                    self, "Błąd", "Proszę wybrać co najmniej dwa raporty do porównania."
                )
                return

            report_paths = []
            for item in selected_items:
                row = item.row()
                report_name = self.results_table.item(row, 0).text()
                report_path = os.path.join(
                    self.settings.get("reports_dir", "data/reports"),
                    f"{report_name}.json",
                )
                if os.path.exists(report_path):
                    report_paths.append(report_path)

            if len(report_paths) < 2:
                QMessageBox.warning(
                    self,
                    "Błąd",
                    "Nie znaleziono wystarczającej liczby raportów do porównania.",
                )
                return

            self._generate_comparison_report(report_paths)

        except Exception as e:
            QMessageBox.critical(
                self, "Błąd", f"Wystąpił błąd podczas porównywania raportów: {str(e)}"
            )

    def _generate_comparison_report(self, report_paths):
        """Generuje raport porównawczy na podstawie wybranych raportów."""
        try:
            comparison_data = {"reports": [], "summary": {}, "charts": {}}

            for report_path in report_paths:
                with open(report_path, "r", encoding="utf-8") as f:
                    report_data = json.load(f)
                    comparison_data["reports"].append(report_data)

            # Generuj podsumowanie
            self._generate_comparison_summary(comparison_data)

            # Generuj wykresy
            if self.include_charts_checkbox.isChecked():
                self._generate_comparison_charts(comparison_data)

            # Zapisz raport
            report_name = (
                f"Porównanie-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
            )
            report_path = os.path.join(
                self.settings.get("reports_dir", "data/reports"), f"{report_name}.json"
            )

            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(comparison_data, f, indent=4, ensure_ascii=False)

            # Generuj HTML
            html_path = report_path.replace(".json", ".html")
            self._generate_html_comparison_report(comparison_data, html_path)

            QMessageBox.information(
                self, "Sukces", "Raport porównawczy został wygenerowany pomyślnie."
            )

            self.refresh()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Błąd",
                f"Wystąpił błąd podczas generowania raportu porównawczego: {str(e)}",
            )

    def _preview_comparison_report(self):
        """Podgląd wygenerowanego raportu porównawczego."""
        try:
            selected_items = self.results_table.selectedItems()
            if not selected_items:
                QMessageBox.warning(self, "Błąd", "Proszę wybrać raport do podglądu.")
                return

            row = selected_items[0].row()
            report_name = self.results_table.item(row, 0).text()
            report_path = os.path.join(
                self.settings.get("reports_dir", "data/reports"), f"{report_name}.html"
            )

            if not os.path.exists(report_path):
                QMessageBox.warning(self, "Błąd", "Nie znaleziono pliku raportu HTML.")
                return

            # Otwórz raport w domyślnej przeglądarce
            webbrowser.open(f"file://{os.path.abspath(report_path)}")

        except Exception as e:
            QMessageBox.critical(
                self, "Błąd", f"Wystąpił błąd podczas podglądu raportu: {str(e)}"
            )

    def _save_comparison_report(self):
        """Zapisuje raport porównawczy w wybranym formacie."""
        try:
            selected_items = self.results_table.selectedItems()
            if not selected_items:
                QMessageBox.warning(self, "Błąd", "Proszę wybrać raport do zapisania.")
                return

            row = selected_items[0].row()
            report_name = self.results_table.item(row, 0).text()
            report_path = os.path.join(
                self.settings.get("reports_dir", "data/reports"), report_name
            )

            # Wybierz format zapisu
            format_dialog = QMessageBox(self)
            format_dialog.setWindowTitle("Wybierz format")
            format_dialog.setText("W jakim formacie chcesz zapisać raport?")
            pdf_button = format_dialog.addButton(
                "PDF", QMessageBox.ButtonRole.ActionRole
            )
            excel_button = format_dialog.addButton(
                "Excel", QMessageBox.ButtonRole.ActionRole
            )
            cancel_button = format_dialog.addButton(
                "Anuluj", QMessageBox.ButtonRole.RejectRole
            )

            format_dialog.exec()

            if format_dialog.clickedButton() == pdf_button:
                self._export_to_pdf()
            elif format_dialog.clickedButton() == excel_button:
                self._export_to_excel()
            else:
                return

        except Exception as e:
            QMessageBox.critical(
                self, "Błąd", f"Wystąpił błąd podczas zapisywania raportu: {str(e)}"
            )

    def _generate_comparison_summary(self, comparison_data):
        """Generuje podsumowanie porównania raportów."""
        try:
            summary = {
                "total_reports": len(comparison_data["reports"]),
                "common_metrics": {},
                "differences": [],
            }

            # Znajdź wspólne metryki
            if comparison_data["reports"]:
                first_report = comparison_data["reports"][0]
                for metric in first_report.get("metrics", {}):
                    if all(
                        metric in report.get("metrics", {})
                        for report in comparison_data["reports"]
                    ):
                        values = [
                            report["metrics"][metric]
                            for report in comparison_data["reports"]
                        ]
                        summary["common_metrics"][metric] = {
                            "min": min(values),
                            "max": max(values),
                            "avg": sum(values) / len(values),
                        }

            # Znajdź różnice
            for i, report1 in enumerate(comparison_data["reports"]):
                for j, report2 in enumerate(comparison_data["reports"][i + 1 :], i + 1):
                    diff = self._compare_two_reports(report1, report2)
                    if diff:
                        summary["differences"].append(
                            {
                                "report1": report1.get("name", f"Raport {i+1}"),
                                "report2": report2.get("name", f"Raport {j+1}"),
                                "differences": diff,
                            }
                        )

            comparison_data["summary"] = summary

        except Exception as e:
            QMessageBox.critical(
                self,
                "Błąd",
                f"Wystąpił błąd podczas generowania podsumowania: {str(e)}",
            )

    def _compare_two_reports(self, report1, report2):
        """Porównuje dwa raporty i zwraca listę różnic."""
        differences = []

        # Porównaj metryki
        metrics1 = report1.get("metrics", {})
        metrics2 = report2.get("metrics", {})

        for metric in set(metrics1.keys()) | set(metrics2.keys()):
            if metric not in metrics1:
                differences.append(f"Brak metryki {metric} w pierwszym raporcie")
            elif metric not in metrics2:
                differences.append(f"Brak metryki {metric} w drugim raporcie")
            elif metrics1[metric] != metrics2[metric]:
                differences.append(
                    f"Różnica w metryce {metric}: "
                    f"{metrics1[metric]} vs {metrics2[metric]}"
                )

        return differences

    def _generate_comparison_charts(self, comparison_data):
        """Generuje wykresy porównawcze."""
        try:
            charts = {
                "metrics_comparison": {},
                "performance_trends": {},
                "distribution_charts": {},
            }

            # Generuj wykresy porównawcze metryk
            for metric, values in comparison_data["summary"]["common_metrics"].items():
                charts["metrics_comparison"][metric] = {
                    "min": values["min"],
                    "max": values["max"],
                    "avg": values["avg"],
                }

            # Generuj wykresy trendów wydajności
            for report in comparison_data["reports"]:
                if "performance_history" in report:
                    for metric, history in report["performance_history"].items():
                        if metric not in charts["performance_trends"]:
                            charts["performance_trends"][metric] = []
                        charts["performance_trends"][metric].append(history)

            # Generuj wykresy rozkładu
            for report in comparison_data["reports"]:
                if "distributions" in report:
                    for metric, distribution in report["distributions"].items():
                        if metric not in charts["distribution_charts"]:
                            charts["distribution_charts"][metric] = []
                        charts["distribution_charts"][metric].append(distribution)

            comparison_data["charts"] = charts

        except Exception as e:
            QMessageBox.critical(
                self, "Błąd", f"Wystąpił błąd podczas generowania wykresów: {str(e)}"
            )
