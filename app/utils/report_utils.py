"""
Moduł zawierający funkcje pomocnicze do generowania raportów.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.core.logger import Logger

logger = Logger()


def generate_report_pdf(
    output_path: str, title: str, data: Dict, template: Optional[str] = None
) -> bool:
    """
    Generuje raport PDF na podstawie przekazanych danych.

    Args:
        output_path (str): Ścieżka do zapisu pliku PDF
        title (str): Tytuł raportu
        data (Dict): Dane do umieszczenia w raporcie
        template (Optional[str]): Ścieżka do szablonu raportu

    Returns:
        bool: True jeśli raport został wygenerowany pomyślnie, False w przeciwnym razie
    """
    try:
        # Utworzenie dokumentu PDF
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72,
        )

        # Style
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "CustomTitle", parent=styles["Heading1"], fontSize=24, spaceAfter=30
        )

        # Elementy dokumentu
        elements = []

        # Tytuł
        elements.append(Paragraph(title, title_style))
        elements.append(Spacer(1, 12))

        # Data wygenerowania
        date_style = ParagraphStyle(
            "Date", parent=styles["Normal"], fontSize=10, textColor=colors.gray
        )
        elements.append(
            Paragraph(
                f"Wygenerowano: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                date_style,
            )
        )
        elements.append(Spacer(1, 20))

        # Dane
        if isinstance(data, dict):
            table_data = [
                [Paragraph(k, styles["Normal"]), Paragraph(str(v), styles["Normal"])]
                for k, v in data.items()
            ]

            table = Table(table_data, colWidths=[2 * inch, 4 * inch])
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
                        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                        ("FONTSIZE", (0, 0), (-1, -1), 10),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ]
                )
            )
            elements.append(table)

        # Generowanie PDF
        doc.build(elements)
        logger.info(f"Raport został wygenerowany: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Błąd podczas generowania raportu: {str(e)}")
        return False


def load_report_template(template_path: str) -> Optional[Dict]:
    """
    Ładuje szablon raportu z pliku JSON.

    Args:
        template_path (str): Ścieżka do pliku szablonu

    Returns:
        Optional[Dict]: Wczytany szablon lub None w przypadku błędu
    """
    try:
        if os.path.exists(template_path):
            with open(template_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None
    except Exception as e:
        logger.error(f"Błąd podczas ładowania szablonu raportu: {str(e)}")
        return None


def save_report_template(template_path: str, template_data: Dict) -> bool:
    """
    Zapisuje szablon raportu do pliku JSON.

    Args:
        template_path (str): Ścieżka do zapisu szablonu
        template_data (Dict): Dane szablonu do zapisania

    Returns:
        bool: True jeśli szablon został zapisany pomyślnie, False w przeciwnym razie
    """
    try:
        with open(template_path, "w", encoding="utf-8") as f:
            json.dump(template_data, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Błąd podczas zapisywania szablonu raportu: {str(e)}")
        return False
