"""Style przycisków dla aplikacji."""

# Kolory zgodne z Material Design i VS Code
PRIMARY_COLOR = "#007ACC"  # Niebieski VS Code
SUCCESS_COLOR = "#10B981"  # Zielony
WARNING_COLOR = "#DC2626"  # Czerwony
BACKGROUND = "#1E1E1E"  # Ciemne tło
SURFACE = "#252526"  # Lekko jaśniejsze tło dla paneli
BORDER_COLOR = "#3F3F46"  # Kolor obramowania
TEXT_COLOR = "#CCCCCC"  # Kolor tekstu

# Style przycisków
BUTTON_STYLES = {
    "default": f"""
        QPushButton {{
            background-color: {SURFACE};
            color: {TEXT_COLOR};
            border: 1px solid {BORDER_COLOR};
            border-radius: 2px;
            padding: 4px 12px;
            min-height: 24px;
            max-height: 24px;
        }}
        QPushButton:hover {{
            background-color: #2A2D2E;
        }}
        QPushButton:pressed {{
            background-color: #3E3E40;
        }}
    """,
    
    "primary": f"""
        QPushButton {{
            background-color: {PRIMARY_COLOR};
            color: white;
            border: none;
            border-radius: 2px;
            padding: 4px 12px;
            min-height: 24px;
            max-height: 24px;
        }}
        QPushButton:hover {{
            background-color: #1C97EA;
        }}
        QPushButton:pressed {{
            background-color: #005A9E;
        }}
    """,
    
    "success": f"""
        QPushButton {{
            background-color: {SUCCESS_COLOR};
            color: white;
            border: none;
            border-radius: 2px;
            padding: 4px 12px;
            min-height: 24px;
            max-height: 24px;
        }}
        QPushButton:hover {{
            background-color: #059669;
        }}
        QPushButton:pressed {{
            background-color: #047857;
        }}
    """,
    
    "warning": f"""
        QPushButton {{
            background-color: {WARNING_COLOR};
            color: white;
            border: none;
            border-radius: 2px;
            padding: 4px 12px;
            min-height: 24px;
            max-height: 24px;
        }}
        QPushButton:hover {{
            background-color: #EF4444;
        }}
        QPushButton:pressed {{
            background-color: #B91C1C;
        }}
    """,
    
    "stop": f"""
        QPushButton {{
            background-color: {WARNING_COLOR};
            color: white;
            border: none;
            border-radius: 4px;
            padding: 4px 8px;
            min-height: 24px;
            max-height: 24px;
        }}
        QPushButton:hover {{
            background-color: #EF4444;
        }}
        QPushButton:disabled {{
            background-color: #4B5563;
            color: #9CA3AF;
        }}
    """
} 