import os
import sqlite3
from datetime import datetime


class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self._create_tables()

    def _create_tables(self):
        """Tworzy tabele w bazie danych jeśli nie istnieją"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Tabela klasyfikacji
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS classifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_path TEXT NOT NULL,
                    class_id INTEGER NOT NULL,
                    class_name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Tabela kategorii
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS categories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT
                )
            """
            )

            conn.commit()

    def add_classification(self, image_path, class_id, class_name, confidence):
        """Dodaje nową klasyfikację do bazy danych."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO classifications (image_path, class_id, class_name, confidence)
                VALUES (?, ?, ?, ?)
                """,
                (image_path, class_id, class_name, confidence),
            )
            conn.commit()
            return cursor.lastrowid

    def get_classifications(self, limit=100):
        """Pobiera ostatnie klasyfikacje"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM classifications
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (limit,),
            )
            return cursor.fetchall()

    def add_category(self, name, description=None):
        """Dodaje nową kategorię do bazy danych."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO categories (name, description)
                VALUES (?, ?)
                """,
                (name, description),
            )
            conn.commit()
            return cursor.lastrowid

    def get_categories(self):
        """Pobiera wszystkie kategorie"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM categories")
            return cursor.fetchall()
