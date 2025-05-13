import json
import os
from datetime import datetime

import piexif
from PIL import Image
from PIL.ExifTags import TAGS


class MetadataManager:
    """Klasa do zarządzania metadanymi plików graficznych."""

    def __init__(self):
        """Inicjalizacja menedżera metadanych."""
        pass

    def add_category_to_image(self, image_path, category_data):
        """
        Dodaje informację o kategorii do metadanych obrazu.

        Args:
            image_path: Ścieżka do pliku obrazu
            category_data: Nazwa kategorii jako string lub słownik z metadanymi kategorii

        Returns:
            bool: True jeśli operacja się powiodła, False w przeciwnym wypadku
        """
        try:
            # Konwersja category_data na format JSON, jeśli to słownik
            category_value = (
                json.dumps(category_data)
                if isinstance(category_data, dict)
                else str(category_data)
            )

            # Sprawdź format pliku
            _, ext = os.path.splitext(image_path.lower())

            if ext in [".jpg", ".jpeg"]:
                # Dla plików JPEG używamy biblioteki piexif
                try:
                    # Wczytaj istniejące dane EXIF
                    exif_dict = piexif.load(image_path)
                except Exception as e:
                    print(f"Nie można wczytać danych EXIF z {image_path}: {str(e)}")
                    # Jeśli nie ma danych EXIF, utwórz pusty słownik
                    exif_dict = {
                        "0th": {},
                        "Exif": {},
                        "GPS": {},
                        "1st": {},
                        "thumbnail": None,
                    }

                # Dodaj kategorię do pola UserComment w sekcji Exif
                category_bytes = category_value.encode("utf-8")
                exif_dict["Exif"][piexif.ExifIFD.UserComment] = category_bytes

                # Zapisz zmodyfikowane dane EXIF
                exif_bytes = piexif.dump(exif_dict)
                piexif.insert(exif_bytes, image_path)
                return True

            elif ext in [".png", ".tiff", ".tif", ".webp"]:
                # Dla innych obsługiwanych formatów używamy PIL
                try:
                    img = Image.open(image_path)

                    # Sprawdź czy obraz ma już metadane
                    metadata = img.info or {}

                    # Dodaj kategorię do metadanych
                    metadata["Category"] = category_value

                    # Zapisz obraz z nowymi metadanymi
                    img.save(image_path, **metadata)
                    return True
                except Exception as e:
                    print(f"Nie można zapisać metadanych do {image_path}: {str(e)}")
                    return False

            else:
                print(f"Nieobsługiwany format pliku: {ext}")
                return False

        except Exception as e:
            print(f"Błąd podczas dodawania metadanych do {image_path}: {str(e)}")
            return False

    def get_category_from_image(self, image_path):
        """
        Odczytuje kategorię z metadanych obrazu.

        Args:
            image_path: Ścieżka do pliku obrazu

        Returns:
            str/dict: Nazwa kategorii, słownik z metadanymi lub None jeśli nie znaleziono
        """
        try:
            # Sprawdź format pliku
            _, ext = os.path.splitext(image_path.lower())

            if ext in [".jpg", ".jpeg"]:
                # Dla plików JPEG używamy biblioteki piexif
                try:
                    exif_dict = piexif.load(image_path)
                    if piexif.ExifIFD.UserComment in exif_dict["Exif"]:
                        category_bytes = exif_dict["Exif"][piexif.ExifIFD.UserComment]
                        category_str = category_bytes.decode("utf-8")

                        # Sprawdź, czy wartość to JSON
                        try:
                            return json.loads(category_str)
                        except json.JSONDecodeError:
                            # Jeśli nie jest JSON, zwróć jako string
                            return category_str
                except:
                    pass

            elif ext in [".png", ".tiff", ".tif", ".webp"]:
                # Dla innych obsługiwanych formatów używamy PIL
                with Image.open(image_path) as img:
                    if "Category" in img.info:
                        category_str = img.info["Category"]
                        # Sprawdź, czy wartość to JSON
                        try:
                            return json.loads(category_str)
                        except json.JSONDecodeError:
                            # Jeśli nie jest JSON, zwróć jako string
                            return category_str

            return None

        except Exception as e:
            print(f"Błąd podczas odczytywania metadanych: {str(e)}")
            return None

    def add_model_info_to_image(self, image_path, classifier):
        """
        Dodaje informacje o modelu do metadanych obrazu.

        Args:
            image_path: Ścieżka do pliku obrazu
            classifier: Instancja ImageClassifier

        Returns:
            bool: True jeśli operacja się powiodła, False w przeciwnym wypadku
        """
        try:
            # Pobierz informacje o modelu
            model_info = {
                "model_type": classifier.model_type,
                "model_version": datetime.now().strftime("%Y%m%d"),
                "classifier_info": "AI Image Classifier",
            }

            # Serializuj do JSON
            model_info_str = json.dumps(model_info)

            # Dodaj do metadanych
            _, ext = os.path.splitext(image_path.lower())

            if ext in [".jpg", ".jpeg"]:
                try:
                    exif_dict = piexif.load(image_path)
                except:
                    exif_dict = {
                        "0th": {},
                        "Exif": {},
                        "GPS": {},
                        "1st": {},
                        "thumbnail": None,
                    }

                # Dodaj do pola ExifVersion w sekcji Exif
                model_bytes = model_info_str.encode("utf-8")
                exif_dict["Exif"][piexif.ExifIFD.ExifVersion] = model_bytes

                exif_bytes = piexif.dump(exif_dict)
                piexif.insert(exif_bytes, image_path)
                return True

            elif ext in [".png", ".tiff", ".tif", ".webp"]:
                with Image.open(image_path) as img:
                    metadata = img.info or {}
                    metadata["ModelInfo"] = model_info_str
                    img.save(image_path, **metadata)
                    return True

            return False
        except Exception as e:
            print(f"Błąd podczas dodawania informacji o modelu: {str(e)}")
            return False
