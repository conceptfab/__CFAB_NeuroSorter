Znalezione błędy i proponowane zmiany
1. Błąd w ai/classifier.py, w metodzie predict
Problem polega na tym, że funkcja predict nie sprawdza poprawnie przypadku gdy klasyfikacja zwraca None. Spróbujmy to naprawić:
pythondef predict(self, image_path):
    """
    Klasyfikuje pojedynczy obraz.

    Args:
        image_path: Ścieżka do pliku obrazu

    Returns:
        Słownik z wynikami klasyfikacji lub None w przypadku błędu
    """
    try:
        # Sprawdź czy plik istnieje
        if not os.path.exists(image_path):
            print(f"Błąd: Plik nie istnieje: {image_path}")
            return None

        # Sprawdź czy plik jest obrazem
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Błąd podczas otwierania obrazu {image_path}: {e}")
            return None

        # Wczytaj i przekształć obraz
        try:
            image_tensor = self.transform(image).unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
        except Exception as e:
            print(f"Błąd podczas przetwarzania obrazu {image_path}: {e}")
            return None

        # Przełącz model w tryb ewaluacji
        self.model.eval()

        # Wykonaj predykcję
        try:
            with torch.no_grad():
                if self.precision == "half" and self.device.type == "cuda":
                    image_tensor = image_tensor.half()
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
        except Exception as e:
            print(f"Błąd podczas predykcji dla obrazu {image_path}: {e}")
            return None

        # Konwertuj indeks klasy na nazwę
        key_to_find = str(predicted_class)  # Klucz, którego szukamy
        class_name = None

        # Sprawdź czy mamy mapowanie klas
        if not hasattr(self, "class_names") or not self.class_names:
            print("OSTRZEŻENIE: Brak mapowania klas w modelu!")
            class_name = f"Klasa_{predicted_class}"
        else:
            # Najpierw szukaj dokładnie
            if key_to_find in self.class_names:
                class_name = self.class_names[key_to_find]
            else:
                # Próba konwersji kluczy
                for k, v in self.class_names.items():
                    try:
                        if str(k) == key_to_find:
                            class_name = v
                            break
                    except:
                        continue

                if class_name is None:
                    class_name = f"Klasa_{predicted_class}"

        return {
            "class_name": class_name,
            "class_id": predicted_class,
            "confidence": confidence,
        }

    except Exception as e:
        print(f"Błąd podczas klasyfikacji obrazu {image_path}: {e}")
        traceback.print_exc()  # Dodajemy wydruk pełnego traceback'a
        return None
2. Problem w app/sorter/image_sorter.py w metodzie _process_image
W metodzie _process_image klasy ImageSorter jest problem z obsługą przypadku, gdy klasyfikator zwraca None. Naprawmy to:
pythondef _process_image(self, image_path: str) -> Dict[str, Any]:
    """
    Przetwarza pojedynczy obraz.

    Args:
        image_path: Ścieżka do obrazu

    Returns:
        Dict[str, Any]: Słownik z wynikami przetwarzania
    """
    result = {
        "status": "error",
        "message": "",
        "image_path": image_path,
        "category": None,
        "confidence": None,
    }

    try:
        # Sprawdź czy plik istnieje i ma uprawnienia do odczytu
        if not os.path.exists(image_path):
            result["message"] = f"Plik nie istnieje: {image_path}"
            return result

        if not os.access(image_path, os.R_OK):
            result["message"] = f"Brak uprawnień do odczytu pliku: {image_path}"
            return result

        # Klasyfikuj obraz
        logger.info(f"Klasyfikacja obrazu: {image_path}")
        classification_result = self.classifier.predict(image_path)

        # Sprawdź czy klasyfikacja się powiodła
        if classification_result is None:
            result["message"] = "Nie można sklasyfikować obrazu."
            logger.info("Nie można sklasyfikować obrazu.")
            return result

        # Sprawdź czy wynik zawiera wymagane pola
        if not all(
            key in classification_result for key in ["class_name", "confidence"]
        ):
            result["message"] = "Nieprawidłowy format wyniku klasyfikacji"
            logger.warning(f"Nieprawidłowy format wyniku: {classification_result}")
            return result

        # Pobierz mapowanie klas
        class_mapping = {}
        if hasattr(self.classifier, "get_class_mapping"):
            class_mapping = self.classifier.get_class_mapping() or {}

        # Aktualizuj wynik
        category = classification_result["class_name"]
        result.update(
            {
                "status": "success",
                "category": category,
                "confidence": classification_result["confidence"],
            }
        )

        logger.info(
            f"Obraz sklasyfikowany jako {category} "
            f"(pewność: {classification_result['confidence']:.2f})"
        )

    except Exception as e:
        result["message"] = f"Błąd klasyfikacji: {str(e)}"
        logger.error(f"Błąd podczas przetwarzania obrazu {image_path}: {e}")
        logger.error(traceback.format_exc())

    return result
3. Problemy z walidacją kategorii w app/sorter/image_sorter.py
Metoda sprawdzania czy kategoria należy do mapowania jest zbyt restrykcyjna. Naprawmy funkcję _process_image:
python# Pobierz mapowanie klas
class_mapping = {}
if hasattr(self.classifier, "get_class_mapping"):
    class_mapping = self.classifier.get_class_mapping() or {}

# Aktualizuj wynik - usuńmy sprawdzanie czy kategoria jest w mapowaniu
category = classification_result["class_name"]
result.update(
    {
        "status": "success",
        "category": category,
        "confidence": classification_result["confidence"],
    }
)
4. Problem z sortowaniem plików w app/sorter/image_sorter.py
Należy upewnić się, że katalogi kategorii są poprawnie tworzone:
pythonif result["status"] == "success":
    category = result["category"]
    
    # Upewnij się, że kategoria nie jest None
    if category is None:
        category = self.uncategorized_dir
        
    target_dir = os.path.join(output_dir, category)
    os.makedirs(target_dir, exist_ok=True)
5. Dodanie debugowania w kluczowych miejscach
Aby lepiej zrozumieć, co się dzieje podczas klasyfikacji, warto dodać więcej logowania w kluczowych miejscach:
python# W metodzie predict klasy ImageClassifier, przed wykonaniem predykcji:
print(f"DEBUG: Klasyfikacja obrazu: {image_path}")
print(f"DEBUG: Typ modelu: {self.model_type}")
print(f"DEBUG: Liczba klas: {self.num_classes}")
print(f"DEBUG: Mapowanie klas: {self.class_names}")

# Po wykonaniu predykcji:
print(f"DEBUG: Przewidziana klasa: {predicted_class}")
print(f"DEBUG: Pewność: {confidence:.4f}")
print(f"DEBUG: Nazwa klasy: {class_name}")
Podsumowanie zmian

W metodzie predict klasy ImageClassifier - dodano lepszą obsługę błędów i debugowanie
W metodzie _process_image klasy ImageSorter - poprawiono obsługę przypadku, gdy classification_result jest None, dodano debugowanie
Usunięto zbyt restrykcyjne sprawdzanie, czy kategoria należy do mapowania
Poprawiono tworzenie katalogów kategorii w metodzie sort_directory klasy ImageSorter
Dodano więcej logowania i debugowania w kluczowych miejscach

Te zmiany powinny rozwiązać problem z klasyfikacją obrazów. Głównym problemem był brak odpowiedniej obsługi przypadku, gdy metoda predict zwraca None lub gdy obiekt class_names nie jest poprawnie zainicjalizowany.