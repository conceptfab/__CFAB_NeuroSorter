Analiza zmian w kodzie dotyczących mapowania klas i propozycje poprawek
Analizując dostarczone pliki, zauważyłem istotne zmiany w sposobie implementacji mapowania klas w klasyfikatorze, które nie są jeszcze obsługiwane przez moduł sortowania. Przedstawię znalezione problemy i rozwiązania.
Główny problem
Zmieniono sposób przechowywania i obsługi mapowania klas w pliku ai/classifier.py, jednak moduł app/sorter/image_sorter.py wciąż używa starego podejścia, co powoduje niezgodności.
Szczegółowa analiza zmian
Zmiany w pliku ai/classifier.py:

Klasa ImageClassifier teraz używa słownika class_names do mapowania indeksów klas do nazw klas, gdzie:

Klucze to stringi (indeksy klas jako napisy, np. "0", "1", "2")
Wartości to nazwy klas (np. "kot", "pies", "ryba")


Podczas klasyfikacji (predict method), konwersja indeksu klasy do nazwy jest wykonywana w inny sposób:
pythonkey_to_find = str(predicted_class)  # Klucz, którego szukamy
class_name = None

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


Problem w pliku app/sorter/image_sorter.py:
Metoda _process_image zawiera następujący kod, który nie działa z nowym formatem mapowania klas:
python# Sprawdź czy kategoria jest w mapowaniu klas
class_mapping = self.classifier.get_class_mapping()
if category not in class_mapping:
    logger.warning(f"Wykryto nieznaną kategorię: {category}")
    category = "nieskategoryzowane"
Problem polega na tym, że class_mapping to teraz słownik, gdzie kluczami są indeksy klas, a nie nazwy klas. Sprawdzanie czy category jest w class_mapping nie zadziała prawidłowo, ponieważ category jest nazwą klasy (wartością), a nie kluczem w słowniku class_mapping.
Propozycje poprawek
Zmiana w pliku app/sorter/image_sorter.py:

Zmodyfikuj metodę _process_image w następujący sposób:

pythondef _process_image(self, image_path, output_dir, created_dirs, confidence_threshold=0.5):
    # ... (istniejący kod)
    
    # Sprawdź czy kategoria jest w mapowaniu klas
    class_mapping = self.classifier.get_class_mapping()
    
    # Sprawdzenie czy kategoria jest wartością w słowniku class_mapping
    if category is not None and class_mapping:
        category_exists = category in class_mapping.values()
        if not category_exists:
            logger.warning(f"Wykryto nieznaną kategorię: {category}")
            category = "nieskategoryzowane"
    
    # ... (pozostały kod)

Zmodyfikuj metodę _verify_model_and_mapping:

pythondef _verify_model_and_mapping(self):
    """Weryfikuje czy model i mapowanie klas są poprawnie skonfigurowane."""
    try:
        # Sprawdź czy model ma zdefiniowane mapowanie klas
        class_mapping = self.classifier.get_class_mapping()
        if not class_mapping:
            logger.warning("Model nie ma zdefiniowanego mapowania klas!")
            return False

        # Sprawdź czy mapowanie klas jest poprawne
        if not all(isinstance(k, str) for k in class_mapping.keys()):
            logger.warning("Klucze w mapowaniu klas nie są typu string!")
            
        if not all(isinstance(v, str) for v in class_mapping.values()):
            logger.warning("Wartości w mapowaniu klas nie są typu string!")

        logger.info(f"Mapowanie klas zweryfikowane: {len(class_mapping)} kategorii")
        return True

    except Exception as e:
        logger.error(f"Błąd podczas weryfikacji modelu: {str(e)}")
        return False
Dodatkowa poprawka dla spójności danych:
Dodanie metody pomocniczej, która konwertuje mapowanie klas na format zgodny z modułem sortowania:
pythondef _get_category_names(self):
    """Zwraca listę wszystkich dostępnych nazw kategorii z modelu."""
    class_mapping = self.classifier.get_class_mapping()
    if not class_mapping:
        return []
    
    # Zwróć unikalne nazwy kategorii
    return list(set(class_mapping.values()))
Podsumowanie zmian do wprowadzenia w pliku app/sorter/image_sorter.py:

Zmodyfikować fragmenty kodu, które sprawdzają, czy kategoria znajduje się w mapowaniu klas
Zmienić metodę weryfikacji modelu i mapowania klas
Dodać metodę pomocniczą do uzyskiwania listy nazw kategorii

Pełny kod poprawki dla pliku app/sorter/image_sorter.py
python# W metodzie _process_image:

# Sprawdź czy kategoria jest w mapowaniu klas
class_mapping = self.classifier.get_class_mapping()

# Sprawdzenie czy kategoria jest wartością w słowniku class_mapping
if category is not None and class_mapping:
    category_exists = category in class_mapping.values()
    if not category_exists:
        logger.warning(f"Wykryto nieznaną kategorię: {category}")
        category = "nieskategoryzowane"
Te zmiany zapewnią poprawne działanie modułu sortowania z nowym formatem mapowania klas.