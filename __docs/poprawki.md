Podobnie jak poprzednio z plikiem ai/preprocessing.py, zamiast kontynuować próby z automatyczną edycją, która zawodzi, sugeruję, abyś ręcznie wprowadził te zmiany.
Przygotuję teraz listę konkretnych modyfikacji (co usunąć, co zamienić), abyś mógł je łatwo zastosować w swoim edytorze kodu.
Proponowane ręczne zmiany w ai/fine_tuning.py:
Sekcja importów (początek pliku):
Usuń balanced_accuracy_score, z importu sklearn.metrics.
Usuń confusion_matrix, z importu sklearn.metrics.
W imporcie from .catastrophic_forgetting import (:
Usuń ElasticWeightConsolidation,
Usuń KnowledgeDistillationLoss,
Usuń RehearsalMemory,
Zamiana f-stringów na zwykłe stringi (przykłady, przejrzyj cały plik pod kątem podobnych):
print(f"\n=== INICJALIZACJA FINE-TUNINGU ===") zmień na print("\n=== INICJALIZACJA FINE-TUNINGU ===")
print(f"\n=== WERYFIKACJA STRUKTURY KATALOGÓW ===") zmień na print("\n=== WERYFIKACJA STRUKTURY KATALOGÓW ===")
print(f"\n=== KONFIGURACJA REHEARSAL ===") zmień na print("\n=== KONFIGURACJA REHEARSAL ===")
I tak dalej dla wszystkich print(f"...") gdzie ... nie zawiera żadnych zmiennych {}.
Usunięcie nieużywanych zmiennych (okolice linii 1024-1025, numeracja może być już inna po poprzednich zmianach):
Znajdź sekcję:
Apply to poprawki.md
Usuń linie:
Apply to poprawki.md
Tak aby zostało:
Apply to poprawki.md
