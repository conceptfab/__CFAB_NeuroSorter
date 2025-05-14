TODO 0.4

- optymalizacja sprzętowa -> wartość uzytkownika -> wartość z profilu sprzętowego
- nie dziala przycisk zastosuj optymalizacje
- usun przycisk pokaz log
- do sprawdzenia czy to potrzebne w pliku zadania: },
        "optimization": {
            "batch_size": 128,
            "num_workers": 16,
            "use_mixed_precision": true
        }

- ikona i nazwa data splitter

- AI usunąć wczytywanie parametrów profili sprzętowego - wszystkie parametry są z pliku json zadania
- nie dziala kolejkowanie
- przy skonczeniu zadania brakuje zmiany statusu NOWY/ZAKOŃCZONY
- plik log w JSON classifier.py - dodanie incfo z loga
- brak czasu trening w zadaniau normalnym
- uprządkować strukturę JSON razem z logiem
- nie dziala wsadowe przetwarzanie

- optymalizacja CUDA w tworzeniu zadań
- kolejka wsadowa nie zmienia statusu zadadani, nie zapisuje renderu treningu

- sortowanie poprawic - pliki o niskim ratingu do osobnego folderu - sprawdzic czy to dziala
- sortw
- pomoc dodatkowe narzędzia

- skalowanie i skaner w jednej zakładce
- skąd ten log?
  Konfiguracja modelu: - Architektura: b0 - Liczba klas: 10 - Drop connect rate: 0.2 - Dropout rate: 0.3

<!-- - fine-tuning dopracowac - dialog do poprawy ładowanie profili nie dziala -->

- dodatkowy algorytm do przygotowania danych - podstawowy minimalna ilość plików podził procentowy
- opcja zapisuj render treningu
- dodac metodę testowania
- mapa projektu
- optymalizacja zadan treningowych - nowy przycisk, pobieranie danych o sprzęcie
- walidacja danych treningowych przycisk dane w liscie zadań
- update dokumentacji, do opisania opcje doszkalania i treningu
