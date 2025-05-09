# Analiza projektu CFAB_NeuroSorter

## 1. Struktura główna projektu

### Pliki główne

- `cfabNS.py` - główny plik aplikacji
- `scaller.py` - moduł skalowania obrazów
- `fix_png.py` - narzędzie do naprawy plików PNG
- `run_resolution_scanner.py` - narzędzie diagnostyczne
- `przygotuj_offline.py` - narzędzie do przygotowania trybu offline
- `__init__.py` - inicjalizacja pakietu

### Katalogi główne

- `app/` - główny katalog aplikacji
- `ai/` - moduły sztucznej inteligencji
- `models/` - modele i wagi
- `config/` - konfiguracja
- `data/` - dane
- `logs/` - logi
- `reports/` - raporty

### Szczegółowa struktura katalogów

```
CFAB_NeuroSorter/
├── app/                    # Główny katalog aplikacji
│   ├── gui/               # Interfejs użytkownika
│   │   ├── dialogs/       # Okna dialogowe
│   │   ├── widgets/       # Komponenty interfejsu
│   │   ├── tabs/          # Zakładki interfejsu
│   │   └── main_window.py # Główne okno aplikacji
│   ├── core/              # Rdzeń aplikacji
│   │   ├── workers/       # Wątki robocze
│   │   ├── logger.py      # System logowania
│   │   ├── notifications.py # System powiadomień
│   │   └── optimizations.py # Optymalizacje
│   ├── utils/             # Narzędzia pomocnicze
│   │   ├── file_utils.py  # Operacje na plikach
│   │   └── image_utils.py # Przetwarzanie obrazów
│   ├── sorter/            # Moduł sortowania
│   ├── resources/         # Zasoby
│   ├── metadata/          # Metadane
│   └── database/          # Baza danych
├── ai/                    # Moduły AI
│   ├── classifier.py      # Klasyfikator obrazów
│   ├── training.py        # Trenowanie modeli
│   └── models.py          # Definicje modeli
├── models/                # Modele i wagi
├── config/                # Konfiguracja
├── data/                  # Dane
│   ├── models/           # Wytrenowane modele
│   └── tasks/            # Zadania
├── logs/                  # Logi
└── reports/               # Raporty
```

## 2. Analiza katalogu app/

### Struktura katalogu app/

- `gui/` - interfejs użytkownika
- `core/` - rdzeń aplikacji
- `utils/` - narzędzia pomocnicze
- `sorter/` - moduł sortowania
- `resources/` - zasoby
- `metadata/` - metadane
- `database/` - baza danych

### Analiza plików w app/gui/

- `main_window.py` - główne okno aplikacji
- `tab_interface.py` - interfejs zakładek
- `__init__.py` - inicjalizacja pakietu GUI

### Analiza plików w app/core/

- `logger.py` - system logowania
- `notifications.py` - system powiadomień
- `optimizations.py` - optymalizacje
- `state_manager.py` - zarządzanie stanem
- `__init__.py` - inicjalizacja pakietu core

## 3. Zależności i użycie plików

### Aktywnie używane pliki

1. `cfabNS.py`

   - Importuje: PyQt6, app.core.logger, app.gui.main_window, app.utils.file_utils
   - Jest głównym punktem wejścia aplikacji
   - Zarządza inicjalizacją aplikacji i obsługą błędów

2. Pliki w katalogu app/

   - Stanowią rdzeń funkcjonalności aplikacji
   - Są aktywnie używane przez główny plik aplikacji
   - Zawierają kluczowe komponenty:
     - Interfejs użytkownika (GUI)
     - System logowania
     - Zarządzanie bazą danych
     - Przetwarzanie obrazów
     - Obsługa modeli AI

3. `scaller.py`
   - Używany do skalowania obrazów
   - Jest częścią procesu przetwarzania obrazów

### Potencjalnie nieużywane pliki

1. `fix_png.py`

   - Narzędzie jednorazowego użytku do naprawy plików PNG
   - Nie jest częścią głównego przepływu aplikacji
   - Można przenieść do katalogu `app/utils/tools/`

2. `run_resolution_scanner.py`

   - Narzędzie diagnostyczne
   - Używane tylko podczas debugowania
   - Można przenieść do katalogu `app/utils/diagnostics/`

3. `przygotuj_offline.py`
   - Narzędzie pomocnicze do przygotowania trybu offline
   - Nie jest częścią głównego przepływu aplikacji
   - Można rozważyć integrację z główną aplikacją w module `app/core/offline/`

## 4. Analiza logów

### Aktywne komponenty (na podstawie logów)

1. Moduły GUI:

   - `app.gui.dialogs.training_task_config_dialog`
   - `app.gui.main_window`
   - `app.gui.widgets`

2. Moduły core:
   - `app.core.logger`
   - `app.core.workers`
   - `app.core.optimizations`

### Nieaktywne komponenty

1. Narzędzia pomocnicze:
   - `fix_png.py`
   - `run_resolution_scanner.py`
   - `przygotuj_offline.py`

## 5. Rekomendacje

### Pliki do zachowania

- Wszystkie pliki w katalogu `app/`
- `cfabNS.py`
- `scaller.py`

### Pliki do weryfikacji przed usunięciem

1. `fix_png.py`

   - Sprawdzić historię użycia
   - Rozważyć przeniesienie do katalogu `app/utils/tools/`

2. `run_resolution_scanner.py`

   - Rozważyć przeniesienie do katalogu `app/utils/diagnostics/`
   - Zachować jeśli używane podczas debugowania

3. `przygotuj_offline.py`
   - Sprawdzić częstotliwość użycia
   - Rozważyć integrację z główną aplikacją w module `app/core/offline/`

## 6. Następne kroki

1. Utworzenie katalogów dla narzędzi pomocniczych:

   - `app/utils/tools/` - dla narzędzi jednorazowego użytku
   - `app/utils/diagnostics/` - dla narzędzi diagnostycznych

2. Przeniesienie plików do odpowiednich katalogów:

   - `fix_png.py` -> `app/utils/tools/`
   - `run_resolution_scanner.py` -> `app/utils/diagnostics/`
   - `przygotuj_offline.py` -> `app/core/offline/`

3. Aktualizacja dokumentacji:

   - Dodanie informacji o nowej strukturze katalogów
   - Aktualizacja instrukcji instalacji i uruchamiania

4. Testy po reorganizacji:
   - Weryfikacja działania głównej aplikacji
   - Testy narzędzi pomocniczych w nowych lokalizacjach
   - Sprawdzenie importów i zależności
