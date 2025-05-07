# Analiza narzędzi pomocniczych i propozycje integracji

## Dostępne narzędzia

### 1. Skaner rozdzielczości (ResolutionScanner)

- **Funkcjonalność**: Skanuje katalog w poszukiwaniu obrazów i analizuje ich rozdzielczości
- **Możliwości**:
  - Wykrywanie obrazów o zbyt małej/dużej rozdzielczości
  - Wizualizacja rozkładu rozdzielczości na wykresie
  - Automatyczne przeskalowywanie zbyt dużych obrazów
- **Status**: Gotowe do integracji

### 2. Narzędzia do obsługi plików (FileTools)

- **Konwerter JPEG do JPG**
- **Kopiarka obrazów**
- **Przenośnik JPEG**
- **Status**: Wymagają integracji

### 3. Splitter danych (DataSplitter)

- **Funkcjonalność**: Dzieli dane na zbiory treningowe i walidacyjne
- **Możliwości**:
  - Podział procentowy lub na podstawie liczby plików
  - Zachowanie struktury katalogów
  - Generowanie raportów
- **Status**: Wymaga integracji

### 4. Walidator danych (MainValidator)

- **Funkcjonalność**: Sprawdza poprawność struktury danych treningowych
- **Możliwości**:
  - Weryfikacja zgodności kategorii
  - Sprawdzanie pustych folderów
  - Analiza liczby plików
- **Status**: Wymaga integracji

## Propozycje integracji

### 1. Menu "Narzędzia" w głównym interfejsie

```
Narzędzia
├── Skaner rozdzielczości
├── Zarządzanie danymi
│   ├── Splitter danych
│   ├── Walidator danych
│   └── Konwerter formatów
└── Ustawienia
```

### 2. Integracja skanera rozdzielczości

- Dodać jako zakładkę w głównym interfejsie
- Zachować obecną funkcjonalność
- Dodać możliwość automatycznego przeskalowywania podczas importu danych

### 3. Integracja narzędzi do obsługi plików

- Połączyć w jeden moduł "Zarządzanie danymi"
- Dodać wspólny interfejs z zakładkami
- Zachować możliwość uruchamiania jako osobne aplikacje

### 4. Integracja splittera i walidatora

- Dodać jako część procesu przygotowania danych
- Zintegrować z systemem raportowania
- Dodać możliwość automatycznej walidacji po podziale danych

## Priorytety integracji

1. Skaner rozdzielczości (najprostszy do integracji)
2. Walidator danych (krytyczny dla jakości danych)
3. Splitter danych (ważny dla procesu treningu)
4. Narzędzia do obsługi plików (pomocnicze)

## Uwagi

- Zachować możliwość uruchamiania narzędzi jako osobnych aplikacji
- Dodać system logowania operacji
- Zintegrować z systemem raportowania
- Dodać możliwość konfiguracji domyślnych ustawień
