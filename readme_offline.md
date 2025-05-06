# Dokumentacja procesu instalacji offline dla projektu CFAB NeuroSorter

_Ostatnia aktualizacja: 2024_

---

## Spis treści

1. [Wprowadzenie](#wprowadzenie)
2. [Wymagania wstępne](#wymagania-wstępne)
3. [Proces instalacji w dwóch krokach](#proces-instalacji-w-dwóch-krokach)
   - [Krok 1: Przygotowanie pakietów](#krok-1-przygotowanie-pakietów-na-komputerze-z-dostępem-do-internetu)
   - [Krok 2: Instalacja na komputerze bez dostępu do internetu](#krok-2-instalacja-na-komputerze-bez-dostępu-do-internetu)
4. [Skrypt automatyzujący proces](#skrypt-automatyzujący-proces)
   - [Opis skryptu](#opis-skryptu)
   - [Opcje skryptu](#opcje-skryptu)
5. [Instrukcja krok po kroku](#instrukcja-krok-po-kroku)
   - [Przygotowanie pakietów (tryb online)](#przygotowanie-pakietów-tryb-online)
   - [Instalacja pakietów (tryb offline)](#instalacja-pakietów-tryb-offline)
6. [Rozwiązywanie problemów](#rozwiązywanie-problemów)
7. [Dodatkowe informacje](#dodatkowe-informacje)

---

## 1. Wprowadzenie

CFAB NeuroSorter to aplikacja wykorzystująca sieci neuronowe do klasyfikacji i sortowania obrazów. Aby umożliwić korzystanie z aplikacji na komputerach bez dostępu do internetu, przygotowano proces instalacji offline, który opisuje poniższa dokumentacja.

---

## 2. Wymagania wstępne

- Python 3.8 lub nowszy
- Dostęp do komputera z internetem (do pobrania pakietów)
- Możliwość przeniesienia plików między komputerami (np. pendrive, dysk zewnętrzny)
- Uprawnienia administratora na obu komputerach (zalecane)

---

## 3. Proces instalacji w dwóch krokach

Proces instalacji offline składa się z dwóch głównych kroków:

1. Pobranie wszystkich wymaganych pakietów na komputerze z dostępem do internetu
2. Instalacja pakietów na komputerze docelowym bez dostępu do internetu

### 3.1 Krok 1: Przygotowanie pakietów na komputerze z dostępem do internetu

Na komputerze, który ma dostęp do internetu, wykonujemy następujące kroki:

1. Pobieramy kod źródłowy projektu CFAB NeuroSorter
2. Tworzymy katalog na pakiety (np. wheelhouse)
3. Pobieramy wszystkie wymagane pakiety z pliku requirements.txt
4. Kopiujemy cały katalog projektu wraz z pobranymi pakietami na nośnik przenośny

### 3.2 Krok 2: Instalacja na komputerze bez dostępu do internetu

Na komputerze docelowym (bez dostępu do internetu):

1. Kopiujemy cały katalog projektu z nośnika przenośnego
2. Tworzymy wirtualne środowisko Python
3. Instalujemy wszystkie pakiety z lokalnego katalogu
4. Uruchamiamy aplikację

---

## 4. Skrypt automatyzujący proces

Aby uprościć proces instalacji, przygotowano skrypt `przygotuj_offline.py`, który automatyzuje wszystkie opisane powyżej kroki.

### 4.1 Opis skryptu

Skrypt `przygotuj_offline.py` obsługuje dwa tryby działania:

- Tryb online - pobiera pakiety na komputerze z internetem
- Tryb offline - instaluje pakiety na komputerze bez internetu

### 4.2 Opcje skryptu

```bash
python przygotuj_offline.py [opcje]
```

Opcje:

- `--offline` Tryb instalacji offline (na komputerze bez internetu)
- `--env NAZWA` Nazwa wirtualnego środowiska (domyślnie: venv)
- `--requirements PLIK` Ścieżka do pliku requirements.txt (domyślnie: requirements.txt)
- `--packages-dir DIR` Ścieżka do katalogu z pakietami (domyślnie: wheelhouse)

---

## 5. Instrukcja krok po kroku

### 5.1 Przygotowanie pakietów (tryb online)

1. Umieść skrypt `przygotuj_offline.py` w katalogu głównym projektu (tam gdzie znajduje się plik requirements.txt)
2. Otwórz terminal/konsolę i przejdź do katalogu projektu
3. Uruchom skrypt w trybie domyślnym (online):

```bash
python przygotuj_offline.py
```

4. Skrypt pobierze wszystkie wymagane pakiety do katalogu wheelhouse
5. Po zakończeniu, skopiuj cały katalog projektu (wraz z katalogiem wheelhouse) na nośnik przenośny

### 5.2 Instalacja pakietów (tryb offline)

1. Przenieś cały katalog projektu z nośnika przenośnego na komputer docelowy
2. Otwórz terminal/konsolę i przejdź do katalogu projektu
3. Uruchom skrypt w trybie offline:

```bash
python przygotuj_offline.py --offline
```

Skrypt:

- Utworzy wirtualne środowisko Python
- Zainstaluje wszystkie pakiety z katalogu wheelhouse
- Utworzy pliki ułatwiające uruchomienie aplikacji (uruchom.bat lub uruchom.sh)

Uruchom aplikację:

- Windows: kliknij dwukrotnie na plik `uruchom.bat`
- Linux/Mac: wykonaj w terminalu `./uruchom.sh`

---

## 6. Rozwiązywanie problemów

### Problem: Brak modułu venv

- **Objaw**: Komunikat błędu o braku modułu venv
- **Rozwiązanie**: Zainstaluj moduł venv:

  ```bash
  # Windows
  python -m pip install --user virtualenv

  # Debian/Ubuntu
  sudo apt-get install python3-venv

  # Fedora
  sudo dnf install python3-venv
  ```

### Problem: Brak uprawnień do tworzenia katalogów

- **Objaw**: Komunikat błędu o braku uprawnienia do zapisu
- **Rozwiązanie**: Uruchom skrypt z uprawnieniami administratora lub zmień uprawnienia do katalogu

### Problem: Brakujące pakiety w trybie offline

- **Objaw**: Błędy instalacji pakietów w trybie offline
- **Rozwiązanie**: Upewnij się, że wymagane pakiety zostały poprawnie pobrane w trybie online i katalog wheelhouse został przeniesiony na komputer docelowy

---

## 7. Dodatkowe informacje

### Automatyczne uruchamianie aplikacji

Skrypt tworzy pliki ułatwiające uruchomienie aplikacji:

- Windows: `uruchom.bat`
- Linux/Mac: `uruchom.sh`

Te pliki aktywują wirtualne środowisko Python i uruchamiają aplikację CFAB NeuroSorter.

### Dostosowywanie procesu instalacji

Możesz dostosować proces instalacji za pomocą opcji skryptu:

```bash
# Zmiana nazwy wirtualnego środowiska na "cfab_env"
python przygotuj_offline.py --env cfab_env

# Zmiana nazwy katalogu pakietów na "packages"
python przygotuj_offline.py --packages-dir packages

# Instalacja w trybie offline z niestandardowymi opcjami
python przygotuj_offline.py --offline --env cfab_env --packages-dir packages
```

### Ręczna instalacja (bez skryptu)

Jeśli z jakiegoś powodu skrypt `przygotuj_offline.py` nie działa, możesz wykonać proces ręcznie:

Na komputerze z internetem:

```bash
mkdir wheelhouse
python -m pip download -r requirements.txt -d wheelhouse
```

Na komputerze bez internetu:

```bash
python -m venv venv

# Aktywacja środowiska (Windows)
venv\Scripts\activate

# Aktywacja środowiska (Linux/Mac)
source venv/bin/activate

# Instalacja pakietów
pip install --no-index --find-links=wheelhouse -r requirements.txt
```

---

_Dokument przygotowany dla projektu CFAB NeuroSorter przez Zespół Wsparcia Technicznego._
