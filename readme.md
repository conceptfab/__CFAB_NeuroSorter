# CFAB NeuroSorter

![CFAB NeuroSorter](resources/img/splash_doc.jpg)

## Nowości w wersji 0.4

- **Ulepszony Interfejs Użytkownika:** Wprowadzono liczne poprawki w interfejsie graficznym, mające na celu zwiększenie intuicyjności i komfortu pracy. Nazewnictwo oraz opisy funkcji w aplikacji zostały ujednolicone.
- **Rozbudowana Dokumentacja:** Dokumentacja użytkownika (dostępna w zakładce Pomoc) została znacząco rozszerzona i zaktualizowana, aby lepiej odzwierciedlać obecne możliwości aplikacji. Usunięto techniczne odwołania do nazw plików, skupiając się na opisach funkcjonalności z perspektywy użytkownika.
- **Zarządzanie Konfiguracją Modeli:** Poprawiono mechanizmy związane z konfiguracją i zarządzaniem modelami AI, w tym ich importem, eksportem oraz wyborem do zadań treningowych i klasyfikacyjnych.
- **Optymalizacja Procesu Trenowania:** Wprowadzono usprawnienia w module treningowym, w tym bardziej szczegółowe opcje konfiguracji zadań treningu od podstaw i fine-tuningu. Zaktualizowano zalecane parametry startowe dla różnych scenariuszy.
- **Usprawniona Klasyfikacja wsadowa:** Zoptymalizowano działanie funkcji klasyfikacji wsadowej, oferując bardziej granularne opcje konfiguracji sortowania i obsługi plików.

## Opis projektu

CFAB NeuroSorter to zaawansowana aplikacja desktopowa z interfejsem graficznym, stworzona z myślą o automatyzacji procesów sortowania i klasyfikacji danych, w szczególności obrazów neuronowych. Wykorzystuje modele sztucznej inteligencji do analizy i kategoryzacji, oferując jednocześnie rozbudowane narzędzia do zarządzania zadaniami, modelami AI, przetwarzania danych oraz obsługi plików. Aplikacja umożliwia efektywne trenowanie, dostrajanie (fine-tuning) oraz testowanie modeli w różnych wariantach konfiguracyjnych.

## Jak to działa

1.  Użytkownik wskazuje folder zawierający obrazy do analizy.
2.  Aplikacja analizuje każdy obraz przy użyciu wybranego modelu AI.
3.  Dla każdego obrazu:
    - Określa najbardziej prawdopodobną kategorię na podstawie jego zawartości.
    - Opcjonalnie dodaje metadane do pliku (funkcjonalność może być rozwijana).
    - Zapisuje informacje o klasyfikacji w wewnętrznej bazie danych (dla celów raportowania i historii).
4.  Umożliwia automatyczne sortowanie plików (kopiowanie lub przenoszenie) do folderów odpowiadających przypisanym kategoriom.

## Główne Funkcjonalności

- **Automatyczna klasyfikacja obrazów:** Analiza pojedynczych obrazów oraz całych folderów przy użyciu modeli AI.
- **Zarządzanie modelami AI:** Importowanie, eksportowanie, usuwanie oraz wybór aktywnych modeli. Dostęp do szczegółowych informacji o modelach.
- **Trenowanie i fine-tuning modeli:** Możliwość trenowania modeli od podstaw oraz dostrajania istniejących modeli na własnych zbiorach danych. Zaawansowana konfiguracja parametrów treningowych (architektura, epoki, współczynnik uczenia, optymalizatory, harmonogramy uczenia, augmentacja danych, itp.).
- **Wizualizacja treningu:** Śledzenie postępów uczenia modeli poprzez dynamiczne wykresy metryk (np. strata, dokładność).
- **Klasyfikacja wsadowa:** Efektywna klasyfikacja i sortowanie dużych kolekcji obrazów z opcjami kopiowania lub przenoszenia plików, obsługą nieklasyfikowanych obrazów i tworzeniem struktury folderów dla klas.
- **Narzędzia do przygotowania danych:** Dedykowane narzędzie ("Przygotowanie danych AI" dostępne z menu "Narzędzia") do importu, organizacji i podziału zbiorów danych na potrzeby treningu (zbiory treningowe, walidacyjne, testowe) z wyborem trybu podziału (procentowy lub limit plików).
- **Generowanie raportów:** Możliwość tworzenia podsumowań i raportów z przeprowadzonych operacji klasyfikacji i treningu (funkcjonalność w rozwoju, obecnie nieaktywna w UI).
- **Zarządzanie ustawieniami aplikacji:** Konfiguracja globalnych parametrów działania NeuroSortera (dostępna poprzez menu "Ustawienia" -> "Ustawienia globalne").
- **Pomoc i dokumentacja:** Zintegrowana pomoc użytkownika wyjaśniająca działanie poszczególnych modułów.
- **Interfejs użytkownika oparty na głównych zakładkach:**
  - **Modele:** Zarządzanie modelami AI.
  - **Trenowanie:** Konfiguracja i monitorowanie zadań treningowych i fine-tuningu.
  - **Klasyfikacja:** Analiza pojedynczych obrazów.
  - **Klasyfikacja wsadowa:** Automatyzacja klasyfikacji i sortowania dla dużych zbiorów.
  - **Raporty:** Generowanie i przeglądanie raportów (funkcjonalność w rozwoju, obecnie nieaktywna w UI).
  - **Pomoc:** Dostęp do dokumentacji.
- **Obsługa różnych architektur modeli:** Wsparcie dla popularnych architektur sieci neuronowych (np. ResNet, EfficientNet, MobileNet, Vision Transformer, ConvNeXt), wstępnie trenowanych na ImageNet.
- **Optymalizacja pod GPU:** Automatyczne wykrywanie i wykorzystanie dostępności CUDA do przyspieszenia obliczeń, w tym wsparcie dla mieszanej precyzji.
- **Logowanie zdarzeń:** Szczegółowe zapisywanie przebiegu operacji i ewentualnych błędów.

## Architektury Modeli AI

Aplikacja oferuje wybór spośród kilku zaawansowanych architektur sieci neuronowych, wstępnie trenowanych na zbiorze danych ImageNet, które mogą być następnie trenowane od podstaw lub dostrajane:

- **Model `50` (np. ResNet-50):** Zapewnia dobrą równowagę między dokładnością a zasobami obliczeniowymi. Uniwersalny wybór dla wielu zadań klasyfikacji.
- **Model `b0` (np. EfficientNet-B0):** Efektywna architektura, optymalizująca dokładność przy ograniczonych zasobach. Zalecana dla większości zadań, z możliwością wyboru większych wariantów (B1-B7) dla wyższej dokładności.
- **Model `mobile3l` (np. MobileNetV3 Large):** Lekka architektura zoptymalizowana pod kątem szybkości wnioskowania i małego rozmiaru, idealna dla środowisk z ograniczonymi zasobami.
- **Model `vitb16` (np. Vision Transformer Base):** Nowoczesna architektura oparta na mechanizmie uwagi, skuteczna w złożonych zadaniach, szczególnie przy dużych zbiorach danych.
- **Model `tiny` (np. ConvNeXt Tiny):** Nowoczesna architektura konwolucyjna inspirowana Transformerami, oferująca wysoką wydajność, często przewyższająca tradycyjne CNN przy podobnej liczbie parametrów.

Każdy model może być dalej dostosowywany poprzez proces treningu od podstaw lub fine-tuningu. Aplikacja wspiera optymalizacje takie jak automatyczny wybór precyzji (pełna/mieszana) i dynamiczne dostosowywanie rozmiaru wsadu (batch size) w zależności od dostępnych zasobów GPU.

## Wymagania systemowe

- Python 3.8 lub nowszy.
- Biblioteki wymienione w pliku `requirements.txt` (instalacja poprzez `pip install -r requirements.txt`).
- Opcjonalnie, ale zalecane: karta graficzna NVIDIA z obsługą CUDA w celu znacznego przyspieszenia procesów treningu i klasyfikacji modeli AI.

## Uruchomienie aplikacji

Aby uruchomić aplikację z interfejsem graficznym, należy wykonać skrypt główny (np. `cfabNS.py` lub analogiczny) w środowisku Python:

```bash
python <nazwa_pliku_uruchomieniowego>.py
```

## Główne Sekcje Interfejsu (Zakładki)

### Zakładka Modele

- Przeglądanie, importowanie, eksportowanie i usuwanie modeli AI.
- Wybór aktywnego modelu do klasyfikacji lub fine-tuningu.
- Podgląd szczegółowych informacji i statystyk modeli.
- Zarządzanie mapowaniem klas (jeśli dotyczy).

### Zakładka Trenowanie

- Konfiguracja i uruchamianie zadań treningu modeli od podstaw oraz fine-tuningu.
- Dostęp do zaawansowanych parametrów uczenia (architektura, epoki, batch size, learning rate, optymalizator, harmonogram uczenia, augmentacja danych, zamrażanie warstw itp.).
- Monitorowanie postępu treningu w czasie rzeczywistym (np. wykresy straty i dokładności).
- Możliwość tworzenia i zarządzania kolejką zadań treningowych.

### Zakładka Klasyfikacja

- Ładowanie i klasyfikacja pojedynczych obrazów przy użyciu aktywnego modelu.
- Wyświetlanie obrazu i wyników predykcji (przewidziana klasa, prawdopodobieństwa dla poszczególnych klas).
- Opcjonalne sortowanie sklasyfikowanego obrazu do odpowiedniego folderu.

### Zakładka Klasyfikacja wsadowa

- Automatyczna klasyfikacja i sortowanie obrazów z wybranego folderu źródłowego do folderu docelowego.
- Konfiguracja opcji (wybór modelu, akcja: kopiuj/przenieś, obsługa nieklasyfikowanych plików, tworzenie podfolderów dla klas, próg pewności).
- Monitorowanie postępu przetwarzania dużych zbiorów danych.

### Zakładka Zbiory Danych

- Narzędzia do zarządzania zbiorami danych wykorzystywanymi w procesach AI.
- Import danych, organizacja struktury folderów.
- Podział danych na zbiory treningowe, walidacyjne i testowe, z uwzględnieniem proporcji klas.
- Walidacja poprawności przygotowanych zbiorów.

### Zakładka Raporty (w trakcie implementacji)

- Generowanie i przeglądanie raportów z przeprowadzonych operacji klasyfikacji i treningu.
- Prezentacja statystyk, wykresów i podsumowań.
- Eksport danych raportowych.

### Zakładka Ustawienia

- Konfiguracja globalnych ustawień aplikacji, np. domyślne ścieżki, parametry logowania, preferencje interfejsu.

### Zakładka Pomoc

- Dostęp do zintegrowanej dokumentacji użytkownika, informacji o programie, FAQ i wskazówek dotyczących rozwiązywania problemów.

## Opcje Treningowe

- **Wybór architektury modelu:** Możliwość wyboru spośród predefiniowanych, wydajnych architektur (Model 50, B0, Mobile3L, ViTB16, Tiny).
- **Augmentacja danych:** Konfigurowalne techniki augmentacji (np. zmiany jasności, kontrastu, nasycenia, obroty, odbicia) z trybami: brak, podstawowy, zaawansowany, aby zwiększyć różnorodność danych treningowych i poprawić generalizację modelu.
- **Dostosowanie hiperparametrów:** Pełna kontrola nad kluczowymi parametrami uczenia, takimi jak współczynnik uczenia, rozmiar wsadu, liczba epok, wybór optymalizatora i harmonogramu uczenia.
- **Fine-tuning:** Możliwość zamrożenia warstw bazowych modelu (backbone) podczas dostrajania, aby zachować wyuczone cechy niskiego poziomu, szczególnie przy mniejszych zbiorach danych.
- **Wczesne zatrzymywanie:** Automatyczne zakończenie treningu, gdy model przestaje wykazywać poprawę na zbiorze walidacyjnym, zapobiegając przeuczeniu.
- **Mieszana precyzja:** Opcja wykorzystania mieszanej precyzji (np. FP16) na wspieranych GPU w celu przyspieszenia treningu i zmniejszenia zużycia pamięci VRAM.

## Opcje Klasyfikacji

- **Klasyfikacja pojedynczego obrazu:** Szybka analiza wybranego pliku.
- **Klasyfikacja wsadowa:** Przetwarzanie całych folderów obrazów.
- **Sortowanie plików:** Automatyczne kopiowanie lub przenoszenie sklasyfikowanych obrazów do folderów odpowiadających ich kategoriom.
- **Próg pewności:** Możliwość ustawienia minimalnego poziomu ufności, jaki musi osiągnąć model, aby jego predykcja została uznana za wiarygodną.
- **Obsługa metadanych:** Potencjalna integracja z systemem metadanych plików (w zależności od rozwoju funkcjonalności).

## Dodatkowe Funkcje

- **Ciemny motyw interfejsu:** Dla komfortu pracy w różnych warunkach oświetleniowych.
- **Monitorowanie zasobów systemowych:** Informacje o wykorzystaniu CPU, GPU, RAM (w zależności od implementacji profilera).
- **Panel konsoli:** Wyświetlanie logów systemowych i komunikatów o przebiegu operacji aplikacji.
- **Profiler sprzętowy:** Potencjalna funkcja analizy konfiguracji sprzętowej i automatycznej optymalizacji parametrów pod kątem wydajności.
- **Zapisywanie ustawień:** Utrzymywanie konfiguracji aplikacji między sesjami.

## Zarządzanie Bazą Danych i Metadanymi

Aplikacja może wykorzystywać wewnętrzną bazę danych (np. SQLite) do przechowywania historii przeprowadzonych operacji klasyfikacji, informacji o modelach, zadaniach treningowych oraz potencjalnie metadanych powiązanych z przetwarzanymi obrazami. Umożliwia to generowanie raportów, śledzenie postępów i analizę wyników w czasie.

## Zależności Główne

- **PyTorch (`torch`, `torchvision`):** Podstawowy framework do uczenia maszynowego i przetwarzania obrazów.
- **Pillow:** Biblioteka do zaawansowanej manipulacji plikami graficznymi.
- **NumPy:** Niezbędna do operacji numerycznych.
- **PyQt6 (lub alternatywa):** Framework do budowy interfejsu graficznego.
- **Piexif (lub podobne):** Do obsługi metadanych EXIF w obrazach (jeśli zaimplementowano).
- **Psutil:** Do monitorowania zasobów systemowych (jeśli zaimplementowano).

Pełna lista zależności znajduje się w pliku `requirements.txt`.

## Licencja

MIT License

Copyright (c) 2023-2024 CFAB NeuroSorter Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Aktualny stan projektu

- Wersja: 0.4 (zgodnie z niniejszym dokumentem)
- Status: W trakcie aktywnego rozwoju
- Ostatnia aktualizacja: Zgodna z datą tego dokumentu
