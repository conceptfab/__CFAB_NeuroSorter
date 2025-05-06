# Schemat procesu po naciśnięciu "Uruchom kolejkę"

Oto schemat procesów i wywołań funkcji po naciśnięciu przycisku "Uruchom kolejkę" w klasie `TrainingManager` w pliku `app/gui/tabs/training_manager.py`:

1.  **Kliknięcie przycisku `start_queue_btn`:**

    - Sygnał `clicked` tego przycisku jest połączony z metodą `_start_task_queue`.

2.  **Wywołanie metody `_start_task_queue(self)`:**

    - **Sprawdzenie:** Sprawdza, czy inny wątek treningowy (`self.training_thread`) już działa (`isRunning()`). Jeśli tak, wyświetla ostrzeżenie (`QMessageBox.warning`) i kończy działanie.
    - **Pobranie zadań:** Określa ścieżkę do katalogu zadań (`data/tasks`).
    - Używa `glob.glob` do znalezienia wszystkich plików `*.json` w tym katalogu, sortuje je i zapisuje do listy `task_files`.
    - **Sprawdzenie:** Sprawdza, czy lista `task_files` jest pusta. Jeśli tak, wyświetla informację (`QMessageBox.information`) i kończy działanie.
    - Loguje informację o liczbie znalezionych zadań (`self.parent.logger.info`).
    - **Utworzenie wątku:** Tworzy instancję klasy `BatchTrainingThread`, przekazując jej listę plików zadań (`task_files`).
    - **Podłączenie sygnałów:** Łączy sygnały emitowane przez obiekt `BatchTrainingThread` z odpowiednimi metodami (slotami) w klasie `TrainingManager`:
      - `task_started` -> `_training_task_started`
      - `task_progress` -> `_training_task_progress`
      - `task_completed` -> `_training_task_completed`
      - `all_tasks_completed` -> `_all_training_tasks_completed`
      - `error` -> `_training_task_error`
      - `log_message_signal` -> `self.parent.logger.info` (bezpośrednie logowanie komunikatów z wątku)
    - **Uruchomienie wątku:** Wywołuje metodę `start()` na obiekcie `BatchTrainingThread`, co rozpoczyna wykonywanie jego metody `run()` w osobnym wątku.
    - **Aktualizacja UI:** Ustawia tekst w `self.parent.current_task_info` na "Rozpoczynanie przetwarzania kolejki...".
    - Loguje informację o uruchomieniu przetwarzania kolejki (`self.parent.logger.info`).
    - _Obsługa błędów:_ W bloku `try...except` łapie ewentualne wyjątki podczas tworzenia lub uruchamiania wątku i wyświetla komunikat błędu (`QMessageBox.critical`).

3.  **Działanie wątku `BatchTrainingThread` (w tle):**

    - Wątek iteruje po liście plików zadań (`task_files`).
    - Dla każdego pliku:
      - Wczytuje konfigurację zadania z pliku JSON.
      - Emituje sygnał `task_started`, przekazując nazwę i typ zadania.
      - Rozpoczyna właściwy proces treningu/doszkalania (prawdopodobnie wywołując funkcje z innych modułów, np. `app.core.trainer`).
      - W trakcie treningu (np. po każdej epoce) emituje sygnał `task_progress`, przekazując nazwę zadania, procent postępu i szczegóły (epoka, strata, dokładność).
      - Loguje postęp i inne komunikaty, emitując `log_message_signal`.
      - W przypadku błędu emituje sygnał `error`, przekazując nazwę zadania i treść błędu.
      - Po pomyślnym zakończeniu zadania emituje sygnał `task_completed`, przekazując nazwę zadania i wyniki (np. ścieżkę do zapisanego modelu, końcową dokładność).
    - Po przetworzeniu wszystkich zadań emituje sygnał `all_tasks_completed`.

4.  **Obsługa sygnałów przez `TrainingManager` (reakcje na zdarzenia z wątku):**
    - **`_training_task_started(self, task_name, task_type)`:**
      - Loguje rozpoczęcie zadania.
      - Aktualizuje `self.parent.current_task_info`, wyświetlając nazwę i typ bieżącego zadania.
      - Resetuje pasek postępu (`self.parent.task_progress_bar`).
      - Aktywuje przycisk zatrzymania zadania (`self.parent.stop_task_btn`).
    - **`_training_task_progress(self, task_name, progress, details)`:**
      - Loguje szczegóły postępu (epoka, strata, dokładność).
      - Aktualizuje etykietę ze szczegółami postępu (`self.parent.task_progress_details`).
      - Aktualizuje wartość paska postępu (`self.parent.task_progress_bar`).
    - **`_training_task_completed(self, task_name, result)`:**
      - Wywołuje `self.parent.model_manager_tab.refresh()` w celu odświeżenia listy modeli.
      - Loguje informacje o zakończonym zadaniu i jego wynikach.
      - Resetuje UI związane z postępem zadania (`self.parent.current_task_info`, `self.parent.task_progress_bar`, `self.parent.task_progress_details`).
      - Dezaktywuje przycisk zatrzymania zadania (`self.parent.stop_task_btn`).
      - Wywołuje `_set_task_status(task_name, "Zakończony")`, aby zaktualizować status zadania w pliku JSON.
    - **`_all_training_tasks_completed(self)`:**
      - Loguje zakończenie wszystkich zadań.
      - Wyświetla informację dla użytkownika (`QMessageBox.information`).
      - Resetuje UI związane z postępem zadania.
    - **`_training_task_error(self, task_name, error_message)`:**
      - Loguje błąd.
      - Wyświetla krytyczny komunikat błędu (`QMessageBox.critical`).
      - Odświeża listę zadań w tabeli (`self.refresh()`).
    - **`self.parent.logger.info(message)`:** Loguje komunikaty wysłane przez sygnał `log_message_signal` z wątku.
