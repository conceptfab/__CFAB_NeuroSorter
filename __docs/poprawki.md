Rozumiem, że zaproponowane wcześniej zmiany nie poprawiły problemu z doszkalaniem modelu. Przyjrzyjmy się głębiej kodowi funkcji doszkalania, aby znaleźć faktyczny problem.
Po dokładniejszej analizie zauważyłem, że problem może leżeć w implementacji funkcji _run_finetuning_task w klasie BatchTrainingThread. Kiedy trafia tam zadanie typu "Doszkalanie", prawdopodobnie występuje niezgodność z oczekiwaną strukturą danych lub logika funkcji nie obsługuje właściwie tego typu zadania.
Spróbujmy zidentyfikować dokładne miejsce problemu w kodzie:

W pliku app/core/workers/batch_training_thread.py funkcja _run_finetuning_task prawdopodobnie nie obsługuje prawidłowo struktury danych przekazywanej przez _configure_finetuning_task.
Istnieje również możliwość, że w funkcji run w klasie BatchTrainingThread występuje problem z identyfikacją typu zadania - może nie rozpoznawać poprawnie zadania "Doszkalanie".

Przeanalizujmy dokładniej kod błędnej funkcji w pliku batch_training_thread.py:
pythondef _run_finetuning_task(self, task_data, task_name):
    # Kluczowe problemy mogą być tutaj
    # 1. Czy funkcja właściwie odczytuje dane z task_data?
    # 2. Czy klucze używane w tej funkcji odpowiadają kluczom w przekazywanym obiekcie?
    # 3. Czy podmiana modelu z base_model na model_path jest wykonywana prawidłowo?
Proponuję wprowadzenie następujących poprawek w pliku app/core/workers/batch_training_thread.py:

Dodać kod, który obsługuje różne struktury danych wejściowych (zarówno format z zagnieżdżonym polem "config", jak i bez niego)
Wprowadzić bardziej szczegółowe logowanie, aby dokładnie wiedzieć, na którym etapie występuje problem
Zmodyfikować funkcję _run_finetuning_task, aby obsługiwała obie struktury (z zagnieżdżonym "config" i bez)

Poprawki w pliku app/core/workers/batch_training_thread.py:
pythondef _run_finetuning_task(self, task_data, task_name):
    """Wykonuje zadanie doszkalania istniejącego modelu."""
    try:
        # Dodaj szczegółowe logowanie
        self.log_message_signal.emit(f"Rozpoczynam zadanie doszkalania: {task_name}")
        self.log_message_signal.emit(f"Struktura danych zadania: {json.dumps(task_data, indent=2)}")
        
        # Obsługa różnych struktur danych wejściowych
        config = task_data.get("config", {})
        
        # Jeśli dane są w polu config, używamy ich, w przeciwnym razie używamy bezpośrednio task_data
        base_model_path = config.get("base_model", task_data.get("model_path", ""))
        training_dir = config.get("train_dir", config.get("training_dir", task_data.get("training_dir", "")))
        validation_dir = config.get("val_dir", task_data.get("val_dir", ""))
        
        self.log_message_signal.emit(f"Odczytane parametry:")
        self.log_message_signal.emit(f"- Model: {base_model_path}")
        self.log_message_signal.emit(f"- Katalog treningowy: {training_dir}")
        self.log_message_signal.emit(f"- Katalog walidacyjny: {validation_dir}")
        
        # Walidacja ścieżek
        if not base_model_path or not os.path.exists(base_model_path):
            error_msg = f"Plik modelu nie istnieje: {base_model_path}"
            self.log_message_signal.emit(f"BŁĄD: {error_msg}")
            raise ValueError(error_msg)
            
        if not training_dir or not os.path.exists(training_dir):
            error_msg = f"Katalog treningowy nie istnieje: {training_dir}"
            self.log_message_signal.emit(f"BŁĄD: {error_msg}")
            raise ValueError(error_msg)
            
        # Dalsze sprawdzenia...
        
        # Pozostała część funkcji...
    except Exception as e:
        self.log_message_signal.emit(f"BŁĄD w _run_finetuning_task: {str(e)}")
        self.log_message_signal.emit(f"TRACEBACK: {traceback.format_exc()}")
        raise
Dodatkowo, w funkcji run w tej samej klasie, upewnijmy się, że rozpoznawanie typu zadania działa poprawnie:
pythondef run(self):
    """Wykonuje wsadowe zadania treningowe."""
    self.log_message_signal.emit("BatchTrainingThread.run: Rozpoczęto działanie wątku.")

    try:
        for task_path in self.task_paths_or_data:
            # Sprawdź zatrzymanie
            if self._stopped:
                self.log_message_signal.emit("BatchTrainingThread.run: Wątek zatrzymany przedwcześnie.")
                break

            # Wczytaj dane zadania z pliku JSON
            try:
                with open(task_path, "r", encoding="utf-8") as f:
                    task_data = json.load(f)
                self.log_message_signal.emit(f"Wczytano dane zadania z: {task_path}")
                self.log_message_signal.emit(f"Typ zadania wg danych: {task_data.get('type', 'Nieznany')}")
            except Exception as e:
                self.log_message_signal.emit(f"BatchTrainingThread.run: BŁĄD wczytywania pliku {task_path}: {str(e)}")
                self.error.emit("Główny wątek", f"Błąd wczytywania pliku: {str(e)}")
                continue

            task_name = task_data.get("name", "Bez nazwy")
            task_type = task_data.get("type", "Trening")
            
            # Ujednolicenie formatu typu zadania
            if task_type.lower() in ["doszkalanie", "finetuning"]:
                task_type = "Doszkalanie"
            elif task_type.lower() in ["trening", "training"]:
                task_type = "Trening"

            self.log_message_signal.emit(f"Rozpoznany typ zadania: {task_type}")
            
            # Powiadom o rozpoczęciu zadania
            self.task_started.emit(task_name, task_type)
            
            # Wykonaj zadanie w zależności od typu
            result = None
            if task_type == "Trening":
                result = self._run_training_task(task_data, task_name)
            elif task_type == "Doszkalanie":
                result = self._run_finetuning_task(task_data, task_name)
            else:
                self.error.emit(task_name, f"Nieznany typ zadania: {task_type}")
                continue
                
            # Reszta kodu...
Te poprawki powinny zwiększyć odporność kodu na różne struktury danych wejściowych i pomóc zidentyfikować faktyczne źródło problemu.
Jeśli powyższe zmiany nie rozwiążą problemu, warto również sprawdzić:

Czy plik z zadaniem doszkalania jest prawidłowo zapisywany (czy format JSON jest poprawny)
Czy zadanie jest prawidłowo wczytywane przez BatchTrainingThread
Czy parametry przekazywane do funkcji train_model_optimized są zgodne z jej oczekiwaniami