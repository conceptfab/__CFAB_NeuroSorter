Analiza i propozycje poprawek do kodu trenowania modeli
Po przeanalizowaniu kodu związanego z trenowaniem modeli, zidentyfikowałam kilka obszarów do poprawy. Poniżej przedstawiam proponowane zmiany uporządkowane według plików i funkcji.
1. Plik ai/optimized_training.py
Problem z obsługą błędów w pętli po batchach
W funkcji train_model_optimized występuje problem z obsługą błędów w pętli po batchach. Obecnie, jeśli wystąpi błąd, wyświetlany jest komunikat, ale pętla nie jest przerywana, co może prowadzić do nieprzewidywalnych zachowań.
python# Zmiana w pliku ai/optimized_training.py, w funkcji train_model_optimized
# Obecny kod w pętli po batchach:
except Exception as e_batch:
    print(
        f"!!!!!!!!!! DEBUG optimized_training: BŁĄD KRYTYCZNY W PĘTLI PO BATCHACH (epoka {epoch + 1}) !!!!!!!!!!"
    )
    print(f"Błąd: {str(e_batch)}")
    print(traceback.format_exc())
    # Można tutaj dodać break lub return, jeśli błąd jest tak poważny, że dalszy trening nie ma sensu
Proponowana zmiana:
pythonexcept Exception as e_batch:
    print(
        f"!!!!!!!!!! DEBUG optimized_training: BŁĄD KRYTYCZNY W PĘTLI PO BATCHACH (epoka {epoch + 1}) !!!!!!!!!!"
    )
    print(f"Błąd: {str(e_batch)}")
    print(traceback.format_exc())
    # Przerywamy pętlę, aby uniknąć dalszych błędów
    break
Problem z konfiguracją scheduler
W funkcji configure_scheduler używana jest zmienna train_loader, która nie jest dostępna w bieżącym zakresie podczas konfiguracji OneCycleLR.
python# Zmiana w pliku ai/optimized_training.py, w funkcji configure_scheduler
# Obecny kod:
elif scheduler_type == "onecycle":
    # OneCycleLR - często najlepszy wybór dla krótszych treningów
    return optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=optimizer.param_groups[0]["lr"] * 10,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=10000.0,
        verbose=True,
    )
Proponowana zmiana:
pythonelif scheduler_type == "onecycle":
    # OneCycleLR - często najlepszy wybór dla krótszych treningów
    # Potrzebujemy dodatkowego parametru steps_per_epoch
    if train_loader is not None:
        steps_per_epoch = len(train_loader)
    else:
        # Wartość domyślna, jeśli train_loader jest niedostępny
        steps_per_epoch = 100
        
    return optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=optimizer.param_groups[0]["lr"] * 10,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=10000.0,
        verbose=True,
    )
Alternatywnie, można zmodyfikować całą funkcję, aby przyjmowała steps_per_epoch jako parametr:
pythondef configure_scheduler(optimizer, scheduler_type, epochs, patience=3, steps_per_epoch=100):
    """Konfiguruje scheduler learning rate odpowiedni do typu optymalizatora i długości treningu."""
    # ... reszta kodu pozostaje bez zmian
    
    elif scheduler_type == "onecycle":
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]["lr"] * 10,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=10000.0,
            verbose=True,
        )
2. Plik ai/models.py
Problem z przestarzałymi parametrami w funkcjach tworzących modele
W funkcji get_model używane są przestarzałe parametry do tworzenia modeli, co może powodować ostrzeżenia i potencjalnie problemy w przyszłości.
python# Zmiana w pliku ai/models.py, w funkcji get_model
# Obecny kod:
model_factories = {
    "efficientnet_b0": lambda: models.efficientnet_b0(pretrained=True),
    "efficientnet_b1": lambda: models.efficientnet_b1(pretrained=True),
    # ... i tak dalej
}
Proponowana zmiana (dla wszystkich modeli):
pythonmodel_factories = {
    "efficientnet_b0": lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "efficientnet_b1": lambda: models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1),
    "efficientnet_b2": lambda: models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1),
    "efficientnet_b3": lambda: models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1),
    "efficientnet_b4": lambda: models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1),
    "efficientnet_b5": lambda: models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.IMAGENET1K_V1),
    "efficientnet_b6": lambda: models.efficientnet_b6(weights=models.EfficientNet_B6_Weights.IMAGENET1K_V1),
    "efficientnet_b7": lambda: models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1),
    "resnet18": lambda: models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": lambda: models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1),
    "resnet101": lambda: models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1),
    "resnet152": lambda: models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1),
    "mobilenet_v2": lambda: models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1),
    "mobilenet_v3_large": lambda: models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1),
    "mobilenet_v3_small": lambda: models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1),
    "convnext_tiny": lambda: models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
    "convnext_small": lambda: models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1),
    "convnext_base": lambda: models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1),
    "convnext_large": lambda: models.convnext_large(weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1),
    "vit_b_16": lambda: models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1),
    "vit_b_32": lambda: models.vit_b_32(weights=models.ViT_B_32_Weights.IMAGENET1K_V1),
    "vit_l_16": lambda: models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1),
    "vit_l_32": lambda: models.vit_l_32(weights=models.ViT_L_32_Weights.IMAGENET1K_V1),
}
3. Plik app/core/workers/batch_training_thread.py
Problem z walidacją modelu
W funkcji _run_training_task jest problem z tworzeniem modelu - nie ma zdefiniowanego zachowania w przypadku, gdy nie można utworzyć ImageClassifier.
python# Zmiana w pliku app/core/workers/batch_training_thread.py, w funkcji _run_training_task
# Obecny kod:
# Utwórz model
self.logger.info(
    f"\nTworzenie modelu {model_type} dla {num_classes} klas..."
)
try:
    model = ImageClassifier(model_type=model_type, num_classes=num_classes)
    self.logger.info(f"Model utworzony pomyślnie: {model}")
    model_info = (
        model.get_model_info() if hasattr(model, "get_model_info") else {}
    )
    self.logger.info(
        f"DEBUG: Informacje o modelu: {json.dumps(model_info, indent=2)}"
    )
except Exception as e:
    self.logger.error(f"BŁĄD podczas tworzenia modelu: {e}")
    self.logger.error(f"TRACEBACK: {traceback.format_exc()}")
    raise
Proponowana zmiana:
python# Utwórz model
self.logger.info(f"\nTworzenie modelu {model_type} dla {num_classes} klas...")
try:
    model = ImageClassifier(model_type=model_type, num_classes=num_classes)
    self.logger.info(f"Model utworzony pomyślnie: {model}")
    model_info = model.get_model_info() if hasattr(model, "get_model_info") else {}
    self.logger.info(f"DEBUG: Informacje o modelu: {json.dumps(model_info, indent=2)}")
except Exception as e:
    self.logger.error(f"BŁĄD podczas tworzenia modelu: {e}")
    self.logger.error(f"TRACEBACK: {traceback.format_exc()}")
    
    # Dodajemy szczegółowe informacje o błędzie i wskazówki dla użytkownika
    error_details = str(e).lower()
    if "cuda" in error_details or "gpu" in error_details:
        additional_info = "Wykryto problem z GPU/CUDA. Spróbuj uruchomić trening z wyłączoną obsługą GPU."
        self.logger.error(additional_info)
        raise ValueError(f"{str(e)}. {additional_info}")
    elif "memory" in error_details or "pamięć" in error_details:
        additional_info = "Wykryto problem z pamięcią. Spróbuj zmniejszyć rozmiar wsadu (batch_size)."
        self.logger.error(additional_info)
        raise ValueError(f"{str(e)}. {additional_info}")
    else:
        raise
Problem z synchronizacją rezultatu treningu
W funkcji _run_training_task wyniki treningu są zapisywane do pliku zadania, ale nie ma sprawdzania, czy rezultat zawiera właściwe dane.
python# Zmiana w pliku app/core/workers/batch_training_thread.py, w funkcji _save_training_time
# Obecny kod:
# Dodaj dokładność walidacyjną jeśli jest dostępna
if result and "history" in result:
    history = result["history"]
    if history.get("train_acc"):
        task_data["train_accuracy"] = history["train_acc"][-1]
    if history.get("train_loss"):
        task_data["train_loss"] = history["train_loss"][-1]
    if history.get("val_acc"):
        task_data["validation_accuracy"] = history["val_acc"][-1]
    if history.get("val_loss"):
        task_data["validation_loss"] = history["val_loss"][-1]
Proponowana zmiana:
python# Dodaj dokładność walidacyjną jeśli jest dostępna
if result and "history" in result:
    history = result["history"]
    
    # Sprawdź, czy historia zawiera potrzebne dane
    has_train_acc = history.get("train_acc") and len(history["train_acc"]) > 0
    has_train_loss = history.get("train_loss") and len(history["train_loss"]) > 0
    has_val_acc = history.get("val_acc") and len(history["val_acc"]) > 0
    has_val_loss = history.get("val_loss") and len(history["val_loss"]) > 0
    
    if has_train_acc:
        task_data["train_accuracy"] = history["train_acc"][-1]
    else:
        task_data["train_accuracy"] = 0.0
        self.logger.warning("Brak danych dokładności treningowej w wynikach.")
        
    if has_train_loss:
        task_data["train_loss"] = history["train_loss"][-1]
    else:
        task_data["train_loss"] = 0.0
        self.logger.warning("Brak danych straty treningowej w wynikach.")
        
    if has_val_acc:
        task_data["validation_accuracy"] = history["val_acc"][-1]
    else:
        task_data["validation_accuracy"] = 0.0
        self.logger.warning("Brak danych dokładności walidacyjnej w wynikach.")
        
    if has_val_loss:
        task_data["validation_loss"] = history["val_loss"][-1]
    else:
        task_data["validation_loss"] = 0.0
        self.logger.warning("Brak danych straty walidacyjnej w wynikach.")
4. Plik app/gui/tabs/training_manager.py
Problem z konfiguracją zadania
W funkcji _handle_dialog_accept brakuje walidacji parametrów zanim zostaną one zapisane do pliku konfiguracyjnego zadania.
python# Zmiana w pliku app/gui/tabs/training_manager.py, w funkcji _handle_dialog_accept
# Dodać przed zapisem konfiguracji:
# Walidacja parametrów przed zapisem
if data_dir and not os.path.exists(data_dir):
    QMessageBox.warning(
        self, "Błąd", "Katalog treningowy nie istnieje."
    )
    return

if val_dir and not os.path.exists(val_dir):
    QMessageBox.warning(
        self, "Błąd", "Katalog walidacyjny nie istnieje."
    )
    return

if epochs <= 0:
    QMessageBox.warning(
        self, "Błąd", "Liczba epok musi być większa od zera."
    )
    return

if batch_size <= 0:
    QMessageBox.warning(
        self, "Błąd", "Rozmiar wsadu musi być większy od zera."
    )
    return

# Walidacja learning_rate (powinien być większy od zera)
try:
    lr = float(learning_rate_combo.currentText())
    if lr <= 0:
        raise ValueError("Learning rate musi być większy od zera.")
except ValueError as e:
    QMessageBox.warning(self, "Błąd", str(e))
    return
Problem z uruchamianiem zadań
W funkcji _run_task_from_queue brakuje sprawdzenia, czy plik zadania istnieje i czy jest poprawny.
python# Zmiana w pliku app/gui/tabs/training_manager.py, w funkcji _run_task_from_queue
# Dodać na początku funkcji:
# Sprawdź, czy plik zadania istnieje
if not os.path.exists(task_file):
    QMessageBox.warning(
        self, "Błąd", f"Plik zadania nie istnieje: {task_file}"
    )
    return

# Sprawdź, czy plik zadania jest poprawny
try:
    with open(task_file, "r", encoding="utf-8") as f:
        task_data = json.load(f)
    
    # Sprawdź podstawowe pola
    required_fields = ["name", "type", "status"]
    for field in required_fields:
        if field not in task_data:
            QMessageBox.warning(
                self, "Błąd", f"Plik zadania nie zawiera pola '{field}'"
            )
            return
except Exception as e:
    QMessageBox.warning(
        self, "Błąd", f"Nie można odczytać pliku zadania: {str(e)}"
    )
    return
5. Ulepszone zarządzanie pamięcią w trakcie treningu
W train_model_optimized proponuję dodać lepsze zarządzanie pamięcią GPU podczas treningu, aby zmniejszyć ryzyko wycieków pamięci.
python# Zmiana w pliku ai/optimized_training.py, w funkcji train_model_optimized
# Dodać na końcu każdej epoki, po zakończeniu pętli batch:

# Wyczyść nieużywaną pamięć GPU po każdej epoce
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    # Opcjonalnie sprawdź i wyświetl aktualnie używaną pamięć
    if device.type == "cuda":
        memory_allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
        memory_reserved = torch.cuda.memory_reserved(device) / (1024 * 1024)
        print(f"GPU memory: allocated={memory_allocated:.2f}MB, reserved={memory_reserved:.2f}MB")
6. Ulepszone zapisywanie modelu
W funkcji _run_training_task proponuję dodać zabezpieczenie na wypadek problemów z zapisem modelu.
python# Zmiana w pliku app/core/workers/batch_training_thread.py, w funkcji _run_training_task
# Modyfikacja fragmentu odpowiedzialnego za zapis modelu:

# Zapisz model
try:
    # Zapisz model do katalogu wyjściowego
    output_dir = os.path.join("data", "models")
    os.makedirs(output_dir, exist_ok=True)

    # Upewniamy się, że model ma przypisane class_names
    if "class_names" in result and result["class_names"]:
        model.class_names = result["class_names"]
        self.logger.info(f"Przypisano mapowanie klas: {result['class_names']}")
    elif hasattr(model, "class_names") and model.class_names:
        self.logger.info(f"Użyto istniejącego mapowania klas modelu: {model.class_names}")
    else:
        self.logger.warning("Brak mapowania klas w wynikach treningu i modelu!")
        
        # Dodane: Próbuj utworzyć proste mapowanie bazując na danych
        try:
            temp_transform = get_default_transforms()
            train_dataset = datasets.ImageFolder(training_dir, transform=temp_transform)
            class_names = {}
            for idx, class_name in enumerate(train_dataset.classes):
                class_names[str(idx)] = class_name
            
            model.class_names = class_names
            self.logger.info(f"Utworzono mapowanie klas na podstawie katalogów: {class_names}")
        except Exception as mapping_error:
            self.logger.error(f"Nie udało się utworzyć mapowania klas: {mapping_error}")
            model.class_names = {}

    # Generuj nazwę pliku modelu
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    model_filename = f"{model_type}_{epochs}epok_{timestamp}.pt"
    model_path = os.path.join(output_dir, model_filename)

    self.logger.info(f"Generowana nazwa modelu: {model_filename}")

    # Zapisz stan modelu z dokładnym śledzeniem błędów
    try:
        model.save(
            model_path,
            metadata={
                "accuracy": result.get("val_acc", 0),
                "training_time": training_time,
                "training_params": task_data,
                "timestamp": timestamp,
                "class_names": model.class_names,
            },
        )
        
        # Sprawdź, czy plik został rzeczywiście zapisany
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Plik modelu nie został utworzony: {model_path}")
            
        # Sprawdź rozmiar pliku
        file_size = os.path.getsize(model_path)
        if file_size < 1000:  # Mniej niż 1KB - prawdopodobnie uszkodzony
            raise ValueError(f"Zapisany plik modelu jest zbyt mały: {file_size} bajtów")
            
        self.logger.info(f"Model zapisany w: {model_path}, rozmiar: {file_size/1024/1024:.2f} MB")
        
    except Exception as save_error:
        self.logger.error(f"Błąd podczas zapisywania modelu: {save_error}")
        self.logger.error(f"TRACEBACK: {traceback.format_exc()}")
        
        # Spróbuj zapisać model w innej lokalizacji jako awaryjny
        try:
            backup_path = os.path.join(output_dir, f"backup_{model_filename}")
            torch.save(model.model.state_dict(), backup_path)
            self.logger.info(f"Zapisano kopię zapasową modelu w: {backup_path}")
        except Exception as backup_error:
            self.logger.error(f"Nie udało się zapisać kopii zapasowej: {backup_error}")
        
        raise

except Exception as e:
    self.logger.error(f"BŁĄD podczas zapisywania modelu: {e}")
    self.logger.error(f"TRACEBACK: {traceback.format_exc()}")
    raise
Podsumowanie
Przedstawiłam najważniejsze problemy i propozycje ich rozwiązania w kodzie trenowania modeli. Główne obszary ulepszeń to:

Lepsza obsługa błędów w kluczowych operacjach treningu
Aktualizacja przestarzałych parametrów modeli
Ulepszone zarządzanie pamięcią GPU
Dokładniejsza walidacja parametrów treningu
Lepsze zarządzanie zapisem i odczytem plików modeli
Zabezpieczenie przed niepoprawnymi wartościami parametrów

Wprowadzenie tych zmian powinno znacząco poprawić stabilność i niezawodność procesu trenowania modeli w aplikacji.