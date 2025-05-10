python# Zaraz po print("\n=== INICJALIZACJA FINE-TUNINGU ===")
# i data rozpoczęcia, dodaj:
start_training_time = time.time()
Następnie, w części zapisującej model (około linii 870), dodaj kod obliczający czas treningu i aktualizujący metadane:
python# Przed zapisem finalnego modelu, oblicz czas treningu
training_time = time.time() - start_training_time

# Przygotuj nowe metadane
new_metadata = {}
# Zachowaj wszystkie oryginalne metadane
if "metadata" in original_config:
    new_metadata = deepcopy(original_config["metadata"])

# Dodaj lub zaktualizuj czas treningu
if "training_time" in new_metadata:
    # Dodaj nowy czas treningu do istniejącego
    new_metadata["training_time"] += training_time
else:
    # Utwórz nowy element czasu treningu
    new_metadata["training_time"] = training_time

# Dodaj lub aktualizuj tylko te elementy, które się zmieniły
new_metadata["finetuning_history"] = finetuning_history
new_metadata["training_details"] = training_details
if task_name:
    new_metadata["task_name"] = task_name

# Zapisz model z kompletnymi metadanymi z oryginalnego modelu plus nowe informacje
final_classifier.save_with_original_config(final_model_path, original_config, new_metadata)