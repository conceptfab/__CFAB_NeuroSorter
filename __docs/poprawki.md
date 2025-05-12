Poprawki do pliku app/core/workers/single_training_thread.py
pythondef _run_finetuning_task(self, task_data, task_name, task_path):
    # [...istniejący kod...]
    
    config = task_data.get("config", {})
    
    # Podstawowe parametry (już istniejące)
    base_model_path = config.get("base_model", task_data.get("model_path", ""))
    training_dir = config.get("train_dir", config.get("training_dir", task_data.get("training_dir", "")))
    validation_dir = config.get("val_dir", task_data.get("val_dir", None))
    
    # Parametry treningu
    training_config = config.get("training", {})
    epochs = training_config.get("epochs", task_data.get("epochs", 5))
    batch_size = training_config.get("batch_size", task_data.get("batch_size", 32))
    learning_rate = training_config.get("learning_rate", task_data.get("learning_rate", 0.0001))
    optimizer_type = training_config.get("optimizer", "adamw").lower()
    scheduler_type = training_config.get("scheduler", "plateau").lower()
    warmup_epochs = training_config.get("warmup_epochs", 1)
    use_mixed_precision = training_config.get("mixed_precision", True)
    
    # Parametry regularizacji
    reg_config = config.get("regularization", {})
    weight_decay = reg_config.get("weight_decay", 0.01)
    label_smoothing = reg_config.get("label_smoothing", 0.1)
    
    # Parametry zapobiegania zapominaniu katastroftalnemu
    advanced_config = config.get("advanced", {})
    cf_prevention = advanced_config.get("catastrophic_forgetting_prevention", {})
    prevent_forgetting = cf_prevention.get("enable", True)
    preserve_original_classes = cf_prevention.get("preserve_original_classes", True)
    
    # Konfiguracja EWC
    ewc_config = cf_prevention.get("ewc_regularization")
    
    # Konfiguracja destylacji wiedzy
    knowledge_distillation_config = cf_prevention.get("knowledge_distillation")
    
    # Konfiguracja rehearsal
    rehearsal_config = cf_prevention.get("rehearsal")
    
    # Konfiguracja zamrażania warstw
    layer_freezing_config = cf_prevention.get("layer_freezing", {})
    freeze_ratio = layer_freezing_config.get("freeze_ratio", task_data.get("freeze_ratio", 0.8))
    
    # Parametry augmentacji
    augmentation_params = config.get("augmentation", {})
    
    # Parametry preprocessingu
    preprocessing_params = config.get("preprocessing", {})
    
    # Parametry monitorowania
    monitoring_config = config.get("monitoring", {})
    early_stopping_config = monitoring_config.get("early_stopping", {})
    early_stopping_patience = early_stopping_config.get("patience", 5)
    
    # Parametr użycia green diffusion
    use_green_diffusion = False  # Brak bezpośredniego odpowiednika w JSON
    
    # [...pozostały kod...]
    
    # Zmodyfikowane wywołanie funkcji z wszystkimi parametrami
    result = fine_tune_model(
        base_model_path=base_model_path,
        train_dir=training_dir,
        val_dir=validation_dir,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        freeze_ratio=freeze_ratio,
        optimizer_type=optimizer_type,
        scheduler_type=scheduler_type,
        label_smoothing=label_smoothing,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        use_mixed_precision=use_mixed_precision,
        prevent_forgetting=prevent_forgetting,
        preserve_original_classes=preserve_original_classes,
        rehearsal_config=rehearsal_config,
        knowledge_distillation_config=knowledge_distillation_config,
        ewc_config=ewc_config,
        layer_freezing_config=layer_freezing_config,
        augmentation_params=augmentation_params,
        preprocessing_params=preprocessing_params,
        early_stopping_patience=early_stopping_patience,
        use_green_diffusion=use_green_diffusion,
        task_name=task_name,
        progress_callback=progress_callback,
        should_stop_callback=lambda: self._stopped,
    )