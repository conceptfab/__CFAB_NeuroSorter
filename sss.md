Wczytano plik ustawień: settings.json
2025-05-12 10:25:35,898 [INFO] Wczytano plik ustawień: settings.json
2025-05-12 10:25:36,958 [INFO] MainWindow __init__: Start
2025-05-12 10:25:36,958 [INFO] MainWindow __init__: self.settings zainicjalizowane jako: {'data_dir': 'data', 'models_dir': 'data/models', 'reports_dir': 'data/reports', 'log_level': 'INFO', 'log_file': 'app.log', 'chart_train_loss_color': 'b', 'chart_val_loss_color': 'r', 'chart_train_acc_color': 'g', 'chart_val_acc_color': 'm', 'chart_plot_area_background_color': 'w', 'autosave': True, 'confirm_exit': True, 'notifications': True}
2025-05-12 10:25:36,977 [INFO] Ustawiono limit cache modeli na 3.60 GB (30% GPU RAM)
2025-05-12 10:25:36,977 [INFO] Rozpoczynam ładowanie profilu sprzętowego...
2025-05-12 10:25:37,153 [INFO] Pomyślnie załadowano profil sprzętowy dla machine_id: a5323828-685b-5ae9-854e-b2a0eb8e8ae9
2025-05-12 10:25:37,155 [INFO] MainWindow __init__: Przed _create_central_widget(), self.settings = {'data_dir': 'data', 'models_dir': 'data/models', 'reports_dir': 'data/reports', 'log_level': 'INFO', 'log_file': 'app.log', 'chart_train_loss_color': 'b', 'chart_val_loss_color': 'r', 'chart_train_acc_color': 'g', 'chart_val_acc_color': 'm', 'chart_plot_area_background_color': 'w', 'autosave': True, 'confirm_exit': True, 'notifications': True}
2025-05-12 10:25:37,846 [INFO] Zaktualizowano profil sprzętowy w zakładce treningu
2025-05-12 10:25:37,861 [INFO] Pomyślnie załadowano profil sprzętowy
2025-05-12 10:25:41,268 [INFO] DEBUG: run_btn.clicked - Lambda wywołana dla pliku: data\tasks\EfficientNet-B2_32_2025-05-12_10-15.json
2025-05-12 10:25:41,269 [INFO] DEBUG: _run_task_from_queue wywołane dla pliku: data\tasks\EfficientNet-B2_32_2025-05-12_10-15.json
2025-05-12 10:25:41,269 [INFO] DEBUG: _run_task_from_queue - PRZED podłączeniem sygnałów dla data\tasks\EfficientNet-B2_32_2025-05-12_10-15.json
2025-05-12 10:25:41,269 [INFO] DEBUG: _run_task_from_queue - PO podłączeniu sygnałów dla data\tasks\EfficientNet-B2_32_2025-05-12_10-15.json
2025-05-12 10:25:41,270 [INFO] DEBUG: _run_task_from_queue - PRZED self.training_thread.start() dla data\tasks\EfficientNet-B2_32_2025-05-12_10-15.json
2025-05-12 10:25:41,270 [INFO] DEBUG: _run_task_from_queue - PO self.training_thread.start() dla data\tasks\EfficientNet-B2_32_2025-05-12_10-15.json
2025-05-12 10:25:41,271 [INFO] Uruchomiono pojedyncze zadanie.
2025-05-12 10:25:41,271 [INFO] === ROZPOCZYNAM WYKONYWANIE ZADANIA ===
2025-05-12 10:25:41,272 [INFO] SingleTrainingThread.run: Rozpoczęto działanie wątku.
2025-05-12 10:25:41,272 [INFO] EMITOWANO task_started dla: EfficientNet-B2_32_2025-05-12_10-15, typ: training
2025-05-12 10:25:41,272 [INFO] === INFORMACJE O ZADANIU ===
2025-05-12 10:25:41,272 [INFO] Nazwa zadania: EfficientNet-B2_32_2025-05-12_10-15
2025-05-12 10:25:41,272 [INFO] Typ zadania: training
2025-05-12 10:25:41,273 [INFO] Pełne dane zadania: {
  "name": "EfficientNet-B2_32_2025-05-12_10-15",
  "type": "training",
  "status": "Nowy",
  "priority": 0,
  "created_at": "2025-05-12 10:15:28",
  "config": {
    "train_dir": "F:/_set_A/training_data",
    "data_dir": "F:/_set_A/training_data",
    "val_dir": "F:/_set_A/validation_data",
    "model": {
      "architecture": "EfficientNet",
      "variant": "EfficientNet-B2",
      "input_size": 288,
      "num_classes": 32
    },
    "training": {
      "epochs": 120,
      "batch_size": 32,
      "learning_rate": 0.00015,
      "optimizer": "AdamW",
      "scheduler": "CosineAnnealingWarmRestarts",
      "num_workers": 8,
      "warmup_epochs": 8,
      "mixed_precision": true,
      "freeze_base_model": true,
      "unfreeze_layers": 15,
      "unfreeze_strategy": "unfreeze_all"
    },
    "regularization": {
      "weight_decay": 0.00015,
      "gradient_clip": 1.0,
      "label_smoothing": 0.1,
      "drop_connect_rate": 0.2,
      "dropout_rate": 0.3,
      "momentum": 0.9,
      "epsilon": 1e-06,
      "swa": {
        "use": true,
        "start_epoch": 60
      }
    },
    "augmentation": {
      "basic": {
        "use": true,
        "rotation": 25,
        "brightness": 0.25,
        "shift": 0.1,
        "zoom": 0.15,
        "horizontal_flip": true,
        "vertical_flip": false
      },
      "mixup": {
        "use": true,
        "alpha": 0.3
      },
      "cutmix": {
        "use": true,
        "alpha": 0.7
      },
      "autoaugment": {
        "use": false
      },
      "randaugment": {
        "use": false,
        "n": 2,
        "m": 9
      }
    },
    "preprocessing": {
      "normalization": "RGB",
      "scaling": {
        "method": "Bilinear",
        "maintain_aspect_ratio": true,
        "pad_to_square": false,
        "pad_mode": "constant",
        "pad_value": 0
      }
    },
    "monitoring": {
      "metrics": {
        "accuracy": true,
        "precision": true,
        "recall": true,
        "f1": true,
        "topk": true,
        "confusion_matrix": true
      },
      "early_stopping": {
        "patience": 20,
        "min_delta": 0.001,
        "monitor": "val_f1"
      },
      "checkpointing": {
        "best_only": true,
        "save_frequency": 1,
        "metric": "val_f1"
      },
      "tensorboard": {
        "use": false,
        "log_dir": ""
      },
      "save_dir": "",
      "save_logs": false
    },
    "advanced": {
      "scheduler": {
        "patience": 5,
        "factor": 0.1,
        "min_lr": 0.0,
        "cooldown": 0
      },
      "weights": {
        "init_method": "kaiming_normal",
        "freeze_cnn": false
      },
      "cross_validation": {
        "use": false,
        "folds": 5
      },
      "distributed": {
        "use": false,
        "backend": "nccl",
        "strategy": "ddp"
      },
      "gradients": {
        "clip": 1.0,
        "accumulation": 1
      },
      "online_validation": {
        "use": false,
        "frequency": 10
      }
    }
  }
}
2025-05-12 10:25:41,274 [INFO] ============ ROZPOZNANO ZADANIE TRENINGU ============
2025-05-12 10:25:41,274 [INFO] Zadanie będzie wykonywane przez skrypt: optimized_training.py
2025-05-12 10:25:41,274 [INFO] ============ ROZPOCZYNAM TRENING NOWEGO MODELU ============
2025-05-12 10:25:41,274 [INFO] Używam skryptu: optimized_training.py
2025-05-12 10:25:41,274 [INFO] Zadanie: EfficientNet-B2_32_2025-05-12_10-15
2025-05-12 10:25:41,274 [INFO] Ścieżka zadania: data\tasks\EfficientNet-B2_32_2025-05-12_10-15.json
2025-05-12 10:25:41,275 [INFO] DEBUG: _training_task_started wywołane dla zadania: EfficientNet-B2_32_2025-05-12_10-15, typ: training
2025-05-12 10:25:41,275 [INFO] DEBUG: _training_task_started - Stan przycisku stop PRZED ustawieniem: False
2025-05-12 10:25:41,275 [INFO] DEBUG: _training_task_started - Stan przycisku stop PO ustawieniu na True: True
2025-05-12 10:25:41,337 [INFO] 
Ładowanie danych treningowych w celu określenia liczby klas...
2025-05-12 10:25:41,358 [INFO] Wykryto 32 klas w katalogu treningowym:
2025-05-12 10:25:41,358 [INFO]   - Klasa 0: animal
2025-05-12 10:25:41,358 [INFO]   - Klasa 1: arm_chair
2025-05-12 10:25:41,359 [INFO]   - Klasa 2: bathroom
2025-05-12 10:25:41,359 [INFO]   - Klasa 3: bed
2025-05-12 10:25:41,359 [INFO]   - Klasa 4: building
2025-05-12 10:25:41,359 [INFO]   - Klasa 5: cabinet_shelf_wardrobe
2025-05-12 10:25:41,359 [INFO]   - Klasa 6: car
2025-05-12 10:25:41,359 [INFO]   - Klasa 7: childroom
2025-05-12 10:25:41,359 [INFO]   - Klasa 8: clothing_accessories
2025-05-12 10:25:41,359 [INFO]   - Klasa 9: decoration
2025-05-12 10:25:41,359 [INFO]   - Klasa 10: deocorative_greenery
2025-05-12 10:25:41,360 [INFO]   - Klasa 11: flower
2025-05-12 10:25:41,360 [INFO]   - Klasa 12: food
2025-05-12 10:25:41,360 [INFO]   - Klasa 13: grass
2025-05-12 10:25:41,360 [INFO]   - Klasa 14: indoor_plant
2025-05-12 10:25:41,360 [INFO]   - Klasa 15: lamp
2025-05-12 10:25:41,360 [INFO]   - Klasa 16: material_preview
2025-05-12 10:25:41,360 [INFO]   - Klasa 17: outdoor_plant
2025-05-12 10:25:41,360 [INFO]   - Klasa 18: pillow
2025-05-12 10:25:41,360 [INFO]   - Klasa 19: pouffe_or_other_seat
2025-05-12 10:25:41,360 [INFO]   - Klasa 20: scene
2025-05-12 10:25:41,360 [INFO]   - Klasa 21: sofa
2025-05-12 10:25:41,361 [INFO]   - Klasa 22: table_chair
2025-05-12 10:25:41,361 [INFO]   - Klasa 23: technology
2025-05-12 10:25:41,361 [INFO]   - Klasa 24: texture_alpha
2025-05-12 10:25:41,361 [INFO]   - Klasa 25: texture_color
2025-05-12 10:25:41,361 [INFO]   - Klasa 26: texture_glossiness
2025-05-12 10:25:41,361 [INFO]   - Klasa 27: texture_metallic
2025-05-12 10:25:41,361 [INFO]   - Klasa 28: texture_normal
2025-05-12 10:25:41,361 [INFO]   - Klasa 29: texture_preview
2025-05-12 10:25:41,361 [INFO]   - Klasa 30: tree
2025-05-12 10:25:41,361 [INFO]   - Klasa 31: vehicle
2025-05-12 10:25:41,361 [INFO]
Tworzenie modelu resnet18 dla 32 klas...
CUDA jest dostępne. Wykryto urządzenie: NVIDIA GeForce RTX 4070
Wersja CUDA: 12.1

Konfiguracja modelu:
- Architektura: 18
- Liczba klas: 32
2025-05-12 10:25:41,561 [INFO] Model utworzony pomyślnie: <ai.classifier.ImageClassifier object at 0x00000154D479B020>
2025-05-12 10:25:41,561 [INFO] DEBUG: Informacje o modelu: {
  "model_type": "18",
  "num_classes": 32,
  "total_parameters": 11736672,
  "trainable_parameters": 11736672,
  "device": "cuda",
  "input_size": [
    224,
    224
  ],
  "class_names": {}
}
2025-05-12 10:25:41,563 [INFO] DEBUG: Rozpoczynam trening z parametrami:
2025-05-12 10:25:41,563 [INFO] DEBUG: model=<class 'torchvision.models.resnet.ResNet'>
2025-05-12 10:25:41,563 [INFO] DEBUG: train_dir=F:/_set_A/training_data
2025-05-12 10:25:41,563 [INFO] DEBUG: val_dir=F:/_set_A/validation_data
2025-05-12 10:25:41,563 [INFO] DEBUG: num_epochs=10
2025-05-12 10:25:41,563 [INFO] DEBUG: batch_size=32
2025-05-12 10:25:41,563 [INFO] DEBUG: learning_rate=0.001
DEBUG: task_config przed zmianami: False
DEBUG: Parametry po zmianach: num_epochs=10, batch_size=32, learning_rate=0.001
Walidacja - Strata: 1.1587
Walidacja - Dokładność: 0.8541

--- PODSUMOWANIE EPOKI ---
Czas trwania: 62.76s
Średnia strata: 1.5150
Średnia dokładność: 0.7133
DEBUG optimized_training: Koniec epoki 1. Zaraz wywołam progress_callback (jeśli istnieje).
2025-05-12 10:26:44,580 [INFO] EMITOWANO task_progress dla: EfficientNet-B2_32_2025-05-12_10-15, epoka: 1
2025-05-12 10:26:44,583 [INFO] Epoka 1/10 | Strata: 1.5150, Dokładność: 71.33% | Val Strata: 1.1587, Val Acc: 85.41% | Top-3: 95.61%, Top-5: 97.62% | Precision: 81.24%, Recall: 75.79% | F1: 76.93%, AUC: 99.04%
GPU memory: allocated=168.42MB, reserved=200.00MB
Walidacja - Strata: 1.1137
Walidacja - Dokładność: 0.8757

--- PODSUMOWANIE EPOKI ---
Czas trwania: 62.04s
Średnia strata: 1.2642
Średnia dokładność: 0.8018
DEBUG optimized_training: Koniec epoki 2. Zaraz wywołam progress_callback (jeśli istnieje).
2025-05-12 10:27:46,625 [INFO] EMITOWANO task_progress dla: EfficientNet-B2_32_2025-05-12_10-15, epoka: 2
2025-05-12 10:27:46,628 [INFO] Epoka 2/10 | Strata: 1.2642, Dokładność: 80.18% | Val Strata: 1.1137, Val Acc: 87.57% | Top-3: 96.72%, Top-5: 98.51% | Precision: 86.26%, Recall: 82.04% | F1: 82.51%, AUC: 99.33%
GPU memory: allocated=168.42MB, reserved=200.00MB
