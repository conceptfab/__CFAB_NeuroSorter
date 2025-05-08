Podział okna konfiguracji doszkalania
Oto propozycja podziału okna konfiguracji doszkalania na logiczne sekcje z uwzględnieniem dobrych praktyk UI/UX:
1. Zakładki główne
Podzielmy interfejs na kilka głównych zakładek, które grupują powiązane ustawienia:
+---------------------------------------------------------------------------------------+
|                                                                                       |
| [Podstawowe] [Model] [Trening] [Dane] [Augmentacja] [Zaawansowane] [Monitoring]       |
|                                                                                       |
+---------------------------------------------------------------------------------------+
2. Zawartość zakładek
Zakładka: Podstawowe
+---------------------------------------------------------------------------------------+
| ■ Informacje ogólne                                                                   |
|   Nazwa projektu: [_____________________] Wersja: [_________]                         |
|   Opis: [________________________________________________________________]           |
|                                                                                       |
| ■ Ścieżki                                                                             |
|   Model bazowy: [_________________________] [Przeglądaj...]                           |
|   Katalog zapisu: [______________________] [Przeglądaj...]                           |
|                                                                                       |
| ■ Sprzęt                                                                              |
|   GPU: [Wybierz GPU ▼]                                                                |
|   Pamięć RAM: [________] GB                                                           |
|   Liczba rdzeni CPU: [__]                                                             |
|                                                                                       |
| [      Zapisz konfigurację      ]        [      Wczytaj konfigurację      ]           |
+---------------------------------------------------------------------------------------+
Zakładka: Model
+---------------------------------------------------------------------------------------+
| ■ Architektura                                                                        |
|   Model: [ConvNeXt ▼]                                                                 |
|   Wariant: [ConvNeXt-Tiny ▼]                                                          |
|   Rozmiar wejścia: [224] x [224] px                                                   |
|   Liczba klas: [38]                                                                   |
|                                                                                       |
| ■ Inicjalizacja wag                                                                   |
|   [x] Użyj wag pretrenowanych                                                         |
|   Metoda inicjalizacji: [kaiming_normal ▼]                                            |
|                                                                                       |
| ■ Strategia doszkalania                                                               |
|   [x] Zamroź model bazowy                                                             |
|   Odmrażane warstwy: [last3 ▼]  Strategia odmrażania: [Po 5 epokach ▼]               |
|   Strategia transferu: [feature_extraction ▼]                                         |
|   [x] Dostosuj warstwę wyjściową                                                      |
|                                                                                       |
| ■ Różne współczynniki uczenia dla warstw                                              |
|   [x] Użyj layer decay                                                                |
|   Współczynnik zaniku: [0.75]                                                         |
+---------------------------------------------------------------------------------------+
Zakładka: Trening
+---------------------------------------------------------------------------------------+
| ■ Harmonogram treningu                                                                |
|   Liczba epok: [40]                                                                   |
|   Rozmiar batcha: [16]                                                                |
|   [x] Mixed precision                                                                 |
|                                                                                       |
| ■ Optymalizacja                                                                       |
|   Optymalizator: [AdamW ▼]                                                            |
|   Learning rate: [0.00005]                                                            |
|   Scheduler: [CosineAnnealingLR ▼]                                                    |
|                                                                                       |
| ■ Parametry schedulera                                                                |
|   T_max: [40]                                                                         |
|   Eta_min: [0.0000001]                                                                |
|   Epoki rozgrzewki: [3]                                                               |
|                                                                                       |
| ■ Regularyzacja                                                                       |
|   Weight decay: [0.001]                                                               |
|   Dropout: [0.2]                                                                      |
|   Label smoothing: [0.05]                                                             |
|   Gradient clip: [0.5]                                                                |
|                                                                                       |
| ■ Stochastic Weight Averaging (SWA)                                                   |
|   [x] Użyj SWA                                                                        |
|   Epoka startowa: [25]                                                                |
+---------------------------------------------------------------------------------------+
Zakładka: Dane
+---------------------------------------------------------------------------------------+
| ■ Zbiór danych                                                                        |
|   Ścieżka do danych: [_________________________] [Przeglądaj...]                      |
|   Liczba klas: [38]                                                                   |
|                                                                                       |
| ■ Podział danych                                                                      |
|   [x] Automatyczny podział                                                            |
|   Train: [95] obrazów/klasę   Val: [5] obrazów/klasę                                  |
|                                                                                       |
| ■ Preprocessing                                                                       |
|   Normalizacja: [ImageNet ▼]                                                          |
|   Metoda skalowania: [Bicubic ▼]                                                      |
|   [x] Zachowaj proporcje                                                              |
|   [x] Dopełnij do kwadratu                                                            |
|   Tryb dopełniania: [reflection ▼]                                                    |
|   Wartość dopełnienia: [0]                                                            |
|                                                                                       |
| ■ Wczytywanie danych                                                                  |
|   Liczba workerów: [14]                                                               |
|   [x] Prefetching                                                                     |
|   Cache wielkość: [1000]                                                              |
+---------------------------------------------------------------------------------------+
Zakładka: Augmentacja
+---------------------------------------------------------------------------------------+
| ■ Podstawowe augmentacje                                                              |
|   [x] Włącz podstawowe augmentacje                                                    |
|                                                                                       |
|   Rotacja: [10]°                        [x] Horizontal flip    [ ] Vertical flip      |
|   Jasność: [0.2]                        Kontrast: [0.2]                               |
|   Nasycenie: [0.1]                      Odcień: [0.05]                                |
|   Przesunięcie: [0.1]                   Zoom: [0.15]                                  |
|                                                                                       |
| ■ Zaawansowane augmentacje                                                            |
|   [ ] Mixup          Alpha: [0.5]                                                     |
|   [ ] CutMix         Alpha: [1.0]                                                     |
|   [ ] AutoAugment                                                                     |
|   [x] RandAugment    N: [2]    M: [9]                                                 |
|   [ ] TrivialAugment                                                                  |
|                                                                                       |
|   [      Podgląd augmentacji      ]                                                   |
+---------------------------------------------------------------------------------------+
Zakładka: Zaawansowane
+---------------------------------------------------------------------------------------+
| ■ Walidacja                                                                           |
|   [ ] Cross-validation    Liczba foldów: [0]                                          |
|   [x] Walidacja online    Częstotliwość: [1] epok                                     |
|                                                                                       |
| ■ Gradienty                                                                           |
|   Clip norm: [0.5]                                                                    |
|   Akumulacja gradientu: [2] batchy                                                    |
|                                                                                       |
| ■ Distributed training                                                                |
|   [ ] Włącz distributed training                                                      |
|   Backend: [nccl ▼]                                                                   |
|   Strategia: [ddp ▼]                                                                  |
|                                                                                       |
| ■ Eksport modelu                                                                      |
|   Format eksportu: [ONNX ▼]                                                           |
|   [x] Optymalizacja modelu                                                            |
|   [x] Kwantyzacja                                                                     |
|   Dokładność kwantyzacji: [INT8 ▼]                                                    |
|                                                                                       |
| ■ Debugowanie                                                                         |
|   [ ] Trace gradients                                                                 |
|   [ ] Profile performance                                                             |
|   Profiler steps: [100]                                                               |
+---------------------------------------------------------------------------------------+
Zakładka: Monitoring
+---------------------------------------------------------------------------------------+
| ■ Metryki                                                                             |
|   [x] Accuracy        [x] Precision      [x] Recall                                   |
|   [x] F1 Score        [x] Top-K          [x] Confusion Matrix                         |
|                                                                                       |
| ■ Early Stopping                                                                      |
|   Monitorowana metryka: [val_f1 ▼]                                                    |
|   Cierpliwość: [10] epok                                                              |
|   Min. delta: [0.001]                                                                 |
|                                                                                       |
| ■ Checkpointing                                                                       |
|   [x] Zapisuj tylko najlepszy model                                                   |
|   Częstotliwość zapisu: [1] epok                                                      |
|   Monitorowana metryka: [val_f1 ▼]                                                    |
|                                                                                       |
| ■ Logowanie                                                                           |
|   [x] Zapisuj logi                                                                    |
|   [x] TensorBoard                                                                     |
|   Katalog logów: [logs/convnext_finetuning]                                           |
|                                                                                       |
| ■ Notyfikacje                                                                         |
|   [ ] Email          [ ] Slack           [ ] Discord                                  |
|   Email: [______________________]                                                     |
|   Webhook URL: [______________________]                                               |
+---------------------------------------------------------------------------------------+
3. Panel kontrolny
U dołu lub z boku można dodać panel kontrolny z najważniejszymi funkcjami:
+---------------------------------------------------------------------------------------+
| [   Rozpocznij trening   ]    [   Wstrzymaj   ]    [   Zatrzymaj   ]                  |
|                                                                                       |
| Status: [Oczekiwanie...]                                      Postęp: [█████░░░░░] 50% |
|                                                                                       |
| [   Zapisz model   ]    [   Eksportuj model   ]    [   Wizualizacja wyników   ]       |
+---------------------------------------------------------------------------------------+
4. Podgląd w czasie rzeczywistym
Dodajmy zakładkę lub panel boczny z wykresami i statystykami w czasie rzeczywistym:
+---------------------------------------------------------------------------------------+
|                                                                                       |
|  ┌─────────────────────────┐  ┌─────────────────────────┐                            |
|  │                         │  │                         │                            |
|  │                         │  │                         │                            |
|  │   Wykres straty (Loss)  │  │  Wykres dokładności     │                            |
|  │                         │  │                         │                            |
|  │                         │  │                         │                            |
|  └─────────────────────────┘  └─────────────────────────┘                            |
|                                                                                       |
|  ┌─────────────────────────┐  ┌─────────────────────────┐                            |
|  │                         │  │                         │                            |
|  │                         │  │                         │                            |
|  │   Learning Rate         │  │  Macierz pomyłek        │                            |
|  │                         │  │                         │                            |
|  │                         │  │                         │                            |
|  └─────────────────────────┘  └─────────────────────────┘                            |
|                                                                                       |
+---------------------------------------------------------------------------------------+
5. Dodatkowe funkcje
Można również dodać przycisk "Porównaj z oryginalnym profilem", który pokaże różnice między konfiguracją treningu a konfiguracją doszkalania:
+---------------------------------------------------------------------------------------+
|                                                                                       |
|  [   Porównaj z profilem treningu   ]                                                 |
|                                                                                       |
|  Parametr             |  Profil treningu  |  Profil doszkalania  |  Zmiana            |
|  --------------------|-------------------|---------------------|---------------------|
|  Liczba epok          |  120              |  40                  |  -80               |
|  Learning rate        |  0.0001           |  0.00005             |  -50%              |
|  Batch size           |  32               |  16                  |  -50%              |
|  Weight decay         |  0.01             |  0.001               |  -90%              |
|  ...                  |  ...              |  ...                 |  ...               |
|                                                                                       |
+---------------------------------------------------------------------------------------+
6. Szybki dostęp
Dodatkowy panel z szablonami konfiguracji i często używanymi ustawieniami:
+---------------------------------------------------------------------------------------+
| Szablony konfiguracji:                                                                |
| [Szybkie doszkalanie]  [Dokładne doszkalanie]  [Małe zbiory danych]  [Duże zbiory]   |
|                                                                                       |
| Często używane ustawienia:                                                            |
| [Mały learning rate]  [Duża augmentacja]  [Tylko ostatnia warstwa]  [Transfer]       |
+---------------------------------------------------------------------------------------+
7. Pomoc i dokumentacja
Panel z kontekstową pomocą i poradami:
+---------------------------------------------------------------------------------------+
| ℹ️ Wskazówka: Przy doszkalaniu modelu zalecane jest zamrożenie większości warstw      |
| i zastosowanie niższego learning rate.                                                |
|                                                                                       |
| 📘 Dokumentacja: [Strategia doszkalania]  [Najlepsze praktyki]  [Przykłady]           |
+---------------------------------------------------------------------------------------+




Cześć! Przeanalizowałem podany profil treningu dla modelu ConvNeXt-Tiny. Profil doszkalania (fine-tuning) powinien się różnić od profilu pełnego treningu w kilku kluczowych aspektach. Oto proponowane zmiany, które należy wprowadzić w pliku konfiguracyjnym:

python"training": {
  "epochs": 40,                    # Zmniejszona liczba epok (z 120 na 40)
  "batch_size": 16,                # Mniejszy rozmiar batcha (z 32 na 16)
  "learning_rate": 0.00005,        # Niższy learning rate (z 0.0001 na 0.00005)
  "optimizer": "AdamW",            # Bez zmian
  "scheduler": "CosineAnnealingLR", # Zmiana z OneCycleLR na CosineAnnealingLR
  "num_workers": 14,               # Bez zmian
  "warmup_epochs": 3,              # Mniej epok rozgrzewki (z 8 na 3)
  "mixed_precision": true,         # Bez zmian
  "freeze_base_model": true,       # Bez zmian
  "unfreeze_layers": "last3",      # Zmiana z "all" na "last3" - odmrażamy tylko ostatnie 3 warstwy
  "unfreeze_strategy": "Po 5 epokach" # Wcześniejsze odmrażanie (z 15 na 5 epok)
}
Zmiana w pliku config.json, sekcja "regularization":
python"regularization": {
  "weight_decay": 0.001,           # Mniejszy weight decay (z 0.01 na 0.001)
  "gradient_clip": 0.5,            # Mniejszy gradient clip (z 1.0 na 0.5)
  "label_smoothing": 0.05,         # Mniejsze label smoothing (z 0.1 na 0.05)
  "dropout_rate": 0.2,             # Mniejszy dropout (z 0.3 na 0.2)
  "momentum": 0.9,                 # Bez zmian
  "epsilon": 1e-8,                 # Bez zmian
  "swa": {
    "use": true,                   # Bez zmian
    "start_epoch": 25              # Wcześniejsze rozpoczęcie SWA (z 70 na 25)
  }
}
Zmiana w pliku config.json, sekcja "augmentation":
python"augmentation": {
  "basic": {
    "use": true,                   # Bez zmian
    "rotation": 10,                # Mniejsza rotacja (z 20 na 10)
    "brightness": 0.2,             # Mniejsza zmiana jasności (z 0.3 na 0.2)
    "contrast": 0.2,               # Mniejsza zmiana kontrastu (z 0.3 na 0.2)
    "saturation": 0.1,             # Mniejsza zmiana nasycenia (z 0.2 na 0.1)
    "hue": 0.05,                   # Mniejsza zmiana odcienia (z 0.1 na 0.05)
    "shift": 0.1,                  # Mniejsze przesunięcie (z 0.15 na 0.1)
    "zoom": 0.15,                  # Mniejszy zoom (z 0.25 na 0.15)
    "horizontal_flip": true,       # Bez zmian
    "vertical_flip": false         # Bez zmian
  },
  "mixup": {
    "use": false                   # Wyłączenie mixup (z true na false)
  },
  "cutmix": {
    "use": false                   # Wyłączenie cutmix (z true na false)
  },
  "autoaugment": {
    "use": false                   # Wyłączenie autoaugment (z true na false)
  },
  "randaugment": {
    "use": true,                   # Bez zmian
    "n": 2,                        # Mniej transformacji (z 3 na 2)
    "m": 9                         # Mniejsza intensywność (z 12 na 9)
  },
  "trivialaugment": {
    "use": false                   # Wyłączenie trivialaugment (z true na false)
  }
}
Zmiana w pliku config.json, sekcja "advanced":
python"advanced": {
  "scheduler": {
    "T_max": 40,                   # Nowy parametr dla CosineAnnealingLR
    "eta_min": 1e-7                # Nowy parametr dla CosineAnnealingLR
    # Usunięto parametry OneCycleLR, które nie są już potrzebne
  },
  "weights": {
    "init_method": "kaiming_normal", # Bez zmian
    "freeze_cnn": true,            # Zamrożenie CNN (z false na true)
    "pretrained": true             # Bez zmian
  },
  "cross_validation": {
    "use": false,                  # Wyłączenie cross-validation (z true na false)
    "folds": 0                     # Brak foldów (z 5 na 0)
  },
  "distributed": {
    "use": false,                  # Bez zmian
    "backend": "nccl",             # Bez zmian
    "strategy": "ddp"              # Bez zmian
  },
  "gradients": {
    "clip": 0.5,                   # Mniejszy clip gradientu (z 1.0 na 0.5)
    "accumulation": 2              # Zwiększona akumulacja gradientu (z 1 na 2)
  },
  "online_validation": {
    "use": true,                   # Bez zmian
    "frequency": 1                 # Częstsza walidacja (z 2 na 1)
  }
}
Dodanie nowej sekcji w pliku config.json:
python"fine_tuning": {
  "enabled": true,
  "base_model_path": "checkpoints/convnext/best_model.pth",
  "transfer_strategy": "feature_extraction",
  "adapt_final_layer": true,
  "layer_decay": {
    "use": true,
    "decay_factor": 0.75
  }
}
Główne różnice między treningiem a doszkalaniem:

Krótszy czas treningu (mniej epok)
Niższy learning rate
Mniejsza intensywność augmentacji danych
Mniej agresywna regularyzacja
Zamrożenie większości warstw modelu (odmrażanie tylko ostatnich warstw)
Dodanie strategi doszkalania (transfer learning)
Wyłączenie zaawansowanych technik augmentacji jak mixup i cutmix
Zastosowanie akumulacji gradientu
Wyłączenie walidacji krzyżowej

Te zmiany pozwolą na efektywne doszkolenie modelu ConvNeXt-Tiny na nowym zbiorze danych bez ryzyka przeuczenia i z zachowaniem wiedzy zdobytej podczas wstępnego treningu.RetryMmozesz mi przygotowac schemat dla całego profilu razem z wyjaśnieniem?EditSchemat profilu doszkalania modelu ConvNeXt-Tiny
Poniżej przedstawiam kompletny schemat profilu doszkalania wraz ze szczegółowym wyjaśnieniem każdej sekcji.
python{
  "info": "Profil doszkalania dla ConvNeXt-Tiny",
  "description": "Profil zoptymalizowany do fine-tuningu modelu ConvNeXt na 38 kategorii z małym zbiorem danych",
  "data_required": "38 klas, 95 obrazów treningowych i 5 walidacyjnych na klasę",
  "hardware_required": "GPU RTX 4700, RYZEN 9 16-rdzeniowy, RAM 128GB",
  "config": {
    "model": {
      "architecture": "ConvNeXt",
      "variant": "ConvNeXt-Tiny",
      "input_size": 224,
      "num_classes": 38
    },
    "training": {
      "epochs": 40,
      "batch_size": 16,
      "learning_rate": 0.00005,
      "optimizer": "AdamW",
      "scheduler": "CosineAnnealingLR",
      "num_workers": 14,
      "warmup_epochs": 3,
      "mixed_precision": true,
      "freeze_base_model": true,
      "unfreeze_layers": "last3",
      "unfreeze_strategy": "Po 5 epokach"
    },
    "regularization": {
      "weight_decay": 0.001,
      "gradient_clip": 0.5,
      "label_smoothing": 0.05,
      "dropout_rate": 0.2,
      "momentum": 0.9,
      "epsilon": 1e-8,
      "swa": {
        "use": true,
        "start_epoch": 25
      }
    },
    "augmentation": {
      "basic": {
        "use": true,
        "rotation": 10,
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.1,
        "hue": 0.05,
        "shift": 0.1,
        "zoom": 0.15,
        "horizontal_flip": true,
        "vertical_flip": false
      },
      "mixup": {
        "use": false
      },
      "cutmix": {
        "use": false
      },
      "autoaugment": {
        "use": false
      },
      "randaugment": {
        "use": true,
        "n": 2,
        "m": 9
      },
      "trivialaugment": {
        "use": false
      }
    },
    "preprocessing": {
      "normalization": "ImageNet",
      "scaling": {
        "method": "Bicubic",
        "maintain_aspect_ratio": true,
        "pad_to_square": true,
        "pad_mode": "reflection",
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
        "patience": 10,
        "min_delta": 0.001,
        "monitor": "val_f1"
      },
      "checkpointing": {
        "best_only": true,
        "save_frequency": 1,
        "metric": "val_f1"
      },
      "tensorboard": {
        "use": true,
        "log_dir": "logs/convnext_finetuning"
      },
      "save_dir": "checkpoints/convnext_finetuned",
      "save_logs": true
    },
    "advanced": {
      "scheduler": {
        "T_max": 40,
        "eta_min": 1e-7
      },
      "weights": {
        "init_method": "kaiming_normal",
        "freeze_cnn": true,
        "pretrained": true
      },
      "cross_validation": {
        "use": false,
        "folds": 0
      },
      "distributed": {
        "use": false,
        "backend": "nccl",
        "strategy": "ddp"
      },
      "gradients": {
        "clip": 0.5,
        "accumulation": 2
      },
      "online_validation": {
        "use": true,
        "frequency": 1
      }
    },
    "fine_tuning": {
      "enabled": true,
      "base_model_path": "checkpoints/convnext/best_model.pth",
      "transfer_strategy": "feature_extraction",
      "adapt_final_layer": true,
      "layer_decay": {
        "use": true,
        "decay_factor": 0.75
      }
    }
  }
}
Szczegółowe wyjaśnienie parametrów
1. Sekcja informacyjna
python"info": "Profil doszkalania dla ConvNeXt-Tiny",
"description": "Profil zoptymalizowany do fine-tuningu modelu ConvNeXt na 38 kategorii z małym zbiorem danych",
"data_required": "38 klas, 95 obrazów treningowych i 5 walidacyjnych na klasę",
"hardware_required": "GPU RTX 4700, RYZEN 9 16-rdzeniowy, RAM 128GB",
Wyjaśnienie: Ta sekcja zawiera ogólne informacje o profilu. Określamy, że jest to profil doszkalania, a nie treningu od podstaw. Ilość danych i wymagany sprzęt pozostają takie same jak w oryginalnym profilu.
2. Model
python"model": {
  "architecture": "ConvNeXt",
  "variant": "ConvNeXt-Tiny",
  "input_size": 224,
  "num_classes": 38
}
Wyjaśnienie: Parametry modelu pozostają bez zmian w stosunku do oryginalnego profilu. Używamy tego samego modelu ConvNeXt-Tiny z tą samą liczbą klas (38) i rozmiarem wejścia (224x224 pikseli).
3. Trening
python"training": {
  "epochs": 40,
  "batch_size": 16,
  "learning_rate": 0.00005,
  "optimizer": "AdamW",
  "scheduler": "CosineAnnealingLR",
  "num_workers": 14,
  "warmup_epochs": 3,
  "mixed_precision": true,
  "freeze_base_model": true,
  "unfreeze_layers": "last3",
  "unfreeze_strategy": "Po 5 epokach"
}
Wyjaśnienie:

epochs: Zmniejszono z 120 do 40, ponieważ doszkalanie wymaga mniej epok niż trening od podstaw.
batch_size: Zmniejszono z 32 do 16, co pozwala na bardziej precyzyjne aktualizacje wag.
learning_rate: Obniżono z 0.0001 do 0.00005, ponieważ przy doszkalaniu potrzebujemy delikatniejszych zmian wag.
scheduler: Zmieniono z OneCycleLR na CosineAnnealingLR, który lepiej sprawdza się przy doszkalaniu.
warmup_epochs: Zmniejszono z 8 do 3, gdyż przy doszkalaniu nie potrzebujemy długiego okresu rozgrzewki.
freeze_base_model: Pozostawiono jako true, aby zachować wiedzę z pretrenowanego modelu.
unfreeze_layers: Zmieniono z "all" na "last3", co oznacza, że trenujemy tylko ostatnie 3 warstwy.
unfreeze_strategy: Zmieniono z "Po 15 epokach" na "Po 5 epokach", co pozwala na wcześniejsze dostosowanie końcowych warstw.

4. Regularyzacja
python"regularization": {
  "weight_decay": 0.001,
  "gradient_clip": 0.5,
  "label_smoothing": 0.05,
  "dropout_rate": 0.2,
  "momentum": 0.9,
  "epsilon": 1e-8,
  "swa": {
    "use": true,
    "start_epoch": 25
  }
}
Wyjaśnienie:

weight_decay: Zmniejszono z 0.01 do 0.001, aby zmniejszyć regularyzację i umożliwić lepsze dostosowanie do nowych danych.
gradient_clip: Zmniejszono z 1.0 do 0.5, aby ograniczyć duże zmiany wag.
label_smoothing: Zmniejszono z 0.1 do 0.05, co daje mniejszą regularyzację funkcji straty.
dropout_rate: Zmniejszono z 0.3 do 0.2, co oznacza mniejszą regularyzację sieci.
swa.start_epoch: Zmieniono z 70 na 25, aby wcześniej rozpocząć uśrednianie wag w krótszym treningu.

5. Augmentacja
python"augmentation": {
  "basic": {
    "use": true,
    "rotation": 10,
    "brightness": 0.2,
    "contrast": 0.2,
    "saturation": 0.1,
    "hue": 0.05,
    "shift": 0.1,
    "zoom": 0.15,
    "horizontal_flip": true,
    "vertical_flip": false
  },
  "mixup": {
    "use": false
  },
  "cutmix": {
    "use": false
  },
  "autoaugment": {
    "use": false
  },
  "randaugment": {
    "use": true,
    "n": 2,
    "m": 9
  },
  "trivialaugment": {
    "use": false
  }
}
Wyjaśnienie:

Podstawowe augmentacje: Wszystkie parametry zostały zmniejszone, aby transformacje były mniej agresywne.
mixup, cutmix, autoaugment, trivialaugment: Wyłączono te techniki (zmieniono z true na false), ponieważ są zbyt agresywne dla doszkalania.
randaugment: Pozostawiono włączone, ale zmniejszono liczbę transformacji (n) z 3 do 2 i intensywność (m) z 12 do 9.

6. Preprocessing
python"preprocessing": {
  "normalization": "ImageNet",
  "scaling": {
    "method": "Bicubic",
    "maintain_aspect_ratio": true,
    "pad_to_square": true,
    "pad_mode": "reflection",
    "pad_value": 0
  }
}
Wyjaśnienie: Parametry preprocessingu pozostają bez zmian, ponieważ sposób przygotowania danych powinien być taki sam jak podczas treningu oryginalnego modelu.
7. Monitoring
python"monitoring": {
  "metrics": {
    "accuracy": true,
    "precision": true,
    "recall": true,
    "f1": true,
    "topk": true,
    "confusion_matrix": true
  },
  "early_stopping": {
    "patience": 10,
    "min_delta": 0.001,
    "monitor": "val_f1"
  },
  "checkpointing": {
    "best_only": true,
    "save_frequency": 1,
    "metric": "val_f1"
  },
  "tensorboard": {
    "use": true,
    "log_dir": "logs/convnext_finetuning"
  },
  "save_dir": "checkpoints/convnext_finetuned",
  "save_logs": true
}
Wyjaśnienie:

early_stopping.patience: Zmniejszono z 20 do 10, ponieważ przy krótszym treningu chcemy wcześniej zatrzymać proces w przypadku braku poprawy.
tensorboard.log_dir: Zmieniono ścieżkę z "logs/convnext" na "logs/convnext_finetuning", aby oddzielić logi doszkalania.
save_dir: Zmieniono ścieżkę z "checkpoints/convnext" na "checkpoints/convnext_finetuned", aby oddzielić zapisane modele.

8. Zaawansowane ustawienia
python"advanced": {
  "scheduler": {
    "T_max": 40,
    "eta_min": 1e-7
  },
  "weights": {
    "init_method": "kaiming_normal",
    "freeze_cnn": true,
    "pretrained": true
  },
  "cross_validation": {
    "use": false,
    "folds": 0
  },
  "distributed": {
    "use": false,
    "backend": "nccl",
    "strategy": "ddp"
  },
  "gradients": {
    "clip": 0.5,
    "accumulation": 2
  },
  "online_validation": {
    "use": true,
    "frequency": 1
  }
}
Wyjaśnienie:

scheduler: Zmieniono parametry dla CosineAnnealingLR zamiast OneCycleLR.
weights.freeze_cnn: Zmieniono z false na true, aby zamrozić wagi sieci konwolucyjnej.
cross_validation: Wyłączono walidację krzyżową, ponieważ przy doszkalaniu zwykle nie jest ona konieczna.
gradients.clip: Zmniejszono z 1.0 do 0.5, aby ograniczyć duże zmiany wag.
gradients.accumulation: Zwiększono z 1 do 2, co oznacza akumulację gradientu z dwóch mini-batchy przed aktualizacją wag.
online_validation.frequency: Zmniejszono z 2 do 1, aby wykonywać walidację po każdej epoce.

9. Fine-tuning (Nowa sekcja)
python"fine_tuning": {
  "enabled": true,
  "base_model_path": "checkpoints/convnext/best_model.pth",
  "transfer_strategy": "feature_extraction",
  "adapt_final_layer": true,
  "layer_decay": {
    "use": true,
    "decay_factor": 0.75
  }
}
Wyjaśnienie:

enabled: Włącza tryb doszkalania.
base_model_path: Ścieżka do wczytania wag modelu bazowego.
transfer_strategy: Strategia "feature_extraction" oznacza, że używamy modelu głównie jako ekstraktora cech.
adapt_final_layer: Dostosowujemy warstwę wyjściową do nowego zbioru danych.
layer_decay: Włączamy mechanizm różnych współczynników uczenia dla różnych warstw, głębsze warstwy (bliżej wejścia) mają mniejszy współczynnik uczenia.

Podsumowanie głównych różnic między treningiem a doszkalaniem:

Krótszy czas treningu: 40 epok zamiast 120.
Mniejszy learning rate: 0.00005 zamiast 0.0001.
Modyfikacja tylko części modelu: Trenujemy tylko ostatnie 3 warstwy.
Łagodniejsza regularyzacja: Mniejszy weight decay, mniejszy dropout.
Mniej agresywna augmentacja: Wyłączenie zaawansowanych technik jak mixup i cutmix.
Inny scheduler: CosineAnnealingLR zamiast OneCycleLR.
Akumulacja gradientu: Ustawienie accumulation=2 dla stabilniejszego treningu.
Dodatkowa sekcja fine_tuning: Parametry specyficzne dla procesu doszkalania.
Wyłączenie walidacji krzyżowej: Nie jest potrzebna przy doszkalaniu.
Częstsza walidacja: Sprawdzanie wyników po każdej epoce.

Ten profil jest zoptymalizowany do efektywnego doszkalania modelu ConvNeXt-Tiny na nowym zbiorze danych, z zachowaniem wiedzy z pretrenowanego modelu i minimalizacją ryzyka przeuczenia.