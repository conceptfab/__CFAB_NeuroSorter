Zmiany w pliku ai/fine_tuning.py
Znalazłem krytyczny błąd w pliku ai/fine_tuning.py, który powoduje problemy z funkcją treningu. Występują dwa główne problemy:
1. Ostrzeżenie o deprecacji torch.cuda.amp.autocast
W pliku obecna jest zdeprecjonowana funkcja torch.cuda.amp.autocast(), którą należy zaktualizować do nowszej wersji API.
Zmiana w funkcji fine_tune_model (linia 633):
python# Stary kod:
with torch.cuda.amp.autocast():
    outputs = model(inputs)
    # pozostały kod...

# Nowy kod:
with torch.amp.autocast(device_type='cuda'):
    outputs = model(inputs)
    # pozostały kod...
2. Błąd w wywołaniu progress_callback
Występuje poważny błąd w wywołaniu funkcji progress_callback, który powoduje przerwanie treningu z powodu brakujących argumentów.
Zmiana w funkcji fine_tune_model (linia 726):
python# Stary kod (powodujący błąd):
progress_callback(progress)

# Nowy kod (z pełnymi argumentami):
if progress_callback:
    try:
        # Pełne wywołanie z wszystkimi wymaganymi argumentami
        progress_callback(
            epoch + 1,
            num_epochs,
            train_loss,
            train_acc,
            val_loss if val_loader else 0,
            val_acc if val_loader else 0,
            0,  # top3
            0,  # top5
            0,  # precision
            0,  # recall
            0,  # f1
            0   # auc
        )
    except Exception as e:
        print(f"Błąd podczas wywołania progress_callback: {str(e)}")
To wywołanie powinno wystąpić w miejscu, gdzie wcześniej było prostsze wywołanie progress_callback(progress).
Te dwie zmiany powinny rozwiązać główne problemy w kodzie, które powodują błędy w procesie fine-tuningu modelu.