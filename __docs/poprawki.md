Przeanalizuję problem z wyświetlaniem danych z treningu w pliku training_visualization.py. Na podstawie dostarczonego kodu mogę zidentyfikować kilka potencjalnych przyczyn, dlaczego wizualizacja treningu może nie działać prawidłowo.
Analiza problemu z wizualizacją treningu
Główne problemy zidentyfikowane w kodzie:
1. Problem z przepływem danych
W metodzie _training_task_progress w klasie TrainingManager dane są przekazywane do wizualizacji, ale występują problemy z typami danych. Wartości są najpierw konwertowane:
python# Upewnij się, że wartości są liczbami
try:
    loss = float(loss) if loss is not None else 0.0
    accuracy = float(accuracy) if accuracy is not None else 0.0
    val_loss = float(val_loss) if val_loss is not None else None
    val_acc = float(val_acc) if val_acc is not None else None
except (ValueError, TypeError) as e:
    print(f"BŁĄD konwersji wartości: {e}")
    self.parent.logger.error(
        f"Błąd konwersji wartości w _training_task_progress: {e}"
    )
    return
2. Walidacja danych przed aktualizacją wykresu
W metodzie update_data w TrainingVisualization istnieje problematyczna walidacja danych:
python# Sprawdź czy dane są sensowne
if train_loss <= 0 or train_acc < 0 or train_acc > 1:
    print("BŁĄD: Nieprawidłowe wartości danych treningowych")
    return

if val_loss is not None and val_loss <= 0:
    print("BŁĄD: Nieprawidłowa wartość straty walidacyjnej")
    return
W rzeczywistości, wartość train_loss może być bardzo mała (bliska zeru) w pewnych przypadkach, a warunek train_loss <= 0 odrzuci takie prawidłowe wartości.
3. Mechanizm aktualizacji wykresu
Widget używa timera do odświeżania, który może nie być optymalny:
python# Timer do automatycznego odświeżania
self.update_timer = QTimer()
self.update_timer.timeout.connect(self.update_plot)
self.update_timer.start(500)  # Odświeżaj co 500ms

# Flaga wskazująca czy dane zostały zaktualizowane
self.data_updated = False
4. Brak synchronizacji między wątkami
Ponieważ trening działa w osobnym wątku (BatchTrainingThread), a wizualizacja jest częścią głównego wątku GUI, może występować problem z synchronizacją danych. Widget wizualizacji nie ma bezpośredniego mechanizmu synchronizacji z wątkiem trenującym.
Proponowane zmiany w pliku training_visualization.py:
python# Poprawienie walidacji w metodzie update_data
def update_data(self, epoch, train_loss, train_acc, val_loss=None, val_acc=None):
    """Aktualizuje dane wykresu."""
    print(f"\nDEBUG update_data:")
    print(f"Epoka: {epoch}")
    print(f"Strata treningowa: {train_loss}")
    print(f"Dokładność treningowa: {train_acc}")
    print(f"Strata walidacyjna: {val_loss}")
    print(f"Dokładność walidacyjna: {val_acc}")

    # Sprawdź poprawność danych
    try:
        epoch = int(epoch)
        train_loss = float(train_loss)
        train_acc = float(train_acc)
        if val_loss is not None:
            val_loss = float(val_loss)
        if val_acc is not None:
            val_acc = float(val_acc)
    except (ValueError, TypeError) as e:
        print(f"BŁĄD konwersji danych: {e}")
        return

    # Sprawdź czy dane są sensowne - POPRAWIONA WALIDACJA
    if train_loss < 0 or train_acc < 0 or train_acc > 1:  # Zmieniamy <= na 
        print(f"BŁĄD: Nieprawidłowe wartości danych treningowych: loss={train_loss}, acc={train_acc}")
        return

    if val_loss is not None and val_loss < 0:  # Zmieniamy <= na 
        print(f"BŁĄD: Nieprawidłowa wartość straty walidacyjnej: {val_loss}")
        return

    if val_acc is not None and (val_acc < 0 or val_acc > 1):
        print(f"BŁĄD: Nieprawidłowa wartość dokładności walidacyjnej: {val_acc}")
        return

    # Dodaj nowe dane
    self.epochs.append(epoch)
    self.train_loss_data.append(train_loss)
    self.train_acc_data.append(train_acc)

    if val_loss is not None:
        self.val_loss_data.append(val_loss)
    if val_acc is not None:
        self.val_acc_data.append(val_acc)

    print("\nDEBUG: Stan danych po aktualizacji:")
    print(f"Epoki: {self.epochs}")
    print(f"Strata treningowa: {self.train_loss_data}")
    print(f"Dokładność treningowa: {self.train_acc_data}")
    print(f"Strata walidacyjna: {self.val_loss_data}")
    print(f"Dokładność walidacyjna: {self.val_acc_data}")

    # Oznacz, że dane zostały zaktualizowane i odśwież wykres
    self.data_updated = True
    self.update_plot()

    # Wymuś odświeżenie wykresu
    self.plot_widget.replot()
Poprawki w pliku batch_training_thread.py:
Należy również upewnić się, że w klasie BatchTrainingThread sygnał task_progress zwraca poprawne dane. W pliku app/core/workers/batch_training_thread.py powinniśmy dodać dodatkowe logowanie wartości przed emisją sygnału:
python# W funkcji progress_callback w metodzie train_model_optimized
def progress_callback(epoch, num_epochs, train_loss, train_acc, val_loss, val_acc):
    self.log_message_signal.emit(f"DEBUG: Emitowanie sygnału task_progress:")
    self.log_message_signal.emit(f"- epoch: {epoch}, num_epochs: {num_epochs}")
    self.log_message_signal.emit(f"- train_loss: {train_loss}, train_acc: {train_acc}")
    self.log_message_signal.emit(f"- val_loss: {val_loss}, val_acc: {val_acc}")
    
    # Sprawdź czy wartości są poprawne przed emisją sygnału
    if train_loss <= 0:
        self.log_message_signal.emit(f"UWAGA: Ujemna lub zerowa wartość straty: {train_loss}")
    
    # Emituj sygnał
    progress = int((epoch / num_epochs) * 100) if num_epochs > 0 else 0
    self.task_progress.emit(
        task_name,
        progress,
        {
            "epoch": epoch,
            "total_epochs": num_epochs,
            "train_loss": max(train_loss, 0.0001),  # Zapewnij minimalną wartość dodatnią
            "train_acc": max(min(train_acc, 1.0), 0.0),  # Ogranicz do [0,1]
            "val_loss": val_loss,
            "val_acc": val_acc,
        },
    )