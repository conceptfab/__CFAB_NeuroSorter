Proponowane rozwiązanie
Zmiany należy wprowadzić w trzech kluczowych miejscach:

W app/gui/widgets/training_visualization.py:

Całkowicie przebudować metodę update_data aby obsługiwała różne wartości danych
Dodać debugging i lepszą obsługę błędów
Zaimplementować bardziej elastyczną aktualizację wykresu


W app/gui/tabs/training_manager.py:

Poprawić metodę _training_task_progress tak, aby zapewniała bardziej niezawodne przekazywanie danych


W app/core/workers/batch_training_thread.py:

Zmodyfikować sposób generowania i wysyłania danych



Przyjrzyjmy się szczegółowo poprawkom do każdego z tych plików:
1. Zmiana w app/gui/widgets/training_visualization.py
pythondef update_data(self, epoch, train_loss, train_acc, val_loss=None, val_acc=None):
    """Aktualizuje dane wykresu."""
    try:
        # Konwersja i walidacja danych
        try:
            epoch = int(epoch)
            train_loss = float(train_loss) if train_loss is not None else None
            train_acc = float(train_acc) if train_acc is not None else None
            val_loss = float(val_loss) if val_loss is not None else None
            val_acc = float(val_acc) if val_acc is not None else None
        except (ValueError, TypeError) as e:
            print(f"BŁĄD konwersji danych: {e}")
            # Nie przerywamy, tylko ustawiamy wartości domyślne
            if train_loss is None or not isinstance(train_loss, (int, float)):
                train_loss = 1.0  # Wartość domyślna
            if train_acc is None or not isinstance(train_acc, (int, float)):
                train_acc = 0.5  # Wartość domyślna

        # Dodaj nowe dane tylko jeśli epoka jest dodatnia
        if epoch > 0:
            # Sprawdź czy ta epoka już istnieje
            if epoch in self.epochs:
                # Znajdź indeks dla tej epoki
                idx = self.epochs.index(epoch)
                # Zaktualizuj istniejące dane
                self.train_loss_data[idx] = train_loss
                self.train_acc_data[idx] = train_acc
                if val_loss is not None and idx < len(self.val_loss_data):
                    self.val_loss_data[idx] = val_loss
                elif val_loss is not None:
                    # Rozszerz listę jeśli potrzeba
                    while len(self.val_loss_data) < idx:
                        self.val_loss_data.append(None)
                    self.val_loss_data.append(val_loss)
                
                if val_acc is not None and idx < len(self.val_acc_data):
                    self.val_acc_data[idx] = val_acc
                elif val_acc is not None:
                    # Rozszerz listę jeśli potrzeba
                    while len(self.val_acc_data) < idx:
                        self.val_acc_data.append(None)
                    self.val_acc_data.append(val_acc)
            else:
                # Dodaj nowe dane na końcu list
                self.epochs.append(epoch)
                self.train_loss_data.append(train_loss)
                self.train_acc_data.append(train_acc)
                if val_loss is not None:
                    # Rozszerz listę val_loss_data jeśli potrzeba
                    while len(self.val_loss_data) < len(self.epochs) - 1:
                        self.val_loss_data.append(None)
                    self.val_loss_data.append(val_loss)
                
                if val_acc is not None:
                    # Rozszerz listę val_acc_data jeśli potrzeba
                    while len(self.val_acc_data) < len(self.epochs) - 1:
                        self.val_acc_data.append(None)
                    self.val_acc_data.append(val_acc)
        
        # Oznacz, że dane zostały zaktualizowane
        self.data_updated = True
        
        # Ręczne wywołanie update_plot
        self.update_plot()
    
    except Exception as e:
        import traceback
        print(f"Błąd w update_data: {e}")
        print(traceback.format_exc())
2. Zmiana w app/gui/tabs/training_manager.py
pythondef _training_task_progress(self, task_name, progress, details):
    """Obsługa postępu zadania treningowego."""
    try:
        # Pobierz dane z details i upewnij się, że mają prawidłowe wartości
        epoch = int(details.get("epoch", 0))
        total_epochs = int(details.get("total_epochs", 1))
        
        # Zabezpieczenie przed dzieleniem przez zero
        if total_epochs <= 0:
            total_epochs = 1
            
        # Pobierz i weryfikuj wartości loss i accuracy
        train_loss = details.get("train_loss")
        train_acc = details.get("train_acc")
        val_loss = details.get("val_loss")
        val_acc = details.get("val_acc")
        
        # Aktualizacja paska postępu
        percentage = min(100, max(0, int((epoch / total_epochs) * 100)))
        self.parent.task_progress_bar.setValue(percentage)
        
        # Aktualizacja opisu postępu
        if epoch > 0:
            loss_text = f"{train_loss:.4f}" if train_loss is not None else "N/A"
            acc_text = f"{train_acc:.2%}" if train_acc is not None else "N/A"
            details_text = f"Epoka {epoch}/{total_epochs} | Strata: {loss_text}, Dokładność: {acc_text}"
            self.parent.task_progress_details.setText(details_text)
            self.parent.logger.info(details_text)
        
        # Aktualizacja wizualizacji jeśli istnieje
        if hasattr(self, "training_visualization") and self.training_visualization:
            # Upewnij się, że epoka jest większa od zera
            if epoch > 0:
                try:
                    self.training_visualization.update_data(
                        epoch=epoch,
                        train_loss=train_loss,
                        train_acc=train_acc,
                        val_loss=val_loss,
                        val_acc=val_acc
                    )
                except Exception as vis_error:
                    self.parent.logger.error(f"Błąd aktualizacji wizualizacji: {vis_error}")
    
    except Exception as e:
        import traceback
        self.parent.logger.error(f"Błąd w _training_task_progress: {e}")
        self.parent.logger.error(traceback.format_exc())
3. Zmiana w app/core/workers/batch_training_thread.py
pythondef _run_training_task(self, task_data, task_name, task_path):
    # ... kod przed funkcją callback ...
    
    def progress_callback(epoch, num_epochs, train_loss, train_acc, val_loss, val_acc):
        """Callback do śledzenia postępu treningu."""
        try:
            # Weryfikacja danych przed emisją sygnału
            epoch = max(0, int(epoch))
            num_epochs = max(1, int(num_epochs))
            
            # Weryfikacja wartości loss i accuracy
            if train_loss is None or not isinstance(train_loss, (int, float)) or train_loss <= 0:
                train_loss = 0.01  # domyślna wartość
            
            if train_acc is None or not isinstance(train_acc, (int, float)):
                train_acc = 0.0  # domyślna wartość
            train_acc = max(0.0, min(1.0, train_acc))  # Upewnij się, że jest w zakresie [0,1]
            
            # Nie weryfikujemy val_loss i val_acc, mogą być None
            
            # Oblicz postęp treningu
            progress = int((epoch / num_epochs) * 100) if num_epochs > 0 else 0
            progress = max(0, min(100, progress))  # Upewnij się, że jest w zakresie [0,100]
            
            # Emituj sygnał z danymi
            self.task_progress.emit(
                task_name,
                progress,
                {
                    "epoch": epoch,
                    "total_epochs": num_epochs,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
            )
        except Exception as e:
            import traceback
            self.log_message_signal.emit(f"BŁĄD w callback: {str(e)}")
            self.log_message_signal.emit(traceback.format_exc())

    # ... kod po funkcji callback ...
Dodatkowe zmiany w update_plot w app/gui/widgets/training_visualization.py
pythondef update_plot(self):
    """Aktualizuje wykres na podstawie wybranej metryki i zestawu danych."""
    try:
        # Wyczyść wykres
        self.plot_widget.clear()
        
        # Sprawdź czy mamy dane do wyświetlenia
        if not self.epochs or len(self.epochs) == 0:
            return
            
        # Przygotuj dane X (epoki)
        x_data = np.array(self.epochs)
        
        # Pobierz aktualne wybory z comboboxów
        metric = self.metric_combo.currentText()
        # dataset = self.dataset_combo.currentText()  # Nieużywane w obecnej implementacji
        
        # Dodaj legendę
        self.plot_widget.addLegend()
        
        # Rysuj odpowiednie dane
        if metric == "Strata":
            # Weryfikacja danych
            if len(self.train_loss_data) > 0 and all(isinstance(v, (int, float)) for v in self.train_loss_data):
                y_data = np.array(self.train_loss_data)
                self.plot_widget.plot(
                    x_data[:len(y_data)],  # Upewnij się, że długości się zgadzają
                    y_data,
                    pen=pg.mkPen(color="b", width=2),
                    name="Strata treningowa",
                    symbol="o",
                )
                
            # Dodaj dane walidacyjne jeśli są dostępne
            if len(self.val_loss_data) > 0 and all(v is None or isinstance(v, (int, float)) for v in self.val_loss_data):
                # Odfiltruj wartości None
                x_val = []
                y_val = []
                for i, val in enumerate(self.val_loss_data):
                    if val is not None and i < len(self.epochs):
                        x_val.append(self.epochs[i])
                        y_val.append(val)
                
                if len(x_val) > 0:
                    self.plot_widget.plot(
                        np.array(x_val),
                        np.array(y_val),
                        pen=pg.mkPen(color="r", width=2),
                        name="Strata walidacyjna",
                        symbol="o",
                    )
        else:  # Dokładność
            # Weryfikacja danych
            if len(self.train_acc_data) > 0 and all(isinstance(v, (int, float)) for v in self.train_acc_data):
                y_data = np.array(self.train_acc_data)
                self.plot_widget.plot(
                    x_data[:len(y_data)],  # Upewnij się, że długości się zgadzają
                    y_data,
                    pen=pg.mkPen(color="g", width=2),
                    name="Dokładność treningowa",
                    symbol="o",
                )
                
            # Dodaj dane walidacyjne jeśli są dostępne
            if len(self.val_acc_data) > 0 and all(v is None or isinstance(v, (int, float)) for v in self.val_acc_data):
                # Odfiltruj wartości None
                x_val = []
                y_val = []
                for i, val in enumerate(self.val_acc_data):
                    if val is not None and i < len(self.epochs):
                        x_val.append(self.epochs[i])
                        y_val.append(val)
                
                if len(x_val) > 0:
                    self.plot_widget.plot(
                        np.array(x_val),
                        np.array(y_val),
                        pen=pg.mkPen(color="m", width=2),
                        name="Dokładność walidacyjna",
                        symbol="o",
                    )
        
        # Dostosuj widok do danych
        self.plot_widget.autoRange()
        
        # Resetuj flagę aktualizacji
        self.data_updated = False
    
    except Exception as e:
        import traceback
        print(f"Błąd w update_plot: {e}")
        print(traceback.format_exc())
Powyższe zmiany powinny znacznie poprawić niezawodność systemu przez:

Dodanie szczegółowej walidacji danych na każdym etapie
Lepszą obsługę błędów z dokładnym loggingiem
Bardziej elastyczne zarządzanie listami danych
Lepszą weryfikację przed rysowaniem wykresu

Te zmiany są znacznie bardziej kompleksowe niż poprzednio zaproponowane rozwiązanie i powinny rozwiązać problem związany z przekazywaniem danych z wątku treningowego do UI.