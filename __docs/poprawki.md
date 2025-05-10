Zmiana w pliku app/gui/widgets/training_visualization.py, w metodzie save_plot
pythondef save_plot(self, filename):
    """Zapisuje wykres do pliku PNG.

    Args:
        filename (str): Ścieżka do pliku, w którym zostanie zapisany wykres.
    """
    try:
        # Upewnij się, że wykres jest aktualny
        if self.data_updated:
            self.update_plot()

        # Zapamiętaj oryginalny rozmiar
        original_width = self.plot_widget.width()
        original_height = self.plot_widget.height()

        # Zapisz wykres w formacie PNG z minimalną szerokością 3000px
        self.plot_widget.setFixedWidth(3000)  # Ustaw minimalną szerokość
        export_image = self.plot_widget.grab()
        export_image.save(filename)

        # Przywróć oryginalny rozmiar
        self.plot_widget.setFixedWidth(-1)  # Przywróć automatyczny rozmiar
        
        # Upewnij się, że wykres jest nadal widoczny
        self.update_plot()

        return True
    except Exception as e:
        print(f"Błąd podczas zapisywania wykresu: {e}")
        import traceback

        print(traceback.format_exc())
        return False
Główne zmiany obejmują:

Zapamiętanie oryginalnego rozmiaru widgetu przed zmianą
Tworzenie oddzielnej zmiennej dla zrzutu ekranu (export_image) zamiast bezpośredniego zapisywania
Dodanie jawnego wywołania update_plot() po przywróceniu rozmiaru, aby upewnić się, że wykres zostanie ponownie narysowany

Problem prawdopodobnie wynikał z tego, że po zmianie stałego rozmiaru widgetu i przywróceniu go do wartości -1, nie następowało ponowne rysowanie wykresu. Dodatkowe wywołanie update_plot() powinno rozwiązać ten problem.