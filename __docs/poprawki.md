Oto proponowana zmiana:
W pliku tools/data_splitter_gui.py, w klasie ScallerWorker, metoda run() zawiera logikę skalowania obrazu. Aktualna implementacja sprawdza, który bok jest dłuższy, a następnie ustawia ten dłuższy bok na wartość docelową, proporcjonalnie zmniejszając drugi wymiar:
pythonif width > height:
    new_width = self.target_dimension  # MODIFIED
    new_height = int(height * (self.target_dimension / width))  # MODIFIED
else:
    new_height = self.target_dimension  # MODIFIED
    new_width = int(width * (self.target_dimension / height))  # MODIFIED
Proponuję zmianę tej logiki, aby ustawiała najkrótszy bok na wartość docelową:
pythonif width < height:  # Zmiana z width > height na width < height
    new_width = self.target_dimension  # Ustawiamy krótszy wymiar
    new_height = int(height * (self.target_dimension / width))
else:
    new_height = self.target_dimension  # Ustawiamy krótszy wymiar
    new_width = int(width * (self.target_dimension / height))
Należy również zaktualizować komunikaty w interfejsie użytkownika, które informują użytkownika o tym, jak działa skalowanie. Zmiana w opisie funkcji powinna być wprowadzona w następujących miejscach:

W pliku tools/data_splitter_gui.py, w klasie ScallerApp, metoda _init_ui() zawiera opis w polu:

pythontarget_dim_info_label = QLabel(
    "Mniejsze pliki (których oba wymiary są mniejsze lub równe wartości docelowej) zostaną pominięte."
)
Proponuję zmienić ten tekst na:
pythontarget_dim_info_label = QLabel(
    "Krótszy bok obrazu zostanie przeskalowany do wartości docelowej. Pliki, których oba wymiary są mniejsze lub równe wartości docelowej, zostaną pominięte."
)

Podobnie w komunikatach logowania, gdzie obecnie pojawia się opis skalowania dłuższego boku, należy zaktualizować komunikaty, by odzwierciedlały nową logikę.