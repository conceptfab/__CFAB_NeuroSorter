Analiza błędu dotyczącego katalogów danych treningowych
Na podstawie przedstawionego błędu i kodu źródłowego, mogę wskazać przyczynę problemu oraz zaproponować rozwiązanie. Problem dotyczy funkcji fine_tune_model() w pliku ai/fine_tuning.py, która nie może znaleźć katalogów klas w folderze treningowym.
Przyczyna błędu
Błąd: FileNotFoundError: Couldn't find any class folder in F:/__Set_gardener/__dane_treningowe.
Problem wynika z tego, że PyTorch oczekuje struktury katalogów, gdzie wewnątrz katalogu treningowego (train_dir) znajdują się foldery reprezentujące poszczególne klasy, a w nich obrazy. Funkcja datasets.ImageFolder() nie może znaleźć żadnego katalogu klasy w podanej ścieżce.
Proponowane zmiany
Zmiana w pliku ai/fine_tuning.py
Należy dodać funkcję pomocniczą, która sprawdzi strukturę katalogów i w razie potrzeby utworzy odpowiednią strukturę:
pythondef ensure_class_folder_structure(directory):
    """
    Sprawdza i naprawia strukturę katalogów dla treningu.
    Jeśli w katalogu są bezpośrednio pliki obrazów (bez podkatalogów),
    tworzy podkatalog 'default_class' i przenosi tam wszystkie obrazy.

    Args:
        directory: Ścieżka do katalogu z danymi

    Returns:
        bool: True jeśli struktura była poprawna lub została naprawiona, False w przypadku błędu
    """
    import os
    import shutil

    # Sprawdź, czy istnieją jakiekolwiek podkatalogi
    has_subdirs = False
    has_images = False
    
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            has_subdirs = True
        elif os.path.isfile(item_path) and item.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            has_images = True
    
    # Jeśli nie ma podkatalogów, ale są obrazy, utwórz katalog 'default_class'
    if not has_subdirs and has_images:
        print(f"Wykryto obrazy bez struktury klas. Tworzenie katalogu 'default_class'...")
        default_class_dir = os.path.join(directory, "default_class")
        
        try:
            # Utwórz katalog 'default_class'
            os.makedirs(default_class_dir, exist_ok=True)
            
            # Przenieś wszystkie pliki obrazów do tego katalogu
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                if os.path.isfile(item_path) and item.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    shutil.move(item_path, os.path.join(default_class_dir, item))
            
            print(f"Utworzono domyślną klasę i przeniesiono do niej obrazy.")
            return True
        except Exception as e:
            print(f"Błąd podczas tworzenia struktury katalogów: {e}")
            return False
    
    return True
Następnie należy zmodyfikować funkcję fine_tune_model(), dodając wywołanie tej funkcji przed ładowaniem danych:
python# W funkcji fine_tune_model, przed ładowaniem danych:
print("\nSprawdzanie struktury katalogów...")
if not ensure_class_folder_structure(train_dir):
    raise ValueError(f"Nie udało się przygotować katalogu {train_dir} do treningu.")

if val_dir and not ensure_class_folder_structure(val_dir):
    raise ValueError(f"Nie udało się przygotować katalogu {val_dir} do treningu.")
To wywołanie należy umieścić przed linią:
python# 8. Załaduj dane
print("\nŁadowanie danych...")
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
Pełna poprawka
Zmiana w pliku ai/fine_tuning.py:

Dodaj nową funkcję ensure_class_folder_structure po istniejącej funkcji verify_directory_structure
Zmodyfikuj funkcję fine_tune_model dodając wywołanie tej funkcji przed ładowaniem danych

Ta zmiana automatycznie wykryje i naprawi brak struktury katalogów, tworząc klasę domyślną, jeśli obrazy znajdują się bezpośrednio w głównym katalogu zamiast w 