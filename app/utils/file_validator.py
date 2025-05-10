import os
from typing import Dict, List, Tuple


def check_directory_files(directory_path: str) -> Dict[str, List[str]]:
    """
    Sprawdza zawartość katalogu i zwraca informacje o plikach.

    Args:
        directory_path: Ścieżka do katalogu do sprawdzenia

    Returns:
        Dict zawierający:
        - 'valid_images': lista plików z poprawnymi rozszerzeniami
        - 'invalid_files': lista plików z niepoprawnymi rozszerzeniami
        - 'total_files': całkowita liczba plików
        - 'valid_extensions': lista poprawnych rozszerzeń
        - 'directory_tree': drzewo katalogów z liczbą plików
    """
    if not os.path.exists(directory_path):
        return {
            "error": f"Katalog {directory_path} nie istnieje",
            "valid_images": [],
            "invalid_files": [],
            "total_files": 0,
            "valid_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".webp"],
            "directory_tree": {},
        }

    if not os.path.isdir(directory_path):
        return {
            "error": f"{directory_path} nie jest katalogiem",
            "valid_images": [],
            "invalid_files": [],
            "total_files": 0,
            "valid_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".webp"],
            "directory_tree": {},
        }

    valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    valid_images = []
    invalid_files = []
    directory_tree = {}
    total_files = 0

    try:
        for root, dirs, files in os.walk(directory_path):
            relative_path = os.path.relpath(root, directory_path)
            if relative_path == ".":
                relative_path = ""

            valid_count = 0
            invalid_count = 0

            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in valid_extensions:
                    valid_images.append(file_path)
                    valid_count += 1
                else:
                    invalid_files.append(file_path)
                    invalid_count += 1

            total_files += len(files)

            # Dodaj informacje o katalogu do drzewa
            if relative_path not in directory_tree:
                directory_tree[relative_path] = {
                    "total_files": len(files),
                    "valid_files": valid_count,
                    "invalid_files": invalid_count,
                    "subdirectories": [],
                }

            # Dodaj informacje o podkatalogach
            for dir_name in dirs:
                dir_path = os.path.join(relative_path, dir_name)
                if relative_path:
                    dir_path = os.path.join(relative_path, dir_name)
                else:
                    dir_path = dir_name
                directory_tree[relative_path]["subdirectories"].append(dir_path)

        return {
            "valid_images": valid_images,
            "invalid_files": invalid_files,
            "total_files": total_files,
            "valid_extensions": valid_extensions,
            "directory_tree": directory_tree,
        }
    except Exception as e:
        return {
            "error": f"Błąd podczas sprawdzania katalogu: {str(e)}",
            "valid_images": [],
            "invalid_files": [],
            "total_files": 0,
            "valid_extensions": valid_extensions,
            "directory_tree": {},
        }


def print_directory_report(directory_path: str) -> None:
    """
    Wyświetla raport o zawartości katalogu.

    Args:
        directory_path: Ścieżka do katalogu do sprawdzenia
    """
    print(f"\n=== RAPORT KATALOGU: {directory_path} ===")

    result = check_directory_files(directory_path)

    if "error" in result:
        print(f"❌ {result['error']}")
        return

    print(f"\n📊 Statystyki:")
    print(f"   Łączna liczba plików: {result['total_files']}")
    print(f"   Poprawne pliki obrazów: {len(result['valid_images'])}")
    print(f"   Niepoprawne pliki: {len(result['invalid_files'])}")

    print(f"\n📁 Poprawne rozszerzenia: {', '.join(result['valid_extensions'])}")

    print("\n🌳 Struktura katalogów:")

    def print_tree(tree, prefix=""):
        for dir_path, info in tree.items():
            if dir_path == "":
                dir_name = os.path.basename(directory_path)
            else:
                dir_name = os.path.basename(dir_path)

            print(f"{prefix}├── {dir_name}/")
            print(f"{prefix}│   ├── Pliki: {info['total_files']}")
            print(f"{prefix}│   ├── Poprawne: {info['valid_files']}")
            print(f"{prefix}│   └── Niepoprawne: {info['invalid_files']}")

            for subdir in info["subdirectories"]:
                if subdir in tree:
                    print_tree({subdir: tree[subdir]}, prefix + "│   ")

    print_tree(result["directory_tree"])

    if result["valid_images"]:
        print("\n✅ Przykładowe poprawne pliki obrazów:")
        for img in result["valid_images"][:5]:
            print(f"   - {img}")
        if len(result["valid_images"]) > 5:
            print(f"   ... i {len(result['valid_images']) - 5} więcej")

    if result["invalid_files"]:
        print("\n❌ Przykładowe niepoprawne pliki:")
        for file in result["invalid_files"][:5]:
            print(f"   - {file}")
        if len(result["invalid_files"]) > 5:
            print(f"   ... i {len(result['invalid_files']) - 5} więcej")

    print("\n===========================")


def validate_training_structure(
    train_dir: str, val_dir: str = None
) -> Tuple[bool, str]:
    """
    Sprawdza strukturę katalogów treningowych i walidacyjnych przed rozpoczęciem treningu.

    Args:
        train_dir: Ścieżka do katalogu treningowego
        val_dir: Ścieżka do katalogu walidacyjnego (opcjonalnie)

    Returns:
        Tuple[bool, str]: (czy struktura jest poprawna, komunikat błędu)
    """
    print("\n=== WALIDACJA STRUKTURY KATALOGÓW ===")

    # Sprawdź katalog treningowy
    print(f"\n📁 Katalog treningowy: {train_dir}")
    train_result = check_directory_files(train_dir)

    if "error" in train_result:
        return False, f"Błąd w katalogu treningowym: {train_result['error']}"

    if not train_result["valid_images"]:
        # Usuń niepoprawne pliki
        removed_count, removed_files = remove_invalid_files(train_dir)
        if removed_count > 0:
            print(f"\nUsunięto {removed_count} niepoprawnych plików:")
            for file in removed_files:
                print(f"  - {file}")
            
            # Sprawdź ponownie po usunięciu
            train_result = check_directory_files(train_dir)
            if not train_result["valid_images"]:
                return False, (
                    f"Brak poprawnych plików obrazów w katalogu treningowym. "
                    f"Znaleziono {train_result['total_files']} plików, "
                    f"ale żaden nie ma poprawnego rozszerzenia "
                    f"({', '.join(train_result['valid_extensions'])})"
                )
        else:
            return False, (
                f"Brak poprawnych plików obrazów w katalogu treningowym. "
                f"Znaleziono {train_result['total_files']} plików, "
                f"ale żaden nie ma poprawnego rozszerzenia "
                f"({', '.join(train_result['valid_extensions'])})"
            )

    # Sprawdź katalog walidacyjny jeśli istnieje
    if val_dir:
        print(f"\n📁 Katalog walidacyjny: {val_dir}")
        val_result = check_directory_files(val_dir)

        if "error" in val_result:
            return False, f"Błąd w katalogu walidacyjnym: {val_result['error']}"

        if not val_result["valid_images"]:
            # Usuń niepoprawne pliki
            removed_count, removed_files = remove_invalid_files(val_dir)
            if removed_count > 0:
                print(f"\nUsunięto {removed_count} niepoprawnych plików:")
                for file in removed_files:
                    print(f"  - {file}")
                
                # Sprawdź ponownie po usunięciu
                val_result = check_directory_files(val_dir)
                if not val_result["valid_images"]:
                    return False, (
                        f"Brak poprawnych plików obrazów w katalogu walidacyjnym. "
                        f"Znaleziono {val_result['total_files']} plików, "
                        f"ale żaden nie ma poprawnego rozszerzenia "
                        f"({', '.join(val_result['valid_extensions'])})"
                    )
            else:
                return False, (
                    f"Brak poprawnych plików obrazów w katalogu walidacyjnym. "
                    f"Znaleziono {val_result['total_files']} plików, "
                    f"ale żaden nie ma poprawnego rozszerzenia "
                    f"({', '.join(val_result['valid_extensions'])})"
                )

        # Porównaj strukturę katalogów
        print("\n🔍 Porównanie struktury katalogów:")
        
        # Pobierz listy kategorii z obu katalogów
        train_categories = set(train_result['directory_tree'].keys())
        val_categories = set(val_result['directory_tree'].keys())
        
        # Sprawdź czy kategorie są takie same
        if train_categories != val_categories:
            missing_in_val = train_categories - val_categories
            missing_in_train = val_categories - train_categories
            
            error_msg = "Różne struktury katalogów:\n"
            if missing_in_val:
                error_msg += f"Kategorie brakujące w walidacji: {', '.join(missing_in_val)}\n"
            if missing_in_train:
                error_msg += f"Kategorie brakujące w treningu: {', '.join(missing_in_train)}"
            return False, error_msg
        
        # Porównaj liczbę plików w każdej kategorii
        for category in train_categories:
            train_info = train_result['directory_tree'][category]
            val_info = val_result['directory_tree'][category]
            
            if train_info['total_files'] != val_info['total_files']:
                return False, (
                    f"Różna liczba plików w kategorii {category}: "
                    f"trening: {train_info['total_files']}, "
                    f"walidacja: {val_info['total_files']}"
                )

    print("\n✅ Struktura katalogów jest poprawna!")
    return True, ""


def print_validation_report(train_dir: str, val_dir: str = None) -> None:
    """
    Wyświetla szczegółowy raport walidacji struktury katalogów.

    Args:
        train_dir: Ścieżka do katalogu treningowego
        val_dir: Ścieżka do katalogu walidacyjnego (opcjonalnie)
    """
    print("\n=== RAPORT WALIDACJI STRUKTURY KATALOGÓW ===")

    # Sprawdź katalog treningowy
    print(f"\n📁 Katalog treningowy: {train_dir}")
    print_directory_report(train_dir)

    # Sprawdź katalog walidacyjny jeśli istnieje
    if val_dir:
        print(f"\n📁 Katalog walidacyjny: {val_dir}")
        print_directory_report(val_dir)

    # Sprawdź całą strukturę
    is_valid, error_msg = validate_training_structure(train_dir, val_dir)

    if is_valid:
        print("\n✅ Struktura katalogów jest poprawna!")
    else:
        print(f"\n❌ Błąd walidacji: {error_msg}")

    print("\n===========================")


def remove_invalid_files(directory: str) -> Tuple[int, List[str]]:
    """
    Usuwa pliki z niepoprawnymi rozszerzeniami z katalogu.

    Args:
        directory: Ścieżka do katalogu

    Returns:
        Tuple[int, List[str]]: (liczba usuniętych plików, lista usuniętych plików)
    """
    valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    removed_files = []

    print(f"\nSprawdzanie katalogu: {directory}")
    if not os.path.exists(directory):
        print(f"❌ Katalog nie istnieje: {directory}")
        return 0, []

    if not os.path.isdir(directory):
        print(f"❌ Ścieżka nie jest katalogiem: {directory}")
        return 0, []

    try:
        for root, _, files in os.walk(directory):
            print(f"\nSprawdzanie katalogu: {root}")
            print(f"Znalezione pliki: {len(files)}")

            for file in files:
                file_path = os.path.join(root, file)
                print(f"Sprawdzanie pliku: {file}")

                if not any(file.lower().endswith(ext) for ext in valid_extensions):
                    print(f"Próba usunięcia pliku: {file_path}")
                    try:
                        os.remove(file_path)
                        removed_files.append(file_path)
                        print(f"✅ Usunięto plik: {file_path}")
                    except Exception as e:
                        print(f"❌ Nie można usunąć pliku {file_path}: {str(e)}")
                else:
                    print(f"✅ Plik ma poprawne rozszerzenie: {file}")

    except Exception as e:
        print(f"❌ Błąd podczas przetwarzania katalogu: {str(e)}")

    print(f"\nPodsumowanie:")
    print(f"Usunięto plików: {len(removed_files)}")
    if removed_files:
        print("Usunięte pliki:")
        for file in removed_files:
            print(f"  - {file}")

    return len(removed_files), removed_files
