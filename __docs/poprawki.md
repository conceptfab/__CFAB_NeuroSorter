2. Modyfikacja funkcji verify_directory_structure
Jeśli masz dostęp do kodu źródłowego i możesz go zmodyfikować, możesz zmodyfikować funkcję verify_directory_structure w pliku ai/fine_tuning.py, aby dodać parametr allow_empty:
pythondef verify_directory_structure(directory, allow_empty=False):
    """
    Sprawdza czy struktura katalogów jest płaska (kategoria/obrazy).

    Args:
        directory: Ścieżka do katalogu z danymi
        allow_empty: Czy zezwalać na puste katalogi kategorii

    Returns:
        bool: True jeśli struktura jest poprawna, False w przeciwnym razie
    """
    for root, dirs, files in os.walk(directory):
        # Pomijamy główny katalog
        if root == directory:
            continue

        # Sprawdzamy czy są podkatalogi
        if dirs:
            return False

        # Sprawdzamy czy są pliki obrazów
        has_images = any(
            f.lower().endswith((".jpg", ".webp", ".jpeg", ".png", ".bmp")) for f in files
        )
        if not has_images and not allow_empty:
            return False

    return True