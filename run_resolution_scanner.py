import os
import sys

# Dodaj główny katalog projektu do ścieżki Pythona
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from app.utils.test_resolution_scanner import main

if __name__ == "__main__":
    main()
