#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import shutil
import argparse
from pathlib import Path


def print_info(message):
    """Wypisuje informację z formatowaniem."""
    print(f"\033[94m[INFO]\033[0m {message}")


def print_success(message):
    """Wypisuje komunikat sukcesu z formatowaniem."""
    print(f"\033[92m[SUKCES]\033[0m {message}")


def print_warning(message):
    """Wypisuje ostrzeżenie z formatowaniem."""
    print(f"\033[93m[OSTRZEŻENIE]\033[0m {message}")


def print_error(message):
    """Wypisuje błąd z formatowaniem."""
    print(f"\033[91m[BŁĄD]\033[0m {message}")


def check_python_version():
    """Sprawdza wersję Pythona."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_error(
            f"Wymagana jest wersja Python 3.8 lub nowsza. Twoja wersja: {version.major}.{version.minor}"
        )
        return False
    return True


def create_virtual_env(env_name):
    """Tworzy wirtualne środowisko."""
    print_info(f"Tworzenie wirtualnego środowiska '{env_name}'...")
    try:
        subprocess.run([sys.executable, "-m", "venv", env_name], check=True)
        print_success(f"Utworzono wirtualne środowisko: {env_name}")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Nie udało się utworzyć wirtualnego środowiska: {e}")
        return False


def get_activation_script(env_name):
    """Zwraca ścieżkę do skryptu aktywacyjnego w zależności od systemu."""
    if os.name == "nt":  # Windows
        return os.path.join(env_name, "Scripts", "activate.bat")
    else:  # Linux/Mac
        return os.path.join(env_name, "bin", "activate")


def download_packages(requirements_file, output_dir):
    """Pobiera pakiety na podstawie pliku requirements.txt."""
    os.makedirs(output_dir, exist_ok=True)
    print_info(f"Pobieranie pakietów z {requirements_file} do {output_dir}...")

    try:
        # Użyj pip do pobrania pakietów
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "download",
                "-r",
                requirements_file,
                "-d",
                output_dir,
            ],
            check=True,
        )
        print_success("Pakiety zostały pobrane pomyślnie.")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Błąd podczas pobierania pakietów: {e}")
        return False


def install_packages_offline(requirements_file, packages_dir, env_name):
    """Instaluje pakiety w trybie offline."""
    print_info(f"Instalacja pakietów w trybie offline z {packages_dir}...")

    activation_script = get_activation_script(env_name)

    if os.name == "nt":  # Windows
        # W Windows używamy cmd /c do uruchomienia komendy aktywacyjnej
        install_cmd = f'cmd /c "{activation_script} && pip install --no-index --find-links={packages_dir} -r {requirements_file}"'
    else:  # Linux/Mac
        # W Linux/Mac używamy source do aktywacji środowiska
        install_cmd = f'source "{activation_script}" && pip install --no-index --find-links={packages_dir} -r {requirements_file}'

    try:
        # Wykonaj komendę w powłoce systemowej
        if os.name == "nt":  # Windows
            process = subprocess.run(install_cmd, shell=True, check=True)
        else:  # Linux/Mac
            process = subprocess.run(
                install_cmd, shell=True, executable="/bin/bash", check=True
            )

        print_success("Pakiety zostały zainstalowane pomyślnie.")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Błąd podczas instalacji pakietów: {e}")
        return False


def create_script_files(env_name):
    """Tworzy pliki skryptów do aktywacji środowiska i uruchomienia aplikacji."""
    if os.name == "nt":  # Windows
        with open("uruchom.bat", "w") as f:
            f.write(f"@echo off\n")
            f.write(f"call {env_name}\\Scripts\\activate.bat\n")
            f.write(f"python cfabNS.py\n")
            f.write(f"pause\n")
        print_success("Utworzono plik 'uruchom.bat'")
    else:  # Linux/Mac
        with open("uruchom.sh", "w") as f:
            f.write(f"#!/bin/bash\n")
            f.write(f"source {env_name}/bin/activate\n")
            f.write(f"python cfabNS.py\n")
        # Nadaj uprawnienia do wykonania
        os.chmod("uruchom.sh", 0o755)
        print_success("Utworzono plik 'uruchom.sh'")


def prepare_online_mode(args):
    """Tryb przygotowania pakietów online."""
    if not check_python_version():
        return

    # Pobieranie pakietów
    if download_packages(args.requirements, args.packages_dir):
        print_success(
            f"Wszystkie pakiety zostały pobrane do katalogu '{args.packages_dir}'"
        )
        print_info(
            f"Skopiuj cały projekt wraz z katalogiem '{args.packages_dir}' na komputer docelowy i uruchom:"
        )
        print_info(
            f"python przygotuj_offline.py --offline --env={args.env} --packages-dir={args.packages_dir}"
        )


def prepare_offline_mode(args):
    """Tryb instalacji pakietów offline."""
    if not check_python_version():
        return

    # Sprawdź czy katalog z pakietami istnieje
    if not os.path.exists(args.packages_dir):
        print_error(f"Katalog z pakietami '{args.packages_dir}' nie istnieje!")
        return

    # Sprawdź czy plik requirements.txt istnieje
    if not os.path.exists(args.requirements):
        print_error(f"Plik '{args.requirements}' nie istnieje!")
        return

    # Utwórz wirtualne środowisko
    if create_virtual_env(args.env):
        # Zainstaluj pakiety offline
        if install_packages_offline(args.requirements, args.packages_dir, args.env):
            # Utwórz pliki skryptów
            create_script_files(args.env)
            print_success(
                f"Gotowe! Możesz uruchomić aplikację używając pliku 'uruchom.bat' (Windows) lub './uruchom.sh' (Linux/Mac)"
            )


def main():
    """Główna funkcja skryptu."""
    parser = argparse.ArgumentParser(
        description="Przygotowanie środowiska dla CFAB NeuroSorter"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Tryb instalacji offline (na komputerze bez internetu)",
    )
    parser.add_argument(
        "--env", default="venv", help="Nazwa wirtualnego środowiska (domyślnie: venv)"
    )
    parser.add_argument(
        "--requirements",
        default="requirements.txt",
        help="Ścieżka do pliku requirements.txt (domyślnie: requirements.txt)",
    )
    parser.add_argument(
        "--packages-dir",
        default="wheelhouse",
        help="Ścieżka do katalogu z pakietami (domyślnie: wheelhouse)",
    )

    args = parser.parse_args()

    print("\n=== CFAB NeuroSorter - Przygotowanie środowiska ===\n")

    if args.offline:
        prepare_offline_mode(args)
    else:
        prepare_online_mode(args)


if __name__ == "__main__":
    main()
