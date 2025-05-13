import json
import logging
import os
import platform
import sqlite3
import subprocess
import time
import uuid

import numpy as np
import psutil
import torch

from app.utils.config import DEFAULT_TRAINING_PARAMS

# Konfiguracja loggera dla tego modułu
logger = logging.getLogger(__name__)


class HardwareProfiler:
    """Klasa do profilowania sprzętu i optymalizacji ustawień."""

    def __init__(self, db_path="data/database.sqlite"):
        """Inicjalizacja profilera."""
        logger.info("Inicjalizacja HardwareProfiler...")
        self.db_path = db_path
        self._create_tables()
        self.machine_id = None  # Inicjalizacja na None

        # Próba wygenerowania machine_id
        try:
            self.machine_id = self._generate_machine_id()
            if not self.machine_id:
                logger.warning(
                    "Nie udało się wygenerować machine_id, używam tymczasowego..."
                )
                self.machine_id = str(uuid.uuid4())
            logger.info("Inicjalizacja zakończona. Machine ID: %s", self.machine_id)
        except Exception as e:
            logger.error("Błąd podczas generowania machine_id: %s", e, exc_info=True)
            # Generuj tymczasowy ID w przypadku błędu
            self.machine_id = str(uuid.uuid4())
            logger.warning("Użyto tymczasowego machine_id: %s", self.machine_id)

    def _create_tables(self):
        """Tworzy lub aktualizuje tabelę profili w bazie danych."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Najpierw stwórz tabelę, jeśli nie istnieje (bez nowych kolumn)
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS hardware_profiles (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        machine_id TEXT NOT NULL UNIQUE,
                        cpu_info TEXT NOT NULL,
                        ram_total REAL NOT NULL,
                        gpu_info TEXT,
                        gpu_memory REAL,
                        cpu_score REAL NOT NULL,
                        gpu_score REAL,
                        overall_score REAL NOT NULL,
                        recommended_batch_size INTEGER NOT NULL,
                        recommended_workers INTEGER NOT NULL,
                        use_mixed_precision BOOLEAN NOT NULL,
                        profile_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        learning_rate REAL,
                        optimizer TEXT,
                        scheduler TEXT,
                        weight_decay REAL,
                        gradient_clip_val REAL,
                        early_stopping_patience INTEGER,
                        max_epochs INTEGER,
                        validation_split REAL,
                        training_params TEXT
                    )
                """
                )

                # Sprawdź i dodaj brakujące kolumny
                cursor.execute("PRAGMA table_info(hardware_profiles)")
                columns = [column[1] for column in cursor.fetchall()]

                if "ram_info" not in columns:
                    logger.info("Aktualizacja tabeli: dodawanie 'ram_info'...")
                    cursor.execute(
                        "ALTER TABLE hardware_profiles ADD COLUMN ram_info TEXT"
                    )

                if "additional_recommendations" not in columns:
                    logger.info(
                        "Aktualizacja tabeli: dodawanie 'additional_recommendations'..."
                    )
                    cursor.execute(
                        "ALTER TABLE hardware_profiles ADD COLUMN additional_recommendations TEXT"
                    )

                conn.commit()
                logger.info("Tabela 'hardware_profiles' jest aktualna.")

        except sqlite3.Error as e:
            logger.error("Błąd SQLite (create/alter table): %s", e, exc_info=True)
        except Exception as e:
            logger.error(
                "Nieoczekiwany błąd DB (create/alter table): %s", e, exc_info=True
            )

    def _generate_machine_id(self):
        """Generuje unikalny identyfikator komputera."""
        logger.info("Generowanie machine_id...")
        try:
            # Zbierz podstawowe informacje o systemie
            system_info = {}

            # Dodaj informacje o procesorze
            try:
                system_info["processor"] = platform.processor()
            except Exception as e:
                logger.warning("Nie udało się pobrać informacji o procesorze: %s", e)
                system_info["processor"] = "unknown"

            # Dodaj informacje o maszynie
            try:
                system_info["machine"] = platform.machine()
            except Exception as e:
                logger.warning("Nie udało się pobrać informacji o maszynie: %s", e)
                system_info["machine"] = "unknown"

            # Dodaj informacje o węźle
            try:
                system_info["node"] = platform.node()
            except Exception as e:
                logger.warning("Nie udało się pobrać informacji o węźle: %s", e)
                system_info["node"] = "unknown"

            # Dodaj informacje o dysku
            try:
                system_info["disk_serial"] = self._get_disk_serial()
            except Exception as e:
                logger.warning("Nie udało się pobrać informacji o dysku: %s", e)
                system_info["disk_serial"] = "unknown"

            # Dodaj informacje o CPU
            try:
                system_info["cpu_count"] = os.cpu_count()
            except Exception as e:
                logger.warning("Nie udało się pobrać informacji o CPU: %s", e)
                system_info["cpu_count"] = 0

            # Dodaj informacje o RAM
            try:
                system_info["total_ram"] = psutil.virtual_memory().total
            except Exception as e:
                logger.warning("Nie udało się pobrać informacji o RAM: %s", e)
                system_info["total_ram"] = 0

            # Sprawdź czy mamy wystarczająco informacji
            if all(v == "unknown" or v == 0 for v in system_info.values()):
                logger.warning(
                    "Nie udało się zebrać wystarczających informacji o systemie"
                )
                return str(uuid.uuid4())

            # Tworzenie hasha na podstawie informacji o sprzęcie
            machine_str = json.dumps(system_info, sort_keys=True)
            machine_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, machine_str))
            logger.info("Wygenerowano machine_id: %s", machine_id)
            return machine_id
        except Exception as e:
            logger.error("Błąd podczas generowania machine_id: %s", e, exc_info=True)
            # Generuj tymczasowy ID w przypadku błędu
            temp_id = str(uuid.uuid4())
            logger.warning("Użyto tymczasowego machine_id: %s", temp_id)
            return temp_id

    def _get_disk_serial(self):
        """Próbuje odczytać numer seryjny dysku."""
        try:
            if platform.system() == "Windows":
                output = subprocess.check_output(
                    "wmic diskdrive get SerialNumber", shell=True
                )
                return output.decode().strip().split("\n")[1]
            elif platform.system() == "Linux":
                # Zwykle /etc/machine-id jest bardziej unikalne niż serial dysku
                with open("/etc/machine-id", "r", encoding="utf-8") as f:
                    return f.read().strip()
            else:
                logger.warning("Nieznany system, nie można pobrać serialu dysku.")
                return ""
        except FileNotFoundError:
            logger.warning("Nie znaleziono /etc/machine-id (Linux).", exc_info=True)
            return ""
        except subprocess.CalledProcessError as e:
            logger.warning(
                "Błąd wmic podczas pobierania serialu dysku (Windows): %s",
                e,
                exc_info=True,
            )
            return ""
        except Exception as e:
            logger.error(
                "Nieoczekiwany błąd podczas pobierania serialu dysku: %s",
                e,
                exc_info=True,
            )
            return ""

    def run_profile(self):
        """Profiluje sprzęt i generuje zalecane ustawienia."""
        logger.info("=== ROZPOCZYNAM PROCES PROFILOWANIA SPRZĘTU ===")
        logger.info("Czas rozpoczęcia: %s", time.strftime("%Y-%m-%d %H:%M:%S"))

        # Sprawdź i wygeneruj machine_id jeśli brakuje
        if not hasattr(self, "machine_id") or not self.machine_id:
            logger.warning("Brak machine_id, generuję nowy...")
            try:
                self.machine_id = self._generate_machine_id()
                if not self.machine_id:
                    logger.warning(
                        "Nie udało się wygenerować machine_id, używam tymczasowego..."
                    )
                    self.machine_id = str(uuid.uuid4())
                logger.info("Wygenerowano nowy machine_id: %s", self.machine_id)
            except Exception as e:
                logger.error(
                    "Błąd podczas generowania machine_id: %s", e, exc_info=True
                )
                self.machine_id = str(uuid.uuid4())
                logger.warning("Użyto tymczasowego machine_id: %s", self.machine_id)

        if not self.machine_id:
            logger.error("Nie udało się uzyskać machine_id, przerywam profilowanie")
            raise ValueError("Nie udało się uzyskać machine_id")

        logger.info("Aktualny machine_id: %s", self.machine_id)

        # 1. Podstawowe informacje o systemie
        logger.info("=== ETAP 1: ZBIERANIE INFORMACJI O SYSTEMIE ===")
        cpu_freq = psutil.cpu_freq()
        cpu_info = {
            "processor": platform.processor(),
            "name": self._get_cpu_name(),
            "cores_physical": psutil.cpu_count(logical=False),
            "cores_logical": psutil.cpu_count(logical=True),
            "frequency": cpu_freq.max if cpu_freq else "Nieznana",
        }
        logger.info("Informacje o CPU: %s", json.dumps(cpu_info, indent=2))

        # Dodany dokładniejszy odczyt pamięci RAM
        logger.info("=== ETAP 2: ANALIZA PAMIĘCI RAM ===")
        vmem = psutil.virtual_memory()
        ram_info = {
            "total_gb": vmem.total / (1024**3),  # GB
            "available_gb": vmem.available / (1024**3),  # GB
            "frequency": self._get_ram_frequency(),  # Dodana funkcja
        }
        ram_total = ram_info["total_gb"]
        logger.info("Informacje o RAM: %s", json.dumps(ram_info, indent=2))

        # 2. Dokładniejsze informacje o GPU
        logger.info("=== ETAP 3: ANALIZA KARTY GRAFICZNEJ ===")
        gpu_info = {}
        gpu_memory = 0
        has_cuda = torch.cuda.is_available()
        logger.info("CUDA dostępne: %s", has_cuda)

        if has_cuda:
            gpu_info = {
                "name": torch.cuda.get_device_name(0),
                "count": torch.cuda.device_count(),
                "cuda_version": torch.version.cuda,
                "compute_capability": self._get_cuda_compute_capability(),
            }
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)

            gpu_info["vram_gb"] = gpu_memory
            gpu_info["memory_bandwidth"] = self._estimate_gpu_memory_bandwidth()
            logger.info("Informacje o GPU: %s", json.dumps(gpu_info, indent=2))

        # 3. Test wydajności CPU
        logger.info("=== ETAP 4: TEST WYDAJNOŚCI CPU ===")
        cpu_score = self._benchmark_cpu()
        logger.info("Wynik testu CPU: %.2f", cpu_score)

        # 4. Test wydajności GPU
        logger.info("=== ETAP 5: TEST WYDAJNOŚCI GPU ===")
        gpu_score = 0
        if has_cuda:
            gpu_score = self._benchmark_gpu()
            logger.info("Wynik testu GPU: %.2f", gpu_score)
        else:
            logger.info("Pomijam test GPU - CUDA niedostępne")

        # 5. Obliczanie ogólnego wyniku
        logger.info("=== ETAP 6: OBLICZANIE WYNIKU KOŃCOWEGO ===")
        if has_cuda and gpu_score > 0:
            overall_score = 0.7 * gpu_score + 0.3 * cpu_score
            logger.info("Wynik końcowy (z GPU): %.2f", overall_score)
        else:
            overall_score = cpu_score
            logger.info("Wynik końcowy (tylko CPU): %.2f", overall_score)

        # 6. Obliczanie zalecanych parametrów
        logger.info("=== ETAP 7: OBLICZANIE ZALECANYCH PARAMETRÓW ===")
        recommended_batch_size = self._calculate_batch_size(gpu_memory, overall_score)
        recommended_workers = self._calculate_workers(cpu_info)
        use_mixed_precision = has_cuda and self._supports_mixed_precision()

        logger.info("Zalecane parametry:")
        logger.info("- Rozmiar batcha: %d", recommended_batch_size)
        logger.info("- Liczba workerów: %d", recommended_workers)
        logger.info("- Mixed precision: %s", "Tak" if use_mixed_precision else "Nie")

        # Dodane: Więcej rekomendacji sprzętowych
        additional_recommendations = {
            "recommended_model": self._recommend_model_architecture(
                gpu_memory, overall_score
            ),
            "recommended_precision": "half" if use_mixed_precision else "full",
            "recommended_augmentation": self._recommend_augmentation_level(
                overall_score
            ),
        }

        # Dodane: Parametry treningowe
        training_params = {
            "learning_rate": self._calculate_learning_rate(overall_score),
            "optimizer": self._recommend_optimizer(overall_score),
            "scheduler": self._recommend_scheduler(overall_score),
            "weight_decay": self._calculate_weight_decay(overall_score),
            "gradient_clip_val": self._calculate_gradient_clip(overall_score),
            "early_stopping_patience": self._calculate_early_stopping(overall_score),
            "max_epochs": self._calculate_max_epochs(overall_score),
            "validation_split": 0.2,  # Standardowa wartość
        }

        logger.info(
            "Dodatkowe rekomendacje: %s",
            json.dumps(additional_recommendations, indent=2),
        )
        logger.info(
            "Parametry treningowe: %s",
            json.dumps(training_params, indent=2),
        )

        # 7. Zapisz profil do bazy
        logger.info("=== ETAP 8: ZAPISYWANIE PROFILU ===")
        profile = {
            "machine_id": self.machine_id,
            "cpu_info": json.dumps(cpu_info),
            "ram_info": json.dumps(ram_info),
            "ram_total": ram_total,
            "gpu_info": json.dumps(gpu_info) if gpu_info else None,
            "gpu_memory": gpu_memory if gpu_memory > 0 else None,
            "cpu_score": cpu_score,
            "gpu_score": gpu_score if gpu_score > 0 else None,
            "overall_score": overall_score,
            "recommended_batch_size": recommended_batch_size,
            "recommended_workers": recommended_workers,
            "use_mixed_precision": use_mixed_precision,
            "additional_recommendations": json.dumps(additional_recommendations),
            "learning_rate": training_params["learning_rate"],
            "optimizer": training_params["optimizer"],
            "scheduler": training_params["scheduler"],
            "weight_decay": training_params["weight_decay"],
            "gradient_clip_val": training_params["gradient_clip_val"],
            "early_stopping_patience": training_params["early_stopping_patience"],
            "max_epochs": training_params["max_epochs"],
            "validation_split": training_params["validation_split"],
            "training_params": json.dumps(training_params),
        }

        logger.info("Przygotowany profil z machine_id: %s", profile["machine_id"])
        self._save_profile(profile)
        logger.info("=== PROFILOWANIE ZAKOŃCZONE ===")
        logger.info("Czas zakończenia: %s", time.strftime("%Y-%m-%d %H:%M:%S"))

        # Zwróć pełny raport
        return {
            "machine_id": self.machine_id,
            "cpu_info": cpu_info,
            "ram_info": ram_info,
            "ram_total": ram_total,
            "gpu_info": gpu_info,
            "gpu_memory": gpu_memory if gpu_memory > 0 else None,
            "cpu_score": cpu_score,
            "gpu_score": gpu_score if gpu_score > 0 else None,
            "overall_score": overall_score,
            "recommended_batch_size": recommended_batch_size,
            "recommended_workers": recommended_workers,
            "use_mixed_precision": use_mixed_precision,
            "additional_recommendations": additional_recommendations,
        }

    def _benchmark_cpu(self):
        """Wykonuje test wydajności procesora."""
        logger.info("=== ROZPOCZYNAM TEST WYDAJNOŚCI CPU ===")
        logger.info(
            "Czas rozpoczęcia testu CPU: %s", time.strftime("%Y-%m-%d %H:%M:%S")
        )

        # Określenie rozmiaru macierzy na podstawie dostępnej pamięci RAM
        ram_gb = psutil.virtual_memory().total / (1024**3)
        cpu_cores = psutil.cpu_count(logical=True)
        logger.info("Dostępna pamięć RAM: %.1f GB", ram_gb)
        logger.info("Liczba rdzeni CPU: %d", cpu_cores)

        # Dynamiczna regulacja rozmiaru macierzy
        if ram_gb > 16 and cpu_cores > 8:
            matrix_size = 3000  # Większe macierze dla mocnych komputerów
        elif ram_gb > 8 and cpu_cores > 4:
            matrix_size = 2000  # Standardowy rozmiar
        else:
            matrix_size = 1000  # Mniejsze macierze dla słabszych komputerów
        logger.info(
            "Wybrany rozmiar macierzy testowej: %dx%d", matrix_size, matrix_size
        )

        # Test mnożenia macierzy
        start_time = time.time()
        iterations = 3
        logger.info("Rozpoczynam %d iteracji testu mnożenia macierzy...", iterations)

        for i in range(iterations):
            logger.info("Iteracja %d/%d...", i + 1, iterations)
            A = np.random.rand(matrix_size, matrix_size)
            B = np.random.rand(matrix_size, matrix_size)
            C = np.dot(A, B)
            logger.info("Iteracja %d zakończona", i + 1)

        matrix_time = (time.time() - start_time) / iterations

        # Normalizacja wyniku (niższy czas = lepszy wynik)
        # Wynik 100 to benchmarkowy czas 1 sekunda
        matrix_score = 100 / max(matrix_time, 0.1)

        logger.info("=== WYNIKI TESTU CPU ===")
        logger.info("Średni czas mnożenia macierzy: %.2fs", matrix_time)
        logger.info("Wynik końcowy: %.1f", matrix_score)
        logger.info(
            "Czas zakończenia testu CPU: %s", time.strftime("%Y-%m-%d %H:%M:%S")
        )

        return matrix_score

    def profile_cpu(self):
        """Alias do _benchmark_cpu dla kompatybilności wstecznej."""
        return self._benchmark_cpu()

    def profile_memory(self):
        """Profiluje dostępną pamięć RAM."""
        logger.info("Profilowanie pamięci RAM...")

        # Pobierz informacje o pamięci
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        used_gb = memory.used / (1024**3)

        logger.info("Całkowita pamięć RAM: %.1f GB", total_gb)
        logger.info("Dostępna pamięć RAM: %.1f GB", available_gb)
        logger.info("Używana pamięć RAM: %.1f GB", used_gb)

        # Zwróć wynik jako procent dostępnej pamięci
        memory_score = (available_gb / total_gb) * 100

        return memory_score

    def _benchmark_gpu(self):
        """Wykonuje test wydajności karty graficznej."""
        logger.info("=== ROZPOCZYNAM TEST WYDAJNOŚCI GPU ===")
        logger.info(
            "Czas rozpoczęcia testu GPU: %s", time.strftime("%Y-%m-%d %H:%M:%S")
        )

        if not torch.cuda.is_available():
            logger.warning("CUDA nie jest dostępne. Pomijam test GPU.")
            return 0

        try:
            # Dobór rozmiaru macierzy na podstawie VRAM
            gpu_props = torch.cuda.get_device_properties(0)
            vram_gb = gpu_props.total_memory / (1024**3)
            logger.info("Dostępna pamięć VRAM: %.1f GB", vram_gb)

            if vram_gb >= 8:
                matrix_size = 4096  # Większa dla mocnych GPU
            elif vram_gb >= 4:
                matrix_size = 2048  # Średnia
            else:
                matrix_size = 1024  # Mniejsza dla słabszych GPU
            logger.info(
                "Wybrany rozmiar macierzy testowej: %dx%d", matrix_size, matrix_size
            )

            A = torch.randn(matrix_size, matrix_size, device="cuda")
            B = torch.randn(matrix_size, matrix_size, device="cuda")

            # Przeprowadź kilka iteracji, aby uśrednić czas
            iterations = 5
            times = []
            logger.info(
                "Rozpoczynam %d iteracji testu mnożenia macierzy na GPU...", iterations
            )

            # Rozgrzewka
            logger.info("Wykonuję rozgrzewkę GPU...")
            for i in range(2):
                C = torch.matmul(A, B)
                torch.cuda.synchronize()
                logger.info("Rozgrzewka %d/2 zakończona", i + 1)

            for i in range(iterations):
                logger.info("Iteracja %d/%d...", i + 1, iterations)
                start = time.time()
                C = torch.matmul(A, B)
                torch.cuda.synchronize()  # Czekaj na zakończenie operacji GPU
                times.append(time.time() - start)
                logger.info("Iteracja %d zakończona w %.4fs", i + 1, times[-1])

            avg_time = sum(times) / iterations

            # Normalizacja wyniku (niższy czas = lepszy wynik)
            # Wynik 100 to benchmarkowy czas 0.1 sekundy dla matrix_size=2048
            # Skalujemy wynik bazowy proporcjonalnie do kwadratu rozmiaru macierzy
            base_time_for_100 = 0.1 * (matrix_size / 2048) ** 2
            gpu_score = (base_time_for_100 / max(avg_time, 0.001)) * 100

            logger.info("=== WYNIKI TESTU GPU ===")
            logger.info("Średni czas mnożenia macierzy: %.4fs", avg_time)
            logger.info("Wynik końcowy: %.1f", gpu_score)
            logger.info(
                "Czas zakończenia testu GPU: %s", time.strftime("%Y-%m-%d %H:%M:%S")
            )

            return gpu_score
        except Exception as e:
            logger.error("Błąd podczas testowania GPU: %s", e, exc_info=True)
            return 0

    def profile_gpu(self):
        """Alias do _benchmark_gpu dla kompatybilności wstecznej."""
        return self._benchmark_gpu()

    def _supports_mixed_precision(self):
        """Sprawdza, czy GPU obsługuje mixed precision (Ampere lub nowsze)."""
        if not torch.cuda.is_available():
            return False
        try:
            # Sprawdź capability. Compute Capability >= 7.0 obsługuje Tensor Cores
            major, _ = torch.cuda.get_device_capability(0)
            return major >= 7
        except Exception as e:
            logger.error(
                "Błąd podczas sprawdzania mixed precision: %s", e, exc_info=True
            )
            return False

    def _calculate_batch_size(self, gpu_memory, score):
        """Oblicza zalecany rozmiar batcha na podstawie pamięci GPU i wydajności."""
        # Uwzględnij indeks wydajności (score) przy określaniu rozmiaru wsadu

        # Dla systemów z wykorzystaniem GPU
        if gpu_memory > 0:
            # Dla wysokich indeksów wydajności (>800) zawsze proponuj większe batche
            if score > 800:
                if gpu_memory >= 16:  # 16+ GB VRAM
                    return 256
                elif gpu_memory >= 12:  # 12-16 GB VRAM
                    return 192
                elif gpu_memory >= 8:  # 8-12 GB VRAM
                    return 128
                elif gpu_memory >= 6:  # 6-8 GB VRAM
                    return 96
                elif gpu_memory >= 4:  # 4-6 GB VRAM
                    return 64
                else:  # Mniej niż 4 GB VRAM
                    return 32

            # Dla średnich indeksów (>300)
            elif score > 300:
                if gpu_memory >= 16:  # 16+ GB VRAM
                    return 192
                elif gpu_memory >= 12:  # 12-16 GB VRAM
                    return 128
                elif gpu_memory >= 8:  # 8-12 GB VRAM
                    return 96
                elif gpu_memory >= 6:  # 6-8 GB VRAM
                    return 64
                elif gpu_memory >= 4:  # 4-6 GB VRAM
                    return 32
                else:  # Mniej niż 4 GB VRAM
                    return 16

            # Dla niskich indeksów (<300)
            else:
                if gpu_memory >= 16:  # 16+ GB VRAM
                    return 128
                elif gpu_memory >= 12:  # 12-16 GB VRAM
                    return 96
                elif gpu_memory >= 8:  # 8-12 GB VRAM
                    return 64
                elif gpu_memory >= 6:  # 6-8 GB VRAM
                    return 32
                elif gpu_memory >= 4:  # 4-6 GB VRAM
                    return 16
                else:  # Mniej niż 4 GB VRAM
                    return 8

        # Dla systemów bez GPU - bazuj na ocenie CPU
        else:
            if score > 500:  # Bardzo mocny CPU
                return 64
            elif score > 300:  # Mocny CPU
                return 32
            elif score > 150:  # Średni CPU
                return 16
            else:  # Słaby CPU
                return 8

    def _calculate_workers(self, cpu_info):
        """Oblicza zalecana liczbę workerów do ładowania danych."""
        logical_cores = cpu_info.get("cores_logical", os.cpu_count() or 4)

        # Standardowe zalecenie to liczba rdzeni - 1, ale nie mniej niż 4
        workers = max(1, min(logical_cores - 1, logical_cores // 2))

        # Dla dużych maszyn ograniczamy maksymalną ilość
        if workers > 16:
            workers = 16

        return workers

    def _save_profile(self, profile):
        """Zapisuje profil sprzętowy do bazy danych."""
        logger.info("=== ZAPISYWANIE PROFILU DO BAZY DANYCH ===")
        logger.info("Czas rozpoczęcia zapisu: %s", time.strftime("%Y-%m-%d %H:%M:%S"))
        logger.info("Machine ID: %s", profile["machine_id"])

        required_keys = [
            "machine_id",
            "cpu_info",
            "ram_total",
            "gpu_info",
            "gpu_memory",
            "cpu_score",
            "gpu_score",
            "overall_score",
            "recommended_batch_size",
            "recommended_workers",
            "use_mixed_precision",
        ]

        # Nowe, opcjonalne klucze dodane w tej wersji
        optional_keys = ["ram_info", "additional_recommendations"]

        # Weryfikacja, czy wszystkie wymagane klucze są obecne
        missing_keys = [key for key in required_keys if key not in profile]
        if missing_keys:
            logger.error("Brakujące klucze w profilu: %s", ", ".join(missing_keys))
            return False

        # Przygotowanie danych do wstawienia, obsługa nowych opcjonalnych pól
        sql = """
            INSERT INTO hardware_profiles (
                machine_id, cpu_info, ram_total, gpu_info, gpu_memory,
                cpu_score, gpu_score, overall_score,
                recommended_batch_size, recommended_workers, use_mixed_precision,
                ram_info, additional_recommendations, profile_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(machine_id) DO UPDATE SET
                cpu_info=excluded.cpu_info,
                ram_total=excluded.ram_total,
                gpu_info=excluded.gpu_info,
                gpu_memory=excluded.gpu_memory,
                cpu_score=excluded.cpu_score,
                gpu_score=excluded.gpu_score,
                overall_score=excluded.overall_score,
                recommended_batch_size=excluded.recommended_batch_size,
                recommended_workers=excluded.recommended_workers,
                use_mixed_precision=excluded.use_mixed_precision,
                ram_info=excluded.ram_info,
                additional_recommendations=excluded.additional_recommendations,
                profile_date=CURRENT_TIMESTAMP;
        """

        try:
            logger.info("Łączenie z bazą danych...")
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Zdefiniowanie pól, które mają być wstawione/zaktualizowane
                fields = [
                    profile["machine_id"],
                    json.dumps(profile["cpu_info"]),
                    profile["ram_total"],
                    json.dumps(profile.get("gpu_info", {})),
                    profile.get("gpu_memory", 0),
                    profile["cpu_score"],
                    profile.get("gpu_score", 0),
                    profile["overall_score"],
                    profile["recommended_batch_size"],
                    profile["recommended_workers"],
                    int(profile["use_mixed_precision"]),
                    json.dumps(profile.get("ram_info", {})),
                    json.dumps(profile.get("additional_recommendations", {})),
                ]

                logger.info("Wykonywanie zapytania SQL...")
                cursor.execute(sql, fields)
                conn.commit()
                logger.info("Profil sprzętowy został zapisany/zaktualizowany pomyślnie")
                logger.info(
                    "Czas zakończenia zapisu: %s", time.strftime("%Y-%m-%d %H:%M:%S")
                )
                return True
        except sqlite3.Error as e:
            logger.error("Błąd SQLite podczas zapisu profilu: %s", e, exc_info=True)
            return False
        except Exception as e:
            logger.error(
                "Nieoczekiwany błąd podczas zapisu profilu: %s", e, exc_info=True
            )
            return False

    # Przywrócenie publicznego aliasu dla kompatybilności wstecznej
    def save_profile(self, profile):
        """Publiczny alias do _save_profile."""
        return self._save_profile(profile)

    def load_profile(self, machine_id=None):
        """Wczytuje profil sprzętowy z bazy danych."""
        target_id = machine_id if machine_id else self.machine_id
        logger.info("=== WCZYTYWANIE PROFILU Z BAZY DANYCH ===")
        logger.info(
            "Czas rozpoczęcia wczytywania: %s", time.strftime("%Y-%m-%d %H:%M:%S")
        )
        logger.info("Szukam profilu dla machine_id: %s", target_id)

        if not target_id:
            logger.error("Brak machine_id do wczytania profilu")
            return None

        try:
            # Sprawdź czy baza danych istnieje
            if not os.path.exists(self.db_path):
                logger.error("Baza danych nie istnieje: %s", self.db_path)
                return None

            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Sprawdź czy tabela istnieje
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='hardware_profiles'"
                )
                if not cursor.fetchone():
                    logger.error("Tabela hardware_profiles nie istnieje!")
                    return None

                # Zaktualizuj zapytanie, aby pobierać wszystkie kolumny
                logger.info("Wykonywanie zapytania SQL...")
                cursor.execute(
                    "SELECT * FROM hardware_profiles WHERE machine_id=?", (target_id,)
                )
                row = cursor.fetchone()

                if row:
                    logger.info("Znaleziono profil dla machine_id: %s", target_id)
                    # Deserializuj pola JSON z powrotem do obiektów Python
                    profile = dict(row)

                    # Deserializacja JSON, obsługa błędów
                    for key in [
                        "cpu_info",
                        "gpu_info",
                        "ram_info",
                        "additional_recommendations",
                    ]:
                        if profile.get(key):
                            try:
                                profile[key] = json.loads(profile[key])
                                logger.info("Zdeserializowano pole %s", key)
                            except json.JSONDecodeError:
                                logger.warning(
                                    "Nie można zdeserializować JSON dla '%s' w profilu %s.",
                                    key,
                                    target_id,
                                    exc_info=True,
                                )
                                profile[key] = None
                        else:
                            profile[key] = None  # Ustaw na None jeśli brak

                    # Konwersja wartości boolowskich z int (0/1) na bool
                    if (
                        "use_mixed_precision" in profile
                        and profile["use_mixed_precision"] is not None
                    ):
                        profile["use_mixed_precision"] = bool(
                            profile["use_mixed_precision"]
                        )

                    logger.info("Profil został wczytany pomyślnie")
                    logger.info(
                        "Czas zakończenia wczytywania: %s",
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                    )
                    return profile
                else:
                    logger.warning(
                        "Nie znaleziono profilu dla machine_id: %s", target_id
                    )
                    return None
        except sqlite3.Error as e:
            logger.error(
                "Błąd SQLite podczas wczytywania profilu: %s", e, exc_info=True
            )
            return None
        except Exception as e:
            logger.error("Nieoczekiwany błąd wczytywania profilu: %s", e, exc_info=True)
            return None

    def get_optimal_parameters(self):
        """
        Zwraca optymalne parametry dla bieżącego sprzętu.

        Returns:
            dict: Słownik z optymalnymi parametrami lub domyślnymi jeśli profil nie istnieje
        """
        logger.info("Pobieranie optymalnych parametrów...")
        try:
            if not hasattr(self, "machine_id") or not self.machine_id:
                logger.warning("Brak machine_id, generuję nowy...")
                self.machine_id = self._generate_machine_id()
            logger.info("Używam machine_id: %s", self.machine_id)

            profile = self.load_profile()
            logger.info("Profil %s", "znaleziony" if profile else "nie znaleziony")

            if profile:
                # Walidacja wartości z profilu
                params = {}

                # Batch size
                batch_size = profile.get("recommended_batch_size")
                if batch_size is not None and batch_size > 0:
                    params["batch_size"] = batch_size
                else:
                    params["batch_size"] = (
                        32  # Zmienione z DEFAULT_TRAINING_PARAMS["batch_size"]
                    )

                # Number of workers
                num_workers = profile.get("recommended_workers")
                if num_workers is not None and num_workers >= 0:
                    params["num_workers"] = num_workers
                else:
                    params["num_workers"] = (
                        0  # Zmienione z max(1, (os.cpu_count() or 4) // 2)
                    )

                # Mixed precision
                mixed_precision = profile.get("use_mixed_precision")
                if mixed_precision is not None:
                    params["mixed_precision"] = mixed_precision
                else:
                    params["mixed_precision"] = (
                        False  # Zmienione z torch.cuda.is_available()
                    )

                # GPU availability
                gpu_score = profile.get("gpu_score")
                params["gpu_available"] = gpu_score is not None and gpu_score > 0

                # Learning rate
                learning_rate = profile.get("learning_rate")
                if learning_rate is not None and learning_rate > 0:
                    params["learning_rate"] = learning_rate
                else:
                    params["learning_rate"] = (
                        1e-4  # Domyślne parametry jeśli nie ma profilu
                    )

                # Optimizer
                optimizer = profile.get("optimizer")
                if optimizer is not None:
                    params["optimizer"] = optimizer
                else:
                    params["optimizer"] = (
                        "RMSprop"  # Domyślne parametry jeśli nie ma profilu
                    )

                # Scheduler
                scheduler = profile.get("scheduler")
                if scheduler is not None:
                    params["scheduler"] = scheduler
                else:
                    params["scheduler"] = (
                        "cosine"  # Domyślne parametry jeśli nie ma profilu
                    )

                # Weight decay
                weight_decay = profile.get("weight_decay")
                if weight_decay is not None and weight_decay >= 0:
                    params["weight_decay"] = weight_decay
                else:
                    params["weight_decay"] = (
                        1e-5  # Domyślne parametry jeśli nie ma profilu
                    )

                # Gradient clip value
                gradient_clip_val = profile.get("gradient_clip_val")
                if gradient_clip_val is not None and gradient_clip_val >= 0:
                    params["gradient_clip_val"] = gradient_clip_val
                else:
                    params["gradient_clip_val"] = (
                        0.1  # Domyślne parametry jeśli nie ma profilu
                    )

                # Early stopping patience
                early_stopping_patience = profile.get("early_stopping_patience")
                if early_stopping_patience is not None and early_stopping_patience > 0:
                    params["early_stopping_patience"] = early_stopping_patience
                else:
                    params["early_stopping_patience"] = (
                        5  # Domyślne parametry jeśli nie ma profilu
                    )

                # Max epochs
                max_epochs = profile.get("max_epochs")
                if max_epochs is not None and max_epochs > 0:
                    params["max_epochs"] = max_epochs
                else:
                    params["max_epochs"] = 30  # Domyślne parametry jeśli nie ma profilu

                # Validation split
                validation_split = profile.get("validation_split")
                if validation_split is not None and 0 < validation_split < 1:
                    params["validation_split"] = validation_split
                else:
                    params["validation_split"] = (
                        0.2  # Domyślne parametry jeśli nie ma profilu
                    )

                logger.info("Zwracam parametry z profilu.")
                return params
            else:
                # Domyślne parametry jeśli nie ma profilu
                params = {
                    "batch_size": 32,  # Zmienione z DEFAULT_TRAINING_PARAMS["batch_size"]
                    "num_workers": 0,  # Zmienione z max(1, (os.cpu_count() or 4) // 2)
                    "mixed_precision": False,  # Zmienione z torch.cuda.is_available()
                    "gpu_available": torch.cuda.is_available(),
                    "learning_rate": 1e-4,  # Domyślne parametry jeśli nie ma profilu
                    "optimizer": "RMSprop",  # Domyślne parametry jeśli nie ma profilu
                    "scheduler": "cosine",  # Domyślne parametry jeśli nie ma profilu
                    "weight_decay": 1e-5,  # Domyślne parametry jeśli nie ma profilu
                    "gradient_clip_val": 0.1,  # Domyślne parametry jeśli nie ma profilu
                    "early_stopping_patience": 5,  # Domyślne parametry jeśli nie ma profilu
                    "max_epochs": 30,  # Domyślne parametry jeśli nie ma profilu
                    "validation_split": 0.2,  # Domyślne parametry jeśli nie ma profilu
                }
                logger.info("Zwracam domyślne parametry.")
                return params

        except Exception as e:
            logger.error(f"Błąd podczas pobierania optymalnych parametrów: {str(e)}")
            # Zwróć bezpieczne domyślne parametry w przypadku błędu
            return {
                "batch_size": 32,
                "num_workers": 0,
                "mixed_precision": False,
                "gpu_available": torch.cuda.is_available(),
                "learning_rate": 1e-4,
                "optimizer": "RMSprop",
                "scheduler": "cosine",
                "weight_decay": 1e-5,
                "gradient_clip_val": 0.1,
                "early_stopping_patience": 5,
                "max_epochs": 30,
                "validation_split": 0.2,
            }

    def generate_recommendations(self):
        """
        Generuje zalecenia dotyczące optymalizacji ustawień na podstawie profilu sprzętowego.

        Returns:
            dict: Słownik z zaleceniami i wyjaśnieniami
        """
        logger.info("=== GENEROWANIE ZALECEŃ OPTYMALIZACYJNYCH ===")
        logger.info("Czas rozpoczęcia: %s", time.strftime("%Y-%m-%d %H:%M:%S"))

        if not hasattr(self, "machine_id") or not self.machine_id:
            logger.warning("Brak machine_id, generuję nowy...")
            self.machine_id = self._generate_machine_id()
        logger.info("Aktualny machine_id: %s", self.machine_id)

        # Pobierz aktualny profil
        profile = self.load_profile()
        if not profile:
            logger.warning(
                "Nie znaleziono profilu sprzętowego. Uruchamiam profilowanie..."
            )
            profile = self.run_profile()
            if not profile:
                raise Exception("Nie udało się utworzyć profilu sprzętowego")

        logger.info(
            "Używam profilu z machine_id: %s", profile.get("machine_id", "BRAK")
        )

        # Przygotuj podstawowe informacje
        cpu_info = profile.get("cpu_info", {})
        if isinstance(cpu_info, str):
            try:
                cpu_info = json.loads(cpu_info)
                logger.info("Zdeserializowano informacje o CPU")
            except:
                logger.warning("Nie udało się zdeserializować informacji o CPU")
                cpu_info = {}

        gpu_info = profile.get("gpu_info", {})
        if isinstance(gpu_info, str):
            try:
                gpu_info = json.loads(gpu_info)
                logger.info("Zdeserializowano informacje o GPU")
            except:
                logger.warning("Nie udało się zdeserializować informacji o GPU")
                gpu_info = {}

        # Bezpieczna konwersja wartości liczbowych
        def safe_float(value, default=0.0):
            if value is None:
                return default
            try:
                return float(value)
            except (ValueError, TypeError):
                return default

        def safe_int(value, default=0):
            if value is None:
                return default
            try:
                return int(value)
            except (ValueError, TypeError):
                return default

        # Pobierz wartości z profilu
        cpu_score = safe_float(profile.get("cpu_score", 0))
        gpu_score = safe_float(profile.get("gpu_score", 0))
        overall_score = safe_float(profile.get("overall_score", 0))
        ram_total = safe_float(profile.get("ram_total", 0))

        logger.info("Wyniki wydajności:")
        logger.info("- CPU Score: %.1f", cpu_score)
        logger.info("- GPU Score: %.1f", gpu_score)
        logger.info("- Overall Score: %.1f", overall_score)
        logger.info("- RAM Total: %.1f GB", ram_total)

        # Przygotuj rekomendacje
        recommendations = {
            "recommended_batch_size": safe_int(
                profile.get("recommended_batch_size", 32)
            ),
            "recommended_workers": safe_int(profile.get("recommended_workers", 4)),
            "use_mixed_precision": bool(profile.get("use_mixed_precision", False)),
            "optimization_tips": [],
        }

        # Dodaj wskazówki optymalizacyjne
        if gpu_score > 800:
            recommendations["optimization_tips"].append(
                "Wysoka wydajność GPU - możesz zwiększyć rozmiar batcha"
            )
        elif gpu_score < 300:
            recommendations["optimization_tips"].append(
                "Niska wydajność GPU - zalecane zmniejszenie rozmiaru batcha"
            )

        if cpu_score > 800:
            recommendations["optimization_tips"].append(
                "Wysoka wydajność CPU - możesz zwiększyć liczbę workerów"
            )
        elif cpu_score < 300:
            recommendations["optimization_tips"].append(
                "Niska wydajność CPU - zalecane zmniejszenie liczby workerów"
            )

        if ram_total < 8:
            recommendations["optimization_tips"].append(
                "Mało pamięci RAM - zalecane zmniejszenie rozmiaru batchy"
            )

        # Wyświetl zalecenia
        logger.info("\n=== ZALECENIA OPTYMALIZACYJNE ===")
        logger.info("CPU: %s", cpu_info.get("processor", "Nieznany"))
        logger.info("GPU: %s", gpu_info.get("name", "Brak") if gpu_info else "Brak")
        logger.info("RAM: %.1f GB", ram_total)
        logger.info("\nWyniki wydajności:")
        logger.info("- CPU Score: %.1f", cpu_score)
        if gpu_score > 0:
            logger.info("- GPU Score: %.1f", gpu_score)
        logger.info("- Overall Score: %.1f", overall_score)
        logger.info("\nZaleconane ustawienia:")
        logger.info("- Batch Size: %d", recommendations["recommended_batch_size"])
        logger.info("- Workers: %d", recommendations["recommended_workers"])
        logger.info(
            "- Mixed Precision: %s",
            "Tak" if recommendations["use_mixed_precision"] else "Nie",
        )
        logger.info("\nWskazówki optymalizacyjne:")
        for tip in recommendations["optimization_tips"]:
            logger.info("- %s", tip)

        # Wyświetl parametry treningowe
        print("\nPARAMETRY TRENINGOWE:")
        print("-" * 30)
        print(f"Learning Rate: {profile.get('learning_rate', 1e-4):.2e}")
        print(f"Optimizer: {profile.get('optimizer', 'Adam')}")
        print(f"Scheduler: {profile.get('scheduler', 'ReduceLROnPlateau')}")
        print(f"Weight Decay: {profile.get('weight_decay', 1e-5):.2e}")
        print(f"Gradient Clip: {profile.get('gradient_clip_val', 0.1)}")
        print(f"Early Stopping Patience: {profile.get('early_stopping_patience', 5)}")
        print(f"Max Epochs: {profile.get('max_epochs', 30)}")
        print(f"Validation Split: {profile.get('validation_split', 0.2)}")

        logger.info(
            "Czas zakończenia generowania zaleceń: %s",
            time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        return recommendations

    def _get_cpu_name(self):
        """Próbuje odczytać nazwę procesora."""
        logger.info("Pobieranie nazwy procesora...")
        try:
            if platform.system() == "Windows":
                logger.info("System Windows - używam wmic")
                output = subprocess.check_output(
                    ["wmic", "cpu", "get", "name"]
                ).decode()
                cpu_name = output.strip().split("\n")[1]
                logger.info("Nazwa procesora: %s", cpu_name)
                return cpu_name
            elif platform.system() == "Linux":
                logger.info("System Linux - czytam /proc/cpuinfo")
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            cpu_name = line.split(":")[1].strip()
                            logger.info("Nazwa procesora: %s", cpu_name)
                            return cpu_name
            logger.info("Używam platform.processor() jako fallback")
            return platform.processor()
        except FileNotFoundError:
            logger.warning("Nie znaleziono /proc/cpuinfo (Linux).", exc_info=True)
            return platform.processor()
        except subprocess.CalledProcessError as e:
            logger.warning("Błąd wmic przy nazwie CPU (Windows): %s", e, exc_info=True)
            return platform.processor()
        except Exception as e:
            logger.warning("Nie udało się odczytać nazwy CPU: %s", e, exc_info=True)
            return platform.processor()

    def _get_ram_frequency(self):
        """Próbuje odczytać częstotliwość pamięci RAM."""
        logger.info("Pobieranie częstotliwości RAM...")
        try:
            if platform.system() == "Windows":
                logger.info("System Windows - używam wmic")
                output = subprocess.check_output(
                    ["wmic", "memorychip", "get", "speed"]
                ).decode()
                speeds = [
                    int(x.strip()) for x in output.strip().split("\n")[1:] if x.strip()
                ]
                if speeds:
                    avg_speed = sum(speeds) / len(speeds)
                    logger.info("Średnia częstotliwość RAM: %d MHz", avg_speed)
                    return avg_speed
                else:
                    logger.warning("Nie znaleziono informacji o częstotliwości RAM")
                    return None
            else:
                logger.warning(
                    "System nie jest Windows - nie można pobrać częstotliwości RAM"
                )
                return None
        except subprocess.CalledProcessError as e:
            logger.warning(
                "Błąd wmic przy częstotliwości RAM (Windows): %s", e, exc_info=True
            )
            return None
        except Exception as e:
            logger.warning("Nie odczytano częstotliwości RAM: %s", e, exc_info=True)
            return None

    def _get_cuda_compute_capability(self):
        """Pobiera compute capability karty CUDA."""
        logger.info("Pobieranie compute capability CUDA...")
        if not torch.cuda.is_available():
            logger.warning(
                "CUDA nie jest dostępne - nie można pobrać compute capability"
            )
            return None
        try:
            props = torch.cuda.get_device_properties(0)
            capability = f"{props.major}.{props.minor}"
            logger.info("Compute capability: %s", capability)
            return capability
        except Exception as e:
            logger.warning("Nie pobrano compute capability CUDA: %s", e, exc_info=True)
            return None

    def _estimate_gpu_memory_bandwidth(self):
        """Szacuje przepustowość pamięci GPU na podstawie testu."""
        logger.info("Szacowanie przepustowości pamięci GPU...")
        if not torch.cuda.is_available():
            logger.warning("CUDA nie jest dostępne - pomijam test przepustowości")
            return None
        try:
            start_time = time.time()
            tensor_size = 1000
            iterations = 3
            logger.info(
                "Rozpoczynam test przepustowości (rozmiar tensora: %d)", tensor_size
            )

            # Rozgrzewka
            logger.info("Wykonuję rozgrzewkę...")
            for i in range(2):
                x = torch.randn(
                    tensor_size, tensor_size, device="cuda", dtype=torch.float32
                )
                y = torch.randn(
                    tensor_size, tensor_size, device="cuda", dtype=torch.float32
                )
                z = x + y
                torch.cuda.synchronize()
                logger.info("Rozgrzewka %d/2 zakończona", i + 1)

            total_elapsed = 0
            for i in range(iterations):
                logger.info("Iteracja %d/%d...", i + 1, iterations)
                x = torch.randn(
                    tensor_size, tensor_size, device="cuda", dtype=torch.float32
                )
                y = torch.randn(
                    tensor_size, tensor_size, device="cuda", dtype=torch.float32
                )
                torch.cuda.synchronize()
                iter_start = time.time()
                z = x + y
                torch.cuda.synchronize()
                iter_time = time.time() - iter_start
                total_elapsed += iter_time
                logger.info("Iteracja %d zakończona w %.4fs", i + 1, iter_time)

            # Oblicz szacunkową przepustowość w GB/s
            avg_elapsed = total_elapsed / iterations
            bytes_processed = 3 * 4 * tensor_size * tensor_size
            if avg_elapsed > 0:
                bandwidth = bytes_processed / avg_elapsed / (1024**3)  # GB/s
                logger.info("Szacowana przepustowość pamięci GPU: %.2f GB/s", bandwidth)
                return bandwidth
            else:
                logger.warning(
                    "Średni czas operacji GPU wyniósł zero, nie można obliczyć przepustowości."
                )
                return None
        except Exception as e:
            logger.error(
                "Błąd podczas testowania przepustowości GPU: %s", e, exc_info=True
            )
            return None

    def _recommend_model_architecture(self, gpu_memory, overall_score):
        """Rekomenduje architekturę modelu na podstawie sprzętu."""
        logger.info("=== REKOMENDACJA ARCHITEKTURY MODELU ===")
        logger.info("Zgodnie z dokumentacją, zawsze zwracamy efficientnet")
        return "efficientnet"

    def _recommend_augmentation_level(self, overall_score):
        """Rekomenduje poziom augmentacji na podstawie wydajności."""
        logger.info("=== REKOMENDACJA POZIOMU AUGMENTACJI ===")
        logger.info("Overall Score: %.1f", overall_score)

        if overall_score > 800:
            recommendation = "high"
            logger.info(
                "Rekomendowany poziom augmentacji: %s (wysoka wydajność)",
                recommendation,
            )
        elif overall_score > 500:
            recommendation = "medium"
            logger.info(
                "Rekomendowany poziom augmentacji: %s (średnia wydajność)",
                recommendation,
            )
        else:
            recommendation = "low"
            logger.info(
                "Rekomendowany poziom augmentacji: %s (podstawowa wydajność)",
                recommendation,
            )

        return recommendation

    def display_settings(self):
        """Wyświetla ustawienia w czytelnej formie w konsoli."""
        print("\n" + "=" * 50)
        print("ZALECANE USTAWIENIA SPRZĘTOWE")
        print("=" * 50)

        # Pobierz aktualny profil
        profile = self.load_profile()
        if not profile:
            print("Nie znaleziono profilu sprzętowego. Uruchamiam profilowanie...")
            profile = self.run_profile()
            if not profile:
                print("Błąd: Nie udało się utworzyć profilu sprzętowego")
                return

        # Wyświetl informacje o sprzęcie
        print("\nINFORMACJE O SPRZĘCIE:")
        print("-" * 30)
        cpu_info = json.loads(profile.get("cpu_info", "{}"))
        print(f"CPU: {cpu_info.get('name', 'Nieznany')}")
        print(
            f"Rdzenie: {cpu_info.get('cores_physical', '?')} fizyczne, {cpu_info.get('cores_logical', '?')} logiczne"
        )

        gpu_info = json.loads(profile.get("gpu_info", "{}"))
        if gpu_info:
            print(f"GPU: {gpu_info.get('name', 'Nieznana')}")
            print(f"VRAM: {gpu_info.get('vram_gb', '?')} GB")
        else:
            print("GPU: Brak")

        ram_total = profile.get("ram_total", 0)
        print(f"RAM: {ram_total:.1f} GB")

        # Wyświetl wyniki testów
        print("\nWYNIKI TESTOW WYDAJNOŚCI:")
        print("-" * 30)
        print(f"CPU Score: {profile.get('cpu_score', 0):.1f}")
        if profile.get("gpu_score"):
            print(f"GPU Score: {profile.get('gpu_score'):.1f}")
        print(f"Overall Score: {profile.get('overall_score', 0):.1f}")

        # Wyświetl zalecane ustawienia
        print("\nZALECANE USTAWIENIA:")
        print("-" * 30)
        print(f"Batch Size: {profile.get('recommended_batch_size', 32)}")
        print(f"Workers: {profile.get('recommended_workers', 4)}")
        print(
            f"Mixed Precision: {'Tak' if profile.get('use_mixed_precision') else 'Nie'}"
        )

        # Wyświetl parametry treningowe
        print("\nPARAMETRY TRENINGOWE:")
        print("-" * 30)
        print(f"Learning Rate: {profile.get('learning_rate', 1e-4):.2e}")
        print(f"Optimizer: {profile.get('optimizer', 'Adam')}")
        print(f"Scheduler: {profile.get('scheduler', 'ReduceLROnPlateau')}")
        print(f"Weight Decay: {profile.get('weight_decay', 1e-5):.2e}")
        print(f"Gradient Clip: {profile.get('gradient_clip_val', 0.1)}")
        print(f"Early Stopping Patience: {profile.get('early_stopping_patience', 5)}")
        print(f"Max Epochs: {profile.get('max_epochs', 30)}")
        print(f"Validation Split: {profile.get('validation_split', 0.2)}")

        # Wyświetl dodatkowe rekomendacje
        additional = json.loads(profile.get("additional_recommendations", "{}"))
        if additional:
            print("\nDODATKOWE REKOMENDACJE:")
            print("-" * 30)
            print(f"Model: {additional.get('recommended_model', 'Nieznany')}")
            print(f"Precyzja: {additional.get('recommended_precision', 'Nieznana')}")
            print(
                f"Augmentacja: {additional.get('recommended_augmentation', 'Nieznana')}"
            )

        print("\n" + "=" * 50)

    def _calculate_learning_rate(self, overall_score):
        """Oblicza zalecaną wartość learning rate na podstawie wydajności."""
        if overall_score > 800:
            return 1e-3  # Wysoka wydajność - większy learning rate
        elif overall_score > 500:
            return 5e-4  # Średnia wydajność
        else:
            return 1e-4  # Niska wydajność - mniejszy learning rate

    def _recommend_optimizer(self, overall_score):
        """Rekomenduje optymalizator na podstawie wydajności."""
        if overall_score > 800:
            return "AdamW"  # Wysoka wydajność - bardziej zaawansowany optymalizator
        elif overall_score > 500:
            return "Adam"  # Średnia wydajność
        else:
            return "SGD"  # Niska wydajność - prostszy optymalizator

    def _recommend_scheduler(self, overall_score):
        """Rekomenduje scheduler na podstawie wydajności."""
        if overall_score > 800:
            return "OneCycleLR"  # Wysoka wydajność - zaawansowany scheduler
        elif overall_score > 500:
            return "CosineAnnealingLR"  # Średnia wydajność
        else:
            return "ReduceLROnPlateau"  # Niska wydajność - prostszy scheduler

    def _calculate_weight_decay(self, overall_score):
        """Oblicza wartość weight decay na podstawie wydajności."""
        if overall_score > 800:
            return 1e-4  # Wysoka wydajność - większy weight decay
        elif overall_score > 500:
            return 5e-5  # Średnia wydajność
        else:
            return 1e-5  # Niska wydajność - mniejszy weight decay

    def _calculate_gradient_clip(self, overall_score):
        """Oblicza wartość gradient clipping na podstawie wydajności."""
        if overall_score > 800:
            return 1.0  # Wysoka wydajność - większy gradient clip
        elif overall_score > 500:
            return 0.5  # Średnia wydajność
        else:
            return 0.1  # Niska wydajność - mniejszy gradient clip

    def _calculate_early_stopping(self, overall_score):
        """Oblicza wartość patience dla early stopping na podstawie wydajności."""
        if overall_score > 800:
            return 10  # Wysoka wydajność - większa cierpliwość
        elif overall_score > 500:
            return 7  # Średnia wydajność
        else:
            return 5  # Niska wydajność - mniejsza cierpliwość

    def _calculate_max_epochs(self, overall_score):
        """Oblicza maksymalną liczbę epok na podstawie wydajności."""
        if overall_score > 800:
            return 100  # Wysoka wydajność - więcej epok
        elif overall_score > 500:
            return 50  # Średnia wydajność
        else:
            return 30  # Niska wydajność - mniej epok


if __name__ == "__main__":
    # Konfiguracja podstawowego logowania na potrzeby testów
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    profiler = HardwareProfiler()
    profile_report = profiler.run_profile()

    if profile_report:
        logger.info("\n--- Pełny raport profilowania ---")
        for key, value in profile_report.items():
            if isinstance(value, dict):
                logger.info(f"{key}:")
                for sub_key, sub_value in value.items():
                    logger.info(f"  {sub_key}: {sub_value}")
            elif isinstance(value, float):
                logger.info(f"{key}: {value:.2f}")  # Formatuj floaty
            else:
                logger.info(f"{key}: {value}")
    else:
        logger.error("Nie udało się wygenerować raportu profilowania.")
