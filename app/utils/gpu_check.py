#!/usr/bin/env python
import sys

import torch


def check_gpu():
    print(f"Python wersja: {sys.version}")
    print(f"PyTorch wersja: {torch.__version__}")

    if torch.cuda.is_available():
        print("CUDA jest dostępne.")
        print(f"CUDA wersja: {torch.version.cuda}")
        print(f"Liczba urządzeń CUDA: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"Urządzenie {i}: {torch.cuda.get_device_name(i)}")

        # Sprawdź czy możemy wykonać obliczenia na GPU
        try:
            # Jawnie definiujemy dtype dla bezpieczeństwa
            x = torch.randn(1000, 1000, device="cuda", dtype=torch.float32)
            y = torch.randn(1000, 1000, device="cuda", dtype=torch.float32)
            z = torch.matmul(x, y)
            print("Test obliczeń na GPU zakończony powodzeniem.")
        except Exception as e:
            print(f"Błąd podczas testowania obliczeń na GPU: {e}")
    else:
        print("CUDA nie jest dostępne.")
        print("Możliwe przyczyny:")
        print("1. Nie zainstalowano sterowników NVIDIA.")
        print("2. PyTorch zainstalowano bez wsparcia dla CUDA.")
        print(
            "3. Wersja CUDA używana przez PyTorch jest niekompatybilna z zainstalowaną wersją sterowników."
        )

    # Sprawdź, czy PyTorch został skompilowany z CUDA
    print(f"PyTorch skompilowany z CUDA: {torch.backends.cuda.is_built()}")

    # Sprawdź CUDNN
    if torch.backends.cudnn.is_available():
        print(f"CUDNN jest dostępny. Wersja: {torch.backends.cudnn.version()}")
        print(f"CUDNN włączony: {torch.backends.cudnn.enabled}")
    else:
        print("CUDNN nie jest dostępny.")


if __name__ == "__main__":
    check_gpu()
