import json
import os
import shutil

import torch


def _create_dummy_input(classifier, device_type="cpu"):
    """
    Optymalizacja: Uproszczenie funkcji poprzez usunięcie zbędnych bloków try-except.
    """
    target_device = getattr(classifier, "device", torch.device("cpu"))

    if target_device.type == "cuda" and torch.cuda.is_available():
        return torch.randn(1, 3, 224, 224, device="cuda", dtype=torch.float32)
    else:
        return torch.randn(1, 3, 224, 224, dtype=torch.float32, device="cpu")


def export_model(classifier, export_dir, include_sample_code=True, formats=None):
    """
    Eksportuje model do formatu gotowego do użycia w innych projektach.

    Args:
        classifier: Instancja ImageClassifier
        export_dir: Katalog docelowy
        include_sample_code: Czy dołączyć przykładowy kod
        formats: Lista formatów do eksportu (dostępne: 'pytorch', 'onnx', 'torchscript', 'jit')

    Returns:
        Ścieżka do wyeksportowanego modelu lub None w przypadku błędu
    """
    try:
        # Utwórz katalog eksportu
        os.makedirs(export_dir, exist_ok=True)

        # Domyślnie eksportuj wszystkie formaty
        if formats is None:
            formats = ["pytorch", "onnx", "torchscript", "jit"]

        # Utwórz tensor testowy
        dummy_input = _create_dummy_input(classifier)
        if dummy_input is None:  # Sprawdzenie czy tensor został utworzony
            raise RuntimeError("Nie udało się utworzyć tensora testowego.")

        export_paths = {}

        # Eksport do formatu PyTorch
        if "pytorch" in formats:
            try:
                model_path = os.path.join(export_dir, "model.pt")
                torch.save(
                    {
                        "model_type": classifier.model_type,
                        "num_classes": classifier.num_classes,
                        "model_state_dict": classifier.model.state_dict(),
                        "class_names": classifier.class_names,
                    },
                    model_path,
                )
                export_paths["pytorch"] = model_path
            except Exception as e:
                print(f"Błąd podczas eksportu do PyTorch: {e}")  # Logowanie błędu
                # Docelowo: logger.error(..., func_name="export_model", file_name="ai/export.py")

        # Eksportuj do ONNX
        if "onnx" in formats:
            try:
                onnx_path = os.path.join(export_dir, "model.onnx")
                torch.onnx.export(
                    classifier.model,
                    dummy_input,
                    onnx_path,
                    input_names=["input"],
                    output_names=["output"],
                    export_params=True,
                    opset_version=12,  # Nowszy opset dla lepszej kompatybilności
                    do_constant_folding=True,  # Optymalizacja modelu
                )
                export_paths["onnx"] = onnx_path
            except Exception as e:
                print(f"Błąd podczas eksportu do ONNX: {e}")  # Logowanie błędu
                # Docelowo: logger.error(..., func_name="export_model", file_name="ai/export.py")

        # Eksportuj do TorchScript
        if "torchscript" in formats:
            try:
                # Tryb śledzenia
                trace_path = os.path.join(export_dir, "model_trace.pt")
                traced_model = torch.jit.trace(classifier.model, dummy_input)
                traced_model.save(trace_path)
                export_paths["torchscript_trace"] = trace_path
            except Exception as e:
                print(
                    f"Błąd podczas eksportu do TorchScript (trace): {e}"
                )  # Logowanie błędu
                # Docelowo: logger.error(..., func_name="export_model", file_name="ai/export.py")

            try:
                # Tryb skryptu (bardziej elastyczny)
                script_path = os.path.join(export_dir, "model_script.pt")
                script_model = torch.jit.script(classifier.model)
                script_model.save(script_path)
                export_paths["torchscript_script"] = script_path
            except Exception as e:
                print(
                    f"Błąd podczas eksportu do TorchScript (script): {e}"
                )  # Logowanie błędu
                # Docelowo: logger.error(..., func_name="export_model", file_name="ai/export.py")

        # Kompilacja JIT dla lepszej wydajności (używana w TorchScript)
        if "jit" in formats:
            try:
                jit_path = os.path.join(export_dir, "model_jit.pt")
                # Używamy tej samej logiki co TorchScript (script)
                if "torchscript_script" in export_paths:
                    # Jeśli model skryptowy już istnieje, po prostu go skopiuj lub użyj
                    # Dla uproszczenia, zakładamy że export_paths["torchscript_script"] zawiera poprawną ścieżkę
                    # Można by też po prostu zapisać go ponownie, ale to zbędne
                    shutil.copyfile(export_paths["torchscript_script"], jit_path)
                else:
                    # Jeśli nie był eksportowany jako script, zrób to teraz
                    jit_model = torch.jit.script(classifier.model)
                    jit_model.save(jit_path)
                export_paths["jit"] = jit_path
            except Exception as e:
                print(f"Błąd podczas eksportu do JIT: {e}")  # Logowanie błędu
                # Docelowo: logger.error(..., func_name="export_model", file_name="ai/export.py")

        # Zapisz konfigurację w JSON
        try:
            config_path = os.path.join(export_dir, "model_config.json")
            config = {
                "model_type": classifier.model_type,
                "num_classes": classifier.num_classes,
                "class_names": classifier.class_names,
                "input_size": [224, 224],
                "channels": 3,
                "exported_formats": list(export_paths.keys()),
            }

            with open(config_path, "w") as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            print(f"Błąd podczas zapisywania konfiguracji JSON: {e}")  # Logowanie błędu
            # Docelowo: logger.error(..., func_name="export_model", file_name="ai/export.py")

        # Dodaj przykładowy kod użycia
        if include_sample_code:
            try:
                example_code = """
from image_classifier import ImageClassifier

# Załaduj model
classifier = ImageClassifier(weights_path='model.pt')

# Klasyfikacja pojedynczego obrazu
result = classifier.predict('sciezka/do/obrazu.jpg')
print(f"Klasa: {result['class_name']}")
print(f"Pewność: {result['confidence']:.2f}")

# Klasyfikacja wielu obrazów
image_paths = ['obraz1.jpg', 'obraz2.jpg', 'obraz3.jpg']
results = classifier.batch_predict(image_paths)
for i, res in enumerate(results):
    print(f"Obraz {i+1}: {res['class_name']} ({res['confidence']:.2f})")
"""
                with open(os.path.join(export_dir, "example.py"), "w") as f:
                    f.write(example_code)
            except Exception as e:
                print(
                    f"Błąd podczas zapisywania pliku example.py: {e}"
                )  # Logowanie błędu
                # Docelowo: logger.error(..., func_name="export_model", file_name="ai/export.py")

        # Utwórz README.md
        try:
            readme_content = f"""# Model klasyfikacji obrazów

Ten model został wytrenowany do klasyfikacji obrazów w następujących kategoriach:
{json.dumps(classifier.class_names, indent=4)}

## Instalacja

1. Zainstaluj wymagane pakiety:
```bash
pip install torch torchvision pillow numpy
```

2. Skopiuj pliki modelu do swojego projektu.

## Dostępne formaty

- `model.pt` - Model w formacie PyTorch
- `model.onnx` - Model w formacie ONNX (współpracuje z różnymi frameworkami)
- `model_config.json` - Konfiguracja modelu

## Użycie

Zobacz plik `example.py` dla przykładowego kodu użycia.
"""

            with open(os.path.join(export_dir, "README.md"), "w") as f:
                f.write(readme_content)
        except Exception as e:
            print(f"Błąd podczas zapisywania pliku README.md: {e}")  # Logowanie błędu
            # Docelowo: logger.error(..., func_name="export_model", file_name="ai/export.py")

        return export_paths

    except Exception as main_export_error:
        error_msg = f"Krytyczny błąd podczas eksportu modelu: {main_export_error}"
        print(error_msg)
        # Docelowo:
        # logger.error(error_msg, func_name="export_model", file_name="ai/export.py")
        return None  # Zwróć None w przypadku krytycznego błędu
