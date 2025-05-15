import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms import AutoAugment, AutoAugmentPolicy, TrivialAugmentWide


def get_default_transforms(config=None):
    """
    Zwraca standardowe przekształcenia dla obrazów używane w treningu.

    Args:
        config (dict, optional): Słownik konfiguracyjny. Może zawierać klucz 'image_size'
                                 (krotka (szerokość, wysokość)). Domyślnie None.

    Returns:
        Obiekt transforms.Compose z przekształceniami
    """
    image_size = (224, 224)  # Domyślny rozmiar
    if config and "image_size" in config:
        image_size = tuple(config["image_size"])

    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def get_augmentation_transforms(config=None):
    """
    Zwraca przekształcenia z augmentacją danych dla treningu.
    Uwaga: Wiele parametrów augmentacji (np. rotacja, ColorJitter) jest tutaj
    ustawionych na stałe. Dla pełnej konfiguracji zobacz get_extended_augmentation_transforms.

    Args:
        config (dict, optional): Słownik konfiguracyjny. Może zawierać klucz 'image_size'
                                 (krotka (szerokość, wysokość)). Domyślnie None.


    Returns:
        Obiekt transforms.Compose z przekształceniami
    """
    image_size = (224, 224)  # Domyślny rozmiar
    if config and "image_size" in config:
        image_size = tuple(config["image_size"])

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.7),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(25),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
        ]
    )


def get_extended_augmentation_transforms(image_size=(224, 224), params=None):
    """
    Optymalizacja: Uproszczenie logiki konstruowania listy transformacji.
    """
    if params is None:
        params = {}

    # Domyślne wartości z dykcjonariuszem params.get()
    brightness = params.get("brightness", 0.2)
    contrast = params.get("contrast", 0.2)
    saturation = params.get("saturation", 0.2)
    hue = params.get("hue", 0.1)
    rotation = params.get("rotation", 15)
    horizontal_flip = params.get("horizontal_flip", True)
    vertical_flip = params.get("vertical_flip", False)
    shift = params.get("shift", 0.1)
    zoom = params.get("zoom", 0.1)
    grayscale = params.get("grayscale", False)
    perspective = params.get("perspective", False)

    # Prosta konstrukcja listy transformacji
    transform_list = [
        transforms.RandomResizedCrop(image_size, scale=(1.0 - zoom, 1.0 + zoom)),
    ]

    # Dodawanie opcjonalnych transformacji
    if horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    if vertical_flip:
        transform_list.append(transforms.RandomVerticalFlip(p=0.5))

    if rotation > 0:
        transform_list.append(transforms.RandomRotation(rotation))

    if shift > 0:
        transform_list.append(transforms.RandomAffine(0, translate=(shift, shift)))

    if params.get("trivialaugment", {}).get("use", False):
        transform_list.append(TrivialAugmentWide())

    if params.get("autoaugment", {}).get("use", False):
        transform_list.append(AutoAugment(AutoAugmentPolicy.IMAGENET))

    if params.get("randaugment", {}).get("use", False):
        transform_list.append(
            transforms.RandAugment(
                num_ops=params.get("randaugment", {}).get("n", 2),
                magnitude=params.get("randaugment", {}).get("m", 9),
            )
        )

    # Dodaj ColorJitter
    transform_list.append(
        transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )
    )

    if grayscale:
        transform_list.append(transforms.RandomGrayscale(p=0.1))

    if perspective:
        transform_list.append(transforms.RandomPerspective(distortion_scale=0.2, p=0.5))

    # Standardowe końcowe transformacje
    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return transforms.Compose(transform_list)


def preprocess_image(image_path, transform=None):
    """
    Wstępnie przetwarza obraz do formatu wymaganego przez model.

    Args:
        image_path: Ścieżka do pliku obrazu
        transform: Opcjonalne przekształcenia do zastosowania

    Returns:
        Przekształcony tensor PyTorch
    """
    # Wczytaj obraz
    image = Image.open(image_path).convert("RGB")

    # Jeśli nie podano przekształceń, użyj domyślnych
    if transform is None:
        transform = get_default_transforms()

    # Zastosuj przekształcenia
    return transform(image)


def batch_preprocess_images(image_paths, transform=None, batch_size=16):
    """
    Wstępnie przetwarza wiele obrazów w trybie wsadowym.

    Args:
        image_paths: Lista ścieżek do plików obrazów
        transform: Opcjonalne przekształcenia do zastosowania
        batch_size: Rozmiar wsadu do przetwarzania

    Returns:
        Lista przekształconych tensorów PyTorch
    """
    results = []

    # Przetwarzanie wsadowe
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_results = [preprocess_image(path, transform) for path in batch_paths]
        results.extend(batch_results)

    return results
