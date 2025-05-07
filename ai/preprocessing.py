import numpy as np
import torchvision.transforms as transforms
from PIL import Image


def get_default_transforms(image_size=(224, 224)):
    """
    Zwraca standardowe przekształcenia dla obrazów używane w treningu.

    Args:
        image_size: Krotka (szerokość, wysokość)

    Returns:
        Obiekt transforms.Compose z przekształceniami
    """
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def get_augmentation_transforms(image_size=(224, 224)):
    """
    Zwraca przekształcenia z augmentacją danych dla treningu.

    Args:
        image_size: Krotka (szerokość, wysokość)

    Returns:
        Obiekt transforms.Compose z przekształceniami
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def get_extended_augmentation_transforms(image_size=(224, 224), params=None):
    """
    Zwraca rozszerzone przekształcenia z konfigurowalnymi parametrami.

    Args:
        image_size: Krotka (szerokość, wysokość)
        params: Słownik z parametrami augmentacji

    Returns:
        Obiekt transforms.Compose z przekształceniami
    """
    if params is None:
        params = {}

    # Domyślne wartości
    brightness = params.get("brightness", 0.2)
    contrast = params.get("contrast", 0.2)
    saturation = params.get("saturation", 0.2)
    hue = params.get("hue", 0.1)
    rotation = params.get("rotation", 15)
    vertical_flip = params.get("vertical_flip", False)
    grayscale = params.get("grayscale", False)
    perspective = params.get("perspective", False)

    transform_list = [
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
    ]

    # Dodaj opcjonalne transformacje na podstawie parametrów
    if vertical_flip:
        transform_list.append(transforms.RandomVerticalFlip())

    if rotation > 0:
        transform_list.append(transforms.RandomRotation(rotation))

    transform_list.append(
        transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )
    )

    if grayscale:
        transform_list.append(transforms.RandomGrayscale(p=0.1))

    if perspective:
        transform_list.append(transforms.RandomPerspective(distortion_scale=0.2, p=0.5))

    # Dodaj standardowe transformacje końcowe
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
    errors = []

    # Przetwarzanie wsadowe
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_results = []

        for path in batch_paths:
            try:
                tensor = preprocess_image(path, transform)
                batch_results.append((path, tensor))
            except Exception as e:
                errors.append((path, str(e)))
                continue

        results.extend([tensor for _, tensor in batch_results])

    if errors:
        print(f"Napotkano błędy podczas przetwarzania {len(errors)} obrazów:")
        for path, error in errors[:5]:  # Pokaż pierwsze 5 błędów
            print(f"  - {path}: {error}")
        if len(errors) > 5:
            print(f"  - ... i {len(errors) - 5} więcej")

    return results
