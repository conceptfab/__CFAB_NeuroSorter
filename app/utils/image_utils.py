import numpy as np
from PIL import Image


def resize_image(image, size=(224, 224), keep_aspect_ratio=True):
    """
    Zmienia rozmiar obrazu zachowując proporcje.

    Args:
        image: Obraz PIL.Image
        size: Krotka (szerokość, wysokość)
        keep_aspect_ratio: Czy zachować proporcje

    Returns:
        Obraz PIL.Image o nowym rozmiarze
    """
    if keep_aspect_ratio:
        width, height = image.size
        scale = min(size[0] / width, size[1] / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return image.resize((new_width, new_height), Image.LANCZOS)
    else:
        return image.resize(size, Image.LANCZOS)


def normalize_image(image):
    """
    Normalizuje obraz do zakresu [0, 1].

    Args:
        image: Obraz PIL.Image

    Returns:
        Znormalizowany obraz jako tablica numpy
    """
    # Konwertuj do tablicy numpy
    img_array = np.array(image)

    # Normalizuj do zakresu [0, 1]
    if img_array.dtype == np.uint8:
        img_array = img_array.astype(np.float32) / 255.0

    return img_array
