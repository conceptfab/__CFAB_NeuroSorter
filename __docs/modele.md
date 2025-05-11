Warianty ResNet
ResNet (Residual Network) została wprowadzona w 2015 roku i ma kilka kluczowych wariantów:

ResNet-18: Najprostsza wersja z 18 warstwami
ResNet-34: Średnia wersja z 34 warstwami
ResNet-50: Najpopularniejsza wersja z 50 warstwami i około 23-25 milionami parametrów
ResNet-101: Głębsza wersja ze 101 warstwami
ResNet-152: Najgłębsza standardowa wersja ze 152 warstwami

ResNet-50-vd: Zmodyfikowana wersja z ulepszonym blokiem początkowym (Stem block)
ResNetV2: Ulepszona wersja z modyfikacją kolejności warstw batch normalization, aktywacji i konwolucji
ResNet-D: Wersja z ulepszonym downsamplingiem
ResNeXt: Wersja ResNet używająca kardynalności (grupowanych konwolucji)

Vision Transformer to stosunkowo nowa architektura wprowadzona w 2020 roku:
ViT-Base (ViT-B): Standardowy model z 12 blokami transformerowymi, 768 wymiarami ukrytymi i 12 głowicami uwagi
ViT-Large (ViT-L): Większy model z 24 blokami, 1024 wymiarami ukrytymi i 16 głowicami
ViT-Huge (ViT-H): Największy model z 32 blokami, 1280 wymiarami ukrytymi i większą liczbą głowic

Swin Transformer: Hierarchiczna wersja ViT używająca przesuwanych okien uwagi
SwinV2: Ulepszony model Swin z lepszą skalowalnością
CrossViT: Model wykorzystujący uwagę krzyżową dla klasyfikacji obrazów
DeiT: Wersja ViT trenowana efektywnie na mniejszych zbiorach danych
DiT: Diffusion Transformer - wersja dla generatywnych modeli dyfuzyjnych

Warianty MobileNet
MobileNet został zaprojektowany dla urządzeń mobilnych i ma kilka generacji:

MobileNetV1: Pierwsza wersja oparta na depthwise separable convolutions
MobileNetV2: Druga generacja wprowadzająca inverted residual block i linear bottlenecks
MobileNetV3: Trzecia generacja, dostępna w dwóch wariantach:

MobileNetV3-Small: Mniejszy i szybszy, ale mniej dokładny
MobileNetV3-Large: Większy i dokładniejszy, ale wolniejszy

MobileNetV4: Najnowsza generacja z różnymi podwariantami:

MobileNetV4-Conv-Small: Mniejsza wersja z naciskiem na efektywność
MobileNetV4-Conv-Small-050: Wersja o zmniejszonych wymiarach (0.5x)

Każdy model MobileNet może być dodatkowo skalowany za pomocą parametru alfa (mnożnik szerokości), który kontroluje liczbę filtrów w każdej warstwie. Na przykład:

MobileNetV3-Large-100: Standardowa wersja (α=1.0)
MobileNetV3-Large-150d: Wersja o zwiększonej szerokości (α=1.5)
