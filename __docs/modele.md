Kluczowa idea EfficientNet:

Zamiast skalować tylko jeden wymiar sieci (głębokość, szerokość lub rozdzielczość wejściową) niezależnie, jak to robiono wcześniej, autorzy EfficientNet zaproponowali metodę jednoczesnego i zrównoważonego skalowania wszystkich trzech wymiarów za pomocą jednego współczynnika, nazwanego Φ (Phi).

EfficientNet-B0 to model bazowy, najmniejszy i najszybszy.

EfficientNet-B1 do B7 to modele coraz większe i potężniejsze, uzyskane przez zwiększanie współczynnika Φ.

Co konkretnie się zmienia wraz ze wzrostem "B" (od B0 do B7):

Głębokość sieci (Depth):

Liczba warstw w sieci rośnie. Głębsze sieci mogą uczyć się bardziej złożonych cech.

Przykład: B0 ma mniej bloków MBConv niż B7.

Szerokość sieci (Width):

Liczba kanałów (filtrów) w każdej warstwie konwolucyjnej rośnie. Szersze sieci mogą uczyć się bardziej zróżnicowanych cech na każdym poziomie.

Przykład: Warstwy w B7 mają więcej kanałów niż odpowiadające im warstwy w B0.

Rozdzielczość obrazu wejściowego (Resolution):

Modele są trenowane i oczekują obrazów o wyższej rozdzielczości. Wyższa rozdzielczość pozwala modelowi dostrzec drobniejsze detale.

Przykład:

B0: 224x224

B1: 240x240

B2: 260x260

B3: 300x300

B4: 380x380

B5: 456x456

B6: 528x528

B7: 600x600

Konsekwencje tych zmian:

Cecha	EfficientNet-B0 (mniejsze B)	EfficientNet-B7 (większe B)
Dokładność	Niższa	Wyższa (generalnie)
Liczba parametrów	Mniejsza	Znacznie większa
Koszt obliczeniowy (FLOPs)	Niski	Bardzo wysoki
Prędkość inferencji	Szybka	Wolniejsza
Prędkość trenowania	Szybsza	Wolniejsza
Zapotrzebowanie na pamięć (RAM/VRAM)	Niskie	Wysokie
Jak to wygląda w praktyce (przybliżone wartości dla ImageNet):

Model	Rozdzielczość	Parametry (mln)	Top-1 Accuracy (ImageNet)
EfficientNet-B0	224x224	5.3	77.1%
EfficientNet-B1	240x240	7.8	79.1%
EfficientNet-B2	260x260	9.2	80.1%
EfficientNet-B3	300x300	12	81.6%
EfficientNet-B4	380x380	19	82.9%
EfficientNet-B5	456x456	30	83.6%
EfficientNet-B6	528x528	43	84.0%
EfficientNet-B7	600x600	66	84.3%
(Dokładne wartości mogą się nieznacznie różnić w zależności od implementacji i frameworka).

Podsumowując:

Wybierz B0-B2: Jeśli zależy Ci na szybkości, małym rozmiarze modelu i ograniczonych zasobach obliczeniowych (np. aplikacje mobilne, szybkie prototypowanie).

Wybierz B3-B5: Jeśli szukasz dobrego kompromisu między dokładnością a zasobami. Są to często bardzo dobre wybory dla wielu praktycznych zastosowań.

Wybierz B6-B7: Jeśli priorytetem jest maksymalna dokładność i masz dostęp do odpowiednio mocnego sprzętu do trenowania i inferencji.

Pamiętaj, że nawet przy transfer learningu, większe modele będą wymagały więcej danych i dłuższego czasu trenowania, aby w pełni wykorzystać swój potencjał. Zawsze warto zacząć od mniejszego modelu i stopniowo go zwiększać, jeśli wyniki nie są satysfakcjonujące i zasoby na to pozwalają.


ResNet (Residual Networks)
Kluczowa idea ResNet:

Głównym problemem, który ResNet rozwiązuje, jest degradacja dokładności (degradation problem) w bardzo głębokich sieciach neuronowych. Kiedy sieci stają się coraz głębsze, trenowanie ich staje się trudniejsze, a dokładność na zbiorze treningowym (i walidacyjnym) zaczyna spadać, mimo dodawania kolejnych warstw. ResNet wprowadza "połączenia resztkowe" (residual connections) lub "skróty" (skip connections). Pozwalają one gradientom płynąć łatwiej przez sieć podczas propagacji wstecznej, umożliwiając trenowanie znacznie głębszych modeli. Zamiast uczyć się bezpośredniego mapowania H(x), warstwy uczą się mapowania resztkowego F(x) = H(x) - x, a następnie wynik to F(x) + x. Jeśli optymalne jest mapowanie tożsamościowe, warstwy mogą łatwo nauczyć się F(x) = 0.

Jak różnią się poszczególne warianty ResNet (np. ResNet-18, -34, -50, -101, -152):

Główną różnicą między wariantami ResNet jest głębokość sieci, czyli liczba warstw konwolucyjnych i w pełni połączonych. Drugą istotną różnicą jest typ bloku resztkowego:

Głębokość (Liczba warstw):

Liczba w nazwie (np. 18, 34, 50) odnosi się do liczby "ważonych" warstw (konwolucyjnych i w pełni połączonych).

ResNet-18: 18 warstw.

ResNet-34: 34 warstwy.

ResNet-50: 50 warstw.

ResNet-101: 101 warstw.

ResNet-152: 152 warstwy.

Typ Bloku Resztkowego:

Basic Block (Blok Podstawowy): Używany w ResNet-18 i ResNet-34. Składa się z dwóch warstw konwolucyjnych 3x3.

Input -> Conv 3x3 -> BN -> ReLU -> Conv 3x3 -> BN -> (+) -> ReLU -> Output
  |                                                  ^
  |--------------------------------------------------| (skip connection)
Use code with caution.
Bottleneck Block (Blok "Wąskiego Gardła"): Używany w ResNet-50, ResNet-101 i ResNet-152. Jest bardziej wydajny obliczeniowo dla głębszych sieci. Składa się z sekwencji warstw konwolucyjnych 1x1, 3x3, 1x1. Warstwa 1x1 najpierw redukuje liczbę kanałów (wymiarowość), następnie konwolucja 3x3 operuje na mniejszej liczbie kanałów, a na końcu druga warstwa 1x1 przywraca pierwotną liczbę kanałów.

Input -> Conv 1x1 -> BN -> ReLU -> Conv 3x3 -> BN -> ReLU -> Conv 1x1 -> BN -> (+) -> ReLU -> Output
  |                                                                       ^
  |-----------------------------------------------------------------------| (skip connection)
Use code with caution.
Ten projekt pozwala na budowanie głębszych sieci bez proporcjonalnie dużego wzrostu liczby parametrów i obliczeń w porównaniu do używania tylko bloków podstawowych.

Konsekwencje tych zmian:

Cecha	ResNet-18/34 (płytsze)	ResNet-50/101/152 (głębsze)
Dokładność	Niższa (potencjalnie)	Wyższa (potencjalnie)
Liczba parametrów	Mniejsza	Większa
Koszt obliczeniowy (FLOPs)	Niski	Wyższy
Prędkość inferencji	Szybsza	Wolniejsza
Prędkość trenowania	Szybsza	Wolniejsza
Zapotrzebowanie na pamięć (RAM/VRAM)	Niskie	Wyższe
Typ bloku	Basic	Bottleneck
Przybliżone wartości dla ImageNet:

Model	Liczba Warstw	Typ Bloku	Parametry (mln)	Top-1 Accuracy (ImageNet)
ResNet-18	18	Basic	~11.7	~69.8%
ResNet-34	34	Basic	~21.8	~73.3%
ResNet-50	50	Bottleneck	~25.6	~76.1% (z wagami V2 ~80.8%)
ResNet-101	101	Bottleneck	~44.5	~77.4% (z wagami V2 ~81.8%)
ResNet-152	152	Bottleneck	~60.2	~78.3% (z wagami V2 ~82.2%)
(Dokładność może się różnić w zależności od wersji wag torchvision, np. V1 vs V2)				
Podsumowując ResNet:
Wybierasz wariant ResNet głównie na podstawie kompromisu między pożądaną dokładnością a dostępnymi zasobami obliczeniowymi. ResNet-50 jest bardzo popularnym i solidnym wyborem jako punkt wyjściowy.

ConvNeXt
Kluczowa idea ConvNeXt:

ConvNeXt to rodzina architektur konwolucyjnych (CNN), która została zaprojektowana poprzez "modernizację" standardowych CNN (takich jak ResNet), czerpiąc inspiracje z projektów Transformerów Wizyjnych (ViT). Autorzy stopniowo modyfikowali architekturę ResNet, wprowadzając zmiany w makro-projekcie, bloku ResNeXt (grupowane konwolucje), odwróconym bloku "bottleneck" (jak w MobileNetV2), większych rozmiarach kerneli konwolucyjnych, oraz różnych drobnych zmianach w warstwach normalizacji i funkcjach aktywacji. Celem było sprawdzenie, czy "czyste" CNN mogą osiągnąć wydajność porównywalną lub lepszą od Transformerów na zadaniach wizyjnych, przy zachowaniu prostoty i efektywności konwolucji.

Jak różnią się poszczególne warianty ConvNeXt (np. Tiny, Small, Base, Large, XLarge):

Warianty ConvNeXt są skalowane w sposób systematyczny, podobnie do EfficientNet, ale skupiając się głównie na szerokości (liczbie kanałów / wymiarze osadzenia) oraz głębokości (liczbie bloków w poszczególnych etapach).

Szerokość (Channel Counts / Embed Dim C):

Liczba kanałów na wyjściu pierwszego "stem" i w kolejnych etapach.

ConvNeXt-T (Tiny): C = (96, 192, 384, 768)

ConvNeXt-S (Small): C = (96, 192, 384, 768) - taka sama jak Tiny, ale więcej bloków

ConvNeXt-B (Base): C = (128, 256, 512, 1024)

ConvNeXt-L (Large): C = (192, 384, 768, 1536)

ConvNeXt-XL (XLarge): C = (256, 512, 1024, 2048)

Głębokość (Number of Blocks per Stage B):

Liczba bloków ConvNeXt w każdym z 4 etapów (stages).

ConvNeXt-T: B = (3, 3, 9, 3)

ConvNeXt-S: B = (3, 3, 27, 3)

ConvNeXt-B: B = (3, 3, 27, 3)

ConvNeXt-L: B = (3, 3, 27, 3)

ConvNeXt-XL: B = (3, 3, 27, 3)

Jak widać, dla S, B, L, XL liczba bloków jest taka sama, a skalowanie odbywa się głównie przez szerokość. Tiny ma mniej bloków w trzecim etapie.

Rozdzielczość wejściowa: Standardowo trenowane na 224x224 lub 384x384. Większe modele często korzystają z większej rozdzielczości dla lepszych wyników.

Konsekwencje tych zmian:

Cecha	ConvNeXt-T (mniejsze)	ConvNeXt-L/XL (większe)
Dokładność	Niższa	Wyższa
Liczba parametrów	Mniejsza	Znacznie większa
Koszt obliczeniowy (FLOPs)	Niski	Bardzo wysoki
Prędkość inferencji	Szybsza	Wolniejsza
Prędkość trenowania	Szybsza	Wolniejsza
Zapotrzebowanie na pamięć (RAM/VRAM)	Niskie	Wysokie
Przybliżone wartości dla ImageNet-1K (224x224):

Model	Parametry (mln)	Top-1 Accuracy (ImageNet-1K)
ConvNeXt-T (Tiny)	~28.6	~82.1%
ConvNeXt-S (Small)	~50.2	~83.1%
ConvNeXt-B (Base)	~88.6	~83.8% (do ~85.8% z pre-treningiem na IN-22K)
ConvNeXt-L (Large)	~197.8	~84.3% (do ~86.8% z pre-treningiem na IN-22K)
(Wyniki dla większych modeli ConvNeXt (Base, Large) znacząco zyskują na pre-treningu na większym zbiorze ImageNet-22K, a następnie fine-tuningu na ImageNet-1K. torchvision dostarcza głównie wagi pre-trenowane na ImageNet-1K).		
Podsumowując ConvNeXt:
Warianty ConvNeXt oferują bardzo wysoką wydajność, często przewyższając ResNety i EfficientNety o podobnej liczbie parametrów. Wybór zależy od kompromisu między dokładnością a zasobami. ConvNeXt-Tiny i ConvNeXt-Small to doskonałe punkty startowe oferujące silną wydajność przy rozsądnych kosztach. Większe modele są dla tych, którzy dążą do state-of-the-art i dysponują odpowiednimi zasobami.



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
