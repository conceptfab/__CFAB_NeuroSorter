Podsumowanie
Z profilu sprzętowego pobierane są tylko 3 parametry:

recommended_batch_size (domyślnie 32)
recommended_workers (domyślnie 0)
use_mixed_precision (domyślnie False)

Wszystkie pozostałe parametry treningu są przekazywane bezpośrednio do funkcji lub generowane wewnętrznie przez inne mechanizmy, jak opisywałem wcześniej. Nie ma kompleksowego "profilu zadania" czy "profilu treningu", który zawierałby wszystkie ustawienia w jednym miejscu.