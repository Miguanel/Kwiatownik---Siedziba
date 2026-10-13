"""Sieci neuronowe Siedziby: male, wlasne modele, ktore przejmuja czesc pracy modeli LLM.

Idea (destylacja wiedzy, Hinton i in. 2015): odpowiedzi LLM zapisane w bazie sa etykietami "nauczyciela".
Mala siec (enkoder tekstu + glowica MLP) uczy sie ich i odpowiada w milisekundach, bez limitow API.

Warstwy:
  encoders.py  - zamiana tekstu na wektor: multilingual-e5-small (ONNX, transformer) albo n-gramy znakow
                 (fastText-podobny "worek n-gramow", dziala od razu, bez pobierania czegokolwiek)
  mlp.py       - glowica MLP w numpy: AdamW, wagi klas, wygladzanie etykiet, wczesne zatrzymanie,
                 kalibracja temperatura (Guo i in. 2017)
  evaluate.py  - walidacja krzyzowa z grupowaniem (roslina / zrodlo - bez przecieku), F1, ECE,
                 krzywa pokrycie-trafnosc i dobor progu pewnosci (klasyfikacja selektywna)
  tasks.py     - zadania: rodzaj strony, typ przepisu, sekcja i czesc rosliny informacji
  train.py     - trening wersji, porownanie z obecna (mistrz / pretendent), zapis artefaktow
  runtime.py   - bramka pewnosci w potoku: wylaczona / cien / aktywna, dziennik kazdej decyzji
"""
