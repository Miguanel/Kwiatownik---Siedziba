"""Pora roku i kalendarz ludowy - tlo dla postow na fanpage (bez LLM, deterministycznie)."""
from dataclasses import dataclass, field
from datetime import date

MONTHS = {
    1: ("styczeń", "stycznia", "zima", "Głęboka zima: mróz, śnieg, przyroda śpi. Czas na suszone zioła, napary "
        "i nalewki zrobione latem, przegląd zapasów i planowanie ogródka."),
    2: ("luty", "lutego", "zima", "Przedwiośnie się szykuje: dni dłuższe, pierwsze odwilże, pąki na drzewach "
        "nabrzmiewają. Czas cięcia drzew i krzewów, zanim ruszą soki."),
    3: ("marzec", "marca", "przedwiośnie", "Budzi się przyroda: ruszają soki w brzozach, wychodzą pierwsze "
        "pędy, kwitnie leszczyna i podbiał. Pierwsze wiosenne zioła i porządki w ogrodzie."),
    4: ("kwiecień", "kwietnia", "wiosna", "Pełna wiosna: młode liście pokrzywy, mniszka i czosnku niedźwiedziego, "
        "kwitną drzewa owocowe. Czas na młode pędy i wiosenne oczyszczanie."),
    5: ("maj", "maja", "wiosna", "Wszystko kwitnie: głóg, bez, mniszek, konwalie. Najlepszy czas na zbiór "
        "kwiatów i młodych liści, majowe wieczory, chrząszcze i słowiki."),
    6: ("czerwiec", "czerwca", "lato", "Początek lata, najdłuższe dni w roku. Zioła są najpełniejsze w olejki - "
        "czas zbiorów ziół świętojańskich, kwiatów lipy i bzu."),
    7: ("lipiec", "lipca", "lato", "Pełnia lata, upały i burze. Kwitnie lipa, dziurawiec, krwawnik; dojrzewają "
        "jagody i poziomki. Suszenie ziół na zimę idzie pełną parą."),
    8: ("sierpień", "sierpnia", "lato", "Koniec lata, żniwa. Dojrzewają owoce i nasiona, dni robią się krótsze. "
        "Czas bukietów ziół na Matki Boskiej Zielnej i pierwszych przetworów."),
    9: ("wrzesień", "września", "jesień", "Wczesna jesień, babie lato. Dojrzewają owoce bzu, jarzębiny, głogu "
        "i dzikiej róży, grzyby w lesie. Wykopki i robienie zapasów."),
    10: ("październik", "października", "jesień", "Złota polska jesień: liście żółkną i opadają, poranne mgły "
         "i przymrozki. Zbiór korzeni, owoców po pierwszym przymrozku (tarnina, dzika róża), nalewki i syropy na zimę."),
    11: ("listopad", "listopada", "późna jesień", "Szaro, mokro i wietrznie, przyroda szykuje się do snu. Czas "
         "rozgrzewających naparów, syropów na przeziębienie i okrywania roślin przed zimą."),
    12: ("grudzień", "grudnia", "zima", "Najkrótsze dni w roku, adwent i święta. Gałązki w wazonie, przyprawy, "
         "rozgrzewające napary i opowieści przy piecu."),
}

# (miesiac, dzien_od, dzien_do, opis) - tradycje i swieta zwiazane z roslinami, o ktorych dziadek moze wspomniec
FOLK_DAYS = [
    (1, 6, 6, "Trzech Króli (6 stycznia) - święcenie kredy i kadzidła"),
    (2, 2, 2, "Matki Boskiej Gromnicznej (2 lutego) - wedle ludu połowa zimy"),
    (3, 21, 21, "pierwszy dzień wiosny (21 marca) - topienie Marzanny"),
    (6, 21, 24, "Noc Kupały / Sobótka (21-24 czerwca) - wianki, ziele świętojańskie, szukanie kwiatu paproci"),
    (8, 15, 15, "Matki Boskiej Zielnej (15 sierpnia) - święcenie bukietów ziół, zbóż i owoców"),
    (9, 8, 8, "Matki Boskiej Siewnej (8 września) - siew ozimin, odlot bocianów"),
    (9, 1, 30, "babie lato - nitki pajęczyn na łąkach, ciepłe dni na początku jesieni"),
    (10, 1, 15, "babie lato i złota polska jesień"),
    (11, 1, 2, "Wszystkich Świętych i Zaduszki (1-2 listopada) - wrzos i chryzantemy na grobach"),
    (11, 11, 11, "świętego Marcina (11 listopada)"),
    (11, 29, 30, "Andrzejki (29-30 listopada) - wróżby, lanie wosku"),
    (12, 4, 4, "świętej Barbary (4 grudnia) - gałązki wiśni wstawione do wody mają zakwitnąć na Wigilię"),
    (12, 24, 24, "Wigilia - sianko pod obrusem, dwanaście potraw"),
]


@dataclass
class Season:
    day: date
    month: int
    month_name: str        # "październik"
    month_gen: str         # "października"
    season: str            # "jesień"
    phase: str             # "początek" | "połowa" | "koniec"
    description: str
    folk: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.phase} {self.month_gen} ({self.season})"


def _folk_near(d: date, days: int = 7) -> list[str]:
    out = []
    for m, d1, d2, text in FOLK_DAYS:
        for y in (d.year - 1, d.year, d.year + 1):
            try:
                start, end = date(y, m, d1), date(y, m, d2)
            except ValueError:
                continue
            if (start - d).days <= days and (d - end).days <= 3:  # do tygodnia przed i 3 dni po
                out.append(text)
                break
    return out


def season_for(d: date) -> Season:
    name, gen, season, desc = MONTHS[d.month]
    phase = "początek" if d.day <= 10 else "połowa" if d.day <= 20 else "koniec"
    return Season(day=d, month=d.month, month_name=name, month_gen=gen, season=season, phase=phase,
                  description=desc, folk=_folk_near(d))
