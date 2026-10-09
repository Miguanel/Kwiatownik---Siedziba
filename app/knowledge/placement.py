"""Rozmieszczanie wiedzy z sieci w rozdzialach strony rosliny Kwiatownika (blok "rozmieszczenie").

Scalanie (merge.py) wplata punkty z sieci w TEKST istniejacych podrozdzialow. Czesc punktow tam nie trafia:
nie ma dla nich podrozdzialu (wystepowanie, historia, nazwy ludowe, pasza, zapylanie...), LLM uznal, ze "nie
pasuja" do tekstu, albo regula ich nie przydzielila. Do tej pory zostawaly w osobnej zakladce "Wiedza z sieci".
Ten modul wyciaga je stamtad i ustawia w konkretnych miejscach strony:
  - w istniejacym podrozdziale (np. "Natura i siedlisko -> Wymagania siedliskowe", "Surowce -> Owoce") jako
    lista "z sieci" pod tekstem Kwiatownika,
  - albo jako NOWY podrozdzial rozdzialu, do ktorego naleza (np. "Wiedza tajemna -> Historia i nazwa",
    "Cykl zycia -> Kwitnienie, zapylanie i rozsiewanie").

Algorytm (bez LLM dziala sam, LLM tylko poprawia):
1. punkty: z wiedza.sekcje, ktorych informacje nie sa uzyte w aktualnym bloku "scalone";
2. ocena kazdego miejsca z katalogu MIEJSCA: sekcja wiedzy (waga), slowa kluczowe w tekscie, czesc rosliny
   (podrozdzial tej czesci w "Surowcach"), kara za podrozdzial, do ktorego scalanie juz ten punkt odrzucilo;
3. spojnosc sekcji: punkty jednej sekcji trzymaja sie razem, gdy wiekszosc trafia w jedno miejsce, a dla reszty
   to miejsce jest niewiele gorsze (sekcje "wyciagane" w calosci, nie rozrzucane po stronie);
4. LLM (opcjonalnie) dostaje liste miejsc i punkty z propozycja reguly - moze zmienic miejsce, ale tylko na
   miejsce z listy; zla odpowiedz = zostaje regula;
5. punkt prawie identyczny z tekstem juz scalonym w tym miejscu nie jest powtarzany (duplikat), a podobne punkty
   w jednym miejscu lacza sie (zrodla razem);
6. punkt, ktory nigdzie nie pasuje (ocena ponizej progu i brak miejsca domyslnego sekcji), zostaje w zakladce
   "Wiedza z sieci".

Blok w pliku rosliny (reczne pola nietkniete, tekst punktow bez zmian - przepisany z wiedza.sekcje):
  "rozmieszczenie": {"wersja": 1, "zaktualizowano": "2026-10-09", "jak": "reguly" | "llm:groq/...",
      "wejscie": "<odcisk>",
      "wstawki": [{"miejsce": "tajemna/historia", "rozdzial": "Wiedza tajemna", "tytul": "Historia i nazwa",
                   "ikona": "ra-hourglass", "nowy": true,
                   "punkty": [{"tekst", "zrodla": [nr], "fakty": ["s12"], "sekcja", "czesc"}]}],
      "duplikaty": ["s7"]}
Kwiatownik2 (app/utils/merged.py) pokazuje wstawke w srodku podrozdzialu "miejsce", a gdy takiego podrozdzialu na
stronie nie ma - jako nowy podrozdzial na koncu rozdzialu.
"""
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date

from app.knowledge.schema import filled, get_path, part_key_for
from app.knowledge.sections import norm_part, norm_text, similar

VERSION = 1
MIN_SCORE = 2.0          # ponizej - miejsce domyslne sekcji (FALLBACK) albo zostaje w "Wiedzy z sieci"
KW_WEIGHT = 1.2          # za kazde trafione slowo kluczowe (najwyzej MAX_HITS)
STRONG_WEIGHT = 2.5      # slowo jednoznaczne dla miejsca ("mylony z", "chwast", "legenda"...) - raz
MAX_HITS = 3
REJECTED = 2.0           # kara: scalanie mialo ten punkt w tym podrozdziale i go nie uzylo
COHESION = 0.6           # tyle punktow sekcji w jednym miejscu -> reszta sekcji dolacza...
COHESION_GAP = 1.0       # ...jesli to miejsce jest dla niej gorsze najwyzej o tyle
DUPLICATE = 0.6          # podobienstwo do zdania juz scalonego w tym miejscu -> punkt pomijany
SAME_POINT = 0.5         # podobienstwo dwoch punktow jednego miejsca -> jeden punkt, zrodla razem
MAX_TEXT = 700
LLM_BATCH = 30

CHAPTERS = {
    "natura": "Natura i siedlisko",
    "rozpoznawanie": "Rozpoznawanie w terenie",
    "surowce": "Surowce i zbiory",
    "zastosowanie": "Praktyczne zastosowanie",
    "bezpieczenstwo": "Bezpieczeństwo i interakcje",
    "cykl": "Cykl życia i ekosystem",
    "tajemna": "Wiedza tajemna",
}
CHAPTER_ICONS = {"natura": "ra-moon-sun", "rozpoznawanie": "ra-eye-monster", "surowce": "ra-fizzing-flask",
                 "zastosowanie": "ra-anvil", "bezpieczenstwo": "ra-bleeding-eye", "cykl": "ra-cycle",
                 "tajemna": "ra-scroll"}


@dataclass(frozen=True)
class Place:
    id: str                          # "rozdzial/podrozdzial"
    title: str
    icon: str                        # klasa RPG Awesome (Kwiatownik)
    builtin: bool                    # podrozdzial jest w szablonie Kwiatownika (wstawka idzie do jego srodka)
    prior: dict = field(default_factory=dict)     # sekcja wiedzy -> waga
    words: tuple = ()                # rdzenie slow (bez polskich znakow, jak norm_text)
    hint: str = ""
    chapter_title: str = ""          # rozdzial wlasny (catalog.py): tytul, ikona, po ktorym rozdziale stoi
    chapter_icon: str = ""
    chapter_after: str = ""
    source: str = "wbudowany"        # wbudowany | uzytkownik | agent

    @property
    def chapter_key(self) -> str:
        return self.id.split("/", 1)[0]

    @property
    def chapter(self) -> str:
        return self.chapter_title or CHAPTERS.get(self.chapter_key, self.chapter_key)

    @property
    def custom_chapter(self) -> bool:
        return self.chapter_key not in CHAPTERS


# Kolejnosc = kolejnosc na stronie rosliny (templates/plant_detail.html w Kwiatownik2)
MIEJSCA: tuple[Place, ...] = (
    Place("natura/profil", "Profil energetyczny", "ra-burning-embers", True,
          {"medycyna_wschodu": 3.0},
          ("smak", "gorzk", "chlodz", "chlodn", "rozgrzew", "meridian", "yin", "yang", "qi ", "zywiol", "kampo",
           "ajurwed", "dosz", "tcm", "medycyn chinsk", "medycynie chinsk"),
          "smak, natura termiczna, meridiany, zywiol, medycyna chinska/kampo/ajurweda"),
    Place("natura/wymagania", "Wymagania siedliskowe", "ra-compass", True,
          {"uprawa": 1.5, "wystepowanie": 1.0},
          ("gleb", "stanowisk", "slonc", "polcien", "cienist", "wilgotn", "mroz", "odczyn", "ph ", "piaszczyst",
           "gliniast", "wapienn", "susz"),
          "stanowisko, gleba, woda, pH, mrozoodpornosc"),
    Place("natura/wystepowanie", "Występowanie i zasięg", "ra-compass", False,
          {"wystepowanie": 3.0},
          ("wystepuj", "wystepowan", "rosnie w", "rosna w", "zasieg", "pochodz", "rodzim", "introdukow",
           "zawleczon", "prowincj", "kontynent", "afryk", "azj", "europ", "ameryk", "chin", "japoni", "kanad",
           "stanach", "region", "wyspach", "gorach", "nizin"),
          "gdzie roslina rosnie na swiecie, pochodzenie, zasieg, zawleczenie"),
    Place("rozpoznawanie/cechy", "Cechy kluczowe", "ra-leaf", True,
          {"opis": 2.0},
          ("lisc", "lodyg", "wlosk", "ksztalt", "barw", "dlugos", "wysokos", "cm ", "mm ", "kora ", "kory ",
           "pien", "zapach", "brzeg", "ulistn"),
          "cechy morfologiczne, po ktorych rozpoznaje sie rosline"),
    Place("rozpoznawanie/pomylki", "Możliwe pomyłki", "ra-skull", True,
          {"bezpieczenstwo": 0.5, "opis": 0.3},
          ("podobn", "przypomina", "syberyjsk"),
          "z czym mozna rosline pomylic i jak odroznic"),
    Place("rozpoznawanie/budowa", "Budowa: kwiaty, owoce i nasiona", "ra-flower", False,
          {"opis": 1.6, "czesci_rosliny": 0.8, "ciekawostki": 0.8},
          ("owoc", "nasion", "nasionk", "rozlup", "torebk", "kotk", "jednopienn", "dwupienn", "precik", "slupk",
           "kwiatostan", "szypul", "rozet", "platk", "kielich", "skrzydelk", "baldach", "koszycz"),
          "budowa kwiatow, owocow i nasion (opis botaniczny organow)"),
    Place("surowce/sklad", "Skład i substancje czynne", "ra-fizzing-flask", False,
          {"sklad": 2.5},
          ("zawier", "substancj", "zwiazk", "witamin", "flawonoid", "saponin", "kumaryn", "garbnik", "olejek",
           "alkaloid", "glikozyd", "kwas ", "kwasy", "mineral"),
          "sklad chemiczny calej rosliny (bez wskazanej czesci)"),
    Place("surowce/inne", "Surowce całej rośliny", "ra-potion", False,
          {"czesci_rosliny": 2.2},
          ("surow", "zbier", "zbior", "susz", "olej", "wyciag", "koncentrat", "herbat"),
          "co sie zbiera z rosliny i co sie z tego robi (bez jednej konkretnej czesci)"),
    Place("zastosowanie/medycyna", "Medycyna", "ra-health", True,
          {"zastosowanie_lecznicze": 2.0, "sklad": 0.8, "medycyna_wschodu": 1.0},
          ("lecz", "terap", "choro", "dolegl", "zapalen", "przeciwzapal", "wspomag", "preparat", "napar", "odwar",
           "wyciag", "kaszel", "goraczk", "rany", "ran ", "oparzen", "trawien", "moczopęd", "moczoped", "lekarst",
           "ziolow", "medycyn"),
          "dzialanie lecznicze calej rosliny, ziololecznictwo, medycyna ludowa"),
    Place("zastosowanie/kuchnia", "Kuchnia", "ra-meat", True,
          {"zastosowanie_lecznicze": 0.5, "historia": 0.3},
          ("jadal", "spozyw", "kuchn", "potraw", "zup", "salat", "syrop", "nalewk", "napoj", "likier", "wino ",
           "wina ", "brandy", "ciast", "przetwor", "dzem", "konfitur", "herbat", "przypraw", "smaz", "gotuj",
           "piwo", "sok "),
          "jadalnosc, kuchnia, napoje, przetwory"),
    Place("zastosowanie/rzemioslo", "Rzemiosło", "ra-hammer", True,
          {"barwienie": 3.0},
          ("barwi", "barwnik", "farb", "drewn", "wlokn", "tkanin", "garbow", "przedz", "wyplat", "mikroskop",
           "rekodziel", "narzedz", "rdzen", "atrament", "indyg"),
          "barwienie, drewno, wlokna, inne rzemiosla i uzytki techniczne"),
    Place("zastosowanie/kosmetyka", "Kosmetyka", "ra-flower", True,
          {"kosmetyka": 3.0},
          ("kosmet", "cer ", "cery", "skor", "wlos", "pielegn", "krem", "maseczk", "mydl"),
          "kosmetyki, pielegnacja skory i wlosow"),
    Place("zastosowanie/ogrod", "Ogród", "ra-pine-tree", True,
          {"uprawa": 2.0},
          ("ogrod", "sadz", "uprawi", "uprawa ", "rozmnaz", "ozdob", "miododaj", "zywoplot", "kompost", "nawoz",
           "skarp", "wydm", "erozj"),
          "uprawa w ogrodzie, rozmnazanie, roslina ozdobna lub miododajna"),
    Place("zastosowanie/gospodarstwo", "Pasza i gospodarstwo", "ra-wooden-sign", False,
          {"uprawa": 0.8},
          ("pasz", "krow", "bydl", "zwierz", "drob", "ptak", "gryzon", "mlecznos", "sera", "serow", "ser ",
           "podpuszczk", "owiec", "owce", "kozy", "koni", "hodowl", "much", "odstrasz", "insektycyd"),
          "pasza dla zwierzat, uzytek w gospodarstwie i hodowli, odstraszanie owadow"),
    Place("bezpieczenstwo/interakcje", "Interakcje z lekami", "ra-potion", True,
          {"bezpieczenstwo": 1.0},
          ("interakc", "lekami", "lekow", "warfaryn", "antykoag", "cytochrom"),
          "interakcje z lekami"),
    Place("bezpieczenstwo/przeciwwskazania", "Przeciwwskazania", "ra-lightning-bolt", True,
          {"bezpieczenstwo": 2.0},
          ("trujac", "toksycz", "zatru", "przeciwwskaz", "ciaz", "uczul", "alerg", "szkodliw", "dawk", "wymiot",
           "biegunk", "podrazn"),
          "toksycznosc, przeciwwskazania, skutki uboczne dla ludzi"),
    Place("bezpieczenstwo/chwast", "Chwast i inwazyjność", "ra-bleeding-eye", False,
          {"bezpieczenstwo": 0.5},
          ("chwast", "inwazyj", "plon", "w uprawach", "uprawach", "szkod", "wektor", "wirus", "zachwaszcz",
           "uciazliw", "zwalcz"),
          "roslina jako chwast lub gatunek inwazyjny, szkody w uprawach"),
    Place("cykl/kalendarz", "Kalendarz", "ra-sickle", True,
          {},
          ("kalendarz", "termin", "miesiac"),
          "kiedy co robic z roslina w ciagu roku"),
    Place("cykl/permakultura", "Permakultura i gildie", "ra-trefoil", True,
          {"uprawa": 0.3},
          ("azot", "okryw", "schronien", "ekosystem", "gildi", "towarzysz", "sasiedztw"),
          "rola w ekosystemie i ogrodzie permakulturowym"),
    Place("cykl/biologia", "Kwitnienie, zapylanie i rozsiewanie", "ra-cycle", False,
          {"ciekawostki": 0.5, "opis": 0.3},
          ("zapyl", "nektar", "pylek", "rozsiew", "rozprzestrz", "roznos", "mrowk", "motyl", "gasienic", "siewk",
           "kielk", "wegetatywn", "plec", "zyja", "zyje", "zycia", "dozyw", "lat ", "samopyl", "samoniewlasciw",
           "kwitnien", "kwitnie", "nasion", "wyrzuc", "ukorzen", "owad", "wyprodukow", "produkuj"),
          "biologia rosliny: kwitnienie, zapylanie, rozsiewanie, rozmnazanie w naturze, dlugosc zycia"),
    Place("tajemna/ciekawostki", "Ciekawostki", "ra-emerald", True,
          {"ciekawostki": 1.5, "kultura": 0.5, "historia": 0.5},
          ("ciekaw", "rekord", "najwiek", "serial", "film", "ksiazk"),
          "inne ciekawostki"),
    Place("tajemna/historia", "Historia i nazwa", "ra-hourglass", False,
          {"historia": 3.0, "nazwy_ludowe": 0.5},
          ("dawniej", "sredniowiecz", "starozyt", "wieku", " xix", "xviii", "xvii", "histor", "egipt", "rzym",
           "grek", "greck", "tradycyj", "lacinsk", "epitet", "etymolog", "oznacza", "pochodzi od", "bach",
           "uzywan", "wykorzystywan"),
          "historia uzycia, etymologia nazwy lacinskiej i polskiej, dawne zastosowania"),
    Place("tajemna/kultura", "Wierzenia i legendy", "ra-crystal-ball", False,
          {"kultura": 3.0},
          ("legend", "wierzen", "wierzon", "obrzed", "judasz", "magi", "czarow", "przesad", "swiet", "chrzesc",
           "symbol", "mit ", "mity", "duch", "bostw", "bogin", "amulet", "zwyczaj"),
          "wierzenia, obrzedy, legendy, symbolika"),
    Place("tajemna/nazwy", "Nazwy ludowe", "ra-quill-ink", False,
          {"nazwy_ludowe": 3.5},
          ("nazw", "nazywan", "okresla", "dialekt", "gwar", "zwan"),
          "nazwy ludowe i regionalne w roznych jezykach"),
)
PLACE_BY_ID = {p.id: p for p in MIEJSCA}
# slowa jednoznaczne: jedno trafienie przesadza o miejscu mocniej niz sekcja, do ktorej trafil punkt
STRONG = {
    "rozpoznawanie/pomylki": ("myl", "pomyl", "odrozni"),
    "bezpieczenstwo/chwast": ("chwast", "inwazyj"),
    "bezpieczenstwo/interakcje": ("interakc",),
    "tajemna/kultura": ("legend", "wierzen", "obrzed"),
    "tajemna/nazwy": ("nazw ludow", "ludowych nazw", "ludowe nazw", "dialekt"),
    "zastosowanie/kosmetyka": ("kosmet", "pielegn"),
    "zastosowanie/gospodarstwo": ("pasz", "podpuszczk"),
    "cykl/biologia": ("zapyl", "nektar", "rozsiew"),
    "natura/wystepowanie": ("wystepowan", "introdukow", "zawleczon"),
}
LAST_RESORT = "tajemna/ciekawostki"        # zakladki "Wiedza z sieci" juz nie ma - kazdy punkt musi gdzies trafic

# miejsce domyslne sekcji, gdy zadne nie przekroczylo progu (sekcja i tak gdzies nalezy)
FALLBACK = {
    "ciekawostki": "tajemna/ciekawostki", "historia": "tajemna/historia", "kultura": "tajemna/kultura",
    "nazwy_ludowe": "tajemna/nazwy", "wystepowanie": "natura/wystepowanie", "barwienie": "zastosowanie/rzemioslo",
    "kosmetyka": "zastosowanie/kosmetyka", "medycyna_wschodu": "natura/profil", "uprawa": "zastosowanie/ogrod",
    "bezpieczenstwo": "bezpieczenstwo/przeciwwskazania", "zastosowanie_lecznicze": "zastosowanie/medycyna",
    "sklad": "surowce/sklad", "opis": "rozpoznawanie/budowa", "czesci_rosliny": "surowce/inne",
}

# czesc rosliny -> podrozdzial tej czesci w "Surowcach" (wagi sekcji)
PART_PRIOR = {"czesci_rosliny": 2.5, "sklad": 2.5, "zastosowanie_lecznicze": 1.8, "bezpieczenstwo": 1.2,
              "opis": 0.6, "uprawa": 0.3}
PART_WORDS = ("surow", "zbier", "zbior", "susz", "wyciag", "napar", "odwar", "olej", "zawier")

# podrozdzial scalania (merge/schema) -> miejsce strony (do kary za odrzucenie przy scalaniu i duplikatow)
SLOT_PLACE = {
    "opis": None, "identyfikacja.cechy_kluczowe": "rozpoznawanie/cechy", "zastosowanie.medyczne": "zastosowanie/medycyna",
    "zastosowanie.kulinarne": "zastosowanie/kuchnia", "zastosowanie.rzemieslnicze": "zastosowanie/rzemioslo",
    "zastosowanie.kosmetyczne": "zastosowanie/kosmetyka", "zastosowanie.ogrodowe": "zastosowanie/ogrod",
    "interakcje": "bezpieczenstwo/interakcje", "ostrzezenia": "bezpieczenstwo/przeciwwskazania",
    "permakultura.funkcje": "cykl/permakultura", "ciekawostki": "tajemna/ciekawostki",
}


def slot_place(path: str | None) -> str | None:
    if not path:
        return None
    if path in SLOT_PLACE:
        return SLOT_PLACE[path]
    if path.startswith("profil_energetyczny."):
        return "natura/profil"
    if path.startswith("wymagania."):
        return "natura/wymagania"
    bits = path.split(".")
    if len(bits) == 3 and bits[0] == "czesci_rosliny":
        return f"surowce/czesc:{bits[1]}"
    return None


# ------------------------------------------------------------------ punkty
def _current_merged(data: dict) -> dict[str, dict]:
    """Aktualne podrozdzialy bloku "scalone" (jak Kwiatownik: oryginal == obecna wartosc pola)."""
    sc = data.get("scalone") if isinstance(data.get("scalone"), dict) else {}
    out = {}
    for path, entry in (sc.get("pola") or {}).items():
        if not isinstance(entry, dict) or not entry.get("tresc"):
            continue
        cur = get_path(data, path)
        orig = entry.get("oryginal")
        if (orig is None and not filled(cur)) or orig == cur:
            out[path] = entry
    return out


def used_facts(data: dict) -> set[str]:
    """Informacje wmontowane w aktualne podrozdzialy (blok "scalone")."""
    return {f for e in _current_merged(data).values() for t in e.get("tresc") or [] for f in t.get("fakty") or []}


def leftover_points(data: dict) -> list[dict]:
    """Punkty wiedzy z sieci, ktorych scalanie nie wmontowalo w zaden podrozdzial."""
    used = used_facts(data)
    out = []
    for sec in ((data.get("wiedza") or {}).get("sekcje") or []):
        for p in sec.get("punkty") or []:
            fakty = [f for f in p.get("fakty") or [] if f]
            if not p.get("tekst") or not p.get("zrodla") or not fakty or set(fakty) <= used:
                continue
            out.append({"pid": len(out) + 1, "sekcja": sec.get("klucz"), "czesc": p.get("czesc"),
                        "tekst": str(p["tekst"]), "zrodla": list(p.get("zrodla") or []), "fakty": fakty})
    return out


# ------------------------------------------------------------------ miejsca czesci rosliny
def _part_label(data: dict, key: str) -> str:
    parts = data.get("czesci_rosliny") if isinstance(data.get("czesci_rosliny"), dict) else {}
    pd = parts.get(key) if isinstance(parts.get(key), dict) else {}
    return str(pd.get("nazwa_surowca") or key.replace("_", " ")).strip().capitalize()


def part_place(data: dict, czesc: str | None) -> Place | None:
    """Podrozdzial czesci rosliny w "Surowcach": istniejaca czesc pliku albo nowa (nazwa ze slownika czesci)."""
    part = norm_part(czesc)
    if not part:
        return None
    key = part_key_for(data, part)
    if key:
        return Place(f"surowce/czesc:{key}", _part_label(data, key), "ra-potion", True, dict(PART_PRIOR),
                     PART_WORDS, f"surowiec: {key.replace('_', ' ')}")
    return Place(f"surowce/czesc:{part}", part.capitalize(), "ra-potion", False, dict(PART_PRIOR), PART_WORDS,
                 f"surowiec: {part}")


def places_for(data: dict, points: list[dict], cat: dict | None = None) -> list[Place]:
    """Miejsca dla tej rosliny w kolejnosci strony: wbudowane (bez wylaczonych w "Ukladzie strony"), podrozdzialy
    czesci rosliny, o ktorych mowia punkty (na poczatku "Surowcow"), i wlasne podrozdzialy z catalog.py
    (na koncu swojego rozdzialu; wlasne rozdzialy po rozdziale "po")."""
    from app.knowledge import catalog
    cat = cat if cat is not None else catalog.load()
    off = catalog.disabled(cat)
    extra: dict[str, Place] = {}
    for p in points:
        pp = part_place(data, p.get("czesc"))
        if pp and pp.id not in extra:
            extra[pp.id] = pp
    by_chapter: dict[str, list[Place]] = {}
    for pl in MIEJSCA:
        if pl.builtin or pl.id not in off:
            by_chapter.setdefault(pl.chapter_key, []).append(pl)
    by_chapter.setdefault("surowce", [])
    by_chapter["surowce"] = list(extra.values()) + by_chapter["surowce"]
    for pl in catalog.custom_places(cat):
        by_chapter.setdefault(pl.chapter_key, []).append(pl)
    out: list[Place] = []
    for ch in catalog.chapters(cat):
        out.extend(by_chapter.get(ch["id"], []))
    return out


def place_by_id(data: dict, pid: str, places: list[Place] | None = None) -> Place | None:
    for pl in places or []:
        if pl.id == pid:
            return pl
    if pid in PLACE_BY_ID:
        return PLACE_BY_ID[pid]
    if pid.startswith("surowce/czesc:"):
        key = pid.split(":", 1)[1]
        parts = data.get("czesci_rosliny") if isinstance(data.get("czesci_rosliny"), dict) else {}
        if key in parts:
            return Place(pid, _part_label(data, key), "ra-potion", True, dict(PART_PRIOR), PART_WORDS)
        return part_place(data, key)
    return None


# ------------------------------------------------------------------ ocena
def _hits(ntext: str, words: tuple) -> int:
    t = f" {ntext} "
    return sum(1 for w in words if f" {w}" in t)


def rejected_place(data: dict, point: dict, merged: dict[str, dict]) -> str | None:
    """Miejsce, do ktorego scalanie przydzielilo punkt (regula) i go nie uzylo - "nie pasuje" do tego tekstu."""
    from app.knowledge.merge import default_route
    path = default_route(data, point)
    return slot_place(path) if path in merged else None


def score_point(data: dict, point: dict, places: list[Place], merged: dict[str, dict]) -> dict[str, float]:
    """{miejsce: ocena} dla jednego punktu."""
    ntext = norm_text(point["tekst"])
    sec = point.get("sekcja")
    own_part = part_place(data, point.get("czesc"))
    bad = rejected_place(data, point, merged)
    out: dict[str, float] = {}
    for pl in places:
        if pl.id.startswith("surowce/czesc:") and (own_part is None or pl.id != own_part.id):
            continue                                   # podrozdzial innej czesci rosliny
        s = pl.prior.get(sec, 0.0) + KW_WEIGHT * min(_hits(ntext, pl.words), MAX_HITS)
        if _hits(ntext, STRONG.get(pl.id, ())):
            s += STRONG_WEIGHT
        if pl.id == bad and pl.id != "tajemna/ciekawostki":
            s -= REJECTED
        if pl.id in ("surowce/inne", "surowce/sklad") and own_part is not None:
            s -= 1.0                                   # punkt o konkretnej czesci -> raczej podrozdzial tej czesci
        out[pl.id] = round(s, 2)
    return out


def rule_places(data: dict, points: list[dict], places: list[Place]) -> tuple[dict[int, str | None], dict[int, dict]]:
    """Przydzial regulami: {pid: miejsce|None} + oceny (do LLM i do raportu)."""
    merged = _current_merged(data)
    scores = {p["pid"]: score_point(data, p, places, merged) for p in points}
    choice: dict[int, str | None] = {}
    for p in points:
        sc = scores[p["pid"]]
        best = max(sc, key=lambda k: (sc[k], -_order(places, k)), default=None)
        if best is not None and sc[best] >= MIN_SCORE:
            choice[p["pid"]] = best
        else:
            fb = FALLBACK.get(p.get("sekcja"))
            if fb == "surowce/inne" or fb == "surowce/sklad":
                own = part_place(data, p.get("czesc"))
                fb = own.id if own else fb
            known = {pl.id for pl in places}
            choice[p["pid"]] = fb if fb in known or (fb or "").startswith("surowce/czesc:") else LAST_RESORT
    return cohesion(points, choice, scores), scores


def _order(places: list[Place], pid: str) -> int:
    for i, pl in enumerate(places):
        if pl.id == pid:
            return i
    return len(places)


def cohesion(points: list[dict], choice: dict[int, str | None], scores: dict[int, dict]) -> dict[int, str | None]:
    """Punkty jednej sekcji trzymaja sie razem: gdy >= COHESION z nich ma to samo miejsce, pozostale dolaczaja,
    jesli to miejsce jest dla nich gorsze najwyzej o COHESION_GAP (i przekracza prog)."""
    out = dict(choice)
    by_sec: dict[str, list[dict]] = {}
    for p in points:
        by_sec.setdefault(p.get("sekcja") or "", []).append(p)
    for pts in by_sec.values():
        if len(pts) < 3:
            continue
        counts: dict[str, int] = {}
        for p in pts:
            if out.get(p["pid"]):
                counts[out[p["pid"]]] = counts.get(out[p["pid"]], 0) + 1
        if not counts:
            continue
        top, n = max(counts.items(), key=lambda kv: kv[1])
        if n / len(pts) < COHESION:
            continue
        for p in pts:
            cur = out.get(p["pid"])
            sc = scores[p["pid"]]
            if cur == top or top not in sc:
                continue
            best = sc.get(cur, 0.0) if cur else 0.0
            if sc[top] >= MIN_SCORE and sc[top] >= best - COHESION_GAP:
                out[p["pid"]] = top
    return out


# ------------------------------------------------------------------ LLM
PLACE_SYSTEM = """Jestes redaktorem polskiego zielnika "Kwiatownik". Strona rosliny ma rozdzialy i podrozdzialy \
(lista M1, M2...; "nowy" = podrozdzial, ktory dopiero powstanie na stronie). Ponizsze informacje z sieci nie \
pasowaly do tekstu zadnego podrozdzialu. Dla KAZDEJ wybierz miejsce na stronie, do ktorego tresciowo nalezy. \
Kazda informacja ma propozycje - zmien ja tylko, gdy inne miejsce pasuje wyraznie lepiej. Informacja o konkretnej \
czesci rosliny (owocach, korzeniu, lisciach...) nalezy do podrozdzialu tej czesci, jesli jest na liscie. \
Gdy ZADNE miejsce nie pasuje, a kilka informacji tworzy wyrazny wspolny temat, mozesz zalozyc nowy podrozdzial: \
{"id": 3, "miejsce": "NOWY", "rozdzial": "<klucz rozdzialu z listy ROZDZIALY>", "tytul": "<krotki tytul po polsku>"} \
- uzywaj tego rzadko. Nie zmieniaj tekstu informacji.
Zwroc TYLKO JSON: {"miejsca": [{"id": 1, "miejsce": "M3"}, {"id": 2, "miejsce": "M7"}]}"""


def place_prompt(plant_pl: str, places: list[Place], points: list[dict], choice: dict[int, str | None],
                 chapters: list[dict] | None = None) -> str:
    label = {pl.id: f"M{i}" for i, pl in enumerate(places, 1)}
    m_lines = [f"M{i} | {pl.chapter} -> {pl.title}{' (nowy)' if not pl.builtin else ''} | {pl.hint}"
               for i, pl in enumerate(places, 1)]
    c_lines = [f"{c['id']} = {c['tytul']}" for c in chapters or []]
    p_lines = [f"{p['pid']} | {p['sekcja']} | {p.get('czesc') or '-'} | propozycja: "
               f"{label.get(choice.get(p['pid']) or '', LAST_RESORT)} | {p['tekst']}" for p in points]
    return (f"ROSLINA: {plant_pl}\n\nROZDZIALY: " + "; ".join(c_lines) + "\n\nMIEJSCA NA STRONIE:\n" +
            "\n".join(m_lines) + "\n\nINFORMACJE (id | sekcja | czesc | propozycja | tekst):\n" + "\n".join(p_lines))


def apply_llm(raw: str, places: list[Place], points: list[dict], choice: dict[int, str | None],
              scores: dict[int, dict]) -> tuple[dict[int, str | None], int, list[dict]]:
    """Odpowiedz LLM -> nowy przydzial (tylko poprawne wpisy), liczba zmian i propozycje nowych podrozdzialow
    [{"pids": [...], "rozdzial", "tytul"}] (zaklada je build_placement przez catalog.py)."""
    from app.knowledge.merge import _parse
    data = _parse(raw)
    ids = {p["pid"] for p in points}
    allowed = {f"M{i}": pl.id for i, pl in enumerate(places, 1)}
    out = dict(choice)
    changed = 0
    proposals: dict[tuple[str, str], dict] = {}
    for row in data.get("miejsca") or []:
        if not isinstance(row, dict):
            continue
        try:
            pid = int(row.get("id"))
        except (TypeError, ValueError):
            continue
        if pid not in ids:
            continue
        lab = str(row.get("miejsce") or "").strip().upper()
        if lab in ("NOWY", "NEW"):
            ch, title = str(row.get("rozdzial") or "").strip(), re.sub(r"\s+", " ", str(row.get("tytul") or "")).strip()
            if ch and 3 <= len(title) <= 60:
                proposals.setdefault((ch, norm_text(title)), {"rozdzial": ch, "tytul": title, "pids": []})["pids"].append(pid)
            continue
        m = re.fullmatch(r"M?(\d+)", lab)
        new = allowed.get(f"M{m.group(1)}") if m else None
        if new and new in scores[pid] and new != out.get(pid):
            out[pid] = new
            changed += 1
    return out, changed, list(proposals.values())


def _agent_new_places(props: list[dict], data: dict, places: list[Place], choice: dict[int, str | None],
                      say) -> tuple[list[Place], int]:
    """Propozycje LLM nowych podrozdzialow -> wpisy w "Ukladzie strony" (zrodlo: agent) + przydzial punktow.
    Najwyzej catalog.MAX_AGENT_NEW na rosline; podrozdzial o tym samym tytule w rozdziale = uzyty istniejacy."""
    from app.knowledge import catalog
    made = 0
    for prop in sorted(props, key=lambda x: -len(x["pids"])):
        ch = catalog.chapter(prop["rozdzial"])
        if ch is None:
            continue
        same = next((pl for pl in places if pl.chapter_key == ch["id"] and norm_text(pl.title) == norm_text(prop["tytul"])), None)
        if same is None:
            if made >= catalog.MAX_AGENT_NEW:
                continue
            try:
                row = catalog.add_subchapter(ch["id"], prop["tytul"], "ra-leaf", zrodlo="agent",
                                             opis="zalozony przez agenta rozmieszczania - sprawdz i zatwierdz")
            except catalog.CatalogError as exc:
                say(f"  rozmieszczenie: nie zalozono podrozdzialu '{prop['tytul']}' ({exc})")
                continue
            made += 1
            same = next((pl for pl in catalog.custom_places() if pl.id == row["id"]), None)
            if same is None:
                continue
            places.append(same)
            say(f"  rozmieszczenie: agent zalozyl nowy podrozdzial '{ch['tytul']} -> {same.title}' (do zatwierdzenia w Ukladzie strony)")
        for pid in prop["pids"]:
            choice[pid] = same.id
    return places, made


# ------------------------------------------------------------------ wstawki
def _merged_texts(data: dict, place_id: str, merged: dict[str, dict]) -> list[str]:
    """Zdania juz na stronie w tym miejscu: tekst scalony + reczne pole (do wykrywania duplikatow)."""
    out = []
    for path, entry in merged.items():
        if slot_place(path) == place_id:
            out += [str(t.get("tekst") or "") for t in entry.get("tresc") or []]
    for path, target in SLOT_PLACE.items():
        if target == place_id and path not in merged:
            v = get_path(data, path)
            if isinstance(v, str):
                out += [s for s in re.split(r"(?<=[.!?])\s+", v) if s]
            elif isinstance(v, list):
                out += [str(x) for x in v if isinstance(x, str)]
    return out


def build_inserts(data: dict, points: list[dict], choice: dict[int, str | None], places: list[Place]) -> tuple[list[dict], list[str]]:
    """Wstawki (kolejnosc strony) + informacje pominiete jako duplikaty tekstu juz scalonego."""
    merged = _current_merged(data)
    by_place: dict[str, list[dict]] = {}
    dup: list[str] = []
    for p in points:
        pid = choice.get(p["pid"]) or LAST_RESORT
        existing = _merged_texts(data, pid, merged)
        if any(similar(p["tekst"], e) >= DUPLICATE for e in existing):
            dup += [f for f in p["fakty"] if f not in dup]
            continue
        rows = by_place.setdefault(pid, [])
        twin = next((r for r in rows if similar(r["tekst"], p["tekst"]) >= SAME_POINT), None)
        if twin:
            if len(p["tekst"]) > len(twin["tekst"]):
                twin["tekst"] = p["tekst"][:MAX_TEXT]
            twin["zrodla"] = sorted(set(twin["zrodla"]) | set(p["zrodla"]))
            twin["fakty"] += [f for f in p["fakty"] if f not in twin["fakty"]]
            continue
        rows.append({"tekst": p["tekst"][:MAX_TEXT], "zrodla": sorted(set(p["zrodla"])), "fakty": list(p["fakty"]),
                     "sekcja": p.get("sekcja"), "czesc": p.get("czesc")})
    order = {pl.id: i for i, pl in enumerate(places)}
    out = []
    for pid in sorted(by_place, key=lambda k: order.get(k, 999)):
        pl = place_by_id(data, pid, places)
        if pl is None:
            continue
        w = {"miejsce": pid, "rozdzial": pl.chapter, "tytul": pl.title, "ikona": pl.icon,
             "nowy": not pl.builtin, "punkty": by_place[pid]}
        if pl.custom_chapter:                          # rozdzial dodany w "Ukladzie strony"
            w.update(rozdzial_ikona=pl.chapter_icon or "ra-leaf", rozdzial_po=pl.chapter_after or "tajemna")
        if pl.source == "agent":
            w["od_agenta"] = True
        out.append(w)
    return out, dup


def fingerprint(points: list[dict], places: list[Place] | None = None) -> str:
    raw = json.dumps([[p["sekcja"], p.get("czesc"), p["tekst"], p["fakty"], p["zrodla"]] for p in points]
                     + [VERSION, sorted(pl.id + "|" + pl.title for pl in places or [])],
                     ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


async def build_placement(llm, data: dict, plant_pl: str, log=None, today: str | None = None,
                          prefer=None, allow_new: bool = True, force: bool = False) -> dict | None:
    """Blok "rozmieszczenie" (None, gdy wszystko jest juz wmontowane w rozdzialy albo nie ma wiedzy).
    allow_new: agent moze zalozyc nowy podrozdzial (catalog.py), gdy LLM uzna, ze zaden nie pasuje."""
    from app.knowledge import catalog
    say = log or (lambda m: None)
    points = leftover_points(data)
    if not points:
        return None
    cat = catalog.load()
    places = places_for(data, points, cat)
    key = fingerprint(points, places)
    old = data.get("rozmieszczenie") if isinstance(data.get("rozmieszczenie"), dict) else None
    if not force and old and old.get("wejscie") == key and old.get("wersja") == VERSION and \
            (llm is None or str(old.get("jak") or "").startswith("llm:")):
        return old                                    # te same punkty i miejsca - bez ponownego liczenia
    choice, scores = rule_places(data, points, places)
    how = "reguly"
    if llm is not None:
        from app.knowledge.merge import _complete
        changed = 0
        model = None
        props: list[dict] = []
        for i in range(0, len(points), LLM_BATCH):
            batch = points[i:i + LLM_BATCH]
            try:
                res = await _complete(llm, place_prompt(plant_pl, places, batch, choice, catalog.chapters(cat)),
                                      PLACE_SYSTEM, prefer)
                choice, n, pr = apply_llm(res.text, places, batch, choice, scores)
                changed += n
                props += pr
                model = f"{res.provider}/{res.model}"
            except Exception as exc:                   # reguly wystarcza
                say(f"  rozmieszczenie: LLM niedostepny ({str(exc)[:80]}) - reguly")
        if props and allow_new:
            places, _made = _agent_new_places(props, data, places, choice, say)
        if model:
            how = f"llm:{model}"
            if changed:
                say(f"  rozmieszczenie: LLM zmienil miejsce {changed} z {len(points)} punktow")
    inserts, dup = build_inserts(data, points, choice, places)
    if not inserts and not dup:
        return None
    n_new = sum(1 for w in inserts if w["nowy"])
    n_pts = sum(len(w["punkty"]) for w in inserts)
    say(f"  rozmieszczenie: {n_pts} punktow w {len(inserts)} miejscach ({n_new} nowych podrozdzialow), "
        f"duplikaty tekstu juz scalonego: {len(dup)}")
    return {"wersja": VERSION, "zaktualizowano": today or date.today().isoformat(), "jak": how,
            "wejscie": fingerprint(points, places), "wstawki": inserts, "duplikaty": dup}


def with_placement(cand: dict, block: dict | None) -> dict:
    if block:
        cand["rozmieszczenie"] = block
    else:
        cand.pop("rozmieszczenie", None)
    return cand


PLACE_RE = re.compile(r"^[a-z0-9_]+/[a-z0-9_:À-ſ]+$")


def validate_placement(cand: dict) -> list[str]:
    """Kontrola bloku "rozmieszczenie" w kopii pliku (plantfile.validate_candidate)."""
    blk = cand.get("rozmieszczenie")
    if blk is None:
        return []
    if not isinstance(blk, dict) or not isinstance(blk.get("wstawki"), list):
        return ["zly blok rozmieszczenie"]
    wiedza = cand.get("wiedza") or {}
    nrs = {z.get("nr") for z in wiedza.get("zrodla") or [] if isinstance(z, dict)}
    fact_ids = {f.get("id") for f in wiedza.get("fakty") or [] if isinstance(f, dict)}
    used = used_facts(cand)
    errors: list[str] = []
    seen: set[str] = set()
    for w in blk["wstawki"]:
        mid = str(w.get("miejsce") or "") if isinstance(w, dict) else ""
        if not PLACE_RE.match(mid) or not (mid.split("/", 1)[0] in CHAPTERS or w.get("rozdzial")):
            errors.append(f"rozmieszczenie: nieznane miejsce {mid[:40]}")
            continue
        if not str(w.get("tytul") or "").strip() or re.search(r"[<>]", str(w.get("tytul")) + str(w.get("rozdzial"))):
            errors.append(f"rozmieszczenie: zly tytul wstawki {mid}")
        if mid in seen:
            errors.append(f"rozmieszczenie: powtorzone miejsce {mid}")
        seen.add(mid)
        pts = w.get("punkty")
        if not isinstance(pts, list) or not pts:
            errors.append(f"rozmieszczenie: pusta wstawka {mid}")
            continue
        for p in pts:
            text = str((p or {}).get("tekst") or "")
            if not (3 <= len(text) <= MAX_TEXT) or re.search(r"<[a-z/!]", text, re.I):
                errors.append(f"rozmieszczenie: zly tekst w {mid}")
            if not p.get("zrodla") or not set(p.get("zrodla") or []) <= nrs:
                errors.append(f"rozmieszczenie: zrodlo spoza listy w {mid}")
            fk = set(p.get("fakty") or [])
            if not fk or not fk <= fact_ids:
                errors.append(f"rozmieszczenie: nieistniejaca informacja w {mid}")
            elif fk <= used:
                errors.append(f"rozmieszczenie: informacja juz scalona z rozdzialem ({mid})")
    if not set(blk.get("duplikaty") or []) <= fact_ids:
        errors.append("rozmieszczenie: duplikaty spoza wiedzy")
    return errors


def placed_facts(data: dict) -> set[str]:
    blk = data.get("rozmieszczenie") if isinstance(data, dict) and isinstance(data.get("rozmieszczenie"), dict) else {}
    out = {f for w in blk.get("wstawki") or [] for p in w.get("punkty") or [] for f in p.get("fakty") or []}
    return out | set(blk.get("duplikaty") or [])
