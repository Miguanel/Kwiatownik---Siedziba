"""Tezaurus magii roslin: rodzaje praktyk, pojecia w wielu jezykach, zapytania "od praktyki" i glebokie archiwa.

Ta sama praktyka ma w kazdym kraju inna nazwe (urok / boeser Blick / сглаз / malocchio / mal de ojo) - generator
zapytan (magic.py) laczy nazwe rosliny w danym jezyku z pojeciami tego jezyka, a tropy znalezione na stronach
(lokalne nazwy rosliny i praktyk, odpowiedniki w innych kulturach) dokladaja kolejne zapytania.
"""
from dataclasses import dataclass

# rodzaj praktyki -> (tytul podrozdzialu na stronie rosliny, ikona RPG Awesome, opis dla LLM, rdzenie slow PL)
RODZAJE: dict[str, tuple[str, str, str, tuple[str, ...]]] = {
    "ochrona": ("Ochrona i odpędzanie zła", "ra-tower",
                "odpedzanie zlych mocy, czarownic, demonow i piorunow; amulety, wieszanie nad drzwiami, ochrona domu i bydla",
                ("odpedz", "chroni", "ochron", "amulet", "talizman", "nad drzwi", "pioru", "bydl", "obor")),
    "uroki": ("Uroki i ich odczynianie", "ra-bleeding-eye",
              "rzucanie i zdejmowanie urokow, zle oko, zamawianie, okadzanie, odczynianie",
              ("urok", "zle oko", "zlego oka", "odczyn", "zamaw", "okadz", "uroczn")),
    "milosna": ("Magia miłosna", "ra-health",
                "lubczyki, napoje milosne, przyciaganie i zatrzymywanie ukochanej osoby, wrozby o malzenstwie",
                ("milos", "lubcz", "kochank", "ukochan", "zalot", "mezatk", "zaslubin", "wesel", "panny", "kawaler")),
    "wrozby": ("Wróżby, sny i przepowiednie", "ra-crystal-ball",
               "wrozenie z roslin, wianki puszczane na wode, sny prorocze, przepowiadanie pogody i losu",
               ("wroz", "przepowiad", "sen ", "sny", "snie", "wianek", "wianki", "przyszl", "los ")),
    "obrzedy": ("Obrzędy i święta", "ra-candle",
                "rosliny w obrzedach dorocznych i rodzinnych: noc swietojanska (Kupala, sobotki), Matki Boskiej Zielnej, "
                "Wigilia, Wielkanoc, wesele, pogrzeb; swiecenie ziol",
                ("obrzed", "swiet", "kupal", "sobotk", "zieln", "wigili", "wielkanoc", "palm", "wesel", "pogrzeb", "dozynk")),
    "czarownice": ("Rośliny czarownic", "ra-cauldron",
                   "rosliny przypisywane czarownicom: masci do latania, sabaty, zbieranie ziol o polnocy, procesy czarownic",
                   ("czarown", "wiedzm", "sabat", "masc", "latan", "proces", "lysa gor", "miotl")),
    "czarna": ("Czarna magia i klątwy", "ra-skull",
               "szkodzenie: klatwy, rzucanie choroby, odbieranie mleka krowom, trucicielstwo w wierzeniach",
               ("klatw", "przeklen", "szkodz", "czarna magi", "zaszkod", "odebra", "mleko krow", "smierc")),
    "lecznicza": ("Magia lecznicza i zamawianie", "ra-potion",
                  "leczenie magiczne: zamawianie chorob, szeptuchy i znachorzy, amulety na choroby, zaklecia przy zbiorze",
                  ("szeptuch", "znachor", "zamawia", "zaklec", "przy zbior", "febr", "zaraz", "leczyc magi")),
    "duchy": ("Duchy, demony i zaświaty", "ra-moon-sun",
              "rosliny a zmarli, upiory, rusalki, wampiry, diably i inne istoty; rosliny na grobach",
              ("duch", "demon", "upior", "rusalk", "wampir", "diab", "czart", "zmarl", "grob", "dusz", "zaswiat")),
    "zabobony": ("Zabobony i przesądy", "ra-clover",
                 "przesady i znaki: szczescie i pech, zakazy, wierzenia o roslinie niewpisujace sie w inne rodzaje",
                 ("przesad", "zabobon", "wierzon", "wierzono", "szczesc", "pech", "nie wolno", "zakaz", "znak")),
}
RODZAJ_KEYS = tuple(RODZAJE)
CHAPTER = ("magia", "Magia i zabobony", "ra-cauldron", "tajemna")      # id, tytul, ikona, po ktorym rozdziale

# jezyki wyszukiwania magii (kolejnosc = priorytet, gdy pojecie nie mowi inaczej)
LANGS = ("pl", "de", "ru", "uk", "en", "cs", "it", "es", "fr", "ja", "zh", "la")


@dataclass(frozen=True)
class Concept:
    key: str
    rodzaj: str
    terms: dict                  # jezyk -> slowa dopisywane do nazwy rosliny
    langs: tuple                 # jezyki, w ktorych temat jest najlepiej opisany (kolejnosc prob)


CONCEPTS: tuple[Concept, ...] = (
    Concept("czarownice", "czarownice", {
        "pl": "czarownice zioła czarownic", "de": "Hexen Hexenkraut", "en": "witches witchcraft folklore",
        "ru": "ведьмы колдовство трава", "uk": "відьми відьомське зілля", "cs": "čarodějnice čarodějnické byliny",
        "it": "streghe erba delle streghe", "es": "brujas brujería hierba", "fr": "sorcières sorcellerie herbe",
        "ja": "魔女 薬草 伝承", "zh": "巫术 女巫 草药"}, ("de", "pl", "en", "cs", "it")),
    Concept("ochrona", "ochrona", {
        "pl": "chroni przed złem amulet wierzenia", "de": "Abwehrzauber Schutz böse Geister", "en": "protection against evil spirits amulet folklore",
        "ru": "оберег от нечистой силы", "uk": "оберіг від нечистої сили", "cs": "ochrana před zlými duchy",
        "it": "amuleto protezione spiriti maligni", "es": "amuleto protección malos espíritus", "fr": "protection contre les mauvais esprits",
        "ja": "魔除け 厄除け", "zh": "辟邪 驱鬼 民俗"}, ("uk", "ru", "pl", "de", "ja", "zh")),
    Concept("uroki", "uroki", {
        "pl": "urok odczynianie uroków", "de": "böser Blick Besprechen", "en": "evil eye charm folk magic",
        "ru": "сглаз заговор", "uk": "пристріт замовляння", "cs": "uhranutí zaříkávání",
        "it": "malocchio", "es": "mal de ojo", "fr": "mauvais œil conjuration",
        "ja": "呪い 祓い", "zh": "驱邪 符咒"}, ("it", "es", "ru", "uk", "pl")),
    Concept("milosna", "milosna", {
        "pl": "magia miłosna lubczyk wróżby", "de": "Liebeszauber Liebesorakel", "en": "love charm love divination folklore",
        "ru": "приворот любовная магия", "uk": "приворот любовна магія", "cs": "milostná magie",
        "it": "filtro d'amore magia", "es": "hechizo de amor", "fr": "philtre d'amour",
        "ja": "恋のおまじない 惚れ薬", "zh": "媚药 爱情 巫术"}, ("ru", "uk", "de", "es", "pl")),
    Concept("wrozby", "wrozby", {
        "pl": "wróżby wianki noc świętojańska", "de": "Orakel Johannisnacht", "en": "divination midsummer folklore",
        "ru": "гадание Иван Купала", "uk": "ворожіння Купала", "cs": "věštění svatojánská noc",
        "it": "divinazione notte di San Giovanni", "es": "adivinación noche de San Juan", "fr": "divination nuit de la Saint-Jean",
        "ja": "占い 民俗", "zh": "占卜 民俗"}, ("uk", "ru", "pl", "cs", "de")),
    Concept("obrzedy", "obrzedy", {
        "pl": "obrzędy zwyczaje święcenie ziół", "de": "Brauchtum Kräuterweihe Ritual", "en": "ritual folk custom",
        "ru": "обряд обычай", "uk": "обряд звичай Маковія", "cs": "obřad lidové zvyky svěcení",
        "it": "rito tradizione popolare", "es": "ritual tradición popular", "fr": "rite tradition populaire",
        "ja": "儀式 行事 風習", "zh": "仪式 习俗 端午"}, ("pl", "uk", "ru", "de", "cs", "ja", "zh")),
    Concept("czarna", "czarna", {
        "pl": "klątwa czary szkodzenie", "de": "Schadenzauber Fluch", "en": "curse black magic hex",
        "ru": "порча чёрная магия", "uk": "порча чорна магія", "cs": "kletba černá magie",
        "it": "maleficio fattura", "es": "maleficio magia negra", "fr": "maléfice envoûtement",
        "ja": "呪詛 呪い", "zh": "巫蛊 诅咒"}, ("ru", "uk", "it", "zh", "de")),
    Concept("lecznicza", "lecznicza", {
        "pl": "zamawianie chorób szeptucha znachor", "de": "Volksmagie Heilzauber Besprechung", "en": "cunning folk charm healing magic",
        "ru": "знахарство заговор от болезни", "uk": "знахарство шептуха замовляння", "cs": "lidová magie zaříkávání nemocí",
        "it": "medicina magica popolare", "es": "curanderismo", "fr": "médecine magique guérisseur",
        "ja": "まじない 民間療法", "zh": "祝由 巫医"}, ("ru", "uk", "pl", "de", "es")),
    Concept("duchy", "duchy", {
        "pl": "demony duchy zmarli wierzenia", "de": "Geister Dämonen Totenbrauch", "en": "plant lore spirits fairies dead",
        "ru": "нечистая сила русалки", "uk": "нечиста сила мавки русалки", "cs": "duchové démoni",
        "it": "spiriti demoni", "es": "espíritus demonios", "fr": "esprits démons fées",
        "ja": "妖怪 霊", "zh": "鬼 精怪"}, ("uk", "ru", "ja", "zh", "en")),
    Concept("zabobony", "zabobony", {
        "pl": "przesądy zabobony wierzenia ludowe", "de": "Aberglaube Volksglaube", "en": "superstitions folk beliefs",
        "ru": "приметы суеверия поверья", "uk": "прикмети повір'я", "cs": "pověry lidová víra",
        "it": "superstizioni credenze popolari", "es": "supersticiones creencias populares", "fr": "superstitions croyances populaires",
        "ja": "迷信 俗信 言い伝え", "zh": "迷信 禁忌 民间信仰"}, ("pl", "de", "ru", "cs", "ja", "zh")),
    Concept("starozytni", "zabobony", {
        "la": "magia superstitio", "en": "Pliny magi herb magical virtues", "de": "Zauberpflanze Antike Magie",
        "fr": "plante magique Antiquité", "it": "pianta magica antichità", "pl": "starożytni magia roślina",
        "ru": "магические растения античность", "zh": "本草 辟邪", "ja": "本草 呪術"}, ("la", "en", "de", "zh")),
)
CONCEPT_BY_KEY = {c.key: c for c in CONCEPTS}

# zapytania "od praktyki" - bez nazwy rosliny; z jednej strony LLM wypisuje wiele roslin naraz
SWEEPS: tuple[tuple[str, str, str], ...] = (
    ("pl", "rośliny w wierzeniach ludowych czarownice zioła", "czarownice"),
    ("pl", "zioła święcone na Matkę Boską Zielną znaczenie magiczne", "obrzedy"),
    ("pl", "rośliny chroniące przed urokami wierzenia ludowe", "uroki"),
    ("pl", "rośliny w magii ludowej Słowian", "ochrona"),
    ("pl", "zioła sobótkowe noc świętojańska wierzenia", "wrozby"),
    ("de", "Pflanzen im Aberglauben Volkskunde Kräuter", "zabobony"),
    ("de", "Hexensalbe Pflanzen Geschichte Hexenprozesse", "czarownice"),
    ("de", "Kräuterbuschen Kräuterweihe Pflanzen Bedeutung Brauchtum", "obrzedy"),
    ("de", "Liebeszauber Pflanzen Volksglaube", "milosna"),
    ("de", "Pflanzen gegen Hexen und böse Geister Volksglaube", "ochrona"),
    ("ru", "обережные травы народные поверья славян", "ochrona"),
    ("ru", "травы Ивана Купалы поверья обряды", "wrozby"),
    ("ru", "приворотные травы русский фольклор", "milosna"),
    ("ru", "колдовские травы ведьмы народные поверья", "czarownice"),
    ("uk", "трави-обереги українців повір'я", "ochrona"),
    ("uk", "купальські трави повір'я обряди", "wrozby"),
    ("uk", "рослини в українській народній магії", "lecznicza"),
    ("en", "plants in witchcraft folklore history", "czarownice"),
    ("en", "flying ointment plants history witch trials", "czarownice"),
    ("en", "herbs against witches folklore british isles", "ochrona"),
    ("en", "plant superstitions folklore europe", "zabobony"),
    ("cs", "byliny proti čarodějnicím lidová víra", "ochrona"),
    ("it", "erbe contro il malocchio tradizione popolare", "uroki"),
    ("it", "erbe di San Giovanni streghe tradizione", "obrzedy"),
    ("es", "hierbas de San Juan tradición mágica", "obrzedy"),
    ("es", "plantas contra el mal de ojo tradición", "uroki"),
    ("fr", "plantes magiques folklore sorcellerie", "czarownice"),
    ("fr", "herbes de la Saint-Jean croyances populaires", "obrzedy"),
    ("ja", "魔除け 植物 民俗", "ochrona"),
    ("ja", "まじない 植物 伝承", "zabobony"),
    ("zh", "辟邪 植物 民俗", "ochrona"),
    ("zh", "端午 辟邪 草药 习俗", "obrzedy"),
)


@dataclass(frozen=True)
class DeepSource:
    domain: str
    lang: str                    # jezyk tekstow ("la" = lacina, "*" = rozne)
    name: str
    word: str                    # krotkie slowo "magiczne" w jezyku zrodla do zapytania site:


# "konce internetu": biblioteki cyfrowe, stare ksiegi i czasopisma etnograficzne (strony HTML z pelnym tekstem)
DEEP_SOURCES: tuple[DeepSource, ...] = (
    DeepSource("zeno.org", "de", "Zeno.org (klasyka niemiecka, Grimm, Handwörterbuch)", "Aberglaube"),
    DeepSource("de.wikisource.org", "de", "Wikisource (de)", "Hexen"),
    DeepSource("pl.wikisource.org", "pl", "Wikiźródła (pl)", "czary"),
    DeepSource("ru.wikisource.org", "ru", "Викитека (ru)", "поверья"),
    DeepSource("uk.wikisource.org", "uk", "Вікіджерела (uk)", "повір'я"),
    DeepSource("cyberleninka.ru", "ru", "КиберЛенинка (artykuły etnograficzne)", "обряд"),
    DeepSource("gutenberg.org", "en", "Project Gutenberg", "superstition"),
    DeepSource("sacred-texts.com", "en", "Internet Sacred Text Archive", "folklore"),
    DeepSource("penelope.uchicago.edu", "en", "LacusCurtius (Pliniusz, Dioskurides)", "magi"),
    DeepSource("thelatinlibrary.com", "la", "The Latin Library", "magi"),
    DeepSource("ethnobiomed.biomedcentral.com", "en", "Journal of Ethnobiology and Ethnomedicine", "magic ritual"),
    DeepSource("journals.openedition.org", "fr", "OpenEdition (etnologia)", "croyances"),
    DeepSource("ctext.org", "zh", "Chinese Text Project (本草, dawne teksty)", "辟邪"),
    DeepSource("zh.wikisource.org", "zh", "維基文庫 (zh)", "辟邪"),
    DeepSource("aozora.gr.jp", "ja", "青空文庫 (Aozora Bunko)", "まじない"),
    DeepSource("cervantesvirtual.com", "es", "Biblioteca Virtual Miguel de Cervantes", "superstición"),
)

# rdzenie slow "magicznych" (male litery) - strona musi miec co najmniej 2 rozne, zeby byla o magii
STEMS: dict[str, tuple[str, ...]] = {
    "pl": ("czarow", "czary", "czarami", "czarach", "czarodz", "czarnoks", "gusł", "zaklin", "zabobon", "przesąd", "przesad", "urok", "wróż", "wroz", "amulet", "obrzęd", "obrzed",
           "magi", "zaklę", "klątw", "czarci", "diab", "sobótk", "kupał", "zieln", "szeptuch", "znachor", "demon",
           "upiór", "rusałk", "wiedźm", "talizman", "odczyn", "zamawia", "wierzono"),
    "de": ("hexe", "zauber", "aberglaub", "amulett", "brauchtum", "böse blick", "besprech", "teufel", "orakel",
           "johannisnacht", "kräuterweih", "dämon", "unheil", "volksglaub", "abwehr", "walpurgis", "segen"),
    "en": ("witch", "spell", "charm", "superstit", "amulet", "ritual", "divination", "folklore", "magic", "curse",
           "evil eye", "fairy", "fairies", "cunning", "talisman", "ward off", "midsummer", "sorcer"),
    "ru": ("колдов", "ведьм", "заговор", "оберег", "примет", "суевер", "гадан", "обряд", "порч", "сглаз", "нечист",
           "купал", "приворот", "знахар", "магич", "русалк", "поверь"),
    "uk": ("відьм", "чакл", "замовлян", "оберіг", "оберег", "прикмет", "ворож", "обряд", "пристріт", "нечист",
           "купал", "знахар", "магі", "мавк", "повір"),
    "cs": ("čaroděj", "pověr", "kouzl", "zaříkáv", "uhranut", "amulet", "věšt", "obřad", "magi", "čarod"),
    "it": ("strega", "streghe", "malocchio", "superstiz", "incantes", "amulet", "rito ", "riti ", "magia", "sortileg",
           "fattur", "divinaz"),
    "es": ("bruj", "hechiz", "mal de ojo", "supersti", "amulet", "ritual", "magia", "maleficio", "adivinac", "curander",
           "conjur"),
    "fr": ("sorcel", "sorci", "superstit", "amulette", "envoût", "maléfic", " rite", "magie", "charme", "divinat",
           "conjur"),
    "ja": ("魔除", "呪", "まじない", "迷信", "俗信", "占い", "魔女", "儀式", "祓", "厄除", "お守り", "妖"),
    "zh": ("辟邪", "巫", "咒", "迷信", "驱邪", "占卜", "符咒", "蛊", "祭祀", "鬼", "禁忌", "民俗"),
    "la": ("magi", "venefic", "incant", "amulet", "superstiti", "malefic", "strig", "carmen", "carmina"),
}
ALL_STEMS = tuple(dict.fromkeys(s for v in STEMS.values() for s in v))

# rodzaje roslin slynne w magii ludowej (rodzaj lacinski) - pierwszenstwo przy wyborze roslin
FAMOUS_GENERA = frozenset((
    "artemisia", "hypericum", "verbena", "mandragora", "hyoscyamus", "atropa", "datura", "aconitum", "allium",
    "viscum", "sorbus", "sambucus", "dryopteris", "pteridium", "ruta", "levisticum", "achillea", "salvia",
    "rosmarinus", "verbascum", "pimpinella", "valeriana", "juniperus", "corylus", "populus", "betula", "quercus",
    "tilia", "fraxinus", "galium", "thymus", "mentha", "paeonia", "lilium", "papaver", "conium",
    "helleborus", "euphorbia", "urtica", "carlina", "angelica", "nigella", "calendula", "chelidonium",
    "lycopodium", "peucedanum", "matricaria", "tanacetum", "acorus", "sempervivum", "plantago", "crataegus",
    "prunus", "ilex", "hedera", "taxus", "laurus", "myrtus", "rosa", "trifolium", "bryonia", "solanum",
))


def concept_order() -> list[str]:
    """Kolejnosc pojec przy rotacji (najpierw te, ktore najczesciej daja wiedze o konkretnej roslinie)."""
    return ["czarownice", "ochrona", "obrzedy", "zabobony", "uroki", "lecznicza", "wrozby", "milosna", "duchy",
            "czarna", "starozytni"]


def magic_score(text: str) -> int:
    """Ile roznych rdzeni slow "magicznych" (we wszystkich jezykach) ma tekst."""
    low = (text or "").lower()
    return sum(1 for s in ALL_STEMS if s in low)
