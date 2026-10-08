"""Szukanie BRAKUJACYCH informacji o roslinie: luki w schemacie pliku (app/knowledge/schema.py) -> zapytania
do wyszukiwarki skupione na roslinie i na tym, czego brakuje - w wielu jezykach, takze po chinsku i japonsku
(tam jest wiele wiedzy zielarskiej - TCM, kampo, 草木染め - trudno dostepnej dla polskich czytelnikow).

Zapytanie = nazwa rosliny w danym jezyku (z Wikidata: etykieta / tytul artykulu Wikipedii, inaczej nazwa
lacinska) + slowa tematu luki w tym jezyku, np. "贯叶连翘" 性味归经, "セイヨウオトギリソウ" 草木染め.
"""
from dataclasses import dataclass

from app.knowledge.schema import PART_FIELDS, SLOTS, filled, get_path

# temat -> slowa w jezykach (zapytanie: "<nazwa rosliny>" + slowa)
TOPICS: dict[str, dict[str, str]] = {
    "lecznicze": {"pl": "właściwości lecznicze zastosowanie", "en": "medicinal uses herbal medicine",
                  "de": "Heilwirkung Anwendung Heilpflanze", "ru": "лечебные свойства применение",
                  "uk": "лікувальні властивості застосування", "fr": "propriétés médicinales utilisation",
                  "zh": "功效与作用 药用", "ja": "薬効 効能 生薬", "cs": "léčivé účinky", "it": "proprietà medicinali"},
    "wschod": {"pl": "medycyna chińska natura smak", "en": "traditional chinese medicine nature flavor meridian",
               "de": "TCM Wirkung Geschmack Meridian", "ru": "китайская медицина свойства вкус меридиан",
               "uk": "китайська медицина", "fr": "médecine chinoise saveur nature méridien",
               "zh": "性味归经 功能主治", "ja": "性味 帰経 漢方 効能"},
    "sklad": {"pl": "skład chemiczny substancje czynne", "en": "chemical constituents active compounds",
              "de": "Inhaltsstoffe Wirkstoffe", "ru": "химический состав действующие вещества",
              "uk": "хімічний склад", "fr": "composition chimique principes actifs", "zh": "化学成分 有效成分",
              "ja": "成分 有効成分"},
    "zbior": {"pl": "zbiór suszenie surowca", "en": "harvesting drying herb", "de": "Ernte Trocknung Droge",
              "ru": "сбор сушка сырья", "uk": "заготівля сушіння", "fr": "récolte séchage", "zh": "采收 加工 炮制",
              "ja": "採取時期 乾燥 調製"},
    "barwienie": {"pl": "barwienie barwnik naturalny", "en": "natural dye dyeing wool", "de": "Färberpflanze färben Wolle",
                  "ru": "натуральный краситель окрашивание", "uk": "фарбування природний барвник",
                  "fr": "teinture végétale", "zh": "植物染 染色 草木染", "ja": "草木染め 染料"},
    "kosmetyka": {"pl": "kosmetyka pielęgnacja skóry włosów", "en": "cosmetic uses skin hair care",
                  "de": "Kosmetik Hautpflege Haare", "ru": "косметология уход за кожей волосами",
                  "uk": "косметологія", "fr": "cosmétique soin peau cheveux", "zh": "美容 护肤 护发",
                  "ja": "化粧品 美容 スキンケア"},
    "kuchnia": {"pl": "jadalna zastosowanie w kuchni", "en": "edible uses culinary", "de": "essbar Küche Verwendung",
                "ru": "съедобность в кулинарии", "uk": "їстівна кулінарія", "fr": "comestible cuisine",
                "zh": "食用 食疗", "ja": "食用 料理 山菜"},
    "uprawa": {"pl": "uprawa wymagania stanowisko gleba", "en": "growing conditions soil sun hardiness",
               "de": "Anbau Standort Boden winterhart", "ru": "выращивание почва освещение зимостойкость",
               "uk": "вирощування ґрунт", "fr": "culture exposition sol rusticité", "zh": "栽培 生长习性 土壤",
               "ja": "育て方 栽培 日当たり 土"},
    "toksycznosc": {"pl": "toksyczność przeciwwskazania skutki uboczne", "en": "toxicity side effects contraindications",
                    "de": "Giftigkeit Nebenwirkungen Gegenanzeigen", "ru": "токсичность противопоказания",
                    "uk": "токсичність протипоказання", "fr": "toxicité contre-indications effets secondaires",
                    "zh": "毒性 禁忌 副作用", "ja": "毒性 副作用 禁忌"},
    "interakcje": {"pl": "interakcje z lekami", "en": "drug interactions", "de": "Wechselwirkungen Medikamente",
                   "ru": "взаимодействие с лекарствами", "uk": "взаємодія з ліками",
                   "fr": "interactions médicamenteuses", "zh": "药物相互作用", "ja": "医薬品 相互作用"},
    "rozpoznawanie": {"pl": "opis morfologia rozpoznawanie", "en": "identification botanical description",
                      "de": "Merkmale Beschreibung Bestimmung", "ru": "ботаническое описание морфология",
                      "uk": "ботанічний опис", "fr": "description botanique identification", "zh": "形态特征 识别",
                      "ja": "特徴 見分け方 形態"},
    "pomylki": {"pl": "pomylenie z podobne rośliny", "en": "lookalikes confused with", "de": "Verwechslung ähnliche",
                "ru": "можно спутать с похожие растения", "uk": "можна сплутати", "fr": "confusion plantes semblables",
                "zh": "易混淆 鉴别", "ja": "似た植物 誤食 見分け"},
    "folklor": {"pl": "wierzenia legendy historia nazwy ludowe", "en": "folklore history legends traditions",
                "de": "Volksglaube Brauchtum Geschichte", "ru": "народные поверья легенды история",
                "uk": "народні повір'я легенди", "fr": "folklore légendes histoire", "zh": "民俗 传说 历史 典故",
                "ja": "民俗 伝承 歴史 由来"},
    "ekologia": {"pl": "roślina miododajna towarzysząca", "en": "pollinators companion planting wildlife",
                 "de": "Bienenweide Mischkultur", "ru": "медонос растения-компаньоны", "uk": "медонос",
                 "fr": "mellifère plantes compagnes", "zh": "蜜源植物 生态", "ja": "蜜源植物 コンパニオンプランツ"},
}

# podrozdzial schematu -> temat wyszukiwania
SLOT_TOPIC: dict[str, str] = {
    "opis": "rozpoznawanie", "identyfikacja.cechy_kluczowe": "rozpoznawanie", "identyfikacja.mozliwe_pomyłki": "pomylki",
    "profil_energetyczny.smak": "wschod", "profil_energetyczny.termika": "wschod",
    "profil_energetyczny.wilgotnosc": "wschod", "profil_energetyczny.zywiol": "wschod",
    "profil_energetyczny.opis": "wschod",
    "wymagania.stanowisko": "uprawa", "wymagania.gleba": "uprawa", "wymagania.ph_gleby": "uprawa",
    "wymagania.mrozoodpornosc": "uprawa", "wymagania.woda": "uprawa", "zastosowanie.ogrodowe": "uprawa",
    "czesci_rosliny": "lecznicze", "zastosowanie.medyczne": "lecznicze", "zastosowanie.kulinarne": "kuchnia",
    "zastosowanie.rzemieslnicze": "barwienie", "zastosowanie.kosmetyczne": "kosmetyka",
    "interakcje": "interakcje", "ostrzezenia": "toksycznosc", "kalendarz_ogrodnika.zadania": "zbior",
    "permakultura.funkcje": "ekologia", "permakultura.gildie": "ekologia", "ciekawostki": "folklor",
}
PART_FIELD_TOPIC = {"opis_botaniczny": "rozpoznawanie", "wlasciwości": "lecznicze", "wlasciwosci": "lecznicze",
                    "skladniki_aktywne": "sklad", "czas_zbioru": "zbior", "ostrzezenia": "toksycznosc"}

# tematy, dla ktorych najwiecej wiedzy jest w danych jezykach (kolejnosc prob)
TOPIC_LANGS: dict[str, tuple[str, ...]] = {
    "wschod": ("zh", "ja", "en"), "barwienie": ("ja", "zh", "en", "de", "fr"),
    "lecznicze": ("zh", "ja", "ru", "de", "uk", "en"), "sklad": ("zh", "ja", "de", "en", "ru"),
    "zbior": ("de", "ru", "zh", "ja", "uk"), "folklor": ("uk", "ru", "ja", "zh", "de", "fr"),
    "kosmetyka": ("ja", "zh", "fr", "en"), "toksycznosc": ("en", "de", "zh", "ja"),
    "interakcje": ("en", "de", "zh"), "uprawa": ("de", "en", "ja", "ru"), "kuchnia": ("ja", "zh", "fr", "de"),
}
# kolejnosc waznosci luk (Kwiatownik = ziololecznictwo, barwienie, kosmetyka)
PRIORITY = ("lecznicze", "toksycznosc", "wschod", "barwienie", "sklad", "zbior", "interakcje", "kosmetyka",
            "rozpoznawanie", "folklor", "uprawa", "pomylki", "kuchnia", "ekologia")


@dataclass
class Gap:
    path: str          # podrozdzial schematu (albo czesci_rosliny.<czesc>.<pole>)
    title: str
    topic: str


@dataclass
class GapQuery:
    gap: Gap
    lang: str
    query: str


def part_gaps(data: dict) -> list[Gap]:
    out = []
    parts = data.get("czesci_rosliny") if isinstance(data.get("czesci_rosliny"), dict) else {}
    for key, pdata in parts.items():
        if not isinstance(pdata, dict):
            continue
        for field, title, _kind, _hint in PART_FIELDS:
            value = pdata.get(field) if field in pdata else pdata.get("wlasciwosci" if field == "wlasciwości" else field)
            if not filled(value):
                out.append(Gap(f"czesci_rosliny.{key}.{field}", f"{key.replace('_', ' ')}: {title}",
                               PART_FIELD_TOPIC.get(field, "lecznicze")))
    return out


def find_gaps(data: dict | None, coverage_rows: list[dict] | None = None) -> list[Gap]:
    """Puste podrozdzialy (bez recznej tresci, bez wiedzy z sieci) - od najwazniejszych."""
    data = data or {}
    status = {r["path"]: r["status"] for r in coverage_rows or []}
    gaps = []
    for slot in SLOTS:
        st = status.get(slot.path)
        empty = st == "brak" if st else not filled(get_path(data, slot.path))
        if empty:
            gaps.append(Gap(slot.path, f"{slot.chapter} - {slot.title}", SLOT_TOPIC.get(slot.path, "lecznicze")))
    gaps += part_gaps(data)
    return sorted(gaps, key=lambda g: PRIORITY.index(g.topic) if g.topic in PRIORITY else len(PRIORITY))


def plant_names(latin: str | None, nazwa_pl: str | None, labels: dict | None, titles: dict | None) -> dict[str, str]:
    """Nazwa rosliny w jezykach: etykieta Wikidata, tytul artykulu Wikipedii, inaczej lacinska."""
    names: dict[str, str] = {}
    for lang, title in (titles or {}).items():
        if title:
            names[lang] = title.split(" (")[0]
    for lang, label in (labels or {}).items():
        if label:
            names[lang] = label
    if nazwa_pl:
        names.setdefault("pl", nazwa_pl)
    if latin:
        names["la"] = latin
    return names


def build_queries(gaps: list[Gap], names: dict[str, str], langs: list[str], done: set[str],
                  max_gaps: int = 4, per_gap: int = 2) -> list[GapQuery]:
    """Zapytania dla najwazniejszych luk: kazda luka w 1-2 jezykach (najpierw te, w ktorych temat jest
    najlepiej opisany), bez zapytan zadanych niedawno (done). Jeden temat = jedna seria zapytan."""
    out: list[GapQuery] = []
    topics_seen: set[str] = set()
    latin = names.get("la")
    for gap in gaps:
        if len({q.gap.topic for q in out}) >= max_gaps:
            break
        if gap.topic in topics_seen:
            continue
        topics_seen.add(gap.topic)
        order = [x for x in TOPIC_LANGS.get(gap.topic, ()) if x in langs] + [x for x in langs]
        n = 0
        for lang in dict.fromkeys(order):
            words = TOPICS.get(gap.topic, {}).get(lang)
            name = names.get(lang) or latin
            if not words or not name:
                continue
            q = f"\"{name}\" {words}"
            if q in done:
                continue
            out.append(GapQuery(gap, lang, q))
            n += 1
            if n >= per_gap:
                break
    return out


def page_about_plant(text: str, names: dict[str, str], lang: str) -> bool:
    """Strona naprawde o tej roslinie: nazwa lacinska (rodzaj) albo nazwa w jezyku strony w tekscie."""
    low = text.lower()
    latin = names.get("la") or ""
    if latin and latin.split()[0].lower() in low:
        return True
    local = names.get(lang)
    return bool(local) and local.lower() in low


def gaps_for_focus(gaps: list[Gap], topic: str | None = None, limit: int = 6) -> list[str]:
    """Opisy luk dla promptu LLM wyciagajacego informacje ("SZCZEGOLNIE SZUKAMY")."""
    chosen = [g for g in gaps if topic is None or g.topic == topic] or gaps
    return [g.title for g in chosen[:limit]]
