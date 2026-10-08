"""Wikidata + Wikipedia (oficjalne API, bez scrapowania HTML).

1. Wikidata: gatunek po nazwie lacinskiej (P225 = nazwa taksonu) -> identyfikator Q, polska nazwa,
   linki do artykulow Wikipedii we wszystkich jezykach.
2. Wikipedia: czysty tekst artykulu (prop=extracts&explaintext) podzielony na sekcje; do LLM idzie
   wstep + sekcje o zastosowaniu, historii, wierzeniach, toksycznosci itd. (bez bibliografii).
"""
import asyncio
import logging
import re
from dataclasses import dataclass, field

import httpx

log = logging.getLogger(__name__)

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
SKIP_SECTIONS = ("przypisy", "bibliografia", "linki zewn", "zobacz te", "uwagi", "galeria", "literatura",
                 "references", "see also", "external links", "further reading", "notes", "gallery", "sources",
                 "einzelnachweise", "literatur", "weblinks", "siehe auch", "примітки", "посилання", "джерела",
                 "література", "див. також", "примечания", "ссылки", "литература", "см. также", "références",
                 "liens externes", "voir aussi", "bibliographie", "reference", "odkazy", "externí odkazy",
                 "poznámky", "šaltiniai", "nuorodos", "išnašos", "note", "collegamenti esterni", "referencias",
                 "enlaces externos", "jegyzetek", "források", "бележки", "източници", "systematyka", "taxonomy",
                 "synonyms", "synonimy", "synonyme", "синоніми", "синонимы",
                 "参考文献", "参见", "参考资料", "外部链接", "注释", "脚注", "关联项目", "関連項目", "外部リンク",
                 "出典", "注釈", "脚注", "参考")
PRIORITY_WORDS = ("zastos", "lecz", "medyc", "zielar", "ludow", "histor", "nazw", "etym", "wierz", "obrz", "legend",
                  "mit", "barw", "farb", "kosmet", "trują", "toksy", "bezpiecz", "skład", "chemi", "kultur",
                  "use", "medic", "herbal", "folk", "histor", "etym", "name", "culture", "myth", "dye", "cosmet",
                  "toxic", "chemi", "constitu", "tradition", "verwend", "heil", "volks", "geschicht", "namen",
                  "farb", "giftig", "inhaltsst", "застос", "лікув", "народн", "істор", "назв", "фарб", "отру",
                  "примен", "лечеб", "истор", "назван", "utilis", "médic", "histoire", "étymol", "toxic",
                  "použit", "léčiv", "histor", "naudoj", "gydom", "uso", "medicin", "storia", "hasznal", "gyógy",
                  "употреб", "лечеб", "utiliz", "medicina",
                  "药用", "功效", "用途", "利用", "价值", "毒", "成分", "染", "文化", "民俗", "栽培", "性味",
                  "薬用", "効能", "利用", "民俗", "文化", "毒性", "染料", "生薬", "漢方", "伝承", "名称", "由来")


@dataclass
class WikiArticle:
    lang: str
    title: str
    url: str
    text: str                     # wybrane sekcje, przyciete do limitu
    sections: list[str] = field(default_factory=list)


@dataclass
class WikiPlant:
    qid: str
    taxon: str | None
    label_pl: str | None
    titles: dict[str, str]        # jezyk -> tytul artykulu
    image: str | None = None      # Wikidata P18 - glowne zdjecie gatunku (nazwa pliku na Commons)
    commons_category: str | None = None   # Wikidata P373 - kategoria gatunku na Commons
    labels: dict[str, str] = field(default_factory=dict)   # nazwa gatunku w jezykach (zh, ja, de...) - do wyszukiwania


def split_sections(text: str) -> list[tuple[str, str]]:
    """Tekst z explaintext -> [(tytul sekcji, tresc)]; wstep ma tytul ''. Podsekcje trafiaja do rodzica."""
    parts: list[tuple[str, str]] = []
    title, buf = "", []
    for line in text.splitlines():
        m = re.match(r"^(={2,6})\s*(.+?)\s*\1\s*$", line)
        if m and len(m.group(1)) == 2:
            parts.append((title, "\n".join(buf).strip()))
            title, buf = m.group(2), []
        elif m:
            buf.append(m.group(2) + ":")
        else:
            buf.append(line)
    parts.append((title, "\n".join(buf).strip()))
    return [(t, b) for t, b in parts if b]


def select_sections(text: str, max_chars: int = 9000) -> tuple[str, list[str]]:
    """Wstep + najciekawsze sekcje (zastosowanie, historia, wierzenia...) do limitu znakow."""
    secs = [(t, b) for t, b in split_sections(text) if not any(t.lower().startswith(s) for s in SKIP_SECTIONS)]
    if not secs:
        return "", []
    intro, rest = secs[0], secs[1:]
    rest.sort(key=lambda s: 0 if any(w in s[0].lower() for w in PRIORITY_WORDS) else 1)
    out, used, size = [], [], 0
    for t, b in [intro] + rest:
        chunk = (f"## {t}\n" if t else "") + b
        if size + len(chunk) > max_chars:
            if not out:
                out.append(chunk[:max_chars])
                used.append(t or "wstep")
            continue
        out.append(chunk)
        used.append(t or "wstep")
        size += len(chunk)
    return "\n\n".join(out), used


class WikiClient:
    def __init__(self, user_agent: str, client: httpx.AsyncClient | None = None, delay_s: float = 0.3):
        self.client = client or httpx.AsyncClient(timeout=30, headers={"User-Agent": user_agent},
                                                  follow_redirects=True)
        self.delay_s = delay_s

    async def aclose(self) -> None:
        await self.client.aclose()

    async def _get(self, url: str, params: dict) -> dict:
        r = await self.client.get(url, params=params | {"format": "json"})
        r.raise_for_status()
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        return r.json()

    async def find_plant(self, latin: str | None, name_pl: str | None = None) -> WikiPlant | None:
        """Szuka gatunku w Wikidata: najpierw po nazwie lacinskiej (musi zgadzac sie P225), potem po polskiej."""
        queries = [(latin, "en")] if latin else []
        if name_pl:
            queries.append((name_pl, "pl"))
        for q, lang in queries:
            found = await self._get(WIKIDATA_API, {"action": "wbsearchentities", "search": q, "language": lang,
                                                   "type": "item", "limit": 7})
            ids = [x["id"] for x in found.get("search", [])]
            if not ids:
                continue
            ents = (await self._get(WIKIDATA_API, {"action": "wbgetentities", "ids": "|".join(ids),
                                                   "props": "sitelinks|claims|labels",
                                                   "languages": LABEL_LANGS}))
            for qid in ids:
                ent = ents.get("entities", {}).get(qid, {})
                taxon = _claim(ent, "P225")
                if not taxon:
                    continue                              # to nie takson (np. miejscowosc o tej nazwie)
                if latin and taxon.lower() != latin.lower() and lang == "en":
                    continue
                titles = {k[:-4]: v["title"] for k, v in ent.get("sitelinks", {}).items()
                          if k.endswith("wiki") and k not in ("commonswiki", "specieswiki") and "_" not in k[:-4]}
                label = (ent.get("labels", {}).get("pl") or {}).get("value")
                return WikiPlant(qid, taxon, label, titles, _claim(ent, "P18"), _claim(ent, "P373"),
                                 labels_by_lang(ent.get("labels") or {}))
        return None

    async def article(self, lang: str, title: str, max_chars: int = 9000) -> WikiArticle | None:
        data = await self._get(f"https://{lang}.wikipedia.org/w/api.php", {
            "action": "query", "prop": "extracts|info", "explaintext": 1, "exsectionformat": "wiki",
            "inprop": "url", "redirects": 1, "titles": title})
        for page in (data.get("query", {}).get("pages") or {}).values():
            text = page.get("extract") or ""
            if len(text) < 200:
                return None
            chosen, used = select_sections(text, max_chars)
            url = page.get("fullurl") or f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"
            return WikiArticle(lang, page.get("title") or title, url, chosen, used)
        return None


LABEL_LANGS = "pl|en|de|ru|uk|fr|cs|it|es|ja|zh|zh-hans|zh-cn|ko"


def labels_by_lang(labels: dict) -> dict[str, str]:
    """Etykiety Wikidata -> {jezyk: nazwa}; chinski: najpierw uproszczony (zh-hans / zh-cn)."""
    out: dict[str, str] = {}
    for code in ("zh", "zh-cn", "zh-hans"):              # ostatni wygrywa
        v = (labels.get(code) or {}).get("value")
        if v:
            out["zh"] = v
    for code, v in labels.items():
        if not code.startswith("zh") and isinstance(v, dict) and v.get("value"):
            out[code] = v["value"]
    return out


def _claim(ent: dict, prop: str) -> str | None:
    try:
        v = ent["claims"][prop][0]["mainsnak"]["datavalue"]["value"]
        return v if isinstance(v, str) else None
    except (KeyError, IndexError, TypeError):
        return None
