"""Filtr nowosci przed ekstrakcja (etap 2 diagnostyki 2026-11-02).

Diagnoza: 712 z 887 wywolan ekstrakcji (80%) nie dalo zadnej nowej informacji; 639 z 847 odczytow Wikipedii to
ponowne czytanie tych samych artykulow (pl "Swierzabek bulwiasty" 23 razy). Teraz kazde zrodlo ma liste skrotow
akapitow juz przeczytanych dla tej rosliny (tabela SourceRead) i do LLM trafiaja tylko:
- akapity nowe albo zmienione od ostatniego odczytu,
- na stronach ogolnych (wyszukiwarka, ciekawostki ze skanera) tylko akapity, ktore wymieniaja rosline
  (i akapit po nich - czesto ciag dalszy), bo reszta strony mowi o czyms innym,
- gdy zostaje mniej niz KNOWLEDGE_MIN_NEW_CHARS znakow - bez wywolania LLM.
Wszystko bez modelu: skroty i zwykle porownanie tekstu. Sprawdzanie cytatow i liczb dziala jak dawniej.
"""
import hashlib
import json
import re
from dataclasses import dataclass, field

from app.knowledge.extract import is_cjk
from app.knowledge.sections import norm_text

MIN_PARA = 200            # krotkie linie (naglowki, punkty listy) laczone z kolejnymi do takiej dlugosci
MAX_PARA = 1500


def paragraphs(text: str) -> list[str]:
    """Tekst -> akapity (puste linie / nowe linie), krotkie kawalki laczone z nastepnymi."""
    parts = [p.strip() for p in re.split(r"\n\s*\n|\n", text or "") if p.strip()]
    out: list[str] = []
    buf = ""
    for p in parts:
        buf = f"{buf}\n{p}" if buf else p
        if len(buf) >= (MIN_PARA // 3 if is_cjk(buf) else MIN_PARA):
            out.extend(_cut(buf))
            buf = ""
    if buf:
        if out and len(buf) < 60:
            out[-1] = f"{out[-1]}\n{buf}"
        else:
            out.append(buf)
    return out


def _cut(p: str) -> list[str]:
    if len(p) <= MAX_PARA:
        return [p]
    sentences = re.split(r"(?<=[.!?。！？])\s*", p)
    out, buf = [], ""
    for s in sentences:
        if buf and len(buf) + len(s) > MAX_PARA:
            out.append(buf.strip())
            buf = ""
        buf += s + " "
    if buf.strip():
        out.append(buf.strip())
    return out


def para_hash(p: str) -> str:
    return hashlib.sha1(norm_text(p).encode("utf-8")).hexdigest()[:16]


def mentions(p: str, names: list[str]) -> bool:
    low = p.lower()
    return any(n and n in low for n in names)


@dataclass
class Novelty:
    text: str                                         # to, co pojdzie do LLM ("" = nic)
    done: list[str] = field(default_factory=list)     # skroty do zapamietania po odczycie: przeczytane wczesniej,
    #                                                   wyslane teraz i nie o tej roslinie (za krotkie nowe - nie)
    total: int = 0                                    # akapitow w zrodle
    seen: int = 0                                     # z nich juz przeczytanych
    off_topic: int = 0                                # pominietych, bo nie wymieniaja rosliny

    @property
    def why_skip(self) -> str | None:
        if not self.total:
            return "pusty tekst"
        if self.seen == self.total:
            return "bez zmian od ostatniego odczytu"
        if not self.text:
            return "nowe akapity nie dotycza tej rosliny" if self.off_topic else "za malo nowego tekstu"
        return None


def select_new(text: str, seen: set[str], names: list[str] | None = None, min_chars: int = 200) -> Novelty:
    """Akapity do wyslania do LLM. names (male litery) = strona ogolna: tylko akapity z nazwa rosliny."""
    paras = paragraphs(text)
    hashes = [para_hash(p) for p in paras]
    fresh = {i for i, h in enumerate(hashes) if h not in seen}
    keep = set(fresh)
    if names:
        hit = {i for i in fresh if mentions(paras[i], names)}
        keep = hit | {i + 1 for i in hit if i + 1 in fresh}          # akapit po wzmiance - zwykle ciag dalszy
    off_topic = fresh - keep
    chosen = "\n\n".join(paras[i] for i in sorted(keep))
    if len(chosen) < (min_chars // 3 if is_cjk(chosen) else min_chars):
        chosen, keep = "", set()      # drobna zmiana: bez LLM; te akapity poczekaja, az dojdzie wiecej nowego
    done = [h for i, h in enumerate(hashes) if i not in fresh or i in keep or i in off_topic]
    return Novelty(chosen, done, len(paras), len(paras) - len(fresh), len(off_topic))


def loads(paragraphs_json: str | None) -> set[str]:
    try:
        return set(json.loads(paragraphs_json or "[]"))
    except ValueError:
        return set()
