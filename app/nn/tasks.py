"""Zadania sieci: co siec przewiduje, skad bierze przyklady i jak buduje tekst wejsciowy.

Kazde zadanie zastepuje (albo kontroluje) jedno konkretne pytanie do LLM:
  page_kind     - classify.py: "czy ta strona to przepis, ciekawostka, czy nic ciekawego"   (decyzja)
  recipe_type   - fast_translate.llm_classify: "jaki to typ przepisu"                        (decyzja)
  fact_section  - extract.py: sekcja informacji o roslinie wybrana przez LLM                 (kontrola)
  fact_part     - extract.py: czesc rosliny, ktorej dotyczy informacja                       (kontrola)
Etykiety pochodza od "nauczyciela": LLM (zweryfikowany drugim modelem, gdzie sie da), archiwum Kwiatownika 1,
dane strukturalne JSON-LD, a najwyzsza wage maja poprawki czlowieka z panelu.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from urllib.parse import urlparse

from sqlmodel import Session, col, select

from app.models import Item, ItemKind, ItemStatus, NNSample, PlantFact

RARE = "_inne"            # zbiorcza etykieta rzadkich klas - siec nigdy nie przyznaje jej sama (zawsze LLM)
HUMAN_WEIGHT = 3.0


@dataclass
class Dataset:
    texts: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    groups: list[str] = field(default_factory=list)
    refs: list[str] = field(default_factory=list)
    origins: list[str] = field(default_factory=list)

    def add(self, text: str, label: str, weight: float, group: str, ref: str, origin: str) -> None:
        if not text or not text.strip() or not label:
            return
        self.texts.append(text)
        self.labels.append(label)
        self.weights.append(float(weight))
        self.groups.append(group or "?")
        self.refs.append(ref)
        self.origins.append(origin)

    def __len__(self) -> int:
        return len(self.texts)

    def merge_rare(self, min_per_class: int, allow_rare: bool) -> dict[str, int]:
        """Klasy z mniej niz min_per_class przykladami -> RARE (albo usuniete). Zwraca, co polaczono."""
        counts = Counter(self.labels)
        rare = {k: v for k, v in counts.items() if v < min_per_class and k != RARE}
        if not rare:
            return {}
        keep = [i for i, lab in enumerate(self.labels) if lab not in rare or allow_rare]
        for name in ("texts", "weights", "groups", "refs", "origins", "labels"):
            setattr(self, name, [getattr(self, name)[i] for i in keep])
        if allow_rare:
            self.labels = [RARE if lab in rare else lab for lab in self.labels]
        return rare


# ------------------------------------------------------------------ teksty wejsciowe (wspolne dla treningu i potoku)
def page_input(url: str, title: str | None, text: str | None) -> str:
    path = urlparse(url or "").path.replace("-", " ").replace("_", " ").replace("/", " ")
    return f"{path.strip()}\n{(title or '').strip()}\n{(text or '').strip()[:1500]}"


def recipe_input(title: str | None, ingredients: list[str], steps: list[str]) -> str:
    return (f"{(title or '').strip()}\nSkladniki: {', '.join(x for x in ingredients if x)[:600]}\n"
            f"Przygotowanie: {' '.join(x for x in steps if x)[:800]}")


def fact_input(text: str | None, quote: str | None) -> str:
    return f"{(text or '').strip()}\n{(quote or '').strip()[:200]}"


def title_group(title: str | None) -> str:
    from app.knowledge.sections import norm_text
    words = [w for w in norm_text(title or "").split() if len(w) > 2][:3]
    return "t:" + " ".join(w[:6] for w in words) if words else "t:?"


def _domain(url: str) -> str:
    return urlparse(url or "").netloc.lower().removeprefix("www.") or "?"


def _names(skl) -> list[str]:
    out = []
    for x in skl or []:
        if isinstance(x, dict):
            out.append(str(x.get("nazwa") or x.get("name") or ""))
        elif x:
            out.append(str(x))
    return out


def _steps(v) -> list[str]:
    if isinstance(v, list):
        return [str(x.get("opis") if isinstance(x, dict) else x) for x in v if x]
    return [str(v)] if v else []


def recipe_fields(data: dict) -> tuple[str, list[str], list[str]]:
    return (str(data.get("tytul") or ""), _names(data.get("skladniki")), _steps(data.get("sposob_przygotowania")))


# ------------------------------------------------------------------ zbiory danych
def _human(s: Session, task: str, ds: Dataset, taken: set[str]) -> None:
    for smp in s.exec(select(NNSample).where(NNSample.task == task, NNSample.origin == "czlowiek")).all():
        ds.add(smp.text, smp.label, smp.weight or HUMAN_WEIGHT, smp.group or "?", smp.ref, "czlowiek")
        taken.add(smp.ref)


def build_page_kind(s: Session) -> Dataset:
    ds, taken = Dataset(), set()
    _human(s, "page_kind", ds, taken)
    for smp in s.exec(select(NNSample).where(NNSample.task == "page_kind", NNSample.origin != "czlowiek")
                      .order_by(col(NNSample.id).desc())).all():
        if smp.ref in taken or smp.origin.startswith("nn:") or not smp.weight:
            continue                      # decyzje samej sieci nie sa etykietami (bez uczenia sie od siebie)
        taken.add(smp.ref)
        ds.add(smp.text, smp.label, smp.weight, smp.group or _domain(smp.ref), smp.ref, smp.origin)
    # rozruch: strony z tekstem juz zapisanym w bazie (przepisy i ciekawostki, przepisy kulinarne odrzucone pozniej)
    rows = s.exec(select(Item).where(col(Item.raw_text).is_not(None),
                                     col(Item.kind).in_([ItemKind.recipe, ItemKind.fact]))).all()
    for it in rows:
        if it.url in taken or not it.raw_text or len(it.raw_text) < 200:
            continue
        method = it.method or ""
        reason = (it.reason or "").lower()
        if it.kind == ItemKind.recipe and it.status == ItemStatus.skipped and (
                reason.startswith("przepis kulinarny") or reason.startswith("rosliny nie sa kluczowym")):
            label, w = "other", 0.8                         # tlumaczenie pokazalo, ze to zwykly przepis kulinarny
        elif method.startswith("llm:") or method == "jsonld":
            label, w = it.kind.value, 1.0
        elif method in ("heuristic", "podzial-artykulu", "profile:artykul"):
            label, w = it.kind.value, 0.5
        else:
            continue
        ds.add(page_input(it.url, it.title, it.raw_text), label, w, _domain(it.url), it.url, method or "?")
    return ds


def build_recipe_type(s: Session) -> Dataset:
    ds, taken = Dataset(), set()
    _human(s, "recipe_type", ds, taken)
    rows = s.exec(select(Item).where(Item.kind == ItemKind.recipe, col(Item.data_json).is_not(None),
                                     col(Item.duplicate_of).is_(None))).all()
    for it in rows:
        ref = f"item:{it.id}"
        if ref in taken:
            continue
        try:
            data = json.loads(it.data_json)
        except ValueError:
            continue
        if not isinstance(data, dict):
            continue
        typ = str(data.get("typ") or "").strip().lower()
        if not typ or typ == "inne":
            continue
        how = (data.get("siedziba") or {}).get("typ_ustalono") if isinstance(data.get("siedziba"), dict) else None
        approved = it.status in (ItemStatus.approved, ItemStatus.exported)
        if str(how or "").startswith("nn:") and not approved:
            continue                     # wlasna decyzja sieci bez potwierdzenia - nie uczymy sie od siebie
        if str(how or "").startswith("nn:"):
            origin, w = "zatwierdzony", 1.0
        elif it.method == "archiwum":
            origin, w = "archiwum", 1.0
        elif how == "reguly":
            origin, w = "reguly", 0.7
        else:
            origin, w = f"llm:{it.translated_by or '?'}", 1.0
        title, ingr, steps = recipe_fields(data)
        group = title_group(title)        # ten sam przepis z kilku stron (prawie ten sam tytul) - jedna grupa
        ds.add(recipe_input(title, ingr, steps), typ, w, group, ref, origin)
    return ds


def _facts(s: Session, task: str, label_of) -> Dataset:
    ds, taken = Dataset(), set()
    _human(s, task, ds, taken)
    rows = s.exec(select(PlantFact).where(col(PlantFact.status).in_(["verified", "applied"]))).all()
    for f in rows:
        ref = f"fact:{f.id}"
        if ref in taken:
            continue
        ds.add(fact_input(f.text, f.quote), label_of(f), 1.0, f.plant_id, ref, f.extracted_by or "llm")
    return ds


def part_label(part: str | None) -> str:
    from app.knowledge.sections import norm_part
    return norm_part(part) or "brak"


def build_fact_section(s: Session) -> Dataset:
    return _facts(s, "fact_section", lambda f: f.section)


def build_fact_part(s: Session) -> Dataset:
    return _facts(s, "fact_part", lambda f: part_label(f.part))


# ------------------------------------------------------------------ rejestr zadan
@dataclass(frozen=True)
class TaskSpec:
    key: str
    title: str
    kind: str                     # decyzja (zastepuje pytanie do LLM) | kontrola (sprawdza odpowiedz LLM)
    replaces: str                 # jakie pytanie do LLM przejmuje
    build: object
    min_per_class: int = 15
    min_total: int = 120
    min_classes: int = 2
    allow_rare: bool = True


TASKS: dict[str, TaskSpec] = {
    "page_kind": TaskSpec(
        "page_kind", "Rodzaj strony", "decyzja",
        "klasyfikacja strony przy skanie (przepis / ciekawostka / nic) - stronom niejednoznacznym dla regul",
        build_page_kind, min_per_class=30, min_total=200, min_classes=3, allow_rare=False),
    "recipe_type": TaskSpec(
        "recipe_type", "Typ przepisu", "decyzja",
        "typ przepisu po tlumaczeniu NLLB, gdy reguly nie rozstrzygna (napar, nalewka, masc, kosmetyk...)",
        build_recipe_type, min_per_class=20, min_total=150),
    "fact_section": TaskSpec(
        "fact_section", "Sekcja informacji", "kontrola",
        "sekcja informacji o roslinie wybrana przez LLM przy wyciaganiu wiedzy - siec zglasza rozbieznosci",
        build_fact_section, min_per_class=15, min_total=200),
    "fact_part": TaskSpec(
        "fact_part", "Czesc rosliny", "kontrola",
        "czesc rosliny przypisana informacji - siec uzupelnia brakujace i zglasza rozbieznosci",
        build_fact_part, min_per_class=15, min_total=200),
}

LABEL_NAMES = {"recipe": "przepis", "fact": "ciekawostka", "other": "nic ciekawego", RARE: "inne (rzadkie)"}


def label_name(label: str | None) -> str:
    return LABEL_NAMES.get(label or "", label or "-")
