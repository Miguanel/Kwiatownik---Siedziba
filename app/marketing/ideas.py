"""Zbieranie potencjalnych ciekawostek do postow (zadanie 'fb_ideas', widok /posts/ideas).

1. Kandydaci:
   - zweryfikowane informacje o roslinach (PlantFact: ciekawostki, kultura, historia, barwienie...) - juz po polsku,
     ze zrodlem i cytatem;
   - artykuly z zagranicznych stron zebrane skanem (Item kind=fact): model wyciaga z nich do 3 ciekawostek
     o roslinach PO POLSKU + DOSLOWNY cytat z oryginalu; kod sprawdza, czy cytat naprawde jest w artykule.
2. Ocena: kod (zagranica, rodzaj informacji, konkret, dlugosc, roslina w Kwiatowniku, swiezosc) + model
   (ciekawosc, zaskoczenie, zrozumialosc 1-10, zajawka, miesiace "na czasie", ryzyko obietnic zdrowotnych).
3. Pula: najlepsze nieuzyte ciekawostki trafiaja do modeli piszacych posty (topics.ciekawostka) - z premia,
   gdy pasuja do biezacego miesiaca."""
import json
import re
from datetime import datetime, timedelta, timezone

from sqlmodel import Session, col, func, select

from app.models import FbIdea, Item, ItemKind, ItemStatus, PlantFact

IDEA_SECTIONS = {"ciekawostki": 25, "kultura": 20, "historia": 20, "barwienie": 18, "nazwy_ludowe": 16,
                 "kosmetyka": 14, "medycyna_wschodu": 14, "wystepowanie": 8, "zastosowanie_lecznicze": 8}
ITEM_WEIGHT = 18
SPECIAL = re.compile(r"najstarsz|największ|najwięks|jedyn|rekord|legend|wierzon|wierzyli|zakazan|tradycj|święt|"
                     r"magi|czarow|obrzęd|zwycza|król|cesar|mnis|klasztor|wojn|żołnierz|starożytn|średniow",
                     re.I)
CYRILLIC = re.compile(r"[а-яА-ЯіїєґІЇЄҐ]")
CJK = re.compile(r"[぀-ヿ一-鿿]")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[„”\"'«»…’`]", "", s or "")).strip().lower()


def is_foreign(lang: str | None, original: str | None = None) -> bool:
    lang = (lang or "").lower()
    if lang:
        return not lang.startswith("pl")
    return bool(original and (CYRILLIC.search(original) or CJK.search(original)))


def code_score(idea: FbIdea, plant_ids: set[str], now: datetime | None = None) -> float:
    """0-100 bez modelu: z zagranicy, rodzaj informacji, konkret, dlugosc, roslina w Kwiatowniku, swiezosc."""
    sc = 0.0
    if is_foreign(idea.language, idea.original):
        sc += 25
    sc += IDEA_SECTIONS.get(idea.category or "", ITEM_WEIGHT if idea.source_type == "item" else 10)
    t = idea.text or ""
    if SPECIAL.search(t):
        sc += 10
    if re.search(r"\d", t) or re.search(r"\s[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]{2,}", t[1:]):
        sc += 8
    sc += 15 if 60 <= len(t) <= 350 else 5
    if idea.plant_id and idea.plant_id in plant_ids:
        sc += 10
    created = idea.created_at.replace(tzinfo=idea.created_at.tzinfo or timezone.utc) if idea.created_at else None
    if created and (now or datetime.now(timezone.utc)) - created < timedelta(days=14):
        sc += 7
    return round(min(100.0, sc), 1)


def final_score(idea: FbIdea) -> float:
    if idea.llm_score is None:
        return round(idea.code_score * 0.8, 1)
    return round(0.4 * idea.code_score + 0.6 * idea.llm_score * 10, 1)


# ------------------------------------------------------------------ 1a. kandydaci z PlantFact
def sync_plant_facts(s: Session, plant_names: dict[str, str]) -> int:
    known = set(s.exec(select(FbIdea.key).where(col(FbIdea.key).startswith("fact:"))).all())
    facts = s.exec(select(PlantFact).where(col(PlantFact.status).in_(["verified", "applied"]),
                                           col(PlantFact.section).in_(list(IDEA_SECTIONS)))).all()
    n = 0
    for f in facts:
        key = f"fact:{f.id}"
        if key in known:
            continue
        s.add(FbIdea(key=key, source_type="fact", source_id=f.id, plant_id=f.plant_id,
                     plant_name=plant_names.get(f.plant_id), text=f.text, original=f.quote,
                     source_url=f.source_url, source_name=f.source_name, language=f.language, category=f.section))
        n += 1
    s.commit()
    return n


# ------------------------------------------------------------------ 1b. wyciaganie z artykulow
EXTRACT_SYSTEM = (
    "Jesteś redaktorem fanpage'a o ziołach i roślinach. Z artykułu (w dowolnym języku) wybierz najwyżej 3 "
    "NAJCIEKAWSZE informacje o roślinach lub ziołach: zaskakujące zwyczaje, historię, legendy, dawne zastosowania, "
    "nazwy ludowe, barwienie, rekordy. Pomijaj ogólniki, reklamy i porady zdrowotne bez konkretu; nie wybieraj "
    "obietnic leczenia. Każdą opisz PO POLSKU w 1–3 zdaniach, tylko na podstawie artykułu, i podaj DOSŁOWNY "
    "fragment oryginału (w języku artykułu, 20–300 znaków), na którym się opiera.\n"
    "Odpowiedz WYŁĄCZNIE JSON-em: {\"ciekawostki\": [{\"tekst\": \"...\", \"cytat\": \"...\", "
    "\"roslina\": \"polska nazwa rośliny albo null\"}]} – pusta lista, gdy nic ciekawego o roślinach."
)


def _plant_match(name: str | None, plants: dict[str, str]) -> str | None:
    if not name:
        return None
    n = _norm(name)
    for pid, pl in plants.items():
        if _norm(pl) == n or _norm(pl).split()[0] == n.split()[0]:
            return pid
    return None


async def extract_from_items(s: Session, llm, plant_names: dict[str, str], limit: int, log) -> dict:
    out = {"articles": 0, "ideas": 0, "bad_quote": 0, "empty": 0}
    done_ids = {int(k.split(":")[1].split("#")[0]) for k in
                s.exec(select(FbIdea.key).where(col(FbIdea.key).startswith("item:"))).all()}
    items = s.exec(select(Item).where(Item.kind == ItemKind.fact, col(Item.raw_text).is_not(None),
                                      col(Item.status).not_in([ItemStatus.skipped, ItemStatus.error]))
                   .order_by(col(Item.id).desc())).all()
    todo = [it for it in items if it.id not in done_ids][:limit]
    for it in todo:
        out["articles"] += 1
        text = (it.raw_text or "")[:6000]
        try:
            res = await llm.complete(f"TYTUŁ: {it.title or '-'}\nADRES: {it.url}\n\nARTYKUŁ:\n{text}",
                                     system=EXTRACT_SYSTEM, json_mode=True)
            data = json.loads(re.sub(r"^```(?:json)?|```$", "", res.text.strip(), flags=re.M).strip())
            model = f"{res.provider}:{res.model}"
        except Exception as exc:  # brak modeli / zly JSON - artykul zostaje na pozniej
            log(f"Artykul #{it.id}: {type(exc).__name__}: {exc}")
            continue
        body, n = _norm(it.raw_text or ""), 0
        for c in (data.get("ciekawostki") or [])[:3]:
            if not isinstance(c, dict):
                continue
            tekst, cytat = str(c.get("tekst") or "").strip(), str(c.get("cytat") or "").strip()
            if len(tekst) < 30 or len(_norm(cytat)) < 20 or _norm(cytat) not in body:
                out["bad_quote"] += 1           # bez prawdziwego cytatu = mozliwe zmyslenie
                continue
            n += 1
            pid = _plant_match(c.get("roslina"), plant_names)
            s.add(FbIdea(key=f"item:{it.id}#{n}", source_type="item", source_id=it.id, plant_id=pid,
                         plant_name=plant_names.get(pid) if pid else (str(c.get("roslina") or "")[:80] or None),
                         text=tekst[:700], original=cytat[:400], source_url=it.url,
                         source_name=it.title, language=it.language, extracted_by=model))
        if not n:
            s.add(FbIdea(key=f"item:{it.id}", source_type="item", source_id=it.id, status="empty",
                         source_url=it.url, extracted_by=model))
            out["empty"] += 1
        out["ideas"] += n
        s.commit()
        log(f"Artykul #{it.id} ({it.language or '?'}): ciekawostek {n}")
    return out


# ------------------------------------------------------------------ 2. ocena
RATE_SYSTEM = (
    "Jesteś redaktorem fanpage'a Kwiatownika (cyfrowy zielnik, ziołolecznictwo, dawne przepisy). Oceń, czy "
    "każda ciekawostka nadaje się na post, który zatrzyma czytelnika Facebooka. Skala 1–10:\n"
    "- ciekawosc: czy ludzie zechcą to przeczytać i udostępnić,\n"
    "- zaskoczenie: czy to coś, czego przeciętny Polak nie wie,\n"
    "- zrozumialosc: czy da się to opowiedzieć prosto, bez fachowej wiedzy.\n"
    "ryzyko = true, gdy ciekawostka obiecuje leczenie, podaje dawki lub może zaszkodzić. Dla każdej podaj "
    "zajawkę (1 krótkie zdanie po polsku, haczyk) i miesiące (1–12), w których temat jest na czasie "
    "(pusta lista = cały rok).\n"
    "Odpowiedz WYŁĄCZNIE JSON-em: {\"oceny\": [{\"id\": 1, \"ciekawosc\": 7, \"zaskoczenie\": 6, "
    "\"zrozumialosc\": 8, \"ryzyko\": false, \"zajawka\": \"...\", \"miesiace\": [9, 10]}]}"
)


def _clamp(v) -> int | None:
    try:
        return max(1, min(10, int(round(float(v)))))
    except (TypeError, ValueError):
        return None


async def rate_ideas(s: Session, llm, limit: int, batch: int, log) -> int:
    ideas = s.exec(select(FbIdea).where(FbIdea.status == "new").order_by(col(FbIdea.code_score).desc())
                   .limit(limit)).all()
    rated = 0
    for i in range(0, len(ideas), batch):
        chunk = ideas[i:i + batch]
        listing = "\n".join(f"ID {x.id}: [{x.plant_name or 'roślina?'}; {x.category or x.source_type}; "
                            f"język źródła {x.language or '?'}] {x.text}" for x in chunk)
        try:
            res = await llm.complete(f"CIEKAWOSTKI:\n{listing}", system=RATE_SYSTEM, json_mode=True)
            data = json.loads(re.sub(r"^```(?:json)?|```$", "", res.text.strip(), flags=re.M).strip())
        except Exception as exc:
            log(f"Ocena paczki nieudana: {type(exc).__name__}: {exc}")
            continue
        by_id = {x.id: x for x in chunk}
        for o in data.get("oceny") or []:
            if not isinstance(o, dict):
                continue
            try:
                x = by_id.get(int(o.get("id")))
            except (TypeError, ValueError):
                continue
            vals = [v for v in (_clamp(o.get(k)) for k in ("ciekawosc", "zaskoczenie", "zrozumialosc"))
                    if v is not None]
            if not x or not vals:
                continue
            months = [int(m) for m in (o.get("miesiace") or []) if str(m).isdigit() and 1 <= int(m) <= 12]
            x.llm_score = round(sum(vals) / len(vals), 1)
            x.risk = bool(o.get("ryzyko"))
            x.teaser = str(o.get("zajawka") or "").strip()[:240] or None
            x.months_json = json.dumps(sorted(set(months)))
            x.rated_by = f"{res.provider}:{res.model}"
            x.rated_at = datetime.now(timezone.utc)
            x.status = "rated"
            x.score = final_score(x)
            s.add(x)
            rated += 1
        s.commit()
        log(f"Ocenione: {rated}/{len(ideas)}")
    return rated


def rescore(s: Session, plant_ids: set[str]) -> None:
    for x in s.exec(select(FbIdea).where(FbIdea.status != "empty")).all():
        x.code_score = code_score(x, plant_ids)
        x.score = final_score(x)
        s.add(x)
    s.commit()


# ------------------------------------------------------------------ 3. pula dla postow
def used_idea_ids(s: Session) -> set[int]:
    from app.models import FbPost
    out = set()
    for ref in s.exec(select(FbPost.ref).where(col(FbPost.ref).startswith("idea:"),
                                               FbPost.status != "rejected")).all():
        try:
            out.add(int(ref.split(":")[1]))
        except (IndexError, ValueError):
            pass
    return out


def pool(s: Session, month: int | None = None, plant_id: str | None = None, limit: int = 50,
         include_used: bool = False) -> list[tuple[FbIdea, float]]:
    """Najlepsze ciekawostki do podsuniecia modelowi: (ciekawostka, ocena z premia za pore roku)."""
    used = set() if include_used else used_idea_ids(s)
    used_facts = set()
    if not include_used:   # informacje uzyte juz w postach starsza droga (ref "fact:...")
        from app.models import FbPost
        for ref in s.exec(select(FbPost.ref).where(col(FbPost.ref).startswith("fact:"),
                                                   FbPost.status != "rejected")).all():
            used_facts |= {f"fact:{x}" for x in ref[5:].split(",") if x}
    stmt = select(FbIdea).where(col(FbIdea.status).in_(["rated", "new"]), FbIdea.risk == False)  # noqa: E712
    if plant_id:
        stmt = stmt.where(FbIdea.plant_id == plant_id)
    out = []
    for x in s.exec(stmt).all():
        if x.id in used or x.key in used_facts or not x.text:
            continue
        sc = x.score
        months = json.loads(x.months_json or "[]")
        if month and months and month in months:
            sc += 8
        out.append((x, round(sc, 1)))
    return sorted(out, key=lambda t: -t[1])[:limit]


def counts(s: Session) -> dict:
    rows = dict(s.exec(select(FbIdea.status, func.count()).group_by(FbIdea.status)).all())
    langs = s.exec(select(FbIdea.language).where(col(FbIdea.status).in_(["new", "rated"]))).all()
    foreign = {(lg or "?").split("-")[0].lower() for lg in langs if is_foreign(lg)}
    return {"new": rows.get("new", 0), "rated": rows.get("rated", 0), "rejected": rows.get("rejected", 0),
            "articles_empty": rows.get("empty", 0), "total": rows.get("new", 0) + rows.get("rated", 0),
            "used": len(used_idea_ids(s)), "foreign_langs": sorted(foreign)}


async def collect(engine, llm, plants: dict[str, dict], log=lambda m: None, items_limit: int = 10,
                  rate_limit: int = 48, batch: int = 8, should_stop=lambda: False) -> dict:
    names = {pid: d.get("nazwa_pl") for pid, d in plants.items()}
    out = {}
    with Session(engine) as s:
        out["z_wiedzy"] = sync_plant_facts(s, names)
        log(f"Nowi kandydaci z wiedzy o roslinach: {out['z_wiedzy']}")
        if llm is not None and items_limit and not should_stop():
            out.update(await extract_from_items(s, llm, names, items_limit, log))
        rescore(s, set(plants))
        if llm is not None and rate_limit and not should_stop():
            out["ocenione"] = await rate_ideas(s, llm, rate_limit, max(1, batch), log)
        top = pool(s, limit=5)
        log("Najlepsze: " + " | ".join(f"{sc:.0f} {x.plant_name or '?'}: {(x.teaser or x.text)[:60]}"
                                       for x, sc in top))
        out.update(counts(s))
    return out
