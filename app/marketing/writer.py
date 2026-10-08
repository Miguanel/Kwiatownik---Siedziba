"""Pisanie posta na fanpage glosem postaci (LLM) + kontrola kodem i skladanie gotowego tekstu."""
import json
import re
from dataclasses import dataclass, field

from app.marketing.persona import Persona
from app.marketing.picker import PlantPick
from app.marketing.season import Season

_WORD = re.compile(r"[\wąćęłńóśźżĄĆĘŁŃÓŚŹŻ'-]+", re.U)
_URL = re.compile(r"(https?://\S+|www\.\S+)", re.I)
_EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
_TAG = re.compile(r"#[\wąćęłńóśźżĄĆĘŁŃÓŚŹŻ]+", re.U)


def word_count(text: str) -> int:
    return len(_WORD.findall(text or ""))


@dataclass
class Material:
    """Temat posta przygotowany kodem: dane dla modelu + link i zrodlo doklejane przez system."""
    kind: str                         # siedziba | pora_roku | ciekawostka | przepis
    label: str                        # krotki opis do logu i panelu
    task: str                         # co napisac (polecenie dla modelu)
    data: str                         # fakty, na ktorych model ma sie oprzec
    season: Season | None = None
    main_name: str | None = None      # nazwa, ktora musi pasc w tresci (kontrola kodem)
    main_latin: str | None = None
    link_text: str | None = None      # None = Persona.link_text
    link_url: str | None = None
    sources: list[str] = field(default_factory=list)   # adresy zrodel (linia "Zrodlo: ...")
    plants: list[dict] = field(default_factory=list)   # [{id, nazwa_pl, nazwa_lat, url}]
    ref: str | None = None            # co wykorzystano (fakty/przepis) - zeby sie nie powtarzac


def system_prompt(p: Persona) -> str:
    return (
        f"Jesteś „{p.name}” – postacią, która pisze posty na fanpage Kwiatownika (cyfrowy zielnik o roślinach, "
        f"ziołach i dawnych przepisach).\n\nKIM JESTEŚ: {p.bio}\n\nJAK MÓWISZ: {p.style}\n\n"
        "ZASADY:\n"
        "- Piszesz po polsku, w pierwszej osobie, jak gawędę starego człowieka do sąsiadów. Bez nagłówków, list i punktorów.\n"
        "- Pierwsze zdanie ma zaciekawić (krótkie, najwyżej 15 słów) – na telefonie widać tylko początek posta.\n"
        "- Podziel tekst na 2–4 krótkie akapity oddzielone pustą linią – żadnej ściany tekstu.\n"
        "- Zakończ pytaniem do czytelników (np. jak u nich na to mówią, czy znają taki zwyczaj).\n"
        "- Wiedzę bierzesz TYLKO z sekcji DANE. Niczego nie zmyślaj – ani o roślinach, ani o tym, jak coś działa; "
        "wspomnienia, porównania i humor mogą być twoje.\n"
        "- Nazwy roślin i przepisów podawaj poprawnie (raz wprost albo w nawiasie), żeby czytelnik wiedział, o czym mowa.\n"
        "- Nie wymyślaj „starych przysłów” udając, że są prawdziwe – jak chcesz, powiedz własną mądrość („jo to godom, że…”).\n"
        "- Żadnych obietnic wyleczenia ani dawek leków. Jeśli w danych jest OSTRZEŻENIE, wspomnij o nim krótko po swojemu.\n"
        "- Nie podpisuj się i nie wstawiaj linków – podpis, link i źródło doklei system. Możesz zachęcić, "
        "żeby zajrzeć do Kwiatownika.\n"
        f"- Długość treści: {p.min_words}–{p.max_words} słów. Najwyżej {p.max_emoji} emoji.\n"
        "- Odpowiedz WYŁĄCZNIE JSON-em: {\"tresc\": \"tekst posta\", \"hashtagi\": [\"#...\"]} "
        "– 2 do 4 hashtagów pasujących do tematu, bez polskich znaków w hashtagach."
    )


def user_prompt(m: Material, hint: str | None = None, feedback: str | None = None) -> str:
    parts = []
    if m.season:
        folk = ("\nŚWIĘTA I TRADYCJE W POBLIŻU: " + "; ".join(m.season.folk)) if m.season.folk else ""
        parts.append(f"DZIŚ: {m.season.day.strftime('%d.%m.%Y')} – {m.season.label}. "
                     f"Możesz nawiązać do pory roku: {m.season.description}{folk}")
    parts.append(f"\nZADANIE: {m.task}")
    parts.append(f"\nDANE:\n{m.data}")
    if hint:
        parts.append(f"\nŻYCZENIE REDAKTORA: {hint}")
    if feedback:
        parts.append(f"\nPOPRAW POPRZEDNIĄ WERSJĘ: {feedback}")
    parts.append("\nNapisz post.")
    return "\n".join(parts)


def plant_block(p: PlantPick) -> str:
    lines = [f"- {p.nazwa_pl}" + (f" ({p.nazwa_lat})" if p.nazwa_lat else "")]
    for t in p.tasks:
        bits = [t.get("czynnosc", "")]
        if t.get("opis"):
            bits.append(t["opis"])
        if t.get("pora_dnia"):
            bits.append(f"pora: {t['pora_dnia']}")
        if t.get("faza_ksiezyca"):
            bits.append(f"księżyc: {t['faza_ksiezyca']}")
        lines.append("  teraz: " + " – ".join(b for b in bits if b))
    for k, v in p.uses.items():
        lines.append(f"  zastosowanie {k}: {v}")
    for f in p.facts:
        lines.append(f"  ciekawostka: {f}")
    if p.recipes:
        lines.append("  przepisy w Kwiatowniku: " + "; ".join(p.recipes))
    if p.warning:
        lines.append(f"  OSTRZEŻENIE: {p.warning}")
    return "\n".join(lines)


@dataclass
class Draft:
    body: str
    hashtags: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    changes: list[str] = field(default_factory=list)   # "wdrozone": jak model wdrozyl uwagi z raportu


def _stem(name: str) -> str:
    word = (name or "").split()[0].lower() if name else ""
    return word[:4]


def parse(text: str, persona: Persona, m: Material) -> Draft:
    raw = (text or "").strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    try:
        data = json.loads(raw)
    except ValueError:
        m = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(m.group(0)) if m else {"tresc": raw}
    body = str(data.get("tresc") or data.get("text") or "").strip()
    body = _URL.sub("", body)
    # model podpisal sie mimo zakazu - usun ostatnia linijke z imieniem postaci
    lines = body.rstrip().splitlines()
    if lines and persona.name.lower() in lines[-1].lower() and len(lines[-1]) < len(persona.name) + 25:
        body = "\n".join(lines[:-1]).rstrip()
    tags = data.get("hashtagi") or data.get("hashtags") or []
    if isinstance(tags, str):
        tags = tags.split()
    tags = ["#" + t.lstrip("#").replace(" ", "") for t in tags if isinstance(t, str) and t.strip("# ")]
    body = re.sub(r"(\s*#[\w]+)+\s*$", "", body, flags=re.U).rstrip()   # hashtagi w tresci -> do listy
    changes = data.get("wdrozone") or []
    if isinstance(changes, str):
        changes = [changes]
    d = Draft(body=body, hashtags=tags[:4], changes=[str(c).strip()[:300] for c in changes if str(c).strip()][:15])
    n = word_count(body)
    lo, hi = int(persona.min_words * 0.85), int(persona.max_words * 1.15)
    if not body:
        d.problems.append("pusta treść")
    elif n < lo:
        d.problems.append(f"za krótki ({n} słów, potrzeba {persona.min_words}–{persona.max_words})")
    elif n > hi:
        d.problems.append(f"za długi ({n} słów, potrzeba {persona.min_words}–{persona.max_words})")
    if m.main_name:
        names = [_stem(m.main_name)] + ([m.main_latin.split()[0].lower()] if m.main_latin else [])
        if body and not any(s and s in body.lower() for s in names):
            d.problems.append(f"nie pada nazwa: {m.main_name}")
    if len(_EMOJI.findall(body)) > persona.max_emoji + 1:
        d.problems.append(f"za dużo emoji (najwyżej {persona.max_emoji})")
    return d


def compose(draft: Draft, persona: Persona, m: Material) -> str:
    """Gotowy tekst do wklejenia na Facebooka: tresc + podpis + link + zrodlo + hashtagi."""
    tags, seen = [], set()
    for t in _TAG.findall(persona.hashtags or "") + draft.hashtags:
        if t.lower() not in seen:
            seen.add(t.lower())
            tags.append(t)
    parts = [draft.body.strip()]
    if persona.signature.strip():
        parts.append(persona.signature.strip())
    if m.link_url:
        parts.append(f"{(m.link_text if m.link_text is not None else persona.link_text).strip()} {m.link_url}".strip())
    if m.sources:
        parts.append(("Źródło: " if len(m.sources) == 1 else "Źródła: ") + " ".join(m.sources))
    if tags:
        parts.append(" ".join(tags))
    return "\n\n".join(parts)


async def write_post(llm, persona: Persona, m: Material, hint: str | None = None,
                     log=lambda msg: None, attempts: int = 2, revise: str | None = None) -> tuple[Draft, str]:
    """Zwraca (szkic, model). Gdy kontrola znajdzie problemy - jedna poprawka z uwagami dla modelu.
    revise: poprzednia wersja + uwagi recenzenta (przepisanie posta wedlug oceny)."""
    system = system_prompt(persona)
    feedback, draft, model = revise, None, ""
    for n in range(1, attempts + 1):
        res = await llm.complete(user_prompt(m, hint, feedback), system=system, json_mode=True)
        model = f"{res.provider}:{res.model}"
        try:
            draft = parse(res.text, persona, m)
        except ValueError:
            draft = Draft(body="", problems=["odpowiedź nie jest poprawnym JSON-em"])
        log(f"Proba {n}: {model}, {word_count(draft.body)} slow" +
            (f", uwagi: {'; '.join(draft.problems)}" if draft.problems else ", OK"))
        if not draft.problems:
            break
        feedback = ((revise + "\n\nDODATKOWO: ") if revise else "") + "; ".join(draft.problems)
    return draft, model
