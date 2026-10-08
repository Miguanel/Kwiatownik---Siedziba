"""Czy taki przepis juz istnieje? Porownanie PO POLSKU (po tlumaczeniu) z:
 - przepisami, ktore juz sa w Kwiatowniku (data/przepisy/*.json),
 - innymi przetlumaczonymi przepisami z Siedziby (tez z innych jezykow!).
Jasne przypadki rozstrzyga podobienstwo tytulu i skladnikow; watpliwe - LLM ("czy to ten sam przepis?").
"""
import json
import logging
import re
import unicodedata
from dataclasses import dataclass

log = logging.getLogger(__name__)

PL_STOP = {
    "przepis", "domowy", "domowa", "domowe", "prosty", "prosta", "najlepszy", "sposob", "sposób", "na", "z", "ze",
    "do", "w", "we", "i", "oraz", "lub", "albo", "dla", "po", "od", "bez", "the", "and", "with", "of", "a",
    "lyzka", "lyzki", "lyzek", "lyzeczka", "lyzeczki", "szklanka", "szklanki", "garsc", "szczypta", "g", "kg", "ml",
    "l", "litr", "litra", "sztuka", "sztuki", "okolo", "ok", "swiezy", "swieze", "swiezych", "suszony", "suszone",
    "suszonych", "posiekany", "drobno", "duzy", "duze", "maly", "male", "opcjonalnie", "smak", "do_smaku", "cup",
}


def pl_tokens(text: str) -> set[str]:
    t = unicodedata.normalize("NFKD", (text or "").lower().replace("ł", "l"))
    t = "".join(c for c in t if not unicodedata.combining(c))
    words = re.findall(r"[a-z]+", t)
    # prymitywny stemming dla polskiej odmiany: syrop/syropu -> "syro", woda/wody -> "wod", kwiaty/kwiatow -> "kwia"
    return {w[:min(4, len(w) - 1)] for w in words if len(w) > 2 and w not in PL_STOP}


def fingerprint(title: str, ingredients: list[str]) -> tuple[set[str], set[str]]:
    return pl_tokens(title), pl_tokens(" ".join(ingredients))


def _j(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def score(fp1, fp2) -> tuple[float, float]:
    return _j(fp1[0], fp2[0]), _j(fp1[1], fp2[1])


def verdict(t: float, i: float) -> str:
    if (t >= 0.5 and i >= 0.5) or (i >= 0.8 and t >= 0.3):
        return "same"
    if (t >= 0.3 and i >= 0.25) or t >= 0.6 or i >= 0.6:
        return "maybe"
    return "no"


@dataclass
class Candidate:
    kind: str            # "kwiatownik" | "siedziba"
    ref: str             # plik#id albo id itemu
    title: str
    ingredients: list[str]


@dataclass
class Match:
    candidate: Candidate
    t: float
    i: float
    how: str             # "podobienstwo" | "llm"
    reason: str = ""


JUDGE_PROMPT = """Porownaj dwa przepisy. Czy to TEN SAM przepis (ta sama potrawa/preparat, te same glowne skladniki \
i metoda; drobne roznice ilosci lub dodatkow nie maja znaczenia)? Odpowiedz TYLKO JSON: \
{"same": true/false, "reason": "max 15 slow"}"""


async def find_match(title: str, ingredients: list[str], candidates: list[Candidate], llm=None,
                     max_llm_checks: int = 3) -> Match | None:
    fp = fingerprint(title, ingredients)
    scored = []
    for c in candidates:
        t, i = score(fp, fingerprint(c.title, c.ingredients))
        v = verdict(t, i)
        if v != "no":
            scored.append((t + i, v, t, i, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    for _, v, t, i, c in scored:
        if v == "same":
            return Match(c, round(t, 2), round(i, 2), "podobienstwo")
    if llm is None:
        return None
    for _, v, t, i, c in scored[:max_llm_checks]:
        prompt = (f"PRZEPIS A: {title}\nSkladniki: {'; '.join(ingredients)}\n\n"
                  f"PRZEPIS B: {c.title}\nSkladniki: {'; '.join(c.ingredients)}")
        try:
            res = await llm.complete(prompt, system=JUDGE_PROMPT, json_mode=True)
            data = json.loads(res.text.strip().removeprefix("```json").removesuffix("```"))
        except Exception as exc:
            log.warning("LLM nie ocenil podobienstwa: %s", exc)
            continue
        if data.get("same") is True:
            return Match(c, round(t, 2), round(i, 2), "llm", str(data.get("reason", ""))[:200])
    return None
