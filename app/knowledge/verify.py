"""Weryfikacja informacji przez drugi model LLM (inny niz ten, ktory je wyciagnal).

Model sprawdza kazda informacje z jej cytatem ze zrodla: czy cytat ja potwierdza, czy dotyczy tej rosliny,
czy tekst jest poprawna polszczyzna i czy nie brzmi jak niebezpieczna porada. Niepotwierdzone -> odrzucone.
"""
import json

BATCH = 12

SYSTEM_PROMPT = """Jestes weryfikatorem faktow polskiego zielnika. Dla kazdej informacji dostajesz jej tekst \
po polsku i CYTAT ze zrodla (moze byc w innym jezyku). Ocen kazda informacje:
- ok=true tylko jesli: cytat POTWIERDZA tresc (bez dodanych szczegolow), dotyczy wskazanej rosliny,
  tekst jest poprawna polszczyzna i nie zawiera niebezpiecznej porady (np. dawkowania trujacej rosliny
  bez ostrzezenia, zachety do stosowania zamiast leczenia).
- W przeciwnym razie ok=false i krotki powod.
Zwroc TYLKO JSON: {"oceny": [{"id": 1, "ok": true, "powod": ""}]}"""


def build_prompt(plant_pl: str, latin: str | None, facts: list) -> str:
    items = "\n\n".join(f"ID {f.id}\nINFORMACJA: {f.text}\nCYTAT ({f.language or '?'}): {f.quote}" for f in facts)
    return f"ROSLINA: {plant_pl}" + (f" ({latin})" if latin else "") + f"\n\n{items}"


def parse_verdicts(raw: str, ids: list[int]) -> dict[int, tuple[bool, str]]:
    data = json.loads(raw.strip().removeprefix("```json").removesuffix("```"))
    out: dict[int, tuple[bool, str]] = {}
    if not isinstance(data, dict):
        return out
    for v in data.get("oceny") or []:
        if not isinstance(v, dict):
            continue
        try:
            fid = int(v.get("id"))
        except (TypeError, ValueError):
            continue
        if fid in ids:
            out[fid] = (v.get("ok") is True, str(v.get("powod") or "")[:200])
    return out            # brak oceny = nie zweryfikowano (zostaje do kolejnej proby)


async def verify_facts(llm, plant_pl: str, latin: str | None, facts: list,
                       avoid: set[str] | None = None) -> tuple[dict[int, tuple[bool, str]], str]:
    extra = {"avoid": avoid} if avoid else {}
    res = await llm.complete(build_prompt(plant_pl, latin, facts), system=SYSTEM_PROMPT, json_mode=True, **extra)
    return parse_verdicts(res.text, [f.id for f in facts]), f"{res.provider}/{res.model}"
