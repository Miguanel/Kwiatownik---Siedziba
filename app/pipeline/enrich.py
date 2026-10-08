"""Opracowanie przepisu w ujeciu tradycyjnych systemow (LLM) - po tlumaczeniu, przed eksportem.

Tlumaczenie (translate.py) oddaje wiernie to, co jest w zrodle. Tutaj LLM DOPISUJE interpretacje w stylu
Kwiatownika (wzorzec_przepisu.json): Piec Przemian (Wu Xing), triada Kampo, doktryna sygnatur, humory Galena,
alchemia spagiryczna, chronoterapia - oraz ostrzezenia.
Celowo BEZ ajurwedy i wzorca 5 filarow: dla ziol europejskich nie ma ich w zrodlach (model je wymyslal), wiec
skladniki nie dostaja pol "filar", "tropizm_organowy" ani "smak_ajurweda" - a stare sa usuwane (strip_invented).

Zasady bezpieczenstwa (wymuszane tez w kodzie):
- nie zmienia skladnikow, ilosci ani krokow ze zrodla - tylko dodaje pola interpretacyjne,
- dawkowania nie wymysla: gdy zrodlo go nie podaje, zostaje "Źródło nie podaje dawkowania",
- kazde opracowanie ma blok "opracowanie" z uwaga, ze to interpretacja tradycyjna, a nie porada medyczna.
"""
import json
import re
from datetime import date

from app.llm.base import InvalidOutputError

SYSTEMS: dict[str, str] = {
    "wu_xing": "Pięć Przemian (Wu Xing): Drewno, Ogień, Ziemia, Metal, Woda; cykl odżywiania matka-syn -> "
               "klasyfikacja_dzialania.matryca_wu_xing {zywiol_leczony, narzad_matczyny_do_wsparcia, porada_matrycy}",
    "kampo": "Triada Kampo: Ki (energia), Ketsu (krew), Sui (płyny) -> klasyfikacja_dzialania.triada_kampo "
             "{cel_glowny, wyjasnienie}",
    "doktryna_sygnatur": "Doktryna Sygnatur (Paracelsus): wygląd/kolor/siedlisko rośliny a leczony organ -> "
                         "klasyfikacja_dzialania.doktryna_sygnatur {sygnatura, interpretacja}",
    "humory": "Humoralizm Galena: jakości ciepłe/zimne, suche/wilgotne i stopień (I-IV) -> "
              "klasyfikacja_dzialania.humory_galena {jakosc, stopien, wyjasnienie}",
    "spagiryka": "Alchemia spagiryczna: Sal, Sulphur, Mercurius; separacja, kalcynacja, kohobacja -> "
                 "klasyfikacja_dzialania.alchemia_spageryczna {zasada, wyjasnienie}",
    "chronoterapia": "Chronoterapia (zegar narządów TCM), tradycje zbioru (fazy księżyca, pora dnia) i zakazy "
                     "dietetyczne -> wymogi_szamanskie_i_czasowe {chronoterapia, astrologia_zbioru, dieta_i_zakazy}",
}
# Pola i systemy, ktorych Siedziba juz nie tworzy (wymyslane przez model) - usuwane tez z zapisanych przepisow
INVENTED_INGREDIENT_KEYS = ("filar", "tropizm_organowy", "smak_ajurweda")
INVENTED_KD_KEYS = ("ajurweda",)
NO_DOSAGE = "Źródło nie podaje dawkowania."
DISCLAIMER = ("Opracowanie jest interpretacją w ujęciu tradycyjnych systemów ziołolecznictwa (m.in. Pięć Przemian, "
              "Kampo, humory Galena) przygotowaną automatycznie na podstawie przepisu źródłowego. Ma charakter "
              "edukacyjny i historyczny - nie jest poradą medyczną. Przed stosowaniem, zwłaszcza w ciąży, "
              "u dzieci, przy chorobach przewlekłych i przyjmowanych lekach, skonsultuj się z lekarzem lub farmaceutą.")

SYSTEM_PROMPT = """Jestes zielarzem-redaktorem polskiego serwisu "Kwiatownik", ktory opisuje przepisy \
ziolowe (lecznicze, nalewki, barwierskie, kosmetyczne) w ujeciu dawnych systemow ziololecznictwa. \
Dostajesz przepis juz przetlumaczony na polski. Twoje zadanie: DOPISAC opracowanie interpretacyjne.
ZASADY:
- NIE zmieniaj skladnikow, ilosci ani krokow. Nie wymyslaj dawkowania - jesli nie ma go w przepisie,
  w "dawkowanie_standardowe" wpisz null.
- Opracowanie to interpretacja tradycyjna - pisz "w tradycji...", "wedlug Wu Xing..."; nie obiecuj wyleczenia.
- Bezpieczenstwo ma pierwszenstwo: wymien znane przeciwwskazania, toksycznosc (np. rosliny trujace, alkohol,
  ciaza, dzieci, interakcje np. z lekami przeciwzakrzepowymi), a przy barwieniu - srodki ostroznosci przy
  zaprawach (np. alun, siarczan zelaza/miedzi) i probe uczuleniowa. Jesli nic nie wiesz - napisz to wprost.
- Nie przypisuj skladnikom rol, filarow, smakow ajurwedyjskich ani tropizmu - tylko pola ponizej.
- Uzywaj tylko systemow z listy SYSTEMY. Gdy system nie pasuje do przepisu (np. barwienie tkanin a Kampo),
  daj null zamiast naciagania.
- Po polsku, zwiezle (kazde pole 1-3 zdania).
Zwroc TYLKO JSON:
{"mechanizm_tworzenia": ["np. Ekstrakcja alkoholowa", "Maceracja"],
 "klasyfikacja_dzialania": {
   "skalowanie_toksykologiczne": "...",
   "matryca_wu_xing": {"zywiol_leczony": "...", "narzad_matczyny_do_wsparcia": "...", "porada_matrycy": "..."} lub null,
   "triada_kampo": {"cel_glowny": "...", "wyjasnienie": "..."} lub null,
   "doktryna_sygnatur": {"sygnatura": "...", "interpretacja": "..."} lub null,
   "humory_galena": {"jakosc": "...", "stopien": "...", "wyjasnienie": "..."} lub null,
   "alchemia_spageryczna": {"zasada": "...", "wyjasnienie": "..."} lub null},
 "wymogi_szamanskie_i_czasowe": {"chronoterapia": "...", "astrologia_zbioru": "...", "dieta_i_zakazy": "..."} lub null,
 "stosowanie_i_dawkowanie": {"okolicznosci_stosowania": "...", "dawkowanie_standardowe": "... lub null",
                             "skalowanie_pacjenta": "ostroznosci: dzieci, ciaza, seniorzy"},
 "wlasciwosci_fizyczne": {"konsystencja_i_slady": "...", "barwienie_skory": "...", "barwienie_ubran": "..."},
 "bezpieczenstwo_i_interakcje": {"interakcje_z_lekami": "...", "ostrzezenia": "..."},
 "wskazowki_tradycyjne": ["krotka wskazowka przygotowania wg tradycji, np. zbior, maceracja, kohobacja"]}"""

KD_KEYS = {"matryca_wu_xing": ("zywiol_leczony", "narzad_matczyny_do_wsparcia", "porada_matrycy"),
           "triada_kampo": ("cel_glowny", "wyjasnienie"),
           "doktryna_sygnatur": ("sygnatura", "interpretacja"),
           "humory_galena": ("jakosc", "stopien", "wyjasnienie"),
           "alchemia_spageryczna": ("zasada", "wyjasnienie")}
KD_SYSTEM = {"matryca_wu_xing": "wu_xing", "triada_kampo": "kampo",
             "doktryna_sygnatur": "doktryna_sygnatur", "humory_galena": "humory", "alchemia_spageryczna": "spagiryka"}


def parse_systems(value: str | None) -> list[str]:
    names = [x.strip().lower() for x in (value or "").split(",") if x.strip()]
    return [n for n in names if n in SYSTEMS] or list(SYSTEMS)


def _txt(v, limit: int = 600) -> str | None:
    if v is None or isinstance(v, (dict, list)):
        return None
    s = re.sub(r"\s+", " ", str(v)).strip()
    return s[:limit] if s and s.lower() not in ("null", "none", "brak", "-") else None


def _block(d, keys: tuple[str, ...]) -> dict | None:
    if not isinstance(d, dict):
        return None
    out = {k: _txt(d.get(k)) for k in keys}
    out = {k: v for k, v in out.items() if v}
    return out or None


def strip_invented(rec: dict) -> dict:
    """Usuwa z przepisu pola, ktorych Siedziba juz nie tworzy (filar, tropizm, smak ajurwedy, ajurweda, "Wzorzec
    5 filarow") oraz puste "link_id": null. Zwraca nowy slownik; reszta przepisu bez zmian."""
    out = dict(rec)
    skl = []
    for s in out.get("skladniki") or []:
        if isinstance(s, dict):
            s = {k: v for k, v in s.items() if k not in INVENTED_INGREDIENT_KEYS and not (k == "link_id" and not v)}
        skl.append(s)
    if "skladniki" in out:
        out["skladniki"] = skl
    kd = out.get("klasyfikacja_dzialania")
    if isinstance(kd, dict) and any(k in kd for k in INVENTED_KD_KEYS):
        kd = {k: v for k, v in kd.items() if k not in INVENTED_KD_KEYS}
        if kd:
            out["klasyfikacja_dzialania"] = kd
        else:
            out.pop("klasyfikacja_dzialania")
    mech = out.get("mechanizm_tworzenia")
    if isinstance(mech, list):
        mech = [m for m in mech if not (isinstance(m, str) and "filar" in m.lower())]
        if mech:
            out["mechanizm_tworzenia"] = mech
        else:
            out.pop("mechanizm_tworzenia")
    op = out.get("opracowanie")
    if isinstance(op, dict) and isinstance(op.get("systemy"), list):
        out["opracowanie"] = {**op, "systemy": [x for x in op["systemy"] if x not in ("ajurweda", "piec_filarow")]}
    return out


def build_prompt(rec: dict, systems: list[str], source_text: str | None = None) -> str:
    base = {k: rec.get(k) for k in ("tytul", "typ", "metoda", "roslina", "opis", "skladniki", "sposob_przygotowania",
                                    "stosowanie_i_dawkowanie", "uwagi") if rec.get(k)}
    base["skladniki"] = [{k: s.get(k) for k in ("nazwa", "ilosc", "czesc_rosliny") if s.get(k)}
                         for s in rec.get("skladniki", [])]
    sys_txt = "\n".join(f"- {n}: {SYSTEMS[n]}" for n in systems)
    extra = f"\n\nFRAGMENT ORYGINALU (kontekst):\n{source_text[:1500]}" if source_text else ""
    return f"SYSTEMY:\n{sys_txt}\n\nPRZEPIS:\n{json.dumps(base, ensure_ascii=False, indent=1)}{extra}"


def merge_enrichment(rec: dict, data: dict, systems: list[str], model: str) -> dict:
    """Dokleja opracowanie do rekordu; pola ze zrodla (skladniki, ilosci, kroki, dawkowanie) maja pierwszenstwo."""
    if not isinstance(data, dict):
        raise InvalidOutputError("opracowanie: odpowiedz nie jest obiektem JSON")
    out = dict(rec)

    mech = [m for m in (_txt(x, 80) for x in (data.get("mechanizm_tworzenia") or []) if isinstance(x, str)) if m]
    if mech:
        out["mechanizm_tworzenia"] = mech[:5]

    kd_in = data.get("klasyfikacja_dzialania") if isinstance(data.get("klasyfikacja_dzialania"), dict) else {}
    kd = {}
    if _txt(kd_in.get("skalowanie_toksykologiczne")):
        kd["skalowanie_toksykologiczne"] = _txt(kd_in.get("skalowanie_toksykologiczne"))
    for key, fields in KD_KEYS.items():
        if KD_SYSTEM[key] in systems:
            b = _block(kd_in.get(key), fields)
            if b:
                kd[key] = b
    if kd:
        out["klasyfikacja_dzialania"] = kd

    if "chronoterapia" in systems:
        w = _block(data.get("wymogi_szamanskie_i_czasowe"), ("chronoterapia", "astrologia_zbioru", "dieta_i_zakazy"))
        if w:
            out["wymogi_szamanskie_i_czasowe"] = w


    src_sd = rec.get("stosowanie_i_dawkowanie") if isinstance(rec.get("stosowanie_i_dawkowanie"), dict) else {}
    new_sd = _block(data.get("stosowanie_i_dawkowanie"),
                    ("okolicznosci_stosowania", "skalowanie_pacjenta")) or {}
    sd = {**new_sd, **{k: v for k, v in src_sd.items() if v}}          # zrodlo wygrywa
    sd.pop("dawkowanie_standardowe", None)
    if src_sd.get("dawkowanie_standardowe"):                            # dawkowanie tylko ze zrodla, nigdy z LLM
        sd["dawkowanie_standardowe"] = src_sd["dawkowanie_standardowe"]
    elif str(rec.get("typ", "")).startswith(("medyczne", "kulinarno_medyczne")):
        sd["dawkowanie_standardowe"] = NO_DOSAGE
    if sd:
        out["stosowanie_i_dawkowanie"] = sd

    wf = _block(data.get("wlasciwosci_fizyczne"), ("konsystencja_i_slady", "barwienie_skory", "barwienie_ubran"))
    if wf:
        out["wlasciwosci_fizyczne"] = wf
    bez = _block(data.get("bezpieczenstwo_i_interakcje"), ("interakcje_z_lekami", "ostrzezenia"))
    out["bezpieczenstwo_i_interakcje"] = bez or {
        "ostrzezenia": "Brak opracowanych ostrzeżeń - przed użyciem sprawdź przeciwwskazania każdej rośliny."}
    tips = [t for t in (_txt(x, 300) for x in (data.get("wskazowki_tradycyjne") or []) if isinstance(x, str)) if t]
    if tips:
        out["wskazowki_tradycyjne"] = tips[:6]

    out["opracowanie"] = {"systemy": systems, "model": model, "data": date.today().isoformat(),
                          "uwaga": DISCLAIMER}
    return strip_invented(out)


async def enrich_recipe(llm, rec: dict, systems: list[str], source_text: str | None = None) -> tuple[dict, str]:
    """Zwraca (rekord z opracowaniem, 'provider/model'). Rzuca LLMError / InvalidOutputError."""
    res = await llm.complete(build_prompt(rec, systems, source_text), system=SYSTEM_PROMPT, json_mode=True)
    try:
        data = json.loads(res.text.strip().removeprefix("```json").removesuffix("```"))
    except ValueError as exc:
        raise InvalidOutputError("opracowanie: LLM zwrocil niepoprawny JSON") from exc
    model = f"{res.provider}/{res.model}"
    return merge_enrichment(rec, data, systems, model), model
