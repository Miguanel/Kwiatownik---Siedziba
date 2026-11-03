"""Stare opracowanie przepisow przez LLM - WYLACZONE (decyzja 2026-11-02: zadnych halucynacji LLM, tylko
informacje wyciagniete ze zrodel). Zostaja stale i strip_invented: czyszczenie pol, ktore dawniej dopisywal model.
Nowe opracowanie liczy bez LLM app/pipeline/opracowanie.py (pliki roslin Kwiatownika + zrodlo przepisu)."""

# Pola i systemy, ktorych Siedziba juz nie tworzy (wymyslane przez model) - usuwane tez z zapisanych przepisow
INVENTED_INGREDIENT_KEYS = ("filar", "tropizm_organowy", "smak_ajurweda")
INVENTED_KD_KEYS = ("ajurweda",)
ALGORITHMIC_SYSTEMS = ("wu_xing", "humory")
ALGORITHMIC_KD_KEYS = ("matryca_wu_xing", "humory_galena")
NO_DOSAGE = "Źródło nie podaje dawkowania."


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
    drop = INVENTED_KD_KEYS + ALGORITHMIC_KD_KEYS
    if isinstance(kd, dict) and any(k in kd for k in drop):
        kd = {k: v for k, v in kd.items() if k not in drop}
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
        out["opracowanie"] = {**op, "systemy": [x for x in op["systemy"]
                                                if x not in ("ajurweda", "piec_filarow") + ALGORITHMIC_SYSTEMS]}
    return out
