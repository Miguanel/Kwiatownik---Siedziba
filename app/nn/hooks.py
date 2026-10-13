"""Punkty styku sieci z potokiem wiedzy o roslinach (kontrola odpowiedzi LLM)."""
from __future__ import annotations

import logging

from sqlmodel import Session, col, select

from app.db import engine
from app.models import NNPrediction, NNSample, PlantFact
from app.nn.tasks import HUMAN_WEIGHT, fact_input, part_label

log = logging.getLogger(__name__)


def review_facts(fact_ids: list[int], teacher: str | None) -> dict:
    """Sekcja i czesc rosliny nowych informacji: siec porownuje je z odpowiedzia LLM.
    W trybie aktywnym uzupelnia brakujaca czesc rosliny (gdy jest pewna), rozbieznosci zglasza do panelu."""
    try:
        from app.nn.runtime import brain
        with Session(engine) as s:
            rows = s.exec(select(PlantFact).where(col(PlantFact.id).in_(fact_ids))).all()
            items = [(f"fact:{f.id}", fact_input(f.text, f.quote), f) for f in rows]
        stats = {"sekcja": {}, "czesc": {}}
        for d in brain.check("fact_section", [(r, t, f.section) for r, t, f in items], teacher):
            stats["sekcja"][d.decision] = stats["sekcja"].get(d.decision, 0) + 1
        filled = []
        for d, (ref, _, f) in zip(brain.check("fact_part", [(r, t, part_label(f.part)) for r, t, f in items], teacher),
                                  items):
            stats["czesc"][d.decision] = stats["czesc"].get(d.decision, 0) + 1
            if d.use and d.decision == "uzupelnienie":
                filled.append((f.id, d.label))
        if filled:
            with Session(engine) as s:
                for fid, part in filled:
                    f = s.get(PlantFact, fid)
                    if f and not f.part:
                        f.part = part
                        s.add(f)
                s.commit()
        return stats
    except Exception as exc:  # noqa: BLE001 - kontrola sieci nie moze zatrzymac zbierania wiedzy
        log.warning("Kontrola sieci dla informacji %s nie powiodla sie: %s", fact_ids[:5], exc)
        return {}


def resolve(pred_id: int, choice: str) -> str:
    """Rozstrzygniecie czlowieka w panelu: 'nn' (siec ma racje), 'llm' (LLM ma racje) albo konkretna etykieta.
    Poprawia informacje w bazie (sekcja / czesc) i zapisuje przyklad o najwyzszej wadze do nastepnego treningu."""
    from app.nn.runtime import record_sample
    from app.nn.tasks import TASKS
    with Session(engine) as s:
        p = s.get(NNPrediction, pred_id)
        if not p:
            return "Nie ma takiej decyzji."
        label = p.label if choice == "nn" else (p.teacher_label if choice == "llm" else choice)
        if not label:
            return "Brak etykiety do zapisania."
        p.human_label = label
        s.add(p)
        changed = ""
        group = None
        text = p.preview or ""
        if p.ref.startswith("fact:") and p.task in ("fact_section", "fact_part"):
            f = s.get(PlantFact, int(p.ref.split(":", 1)[1]))
            if f:
                group = f.plant_id
                text = fact_input(f.text, f.quote)
                if p.task == "fact_section" and f.section != label:
                    changed = f"sekcja: {f.section} -> {label}"
                    f.section = label
                elif p.task == "fact_part" and part_label(f.part) != label:
                    changed = f"czesc: {f.part or 'brak'} -> {label}"
                    f.part = None if label == "brak" else label
                s.add(f)
        elif p.ref.startswith("item:") and p.ref.split(":", 1)[1].isdigit():
            import json

            from app.models import Item
            from app.nn.tasks import recipe_fields, recipe_input, title_group
            it = s.get(Item, int(p.ref.split(":", 1)[1]))
            if it and it.data_json:
                text = recipe_input(*recipe_fields(json.loads(it.data_json)))
                group = title_group(recipe_fields(json.loads(it.data_json))[0])
        else:
            smp = s.exec(select(NNSample).where(NNSample.task == p.task, NNSample.ref == p.ref)).first()
            if smp:
                text, group = smp.text, smp.group
        s.commit()
        task, ref = p.task, p.ref
    record_sample(task, ref, text, label, "czlowiek", HUMAN_WEIGHT, group)
    title = TASKS[task].title if task in TASKS else task
    return f"{title}: zapisano '{label}'" + (f" ({changed})" if changed else "") + " - trafi do nastepnego treningu."
