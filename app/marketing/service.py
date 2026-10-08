"""Generowanie posta: pora roku -> dobor roslin -> LLM glosem postaci -> zapis szkicu w bazie."""
import asyncio
import json
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlmodel import Session, col, select

from app.config import settings
from app.db import engine
from app.marketing import history, picker, topics
from app.marketing.persona import Persona
from app.marketing.season import season_for
from app.marketing.writer import Material, compose, write_post
from app.models import FbPost, FbPostReview

SITE_URL = topics.SITE_URL


def persona_path() -> Path:
    return Path(settings.data_dir) / "marketing_persona.json"


def get_persona() -> Persona:
    return Persona.load(persona_path())


def today():
    return datetime.now(ZoneInfo(settings.timezone)).date()


def recent_plant_ids(limit: int = 10) -> set[str]:
    with Session(engine) as s:
        rows = s.exec(select(FbPost.plants_json).where(FbPost.status != "rejected")
                      .order_by(col(FbPost.id).desc()).limit(limit)).all()
    out = set()
    for r in rows:
        try:
            out |= {p["id"] for p in json.loads(r or "[]")[:1]}   # tylko glowne rosliny
        except (ValueError, KeyError, TypeError):
            pass
    return out


def season_candidates(limit: int = 30) -> list[picker.PlantPick]:
    plants = picker.load_plants(settings.kwiatownik_plants_dir)
    return picker.candidates(plants, season_for(today()), SITE_URL, recent_plant_ids(), seed=0)[:limit]


def build_material(s: Session, kind: str, plant_id: str | None, day=None, log=lambda m: None,
                   idea_id: int | None = None):
    season = season_for(day or today())
    plants = picker.load_plants(settings.kwiatownik_plants_dir)
    order = topics.auto_kind(s) if kind == "auto" else [kind]
    if kind == "auto":
        order += [k for k in topics.AUTO_ROTATION if k not in order]   # brak danych -> nastepny temat
    for k in order:
        if k == "siedziba":
            m = topics.siedziba(s, season)
        elif k == "pora_roku":
            m = topics.pora_roku(s, season, plants, plant_id, recent_plant_ids())
        elif k == "ciekawostka":
            m = topics.ciekawostka(s, season, plants, plant_id, idea_id=idea_id)
        elif k == "przepis":
            m = topics.przepis(s, season)
        else:
            raise ValueError(f"Nieznany temat posta: {k}")
        if m:
            return m
        log(f"Temat '{topics.KINDS.get(k, k)}': brak nowych danych, pomijam")
    return None


def _material_dict(m: Material) -> dict:
    d = asdict(m)
    d.pop("season", None)
    return d


def _material_from(post: FbPost, day=None) -> Material | None:
    try:
        d = json.loads(post.source_json or "")
    except ValueError:
        return None
    if not isinstance(d, dict) or "task" not in d or "kind" not in d:
        return None
    known = {f.name for f in fields(Material)} - {"season"}
    return Material(**{k: v for k, v in d.items() if k in known}, season=season_for(day or today()))


_LOCK = asyncio.Lock()   # posty po kolei: temat "auto" i niepowtarzanie zaleza od juz zapisanych postow


async def generate_post(llm, kind: str = "auto", plant_id: str | None = None, hint: str | None = None,
                        log=lambda m: None, job_id: int | None = None, day=None, idea_id: int | None = None) -> int:
    """idea_id: post z konkretnej ciekawostki z puli (/posts/ideas)."""
    if llm is None:
        raise RuntimeError("Brak modeli LLM (ustaw klucze Gemini/Groq albo Ollame)")
    async with _LOCK:
        return await _generate(llm, "ciekawostka" if idea_id else kind, plant_id, hint, log, job_id, day, idea_id)


async def _generate(llm, kind, plant_id, hint, log, job_id, day, idea_id=None) -> int:
    persona = get_persona()
    with Session(engine) as s:
        m = build_material(s, kind or "auto", plant_id or None, day, log, idea_id)
    if m is None:
        raise RuntimeError("Brak danych do posta (zadna roslina / ciekawostka / przepis nie pasuje)")
    log(f"Temat: {topics.KINDS.get(m.kind, m.kind)} - {m.label}" + (f"; {m.season.label}" if m.season else ""))
    draft, model = await write_post(llm, persona, m, hint, log)
    text = compose(draft, persona, m) if draft.body else ""
    post = FbPost(kind=m.kind, ref=m.ref, persona=persona.name, season=m.season.label if m.season else None,
                  hint=hint or None, body=draft.body, text=text, model=model,
                  problems="; ".join(draft.problems) or None, job_id=job_id,
                  plants_json=json.dumps(m.plants, ensure_ascii=False),
                  source_json=json.dumps(_material_dict(m), ensure_ascii=False))
    with Session(engine) as s:
        s.add(post)
        s.commit()
        history.record(s, post, "generated", model)
        s.commit()
        pid = post.id
    log(f"Zapisano szkic posta #{pid}" + (f" (uwagi: {post.problems})" if post.problems else ""))
    return pid


async def review_posts(llm, post_ids: list[int] | None = None, log=lambda m: None, should_stop=lambda: False) -> int:
    """Ocena wskazanych postow (None = wszystkie bez aktualnej oceny). Zwraca liczbe ocenionych."""
    from app.marketing.review import needs_review, review_post
    if post_ids is None:
        with Session(engine) as s:
            post_ids = needs_review(s)
    log(f"Do oceny: {len(post_ids)} postow")
    persona, done = get_persona(), 0
    for pid in post_ids:
        if should_stop():
            log("Zatrzymano")
            break
        if await review_post(engine, llm, pid, persona, log):
            done += 1
    return done


def split_text(old: FbPost, persona) -> tuple[str, str]:
    """Gotowy tekst -> (tresc, ogon). Ogon = podpis, link, zrodlo, hashtagi - zostaja bez zmian przy redakcji."""
    text = (old.text or "").replace("\r\n", "\n")
    if old.body and text.startswith(old.body):
        return old.body, text[len(old.body):]
    sig = (persona.signature or "").strip()
    if sig and sig in text:
        i = text.rindex(sig)
        return text[:i].rstrip(), "\n\n" + text[i:]
    return text.strip(), ""


def _fallback_material(old: FbPost) -> Material:
    """Post sprzed zapisywania danych zrodlowych: redakcja bez nowych faktow."""
    try:
        main = (json.loads(old.plants_json or "[]") or [{}])[0]
    except ValueError:
        main = {}
    return Material(kind=old.kind or "auto", label="redakcja", task="Popraw tekst posta według uwag redaktora.",
                    data="(brak zapisanych danych źródłowych – NIE dodawaj żadnych nowych faktów, liczb ani nazw; "
                         "korzystaj tylko z tego, co jest w tekście)",
                    season=season_for(today()), main_name=main.get("nazwa_pl"), main_latin=main.get("nazwa_lat"),
                    plants=[main] if main else [], ref=old.ref)


IMPROVE_RULES = (
    "ULEPSZ CAŁY TEKST na podstawie raportu redaktora – nie tylko wskazane zdania. Wdróż WSZYSTKIE uwagi, "
    "w kolejności ważności:\n"
    "1) [BŁĄD] i [RYZYKO] – usuń albo popraw (żadnych zmyśleń, obietnic leczenia, dawek),\n"
    "2) kryteria z oceną poniżej 8 – podnieś każde do co najmniej 8/10 (opis kryterium mówi, czego brakuje),\n"
    "3) [SUGESTIA], [SŁABOŚĆ], [POMIAR] i [POCZĄTEK],\n"
    "4) [ZACHOWAJ] – mocne strony muszą zostać.\n"
    "Przejrzyj cały tekst: pierwsze zdanie (zaciekawia, do 15 słów), układ 2–4 akapitów, rytm zdań, naturalną "
    "i czytelną gwarę, konkret i wartość dla czytelnika, przejście do Kwiatownika, zakończenie pytaniem. "
    "Fakty tylko z DANYCH i z tekstu. Do JSON-a dodaj pole \"wdrozone\": lista krótkich zdań – jak wdrożyłeś "
    "każdą uwagę (np. „[CZYTELNOŚĆ] podzieliłem na 3 akapity”)."
)


def _changes_from_review(rev: FbPostReview | None) -> dict:
    if not rev:
        return {}
    return {"ocena": rev.overall, "werdykt": rev.verdict, "oceny": json.loads(rev.scores_json or "{}")}


async def improve_post(llm, post_id: int, log=lambda m: None, job_id: int | None = None, mode: str = "nowa",
                       rounds: int | None = None, target: float | None = None) -> int:
    """Wdrozenie poprawek z raportu - ulepszenie CALEGO tekstu, w rundach "popraw -> ocen":
    kazda runda dostaje raport poprzedniej wersji; konczy sie, gdy wersja osiagnie ocene docelowa
    (MARKETING_TARGET_SCORE) albo po MARKETING_IMPROVE_ROUNDS rundach. Zapisywana jest NAJLEPSZA wersja
    (z oceną, bez ponownej recenzji); stara dostaje status 'odrzucony' tylko, gdy nowa jest lepsza.
    mode="redaguj": punktem wyjscia jest AKTUALNY tekst z pola (z recznymi poprawkami), ogon bez zmian;
    mode="nowa":    mozna napisac od nowa na tych samych danych."""
    from app.config import settings
    from app.marketing.review import evaluate, improvement_notes, latest_reviews
    if llm is None:
        raise RuntimeError("Brak modeli LLM (ustaw klucze Gemini/Groq albo Ollame)")
    rounds = max(1, rounds or settings.marketing_improve_rounds)
    target = settings.marketing_target_score if target is None else target
    async with _LOCK:
        with Session(engine, expire_on_commit=False) as s:
            old = s.get(FbPost, post_id)
            if not old:
                raise RuntimeError(f"Brak posta #{post_id}")
            rev = latest_reviews(s, [post_id]).get(post_id)
            history.ensure_initial(s, old)
            s.commit()
            s.expunge_all()
        if mode == "redaguj" and not rev:
            raise RuntimeError(f"Post #{post_id} nie ma raportu oceny - najpierw kliknij 'Ocen'")
        persona = get_persona()
        m = _material_from(old)
        keep_tail = mode == "redaguj" or m is None
        if m is None:
            log("Brak zapisanych danych zrodlowych - poprawa bez nowych faktow")
            m = _fallback_material(old)
        body, tail = split_text(old, persona)
        start = rev.overall if rev and rev.text_hash == history.text_hash(old.text) else None
        cur_body, cur_rev = body, rev
        best, tried = None, []
        for n in range(1, rounds + 1):
            notes = improvement_notes(cur_rev)
            base = "TEKST DO ULEPSZENIA" if (mode == "redaguj" or n > 1) else "POPRZEDNIA WERSJA (możesz napisać od nowa)"
            revise = (IMPROVE_RULES + (" Zachowaj sens ręcznych poprawek autora." if mode == "redaguj" else "")
                      + "\n\nUWAGI Z RAPORTU:\n" + ("\n".join(notes) or "- popraw cały tekst według zasad")
                      + f"\n\n{base}:\n{cur_body}")
            log(f"Runda {n}/{rounds}: wdrazam {len(notes)} uwag" + (f" (ocena wyjsciowa {cur_rev.overall:.0f})"
                                                                     if cur_rev else ""))
            draft, model = await write_post(llm, persona, m, old.hint, log, revise=revise)
            if not draft.body:
                log(f"Runda {n}: pusta odpowiedz modelu")
                continue
            if keep_tail:
                text = draft.body.strip() + (tail if tail.startswith("\n") else ("\n\n" + tail if tail else ""))
            else:
                text = compose(draft, persona, m)
            cand = FbPost(kind=old.kind or m.kind, ref=old.ref or m.ref, persona=persona.name,
                          season=m.season.label if m.season else old.season, hint=old.hint, body=draft.body,
                          text=text, model=model, problems="; ".join(draft.problems) or None, job_id=job_id,
                          plants_json=old.plants_json, source_json=old.source_json, parent_id=post_id,
                          root_id=old.root_id or old.id)
            ev = await evaluate(llm, cand, persona, log, label=f"runda {n}")
            tried.append({"runda": n, "ocena": ev.overall, "werdykt": ev.verdict, "model": model,
                          "wdrozone": draft.changes, "cand": cand, "ev": ev, "uwagi": len(notes)})
            if best is None or ev.overall > best["ocena"]:
                best = tried[-1]
            if ev.overall >= target and ev.verdict == "publikuj":
                log(f"Runda {n}: osiagnieto ocene docelowa {target:.0f} - koniec")
                break
            cur_body, cur_rev = draft.body, ev
        if best is None:
            raise RuntimeError("Model nie zwrocil zadnej poprawionej wersji")
        post, ev = best["cand"], best["ev"]
        better = start is None or ev.overall > start
        post.revision_json = json.dumps({
            "z_posta": post_id, "tryb": mode, "ocena_przed": start, "ocena_po": ev.overall,
            "przed": _changes_from_review(rev), "wdrozone": best["wdrozone"], "najlepsza_runda": best["runda"],
            "rundy": [{"runda": t["runda"], "ocena": t["ocena"], "werdykt": t["werdykt"], "model": t["model"],
                       "uwagi": t["uwagi"]} for t in tried],
        }, ensure_ascii=False)
        kind = "rewrite" if mode == "redaguj" else "improve"
        after = ev.overall
        with Session(engine) as s:
            s.add(post)
            s.commit()
            for t in tried:   # oceny odrzuconych rund (do historii), ocena wybranej wersji na koncu = aktualna
                if t is not best:
                    t["ev"].post_id = post.id
                    s.add(t["ev"])
            s.flush()
            ev.post_id = post.id
            s.add(ev)
            for t in tried:   # odrzucone rundy tez trafiaja do historii - mozna je porownac
                if t is not best:
                    history.record(s, post, "round", f"runda {t['runda']}: {t['ocena']:.0f}/100",
                                   text=t["cand"].text)
            history.record(s, post, kind, f"z #{post_id}, runda {best['runda']}: "
                           + (f"{start:.0f} → " if start is not None else "") + f"{after:.0f}/100")
            o = s.get(FbPost, post_id)
            if better and o.status in ("draft", "approved"):
                o.status = "rejected"
                o.problems = ((o.problems + "; ") if o.problems else "") + f"zastąpiony lepszą wersją #{post.id}"
                s.add(o)
            s.commit()
            pid = post.id
        log(f"Zapisano wersje #{pid}: " + (f"{start:.0f} -> " if start is not None else "") + f"{after:.0f}/100"
            + ("" if better else f" - NIE lepsza od #{post_id}, stara wersja zostaje"))
        return pid


async def collect_ideas(llm, log=lambda m: None, should_stop=lambda: False) -> dict:
    """Zbieranie i ocena kandydatow na ciekawostki (zadanie fb_ideas)."""
    from app.marketing import ideas
    plants = picker.load_plants(settings.kwiatownik_plants_dir)
    return await ideas.collect(engine, llm, plants, log, settings.ideas_items_per_run, settings.ideas_rate_per_run,
                               settings.ideas_rate_batch, should_stop)
