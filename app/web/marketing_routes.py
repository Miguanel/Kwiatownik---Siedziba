"""Widok "Fanpage": posty Lesnego Dziadygi na fanpage Kwiatownika (generowanie, edycja, kopiowanie)."""
import json
from dataclasses import fields
from datetime import datetime, timezone

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse
from sqlmodel import Session, col, func, select

from app.db import engine
from app.marketing import history, metrics, review, service, topics
from app.marketing import stats as mstats
from app.marketing.persona import Persona
from app.marketing.season import season_for
from app.marketing.writer import word_count
from app.models import FbPost, FbPostVersion, Job, JobStatus
from app.web.templating import templates
from app.worker import actions

router = APIRouter()
STATUSES = {"draft": "szkic", "approved": "zatwierdzony", "published": "opublikowany", "rejected": "odrzucony"}


def _redirect(url: str) -> RedirectResponse:
    return RedirectResponse(url, status_code=303)


VERDICT_LABELS = {"publikuj": "gotowy do publikacji", "popraw": "do poprawy", "odrzuc": "do odrzucenia"}


def _review_view(r, post) -> dict | None:
    if r is None:
        return None
    m = json.loads(r.metrics_json or "{}")
    scores = json.loads(r.scores_json or "{}")
    return {
        "overall": r.overall, "code": r.code_score, "expert": r.expert_score, "verdict": r.verdict,
        # srednia z ocen eksperta 1-10 (bez eksperta: ocena koncowa przeliczona na skale 10)
        "avg": round(sum(scores.values()) / len(scores), 1) if scores else round(r.overall / 10, 1),
        "avg_n": len(scores),
        "model": r.model, "at": r.created_at, "stale": r.text_hash != review.text_hash(post.text),
        "scores": [{"key": k, "label": lab, "v": scores[k]} for k, (lab, _, _) in review.CRITERIA.items()
                   if k in scores],
        "sub": [{"label": metrics.SUBSCORE_LABELS.get(k, k), "v": v} for k, v in m.get("podwyniki", {}).items()],
        "m": m, "issues": [{"label": metrics.ISSUE_LABELS.get(i["kod"], i["kod"]), "opis": i["opis"]}
                           for i in m.get("problemy", [])],
        "rev": json.loads(r.review_json or "{}"),
    }


def _list_ctx(status: str = "") -> dict:
    with Session(engine) as s:
        stmt = select(FbPost).order_by(col(FbPost.id).desc()).limit(60)
        if status:
            stmt = stmt.where(FbPost.status == status)
        posts = s.exec(stmt).all()
        counts = dict(s.exec(select(FbPost.status, func.count()).group_by(FbPost.status)).all())
        reviews = review.latest_reviews(s, [p.id for p in posts])
        roots = [p.root_id or p.id for p in posts]
        vcount = dict(s.exec(select(FbPostVersion.root_id, func.count()).where(
            col(FbPostVersion.root_id).in_(roots)).group_by(FbPostVersion.root_id)).all()) if roots else {}
        to_review = len(review.needs_review(s))
    posts = [{**p.model_dump(), "plants": json.loads(p.plants_json or "[]"), "words": word_count(p.body),
              "review": _review_view(reviews.get(p.id), p), "versions": vcount.get(p.root_id or p.id, 0),
              "revision": json.loads(p.revision_json) if p.revision_json else None} for p in posts]
    return {"posts": posts, "counts": counts, "status": status, "statuses": STATUSES,
            "kinds": topics.KINDS, "to_review": to_review, "verdicts": VERDICT_LABELS, **_status_ctx(status)}


MARKETING_JOBS = ["fb_post", "fb_review", "fb_ideas"]


def _status_ctx(status: str = "") -> dict:
    """Maly pasek stanu zadan - odpytywany co kilka sekund ZAMIAST calej listy postow."""
    with Session(engine) as s:
        running = s.exec(select(Job).where(col(Job.kind).in_(MARKETING_JOBS), col(Job.status).in_(
            [JobStatus.queued, JobStatus.running])).order_by(Job.id)).all()
        last = s.exec(select(Job).where(col(Job.kind).in_(MARKETING_JOBS), col(Job.status).in_(
            [JobStatus.done, JobStatus.failed, JobStatus.cancelled])).order_by(col(Job.id).desc()).limit(1)).first()
    failed = last if last and last.status == JobStatus.failed else None
    return {"running": running, "failed": failed, "seen": last.id if last else 0, "status": status}


@router.get("/posts")
def posts_page(request: Request, status: str = "", msg: str | None = None):
    persona = service.get_persona()
    season = season_for(service.today())
    ctx = _list_ctx(status)
    with Session(engine) as s:
        next_kind = topics.auto_kind(s)[0]
    ctx.update({"kinds": topics.KINDS, "next_kind": next_kind, "persona": persona, "season": season, "candidates": service.season_candidates(), "msg": msg})
    return templates.TemplateResponse(request, "posts.html", ctx)


@router.get("/posts/status")
def posts_status(request: Request, seen: int = 0, status: str = ""):
    """Pasek stanu. Gdy skonczylo sie nowe zadanie (pisanie/ocena) -> zdarzenie 'posts-changed':
    lista odswiezy sie RAZ (z zachowaniem rozwinietych raportow i pozycji ekranu)."""
    ctx = _status_ctx(status)
    resp = templates.TemplateResponse(request, "posts_status.html", ctx)
    if ctx["seen"] > seen:
        resp.headers["HX-Trigger"] = "posts-changed"
    return resp


@router.get("/posts/list")
def posts_list(request: Request, status: str = ""):
    return templates.TemplateResponse(request, "posts_list.html", _list_ctx(status))


@router.post("/posts/generate")
async def posts_generate(request: Request, kind: str = Form("auto"), plant_id: str = Form(""), hint: str = Form(""),
                   count: int = Form(1)):
    kind = kind if kind in topics.KINDS or kind == "auto" else "auto"
    if kind == "auto" and plant_id.strip():   # wybrana roslina = post o niej (ziele na te pore roku)
        kind = "pora_roku"
    actions.start_fb_post(request.app.state, kind, plant_id.strip() or None, hint.strip() or None,
                          count=max(1, min(count, 5)))
    return _redirect("/posts?msg=Dziadyga+pisze...")


@router.post("/posts/{post_id}/save")
def posts_save(post_id: int, text: str = Form(...), status: str = Form("")):
    with Session(engine) as s:
        p = s.get(FbPost, post_id)
        if p:
            if text.replace("\r\n", "\n") != p.text:
                history.ensure_initial(s, p)
                p.text, p.edited = text.replace("\r\n", "\n"), True
                history.record(s, p, "edit")
            if status in STATUSES:
                p.status = status
                p.published_at = datetime.now(timezone.utc) if status == "published" else p.published_at
            s.add(p)
            s.commit()
    return _redirect(f"/posts#post-{post_id}")


@router.post("/posts/{post_id}/again")
async def posts_again(request: Request, post_id: int):
    """Nowa wersja z ta sama glowna roslina i zyczeniem."""
    with Session(engine) as s:
        p = s.get(FbPost, post_id)
        if not p:
            return _redirect("/posts")
        plants = json.loads(p.plants_json or "[]")
        main_id, hint, kind = (plants[0]["id"] if plants else None), p.hint, p.kind or "auto"
        if kind == "siedziba" and p.status != "rejected":   # ta sama czesc o Siedzibie: odrzuc stara wersje
            p.status = "rejected"
            s.add(p)
            s.commit()
    actions.start_fb_post(request.app.state, kind, main_id if kind in ("pora_roku", "ciekawostka") else None, hint)
    return _redirect("/posts?msg=Nowa+wersja+w+drodze")


@router.post("/posts/persona")
async def posts_persona(request: Request):
    form = await request.form()
    current = service.get_persona()
    data = {}
    for f in fields(Persona):
        if f.name not in form:
            data[f.name] = getattr(current, f.name)
            continue
        val = str(form[f.name]).replace("\r\n", "\n").strip()
        if f.type in (int, "int"):
            try:
                val = max(0, int(val))
            except ValueError:
                val = getattr(current, f.name)
        data[f.name] = val
    persona = Persona(**data)
    if persona.max_words < persona.min_words:
        persona.max_words = persona.min_words
    persona.save(service.persona_path())
    return _redirect("/posts?msg=Zapisano+postac")


@router.post("/posts/persona/reset")
def posts_persona_reset():
    Persona().save(service.persona_path())
    return _redirect("/posts?msg=Przywrocono+domyslna+postac")


@router.post("/posts/review")
async def posts_review_all(request: Request):
    actions.start_fb_review(request.app.state, None)
    return _redirect("/posts?msg=Ocena+postow+w+toku")


@router.post("/posts/{post_id}/review")
async def posts_review(request: Request, post_id: int):
    actions.start_fb_review(request.app.state, [post_id])
    return _redirect(f"/posts?msg=Ocena+posta+%23{post_id}+w+toku#post-{post_id}")


@router.post("/posts/{post_id}/rewrite")
async def posts_rewrite(request: Request, post_id: int, text: str = Form(...)):
    """Przeredaguj aktualny tekst z pola (z recznymi poprawkami) wedlug ostatniego raportu oceny."""
    with Session(engine) as s:
        p = s.get(FbPost, post_id)
        if not p:
            return _redirect("/posts")
        text = text.replace("\r\n", "\n")
        if text != p.text:            # najpierw zapisz to, co jest w polu - redakcja idzie od tej wersji
            history.ensure_initial(s, p)
            p.text, p.edited = text, True
            history.record(s, p, "edit")
            s.add(p)
            s.commit()
        kind = p.kind or "auto"
    actions.start_fb_post(request.app.state, kind, improve=post_id, mode="redaguj")
    return _redirect("/posts?msg=Dziadyga+redaguje+tekst+wedlug+raportu")


@router.post("/posts/{post_id}/improve")
async def posts_improve(request: Request, post_id: int):
    """Przepisanie posta wedlug uwag z oceny (te same dane, stara wersja -> odrzucona)."""
    with Session(engine) as s:
        p = s.get(FbPost, post_id)
        kind = (p.kind if p else None) or "auto"
    actions.start_fb_post(request.app.state, kind, improve=post_id)
    return _redirect("/posts?msg=Dziadyga+poprawia+post+wedlug+uwag")


def _int(v: str) -> int | None:
    v = (v or "").strip().replace(" ", "")
    return int(v) if v.isdigit() else None


@router.post("/posts/{post_id}/results")
def posts_results(post_id: int, reach: str = Form(""), reactions: str = Form(""), comments: str = Form(""),
                  shares: str = Form("")):
    """Wyniki z Facebooka wpisane recznie (do statystyk: ocena vs rzeczywiste zaangazowanie)."""
    with Session(engine) as s:
        p = s.get(FbPost, post_id)
        if p:
            p.reach, p.reactions, p.comments, p.shares = _int(reach), _int(reactions), _int(comments), _int(shares)
            s.add(p)
            s.commit()
    return _redirect(f"/posts?status=published#post-{post_id}")


@router.get("/posts/stats")
def posts_stats(request: Request):
    from zoneinfo import ZoneInfo

    from app.config import settings
    with Session(engine) as s:
        st = mstats.build(s, ZoneInfo(settings.timezone))
        to_review = len(review.needs_review(s))
        running = s.exec(select(Job).where(Job.kind == "fb_review", col(Job.status).in_(
            [JobStatus.queued, JobStatus.running]))).first()
    return templates.TemplateResponse(request, "posts_stats.html",
                                      {"st": st, "to_review": to_review, "running": running,
                                       "verdicts": VERDICT_LABELS, "kinds": topics.KINDS})


def _version_view(v: FbPostVersion, idx: int, posts: dict, revs: dict) -> dict:
    r = revs.get(history.text_hash(v.text))
    p = posts.get(v.post_id)
    return {"id": v.id, "n": idx, "post_id": v.post_id, "kind": v.kind, "label": history.KIND_LABELS.get(v.kind, v.kind),
            "note": v.note, "at": v.created_at, "text": v.text, "words": word_count(v.text),
            "current": bool(p and p.text == v.text), "post_status": p.status if p else None,
            "overall": r.overall if r else None, "verdict": r.verdict if r else None,
            "scores": json.loads(r.scores_json or "{}") if r else {}}


@router.get("/posts/{post_id}/history")
def posts_history(request: Request, post_id: int, a: int | None = None, b: int | None = None, view: str = "inline"):
    """Historia edycji calej rodziny wersji posta + porownanie dwoch wersji (slowo po slowie)."""
    with Session(engine) as s:
        posts, versions = history.family(s, post_id)
        if not posts:
            return _redirect("/posts")
        pmap = {p.id: p for p in posts}
        revs = history.reviews_by_hash(s, list(pmap))
        revisions = {p.id: json.loads(p.revision_json) for p in posts if p.revision_json}
    vs = [_version_view(v, i + 1, pmap, revs) for i, v in enumerate(versions)]
    by_id = {v["id"]: v for v in vs}
    cur = [v for v in vs if v["post_id"] == post_id and v["current"]]
    vb = by_id.get(b) if b else (cur[-1] if cur else (vs[-1] if vs else None))
    if a:
        va = by_id.get(a)
    else:   # domyslnie: poprzednia "prawdziwa" wersja (bez odrzuconych rund) - np. tekst sprzed poprawki
        before = [v for v in vs if vb and v["n"] < vb["n"] and v["kind"] != "round"]
        va = before[-1] if before else (vs[0] if vs else None)
    diff = history.word_diff(va["text"], vb["text"]) if va and vb else None
    crit = [{"label": lab, "a": (va or {}).get("scores", {}).get(k), "b": (vb or {}).get("scores", {}).get(k)}
            for k, (lab, _, _) in review.CRITERIA.items()]
    crit = [c for c in crit if c["a"] is not None or c["b"] is not None]
    return templates.TemplateResponse(request, "posts_history.html", {
        "post_id": post_id, "versions": vs, "va": va, "vb": vb, "diff": diff, "crit": crit, "view": view,
        "revisions": revisions, "statuses": STATUSES, "verdicts": VERDICT_LABELS})


@router.post("/posts/{post_id}/restore/{version_id}")
def posts_restore(post_id: int, version_id: int):
    """Przywraca tekst wybranej wersji do posta (jako nowa wersja w historii - nic nie ginie)."""
    with Session(engine) as s:
        p, v = s.get(FbPost, post_id), s.get(FbPostVersion, version_id)
        if p and v and v.root_id == history.root_of(p):
            history.ensure_initial(s, p)
            p.text, p.edited = v.text, True
            if p.status == "rejected":
                p.status = "draft"
            history.record(s, p, "restore", f"z wersji {version_id}")
            s.add(p)
            s.commit()
    return _redirect(f"/posts/{post_id}/history")


@router.get("/posts/ideas")
def posts_ideas(request: Request, show: str = "pool", msg: str | None = None):
    """Pula ciekawostek: kandydaci zebrani przez Siedzibe, ocena kodem i modelem, podsuwani modelom piszacym."""
    from app.marketing import ideas
    from app.models import FbIdea
    month = service.today().month
    with Session(engine) as s:
        c = ideas.counts(s)
        used = ideas.used_idea_ids(s)
        if show == "pool":
            rows = ideas.pool(s, month, limit=120)
        else:
            stmt = select(FbIdea).where(FbIdea.status != "empty")
            if show == "rejected":
                stmt = stmt.where(FbIdea.status == "rejected")
            elif show == "used":
                stmt = stmt.where(col(FbIdea.id).in_(used or [-1]))
            elif show == "risk":
                stmt = stmt.where(FbIdea.risk == True)  # noqa: E712
            rows = [(x, x.score) for x in s.exec(stmt.order_by(col(FbIdea.score).desc()).limit(200)).all()]
        running = s.exec(select(Job).where(Job.kind == "fb_ideas", col(Job.status).in_(
            [JobStatus.queued, JobStatus.running]))).first()
        last = s.exec(select(Job).where(Job.kind == "fb_ideas").order_by(col(Job.id).desc())).first()
    items = [{"x": x, "sc": sc, "months": json.loads(x.months_json or "[]"), "used": x.id in used,
              "foreign": ideas.is_foreign(x.language, x.original),
              "season": month in json.loads(x.months_json or "[]")} for x, sc in rows]
    return templates.TemplateResponse(request, "posts_ideas.html", {
        "items": items, "c": c, "show": show, "msg": msg, "running": running, "last": last, "month": month})


@router.post("/posts/ideas/collect")
async def posts_ideas_collect(request: Request):
    actions.start_fb_ideas(request.app.state)
    return _redirect("/posts/ideas?msg=Siedziba+zbiera+i+ocenia+ciekawostki")


@router.post("/posts/ideas/{idea_id}/write")
async def posts_ideas_write(request: Request, idea_id: int, hint: str = Form("")):
    actions.start_fb_post(request.app.state, "ciekawostka", hint=hint.strip() or None, idea_id=idea_id)
    return _redirect("/posts?msg=Dziadyga+pisze+post+z+ciekawostki")


@router.post("/posts/ideas/{idea_id}/status")
def posts_ideas_status(idea_id: int, status: str = Form(...), show: str = Form("pool")):
    from app.marketing import ideas
    from app.models import FbIdea
    with Session(engine) as s:
        x = s.get(FbIdea, idea_id)
        if x and status in ("rejected", "restore"):
            x.status = "rejected" if status == "rejected" else ("rated" if x.llm_score is not None else "new")
            x.score = ideas.final_score(x)
            s.add(x)
            s.commit()
    return _redirect(f"/posts/ideas?show={show}#idea-{idea_id}")
